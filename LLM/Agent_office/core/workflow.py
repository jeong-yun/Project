"""조직도(트리) 워크플로우 — LangGraph.

    START → brief → research → work ⟲ (할 일이 남아 있는 동안 반복) → human_decision(interrupt) → feedback → END

  brief    : 질문을 '최종 상태 정의(Desired State)' 브리프로 정리 (core/brief.py) — 모든 에이전트가 이 브리프로 일한다
  research : 웹 검색(선택) — 결과를 모든 에이전트 프롬프트에 근거 자료로 넣는다

동작 (첨부 그림의 Multi-Agent Team 구조)
  👤 사용자 ↔ supervisor(클릭한 에이전트, 기본은 팀장)
  위 → 아래 : 관리자가 질문을 보고 필요한 하위 에이전트를 골라 업무를 나눠 준다          (plan)
  맨 아래   : 하위가 없는 실무자가 초안을 쓴다                                           (draft)
  아래 → 위 : 관리자가 하위 결과를 검토(1명) 또는 통합(여러 명)해 위로 올린다           (review / integrate)
  꼭대기    : supervisor 의 통합 결과가 최종 답변 → 사람이 채택/수정/반려 → 교훈 피드백

'할 일 스택(agenda)' 방식
  재귀 함수 대신 스택에 할 일을 쌓고 work 노드가 한 번에 하나씩 꺼내 처리한다.
    visit(X)     : X 가 관리자면 업무 분배 후 [integrate(X), visit(자식들)…] 을 쌓는다 / 실무자면 초안 작성
    integrate(X) : 자식들 결과가 다 모이면 X 가 검토·통합
  → 단계마다 LangGraph 체크포인트가 남아 진행 상황을 화면에 실시간으로 보여 주고, 중간에 멈춰도 이어서 진행할 수 있다.

검증 (core/verification.py, core/harness)
  초안 → 하네스(결정론적 검증) 실패 시 에러 로그를 돌려주고 재작성 (Backtracking)
  검토·통합 → 비평가(Critic) 채점 → 불합격이면 자가 교정 → 하네스 검증
"""

import operator
from collections.abc import Iterator
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from core import harness, search, verification
from core.agent import OrgAgent, chunk_text, with_feedback
from core.llm_factory import NoAvailableModelError
from core.brief import DesiredState, draft_brief, empty_state
from core.config import load_skills, parent_map, subtree
from core.feedback import generate_lessons
from core.schemas import AgentAnswer, HumanDecision, Lesson, ModelsConfig, OrgNode, Rank, Team, ToolResult
from core.search import WebContext
from core.storage import Storage
from core.tools import run_tools

WEB_SEARCH_TOOL = "web_search"
DRAFT_TAG = "draft:"  # 초안 스트리밍 토큰에 붙이는 태그 접두어 (뒤에 에이전트 id)


class OfficeState(TypedDict, total=False):
    """그래프가 단계마다 저장·전달하는 상태."""

    question: str  # 사용자가 입력한 원 질문
    root_id: str  # 이번 실행의 supervisor (클릭한 에이전트)
    brief_input: DesiredState | None  # 사용자가 화면에서 직접 채운 브리프 (일부만 채워도 됨)
    auto_brief: bool  # 빈 항목을 보조 모델이 채울지
    brief: DesiredState  # 완성된 브리프 → 에이전트들이 실제로 받는 업무 지시
    use_web: bool
    web: WebContext | None
    tool_inputs: dict  # 팀 업무 양식의 입력값 → 데이터 툴 입력 (예: stock, quantity, start_date)
    tool_results: list[ToolResult]  # 데이터 툴 결과 (주가·여행 달력 등)
    extra_validators: list[str]  # 업무 양식이 추가로 요구하는 하네스 검증기
    harness_options: dict  # 업무·툴이 정한 하네스 옵션 (예: required_terms, trip_days)
    agenda: list[dict]  # 할 일 스택 [{"op": "visit"|"integrate", "node": id}]
    tasks: dict[str, str]  # 에이전트 id → 맡은 업무 (위에서 내려온 지시)
    assigned: dict[str, list[str]]  # 관리자 id → 업무를 맡긴 하위 id 목록
    results: dict[str, AgentAnswer]  # 에이전트 id → 그 에이전트가 위로 올린 최종 결과
    steps: Annotated[list[AgentAnswer], operator.add]  # 타임라인 (모든 단계, 누적)
    decision: HumanDecision
    lessons: list[Lesson]
    notes: list[str]


def make_checkpointer(db_path):
    """사람 결정 대기 상태를 저장할 체크포인터. 서버를 재시작해도 이어서 진행할 수 있도록 SQLite 를 쓴다.

    langgraph-checkpoint-sqlite 가 없으면 메모리 체크포인터로 대체 (서버 재시작 시 대기 상태 소실).
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    # Streamlit 은 매 실행을 다른 스레드에서 돌리므로 check_same_thread=False
    return SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))


def build_office_graph(
    org: OrgNode,
    ranks: list[Rank],
    team: Team,
    models: ModelsConfig,
    settings: dict,
    verification_cfg: dict | None = None,
    storage: Storage | None = None,
    checkpointer=None,
):
    """조직도 전체로 그래프를 만든다. 어느 에이전트부터 시작할지는 실행할 때 root_id 로 정한다."""
    verification_cfg = verification_cfg or {}
    rewrite_answer = settings.get("review", {}).get("rewrite_answer", True)
    max_lessons = settings.get("feedback", {}).get("max_lessons_in_prompt", 5)
    critic_cfg = verification.critic_settings(verification_cfg, team.verification)
    rubric = team.rubric or verification_cfg.get("review_rubric")
    with_human = checkpointer is not None  # interrupt 는 체크포인터가 있어야 동작한다

    ranks_by_id = {r.id: r for r in ranks}
    skills = load_skills(team.id)  # teams/<팀>/skills/*.md
    nodes = {n.id: n for n in org.walk()}
    names = {n.id: n.name or ranks_by_id[n.rank].name for n in nodes.values()}
    agents = {
        n.id: OrgAgent(n, ranks_by_id[n.rank], team, models, rubric=rubric, names=names, skills=skills)
        for n in nodes.values()
    }
    parents = parent_map(org)

    # ── 작은 도우미 ──────────────────────────────────────────────

    def lessons_for(node_id: str) -> list[str] | None:
        return storage.recent_lessons(team.id, node_id, max_lessons) if storage else None

    def evidence_parts(state: OfficeState) -> list[str]:
        """근거 자료 묶음 = 데이터 툴 결과(사실 수치)마다 1개 + 웹 검색 결과 1개."""
        parts = [f"## {t.title}\n{t.content}" for t in state.get("tool_results") or [] if t.content]
        web = state.get("web")
        if web and web.results:
            parts.append("## 🌐 웹 검색 결과\n" + web.as_prompt())
        return parts

    def web_text(state: OfficeState) -> str:
        """에이전트에게 줄 근거 자료 = 데이터 툴 결과(사실 수치) + 웹 검색 결과."""
        return "\n\n".join(evidence_parts(state))

    def emit(event: dict) -> None:
        """화면에 '누가 지금 무엇을 하는지' 알린다 (stream_mode='custom' 이벤트)."""
        get_stream_writer()(event)

    def harness_plan(state: OfficeState, node_id: str | None = None) -> tuple[list[str], dict]:
        """이번 실행에서 돌릴 검증기와 옵션 = 팀 기본 + 업무 양식 추가분 + 툴이 계산한 값.

        node_id 가 최종 답변자(root)가 아니면 final_only 검증기는 뺀다.
        """
        validators = list(dict.fromkeys([*team.harness.validators, *(state.get("extra_validators") or [])]))
        if node_id is not None and node_id != state["root_id"]:
            validators = [v for v in validators if v not in team.harness.final_only]
        options = {**team.harness.options, **(state.get("harness_options") or {})}
        for t in state.get("tool_results") or []:
            options = {**t.harness_options, **options}
        return validators, options

    def run_harness(answer: AgentAnswer, redo, state: OfficeState) -> AgentAnswer:
        """하네스 검증 → 실패하면 에러 로그를 돌려주고 다시 쓰게 한다 (최대 max_retries 회).

        redo(feedback, previous) 는 '피드백을 반영해 다시 작성'하는 함수 (초안/검토마다 다름).
        """
        cfg = team.harness
        validators, options = harness_plan(state, answer.node_id)
        if not validators:
            return answer
        log: list[str] = []
        for attempt in range(cfg.max_retries + 1):
            results = harness.run_validators(validators, answer.content, options)
            if harness.all_passed(results) or attempt == cfg.max_retries:
                break
            feedback = harness.harness_feedback(results)
            log.append(f"하네스 실패 → 재작성 {attempt + 1}회: " + " / ".join(r.log for r in results if not r.passed))
            emit({"type": "retry", "node": answer.node_id, "reason": "harness"})
            previous = answer
            answer = redo(feedback, previous)
            answer.attempts = previous.attempts + 1
            answer.verification_log = previous.verification_log
        answer.harness = results
        answer.verification_log = [*answer.verification_log, *log]
        if not harness.all_passed(results):
            answer.verification_log.append("하네스 재시도 한도 초과 — 검증 실패 상태로 진행")
        return answer

    def keep_best_attempt(tried: list[AgentAnswer], passed: bool) -> AgentAnswer:
        """자가 교정이 끝난 뒤 내보낼 답변. 마지막 교정본이 불합격이면 비평 점수가 가장 높은 답변을 유지한다.

        교정본이 원본보다 나빠질 수 있다 (실측: 개념 실행에서 4/5 → 교정 후 2/5 인데 2/5 가 나감).
        """
        last = tried[-1]
        if passed:
            if len(tried) > 1 and last.critic:
                last.verification_log.append(f"자가 교정 후 비평가 통과({last.critic.score}/5)")
            return last
        best = tried[verification.best_attempt([a.critic for a in tried])]
        scores = " → ".join(f"{a.critic.score}/5" if a.critic else "-" for a in tried)
        note = f"최종 비평가 여전히 불합격 (비평 점수 {scores})"
        if best is not last:
            note += f" → 교정본이 나아지지 않아 {tried.index(best) + 1}번째 답변을 유지"
        best.verification_log = [*last.verification_log, note]
        best.attempts = last.attempts
        return best

    # ── 노드 ────────────────────────────────────────────────────

    def brief_node(state: OfficeState) -> dict:
        """질문 → 최종 상태 정의(Desired State) 브리프.

        사용자가 목표와 완료 기준을 직접 채웠으면 그대로 쓰고(LLM 호출 없음),
        비었거나 일부만 채웠으면 보조 모델이 빈 항목을 채운다 (auto_brief=False 면 최소 브리프).
        """
        partial = state.get("brief_input") or empty_state()
        if partial.goal.strip() and partial.success_criteria:
            return {"brief": partial, "notes": []}
        if not state.get("auto_brief", True):
            return {"brief": partial.model_copy(update={"goal": partial.goal or state["question"]}), "notes": []}
        emit({"type": "start", "node": state["root_id"], "kind": "brief"})
        brief, notes = draft_brief(state["question"], models, partial)
        return {"brief": brief, "notes": notes}

    def research_node(state: OfficeState) -> dict:
        """자료 조사 — 데이터 툴(주가·여행 달력 등) + 웹 검색(선택) 후 할 일 스택을 준비한다."""
        root = state["root_id"]
        brief = state["brief"]
        data_tools = [t for t in team.tools if t != WEB_SEARCH_TOOL]
        tool_results = []
        if data_tools:
            emit({"type": "start", "node": root, "kind": "tools"})
            tool_results = run_tools(data_tools, state["question"], state.get("tool_inputs") or {}, models)
        web = None
        if state.get("use_web") and WEB_SEARCH_TOOL in team.tools:
            emit({"type": "start", "node": root, "kind": "research"})
            # 검색어는 원 질문 + 목표로 만든다 (브리프 전체는 검색어 생성에 너무 길다). 팀별 검색 설정을 덮어쓴다
            web = search.research(
                f"{state['question']}\n(목표: {brief.goal})", models, {**settings, "web_search": {**settings.get("web_search", {}), **team.search}}
            )
        return {
            "web": web,
            "tool_results": tool_results,
            "agenda": [{"op": "visit", "node": root}],
            "tasks": {root: brief.goal or state["question"]},  # supervisor 의 업무 = 최종 목표
            "assigned": {},
            "results": {},
        }

    def work_node(state: OfficeState) -> dict:
        """할 일 스택에서 하나를 꺼내 처리한다 (LLM 호출 1회 + 필요하면 검증 재시도)."""
        agenda = list(state["agenda"])
        item = agenda.pop()  # 스택: 마지막에 쌓은 것부터 (깊이 우선)
        node_id = item["node"]
        node, agent = nodes[node_id], agents[node_id]
        tasks, assigned, results = dict(state["tasks"]), dict(state["assigned"]), dict(state["results"])
        # 에이전트들은 원 질문 대신 '업무 브리프'(목표·완료 기준·제약·형식)를 받는다
        brief: DesiredState = state["brief"]
        question, web = brief.as_prompt(state["question"]), web_text(state)
        new_steps: list[AgentAnswer] = []

        if item["op"] == "visit" and node.children:
            # ── 위 → 아래: 업무 분배 ──
            if len(node.children) == 1:
                # 하위가 한 명이면 LLM 을 부르지 않고 업무를 그대로 넘긴다 (비용 절약)
                chosen = [(node.children[0].id, tasks[node_id])]
            else:
                emit({"type": "start", "node": node_id, "kind": "plan"})
                plan = agent.plan(question, tasks[node_id], web, lessons_for(node_id))
                valid = {c.id for c in node.children}
                chosen, seen = [], set()
                for a in plan.delegation.assignments:
                    if a.child_id in valid and a.child_id not in seen:
                        chosen.append((a.child_id, a.subtask))
                        seen.add(a.child_id)
                if not chosen:  # 잘못된 id 만 골랐으면 전원에게 같은 업무
                    chosen = [(c.id, tasks[node_id]) for c in node.children]
                new_steps.append(plan)
            assigned[node_id] = [cid for cid, _ in chosen]
            for cid, subtask in chosen:
                tasks[cid] = subtask
            # 화면: 관리자는 '하위 결과 대기', 업무를 받은 하위는 '대기' 로 표시
            emit({"type": "delegated", "node": node_id, "children": assigned[node_id]})
            # 나중에 할 '통합'을 먼저 쌓고, 그 위에 자식 방문을 쌓는다 → 자식들이 먼저 처리된다
            agenda.append({"op": "integrate", "node": node_id})
            agenda += [{"op": "visit", "node": cid} for cid, _ in reversed(chosen)]

        elif item["op"] == "visit":
            # ── 맨 아래: 실무자 초안 (+ 하네스) ──
            emit({"type": "start", "node": node_id, "kind": "draft"})
            messages = agent.draft_messages(question, tasks[node_id], web, lessons_for(node_id))
            tag = DRAFT_TAG + node_id

            def restart() -> None:
                # 모델이 쓰는 도중 끊겨 처음부터 다시 쓸 때 — 화면의 '작성 중' 미리보기를 비운다
                emit({"type": "retry", "node": node_id, "reason": "model"})

            try:
                answer = agent.draft(messages, tag, restart)

                def redo_draft(feedback: str, previous: AgentAnswer) -> AgentAnswer:
                    return agent.draft(with_feedback(messages, previous.content, feedback), tag, restart)

                answer = run_harness(answer, redo_draft, state)
            except NoAvailableModelError as e:
                # 실무자 한 명의 모든 모델이 실패해도 팀 전체를 멈추지 않는다 → 실패 표시를 남기고 관리자가 나머지로 진행
                answer = AgentAnswer(
                    node_id=node_id, rank_id=agent.rank.id, kind="draft",
                    content=f"⚠️ {agent.name}: 모델 오류로 초안을 작성하지 못했습니다. 이 부분은 다른 결과로 보완해야 합니다.",
                    model=agent.rank_models.primary, errors=str(e).splitlines()[1:][:6],
                    verification_log=["모든 후보 모델 호출 실패 → 이 에이전트 없이 진행"],
                )
            results[node_id] = answer
            new_steps.append(answer)

        else:
            # ── 아래 → 위: 검토(1명) / 통합(여러 명) (+ 비평가·자가 교정 + 하네스) ──
            children = [results[cid] for cid in assigned[node_id]]
            kind = "integrate" if len(children) > 1 else "review"
            emit({"type": "start", "node": node_id, "kind": kind})
            messages = agent.review_messages(question, tasks[node_id], children, web, lessons_for(node_id), rewrite_answer)
            answer = agent.review(messages, children, rewrite_answer)

            # 비평가(Critic): 검토 과정의 논리 비약을 채점 → 불합격이면 자가 교정
            use_critic = critic_cfg["enabled"] and (critic_cfg["scope"] == "all" or node_id == state["root_id"])
            if use_critic:
                reviewed_input = "\n\n".join(c.content for c in children)
                # 비평가도 근거 자료를 봐야 '숫자에는 출처·기준일' 같은 규칙을 실제로 대조할 수 있다 (길이는 예산 안으로)
                evidence = verification.trim_evidence(evidence_parts(state), critic_cfg["evidence_chars"])
                tried: list[AgentAnswer] = []  # 비평을 받은 답변들 (원본 → 교정본 순)
                for round_ in range(critic_cfg["max_self_corrections"] + 1):
                    emit({"type": "start", "node": node_id, "kind": "critic"})
                    critic, errs = verification.critique(
                        # 채점 기준 = 팀 검토 기준 + 브리프의 완료 기준
                        question, reviewed_input, answer.review, [*team.review_criteria, *brief.success_criteria], models,
                        critic_cfg.get("rules"), evidence,
                    )
                    answer.critic = critic
                    answer.verification_log += errs
                    tried.append(answer)
                    passed = verification.critic_passed(critic, critic_cfg["min_score"], critic_cfg["fail_on"])
                    if passed or round_ == critic_cfg["max_self_corrections"]:
                        break
                    emit({"type": "retry", "node": node_id, "reason": "critic"})
                    previous = answer
                    answer = agent.review(
                        with_feedback(messages, previous.review.model_dump_json(), verification.correction_message(critic)),
                        children,
                        rewrite_answer,
                    )
                    answer.attempts = previous.attempts + 1
                    answer.verification_log = [
                        *previous.verification_log,
                        f"비평가 불합격({critic.score}/5) → 자가 교정 {round_ + 1}회",
                    ]
                answer = keep_best_attempt(tried, passed)

            def redo_review(feedback: str, previous: AgentAnswer) -> AgentAnswer:
                return agent.review(with_feedback(messages, previous.review.model_dump_json(), feedback), children, rewrite_answer)

            answer = run_harness(answer, redo_review, state)
            results[node_id] = answer
            new_steps.append(answer)

        return {"agenda": agenda, "tasks": tasks, "assigned": assigned, "results": results, "steps": new_steps}

    def human_decision_node(state: OfficeState) -> dict:
        # 여기서 그래프가 멈추고, UI 가 Command(resume=HumanDecision 값)으로 재개한다
        value = interrupt({"type": "human_decision", "root": state["root_id"]})
        return {"decision": HumanDecision.model_validate(value)}

    def feedback_node(state: OfficeState, config: RunnableConfig) -> dict:
        lessons, notes = generate_lessons(
            state["question"], state["steps"], state["decision"], names, parents, team, models
        )
        if storage:
            run_id = config.get("configurable", {}).get("thread_id")
            storage.add_lessons(team.id, run_id, lessons)
        return {"lessons": lessons, "notes": notes}

    def after_work(state: OfficeState) -> Literal["work", "human_decision", "__end__"]:
        if state["agenda"]:
            return "work"
        return "human_decision" if with_human else END

    graph = StateGraph(OfficeState)
    graph.add_node("brief", brief_node)
    graph.add_node("research", research_node)
    graph.add_node("work", work_node)
    graph.add_edge(START, "brief")
    graph.add_edge("brief", "research")
    graph.add_edge("research", "work")
    targets = ["work", "human_decision", END] if with_human else ["work", END]
    graph.add_conditional_edges("work", after_work, targets)
    if with_human:
        graph.add_node("human_decision", human_decision_node)
        graph.add_node("feedback", feedback_node)
        graph.add_edge("human_decision", "feedback")
        graph.add_edge("feedback", END)
    return graph.compile(checkpointer=checkpointer)


# ─── 실행·이벤트 변환 (UI 가 쓰는 함수) ─────────────────────────────


def _config(org: OrgNode, thread_id: str | None) -> dict:
    # 단계 수 상한: 에이전트마다 visit + integrate 2단계 + 여유
    config = {"recursion_limit": len(org.walk()) * 2 + 10}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return config


def _events(graph, inputs, config) -> Iterator[tuple[str, object]]:
    """LangGraph 스트림을 화면에서 쓰기 쉬운 이벤트로 바꾼다."""
    for mode, data in graph.stream(inputs, config, stream_mode=["custom", "messages", "updates"]):
        if mode == "custom":  # emit() 으로 보낸 '시작/재시도' 알림
            yield data["type"], data
            continue
        if mode == "messages":
            chunk, metadata = data
            # 초안 호출에만 DRAFT_TAG 를 붙였으므로, 검토·분배(JSON) 토큰은 여기서 걸러진다
            tag = next((t for t in metadata.get("tags", []) if t.startswith(DRAFT_TAG)), None)
            if tag:
                text = chunk_text(chunk.content)
                if text:
                    yield "token", (tag[len(DRAFT_TAG):], text)
            continue
        for node, update in data.items():
            if node == "__interrupt__":
                yield "interrupt", update
            elif node == "brief":
                yield "brief", update.get("brief")
                for note in update.get("notes", []):
                    yield "note", note
            elif node == "research":
                if update.get("tool_results"):
                    yield "tools", update["tool_results"]
                yield "research", update.get("web")
            elif node == "feedback":
                yield "lessons", update["lessons"]
                for note in update.get("notes", []):
                    yield "note", note
            elif isinstance(update, dict):
                for step in update.get("steps", []):
                    yield "step", step


def stream_office(
    graph,
    org: OrgNode,
    question: str,
    root_id: str,
    thread_id: str | None = None,
    use_web: bool = False,
    brief: DesiredState | None = None,
    auto_brief: bool = True,
    tool_inputs: dict | None = None,
    extra_validators: list[str] | None = None,
    harness_options: dict | None = None,
) -> Iterator[tuple[str, object]]:
    """질문 → 브리프 → (데이터 툴·웹 검색) → 분배·초안·검토·통합 진행 이벤트.

    Args:
        brief:            사용자가 화면에서 채운 최종 상태 정의 (없거나 일부만 있어도 됨)
        auto_brief:       빈 항목을 보조 모델이 채울지
        tool_inputs:      팀 업무 양식 입력값 (데이터 툴이 사용)
        extra_validators: 업무 양식이 추가로 요구하는 하네스 검증기
        harness_options:  하네스 옵션 (예: required_terms)

    Yields:
        ("start",   {"node", "kind"})  — 어떤 에이전트가 어떤 일을 시작함 (brief/tools/research/plan/draft/review/integrate/critic)
        ("retry",   {"node", "reason"}) — 자가 교정(critic) 또는 하네스(harness) 실패로 다시 씀
        ("delegated", {"node", "children"}) — 관리자가 하위에게 업무를 넘김 (하위 결과를 기다리는 중)
        ("brief",   DesiredState)       — 정리된 업무 브리프
        ("tools",   list[ToolResult])   — 데이터 툴 결과 (주가·여행 달력 등)
        ("research", WebContext | None) — 웹 검색 결과
        ("token",   (에이전트 id, 텍스트)) — 실무자 초안 토큰
        ("step",    AgentAnswer)        — 한 단계 완료
        ("interrupt", ...)              — 사람의 결정을 기다리며 멈춤 (checkpointer 사용 시)
    """
    subtree(org, root_id)  # 없는 id 면 KeyError
    inputs = {
        "question": question,
        "root_id": root_id,
        "brief_input": brief,
        "auto_brief": auto_brief,
        "use_web": use_web,
        "web": None,
        "tool_inputs": tool_inputs or {},
        "tool_results": [],
        "extra_validators": extra_validators or [],
        "harness_options": harness_options or {},
        "steps": [],
    }
    yield from _events(graph, inputs, _config(org, thread_id))


def resume_with_decision(graph, org: OrgNode, decision: HumanDecision, thread_id: str) -> Iterator[tuple[str, object]]:
    """사람의 결정으로 멈춘 그래프를 재개한다.

    Yields:
        ("lessons", list[Lesson]) — 저장된 에이전트별 교훈
        ("note", str)             — 피드백 생성 중 경고 (fallback 사용 등)
    """
    yield from _events(graph, Command(resume=decision.model_dump()), _config(org, thread_id))
