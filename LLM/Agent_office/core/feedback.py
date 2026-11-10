"""사람의 최종 결정 → 에이전트별 교훈(Lesson) 생성.

1) 보조(utility) 모델이 결정 내용과 단계별 결과를 보고, 참여한 에이전트마다 한 문장의 교훈을 만든다.
2) 모델이 빠뜨린 에이전트가 있거나 호출이 모두 실패하면 규칙 기반 교훈으로 채운다.
3) 사람의 코멘트는 채택된 에이전트(반려면 전원)의 교훈에 원문 그대로 붙인다 (모델이 빠뜨려도 보장).
→ 어떤 경우에도 참여한 모든 에이전트가 피드백을 받고, 다음 실행 때 시스템 프롬프트에 주입된다.

프롬프트에 넣는 교훈 (team_lessons)
  교훈 = 규칙 문장(짧은 일반 지침) + 세부 코멘트(끝에 붙은 '(사람 코멘트: …)' 원문 — 길다).
  글자 한도(settings.yaml feedback.lesson_chars_in_prompt) 안에서 규칙 문장은 모두 넣고, 세부 코멘트는 이번 질문과 관련 높은 것 →
  반복 많은 것 → 최신 순으로 붙인다. 같은 피드백을 여러 번 받은 교훈은 '[중요 — 피드백 N회 반복]'을 달아 맨 앞에 둔다.
  빠진 것(세부 코멘트만 · 통째로)은 화면(📜 프롬프트·📚 교훈)이 이유와 함께 알린다 — 조용히 버리지 않는다.
  실행(core/workflow.py)과 화면(prompt_ui.py)이 같은 함수(team_lessons · team_prompts)를 쓴다 (화면은 질문이 없어 반복·최신 순).

결정 직후 정리 (propose_cleanup → absorb_duplicates)
  새 교훈과 겹치거나 충돌하는 옛 교훈을 보조 모델이 찾고(실패하면 문장 비교), 코드가 번호·에이전트를 확인한다.
  겹침 → 바로 1개로 합친다 (반복 횟수 합, 사람 코멘트 원문은 그대로, 원래 교훈은 끔 — 화면에서 되돌리기 가능).
  충돌 → 바꾸지 않고, 교훈마다 원래 피드백을 보여 주며 사용자가 남길 쪽을 고른다 (apply_cleanup).
  📚 교훈 관리의 '🧹 정리 제안 받기'(쌓인 교훈)는 예전처럼 사용자가 고른 것만 반영한다.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core import llm_factory
from core.agent import OrgAgent
from core.config import load_skills, team_org
from core.feedback_sync import COMMENT, public_lesson
from core.llm_factory import NoAvailableModelError
from core.schemas import (
    AgentAnswer, CleanupProposal, HumanDecision, Lesson, LessonGroup, LessonGroupSet, LessonSet, ModelsConfig, Rank, Team,
)

ANSWER_PREVIEW_CHARS = 600
KIND_LABELS = {"plan": "업무 분배", "draft": "초안 작성", "review": "검토", "integrate": "통합"}


def participants(steps: list[AgentAnswer]) -> list[str]:
    """이번 실행에 참여한 에이전트 id (처음 등장한 순서, 중복 제거)."""
    seen: list[str] = []
    for step in steps:
        if step.node_id not in seen:
            seen.append(step.node_id)
    return seen


def final_steps(steps: list[AgentAnswer]) -> dict[str, AgentAnswer]:
    """에이전트별 마지막(=위로 올린) 단계. 관리자는 분배(plan) 뒤 검토·통합이 마지막이다."""
    return {s.node_id: s for s in steps}


def _step_summary(i: int, step: AgentAnswer, names: dict[str, str]) -> str:
    head = f"{i + 1}. {names.get(step.node_id, step.node_id)}({step.node_id}) — {KIND_LABELS[step.kind]}"
    if step.review:
        r = step.review
        verdict = "승인" if r.verdict == "approve" else "보완/통합"
        head += f": {verdict}, 하위 결과 {r.score}/10점, 코멘트: {r.comment}"
        if r.issues:
            head += "\n   지적: " + " / ".join(r.issues)
    preview = step.content[:ANSWER_PREVIEW_CHARS] + ("…" if len(step.content) > ANSWER_PREVIEW_CHARS else "")
    return f"{head}\n   결과: {preview}"


def build_feedback_messages(
    question: str, steps: list[AgentAnswer], decision: HumanDecision, names: dict[str, str], team: Team
) -> list[BaseMessage]:
    system = "\n".join(
        [
            f"당신은 '{team.name}' 팀의 업무 코치다. 사람이 내린 최종 결정을 바탕으로,",
            "참여한 각 에이전트가 **다음 업무에서 바로 적용할 교훈**을 만든다.",
            "",
            "규칙:",
            "- 참여한 에이전트마다 정확히 1개의 교훈을 만든다 (agent_id 는 괄호 안의 id 를 그대로 사용).",
            "- 이번 질문 내용에만 해당하는 말이 아니라, 같은 팀의 다른 업무에도 통하는 행동 지침 한 문장으로 쓴다.",
            "- 잘해서 유지할 점이면 kind=praise, 고칠 점이면 kind=improve.",
            "- 사람의 코멘트와 수정 내용을 가장 우선해서 반영한다.",
            "- 사람이 채택안을 직접 수정했거나 코멘트에 지적이 있으면, 채택된 에이전트의 교훈은 kind=improve 로 쓰고",
            "  그 지적(빠진 내용, 고친 방향)을 구체적으로 담는다. 칭찬만 하고 지적을 빼지 않는다.",
            "- 채택된 안보다 위에서 답변을 바꾼 관리자는 과도한 수정이 없었는지 짚는다.",
            "- 한국어로 쓴다.",
        ]
    )
    if decision.action == "adopt":
        adopted = names.get(decision.adopted_node_id or "", decision.adopted_node_id)
        decision_text = f"채택 — {adopted}({decision.adopted_node_id}) 의 안을 최종안으로 채택"
        if decision.edited:
            decision_text += f"\n사람이 직접 수정한 최종 답변:\n{decision.final_answer[:ANSWER_PREVIEW_CHARS * 2]}"
    else:
        decision_text = "반려 — 어느 안도 채택하지 않음"
    human = "\n\n".join(
        [
            f"[질문]\n{question}",
            "[단계별 결과]\n" + "\n".join(_step_summary(i, s, names) for i, s in enumerate(steps)),
            f"[사람의 결정]\n{decision_text}\n만족도: {decision.score}/5\n코멘트: {decision.comment or '(없음)'}",
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


def rule_based_lessons(
    steps: list[AgentAnswer], decision: HumanDecision, names: dict[str, str], parents: dict[str, str]
) -> list[Lesson]:
    """LLM 없이 만드는 기본 교훈 (LLM 실패·누락 시 사용).

    채택: 채택된 에이전트 → 칭찬(수정됐으면 개선) / 채택안 위의 관리자 → 과도한 수정 주의 /
          그 외 → 내 결과를 받은 상위 관리자의 판정(승인·보완)에 따라 칭찬·개선
    반려: 전원 개선
    """
    comment = f" 사람 코멘트: {decision.comment}" if decision.comment else ""
    ids = participants(steps)
    if decision.action == "reject":
        return [
            Lesson(agent_id=nid, kind="improve", lesson=f"최종 결과가 반려되었다. 질문의 의도와 요구 수준을 먼저 확인하고 일할 것.{comment}")
            for nid in ids
        ]

    adopted = decision.adopted_node_id if decision.adopted_node_id in ids else ids[0]
    adopted_name = names.get(adopted, adopted)
    ancestors = set()  # 채택안 위에서 그 결과를 다시 고친 관리자들
    cur = adopted
    while cur in parents:
        cur = parents[cur]
        ancestors.add(cur)
    last = final_steps(steps)

    lessons = []
    for nid in ids:
        if nid == adopted:
            if decision.edited:
                lessons.append(Lesson(agent_id=nid, kind="improve", lesson=f"작성한 안이 채택되었지만 사람이 직접 수정했다. 수정된 방향을 참고할 것.{comment}"))
            else:
                lessons.append(Lesson(agent_id=nid, kind="praise", lesson=f"작성한 안이 최종 채택되었다(만족도 {decision.score}/5). 이 수준과 방식을 유지할 것."))
            continue
        if nid in ancestors:
            lessons.append(Lesson(agent_id=nid, kind="improve", lesson=f"사람은 하위 {adopted_name}의 안을 더 선호했다. 과도한 수정을 피하고 꼭 필요한 보완만 할 것.{comment}"))
            continue
        parent_step = last.get(parents.get(nid, ""))
        if parent_step and parent_step.review and parent_step.review.verdict == "revise":
            issues = parent_step.review.issues[:3]
            text = f"상위 {names.get(parent_step.node_id, parent_step.node_id)}의 검토에서 보완된 점을 처음부터 반영할 것"
            text += f": {' / '.join(issues)}." if issues else "."
            lessons.append(Lesson(agent_id=nid, kind="improve", lesson=text))
        else:
            lessons.append(Lesson(agent_id=nid, kind="praise", lesson="결과가 상위 검토에서 그대로 승인되었다. 이 방식을 유지할 것."))
    return lessons


def generate_lessons(
    question: str,
    steps: list[AgentAnswer],
    decision: HumanDecision,
    names: dict[str, str],
    parents: dict[str, str],
    team: Team,
    models: ModelsConfig,
) -> tuple[list[Lesson], list[str]]:
    """에이전트별 교훈 생성. Returns: (참여 순서대로 정렬된 교훈 목록, 경고/에러 메시지)"""
    ids = participants(steps)
    fallback = {l.agent_id: l for l in rule_based_lessons(steps, decision, names, parents)}
    notes: list[str] = []
    generated: dict[str, Lesson] = {}
    try:
        call = llm_factory.invoke_structured(
            models.utility, models, LessonSet, build_feedback_messages(question, steps, decision, names, team), "피드백 생성"
        )
        notes += call.errors
        for lesson in call.parsed.lessons:
            if lesson.agent_id in ids and lesson.agent_id not in generated and lesson.lesson.strip():
                generated[lesson.agent_id] = lesson
    except NoAvailableModelError as e:
        notes.append(f"피드백 모델 호출 실패 → 규칙 기반 교훈 사용: {e}")

    missing = [nid for nid in ids if nid not in generated]
    if missing and generated:
        notes.append(f"모델이 빠뜨린 에이전트는 규칙 기반 교훈 사용: {missing}")
    lessons = [generated.get(nid) or fallback[nid] for nid in ids]
    return [_with_human_comment(l, decision) for l in lessons], notes


def _with_human_comment(lesson: Lesson, decision: HumanDecision) -> Lesson:
    """사람의 코멘트가 누락·왜곡되지 않도록 원문을 붙인다.

    대상: 반려면 모든 에이전트, 채택이면 채택된 에이전트 (모델이 코멘트를 빠뜨리는 경우가 있어 코드로 보장).
    """
    comment = decision.comment.strip()
    targeted = decision.action == "reject" or lesson.agent_id == decision.adopted_node_id
    if not comment or not targeted or comment in lesson.lesson:
        return lesson
    kind = "improve" if decision.edited or decision.action == "reject" else lesson.kind
    return lesson.model_copy(update={"kind": kind, "lesson": f"{lesson.lesson} (사람 코멘트: {comment})"})


# ─── 프롬프트에 넣을 교훈 (글자 한도) ───────────────────────────────────

DEFAULT_LESSON_CHARS = 20000


def lesson_budget(settings: dict) -> int:
    """에이전트 한 명의 프롬프트에 넣을 교훈 글자 수 한도 (settings.yaml feedback.lesson_chars_in_prompt)."""
    return int((settings.get("feedback") or {}).get("lesson_chars_in_prompt", DEFAULT_LESSON_CHARS))


def split_lesson(text: str) -> tuple[str, str]:
    """교훈 = (규칙 문장, 세부 코멘트). 세부 코멘트는 끝에 붙은 '(사람 코멘트: …)' 원문 — 없으면 빈 문자열.
    규칙 문장은 짧은 일반 지침(개념 편집 팀 실측 평균 약 100자), 세부 코멘트는 그 업무의 구체 지적이라 길다(2천~6천 자)."""
    m = COMMENT.search(text)
    return (text, "") if not m else (text[:m.start()].rstrip(), text[m.start():])


def comment_body(detail: str) -> str:
    """세부 코멘트 ' (사람 코멘트: …)' 에서 괄호·머리말을 뗀 본문."""
    body = detail.strip()
    closed = body.startswith("(")
    body = body.lstrip("(").removeprefix("사람 코멘트:").strip()
    return body[:-1].rstrip() if closed and body.endswith(")") else body


def emphasis(row: dict) -> str:
    """같은 지시를 2번 이상 받은 교훈은 앞에 강조 표시를 단다 (겹치는 교훈을 합칠 때 횟수를 더함)."""
    hits = row.get("hits", 1)
    return f"[중요 — 피드백 {hits}회 반복] " if hits > 1 else ""


def relevance(query: str, text: str) -> float:
    """질문과 교훈의 관련도 (0~1) — 공백 뺀 글자 2개 묶음의 겹침(코사인). 임베딩 모델 없이 코드로 계산한다."""
    q, t = ({s[i:i + 2] for i in range(len(s) - 1)} for s in (re.sub(r"\s+", "", x.lower()) for x in (query, text)))
    return len(q & t) / ((len(q) * len(t)) ** 0.5) if q and t else 0.0


@dataclass
class PromptLessons:
    """한 에이전트의 프롬프트에 들어갈 교훈."""

    kept: list[dict]  # 들어가는 교훈 (프롬프트 순서: 반복 많은 것 먼저, 같으면 오래된 것 → 최신). r["prompt"] = 실제로 넣는 글
    left_out: list[dict]  # 규칙 문장까지 글자 한도를 넘어 통째로 빠진 교훈 (최신 → 오래된)
    budget: int

    @property
    def texts(self) -> list[str]:
        return [r["prompt"] for r in self.kept]

    @property
    def chars(self) -> int:
        return sum(map(len, self.texts))

    @property
    def cut(self) -> list[dict]:
        """규칙 문장은 들어가고 세부 코멘트가 다 들어가지 못한 교훈 (r["partial"] = 들어간 앞부분 글자 수, 0 이면 통째로 빠짐)."""
        return [r for r in self.kept if r["cut"]]


MIN_PARTIAL_CHARS = 200  # 남은 자리가 이보다 작으면 세부 코멘트를 잘라 넣지 않는다 (몇 낱말만 들어가면 뜻이 통하지 않음 — 추정값)
CUT_NOTE = " …(뒤는 글자 한도로 생략)"  # 잘라 넣은 세부 코멘트 끝에 붙여 모델이 '여기서 끊겼다'를 알게 한다


def fit_lessons(rows: list[dict], budget: int, query: str = "") -> PromptLessons:
    """rows = 켜 둔 교훈 (최신 먼저). 두 단계로 글자 한도(budget) 안을 채운다.

    1) 규칙 문장(+반복 강조 표시)을 최신부터 모두 넣는다 — 짧아서 거의 항상 다 들어간다.
       넘치는 교훈은 건너뛰고 계속 채운다 (긴 교훈 하나 때문에 짧은 옛 교훈까지 빠지지 않게).
    2) 남은 글자 안에서 세부 코멘트를 붙인다. 순서: 이번 질문(query)과 관련 높은 것 → 반복 많은 것 → 최신.
       통째로 들어가는 것을 먼저 붙이고, 그래도 남은 자리에는 못 붙인 세부 코멘트의 앞부분을 같은 순서로 잘라 넣는다
       (합친 교훈은 코멘트 여러 개가 한 덩어리라 '전부 아니면 빼기'만 하면 통째로 사라진다 — 2026-11-10 점검).
       다 못 붙인 세부 코멘트는 cut 으로 알린다 (조용히 버리지 않는다).
    """
    kept, left_out, used = [], [], 0
    for row in rows:
        rule, detail = split_lesson(row["lesson"])
        head = emphasis(row) + rule
        if used + len(head) <= budget:
            kept.append({**row, "prompt": head, "cut": bool(detail), "detail": detail, "partial": 0})
            used += len(head)
        else:
            left_out.append(row)
    order = sorted((r for r in kept if r["detail"]),
                   key=lambda r: (relevance(query, r["lesson"]) if query else 0.0, r.get("hits", 1), r["id"]), reverse=True)
    for r in order:
        if used + len(r["detail"]) <= budget:
            r["prompt"] += r["detail"]
            r["cut"] = False
            used += len(r["detail"])
    for r in order:
        if r["cut"] and budget - used >= MIN_PARTIAL_CHARS:
            tail = CUT_NOTE + (")" if r["detail"].rstrip().endswith(")") else "")
            r["partial"] = budget - used - len(tail)
            r["prompt"] += r["detail"][:r["partial"]] + tail
            used = budget
    kept.sort(key=lambda r: (-r.get("hits", 1), r["id"]))
    return PromptLessons(kept, left_out, budget)


def active_lessons(storage, team_id: str) -> dict[str, list[dict]]:
    """{에이전트 id: 켜 둔 교훈 (최신 먼저)} — 저장소를 한 번만 읽는다 (나뉜 팀은 옛 팀 교훈도 함께)."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in storage.list_lessons(team_id):  # 최신 먼저
        if row["active"]:
            by_agent[row["agent_id"]].append(row)
    return dict(by_agent)


def team_lessons(storage, team_id: str, budget: int, query: str = "") -> dict[str, PromptLessons]:
    """팀 에이전트별 프롬프트 교훈 — 실행과 화면이 같이 쓴다. 교훈이 없는 에이전트는 빠져 있다.
    query: 이번 질문 — 한도를 넘을 때 관련 높은 세부 코멘트부터 넣는다 (화면은 질문이 없어 반복·최신 순)."""
    return {agent: fit_lessons(rows, budget, query) for agent, rows in active_lessons(storage, team_id).items()}


@dataclass
class AgentPrompt:
    """팀원 한 명이 실행 때 받는 시스템 프롬프트 (화면 표시용)."""

    node_id: str
    name: str
    rank: str
    specialty: str
    prompt: str
    lessons: PromptLessons


def team_prompts(team: Team, ranks: list[Rank], models: ModelsConfig, storage, budget: int) -> list[AgentPrompt]:
    """팀원마다 실행 때 받는 시스템 프롬프트 — 실행(core/workflow.py)과 같은 스킬·OrgAgent.system_prompt·team_lessons 로 만든다."""
    ranks_by_id = {r.id: r for r in ranks}
    nodes = list(team_org(team, ranks).walk())
    names = {n.id: n.name or ranks_by_id[n.rank].name for n in nodes}
    skills = load_skills(team.id)
    picks = team_lessons(storage, team.id, budget) if storage else {}
    out = []
    for n in nodes:
        rank = ranks_by_id[n.rank]
        agent = OrgAgent(n, rank, team, models, names=names, skills=skills)
        pick = picks.get(n.id) or PromptLessons([], [], budget)
        out.append(AgentPrompt(n.id, agent.name, rank.name, n.specialty or rank.role, agent.system_prompt(pick.texts), pick))
    return out


def orphan_lessons(storage, team: Team, ranks: list[Rank]) -> dict[str, list[dict]]:
    """지금 조직도에 없는 팀원 id 로 남은 켜 둔 교훈 {옛 id: 교훈들} — 어느 프롬프트에도 들어가지 않는다.
    (예: 일반 팀은 조직도가 생기기 전 직급 id(team_lead·gm…)로 교훈이 쌓였다. 옮길지는 사용자가 화면에서 고른다)"""
    ids = {n.id for n in team_org(team, ranks).walk()}
    return {agent: rows for agent, rows in active_lessons(storage, team.id).items() if agent not in ids}


def suggest_owner(old_id: str, team: Team, ranks: list[Rank]) -> str:
    """옛 id 의 교훈을 받을 팀원 제안 — 같은 직급의 첫 팀원 (옛 id 는 대개 직급 id), 없으면 팀장."""
    org = team_org(team, ranks)
    return next((n.id for n in org.walk() if n.rank == old_id), org.id)


# ─── 교훈 정리 제안 (겹침·충돌) — 사용자가 고르기 전에는 아무것도 바꾸지 않는다 ─────────────

SIMILAR = 0.6  # 문장 비교(코드)로 '거의 같은 교훈'으로 볼 기준 — 글자 2개 묶음이 이만큼 겹치면
LESSON_PREVIEW_CHARS = 4000  # 정리 모델에 보낼 교훈 한 개의 최대 글자 수


def similarity(a: str, b: str) -> float:
    """두 교훈 문장(사람 코멘트 뺀 부분)의 글자 2개 묶음 겹침 비율 (0~1)."""
    grams = [{t[i:i + 2] for i in range(len(t) - 1)} for t in (re.sub(r"\s+", "", public_lesson(x)) for x in (a, b))]
    union = grams[0] | grams[1]
    return len(grams[0] & grams[1]) / len(union) if union else 0.0


def similar_groups(targets: dict[str, list[dict]], new_ids: dict[str, int]) -> list[LessonGroup]:
    """모델 없이 — 문장이 거의 같은 교훈끼리 묶는다. 새 교훈이 있으면 그것과 비슷한 옛 교훈만."""
    groups = []
    for agent, rows in targets.items():
        heads = [r for r in rows if r["id"] == new_ids[agent]] if agent in new_ids else rows
        used: set[int] = set()
        for head in heads:  # 최신 먼저 → 합친 문장은 묶음의 가장 최근 교훈
            if head["id"] in used:
                continue
            same = [r["id"] for r in rows if r["id"] != head["id"] and r["id"] not in used
                    and similarity(head["lesson"], r["lesson"]) >= SIMILAR]
            if same:
                used |= {head["id"], *same}
                groups.append(LessonGroup(agent_id=agent, ids=[head["id"], *same], relation="overlap", merged=head["lesson"],
                                          detail="", reason="문장이 거의 같습니다 (문장 비교)"))
    return groups


def cleanup_messages(team: Team, targets: dict[str, list[dict]], names: dict[str, str], new_ids: dict[str, int]) -> list[BaseMessage]:
    system = "\n".join([
        f"당신은 '{team.name}' 팀의 교훈 정리 담당이다. 교훈은 지난 업무에서 사람이 준 피드백으로, 에이전트의 프롬프트에 들어간다.",
        "에이전트마다 아래 둘을 찾아 묶고, 묶음을 대신할 문장을 제안한다.",
        "- overlap: 같은 지시를 되풀이하는 교훈들",
        "- conflict: 서로 반대되는 지시를 하는 교훈들",
        "",
        "규칙:",
        "- 같은 에이전트의 교훈끼리만 묶는다 (agent_id 와 번호를 그대로 쓴다). 한 교훈은 한 묶음에만 넣는다.",
        "- 겹치거나 충돌하지 않는 교훈은 묶지 않는다. 묶을 것이 없으면 groups 를 비운다.",
        "- [새 교훈] 표시가 있는 에이전트는 그 새 교훈이 들어간 묶음만 만든다.",
        "- merged: 묶음의 지시를 빠짐없이 담은 일반 행동 지침 한두 문장. 충돌이면 더 최근 교훈을 따른다.",
        "  사람 코멘트 원문은 코드가 그대로 붙이므로 merged 에 옮겨 적지 않는다.",
        "- detail: 교훈에 사람 코멘트가 없을 때만, 꼭 남길 구체 규칙(형식·기호·예시 등)을 짧게. 그 밖에는 빈 문자열.",
        "- 같은 방향의 지시를 다른 말로 되풀이하면 overlap, 한쪽을 따르면 다른 쪽을 어기게 되면 conflict 다. 애매하면 conflict 로 둔다.",
        "- reason: 왜 묶었는지 한 문장. 한국어로 쓴다.",
    ])
    blocks = []
    for agent, rows in targets.items():
        lines = [f"## {names.get(agent, agent)}({agent})"]
        for r in reversed(rows):  # 오래된 것 → 최신
            mark = " [새 교훈]" if new_ids.get(agent) == r["id"] else ""
            text = r["lesson"] if len(r["lesson"]) <= LESSON_PREVIEW_CHARS else r["lesson"][:LESSON_PREVIEW_CHARS] + "…"
            lines.append(f"[번호 {r['id']}]{mark} ({r['created_at'][:10]}, {'유지할 점' if r['kind'] == 'praise' else '고칠 점'}) {text}")
        blocks.append("\n".join(lines))
    return [SystemMessage(system), HumanMessage("\n\n".join(blocks))]


def _merged_text(merged: str, detail: str, rows: list[dict]) -> str:
    """합친 문장 = 모델이 낸 규칙 문장 + 원래 교훈들의 사람 코멘트 원문 (모델이 요약하다 빠뜨리지 않게 코드가 그대로 붙인다).
    원래 교훈에 코멘트가 없을 때만 모델이 고른 detail 을 붙인다. '(사람 코멘트: …)' 부분은 공개 knowledge/ 로 나가지 않는다.
    코멘트는 최신 교훈 것부터 붙인다 — 프롬프트 글자 한도로 뒤가 잘려도(fit_lessons) 최신 피드백이 남게."""
    comments = list(dict.fromkeys(b for b in (comment_body(split_lesson(r["lesson"])[1]) for r in reversed(rows)) if b))
    tail = " / ".join(comments) or detail.strip()
    return split_lesson(merged.strip())[0] + (f" (사람 코멘트: {tail})" if tail else "")


def _proposals(team_id: str, groups: list[LessonGroup], targets: dict[str, list[dict]], new_ids: dict[str, int],
               by: str) -> list[CleanupProposal]:
    """모델이 낸 묶음을 코드로 확인 — 그 에이전트의 켜 둔 교훈 번호만, 2개 이상, 한 교훈은 한 묶음, 새 교훈이 있으면 그것이 낀 묶음만."""
    out, used = [], set()
    for g in groups:
        rows = {r["id"]: r for r in targets.get(g.agent_id, [])}
        ids = [i for i in dict.fromkeys(g.ids) if i in rows]
        new = new_ids.get(g.agent_id)
        if len(ids) < 2 or used & set(ids) or (new_ids and new not in ids) or not split_lesson(g.merged.strip())[0]:
            continue
        used |= set(ids)
        picked = [rows[i] for i in sorted(ids)]
        out.append(CleanupProposal(
            team_id=team_id, agent_id=g.agent_id, ids=sorted(ids), new_id=new, relation=g.relation,
            kind="improve" if any(r["kind"] == "improve" for r in picked) else "praise",
            merged=_merged_text(g.merged, g.detail, picked), reason=g.reason.strip(), by=by,
            hits=sum(r.get("hits", 1) for r in picked),
        ))
    return out


def propose_cleanup(
    team: Team, lessons: dict[str, list[dict]], names: dict[str, str], models: ModelsConfig, new_ids: dict[str, int] | None = None
) -> tuple[list[CleanupProposal], list[str]]:
    """교훈 정리 제안 — 저장소는 바꾸지 않는다.

    lessons: {에이전트 id: 켜 둔 교훈 (최신 먼저)} (active_lessons)
    new_ids: {에이전트 id: 이번 결정에서 생긴 교훈 번호} — 주면 그 교훈과 겹치거나 충돌하는 옛 교훈만 찾는다.
    교훈이 2개 이상인 에이전트만 보조 모델에 한 번에 묻는다. 모델을 못 부르면 문장이 거의 같은 것만 코드로 찾는다.
    """
    new_ids = new_ids or {}
    targets = {a: rows for a, rows in lessons.items() if len(rows) >= 2 and (not new_ids or a in new_ids)}
    if not targets:
        return [], []
    notes: list[str] = []
    try:
        call = llm_factory.invoke_structured(
            models.utility, models, LessonGroupSet, cleanup_messages(team, targets, names, new_ids), "교훈 정리 제안"
        )
        notes += call.errors
        return _proposals(team.id, call.parsed.groups, targets, new_ids, "model"), notes
    except NoAvailableModelError as e:
        notes.append(f"교훈 정리 제안 모델을 부르지 못해 문장이 거의 같은 교훈만 찾았습니다: {e}")
        return _proposals(team.id, similar_groups(targets, new_ids), targets, new_ids, "code"), notes


def merge_lessons(storage, proposal: CleanupProposal, text: str = "") -> int:
    """묶인 교훈을 합친 교훈 1개로 — 새 교훈을 넣고(반복 횟수는 합) 원래 교훈은 지우지 않고 끈다. 새 교훈 번호를 돌려준다."""
    text = (text or proposal.merged).strip()
    new_id = storage.add_lessons(proposal.team_id, None, [Lesson(agent_id=proposal.agent_id, kind=proposal.kind, lesson=text)],
                                 hits=proposal.hits)[0]
    for i in proposal.ids:
        storage.update_lesson(i, active=False)
    return new_id


def apply_cleanup(storage, proposal: CleanupProposal, choice: str, text: str = "", keep: int | None = None) -> str:
    """사용자가 화면에서 고른 정리를 반영한다 (버튼을 누를 때만 부른다). 지우지 않고 끈다(active=0). 화면에 보일 한 줄을 돌려준다.

    keep    : 아무것도 바꾸지 않음
    merge   : 합친 문장(text — 사용자가 고친 것, 없으면 제안 그대로)을 새 교훈으로 넣고 묶인 교훈을 모두 끔
    off_old : 새 교훈은 두고 묶인 옛 교훈만 끔 (새 교훈이 없는 정리 제안에서는 쓰지 않음)
    only    : 충돌하는 교훈 중 keep 번호 하나만 남기고 나머지를 끔
    """
    if choice == "keep":
        return "그대로 두었습니다."
    if choice in ("off_old", "only"):
        stay = proposal.new_id if choice == "off_old" else keep
        off = [i for i in proposal.ids if i != stay]
        for i in off:
            storage.update_lesson(i, active=False)
        return (f"교훈 #{stay}만 남기고 {len(off)}개를 껐습니다. 지운 것이 아니라서 📚 교훈 관리에서 다시 켤 수 있습니다."
                if choice == "only" else f"옛 교훈 {len(off)}개를 껐습니다. 지운 것이 아니라서 📚 교훈 관리에서 다시 켤 수 있습니다.")
    if choice == "merge":
        if not (text or proposal.merged).strip():
            return "합친 문장이 비어 있어 바꾸지 않았습니다. 문장을 적고 다시 눌러 주세요."
        merge_lessons(storage, proposal, text)
        return f"교훈 {len(proposal.ids)}개를 1개로 합쳤습니다. 원래 교훈은 꺼 두었습니다 (📚 교훈 관리에서 다시 켤 수 있음)."
    raise ValueError(f"모르는 선택: {choice}")


# ─── 결정 직후: 겹치면 자동으로 합쳐 강조 · 충돌은 사용자에게 ─────────────────────


def absorb_duplicates(storage, proposals: list[CleanupProposal]) -> tuple[list[CleanupProposal], list[CleanupProposal]]:
    """결정 직후 정리 제안 중 겹침(overlap)은 바로 합친다 — 같은 지시를 또 받았다는 뜻이라 반복 횟수를 더해 강조한다.
    충돌(conflict)은 어느 쪽이 맞는지 사람만 알므로 바꾸지 않고 돌려준다.
    Returns: (자동으로 합친 것 — merged_id 가 채워짐, 사용자가 고를 충돌)"""
    merged, conflicts = [], []
    for p in proposals:
        if p.relation == "overlap":
            merged.append(p.model_copy(update={"merged_id": merge_lessons(storage, p)}))
        else:
            conflicts.append(p)
    return merged, conflicts


def undo_merge(storage, proposal: CleanupProposal) -> str:
    """자동 합치기 되돌리기 — 원래 교훈을 다시 켜고 합친 교훈을 끈다 (어느 것도 지우지 않는다)."""
    for i in proposal.ids:
        storage.update_lesson(i, active=True)
    if proposal.merged_id:
        storage.update_lesson(proposal.merged_id, active=False)
    return f"합치기를 되돌렸습니다. 원래 교훈 {len(proposal.ids)}개를 다시 켜고 합친 교훈 #{proposal.merged_id}은(는) 꺼 두었습니다."


def lesson_origin(storage, row: dict) -> dict:
    """교훈이 나온 원래 피드백 — 충돌을 고를 때 보여 준다. 실행 기록이 없으면(합친 교훈·가져온 교훈) 날짜만."""
    run = storage.get_run(row["run_id"]) if row.get("run_id") else None
    decision = run and run["decision"]
    return {
        "date": ((run and run["decided_at"]) or row["created_at"])[:16].replace("T", " "),
        "run_id": row.get("run_id") or "",
        "question": run["question"] if run else "",
        "action": ("채택" if decision.action == "adopt" else "반려") if decision else "",
        "score": decision.score if decision else None,
        "comment": decision.comment if decision else comment_body(split_lesson(row["lesson"])[1]),
    }
