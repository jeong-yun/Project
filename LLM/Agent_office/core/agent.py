"""조직도의 에이전트 한 명 (OrgAgent).

에이전트가 하는 일은 세 가지뿐이다.
  plan()   : [관리자] 질문을 보고 어떤 하위 에이전트에게 무슨 일을 맡길지 정한다      (위 → 아래)
  draft    : [실무자] 맡은 업무의 초안을 쓴다 (토큰 스트리밍)                          (맨 아래)
  review() : [관리자] 하위 결과를 검토(1명)하거나 통합(여러 명)해 더 나은 답을 만든다 (아래 → 위)

각 메서드는 'LLM 한 번 호출'만 담당한다. 재시도·자가 교정·하네스 같은 반복 제어는 core/workflow.py 가 한다.
모델 선택과 실패 시 대안 모델 전환은 core/llm_factory.py 가 맡는다.
"""

import time
from collections.abc import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from core import llm_factory
from core.config import date_label, rank_models, today
from core.llm_factory import NoAvailableModelError
from core.schemas import AgentAnswer, Delegation, ModelsConfig, ModelSpec, OrgNode, Rank, ReviewResult, Team
from core.verification import rubric_text

__all__ = ["NoAvailableModelError", "OrgAgent", "AnswerStream", "chunk_text", "with_feedback"]


def chunk_text(content) -> str:
    """메시지 content 에서 텍스트만 추출 (provider 에 따라 str 또는 block 리스트)."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
        if not isinstance(block, dict) or block.get("type", "text") == "text"
    )


def with_feedback(messages: list[BaseMessage], previous_output: str, feedback: str) -> list[BaseMessage]:
    """이전 출력 + 피드백을 대화 뒤에 붙인다 — 자가 교정·하네스 재시도(Backtracking)에 쓴다.

    [시스템, 질문] → [시스템, 질문, (AI) 이전 출력, (사람) "A 부분이 B 규칙에 어긋난다. 고쳐라"]
    """
    return [*messages, AIMessage(previous_output), HumanMessage(feedback)]


class OrgAgent:
    def __init__(
        self,
        node: OrgNode,
        rank: Rank,
        team: Team,
        models: ModelsConfig,
        rubric: dict | None = None,
        names: dict[str, str] | None = None,
        skills: str = "",
    ):
        self.node = node
        self.rank = rank
        self.team = team
        self.models = models
        self.rubric = rubric  # 채점 루브릭(퓨샷 예시 포함) — 검토할 때 프롬프트에 들어간다
        self.names = names or {}  # 에이전트 id → 표시 이름 (하위 에이전트 소개용)
        self.skills = skills  # 팀 스킬(teams/<팀>/skills/*.md) — 모든 업무에 적용되는 고정 지침
        self.rank_models = rank_models(rank.id, models, team)  # 이 직급이 쓸 모델 후보

    @property
    def name(self) -> str:
        return self.node.name or self.rank.name

    @property
    def is_manager(self) -> bool:
        return bool(self.node.children)

    # ─── 공통 프롬프트 ─────────────────────────────────────────────

    def system_prompt(self, lessons: list[str] | None = None) -> str:
        """이 에이전트의 '정체성' — 페르소나, 직급, 전문 분야, 지난 교훈."""
        persona = self.node.persona or self.team.personas.get(self.rank.id) or f"당신은 {self.rank.name}입니다."
        lines = [
            persona,
            f"이름: {self.name} (직급: {self.rank.name})",
            f"전문 분야: {self.node.specialty or self.rank.role}",
            f"소속 팀: {self.team.name} — {self.team.description}",
            f"오늘 날짜: {date_label(today())} — '현재·최근·향후'는 이 날짜 기준이다. 학습 지식이 아니라 근거 자료의 날짜를 우선한다.",
            "",
            "업무 규칙:",
            "- 한국어로 쓴다.",
            f"- {self.team.output_format} 형식으로 작성한다.",
            "- 모르는 내용은 추측하지 말고 모른다고 밝히며, 근거가 있으면 함께 제시한다.",
            "- 업무 브리프의 [최종 목표]와 [완료 기준]을 만족하는 것이 이 일의 끝이다. 과정은 스스로 판단한다.",
        ]
        if self.skills:  # 사람이 작성한 팀 업무 지침
            lines += ["", "[팀 업무 지침(스킬) — 항상 따른다]", self.skills]
        if lessons:  # 사람의 채택 결과에서 얻은 교훈 (core/feedback.py 가 생성)
            lines += ["", "지난 업무에서 받은 피드백 (반드시 반영):"]
            lines += [f"- {lesson}" for lesson in lessons]
        return "\n".join(lines)

    @staticmethod
    def _web_block(web_context: str, purpose: str) -> list[str]:
        return [f"[근거 자료 — {purpose}]\n{web_context}"] if web_context else []

    # ─── 위 → 아래: 업무 분배 ─────────────────────────────────────

    def plan(self, question: str, task: str, web_context: str = "", lessons: list[str] | None = None) -> AgentAnswer:
        """관리자가 하위 에이전트들에게 업무를 나눠 준다 (구조화 출력 Delegation)."""
        children = "\n".join(
            f"- id: {c.id} / 이름: {self.names.get(c.id, c.id)} / 전문 분야: {c.specialty or '-'}" for c in self.node.children
        )
        system = "\n".join(
            [
                self.system_prompt(lessons),
                "",
                "지금은 관리자로서 하위 에이전트에게 업무를 나눠 주는 단계다.",
                "- 먼저 analysis 에 질문 의도와 필요한 전문 분야를 분석한다.",
                "- 질문에 필요한 하위 에이전트만 고른다 (모두 고를 필요 없음, 최소 1명).",
                "- 각자 전문 분야에 맞는 구체적인 업무 지시(subtask)를 준다. 서로 겹치지 않게 나눈다.",
            ]
        )
        human = "\n\n".join(
            [
                f"[업무 브리프]\n{question}",
                f"[내가 맡은 업무]\n{task}",
                f"[하위 에이전트 목록]\n{children}",
                *self._web_block(web_context, "업무 분배 참고"),
            ]
        )
        call = llm_factory.invoke_structured(
            self.rank_models, self.models, Delegation, [SystemMessage(system), HumanMessage(human)], self.name
        )
        delegation: Delegation = call.parsed
        summary = "\n".join(f"- **{self.names.get(a.child_id, a.child_id)}**: {a.subtask}" for a in delegation.assignments)
        return AgentAnswer(
            node_id=self.node.id, rank_id=self.rank.id, kind="plan", content=summary, delegation=delegation,
            model=call.spec, is_fallback=call.is_fallback, elapsed_sec=call.elapsed_sec,
            input_tokens=call.input_tokens, output_tokens=call.output_tokens, errors=call.errors,
        )

    # ─── 맨 아래: 실무자 초안 ─────────────────────────────────────

    def draft_messages(
        self, question: str, task: str, web_context: str = "", lessons: list[str] | None = None
    ) -> list[BaseMessage]:
        parts = [f"[업무 브리프]\n{question}"]
        if task and task != question:
            parts.append(f"[상사가 맡긴 업무 — 이 범위에 집중]\n{task}")
        if web_context:
            parts += self._web_block(web_context, "답변 근거로 활용")
            parts.append(
                "위 자료를 근거로 답한다. 웹 검색 자료는 해당 문장 끝에 [번호]로, 데이터 툴 자료(시장 데이터·여행 달력 등)는 "
                "'(시장 데이터, 기준일)'처럼 출처를 표시한다. 데이터 툴의 숫자는 다시 계산하지 말고 그대로 인용한다. "
                "자료에 없는 내용은 추측임을 밝히고, 질문과 무관한 동음이의어 자료는 무시한다."
            )
        return [SystemMessage(self.system_prompt(lessons)), HumanMessage("\n\n".join(parts))]

    def stream_draft(self, messages: list[BaseMessage], tag: str = "", on_restart=None) -> "AnswerStream":
        """초안을 토큰 단위로 스트리밍한다. tag 는 화면에서 '누가 쓰는 중인지' 구분하는 표시."""
        return AnswerStream(self, messages, tag, on_restart)

    def draft(self, messages: list[BaseMessage], tag: str = "", on_restart=None) -> AgentAnswer:
        stream = self.stream_draft(messages, tag, on_restart)
        for _ in stream:  # 끝까지 소비해야 stream.result 가 채워진다
            pass
        return stream.result

    # ─── 아래 → 위: 검토·통합 ─────────────────────────────────────

    def review_messages(
        self,
        question: str,
        task: str,
        child_results: list[AgentAnswer],
        web_context: str = "",
        lessons: list[str] | None = None,
        rewrite_answer: bool = True,
    ) -> list[BaseMessage]:
        """하위 결과를 검토(1명) 또는 통합(여러 명)하는 프롬프트."""
        integrate = len(child_results) > 1
        criteria = "\n".join(f"- {c}" for c in self.team.review_criteria) or "- (팀 기준 없음)"
        # 적대적 리뷰어: 결과를 그대로 받아들이지 않고 취약점부터 찾는다
        stance = [
            "너는 적대적 리뷰어(Adversarial Reviewer)다. 하위 결과를 그대로 믿지 말고",
            "취약점·논리적 오류·근거 없는 주장·[완료 기준] 미충족 항목을 먼저 찾아낸 뒤 판단한다.",
        ]
        if integrate:
            job = [
                *stance,
                "지금은 관리자로서 하위 에이전트들의 결과를 하나로 통합하는 단계다.",
                "- 각 결과의 오류·중복·모순을 analysis 에 먼저 분석하고 issues 에 적는다.",
                "- revised_answer 에 모든 결과를 통합한 **하나의 완성 답변 전문**을 쓴다 (verdict 는 revise).",
                "- 하위 결과에 있는 사실·의미를 근거 없이 버리지 않는다. 하위끼리 해석이 다르면 모두 정리하고,",
                "  웹 검색의 최신 뉴스·동향 기준으로 가장 유력한 해석을 앞에 둔다 (버린 내용이 있으면 이유를 issues 에 적는다).",
                "- 출처 [번호] 표기는 유지한다.",
            ]
        else:
            rule = (
                "revise 이면 revised_answer 에 지적을 모두 반영한 **완성 답변 전문**을 쓴다. approve 이면 빈 문자열."
                if rewrite_answer
                else "답변을 직접 고쳐 쓰지 말고 revised_answer 는 빈 문자열로 둔다. 지적은 issues 에 남긴다."
            )
            job = [
                *stance,
                "지금은 관리자로서 하위 에이전트의 결과를 검토하는 단계다.",
                "- 점수를 매기기 전에 analysis 에 취약점(오류·누락·근거 부족)을 3문장 이내로 먼저 분석한다.",
                "- 고칠 것이 없으면 verdict=approve, 있으면 verdict=revise.",
                f"- {rule}",
            ]
        system = "\n".join(
            [
                self.system_prompt(lessons),
                "",
                *job,
                "- score 는 아래 채점 기준에 맞춰 하위 결과의 품질을 매긴다 (분석 → 문제점 → 점수 순서).",
                "- comment 는 하위 에이전트가 다음에 참고할 한두 문장의 조언이다.",
                *(["- [근거 자료](데이터·웹 검색)와 어긋나는 내용, 숫자를 잘못 옮긴 부분은 issues 에 지적하고 바로잡는다."] if web_context else []),
                "",
                rubric_text(self.rubric),
            ]
        )
        results = "\n\n".join(
            f"[하위 결과 {i}] {self.names.get(r.node_id, r.node_id)}\n{r.content}" for i, r in enumerate(child_results, 1)
        )
        human = "\n\n".join(
            [
                f"[업무 브리프]\n{question}",
                f"[내가 맡은 업무]\n{task}",
                results,
                f"[검토 기준]\n{criteria}",
                *self._web_block(web_context, "사실 검증용"),
            ]
        )
        return [SystemMessage(system), HumanMessage(human)]

    def review(self, messages: list[BaseMessage], child_results: list[AgentAnswer], rewrite_answer: bool = True) -> AgentAnswer:
        """검토·통합 LLM 호출 1회. 승인이면 하위 답변을 그대로 올린다."""
        call = llm_factory.invoke_structured(self.rank_models, self.models, ReviewResult, messages, self.name)
        review: ReviewResult = call.parsed
        review.score = min(max(review.score, 1), 10)
        integrate = len(child_results) > 1

        if review.revised_answer.strip() and (integrate or (review.verdict == "revise" and rewrite_answer)):
            content = review.revised_answer
        elif integrate:
            # 통합인데 답을 안 썼으면 하위 결과를 이어 붙여서라도 올린다
            content = "\n\n".join(f"### {self.names.get(r.node_id, r.node_id)}\n{r.content}" for r in child_results)
        else:
            # 승인 또는 고쳐 쓰기 끔 → 하위 답변 유지 (모델이 옮겨 쓰며 생기는 변형 방지)
            content = child_results[0].content
        review.revised_answer = content
        return AgentAnswer(
            node_id=self.node.id, rank_id=self.rank.id, kind="integrate" if integrate else "review",
            content=content, review=review,
            model=call.spec, is_fallback=call.is_fallback, elapsed_sec=call.elapsed_sec,
            input_tokens=call.input_tokens, output_tokens=call.output_tokens, errors=call.errors,
        )

    def candidates(self) -> list[tuple[ModelSpec, bool]]:
        return llm_factory.require_candidates(self.rank_models, self.models, self.name)


class AnswerStream:
    """텍스트 조각을 yield 하고, 끝나면 .result 에 AgentAnswer 를 담는 이터레이터.

    실패 처리 (예: Gemini 503 '요청 폭주', 429, 연결 끊김)
      - 일시적 오류  : 같은 모델로 잠깐 기다렸다 다시 (llm_factory.RETRY_DELAYS)
      - 그 밖의 오류 : 다음 후보(대안 모델)로
      - 쓰는 도중 끊김: 반쯤 쓴 초안은 버리고 처음부터 다시 쓴다 → on_restart() 로 화면의 미리보기를 지우게 알린다
        (yield 된 조각에는 버려진 초안이 섞일 수 있으므로, 최종 답변은 항상 .result 를 쓴다)
    """

    def __init__(self, agent: OrgAgent, messages: list[BaseMessage], tag: str = "", on_restart=None):
        self.agent = agent
        self.messages = messages
        self.tag = tag
        self.on_restart = on_restart  # 다시 쓰기 시작할 때 부르는 함수 (화면 미리보기 초기화용)
        self.result: AgentAnswer | None = None

    def __iter__(self) -> Iterator[str]:
        candidates = self.agent.candidates()

        errors: list[str] = []
        for spec, is_fallback in candidates:
            for attempt in range(len(llm_factory.RETRY_DELAYS) + 1):
                started = time.perf_counter()
                full: AIMessageChunk | None = None
                try:
                    llm = llm_factory.create_chat_model(spec, self.agent.models)
                    # tags: LangGraph 스트림에서 '이 토큰은 누구의 초안인지' 알아보는 표시
                    config = {"tags": [self.tag]} if self.tag else None
                    for chunk in llm.stream(self.messages, config=config):
                        full = chunk if full is None else full + chunk
                        text = chunk_text(chunk.content)
                        if text:
                            yield text
                except Exception as e:  # noqa: BLE001 — provider 별 예외 타입이 제각각
                    # provider 에러 메시지는 매우 길 수 있어 앞부분만 남긴다
                    started_writing = full is not None and bool(chunk_text(full.content))
                    errors.append(
                        f"{spec.label}: {type(e).__name__}: {str(e)[:300]}"
                        + (" (작성 도중 끊김 → 처음부터 다시)" if started_writing else "")
                    )
                    llm_factory.note_failure(spec, e)
                    if started_writing and self.on_restart:
                        self.on_restart()  # 반쯤 쓴 초안은 버린다
                    if attempt < len(llm_factory.RETRY_DELAYS) and llm_factory.is_transient(e):
                        llm_factory.sleep(llm_factory.RETRY_DELAYS[attempt])  # 잠깐 기다렸다 같은 모델로 다시
                        continue
                    break  # 영구 오류이거나 재시도를 다 썼으면 다음 후보로

                usage = (full.usage_metadata or {}) if full is not None else {}
                self.result = AgentAnswer(
                    node_id=self.agent.node.id,
                    rank_id=self.agent.rank.id,
                    kind="draft",
                    content=chunk_text(full.content) if full is not None else "",
                    model=spec,
                    is_fallback=is_fallback,  # 후보 순서로 판단 (같은 모델 재시도 성공은 대안이 아님)
                    elapsed_sec=round(time.perf_counter() - started, 2),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    errors=errors,
                )
                return

        raise NoAvailableModelError(
            f"{self.agent.name}: 모든 후보 모델 호출에 실패했습니다.\n" + "\n".join(errors)
        )
