"""최종 결과에 대한 후속 대화 — 사람이 채택·반려하기 전에 결과에 대해 묻거나 고쳐 달라고 요청한다.

흐름 (워크플로우 그래프 밖에서 동작 — 그래프는 '사람 결정' 단계에서 멈춰 기다리는 중)
  사람의 말 → supervisor(최종 답변자) 에이전트가 한 번에 판단
     · 질문이면  : reply 에 근거와 함께 답만 한다 (결과는 그대로)
     · 수정 요청이면: revised_answer 에 요청을 반영한 '완성 전문'을 쓰고 changes 에 바뀐 점을 적는다
  수정본은 하네스(팀 검증기)로 검사하고, 실패하면 오류를 돌려줘 한 번 더 고치게 한다.
  수정본은 채택 후보(수정본 v1, v2 …)가 되고, 사람이 그중 하나를 채택하거나 전부 버린다.
  대화 내용은 결정 코멘트에 붙어 에이전트별 교훈(core/feedback.py)에도 반영된다.

추가/수정을 사람이 고르지 않아도 되게 한 것처럼(개념 편집 팀), 질문/수정도 사람이 고르지 않는다.
"""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import harness, llm_factory
from core.agent import OrgAgent, with_feedback
from core.schemas import HarnessResult, ModelSpec

QUESTION_CHARS = 4000  # 원 질문(개념 편집 팀은 초안·관련 항목까지 들어 있어 길다)
HISTORY_TURNS = 6  # 프롬프트에 넣을 최근 대화 수


class FollowupResult(BaseModel):
    """후속 대화 한 번의 구조화 출력."""

    analysis: str = Field(description="사람의 말이 질문인지 수정 요청인지, 무엇을 원하는지 1~3문장 분석")
    revised: bool = Field(description="결과를 고쳐 썼으면 true (질문에 답만 했으면 false)")
    reply: str = Field(description="사람에게 하는 답변. 질문이면 근거와 함께 답하고, 수정이면 무엇을 왜 고쳤는지 한두 문장")
    changes: list[str] = Field(default_factory=list, description="수정했을 때 바뀐 점 (짧게, 항목마다 한 줄)")
    revised_answer: str = Field(default="", description="수정했을 때 요청을 반영한 완성 답변 전문. 질문이면 빈 문자열")


@dataclass
class FollowupTurn:
    """화면에 쌓이는 대화 한 줄 (사람 요청 + 에이전트 답)."""

    request: str
    reply: str
    revised: bool = False
    content: str = ""  # 수정본 전문 (revised 일 때)
    changes: list[str] = field(default_factory=list)
    harness: list[HarnessResult] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    model: ModelSpec | None = None
    elapsed_sec: float = 0.0
    version: int = 0  # 수정본 번호 (1부터, 질문이면 0)


def history_text(turns: list[FollowupTurn]) -> str:
    lines = []
    for t in turns[-HISTORY_TURNS:]:
        lines.append(f"- 사람: {t.request}")
        lines.append(f"  답변: {t.reply}" + (f" (수정본 v{t.version} 작성: {'; '.join(t.changes)})" if t.revised else ""))
    return "\n".join(lines)


def build_messages(agent: OrgAgent, question: str, current: str, request: str, turns: list[FollowupTurn],
                   evidence: str = "", lessons: list[str] | None = None) -> list[BaseMessage]:
    system = "\n".join(
        [
            agent.system_prompt(lessons),
            "",
            "지금은 팀이 낸 최종 결과를 사람이 받아 본 뒤, 채택하기 전에 묻거나 고쳐 달라고 하는 단계다.",
            "- 먼저 analysis 에 사람의 말이 '질문'인지 '수정 요청'인지 판단한다 (둘 다일 수 있다).",
            "- 질문이면 revised=false, reply 에 [현재 결과]와 [근거 자료]를 바탕으로 답한다. 근거가 없으면 모른다고 밝힌다.",
            "- 수정 요청이면 revised=true, revised_answer 에 요청을 반영한 **완성 답변 전문**을 쓴다.",
            "  · [현재 결과]의 형식·구조·팀 업무 지침(스킬)을 그대로 지킨다. 요청과 무관한 부분은 바꾸지 않는다.",
            "  · 무엇을 왜 고쳤는지는 changes·reply 에만 쓰고 revised_answer 본문에는 넣지 않는다 (수정 표시·이유 금지).",
            "- 요청이 [근거 자료]나 사실과 어긋나면 그대로 따르지 말고 reply 에서 이유를 설명한다.",
            "  (그래도 사람이 원하는 방향이면 반영하되, 근거가 없다는 점을 reply 에 남긴다)",
        ]
    )
    parts = [
        f"[원래 질문]\n{question[:QUESTION_CHARS]}" + ("…" if len(question) > QUESTION_CHARS else ""),
        f"[현재 결과]\n{current}",
    ]
    if evidence:
        parts.append(f"[근거 자료]\n{evidence}")
    if turns:
        parts.append(f"[지금까지의 대화]\n{history_text(turns)}")
    parts.append(f"[사람의 말]\n{request}")
    return [SystemMessage(system), HumanMessage("\n\n".join(parts))]


def respond(agent: OrgAgent, question: str, current: str, request: str, turns: list[FollowupTurn],
            evidence: str = "", lessons: list[str] | None = None,
            validators: list[str] | None = None, options: dict | None = None, max_retries: int = 1) -> FollowupTurn:
    """사람의 말 한 번에 답한다. 수정했으면 하네스로 검사하고, 실패하면 오류를 돌려줘 다시 고치게 한다."""
    messages = build_messages(agent, question, current, request, turns, evidence, lessons)
    log: list[str] = []
    results: list[HarnessResult] = []
    for attempt in range(max_retries + 1):
        call = llm_factory.invoke_structured(agent.rank_models, agent.models, FollowupResult, messages, agent.name)
        out: FollowupResult = call.parsed
        revised = out.revised and bool(out.revised_answer.strip())
        if not revised or not validators:
            break
        results = harness.run_validators(validators, out.revised_answer, options or {})
        if harness.all_passed(results) or attempt == max_retries:
            if not harness.all_passed(results):
                log.append("하네스 검사 실패 상태로 수정본을 냄 — 채택 전에 확인하세요")
            break
        log.append("하네스 실패 → 다시 고침: " + " / ".join(r.log for r in results if not r.passed))
        messages = with_feedback(messages, out.model_dump_json(), harness.harness_feedback(results))
    version = sum(t.revised for t in turns) + 1 if revised else 0
    return FollowupTurn(
        request=request, reply=out.reply.strip(), revised=revised,
        content=out.revised_answer.strip() if revised else "", changes=out.changes if revised else [],
        harness=results if revised else [], log=log, model=call.spec, elapsed_sec=call.elapsed_sec, version=version,
    )


def decision_note(turns: list[FollowupTurn]) -> str:
    """결정 코멘트에 붙일 대화 요약 — 교훈 생성(core/feedback.py)이 사람의 요구를 알 수 있게."""
    if not turns:
        return ""
    lines = [f"{i}) {t.request}" + (f" → 수정본 v{t.version}" if t.revised else " (질문)") for i, t in enumerate(turns, 1)]
    return "[결정 전 문의·수정 요청]\n" + "\n".join(lines)
