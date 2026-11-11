"""최종 결과에 대한 후속 대화 — 사람이 채택·반려하기 전에 결과에 대해 묻거나 고쳐 달라고 요청한다.

흐름 (워크플로우 그래프 밖에서 동작 — 그래프는 '사람 결정' 단계에서 멈춰 기다리는 중)
  사람의 말 → supervisor(최종 답변자) 에이전트가 한 번에 판단
     · 질문이면  : reply 에 근거와 함께 답만 한다 (결과는 그대로)
     · 수정 요청이면: revised_answer 에 요청을 반영한 '완성 전문'을 쓰고 changes 에 바뀐 점을 적는다
  수정본은 하네스(팀 검증기)로 검사하고, 실패하면 오류를 돌려줘 한 번 더 고치게 한다.
  👔 수정본은 그 팀이 속한 부서의 부장(비평가)이 마지막으로 검토한다 (질문에 답만 했으면 부르지 않는다) — HeadReview
     · 본 업무의 비평(core/workflow.py integrate)과 같은 부장 모델·관점·검토 기준·비평 설정(꺼져 있으면 부르지 않음)을 쓴다
     · 보는 것 ① 사람의 수정 요청이 제대로 반영됐는가 (지적) ② 요청과 별개로 수정본에서 추가로 고칠 점 (extra_fixes)
     · 긴 글은 비평 글자 한도까지만 넣되, <final> 최종본부터 끝까지는 자르지 않고 그 앞의 검증 로그만 줄인다 (verification.trim_keep_final)
     · 불합격이면 팀장이 지적을 받아 다시 고친다 (고친 뒤 하네스도 다시) → 부장이 다시 검토
     · 합격인데 '추가로 고칠 점'이 있으면 부장이 그 점들만 직접 고친다 (HeadFix — 완성 전문, 하네스는 검사만) → 부장이 다시 검토
       - 사람에게 버튼으로 묻지 않는다 (2026-11-10 사용자 요청: 추가로 고칠 점이 나올 때마다 버튼을 누르기 불편하다 → 부장이 고치고 다시 검토)
       - 한 검토에 불합격 지적과 추가 점이 함께 있으면 팀장이 지적을 고치는 쪽이 먼저다. 추가 점은 넘기지 않고 다음 검토 결과를 따른다
       - 다시 검토도 처음과 같은 기준의 온전한 검토다 (점수·합격·지적·새 추가 점). 부장이 직접 고친 점만 알려 줘 '요청과 무관한 변경'으로 지적하지 않게 한다
       - 부장이 고치지 못하면(모델 실패·끊음·건너뜀·하네스 실패·빈 글이나 많이 짧아진 글) 직전 수정본을 그대로 내고 남은 점을 보여 준다
     · 상한은 본 업무와 같은 max_self_corrections 하나 — 팀장이 다시 고친 횟수와 부장이 직접 고친 횟수를 합쳐서 센다 (새 설정을 만들지 않음)
       - 다시 고친 글이 불합격이면 앞선 수정본 중 나은 것을 낸다 (합격한 것 먼저, 그다음 verification.best_attempt)
       - 상한에 닿고도 남은 점만 화면에 보이고 버튼으로 다음 수정 요청에 넣을 수 있다 (next_request)
       - 요청 밖의 변경이므로 부장이 무엇을 직접 고쳤는지 FollowupTurn.auto_fixed 에 남겨 화면에 보여 준다
     · 부장 검토 호출이 실패하거나 사람이 건너뛰면(⏱ 호출 모니터) 수정본은 그대로 내고 이유를 log 에 남긴다
     모델 호출: 수정 한 번에 부장 검토 +1. 불합격(팀장 다시 쓰기 1 + 부장 다시 검토 1)이나 추가로 고칠 점(부장 직접 수정 1 + 부장 다시 검토 1)이
       나온 회차마다 +2 — 상한 1회(기본)면 최대 +3, 2회면 최대 +5. 팀장이 다시 쓴 글이 하네스에 걸리면 재작성마다 +1 (부장의 직접 수정은 다시 쓰지 않는다).
  수정본은 채택 후보(수정본 v1, v2 …)가 되고, 사람이 그중 하나를 채택하거나 전부 버린다.
  대화 내용은 결정 코멘트에 붙어 에이전트별 교훈(core/feedback.py)에도 반영된다 (부장 검토 의견은 넣지 않는다 — decision_note).

추가/수정을 사람이 고르지 않아도 되게 한 것처럼(개념 편집 팀), 질문/수정도 사람이 고르지 않는다.
"""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import call_monitor, harness, llm_factory, verification
from core.agent import OrgAgent, with_feedback
from core.dept_heads import CriticPlan
from core.llm_factory import NoAvailableModelError
from core.schemas import CriticResult, HarnessResult, ModelSpec

QUESTION_CHARS = 4000  # 원 질문(개념 편집 팀은 초안·관련 항목까지 들어 있어 길다)
HISTORY_TURNS = 6  # 프롬프트에 넣을 최근 대화 수


class FollowupResult(BaseModel):
    """후속 대화 한 번의 구조화 출력."""

    analysis: str = Field(description="사람의 말이 질문인지 수정 요청인지, 무엇을 원하는지 1~3문장 분석")
    revised: bool = Field(description="결과를 고쳐 썼으면 true (질문에 답만 했으면 false)")
    reply: str = Field(description="사람에게 하는 답변. 질문이면 근거와 함께 답하고, 수정이면 무엇을 왜 고쳤는지 한두 문장")
    changes: list[str] = Field(default_factory=list, description="수정했을 때 바뀐 점 (짧게, 항목마다 한 줄)")
    revised_answer: str = Field(default="", description="수정했을 때 요청을 반영한 완성 답변 전문. 질문이면 빈 문자열")


class RevisionCritic(CriticResult):
    """👔 부서 부장의 수정본 검토 — 본 업무 비평(CriticResult)과 같은 합격 기준을 쓰고 '추가로 고칠 점'만 더한다."""

    analysis: str = Field(description="점수 전에, 사람의 수정 요청이 수정본에 제대로 반영됐는지 3문장 이내로 먼저 분석")
    logical_leaps: list[str] = Field(description="수정본에 근거 없이 새로 들어간 주장. 없으면 빈 목록")
    rule_violations: list[str] = Field(
        description="요청이 빠졌거나 다르게 반영된 곳, 요청과 무관하게 바뀐 곳, 검토 기준을 어긴 곳을 "
                    "'A 부분이 B 에 어긋남' 형식으로. 없으면 빈 목록"
    )
    score: int = Field(description="수정 요청을 얼마나 제대로 반영했는가 (1~5, 5=빠짐없이 반영). 위 분석에 근거해 마지막에 매긴다")
    extra_fixes: list[str] = Field(
        default_factory=list,
        description="사람의 요청과는 별개로, 수정본에서 추가로 고쳐야 할 점을 '어디를 → 어떻게' 한 줄씩. 없으면 빈 목록",
    )


class HeadFix(BaseModel):
    """👔 부장이 자기가 짚은 '추가로 고칠 점'을 직접 고친 결과."""

    revised_answer: str = Field(description="[추가로 고칠 점]만 반영한 완성 답변 전문 (처음부터 끝까지, 생략 없이)")


# 부장이 고친 글이 고치기 전 길이의 이 비율보다 짧으면 잘린 것으로 보고 버린다 (추정값 — 실제 모델로 재지 않음)
HEAD_FIX_MIN_RATIO = 0.7


@dataclass
class HeadReview:
    """부장 검토에 필요한 것 — 본 업무의 비평과 같은 값을 app.py 가 넘긴다."""

    plan: CriticPlan  # 누가(부서 부장)·어느 모델로 (core/dept_heads.critic_plan)
    cfg: dict  # 팀의 비평 설정 (verification.critic_settings) — enabled 가 꺼져 있으면 검토하지 않는다
    criteria: list[str] = field(default_factory=list)  # 팀 검토 기준 + 브리프의 완료 기준


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
    # 👔 부장 검토 (AgentAnswer 와 같은 이름). critic 이 없으면 검토를 안 했거나 못 한 것 — 이유는 log
    critic: RevisionCritic | None = None
    critic_by: str = ""
    critic_model: ModelSpec | None = None
    critic_passed: bool = True
    auto_fixed: list[str] = field(default_factory=list)  # 부장이 추가로 짚어 직접 고친 점 (이 수정본에 반영됨 — 사람이 요청하지 않은 변경)


def history_text(turns: list[FollowupTurn]) -> str:
    lines = []
    for t in turns[-HISTORY_TURNS:]:
        lines.append(f"- 사람: {t.request}")
        lines.append(f"  답변: {t.reply}" + (f" (수정본 v{t.version} 작성: {'; '.join(t.changes)})" if t.revised else ""))
    return "\n".join(lines)


def build_messages(agent: OrgAgent, question: str, current: str, request: str, turns: list[FollowupTurn],
                   evidence: str = "", lessons: list[str] | None = None, criteria: list[str] | None = None) -> list[BaseMessage]:
    """criteria = 부장이 수정본을 검토할 기준 (HeadReview.criteria — 팀 검토 기준 + 브리프의 완료 기준).
    팀장은 원래 질문과 현재 결과만 받으므로, 이것을 주지 않으면 모르는 기준으로 불합격을 받는다."""
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
    if criteria:
        parts.append("[검토 기준 — 고친 부분도 이 기준을 지킨다 (부장이 이 기준으로 수정본을 검토한다)]\n"
                     + "\n".join(f"- {c}" for c in criteria))
    if evidence:
        parts.append(f"[근거 자료]\n{evidence}")
    if turns:
        parts.append(f"[지금까지의 대화]\n{history_text(turns)}")
    parts.append(f"[사람의 말]\n{request}")
    return [SystemMessage(system), HumanMessage("\n\n".join(parts))]


def revision_messages(head: HeadReview, question: str, before: str, request: str, out: FollowupResult,
                      evidence: str = "", head_fixed: list[str] | None = None) -> list[BaseMessage]:
    """부장에게 보내는 수정본 검토 프롬프트. 본 업무의 비평은 검토자의 '지적 → 수정' 과정도 보지만, 여기서는 수정본 자체만 본다.

    head_fixed = 앞선 검토에서 부장이 짚어 직접 고친 '추가로 고칠 점'. 다시 검토도 같은 기준의 온전한 검토이고,
    이 목록은 그 수정으로 바뀐 곳을 '요청과 무관한 변경'으로 잘못 지적하지 않게 알려 주는 것뿐이다.
    글이 한도(input_chars·answer_chars)보다 길면 줄이되, <final> 최종본부터 끝까지는 통째로 넣는다 (verification.trim_keep_final).
    """
    cfg, rules = head.cfg, head.cfg.get("rules")
    system = "\n".join(
        [
            *([head.plan.head.critic_persona()] if head.plan.head else []),
            "팀장이 사람의 수정 요청을 받아 결과를 고쳐 썼다. 너는 그 수정본을 사람에게 내기 전에 마지막으로 검토한다.",
            "- [사용자의 수정 요청]이 [수정본]에 빠짐없이, 요청한 뜻대로 반영됐는가? ([수정 전 결과]와 견주어 본다)",
            "- 요청과 무관한 부분을 바꾸거나 [수정 전 결과]의 올바른 내용을 버리지 않았는가? 근거 없이 새 주장을 넣지 않았는가?",
            "- 수정본이 [검토 기준]을 여전히 지키는가?",
            "점수를 매기기 전에 반드시 analysis 에 3문장 이내로 분석을 먼저 쓴다.",
            "위 세 가지에 어긋나는 것은 rule_violations 에 'A 부분이 B 에 어긋남' 형식으로 구체적으로 쓴다.",
            *(["- [근거 자료]와 대조한다: 수정본의 숫자·사실이 자료와 맞는지, 출처·기준일이 붙었는지 확인한다."] if evidence else []),
            "severity 는 지적 중 가장 심각한 것의 등급이다 (지적이 없으면 none).",
            "- minor : 표현·형식·사소한 누락처럼 요청은 반영됐고 결론도 그대로 유효한 지적",
            "- severe: 요청을 반영하지 않았거나 다르게 반영함, 사실 오류, 근거 자료와 어긋나는 숫자, 올바른 내용을 버림",
            "사소한(minor) 지적만으로 점수를 크게 깎지 않는다.",
            "extra_fixes 에는 사람의 요청과는 별개로, 수정본에서 추가로 고쳐야 할 점을 '어디를 → 어떻게' 한 줄씩 쓴다 (없으면 빈 목록).",
            "- 요청이 반영되지 않은 문제는 extra_fixes 가 아니라 rule_violations 에 쓴다.",
            "- extra_fixes 는 점수·등급에 넣지 않는다 — 사람에게 묻지 않고 네가 바로 직접 고치게 되니, 취향 차이는 빼고 고쳐야 할 것만 쓴다.",
            verification.TRIM_NOTE,
            *(["[부장이 직접 고친 점]은 앞선 검토에서 네가 짚어 이 수정본에 직접 고쳐 넣은 것이다.",
               "- 그에 따라 바뀐 곳은 '요청과 무관한 변경'으로 보지 않는다. 이미 고쳐진 점은 extra_fixes 에 다시 쓰지 않는다."] if head_fixed else []),
            *(["[이 팀의 추가 비평 규칙 — 하나라도 어기면 rule_violations 에 넣는다]", *(f"- {r}" for r in rules)] if rules else []),
        ]
    )
    human = "\n\n".join(
        [
            f"[원래 질문]\n{question[:QUESTION_CHARS]}" + ("…" if len(question) > QUESTION_CHARS else ""),
            f"[수정 전 결과]\n{verification.trim_keep_final(before, cfg['input_chars'])}",
            f"[사용자의 수정 요청]\n{request}",
            "[팀장이 밝힌 바뀐 점]\n" + ("\n".join(f"- {c}" for c in out.changes) or f"- {out.reply}"),
            *(["[부장이 직접 고친 점]\n" + "\n".join(f"- {p}" for p in head_fixed)] if head_fixed else []),
            f"[수정본]\n{verification.trim_keep_final(out.revised_answer.strip(), cfg['answer_chars'])}",
            "[검토 기준]\n" + ("\n".join(f"- {c}" for c in head.criteria) or "- (없음)"),
            *([f"[근거 자료 — 데이터 툴·웹 검색 (숫자·출처·날짜 대조용)]\n{evidence}"] if evidence else []),
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


def refix_message(critic: CriticResult) -> str:
    """부장 불합격 뒤 팀장에게 돌려주는 글. 본 업무의 자가 교정 글(verification.correction_message)은 검토 출력용이라
    '점수는 마지막에'·'analysis 맨 앞에 교정 사유'를 말하는데, 팀장의 출력(FollowupResult)에는 점수 칸이 없고 analysis 는 다른 뜻이다."""
    return (
        "부장 검토에서 네 수정본의 다음 부분이 지적됐다.\n"
        f"{verification.critic_problems(critic)}\n"
        "지적을 반영해 수정본을 다시 쓴다. 지적마다 어떻게 고쳤는지는 changes 에 한 줄씩 쓰고"
        " (고칠 필요가 없다고 판단한 지적은 '수정 없음: 이유'), revised_answer 에는 고친 완성 전문만 쓴다."
        " 사람의 수정 요청은 그대로 지킨다."
    )


def review_note(name: str, cfg: dict, waiting: bool = False) -> str:
    """💬 수정 요청 화면의 안내 문구 — 이 팀 설정에서 부장(name)이 수정본에 실제로 하는 일.
    비평을 끈 팀은 검토가 없고, 상한(max_self_corrections) 0 인 팀은 검토만 한다. waiting = 답을 기다리는 동안의 문구."""
    if not cfg["enabled"]:
        return "답이 오면 " if waiting else "이 팀은 부장 검토를 꺼 두어, 수정본은 검토 없이 나옵니다."
    if cfg["max_self_corrections"] < 1:
        if waiting:
            return f"수정본이면 {name}의 검토까지 끝난 뒤 답이 오고, "
        return (f"수정본은 **{name}**이(가) 마지막으로 검토만 합니다 — 지적이나 추가로 고칠 점이 나와도 다시 고치지 않고 "
                "수정본 아래에 적어 둡니다.")
    if waiting:
        return f"수정본이면 {name}의 검토(추가로 고칠 점이 있으면 직접 고치고 다시 검토)까지 끝난 뒤 답이 오고, "
    return (f"수정본은 **{name}**이(가) 마지막으로 검토하고, 추가로 고칠 점을 찾으면 직접 고친 뒤 다시 검토합니다 "
            "(무엇을 고쳤는지는 수정본 아래에 적힙니다).")


def _is_revision(out: FollowupResult) -> bool:
    return out.revised and bool(out.revised_answer.strip())


def _write(agent: OrgAgent, messages: list[BaseMessage], validators: list[str] | None, options: dict | None,
           max_retries: int, log: list[str]) -> tuple[FollowupResult, llm_factory.StructuredCall, list[HarnessResult]]:
    """팀장이 한 번 답한다. 수정했으면 하네스로 검사하고, 실패하면 오류를 돌려줘 다시 고치게 한다."""
    results: list[HarnessResult] = []
    for attempt in range(max_retries + 1):
        call = llm_factory.invoke_structured(agent.rank_models, agent.models, FollowupResult, messages, agent.name)
        out: FollowupResult = call.parsed
        if not _is_revision(out) or not validators:
            break
        results = harness.run_validators(validators, out.revised_answer, options or {})
        if harness.all_passed(results) or attempt == max_retries:
            if not harness.all_passed(results):
                log.append("하네스 검사 실패 상태로 수정본을 냄 — 채택 전에 확인하세요")
            break
        log.append("하네스 실패 → 다시 고침: " + " / ".join(r.log for r in results if not r.passed))
        messages = with_feedback(messages, out.model_dump_json(), harness.harness_feedback(results))
    return out, call, results


def head_fix_messages(head: HeadReview, question: str, request: str, answer: str, extras: list[str],
                      evidence: str = "") -> list[BaseMessage]:
    """부장이 자기가 짚은 '추가로 고칠 점'을 직접 고칠 때의 프롬프트. 수정본은 줄이지 않고 전문을 준다 (전문을 다시 써야 하므로)."""
    system = "\n".join(
        [
            *([head.plan.head.critic_persona()] if head.plan.head else []),
            "너는 팀장이 사람의 수정 요청을 받아 쓴 [수정본]을 검토했고, 요청과는 별개로 [추가로 고칠 점]을 짚었다. 이제 그 점들을 네가 직접 고친다.",
            "- [추가로 고칠 점]만 반영한다. 그 밖의 내용과 형식·구조, [사용자의 수정 요청]으로 바뀐 부분은 그대로 둔다.",
            "- revised_answer 에는 고친 **완성 전문**을 처음부터 끝까지 쓴다. 줄이거나 '이하 동일'처럼 생략하지 않는다.",
            "- 무엇을 왜 고쳤는지, 수정 표시는 본문에 넣지 않는다.",
            "- [검토 기준]을 지키고, 근거 없는 숫자·사실을 새로 넣지 않는다" + (" ([근거 자료]와 맞춘다)." if evidence else "."),
        ]
    )
    human = "\n\n".join(
        [
            f"[원래 질문]\n{question[:QUESTION_CHARS]}" + ("…" if len(question) > QUESTION_CHARS else ""),
            f"[사용자의 수정 요청]\n{request}",
            f"[수정본]\n{answer}",
            "[추가로 고칠 점]\n" + "\n".join(f"- {p}" for p in extras),
            "[검토 기준]\n" + ("\n".join(f"- {c}" for c in head.criteria) or "- (없음)"),
            *([f"[근거 자료 — 데이터 툴·웹 검색]\n{evidence}"] if evidence else []),
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


def _head_fix(head: HeadReview, agent: OrgAgent, question: str, request: str, draft: tuple, extras: list[str],
              evidence: str, validators: list[str] | None, options: dict | None) -> tuple[tuple | None, str]:
    """부장이 '추가로 고칠 점'만 반영한 완성 전문을 직접 쓴다 — 모델 1번, 하네스는 검사만 한다 (실패해도 다시 쓰게 하지 않는다).

    (새 수정본, "") 을 돌려준다. 고치지 못했으면 (None, 이유) — 부르는 쪽이 직전 수정본을 그대로 낸다.
    """
    out, call, _ = draft
    before = out.revised_answer.strip()
    try:
        with call_monitor.purpose("self_correct"):  # ⏱ 호출 모니터·진행 막대의 '🔁 자가 교정', ⛔ 부장 검토 건너뛰기로 함께 끊긴다
            fix = llm_factory.invoke_structured(head.plan.chain, agent.models, HeadFix,
                                                head_fix_messages(head, question, request, before, extras, evidence),
                                                f"{head.plan.who} (직접 수정)").parsed
    except call_monitor.CallCancelled:
        return None, "사용자가 끊음"
    except NoAvailableModelError as e:
        return None, (str(e).splitlines() or [""])[0][:120] or "모델 호출 실패"
    text = fix.revised_answer.strip()
    if len(text) < len(before) * HEAD_FIX_MIN_RATIO:
        return None, f"고친 글이 비었거나 많이 짧음 ({len(before):,}자 → {len(text):,}자, 잘린 것으로 봄)"
    results = harness.run_validators(validators, text, options or {}) if validators else []
    if not harness.all_passed(results):
        return None, "하네스 검사 실패 — " + " / ".join(r.log for r in results if not r.passed)
    return (out.model_copy(update={"revised_answer": text}), call, results), ""


def respond(agent: OrgAgent, question: str, current: str, request: str, turns: list[FollowupTurn],
            evidence: str = "", lessons: list[str] | None = None,
            validators: list[str] | None = None, options: dict | None = None, max_retries: int = 1,
            head: HeadReview | None = None) -> FollowupTurn:
    """사람의 말 한 번에 답한다. 수정했으면 하네스로 검사하고, head 가 있으면 부서 부장이 수정본을 마지막으로 검토한다.

    부장 검토가 불합격이면 팀장이 다시 고치고, 합격인데 '추가로 고칠 점'이 있으면 부장이 직접 고친다. 어느 쪽이든 부장이 다시 검토하고,
    두 가지 고침을 합쳐 max_self_corrections 번까지만 한다.
    """
    messages = build_messages(agent, question, current, request, turns, evidence, lessons, head.criteria if head else None)
    log: list[str] = []
    draft = _write(agent, messages, validators, options, max_retries, log)
    reviewed: verification.Critique | None = None  # 지금 수정본(draft)에 대한 부장 검토
    passed = True
    fixed: list[str] = []  # 부장이 지금 수정본(draft)에 직접 고쳐 넣은 '추가로 고칠 점'
    if _is_revision(draft[0]) and head is not None and head.cfg["enabled"]:
        cfg = head.cfg
        log += [f"⚠️ {w}" for w in head.plan.warnings]
        tried: list[tuple] = []  # 부장 검토를 받은 (수정본, 검토, 합격 여부, 부장이 직접 고친 점) — 처음 → 다시 고친 순
        for round_ in range(cfg["max_self_corrections"] + 1):
            reviewed = None
            if call_monitor.skipped("critic") or call_monitor.stopped():
                log.append("⛔ 사용자가 부장 검토를 건너뜀 → 검토 없이 수정본을 냄")
                break
            try:
                with call_monitor.purpose("critic"):
                    done = verification.run_critic(
                        revision_messages(head, question, current, request, draft[0], evidence, fixed),
                        agent.models, head.plan.chain, f"{head.plan.who} (비평가)", RevisionCritic)
            except call_monitor.CallCancelled:
                log.append("⛔ 사용자가 부장 검토를 끊음 → 검토 없이 수정본을 냄")
                break
            if done.critic is None:  # 부장 모델이 모두 실패 — 수정본을 막지 않는다
                lines = (done.errors[0] if done.errors else "").splitlines()  # 첫 줄 = 요약, 다음 줄부터 모델별 오류
                why = (f"{lines[1].split(': ', 1)[0]}: {llm_factory.failure_reason(lines[1])}" if len(lines) > 1
                       else lines[0].split(": ", 2)[-1] if lines else "모델 호출 실패")
                log.append(f"👔 부장 검토를 하지 못했습니다 — {why[:200]} (수정본은 검토 없이 냈으니 채택 전에 직접 확인하세요)")
                break
            if done.is_fallback and done.spec:
                log.append(f"부장 대안 모델 {done.spec.label} 로 검토")
            reviewed = done
            passed = verification.critic_passed(done.critic, cfg["min_score"], cfg["fail_on"])
            tried.append((draft, done, passed, fixed))
            extras = done.critic.extra_fixes
            if (passed and not extras) or round_ == cfg["max_self_corrections"]:
                break
            if passed:  # 합격 + 추가로 고칠 점 → 부장이 직접 고친다 (사람에게 버튼으로 묻지 않는다)
                again, why = _head_fix(head, agent, question, request, draft, extras, evidence, validators, options)
                if again is None:
                    log.append(f"부장이 추가로 고칠 점 {len(extras)}개를 직접 고치지 못해 직전 수정본을 유지 ({why})")
                    break
                log.append(f"부장이 추가로 고칠 점 {len(extras)}개를 직접 고침 {round_ + 1}회")
                fixed = [*fixed, *(p for p in extras if p not in fixed)]
                draft = again
                continue
            # 불합격 → 팀장이 지적을 받아 다시 고친다 (추가로 고칠 점은 넘기지 않는다 — 다음 검토 결과를 따른다)
            failed = f"부장 검토 불합격({done.critic.score}/5)"
            try:
                with call_monitor.purpose("self_correct"):
                    again = _write(agent, with_feedback(messages, draft[0].model_dump_json(), refix_message(done.critic)),
                                   validators, options, max_retries, log)
            except (call_monitor.CallCancelled, NoAvailableModelError) as e:
                log.append(f"{failed} — 다시 고치지 못해 직전 수정본을 유지 ({(str(e).splitlines() or [''])[0][:120]})")
                break
            if not _is_revision(again[0]):
                log.append(f"{failed} — 팀장이 다시 고치지 않아 직전 수정본을 유지")
                break
            log.append(f"{failed} → 팀장이 다시 고침 {round_ + 1}회")
            draft = again
        if reviewed is not None and not passed and len(tried) > 1:
            # 다시 고쳐도 불합격 — 합격한 수정본이 앞에 있으면 그중에서, 없으면 전체에서 부장 점수가 가장 나은 것을 낸다 (본 업무처럼)
            pool = [t for t in tried if t[2]] or tried
            best = pool[verification.best_attempt([t[1].critic for t in pool])]
            if best[1] is not reviewed:
                log.append(f"다시 고친 수정본이 나아지지 않아 {tried.index(best) + 1}번째로 쓴 수정본을 유지")
                draft, reviewed, passed, fixed = best
        if reviewed is not None and not passed:
            log.append("부장 검토 불합격 상태로 수정본을 냄 — 채택 전에 지적을 확인하세요")
        if reviewed is not None:  # 다시 검토가 그대로 '아직 남음'이라고 한 점은 고쳤다고 하지 않는다
            fixed = [p for p in fixed if p not in reviewed.critic.extra_fixes]
    out, call, results = draft
    revised = _is_revision(out)
    version = sum(t.revised for t in turns) + 1 if revised else 0
    return FollowupTurn(
        request=request, reply=out.reply.strip(), revised=revised,
        content=out.revised_answer.strip() if revised else "", changes=out.changes if revised else [],
        harness=results if revised else [], log=log, model=call.spec, elapsed_sec=call.elapsed_sec, version=version,
        critic=reviewed.critic if reviewed else None, critic_by=head.plan.who if reviewed else "",
        critic_model=reviewed.spec if reviewed else None, critic_passed=passed or reviewed is None,
        auto_fixed=fixed,
    )


def head_review_markdown(turn: FollowupTurn, by: str) -> str:
    """화면에 보일 부장 검토 (마크다운). by = 누가·어느 모델로 — 예: '🛡️ 보안부 부장 (gpt-5.5)'.

    검토 결과 · 부장이 직접 고친 점 · (상한에 닿고도) 남은 추가로 고칠 점. 다시 검토를 건너뛰어 critic 이 없어도 직접 고친 점은 보인다.
    """
    c = turn.critic
    lines: list[str] = []
    if c is not None:
        verdict = "✅ 합격" if turn.critic_passed else "❌ 불합격 — 채택하기 전에 아래 지적을 확인하세요"
        lines += [f"**{by or '부장'} 검토** · {c.score}/5 · {verdict}", "", c.analysis]
        points = [*c.rule_violations, *c.logical_leaps]
        if points:
            lines += ["", "**지적**", *(f"- {p}" for p in points)]
    if turn.auto_fixed:
        lines += ["", f"**{by or '부장'}이(가) 추가로 고칠 점 {len(turn.auto_fixed)}개를 직접 고쳤습니다** "
                      "(요청하신 것 밖의 변경이니 채택하기 전에 확인하세요)", *(f"- {p}" for p in turn.auto_fixed)]
    if c is not None and c.extra_fixes:
        lines += ["", "**추가로 고칠 점** (부장이 직접 고치지 못하고 남은 점입니다)", *(f"- {p}" for p in c.extra_fixes)]
    return "\n".join(lines).strip()


def next_request(turn: FollowupTurn) -> str:
    """부장이 직접 고친 뒤에도 남은 것(지적 + 추가로 고칠 점)을 다음 수정 요청 글로. 남은 것이 없으면 ''."""
    c = turn.critic
    points = [*c.rule_violations, *c.logical_leaps, *c.extra_fixes] if c else []
    if not points:
        return ""
    return "부장 검토에서 나온 고칠 점을 반영해 주세요.\n" + "\n".join(f"- {p}" for p in points)


def decision_note(turns: list[FollowupTurn]) -> str:
    """결정 코멘트에 붙일 대화 요약 — 교훈 생성(core/feedback.py)이 사람의 요구를 알 수 있게.

    부장 검토(점수·지적·추가로 고칠 점·부장이 직접 고친 점)는 넣지 않는다 (결정 2026-11-10): 교훈은 사람의 요구에서 만드는데, 모델인 부장의 의견을
    결정 코멘트에 섞으면 사람이 하지 않은 말이 교훈으로 굳는다. 사람이 버튼으로 부장의 지적을 다음 수정 요청으로 보냈다면
    그 글이 request 로 여기 들어오므로, 사람이 고른 것만 남는다.
    """
    if not turns:
        return ""
    lines = [f"{i}) {t.request}" + (f" → 수정본 v{t.version}" if t.revised else " (질문)") for i, t in enumerate(turns, 1)]
    return "[결정 전 문의·수정 요청]\n" + "\n".join(lines)
