"""LLM 검증 장치 — 채점 루브릭(퓨샷), 비평가(Critic), 자가 교정(Self-Correction).

검토자(관리자 에이전트)가 하위 결과를 채점·수정할 때 생기는 문제
  - 점수가 들쭉날쭉하다          → 루브릭 + 실제 고득점/저득점 예시(퓨샷)로 기준점을 맞춘다   (rubric_text)
  - 근거 없이 결론부터 낸다      → 스키마에서 analysis 를 점수보다 먼저 쓰게 한다            (core/schemas.py)
  - '지적 → 수정' 과정이 비약한다 → 별도 비평가가 그 중간 과정만 채점한다                    (critique)
  - 비평가가 불합격을 주면        → "A 부분이 B 규칙에 어긋난다. 수정하라" 를 돌려줘 다시 쓴다 (correction_message)

설정: config/verification.yaml (팀별 루브릭은 team.yaml 의 rubric)
"""

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from core import llm_factory
from core.llm_factory import NoAvailableModelError
from core.schemas import CriticResult, ModelsConfig, ModelSpec, RankModels, ReviewResult
from core.search import WEB_END

DEFAULT_CRITIC = {
    "enabled": True,
    "scope": "final",
    "min_score": 4,
    "max_self_corrections": 1,
    "fail_on": "severe",  # severe: 심각한 위반이 있을 때만 불합격 / any: 지적이 하나라도 있으면 불합격 (이전 동작)
    "evidence_chars": 6000,  # 비평가에게 넘기는 근거 자료(데이터 툴·웹 검색) 최대 글자 수
    # 비평가에게 넘기는 검토 대상(하위 결과 모음)·수정 답변 최대 글자 수 — 팀장 통합본(출력 4만 토큰)과 하위 결과를
    # 그대로 넘기면 입력이 너무 커져 느리고 멈췄다 (2026-11-04 실측). 앞·뒤를 남기고 가운데를 줄인다.
    # 수정 답변(본 업무 비평)과 💬 수정본 검토(core/followup.py)는 한도로 줄이되 <final> 최종본은 자르지 않는다 (trim_keep_final).
    "input_chars": 16000,
    "answer_chars": 24000,
}


def critic_settings(verification: dict, team_override: dict | None = None) -> dict:
    """기본값 ← config/verification.yaml 의 critic ← team.yaml 의 verification.critic 순으로 덮어쓴다."""
    return {**DEFAULT_CRITIC, **(verification.get("critic") or {}), **((team_override or {}).get("critic") or {})}


def rubric_text(rubric: dict | None) -> str:
    """루브릭(점수 기준 + 퓨샷 예시)을 프롬프트에 넣을 텍스트로 만든다."""
    if not rubric:
        return ""
    lines = [f"[채점 기준(루브릭)] {rubric.get('scale', '')}"]
    lines += [f"- {level}" for level in rubric.get("levels", [])]
    examples = rubric.get("examples", [])
    if examples:
        lines.append("[채점 예시 — 이 기준점에 맞춰 채점한다]")
        for ex in examples:
            lines.append(
                f"- 질문: {ex.get('question')}\n  답변: {ex.get('answer')}\n  → {ex.get('score')}점 (이유: {ex.get('reason')})"
            )
    return "\n".join(lines)


def trim_middle(text: str, budget: int) -> str:
    """budget 글자보다 길면 앞 2/3 · 뒤 1/3 을 남기고 가운데를 줄인다 (결론·표 끝부분이 남게). budget 0 = 그대로."""
    if budget <= 0 or len(text) <= budget:
        return text
    head = budget * 2 // 3
    tail = budget - head
    return f"{text[:head].rstrip()}\n…(가운데 {len(text) - budget:,}자 생략 — 비평 입력 한도)…\n{text[-tail:].lstrip()}"


def trim_keep_final(text: str, budget: int) -> str:
    """trim_middle 과 같되, 글에 <final> 이 있으면 거기부터 끝까지는 통째로 두고 그 앞(검증 로그)만 줄인다.

    개념 편집 팀 결과는 검증 로그 → <final> 최종본 → 요약·적용 계획 순(core/concept.py 출력 계약)이라, 가운데를 줄이면
    사람이 실제로 받는 최종본이 빠진다 (2026-11-10 실제 채택본 31,524자: 최종본의 55% 가 빠졌다).
    <final> 부터 끝까지가 budget 보다 길면 앞부분을 통째로 빼고 최종본은 그대로 둔다 (이때만 budget 을 넘는다).
    <final> 이 없는 글(다른 팀)은 trim_middle 그대로다.
    """
    start = text.lower().find("<final>")
    if budget <= 0 or len(text) <= budget or start < 0:
        return trim_middle(text, budget)
    front, kept = text[:start], text[start:]
    left = budget - len(kept)
    if left > 0:
        front = trim_middle(front, left)
    elif front:
        front = f"…(최종본 앞부분 {len(front):,}자 생략 — 비평 입력 한도)…"
    return f"{front.rstrip()}\n{kept}" if front else kept


def trim_evidence(parts: list[str], budget: int, what: str = "근거 자료") -> str:
    """근거 자료 묶음(툴 결과마다 1개 + 웹 검색 1개)을 budget 글자 안으로 줄인다.

    앞에서부터 자르면 뒤의 웹 검색 결과가 통째로 빠지므로, 짧은 묶음은 통째로 넣고 남은 몫을 긴 묶음끼리 나눈다.
    """
    parts = [p for p in parts if p]
    if budget <= 0 or not parts:
        return ""
    total = sum(len(p) for p in parts)
    if total <= budget:
        return "\n\n".join(parts)
    shares, remaining = {}, budget
    order = sorted(range(len(parts)), key=lambda i: len(parts[i]))  # 짧은 묶음부터 몫을 정한다
    for n, i in enumerate(order):
        shares[i] = min(len(parts[i]), remaining // (len(order) - n))
        remaining -= shares[i]
    # 웹 검색 묶음은 잘려도 '데이터 끝' 표시선을 남긴다 (여는 표시선만 남으면 뒤 글까지 데이터로 읽힌다)
    trimmed = [p if shares[i] >= len(p) else p[: shares[i]].rstrip() + " …(생략)" + (f"\n{WEB_END}" if p.endswith(WEB_END) else "")
               for i, p in enumerate(parts)]
    return "\n\n".join(trimmed) + f"\n\n({what}가 길어 {total}자 중 약 {budget}자만 표시)"


def trim_reviewed(parts: list[str], budget: int) -> str:
    """검토 대상(하위 결과 모음)을 budget 글자 안으로 줄인다 — trim_evidence 와 같되, 하위 결과가 하나이고 <final> 이 있으면
    최종본부터 끝까지는 자르지 않는다 (trim_keep_final). 개념 편집 팀 편집장이 받는 부장 결과가 이 모양이다
    (실제 기록 22건 중 4건이 17,578~19,058자로 한도 16,000자를 넘어 최종본 끝·요약이 잘렸다, 2026-11-11 확인)."""
    if len(parts) == 1 and "<final>" in parts[0].lower():
        return trim_keep_final(parts[0], budget)
    return trim_evidence(parts, budget, "검토 대상")


def build_critic_messages(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], rules: list[str] | None = None,
    evidence: str = "", persona: str = "", answer_chars: int = 0,
) -> list:
    """비평가 프롬프트. 최종 답변의 좋고 나쁨이 아니라 '검토 과정의 논리'만 보게 한다.

    rules = 팀 전용 비평 규칙
    evidence = 근거 자료(데이터 툴·웹 검색) — '숫자에는 출처·기준일' 같은 규칙을 비평가가 실제로 대조할 수 있게 한다
    persona = 비평하는 부서 부장의 관점 (core/dept_heads.py) — 예: 보안부 부장은 보안 사실의 정확성을 먼저 본다
    answer_chars = 수정 답변 최대 글자 수 (0 = 그대로) — 넘으면 가운데를 줄이되 <final> 최종본부터 끝까지는 통째로 둔다 (trim_keep_final)
    """
    system = "\n".join(
        [
            *([persona] if persona else []),
            "너는 비평가(Critic)다. 다른 검토자가 작성한 '검토 과정'의 논리만 전문적으로 채점한다.",
            "- 검토자가 찾은 문제점(issues)이 실제 검토 대상 답변에 근거하는가?",
            "- 수정 답변(revised_answer)이 그 문제점들을 실제로 해결했는가? 근거 없이 새 주장을 추가하지 않았는가?",
            "- 검토자의 점수가 분석·문제점과 일관되는가?",
            "점수를 매기기 전에 반드시 analysis 에 3문장 이내로 분석을 먼저 쓴다.",
            "rule_violations 는 'A 부분이 B 규칙에 어긋남' 형식으로 구체적으로 쓴다.",
            *(
                [
                    "- [근거 자료]와 대조한다: 수정 답변의 숫자·사실이 자료와 맞는지, 출처·기준일이 붙었는지 확인한다.",
                    "  자료와 다른 숫자, 자료에 없는데 사실처럼 쓴 수치, 하위 결과의 올바른 사실을 버린 것은 지적한다.",
                ]
                if evidence
                else []
            ),
            "severity 는 지적 중 가장 심각한 것의 등급이다 (지적이 없으면 none).",
            "- minor : 표현·형식·사소한 누락처럼 결론은 그대로 유효한 지적",
            "- severe: 사실 오류, 근거 자료와 어긋나는 숫자, 하위 결과의 올바른 사실을 버림, 결론을 뒤집는 비약",
            "사소한(minor) 지적만으로 점수를 크게 깎지 않는다.",
            *(["[이 팀의 추가 비평 규칙 — 하나라도 어기면 rule_violations 에 넣는다]", *(f"- {r}" for r in rules)] if rules else []),
        ]
    )
    criteria_text = "\n".join(f"- {c}" for c in criteria) or "- (없음)"
    human = "\n\n".join(
        [
            f"[원 질문]\n{question}",
            f"[검토 대상(하위 결과)]\n{reviewed_input}",
            f"[검토 기준]\n{criteria_text}",
            "[검토자의 출력]\n"
            f"분석: {review.analysis}\n문제점: {review.issues}\n점수: {review.score}\n판정: {review.verdict}\n"
            f"수정 답변:\n{trim_keep_final(review.revised_answer, answer_chars) or '(승인 — 수정 없음)'}",
            *([f"[근거 자료 — 데이터 툴·웹 검색 (숫자·출처·날짜 대조용)]\n{evidence}"] if evidence else []),
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


@dataclass
class Critique:
    """비평가 호출 결과 — critic 이 None 이면 모델이 모두 실패해 비평을 건너뜀 (답변을 막지 않음)."""

    critic: CriticResult | None
    errors: list[str] = field(default_factory=list)
    spec: ModelSpec | None = None  # 실제로 비평한 모델
    is_fallback: bool = False


def critique(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], models: ModelsConfig,
    rules: list[str] | None = None, evidence: str = "", *, chain: RankModels | None = None, persona: str = "",
    who: str = "비평가", answer_chars: int = 0,
) -> Critique:
    """비평가 호출. chain = 비평가 모델 후보 (없으면 models.critic → utility, core/dept_heads.critic_plan 참고)."""
    messages = build_critic_messages(question, reviewed_input, review, criteria, rules, evidence, persona, answer_chars)
    return run_critic(messages, models, chain, who)


def run_critic(messages: list, models: ModelsConfig, chain: RankModels | None = None, who: str = "비평가",
               schema: type[CriticResult] = CriticResult) -> Critique:
    """만들어 둔 비평 프롬프트로 비평가를 부른다 — 본 업무(critique)와 💬 수정본 검토(core/followup.py)가 함께 쓴다."""
    chain = chain or models.critic or models.utility
    try:
        call = llm_factory.invoke_structured(chain, models, schema, messages, who)
    except NoAvailableModelError as e:
        return Critique(None, [f"비평가 호출 실패 → 비평 생략: {e}"])
    critic: CriticResult = call.parsed
    critic.score = min(max(critic.score, 1), 5)
    return Critique(critic, call.errors, call.spec, call.is_fallback)


def critic_passed(critic: CriticResult | None, min_score: int, fail_on: str = "severe") -> bool:
    """합격 = 점수가 min_score 이상이고, 불합격 사유가 될 지적이 없음.

    fail_on="severe": severity 가 severe 일 때만 지적을 불합격 사유로 본다 (사소한 지적이 붙은 4/5 는 통과)
    fail_on="any"   : 논리 비약·규칙 위반이 하나라도 있으면 불합격 (이전 동작 — 실측 불합격률 75%)
    """
    if critic is None:
        return True  # 비평을 못 했으면 통과로 본다
    if critic.score < min_score:
        return False
    if fail_on == "any":
        return not critic.logical_leaps and not critic.rule_violations
    return critic.severity != "severe"


def best_attempt(critics: list[CriticResult | None]) -> int:
    """자가 교정 시도들 중 가장 나은 것의 위치 — 비평 점수가 높고, 같으면 severe 가 아닌 것, 그래도 같으면 나중 것."""
    def rank(i: int) -> tuple[int, bool, int]:
        c = critics[i]
        return (c.score if c else 0, bool(c) and c.severity != "severe", i)

    return max(range(len(critics)), key=rank)


def correction_message(critic: CriticResult) -> str:
    """자가 교정 프롬프트 — 비평 결과를 '어디가 어떤 규칙에 어긋나는지' 형태로 돌려준다."""
    problems = [*critic.rule_violations, *(f"{leap} 부분이 '근거 없는 비약 금지' 규칙에 어긋난다" for leap in critic.logical_leaps)]
    if not problems:
        problems = [f"비평 점수 {critic.score}/5 — {critic.analysis}"]
    bullet = "\n".join(f"- {p}" for p in problems)
    return (
        "너의 이전 단계 논리에서 다음 부분이 규칙에 어긋난다.\n"
        f"{bullet}\n"
        "분석(analysis) 맨 앞에 지적마다 '[지적된 부분] → 어떻게 고쳤는지' 교정 사유를 1문장씩 먼저 쓰고"
        " (고칠 필요가 없다고 판단한 지적은 '수정 없음: 이유'), 그 다음 같은 형식으로 다시 작성하라. 점수는 마지막에 매긴다."
    )
