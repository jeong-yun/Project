"""LLM 검증 장치 — 채점 루브릭(퓨샷), 비평가(Critic), 자가 교정(Self-Correction).

검토자(관리자 에이전트)가 하위 결과를 채점·수정할 때 생기는 문제
  - 점수가 들쭉날쭉하다          → 루브릭 + 실제 고득점/저득점 예시(퓨샷)로 기준점을 맞춘다   (rubric_text)
  - 근거 없이 결론부터 낸다      → 스키마에서 analysis 를 점수보다 먼저 쓰게 한다            (core/schemas.py)
  - '지적 → 수정' 과정이 비약한다 → 별도 비평가가 그 중간 과정만 채점한다                    (critique)
  - 비평가가 불합격을 주면        → "A 부분이 B 규칙에 어긋난다. 수정하라" 를 돌려줘 다시 쓴다 (correction_message)

설정: config/verification.yaml (팀별 루브릭은 team.yaml 의 rubric)
"""

from langchain_core.messages import HumanMessage, SystemMessage

from core import llm_factory
from core.llm_factory import NoAvailableModelError
from core.schemas import CriticResult, ModelsConfig, ReviewResult

DEFAULT_CRITIC = {
    "enabled": True,
    "scope": "final",
    "min_score": 4,
    "max_self_corrections": 1,
    "fail_on": "severe",  # severe: 심각한 위반이 있을 때만 불합격 / any: 지적이 하나라도 있으면 불합격 (이전 동작)
    "evidence_chars": 6000,  # 비평가에게 넘기는 근거 자료(데이터 툴·웹 검색) 최대 글자 수
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


def trim_evidence(parts: list[str], budget: int) -> str:
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
    trimmed = [p if shares[i] >= len(p) else p[: shares[i]].rstrip() + " …(생략)" for i, p in enumerate(parts)]
    return "\n\n".join(trimmed) + f"\n\n(근거 자료가 길어 {total}자 중 약 {budget}자만 표시)"


def build_critic_messages(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], rules: list[str] | None = None,
    evidence: str = "",
) -> list:
    """비평가 프롬프트. 최종 답변의 좋고 나쁨이 아니라 '검토 과정의 논리'만 보게 한다.

    rules = 팀 전용 비평 규칙
    evidence = 근거 자료(데이터 툴·웹 검색) — '숫자에는 출처·기준일' 같은 규칙을 비평가가 실제로 대조할 수 있게 한다
    """
    system = "\n".join(
        [
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
            f"수정 답변:\n{review.revised_answer or '(승인 — 수정 없음)'}",
            *([f"[근거 자료 — 데이터 툴·웹 검색 (숫자·출처·날짜 대조용)]\n{evidence}"] if evidence else []),
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


def critique(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], models: ModelsConfig,
    rules: list[str] | None = None, evidence: str = "",
) -> tuple[CriticResult | None, list[str]]:
    """비평가 호출. 모델이 모두 실패하면 (None, 에러) — 비평 없이 진행한다 (답변을 막지 않음)."""
    messages = build_critic_messages(question, reviewed_input, review, criteria, rules, evidence)
    try:
        call = llm_factory.invoke_structured(models.utility, models, CriticResult, messages, "비평가")
    except NoAvailableModelError as e:
        return None, [f"비평가 호출 실패 → 비평 생략: {e}"]
    critic: CriticResult = call.parsed
    critic.score = min(max(critic.score, 1), 5)
    return critic, call.errors


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
