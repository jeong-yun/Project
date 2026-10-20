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

DEFAULT_CRITIC = {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 1}


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


def build_critic_messages(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], rules: list[str] | None = None
) -> list:
    """비평가 프롬프트. 최종 답변의 좋고 나쁨이 아니라 '검토 과정의 논리'만 보게 한다. rules = 팀 전용 비평 규칙."""
    system = "\n".join(
        [
            "너는 비평가(Critic)다. 다른 검토자가 작성한 '검토 과정'의 논리만 전문적으로 채점한다.",
            "- 검토자가 찾은 문제점(issues)이 실제 검토 대상 답변에 근거하는가?",
            "- 수정 답변(revised_answer)이 그 문제점들을 실제로 해결했는가? 근거 없이 새 주장을 추가하지 않았는가?",
            "- 검토자의 점수가 분석·문제점과 일관되는가?",
            "점수를 매기기 전에 반드시 analysis 에 3문장 이내로 분석을 먼저 쓴다.",
            "rule_violations 는 'A 부분이 B 규칙에 어긋남' 형식으로 구체적으로 쓴다.",
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
        ]
    )
    return [SystemMessage(system), HumanMessage(human)]


def critique(
    question: str, reviewed_input: str, review: ReviewResult, criteria: list[str], models: ModelsConfig,
    rules: list[str] | None = None,
) -> tuple[CriticResult | None, list[str]]:
    """비평가 호출. 모델이 모두 실패하면 (None, 에러) — 비평 없이 진행한다 (답변을 막지 않음)."""
    try:
        call = llm_factory.invoke_structured(
            models.utility, models, CriticResult, build_critic_messages(question, reviewed_input, review, criteria, rules), "비평가"
        )
    except NoAvailableModelError as e:
        return None, [f"비평가 호출 실패 → 비평 생략: {e}"]
    critic: CriticResult = call.parsed
    critic.score = min(max(critic.score, 1), 5)
    return critic, call.errors


def critic_passed(critic: CriticResult | None, min_score: int) -> bool:
    if critic is None:
        return True  # 비평을 못 했으면 통과로 본다
    return critic.score >= min_score and not critic.logical_leaps and not critic.rule_violations


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
