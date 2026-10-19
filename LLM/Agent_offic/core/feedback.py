"""사람의 최종 결정 → 에이전트별 교훈(Lesson) 생성.

1) 보조(utility) 모델이 결정 내용과 단계별 결과를 보고, 참여한 에이전트마다 한 문장의 교훈을 만든다.
2) 모델이 빠뜨린 에이전트가 있거나 호출이 모두 실패하면 규칙 기반 교훈으로 채운다.
3) 사람의 코멘트는 채택된 에이전트(반려면 전원)의 교훈에 원문 그대로 붙인다 (모델이 빠뜨려도 보장).
→ 어떤 경우에도 참여한 모든 에이전트가 피드백을 받고, 다음 실행 때 시스템 프롬프트에 주입된다.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core import llm_factory
from core.llm_factory import NoAvailableModelError
from core.schemas import AgentAnswer, HumanDecision, Lesson, LessonSet, ModelsConfig, Team

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
