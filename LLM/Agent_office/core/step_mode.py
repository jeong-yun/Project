"""🔀 단계마다 확인 (step-by-step mode) — 켜면 워크플로우가 세 곳에서 멈추고 사람의 선택을 기다린다.

  ① 브리프 뒤   : 정리된 브리프를 고치고, 팀장이 내놓은 2~3가지 방향(A·B·C) 중 하나를 고르거나 직접 쓴다
  ② 팀장 분배 뒤 : 하위에게 줄 업무를 고치거나 뺀다
  ③ 초안 뒤     : 통합 전에 초안을 직접 고치거나, 통합할 때 참고할 지시를 남긴다

기본은 꺼짐 — 끄면 그래프가 지금과 똑같이 흐른다 (core/workflow.py 의 분기는 state["step_mode"] 일 때만).
멈춤은 LangGraph interrupt 로 하므로 체크포인터가 있어야 한다 (app.py 는 SqliteSaver).
멈추는 노드에는 LLM 호출을 두지 않는다 — 재개하면 그 노드를 처음부터 다시 실행하기 때문 (방향 제안은 앞 노드가 한다).

이 파일은 '멈출 때 화면에 보낼 값'과 '사람의 답을 상태에 반영하는 법'만 맡는다 (순수 함수 → 테스트하기 쉬움).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.brief import DesiredState
from core.schemas import AgentAnswer

CHECK = "step_check"  # interrupt 값의 type
POINTS = {"brief": "① 브리프·방향", "plan": "② 팀장 업무 분배", "drafts": "③ 초안 (통합 전)"}


class DirectionOption(BaseModel):
    title: str = Field(description="방향 이름 (예: 'A: 기술 중심')")
    summary: str = Field(description="이 방향으로 하면 결과가 어떻게 달라지는지 1~2문장")


class DirectionOptions(BaseModel):
    """팀장이 사람에게 고르라고 내놓는 서로 다른 진행 방향 2~3개."""

    options: list[DirectionOption] = Field(description="서로 뚜렷이 다른 방향 2~3개")


def direction_messages(system_prompt: str, brief_text: str) -> list:
    system = "\n".join([
        system_prompt, "",
        "지금은 일을 나누기 전에 사용자에게 진행 방향을 물어보는 단계다.",
        "- 같은 목표를 이룰 수 있는, 서로 뚜렷이 다른 방향 2~3개를 제안한다 (예: 기술 중심 / 성과 중심 / 리더십 중심).",
        "- title 은 'A: ○○ 중심'처럼 짧게, summary 는 그 방향을 고르면 결과가 어떻게 달라지는지 1~2문장.",
    ])
    return [SystemMessage(system), HumanMessage(f"[업무 브리프]\n{brief_text}")]


# ─── 멈출 때 화면에 보낼 값 ─────────────────────────────────────────
def brief_payload(brief: DesiredState, options: list[DirectionOption], root: str) -> dict:
    return {"type": CHECK, "point": "brief", "root": root, "brief": brief.model_dump(),
            "options": [o.model_dump() for o in options]}


def plan_payload(root: str, chosen: list[tuple[str, str]], plan: AgentAnswer | None, children: list[str]) -> dict:
    return {"type": CHECK, "point": "plan", "root": root, "children": children,
            "assignments": [{"child_id": c, "subtask": t} for c, t in chosen],
            "analysis": plan.delegation.analysis if plan and plan.delegation else ""}


def drafts_payload(root: str, results: dict[str, AgentAnswer], leaves: list[str]) -> dict:
    return {"type": CHECK, "point": "drafts", "root": root,
            "drafts": [{"node_id": n, "content": results[n].content} for n in leaves if n in results]}


def is_check(value) -> bool:
    return isinstance(value, dict) and value.get("type") == CHECK


# ─── 사람의 답 → 상태 ──────────────────────────────────────────────
def apply_brief(brief: DesiredState, answer: dict) -> tuple[DesiredState, str]:
    """answer = {"brief": {...고친 브리프}?, "direction": "B: 성과 중심 — …"?}. (새 브리프, 기록 한 줄)"""
    answer = answer or {}
    new = DesiredState.model_validate(answer["brief"]) if answer.get("brief") else brief
    direction = (answer.get("direction") or "").strip()
    note = []
    if new != brief:
        note.append("브리프 고침")
    if direction:
        new = new.model_copy(update={"constraints": [*new.constraints, f"진행 방향(사용자 선택): {direction}"]})
        note.append(f"방향 선택 — {direction[:60]}")
    return new, " · ".join(note) or "그대로 승인"


def apply_plan(chosen: list[tuple[str, str]], answer: dict, valid: set[str]) -> tuple[list[tuple[str, str]], str]:
    """answer = {"assignments": [{"child_id", "subtask"}]} (뺀 하위는 목록에서 빠짐). 비거나 잘못되면 원래 분배."""
    rows = [(a.get("child_id"), (a.get("subtask") or "").strip()) for a in (answer or {}).get("assignments") or []]
    new = [(c, t) for c, t in rows if c in valid and t]
    if not new or new == chosen:
        return chosen, "그대로 승인"
    return new, f"업무 분배 고침 ({len(chosen)}명 → {len(new)}명)"


def apply_drafts(results: dict[str, AgentAnswer], answer: dict) -> tuple[dict[str, AgentAnswer], str, str]:
    """answer = {"edits": {에이전트 id: 고친 초안}, "note": "통합할 때 참고"}. (새 results, 통합 지시, 기록 한 줄)"""
    answer = answer or {}
    out = dict(results)
    edited = []
    for node_id, text in (answer.get("edits") or {}).items():
        if node_id in out and text and text.strip() != out[node_id].content.strip():
            old = out[node_id]
            out[node_id] = old.model_copy(update={"content": text, "verification_log": [*old.verification_log, "✏️ 사용자가 초안을 고침"]})
            edited.append(node_id)
    note = (answer.get("note") or "").strip()
    parts = ([f"초안 {len(edited)}개 고침"] if edited else []) + ([f"통합 지시 — {note[:60]}"] if note else [])
    return out, note, " · ".join(parts) or "그대로 승인"
