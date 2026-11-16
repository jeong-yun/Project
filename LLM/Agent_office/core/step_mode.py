"""🔀 단계마다 확인 (step-by-step mode) — 켜면 워크플로우가 세 곳에서 멈추고 사람의 선택을 기다린다.

  ① 브리프 뒤   : 정리된 브리프를 고치고, 팀장이 내놓은 2~3가지 방향(A·B·C) 중 하나를 고르거나 직접 쓴다
  ② 팀장 분배 뒤 : 하위에게 줄 업무를 고치거나 뺀다
  ③ 초안 뒤     : 통합 전에 초안을 직접 고치거나, 통합할 때 참고할 지시를 남긴다

기본은 꺼짐 — 끄면 그래프가 지금과 똑같이 흐른다 (core/workflow.py 의 분기는 state["step_mode"] 일 때만).
멈춤은 LangGraph interrupt 로 하므로 체크포인터가 있어야 한다 (app.py 는 SqliteSaver).
멈추는 노드에는 LLM 호출을 두지 않는다 — 재개하면 그 노드를 처음부터 다시 실행하기 때문 (방향 제안은 앞 노드가 한다).

🧭 가정 확인 (point="assumptions") — 단계마다 확인을 꺼도, 요청 정리 모델이 '질문이 모호해 스스로 정한 가정'(brief.assumptions)을
냈으면 팀이 일을 시작하기 전에 한 번 멈춰 사람에게 보여 준다 (2026-11-16 — 가정이 틀린 채 끝까지 가서 사람이 다시 고치던 문제).
질문은 이미 나온 가정을 그대로 쓴다 (모델 호출 없음). 묻지 않는 경우는 core/workflow.py 의 asks_assumptions 참고.

이 파일은 '멈출 때 화면에 보낼 값'과 '사람의 답을 상태에 반영하는 법'만 맡는다 (순수 함수 → 테스트하기 쉬움).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.brief import DesiredState
from core.schemas import AgentAnswer

CHECK = "step_check"  # interrupt 값의 type
POINTS = {"assumptions": "🧭 시작 전 확인 (팀이 이해한 내용)", "brief": "① 브리프·방향", "plan": "② 팀장 업무 분배",
          "drafts": "③ 초안 (통합 전)"}
MAX_ASSUMPTIONS = 3  # 🧭 가정 확인에서 묻는 최대 개수 (나머지는 그대로 둔다)
DECIDED = "사람이 정한 해석 (목표·완료 기준보다 우선)"  # 고쳐 쓴 가정이 들어가는 브리프 제약 앞머리


class DirectionOption(BaseModel):
    title: str = Field(description="방향 이름 (예: 'A: 범위를 넓게')")
    summary: str = Field(description="이 방향으로 하면 결과가 어떻게 달라지는지 1~2문장")


class DirectionOptions(BaseModel):
    """팀장이 사람에게 고르라고 내놓는 서로 다른 진행 방향 2~3개."""

    options: list[DirectionOption] = Field(description="서로 뚜렷이 다른 방향 2~3개")


def direction_messages(system_prompt: str, brief_text: str) -> list:
    system = "\n".join([
        system_prompt, "",
        "지금은 일을 나누기 전에 사용자에게 진행 방향을 물어보는 단계다.",
        "- 같은 목표를 이룰 수 있는, 서로 뚜렷이 다른 방향 2~3개를 제안한다 (예: 범위를 넓게 / 한 가지를 깊게 / 빠르게 핵심만).",
        "- title 은 'A: 범위를 넓게'처럼 짧게, summary 는 그 방향을 고르면 결과가 어떻게 달라지는지 1~2문장.",
    ])
    return [SystemMessage(system), HumanMessage(f"[업무 브리프]\n{brief_text}")]


# ─── 멈출 때 화면에 보낼 값 ─────────────────────────────────────────
def brief_payload(brief: DesiredState, options: list[DirectionOption], root: str) -> dict:
    return {"type": CHECK, "point": "brief", "root": root, "brief": brief.model_dump(),
            "options": [o.model_dump() for o in options]}


def new_assumptions(brief: DesiredState, given: DesiredState | None) -> list[str]:
    """사람에게 물을 가정 — 요청 정리 모델이 새로 낸 것만 (사람이 직접 적은 가정은 묻지 않는다)."""
    mine = set((given.assumptions if given else None) or [])
    return [a for a in brief.assumptions if a.strip() and a not in mine]


def assumptions_payload(pending: list[str], root: str) -> dict:
    """화면에 보낼 값 — 앞에서부터 MAX_ASSUMPTIONS 개만 묻고, 나머지 개수(more)는 '그대로 둔다'고 알린다."""
    asked = pending[:MAX_ASSUMPTIONS]
    return {"type": CHECK, "point": "assumptions", "root": root, "assumptions": asked, "more": len(pending) - len(asked)}


def plan_payload(root: str, chosen: list[tuple[str, str]], plan: AgentAnswer | None, children: list[str]) -> dict:
    return {"type": CHECK, "point": "plan", "root": root, "children": children,
            "assignments": [{"child_id": c, "subtask": t} for c, t in chosen],
            "analysis": plan.delegation.analysis if plan and plan.delegation else ""}


def drafts_payload(root: str, results: dict[str, AgentAnswer], leaves: list[str]) -> dict:
    return {"type": CHECK, "point": "drafts", "root": root,
            "drafts": [{"node_id": n, "content": results[n].content} for n in leaves if n in results]}


def is_check(value) -> bool:
    return isinstance(value, dict) and value.get("type") == CHECK


def wait_label(run: dict) -> str:
    """멈춘 업무의 상태 글 — 🧭 가정 확인은 단계마다 확인을 켜지 않은 업무에도 오므로 따로 부른다."""
    return "🧭 시작 전 확인 대기" if (run.get("checkpoint") or {}).get("point") == "assumptions" else "🔀 확인 대기"


# ─── 사람의 답 → 상태 ──────────────────────────────────────────────
def apply_brief(brief: DesiredState, answer: dict) -> tuple[DesiredState, str]:
    """answer = {"brief": {...고친 브리프}?, "direction": "B: 성과 중심 — …"?, "fixes": {가정: 고쳐 쓴 글}?}. (새 브리프, 기록 한 줄)

    단계마다 확인을 켜면 🧭 가정 확인을 따로 멈추지 않고 이 ① 화면에서 함께 묻는다 (fixes — apply_assumptions 와 같은 모양).
    """
    answer = answer or {}
    edited = DesiredState.model_validate(answer["brief"]) if answer.get("brief") else brief
    direction = (answer.get("direction") or "").strip()
    note = []
    if edited != brief:
        note.append("브리프 고침")
    new, fixed = apply_assumptions(edited, answer)
    if new is not edited:
        note.append(fixed)
    if direction:
        new = new.model_copy(update={"constraints": [*new.constraints, f"진행 방향(사용자 선택): {direction}"]})
        note.append(f"방향 선택 — {direction[:60]}")
    return new, " · ".join(note) or "그대로 승인"


def apply_assumptions(brief: DesiredState, answer: dict) -> tuple[DesiredState, str]:
    """answer = {"fixes": {가정 원문: 고쳐 쓴 글}} (비었거나 원문과 같으면 '이대로'). (새 브리프, 기록 한 줄)

    고쳐 쓴 가정은 가정 목록에서 빼고, 모든 에이전트가 받는 브리프 제약에 '사람이 정한 해석'으로 넣는다
    (목표·완료 기준이 틀린 가정을 바탕으로 쓰였을 수 있어 그보다 우선한다고 적는다). 모두 이대로면 브리프를 바꾸지 않는다.
    """
    fixes = {k: (v or "").strip() for k, v in ((answer or {}).get("fixes") or {}).items()}
    fixed = [(a, fixes[a]) for a in brief.assumptions if fixes.get(a) and fixes[a] != a.strip()]
    if not fixed:
        return brief, "모두 이대로 진행"
    gone = {a for a, _ in fixed}
    new = brief.model_copy(update={
        "assumptions": [a for a in brief.assumptions if a not in gone],
        "constraints": [*brief.constraints, *(f"{DECIDED}: {text}" for _, text in fixed)],
    })
    return new, f"가정 {len(fixed)}개 고침 — " + " / ".join(text[:40] for _, text in fixed)


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
