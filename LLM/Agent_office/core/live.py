"""실시간 진행 상태 — 워크플로우를 '백그라운드'에서 돌리고 화면은 1초마다 그 상태를 다시 그린다.

왜 백그라운드인가?
  Streamlit 은 사용자가 버튼·선택 등을 누르면 스크립트를 처음부터 다시 실행한다.
  워크플로우를 화면 스크립트 안에서 돌리면, 진행 중에 에이전트를 클릭하는 순간 작업이 끊긴다.
  그래서 작업은 별도 스레드(start_run)가 하고, 진행 상황은 run(dict)에 기록한다.
  화면은 이 run 을 읽어서 그리기만 하므로, 진행 중에도 자유롭게 에이전트를 눌러 볼 수 있다.

run(dict) 의 주요 키
  status     reviewing → awaiting_human → adopted / rejected   (실패: error)
             awaiting_human ⇄ followup_running (💬 결과에 대해 묻기·수정 요청을 백그라운드에서 답하는 중 — start_followup)
  done       스레드 종료 여부 (True 가 되면 화면이 '사람 결정' 단계를 보여 준다)
  statuses   {에이전트 id: (상태, 표시 문구)}  — 조직도 배지 (working / wait / done / revise)
  activity   {에이전트 id: 지금 하는 일}        — 에이전트 패널의 '진행 중' 표시
  drafting   {에이전트 id: 지금 쓰고 있는 초안}  — 토큰 단위로 늘어남. 형제 실무자들이 동시에 쓰므로 에이전트별로 따로 모은다
  steps      완료된 단계(AgentAnswer) 목록      — 타임라인
  assigned   {관리자 id: [업무를 받은 하위 id]}   — 📊 진행률이 실제로 일을 받은 사람만 세게 (core/progress.py)
  brief / tools / web / notes / error
  checkpoint 🔀 단계마다 확인에서 멈춘 곳 (status=step_check 일 때 — core/step_mode.py 의 payload)
  attempts   ✏️ 고쳐 다시 하기 전의 시도들 [{round, question, brief, steps, ended_at, why}] — 비교용
  followups  💬 결정 전 수정 대화 [FollowupTurn] — 끝날 때마다 실행 기록에 저장하고(app.py), 다시 켜면 restore_run 이 되살린다
  모델 호출 하나하나의 기록·개입 내역은 core/call_monitor.py (run_id 로 묶음)
"""

import threading
import time
import traceback
from collections.abc import Callable, Iterator
from datetime import datetime

from core import call_monitor, step_mode, usage
from core.llm_factory import NoAvailableModelError
from core.schemas import AgentAnswer

WORKING_LABELS = {
    "brief": "🎯 요청 정리 중", "tools": "📊 데이터 조회 중", "research": "🌐 자료 조사 중",
    "plan": "📋 업무 분배 중", "draft": "✍️ 작성 중", "review": "🔍 검토 중", "integrate": "🧩 통합 중",
    "critic": "⚖️ 비평 받는 중",
}
RETRY_LABELS = {"critic": "🔁 자가 교정 중", "harness": "🧪 오류 수정 중", "model": "🔄 모델 오류 — 다시 작성 중"}
KIND_LABELS = {"plan": "📋 업무 분배", "draft": "📝 초안", "review": "🔍 검토", "integrate": "🧩 통합"}

# 화면 위쪽 진행 단계 표시 — 워크플로우가 지나가는 큰 단계 (마지막 '결정'은 사람 몫)
STAGES = ["🎯 요청 정리", "🌐 자료 수집", "📋 업무 분배", "✍️ 작성", "🔍 검토·통합", "⚖️ 검증", "👤 결정"]
_KIND_STAGE = {"brief": 0, "tools": 1, "research": 1, "plan": 2, "draft": 3, "review": 4, "integrate": 4, "critic": 5}
_LABEL_STAGE = {
    **{label: _KIND_STAGE[kind] for kind, label in WORKING_LABELS.items()},
    RETRY_LABELS["critic"]: 5, RETRY_LABELS["harness"]: 5,
}


def stage_of(run: dict) -> int:
    """지금 몇 번째 단계인지 (STAGES 의 위치). 결정까지 끝나면 len(STAGES).

    여러 에이전트가 동시에 다른 단계에 있을 수 있으므로, 지금까지 도달한 가장 뒤 단계를 쓴다.
    """
    if run["status"] in ("adopted", "rejected"):
        return len(STAGES)
    if run["status"] in ("awaiting_human", FOLLOWUP_RUNNING):
        return len(STAGES) - 1
    reached = 1 if run.get("brief") is not None else 0
    for step in run["steps"]:
        reached = max(reached, _KIND_STAGE[step.kind])
    for label in run["activity"].values():
        reached = max(reached, _LABEL_STAGE.get(label, 0))
    return reached


def new_run(run_id: str, question: str, root_id: str, team_id: str = "") -> dict:
    return {
        "run_id": run_id, "team_id": team_id, "question": question, "root_id": root_id, "started_at": time.time(),
        "status": "reviewing", "done": False, "error": None,
        "statuses": {root_id: ("working", WORKING_LABELS["brief"])}, "activity": {root_id: WORKING_LABELS["brief"]},
        "drafting": {},
        "steps": [], "brief": None, "tools": [], "web": None, "notes": [],
        "decision": None, "lessons": [],
        # 후처리(여행 지도 등): post_busy=True 인 동안 화면이 계속 자동 갱신된다
        "post_busy": False, "route": None, "route_error": None, "trip_start": None,
        "saved_files": [],  # 채택해서 저장한 결과 파일 경로
        "harness_options": {},  # 업무 양식의 검증 옵션 (주식 검산표 등)
        "step_mode": False, "checkpoint": None, "attempts": [],
    }


def restore_run(row: dict, followups: list | None = None) -> dict:
    """저장소(runs.db)에 남은 '결정 대기' 실행을 화면용 run 으로 되살린다 (서버를 다시 켠 뒤).

    LangGraph 체크포인트가 파일(checkpoints.db)에 남아 있으므로 사람 결정으로 이어서 재개할 수 있다.
    브리프·근거 자료는 저장하지 않으므로 비어 있다.
    followups = 저장해 둔 💬 수정 대화 (수정본은 다시 채택 후보가 된다). 답을 기다리다 꺼졌으면 저장된 것까지만 — 대기 표시는 되살리지 않는다.
    """
    run = new_run(row["id"], row["question"], row["root_id"], row["team_id"])
    run["followups"] = list(followups or [])
    interrupted = row["status"] == FOLLOWUP_RUNNING  # 💬 추가 질문 중에 서버가 꺼짐 → 결정 대기로 되돌리고 알린다
    if interrupted:
        row = {**row, "status": "awaiting_human"}
        run["followup_error"] = FOLLOWUP_INTERRUPTED
    run.update(status=row["status"], done=True, steps=row["steps"], activity={}, decision=row.get("decision"),
               started_at=datetime.fromisoformat(row["created_at"]).timestamp(), restored=True)
    run["statuses"] = {s.node_id: step_status(s) for s in row["steps"]}
    if row["status"] == "awaiting_human":
        run["statuses"][row["root_id"]] = ("wait", "👤 결정 대기")
    return run


def step_status(step: AgentAnswer) -> tuple[str, str]:
    """완료된 단계 → 조직도 배지."""
    if step.kind == "plan":
        return "wait", "⏳ 하위 결과 대기"
    if step.kind == "draft":
        return "done", "📝 초안 완료"
    if step.review and step.review.verdict == "approve":
        return "done", f"✅ 승인 {step.review.score}점"
    return "revise", f"{'🧩 통합' if step.kind == 'integrate' else '✏️ 보완'} {step.review.score if step.review else ''}점"


def apply_event(run: dict, kind: str, payload) -> None:
    """워크플로우 이벤트 하나를 run 상태에 반영한다 (화면 그리기 없음 → 테스트하기 쉬움)."""
    statuses, activity = run["statuses"], run["activity"]
    if kind == "start":
        label = WORKING_LABELS.get(payload["kind"], "작업 중")
        statuses[payload["node"]] = ("working", label)
        activity[payload["node"]] = label
    elif kind == "delegated":
        statuses[payload["node"]] = ("wait", "⏳ 하위 결과 대기")
        activity[payload["node"]] = "⏳ 하위 에이전트 결과를 기다리는 중"
        run.setdefault("assigned", {})[payload["node"]] = list(payload["children"])
        for cid in payload["children"]:
            statuses[cid] = ("wait", "⏳ 대기")
            activity[cid] = "⏳ 차례를 기다리는 중"
    elif kind == "retry":
        label = RETRY_LABELS[payload["reason"]]
        statuses[payload["node"]] = ("working", label)
        activity[payload["node"]] = label
        run["drafting"].pop(payload["node"], None)  # 다시 쓰는 에이전트의 미리보기만 비운다 (다른 실무자는 계속 작성 중)
    elif kind == "token":
        node, text = payload
        run["drafting"][node] = run["drafting"].get(node, "") + text
    elif kind == "step":
        run["steps"].append(payload)
        statuses[payload.node_id] = step_status(payload)
        activity.pop(payload.node_id, None)
        run["drafting"].pop(payload.node_id, None)
    elif kind == "brief":
        run["brief"] = payload
    elif kind == "tools":
        run["tools"] = payload
    elif kind == "research":
        run["web"] = payload
    elif kind == "note":
        run["notes"].append(payload)
    elif kind == "interrupt":
        value = next((getattr(i, "value", None) for i in payload or ()), None) if isinstance(payload, (tuple, list)) else None
        if step_mode.is_check(value):  # 🔀 단계마다 확인 — 사람의 선택을 기다림 (결정 대기와 다름)
            run["status"], run["checkpoint"] = "step_check", value
            statuses[run["root_id"]] = ("wait", "🔀 확인 대기")
            return
        run["status"], run["checkpoint"] = "awaiting_human", None
        statuses[run["root_id"]] = ("wait", "👤 결정 대기")


def consume(run: dict, events: Iterator[tuple[str, object]], on_step: Callable[[dict], None] | None = None) -> None:
    """이벤트를 끝까지 소비하며 run 을 갱신한다. 실패하면 status=error 로 남긴다 (예외를 밖으로 던지지 않음)."""
    try:
        for kind, payload in events:
            apply_event(run, kind, payload)
            if kind == "step" and on_step:
                on_step(run)
    except NoAvailableModelError as e:
        run["status"], run["error"] = "error", str(e)
    except call_monitor.CallCancelled as e:
        run["status"], run["error"] = "error", f"⏹ {e}"
    except Exception as e:  # noqa: BLE001 — 백그라운드 스레드의 예외는 화면에 보여 줘야 한다
        run["status"], run["error"] = "error", f"진행 중 오류: {type(e).__name__}: {e}"
        run["notes"].append(traceback.format_exc()[-1500:])
    finally:
        run["activity"].clear()
        run["drafting"] = {}
        call_monitor.release(run["run_id"])  # 중단·건너뛰기는 이번 진행에만 — 결정 뒤 교훈 생성·재개는 정상으로
        run["done"] = True


def prepare_resume(run: dict, label: str = "▶ 이어서 진행") -> None:
    """멈춘 run(단계 확인·고쳐 다시)을 다시 '진행 중'으로 — start_run 으로 스레드를 띄우기 전에 부른다."""
    run.update(done=False, status="reviewing", error=None, checkpoint=None)
    run["statuses"][run["root_id"]] = ("working", label)
    run["activity"][run["root_id"]] = label


def archive_attempt(run: dict, why: str) -> dict:
    """✏️ 고쳐 다시 — 지금까지의 시도를 run["attempts"] 에 보관하고 화면 상태를 비운다 (비교용으로 남김)."""
    attempt = {
        "round": len(run["attempts"]) + 1, "question": run["question"], "brief": run.get("brief"),
        "steps": list(run["steps"]), "ended_at": time.time(), "why": why,
    }
    run["attempts"].append(attempt)
    run.update(steps=[], drafting={}, decision=None, error=None, checkpoint=None)
    run["statuses"] = {run["root_id"]: ("working", "✏️ 고쳐 다시")}
    run["activity"] = {run["root_id"]: "✏️ 고쳐 다시"}
    return attempt


# ─── 💬 결과에 대해 묻기·수정 요청 (결정 대기 중 백그라운드) ─────────────────────
# 예전에는 화면 스크립트 안에서 st.spinner 로 기다려서 그동안 사무실 애니메이션·호출 모니터가 멈추고
# 상태가 계속 '결정 대기'로만 보였다 (2026-11-05). 본 업무처럼 스레드로 돌리고 상태를 followup_running 으로 둔다.
FOLLOWUP_RUNNING = "followup_running"
FOLLOWUP_LABEL = "💬 추가 질문 답하는 중"
FOLLOWUP_INTERRUPTED = "💬 추가 질문이 중단됨 — 앱을 다시 켜서 답을 받지 못했습니다. 다시 보내 주세요."


def begin_followup(run: dict, request: str) -> None:
    """결정 대기 → 추가 질문 진행 중. 답하는 사람(최종 답변자)이 사무실에서 일하는 모습으로 보이게 한다."""
    root = run["root_id"]
    run.update(done=False, status=FOLLOWUP_RUNNING, followup_error="")
    run["followup_pending"] = {"request": request, "started_at": time.time()}
    run["statuses"][root] = ("working", FOLLOWUP_LABEL)
    run["activity"][root] = FOLLOWUP_LABEL


def end_followup(run: dict, turn=None, error: str = "") -> None:
    """추가 질문 끝 → 다시 결정 대기. turn 이 있으면 대화에 붙이고, 실패면 error 를 화면에 남긴다."""
    root = run["root_id"]
    run.pop("followup_pending", None)
    if turn is not None:
        run.setdefault("followups", []).append(turn)
    run["followup_error"] = error
    run["status"] = "awaiting_human"
    run["statuses"][root] = ("wait", "👤 결정 대기")
    run["activity"].pop(root, None)
    run["notice"] = "followup"
    run["done"] = True  # 마지막에 — 화면이 done 을 보고 결과를 다시 그린다


def recover_followup(run: dict, why: str = FOLLOWUP_INTERRUPTED) -> bool:
    """끊긴 추가 질문(서버를 다시 켬·스레드가 사라짐)을 결정 대기로 되돌린다. 되돌렸으면 True."""
    if run.get("status") != FOLLOWUP_RUNNING:
        return False
    thread = run.get("followup_thread")
    if thread is not None and thread.is_alive():
        return False
    end_followup(run, error=why)
    run.pop("notice", None)
    return True


def start_followup(run: dict, request: str, work: Callable[[], object], on_finish=None, agent_id: str = "") -> threading.Thread:
    """work() (= followup.respond(...)) 를 백그라운드에서 돌린다. 호출 모니터에 실시간으로 뜨고 바꾸기·중단도 된다."""
    begin_followup(run, request)

    def _target():
        turn, error = None, ""
        with usage.scope(team_id=run.get("team_id", ""), run_id=run["run_id"], agent_id=agent_id or run["root_id"],
                         purpose="결정 전 문의·수정"):
            try:
                turn = work()
            except NoAvailableModelError as e:
                error = f"답할 수 있는 모델이 없습니다: {e}"
            except call_monitor.CallCancelled as e:
                error = f"⏹ 추가 질문을 중단했습니다 ({e})"
            except Exception as e:  # noqa: BLE001 — 백그라운드 스레드의 예외는 화면에 보여 줘야 한다
                error = f"답변 중 오류: {type(e).__name__}: {e}"
            finally:
                call_monitor.release(run["run_id"])  # ⏹ 중단은 이번 질문에만 — 결정·다음 질문은 정상으로
                end_followup(run, turn, error)
                if on_finish:
                    try:
                        on_finish(run)
                    except Exception:  # noqa: BLE001 — 저장 실패가 화면을 막지 않게
                        pass

    thread = threading.Thread(target=_target, name=f"followup-{run['run_id'][:8]}", daemon=True)
    run["followup_thread"] = thread
    thread.start()
    return thread


def start_run(run: dict, events_factory: Callable[[], Iterator], on_step=None, on_finish=None) -> threading.Thread:
    """백그라운드 스레드에서 워크플로우를 실행한다. events_factory 는 stream_office(...) 를 만드는 함수."""

    def _target():
        # 이 실행 안의 모든 모델 호출 기록에 팀·실행 id 를 붙인다 (동시 실행 스레드에도 이어진다)
        with usage.scope(team_id=run.get("team_id", ""), run_id=run["run_id"]):
            consume(run, events_factory(), on_step)
            if on_finish:
                on_finish(run)

    thread = threading.Thread(target=_target, name=f"office-{run['run_id'][:8]}", daemon=True)
    thread.start()
    return thread
