"""실시간 진행 상태 — 워크플로우를 '백그라운드'에서 돌리고 화면은 1초마다 그 상태를 다시 그린다.

왜 백그라운드인가?
  Streamlit 은 사용자가 버튼·선택 등을 누르면 스크립트를 처음부터 다시 실행한다.
  워크플로우를 화면 스크립트 안에서 돌리면, 진행 중에 에이전트를 클릭하는 순간 작업이 끊긴다.
  그래서 작업은 별도 스레드(start_run)가 하고, 진행 상황은 run(dict)에 기록한다.
  화면은 이 run 을 읽어서 그리기만 하므로, 진행 중에도 자유롭게 에이전트를 눌러 볼 수 있다.

run(dict) 의 주요 키
  status     reviewing → awaiting_human → adopted / rejected   (실패: error)
  done       스레드 종료 여부 (True 가 되면 화면이 '사람 결정' 단계를 보여 준다)
  statuses   {에이전트 id: (상태, 표시 문구)}  — 조직도 배지 (working / wait / done / revise)
  activity   {에이전트 id: 지금 하는 일}        — 에이전트 패널의 '진행 중' 표시
  drafting   {에이전트 id: 지금 쓰고 있는 초안}  — 토큰 단위로 늘어남. 형제 실무자들이 동시에 쓰므로 에이전트별로 따로 모은다
  steps      완료된 단계(AgentAnswer) 목록      — 타임라인
  brief / tools / web / notes / error
"""

import threading
import time
import traceback
from collections.abc import Callable, Iterator

from core.llm_factory import NoAvailableModelError
from core.schemas import AgentAnswer

WORKING_LABELS = {
    "brief": "🎯 요청 정리 중", "tools": "📊 데이터 조회 중", "research": "🌐 자료 조사 중",
    "plan": "📋 업무 분배 중", "draft": "✍️ 작성 중", "review": "🔍 검토 중", "integrate": "🧩 통합 중",
    "critic": "⚖️ 비평 받는 중",
}
RETRY_LABELS = {"critic": "🔁 자가 교정 중", "harness": "🧪 오류 수정 중", "model": "🔄 모델 오류 — 다시 작성 중"}
KIND_LABELS = {"plan": "📋 업무 분배", "draft": "📝 초안", "review": "🔍 검토", "integrate": "🧩 통합"}


def new_run(run_id: str, question: str, root_id: str) -> dict:
    return {
        "run_id": run_id, "question": question, "root_id": root_id, "started_at": time.time(),
        "status": "reviewing", "done": False, "error": None,
        "statuses": {root_id: ("working", WORKING_LABELS["brief"])}, "activity": {root_id: WORKING_LABELS["brief"]},
        "drafting": {},
        "steps": [], "brief": None, "tools": [], "web": None, "notes": [],
        "decision": None, "lessons": [],
        # 후처리(여행 지도 등): post_busy=True 인 동안 화면이 계속 자동 갱신된다
        "post_busy": False, "route": None, "route_error": None, "trip_start": None,
        "saved_files": [],  # 채택해서 저장한 결과 파일 경로
        "harness_options": {},  # 업무 양식의 검증 옵션 (주식 검산표 등)
    }


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
        run["status"] = "awaiting_human"
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
    except Exception as e:  # noqa: BLE001 — 백그라운드 스레드의 예외는 화면에 보여 줘야 한다
        run["status"], run["error"] = "error", f"진행 중 오류: {type(e).__name__}: {e}"
        run["notes"].append(traceback.format_exc()[-1500:])
    finally:
        run["activity"].clear()
        run["drafting"] = {}
        run["done"] = True


def start_run(run: dict, events_factory: Callable[[], Iterator], on_step=None, on_finish=None) -> threading.Thread:
    """백그라운드 스레드에서 워크플로우를 실행한다. events_factory 는 stream_office(...) 를 만드는 함수."""

    def _target():
        consume(run, events_factory(), on_step)
        if on_finish:
            on_finish(run)

    thread = threading.Thread(target=_target, name=f"office-{run['run_id'][:8]}", daemon=True)
    thread.start()
    return thread
