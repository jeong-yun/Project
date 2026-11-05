"""⏱ 호출 모니터 — 모델 호출 하나하나를 기록하고, 진행 중인 호출을 사람이 끊거나 다른 모델로 바꾸게 한다.

왜 필요한가 (2026-11-03 실측)
  포트폴리오 업무가 30분 넘게 걸렸는데 화면에는 '오류 수정 중·비평 받는 중'만 10~20분 보였다.
  실제로는 Gemini 호출 하나가 시간 제한 없이 10분 가까이 멈춰 있었다. → 어느 호출이 왜 느린지 보이고, 손쓸 수 있어야 한다.

무엇을 기록하나 (CallRecord — 호출 1회 = 1줄)
  누가(에이전트·비평가·요청 정리…) · 무슨 일(초안·검토·비평·하네스 재작성…) · 어느 모델 · 언제 줄을 섰고(동시 호출 자리 대기)
  언제 시작해서 언제 끝났는지 · 첫 토큰까지 걸린 시간(스트리밍) · 토큰 수 · 상태(대기/진행/작성 중/재시도 대기/끝/실패/중단/바꿈)
  실행(run) id 는 usage.scope 의 run_id 로 묶는다 (core/live.py 가 실행 스레드에서 연다).

사람이 손쓰는 방법 (화면: app.py ⏱ 호출 모니터)
  switch(call_id, 모델)  : 진행 중인 호출을 끊고 고른 모델로 그 단계를 다시 (남은 업무에도 쓰기 선택 가능)
  skip(run_id, "critic" | "harness") : 이번 실행의 비평·하네스 재시도를 건너뜀 (진행 중인 그 호출도 끊음)
  stop(run_id)           : 새 호출을 막고 진행 중인 호출을 끊는다 → 워크플로우가 지금까지의 결과로 마무리
  모든 개입은 interventions 에 '누가·무엇을·언제'로 남는다.

어떻게 끊나 (협조적 취소)
  모델 호출은 보조 스레드에서 돌리고, 원래 스레드는 '결과' 또는 '취소' 중 먼저 오는 것을 기다린다.
  취소되면 원래 스레드는 바로 CallCancelled 를 받고 (반쯤 받은 출력은 버림), 보조 스레드는
    - claude_cli : 프로세스를 강제 종료 (attach_process 로 등록한 claude -p)
    - 스트리밍   : 다음 조각에서 멈추고 스트림을 닫는다 (HTTP 연결 닫힘)
    - 그 밖      : 응답이 올 때까지 기다렸다 버린다 (요청 하나는 끝까지 갈 수 있음 — 토큰은 쓰일 수 있다)
  동시 호출 자리 대기·재시도 대기 중에도 취소를 바로 알아챈다 (provider_slot·sleep 이 check_cancel 을 부름).

부담: 호출마다 dict 1줄 + 보조 스레드 1개 — LLM 호출 시간에 비하면 무시할 만하다. 저장은 실행이 끝날 때 한 번 (Storage.save_monitor).
"""

import contextvars
import itertools
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

from core import usage

# ─── 무슨 일을 하는 호출인가 ──────────────────────────────────────────
KIND_LABELS = {
    "brief": "🎯 요청 정리", "search": "🌐 검색어", "directions": "🔀 방향 제안", "plan": "📋 업무 분배",
    "draft": "✍️ 초안", "review": "🔍 검토·통합", "critic": "⚖️ 비평", "self_correct": "🔁 자가 교정",
    "harness": "🧪 하네스 재작성", "lessons": "📚 교훈", "followup": "💬 문의", "other": "기타",
}
_SCHEMA_KIND = {
    "Delegation": "plan", "ReviewResult": "review", "ReviewEdits": "review", "CriticResult": "critic",
    "DesiredState": "brief", "SearchPlan": "search", "LessonSet": "lessons", "FollowupResult": "followup",
    "DirectionOptions": "directions",
}
STATUS_LABELS = {
    "queued": "⏳ 자리 대기", "running": "▶ 응답 기다림", "streaming": "✍️ 받는 중", "retry_wait": "⏸ 재시도 대기",
    "done": "✅ 끝", "failed": "❌ 실패", "cancelled": "⛔ 중단", "switched": "⏭ 모델 바꿈",
}
ACTIVE = ("queued", "running", "streaming", "retry_wait")
STALL_SEC = 60.0  # 이 시간 동안 아무 응답(토큰)이 없으면 '⚠️ 응답 없음' (settings.yaml call_monitor.stall_sec)
SKIP_KINDS = {"critic": ("critic", "self_correct"), "harness": ("harness",)}  # 건너뛰면 함께 끊을 호출 종류


class CallCancelled(RuntimeError):
    """사람이 호출을 끊음. switch_to 가 있으면 '그 모델로 다시', 없으면 reason(skip·stop·cancel)대로 진행을 멈춘다."""

    transient = False  # llm_factory 가 '잠깐 뒤 다시'로 보지 않게

    def __init__(self, reason: str, switch_to=None, message: str = ""):
        super().__init__(message or {"stop": "사용자가 실행을 중단", "skip": "사용자가 이 단계를 건너뜀"}.get(reason, "사용자가 호출을 끊음"))
        self.reason, self.switch_to = reason, switch_to


_ids = itertools.count(1)


@dataclass
class CallRecord:
    id: str
    run_id: str
    who: str  # 에이전트 이름 · '비평가' · '요청 정리' …
    kind: str  # KIND_LABELS 의 키
    provider: str
    model: str
    agent_id: str = ""
    team_id: str = ""
    attempt: int = 0  # 같은 모델로 다시 한 횟수 (0 = 첫 시도)
    is_fallback: bool = False
    round: int = 1  # 몇 번째 시도(✏️ 고쳐 다시)의 호출인가
    queued_at: float = 0.0
    started_at: float | None = None  # 동시 호출 자리를 잡은 시각
    first_token_at: float | None = None
    last_activity: float | None = None
    ended_at: float | None = None
    status: str = "queued"
    input_tokens: int | None = None
    output_tokens: int | None = None
    chars: int = 0  # 스트리밍으로 받은 글자 수
    error: str = ""
    note: str = ""  # '사용자가 gpt-5 로 바꿈' · '5초 뒤 다시' 같은 한 줄
    retry_wait: float = 0.0  # 재시도 전에 기다린 시간 (초)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._wake: queue.Queue | None = None
        self._cancel: CallCancelled | None = None
        self._proc = None

    # ─── 상태 바꾸기 (호출하는 스레드) ─────────────────────────────
    @property
    def cancelled(self) -> bool:
        return self._cancel is not None

    def running(self) -> None:
        now = time.time()
        self.started_at, self.last_activity, self.status = now, now, "running"

    def token(self, text: str) -> None:
        now = time.time()
        if self.first_token_at is None:
            self.first_token_at = now
            self.status = "streaming"
        self.last_activity = now
        self.chars += len(text)

    def finish(self, input_tokens=None, output_tokens=None) -> None:
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self._end("done")

    def fail(self, error: Exception | str) -> None:
        self.error = (f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error))[:300]
        self._end("failed")

    def waiting_retry(self, seconds: float) -> None:
        """실패 후 같은 모델로 다시 하기 전 기다리는 중 (끝나면 fail 로 마감)."""
        self.status, self.retry_wait, self.note = "retry_wait", seconds, f"{seconds:.0f}초 뒤 같은 모델로 다시"

    def _end(self, status: str) -> None:
        if self.ended_at is None:
            self.ended_at, self.status = time.time(), status
        self._proc = None

    # ─── 끊기 (화면 스레드) ────────────────────────────────────────
    def cancel(self, reason: str, switch_to=None, note: str = "") -> bool:
        """진행 중이면 끊는다. 이미 끝났으면 False."""
        with self._lock:
            if self.ended_at is not None or self._cancel is not None:
                return False
            self._cancel = CallCancelled(reason, switch_to)
            self.note = note or self.note
            wake, proc = self._wake, self._proc
        if wake is not None:
            wake.put(("cancel", None))
        if proc is not None:
            kill_process(proc)
        return True

    def check(self) -> None:
        if self._cancel is not None:
            raise self._cancel

    def attach(self, proc) -> None:
        with self._lock:
            self._proc = proc
            cancelled = self._cancel is not None
        if cancelled:
            kill_process(proc)

    # ─── 보조 스레드에서 실행 ───────────────────────────────────────
    def run(self, fn: Callable[[], object]) -> object:
        """fn() 을 보조 스레드에서 돌리고 결과 또는 취소를 기다린다."""
        wake: queue.Queue = queue.Queue()
        with self._lock:
            self._wake = wake
        self.check()
        token = _current.set(self)
        ctx = contextvars.copy_context()  # usage.scope·LangGraph 설정이 보조 스레드에도 이어지게
        _current.reset(token)

        def work() -> None:
            try:
                wake.put(("done", fn()))
            except BaseException as e:  # noqa: BLE001 — 원래 스레드에서 다시 올린다
                wake.put(("error", e))

        threading.Thread(target=ctx.run, args=(work,), name=f"call-{self.id}", daemon=True).start()
        kind, value = wake.get()
        if kind == "cancel" or self._cancel is not None:
            raise self._cancel  # 늦게 온 결과는 버린다
        if kind == "error":
            raise value
        return value

    def stream(self, make_iter: Callable[[], Iterator]) -> Iterator:
        """make_iter() 의 조각을 보조 스레드에서 받아 넘긴다. 끊기면 다음 조각에서 멈추고 스트림을 닫는다."""
        wake: queue.Queue = queue.Queue()
        stop = threading.Event()
        with self._lock:
            self._wake = wake
        self.check()
        token = _current.set(self)
        ctx = contextvars.copy_context()
        _current.reset(token)

        def pump() -> None:
            it = None
            try:
                it = iter(make_iter())
                for item in it:
                    if stop.is_set():
                        break
                    wake.put(("item", item))
                wake.put(("end", None))
            except BaseException as e:  # noqa: BLE001
                wake.put(("error", e))
            finally:
                close = getattr(it, "close", None)
                if close is not None:
                    try:
                        close()  # 스트림 닫기 → HTTP 연결도 닫힌다
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=ctx.run, args=(pump,), name=f"call-{self.id}", daemon=True).start()
        try:
            while True:
                kind, value = wake.get()
                if self._cancel is not None or kind == "cancel":
                    raise self._cancel
                if kind == "item":
                    yield value
                elif kind == "end":
                    return
                else:
                    raise value
        finally:
            stop.set()

    # ─── 화면용 값 ─────────────────────────────────────────────────
    def waited(self, now: float | None = None) -> float:
        """동시 호출 자리를 기다린 시간 + 재시도 전 기다린 시간."""
        start = self.started_at if self.started_at is not None else (self.ended_at or now or time.time())
        return max(0.0, start - self.queued_at) + self.retry_wait

    def model_time(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at or now or time.time()
        if self.status == "retry_wait":
            end = self.last_activity or end
        return max(0.0, end - self.started_at)

    def ttft(self) -> float | None:
        return None if self.first_token_at is None or self.started_at is None else self.first_token_at - self.started_at

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_FIELD_NAMES = {f.name for f in fields(CallRecord)}


def record_from(data: dict) -> CallRecord:
    return CallRecord(**{k: v for k, v in data.items() if k in _FIELD_NAMES})


def kill_process(proc) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:  # noqa: BLE001 — 이미 끝난 프로세스
        pass


# ─── 실행(run)마다 사람이 정한 것 ────────────────────────────────────
@dataclass
class RunControl:
    stopped: bool = False
    skips: set[str] = field(default_factory=set)
    overrides: dict = field(default_factory=dict)  # who → ModelSpec (남은 업무에도 이 모델)
    round: int = 1
    interventions: list[dict] = field(default_factory=list)


_lock = threading.Lock()
_calls: dict[str, list[CallRecord]] = {}
_by_id: dict[str, CallRecord] = {}
_controls: dict[str, RunControl] = {}
_current: contextvars.ContextVar[CallRecord | None] = contextvars.ContextVar("current_call", default=None)
_purpose: contextvars.ContextVar[str] = contextvars.ContextVar("call_purpose", default="")
KEEP_RUNS = 40  # 메모리에 남길 실행 수 (지난 업무는 저장소에서 읽는다)
KEEP_LOOSE = 60  # 실행 밖 호출(🏠 뉴스 요약 등)은 최근 것만


def control(run_id: str) -> RunControl:
    with _lock:
        return _controls.setdefault(run_id, RunControl())


@contextmanager
def purpose(kind: str) -> Iterator[None]:
    """with call_monitor.purpose("critic"): … — 안쪽 호출을 그 종류로 기록한다 (워크플로우가 붙임)."""
    token = _purpose.set(kind)
    try:
        yield
    finally:
        _purpose.reset(token)


def current() -> CallRecord | None:
    return _current.get()


def begin(spec, who: str, attempt: int = 0, is_fallback: bool = False, schema: str = "", streaming: bool = False) -> CallRecord:
    """호출 기록 시작 (동시 호출 자리를 잡기 전). 이 실행이 중단됐으면 바로 CallCancelled."""
    run_id = usage.current("run_id")
    ctl = control(run_id) if run_id else None
    if ctl is not None and ctl.stopped:
        raise CallCancelled("stop")
    kind = _purpose.get() or _SCHEMA_KIND.get(schema) or ("draft" if streaming else "other")
    if ctl is not None and kind in {k for s in ctl.skips for k in SKIP_KINDS.get(s, ())}:
        raise CallCancelled("skip")
    rec = CallRecord(
        id=f"c{next(_ids)}", run_id=run_id, who=who, kind=kind, provider=spec.provider, model=spec.model,
        agent_id=usage.current("agent_id"), team_id=usage.current("team_id"), attempt=attempt, is_fallback=is_fallback,
        round=ctl.round if ctl else 1, queued_at=time.time(),
    )
    with _lock:
        bucket = _calls.setdefault(run_id, [])
        bucket.append(rec)
        _by_id[rec.id] = rec
        if not run_id and len(bucket) > KEEP_LOOSE:
            for old in bucket[:-KEEP_LOOSE]:
                _by_id.pop(old.id, None)
            del bucket[:-KEEP_LOOSE]
        if len(_calls) > KEEP_RUNS:
            for old_run in [r for r in _calls if r and not any(c.status in ACTIVE for c in _calls[r])][: len(_calls) - KEEP_RUNS]:
                for old in _calls.pop(old_run):
                    _by_id.pop(old.id, None)
                _controls.pop(old_run, None)
    return rec


@contextmanager
def active(rec: CallRecord) -> Iterator[CallRecord]:
    """이 호출을 '지금 호출'로 둔다 — 안쪽의 provider_slot·sleep·claude_cli 가 취소를 알아챈다."""
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)


def check_cancel() -> None:
    """지금 호출이 끊겼으면 CallCancelled (동시 호출 자리 대기·재시도 대기 중에 부른다)."""
    rec = _current.get()
    if rec is not None:
        rec.check()


def wait(seconds: float) -> None:
    """끊기면 바로 깨는 sleep (지금 호출이 없으면 그냥 sleep)."""
    rec = _current.get()
    if rec is None:
        time.sleep(seconds)
        return
    wake: queue.Queue = queue.Queue()
    with rec._lock:
        rec._wake = wake
    rec.check()
    try:
        wake.get(timeout=seconds)
    except queue.Empty:
        return
    rec.check()


def attach_process(proc) -> None:
    """claude_cli 가 띄운 프로세스를 지금 호출에 등록 — 끊으면 프로세스를 강제 종료한다."""
    rec = _current.get()
    if rec is not None:
        rec.attach(proc)


def prefer(candidates: list, who: str) -> list:
    """사람이 '남은 업무에도 이 모델'로 정한 모델을 맨 앞에 둔다 (나머지는 대안)."""
    run_id = usage.current("run_id")
    spec = control(run_id).overrides.get(who) if run_id else None
    if spec is None:
        return candidates
    return [(spec, False), *[(s, True) for s, _ in candidates if s != spec]]


# ─── 화면이 부르는 함수 ──────────────────────────────────────────────
def calls(run_id: str) -> list[CallRecord]:
    with _lock:
        return list(_calls.get(run_id, []))


def get(call_id: str) -> CallRecord | None:
    return _by_id.get(call_id)


def in_flight(run_id: str) -> list[CallRecord]:
    return [c for c in calls(run_id) if c.status in ACTIVE]


def log(run_id: str, what: str, detail: str = "", who: str = "사용자") -> dict:
    entry = {"ts": time.time(), "who": who, "what": what, "detail": detail}
    control(run_id).interventions.append(entry)
    return entry


def interventions(run_id: str) -> list[dict]:
    return list(control(run_id).interventions)


def switch(call_id: str, spec, for_rest: bool = False) -> bool:
    """진행 중인 호출을 끊고 spec 모델로 그 단계를 다시. for_rest = 이 실행의 남은 같은 담당(who) 호출에도."""
    rec = get(call_id)
    if rec is None:
        return False
    label = f"{spec.provider}:{spec.model}"
    if for_rest:
        control(rec.run_id).overrides[rec.who] = spec
    ok = rec.cancel("switch", spec, note=f"사용자가 {label} 로 바꿈")
    if ok or for_rest:
        log(rec.run_id, "⏭ 모델 바꿈",
            f"{rec.who} · {KIND_LABELS.get(rec.kind, rec.kind)} · {rec.provider}:{rec.model} → {label}"
            + (" (남은 업무에도)" if for_rest else "") + ("" if ok else " — 그 호출은 이미 끝나 다음 호출부터"))
    return ok


def skip(run_id: str, what: str) -> int:
    """이번 실행의 비평(critic)·하네스 재시도(harness)를 건너뛴다. 끊은 진행 중 호출 수."""
    control(run_id).skips.add(what)
    kinds = SKIP_KINDS.get(what, ())
    n = sum(c.cancel("skip", note="사용자가 건너뜀") for c in in_flight(run_id) if c.kind in kinds)
    log(run_id, "⛔ 건너뛰기", {"critic": "비평·자가 교정", "harness": "하네스 재시도"}.get(what, what) + f" (끊은 호출 {n}개)")
    return n


def skipped(what: str, run_id: str | None = None) -> bool:
    run_id = usage.current("run_id") if run_id is None else run_id
    return bool(run_id) and what in control(run_id).skips


def stop(run_id: str, why: str = "지금까지 결과로 마무리") -> int:
    control(run_id).stopped = True
    n = sum(c.cancel("stop", note="사용자가 중단") for c in in_flight(run_id))
    log(run_id, "⏹ 중단", f"{why} (끊은 호출 {n}개)")
    return n


def stopped(run_id: str | None = None) -> bool:
    run_id = usage.current("run_id") if run_id is None else run_id
    return bool(run_id) and control(run_id).stopped


def release(run_id: str) -> None:
    """실행 스트림이 멈춘 뒤 (결정 대기·단계 확인) — 중단·건너뛰기 표시를 푼다. 다음 단계(교훈·재개)는 정상으로."""
    ctl = control(run_id)
    ctl.stopped = False
    ctl.skips.clear()


def new_round(run_id: str) -> int:
    """✏️ 고쳐 다시 — 다음 호출부터 새 시도 번호. 바꾼 모델(남은 업무에도)은 그대로 둔다."""
    ctl = control(run_id)
    ctl.round += 1
    ctl.stopped = False
    ctl.skips.clear()
    return ctl.round


# ─── 표시 ──────────────────────────────────────────────────────────
def _fmt(sec: float | None) -> str:
    if sec is None:
        return "-"
    sec = int(round(sec))
    return f"{sec // 60}분 {sec % 60:02d}초" if sec >= 60 else f"{sec}초"


def flags(rec: CallRecord, now: float | None = None, stall_sec: float = STALL_SEC) -> list[str]:
    now = now or time.time()
    out = []
    if rec.status in ("running", "streaming"):
        idle = now - (rec.last_activity or rec.started_at or now)
        if idle >= stall_sec:
            out.append(f"⚠️ 응답 없음 {max(1, int(idle // 60))}분" if idle >= 60 else f"⚠️ 응답 없음 {int(idle)}초")
    if rec.attempt:
        out.append(f"🔁 재시도 {rec.attempt}회")
    if rec.is_fallback:
        out.append("↪️ 대안 모델로")
    if rec.status == "queued" and now - rec.queued_at >= 1:
        out.append("⏳ 동시 호출 자리 대기")
    elif rec.started_at is not None and rec.started_at - rec.queued_at >= 1:
        out.append(f"⏳ 자리 대기 {_fmt(rec.started_at - rec.queued_at)}")
    if rec.status == "retry_wait":
        out.append(f"⏸ {rec.note}")
    if rec.status in ("switched", "cancelled") and rec.note:
        out.append(rec.note)
    return out


def elapsed(rec: CallRecord, now: float | None = None) -> float:
    now = now or time.time()
    return (rec.ended_at or now) - rec.queued_at


def rows(records: list[CallRecord], names: dict[str, str] | None = None, now: float | None = None,
         stall_sec: float = STALL_SEC) -> list[dict]:
    """표 한 줄씩 (최근 시작 순). 진행 중인 호출의 경과 시간은 부를 때마다 늘어난다."""
    now = now or time.time()
    out = []
    for r in sorted(records, key=lambda r: r.queued_at, reverse=True):
        out.append({
            "id": r.id, "상태": STATUS_LABELS.get(r.status, r.status), "누가": r.who,
            "일": KIND_LABELS.get(r.kind, r.kind), "모델": f"{r.provider}:{r.model}",
            "경과": _fmt(elapsed(r, now)), "대기": _fmt(r.waited(now)), "모델 시간": _fmt(r.model_time(now)),
            "첫 토큰": _fmt(r.ttft()), "토큰(입/출)": f"{r.input_tokens or 0:,}/{r.output_tokens or 0:,}" if r.input_tokens or r.output_tokens else (f"{r.chars:,}자" if r.chars else "-"),
            "표시": " · ".join(flags(r, now, stall_sec)), "시도": r.round,
        })
    return out


def summary(records: list[CallRecord], now: float | None = None) -> dict:
    """실행 뒤 요약 — 단계별·모델별 시간, 가장 느린 호출, 기다린 시간 vs 모델 시간."""
    now = now or time.time()
    by_kind: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    for r in records:
        for table, key in ((by_kind, KIND_LABELS.get(r.kind, r.kind)), (by_model, f"{r.provider}:{r.model}")):
            row = table.setdefault(key, {"호출": 0, "모델 시간(초)": 0.0, "대기(초)": 0.0, "실패·중단": 0, "출력 토큰": 0})
            row["호출"] += 1
            row["모델 시간(초)"] += r.model_time(now)
            row["대기(초)"] += r.waited(now)
            row["실패·중단"] += r.status in ("failed", "cancelled", "switched")
            row["출력 토큰"] += r.output_tokens or 0
    for table in (by_kind, by_model):
        for row in table.values():
            row["모델 시간(초)"] = round(row["모델 시간(초)"], 1)
            row["대기(초)"] = round(row["대기(초)"], 1)
    slowest = sorted(records, key=lambda r: r.model_time(now), reverse=True)[:5]
    wall = (max((r.ended_at or now) for r in records) - min(r.queued_at for r in records)) if records else 0.0
    return {
        "calls": len(records), "wall_sec": round(wall, 1),
        "model_sec": round(sum(r.model_time(now) for r in records), 1),
        "wait_sec": round(sum(r.waited(now) for r in records), 1),
        "by_kind": by_kind, "by_model": by_model,
        "slowest": [{"누가": r.who, "일": KIND_LABELS.get(r.kind, r.kind), "모델": f"{r.provider}:{r.model}",
                     "모델 시간": _fmt(r.model_time(now)), "상태": STATUS_LABELS.get(r.status, r.status)} for r in slowest],
    }


def gantt_html(records: list[CallRecord], now: float | None = None, max_rows: int = 40, stall_sec: float = STALL_SEC) -> str:
    """가로 막대 시간표 (HTML) — 회색 = 자리·재시도 대기, 색 = 모델 시간, 빨강 = 응답 없음."""
    import html as _html

    now = now or time.time()
    recs = sorted(records, key=lambda r: r.queued_at)[-max_rows:]
    if not recs:
        return "<div style='opacity:.7;font-size:.85rem'>아직 모델 호출이 없습니다.</div>"
    t0 = min(r.queued_at for r in recs)
    t1 = max(max((r.ended_at or now) for r in recs), t0 + 1)
    span = t1 - t0
    color = {"done": "#36B37E", "failed": "#DE350B", "cancelled": "#8777D9", "switched": "#8777D9",
             "running": "#4C9AFF", "streaming": "#00B8D9", "retry_wait": "#FFAB00", "queued": "#C1C7D0"}
    lines = []
    for r in recs:
        start = r.started_at if r.started_at is not None else (r.ended_at or now)
        end = r.ended_at or now
        wait_l, wait_w = (r.queued_at - t0) / span * 100, max(0.0, (start - r.queued_at) / span * 100)
        run_l, run_w = (start - t0) / span * 100, max(0.4, (end - start) / span * 100) if r.started_at is not None else 0
        stall = any(f.startswith("⚠️") for f in flags(r, now, stall_sec))
        c = "#FF5630" if stall else color.get(r.status, "#999")
        label = _html.escape(f"{r.who} · {KIND_LABELS.get(r.kind, r.kind)}")
        tip = _html.escape(f"{r.provider}:{r.model} · {STATUS_LABELS.get(r.status, r.status)} · 대기 {_fmt(r.waited(now))} · "
                           f"모델 {_fmt(r.model_time(now))}" + (" · " + " · ".join(flags(r, now, stall_sec)) if flags(r, now, stall_sec) else ""))
        pulse = " ao-gt-live" if r.status in ACTIVE else ""
        lines.append(
            f"<div class='ao-gt-row' title='{tip}'><div class='ao-gt-name'>{label}</div><div class='ao-gt-track'>"
            f"<span class='ao-gt-wait' style='left:{wait_l:.2f}%;width:{wait_w:.2f}%'></span>"
            + (f"<span class='ao-gt-bar{pulse}' style='left:{run_l:.2f}%;width:{run_w:.2f}%;background:{c}'></span>" if run_w else "")
            + f"</div><div class='ao-gt-time'>{_fmt(elapsed(r, now))}</div></div>"
        )
    css = (
        "<style>.ao-gt{font-size:.78rem}.ao-gt-row{display:flex;align-items:center;gap:6px;margin:2px 0}"
        ".ao-gt-name{width:30%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".ao-gt-track{position:relative;flex:1;height:12px;background:rgba(127,127,127,.10);border-radius:3px}"
        ".ao-gt-track span{position:absolute;top:0;height:12px;border-radius:3px}"
        ".ao-gt-wait{background:repeating-linear-gradient(45deg,rgba(127,127,127,.35),rgba(127,127,127,.35) 3px,transparent 3px,transparent 6px)}"
        ".ao-gt-live{animation:ao-pulse 1.2s ease-in-out infinite}"
        ".ao-gt-time{width:64px;text-align:right;font-variant-numeric:tabular-nums}"
        "@keyframes ao-pulse{0%,100%{opacity:1}50%{opacity:.55}}</style>"
    )
    head = f"<div style='display:flex;justify-content:space-between;opacity:.7'><span>0초</span><span>{_fmt(span)}</span></div>"
    return css + "<div class='ao-gt'>" + head + "".join(lines) + "</div>"


def headline(records: list[CallRecord], now: float | None = None, stall_sec: float = STALL_SEC) -> str:
    """한 줄 요약 — '모델 호출 12번 · 진행 중 2 · ⚠️ 응답 없음 1 · 기다린 시간 40초 · 모델 시간 5분 10초'."""
    now = now or time.time()
    live = [r for r in records if r.status in ACTIVE]
    stalled = [r for r in live if any(f.startswith("⚠️") for f in flags(r, now, stall_sec))]
    queued = [r for r in live if r.status == "queued"]
    parts = [f"모델 호출 {len(records)}번", f"진행 중 {len(live)}"]
    if stalled:
        parts.append(f"⚠️ 응답 없음 {len(stalled)}")
    if queued:
        parts.append(f"⏳ 자리 대기 {len(queued)}")
    parts += [f"기다린 시간 {_fmt(sum(r.waited(now) for r in records))}", f"모델 시간 {_fmt(sum(r.model_time(now) for r in records))}"]
    return " · ".join(parts)


def console_lines(records: list[CallRecord], now: float | None = None, stall_sec: float = STALL_SEC, limit: int = 8) -> list[str]:
    """🖥️ 커맨드 센터 📈 모니터용 — 진행 중인 호출(오래된 순) 한 줄씩."""
    now = now or time.time()
    live = sorted((r for r in records if r.status in ACTIVE), key=lambda r: r.queued_at)[:limit]
    return [
        f"{STATUS_LABELS.get(r.status, r.status)} {r.who} · {KIND_LABELS.get(r.kind, r.kind)} · {r.provider}:{r.model} · "
        f"{_fmt(elapsed(r, now))}" + ("".join(f" · {f}" for f in flags(r, now, stall_sec)))
        for r in live
    ]


# ─── 저장 (실행이 끝날 때 한 번) ─────────────────────────────────────
def dump(run_id: str) -> dict:
    return {"calls": [c.to_dict() for c in calls(run_id)], "interventions": interventions(run_id)}


def merge_saved(saved: dict | None, now: dict) -> dict:
    """저장된 기록 + 지금 메모리의 기록. 서버를 다시 켠 뒤(💬 추가 질문 등) 저장하면 메모리에는 새 호출만 있으므로
    예전 호출을 지우지 않게 합친다 — 같은 호출은 (줄 선 시각·누가·일)로 알아본다 (id 는 서버마다 다시 1부터)."""
    saved = saved or {}

    def key(c: dict) -> tuple:
        return c.get("queued_at"), c.get("who"), c.get("kind")

    fresh = {key(c) for c in now.get("calls") or []}
    calls = [c for c in saved.get("calls") or [] if key(c) not in fresh] + list(now.get("calls") or [])
    seen = {(e.get("ts"), e.get("what")) for e in now.get("interventions") or []}
    entries = [e for e in saved.get("interventions") or [] if (e.get("ts"), e.get("what")) not in seen]
    return {**now, "calls": sorted(calls, key=lambda c: c.get("queued_at") or 0),
            "interventions": sorted([*entries, *(now.get("interventions") or [])], key=lambda e: e.get("ts") or 0)}


def load(data: dict | None) -> tuple[list[CallRecord], list[dict]]:
    data = data or {}
    return [record_from(c) for c in data.get("calls") or []], list(data.get("interventions") or [])


def reset() -> None:
    """테스트용 — 기록을 모두 비운다."""
    with _lock:
        _calls.clear()
        _by_id.clear()
        _controls.clear()


__all__ = [
    "CallCancelled", "CallRecord", "KIND_LABELS", "STATUS_LABELS", "active", "attach_process", "begin", "calls",
    "check_cancel", "control", "dump", "flags", "gantt_html", "get", "in_flight", "interventions", "load", "log",
    "new_round", "prefer", "purpose", "release", "rows", "skip", "skipped", "stop", "stopped", "summary", "switch",
    "wait",
]
