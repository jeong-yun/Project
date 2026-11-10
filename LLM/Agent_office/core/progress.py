"""📊 진행률·남은 시간 추정 — '전체 몇 단계 중 몇 단계가 끝났고, 대략 얼마나 남았나'.

왜 필요한가 (2026-11-05 사용자 요청)
  ⏱ 호출 모니터의 시간표에는 끝났거나 진행 중인 호출만 보여서, 전체가 몇 단계인지·얼마나 남았는지 알 수 없었다.

무엇을 '단계'로 세나 (Unit — 모델 호출 한 번이 보통 한 단계)
  🎯 요청 정리(브리프를 AI 가 채울 때) · 🌐 검색어(웹 검색을 켰을 때) · 🔀 방향 제안(단계마다 확인)
  조직도(업무를 받은 사람만 — 분배가 끝나면 실제로 받은 사람으로 다시 센다):
    하위가 2명 이상인 관리자 = 📋 분배 1 + 🔍 검토·통합 1, 하위가 1명이면 분배 없이 검토 1, 맨 아래 실무자 = ✍️ 초안 1
  ⚖️ 비평 = 비평을 받는 사람마다 1 (settings scope: final = 최종 답변자만, all = 관리자 모두)
    불합격으로 🔁 자가 교정이 시작되면 그때마다 (교정 1 + 다시 비평 1) 을 더한다 → 남은 단계가 늘어난다
  🧪 하네스 재작성이 실제로 일어나면 그만큼 더한다
  '최대' = 비평이 매번 불합격하고 하네스가 매번 실패할 때 (max_self_corrections · max_retries 끝까지)
  💬 추가 질문(결정 전 문의·수정) 중에는 그 한 번만 센다 (답 1 + 하네스 실패 시 다시 고침 + 수정본이면 부장 검토,
    불합격이면 팀장이 다시 고침 · 추가로 고칠 점이 있으면 부장이 직접 고침 → 다시 검토 — followup_units)

남은 시간 (추정)
  단계 종류·모델마다 '걸린 시간의 중앙값'을 쓴다 — 같은 팀의 지난 실행 기록(runs.db 의 ⏱ 호출 기록) + 이번 실행에서 끝난 호출.
  기록이 2개 미만이면 같은 종류의 다른 모델 중앙값, 그것도 없으면 기본값(DEFAULT_SEC)을 쓰고 '기록 부족'이라고 밝힌다.
  형제 팀원은 동시에 일하므로 조직도를 따라 '가장 오래 걸리는 줄기(임계 경로)'로 더한다.
  같은 provider 를 쓰는 형제가 동시 호출 한도(settings llm_concurrency)보다 많으면 그만큼 줄을 서므로 나눠서 더한다.
  진행 중인 호출은 '예상 시간 − 지금까지 걸린 시간'만 남은 것으로 본다 (이미 넘겼으면 약간 남은 것으로).

화면: run_control_ui.render_progress (진행 막대) · 📊 시간표 탭의 남은 단계 표.
"""

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from core.schemas import OrgNode

# 기록이 없을 때의 기본값 (초) — 2026-11-03~05 실측 대략값 (claude_cli 는 호출마다 시작 시간 약 8초가 더해진다)
DEFAULT_SEC = {
    "brief": 20.0, "search": 15.0, "directions": 30.0, "plan": 40.0, "draft": 90.0, "review": 80.0,
    "critic": 45.0, "self_correct": 80.0, "harness": 80.0, "followup": 45.0,
}
CLI_OVERHEAD_SEC = 8.0
MIN_SAMPLES = 2  # 이보다 적으면 그 모델의 기록은 믿지 않는다
LEFT_FLOOR = 0.15  # 진행 중인 호출이 예상 시간을 넘겼을 때 남은 것으로 볼 비율
LABELS = {
    "brief": "🎯 요청 정리", "search": "🌐 검색어", "directions": "🔀 방향 제안", "plan": "📋 분배", "draft": "✍️ 초안",
    "review": "🔍 검토·통합", "critic": "⚖️ 비평", "self_correct": "🔁 자가 교정", "harness": "🧪 하네스 재작성",
    "followup": "💬 추가 질문",
}
DONE = "done"
ACTIVE = ("queued", "running", "streaming", "retry_wait")


@dataclass
class Plan:
    """이번 실행의 모양 — 조직도·설정에서 정해진다 (화면이 run 마다 만든다)."""

    org: OrgNode
    root_id: str
    node_models: dict[str, str] = field(default_factory=dict)  # 에이전트 id → "provider:model" (맡길 때의 primary)
    critic_model: str = ""
    utility_model: str = ""
    critic_enabled: bool = True
    critic_scope: str = "final"  # final | all
    max_self_corrections: int = 1
    harness: bool = False  # 검증기가 있는 팀인가
    max_retries: int = 0
    brief: bool = True  # 요청 정리를 AI 가 하는가
    web: bool = False
    step_mode: bool = False
    concurrency: dict = field(default_factory=dict)  # settings llm_concurrency


@dataclass
class Unit:
    kind: str
    node: str  # 에이전트 id ("" = 팀 공통)
    model: str
    done: bool = False
    running_sec: float | None = None  # 진행 중이면 지금까지 걸린 시간
    est_sec: float = 0.0
    basis: str = ""  # history | kind | default

    @property
    def left_sec(self) -> float:
        if self.done:
            return 0.0
        if self.running_sec is None:
            return self.est_sec
        return max(self.est_sec - self.running_sec, self.est_sec * LEFT_FLOOR)


@dataclass
class Estimate:
    units: list[Unit]
    done: int
    total: int
    worst: int  # 재시도·비평 반려가 끝까지 일어나면
    remaining_sec: float
    basis: str  # 사람이 읽는 근거 한 줄
    samples: int  # 추정에 쓴 지난 기록 수
    phase: str = "work"  # work | followup | waiting

    @property
    def percent(self) -> int:
        if not self.total:
            return 0
        return min(99, int(self.done * 100 / self.total)) if self.done < self.total else 100

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    def short(self) -> str:
        """'62% · 약 3분 남음 (추정)' — 사무실·막대 옆 한 줄."""
        return f"{self.percent}% · {eta_text(self.remaining_sec)} (추정)"


def eta_text(sec: float) -> str:
    if sec <= 0:
        return "곧 끝남"
    if sec < 60:
        return f"약 {max(10, int(round(sec / 10.0)) * 10)}초 남음"
    minutes = int(round(sec / 60.0))
    return f"약 {minutes}분 남음" if minutes < 60 else f"약 {minutes // 60}시간 {minutes % 60}분 남음"


# ─── 걸린 시간 기록 ─────────────────────────────────────────────────


def model_sec(rec) -> float | None:
    """호출 하나가 모델에서 걸린 시간 (끝난 호출만). CallRecord 또는 저장된 dict."""
    get = rec.get if isinstance(rec, dict) else lambda k, d=None: getattr(rec, k, d)
    if get("status") != DONE or get("started_at") is None or get("ended_at") is None:
        return None
    return max(0.0, float(get("ended_at")) - float(get("started_at")))


class Durations:
    """(단계 종류, 모델) → 걸린 시간 목록. 지난 기록 + 이번 실행."""

    def __init__(self, records=()):
        self.by_model: dict[tuple[str, str], list[float]] = {}
        self.by_kind: dict[str, list[float]] = {}
        self.history_count = 0
        self.add(records)

    def add(self, records, history: bool = False) -> None:
        for rec in records:
            sec = model_sec(rec)
            if sec is None:
                continue
            get = rec.get if isinstance(rec, dict) else lambda k, d=None: getattr(rec, k, d)
            kind, key = get("kind"), f"{get('provider')}:{get('model')}"
            self.by_model.setdefault((kind, key), []).append(sec)
            self.by_kind.setdefault(kind, []).append(sec)
            self.history_count += history

    def guess(self, kind: str, model: str) -> tuple[float, str]:
        own = self.by_model.get((kind, model)) or []
        if len(own) >= MIN_SAMPLES:
            return statistics.median(own), "history"
        same_kind = self.by_kind.get(kind) or []
        if len(same_kind) >= MIN_SAMPLES:
            return statistics.median(same_kind), "kind"
        if len(own) == 1:
            return own[0], "history"
        base = DEFAULT_SEC.get(kind, 60.0)
        return base + (CLI_OVERHEAD_SEC if model.startswith("claude_cli:") else 0.0), "default"


_hist_lock = threading.Lock()
_hist_cache: dict[str, tuple[float, list[dict]]] = {}
HISTORY_TTL = 60.0  # 지난 기록은 1분마다만 다시 읽는다 (화면은 1초마다 다시 그림)


def history(team_id: str, loader: Callable[[str], list[dict]], now: float | None = None) -> list[dict]:
    """같은 팀 지난 실행들의 호출 기록 (dict 목록). loader(team_id) 는 저장소를 읽는 함수 — 1분 동안 캐시."""
    now = time.time() if now is None else now
    with _hist_lock:
        hit = _hist_cache.get(team_id)
        if hit and now - hit[0] < HISTORY_TTL:
            return hit[1]
    try:
        records = list(loader(team_id))
    except Exception:  # noqa: BLE001 — 추정용이라 실패해도 기본값으로
        records = []
    with _hist_lock:
        _hist_cache[team_id] = (now, records)
    return records


def clear_history_cache() -> None:
    with _hist_lock:
        _hist_cache.clear()


# ─── 단계 세기 ─────────────────────────────────────────────────────


def _rget(rec, name):
    return rec.get(name) if isinstance(rec, dict) else getattr(rec, name, None)


def _calls_of(records, kind: str, node: str | None = None) -> list:
    return [r for r in records if _rget(r, "kind") == kind and (node is None or _rget(r, "agent_id") == node)]


def _running(records, kind: str, node: str | None, now: float) -> float | None:
    live = [r for r in _calls_of(records, kind, node) if _rget(r, "status") in ACTIVE]
    if not live:
        return None
    return max(now - (_rget(r, "started_at") or _rget(r, "queued_at") or now) for r in live)


def _n_done(records, kind: str, node: str | None = None) -> int:
    return sum(_rget(r, "status") == DONE for r in _calls_of(records, kind, node))


def _n_started(records, kind: str, node: str | None = None) -> int:
    """끝났거나 진행 중인 호출 수 (실패해서 대안 모델로 다시 한 호출은 한 번으로 — 실패·바꿈은 세지 않음)."""
    return sum(_rget(r, "status") in (DONE, *ACTIVE) for r in _calls_of(records, kind, node))


def build_units(plan: Plan, run: dict, records, now: float | None = None) -> tuple[list[Unit], int]:
    """(단계 목록, 최대일 때 더해질 단계 수). records = 이번 시도의 호출 기록 (CallRecord)."""
    now = time.time() if now is None else now
    steps = run.get("steps") or []
    step_kinds: dict[str, set[str]] = {}
    for s in steps:
        step_kinds.setdefault(s.node_id, set()).add(s.kind)
    assigned: dict[str, list[str]] = run.get("assigned") or {}
    units: list[Unit] = []
    extra_worst = 0
    work_started = bool(steps) or bool(assigned)

    live_given: set[tuple[str, str]] = set()  # 같은 사람의 같은 종류 단계가 여럿이면 첫 번째 남은 단계만 '진행 중'

    def add(kind: str, node: str, model: str, done: bool) -> None:
        running = None
        if not done and (kind, node) not in live_given:
            running = _running(records, kind, node or None, now)
            if running is not None:
                live_given.add((kind, node))
        units.append(Unit(kind, node, model, done=done, running_sec=running))

    if plan.brief:
        add("brief", "", plan.utility_model, run.get("brief") is not None or work_started)
    if plan.web:
        add("search", "", plan.utility_model, run.get("web") is not None or work_started or _n_done(records, "search") > 0)
    if plan.step_mode:
        add("directions", plan.root_id, plan.node_models.get(plan.root_id, ""), work_started or _n_done(records, "directions") > 0)

    nodes = {n.id: n for n in plan.org.walk()}
    root = nodes.get(plan.root_id, plan.org)

    def critic_for(node_id: str) -> bool:
        return plan.critic_enabled and (plan.critic_scope == "all" or node_id == plan.root_id)

    def harness_units(node_id: str, finished: bool) -> int:
        nonlocal extra_worst
        started = _n_started(records, "harness", node_id)
        for i in range(started):
            add("harness", node_id, plan.node_models.get(node_id, ""), finished or i < _n_done(records, "harness", node_id))
        if plan.harness and not finished:
            extra_worst += max(0, plan.max_retries - started)
        return started

    def visit(node: OrgNode) -> None:
        nonlocal extra_worst
        model = plan.node_models.get(node.id, "")
        kinds = step_kinds.get(node.id, set())
        if not node.children:
            finished = "draft" in kinds
            add("draft", node.id, model, finished)
            harness_units(node.id, finished)
            return
        if len(node.children) > 1:
            add("plan", node.id, model, node.id in assigned or "plan" in kinds)
        kids = assigned.get(node.id)
        chosen = [c for c in node.children if kids is None or c.id in kids]
        for child in chosen:
            visit(child)
        finished = bool(kinds & {"review", "integrate"})
        add("review", node.id, model, finished or _n_done(records, "review", node.id) > 0)
        if critic_for(node.id):
            corrections = _n_started(records, "self_correct", node.id)
            n_critic = 1 + corrections
            critic_done = _n_done(records, "critic", node.id)
            for i in range(n_critic):
                add("critic", node.id, plan.critic_model, finished or i < critic_done)
            for i in range(corrections):
                add("self_correct", node.id, model, finished or i < _n_done(records, "self_correct", node.id))
            if not finished:
                extra_worst += 2 * max(0, plan.max_self_corrections - corrections)
        harness_units(node.id, finished)

    visit(root)
    return units, extra_worst


def followup_units(plan: Plan, run: dict, records, now: float | None = None) -> tuple[list[Unit], int]:
    """💬 추가 질문 하나 — 팀장 답 1번 (+ 하네스 실패로 다시 고친 만큼) + 수정본이면 👔 부장 검토
    (+ 불합격이면 팀장이 다시 고침, 합격인데 추가로 고칠 점이 있으면 부장이 직접 고침 → 부장이 다시 검토 — 둘을 합쳐 max_self_corrections 번까지).

    수정본인지는 팀장 답이 와야 알므로, 부장 검토는 시작된 뒤에 단계로 세고 그 전에는 '최대'(worst)에만 넣는다.
    부장의 직접 수정도 팀장의 다시 고침과 같은 종류(self_correct)로 기록되고 상한을 함께 쓰므로 '최대' 계산은 하나다.
    (남은 시간 추정은 그 단계를 팀장 모델의 기록으로 어림한다 — 부장 모델이 다르면 조금 어긋날 수 있다.)
    """
    now = time.time() if now is None else now
    started_at = (run.get("followup_pending") or {}).get("started_at", 0.0)
    recs = [r for r in records if _rget(r, "kind") in ("followup", "critic", "self_correct")
            and (_rget(r, "queued_at") or 0) >= started_at and _rget(r, "status") in (DONE, *ACTIVE)]
    lead = plan.node_models.get(plan.root_id, "")
    units = [Unit(_rget(r, "kind"), plan.root_id, plan.critic_model if _rget(r, "kind") == "critic" else lead,
                  done=_rget(r, "status") == DONE,
                  running_sec=None if _rget(r, "status") == DONE else _running([r], _rget(r, "kind"), None, now))
             for r in recs] or [Unit("followup", plan.root_id, lead)]
    kinds = [u.kind for u in units]
    worst = max(0, plan.max_retries - (kinds.count("followup") - 1)) if plan.harness else 0
    if plan.critic_enabled:  # 검토 1 + (팀장 다시 고침 또는 부장 직접 수정 1 + 다시 검토 1) × 상한
        worst += (max(0, 1 + plan.max_self_corrections - kinds.count("critic"))
                  + max(0, plan.max_self_corrections - kinds.count("self_correct")))
    return units, worst


# ─── 남은 시간 ─────────────────────────────────────────────────────


def _limit(plan: Plan, model: str) -> int:
    conc = plan.concurrency or {}
    value = conc.get(model.partition(":")[0], conc.get("default", 0)) if model else conc.get("default", 0)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else 10**6


def critical_path(plan: Plan, units: list[Unit], run: dict) -> float:
    """조직도를 따라 남은 시간 — 형제는 동시에, 같은 provider 동시 한도만큼만."""
    by_node: dict[str, list[Unit]] = {}
    common = 0.0
    for u in units:
        if u.kind in ("brief", "search", "directions"):
            common += u.left_sec
        else:
            by_node.setdefault(u.node, []).append(u)
    assigned = run.get("assigned") or {}
    nodes = {n.id: n for n in plan.org.walk()}
    root = nodes.get(plan.root_id, plan.org)

    def own(node_id: str, kinds: tuple[str, ...]) -> float:
        return sum(u.left_sec for u in by_node.get(node_id, []) if u.kind in kinds)

    def t(node: OrgNode) -> float:
        if not node.children:
            return own(node.id, ("draft", "harness"))
        kids = assigned.get(node.id)
        chosen = [c for c in node.children if kids is None or c.id in kids]
        times = [(t(c), plan.node_models.get(c.id, "")) for c in chosen]
        longest = max((x for x, _ in times), default=0.0)
        queued = 0.0
        for provider in {m.partition(":")[0] for _, m in times}:
            group = [(x, m) for x, m in times if m.partition(":")[0] == provider]
            queued = max(queued, sum(x for x, _ in group) / min(len(group), _limit(plan, group[0][1])))
        return own(node.id, ("plan",)) + max(longest, queued) + own(node.id, ("review", "critic", "self_correct", "harness"))

    return common + t(root)


def estimate(plan: Plan, run: dict, records, past=(), now: float | None = None) -> Estimate:
    """records = 이번 시도의 호출 기록, past = 같은 팀 지난 실행의 호출 기록 (dict)."""
    now = time.time() if now is None else now
    durations = Durations()
    durations.add(past, history=True)
    durations.add(records)
    followup = run.get("status") == "followup_running"
    units, worst = (followup_units if followup else build_units)(plan, run, records, now)
    bases = set()
    for u in units:
        u.est_sec, u.basis = durations.guess(u.kind, u.model)
        if not u.done:
            bases.add(u.basis)
    remaining = sum(u.left_sec for u in units) if followup else critical_path(plan, units, run)
    done = sum(u.done for u in units)
    if not bases or bases == {"history"}:
        basis = (f"같은 팀 지난 기록 {durations.history_count}개 호출의 중앙값" if durations.history_count
                 else "이번 실행에서 끝난 호출의 중앙값")
    elif bases == {"default"}:
        basis = f"기록 부족 — 대략 {max(1, int(round(remaining / 60)))}분 (기본값)"
    else:
        basis = "지난 기록 중앙값 + 기록이 없는 단계는 " + ("같은 종류 평균" if "default" not in bases else "기본값")
    phase = "followup" if followup else ("waiting" if run.get("status") == "step_check" else "work")
    return Estimate(units, done, len(units), len(units) + worst, remaining, basis, durations.history_count, phase)


def remaining_rows(est: Estimate, names: dict[str, str]) -> list[dict]:
    """📊 시간표 탭의 '남은 단계' 표 — 진행 중 먼저."""
    rows = []
    for u in sorted((u for u in est.units if not u.done), key=lambda u: u.running_sec is None):
        rows.append({
            "상태": "▶ 진행 중" if u.running_sec is not None else "⏳ 남음", "단계": LABELS.get(u.kind, u.kind),
            "누가": names.get(u.node, u.node) if u.node else "팀", "모델": u.model or "-",
            "예상": f"{int(u.est_sec)}초", "남은 예상": f"{int(u.left_sec)}초",
            "근거": {"history": "지난 기록", "kind": "같은 종류 기록", "default": "기본값"}.get(u.basis, u.basis),
        })
    return rows


__all__ = ["DEFAULT_SEC", "Durations", "Estimate", "Plan", "Unit", "build_units", "clear_history_cache", "critical_path",
           "estimate", "eta_text", "followup_units", "history", "remaining_rows"]
