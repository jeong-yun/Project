"""자율 실행(스케줄) — 사람이 묻지 않아도 정해 둔 업무를 정해 둔 주기로 돌린다. 기본은 꺼짐.

설계 문서: docs/AI_전용_업무_설계.md

설정 (config/settings.yaml 의 autonomy)
  enabled          : false 면 scripts/run_autonomous.py 가 아무것도 하지 않는다 (기본)
  daily_budget_usd : 오늘(현지 날짜) 모든 LLM 호출 추정 비용 합계가 이 값 이상이면 새 실행을 시작하지 않는다
  max_runs_per_day : 하루에 시작하는 자율 실행 수 상한 (비용 추정이 빠진 무료·구독 모델도 막기 위해)
  jobs             : [{id, team, task, inputs, every_hours, level}]

자율 수준 (level) — 허용하지 않은 행동은 실행하지 않고 '사람 승인 대기'로만 남긴다 (기본 거부)
  observe : 읽기·분석·보고서 저장만 (read, report)
  propose : 위 + 바꿀 내용을 '제안'으로 저장 (propose — 예: 수정 PR 초안, 정책 문구)
  되돌릴 수 없는 행동(push·merge·삭제·외부 전송·설정 적용·배포)을 스스로 하는 수준은 만들지 않는다.

감사 로그 (audit.jsonl) — 실행 시작·끝·건너뜀·승인 요청·사람 결정을 한 줄씩 남긴다.
  줄마다 앞 줄의 해시를 넣어(해시 체인) 중간 줄을 고치거나 지우면 verify() 가 그 줄 번호를 찾아낸다.
  마지막 실행 시각도 이 로그에서 읽으므로 상태 파일을 따로 두지 않는다.
  화면·CLI·스케줄 스크립트가 같은 파일에 동시에 쓸 수 있어 append 는 잠금 파일(audit.jsonl.lock)을 잡고 쓴다.

승인 요청 — 허용 밖 행동은 request_id 와 함께 approval_requested 로 남기고, 행동의 내용(payload, 예: 패치)은
  autonomy/requests/<id>.json 에 따로 두고 그 sha256 을 로그 줄에 적는다 (승인한 내용 = 실행할 내용인지 확인용).
  승인·기각·실행은 core/approvals.py.

실제 업무 실행(runner)은 함수로 넘겨받는다 — 테스트는 가짜 runner, 스크립트는 워크플로우를 사람 결정 없이 돌리는 runner.
"""

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["observe", "propose"]

ALLOWED_ACTIONS: dict[str, set[str]] = {
    "observe": {"read", "report"},
    "propose": {"read", "report", "propose"},
}
# 어느 수준에서도 스스로 하지 않는 행동 — 화면·문서에 보여 주려고 따로 적어 둔다 (gate 는 허용 목록 밖이면 모두 승인 대기)
IRREVERSIBLE_ACTIONS = ("git_push", "merge", "delete", "send_external", "apply_config", "deploy")


class AutonomyJob(BaseModel):
    id: str
    team: str
    task: str = ""  # 팀 업무 양식 id (비우면 question 으로 자유 질문)
    question: str = ""
    inputs: dict = Field(default_factory=dict)
    every_hours: float = 24
    level: Level = "observe"


class ApprovalsConfig(BaseModel):
    """사람 승인 처리 (core/approvals.py)."""

    expiry_days: float = 14  # 이보다 오래 결정하지 않은 요청은 approval_expired (실행 없음)
    patch_repo: str = ""  # create_branch_with_patch 가 브랜치를 만들 git 저장소 (비우면 이 앱이 들어 있는 저장소)
    worktree_dir: str = ""  # 그 브랜치를 꺼낼 새 작업 폴더의 부모 (비우면 저장소 옆 <저장소>-autonomy-worktrees)


class AutonomyConfig(BaseModel):
    enabled: bool = False
    daily_budget_usd: float = 1.0
    max_runs_per_day: int = 6
    audit_file: str = "autonomy/audit.jsonl"  # 데이터 폴더(data/) 기준
    approvals: ApprovalsConfig = Field(default_factory=ApprovalsConfig)
    jobs: list[AutonomyJob] = Field(default_factory=list)


def load_config(settings: dict) -> AutonomyConfig:
    return AutonomyConfig(**(settings.get("autonomy") or {}))


def gate(level: Level, action: str) -> Literal["allow", "approval"]:
    """이 수준에서 스스로 해도 되는 행동이면 allow, 아니면 approval (실행하지 않고 사람 승인 대기열로)."""
    return "allow" if action in ALLOWED_ACTIONS[level] else "approval"


def is_due(job: AutonomyJob, last_ts: float | None, now: float) -> bool:
    return last_ts is None or now - last_ts >= job.every_hours * 3600


def day_start(now: float) -> float:
    """현지 날짜 0시의 타임스탬프 — 오늘 비용·실행 수를 셀 때 기준."""
    d = datetime.fromtimestamp(now)
    return datetime(d.year, d.month, d.day).timestamp()


# ─── 감사 로그 ────────────────────────────────────────────────────────


def _digest(prev: str, record: dict) -> str:
    body = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_LOCK_DEPTH: dict[str, int] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
LOCK_WAIT_SEC = 30.0  # 다른 쪽이 잠금을 이만큼 놓지 않으면 TimeoutError
LOCK_STALE_SEC = 120.0  # 이보다 오래된 잠금 파일은 죽은 프로세스가 남긴 것으로 보고 지운다


class AuditLog:
    """추가만 하는 JSONL 감사 로그 (해시 체인)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @contextmanager
    def locked(self):
        """이 로그를 읽고 이어 쓰는 동안 다른 스레드·프로세스가 끼어들지 못하게 한다 (같은 스레드에서는 겹쳐 잡아도 됨).

        화면(승인 버튼)·CLI·스케줄 스크립트가 동시에 쓰면 두 줄이 같은 앞 줄 해시를 가져 체인이 갈라지므로 필요하다.
        """
        key = str(self.path.resolve())
        with _THREAD_LOCKS_GUARD:
            rlock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with rlock:
            if _LOCK_DEPTH.get(key, 0) == 0:
                self._acquire_file()
            _LOCK_DEPTH[key] = _LOCK_DEPTH.get(key, 0) + 1
            try:
                yield
            finally:
                _LOCK_DEPTH[key] -= 1
                if _LOCK_DEPTH[key] == 0:
                    self.lock_path.unlink(missing_ok=True)

    def _acquire_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_WAIT_SEC
        while True:
            try:
                os.close(os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > LOCK_STALE_SEC:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"감사 로그 잠금을 {LOCK_WAIT_SEC:g}초 안에 얻지 못했습니다: {self.lock_path}") from None
                time.sleep(0.05)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, event: str, **fields) -> dict:
        with self.locked():
            items = self.entries()
            prev = items[-1]["hash"] if items else ""
            record = {"ts": fields.pop("ts", time.time()), "event": event, **fields, "prev": prev}
            record["hash"] = _digest(prev, record)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def verify(self) -> tuple[bool, int | None]:
        """(체인이 온전한지, 처음 어긋난 줄 번호 1부터)."""
        prev = ""
        for i, rec in enumerate(self.entries(), 1):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev") != prev or rec.get("hash") != _digest(prev, body):
                return False, i
            prev = rec["hash"]
        return True, None

    def last_runs(self) -> dict[str, float]:
        """{작업 id: 마지막으로 시작한 시각} — 실패한 실행도 시작한 것으로 친다 (실패가 반복 호출로 비용을 키우지 않게)."""
        out: dict[str, float] = {}
        for rec in self.entries():
            if rec["event"] == "run_start":
                out[rec["job"]] = rec["ts"]
        return out

    def runs_since(self, since: float) -> int:
        return sum(1 for rec in self.entries() if rec["event"] == "run_start" and rec["ts"] >= since)


# ─── 승인 요청 ─────────────────────────────────────────────────────────

MAX_PAYLOAD_CHARS = 500_000  # 패치 하나가 이보다 크면 내용을 저장하지 않고 요청만 남긴다 (승인해도 실행 실패)


def payload_path(audit: AuditLog, request_id: str) -> Path:
    """승인 요청의 내용(payload) 파일 — 감사 로그 옆 requests/<id>.json."""
    return audit.path.parent / "requests" / f"{request_id}.json"


def request_approval(audit: AuditLog, *, job: str, action: str, summary: str = "", payload=None,
                     ts: float | None = None, **fields) -> dict:
    """허용 밖 행동을 실행하지 않고 사람 승인 대기열에 넣는다 → approval_requested 줄 (request_id 포함).

    payload: dict 또는 pydantic 모델 (create_branch_with_patch·save_proposal 은 core.approvals.Proposal).
    내용은 파일로 따로 두고 sha256 만 로그에 적는다 — 승인 뒤 파일이 바뀌었으면 실행하지 않는다.
    """
    rid = uuid.uuid4().hex[:12]
    extra: dict = {}
    if payload is not None:
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        body = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1)
        if len(body) > MAX_PAYLOAD_CHARS:
            extra["payload_error"] = f"내용이 너무 큼 ({len(body):,}자 > {MAX_PAYLOAD_CHARS:,}자) — 저장하지 않음"
        else:
            path = payload_path(audit, rid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            extra["payload_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return audit.append("approval_requested", ts=time.time() if ts is None else ts, job=job, action=action,
                        summary=str(summary)[:200], request_id=rid, **fields, **extra)


# ─── 실행 ─────────────────────────────────────────────────────────────


@dataclass
class JobResult:
    job: str
    status: Literal["done", "error", "skipped_budget", "skipped_runs"]
    report: str = ""
    pending_approvals: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0


# runner(job) → {"report": 보고서 텍스트, "actions": [{"action": "propose", ...}, ...], "audit": {run_end 줄에 덧붙일 값 (검증 결과 등)}}
Runner = Callable[[AutonomyJob], dict]


def run_due(
    cfg: AutonomyConfig,
    runner: Runner,
    audit: AuditLog,
    spent_today: Callable[[], float],
    now: float | None = None,
) -> list[JobResult]:
    """때가 된 작업을 차례로 실행한다. 꺼져 있으면 아무것도 하지 않고 감사 로그도 남기지 않는다.

    spent_today: 오늘 LLM 추정 비용 합계(달러)를 돌려주는 함수 — 작업마다 다시 불러 예산을 확인한다.
    """
    if not cfg.enabled:
        return []
    now = time.time() if now is None else now
    last = audit.last_runs()
    results: list[JobResult] = []
    for job in cfg.jobs:
        if not is_due(job, last.get(job.id), now):
            continue
        before = spent_today()
        if before >= cfg.daily_budget_usd:
            audit.append("skipped_budget", ts=now, job=job.id, spent_usd=round(before, 4), budget_usd=cfg.daily_budget_usd)
            results.append(JobResult(job.id, "skipped_budget"))
            continue
        if audit.runs_since(day_start(now)) >= cfg.max_runs_per_day:
            audit.append("skipped_runs", ts=now, job=job.id, max_runs_per_day=cfg.max_runs_per_day)
            results.append(JobResult(job.id, "skipped_runs"))
            continue
        audit.append("run_start", ts=now, job=job.id, team=job.team, task=job.task, level=job.level)
        try:
            out = runner(job)
        except Exception as e:  # 한 작업의 실패가 다른 작업을 막지 않게
            cost = max(spent_today() - before, 0.0)
            audit.append("run_error", ts=now, job=job.id, error=f"{type(e).__name__}: {e}", cost_usd=round(cost, 4))
            results.append(JobResult(job.id, "error", report=str(e), cost_usd=cost))
            continue
        pending = []
        for act in out.get("actions", []):
            if gate(job.level, act.get("action", "")) == "approval":
                rec = request_approval(audit, ts=now, job=job.id, action=act.get("action", ""), summary=act.get("summary", ""),
                                       payload=act.get("payload"), level=job.level, team=job.team, task=job.task)
                pending.append({**act, "request_id": rec["request_id"]})
        cost = max(spent_today() - before, 0.0)
        extra = {k: v for k, v in (out.get("audit") or {}).items() if k not in ("ts", "event", "job", "prev", "hash", "cost_usd", "pending")}
        audit.append("run_end", ts=now, job=job.id, cost_usd=round(cost, 4), pending=len(pending), **extra)
        results.append(JobResult(job.id, "done", report=out.get("report", ""), pending_approvals=pending, cost_usd=cost))
    return results


# ─── 화면용 읽기 (autonomy_ui.py · scripts/run_autonomous.py --dry-run) ─────────────


def schedule(cfg: AutonomyConfig, audit: AuditLog, now: float) -> list[dict]:
    """작업별 {job, team, task, level, every_hours, last, next, due, last_result} — 감사 로그만 읽는다 (호출 없음)."""
    last = audit.last_runs()
    finished: dict[str, dict] = {}
    for rec in audit.entries():
        if rec["event"] in ("run_end", "run_error", "skipped_budget", "skipped_runs"):
            finished[rec["job"]] = rec
    out = []
    for job in cfg.jobs:
        ts = last.get(job.id)
        out.append({"job": job.id, "team": job.team, "task": job.task, "level": job.level, "every_hours": job.every_hours,
                    "last": ts, "next": now if ts is None else ts + job.every_hours * 3600,
                    "due": is_due(job, ts, now), "last_result": finished.get(job.id)})
    return out


def approval_queue(audit: AuditLog) -> list[dict]:
    """사람 승인 대기 — 자율 수준 밖이라 실행하지 않고 남긴 행동 중 아직 결정(승인·기각·만료)이 없는 것 (최근 것이 앞).

    줄마다 "id"(요청 id)가 붙는다. 승인·기각·만료 처리는 core/approvals.py.
    """
    from core import approvals  # approvals 가 이 모듈을 쓰므로 여기서 읽는다

    return approvals.pending(audit)


def autonomy_cost_since(audit: AuditLog, since: float) -> float:
    """자율 실행만의 추정 비용 합계 (run_end·run_error 의 cost_usd)."""
    return sum(float(rec.get("cost_usd") or 0) for rec in audit.entries()
               if rec["event"] in ("run_end", "run_error") and rec["ts"] >= since)
