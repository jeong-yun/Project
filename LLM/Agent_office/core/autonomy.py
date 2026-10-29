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

감사 로그 (audit.jsonl) — 실행 시작·끝·건너뜀·승인 요청을 한 줄씩 남긴다.
  줄마다 앞 줄의 해시를 넣어(해시 체인) 중간 줄을 고치거나 지우면 verify() 가 그 줄 번호를 찾아낸다.
  마지막 실행 시각도 이 로그에서 읽으므로 상태 파일을 따로 두지 않는다.

실제 업무 실행(runner)은 함수로 넘겨받는다 — 테스트는 가짜 runner, 스크립트는 워크플로우를 사람 결정 없이 돌리는 runner.
"""

import hashlib
import json
import time
from collections.abc import Callable
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


class AutonomyConfig(BaseModel):
    enabled: bool = False
    daily_budget_usd: float = 1.0
    max_runs_per_day: int = 6
    audit_file: str = "autonomy/audit.jsonl"  # 데이터 폴더(data/) 기준
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


class AuditLog:
    """추가만 하는 JSONL 감사 로그 (해시 체인)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, event: str, **fields) -> dict:
        items = self.entries()
        prev = items[-1]["hash"] if items else ""
        record = {"ts": fields.pop("ts", time.time()), "event": event, **fields, "prev": prev}
        record["hash"] = _digest(prev, record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                pending.append(act)
                audit.append("approval_requested", ts=now, job=job.id, action=act.get("action", ""),
                             summary=str(act.get("summary", ""))[:200])
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
    """사람 승인 대기 — 자율 수준 밖이라 실행하지 않고 남긴 행동 (최근 것이 앞). 승인·기각 처리는 아직 없다 (읽기 전용)."""
    return [rec for rec in reversed(audit.entries()) if rec["event"] == "approval_requested"]


def autonomy_cost_since(audit: AuditLog, since: float) -> float:
    """자율 실행만의 추정 비용 합계 (run_end·run_error 의 cost_usd)."""
    return sum(float(rec.get("cost_usd") or 0) for rec in audit.entries()
               if rec["event"] in ("run_end", "run_error") and rec["ts"] >= since)
