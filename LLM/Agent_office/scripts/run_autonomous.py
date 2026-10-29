"""자율 실행 — settings.yaml 의 autonomy.jobs 중 때가 된 작업만 사람 결정 없이 돌리고 보고서를 남긴다.

실행 (Agent_office 폴더에서):
    python scripts/run_autonomous.py --dry-run   # 호출 없이 때가 된 작업·오늘 비용·예산만 보기
    python scripts/run_autonomous.py             # 실제 실행 (autonomy.enabled: true 일 때만, API 비용)

작업 스케줄러에 1시간마다 등록해 두면 작업마다 every_hours 가 지난 것만 실행된다.
보고서는 data/autonomy/reports/<작업>/<시각>.md, 기록은 data/autonomy/audit.jsonl (해시 체인).
사람 결정 단계(interrupt)는 쓰지 않으므로 교훈은 저장되지 않는다 — 사람이 보고서를 읽고 판단한다.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import autonomy, usage  # noqa: E402
from core.config import load_env, load_models, load_ranks, load_settings, load_team, load_verification, team_org  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.tasks import build_request  # noqa: E402
from core.workflow import build_office_graph, stream_office  # noqa: E402


def make_runner(data_dir: Path):
    """작업 하나 = 팀 워크플로우 1회 (평가 하네스 evals/run_eval.py 와 같은 방식, storage·checkpointer 없음)."""

    def run(job: autonomy.AutonomyJob) -> dict:
        ranks, models, settings = load_ranks(), load_models(), load_settings()
        team = load_team(job.team)
        org = team_org(team, ranks)
        graph = build_office_graph(org, ranks, team, models, settings, load_verification())
        kwargs: dict = {"use_web": "web_search" in team.tools}
        question = job.question
        if job.task:
            req = build_request(next(t for t in team.tasks if t.id == job.task), job.inputs)
            question = req.question
            kwargs.update(brief=req.brief, auto_brief=False, tool_inputs=req.tool_inputs,
                          extra_validators=req.validators, harness_options=req.harness_options)
        run_id = f"auto-{job.id}-{datetime.now():%Y%m%d-%H%M%S}"
        final = ""
        with usage.scope(team_id=team.id, run_id=run_id, purpose="자율 실행"):
            for kind, payload in stream_office(graph, org, question, org.id, **kwargs):
                if kind == "step" and payload.node_id == org.id and payload.kind != "plan":
                    final = payload.content  # supervisor 가 마지막으로 올린 결과 = 최종 답변 (evals RunTrace.final 과 같음)
        out = data_dir / "autonomy" / "reports" / job.id / f"{run_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final, encoding="utf-8")
        return {"report": str(out), "actions": [{"action": "report", "summary": str(out)}]}

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="자율 실행 (때가 된 작업만)")
    parser.add_argument("--dry-run", action="store_true", help="호출 없이 계획만 보기")
    parser.add_argument("--data", default=os.getenv("AGENT_OFFICE_DATA_DIR") or str(ROOT / "data"), help="runs.db 가 있는 폴더")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = load_settings()
    cfg = autonomy.load_config(settings)
    data_dir = Path(args.data)
    storage = Storage(data_dir / settings.get("storage", {}).get("db_file", "runs.db"))
    audit = autonomy.AuditLog(data_dir / cfg.audit_file)
    now = time.time()

    def spent_today() -> float:
        return sum(r["cost_usd"] or 0.0 for r in storage.usage_summary(since=autonomy.day_start(now)))

    last = audit.last_runs()
    due = [j.id for j in cfg.jobs if autonomy.is_due(j, last.get(j.id), now)]
    ok, bad_line = audit.verify()
    print(f"자율 실행 {'켜짐' if cfg.enabled else '꺼짐'} · 작업 {len(cfg.jobs)}개 · 때가 된 작업 {due or '없음'}")
    print(f"오늘 추정 비용 ${spent_today():.4f} / 예산 ${cfg.daily_budget_usd} · 감사 로그 {'온전' if ok else f'{bad_line}번째 줄부터 어긋남'}")
    if args.dry_run or not cfg.enabled:
        return
    if not ok:
        print("감사 로그가 어긋나 실행하지 않습니다 — 로그를 확인하세요.")
        return

    load_env()
    usage.set_sink(storage.record_usage)  # 자율 실행 비용도 🏠 사용량에 '자율 실행'으로 잡힌다
    for r in autonomy.run_due(cfg, make_runner(data_dir), audit, spent_today, now):
        print(f"- {r.job}: {r.status} · ${r.cost_usd:.4f} · 승인 대기 {len(r.pending_approvals)}건 {r.report}")


if __name__ == "__main__":
    main()
