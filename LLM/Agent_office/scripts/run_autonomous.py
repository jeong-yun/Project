"""자율 실행 — settings.yaml 의 autonomy.jobs 중 때가 된 작업만 사람 결정 없이 돌리고 보고서를 남긴다.

실행 (Agent_office 폴더에서):
    python scripts/run_autonomous.py --dry-run   # 호출 없이 때가 된 작업·오늘 비용·예산만 보기
    python scripts/run_autonomous.py             # 실제 실행 (autonomy.enabled: true 일 때만, API 비용)
    python scripts/run_autonomous.py --dry-run --jobs tests/fixtures/autonomy_governance.yaml
                                                 # settings 대신 작업 파일(예시)로 계획만 보기 — 켜짐으로 간주, 호출 없음
    python scripts/run_autonomous.py --approvals                 # 사람 승인 대기열 + 최근 결정 (호출 없음)
    python scripts/run_autonomous.py --approve <id> [--comment 메모]   # 승인 → 안전한 행동만 실행 (core/approvals.py)
    python scripts/run_autonomous.py --reject <id> [--comment 메모]    # 기각 (실행 없음)

작업 스케줄러에 1시간마다 등록해 두면 작업마다 every_hours 가 지난 것만 실행된다.
보고서는 data/autonomy/reports/<작업>/<시각>.md, 기록은 data/autonomy/audit.jsonl (해시 체인).
인프라 보안 팀 수정안은 재검사를 통과한 파일마다 승인 요청(create_branch_with_patch, payload = 승인 대기열의 Proposal 모양)으로
남기고, 통과한 diff 는 보고서에도 붙인다 — 저장소를 바꾸는 행동은 하지 않는다.
사람 결정 단계(interrupt)는 쓰지 않으므로 교훈은 저장되지 않는다 — 사람이 보고서를 읽고 판단한다.
하네스를 통과하지 못한 보고서는 맨 위에 '미검증'을 붙이고, 검증 결과·걸린 시간은 감사 로그 run_end 줄에 남긴다.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import approvals, autonomy, governance, iac_patch, usage  # noqa: E402
from core.config import load_env, load_models, load_ranks, load_settings, load_team, load_verification, team_org  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.tasks import build_request  # noqa: E402
from core.tools import run_tools  # noqa: E402
from core.workflow import build_office_graph, stream_office  # noqa: E402


def make_runner(data_dir: Path, models=None):
    """작업 하나 = 팀 워크플로우 1회 (평가 하네스 evals/run_eval.py 와 같은 방식, storage·checkpointer 없음).

    models 를 넘기면 그 설정을 쓴다 (테스트의 가짜 모델, 저비용 모델로 바꾼 실측 등).
    """

    def run(job: autonomy.AutonomyJob) -> dict:
        started = time.perf_counter()
        ranks, settings = load_ranks(), load_settings()
        model_cfg = models or load_models()
        team = load_team(job.team)
        org = team_org(team, ranks)
        graph = build_office_graph(org, ranks, team, model_cfg, settings, load_verification())
        kwargs: dict = {"use_web": "web_search" in team.tools}
        question = job.question
        options: dict = dict(team.harness.options)
        if job.task:
            req = build_request(next(t for t in team.tasks if t.id == job.task), job.inputs)
            question = req.question
            kwargs.update(brief=req.brief, auto_brief=False, tool_inputs=req.tool_inputs,
                          extra_validators=req.validators, harness_options=req.harness_options)
            options.update(req.harness_options)
        run_id = f"auto-{job.id}-{datetime.now():%Y%m%d-%H%M%S}"
        final = None
        with usage.scope(team_id=team.id, run_id=run_id, purpose="자율 실행"):
            for kind, payload in stream_office(graph, org, question, org.id, **kwargs):
                if kind == "step" and payload.node_id == org.id and payload.kind != "plan":
                    final = payload  # supervisor 가 마지막으로 올린 결과 = 최종 답변 (evals RunTrace.final 과 같음)
        content = final.content if final else ""
        failed = [r.validator for r in (final.harness if final else []) if not r.passed]
        audit_fields: dict = {"seconds": round(time.perf_counter() - started, 1), "harness_failed": failed}
        if "governance_evidence" in team.tools:  # 통제 번호·인용 측정값 (툴은 계산만, 호출 없음 — 실행 때와 같은 증적으로 다시 만든다)
            for t in run_tools(["governance_evidence"], question, kwargs.get("tool_inputs") or {}, model_cfg):
                options = {**t.harness_options, **options}
            audit_fields.update(governance.answer_metrics(content, options))
        actions: list[dict] = []
        report = content
        if "iac_scan" in team.tools:  # 수정안 재검사 표(코드) · propose 면 제안 저장 + 브랜치 만들기 승인 요청 (core/iac_patch.py)
            tool_inputs = {**(kwargs.get("tool_inputs") or {}), "external": False}  # 원본만 다시 읽는다 (외부 스캐너는 재검사 때)
            for t in run_tools(["iac_scan"], question, tool_inputs, model_cfg):
                options = {**t.harness_options, **options}
            iac_out = iac_patch.autonomy_outputs(content, options, job)  # 재검사 통과 파일마다 승인 요청
            audit_fields.update(iac_out["audit"])
            actions = iac_out["actions"]
            report = f"{content}\n\n{iac_out['report']}" if iac_out["report"] else content
        if failed:  # 사람이 없으니 검증 실패를 보고서 맨 위에 드러낸다 (설계 문서 5.4)
            logs = " / ".join(r.log for r in final.harness if not r.passed)
            report = f"> ⚠️ **미검증** — 하네스 검증 실패: {', '.join(failed)}\n> {logs[:600]}\n\n{report}"
        out = data_dir / "autonomy" / "reports" / job.id / f"{run_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        return {"report": str(out), "actions": [{"action": "report", "summary": str(out)}, *actions], "audit": audit_fields}

    return run


def load_jobs_file(path: str, settings: dict) -> autonomy.AutonomyConfig:
    """작업 파일(예시·fixture) — {jobs: [...]} 만 읽고 나머지(예산·로그 위치)는 settings 를 따른다. 켜짐으로 간주."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = settings.get("autonomy") or {}
    return autonomy.AutonomyConfig(**{**base, **data, "enabled": True})


def show_approvals(audit: autonomy.AuditLog, cfg: autonomy.AutonomyConfig) -> None:
    queue = autonomy.approval_queue(audit)
    entries = audit.entries()
    print(f"승인 대기 {len(queue)}건 (만료 {cfg.approvals.expiry_days:g}일)")
    for q in queue:
        run_cost, total = approvals.costs(entries, q)
        cost = "실행 중" if run_cost is None else f"${run_cost:.4f}"
        print(f"  [{q['id']}] {datetime.fromtimestamp(q['ts']):%Y-%m-%d %H:%M} · {q.get('job', '')} · {q.get('action', '')}"
              f" · {q.get('summary', '')}\n      비용 이번 실행 {cost} · 작업 합계 ${total:.4f}\n"
              f"      승인하면: {approvals.on_approve(q.get('action', ''))}")
    hist = approvals.history(audit, entries)[:10]
    if hist:
        print("최근 결정")
    for h in hist:
        print(f"  [{h['id']}] {datetime.fromtimestamp(h['decided']):%Y-%m-%d %H:%M} · {h['action']} · "
              f"{DECISION_TEXT[h['decision']]}{' → ' + h['outcome']['event'] if h['outcome'] else ''}"
              f"{' · ' + h['comment'] if h['comment'] else ''}")


DECISION_TEXT = {"granted": "승인", "rejected": "기각", "expired": "만료"}


def show_decision(out: dict) -> None:
    dec, res = out["decision"], out["outcome"]
    if out["status"] == "already":
        print(f"이미 결정된 요청입니다 — {DECISION_TEXT[approvals.DECISION_EVENTS[dec['event']]]} "
              f"({datetime.fromtimestamp(dec['ts']):%Y-%m-%d %H:%M}). 다시 실행하지 않습니다.")
    elif out["status"] == "expired":
        print(f"만료된 요청입니다 ({dec['age_days']}일 지남) — 실행하지 않았습니다.")
    else:
        print(f"{DECISION_TEXT[out['status']]} 기록 완료 ({dec['request_id']}).")
    if res:
        print(f"결과: {res['event']}" + (f" · {res['error']}" if res.get("error") else ""))
        for key in ("export_dir", "branch", "worktree", "commit"):
            if (res.get("result") or {}).get(key):
                print(f"  {key}: {res['result'][key]}")
        for step in res.get("manual_steps") or []:
            print(f"  사람이 직접: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="자율 실행 (때가 된 작업만)")
    parser.add_argument("--dry-run", action="store_true", help="호출 없이 계획만 보기")
    parser.add_argument("--jobs", help="settings 대신 이 작업 파일(YAML, {jobs: [...]})로 — --dry-run 과 함께만")
    parser.add_argument("--approvals", action="store_true", help="사람 승인 대기열·최근 결정 보기 (호출 없음)")
    parser.add_argument("--approve", metavar="ID", help="승인 요청 하나를 승인 (안전한 행동만 실행)")
    parser.add_argument("--reject", metavar="ID", help="승인 요청 하나를 기각 (실행 없음)")
    parser.add_argument("--comment", default="", help="--approve·--reject 에 붙일 메모")
    parser.add_argument("--data", default=os.getenv("AGENT_OFFICE_DATA_DIR") or str(ROOT / "data"), help="runs.db 가 있는 폴더")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.jobs and not args.dry_run:
        parser.error("--jobs 는 --dry-run 과 함께만 씁니다 (실제 실행은 settings.yaml 의 작업만)")

    settings = load_settings()
    cfg = load_jobs_file(args.jobs, settings) if args.jobs else autonomy.load_config(settings)
    data_dir = Path(args.data)
    storage = Storage(data_dir / settings.get("storage", {}).get("db_file", "runs.db"))
    audit = autonomy.AuditLog(data_dir / cfg.audit_file)
    now = time.time()

    if args.approvals or args.approve or args.reject:
        if args.approve and args.reject:
            parser.error("--approve 와 --reject 는 함께 쓰지 않습니다")
        approvals.expire_stale(audit, cfg.approvals, now)
        if args.approve or args.reject:
            try:
                show_decision(approvals.decide(audit, cfg.approvals, args.approve or args.reject,
                                               "grant" if args.approve else "reject", args.comment, via="cli", now=now))
            except approvals.ApprovalError as e:
                print(f"결정을 기록하지 않았습니다: {e}")
                sys.exit(1)
        else:
            show_approvals(audit, cfg)
        return

    def spent_today() -> float:
        return sum(r["cost_usd"] or 0.0 for r in storage.usage_summary(since=autonomy.day_start(now)))

    plan = autonomy.schedule(cfg, audit, now)
    due = [p["job"] for p in plan if p["due"]]
    ok, bad_line = audit.verify()
    source = f" (작업 파일 {args.jobs})" if args.jobs else ""
    print(f"자율 실행 {'켜짐' if cfg.enabled else '꺼짐'}{source} · 작업 {len(cfg.jobs)}개 · 때가 된 작업 {due or '없음'}")
    for p in plan:
        nxt = "지금" if p["due"] else f"{datetime.fromtimestamp(p['next']):%Y-%m-%d %H:%M}"
        print(f"  - {p['job']}: {p['team']}/{p['task'] or '자유 질문'} · {p['level']} · {p['every_hours']:g}시간마다 · 다음 {nxt}")
    print(f"오늘 추정 비용 ${spent_today():.4f} / 예산 ${cfg.daily_budget_usd} · 감사 로그 {'온전' if ok else f'{bad_line}번째 줄부터 어긋남'}"
          f" · 승인 대기 {len(autonomy.approval_queue(audit))}건")
    if args.dry_run or not cfg.enabled:
        return
    if not ok:
        print("감사 로그가 어긋나 실행하지 않습니다 — 로그를 확인하세요.")
        return
    approvals.expire_stale(audit, cfg.approvals, now)

    load_env()
    usage.set_sink(storage.record_usage)  # 자율 실행 비용도 🏠 사용량에 '자율 실행'으로 잡힌다
    for r in autonomy.run_due(cfg, make_runner(data_dir), audit, spent_today, now):
        print(f"- {r.job}: {r.status} · ${r.cost_usd:.4f} · 승인 대기 {len(r.pending_approvals)}건 {r.report}")


if __name__ == "__main__":
    main()
