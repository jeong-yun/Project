"""자율 실행 테스트 — 기본 꺼짐, 주기, 예산·실행 수 상한, 승인 대기(기본 거부), 감사 로그 해시 체인, 거버넌스 팀 뼈대."""

import json

import pytest

from core import autonomy
from core.autonomy import AuditLog, AutonomyConfig, AutonomyJob, gate, run_due
from core.config import load_settings, load_team, validate_config
from core.tasks import build_request

NOW = 1_800_000_000.0  # 고정 시각 (테스트마다 같은 결과)
HOUR = 3600


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


def make_cfg(**kw) -> AutonomyConfig:
    jobs = kw.pop("jobs", [AutonomyJob(id="drift", team="security_governance", task="control_check", every_hours=24)])
    return AutonomyConfig(**{"enabled": True, "daily_budget_usd": 1.0, "max_runs_per_day": 6, "jobs": jobs, **kw})


class FakeRunner:
    """호출 기록 + 호출마다 비용을 cost 만큼 늘리는 가짜 runner."""

    def __init__(self, actions=None, cost=0.1, fail=False):
        self.calls, self.spent = [], 0.0
        self.actions, self.cost, self.fail = actions or [], cost, fail

    def __call__(self, job):
        self.calls.append(job.id)
        self.spent += self.cost
        if self.fail:
            raise RuntimeError("모델 오류")
        return {"report": f"{job.id} 보고서", "actions": self.actions}

    def spent_today(self):
        return self.spent


def test_settings_default_is_off_and_does_nothing(audit):
    cfg = autonomy.load_config(load_settings())
    assert cfg.enabled is False  # 저장소 기본값은 꺼짐
    runner = FakeRunner()
    cfg.jobs = make_cfg().jobs
    assert run_due(cfg, runner, audit, runner.spent_today, NOW) == []
    assert runner.calls == [] and audit.entries() == []  # 꺼져 있으면 로그도 남기지 않는다


def test_runs_only_when_due(audit):
    cfg, runner = make_cfg(), FakeRunner()
    assert [r.status for r in run_due(cfg, runner, audit, runner.spent_today, NOW)] == ["done"]
    assert run_due(cfg, runner, audit, runner.spent_today, NOW + 23 * HOUR) == []  # 24시간이 안 지남
    assert [r.job for r in run_due(cfg, runner, audit, runner.spent_today, NOW + 24 * HOUR)] == ["drift"]
    assert runner.calls == ["drift", "drift"]


def test_budget_cap_stops_new_runs(audit):
    jobs = [AutonomyJob(id=f"j{i}", team="security_governance", every_hours=1) for i in range(3)]
    cfg, runner = make_cfg(jobs=jobs), FakeRunner(cost=0.6)
    results = run_due(cfg, runner, audit, runner.spent_today, NOW)
    assert [r.status for r in results] == ["done", "done", "skipped_budget"]  # 0.6 → 1.2 ≥ 1.0 이면 세 번째는 시작 안 함
    assert results[0].cost_usd == pytest.approx(0.6)
    assert audit.entries()[-1]["event"] == "skipped_budget"


def test_max_runs_per_day_counts_free_models_too(audit):
    jobs = [AutonomyJob(id=f"j{i}", team="security_governance", every_hours=1) for i in range(3)]
    cfg, runner = make_cfg(jobs=jobs, max_runs_per_day=2), FakeRunner(cost=0.0)  # 무료·구독 모델: 비용 0 으로 잡힘
    assert [r.status for r in run_due(cfg, runner, audit, runner.spent_today, NOW)] == ["done", "done", "skipped_runs"]


def test_irreversible_actions_wait_for_approval(audit):
    acts = [{"action": "report"}, {"action": "propose", "summary": "정책 문구 수정안"}, {"action": "git_push", "summary": "수정 PR"}]
    observe = make_cfg()
    result = run_due(observe, FakeRunner(actions=acts), audit, lambda: 0.0, NOW)[0]
    assert [a["action"] for a in result.pending_approvals] == ["propose", "git_push"]  # observe 는 보고서만

    propose = make_cfg(jobs=[AutonomyJob(id="p", team="security_governance", level="propose")])
    result = run_due(propose, FakeRunner(actions=acts), audit, lambda: 0.0, NOW)[0]
    assert [a["action"] for a in result.pending_approvals] == ["git_push"]
    for action in autonomy.IRREVERSIBLE_ACTIONS:  # 어느 수준에서도 스스로 하지 않는다
        assert gate("observe", action) == gate("propose", action) == "approval"
    assert gate("propose", "모르는_행동") == "approval"  # 기본 거부


def test_runner_error_is_logged_and_counts_as_run(audit):
    cfg, runner = make_cfg(), FakeRunner(fail=True)
    assert [r.status for r in run_due(cfg, runner, audit, runner.spent_today, NOW)] == ["error"]
    assert [e["event"] for e in audit.entries()] == ["run_start", "run_error"]
    assert run_due(cfg, runner, audit, runner.spent_today, NOW + HOUR) == []  # 실패해도 바로 다시 부르지 않는다


def test_audit_chain_detects_tampering(audit):
    cfg, runner = make_cfg(), FakeRunner(actions=[{"action": "git_push"}])
    run_due(cfg, runner, audit, runner.spent_today, NOW)
    assert audit.verify() == (True, None)
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["action"] = "report"  # 승인 요청 기록을 몰래 바꾸면
    lines[1] = json.dumps(rec, ensure_ascii=False)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert audit.verify() == (False, 2)
    del lines[1]  # 줄을 지워도
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert audit.verify() == (False, 2)


def test_governance_team_skeleton_is_valid(ranks, models):
    team = load_team("security_governance")
    validate_config(ranks, models, [team])
    assert "web_search" not in team.tools  # 내부 증적이 검색어로 밖에 나가지 않게
    req = build_request(team.tasks[0], {"framework": "ISMS-P", "evidence": "계정 검토: 연 1회"})
    assert "ISMS-P" in req.question and "점검 범위" not in req.question  # 비운 선택 입력 줄은 빠짐
    assert req.harness_options["required_keys"] == ["summary", "controls"]
