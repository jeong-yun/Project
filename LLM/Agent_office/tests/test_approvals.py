"""사람 승인 처리 테스트 — 승인·기각이 해시 체인에 이어 쓰이는지, 두 번 승인해도 한 번만 실행, 실행기 없는 행동은 기록만,
실행 실패 기록, 만료, 로그 조작 탐지, 패치 스키마 검사, 새 브랜치에만 커밋(push 없음), CLI, 화면(AppTest).

git 이 필요한 테스트는 tmp_path 에 임시 저장소를 만든다 — 실제 저장소·사용자 checkout 은 건드리지 않는다.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core import approvals, autonomy
from core.approvals import ApprovalError, Proposal, RescanResult, decide, expire_stale
from core.autonomy import ApprovalsConfig, AuditLog, AutonomyConfig, AutonomyJob, request_approval, run_due

BASE_DIR = Path(__file__).resolve().parent.parent
NOW = 1_800_000_000.0
DAY = 86400

PATCH = """diff --git a/infra/main.tf b/infra/main.tf
--- a/infra/main.tf
+++ b/infra/main.tf
@@ -1,3 +1,3 @@
 resource "aws_security_group_rule" "ssh" {
-  cidr_blocks = ["0.0.0.0/0"]
+  cidr_blocks = ["10.0.0.0/8"]
 }
"""
MAIN_TF = 'resource "aws_security_group_rule" "ssh" {\n  cidr_blocks = ["0.0.0.0/0"]\n}\n'


def proposal(**kw) -> Proposal:
    return Proposal(**{"title": "SSH 를 사내 대역으로 제한", "target_file": "infra/main.tf", "patch": PATCH,
                       "rationale": "CKV_AWS_24", "findings": ["CKV_AWS_24: 0.0.0.0/0:22"],
                       "rescan": RescanResult(scanner="checkov", passed=True, before=["CKV_AWS_24"], after=[]), **kw})


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "autonomy" / "audit.jsonl")


@pytest.fixture
def cfg(tmp_path):
    return ApprovalsConfig(worktree_dir=str(tmp_path / "wts"))


def events(audit: AuditLog) -> list[str]:
    return [e["event"] for e in audit.entries()]


def ask(audit, action="save_proposal", payload=None, ts=NOW, job="iac_scan"):
    return request_approval(audit, ts=ts, job=job, action=action, summary=f"{action} 요청", payload=payload,
                            level="propose", team="infra_security", task="iac_fix")["request_id"]


# ─── 결정·감사 로그 ────────────────────────────────────────────────────────


def test_run_due_records_request_id_and_payload(audit):
    job = AutonomyJob(id="iac_scan", team="infra_security", level="propose")
    acts = [{"action": "report"}, {"action": "create_branch_with_patch", "summary": "SG 수정", "payload": proposal()}]
    res = run_due(AutonomyConfig(enabled=True, jobs=[job]), lambda j: {"actions": acts}, audit, lambda: 0.0, NOW)[0]
    rid = res.pending_approvals[0]["request_id"]
    req = autonomy.approval_queue(audit)[0]
    assert req["id"] == rid and req["level"] == "propose" and req["payload_sha256"]
    payload, problem = approvals.load_payload(audit, req)
    assert problem == "" and payload["target_file"] == "infra/main.tf" and payload["rescan"]["passed"] is True


def test_grant_and_reject_append_chained_events(audit, cfg):
    a = ask(audit, payload=proposal())
    b = ask(audit, action="git_push", payload={"branch": "autonomy/x"})
    out = decide(audit, cfg, a, "grant", comment="좋음", now=NOW + 60)
    assert out["status"] == "granted" and out["outcome"]["event"] == "action_executed"
    assert out["decision"]["decider"] == "user" and out["decision"]["comment"] == "좋음"
    assert decide(audit, cfg, b, "reject", comment="아직", via="cli", now=NOW + 120)["status"] == "rejected"
    assert events(audit) == ["approval_requested", "approval_requested", "approval_granted", "action_executed", "approval_rejected"]
    assert audit.verify() == (True, None)
    assert autonomy.approval_queue(audit) == []
    hist = approvals.history(audit)
    assert [(h["id"], h["decision"]) for h in hist] == [(b, "rejected"), (a, "granted")]
    assert hist[0]["via"] == "cli" and hist[1]["outcome"]["result"]["export_dir"].endswith(a)
    exported = Path(hist[1]["outcome"]["result"]["patch_file"])
    assert exported.read_text(encoding="utf-8") == PATCH  # 승인한 내용 그대로 저장


def test_double_approve_is_idempotent(audit, cfg):
    rid = ask(audit, payload=proposal())
    decide(audit, cfg, rid, "grant", now=NOW + 1)
    n = len(audit.entries())
    again = decide(audit, cfg, rid, "grant", now=NOW + 2)
    assert again["status"] == "already" and again["outcome"]["event"] == "action_executed"
    assert decide(audit, cfg, rid, "reject", now=NOW + 3)["status"] == "already"  # 승인 뒤 기각으로 바꿀 수도 없다
    assert len(audit.entries()) == n  # 새 줄 없음 = 다시 실행 안 함
    # 실행기 자체도 두 번 불러도 같은 결과 (파일을 다시 쓰지 않음)
    ctx = approvals.ExecContext(audit, cfg, autonomy.AuditLog(audit.path).entries()[0] | {"id": rid},
                                json.loads(autonomy.payload_path(audit, rid).read_text(encoding="utf-8")))
    assert approvals.save_proposal(ctx)["reused"] is True


@pytest.mark.parametrize("action", [*autonomy.IRREVERSIBLE_ACTIONS, "모르는_행동"])
def test_unknown_or_irreversible_action_only_records(audit, cfg, action):
    rid = ask(audit, action=action, payload={"branch": "feature/x", "remote": "origin; rm -rf /"})
    out = decide(audit, cfg, rid, "grant", now=NOW + 1)
    assert out["outcome"]["event"] == "action_not_executed" and "result" not in out["outcome"]
    assert not (audit.path.parent / "proposals").exists()
    steps = " ".join(out["outcome"]["manual_steps"])
    assert "rm -rf" not in steps  # 모델이 낸 문자열을 명령에 그대로 넣지 않는다
    if action == "git_push":
        assert out["outcome"]["manual_steps"] == ["git push origin feature/x"]


def test_executor_failure_is_recorded(audit, cfg):
    rid = ask(audit, action="save_proposal")  # 내용(payload) 없음 → 실행 실패
    out = decide(audit, cfg, rid, "grant", now=NOW + 1)
    assert out["outcome"]["event"] == "action_failed" and "payload" in out["outcome"]["error"]
    # 요청 뒤 내용 파일이 바뀌면 실행하지 않는다 (승인한 것 = 실행할 것)
    rid = ask(audit, payload=proposal())
    path = autonomy.payload_path(audit, rid)
    path.write_text(path.read_text(encoding="utf-8").replace("10.0.0.0/8", "0.0.0.0/0"), encoding="utf-8")
    out = decide(audit, cfg, rid, "grant", now=NOW + 2)
    assert out["outcome"]["event"] == "action_failed" and "sha256" in out["outcome"]["error"]
    assert audit.verify() == (True, None)


def test_expiry(audit, cfg):
    old = ask(audit, payload=proposal(), ts=NOW - 15 * DAY)
    fresh = ask(audit, payload=proposal(), ts=NOW - 1 * DAY)
    assert expire_stale(audit, cfg, NOW) == [old]
    assert expire_stale(audit, cfg, NOW) == []  # 두 번 불러도 한 번만
    assert [q["id"] for q in autonomy.approval_queue(audit)] == [fresh]
    assert decide(audit, cfg, old, "grant", now=NOW)["status"] == "already"
    assert not (audit.path.parent / "proposals").exists()  # 만료는 실행 없음
    # 화면을 안 열어 만료 처리가 안 됐어도, 늦은 승인은 만료로 닫힌다
    late = decide(audit, ApprovalsConfig(expiry_days=0.5), fresh, "grant", now=NOW)
    assert late["status"] == "expired" and late["decision"]["event"] == "approval_expired"
    assert audit.verify() == (True, None)


def test_tampering_still_detected_and_blocks_decisions(audit, cfg):
    a = ask(audit, payload=proposal())
    b = ask(audit, action="git_push")
    decide(audit, cfg, a, "reject", now=NOW + 1)
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["event"] = "approval_granted"  # 기각을 승인으로 몰래 바꾸면
    lines[2] = json.dumps(rec, ensure_ascii=False)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert audit.verify() == (False, 3)
    with pytest.raises(ApprovalError, match="3번째 줄"):
        decide(audit, cfg, b, "grant", now=NOW + 2)
    assert expire_stale(audit, cfg, NOW + 99 * DAY) == []  # 어긋난 로그에는 아무것도 덧붙이지 않는다
    with pytest.raises(ApprovalError):
        decide(AuditLog(audit.path.with_name("other.jsonl")), cfg, "nope", "grant")


def test_legacy_request_without_id_uses_line_hash(audit, cfg):
    rec = audit.append("approval_requested", ts=NOW, job="old", action="apply_config", summary="옛 요청")
    q = autonomy.approval_queue(audit)
    assert q[0]["id"] == rec["hash"][:12]
    assert decide(audit, cfg, rec["hash"][:12], "reject", now=NOW + 1)["status"] == "rejected"


def test_proposal_schema_rejects_unsafe_patches():
    for bad in ["../etc/passwd", "/etc/passwd", "C:/x", ".git/config", "a\\b"]:
        with pytest.raises(ValueError):
            proposal(target_file=bad)
    with pytest.raises(ValueError, match="한 파일만"):
        proposal(patch=PATCH + PATCH.replace("infra/main.tf", "infra/other.tf"))
    with pytest.raises(ValueError, match="심볼릭"):
        proposal(patch="diff --git a/infra/main.tf b/infra/main.tf\nnew file mode 120000\n")
    with pytest.raises(ValueError):
        proposal(patch="", new_content="")
    # hunk 안의 '-- 주석' 줄은 파일 머리줄로 세지 않는다
    sql = "--- a/infra/main.tf\n+++ b/infra/main.tf\n@@ -1,2 +1,2 @@\n--- old comment\n+++ new comment\n x\n"
    assert approvals.patch_paths(sql) == {"infra/main.tf"}
    assert Proposal(title="t", target_file="infra/main.tf", new_content="x").patch == ""


def test_concurrent_appends_keep_chain(audit):
    import threading

    def worker(n):
        for i in range(10):
            audit.append("probe", ts=NOW, n=n, i=i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(audit.entries()) == 40 and audit.verify() == (True, None)
    assert not audit.lock_path.exists()


# ─── create_branch_with_patch (임시 git 저장소) ─────────────────────────────


def git(cwd, *args) -> str:
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git 없음")
    root = tmp_path / "repo"
    (root / "infra").mkdir(parents=True)
    (root / "infra" / "main.tf").write_bytes(MAIN_TF.encode("utf-8"))
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-q", "-m", "init")
    return root


def test_create_branch_with_patch_commits_on_new_branch_only(audit, tmp_path, repo):
    cfg = ApprovalsConfig(patch_repo=str(repo), worktree_dir=str(tmp_path / "wts"))
    head = git(repo, "rev-parse", "HEAD")
    rid = ask(audit, action="create_branch_with_patch", payload=proposal())
    out = decide(audit, cfg, rid, "grant", now=NOW + 1)
    assert out["outcome"]["event"] == "action_executed", out["outcome"]
    res = out["outcome"]["result"]
    assert res["branch"] == f"autonomy/{rid}" and Path(res["worktree"]) == tmp_path / "wts" / rid
    # 사용자 checkout 은 그대로 — 브랜치·HEAD·파일 모두
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main" and git(repo, "rev-parse", "HEAD") == head
    assert (repo / "infra" / "main.tf").read_text(encoding="utf-8") == MAIN_TF
    assert "10.0.0.0/8" in git(repo, "show", f"{res['branch']}:infra/main.tf")
    assert git(repo, "rev-parse", res["branch"]) == res["commit"] and git(repo, "rev-parse", f"{res['branch']}~1") == head
    assert git(repo, "remote") == ""  # push 할 곳도 없고, 하지도 않는다
    assert any("push" in s for s in out["outcome"]["manual_steps"])  # push 는 사람이 할 명령으로만
    # 두 번째 실행(재승인·재실행)은 새 커밋을 만들지 않는다
    ctx = approvals.ExecContext(audit, cfg, approvals._index(audit.entries())[0][rid],
                                approvals.load_payload(audit, approvals._index(audit.entries())[0][rid])[0])
    again = approvals.create_branch_with_patch(ctx)
    assert again["reused"] is True and again["commit"] == res["commit"]


def test_create_branch_failure_cleans_up_and_records(audit, tmp_path, repo):
    cfg = ApprovalsConfig(patch_repo=str(repo), worktree_dir=str(tmp_path / "wts"))
    bad = proposal(patch=PATCH.replace('"0.0.0.0/0"', '"1.2.3.4/32"'))  # 기준 내용과 맞지 않는 패치
    rid = ask(audit, action="create_branch_with_patch", payload=bad)
    out = decide(audit, cfg, rid, "grant", now=NOW + 1)
    assert out["outcome"]["event"] == "action_failed" and "git apply" in out["outcome"]["error"]
    assert git(repo, "branch", "--list", f"autonomy/{rid}") == "" and not (tmp_path / "wts" / rid).exists()
    # 저장소 checkout 안에는 작업 폴더를 만들지 않는다
    inside = ApprovalsConfig(patch_repo=str(repo), worktree_dir=str(repo / "wts"))
    rid = ask(audit, action="create_branch_with_patch", payload=proposal())
    out = decide(audit, inside, rid, "grant", now=NOW + 2)
    assert out["outcome"]["event"] == "action_failed" and "checkout 밖" in out["outcome"]["error"]


# ─── CLI ─────────────────────────────────────────────────────────────────


def test_cli_lists_approves_and_rejects(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    import run_autonomous

    data = tmp_path / "data"
    audit = AuditLog(data / "autonomy" / "audit.jsonl")
    a = ask(audit, payload=proposal(), ts=time.time())
    b = ask(audit, action="git_push", payload={"branch": "autonomy/x"}, ts=time.time())

    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["run_autonomous.py", "--data", str(data), *args])
        run_autonomous.main()
        return capsys.readouterr().out

    out = cli("--approvals")
    assert a in out and b in out and "승인 대기 2건" in out
    out = cli("--approve", a)
    assert "승인" in out and "proposals" in out
    out = cli("--reject", b, "--comment", "아직 이름")
    assert "기각" in out
    assert cli("--approve", a).count("이미") == 1
    assert "승인 대기 0건" in cli("--approvals")
    recs = audit.entries()
    assert [r["event"] for r in recs][-3:] == ["approval_granted", "action_executed", "approval_rejected"]
    assert recs[-1]["via"] == "cli" and recs[-1]["comment"] == "아직 이름" and audit.verify() == (True, None)


# ─── 화면 (AppTest) ──────────────────────────────────────────────────────


def test_panel_buttons_approve_and_reject(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    data = tmp_path / "data"
    audit = AuditLog(data / "autonomy" / "audit.jsonl")
    a = ask(audit, payload=proposal(), ts=time.time() - 60)
    b = ask(audit, action="git_push", payload={"branch": "autonomy/x"}, ts=time.time())
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=90)
        at.run()
        assert not at.exception
        panel = next(e for e in at.expander if e.label.startswith("🤖 자율 실행"))
        assert "승인 대기 2건" in panel.label
        assert any(c.language == "diff" and "10.0.0.0/8" in c.value for c in at.code)  # 패치 미리보기
        at.button(key=f"home_ap_grant_{a}").click().run()  # 승인 → 확인 단계 (아직 실행 안 함)
        assert not at.exception and "approval_granted" not in events(audit)
        at.text_input(key=f"home_ap_comment_{a}").input("검토함")
        at.button(key=f"home_ap_confirm_{a}_go").click().run()
        assert not at.exception
        assert events(audit)[-2:] == ["approval_granted", "action_executed"]
        assert audit.entries()[-2]["comment"] == "검토함" and audit.entries()[-2]["via"] == "ui"
        assert [q["id"] for q in autonomy.approval_queue(audit)] == [b]
        panel = next(e for e in at.expander if e.label.startswith("🤖 자율 실행"))
        assert "승인 대기 1건" in panel.label
        at.button(key=f"home_ap_reject_{b}").click().run()
        assert not at.exception and events(audit)[-1] == "approval_rejected"
        assert autonomy.approval_queue(audit) == [] and audit.verify() == (True, None)
        hist = next(df for df in at.dataframe if "결정" in df.value.columns)
        assert set(hist.value["결정"]) == {"✅ 승인", "❌ 기각"}
    finally:
        st.cache_resource.clear()


def test_command_center_decision_tab_approves(tmp_path, monkeypatch, set_keys):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    set_keys()
    data = tmp_path / "data"
    audit = AuditLog(data / "autonomy" / "audit.jsonl")
    a = ask(audit, action="git_push", payload={"branch": "autonomy/abc"}, ts=time.time())
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=90)
        at.query_params["team"] = "general"
        at.query_params["view"] = "team"
        at.session_state["office_view"] = "office"
        at.session_state["cc_tab_general"] = "ask"
        at.run()
        assert not at.exception, at.exception
        at.button(key=f"cc_ask_ap_grant_{a}").click().run()
        at.button(key=f"cc_ask_ap_confirm_{a}_go").click().run()
        assert not at.exception, at.exception
        assert events(audit)[-2:] == ["approval_granted", "action_not_executed"]  # push 는 기록만
        assert any("git push origin autonomy/abc" in c.value for c in at.code)  # 사람이 할 명령을 보여 준다
        at.session_state["cc_tab_general"] = "triggers"
        at.run()
        assert not at.exception and any("결정" in df.value.columns for df in at.dataframe)  # ⏱ 탭의 결정 이력
    finally:
        st.cache_resource.clear()
