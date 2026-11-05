"""push 범위 계산·출력 해석 (prepush.py) — 범위는 임시 git 저장소로 실제 git 에 물어서 확인한다."""

import subprocess
from pathlib import Path

import pytest

import prepush
import pytest_select

ZERO = prepush.ZERO


def run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """원격(bare) + 내 PC 저장소. main 에 커밋 하나를 올려 둔 상태."""
    remote = tmp_path / "remote.git"
    local = tmp_path / "local"
    run(tmp_path, "init", "--bare", "-b", "main", str(remote))
    run(tmp_path, "init", "-b", "main", str(local))
    for k, v in (("user.name", "t"), ("user.email", "t@example.com"), ("core.autocrlf", "false")):
        run(local, "config", k, v)
    run(local, "remote", "add", "origin", str(remote))
    commit(local, "README.md", "hi")
    run(local, "push", "-q", "origin", "main")
    return remote, local


def commit(repo: Path, path: str, text: str) -> str:
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    run(repo, "add", path)
    run(repo, "commit", "-q", "-m", f"change {path}")
    return run(repo, "rev-parse", "HEAD")


def new_commits(repo, revs):
    return set(run(repo, "rev-list", *revs).split())


def test_parse_push_lines():
    text = f"refs/heads/main abc refs/heads/check/main {ZERO}\n\nbad line\n"
    assert prepush.parse_push_lines(text) == [prepush.Update("refs/heads/main", "abc", "refs/heads/check/main", ZERO)]


def test_delete_push_has_no_range():
    assert prepush.rev_args(prepush.Update("(delete)", ZERO, "refs/heads/x", "abc"), "--remotes=origin", lambda s: True) is None


def test_remote_filter_named_or_url():
    assert prepush.remote_filter("origin", ["origin"]) == "--remotes=origin"
    assert prepush.remote_filter("https://github.com/a/b.git", ["origin"]) == "--remotes"


def test_existing_branch_range_is_remote_to_local(repos):
    _, local = repos
    old = run(local, "rev-parse", "HEAD")
    a = commit(local, "LLM/Agent_office/core/feeds.py", "x = 1")
    b = commit(local, "docs/한글 이름.md", "doc")
    revs = prepush.rev_args(prepush.Update("refs/heads/main", b, "refs/heads/main", old), "--remotes=origin",
                            lambda s: prepush.commit_known(local, s))
    assert new_commits(local, revs) == {a, b}
    assert prepush.changed_files(local, revs) == ["LLM/Agent_office/core/feeds.py", "docs/한글 이름.md"]


def test_new_branch_range_excludes_commits_already_on_remote(repos):
    """main:check/main 처음 push — 원격 main 에 이미 있는 커밋은 빼고 새 커밋만."""
    _, local = repos
    a = commit(local, "a.py", "a")
    revs = prepush.rev_args(prepush.Update("refs/heads/main", a, "refs/heads/check/main", ZERO), "--remotes=origin",
                            lambda s: prepush.commit_known(local, s))
    assert new_commits(local, revs) == {a}


def test_unknown_remote_sha_falls_back_to_not_on_any_remote(repos):
    """다른 PC 가 올린 원격 커밋을 아직 받지 않았을 때도 범위를 정할 수 있다."""
    _, local = repos
    a = commit(local, "a.py", "a")
    unknown = "1234567890" * 4
    assert not prepush.commit_known(local, unknown)
    revs = prepush.rev_args(prepush.Update("refs/heads/main", a, "refs/heads/check/main", unknown), "--remotes=origin",
                            lambda s: prepush.commit_known(local, s))
    assert new_commits(local, revs) == {a}


def test_check_branch_behind_main_skips_commits_already_on_main(repos):
    """main 에 직접 올린 뒤 check/main 으로 올릴 때, main 에 이미 있는 커밋은 다시 보지 않는다."""
    _, local = repos
    run(local, "push", "-q", "origin", "main:check/main")
    check_old = run(local, "rev-parse", "HEAD")
    direct = commit(local, "direct.py", "d")
    run(local, "push", "-q", "origin", "main")
    run(local, "fetch", "-q", "origin")
    mine = commit(local, "mine.py", "m")
    revs = prepush.rev_args(prepush.Update("refs/heads/main", mine, "refs/heads/check/main", check_old), "--remotes=origin",
                            lambda s: prepush.commit_known(local, s))
    assert new_commits(local, revs) == {mine}
    assert direct not in new_commits(local, revs)


def test_worktree_clean_check(repos):
    _, local = repos
    commit(local, "proj/a.py", "a")
    assert prepush.worktree_is_clean(local, ["proj"])
    (local / "proj" / "new_test.py").write_text("x", encoding="utf-8")
    assert not prepush.worktree_is_clean(local, ["proj"])
    assert prepush.worktree_is_clean(local, ["other"])


def test_test_env_blanks_keys_and_reads_env_example(tmp_path, monkeypatch):
    ao = tmp_path / "LLM" / "Agent_office"
    ao.mkdir(parents=True)
    (ao / ".env.example").write_text("# x\nNEW_SERVICE_KEY=\n# OPTIONAL_TOKEN=\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("NEW_SERVICE_KEY", "secret")
    env = prepush.test_env(tmp_path)
    assert env["OPENAI_API_KEY"] == "" and env["NEW_SERVICE_KEY"] == "" and env["OPTIONAL_TOKEN"] == ""
    assert env["LANGSMITH_TRACING"] == "false"


def test_output_parsing():
    out = "....\nFAILED tests/test_a.py::test_x - assert 1 == 2\nERROR tests/test_b.py - ModuleNotFoundError\n"
    assert prepush.failed_lines(out) == ["FAILED tests/test_a.py::test_x - assert 1 == 2",
                                         "ERROR tests/test_b.py - ModuleNotFoundError"]
    assert prepush.collection_error("!!! Interrupted: 1 error during collection !!!")
    assert not prepush.collection_error("1 failed, 3 passed")
    assert prepush.missing_module("E   ModuleNotFoundError: No module named 'pygame'") == "pygame"


def test_select_plugin_matching():
    want = ["tests/test_a.py", "tests/test_b.py::test_y"]
    assert pytest_select.matches("tests/test_a.py::test_x", want)
    assert pytest_select.matches("tests/test_b.py::test_y[1]", want)
    assert not pytest_select.matches("tests/test_b.py::test_yz", want)
    assert not pytest_select.matches("tests/test_ab.py::test_x", want)


def test_fmt_secs():
    assert prepush.fmt_secs(42.4) == "42초"
    assert prepush.fmt_secs(75) == "1분 15초"


def test_pytest_plan_collect_modes():
    import select_tests as st

    chosen = st.RunList(run={"tests/test_a.py": "x", "tests/test_b.py::t": "y"}, left_to_ci=["tests/test_c.py"])
    assert prepush.pytest_plan(chosen, "near", full=False) == (
        ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"], ["tests/test_a.py", "tests/test_b.py::t"])
    assert prepush.pytest_plan(chosen, "all", full=False) == ([], ["tests/test_a.py", "tests/test_b.py::t"])
    assert prepush.pytest_plan(chosen, "near", full=True) == ([], None)
