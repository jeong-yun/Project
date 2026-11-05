"""올리기 전 검사 — git pre-push 훅(.githooks/pre-push)이 부른다. Docker 없이, 사양이 낮은 PC 에서도 1~2분 안을 목표로 한다.

검사 (push 할 커밋들 = 원격에 있던 것..보내는 것)
  1. 비밀 검사 — gitleaks (CI 와 같은 8.30.1, 같은 .gitleaks.toml, 같은 범위, 값은 --redact 로 가림)
  2. 바뀐 부분 테스트 — select_tests.py 의 규칙으로 고른 테스트 + 전체 테스트 '모으기'(import 검사)
     실제 모델·웹을 부르지 않게 검사 중에는 API 키 환경 변수를 비운다 (테스트는 tests/conftest.py 의 가짜 모델을 쓴다).
  3. 핵심 패키지 버전 경고 (막지 않음)
하나라도 실패하면 push 를 막는다. 급하면 SKIP_PREPUSH=1 (훅이 경고만 하고 통과시킨다).

손으로 돌려 보기 (push 하지 않고 검사만):
    python tools/prepush/prepush.py --manual [브랜치·커밋, 기본 HEAD]
환경 변수
    PREPUSH_ALL=1        2순위 테스트도 상한 없이 모두 돌린다
    PREPUSH_FULL=1       Agent Office 테스트 전체를 돌린다 (이 PC 10~17분)
    PREPUSH_MAX_FILES=8  2순위까지 넣을 때의 파일 수 상한
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gitleaks_tool  # noqa: E402
import pytest_select  # noqa: E402
import select_tests  # noqa: E402
import versions  # noqa: E402

ZERO = "0" * 40
# 검사 중 비우는 키 (Agent Office .env.example 의 이름들 + 흔한 것). 빈 값으로 두면 python-dotenv 가 .env 로 덮어쓰지 않는다.
SECRET_ENV = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "UPSTAGE_API_KEY",
    "FREELLMAPI_API_KEY", "FREELLMAPI_BASE_URL", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET",
    "THREADS_ACCESS_TOKEN", "TAVILY_API_KEY",
)
LOAD_SENSITIVE = ".github/ci/agent-office-load-sensitive.txt"


# ─────────────────────────── push 범위 ───────────────────────────

@dataclass
class Update:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def parse_push_lines(text: str) -> list[Update]:
    """pre-push 훅 표준 입력: '<로컬 ref> <로컬 sha> <원격 ref> <원격 sha>' 한 줄씩."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            out.append(Update(*parts))
    return out


def remote_filter(remote: str, named_remotes: list[str]) -> str:
    """원격 이름이면 그 원격의 브랜치들, 주소로 push 하면(이름 없음) 모든 원격의 브랜치들."""
    return f"--remotes={remote}" if remote in named_remotes else "--remotes"


def rev_args(update: Update, remote: str, known) -> list[str] | None:
    """이번 push 로 새로 올라가는 커밋을 고르는 git rev-list 인자. 브랜치 삭제면 None.
    - 원격에 이미 있던 커밋을 이 PC 가 알면: 원격..로컬 (그중 원격의 다른 브랜치에 이미 있는 커밋은 뺀다 —
      예: main 에 직접 올린 뒤 check/main 으로 올릴 때 이미 GitHub 에서 검사한 커밋을 다시 보지 않게)
    - 새 브랜치이거나 원격 커밋을 아직 못 받았으면: 로컬 중 원격 어느 브랜치에도 없는 것
    known(sha) → 이 PC 저장소에 그 커밋이 있는지. remote 는 remote_filter() 결과(--remotes=origin 등)."""
    if update.local_sha == ZERO:
        return None
    if update.remote_sha != ZERO and known(update.remote_sha):
        return [f"{update.remote_sha}..{update.local_sha}", "--not", remote]
    return [update.local_sha, "--not", remote]


def git(*args: str, cwd: Path, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 실패: {res.stderr.strip()}")
    return res.stdout


def commit_known(root: Path, sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=root, capture_output=True).returncode == 0


def changed_files(root: Path, revs: list[str]) -> list[str]:
    out = git("log", "--no-renames", "--name-only", "--format=", "-z", *revs, cwd=root)
    return sorted({p.strip("\n") for p in out.split("\0") if p.strip("\n")})


# ─────────────────────────── 출력 ───────────────────────────

def say(msg: str = "") -> None:
    print(msg, flush=True)


def run_streamed(cmd: list[str], cwd: Path, env: dict | None = None, indent: str = "    ") -> tuple[int, str]:
    """명령을 돌리면서 출력을 들여 써서 보여 주고, 전체 출력도 돌려준다."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        print(indent + line, flush=True)
    return proc.wait(), "\n".join(lines)


def fmt_secs(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}분 {s % 60}초" if s >= 60 else f"{s}초"


@dataclass
class Result:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    secret_found: bool = False


# ─────────────────────────── 1. gitleaks ───────────────────────────

def run_gitleaks(root: Path, revs_list: list[list[str]], tip: str, res: Result) -> None:
    say(f"[1/3] 비밀 검사 (gitleaks {gitleaks_tool.VERSION})")
    exe = gitleaks_tool.exe_path()
    have = gitleaks_tool.installed_version(exe)
    if have != gitleaks_tool.VERSION:
        why = "설치 안 됨" if have is None else f"버전이 {have}"
        res.failures.append(f"gitleaks {gitleaks_tool.VERSION} {why} — tools\\setup_prepush.bat 를 한 번 실행하세요")
        say(f"    ❌ gitleaks {why} ({exe})")
        return
    with tempfile.TemporaryDirectory(prefix="prepush-gl-") as tmp:
        config = Path(tmp) / "gitleaks.toml"
        # 보내는 커밋에 들어 있는 설정을 쓴다 (CI 가 그 커밋을 받아서 쓰는 것과 같게)
        shown = subprocess.run(["git", "show", f"{tip}:.gitleaks.toml"], cwd=root, capture_output=True)
        cmd_cfg: list[str] = []
        if shown.returncode == 0:
            config.write_bytes(shown.stdout)
            cmd_cfg = ["--config", str(config)]
        found = False
        for revs in revs_list:
            cmd = [str(exe), "git", str(root), *cmd_cfg, "--redact", "--no-banner", "--no-color", "--verbose", "--log-opts", " ".join(revs)]
            code, out = run_streamed(cmd, cwd=root)
            if code != 0:
                found = True
                if "leaks found" in out:
                    res.secret_found = True
                    n = re.search(r"leaks found:\s*(\d+)", out)
                    res.failures.append(
                        f"비밀 검사: 키·토큰처럼 보이는 값 {n.group(1) if n else ''}개 (위 Finding·File·Line 참고, 값은 가림). "
                        "진짜 키면 발급처에서 폐기하고 커밋에서 뺀 뒤 다시, 오탐이면 .gitleaks.toml 에 파일+값 모양으로 좁은 예외"
                    )
                else:
                    res.failures.append("비밀 검사: gitleaks 실행 오류 (위 출력 참고)")
        if not found:
            say("    ✅ 통과")


# ─────────────────────────── 2. 테스트 ───────────────────────────

def test_env(root: Path) -> dict:
    env = dict(os.environ)
    names = set(SECRET_ENV)
    example = root / select_tests.AO_DIR / ".env.example"
    if example.is_file():
        for line in example.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", line)
            if m:
                names.add(m.group(1))
    for name in names:
        env[name] = ""
    env.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return env


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return select_tests.read_smoke(path.read_text(encoding="utf-8"))


def failed_lines(output: str) -> list[str]:
    return [ln.strip() for ln in output.splitlines() if ln.startswith(("FAILED ", "ERROR "))]


def collection_error(output: str) -> bool:
    return "ERROR collecting" in output or re.search(r"\d+ errors? during collection", output) is not None


def missing_module(output: str) -> str | None:
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
    return m.group(1) if m else None


def pytest_plan(chosen: select_tests.RunList, collect: str, full: bool) -> tuple[list[str], list[str] | None]:
    """(pytest 에 줄 경로 인자, 실행할 것 목록 — None 이면 모은 것 전부 실행).
    collect="all": 전체 테스트를 모으고(import 검사) 고른 것만 실행.
    collect="near": 고른 파일 + 상한 때문에 CI 에 맡긴 2순위 파일(바뀐 모듈을 import 하는 테스트)만 모으고 고른 것만 실행."""
    if full:
        return [], None
    if collect == "all":
        return [], list(chosen.run)
    files = sorted({k.split("::", 1)[0] for k in chosen.run} | set(chosen.left_to_ci))
    return files, list(chosen.run)


def run_agent_office(root: Path, changed: list[str], res: Result, env: dict) -> None:
    ao_root = root / select_tests.AO_DIR
    tests_dir = ao_root / "tests"
    sources = {
        f"tests/{p.name}": p.read_text(encoding="utf-8", errors="replace") for p in sorted(tests_dir.glob("test_*.py"))
    }
    plan = select_tests.plan_agent_office(changed, sources)
    smoke = read_list(HERE / "smoke_tests.txt")
    full = os.environ.get("PREPUSH_FULL") == "1"
    run_all = os.environ.get("PREPUSH_ALL") == "1"
    collect = "all" if os.environ.get("PREPUSH_COLLECT_ALL") == "1" else "near"
    max_files = int(os.environ.get("PREPUSH_MAX_FILES") or select_tests.DEFAULT_MAX_FILES)
    chosen = select_tests.finalize(plan, smoke, max_files=max_files, run_all=run_all)
    load_sensitive = read_list(root / LOAD_SENSITIVE)
    paths, selected = pytest_plan(chosen, collect, full)

    say("    Agent Office:")
    if full:
        say("      PREPUSH_FULL=1 — 전체 테스트를 돌립니다")
    else:
        for test, why in chosen.run.items():
            say(f"      - {test}  ← {why}")
        if chosen.left_to_ci:
            say(f"      (2순위 {len(chosen.left_to_ci)}개 파일은 상한 {max_files}개를 넘어 실행은 GitHub check/main 검사에 맡기고 "
                f"모으기(import 검사)만 함 — 모두 돌리려면 PREPUSH_ALL=1)")
            say("        " + ", ".join(t.removeprefix("tests/") for t in chosen.left_to_ci))
        if collect == "all":
            say("      (PREPUSH_COLLECT_ALL=1 — 전체 테스트를 모아 import 오류를 봅니다)")
        if not chosen.run and not paths and collect != "all":
            say("      고른 테스트 없음 — 건너뜀")
            return
    with tempfile.TemporaryDirectory(prefix="prepush-sel-") as tmp:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--disable-warnings", "--rootdir", str(ao_root)]
        skipped_ls = list(load_sensitive) if selected is None else             [n for n in load_sensitive if pytest_select.matches(n, selected)]
        for node in load_sensitive:
            cmd += ["--deselect", node]
        run_env = dict(env)
        if selected is not None:
            sel = Path(tmp) / "select.txt"
            sel.write_text("\n".join(selected), encoding="utf-8")
            run_env["PREPUSH_SELECT"] = str(sel)
            run_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), env.get("PYTHONPATH")]))
            cmd += ["-p", "pytest_select"]
        cmd += paths
        if skipped_ls:
            say(f"      (부하·속도에 민감한 테스트 {len(skipped_ls)}개는 빼고 GitHub 검사에 맡김 — .github/ci/agent-office-load-sensitive.txt)")
        started = time.monotonic()
        code, out = run_streamed(cmd, cwd=ao_root, env=run_env, indent="      | ")
    took = fmt_secs(time.monotonic() - started)
    # 5 = 실행한 테스트 없음 (고른 것이 없거나 부하 민감 테스트만 골랐을 때 — 모으기는 성공)
    if code == 0 or (code == 5 and not collection_error(out)):
        say(f"      ✅ 통과 ({took})")
        return
    if collection_error(out):
        lines = failed_lines(out)
        res.failures.append("Agent Office 테스트 모으기(import) 오류 — 실행하지 않는 파일이라도 import 가 깨지면 막습니다:\n"
                            + "\n".join(f"      {ln}" for ln in lines[:15]))
    else:
        lines = failed_lines(out) or [f"pytest 종료 코드 {code}"]
        res.failures.append("Agent Office 테스트 실패:\n" + "\n".join(f"      {ln}" for ln in lines[:15]))
    say(f"      ❌ 실패 ({took})")


def run_other(root: Path, project: str, files: list[str], res: Result, env: dict) -> None:
    say(f"    {project}: (바뀐 파일 {len(files)}개) → {project}/tests 전체")
    started = time.monotonic()
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--disable-warnings", "tests"]
    code, out = run_streamed(cmd, cwd=root / project, env=env, indent="      | ")
    took = fmt_secs(time.monotonic() - started)
    if code == 0:
        say(f"      ✅ 통과 ({took})")
    elif collection_error(out) and (mod := missing_module(out)):
        res.warnings.append(f"{project}: 이 PC 에 '{mod}' 패키지가 없어 테스트를 돌리지 못함 (막지 않음)")
        say(f"      ⚠ 패키지 '{mod}' 없음 — 건너뜀")
    else:
        lines = failed_lines(out) or [f"pytest 종료 코드 {code}"]
        res.failures.append(f"{project} 테스트 실패:\n" + "\n".join(f"      {ln}" for ln in lines[:15]))
        say(f"      ❌ 실패 ({took})")


def worktree_is_clean(root: Path, projects: list[str]) -> bool:
    """검사할 폴더에 커밋 안 한 변경·추적 안 하는 새 파일이 없나 (.gitignore 된 것은 상관없음)."""
    if not projects:
        return True
    out = git("status", "--porcelain", "-z", "--", *projects, cwd=root)
    return not out.strip("\0")


@dataclass
class TestTree:
    root: Path
    temporary: bool


def open_test_tree(root: Path, tip: str, projects: list[str]) -> TestTree:
    """보내는 커밋과 똑같은 파일에서 테스트한다.
    지금 폴더가 그 커밋이고(HEAD) 검사할 폴더에 커밋 안 한 변경이 없으면 그대로 쓰고, 아니면 임시 작업 폴더를 만든다."""
    head = git("rev-parse", "HEAD", cwd=root).strip()
    if head == tip and worktree_is_clean(root, projects):
        return TestTree(root, temporary=False)
    base = gitleaks_tool.tools_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="prepush-tree-", dir=base))
    path.rmdir()
    why = "보내는 커밋이 지금 폴더(HEAD)와 다름" if head != tip else "커밋 안 한 변경이 있음"
    say(f"    ({why} → 보내는 커밋 {tip[:8]} 을 임시 폴더에 꺼내서 테스트)")
    git("worktree", "add", "--detach", "--quiet", str(path), tip, cwd=root)
    return TestTree(path, temporary=True)


def close_test_tree(root: Path, tree: TestTree) -> None:
    if tree.temporary:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree.root)], cwd=root, capture_output=True)
        shutil.rmtree(tree.root, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=root, capture_output=True)


def run_tests(root: Path, tip: str, changed: list[str], res: Result) -> bool:
    """돌렸으면 True (Agent Office 를 건드렸는지 여부는 버전 경고에서 쓴다)."""
    say("[2/3] 바뀐 부분 테스트")
    ao_changed, other_changed = select_tests.split_changed(changed)

    def has_tests_dir(d: str) -> bool:
        return git("ls-tree", "-d", "--name-only", f"{tip}:{d}" if d else tip, cwd=root, check=False).splitlines().count("tests") > 0

    others = select_tests.plan_other(other_changed, has_tests_dir)
    projects = ([select_tests.AO_DIR] if ao_changed else []) + list(others)
    if not projects:
        say("    Agent Office·테스트가 있는 폴더가 바뀌지 않음 — 건너뜀")
        return False
    tree = open_test_tree(root, tip, projects)
    try:
        env = test_env(tree.root)
        if ao_changed:
            run_agent_office(tree.root, ao_changed, res, env)
        for project, files in others.items():
            run_other(tree.root, project, files, res, env)
    finally:
        close_test_tree(root, tree)
    return bool(ao_changed)


# ─────────────────────────── 3. 버전 경고 ───────────────────────────

def warn_versions(root: Path, tip: str, res: Result) -> None:
    say("[3/3] 핵심 패키지 버전 (경고만)")
    shown = subprocess.run(["git", "show", f"{tip}:{select_tests.AO_DIR}/requirements.txt"], cwd=root, capture_output=True)
    if shown.returncode != 0:
        say("    requirements.txt 없음 — 건너뜀")
        return
    lines = versions.check(shown.stdout.decode("utf-8", errors="replace"), gitleaks_tool.tools_dir() / "pypi_cache.json")
    if not lines:
        say("    ✅ CI 와 같거나 확인할 수 없음 (인터넷 안 됨)")
    for ln in lines:
        res.warnings.append(ln)
        say(f"    ⚠ {ln}")


# ─────────────────────────── 실행 ───────────────────────────

def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str], stdin_text: str) -> int:
    setup_console()
    started = time.monotonic()
    root = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
    if len(argv) >= 1 and argv[0] == "--manual":
        remote = "origin"
        ref = argv[1] if len(argv) >= 2 else "HEAD"
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=root).strip()
        updates = [Update(ref, sha, "(검사만)", ZERO)]
        target = f"{ref} (검사만, push 안 함)"
    else:
        remote = argv[0] if argv else "origin"
        updates = parse_push_lines(stdin_text)
        target = ", ".join(f"{u.local_ref.removeprefix('refs/heads/')} → {remote}/{u.remote_ref.removeprefix('refs/heads/')}"
                           for u in updates if u.local_sha != ZERO)
    jobs = []  # (update, revs)
    flt = remote_filter(remote, git("remote", cwd=root).split())
    for u in updates:
        revs = rev_args(u, flt, lambda s: commit_known(root, s))
        if revs is not None:
            jobs.append((u, revs))
    if not jobs:
        return 0
    commits = sorted({c for _, revs in jobs for c in git("rev-list", *revs, cwd=root).split()})
    say("━━━━━━━━ 올리기 전 검사 ━━━━━━━━")
    say(f"대상: {target} · 새 커밋 {len(commits)}개")
    if not commits:
        say("새 커밋이 없어 검사할 것이 없습니다 — 통과")
        return 0

    res = Result()
    run_gitleaks(root, [revs for _, revs in jobs], jobs[0][0].local_sha, res)
    ao_touched = False
    seen_tips: set[str] = set()
    for u, revs in jobs:
        if u.local_sha in seen_tips:
            continue
        seen_tips.add(u.local_sha)
        if len(jobs) > 1:
            say(f"  ({u.local_ref} {u.local_sha[:8]})")
        ao_touched |= run_tests(root, u.local_sha, changed_files(root, revs), res)
    if ao_touched:
        warn_versions(root, jobs[0][0].local_sha, res)
    else:
        say("[3/3] 핵심 패키지 버전 — Agent Office 가 바뀌지 않아 건너뜀")

    took = fmt_secs(time.monotonic() - started)
    say("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if res.failures:
        say(f"❌ push 를 막았습니다 ({took}). 고친 뒤 다시 push 하세요.")
        for f in res.failures:
            say(f"  - {f}")
        if res.secret_found:
            say("  ⚠ 비밀 검사에 걸린 것은 건너뛰지 마세요 — 공개 저장소에 올라가면 지워도 이미 복사된 것으로 봅니다.")
        else:
            say("  급하면: SKIP_PREPUSH=1 git push … (GitHub 의 검사는 그대로 돕니다)")
        return 1
    say(f"✅ 통과 ({took})" + (f" — 경고 {len(res.warnings)}개" if res.warnings else ""))
    return 0


if __name__ == "__main__":
    stdin_text = "" if (len(sys.argv) > 1 and sys.argv[1] == "--manual") else sys.stdin.read()
    try:
        sys.exit(main(sys.argv[1:], stdin_text))
    except KeyboardInterrupt:
        print("\n중단됨 — push 하지 않습니다.")
        sys.exit(130)
