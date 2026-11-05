"""검사 후 올리기 — 바탕화면 Agent_Office_검사후_올리기.bat 이 부른다.

하는 일
  1. 처음 설정이 안 됐으면 먼저 한다 (git 훅 폴더·gitleaks — first_setup.py)
  2. GitHub 최신 상태를 받아(fetch) 올릴 수 있는지 본다
     - 내 main 이 GitHub main 보다 뒤처져 있으면 멈춤 → 먼저 pull
     - GitHub check/main 에 내 main 에 없는 커밋이 있으면(다른 PC 에서 올려 검사 중·실패, 또는 고쳐 쓴 커밋) 보여 주고 덮어쓸지 묻는다
  3. git push origin main:check/main — 이때 pre-push 훅이 '올리기 전 검사'를 돌리고, 실패하면 push 가 멈춘다
  4. GitHub Actions 결과 페이지 주소를 보여 준다. 검사가 모두 통과하면 promote-to-main 이 main 에 반영한다 (빨리 감기만).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import first_setup  # noqa: E402
import gitleaks_tool  # noqa: E402

REPO = HERE.parent.parent
CHECK_BRANCH = "check/main"
WORKFLOW_FILE = "promote-to-main.yml"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and res.returncode != 0:
        raise SystemExit(f"❌ git {' '.join(args)} 실패:\n{res.stderr.strip()}")
    return res


def web_url(remote_url: str) -> str | None:
    """원격 주소 → GitHub 웹 주소. https·ssh 둘 다."""
    remote_url = remote_url.strip()
    m = re.match(r"^(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+)/(.+?)(?:\.git)?/?$", remote_url)
    return f"https://github.com/{m.group(1)}/{m.group(2)}" if m else None


def actions_url(web: str) -> str:
    return f"{web}/actions/workflows/{WORKFLOW_FILE}?query=branch%3Acheck%2Fmain"


def is_ancestor(a: str, b: str) -> bool:
    return git("merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def ask_yes(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes", "ㅛ")
    except EOFError:
        return False


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", **({} if stream.isatty() else {"encoding": "utf-8"}))
        except (AttributeError, ValueError):
            pass
    print(f"저장소: {REPO}")
    hooks = git("config", "--get", "core.hooksPath", check=False).stdout.strip()
    if hooks != ".githooks" or gitleaks_tool.installed_version() != gitleaks_tool.VERSION:
        print("처음 설정이 필요합니다 — 지금 합니다.\n")
        if first_setup.main() != 0:
            print("\n❌ 설정을 마치지 못해 멈춥니다. 위 안내대로 한 뒤 다시 실행하세요.")
            return 1
        print()

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        print(f"ℹ 지금 작업 브랜치는 '{branch}' 입니다. 올리는 것은 main 브랜치입니다.")
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        print("ℹ 커밋하지 않은 변경이 있습니다 — 커밋한 것만 올라갑니다.")

    print("GitHub 최신 상태 받는 중 (git fetch)…")
    git("fetch", "--quiet", "origin")
    if git("rev-parse", "--verify", "--quiet", "origin/main", check=False).returncode != 0:
        print("❌ GitHub 에서 main 을 찾지 못했습니다.")
        return 1
    if not is_ancestor("origin/main", "main"):
        print("❌ GitHub 의 main 에 내 PC 에 없는 커밋이 있습니다 (다른 PC 에서 올린 것).")
        print("   먼저 받으세요:  git pull --ff-only origin main   (충돌이 나면 Claude 에게 알려 주세요)")
        return 1
    new = git("rev-list", "--count", "origin/main..main").stdout.strip()
    if new == "0":
        print("✅ 올릴 새 커밋이 없습니다 — main 이 GitHub 와 같습니다.")
        return 0

    push = ["push"]
    has_check = git("rev-parse", "--verify", "--quiet", f"origin/{CHECK_BRANCH}", check=False)
    if has_check.returncode == 0:
        check_sha = has_check.stdout.strip()
        if not is_ancestor(check_sha, "main"):
            print(f"⚠ GitHub 의 {CHECK_BRANCH} 에 내 main 에 없는 커밋이 있습니다:")
            print(git("log", "--oneline", "-10", f"main..{check_sha}").stdout.rstrip())
            print("   다른 PC 에서 올려 아직 검사 중이거나 검사에 실패한 커밋, 또는 내가 고쳐 쓴(amend·rebase) 커밋입니다.")
            print("   다른 PC 것이면 'N' 을 누르고 그 PC 에서 결과를 확인하세요. 내가 버린 커밋이면 'y'.")
            if not ask_yes(f"   {CHECK_BRANCH} 을 내 main 으로 덮어쓸까요? (main 은 건드리지 않습니다)"):
                print("멈췄습니다.")
                return 1
            # 방금 본 커밋일 때만 덮어쓴다 — 그 사이 누가 또 올렸으면 git 이 거절한다
            push.append(f"--force-with-lease=refs/heads/{CHECK_BRANCH}:{check_sha}")

    print(f"\n새 커밋 {new}개를 {CHECK_BRANCH} 으로 올립니다. 먼저 올리기 전 검사가 돕니다…\n")
    res = subprocess.run(["git", *push, "origin", f"refs/heads/main:refs/heads/{CHECK_BRANCH}"], cwd=REPO)
    if res.returncode != 0:
        print("\n❌ 올리지 못했습니다. 위 '올리기 전 검사' 결과나 git 메시지를 보고 고친 뒤 다시 실행하세요.")
        return 1
    sha = git("rev-parse", "--short", "main").stdout.strip()
    web = web_url(git("remote", "get-url", "origin").stdout)
    print(f"\n✅ {CHECK_BRANCH} 으로 올렸습니다 (커밋 {sha}).")
    print("   GitHub 이 전체 테스트·비밀 검사를 돌리고, 모두 통과하면 main 에 자동 반영합니다 (보통 15~25분).")
    if web:
        print(f"   결과 보기: {actions_url(web)}")
    print("   실패하면 GitHub 에서 메일이 오고 main 은 그대로입니다. 고쳐서 커밋한 뒤 이 파일을 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
