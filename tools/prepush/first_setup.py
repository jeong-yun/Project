"""올리기 전 검사 처음 설정 — PC 마다 한 번 (tools/setup_prepush.bat 이 부른다).

1. git config core.hooksPath .githooks   → push 때 .githooks/pre-push 가 돈다 (이 저장소에만, 다시 해도 같음)
   git config prepush.python <이 파이썬>  → 훅이 이 파이썬으로 검사한다
2. gitleaks 8.30.1 내려받기 (공식 릴리스 zip, SHA256 확인) → %LOCALAPPDATA%\\agent-office-tools\\
3. 테스트 도구 확인 (pytest·Agent Office 패키지) — 없으면 설치 명령을 알려 준다 (직접 설치하지 않음)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gitleaks_tool  # noqa: E402

REPO = HERE.parent.parent
NEEDED = ("pytest", "streamlit", "langgraph", "langchain_core")


def configure_hooks(repo: Path = REPO) -> None:
    cur = subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=repo, capture_output=True, text=True).stdout.strip()
    if cur == ".githooks":
        print("1) git 훅 폴더: 이미 .githooks 로 설정됨")
        return
    if cur:
        print(f"1) git 훅 폴더가 '{cur}' 였습니다 → .githooks 로 바꿉니다")
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=repo, check=True)
    print("1) git 훅 폴더: .githooks 로 설정함 (이제 push 할 때마다 검사가 돕니다)")


def remember_python(repo: Path = REPO) -> None:
    """훅이 이 파이썬(pytest·Agent Office 패키지가 있는 것)을 쓰게 저장소 설정에 적어 둔다."""
    exe = sys.executable.replace("\\", "/")
    subprocess.run(["git", "config", "prepush.python", exe], cwd=repo, check=True)
    print(f"   검사에 쓸 파이썬: {exe}")


def missing_packages() -> list[str]:
    return [m for m in NEEDED if importlib.util.find_spec(m) is None]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", **({} if stream.isatty() else {"encoding": "utf-8"}))
        except (AttributeError, ValueError):
            pass
    print(f"저장소: {REPO}")
    print(f"파이썬: {sys.executable}")
    configure_hooks()
    remember_python()
    print("2) gitleaks:")
    try:
        gitleaks_tool.install()
    except Exception as e:  # noqa: BLE001
        print(f"   ❌ gitleaks 설치 실패: {e}")
        print("   인터넷 연결을 확인하고 다시 실행하세요. 이대로면 push 때 검사가 멈춥니다 (급하면 SKIP_PREPUSH=1).")
        return 1
    miss = missing_packages()
    if miss:
        print(f"3) ⚠ 이 파이썬에 없는 패키지: {', '.join(miss)}")
        print("   Agent Office 테스트를 돌리려면 한 번 설치하세요:")
        print(f'   "{sys.executable}" -m pip install -r LLM\\Agent_office\\requirements-dev.txt')
        return 1
    print("3) 테스트 도구: pytest·Agent Office 패키지 있음")
    print("\n설정 끝. 평소에는 바탕화면의 Agent_Office_검사후_올리기.bat 으로 올리세요 (check/main → 검사 통과 시 main 자동 반영).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
