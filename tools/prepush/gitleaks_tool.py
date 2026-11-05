"""gitleaks 실행 파일 — CI(.github/workflows/secret-scan.yml)와 같은 버전을 체크섬으로 확인해 저장소 밖 도구 폴더에 둔다.

- 버전·체크섬은 공식 릴리스의 checksums.txt 값을 그대로 고정한다:
  https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_checksums.txt
  (linux_x64 값은 secret-scan.yml 의 GITLEAKS_SHA256 과 같다 — 버전을 올릴 때는 두 곳을 함께 바꾼다)
- 내려받기는 사용자가 실행하는 설정(tools/setup_prepush.bat → `python tools/prepush/gitleaks_tool.py install`)만 한다.
  push 때 도는 검사(prepush.py)는 내려받지 않고, 없으면 설정을 먼저 하라고 알린다.
- 도구 폴더: %LOCALAPPDATA%\\agent-office-tools\\gitleaks-<버전>\\ (AGENT_OFFICE_TOOLS_DIR 로 바꿀 수 있음)
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "8.30.1"
SHA256 = {
    "windows_x64": "d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e",
    "windows_arm64": "b95f5e4f5c425cedca7ee203d9afd29597e692c4924a12ed42f970537c72cc0f",
    "windows_x32": "190ad53db301eec3e90afe3a1a75270768b8ebf89e731345e19421c32c1ae1a1",
    "linux_x64": "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
}
URL = "https://github.com/gitleaks/gitleaks/releases/download/v{v}/gitleaks_{v}_{plat}.{ext}"


def tools_dir() -> Path:
    override = os.environ.get("AGENT_OFFICE_TOOLS_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "agent-office-tools"


def exe_path() -> Path:
    name = "gitleaks.exe" if os.name == "nt" else "gitleaks"
    return tools_dir() / f"gitleaks-{VERSION}" / name


def platform_key() -> str:
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64", "x86": "x32", "i386": "x32", "i686": "x32"}.get(machine, machine)
    system = "windows" if os.name == "nt" else platform.system().lower()
    return f"{system}_{arch}"


def installed_version(exe: Path | None = None) -> str | None:
    """설치된 gitleaks 의 버전 (없거나 실행이 안 되면 None)."""
    exe = exe or exe_path()
    if not exe.is_file():
        return None
    try:
        out = subprocess.run([str(exe), "version"], capture_output=True, text=True, timeout=30)
    except OSError:
        return None
    return out.stdout.strip().lstrip("v") or None


def verify_sha256(data: bytes, expected: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"체크섬이 다름 — 받은 파일을 쓰지 않습니다 (기대 {expected[:12]}…, 실제 {actual[:12]}…)")


def extract_binary(archive: bytes, want: str) -> bytes:
    """zip 안에서 실행 파일 하나만 꺼낸다 (다른 경로로 풀지 않으므로 zip 경로 조작 걱정이 없다)."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            if info.filename.replace("\\", "/").rsplit("/", 1)[-1] == want and not info.is_dir():
                return zf.read(info)
    raise ValueError(f"압축 파일 안에 {want} 가 없습니다")


def install(force: bool = False) -> Path:
    exe = exe_path()
    if not force and installed_version(exe) == VERSION:
        print(f"gitleaks {VERSION} 이미 설치됨: {exe}")
        return exe
    key = platform_key()
    if key not in SHA256 or not key.startswith("windows"):
        raise SystemExit(f"이 설치기는 Windows 용입니다 (지금: {key}). 다른 OS 는 gitleaks {VERSION} 을 직접 설치하세요.")
    url = URL.format(v=VERSION, plat=key, ext="zip")
    print(f"내려받는 중: {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — 고정된 공식 https 주소
        data = resp.read()
    verify_sha256(data, SHA256[key])
    print(f"체크섬 확인됨 (SHA256 {SHA256[key][:16]}…)")
    binary = extract_binary(data, exe.name)
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(".tmp")
    tmp.write_bytes(binary)
    os.replace(tmp, exe)
    got = installed_version(exe)
    if got != VERSION:
        raise SystemExit(f"설치한 gitleaks 버전이 {got} 입니다 (기대 {VERSION})")
    print(f"gitleaks {VERSION} 설치됨: {exe}")
    return exe


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "install":
        install(force="--force" in sys.argv)
    elif len(sys.argv) >= 2 and sys.argv[1] == "where":
        print(exe_path())
    else:
        print("사용: python tools/prepush/gitleaks_tool.py install [--force] | where")
        sys.exit(2)
