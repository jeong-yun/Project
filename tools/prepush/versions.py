"""핵심 패키지 버전 경고 — 이 PC 에 설치된 버전과 CI 가 받을(=requirements 범위 안 최신) 버전이 다르면 알려만 준다 (막지 않음).

CI(GitHub Actions)는 매번 requirements 범위 안의 최신 버전을 새로 설치한다. 예: 이 PC streamlit 1.43, CI 1.66.
그래서 이 PC 에서 통과해도 CI 에서 화면 표시가 다를 수 있다 — 그 차이는 check/main 검사(GitHub)가 잡는다.
PyPI 목록은 12시간 동안 도구 폴더에 저장해 두고 다시 쓴다. 인터넷이 안 되면 조용히 건너뛴다.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from importlib import metadata
from pathlib import Path

CORE_PACKAGES = ("streamlit", "langgraph", "langchain-core")
CACHE_TTL = 12 * 3600


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_specs(text: str) -> dict[str, str]:
    """requirements.txt 글자 → {패키지 이름(정규화): 범위 문자열}. 주석·-r 줄은 건너뛴다."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", line)
        if m:
            out[_norm(m.group(1))] = m.group(3).split(";", 1)[0].strip()
    return out


def latest_allowed(versions: list[str], spec: str) -> str | None:
    """정식(시험판 아님) 버전 중 범위 안 최신."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    allowed = SpecifierSet(spec)
    best = None
    for v in versions:
        try:
            ver = Version(v)
        except InvalidVersion:
            continue
        if ver.is_prerelease or ver.is_devrelease or ver not in allowed:
            continue
        if best is None or ver > best:
            best = ver
    return str(best) if best else None


def pypi_versions(name: str, cache_file: Path | None = None, timeout: float = 4.0) -> list[str] | None:
    """PyPI 에 올라온 (철회되지 않은) 버전 목록. 실패하면 None."""
    cache: dict = {}
    if cache_file and cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    hit = cache.get(name)
    if hit and time.time() - hit.get("at", 0) < CACHE_TTL:
        return hit["versions"]
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — 오프라인·차단 등: 경고만 못 할 뿐
        return None
    versions = [v for v, files in data.get("releases", {}).items() if files and not all(f.get("yanked") for f in files)]
    if cache_file:
        cache[name] = {"at": time.time(), "versions": versions}
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    return versions


def installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def compare(installed_versions: dict[str, str | None], ci_versions: dict[str, str | None]) -> list[str]:
    """다른 것만 경고 문장으로."""
    lines = []
    for name in CORE_PACKAGES:
        here, ci = installed_versions.get(name), ci_versions.get(name)
        if not ci:
            continue
        if here is None:
            lines.append(f"{name}: 이 PC 에 설치 안 됨 / CI {ci}")
        elif here != ci:
            lines.append(f"{name}: 이 PC {here} / CI {ci} — 버전 차이로 생기는 동작·화면 차이는 check/main 검사(GitHub)에서 확인됨")
    return lines


def check(requirements_text: str, cache_file: Path | None = None) -> list[str]:
    specs = requirement_specs(requirements_text)
    here, ci = {}, {}
    for name in CORE_PACKAGES:
        spec = specs.get(_norm(name))
        if spec is None:
            continue
        here[name] = installed(name)
        versions = pypi_versions(name, cache_file)
        ci[name] = latest_allowed(versions, spec) if versions else None
    return compare(here, ci)
