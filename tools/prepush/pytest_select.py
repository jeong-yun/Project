"""pytest 플러그인 — 전체 테스트를 '모으기'(import 검사)는 다 하고, 실행은 고른 것만 한다.

PREPUSH_SELECT 환경 변수가 가리키는 파일에 한 줄에 하나씩 테스트 파일 또는 node id 를 적는다.
(모을 때 import 오류가 난 파일이 있으면 pytest 가 실행 전에 멈추므로, 고르지 않은 파일의 import 오류도 잡힌다.)
"""

from __future__ import annotations

import os
from pathlib import Path


def _wanted() -> list[str]:
    path = os.environ.get("PREPUSH_SELECT")
    if not path:
        return []
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def matches(nodeid: str, wanted: list[str]) -> bool:
    return any(nodeid == w or nodeid.startswith(w + "::") or nodeid.startswith(w + "[") for w in wanted)


def pytest_collection_modifyitems(config, items):
    wanted = _wanted()
    keep, drop = [], []
    for item in items:
        (keep if matches(item.nodeid, wanted) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
