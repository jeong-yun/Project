"""push 할 커밋에서 바뀐 파일 → 돌릴 테스트 고르기 (순수 함수 — git·pytest 를 부르지 않는다. 단위 테스트: tests/test_select_tests.py).

고르는 규칙 (Agent Office = LLM/Agent_office 기준 경로)
  1순위 — 늘 돌린다
    - 바뀐 테스트 파일                       tests/test_x.py            → 그 파일
    - 바뀐 파이썬 모듈의 '이름 짝' 테스트      core/feeds.py              → tests/test_feeds.py
                                             edu_hobby_ui.py            → tests/test_edu_hobby_ui.py, tests/test_edu_hobby.py
    - 설정 파일                               config/*.yaml 등           → tests/test_config.py
    - 팀 설정                                 teams/<팀>/…(.py 아닌 것)   → tests/test_teams.py, tests/test_<팀>.py
    - 넓게 영향을 주는 파일 → '대표 묶음'(smoke_tests.txt)
        app.py, tests/conftest.py·fakes.py·fixtures/, requirements*.txt
  2순위 — 다 합쳐 상한(기본 8개 파일) 안이면 돌리고, 넘으면 이름만 보여 주고 check/main 검사(GitHub, 전체 테스트)에 맡긴다
    - 바뀐 모듈을 import 하는 테스트 (직접 import 만 — 거쳐서 쓰는 것은 따지지 않는다)
    - 바뀐 설정 파일 이름(예: models.yaml)이 본문에 나오는 테스트
  그 밖(문서·그림·knowledge 등)은 테스트를 고르지 않는다. 어떤 경우든 전체 테스트를 '모으기만' 해서 import 오류는 잡는다.

Agent Office 밖
  - 바뀐 파일에서 위로 올라가며 처음 만나는 tests/ 폴더가 있는 폴더(저장소 루트 제외)의 테스트를 통째로 돌린다.
    예: GAME/Soccer/x.py → GAME/Soccer/tests, LLM/train_set_making_by_rule(talos, et)/3_0b_mask.py → 그 폴더의 tests
  - .githooks/ 는 이 검사기(tools/prepush)의 테스트로 본다.
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

AO_DIR = "LLM/Agent_office"
PREPUSH_DIR = "tools/prepush"
DEFAULT_MAX_FILES = 8

# 넓게 영향을 주는 Agent Office 파일 → 대표 묶음
BROAD_FILES = {
    "app.py": "app.py(앱 시작 파일)",
    "tests/conftest.py": "tests/conftest.py(공통 테스트 준비물)",
    "tests/fakes.py": "tests/fakes.py(가짜 모델)",
    "tests/__init__.py": "tests/__init__.py",
    "requirements.txt": "requirements.txt(패키지 목록)",
    "requirements-dev.txt": "requirements-dev.txt(패키지 목록)",
}


@dataclass
class AOPlan:
    """Agent Office 테스트 고르기 결과. 경로는 LLM/Agent_office 기준 posix 경로."""

    touched: bool = False                                  # Agent Office 안 파일이 하나라도 바뀌었나
    first: dict[str, str] = field(default_factory=dict)    # 1순위: 테스트 파일 → 고른 이유
    second: dict[str, str] = field(default_factory=dict)   # 2순위: 테스트 파일 → 고른 이유
    smoke_reasons: list[str] = field(default_factory=list)  # 대표 묶음을 넣은 이유 (비어 있으면 안 넣음)
    no_test: list[str] = field(default_factory=list)        # 테스트를 고르지 않은 바뀐 파일 (문서 등)

    def add_first(self, test: str, reason: str) -> None:
        self.first.setdefault(test, reason)
        self.second.pop(test, None)

    def add_second(self, test: str, reason: str) -> None:
        if test not in self.first:
            self.second.setdefault(test, reason)


@dataclass
class RunList:
    run: dict[str, str]        # 실제로 돌릴 것 (테스트 파일 또는 node id → 이유)
    left_to_ci: list[str]      # 상한을 넘어 CI 에 맡긴 2순위 테스트 파일


def split_changed(paths: list[str]) -> tuple[list[str], list[str]]:
    """바뀐 파일(저장소 기준) → (Agent Office 안 파일들[AO 기준 경로], 그 밖 파일들[저장소 기준])."""
    prefix = AO_DIR + "/"
    ao, other = [], []
    for p in paths:
        p = p.replace("\\", "/")
        if p.startswith(prefix):
            ao.append(p[len(prefix):])
        else:
            other.append(p)
    return ao, other


def module_name(rel: str) -> str | None:
    """AO 기준 .py 경로 → import 이름. core/feeds.py → core.feeds, core/tools/__init__.py → core.tools."""
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def imported_modules(source: str) -> set[str]:
    """파이썬 소스가 import 하는 이름들. `from core import feeds` 는 core 와 core.feeds 를 둘 다 넣는다 (함수 안 import 포함)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def imports_module(imported: set[str], module: str) -> bool:
    return any(name == module or name.startswith(module + ".") for name in imported)


def plan_agent_office(changed: list[str], test_sources: dict[str, str]) -> AOPlan:
    """changed: AO 기준 바뀐 경로. test_sources: 지금 있는 tests/test_*.py → 소스 글자."""
    plan = AOPlan(touched=bool(changed))
    tests = set(test_sources)
    imports = {t: imported_modules(src) for t, src in test_sources.items()}

    def name_pair(stem: str, why: str) -> bool:
        found = False
        for cand in (f"tests/test_{stem}.py", f"tests/test_{stem[:-3]}.py" if stem.endswith("_ui") else None):
            if cand and cand in tests:
                plan.add_first(cand, why)
                found = True
        return found

    for rel in changed:
        if rel in BROAD_FILES or rel.startswith("tests/fixtures/"):
            plan.smoke_reasons.append(BROAD_FILES.get(rel, "tests/fixtures/(테스트 자료)"))
            if rel == "tests/fakes.py":
                for t in sorted(tests):
                    if imports_module(imports[t], "tests.fakes"):
                        plan.add_second(t, "tests.fakes 를 import 함")
            continue
        if rel.startswith("tests/"):
            if rel in tests:
                plan.add_first(rel, "바뀐 테스트 파일")
            elif rel.endswith(".py") and not PurePosixPath(rel).name.startswith("test_"):  # 지운 테스트 파일은 무시
                plan.smoke_reasons.append(f"{rel}(테스트 도우미)")
            continue
        parts = rel.split("/")
        if rel.endswith(".py"):
            mod = module_name(rel)
            stem = PurePosixPath(rel).stem
            if stem == "__init__":
                stem = parts[-2] if len(parts) >= 2 else stem
            name_pair(stem, f"{rel} 의 이름 짝")
            if mod:
                for t in sorted(tests):
                    if imports_module(imports[t], mod):
                        plan.add_second(t, f"{mod} 를 import 함")
            continue
        if parts[0] == "config":
            name = parts[-1]
            if "tests/test_config.py" in tests:
                plan.add_first("tests/test_config.py", f"설정 파일 {rel}")
            pat = re.compile(re.escape(name))
            for t in sorted(tests):
                if pat.search(test_sources[t]):
                    plan.add_second(t, f"{name} 이 본문에 나옴")
            continue
        if parts[0] == "teams" and len(parts) >= 2:
            team = parts[1]
            for cand in ("tests/test_teams.py", f"tests/test_{team}.py"):
                if cand in tests:
                    plan.add_first(cand, f"팀 설정 teams/{team}")
            continue
        plan.no_test.append(rel)
    return plan


def finalize(plan: AOPlan, smoke: list[str], max_files: int = DEFAULT_MAX_FILES, run_all: bool = False) -> RunList:
    """1순위 + 대표 묶음은 늘, 2순위는 다 합쳐 max_files 개 파일 이하일 때만 (run_all 이면 상한 없음)."""
    run: dict[str, str] = dict(plan.first)
    if plan.smoke_reasons:
        why = "대표 묶음 — " + ", ".join(dict.fromkeys(plan.smoke_reasons))
        for node in smoke:
            if node.split("::", 1)[0] not in run:
                run.setdefault(node, why)
    files = {k.split("::", 1)[0] for k in run}
    extra = [t for t in plan.second if t not in files]
    if run_all or len(files) + len(extra) <= max_files:
        for t in extra:
            run[t] = plan.second[t]
        return RunList(run=run, left_to_ci=[])
    return RunList(run=run, left_to_ci=extra)


def other_project_of(path: str, has_tests_dir) -> str | None:
    """Agent Office 밖 파일 → 테스트를 돌릴 폴더 (저장소 기준). 위로 올라가며 처음 만나는 tests/ 가 있는 폴더. 없으면 None.
    has_tests_dir(폴더) → 그 폴더 아래에 tests/ 폴더가 있는지."""
    path = path.replace("\\", "/")
    if path.startswith(".githooks/"):
        return PREPUSH_DIR
    d = posixpath.dirname(path)
    while d and d != ".":
        if posixpath.basename(d) == "tests":
            d = posixpath.dirname(d)
            continue
        if has_tests_dir(d):
            return d
        d = posixpath.dirname(d)
    return None


def plan_other(paths: list[str], has_tests_dir) -> dict[str, list[str]]:
    """Agent Office 밖 바뀐 파일들 → {테스트 폴더가 있는 프로젝트: [바뀐 파일들]}."""
    out: dict[str, list[str]] = {}
    for p in paths:
        proj = other_project_of(p, has_tests_dir)
        if proj and proj != AO_DIR:
            out.setdefault(proj, []).append(p)
    return out


def read_smoke(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
