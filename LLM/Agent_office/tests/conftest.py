"""pytest 공용 준비물 (fixture).

- 실제 웹 검색·LLM 호출을 모두 막고 가짜로 바꾼다 → API 키·비용·네트워크 없이 테스트
- 설정은 실제 config/, teams/ 파일을 읽는다 → 설정 파일이 깨지면 테스트도 실패한다
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import list_teams, load_models, load_ranks, team_org  # noqa: E402

ALL_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "UPSTAGE_API_KEY")


@pytest.fixture(autouse=True)
def no_network_search(monkeypatch):
    """테스트 중 실제 웹 검색을 막는다. 호출된 검색어는 반환값 리스트에 기록된다."""
    from core import search

    calls: list[str] = []

    def fake_backend(query, max_results, region, **kwargs):
        calls.append(query)
        return [
            search.SearchResult(title=f"{query} 결과", url=f"https://example.com/{len(calls)}", snippet="요약"),
            search.SearchResult(title="중복", url="https://example.com/dup", snippet="중복 결과"),
        ]

    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", fake_backend)
    return calls


@pytest.fixture
def ranks():
    return load_ranks()


@pytest.fixture
def models():
    return load_models()


@pytest.fixture
def team():
    return next(t for t in list_teams() if t.id == "general")


@pytest.fixture
def org(team, ranks):
    return team_org(team, ranks)


@pytest.fixture
def set_keys(monkeypatch):
    """지정한 provider 의 키만 존재하는 환경을 만든다. 예: set_keys("google", "openai")"""

    def _set(*providers: str):
        for env_key in ALL_KEYS:
            monkeypatch.delenv(env_key, raising=False)
        for provider in providers:
            monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-key")

    return _set


@pytest.fixture
def use_script(set_keys, monkeypatch):
    """모든 모델 호출을 Script 대로 답하는 가짜 모델로 바꾼다. 사용: script = use_script(Script(...))"""
    from core import llm_factory
    from tests.fakes import FakeOfficeModel

    def _install(script):
        set_keys("openai", "google", "upstage")
        monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, models: FakeOfficeModel(script=script))
        return script

    return _install
