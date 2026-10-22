"""pytest 공용 준비물 (fixture).

- 실제 웹 검색·LLM 호출을 모두 막고 가짜로 바꾼다 → API 키·비용·네트워크 없이 테스트
- 설정은 실제 config/, teams/ 파일을 읽는다 → 설정 파일이 깨지면 테스트도 실패한다
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import list_teams, load_models, load_ranks, team_org  # noqa: E402

ALL_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "UPSTAGE_API_KEY", "FREELLMAPI_API_KEY")
ENV_KEY = {"freellm": "FREELLMAPI_API_KEY"}  # 이름 규칙이 다른 provider


@pytest.fixture(autouse=True)
def clear_model_cooldown():
    """할당량 소진으로 쉬게 한 모델 기록을 테스트마다 비운다."""
    from core import llm_factory

    llm_factory._cooldown.clear()
    yield
    llm_factory._cooldown.clear()


@pytest.fixture(autouse=True)
def no_claude_cli(monkeypatch):
    """실제 claude CLI(구독)를 부르지 않는다 — 기본은 '설치 안 됨'. claude_cli 테스트는 가짜 실행 파일로 바꿔 쓴다."""
    from core import claude_cli

    monkeypatch.setattr(claude_cli, "find_executable", lambda: None)
    claude_cli._status_cache.clear()
    yield
    claude_cli._status_cache.clear()


@pytest.fixture(autouse=True)
def no_usage_sink():
    """사용량 기록 연결을 테스트마다 끊는다 — app.py(AppTest)가 연결한 저장소가 다음 테스트로 새지 않게."""
    from core import usage

    usage.set_sink(None)
    usage.set_price_overrides(None)
    yield
    usage.set_sink(None)
    usage.set_price_overrides(None)


@pytest.fixture(autouse=True)
def no_network_search(monkeypatch):
    """테스트 중 실제 웹 검색을 막는다. 호출된 검색어는 반환값 리스트에 기록된다 (검색어들은 동시에 검색되므로 기록 순서는 달라질 수 있음)."""
    from core import search

    calls: list[str] = []

    def fake_backend(query, max_results, region, **kwargs):
        calls.append(query)
        return [
            search.SearchResult(title=f"{query} 결과", url=f"https://example.com/{query}", snippet="요약"),
            search.SearchResult(title="중복", url="https://example.com/dup", snippet="중복 결과"),
        ]

    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", fake_backend)
    return calls


@pytest.fixture(autouse=True)
def no_network_feeds(monkeypatch):
    """🏠 메인 화면 뉴스·SNS 수집이 테스트 중 실제 네트워크·백그라운드 수집을 하지 않게 한다.
    수집기 테스트는 가짜 http·뉴스 검색을 직접 넣는다. start_collect 호출은 반환 리스트에 기록된다."""
    from core import feeds, search

    def blocked(*args, **kwargs):
        raise RuntimeError("테스트 중 네트워크 금지")

    started: list[str] = []
    monkeypatch.setattr(search, "news_search", blocked)
    monkeypatch.setattr(feeds.Http, "_send", blocked)
    monkeypatch.setattr(feeds, "start_collect", lambda settings, models, data_dir, teams=None: started.append(str(data_dir)) or True)
    return started


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
            monkeypatch.setenv(ENV_KEY.get(provider, f"{provider.upper()}_API_KEY"), "test-key")

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
