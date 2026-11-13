"""pytest 공용 준비물 (fixture).

- 실제 웹 검색·LLM 호출을 모두 막고 가짜로 바꾼다 → API 키·비용·네트워크 없이 테스트
- 모든 테스트에서 `.env` 를 읽지 않고 제공처 키를 지운다. 가짜 모델을 끼우지 않고 모델을 만들면 그 테스트는 실패한다
  (no_real_keys · no_real_models — 확인: tests/test_conftest_guards.py)
- 설정은 실제 config/, teams/ 파일을 읽는다 → 설정 파일이 깨지면 테스트도 실패한다
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, llm_factory  # noqa: E402
from core.config import list_teams, load_models, load_ranks, team_org  # noqa: E402

ALL_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "UPSTAGE_API_KEY", "FREELLMAPI_API_KEY")
ENV_KEY = {"freellm": "FREELLMAPI_API_KEY"}  # 이름 규칙이 다른 provider
SECRET_ENV = (*ALL_KEYS, "GEMINI_API_KEY")  # 테스트마다 지우는 키 (GEMINI_API_KEY = 음악 Lyria·채점이 대신 읽는 이름)
REAL_CREATE_CHAT_MODEL = llm_factory.create_chat_model  # no_real_models 가 막기 전의 원래 함수
BLOCKED_MODEL = "테스트 중 실제 모델 금지"


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """`.env` 의 실제 키가 테스트에 들어오지 못하게 한다 — 모든 테스트에서.

    - `.env` 읽기(core.config.load_env 안의 load_dotenv)를 아무것도 하지 않게 바꾼다. app.py·스크립트가 load_env 를
      어떻게 가져다 쓰든 이 한 곳을 지난다. 읽으려던 경로는 반환 리스트에 기록된다.
      (전에는 앱 화면 테스트가 돌 때 app.py 가 `.env` 를 읽어, set_keys() 로 지운 키까지 되살리고 다음 테스트로 넘겼다.)
    - 셸에 있던 제공처 키도 테스트 시작 때 지운다. 키가 필요한 테스트는 set_keys 로 가짜 키("test-key")를 넣는다.
    """
    asked: list[str] = []
    monkeypatch.setattr(config, "load_dotenv", lambda path=None, **kwargs: asked.append(str(path)) or False)
    for name in SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    return asked


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """가짜 모델을 끼우지 않은 테스트가 모델을 만들면(llm_factory.create_chat_model) 실패시킨다 — 모든 테스트에서.

    앱 코드는 모델 오류를 삼키고 다음 후보·낱말 비교로 넘어가므로, 만들 때 오류를 내는 것으로는 부족하다
    → 닿은 모델을 반환 리스트에 적어 두고 테스트가 끝날 때 실패시킨다.
    가짜는 이 위에 덮어쓴다: use_script, 또는 monkeypatch.setattr(llm_factory, "create_chat_model", …).
    모델 객체를 만들어 설정만 확인하는 테스트는 allow_model_objects 를 쓴다.
    """
    reached: list[str] = []

    def blocked(spec, models):
        reached.append(spec.label)
        raise RuntimeError(f"{BLOCKED_MODEL} ({spec.label}) — use_script 같은 가짜 모델을 끼우세요. "
                           "모델 객체만 만들어 설정을 확인하는 테스트는 allow_model_objects 를 쓰세요.")

    monkeypatch.setattr(llm_factory, "create_chat_model", blocked)
    yield reached
    if reached:
        pytest.fail(f"{BLOCKED_MODEL} — 가짜 모델 없이 모델을 만들려 했습니다: {', '.join(reached)}. "
                    "use_script 같은 가짜 모델을 끼우세요 (tests/conftest.py 의 no_real_models).", pytrace=False)


@pytest.fixture
def allow_model_objects(monkeypatch, no_real_models):
    """모델 객체를 '만들기만' 하고 부르지 않는 테스트용 — 원래 create_chat_model 을 되돌린다.

    키는 set_keys 의 가짜 키뿐이고(`.env` 는 읽지 않는다) claude_cli 는 가짜 실행 파일(fake_cli)로만 돈다.
    이 준비물을 쓴 테스트에서 .invoke()·.stream() 을 부르려면 그 모델을 가짜로 바꿔야 한다."""
    monkeypatch.setattr(llm_factory, "create_chat_model", REAL_CREATE_CHAT_MODEL)


@pytest.fixture(autouse=True)
def clear_model_cooldown():
    """할당량 소진으로 쉬게 한 모델 기록을 테스트마다 비운다."""
    llm_factory._cooldown.clear()
    yield
    llm_factory._cooldown.clear()


@pytest.fixture(autouse=True)
def no_model_caps_data():
    """모델 능력 판단의 데이터 폴더 연결(app.py 가 연결)·알림 기록을 테스트마다 끊는다 — 이름 규칙·능력표만 쓴다."""
    from core import model_caps

    model_caps.configure(None)
    model_caps._notices.clear()
    yield
    model_caps.configure(None)
    model_caps._notices.clear()


@pytest.fixture(autouse=True)
def no_model_recs():
    """⭐ 역할별 추천 모델(config/model_recommendations.yaml)을 기본으로 끈다 — 직급 기본을 확인하는 예전 테스트가 그대로 돌게.
    추천 테스트(tests/test_model_recs.py)는 model_recs.configure() 로 켠다."""
    from core import model_recs

    model_recs.configure(None)
    yield
    model_recs.configure(None)


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


@pytest.fixture(autouse=True)
def no_network_career(monkeypatch):
    """👔 커리어 팀의 주소 읽기(GitHub·웹 페이지 — core/career_search.py)가 테스트 중 실제로 나가지 않게 한다 —
    가져오기(_send)와 사이트 이름 풀기(_resolve) 둘 다. 주소 읽기 테스트는 가짜 _send·_resolve 를 직접 넣는다."""
    from core import career_search

    def blocked(*args, **kwargs):
        raise RuntimeError("테스트 중 네트워크 금지")

    monkeypatch.setattr(career_search, "_send", blocked)
    monkeypatch.setattr(career_search, "_resolve", blocked)


@pytest.fixture(autouse=True)
def no_network_market(monkeypatch):
    """📺 보드·🏠 메인의 지수·환율 띠가 테스트 중 Yahoo 에 묻지 않게 한다 (실패 → 그 칸은 '—').
    시세 테스트는 market.fetch_history 를 가짜 일봉으로 바꿔 쓴다."""
    from core import market

    def blocked(ticker, timeout=10):
        raise RuntimeError("테스트 중 네트워크 금지")

    monkeypatch.setattr(market, "fetch_history", blocked)


@pytest.fixture(autouse=True)
def no_model_discovery(monkeypatch):
    """모델 목록 자동 찾기(core/model_catalog.py)가 테스트 중 각 사 API 를 부르거나 백그라운드로 파일을 쓰지 않게 한다.
    목록 테스트는 가짜 http_get·FETCHERS 를 직접 넣고, start_refresh 는 원래 함수(model_catalog.REAL_START)를 부른다.
    start_refresh 호출은 반환 리스트에 기록된다."""
    from core import model_catalog

    def blocked(*args, **kwargs):
        raise RuntimeError("테스트 중 네트워크 금지")

    started: list[str] = []
    monkeypatch.setattr(model_catalog, "http_get", blocked)
    monkeypatch.setattr(model_catalog, "start_refresh", lambda models, data_dir, force=False: started.append(str(data_dir)))
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
    from tests.fakes import FakeOfficeModel

    def _install(script):
        set_keys("openai", "google", "upstage")
        monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, models: FakeOfficeModel(script=script))
        return script

    return _install
