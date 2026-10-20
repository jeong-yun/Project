from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core import llm_factory
from core.schemas import ModelSpec, RankModels


def test_candidates_follow_available_keys(models, set_keys):
    chain = RankModels(
        primary=ModelSpec(provider="upstage", model="a"),
        fallback=[ModelSpec(provider="google", model="b"), ModelSpec(provider="openai", model="c")],
    )
    models.disabled_providers = []

    set_keys("upstage", "google", "openai")
    spec, is_fallback = llm_factory.available_candidates(chain, models)[0]
    assert (spec.provider, is_fallback) == ("upstage", False)

    set_keys("openai")
    spec, is_fallback = llm_factory.available_candidates(chain, models)[0]
    assert (spec.provider, is_fallback) == ("openai", True)

    set_keys()
    assert llm_factory.available_candidates(chain, models) == []


def test_disabled_provider_is_skipped_even_with_key(models, set_keys):
    chain = RankModels(
        primary=ModelSpec(provider="anthropic", model="claude"),
        fallback=[ModelSpec(provider="openai", model="gpt")],
    )
    models.disabled_providers = ["anthropic"]
    set_keys("anthropic", "openai")

    spec, is_fallback = llm_factory.available_candidates(chain, models)[0]
    assert (spec.provider, is_fallback) == ("openai", True)
    assert llm_factory.provider_status(models)["anthropic"] == "disabled"
    assert llm_factory.provider_status(models)["openai"] == "ok"


def test_create_chat_model_per_provider(models, set_keys):
    set_keys("openai", "anthropic", "google", "upstage")

    upstage = llm_factory.create_chat_model(ModelSpec(provider="upstage", model="solar-mini"), models)
    assert isinstance(upstage, ChatOpenAI)
    assert upstage.openai_api_base == llm_factory.UPSTAGE_BASE_URL

    claude = llm_factory.create_chat_model(ModelSpec(provider="anthropic", model="claude-x"), models)
    assert isinstance(claude, ChatAnthropic)
    assert claude.max_tokens == llm_factory.ANTHROPIC_DEFAULT_MAX_TOKENS

    gemini = llm_factory.create_chat_model(
        ModelSpec(provider="google", model="gemini-x", max_tokens=100), models
    )
    assert isinstance(gemini, ChatGoogleGenerativeAI)
    assert gemini.max_output_tokens == 100


def test_temperature_omitted_when_not_set(models, set_keys):
    set_keys("openai")
    gpt = llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5"), models)
    assert gpt.temperature is None


# ─── 무료 모델 (FreeLLMAPI) ───────────────────────────────────────────


def test_freellm_model_uses_proxy_and_json_schema(models, set_keys, monkeypatch):
    set_keys("freellm")
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "http://127.0.0.1:3999/v1")
    spec = ModelSpec(provider="freellm", model="gpt-oss-20b")
    llm = llm_factory.create_chat_model(spec, models)
    assert isinstance(llm, ChatOpenAI) and llm.openai_api_base == "http://127.0.0.1:3999/v1"
    assert llm.max_retries == 0  # 재시도는 공통 로직이 맡는다
    assert llm_factory.structured_output_kwargs(spec) == {"method": "json_schema"}


def test_freellm_is_skipped_when_server_is_off(models, set_keys, monkeypatch):
    set_keys("freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: False)
    assert not llm_factory.is_available("freellm", models)
    assert llm_factory.provider_status(models)["freellm"] == "offline"
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    assert llm_factory.provider_status(models)["freellm"] == "ok"


def test_free_only_keeps_just_free_models(models, set_keys, monkeypatch):
    set_keys("openai", "google", "freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    free = llm_factory.free_only(models)
    for rank_models in [*free.ranks.values(), free.utility]:
        cands = llm_factory.available_candidates(rank_models, free)
        assert cands and all(spec.provider == "freellm" for spec, _ in cands)  # 모든 직급에 무료 대안이 있다
    # 원래 설정에서는 유료가 먼저, 무료는 마지막 대안
    staff = llm_factory.available_candidates(models.ranks["staff"], models)
    assert staff[0][0].provider == "google" and staff[-1][0].provider == "freellm"


def test_free_tier_errors_skip_waiting():
    too_large = Exception("Error code: 413 - Request too large for model gpt-oss-120b: tpm_limit 8000 < estimated 33476")
    exhausted = Exception("Error code: 429 - All models exhausted: 1 route checked (1 rate-limited or on cooldown). Soonest reset ~3h.")
    assert not llm_factory.is_transient(too_large)  # 문맥이 큰 다음 모델로 바로
    assert llm_factory.is_quota_exhausted(exhausted) and not llm_factory.is_transient(exhausted)


def test_freellm_response_without_role_is_repaired():
    import httpx

    body = {"choices": [{"index": 0, "message": {"content": "{\"a\": 1}"}, "finish_reason": "stop"}]}
    inner = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    client = httpx.Client(transport=llm_factory._RoleFixTransport(inner))
    data = client.post("http://x/v1/chat/completions", json={}).json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    # content-type 이 json 이 아니게 와도 보정
    inner2 = httpx.MockTransport(lambda req: httpx.Response(200, content=__import__("json").dumps(body).encode(), headers={"content-type": "text/plain"}))
    data = httpx.Client(transport=llm_factory._RoleFixTransport(inner2)).post("http://x/v1/chat/completions").json()
    assert data["choices"][0]["message"]["role"] == "assistant"
