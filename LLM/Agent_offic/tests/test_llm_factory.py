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
