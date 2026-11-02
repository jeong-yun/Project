"""모델 목록 — 각 사 목록 읽기·거르기, 저장(TTL·오프라인), 🆕·⚠️, 직접 입력, 강도 추정·사람이 고른 급,
상하 관계 규칙, '가격 미등록', 연결 시험, 화면(AppTest). 네트워크는 쓰지 않는다 (가짜 http_get)."""

import json
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from core import llm_factory, member_models, model_catalog, usage
from core.config import list_teams
from core.model_catalog import CustomModelError, CustomStore, guess_tier, merge, name_info

NOW = 1_800_000_000.0
DAY = 86400


# ─── 목록 읽기·거르기 ────────────────────────────────────────────────────


def test_openai_list_keeps_chat_models_only():
    data = {"data": [
        {"id": "gpt-6-luna", "created": 1}, {"id": "gpt-5.5"}, {"id": "gpt-5.5-2026-05-01"}, {"id": "gpt-4-0613"},
        {"id": "text-embedding-3-large"}, {"id": "whisper-1"}, {"id": "tts-1-hd"}, {"id": "dall-e-3"}, {"id": "gpt-image-1"},
        {"id": "omni-moderation-latest"}, {"id": "gpt-4o-realtime-preview"}, {"id": "gpt-4o-transcribe"}, {"id": "babbage-002"},
        {"id": "ft:gpt-5.4-mini:my-org::abc"}, {"id": "sora-2"}, {"id": "o3"},
        {"id": "gpt-5.5-pro"}, {"id": "gpt-5.3-codex"}, {"id": "gpt-3.5-turbo-instruct"},
    ]}
    ids = [f.id for f in model_catalog.parse_openai_like(data, model_catalog.OPENAI_NOT_CHAT)]
    # 날짜 붙은 고정판은 날짜 없는 이름이 있으면 뺀다 (gpt-4-0613 은 gpt-4 가 없어 남는다)
    # Responses 전용(-pro·codex)·Completions 전용(instruct)은 Chat Completions 로 못 부르므로 뺀다
    assert ids == ["gpt-6-luna", "gpt-5.5", "gpt-4-0613", "o3"]


def test_google_list_uses_generation_methods():
    data = {"models": [
        {"name": "models/gemini-3.9-flash", "displayName": "Gemini 3.9 Flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]},
        {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
        {"name": "models/gemini-3.5-flash-native-audio", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash-image", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/nano-banana-pro-preview", "supportedGenerationMethods": ["generateContent"]},
    ]}
    found = model_catalog.parse_google(data)
    assert [(f.id, f.display) for f in found] == [("gemini-3.9-flash", "Gemini 3.9 Flash")]


def test_anthropic_upstage_freellm_lists():
    anthropic = {"data": [{"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-09-01T00:00:00Z"}]}
    found = model_catalog.parse_anthropic(anthropic)
    assert found[0].id == "claude-opus-5-5" and found[0].display == "Claude Opus 5.5" and found[0].created > 0
    upstage = {"data": [{"id": "solar-pro3"}, {"id": "solar-pro3-260806"}, {"id": "embedding-query"}, {"id": "document-parse"},
                        {"id": "solar-mini"}]}
    assert [f.id for f in model_catalog.parse_openai_like(upstage)] == ["solar-pro3", "solar-mini"]
    free = {"data": [{"id": "auto"}, {"id": "free-router"}, {"id": "kilo-auto"}, {"id": "default"},
                     {"id": "gpt-oss-20b", "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}}]}
    found = model_catalog.parse_openai_like(free, model_catalog.FREELLM_ROUTERS)
    assert [f.id for f in found] == ["gpt-oss-20b"]  # 자동 선택·라우터는 쓰지 않는다
    assert found[0].price == {"input": 0.1, "output": 0.5}  # 목록 API 가 준 가격 → 100만 토큰당


# ─── 저장 (TTL · 오프라인) ───────────────────────────────────────────────


@pytest.fixture
def fake_http(monkeypatch, set_keys):
    """가짜 목록 API — 주소별 응답. 응답을 바꾸거나 RuntimeError 로 '네트워크 없음'을 흉내 낸다."""
    set_keys("openai", "google")
    replies = {
        model_catalog.OPENAI_URL: {"data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4"}, {"id": "gpt-5.4-mini"}, {"id": "gpt-5.4-nano"}]},
        model_catalog.GOOGLE_URL: {"models": [
            {"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
            for m in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")
        ]},
    }
    calls = []

    def http_get(url, headers, params=None, timeout=None):
        calls.append(url)
        reply = replies[url]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(model_catalog, "http_get", http_get)
    return replies, calls


def test_refresh_saves_cache_and_only_calls_enabled_providers(tmp_path, models, fake_http):
    replies, calls = fake_http
    data = model_catalog.refresh(models, tmp_path, now=NOW)
    assert sorted(calls) == sorted([model_catalog.OPENAI_URL, model_catalog.GOOGLE_URL])  # 키 있는 provider 만 (anthropic 은 꺼짐)
    saved = json.loads((tmp_path / model_catalog.CATALOG_FILE).read_text(encoding="utf-8"))
    assert saved == data and set(saved["providers"]) == {"openai", "google"}
    assert "gpt-5.5" in saved["providers"]["openai"]["models"]
    assert not model_catalog.is_stale(tmp_path, now=NOW + 3600)
    assert model_catalog.is_stale(tmp_path, now=NOW + model_catalog.TTL_SEC + 1)


def test_start_refresh_respects_ttl(tmp_path, models, fake_http):
    model_catalog.refresh(models, tmp_path, now=time.time())
    assert model_catalog.REAL_START(models, tmp_path) is None  # 아직 하루가 안 지남 → 안 부른다
    thread = model_catalog.REAL_START(models, tmp_path, force=True)  # 🔄 버튼
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_offline_keeps_last_list_then_yaml(tmp_path, models, fake_http):
    replies, _ = fake_http
    assert merge(models, tmp_path).models.model_tiers == models.model_tiers  # 받은 적 없음 → yaml 그대로
    replies[model_catalog.OPENAI_URL]["data"].append({"id": "gpt-6-luna"})
    model_catalog.refresh(models, tmp_path, now=NOW)
    replies[model_catalog.OPENAI_URL] = RuntimeError("ConnectError: 네트워크 없음")
    data = model_catalog.refresh(models, tmp_path, now=NOW + 2 * DAY)
    entry = data["providers"]["openai"]
    assert "ConnectError" in entry["error"] and "gpt-6-luna" in entry["models"]  # 실패 → 예전 목록 그대로
    assert "openai:gpt-6-luna" in member_models.catalog(merge(models, tmp_path).models)
    # 아무것도 못 받고 데이터 폴더도 없으면 yaml 만
    assert merge(models, None).models.model_tiers == models.model_tiers


def test_new_badge_and_missing_model_warning(tmp_path, models, fake_http):
    replies, _ = fake_http
    model_catalog.refresh(models, tmp_path, now=NOW)  # 처음 받은 목록 = 원래 있던 모델 (🆕 아님)
    assert not any(i.new for i in merge(models, tmp_path, now=NOW).info.values())
    replies[model_catalog.OPENAI_URL] = {"data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4-mini"}, {"id": "gpt-5.4-nano"},
                                                  {"id": "gpt-6-luna"}, {"id": "gpt-5.6"}]}  # gpt-5.4 가 사라짐
    model_catalog.refresh(models, tmp_path, now=NOW + DAY)
    merged = merge(models, tmp_path, now=NOW + DAY)
    assert merged.info["openai:gpt-6-luna"].new and merged.info["openai:gpt-6-luna"].source == "discovered"
    assert not merged.info["openai:gpt-5.5"].new
    gone = merged.info["openai:gpt-5.4"]
    assert gone.missing and gone.suggest == "openai:gpt-6-luna"  # 같은 계열·같은 크기에서 가장 새 버전
    assert not merged.info["google:gemini-3.8-flash"].missing
    assert not merge(models, tmp_path, now=NOW + DAY + model_catalog.NEW_DAYS * DAY).info["openai:gpt-6-luna"].new
    # ⚠️ 는 표시만 — 직급의 기본 모델은 그대로
    assert merged.models.ranks == models.ranks and merged.models.utility == models.utility


# ─── 직접 입력 (custom_models.json) ─────────────────────────────────────


def test_custom_models_add_remove_persist(tmp_path, models):
    store = CustomStore(tmp_path)
    key = store.add("openai", " gpt-6-luna ", "GPT-6 Luna", models)
    assert key == "openai:gpt-6-luna"
    store.add("freellm", "llama-4-scout", "", models)
    again = CustomStore(tmp_path)  # 다시 열어도 그대로
    assert again.keys() == ["openai:gpt-6-luna", "freellm:llama-4-scout"]
    assert again.models()[0]["display"] == "GPT-6 Luna"
    store.add("openai", "gpt-6-luna", "새 이름", models)  # 같은 모델 = 덮어쓰기
    assert [m["display"] for m in CustomStore(tmp_path).models() if m["model"] == "gpt-6-luna"] == ["새 이름"]
    store.set_tier(key, "openai:gpt-5.4")
    store.remove(key)
    assert CustomStore(tmp_path).keys() == ["freellm:llama-4-scout"] and CustomStore(tmp_path).tiers() == {}
    with pytest.raises(CustomModelError, match="provider"):
        store.add("mistral", "large", "", models)
    with pytest.raises(CustomModelError, match="이름"):
        store.add("openai", "gpt 6", "", models)
    assert not (Path(model_catalog.__file__).parent.parent / "config" / model_catalog.CUSTOM_FILE).exists()


# ─── 강도 추정 · 사람이 고른 급 ───────────────────────────────────────────


def test_name_info():
    assert name_info("gpt-5.4-mini") == model_catalog.NameInfo("gpt", 5.4, 2)
    assert name_info("claude-opus-5-5") == model_catalog.NameInfo("claude", 5.5, 4)
    assert name_info("gemini-3.5-flash-lite-preview-09-2026").size == 1
    assert name_info("gpt-oss-120b") == model_catalog.NameInfo("gpt-oss", None, 2)
    assert name_info("solar-pro2") == model_catalog.NameInfo("solar", 2.0, 4)
    assert name_info("gpt-5.4-2026-03-05").version == 5.4


def test_tier_guess_from_name(models):
    tiers = models.model_tiers
    row = {k: i for i, r in enumerate(tiers) for k in r}
    assert guess_tier("openai:gpt-6-luna", tiers) == row["openai:gpt-5.5"] - 1  # 새 버전 → gpt-5.5 보다 한 줄 위
    assert guess_tier("openai:gpt-5.4-nano", tiers) == row["openai:gpt-5.4-nano"]  # yaml 과 같은 이름이면 같은 줄
    assert guess_tier("openai:gpt-6-nano", tiers) > guess_tier("openai:gpt-6-mini", tiers) > guess_tier("openai:gpt-6-luna", tiers)
    assert guess_tier("openai:gpt-3.5-turbo", tiers) > row["openai:gpt-5.4"]  # 옛 버전은 낮게
    assert guess_tier("google:gemini-3.9-pro", tiers) < row["google:gemini-3.8-flash"]  # pro 는 높게
    assert guess_tier("google:gemini-3.9-flash-lite", tiers) > guess_tier("google:gemini-3.9-flash", tiers)
    assert guess_tier("claude_cli:haiku", tiers) > guess_tier("claude_cli:sonnet", tiers) > guess_tier("claude_cli:opus", tiers)
    free_rows = {i for i, r in enumerate(tiers) if all(k.startswith("freellm:") for k in r)}
    assert guess_tier("freellm:llama-4-scout", tiers) in free_rows  # 무료 모델은 무료 줄 안에서만
    assert guess_tier("freellm:gemini-9-pro", tiers) in free_rows
    assert guess_tier("upstage:solar-tiny", tiers) not in free_rows  # 유료는 무료 줄 위에서만


def test_merge_precedence_yaml_then_user_then_guess(tmp_path, models):
    store = CustomStore(tmp_path)
    store.add("openai", "gpt-6-luna", "", models)
    store.add("openai", "gpt-5.4-nano", "", models)  # yaml 에 이미 있는 모델
    merged = merge(models, tmp_path)
    tiers = member_models.tier_map(merged.models)
    assert tiers["openai:gpt-6-luna"] == guess_tier("openai:gpt-6-luna", models.model_tiers)
    assert merged.info["openai:gpt-6-luna"].tier_source == "guess" and merged.info["openai:gpt-6-luna"].source == "custom"
    store.set_tier("openai:gpt-6-luna", "openai:gpt-5.4-mini")  # 사람이 고른 급
    store.set_tier("openai:gpt-5.4-nano", "openai:gpt-5.5")  # yaml 모델은 yaml 이 이긴다
    merged = merge(models, tmp_path)
    tiers = member_models.tier_map(merged.models)
    assert tiers["openai:gpt-6-luna"] == tiers["openai:gpt-5.4-mini"] and merged.info["openai:gpt-6-luna"].tier_source == "user"
    assert tiers["openai:gpt-5.4-nano"] == member_models.tier_map(models)["openai:gpt-5.4-nano"]
    store.set_tier("openai:gpt-6-luna", "openai:없는-모델")  # 기준 모델이 yaml 에서 사라지면 → 추정
    assert merge(models, tmp_path).info["openai:gpt-6-luna"].tier_source == "guess"
    assert len(merged.models.model_tiers) == len(models.model_tiers)  # 새 줄을 만들지 않는다


def test_yaml_model_missing_from_tiers_does_not_crash(models, ranks):
    from core.config import validate_config

    tiers = [[k for k in row if k != "openai:gpt-5.4"] for row in models.model_tiers]
    broken = models.model_copy(update={"model_tiers": [r for r in tiers if r]})
    validate_config(ranks, broken, list_teams())  # 오류로 멈추지 않는다
    merged = merge(broken, None, list_teams())
    assert merged.yaml_missing == ["openai:gpt-5.4"]
    assert "openai:gpt-5.4" in member_models.tier_map(merged.models)


# ─── 상하 관계 규칙 + 직접 입력한 모델 ─────────────────────────────────────


def test_hierarchy_rule_with_custom_model(tmp_path, models, org, team):
    store = CustomStore(tmp_path)
    store.add("openai", "gpt-6-luna", "", models)
    store.set_tier("openai:gpt-6-luna", "openai:gpt-5.5")
    merged = merge(models, tmp_path).models
    nodes = {n.id: n for n in org.walk()}
    plans = member_models.plan(org, merged, team, {})
    assert "openai:gpt-6-luna" in member_models.allowed_keys(merged, nodes["gm_research"], plans["gm_research"])
    assert "openai:gpt-6-luna" not in member_models.allowed_keys(merged, nodes["dgm_research"], plans["dgm_research"])  # 기본 부장(gpt-5.4)보다 강함

    choices, _ = member_models.apply_choice(org, merged, team, {"dgm_research": "openai:gpt-5.4"}, "gm_research", "openai:gpt-6-luna")
    plans = member_models.plan(org, merged, team, choices)
    assert plans["gm_research"].effective == "openai:gpt-6-luna"
    chain = member_models.member_rank_models(plans["gm_research"], merged, team)
    assert chain.primary.provider == "openai" and chain.primary.model == "gpt-6-luna" and chain.primary.temperature is None
    tiers = member_models.tier_map(merged)
    assert all(tiers[member_models.model_key(s)] >= tiers["openai:gpt-6-luna"] for s in chain.candidates)  # 대안도 상한 안
    # 상사 = gpt-6-luna(직접 고름) → 그 아래는 같은 급(gpt-5.5)도 안 된다
    assert "openai:gpt-5.5" not in member_models.allowed_keys(merged, nodes["dgm_research"], plans["dgm_research"])
    # 직급 기본 모델은 바뀌지 않는다
    assert merged.ranks == models.ranks


def test_custom_free_model_never_falls_back_to_paid(tmp_path, models, org, team):
    CustomStore(tmp_path).add("freellm", "llama-4-scout", "", models)
    merged = merge(models, tmp_path).models
    plans = member_models.plan(org, merged, team, {"staff_draft": "freellm:llama-4-scout"})
    chain = member_models.member_rank_models(plans["staff_draft"], merged, team)
    assert chain.primary.model == "llama-4-scout" and {s.provider for s in chain.candidates} == {"freellm"}


# ─── 가격 미등록 ─────────────────────────────────────────────────────────


def test_unknown_model_is_unpriced_not_zero(tmp_path):
    events = []
    usage.set_sink(events.append)
    usage.record("openai", "gpt-6-luna", "sk-test-key-123456", 1000, 200)
    assert events[0].cost_usd is None and events[0].input_tokens == 1000  # 토큰은 기록, 비용은 '가격 미등록'
    assert usage.UNPRICED == "가격 미등록"
    assert usage.estimate_cost("freellm", "llama-4-scout", 1000, 200) == 0.0  # 무료 provider 는 0
    # 목록 API 가 가격을 알려 주면 그 값을 쓴다 (가격표에 없는 모델만)
    usage.set_catalog_prices(lambda: {"gpt-6-luna": {"input": 1.0, "output": 2.0}, "gpt-5.5": {"input": 0.0, "output": 0.0}})
    try:
        assert usage.estimate_cost("openai", "gpt-6-luna", 1_000_000, 1_000_000) == 3.0
        assert usage.estimate_cost("openai", "gpt-5.5", 1_000_000, 0) == 5.0  # 가격표(pricing.yaml)가 이긴다
    finally:
        usage.set_catalog_prices(None)


def test_catalog_prices_from_cache(tmp_path, models, monkeypatch, set_keys):
    set_keys("freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    monkeypatch.setattr(model_catalog, "http_get", lambda url, headers, params=None, timeout=None: {
        "data": [{"id": "llama-4-scout", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]})
    model_catalog.refresh(models, tmp_path, now=NOW)
    assert model_catalog.catalog_prices(tmp_path) == {"llama-4-scout": {"input": 1.0, "output": 2.0}}


def test_freellm_found_models_stay_out_of_selector_until_added(tmp_path, models, monkeypatch, set_keys):
    """FreeLLMAPI 는 수백 개라 찾은 모델을 선택지에 바로 넣지 않는다 — 직접 입력(찾은 모델에서 고르기)하면 들어간다."""
    set_keys("freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    monkeypatch.setattr(model_catalog, "http_get", lambda url, headers, params=None, timeout=None: {
        "data": [{"id": "gpt-oss-20b"}, {"id": "llama-4-scout"}, {"id": "qwen3.8-max"}]})
    model_catalog.refresh(models, tmp_path, now=NOW)
    merged = merge(models, tmp_path)
    assert "freellm:llama-4-scout" in merged.info and not merged.info["freellm:llama-4-scout"].selectable
    assert "freellm:llama-4-scout" not in member_models.catalog(merged.models)
    CustomStore(tmp_path).add("freellm", "llama-4-scout", "", models)
    merged = merge(models, tmp_path)
    assert merged.info["freellm:llama-4-scout"].source == "custom" and "freellm:llama-4-scout" in member_models.catalog(merged.models)
    every = merge(models, tmp_path, selector_providers=("freellm",))  # settings.yaml 로 넣을 수도 있다
    assert "freellm:qwen3.8-max" in member_models.catalog(every.models)


# ─── 🧪 연결 시험 ───────────────────────────────────────────────────────


class FakeChat:
    def __init__(self, structured_ok=True, fail=None):
        self.structured_ok, self.fail = structured_ok, fail

    def invoke(self, prompt):
        if self.fail:
            raise self.fail
        return AIMessage(content="OK", usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14})

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        ok = self.structured_ok

        class Bound:
            def invoke(self, prompt):
                raw = AIMessage(content="{}", usage_metadata={"input_tokens": 30, "output_tokens": 8, "total_tokens": 38})
                return {"parsed": schema(ok=True, word="pong") if ok else None, "raw": raw,
                        "parsing_error": None if ok else "not json"}

        return Bound()


def test_connection_test_with_fake_model(models, monkeypatch, set_keys):
    set_keys("openai")
    made = []
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: made.append(spec) or FakeChat())
    assert model_catalog.estimate_test_cost("openai", "gpt-6-luna") is None  # 미리 보기: 가격 미등록
    assert model_catalog.estimate_test_cost("openai", "gpt-5.4-nano") > 0
    result = model_catalog.connection_test("openai", "gpt-6-luna", models)
    assert result.ok and result.text_ok and result.structured_ok and result.reply == "OK"
    assert (result.input_tokens, result.output_tokens) == (42, 10) and result.cost_usd is None
    assert made[0].provider == "openai" and made[0].model == "gpt-6-luna"

    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: FakeChat(structured_ok=False))
    result = model_catalog.connection_test("openai", "gpt-6-luna", models)
    assert result.text_ok and not result.ok and "구조화 출력" in result.error

    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: FakeChat(fail=RuntimeError("404 model_not_found")))
    result = model_catalog.connection_test("openai", "gpt-없음", models)
    assert not result.ok and "model_not_found" in result.error


def test_connection_test_respects_disabled_and_free_only(models, monkeypatch, set_keys):
    set_keys("openai", "anthropic")
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: pytest.fail("부르면 안 됨"))
    assert "비활성" in model_catalog.connection_test("anthropic", "claude-opus-5-5", models).error  # disabled_providers
    assert "비활성" in model_catalog.connection_test("openai", "gpt-6-luna", llm_factory.free_only(models)).error
    assert "키 없음" in model_catalog.connection_test("google", "gemini-9", models).error


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    data = tmp_path / "data"
    data.mkdir(parents=True)
    now = time.time()
    (data / model_catalog.CATALOG_FILE).write_text(json.dumps({"refreshed_at": now, "providers": {"openai": {
        "fetched_at": now, "checked_at": now, "error": "",
        "models": {"gpt-5.5": {"first_seen": 0}, "gpt-5.4": {"first_seen": 0}, "gpt-5.4-mini": {"first_seen": 0},
                   "gpt-5.4-nano": {"first_seen": 0}, "gpt-6-nova": {"first_seen": now - 60}},
        "all_ids": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-6-nova"],
    }}}), encoding="utf-8")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    return at


def test_selector_shows_discovered_and_custom_models(app, tmp_path, no_model_discovery):
    at = app.run()
    assert not at.exception, at.exception
    assert no_model_discovery  # 앱을 켤 때 목록 새로 받기를 (백그라운드로) 요청함
    gm = at.selectbox(key="member_model_general_gm_research")
    nova = [o for o in gm.options if o.startswith("gpt-6-nova")]
    assert nova and "🆕" in nova[0] and "유료 API" in nova[0]  # 찾은 새 모델이 선택지에 🆕 로
    assert gm.value == ""  # 기본은 그대로 (자동으로 바꾸지 않음)

    at = at.toggle(key="member_catalog_general").set_value(True).run()
    assert not at.exception, at.exception
    at = at.selectbox(key="custom_provider_member").set_value("openai").run()
    at = at.text_input(key="custom_name_member").input("gpt-6-luna").run()
    at = at.text_input(key="custom_display_member").input("GPT-6 Luna").run()
    assert any("연결 시험 예상 비용: 가격 미등록" in c.value for c in at.caption)
    at = at.button(key="custom_add_member").click().run()
    assert not at.exception, at.exception
    saved = json.loads((tmp_path / "data" / model_catalog.CUSTOM_FILE).read_text(encoding="utf-8"))
    assert [(m["provider"], m["model"], m["display"]) for m in saved["models"]] == [("openai", "gpt-6-luna", "GPT-6 Luna")]
    assert at.text_input(key="custom_name_member").value == ""  # 추가하면 입력칸을 비운다
    assert any("추가했습니다" in i.value for i in at.main.info) and not any("추가했습니다" in i.value for i in at.sidebar.info)
    gm = at.selectbox(key="member_model_general_gm_research")
    luna = [o for o in gm.options if o.startswith("GPT-6 Luna")]
    assert luna and "✍️" in luna[0]
    at = at.selectbox(key="member_model_general_gm_research").set_value("openai:gpt-6-luna").run()
    assert not at.exception, at.exception
    assert json.loads((tmp_path / "data" / "member_models.json").read_text(encoding="utf-8"))["general"]["gm_research"] == "openai:gpt-6-luna"

    # 📐 급 고르기 — gpt-6-luna 를 gpt-5.4-mini 와 같은 급으로 → 부장이 고른 모델이라 차장은 그보다 낮은 모델만
    at = at.selectbox(key="tier_model_member").set_value("openai:gpt-6-luna").run()
    at = at.selectbox(key="tier_pick_member_openai:gpt-6-luna").set_value("openai:gpt-5.4-mini").run()
    assert not at.exception, at.exception
    assert CustomStore(tmp_path / "data").tiers() == {"openai:gpt-6-luna": "openai:gpt-5.4-mini"}
    dgm = at.selectbox(key="member_model_general_dgm_research")
    assert not any(o.startswith("gpt-5.4-mini") for o in dgm.options) and any(o.startswith("gpt-5.4-nano") for o in dgm.options)


def test_sidebar_panel_and_test_button(app, tmp_path, monkeypatch, set_keys):
    at = app.run()
    assert not at.exception, at.exception
    assert any(e.label == "🧩 모델 목록 · 직접 입력" for e in at.sidebar.expander)
    assert any("gpt-6-nova" in m.value for m in at.sidebar.markdown)  # 🆕 새로 나온 모델
    set_keys("openai")
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: FakeChat())
    at = at.text_input(key="custom_name_side").input("gpt-6-luna").run()
    at = at.button(key="custom_test_btn_side").click().run()
    assert not at.exception, at.exception
    assert any("연결됨" in s.value and "가격 미등록" in s.value for s in at.sidebar.success)
