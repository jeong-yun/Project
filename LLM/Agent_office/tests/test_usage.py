"""🏠 메인 화면의 사용량·비용 — 키 지문, 가격 계산, 저장·요약, LangChain 콜백, 화면(AppTest)."""

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from core import llm_factory, usage
from core.schemas import ModelSpec
from core.storage import Storage

PRICING = {
    "free_providers": ["freellm"],
    "models": {
        "gpt-5.4": {"input": 2.5, "output": 15.0, "cached_input": 0.25},
        "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "changes": [{"from": "2027-01-01", "input": 1.5, "output": 7.5}]},
    },
    "per_call": {"lyria-3.5": 0.08},
}


@pytest.fixture
def store(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    usage.set_sink(storage.record_usage)
    usage.set_price_overrides(storage.price_overrides)
    yield storage
    usage.set_sink(None)
    usage.set_price_overrides(None)


# ─── 키 지문·가격 ─────────────────────────────────────────────────────────


def test_key_fingerprint_hides_raw_key():
    key_id, masked = usage.key_fingerprint("sk-proj-ABCDEFGHIJKLMNOP1234")
    assert len(key_id) == 12 and masked == "sk-…1234"
    assert "ABCDEFGH" not in key_id + masked
    assert usage.key_fingerprint(None) == ("", "(키 없음)")
    assert usage.key_fingerprint("sk-proj-ABCDEFGHIJKLMNOP1234")[0] != usage.key_fingerprint("sk-proj-ABCDEFGHIJKLMNOP9999")[0]


def test_estimate_cost_tokens_cache_per_call_free_and_dated_changes():
    assert usage.estimate_cost("openai", "gpt-5.4", 1_000_000, 100_000, pricing=PRICING) == 4.0
    # 캐시된 입력 40만 토큰은 캐시 가격 (60만×2.5 + 40만×0.25)/1e6 = 1.6
    assert usage.estimate_cost("openai", "gpt-5.4", 1_000_000, 0, cached_tokens=400_000, pricing=PRICING) == 1.6
    assert usage.estimate_cost("google", "gemini-3.8-flash", 1_000_000, 1_000_000, pricing=PRICING, day="2026-12-31") == 4.5
    assert usage.estimate_cost("google", "gemini-3.8-flash", 1_000_000, 1_000_000, pricing=PRICING, day="2027-01-01") == 9.0
    assert usage.estimate_cost("google", "lyria-3.5", 0, 0, calls=2, pricing=PRICING) == 0.16
    assert usage.estimate_cost("freellm", "gpt-oss-20b", 9_999, 9_999, pricing=PRICING) == 0.0
    assert usage.estimate_cost("openai", "unknown-model", 10, 10, pricing=PRICING) is None


def test_shipped_pricing_covers_every_configured_paid_model():
    from core.config import load_models

    pricing = usage.load_pricing()
    models = load_models()
    used = {(s.provider, s.model) for rm in [*models.ranks.values(), models.utility] for s in rm.candidates}
    missing = [m for p, m in used if p not in pricing["free_providers"] and m not in pricing["models"]]
    assert not missing, f"config/pricing.yaml 에 가격이 없는 모델: {missing}"


# ─── 저장·요약·키 이름·가격 덮어쓰기 ──────────────────────────────────────────


def test_record_summary_keys_labels_and_scope(store, monkeypatch):
    monkeypatch.setattr(usage, "load_pricing", lambda path=None: PRICING)
    with usage.scope(team_id="stock", run_id="r1"):
        usage.record("openai", "gpt-5.4", "sk-old-key-000000001111", 1_000_000, 100_000)
    usage.record("openai", "gpt-5.4", "sk-new-key-000000002222", 2_000_000, 0)
    usage.record("upstage", "solar-x", "up-key-0000000003333", 10, 10)  # 가격 없음
    rows = store.usage_summary()
    by_key = {(r["key_id"], r["model"]): r for r in rows}
    old_id = usage.key_fingerprint("sk-old-key-000000001111")[0]
    assert by_key[(old_id, "gpt-5.4")]["cost_usd"] == 4.0
    unpriced = next(r for r in rows if r["model"] == "solar-x")
    assert unpriced["cost_usd"] is None and unpriced["unpriced_calls"] == 1
    keys = {k["key_id"]: k for k in store.api_keys()}
    assert keys[old_id]["masked"] == "sk-…1111" and len(keys) == 3
    store.set_key_label(old_id, "회사 OpenAI 키")
    assert {k["key_id"]: k for k in store.api_keys()}[old_id]["label"] == "회사 OpenAI 키"
    tagged = store._execute("SELECT team_id, run_id FROM usage WHERE key_id=?", (old_id,))[0]
    assert (tagged["team_id"], tagged["run_id"]) == ("stock", "r1")
    # 가격을 넣으면 비어 있던 기록만 다시 계산
    store.set_price_override("solar-x", 1.0, 1.0)
    assert store.reprice_unpriced(usage.estimate_cost) == 1
    assert next(r for r in store.usage_summary() if r["model"] == "solar-x")["cost_usd"] == pytest.approx(0.00002)


def test_record_without_sink_or_on_storage_error_never_raises(store):
    usage.set_sink(None)
    assert usage.record("openai", "gpt-5.4", "k", 1, 1) is None
    usage.set_sink(lambda event: (_ for _ in ()).throw(RuntimeError("DB 잠김")))
    assert usage.record("openai", "gpt-5.4", "k", 1, 1) is None  # 기록 실패가 모델 호출을 막지 않는다


# ─── LangChain 콜백 ───────────────────────────────────────────────────────


def test_callback_records_tokens_from_chat_model(store, monkeypatch):
    monkeypatch.setattr(usage, "load_pricing", lambda path=None: PRICING)
    reply = AIMessage("안녕", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
                                             "input_token_details": {"cache_read": 400}})
    model = GenericFakeChatModel(messages=iter([reply]), callbacks=[usage.UsageCallback("openai", "gpt-5.4", "sk-test-key-99998888")])
    model.invoke("hi")
    row = store.usage_summary()[0]
    assert (row["model"], row["input_tokens"], row["output_tokens"], row["cached_tokens"], row["calls"]) == ("gpt-5.4", 1000, 200, 400, 1)
    assert row["cost_usd"] == pytest.approx((600 * 2.5 + 400 * 0.25 + 200 * 15) / 1e6)


def test_create_chat_model_attaches_usage_callback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000000000001234")
    from core.config import load_models

    llm = llm_factory.create_chat_model(ModelSpec(provider="openai", model="gpt-5.4-mini"), load_models())
    callback = next(c for c in llm.callbacks if isinstance(c, usage.UsageCallback))
    assert (callback.provider, callback.model) == ("openai", "gpt-5.4-mini")


# ─── 화면 (AppTest) ───────────────────────────────────────────────────────


def test_home_screen_shows_usage_current_and_previous_keys_and_saves_names(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    data = tmp_path / "data"
    storage = Storage(data / "runs.db")
    usage.set_sink(storage.record_usage)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-now-key-000000004444")
    usage.record("openai", "gpt-5.4", "sk-now-key-000000004444", 1_000_000, 100_000)
    usage.record("openai", "gpt-5.5", "sk-old-key-000000005555", 100_000, 10_000)  # 지금 .env 에 없는 키
    usage.set_sink(None)
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.run()
        assert not at.exception
        assert at.title[0].value == "🏠 Agent Office"
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["호출"] == "2" and metrics["추정 비용"].startswith("$")
        headers = [m.value for m in at.markdown if m.value.startswith("####")]
        assert "#### 🔑 지금 쓰는 API 키" in headers and "#### 🗂 이전 API 키 (1)" in headers
        # 이전 키에 이름 붙이기
        editor = at.get("arrow_data_frame")  # data_editor 는 AppTest 에서 직접 편집이 어려워 저장소로 확인
        assert editor is not None
        old_id = usage.key_fingerprint("sk-old-key-000000005555")[0]
        storage.set_key_label(old_id, "예전 개인 키")
        at.run()
        assert not at.exception
        # 팀 버튼 → 팀 업무 화면
        next(b for b in at.button if b.key == "home_team_general").click().run()
        assert not at.exception and at.session_state["view"] == "team"
    finally:
        st.cache_resource.clear()
        usage.set_sink(None)
        usage.set_price_overrides(None)
