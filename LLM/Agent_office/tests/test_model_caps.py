"""모델 능력 판단 (core/model_caps.py) — 사진을 못 읽는 모델·제공처에서 사라진 모델을 선택지와 실행 후보에서 뺀다."""

import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from core import claude_cli, llm_factory, model_caps, model_catalog
from core.claude_cli import ClaudeCLIChatModel
from core.schemas import ModelSpec, RankModels
from tests.fakes import FakeOfficeModel, Script
from tests.test_claude_cli import fake_cli, result_event  # noqa: F401 — fixture 재사용

NOW = 1_800_000_000.0
IMAGE_MSG = HumanMessage(content=[{"type": "text", "text": "이 그림을 읽어 줘"},
                                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}])


def write_catalog(data_dir, provider: str, ids: list[str], fetched_at: float = NOW, error: str = "", models=None):
    path = data_dir / model_catalog.CATALOG_FILE
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"providers": {}}
    data["providers"][provider] = {
        "fetched_at": fetched_at, "checked_at": NOW, "error": error, "all_ids": ids,
        "models": models or {i: {"display": "", "created": 0, "price": None, "first_seen": 0} for i in ids},
    }
    data["refreshed_at"] = NOW
    path.write_text(json.dumps(data), encoding="utf-8")


# ─── 능력 판단 순서 ──────────────────────────────────────────────────


def test_heuristic_marks_gpt_oss_and_solar_text_only(monkeypatch):
    monkeypatch.setattr(model_caps, "yaml_rules", lambda path=None: [])  # 이름 규칙만
    for key in ("freellm:gpt-oss-120b", "freellm:gpt-oss-20b", "groq:openai/gpt-oss-120b", "upstage:solar-pro4",
                "upstage:solar-mini"):
        caps = model_caps.caps_for(key, data_dir=None)
        assert caps.text_only and caps.source["vision"] == "name", key
    assert model_caps.caps_for("openai:gpt-4", data_dir=None).text_only
    for key in ("google:gemini-3.5-flash", "openai:gpt-5.4-nano", "anthropic:claude-opus-5-5", "claude_cli:opus",
                "openai:gpt-4-turbo"):
        assert model_caps.caps_for(key, data_dir=None).vision is True, key
    assert model_caps.caps_for("freellm:gemini-3.5-flash", data_dir=None).vision is None  # 프록시 전달은 모름


def test_yaml_table_has_sources_for_configured_text_only_models():
    caps = model_caps.caps_for("freellm:gpt-oss-120b", data_dir=None)
    assert caps.vision is False and caps.source["vision"] == "yaml" and "gpt-oss" in caps.note["vision"]
    assert caps.context == 131072 and caps.structured_output is True
    assert model_caps.caps_for("upstage:solar-pro2", data_dir=None).text_only
    assert model_caps.caps_for("google:gemini-3.8-flash", data_dir=None).context == 1048576


def test_precedence_name_yaml_api_probe_user(tmp_path):
    key = "freellm:gemini-3.5-flash"
    assert model_caps.caps_for(key, tmp_path).vision is None  # 이름·yaml 은 모름
    # 목록 API 가 사진 입력을 알려 줌 → api
    write_catalog(tmp_path, "freellm", ["gemini-3.5-flash"], models={"gemini-3.5-flash": {
        "caps": {"input_modalities": ["text", "image"], "context": 250000}}})
    caps = model_caps.caps_for(key, tmp_path)
    assert (caps.vision, caps.source["vision"], caps.context) == (True, "api", 250000)
    # 🧪 사진 시험이 실패 → probe 가 api 를 덮는다
    store = model_caps.CapsStore(tmp_path)
    store.record_probe(key, {"vision": False}, "글자 K → 답 '?'")
    assert model_caps.caps_for(key, tmp_path).source["vision"] == "probe"
    assert model_caps.caps_for(key, tmp_path).vision is False
    # 사람이 지정 → 가장 우선
    store.set_user(key, "vision", True)
    caps = model_caps.caps_for(key, tmp_path)
    assert (caps.vision, caps.source["vision"]) == (True, "user")
    store.set_user(key, "vision", None)  # 자동으로 되돌림
    assert model_caps.caps_for(key, tmp_path).vision is False
    # yaml 이 이름 규칙을 덮는다 (gpt-oss: 둘 다 False 지만 출처는 yaml)
    assert model_caps.caps_for("freellm:gpt-oss-20b", tmp_path).source["vision"] == "yaml"


def test_catalog_parsers_keep_capability_metadata():
    found = model_catalog.parse_openai_like({"data": [
        {"id": "m1", "context_length": 32768, "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "m2", "architecture": {"modality": "text->text"}},
    ]})
    assert found[0].caps == {"input_modalities": ["text", "image"], "context": 32768}
    assert found[1].caps == {"input_modalities": ["text"]}
    google = model_catalog.parse_google({"models": [{"name": "models/gemini-x", "inputTokenLimit": 1048576,
                                                     "supportedGenerationMethods": ["generateContent"]}]})
    assert google[0].caps == {"context": 1048576}


# ─── 🧪 사진 시험 ───────────────────────────────────────────────────


class EyeModel:
    """가짜 모델 — answer 가 함수면 메시지를 받아 답을 만든다."""

    def __init__(self, answer):
        self.answer = answer
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return AIMessage(self.answer)


def test_vision_probe_pass_records_vision(models, tmp_path):
    fake = EyeModel("K")
    result = model_caps.vision_probe("freellm", "gemini-3.5-flash", models, tmp_path, letter="K", llm=fake)
    assert result.ok is True and result.recorded
    block = fake.seen[0][0].content[1]
    assert block["type"] == "image_url" and block["image_url"]["url"].startswith("data:image/png;base64,")
    caps = model_caps.caps_for("freellm:gemini-3.5-flash", tmp_path)
    assert caps.vision is True and caps.source["vision"] == "probe"
    assert model_caps.can("freellm:gemini-3.5-flash", model_caps.IMAGE_READING, tmp_path)


def test_vision_probe_fail_marks_text_only(models, tmp_path):
    wrong = model_caps.vision_probe("custom", "m", models, tmp_path, letter="K", llm=EyeModel("I cannot see images, sorry. X"))
    assert wrong.ok is False and model_caps.caps_for("custom:m", tmp_path).vision is False
    refused = model_caps.vision_probe(
        "freellm", "gpt-oss-20b", models, tmp_path, letter="K",
        llm=EyeModel(RuntimeError("Error code: 400 - 'messages.0.content' image input is not supported")))
    assert refused.ok is False and refused.recorded
    # 연결 오류·한도는 '모름' — 기록하지 않는다
    flaky = model_caps.vision_probe("freellm", "gemini-3.6-flash", models, tmp_path, letter="K",
                                    llm=EyeModel(RuntimeError("503 overloaded")))
    assert flaky.ok is None and not flaky.recorded
    assert model_caps.caps_for("freellm:gemini-3.6-flash", tmp_path).vision is None


def test_probe_image_is_small_png():
    data = model_caps.probe_image("K")
    assert data.startswith(b"\x89PNG") and len(data) < 5000


def test_connection_test_records_structured_output(tmp_path):
    model_caps.record_connection_test("openai:gpt-6-luna", False, tmp_path)
    reasons = model_caps.reasons("openai:gpt-6-luna", model_caps.TEAM_WORK, tmp_path)
    assert reasons and "형식" in reasons[0]


# ─── 선택지 거르기 ───────────────────────────────────────────────────


def test_selectable_models_hides_text_only_for_image_jobs_with_reasons(tmp_path):
    keys = ["google:gemini-3.5-flash", "freellm:gpt-oss-120b", "upstage:solar-mini", "freellm:gemini-3.5-flash",
            "claude_cli:claude-opus-5-5"]
    shown, hidden = model_caps.selectable_models(model_caps.IMAGE_READING, keys, tmp_path)
    assert shown == ["google:gemini-3.5-flash", "claude_cli:claude-opus-5-5"]
    why = {h.key: h.reasons[0] for h in hidden}
    assert "사진을 못 읽음" in why["freellm:gpt-oss-120b"] and "gpt-oss" in why["freellm:gpt-oss-120b"]
    assert "사진을 못 읽음" in why["upstage:solar-mini"]
    assert "확인 안 됨" in why["freellm:gemini-3.5-flash"]  # 모름 → 사진 일에서는 숨김
    # 팀 업무(텍스트)에는 모두 보인다
    assert model_caps.selectable_models(model_caps.TEAM_WORK, keys, tmp_path) == (keys, [])


def test_member_selector_options_hide_removed_models(models, org, team, tmp_path):
    from core import member_models

    write_catalog(tmp_path, "upstage", ["solar-pro4", "solar-mini4"])
    plans = member_models.plan(org, models, team, {})
    node = next(n for n in org.walk() if n.rank == "staff")
    allowed = member_models.allowed_keys(models, node, plans[node.id])
    shown, hidden = model_caps.selectable_models(model_caps.TEAM_WORK, allowed, tmp_path, now=NOW + 60)
    assert "upstage:solar-mini" in allowed and "upstage:solar-mini" not in shown
    gone = next(h for h in hidden if h.key == "upstage:solar-mini")
    assert "사라짐" in gone.reasons[0] and "solar-mini4" in gone.reasons[0]


# ─── 실행 후보 ───────────────────────────────────────────────────────


def test_runtime_chain_skips_text_only_models_for_images(models, set_keys, monkeypatch):
    set_keys("freellm", "google", "upstage")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    chain = RankModels(primary=ModelSpec(provider="freellm", model="gpt-oss-120b"), fallback=[
        ModelSpec(provider="upstage", model="solar-mini"), ModelSpec(provider="google", model="gemini-3.5-flash")])
    image = model_caps.needs_for_messages([IMAGE_MSG])
    assert image.vision and not model_caps.needs_for_messages([HumanMessage("글만")]).vision
    got = llm_factory.available_candidates(chain, models, needs=image)
    assert [(s.model, fb) for s, fb in got] == [("gemini-3.5-flash", True)]
    assert len(llm_factory.available_candidates(chain, models)) == 3  # 글만 보내면 그대로

    made = []

    class Out(BaseModel):
        text: str

    def fake_create(spec, models):
        made.append(spec.model)
        return FakeOfficeModel(script=Script(Out=Out(text="ok")))

    monkeypatch.setattr(llm_factory, "create_chat_model", fake_create)
    call = llm_factory.invoke_structured(chain, models, Out, [IMAGE_MSG], "사진 읽기")
    assert made == ["gemini-3.5-flash"] and call.is_fallback
    assert any("gpt-oss-120b" in e and "사진을 못 읽음" in e for e in call.errors)


def test_runtime_chain_with_only_text_models_explains(models, set_keys, monkeypatch):
    set_keys("freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    chain = RankModels(primary=ModelSpec(provider="freellm", model="gpt-oss-20b"))
    with pytest.raises(llm_factory.NoAvailableModelError, match="이 일을 할 수 있는 모델"):
        llm_factory.require_candidates(chain, models, "도면", model_caps.IMAGE_READING)


def test_floorplan_vision_models_use_capabilities(models):
    from core import floorplan_import

    picked = floorplan_import.vision_models(models)
    keys = [f"{s.provider}:{s.model}" for s in picked.candidates]
    assert keys and all(model_caps.can(k, model_caps.IMAGE_READING, None) for k in keys)
    assert not any("gpt-oss" in k or "solar" in k for k in keys)


# ─── 사라진 모델 ─────────────────────────────────────────────────────


def test_removed_hidden_only_when_list_is_fresh_and_ok(tmp_path):
    key = "upstage:solar-pro2"
    write_catalog(tmp_path, "upstage", ["solar-pro4", "solar-mini4"])
    gone = model_caps.removal(key, tmp_path, now=NOW + 60)
    assert gone and gone.suggest == "upstage:solar-pro4"
    assert model_caps.removal("upstage:solar-pro4", tmp_path, now=NOW + 60) is None
    # 오래된 목록 (TTL 지남) → 숨기지 않음
    assert model_caps.removal(key, tmp_path, now=NOW + model_catalog.TTL_SEC + 1) is None
    # 목록 API 가 없는 provider·직접 입력한 모델은 숨기지 않음
    write_catalog(tmp_path, "claude_cli", ["opus"])
    assert model_caps.removal("claude_cli:claude-opus-5-5", tmp_path, now=NOW + 60) is None
    model_catalog.CustomStore(tmp_path).add("upstage", "solar-secret", "", _models())
    assert model_caps.removal("upstage:solar-secret", tmp_path, now=NOW + 60) is None


def test_offline_hides_nothing(tmp_path):
    keys = ["upstage:solar-pro2", "upstage:solar-mini"]
    assert model_caps.selectable_models(model_caps.TEAM_WORK, keys, tmp_path, now=NOW) == (keys, [])  # 목록 없음
    write_catalog(tmp_path, "upstage", ["solar-pro4"], error="ConnectError: offline")  # 이번 확인 실패
    assert model_caps.selectable_models(model_caps.TEAM_WORK, keys, tmp_path, now=NOW + 60) == (keys, [])
    assert model_caps.selectable_models(model_caps.TEAM_WORK, keys, None) == (keys, [])  # 데이터 폴더 없음(테스트 기본)


def test_runtime_skips_removed_rank_default_with_one_time_notice(models, set_keys, tmp_path):
    set_keys("upstage", "google")
    write_catalog(tmp_path, "upstage", ["solar-pro4", "solar-mini4"], fetched_at=time.time())
    model_caps.configure(tmp_path)
    chain = RankModels(primary=ModelSpec(provider="upstage", model="solar-mini", temperature=0.7),
                       fallback=[ModelSpec(provider="google", model="gemini-3.5-flash-lite")])
    got = llm_factory.available_candidates(chain, models)
    assert [(s.model, fb) for s, fb in got] == [("gemini-3.5-flash-lite", True)]
    notes = llm_factory.skip_notes(chain, got)
    assert any("solar-mini" in n and "사라짐" in n for n in notes)
    assert "upstage:solar-mini" in model_caps.runtime_notices()
    removed = model_caps.removed_defaults(models)
    assert {r.key for r in removed} == {"upstage:solar-mini", "upstage:solar-pro2"}


def test_replacement_override_is_used_at_runtime(models, set_keys, tmp_path):
    set_keys("upstage", "google")
    write_catalog(tmp_path, "upstage", ["solar-pro4", "solar-mini4"], fetched_at=time.time())
    model_caps.configure(tmp_path)
    model_caps.CapsStore(tmp_path).set_replacement("upstage:solar-mini", "upstage:solar-mini4")
    chain = RankModels(primary=ModelSpec(provider="upstage", model="solar-mini", temperature=0.7),
                       fallback=[ModelSpec(provider="google", model="gemini-3.5-flash-lite")])
    got = llm_factory.available_candidates(chain, models)
    assert [(s.model, s.temperature, fb) for s, fb in got] == [("solar-mini4", 0.7, False), ("gemini-3.5-flash-lite", None, True)]
    assert next(r for r in model_caps.removed_defaults(models) if r.key == "upstage:solar-mini").replacement == "upstage:solar-mini4"
    shown, hidden = model_caps.selectable_models(model_caps.TEAM_WORK, ["upstage:solar-mini"])
    assert not shown and "대체: solar-mini4" in hidden[0].reasons[0]
    # yaml 은 그대로
    assert any(s.model == "solar-mini" for s in models.ranks["staff"].candidates)


def test_saved_choice_removed_or_incapable_warns_and_falls_back(tmp_path):
    write_catalog(tmp_path, "upstage", ["solar-pro4", "solar-mini4"])
    model_caps.CapsStore(tmp_path).record_probe("openai:gpt-6-luna", {"structured_output": False})
    choices = {"lead": "google:gemini-3.8-flash", "staff1": "upstage:solar-mini", "gm": "upstage:solar-pro2",
               "dgm": "openai:gpt-6-luna"}
    model_caps.CapsStore(tmp_path).set_replacement("upstage:solar-pro2", "upstage:solar-pro4")
    use, issues = model_caps.check_choices(choices, model_caps.TEAM_WORK, tmp_path, now=NOW + 60)
    assert use == {"lead": "google:gemini-3.8-flash", "gm": "upstage:solar-pro4"}  # 없는 것은 기본(다음 후보)으로
    by = {i.node_id: i for i in issues}
    assert set(by) == {"staff1", "gm", "dgm"}
    assert by["staff1"].use == "" and "사라짐" in by["staff1"].reasons[0]
    assert by["gm"].use == "upstage:solar-pro4"
    assert "형식" in by["dgm"].reasons[0]
    assert choices["staff1"] == "upstage:solar-mini"  # 저장값은 그대로 (조용히 지우지 않음)


def test_saved_choice_for_image_job_needs_vision(tmp_path):
    use, issues = model_caps.check_choices({"a": "freellm:gpt-oss-120b", "b": "google:gemini-3.5-flash"},
                                           model_caps.IMAGE_READING, tmp_path)
    assert use == {"b": "google:gemini-3.5-flash"} and issues[0].node_id == "a" and "사진" in issues[0].reasons[0]


# ─── Claude 구독(claude_cli) 사진 전달 ─────────────────────────────────


def test_claude_cli_sends_images_as_stream_json(fake_cli):  # noqa: F811
    calls = fake_cli([result_event("K")])
    reply = ClaudeCLIChatModel(model="claude-haiku-5-5").invoke([IMAGE_MSG])
    assert reply.content == "K"
    cmd = calls[0].cmd
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json" and "--verbose" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # 도구는 여전히 끔
    sent = json.loads(calls[0].prompt)
    content = sent["message"]["content"]
    assert sent["type"] == "user" and content[0] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    assert content[-1] == {"type": "text", "text": "이 그림을 읽어 줘"}


def test_claude_cli_text_only_messages_unchanged(fake_cli):  # noqa: F811
    calls = fake_cli()
    ClaudeCLIChatModel(model="claude-haiku-5-5").invoke([HumanMessage("안녕")])
    assert "--input-format" not in calls[0].cmd and calls[0].prompt == "안녕"
    assert claude_cli.images_of([HumanMessage("x")]) == []
    url = claude_cli._image_block({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})
    assert url == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}


def _models():
    from core.config import load_models

    return load_models()
