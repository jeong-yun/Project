"""팀원별 모델 선택 — 강도 순서 검증, 상하 관계 규칙(자동 낮춤), 대안 모델 상한, 저장, 기본 = 지금과 같음, 화면(AppTest)."""

import json
from pathlib import Path

import pytest

from core import llm_factory, member_models, usage
from core.config import ConfigError, list_teams, rank_models, team_org, validate_config, validate_model_tiers
from core.member_models import MemberModelStore, apply_choice, plan, team_rank_models
from core.storage import Storage
from core.workflow import build_office_graph, stream_office
from tests.fakes import FakeOfficeModel, Script, make_delegation, make_review

OPUS = "claude_cli:claude-opus-5-5"
OPUS_API = "anthropic:claude-opus-5-5"


def keys(chain):
    return [member_models.model_key(s) for s in chain.candidates]


# ─── 강도 순서 (models.yaml model_tiers) ─────────────────────────────────


def test_shipped_tiers_cover_every_model(ranks, models):
    validate_config(ranks, models, list_teams())
    tiers = member_models.tier_map(models)
    assert tiers[OPUS] == tiers[OPUS_API] == 0  # 구독·API 의 Opus 는 같은 강도, 가장 강함
    assert tiers[OPUS] < tiers["openai:gpt-5.5"] < tiers["openai:gpt-5.4"] < tiers["google:gemini-3.8-flash"]
    assert tiers["google:gemini-3.8-flash"] < tiers["openai:gpt-5.4-mini"] < tiers["google:gemini-3.5-flash"]
    paid = [t for k, t in tiers.items() if not k.startswith("freellm:")]
    free = [t for k, t in tiers.items() if k.startswith("freellm:")]
    assert max(paid) < min(free)  # 무료 모델은 모두 맨 아래


def test_tier_validation_catches_missing_duplicate_and_bad_format(models):
    missing = models.model_copy(update={"model_tiers": [t for t in models.model_tiers if "openai:gpt-5.5" not in t]})
    with pytest.raises(ConfigError, match="gpt-5.5"):
        validate_model_tiers(missing)
    dup = models.model_copy(update={"model_tiers": [*models.model_tiers, ["openai:gpt-5.5"]]})
    with pytest.raises(ConfigError, match="두 번"):
        validate_model_tiers(dup)
    bad = models.model_copy(update={"model_tiers": [*models.model_tiers, ["nowhere:x"]]})
    with pytest.raises(ConfigError, match="형식"):
        validate_model_tiers(bad)
    with pytest.raises(ConfigError, match="model_tiers"):
        validate_model_tiers(models.model_copy(update={"model_tiers": []}))


def test_team_override_models_must_be_in_tiers(models, team):
    from core.schemas import ModelSpec, RankModels

    odd = team.model_copy(update={"model_overrides": {"staff": RankModels(primary=ModelSpec(provider="openai", model="gpt-9"))}})
    with pytest.raises(ConfigError, match="gpt-9"):
        validate_model_tiers(models, [odd])


# ─── 기본 = 지금과 같음 ────────────────────────────────────────────────


def test_default_is_unchanged_for_every_team(ranks, models):
    for team in list_teams():
        org = team_org(team, ranks)
        chains = team_rank_models(org, models, team, {})
        for node in org.walk():
            assert chains[node.id] == rank_models(node.rank, models, team), (team.id, node.id)
        assert not any(p.limited or p.chosen_on_chain for p in plan(org, models, team, {}).values())


def test_choice_on_other_branch_leaves_branch_unchanged(org, team, models):
    chains = team_rank_models(org, models, team, {"gm_research": OPUS})
    for node_id in ("gm_action", "am_action", "staff_case", "lead"):
        node = next(n for n in org.walk() if n.id == node_id)
        assert chains[node_id] == rank_models(node.rank, models, team)


# ─── 상하 관계 규칙 ──────────────────────────────────────────────────


def test_gm_can_pick_opus_and_subordinates_are_limited(org, team, models):
    """부장 = Opus → 그 부장 아래(차장·대리·사원)는 Opus 보다 낮은 모델만."""
    nodes = {n.id: n for n in org.walk()}
    before = plan(org, models, team, {})
    assert OPUS in member_models.allowed_keys(models, nodes["gm_research"], before["gm_research"])  # 팀장 기본(Opus)과 같은 강도까지

    plans = plan(org, models, team, {"gm_research": OPUS})
    assert plans["gm_research"].effective == OPUS and not plans["gm_research"].limited
    tiers = member_models.tier_map(models)
    for sub in ("dgm_research", "am_fact", "staff_draft"):
        allowed = member_models.allowed_keys(models, nodes[sub], plans[sub])
        assert OPUS not in allowed and OPUS_API not in allowed
        assert all(tiers[k] > tiers[OPUS] for k in allowed)
    assert OPUS in member_models.allowed_keys(models, nodes["gm_action"], plans["gm_action"])  # 다른 부장은 그대로


def test_subordinate_cannot_exceed_default_superior(org, team, models):
    nodes = {n.id: n for n in org.walk()}
    plans = plan(org, models, team, {})
    allowed = member_models.allowed_keys(models, nodes["dgm_research"], plans["dgm_research"])
    assert "openai:gpt-5.4" in allowed  # 기본 부장(gpt-5.4)과 같은 강도까지
    assert "openai:gpt-5.5" not in allowed and OPUS not in allowed


def test_manager_cannot_take_weakest_tier(org, team, models):
    """하위가 있는 팀원은 맨 아래 줄을 고를 수 없다 (하위가 쓸 더 낮은 모델이 남도록). 실무자는 고를 수 있다."""
    nodes = {n.id: n for n in org.walk()}
    plans = plan(org, models, team, {})
    weakest = models.model_tiers[-1][0]
    assert weakest not in member_models.allowed_keys(models, nodes["dgm_research"], plans["dgm_research"])
    assert weakest in member_models.allowed_keys(models, nodes["staff_draft"], plans["staff_draft"])


def test_raising_superior_lowers_chosen_subordinate(org, team, models):
    choices, changes = apply_choice(org, models, team, {}, "dgm_research", "openai:gpt-5.4")
    assert choices == {"dgm_research": "openai:gpt-5.4"} and changes == []

    choices, changes = apply_choice(org, models, team, choices, "gm_research", "openai:gpt-5.4")
    # 부장이 gpt-5.4 를 직접 골랐으니 차장은 더 낮은 모델 중 가장 강한 것으로 (저장값도 바뀜)
    assert choices == {"gm_research": "openai:gpt-5.4", "dgm_research": "google:gemini-3.8-flash"}
    assert [(c.node_id, c.before, c.after) for c in changes] == [("dgm_research", "openai:gpt-5.4", "google:gemini-3.8-flash")]


def test_default_subordinate_is_capped_without_saving_and_restored(org, team, models):
    choices, changes = apply_choice(org, models, team, {}, "gm_research", "openai:gpt-5.4-mini")
    assert choices == {"gm_research": "openai:gpt-5.4-mini"}  # 기본 차장은 저장하지 않는다
    plans = plan(org, models, team, choices)
    dgm = plans["dgm_research"]
    assert dgm.choice == "" and dgm.limited
    assert dgm.effective == "upstage:solar-pro2"  # gpt-5.4-mini 아래 가장 강한 줄 중 차장 직급 목록에 있는 모델
    assert [c.node_id for c in changes] == ["dgm_research"]

    choices, changes = apply_choice(org, models, team, choices, "gm_research", "")
    assert choices == {}
    assert [(c.node_id, c.after) for c in changes] == [("dgm_research", "openai:gpt-5.4-mini")]  # 부장을 되돌리면 기본도 돌아온다


def test_lowering_cascades_down_the_chain(org, team, models):
    choices = {"dgm_research": "google:gemini-3.8-flash", "am_fact": "openai:gpt-5.4-mini"}
    choices, changes = apply_choice(org, models, team, choices, "gm_research", "google:gemini-3.8-flash")
    tiers = member_models.tier_map(models)
    assert tiers[choices["dgm_research"]] == tiers["openai:gpt-5.4-mini"]
    assert tiers[choices["am_fact"]] > tiers[choices["dgm_research"]]
    assert {c.node_id for c in changes} >= {"dgm_research", "am_fact"}


def test_stale_saved_choice_is_lowered_at_runtime(org, team, models):
    """파일을 손으로 고쳐 규칙을 어기는 값이 있어도 실행 때는 규칙대로 낮춘다. 없는 모델은 기본으로."""
    plans = plan(org, models, team, {"gm_research": OPUS, "dgm_research": OPUS, "staff_case": "openai:gpt-99"})
    assert plans["dgm_research"].limited and plans["dgm_research"].effective == "openai:gpt-5.5"
    assert plans["staff_case"].choice == "" and not plans["staff_case"].limited


# ─── 실행 때 모델 후보 (대안 상한) ────────────────────────────────────


def test_chosen_model_is_primary_and_keeps_rank_fallbacks(org, team, models):
    chain = team_rank_models(org, models, team, {"gm_research": OPUS})["gm_research"]
    gm = rank_models("gm", models, team)
    assert member_models.model_key(chain.primary) == OPUS
    assert keys(chain)[1:] == keys(gm)  # 직급의 primary·대안이 순서대로 대안으로


def test_fallbacks_never_escalate_past_superior(org, team, models):
    chains = team_rank_models(org, models, team, {"gm_research": "google:gemini-3.8-flash"})
    dgm = keys(chains["dgm_research"])
    assert dgm[0] == "openai:gpt-5.4-mini"  # 기본 primary 는 허용 (gemini-3.8-flash 보다 낮음)
    assert "google:gemini-3.8-flash" not in dgm  # 차장 직급의 대안이지만 부장이 고른 모델과 같은 강도 → 뺀다
    assert dgm[1:] == [k for k in keys(rank_models("deputy_gm", models, team))[1:] if k != "google:gemini-3.8-flash"]
    tiers = member_models.tier_map(models)
    for sub in ("dgm_research", "am_fact", "staff_draft"):
        assert all(tiers[k] > tiers["google:gemini-3.8-flash"] for k in keys(chains[sub])), sub


def test_capped_chain_still_respects_disabled_and_free_only(org, team, models, set_keys, monkeypatch):
    set_keys("openai", "google", "freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    chains = team_rank_models(org, models, team, {"gm_research": OPUS_API})  # anthropic 은 disabled_providers
    cands = llm_factory.available_candidates(chains["gm_research"], models)
    assert cands[0][0].model == "gpt-5.4" and cands[0][1]  # Opus API 는 비활성 → 직급 primary 가 대안으로
    free = llm_factory.free_only(models)
    cands = llm_factory.available_candidates(team_rank_models(org, free, team, {"gm_research": OPUS})["gm_research"], free)
    assert {c[0].provider for c in cands} == {"freellm"}


# ─── 무료 모델을 고른 팀원 — 유료로 넘어가지 않음 ─────────────────────


FREE_PICK = "freellm:gpt-oss-20b"


def test_free_choice_keeps_only_free_fallbacks(org, team, models):
    chains = team_rank_models(org, models, team, {"staff_draft": FREE_PICK})
    assert keys(chains["staff_draft"])[0] == FREE_PICK
    assert {member_models.split_key(k)[0] for k in keys(chains["staff_draft"])} == {"freellm"}
    assert keys(chains["staff_case"]) == keys(rank_models("staff", models, team))  # 기본 팀원은 그대로 (유료 대안 유지)


def test_auto_lowered_to_free_also_stays_free(org, team, models):
    """상사 규칙으로 무료 모델까지 낮아진 기본 팀원도 대안은 무료만."""
    plans = plan(org, models, team, {"gm_research": "upstage:solar-mini"})
    assert plans["dgm_research"].limited and plans["dgm_research"].effective.startswith("freellm:")
    chain = team_rank_models(org, models, team, {"gm_research": "upstage:solar-mini"})["dgm_research"]
    assert {s.provider for s in chain.candidates} == {"freellm"}


def free_team_script():
    return Script(
        text="초안",
        Delegation=lambda m: make_delegation("gm_research", "gm_action") if "gm_research" in m[-1].content else make_delegation("am_fact", "staff_draft"),
        ReviewResult=make_review(answer="보완안"),
    )


@pytest.fixture
def fake_models(set_keys, monkeypatch):
    """모든 provider 키가 있는 환경 + 호출한 모델 기록."""
    set_keys("openai", "google", "upstage", "freellm")
    called: list[str] = []

    def _install(script):
        def fake_model(spec, models):
            called.append(f"{spec.provider}:{spec.model}")
            return FakeOfficeModel(script=script)

        monkeypatch.setattr(llm_factory, "create_chat_model", fake_model)
        return called

    return _install


def test_free_choice_with_server_off_fails_clearly_without_paid_call(org, ranks, team, models, fake_models, monkeypatch):
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: False)
    called = fake_models(free_team_script())
    chain = team_rank_models(org, models, team, {"staff_draft": FREE_PICK})["staff_draft"]
    with pytest.raises(llm_factory.NoAvailableModelError, match="무료 서버 꺼짐 — Docker Desktop 실행 후 FreeLLMAPI 켜기"):
        llm_factory.require_candidates(chain, models, "사원·초안")

    # 실무자: '작성 실패'로 남고 나머지는 계속, 이유가 본문·기록에 보인다
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}}, member_models={"staff_draft": FREE_PICK})
    steps = [p for k, p in stream_office(graph, org, "질문", "lead") if k == "step"]
    failed = next(s for s in steps if s.node_id == "staff_draft")
    assert "작성하지 못했습니다" in failed.content and "무료 서버 꺼짐" in failed.content
    assert any("무료 서버 꺼짐" in e for e in failed.errors)
    assert steps[-1].node_id == "lead" and steps[-1].content == "보완안"
    assert not any(c.startswith("freellm:") for c in called)  # 꺼진 무료 서버는 부르지 않고
    staff_models = {s.model.provider for s in steps if s.node_id == "staff_draft"}
    assert staff_models == {"freellm"}  # 실패 기록도 고른 무료 모델 — 유료로 넘어가지 않음


def test_free_manager_with_server_off_stops_run_with_reason(org, ranks, team, models, fake_models, monkeypatch):
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: False)
    fake_models(free_team_script())
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}},
                               member_models={"dgm_research": "freellm:gpt-oss-120b"})
    with pytest.raises(llm_factory.NoAvailableModelError, match="차장·분석: 고른 무료 모델을 쓸 수 없습니다 — 무료 서버 꺼짐"):
        list(stream_office(graph, org, "질문", "lead"))


def test_free_choice_with_server_on_works(org, ranks, team, models, fake_models, monkeypatch):
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)
    called = fake_models(free_team_script())
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}}, member_models={"staff_draft": FREE_PICK})
    steps = [p for k, p in stream_office(graph, org, "질문", "lead") if k == "step"]
    assert next(s for s in steps if s.node_id == "staff_draft").model.label == "freellm · gpt-oss-20b"
    assert FREE_PICK in called


# ─── 저장 ───────────────────────────────────────────────────────────


def test_store_roundtrip_keeps_other_teams(tmp_path):
    store = MemberModelStore(tmp_path / "data")
    assert store.load("general") == {}
    store.save("general", {"gm_research": OPUS, "lead": ""})
    store.save("stock", {"lead": "openai:gpt-5.5"})
    again = MemberModelStore(tmp_path / "data")  # 앱을 다시 켜도
    assert again.load("general") == {"gm_research": OPUS}
    assert again.load("stock") == {"lead": "openai:gpt-5.5"}
    again.save("general", {})
    assert json.loads((tmp_path / "data" / "member_models.json").read_text(encoding="utf-8")) == {"stock": {"lead": "openai:gpt-5.5"}}


def test_store_ignores_broken_file(tmp_path):
    (tmp_path / "member_models.json").write_text("{깨진", encoding="utf-8")
    store = MemberModelStore(tmp_path)
    assert store.load("general") == {}
    store.save("general", {"lead": OPUS})
    assert store.load("general") == {"lead": OPUS}


# ─── 워크플로우 반영·팀원별 사용량 ────────────────────────────────────


def test_workflow_uses_member_choice_and_tags_usage(org, ranks, team, models, set_keys, monkeypatch, tmp_path):
    set_keys("openai", "google", "upstage")
    script = Script(
        text="초안",
        Delegation=lambda m: make_delegation("gm_research", "gm_action") if "gm_research" in m[-1].content else make_delegation("am_fact", "staff_draft"),
        ReviewResult=make_review(answer="보완안"),
    )

    def fake_model(spec, models):
        usage.record(spec.provider, spec.model, None, 10, 5)  # 실제 모델처럼 호출마다 기록 (스레드 안 → 에이전트 id 가 붙음)
        return FakeOfficeModel(script=script)

    monkeypatch.setattr(llm_factory, "create_chat_model", fake_model)
    storage = Storage(tmp_path / "runs.db")
    usage.set_sink(storage.record_usage)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}},
                               member_models={"staff_draft": "upstage:solar-mini", "gm_research": "google:gemini-3.8-flash"})
    with usage.scope(team_id="general", run_id="r1"):
        steps = [p for k, p in stream_office(graph, org, "질문", "lead") if k == "step"]
    by_node = {s.node_id: s for s in steps}
    assert by_node["staff_draft"].model.label == "upstage · solar-mini"
    assert by_node["gm_research"].model.label == "google · gemini-3.8-flash"
    assert by_node["staff_case"].model.model == "gemini-3.5-flash-lite"  # 고르지 않은 팀원은 직급 설정 그대로

    used = storage.usage_by_agent("general")
    assert used["staff_draft"][0]["model"] == "solar-mini" and used["staff_draft"][0]["calls"] >= 1
    assert used["gm_research"][0]["model"] == "gemini-3.8-flash"
    assert "" not in used  # 워크플로우 밖 호출(브리프 등)은 팀원별 표에 섞이지 않는다


def test_old_usage_table_gets_agent_column(tmp_path):
    import sqlite3

    db = tmp_path / "runs.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, provider TEXT NOT NULL, "
                     "model TEXT NOT NULL, key_id TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0, "
                     "output_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0, "
                     "calls INTEGER NOT NULL DEFAULT 1, cost_usd REAL, team_id TEXT NOT NULL DEFAULT '', "
                     "run_id TEXT NOT NULL DEFAULT '', purpose TEXT NOT NULL DEFAULT '')")
        conn.execute("INSERT INTO usage (ts, provider, model, key_id, team_id) VALUES (1, 'openai', 'gpt-5.4', '', 'general')")
    storage = Storage(db)
    assert storage.usage_by_agent("general") == {}  # 예전 기록은 agent_id = '' → 팀원별 표에는 안 나온다
    assert storage.usage_summary()[0]["calls"] == 1


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    return at


def test_member_model_selector(app, tmp_path):
    at = app.run()
    assert not at.exception, at.exception
    assert any(e.label == "🎛 팀원 모델 선택" for e in at.expander)
    gm = at.selectbox(key="member_model_general_gm_research")
    assert gm.value == "" and gm.format_func("").startswith("기본 · gpt-5.4 · OpenAI · 유료 API")
    assert any(o.startswith("claude-opus-5-5 · Claude CLI · 구독") for o in gm.options)  # 화면의 선택지 = 표시 이름

    at = at.selectbox(key="member_model_general_dgm_research").set_value("openai:gpt-5.4").run()
    at = at.selectbox(key="member_model_general_gm_research").set_value(OPUS).run()
    assert not at.exception, at.exception
    saved = json.loads((tmp_path / "data" / "member_models.json").read_text(encoding="utf-8"))
    assert saved["general"]["gm_research"] == OPUS
    dgm = at.selectbox(key="member_model_general_dgm_research")
    assert not any(o.startswith("claude-opus-5-5") for o in dgm.options)
    assert dgm.value == "openai:gpt-5.4"  # gpt-5.4 는 Opus 보다 낮으니 그대로

    at = at.selectbox(key="member_model_general_gm_research").set_value("openai:gpt-5.4").run()
    assert not at.exception, at.exception
    dgm = at.selectbox(key="member_model_general_dgm_research")
    assert dgm.value == "google:gemini-3.8-flash"  # 부장과 같은 강도였던 차장이 자동으로 낮아짐
    assert dgm.format_func("").startswith("기본 · gpt-5.4-mini")  # '기본'은 본인 선택이 아니라 직급 설정을 보여 준다
    assert any("차장·분석" in i.value and "gemini-3.8-flash" in i.value for i in at.info)
    assert at.selectbox(key="member_model_general_gm_research").format_func("").startswith("기본 · gpt-5.4 ")

    at = at.selectbox(key="member_model_general_gm_research").set_value("openai:gpt-5.4-mini").run()
    assert at.selectbox(key="member_model_general_dgm_research").value == "upstage:solar-pro2"  # 한 번 더 낮아짐
    am = at.selectbox(key="member_model_general_am_fact")  # 기본 대리(gemini-3.5-flash)는 차장(solar-pro2)과 같은 강도 → 저장 없이 자동 제한
    assert am.value == "" and am.format_func("").startswith("🔽 기본(자동 제한) · gemini-3.5-flash-lite")
    at = at.button(key="member_reset_general").click().run()
    assert not at.exception, at.exception
    assert json.loads((tmp_path / "data" / "member_models.json").read_text(encoding="utf-8")) == {}


def test_selector_and_task_input_show_chosen_model_off(app, tmp_path, set_keys, monkeypatch):
    set_keys("openai", "google", "freellm")
    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: False)  # FreeLLMAPI(Docker) 꺼짐
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "member_models.json").write_text(json.dumps({"general": {"staff_draft": FREE_PICK}}), encoding="utf-8")
    at = app.run()
    assert not at.exception, at.exception
    off = [m.value for m in at.markdown if "고른 모델 꺼짐" in m.value]
    assert len(off) == 1 and "무료 서버 꺼짐" in off[0] and "Docker Desktop" in off[0]  # 사원·초안 한 명만
    warning = next(w.value for w in at.warning if "고른 모델이 꺼져 있는 팀원" in w.value)
    assert "사원·초안" in warning and "실패합니다" in warning and "Docker Desktop" in warning

    monkeypatch.setattr(llm_factory, "freellm_reachable", lambda: True)  # 켜지면 표시가 사라진다
    llm_factory._reachable.clear()
    at = at.run()
    assert not any("고른 모델 꺼짐" in m.value for m in at.markdown)
    assert not any("고른 모델이 꺼져 있는 팀원" in w.value for w in at.warning)
