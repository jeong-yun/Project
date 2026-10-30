"""🏙 빌딩·부서·층 · 여행 팀 나누기(국내/해외) · 옛 travel 기록 이어 받기 · 🧭 부서장 · 화면(AppTest)."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import building
from core.config import BASE_DIR, ConfigError, list_teams, load_models, load_ranks, load_skills, load_team, team_org, validate_config
from core.member_models import MemberModelStore
from core.schemas import HumanDecision, Lesson, OrgNode, Team
from core.storage import Storage
from core.tasks import build_request
from core.usage import UsageEvent
from tests.fakes import Script

APP = str(BASE_DIR / "app.py")


def fake_team(tid: str, **extra) -> Team:
    org = OrgNode(id="lead", rank="team_lead", name="팀장", children=[OrgNode(id="a", rank="staff")])
    return Team(id=tid, name=f"{tid} 팀", description=f"{tid} 설명", org=org, **extra)


def cfg_with(*depts: tuple[str, list[str]], per: int = 4) -> dict:
    return {"rooms_per_floor": per,
            "departments": [{"id": d, "name": f"{d}부", "emoji": "🏢", "color": "#112233", "teams": t} for d, t in depts]}


# ─── 부서 설정 ─────────────────────────────────────────────────────────


def test_real_config_puts_every_team_in_a_department():
    teams = list_teams()
    depts = building.departments([t.id for t in teams], building.load_config())
    assert [d.name for d in depts] == ["리서치·지식부", "투자부", "여행부", "크리에이티브부", "보안부"]  # '기타' 없음
    assert next(d for d in depts if d.id == "travel").teams == ["travel_domestic", "travel_abroad"]
    assert sorted(t for d in depts for t in d.teams) == sorted(t.id for t in teams)


def test_unknown_duplicate_and_missing_teams():
    cfg = cfg_with(("a", ["t1", "gone", "t2"]), ("b", ["t2", "t3"]), ("empty", ["gone2"]))
    cfg["departments"][0]["color"] = "red"  # 잘못된 색 → 기본색
    depts = building.departments(["t1", "t2", "t3", "new1", "new2"], cfg)
    assert [(d.id, d.teams) for d in depts] == [("a", ["t1", "t2"]), ("b", ["t3"]), ("etc", ["new1", "new2"])]
    assert depts[0].color == "#64748B" and depts[-1].name == "기타"  # 두 번 적은 t2 는 처음 부서에만, 팀 없는 부서는 빠짐
    assert building.departments(["x"], {})[0].name == "기타"  # 설정 파일이 없어도 팀은 보인다


# ─── 층 계산 ──────────────────────────────────────────────────────────


def test_floors_grow_when_teams_are_added():
    eight = [(f"d{i}", [f"t{i}a", f"t{i}b"]) for i in range(4)]
    twelve = eight + [("d4", ["t4a", "t4b"]), ("d5", ["t5a", "t5b"])]
    ids8, ids12 = [t for _, ts in eight for t in ts], [t for _, ts in twelve for t in ts]
    f8 = building.plan_floors(building.departments(ids8, cfg_with(*eight)), 4)
    f12 = building.plan_floors(building.departments(ids12, cfg_with(*twelve)), 4)
    assert len(f8) == 2 and len(f12) == 3  # 8팀 → 2~3층, 12팀 → 4층까지 (1층은 로비)
    l8, l12 = building.build_layout(f8, 4), building.build_layout(f12, 4)
    assert l12["h"] - l8["h"] == building.FLOOR_H and l12["w"] == l8["w"]  # 위로 한 층 자란다
    assert [f["n"] for f in l12["floors"]] == [2, 3, 4]
    assert l12["floors"][-1]["y"] < l12["floors"][0]["y"] < l12["lobby"]["y"]  # 위층일수록 위에


def test_department_stays_on_one_floor_and_big_one_spans():
    depts = building.departments(["a1", "a2", "a3", "b1", "b2", "c1", "c2", "c3", "c4", "c5"],
                                 cfg_with(("a", ["a1", "a2", "a3"]), ("b", ["b1", "b2"]), ("c", ["c1", "c2", "c3", "c4", "c5"])))
    floors = building.plan_floors(depts, 4)
    assert [[t for _, t in f] for f in floors] == [["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2", "c3", "c4"], ["c5"]]
    layout = building.build_layout(floors, 4)
    first = layout["floors"][0]
    assert [r["team"] for r in first["rooms"]] == ["a1", "a2", "a3", ""]  # 빈 사무실 한 칸
    assert first["banners"] == [{"dept": "a", "x": 7, "y": first["y"], "w": 30, "h": 1}]  # 부서 띠는 방 3개 너비


def test_real_building_has_lobby_and_three_department_floors():
    teams = list_teams()
    cfg = building.load_config()
    assert building.floor_of("general", teams, cfg) == 2 and building.floor_of("travel_abroad", teams, cfg) == 3
    assert building.floor_of("infra_security", teams, cfg) == 4


# ─── 팀 요약 카드 · 상태 ────────────────────────────────────────────────


def test_team_card_comes_from_team_yaml():
    ranks = load_ranks()
    team = load_team("travel_abroad")
    card = building.team_card(team, team_org(team, ranks), ranks)
    assert "비자/입국" in card["description"]
    assert card["lead"].startswith("팀장·해외여행기획 — 해외 여행 기획 총괄")
    assert card["members"] == "팀장 1 · 부장 2 · 대리 2 · 사원 2" and card["headcount"] == 7
    assert card["tasks"] == ["🌏 해외 여행 계획서"]
    assert {"📅 여행 달력", "🌐 웹 검색", "🗺 경로 지도"} <= set(card["tools"])
    general = load_team("general")
    assert building.team_card(general, team_org(general, ranks), ranks)["tasks"] == ["💬 자유 질문"]


def run_dict(team_id: str, done: bool, error: str = "", activity: dict | None = None) -> dict:
    return {"team_id": team_id, "done": done, "error": error, "activity": activity or {}, "status": "awaiting_human"}


def test_status_lights():
    runs = [run_dict("a", False, activity={"lead": "📝 초안 쓰는 중", "x": "⏳ 하위 기다리는 중", "y": "🌐 자료 조사 중"}),
            run_dict("b", True), run_dict("c", True, error="boom")]
    assert building.team_status("a", runs) == ("running", 2)
    assert building.team_status("b", runs) == ("waiting", 0)
    assert building.team_status("c", runs) == ("error", 0)
    assert building.team_status("d", runs) == ("idle", 0)


def test_snapshot_rooms_costs_and_recent():
    teams = [fake_team("a"), fake_team("b"), fake_team("c")]
    ranks = load_ranks()
    snap = building.snapshot(teams, {t.id: team_org(t, ranks) for t in teams}, ranks, cfg_with(("x", ["a", "b"])),
                             open_runs=[run_dict("a", False, activity={"lead": "📝 초안"})],
                             recent={"b": {"text": "지난 질문", "when": "10-29 10:00", "status": "채택"}},
                             costs={"a": 0.5, "b": 1.25, "c": 0.0})
    assert snap["kind"] == "building" and snap["live"] is True
    assert snap["rooms"]["a"]["status"] == "running" and "팀원 1명" in snap["rooms"]["a"]["statusText"]
    assert snap["rooms"]["b"]["recent"]["status"] == "채택" and snap["rooms"]["c"]["dept"] == "etc"
    assert snap["floorCosts"] == ["$1.75"] and snap["rooms"]["b"]["cost"] == "$1.25"  # a·b·c 가 한 층 (4칸)
    json.dumps(snap, ensure_ascii=False)  # 화면으로 보낼 수 있다


# ─── 여행 팀 나누기 ──────────────────────────────────────────────────────


def test_travel_split_teams_are_valid_and_share_skills():
    ranks, models, teams = load_ranks(), load_models(), list_teams()
    validate_config(ranks, models, teams)
    ids = {t.id for t in teams}
    assert {"travel_domestic", "travel_abroad"} <= ids and "travel" not in ids
    dom, abroad = load_skills("travel_domestic"), load_skills("travel_abroad")
    for text in (dom, abroad):  # 공용 스킬 (teams/_shared/travel) 은 두 팀 모두에
        assert "## 스킬: travel_planning" in text and "## 스킬: adversarial_review" in text
    assert "KTX" in dom and "비자" not in dom
    assert "## 스킬: abroad" in abroad and "eSIM" in abroad and "## 스킬: domestic" not in abroad
    d, a = load_team("travel_domestic"), load_team("travel_abroad")
    assert d.tools == a.tools == ["travel_calendar", "web_search"] and d.features == a.features == ["route_map"]
    assert [n.id for n in team_org(d, ranks).walk()] == [n.id for n in team_org(a, ranks).walk()]  # 옛 교훈이 맞는 id
    q_dom = build_request(d.tasks[0], {"start_date": "2026-11-20", "end_date": "2026-11-22", "transport": "KTX·기차"})
    q_abr = build_request(a.tasks[0], {"start_date": "2026-11-20", "end_date": "2026-11-22", "airport": "인천"})
    assert "국내 여행" in q_dom.question and "KTX·기차" in q_dom.question and q_dom.tool_inputs["scope"] == "국내"
    assert "해외 여행" in q_abr.question and "인천" in q_abr.question
    assert "입국|비자" in q_abr.harness_options["required_terms"]


def test_bad_shared_skills_or_legacy_is_a_config_error():
    ranks, models = load_ranks(), load_models()
    with pytest.raises(ConfigError, match="skills_from"):
        validate_config(ranks, models, [fake_team("x", skills_from=["_shared/없는폴더"])])
    with pytest.raises(ConfigError, match="legacy"):
        validate_config(ranks, models, [fake_team("x"), fake_team("y", legacy={"team": "x"})])


# ─── 옛 travel 기록 이어 받기 (DB 는 그대로) ────────────────────────────────

LINKS = {"travel_domestic": ("travel", "국내 여행"), "travel_abroad": ("travel", "해외 여행")}


def seed_legacy(storage: Storage) -> None:
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer="1일차 …", score=5)
    for rid, q in (("r_dom", "2026-11-20 부터 2026-11-22 까지 2박 3일 국내 여행 계획을 세워줘.\n목적지: 강릉"),
                   ("r_abr", "2026-11-20 부터 2026-11-23 까지 3박 4일 해외 여행 계획을 세워줘.\n목적지: 오사카"),
                   ("r_free", "가을에 갈 만한 곳 추천해줘")):
        storage.create_run(rid, "travel", "lead", q)
        storage.save_decision(rid, decision)
    storage.add_lessons("travel", "r_dom", [Lesson(agent_id="lead", kind="improve", lesson="이동 시간을 꼭 쓴다")])
    storage.record_usage(UsageEvent(ts=1e10, provider="openai", model="gpt-5.4", key_id="k", key_masked="sk-…", input_tokens=10,
                                    output_tokens=5, cached_tokens=0, calls=1, cost_usd=0.25, team_id="travel", run_id="r_dom",
                                    purpose="", agent_id="lead"))


def test_old_travel_runs_lessons_usage_show_in_split_teams(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    seed_legacy(storage)
    before = [dict(r) for r in storage._run_sql("SELECT id, team_id FROM runs ORDER BY id", ())]
    storage.set_team_links(LINKS)
    dom = {r["id"] for r in storage.list_runs("travel_domestic")}
    abr = {r["id"] for r in storage.list_runs("travel_abroad")}
    assert dom == {"r_dom", "r_free"} and abr == {"r_abr", "r_free"}  # 어느 말도 없는 기록은 두 팀 모두에 (숨기지 않음)
    assert storage.recent_lessons("travel_abroad", "lead", 5) == ["이동 시간을 꼭 쓴다"]  # 교훈은 두 팀이 물려받음
    assert storage.lesson_counts("travel_domestic") == {"lead": 1}
    assert storage.usage_by_agent("travel_abroad", since=0)["lead"][0]["cost_usd"] == 0.25
    assert storage.team_for("travel", "… 해외 여행 계획") == "travel_abroad" and storage.team_for("travel") == "travel_abroad"
    assert storage.team_for("stock", "x") == "stock"
    assert storage.list_runs("stock") == []
    after = [dict(r) for r in storage._run_sql("SELECT id, team_id FROM runs ORDER BY id", ())]
    assert before == after  # DB 의 team_id 는 그대로 (지우거나 고치지 않음)


def test_member_model_choices_are_inherited_until_changed(tmp_path):
    store = MemberModelStore(tmp_path)
    store.save("travel", {"lead": "openai:gpt-5.5"})  # 나누기 전 여행 팀에서 고른 모델
    assert store.load("travel_abroad", "travel") == {"lead": "openai:gpt-5.5"}
    store.save("travel_abroad", {}, "travel")  # '기본으로 되돌리기' — 옛 선택이 다시 보이지 않게 기억
    assert store.load("travel_abroad", "travel") == {} and store.load("travel_domestic", "travel") == {"lead": "openai:gpt-5.5"}
    assert store.load("travel") == {"lead": "openai:gpt-5.5"}  # 옛 기록은 그대로
    assert json.loads((tmp_path / "member_models.json").read_text(encoding="utf-8"))["travel_abroad"] == {}


def test_avatar_falls_back_to_old_team_folder(tmp_path, monkeypatch):
    from core import config

    teams = tmp_path / "teams"
    (teams / "new").mkdir(parents=True)
    (teams / "new" / "team.yaml").write_text("name: 새 팀\nlegacy: {team: old}\n", encoding="utf-8")
    (teams / "old" / "avatars").mkdir(parents=True)
    (teams / "old" / "avatars" / "lead.png").write_bytes(b"png")
    monkeypatch.setattr(config, "TEAMS_DIR", teams)
    rank = load_ranks()[-1]
    assert config.resolve_avatar("new", rank, node=OrgNode(id="lead", rank=rank.id)) == teams / "old" / "avatars" / "lead.png"


# ─── 🧭 부서장 ────────────────────────────────────────────────────────


def test_dept_head_routes_with_model_or_keywords(set_keys, use_script):
    dom, abr = load_team("travel_domestic"), load_team("travel_abroad")
    dept = next(d for d in building.departments([dom.id, abr.id], building.load_config()) if d.id == "travel")
    models = load_models()
    one = building.route_request("아무거나", dept, [abr], models)
    assert one["team_id"] == "travel_abroad" and one["how"] == "one"
    set_keys()  # 키 없음 → 모델을 못 써도 낱말 비교로 고른다
    kw = building.route_request("KTX 타고 강릉 국내 여행", dept, [dom, abr], models)
    assert kw["team_id"] == "travel_domestic" and kw["how"] == "keyword" and "낱말 비교" in kw["note"]
    assert building.keyword_pick("비자 환전 필요한 해외 오사카", [dom, abr])[0] == "travel_abroad"
    script = use_script(Script(TeamPick=building.TeamPick(analysis="해외", team_id="travel_abroad", reason="오사카는 해외")))
    picked = building.route_request("오사카 3박 4일", dept, [dom, abr], models)
    assert picked == {"team_id": "travel_abroad", "reason": "오사카는 해외", "how": "llm", "model": picked["model"], "note": ""}
    assert "여행부" in script.prompts("TeamPick")[0] and "travel_domestic" in script.prompts("TeamPick")[0]
    use_script(Script(TeamPick=building.TeamPick(analysis="", team_id="stock", reason="")))  # 목록에 없는 팀 → 낱말 비교
    assert building.route_request("국내 KTX", dept, [dom, abr], models)["how"] == "keyword"


# ─── 화면 (AppTest) ─────────────────────────────────────────────────────


@pytest.fixture
def seeded(tmp_path, monkeypatch, set_keys):
    import streamlit as st

    set_keys()
    data = tmp_path / "data"
    seed_legacy(Storage(data / "runs.db"))
    (tmp_path / "results" / "travel").mkdir(parents=True)
    (tmp_path / "results" / "travel" / "옛_오사카_계획.md").write_text("# 옛 계획", encoding="utf-8")
    MemberModelStore(data).save("travel", {"lead": "openai:gpt-5.4"})
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(data))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def component_args(at: AppTest) -> dict:
    comp = next(e for e in at.get("component_instance") if "pixel_building" in e.proto.component_name)
    return json.loads(comp.proto.json_args)["snapshot"]


def test_home_shows_building_and_list_and_dept_head(seeded):
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    snap = component_args(at)
    assert snap["kind"] == "building" and len(snap["layout"]["floors"]) == 3
    assert snap["rooms"]["travel_abroad"]["recent"]["text"].startswith("2026-11-20 부터 2026-11-23")  # 옛 해외 업무
    assert snap["floorCosts"] == ["$0", "$0.250", "$0"]  # 옛 travel 사용량 → 3층(여행부) 비용 미터
    assert snap["rooms"]["travel_abroad"]["cost"] == snap["rooms"]["travel_domestic"]["cost"] == "$0"  # 어느 팀 몫인지 모름
    assert at.radio(key="home_section").value == "brief"
    # 방을 누름 → 다음 실행 맨 앞에서 그 팀 화면으로
    at.session_state["building_nav"] = {"kind": "team", "id": "travel_abroad"}
    at.run()
    assert not at.exception and at.session_state["view"] == "team" and at.session_state["team_pick"] == "travel_abroad"


def test_list_view_and_dept_head_fill_question(seeded):
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["building_list"] = True
    at.run()
    assert not at.exception, at.exception
    assert any(b.key == "home_team_travel_domestic" for b in at.button)
    assert any(m.value.startswith("##### 3층 — ✈️ 여행부") for m in at.markdown)
    at.selectbox(key="dept_head_pick").set_value("travel").run()
    at.text_area(key="dept_head_text").input("비자랑 환전 필요한 오사카 해외 여행").run()
    at.button(key="dept_head_go").click().run()
    assert not at.exception and "🌏 해외 여행팀" in at.success[0].value  # 키 없음 → 낱말 비교
    at.button(key="dept_head_open").click().run()
    assert not at.exception and at.session_state["view"] == "team" and at.session_state["team_pick"] == "travel_abroad"
    assert at.text_area(key="q_text").value == "비자랑 환전 필요한 오사카 해외 여행"


@pytest.mark.parametrize("team_id, mine, other", [("travel_domestic", "강릉", "오사카"), ("travel_abroad", "오사카", "강릉")])
def test_both_travel_teams_show_their_old_records(seeded, team_id, mine, other):
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["view"] = "team"
    at.query_params["team"] = team_id
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="office_view").value == "office"  # 팀 화면 기본 = 🏢 픽셀 사무실
    history = next(e for e in at.sidebar.expander if e.label.startswith("🕘 지난 업무"))
    assert history.label == "🕘 지난 업무 (2)"  # 내 옛 업무 + 국내/해외 말이 없는 옛 업무
    saved = next(e for e in at.sidebar.expander if e.label.startswith("📂 저장된 결과 파일"))
    assert saved.label == "📂 저장된 결과 파일 (1)"  # 옛 results/travel 의 파일
    assert any(c.value.startswith("지금: 1명 직접 선택.") for c in at.caption)  # 옛 여행 팀의 팀원 모델 선택
    assert any("📚 팀 스킬:" in c.value and "travel_planning" in c.value for c in at.sidebar.caption)
