"""👔 부서장 중심 흐름 — 🛎 인포메이션 센터(부서장 회의) · 부서장 창구(팀 배정) · 🆕 새 부서 만들기/나중에 · 커맨드 센터 범위 확인."""

import json

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from core import autonomy, building, dept_builder, reception
from core.config import BASE_DIR, list_teams, load_models, load_ranks, load_yaml
from core.dept_builder import DeptBuildError
from core.reception import DeptVerdict, MeetingResult, NewDeptProposal, ProposedMember
from tests.fakes import Script

APP = str(BASE_DIR / "app.py")
DOM_Q = "다음 달 KTX 타고 강릉 1박 2일 국내 여행"
ABR_Q = "비자랑 환전 필요한 오사카 해외 여행 3박 4일"
TAX_Q = "우리 가게 부가세 신고서를 써 줘"


@pytest.fixture(scope="module")
def teams():
    return list_teams()


@pytest.fixture(scope="module")
def depts(teams):
    return building.departments([t.id for t in teams], building.load_config())


def verdicts(depts, ok: dict[str, int]) -> list[DeptVerdict]:
    return [DeptVerdict(dept_id=d.id, can_handle=d.id in ok, fit=ok.get(d.id, 0), reason=f"{d.name} 판단") for d in depts]


def proposal(**extra) -> NewDeptProposal:
    base = dict(
        analysis="세무 신고 전문 지식이 필요", dept_id="Tax Dept", dept_name="세무부", emoji="🧾",
        dept_description="세금 신고서 초안을 만든다", head_persona="세법 조항과 숫자 계산을 먼저 본다.",
        team_id="vat team", team_name="🧾 부가세팀", team_description="부가세 신고서 초안", scope_summary="부가세·종합소득세 신고",
        accepts=["부가세 신고서 초안"], rejects=["주식 투자 (→ 투자부)"], keywords=["부가세", "신고서", "세금"],
        members=[ProposedMember(id="lead", rank="team_lead", name="팀장", specialty="세무 총괄"),
                 ProposedMember(id="law", rank="deputy_gm", name="차장·세법", specialty="세법 조사"),
                 ProposedMember(id="law", rank="ceo", name="대리·계산", specialty="세액 계산")],  # id 겹침·없는 직급
        tools_needed=["홈택스 자료 (사람이 올림)", "웹 검색 (필요하면 사람이 켬)"], review_criteria=["세액 계산이 맞는가"],
    )
    return NewDeptProposal(**{**base, **extra})


# ─── 🧭 부서장 배정: 여행부 국내/해외 ───────────────────────────────────


def test_travel_head_routes_domestic_and_abroad(set_keys, use_script, teams, depts):
    travel = next(d for d in depts if d.id == "travel")
    cands = [t for t in teams if t.id in travel.teams]
    set_keys()  # 키 없음 → 낱말 비교
    assert building.route_request(DOM_Q, travel, cands, load_models())["team_id"] == "travel_domestic"
    assert building.route_request(ABR_Q, travel, cands, load_models())["team_id"] == "travel_abroad"
    use_script(Script(TeamPick=building.TeamPick(analysis="해외", team_id="travel_abroad", reason="오사카는 해외")))
    pick = building.route_request(ABR_Q, travel, cands, load_models())
    assert pick["how"] == "llm" and pick["team_id"] == "travel_abroad" and pick["model"]


# ─── 🛎 부서장 회의 ─────────────────────────────────────────────────


def test_meeting_without_model_uses_keywords(set_keys, teams, depts):
    set_keys()
    out = reception.meeting(DOM_Q, depts, teams, load_models())
    assert out["how"] == "keyword" and out["dept_id"] == "travel" and "낱말 비교" in out["note"]
    assert [v["dept_id"] for v in out["verdicts"]] == [d.id for d in depts]
    # 아무 낱말도 안 맞으면 범용 기본 팀(accept_all)의 부서 — 모델 없이는 '맡을 부서 없음'을 단정하지 않는다
    assert reception.meeting("음 그냥 이것저것", depts, teams, load_models())["dept_id"] == "research"
    no_general = [d for d in depts if d.id != "research"]
    assert reception.keyword_meeting("음 그냥 이것저것", no_general, {t.id: t for t in teams})["dept_id"] == ""


def test_meeting_one_call_for_all_departments(use_script, teams, depts):
    script = use_script(Script(MeetingResult=MeetingResult(analysis="해외 여행", verdicts=verdicts(depts, {"travel": 9, "research": 5}),
                                                           best_dept_id="travel")))
    out = reception.meeting(ABR_Q, depts, teams, load_models())
    assert out["how"] == "llm" and out["dept_id"] == "travel" and out["model"]
    assert len(script.prompts("MeetingResult")) == 1  # 부서마다 따로가 아니라 1회
    prompt = script.prompts("MeetingResult")[0]
    for d in depts:  # 모든 부서와 부장 관점이 한 프롬프트에
        assert d.name in prompt and d.head["persona"][:20] in prompt
    assert "해외 여행팀" in prompt and "범용 창구" in prompt


def test_meeting_checks_the_models_pick(use_script, teams, depts):
    ids = [d.id for d in depts]
    # 모델이 고른 부서보다 fit 이 높은 부서가 있으면 그쪽, 목록에 없는 부서·fit 미달은 버린다
    r = MeetingResult(analysis="", verdicts=verdicts(depts, {"travel": 6, "security": 9}), best_dept_id="travel")
    assert reception.pick_best(r, ids) == "security"
    r = MeetingResult(analysis="", verdicts=[DeptVerdict(dept_id="ghost", can_handle=True, fit=10, reason="")], best_dept_id="ghost")
    assert reception.pick_best(r, ids) == ""
    r = MeetingResult(analysis="", verdicts=verdicts(depts, {"research": 3}), best_dept_id="research")
    assert reception.pick_best(r, ids) == ""  # fit 3 < MIN_FIT → 맡을 부서 없음
    use_script(Script(MeetingResult=MeetingResult(analysis="세무", verdicts=verdicts(depts, {}), best_dept_id="")))
    out = reception.meeting(TAX_Q, depts, teams, load_models())
    assert out["dept_id"] == "" and all(not v["can_handle"] for v in out["verdicts"])


def test_propose_cleans_ids_and_members(use_script, teams, depts):
    rank_ids = [r.id for r in load_ranks()]
    taken = {t.id for t in teams} | {d.id for d in depts}
    use_script(Script(NewDeptProposal=proposal()))
    out = reception.propose(TAX_Q, depts, teams, load_models(), rank_ids, taken)
    p = out["proposal"]
    assert out["how"] == "llm" and p["dept_id"] == "tax_dept" and p["team_id"] == "vat_team"
    assert [m["id"] for m in p["members"]] == ["lead", "law", "law_2"]
    assert p["members"][0]["rank"] == "team_lead" and p["members"][2]["rank"] == "staff"  # 없는 직급 → 사원
    use_script(Script(NewDeptProposal=proposal(dept_id="travel", team_id="stock")))
    p = reception.propose(TAX_Q, depts, teams, load_models(), rank_ids, taken)["proposal"]
    assert p["dept_id"] == "travel_2" and p["team_id"] == "stock_2"  # 기존 id 와 겹치지 않게


def test_propose_without_model_gives_editable_draft(set_keys, teams, depts):
    set_keys()
    out = reception.propose(TAX_Q, depts, teams, load_models(), [r.id for r in load_ranks()], {d.id for d in depts})
    assert out["how"] == "fallback" and out["proposal"]["dept_id"] == "new_dept" and len(out["proposal"]["members"]) >= 2


# ─── 🆕 새 부서 파일: 초안 → 검증 → 저장 ──────────────────────────────


def drafted(teams, depts, **extra):
    rank_ids = [r.id for r in load_ranks()]
    p = proposal(**extra).model_dump()
    p["dept_id"], p["team_id"] = reception.slug(p["dept_id"], "x"), reception.slug(p["team_id"], "y")
    p["members"] = reception._clean_members(p["members"], rank_ids)
    return dept_builder.draft(p, TAX_Q, rank_ids, {d.color for d in depts})


def check(dept_text, team_text, teams, depts, tmp_path=None):
    return dept_builder.validate(dept_text, team_text, team_ids={t.id for t in teams}, dept_ids={d.id for d in depts},
                                 ranks=load_ranks(), models=load_models(), teams_dir=tmp_path)


def test_draft_uses_template_and_turns_tools_off(teams, depts):
    dept_text, team_text = drafted(teams, depts)
    dept, team = check(dept_text, team_text, teams, depts)
    assert dept["teams"] == ["vat_team"] and dept["head"]["name"] == "세무부 부장" and dept["color"] not in {d.color for d in depts}
    assert team["tools"] == [] and "웹 검색 (필요하면 사람이 켬)" in team_text  # 위험한 도구는 꺼 두고 주석으로만
    assert team["harness"] == load_yaml(BASE_DIR / "teams" / "_template" / "team.yaml")["harness"]  # _template 바탕
    assert team["scope"]["keywords"] == ["부가세", "신고서", "세금"] and team["free_examples"][0]["question"] == TAX_Q
    assert [c["id"] for c in team["org"]["children"]] == ["law", "law_2"]


@pytest.mark.parametrize("edit, message", [
    (lambda d, t: (d.replace("id: tax_dept", "id: travel"), t), "이미 있습니다"),
    (lambda d, t: (d.replace("- vat_team", "- stock"), t), "이미 있습니다"),
    (lambda d, t: (d.replace("id: tax_dept", "id: Tax-Dept"), t), "영문 소문자"),
    (lambda d, t: (d, t.replace("rank: deputy_gm", "rank: ceo")), "rank"),
    (lambda d, t: (d, t.replace("tools: []", "tools: [rocket_launcher]")), "없는 툴"),
    (lambda d, t: (d, t + "\n  : 깨진 줄"), "YAML"),
])
def test_validate_rejects_bad_drafts(teams, depts, edit, message):
    dept_text, team_text = edit(*drafted(teams, depts))
    with pytest.raises(DeptBuildError, match=message):
        check(dept_text, team_text, teams, depts)


def test_save_writes_team_and_department_keeping_comments(tmp_path, teams, depts):
    cfg = tmp_path / "departments.yaml"
    original = (BASE_DIR / "config" / "departments.yaml").read_text(encoding="utf-8")
    cfg.write_text(original, encoding="utf-8")
    dept_text, team_text = drafted(teams, depts)
    dept, _ = check(dept_text, team_text, teams, depts, tmp_path / "teams")
    path = dept_builder.save(dept, team_text, config_path=cfg, teams_dir=tmp_path / "teams")
    assert path == tmp_path / "teams" / "vat_team" / "team.yaml"
    after = cfg.read_text(encoding="utf-8")
    assert after.startswith(original.split("departments:")[0])  # 위쪽 설명 주석 그대로
    assert after.index("id: tax_dept") < after.index("# 부서에 적지 않은 팀") < after.index("other:")
    data = yaml.safe_load(after)
    assert data["other"]["id"] == "etc" and data["departments"][-1]["id"] == "tax_dept"
    new_ids = [t.id for t in teams] + ["vat_team"]
    got = building.departments(new_ids, data)
    assert got[-1].id == "tax_dept" and got[-1].teams == ["vat_team"] and got[-1].head["name"] == "세무부 부장"
    floors = building.plan_floors(got, building.per_floor(data))
    assert ("tax_dept", "vat_team") in floors[-1]  # 새 층(또는 맨 위층 빈 칸)에 방이 생긴다
    with pytest.raises(DeptBuildError, match="이미 있습니다"):
        dept_builder.save(dept, team_text, config_path=cfg, teams_dir=tmp_path / "teams")


def test_save_rolls_back_when_config_is_broken(tmp_path, teams, depts):
    cfg = tmp_path / "departments.yaml"
    cfg.write_text("departments: [이건, 목록]\nother: {id: etc}\n", encoding="utf-8")  # 블록 목록이 아니라 끼워 넣으면 깨진다
    dept_text, team_text = drafted(teams, depts)
    dept, _ = check(dept_text, team_text, teams, depts, tmp_path / "teams")
    with pytest.raises(DeptBuildError):
        dept_builder.save(dept, team_text, config_path=cfg, teams_dir=tmp_path / "teams")
    assert cfg.read_text(encoding="utf-8").startswith("departments: [이건") and not (tmp_path / "teams" / "vat_team").exists()


def test_later_store_and_audit(tmp_path):
    store = reception.LaterStore(tmp_path)
    a = store.add("질문 1", {"dept_id": "a"})
    b = store.add("질문 2", {"dept_id": "b"})
    assert [i["id"] for i in store.items()] == [b["id"], a["id"]] and store.get(a["id"])["question"] == "질문 1"
    store.remove(a["id"])
    assert [i["question"] for i in store.items()] == ["질문 2"]
    dept_builder.audit(tmp_path, {}, "dept_deferred", later_id=b["id"])
    dept_builder.audit(tmp_path, {}, "dept_created", dept_id="b")
    log = autonomy.AuditLog(tmp_path / "autonomy" / "audit.jsonl")
    assert [e["event"] for e in log.entries()] == ["dept_deferred", "dept_created"] and log.verify() == (True, None)
    assert log.entries()[0]["via"] == "info_center" and log.entries()[0]["decider"] == "user"


# ─── 화면 (AppTest) ─────────────────────────────────────────────────────


@pytest.fixture
def app_env(tmp_path, monkeypatch, set_keys):
    import streamlit as st

    from core import live

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(live, "start_run", lambda run, *a, **k: started.append((run["team_id"], run["question"])))
    st.cache_resource.clear()
    yield {"data": tmp_path / "data", "started": started, "tmp": tmp_path}
    st.cache_resource.clear()


def home() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at


def test_home_is_a_floor_guide_with_info_desk_and_full_rooms(app_env, teams):
    at = home()
    assert at.radio(key="desk_pick").value == "info"  # 처음 창구 = 🛎 인포메이션 센터
    assert at.radio(key="desk_pick").options[0] == "🛎 인포메이션 센터"
    snap = next(e for e in at.get("component_instance") if "pixel_building" in e.proto.component_name)
    snap = json.loads(snap.proto.json_args)["snapshot"]
    assert snap["layout"]["lobby"]["spots"][0]["id"] == "info"
    for t in teams:  # 방 안에 팀원 수만큼 자리 (8명까지)
        room = snap["rooms"][t.id]
        assert len(room["people"]) == min(room["card"]["headcount"], building.MAX_SEATS)
        assert len(room["people"]) + room["more"] == room["card"]["headcount"]
    travel = snap["depts"]["travel"]
    assert travel["head"] == "여행부 부장" and travel["headcount"] == sum(c["headcount"] for c in travel["teamCards"])
    assert travel["teamCards"][0]["tasks"] and travel["state"] == "○ 모두 쉬는 중"
    text = "\n".join(m.value for m in at.markdown)
    assert "🗂 층별 부서 안내" in text and "##### 3층 — ✈️ 여행부" in text
    assert any(b.key == "home_team_travel_domestic" for b in at.button)  # 팀 바로 들어가기 그대로
    at = at.button(key="guide_dept_travel").click().run()  # 부서 카드 → 부서장 창구
    assert not at.exception and at.radio(key="desk_pick").value == "travel"
    assert at.text_area(key="dept_head_text") is not None


def test_dept_desk_assigns_and_starts_the_team(app_env, use_script):
    use_script(Script())  # 모델 답 없음 → 낱말 비교 (.env 의 실제 키로 진짜 호출하지 않게)
    at = home()
    at = at.radio(key="desk_pick").set_value("travel").run()
    assert any("🎛 부서장 모델:" in c.value for c in at.caption)
    at.text_area(key="dept_head_text").input(DOM_Q).run()
    at = at.button(key="dept_head_go").click().run()
    assert not at.exception and "🚄 국내 여행팀" in at.success[0].value
    assert any("판단: 낱말 비교" in c.value for c in at.caption)
    at = at.radio(key="dept_head_team").set_value("travel_abroad").run()  # 다른 팀으로 바꿔 보내기
    assert any("대신 **🌏 해외 여행팀**" in c.value for c in at.caption)
    at = at.button(key="dept_head_start").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "team" and at.session_state["team_pick"] == "travel_abroad"
    assert app_env["started"] == [("travel_abroad", DOM_Q)]  # 팀장이 바로 시작 (기존 workflow)
    assert "여행부 부장" in at.info[0].value and "다른 팀으로 바꿔 보냄" in at.info[0].value


def test_info_center_hands_off_to_department_head(app_env, use_script, depts):
    use_script(Script(MeetingResult=MeetingResult(analysis="국내 여행", verdicts=verdicts(depts, {"travel": 9}), best_dept_id="travel"),
                      TeamPick=building.TeamPick(analysis="국내", team_id="travel_domestic", reason="강릉은 국내")))
    at = home()
    at.text_area(key="info_text").input(DOM_Q).run()
    at = at.button(key="info_go").click().run()
    assert not at.exception, at.exception
    table = next(m.value for m in at.markdown if "부서장 판단" in m.value)
    assert "| ✅ | ✈️ 여행부 | 9 |" in table and "| ❌ | 💹 투자부 | 0 |" in table
    assert "여행부** 이(가) 맡을 수 있습니다" in at.success[0].value
    at = at.button(key="info_handoff").click().run()  # → 여행부 부서장 창구가 열리고 곧바로 팀 배정
    assert not at.exception, at.exception
    assert at.radio(key="desk_pick").value == "travel" and at.text_area(key="dept_head_text").value == DOM_Q
    assert "🚄 국내 여행팀" in at.success[0].value and "강릉은 국내" in at.success[0].value
    assert any("사용 모델" in c.value for c in at.caption)


def test_info_center_no_department_later_then_make(app_env, use_script, depts, monkeypatch):
    use_script(Script(MeetingResult=MeetingResult(analysis="세무", verdicts=verdicts(depts, {}), best_dept_id=""),
                      NewDeptProposal=proposal()))
    saved = {}

    def fake_save(dept, team_text, **_):  # 실제 config/teams 는 건드리지 않는다 (저장 로직은 위 단위 테스트)
        saved.update(dept=dept, team=team_text)
        return app_env["tmp"] / "team.yaml"

    monkeypatch.setattr(dept_builder, "save", fake_save)
    at = home()
    at.text_area(key="info_text").input(TAX_Q).run()
    at = at.button(key="info_go").click().run()
    assert not at.exception, at.exception
    assert "이 일을 할 부서가 없습니다. 새 부서를 만들까요?" in at.warning[0].value
    card = "\n".join(m.value for m in at.markdown)
    assert "🧾 세무부" in card and "🧾 부가세팀" in card and "차장·세법" in card and "홈택스 자료" in card
    at = at.button(key="info_later").click().run()  # [나중에]
    assert not at.exception and "추후 진행 목록에 남겼습니다" in at.success[0].value
    items = reception.LaterStore(app_env["data"]).items()
    assert len(items) == 1 and items[0]["question"] == TAX_Q
    later = next(e for e in at.expander if e.label.startswith("🗂 추후 진행"))
    assert later.label.endswith("(1)")
    at = at.button(key=f"later_make_{items[0]['id']}").click().run()  # 나중에 다시 꺼내 [만들기] → 미리 보기
    assert not at.exception, at.exception
    assert "tax_dept" in at.text_area(key="info_dept_yaml").value and "tools: []" in at.text_area(key="info_team_yaml").value
    assert saved == {}  # 미리 보기만으로는 저장하지 않는다
    at.text_area(key="info_dept_yaml").input(at.text_area(key="info_dept_yaml").value.replace("id: tax_dept", "id: travel")).run()
    at = at.button(key="info_save").click().run()
    assert "저장하지 않았습니다" in at.error[0].value and saved == {}  # 검증 실패 → 저장 안 함
    at.text_area(key="info_dept_yaml").input(at.text_area(key="info_dept_yaml").value.replace("id: travel", "id: tax_dept")).run()
    at = at.button(key="info_save").click().run()  # ✅ 확인하고 저장
    assert not at.exception, at.exception
    assert saved["dept"]["id"] == "tax_dept" and "name: 🧾 부가세팀" in saved["team"]
    assert "새 부서 **🧾 세무부**" in at.success[0].value
    assert any("원래 질문을 새 부서로 넘길까요?" in m.value for m in at.markdown)
    assert reception.LaterStore(app_env["data"]).items() == []  # 만든 후보는 목록에서 빠진다
    events = [e["event"] for e in autonomy.AuditLog(app_env["data"] / "autonomy" / "audit.jsonl").entries()]
    assert events == ["dept_deferred", "dept_created"]
    at = at.button(key="info_forward_no").click().run()
    assert not at.exception and at.text_area(key="info_text") is not None


def test_command_center_line_goes_through_scope_check(app_env):
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["view"] = "team"
    at.query_params["team"] = "stock"
    at.run()
    at.text_input(key="cc_text_stock").input(ABR_Q)
    at = next(b for b in at.button if b.label == "보내기").click().run()
    assert not at.exception, at.exception
    assert app_env["started"] == []  # 범위 밖 → 맡기지 않았다
    text = "\n".join(m.value for m in at.markdown)
    assert "범위가 아닙니다" in text and "해외 여행팀" in text
    at = at.button(key="cc_redirect_go_stock").click().run()  # ➡️ 맞는 팀에 맡기기
    assert not at.exception, at.exception
    assert at.session_state["team_pick"] == "travel_abroad" and app_env["started"] == [("travel_abroad", ABR_Q)]
    at.text_input(key="cc_text_travel_abroad").input("오사카 말고 후쿠오카 해외 여행이면?")
    at = next(b for b in at.button if b.label == "보내기").click().run()  # 범위 안 → 평소처럼 (진행 중이라 안내만)
    assert not at.exception, at.exception
