"""🏢 부서 업무 화면 — 부서 하나 = 사무실 하나, 👔 부서장 배정 → 팀 실행, 부서 업무 목록에서 팀 업무 직접 고르기.

- 배치 단위 테스트 : 부서원 전원(부서장 + 모든 팀의 조직도)이 한 사무실에 자리가 있고, 부서장실·회의실·자료실은 하나씩
- 스냅샷 단위 테스트 : 일하는 팀은 회의실(분배)·자료실(조사)로, 쉬는 팀은 자리에. 쪽지는 사용자 → 부서장 → 팀장
- 화면 연기 테스트(AppTest) : 예전 ?view=team 주소 → 부서 화면, 부서장 배정 → 그 팀 실행, 업무 직접 고르기, 다른 부서 일 넘기기.
  모델은 모두 가짜(use_script) 또는 키 없음 — .env 의 실제 키로 진짜 모델을 부르지 않는다.
"""

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import building, live, office
from core.config import BASE_DIR, list_teams, load_ranks, team_org
from core.schemas import Assignment, Delegation
from tests.fakes import Script

APP = str(BASE_DIR / "app.py")
DOM_Q = "다음 달 KTX 타고 강릉 1박 2일 국내 여행"
ABR_Q = "비자랑 환전 필요한 오사카 해외 여행 3박 4일"


@pytest.fixture(scope="module")
def world():
    teams = {t.id: t for t in list_teams()}
    ranks = load_ranks()
    depts = building.departments(list(teams), building.load_config())
    return teams, ranks, depts


def layout_of(world, dept_id: str) -> tuple[dict, dict]:
    teams, ranks, depts = world
    d = next(x for x in depts if x.id == dept_id)
    orgs = {t: team_org(teams[t], ranks) for t in d.teams}
    return office.build_dept_layout(d.id, d.name, [(t, teams[t].name, d.color, orgs[t]) for t in d.teams]), orgs


# ─── 배치 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("dept_id", ["research", "invest", "travel", "creative", "security", "career"])
def test_every_department_member_has_a_seat_and_rooms_are_shared(world, dept_id):
    layout, orgs = layout_of(world, dept_id)
    expected = {office.HEAD} | {office.agent_key(t, n.id) for t, org in orgs.items() for n in org.walk()}
    assert set(layout["seats"]) == expected  # 부서장 + 모든 팀의 팀원 전원
    kinds = [r["kind"] for r in layout["rooms"]]
    assert kinds.count("lead") == kinds.count("meeting") == kinds.count("library") == 1  # 부서마다 하나씩
    assert [r["team"] for r in layout["rooms"] if r["kind"] == "zone"] == list(orgs)  # 팀마다 구역 하나
    seats = [tuple(s) for s in layout["seats"].values()]
    assert len(seats) == len(set(seats))  # 자리가 겹치지 않는다
    blocked = {(i, j) for x, y, w, h in layout["walls"] for i in range(x, x + w) for j in range(y, y + h)}
    blocked |= {(i, j) for f in layout["furniture"] if f["type"] not in ("rug", "board")
                for i in range(f["x"], f["x"] + f["w"]) for j in range(f["y"], f["y"] + f["h"])}
    assert not blocked & set(seats)  # 벽·가구 위에 앉지 않는다
    for room in (r for r in layout["rooms"] if r["kind"] == "zone"):  # 팀원은 자기 팀 구역 안에
        for key, (x, y) in layout["seats"].items():
            if key.startswith(room["team"] + "/"):
                assert room["x"] <= x < room["x"] + room["w"] and room["y"] <= y < room["y"] + room["h"], key
    assert len(layout["spots"]["meeting"]) >= max(len(o.walk()) for o in orgs.values())  # 한 팀 전원이 회의실에 앉는다


def test_travel_office_holds_both_teams_and_small_department_is_compact(world):
    travel, orgs = layout_of(world, "travel")
    assert len(travel["seats"]) == 1 + sum(len(o.walk()) for o in orgs.values()) == 15
    career, _ = layout_of(world, "career")
    assert career["w"] < travel["w"]  # 사람이 적은 부서는 사무실도 작게 (화면에서는 더 크게 보인다)
    assert any(f["type"] in ("cabinet", "printer") for f in career["furniture"])  # 빈 바닥은 집기로 채운다


def test_snapshot_moves_working_team_and_relays_mail_through_head(world):
    teams, ranks, depts = world
    layout, orgs = layout_of(world, "travel")
    dom, abr = orgs["travel_domestic"], orgs["travel_abroad"]
    run = live.new_run("r" * 32, DOM_Q, dom.id, "travel_domestic")
    run["activity"] = {dom.id: live.WORKING_LABELS["plan"], dom.children[0].id: live.WORKING_LABELS["research"]}

    def view(tid, org, r):
        names = {n.id: n.id for n in org.walk()}
        return office.TeamView(team_id=tid, team_name=teams[tid].name, org=org, names=names,
                               ranks={n.id: (n.rank, n.rank) for n in org.walk()}, specialties={},
                               parents={c.id: n.id for n in org.walk() for c in n.children}, run=r,
                               active_ids={n.id for n in org.walk()} if r else None)

    snap = office.dept_snapshot(layout, [view("travel_domestic", dom, run), view("travel_abroad", abr, None)],
                                head_name="여행부 부장", head_about="", focus="", active_team="travel_domestic")
    by_id = {a["id"]: a for a in snap["agents"]}
    assert len(by_id) == 15 and by_id[office.HEAD]["head"] and by_id[office.HEAD]["bubble"].startswith("👀")
    assert by_id[f"travel_domestic/{dom.id}"]["place"] == "meeting"  # 📋 분배 → 회의실
    assert by_id[f"travel_domestic/{dom.children[0].id}"]["place"] == "library"  # 🌐 자료 조사 → 자료실
    assert all(a["place"] == "desk" for k, a in by_id.items() if k.startswith("travel_abroad/"))  # 쉬는 팀은 자리에
    assert by_id[f"travel_abroad/{abr.id}"]["team"] == "🌏 해외 여행팀"
    hops = [(e["from"], e["to"]) for e in snap["events"]]
    assert hops[:2] == [(office.USER, office.HEAD), (office.HEAD, f"travel_domestic/{dom.id}")]  # 사용자 → 부서장 → 팀장
    assert snap["running"] and "🚄 국내 여행팀" in snap["stage"]
    # 팀장이 업무를 나누면 그 팀 안에서만 쪽지가 오간다
    run["steps"].append(live_step(dom.id, Delegation(analysis="분석", assignments=[Assignment(child_id=dom.children[0].id, subtask="일정")])))
    snap = office.dept_snapshot(layout, [view("travel_domestic", dom, run)], head_name="부장", head_about="", focus="")
    assert (f"travel_domestic/{dom.id}", f"travel_domestic/{dom.children[0].id}") in [(e["from"], e["to"]) for e in snap["events"]]


def live_step(node_id, delegation):
    from core.schemas import AgentAnswer, ModelSpec

    return AgentAnswer(node_id=node_id, rank_id="team_lead", kind="plan", content="분배", delegation=delegation,
                       model=ModelSpec(provider="openai", model="x"))


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app_env(tmp_path, monkeypatch, set_keys):
    """키 없음 · 임시 데이터 폴더 · 실행 시작은 기록만 (팀 id, 질문)."""
    import streamlit as st

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(live, "start_run", lambda run, *a, **k: started.append((run["team_id"], run["question"])))
    st.cache_resource.clear()
    yield {"data": tmp_path / "data", "started": started}
    st.cache_resource.clear()


def open_app(**params) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    for k, v in params.items():
        at.query_params[k] = v
    at.session_state["office_view"] = "office"
    at.run()
    assert not at.exception, at.exception
    return at


def office_snapshot(at: AppTest) -> dict:
    comp = next(e for e in at.get("component_instance") if "pixel_office" in e.proto.component_name)
    return json.loads(comp.proto.json_args)["snapshot"]


def test_old_team_link_opens_department_office_with_everyone(app_env):
    at = open_app(view="team", team="travel_abroad")  # 예전 '🏢 팀 업무' 주소
    assert at.session_state["view"] == "dept" and at.session_state["dept_pick"] == "travel"
    assert at.session_state["team_pick"] == "travel_abroad"
    assert at.sidebar.radio(key="view").format_func("dept") == "🏢 부서 업무"
    assert at.title[0].value == "🏢 ✈️ 여행부"
    snap = office_snapshot(at)
    assert len(snap["agents"]) == 15 and snap["activeTeam"] == "travel_abroad"
    assert sorted(r["label"] for r in snap["layout"]["rooms"] if r["kind"] == "zone") == ["🌏 해외 여행팀", "🚄 국내 여행팀"]
    picker = at.selectbox(key="dept_task_travel")
    assert picker.value == "head"  # 기본 = 👔 부서장에게 맡기기
    assert picker.options[0].startswith("👔 부서장에게 맡기기")
    assert {"🚄 국내 여행팀 › 💬 자유 질문", "🌏 해외 여행팀 › 💬 자유 질문"} <= set(picker.options)
    assert at.text_area(key="dept_head_text") is not None
    assert any(b.key == "see_team_travel_domestic" for b in at.button)  # 팀 상태 줄 — 다른 팀 업무 보기


def test_single_team_department_and_department_org_chart(app_env):
    at = open_app(view="dept", dept="career")
    snap = office_snapshot(at)
    assert len(snap["agents"]) == 5 and snap["agents"][0]["id"] == office.HEAD  # 부서장 + 커리어 팀 4명
    assert not any(b.key and b.key.startswith("see_team_") for b in at.button)  # 팀이 하나면 팀 줄 없음
    at = at.radio(key="office_view").set_value("chart").run()  # 🗂 부서 조직도 (부서장 → 팀장 → 팀원)
    assert not at.exception, at.exception
    chart = next(e for e in at.get("component_instance") if "click_detector" in e.proto.component_name)
    html_text = json.loads(chart.proto.json_args)["html_content"]
    assert "node:__head__" in html_text and "node:career/lead" in html_text and "커리어부 부장" in html_text


def test_head_assigns_team_and_team_workflow_starts(app_env, use_script):
    use_script(Script(TeamPick=building.TeamPick(analysis="해외", team_id="travel_abroad", reason="비자·환전이 필요한 해외 여행")))
    at = open_app(view="dept", dept="travel")
    at.text_area(key="dept_head_text").input(ABR_Q).run()
    at = at.button(key="dept_head_go").click().run()
    assert not at.exception, at.exception
    assert "여행부 부장 → **🌏 해외 여행팀**" in at.success[0].value and "비자·환전" in at.success[0].value
    assert any("판단: 부장 모델" in c.value and "사용 모델" in c.value and "범위 확인" in c.value for c in at.caption)
    at = at.button(key="dept_head_start").click().run()  # ▶ 바로 맡기기 → 해외 팀의 팀장이 곧바로 시작
    assert not at.exception, at.exception
    assert app_env["started"] == [("travel_abroad", ABR_Q)]
    assert at.session_state["dept_pick"] == "travel" and at.session_state["team_pick"] == "travel_abroad"
    assert at.selectbox(key="dept_task_travel").value == "head"  # 부서 화면 그대로
    assert "여행부 부장** → **🌏 해외 여행팀** 배정" in at.info[0].value
    snap = office_snapshot(at)
    assert snap["running"] and snap["activeTeam"] == "travel_abroad"
    assert (office.USER, office.HEAD) in [(e["from"], e["to"]) for e in snap["events"]]  # 사용자 → 부서장 쪽지


def test_direct_task_pick_goes_straight_to_that_team(app_env):
    at = open_app(view="dept", dept="travel")
    at = at.selectbox(key="dept_task_travel").set_value("travel_domestic:free").run()
    assert not at.exception, at.exception
    assert at.session_state["team_pick"] == "travel_domestic"
    assert any("직접 고른 업무" in c.value and "🚄 국내 여행팀" in c.value for c in at.caption)  # 부서장 → ○○팀 표시
    at.text_area(key="q_text").input(DOM_Q)
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    assert app_env["started"] == [("travel_domestic", DOM_Q)]
    assert any("여행부 부장 → **🚄 국내 여행팀**" in c.value for c in at.caption)
    # 팀 업무 양식도 같은 목록에서 — 고르면 그 팀 양식이 그대로 나온다
    at = at.selectbox(key="dept_task_travel").set_value("travel_abroad:plan").run()
    assert not at.exception and at.session_state["team_pick"] == "travel_abroad"
    assert any(k.startswith("tf_travel_abroad_plan_") for k in [w.key or "" for w in at.text_input] + [w.key or "" for w in at.date_input])


def test_request_for_another_department_is_handed_over(app_env):
    at = open_app(view="dept", dept="career")
    at.text_area(key="dept_head_text").input(ABR_Q).run()
    at = at.button(key="dept_head_go").click().run()  # 키 없음 → 낱말 비교
    assert not at.exception, at.exception
    assert "🌏 해외 여행팀" in at.warning[0].value and "여행부" in at.warning[0].value
    assert app_env["started"] == [] and not any(b.key == "dept_head_start" for b in at.button)
    at = at.button(key="dept_head_outside").click().run()  # ➡️ 여행부 부서장에게 넘기기 → 곧바로 배정
    assert not at.exception, at.exception
    assert at.session_state["dept_pick"] == "travel" and at.text_area(key="dept_head_text").value == ABR_Q
    assert "🌏 해외 여행팀" in at.success[0].value


def test_department_records_gather_all_teams(app_env):
    from core.schemas import HumanDecision
    from core.storage import Storage

    storage = Storage(app_env["data"] / "runs.db")
    for i, (tid, q) in enumerate([("travel_domestic", DOM_Q), ("travel_abroad", ABR_Q)]):
        storage.create_run(f"run{i}", tid, "lead", q)
        storage.save_decision(f"run{i}", HumanDecision(action="reject", score=3))
    at = open_app(view="dept", dept="travel")
    history = next(e for e in at.sidebar.expander if e.label.startswith("🕘 지난 업무"))
    assert history.label == "🕘 지난 업무 (2)"  # 두 팀 기록을 부서 단위로
    at = at.sidebar.selectbox(key="hist_team_travel").set_value("travel_abroad").run()  # 팀 필터 (선택)
    history = next(e for e in at.sidebar.expander if e.label.startswith("🕘 지난 업무"))
    assert history.label == "🕘 지난 업무 (1)"
