"""🏢 픽셀 사무실 · 🖥️ 커맨드 센터 테스트.

- core/office.py : 배치(자리·벽·가구), 진행 상태 → 말풍선·갈 곳, ✉️ 쪽지 이벤트, '다들 뭐 해?' 한 줄 보고 (순수 함수)
- 화면 연기 테스트(AppTest) : 팀마다 픽셀 사무실이 그려지는지, 터미널이 모델 없이 '다들 뭐 해?'에 답하는지
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from core import office
from core.config import list_teams, load_ranks, parent_map, team_org
from core.live import WORKING_LABELS, apply_event, new_run
from core.schemas import AgentAnswer, ModelSpec
from core.storage import Storage
from tests.fakes import make_delegation, make_review

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SPEC = ModelSpec(provider="openai", model="x")
BLOCKING = {"desk", "table", "shelf", "plant", "cooler", "sofa"}


def step(node_id, kind="draft", review=None, delegation=None):
    return AgentAnswer(node_id=node_id, rank_id="staff", kind=kind, content="내용", review=review,
                       delegation=delegation, model=SPEC)


def ctx_args(org):
    ranks = {r.id: r for r in load_ranks()}
    nodes = list(org.walk())
    return dict(
        names={n.id: n.name or ranks[n.rank].name for n in nodes},
        ranks={n.id: (n.rank, ranks[n.rank].name) for n in nodes},
        specialties={n.id: n.specialty for n in nodes}, parents=parent_map(org),
    )


def blocked_tiles(layout) -> set[tuple[int, int]]:
    tiles = set()
    for x, y, w, h in layout["walls"]:
        tiles |= {(i, j) for i in range(x, x + w) for j in range(y, y + h)}
    for f in layout["furniture"]:
        if f["type"] in BLOCKING:
            tiles |= {(i, j) for i in range(f["x"], f["x"] + f["w"]) for j in range(f["y"], f["y"] + f["h"])}
    return tiles


# ─── 배치 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("team", list_teams(), ids=lambda t: t.id)
def test_layout_gives_every_agent_a_free_seat_inside_the_office(team):
    org = team_org(team, load_ranks())
    layout = office.build_layout(org)
    ids = [n.id for n in org.walk()]
    seats = [tuple(layout["seats"][i]) for i in ids]
    assert len(set(seats)) == len(ids)  # 한 자리에 한 명
    blocked = blocked_tiles(layout)
    for x, y in seats + [tuple(s[:2]) for spots in layout["spots"].values() for s in spots]:
        assert 0 < x < layout["w"] - 1 and 0 < y < layout["h"] - 1
        assert (x, y) not in blocked  # 앉는 자리·모이는 자리는 벽·가구 위가 아니다
    assert tuple(layout["seats"][org.id])[1] < 8  # 팀장은 위쪽 팀장실
    assert all(tuple(layout["seats"][i])[1] > 8 for i in ids if i != org.id)  # 나머지는 업무 공간
    assert {r["kind"] for r in layout["rooms"]} >= {"lead", "meeting", "library", "floor"}
    assert office.build_layout(org) == layout  # 같은 조직도 → 같은 배치


def test_manager_sits_above_their_staff(org):
    seats = office.build_layout(org)["seats"]
    assert seats["gm_research"][1] < seats["am_fact"][1]
    assert seats["gm_action"][1] < seats["staff_case"][1]


# ─── 진행 상태 → 말풍선·갈 곳 ──────────────────────────────────────────


def test_agent_state_covers_working_waiting_done_blocked_and_out():
    run = new_run("r1", "질문", "lead")
    assert office.agent_state("lead", run, None, {}) == {"tone": "working", "bubble": WORKING_LABELS["brief"], "kind": "brief"}
    apply_event(run, "delegated", {"node": "lead", "children": ["a"]})
    assert office.agent_state("lead", run, None, {})["bubble"] == "⏳ 하위 결과 대기"
    apply_event(run, "step", step("a"))
    assert office.agent_state("a", run, None, {}) == {"tone": "done", "bubble": "📝 초안 완료", "kind": ""}
    assert office.agent_state("z", run, {"lead", "a"}, {})["tone"] == "out"
    assert office.agent_state("b", None, None, {"b": "x — 꺼짐"})["bubble"] == "⛔ 고른 모델 꺼짐"
    assert office.agent_state("b", None, None, {}) == {"tone": "idle", "bubble": "대기", "kind": ""}
    run["error"], run["done"] = "실패", True
    run["activity"].clear()
    assert office.agent_state("lead", run, None, {})["tone"] == "error"


def test_snapshot_sends_agents_to_meeting_table_library_or_desk(org):
    layout = office.build_layout(org)
    run = new_run("r1", "질문", "lead")
    apply_event(run, "start", {"node": "lead", "kind": "plan"})
    apply_event(run, "start", {"node": "gm_research", "kind": "research"})
    apply_event(run, "start", {"node": "staff_draft", "kind": "draft"})
    snap = office.snapshot(layout, org, run, active_ids=None, blocked={}, focus="staff_draft", **ctx_args(org))
    agents = {a["id"]: a for a in snap["agents"]}
    meeting = [s[:2] for s in layout["spots"]["meeting"]]
    library = [s[:2] for s in layout["spots"]["library"]]
    assert agents["lead"]["place"] == "meeting" and agents["lead"]["target"] in meeting
    assert agents["gm_research"]["place"] == "library" and agents["gm_research"]["target"] in library
    assert agents["staff_draft"]["target"] == list(layout["seats"]["staff_draft"])
    assert agents["staff_draft"]["bubble"] == "✍️ 작성 중" and agents["staff_draft"]["tone"] == "working"
    assert agents["am_fact"]["bubble"] == "💤 쉬는 중"
    assert snap["running"] and snap["focus"] == "staff_draft" and snap["stage"]
    targets = [tuple(a["target"]) for a in snap["agents"]]
    assert len(set(targets)) == len(targets)  # 같은 곳에 겹쳐 서지 않는다


def test_envelopes_follow_delegation_and_reports(org):
    parents = parent_map(org)
    assert office.envelope_events(None, parents) == []
    run = new_run("abcdef123", "질문", "lead")
    run["steps"] = [
        step("lead", "plan", delegation=make_delegation("gm_research", "gm_action")),
        step("staff_draft"),
        step("lead", "integrate", review=make_review("approve", 9)),
    ]
    events = office.envelope_events(run, parents)
    pairs = [(e["from"], e["to"], e["kind"]) for e in events]
    assert pairs == [
        (office.USER, "lead", "ask"),
        ("lead", "gm_research", "delegate"), ("lead", "gm_action", "delegate"),
        ("staff_draft", "dgm_research", "report"),
        ("lead", office.USER, "report"),  # 맨 위의 최종 결과는 사용자(출입문)로
    ]
    assert all(e["id"].startswith("abcdef12:") for e in events)  # 다음 업무와 id 가 겹치지 않게
    assert office.envelope_events(run, parents) == events  # 늘 같은 id → 화면은 새 쪽지만 날린다
    assert len(office.envelope_events(run, parents, limit=2)) == 2


# ─── '다들 뭐 해?' ─────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["다들 뭐 해?", "다들 뭐해", "/status", "진행 상황 알려줘", "What are you up to?"])
def test_status_questions_are_recognized(text):
    assert office.is_status_question(text)


@pytest.mark.parametrize("text", ["HBF 가 뭐야? 표로 정리해줘", "서버 상태 점검 체크리스트를 ISMS-P 기준으로 표로 만들어 줘", ""])
def test_tasks_are_not_status_questions(text):
    assert not office.is_status_question(text)


def test_rundown_is_one_line_per_agent_with_next_step(org):
    args = ctx_args(org)
    kw = dict(names=args["names"], parents=args["parents"], active_ids=None, blocked={})
    idle = office.rundown(None, org, **kw)
    assert "쉬는 중" in idle[0] and len(idle) == 1 + len(list(org.walk()))

    run = new_run("r1", "질문", "lead")
    apply_event(run, "delegated", {"node": "lead", "children": ["gm_research", "gm_action"]})
    apply_event(run, "start", {"node": "staff_draft", "kind": "draft"})
    lines = office.rundown(run, org, **kw)
    assert lines[0].startswith("팀원 8명에게 물어봤어요") and "일하는 중 1" in lines[0]
    by_name = {line.split(" — ")[0][2:]: line for line in lines[1:]}
    assert "✍️ 작성 중" in by_name["사원·초안"] and "차장·분석에게 보고" in by_name["사원·초안"]
    assert "팀원 결과 기다리는 중" in by_name["팀장"] and "검토·통합" in by_name["팀장"]
    assert "차례 기다리는 중" in by_name["부장·리서치"]

    run["status"], run["done"] = "awaiting_human", True
    run["activity"].clear()
    run["statuses"]["lead"] = ("wait", "👤 결정 대기")
    assert "사용자 결정 대기" in office.rundown(run, org, **kw)[1]
    blocked = office.rundown(None, org, **{**kw, "blocked": {"am_fact": "x — 서버 꺼짐"}})
    assert any(line.startswith("⛔ 대리·팩트체크") for line in blocked)


# ─── 화면 연기 테스트 ──────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    set_keys()  # API 키 없음 → 모델을 부를 수 없다 (불렀다면 업무가 시작되어 오류가 남는다)
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))

    def make(team_id: str) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=60)
        at.query_params["team"] = team_id
        at.query_params["view"] = "team"
        at.session_state["office_view"] = "office"
        return at

    return make


@pytest.mark.parametrize("team_id", [t.id for t in list_teams()])
def test_office_view_renders_for_each_team(app, team_id):
    at = app(team_id).run()
    assert not at.exception, at.exception
    assert at.radio(key="office_view").value == "office"
    assert any("COMMAND CENTER" in m.value for m in at.markdown)
    assert at.text_input(key=f"cc_text_{team_id}") is not None


def test_terminal_answers_status_question_without_a_model(app, tmp_path):
    at = app("general").run()
    at.text_input(key="cc_text_general").input("다들 뭐 해?")
    at = next(b for b in at.button if b.label == "보내기").click().run()
    assert not at.exception, at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "다들 뭐 해?" in text and "팀원 8명 모두 쉬는 중" in text and "모델 호출 없이" in text
    assert Storage(tmp_path / "data" / "runs.db").list_runs() == []  # 업무를 시작하지 않았다


def test_terminal_queues_other_lines_as_a_task(monkeypatch, org):
    import command_center_ui

    state: dict = {}
    monkeypatch.setattr(command_center_ui.st, "session_state", state)  # 화면 없이 세션 상태만 흉내
    ctx = SimpleNamespace(team_id="general", org=org, names={n.id: n.id for n in org.walk()}, parents={}, blocked={})
    assert not command_center_ui.answer(ctx, None, "lead", None, "다들 뭐 해?")  # 상태 질문 → 업무 아님
    assert command_center_ui.pop_request("general", True) == {}
    assert not command_center_ui.answer(ctx, {"done": False, "status": "reviewing"}, "lead", None, "또 하나")  # 진행 중
    waiting = {"done": True, "status": "awaiting_human", "error": None}
    assert not command_center_ui.answer(ctx, waiting, "lead", None, "새 일")  # 결정 대기 결과를 지우지 않게
    assert command_center_ui.answer(ctx, None, "lead", None, "  HBF 가  뭐야?  ")
    request = command_center_ui.pop_request("general", True)
    assert request["question"] == "HBF 가 뭐야?" and request["auto_brief"] and request["use_web"]
    assert command_center_ui.pop_request("general", True) == {}  # 한 번만 꺼내 간다
    assert [k for k, _ in state["cc_log_general"]].count("me") == 4
    command_center_ui.answer(ctx, None, "lead", None, "/clear")
    assert state["cc_log_general"] == []
