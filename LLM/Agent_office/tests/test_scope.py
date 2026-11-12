"""🧭 팀별 자유 질문 범위 (core/scope.py · team.yaml 의 scope · free_examples) — 화면(AppTest) 포함.

실제 모델 호출 없음: 부서장 배정은 가짜 함수 또는 Script, 키 없음이면 낱말 비교로 떨어진다.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import building, followup, scope
from core.agent import OrgAgent
from core.config import list_teams, load_models, load_ranks, load_team, team_org, validate_config
from core.followup import FollowupResult
from tests.fakes import Script

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TRAVEL_Q = "다음 달 오사카 3박 4일 가족 여행 계획 세워줘"


@pytest.fixture
def teams():
    return list_teams()


def by_id(teams, tid):
    return next(t for t in teams if t.id == tid)


def no_router(*args, **kwargs):
    raise AssertionError("낱말 확인으로 끝나야 하는데 부서장 배정(모델)을 불렀습니다")


def fake_router(team_id, how="llm", reason="가짜 부서장"):
    calls = []

    def route(text, dept, candidates, models):
        calls.append({"text": text, "dept": dept, "candidates": candidates})
        return {"team_id": team_id, "reason": reason, "how": how, "model": "fake" if how == "llm" else "", "note": ""}

    route.calls = calls
    return route


# ─── team.yaml ──────────────────────────────────────────────────────


def test_every_team_has_scope_and_own_examples(teams):
    validate_config(load_ranks(), load_models(), teams)
    for team in teams:
        assert team.scope and team.scope.summary and team.scope.accepts, team.id
        assert 2 <= len(team.free_examples) <= 4, team.id
        for ex in team.free_examples:
            assert ex.title and ex.question and ex.goal and ex.success_criteria and ex.output_format, (team.id, ex.title)
        if team.id == "general":
            assert team.scope.accept_all  # 범용 창구는 설정으로 분명히
        else:
            assert not team.scope.accept_all and team.scope.keywords and team.scope.rejects, team.id
            for ex in team.free_examples:  # 이 팀 예시는 모델 없이도 이 팀 질문으로 통과해야 한다
                decision = scope.check(ex.question, team, teams, None, route=no_router)
                assert decision["in_scope"] and decision["stage"] == "keyword", (team.id, ex.question)


def test_team_card_shows_scope(teams):
    ranks = load_ranks()
    stock = by_id(teams, "stock")
    assert building.team_card(stock, team_org(stock, ranks), ranks)["scope"].startswith("주식")


# ─── 범위 확인 ──────────────────────────────────────────────────────


def test_clear_keywords_pass_without_model(teams):
    d = scope.check("삼성전자 지금 사도 될까? 목표가도 알려줘", by_id(teams, "stock"), teams, None, route=no_router)
    assert d["in_scope"] and d["stage"] == "keyword" and "삼성전자" in d["reason"]
    # 영문 낱말은 다른 낱말 속에서 맞지 않는다 (AI ≠ MAIN)
    assert scope.keyword_hits("MAIN 화면", by_id(teams, "concept")) == []


def test_out_of_scope_is_blocked_with_suggested_team(teams):
    stock = by_id(teams, "stock")
    route = fake_router("travel_abroad", reason="오사카는 해외 여행")
    d = scope.check(TRAVEL_Q, stock, teams, None, route=route)
    assert d == {"in_scope": False, "team_id": "travel_abroad", "stage": "router", "reason": "오사카는 해외 여행",
                 "model": "fake", "note": ""}
    msg = scope.blocked_message(d, stock, teams)
    assert msg == "이 질문은 **주식 팀** 범위가 아닙니다 — **🌏 해외 여행팀**(여행부)이 맞습니다."
    # 부서장 배정은 모든 팀을 대상으로 부른다 (팀 소개에 '다루는 질문'을 붙이는 일은 building.route_messages 가 한다 — 아래 테스트)
    call = route.calls[0]
    assert {t.id for t in call["candidates"]} == {t.id for t in teams}
    # 부서장이 이 팀을 고르면 통과
    assert scope.check(TRAVEL_Q, stock, teams, None, route=fake_router("stock"))["in_scope"]


def test_real_dept_head_router_is_reused(teams, models, use_script):
    script = use_script(Script(TeamPick=building.TeamPick(analysis="여행", team_id="travel_abroad", reason="해외 여행 계획")))
    d = scope.check(TRAVEL_Q, by_id(teams, "stock"), teams, models)
    assert not d["in_scope"] and d["team_id"] == "travel_abroad" and d["stage"] == "router"
    prompt = script.prompts("TeamPick")[0]
    assert "부서장 회의" in prompt and "travel_abroad" in prompt and "다루지 않음" in prompt
    assert prompt.count("다루는 질문: 해외 여행") == 1


def test_without_model_blocks_only_on_clear_other_team(teams, models, set_keys):
    set_keys()  # 키 없음 → 부서장은 낱말 비교로 떨어진다
    stock = by_id(teams, "stock")
    d = scope.check(TRAVEL_Q, stock, teams, models)
    assert not d["in_scope"] and d["team_id"] == "travel_abroad" and d["stage"] == "router_keyword"
    # 어느 팀 낱말도 분명하지 않으면 막지 않는다 (모델 없이 확실하지 않은 판단으로 막지 않게)
    vague = scope.check("요즘 어떻게 생각해?", stock, teams, models)
    assert vague["in_scope"] and vague["stage"] == "router_keyword"


def test_general_accepts_everything(teams):
    general = by_id(teams, "general")
    for q in (TRAVEL_Q, "삼성전자 사도 돼?", "아무 질문"):
        d = scope.check(q, general, teams, None, route=no_router)
        assert d["in_scope"] and d["stage"] == "accept_all"


def test_short_followup_on_last_result_is_lenient(teams):
    stock = by_id(teams, "stock")
    last = "삼성전자 종목 판단\n결론: 보유. 핵심 논거 1) HBM 수요 2) 밸류에이션"
    d = scope.check("이거 표로 다시 정리해줘", stock, teams, None, last_text=last, route=no_router)
    assert d["in_scope"] and d["stage"] == "followup"
    # 다른 팀 낱말이 있으면 후속 표현이 있어도 봐 주지 않는다 → 부서장에게
    route = fake_router("travel_abroad")
    assert not scope.check("그럼 오사카 여행은?", stock, teams, None, last_text=last, route=route)["in_scope"]
    assert route.calls


def test_decisions_are_logged(tmp_path, teams):
    log = tmp_path / "scope_log.jsonl"
    stock = by_id(teams, "stock")
    scope.log_decision(log, stock, TRAVEL_Q, {"in_scope": False, "team_id": "travel_abroad", "stage": "router",
                                              "reason": "r", "model": "", "note": ""})
    rows = scope.read_log(log)
    assert rows[0]["team"] == "stock" and rows[0]["team_id"] == "travel_abroad" and rows[0]["in_scope"] is False


def test_followup_chat_is_not_restricted(teams, ranks, models, use_script, monkeypatch):
    """💬 결정 전 문의는 범위 확인을 거치지 않는다 (다른 분야 말이 섞여도 그대로 답한다)."""
    monkeypatch.setattr(scope, "check", no_router)
    stock = by_id(teams, "stock")
    org = team_org(stock, ranks)
    agent = OrgAgent(org, next(r for r in ranks if r.id == org.rank), stock, models)
    use_script(Script(FollowupResult=FollowupResult(analysis="질문", revised=False, reply="여행 경비는 이 결과와 무관합니다.")))
    turn = followup.respond(agent, "삼성전자 판단", "## 결론\n보유", "이번 달 오사카 여행 경비도 감안해야 할까?", [])
    assert turn.reply.startswith("여행 경비")


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


@pytest.fixture
def app_env(tmp_path, monkeypatch, set_keys):
    """키 없음(부서장 = 낱말 비교) · 임시 데이터 폴더 · 실행 시작은 기록만."""
    import streamlit as st

    from core import live

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    started: list[str] = []
    monkeypatch.setattr(live, "start_run", lambda run, *a, **k: started.append(run["question"]))
    st.cache_resource.clear()
    yield {"data": tmp_path / "data", "started": started}
    st.cache_resource.clear()


def open_team(team_id: str, free: bool = True) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["view"] = "team"
    at.query_params["team"] = team_id
    at.run()
    assert not at.exception, at.exception
    if free:  # 🏢 부서 업무 목록에서 이 팀의 '💬 자유 질문'을 직접 고른다
        at.selectbox(key=f"dept_task_{at.session_state['dept_pick']}").set_value(f"{team_id}:free").run()
        assert not at.exception, at.exception
    return at


def submit(at: AppTest, question: str | None = None) -> AppTest:
    if question is not None:
        at.text_area(key="q_text").input(question)
    return at.button[[b.label for b in at.button].index("맡기기")].click().run()


@pytest.mark.parametrize("team_id", ["stock", "general"])
def test_example_picker_shows_only_this_teams_examples(app_env, team_id):
    at = open_team(team_id)
    titles = [e.title for e in load_team(team_id).free_examples]
    picker = at.selectbox(key=f"example_pick_{team_id}")
    assert picker.options == ["(직접 입력)", *titles]
    assert any(c.value.startswith("🧭 이 팀이 다루는 질문:") for c in at.caption)
    at = picker.set_value(titles[0]).run()
    assert at.text_area(key="q_text").value == load_team(team_id).free_examples[0].question


def test_off_topic_question_is_blocked_and_button_goes_to_right_team(app_env):
    at = open_team("stock")
    at.text_area(key="q_text").input(TRAVEL_Q)
    at = submit(at)
    assert not at.exception, at.exception
    assert app_env["started"] == []  # 실행하지 않았다
    assert "주식 팀** 범위가 아닙니다 — **🌏 해외 여행팀**(여행부)이 맞습니다" in at.error[0].value
    assert scope.read_log(app_env["data"] / "scope_log.jsonl")[-1]["team_id"] == "travel_abroad"
    assert not any(b.label.startswith("그래도") for b in at.button)  # '그래도 실행'은 없다
    at = at.button(key="scope_go_stock").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "dept" and at.session_state["team_pick"] == "travel_abroad"
    assert at.session_state["dept_pick"] == "travel" and at.session_state["dept_task_travel"] == "travel_abroad:free"
    assert at.text_area(key="q_text").value == TRAVEL_Q


def test_edit_button_clears_block_and_keeps_question(app_env):
    at = submit(open_team("stock"), TRAVEL_Q)
    at = at.button(key="scope_edit_stock").click().run()
    assert not at.exception and not at.error
    assert at.text_area(key="q_text").value == TRAVEL_Q


def test_in_scope_question_runs(app_env):
    at = submit(open_team("stock"), "삼성전자 지금 사도 될까?")
    assert not at.exception, at.exception
    assert app_env["started"] == ["삼성전자 지금 사도 될까?"]
    assert scope.read_log(app_env["data"] / "scope_log.jsonl")[-1]["in_scope"] is True


def test_structured_task_form_is_not_scope_checked(app_env):
    at = open_team("stock", free=False)
    at = at.selectbox(key="dept_task_invest").set_value("stock:judge").run()
    at.text_input(key="tf_stock_judge_stock").input("삼성전자")
    at = submit(at)
    assert not at.exception, at.exception
    assert len(app_env["started"]) == 1
    assert not (app_env["data"] / "scope_log.jsonl").exists()
