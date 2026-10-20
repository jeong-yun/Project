"""주식·여행 팀 테스트 — 업무 양식, 데이터 툴, 새 검증기, 워크플로우 연결 (API·네트워크 없이)."""

import pandas as pd
import pytest

from core.config import ConfigError, list_teams, load_team, team_org, validate_config
from core.harness import REGISTRY
from core.schemas import Team
from core.tasks import TaskInputError, build_request, render
from core.tools import run_tools, stock_data, travel_calendar
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_delegation, make_review

NO_CRITIC = {"critic": {"enabled": False}}


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("AGENT_OFFICE_TODAY", "2026-10-19")


@pytest.fixture
def stock():
    return load_team("stock")


@pytest.fixture
def travel():
    return load_team("travel")


@pytest.fixture
def fake_market(monkeypatch):
    """가짜 시세: 100 → 349 로 매일 1씩 오르는 250 거래일. 알 수 없는 티커는 빈 데이터."""
    idx = pd.date_range("2025-10-20", periods=250, freq="B")
    data = pd.DataFrame({"Close": [100.0 + i for i in range(250)], "Volume": [1000] * 250}, index=idx)
    known = {"005930.KS", "247540.KQ"}
    monkeypatch.setattr(stock_data, "fetch_history", lambda t: data if t in known else data.iloc[0:0])
    monkeypatch.setattr(stock_data, "fetch_info", lambda t: {"currency": "KRW", "longName": f"회사 {t}"})


# ─── 팀 설정 ─────────────────────────────────────────────────────────


def test_new_teams_are_valid(ranks, models):
    teams = {t.id: t for t in list_teams()}
    assert {"stock", "travel"} <= teams.keys()
    validate_config(ranks, models, list(teams.values()))
    assert [t.id for t in teams["stock"].tasks] == ["recommend", "judge", "forecast", "trade", "position", "portfolio"]
    assert "travel_calendar" in teams["travel"].tools and "stock_data" in teams["stock"].tools
    assert len(team_org(teams["stock"], ranks).walk()) == 7


def test_unknown_tool_is_rejected(ranks, models, stock):
    bad = Team(**{**stock.model_dump(), "tools": ["nope"]})
    with pytest.raises(ConfigError, match="툴"):
        validate_config(ranks, models, [bad])


# ─── 업무 양식 → 질문·브리프 ───────────────────────────────────────────


def test_render_drops_lines_with_empty_values():
    text = render("목적지: {dest}\n출발지: {dep}", {"dest": "", "dep": "서울"})
    assert text == "출발지: 서울"
    assert render("{qty}주", {"qty": 10.0}) == "10주"  # 10.0 → 10
    assert render("{price}원", {"price": 70000}) == "70,000원"


def task_of(team: Team, task_id: str):
    return next(t for t in team.tasks if t.id == task_id)


def test_position_task_builds_full_brief(stock):
    req = build_request(task_of(stock, "position"), {"stock": "삼성전자", "quantity": 10.0, "buy_price": 70000.0,
                                                     "horizon": "중기(6~12개월)", "risk": "중립형"}, extra_request="배당도 고려")
    assert req.question.startswith("삼성전자 종목을 10주, 평균 단가 70,000에 보유 중이야")
    assert "추가 요청: 배당도 고려" in req.question
    assert "투자 목적" not in req.question  # 비운 선택 입력 줄은 빠짐
    assert req.brief.goal and len(req.brief.success_criteria) == 6  # 브리프가 완성 → 요청 정리 LLM 불필요
    assert "[업무 수행 지침]" in req.brief.context and "position_plan" in req.brief.context
    assert req.tool_inputs["quantity"] == 10.0
    assert req.harness_options["stock_checks"] == ["position_plan"] and "stock_plan" in req.validators


def test_task_harness_options_are_rendered(stock):
    req = build_request(task_of(stock, "recommend"), {"sector": "AI 반도체", "market": "국내", "count": 4})
    assert req.harness_options["expected_count"] == "4"
    assert '"sector": "AI 반도체"' in req.brief.context  # 지침 속 {sector} 도 치환
    assert "후보 종목" not in req.question


def test_required_fields_and_dates_are_checked(stock, travel):
    with pytest.raises(TaskInputError, match="종목"):
        build_request(task_of(stock, "judge"), {"stock": ""})
    with pytest.raises(TaskInputError, match="도착일"):
        build_request(travel.tasks[0], {"scope": "국내", "start_date": "2026-11-23", "end_date": "2026-11-20"})


def test_travel_task_derives_days_for_harness(travel):
    req = build_request(travel.tasks[0], {"scope": "해외", "start_date": "2026-11-20", "end_date": "2026-11-23"})
    assert "3박 4일" in req.question and "목적지" not in req.question  # 비운 목적지 줄은 빠짐
    assert req.harness_options["trip_days"] == 4 and "itinerary_days" in req.validators


# ─── 데이터 툴 ─────────────────────────────────────────────────────────


def test_travel_calendar_lists_every_day_with_weekday(models):
    result = travel_calendar.run("", {"start_date": "2026-11-20", "end_date": "2026-11-23", "scope": "해외"}, models)
    assert "3박 4일" in result.content and "D-32" in result.content
    assert "| 1일차 | 2026-11-20 | 금 | 출발일 |" in result.content
    assert "| 4일차 | 2026-11-23 | 월 | 귀가일 |" in result.content
    assert result.harness_options == {"trip_days": 4, "trip_start": "2026-11-20"}
    assert travel_calendar.run("", {}, models) is None


def test_stock_data_by_code_computes_returns_and_pnl(models, fake_market):
    result = stock_data.run("", {"stock": "005930", "quantity": 10.0, "buy_price": 300.0}, models)
    assert "005930.KS" in result.content and "349 KRW" in result.content  # 마지막 종가
    assert "+16.3%" in result.content  # 평가손익 (349-300)/300 — 코드 계산
    assert "평가손익 490 KRW" in result.content


def test_stock_data_resolves_name_with_llm_and_falls_back_to_kosdaq(models, fake_market, use_script):
    # LLM 이 .KS 로 잘못 추정해도 .KQ 로 재조회해 찾는다
    use_script(Script(TickerResolution=stock_data.TickerResolution(items=[
        stock_data.TickerItem(query="에코프로비엠", name="에코프로비엠", ticker="247540.KS", quantity=0, buy_price=0),
        stock_data.TickerItem(query="없는회사", name="없는회사", ticker="999999.KS", quantity=0, buy_price=0),
    ])))
    result = stock_data.run("", {"holdings": "에코프로비엠 5주, 없는회사 1주"}, models)
    assert "247540.KQ" in result.content
    assert any("없는회사" in n for n in result.notes)


def test_stock_data_skips_when_no_stock(models):
    assert stock_data.run("", {"sector": "AI 반도체"}, models) is None


def test_run_tools_isolates_failures(models, monkeypatch):
    def boom(q, inputs, m):
        raise RuntimeError("yahoo down")

    monkeypatch.setitem(__import__("core.tools", fromlist=["TOOLS"]).TOOLS, "stock_data", boom)
    results = run_tools(["stock_data", "travel_calendar"], "", {"start_date": "2026-11-20", "end_date": "2026-11-21"}, models)
    assert "yahoo down" in results[0].notes[0] and results[1].tool == "travel_calendar"


# ─── 새 검증기 ─────────────────────────────────────────────────────────


def test_required_terms_supports_alternatives():
    v = REGISTRY["required_terms"]
    assert v.check("교통은 지하철, 예산 50만원", {"required_terms": ["예산", "교통|이동"]}).passed
    failed = v.check("예산만 있음", {"required_terms": ["예산", "숙소"]})
    assert not failed.passed and "숙소" in failed.log


def test_itinerary_days_accepts_labels_or_dates():
    v = REGISTRY["itinerary_days"]
    opts = {"trip_days": 3, "trip_start": "2026-11-20"}
    assert v.check("1일차 …\n11월 21일 …\nDay 3 …", opts).passed
    failed = v.check("1일차 …\n2일차 …", opts)
    assert not failed.passed and "3일차" in failed.log
    assert v.check("아무거나", {}).passed  # 일정 정보가 없으면 검사 생략


# ─── 워크플로우 연결 ───────────────────────────────────────────────────


def test_tool_results_reach_prompts_and_harness(ranks, models, travel, use_script):
    plan_text = "1일차 출발\n2일차 관광\n3일차 귀가\n예산 표, 숙소, 교통"
    script = use_script(Script(text=plan_text, ReviewResult=make_review("approve", 9),
                               Delegation=make_delegation("am_spots", "staff_food")))
    org = team_org(travel, ranks)
    req = build_request(travel.tasks[0], {"scope": "국내", "start_date": "2026-11-20", "end_date": "2026-11-22"})
    graph = build_office_graph(org, ranks, travel, models, {}, NO_CRITIC)
    events = list(stream_office(graph, org, req.question, "am_spots", brief=req.brief, auto_brief=False,
                                tool_inputs=req.tool_inputs, extra_validators=req.validators, harness_options=req.harness_options))

    tools = next(p for k, p in events if k == "tools")
    assert tools[0].tool == "travel_calendar"
    assert "| 3일차 | 2026-11-22 | 일 | 귀가일 · 주말 |" in script.prompts("text")[0]  # 달력이 실무자 근거 자료로
    assert "오늘 날짜: 2026-10-19 (월)" in script.prompts("text")[0]  # 현 시점 기준
    draft = next(p for k, p in events if k == "step")
    assert {h.validator for h in draft.harness} == {"itinerary_days", "required_terms"}
    assert all(h.passed for h in draft.harness)
