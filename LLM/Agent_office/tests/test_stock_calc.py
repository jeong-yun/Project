"""주식 업무 숫자 검산 테스트 — 가중 점수·결정 밴드, 분할 매수 평균가·손익비·손절폭, 물타기 평단·추가 매수 한도.

LLM 이 쓴 ```json 요약을 코드로 다시 계산해 틀리면 '정답 값'과 함께 다시 쓰게 하는지(하네스),
하위 실무자에게는 이 검산을 요구하지 않는지(final_only), 시세 툴이 검산용 값을 넘기는지 확인한다.
"""

import json

import pandas as pd
import pytest

from core import search, stock_calc
from core.config import load_team, team_org
from core.harness import REGISTRY
from core.tools import stock_data
from core.verification import build_critic_messages, critic_settings
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_delegation, make_review

NO_CRITIC = {"critic": {"enabled": False}}


def block(data: dict) -> str:
    return "본문\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


EXAMPLE_A = {"valuation": 4, "earnings": 5, "financial": 4, "growth": 5, "industry": 4, "momentum": 4, "risk": 4}
EXAMPLE_B = {"valuation": 2, "earnings": 1, "financial": 1, "growth": 2, "industry": 2, "momentum": 2, "risk": 1}


# ─── 판단: 7개 항목 가중 점수 → 결정 밴드 ─────────────────────────────


def test_composite_matches_prompt_anchor_examples():
    assert stock_calc.composite_score(EXAMPLE_A) == (4.35, [])
    assert stock_calc.composite_score(EXAMPLE_B) == (1.55, [])
    assert stock_calc.verdict_for(4.35, []) == "매수"
    assert stock_calc.verdict_for(3.74, []) == "보유" and stock_calc.verdict_for(2.50, []) == "보유"
    assert stock_calc.verdict_for(2.49, []) == "매도"


def test_missing_items_reweight_or_hold_off():
    two_missing = {**EXAMPLE_A, "momentum": None, "risk": None}
    composite, missing = stock_calc.composite_score(two_missing)
    assert missing == ["momentum", "risk"] and composite == pytest.approx((4 * 20 + 5 * 20 + 4 * 15 + 5 * 15 + 4 * 10) / 80, abs=0.01)
    three_missing = {**two_missing, "industry": None}
    assert stock_calc.verdict_for(*stock_calc.composite_score(three_missing)) == stock_calc.HOLD_OFF


def test_scorecard_check_reports_correct_value_on_wrong_math():
    ok = stock_calc.check_scorecard(block({"scorecard": EXAMPLE_A, "composite": 4.35, "verdict": "매수"}), {})
    assert ok.found and not ok.problems and ok.computed["verdict"] == "매수"

    wrong = stock_calc.check_scorecard(block({"scorecard": EXAMPLE_A, "composite": 4.1, "verdict": "보유"}), {})
    assert any("= 4.35" in p for p in wrong.problems)  # 정답 값을 알려 준다
    assert any("규칙상 '매수'" in p for p in wrong.problems)

    bad_score = stock_calc.check_scorecard(block({"scorecard": {**EXAMPLE_A, "risk": 7}, "composite": 4.35, "verdict": "매수"}), {})
    assert any("1~5 정수" in p for p in bad_score.problems)
    assert not stock_calc.check_scorecard("JSON 없음", {}).found


# ─── 매매 계획: 평균 매수가·손익비·보유 기간별 손절폭 ─────────────────────


def plan(**overrides) -> dict:
    base = {
        "horizon": "중기",
        "tranches": [{"price": 100000, "weight": 40}, {"price": 95000, "weight": 30}, {"price": 90000, "weight": 30}],
        "take_profit": [115000, 125000], "stop_loss": 87000, "risk_reward": 2.29,
    }
    return {"trade_plan": {**base, **overrides}}


def test_trade_plan_computes_average_and_risk_reward():
    res = stock_calc.check_trade_plan(block(plan()), {"current_price": 98000})
    assert not res.problems, res.problems
    assert res.computed["average"] == pytest.approx(95500)
    assert res.computed["risk_reward"] == pytest.approx(round(19500 / 8500, 2))  # (115000-95500)/(95500-87000)
    assert res.computed["stop_pct"] == pytest.approx(8500 / 95500 * 100)  # 중기 8~10% 안


def test_trade_plan_rejects_bad_numbers_with_computed_values():
    res = stock_calc.check_trade_plan(block(plan(risk_reward=3.5)), {})
    assert any("= 2.29" in p for p in res.problems)
    weights = stock_calc.check_trade_plan(block(plan(tranches=[{"price": 100, "weight": 50}] * 3, take_profit=[120, 130], stop_loss=91, risk_reward=2.22)), {})
    assert any("비중 합이 100%가 아니다" in p for p in weights.problems)
    stop_too_tight = stock_calc.check_trade_plan(block(plan(stop_loss=93500, risk_reward=9.75)), {})
    assert any("중기 손절폭" in p for p in stop_too_tight.problems)  # 95500 대비 2.1% → 중기 8~10% 기준 위반
    stop_above_entry = stock_calc.check_trade_plan(block(plan(stop_loss=92000)), {})
    assert any("가장 낮은 매수가" in p for p in stop_above_entry.problems)


# ─── 보유 종목: 물타기 평단·추가 매수 한도 ─────────────────────────────


def position(**overrides) -> dict:
    base = {"action": "추가매수", "add_orders": [{"price": 60000, "quantity": 10}], "new_average": 66667,
            "take_profit_half": 80000, "take_profit_full": 90000, "stop_loss": 55000}
    return {"position_plan": {**base, **overrides}}


def test_position_plan_checks_new_average_and_limit():
    opts = {"quantity": 20, "buy_price": 70000, "current_price": 62000, "max_add_amount": 700000}
    ok = stock_calc.check_position_plan(block(position()), opts)
    assert not ok.problems and ok.computed["new_average"] == pytest.approx(200000 * 10 / 30)

    wrong_avg = stock_calc.check_position_plan(block(position(new_average=65000)), opts)
    assert any("평균 단가가 틀렸다" in p and "66,667" in p for p in wrong_avg.problems)

    over_limit = stock_calc.check_position_plan(block(position(add_orders=[{"price": 60000, "quantity": 20}], new_average=65000)), opts)
    assert any("허용 한도" in p for p in over_limit.problems)

    stop_above_price = stock_calc.check_position_plan(block(position(stop_loss=63000)), opts)
    assert any("현재가" in p for p in stop_above_price.problems)

    high_average_down = stock_calc.check_position_plan(block(position(add_orders=[{"price": 66000, "quantity": 5}], new_average=68800)), opts)
    assert any("물타기 가격" in w for w in high_average_down.warnings)


# ─── 추천: Stage 1 스키마 ─────────────────────────────────────────────


def test_recommend_requires_analysis_before_score_and_count():
    cand = {"ticker": "A", "extract_fundamentals": "사실", "analyze_risk": "리스크", "analyze_opportunity": "기회",
            "final_decision": {"score": 4, "one_line_rationale": "근거", "key_assumption": "전제"}}
    ok = stock_calc.check_recommend(block({"candidates": [cand, {**cand, "ticker": "B"}]}), {"expected_count": "2"})
    assert not ok.problems
    bad = stock_calc.check_recommend(
        block({"candidates": [{**cand, "analyze_risk": "", "final_decision": {**cand["final_decision"], "score": 6}}]}),
        {"expected_count": "2"},
    )
    assert any("2개인데" in p for p in bad.problems)
    assert any("analyze_risk" in p for p in bad.problems) and any("1~5 정수" in p for p in bad.problems)


# ─── 하네스 검증기 · 화면 검산표 ──────────────────────────────────────


def test_stock_plan_validator_and_report():
    v = REGISTRY["stock_plan"]
    assert v.check("아무거나", {}).passed  # 검산 대상이 없는 업무
    answer = block({"scorecard": EXAMPLE_A, "composite": 4.2, "verdict": "매수"})
    failed = v.check(answer, {"stock_checks": ["scorecard"]})
    assert not failed.passed and "4.35" in failed.log
    text = stock_calc.report(answer, {"stock_checks": ["scorecard"]})
    assert "결론: 매수" in text and "① 밸류에이션" in text and "❌" in text


# ─── 시세 툴: 재무 지표·컨센서스 극단값·추가 매수 한도 → 하네스 옵션 ──────────


@pytest.fixture
def fake_market(monkeypatch):
    idx = pd.date_range("2025-10-20", periods=250, freq="B")
    data = pd.DataFrame({"Close": [100.0] * 249 + [80.0], "Volume": [1000] * 250}, index=idx)
    info = {"currency": "KRW", "longName": "테스트전자", "exchange": "KSC", "sector": "Technology",
            "returnOnEquity": 0.18, "debtToEquity": 38.0, "payoutRatio": 0.25, "trailingEps": 5.0,
            "targetMeanPrice": 200.0, "targetHighPrice": 300.0, "targetLowPrice": 100.0, "numberOfAnalystOpinions": 10}
    monkeypatch.setattr(stock_data, "fetch_history", lambda t: data)
    monkeypatch.setattr(stock_data, "fetch_info", lambda t: info)


def test_stock_data_adds_fundamentals_consensus_and_limits(models, fake_market):
    result = stock_data.run("", {"stock": "005930", "quantity": 10, "buy_price": 100, "weight": 20, "max_weight": 30}, models)
    text = result.content
    assert "[종목 식별] 테스트전자 / 005930 / KOSPI / 업종 Technology" in text
    assert "| ROE | 18.0% |" in text and "| 부채비율(부채/자본) | 38.0% |" in text and "| 배당성향 | 25.0% |" in text
    assert "⚠️ 극단값 주의" in text  # 평균 목표가 200 vs 현재가 80 (+150%)
    assert "⚠️ 목표가 편차 큼" in text  # 300/100 = 3배
    # 평가금액 800 / 비중 20% → 예산 4000, 최대 30% = 1200 → 추가 400
    assert "추가 매수 가능 금액 400 KRW" in text and "물타기 — 손실 중" in text
    assert result.harness_options == {"current_price": 80.0, "quantity": 10, "buy_price": 100, "max_add_amount": 400.0}


# ─── 웹 검색: 블로그·커뮤니티 제외 ──────────────────────────────────────


def test_excluded_domains_and_paths():
    rules = ["blog.naver.com", "tistory.com", "finance.naver.com/item/board"]
    assert search.excluded("https://blog.naver.com/abc/123", rules)
    assert search.excluded("https://someone.tistory.com/1", rules)
    assert search.excluded("https://finance.naver.com/item/board.naver?code=005930", rules)
    assert not search.excluded("https://finance.naver.com/item/main.naver?code=005930", rules)
    assert not search.excluded("https://dart.fss.or.kr/x", rules)


# ─── 팀별 비평 설정 · 최종 답변 전용 하네스 ─────────────────────────────


def test_team_critic_override_and_rules():
    stock = load_team("stock")
    cfg = critic_settings({"critic": {"enabled": True, "scope": "final", "max_self_corrections": 1}}, stock.verification)
    assert cfg["scope"] == "all" and cfg["max_self_corrections"] == 2 and cfg["enabled"]
    system = build_critic_messages("q", "x", make_review(), [], cfg["rules"])[0].content
    assert "점수-분석 불일치" in system and "근거 없는 수치" in system


def test_final_only_validators_skip_subordinate_drafts(ranks, models, use_script):
    stock = load_team("stock")
    final = "종목 식별 … 종합 점수 … 신뢰도 상\n" + block({"scorecard": EXAMPLE_A, "composite": 4.35, "verdict": "매수"})
    use_script(Script(text="뉴스 정리 초안", ReviewResult=make_review("revise", 8, answer=final),
                      Delegation=make_delegation("dgm_scenario", "staff_portfolio")))
    org = team_org(stock, ranks)
    graph = build_office_graph(org, ranks, stock, models, {}, NO_CRITIC)
    events = list(stream_office(graph, org, "판단해줘", "gm_risk", auto_brief=False,
                                extra_validators=["stock_plan"], harness_options={"stock_checks": ["scorecard"],
                                                                                  "required_terms": ["종목 식별"]}))
    steps = [p for k, p in events if k == "step"]
    drafts = [s for s in steps if s.kind == "draft"]
    assert drafts and all(not d.harness for d in drafts)  # 하위 초안은 전체 형식 검사를 받지 않는다
    root = steps[-1]
    assert root.node_id == "gm_risk" and {h.validator for h in root.harness} == {"required_terms", "stock_plan"}
    assert all(h.passed for h in root.harness)
