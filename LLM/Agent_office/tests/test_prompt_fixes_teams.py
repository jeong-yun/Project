"""프롬프트 점검(2026-11-10)의 남은 중간 항목 — 리서치·지식부(R3~R9) · 투자·여행부(T4~T12). 가짜 모델만.

글만 바꾼 항목은 실제 teams/ 파일로 조립한 프롬프트에 새 문장이 들어가고 어긋나던 문장이 없어졌는지 본다.
코드가 바뀐 항목(T7 요약 블록 고르기 · T11 종목 찾기)은 동작을 본다. 실제 모델 결과가 좋아지는지는 여기서 확인하지 못한다.
"""

import json

import pytest

from core import harness, stock_calc
from core.agent import OrgAgent
from core.config import load_skills, load_team, team_org
from core.tools import stock_data
from tests.fakes import Script, make_delegation
from tests.test_teams import fake_market, fixed_today  # noqa: F401 — 가짜 시세·고정 날짜(2026-10-19) fixture 재사용


def agents_of(team_id: str, ranks, models) -> dict[str, OrgAgent]:
    """그 팀의 실제 설정·스킬로 만든 에이전트들 (id → OrgAgent)."""
    team, by_rank = load_team(team_id), {r.id: r for r in ranks}
    nodes = list(team_org(team, ranks).walk())
    names = {n.id: n.name or n.id for n in nodes}
    return {n.id: OrgAgent(n, by_rank[n.rank], team, models, names=names, skills=load_skills(team_id)) for n in nodes}


def task_of(team_id: str, task_id: str):
    return next(t for t in load_team(team_id).tasks if t.id == task_id)


def skill_line(agent: OrgAgent, start: str) -> str:
    """스킬에서 start 로 시작하는 줄 (앞 공백 무시)."""
    return next(line.strip() for line in agent.skills.splitlines() if line.strip().startswith(start))


# ─── R4·R5 기본 팀 ─────────────────────────────────────────────────────


def test_general_personas_do_not_fight_the_integration_rule(ranks, models):
    agents = agents_of("general", ranks, models)
    action, research, staff = (agents[i].system_prompt() for i in ("gm_action", "gm_research", "staff_draft"))
    assert "핵심만 남깁니다" not in action and "확인된 사실은 빼지 않고" in action  # 통합 지시 '사실을 근거 없이 버리지 않는다'와 같은 방향
    assert "실행을 책임지는" not in research and "사실관계" in research.splitlines()[0]  # 부장·리서치는 전문 분야에 맞는 페르소나
    assert "빠르게 초안" not in staff and "맡은 범위의 초안을 근거와 함께" in staff


def test_general_answer_style_follows_brief_format_and_has_one_word_for_unknowns(ranks, models):
    agent = agents_of("general", ranks, models)["staff_case"]
    order, cite = skill_line(agent, "- 업무 브리프에 [결과 형식]"), skill_line(agent, "- [근거 자료]가 주어졌으면")
    assert "그 순서를 따른다" in order and "없을 때만" in order  # '결론을 맨 앞에'는 결과 형식이 없을 때만
    assert "[n]" in cite and "'확인 필요'" in cite and "고유명사" not in cite  # 근거 자료가 없는 업무(코드·이메일)에 출처 번호를 요구하지 않는다
    assert "추측" in cite  # 공통 규칙의 '모른다고 밝힌다'·'추측임을 밝힌다'도 이 말로


# ─── R3·R6~R9 개념 편집 팀 ───────────────────────────────────────────────


def test_concept_editor_always_delegates_to_both_members(ranks, models, use_script):
    script = use_script(Script(Delegation=make_delegation("am_fact", "dgm_lineage")))
    agents_of("concept", ranks, models)["gm_editor"].plan("질문", "질문")
    prompt = script.prompts("Delegation")[0]
    assert "항상 둘 다 맡긴다" in prompt and "대리·팩트체크" in prompt and "차장·계보분석" in prompt


def test_concept_citation_numbers_stay_out_of_the_final(ranks, models):
    agents = agents_of("concept", ranks, models)
    for who in ("lead", "gm_editor", "am_fact"):  # 최종본·참고 문헌 후보를 쓰는 사람
        line = skill_line(agents[who], "- 웹 검색 자료의 [번호]")
        assert "<review_log>" in line and "최종본" in line and "참고 문헌" in line, who
    prompt = agents["gm_editor"].system_prompt()
    assert "- markdown 형식으로 작성한다." not in prompt  # '굵게(**) 쓰지 않음'과 부딪히던 줄
    assert "[출력 형식 견본]" in next(line for line in prompt.splitlines() if line.endswith("형식으로 작성한다."))


def test_concept_critic_rules_judge_only_what_the_head_can_see():
    rules = load_team("concept").verification["critic"]["rules"]
    text = "\n".join(rules)
    assert "원전 확인 없이" not in text and "존재 확인이 안 된" not in text  # 부장은 원전을 볼 수 없다
    assert sum("판정 표" in r for r in rules) >= 2 and "출처 미확인" in text
    assert "생략" in text  # 검증 로그가 길어 줄었으면 이 규칙으로 지적하지 않는다
    assert "초안 구조 전제 수용" in text and "이중 해리 미검사" in text and "판정 불일치" in text  # 위반 이름은 그대로


def test_concept_stage1_matches_what_the_model_actually_receives(ranks, models):
    lineage = agents_of("concept", ranks, models)["dgm_lineage"]  # 1단계 담당
    listing, primary, mismatch = (skill_line(lineage, s) for s in ("(1)", "· 판정 근거는", "(4)"))
    assert "3~6개" not in listing and "[입력]" in listing and "비교 대상 없음" in listing  # 사람이 고른 0~6개만 받는다
    assert "따로 열어" not in primary and "[근거 자료]" in primary and "원전 미확인" in primary  # 문서를 여는 도구가 없다
    assert "같이 쓰이는 사례만 보면" in primary  # 뒤 문장은 그대로
    assert "결론을 표에 맞춰 수정" not in mismatch and "(2)(3)" in mismatch and "다시 판정" in mismatch


# ─── T5~T11 주식 팀 ────────────────────────────────────────────────────


def test_stock_skill_says_what_to_do_without_market_data(ranks, models):
    line = skill_line(agents_of("stock", ranks, models)["am_data"], "- [근거 자료]에 시장 데이터가 없")
    assert "자유 질문" in line and "'확인 불가'" in line and "기억하는" in line


def test_stock_sources_use_one_format_and_one_meaning_of_base_date(ranks, models, fake_market):
    agent = agents_of("stock", ranks, models)["am_data"]
    numbers, news = skill_line(agent, "- 숫자("), skill_line(agent, "- 뉴스·리포트는")
    assert "[출처, YYYY-MM-DD]" not in agent.skills and "(시장 데이터, 기준일)" in numbers  # 공통 초안 지시와 같은 형식
    assert "마지막 거래일" in numbers and "조회일" in numbers
    assert "날짜 미상" in news and "최근 1개월" in news  # 일반 검색 결과에는 날짜가 없다
    judge = task_of("stock", "judge")
    assert "[출처, YYYY-MM-DD]" not in "\n".join([*judge.success_criteria, *judge.constraints])
    assert "조회 시점" not in judge.instructions and "기준일(오늘)" not in "\n".join(task_of("stock", "recommend").constraints)
    content = stock_data.run("", {"stock": "005930"}, models).content
    assert "[Yahoo Finance, 2026-10-19]" not in content  # 조회일(오늘)을 기준일로 붙이게 하던 예시
    assert "(시장 데이터, 기준일)" in content and "마지막 거래일" in content


def test_stock_summary_block_is_written_once_by_whoever_writes_the_final_answer(ranks, models):
    agent = agents_of("stock", ranks, models)["staff_portfolio"]
    line = skill_line(agent, "- 업무 지침이 ```json 요약 블록을")
    assert "최종 답변을 쓰는" in line and "블록 없이" in line and "하나로" in line


def trade_block(stop: int, rr: float) -> str:
    plan = {"horizon": "중기", "tranches": [{"price": 100000, "weight": 40}, {"price": 95000, "weight": 30}, {"price": 90000, "weight": 30}],
            "take_profit": [115000, 125000], "stop_loss": stop, "risk_reward": rr}
    return "```json\n" + json.dumps({"trade_plan": plan}) + "\n```"


def test_stock_check_reads_the_last_summary_block():
    # 통합본에 하위 결과의 블록이 섞여도, 지침대로 '맨 끝'에 쓴 블록을 검산한다
    answer = "하위 결과 인용\n" + trade_block(99000, 9.99) + "\n최종 계획\n" + trade_block(87000, 2.29)
    res = stock_calc.check_trade_plan(answer, {})
    assert res.found and not res.problems and res.computed["stop_loss"] == 87000


def test_trade_task_stop_loss_is_based_on_average_price_like_the_check():
    trade = task_of("stock", "trade")
    assert "손절 참고가'를 참고한다" not in trade.instructions
    assert "평균 매수가 × (1 − 손절폭)" in trade.instructions and "현재가 기준" in trade.instructions
    assert "매물대" not in trade.instructions  # 시장 데이터에 없는 값
    assert "중기(6~12개월)" in "\n".join(trade.success_criteria) and "3~6개월" not in "\n".join(trade.success_criteria)  # 다른 업무와 같은 뜻
    example = "```json\n" + next(line for line in trade.instructions.splitlines() if line.startswith('{"trade_plan"')) + "\n```"
    assert not stock_calc.check_trade_plan(example, {}).problems  # 지침의 예시가 검산을 통과한다


def test_position_task_wording_matches_the_check(models, fake_market):
    position = task_of("stock", "position")
    criteria = "\n".join(position.success_criteria)
    assert "50% 익절가와 100% 익절가" not in criteria and "절반" in criteria and "나머지" in criteria
    assert "보수적 한도" not in criteria and "비율로만" in criteria
    assert "전량매도" in position.instructions and "계속 보유한다면" in position.instructions  # 검산은 익절가·손절가가 양수여야 통과
    example = json.loads(next(line for line in position.instructions.splitlines() if line.startswith('{"position_plan"')))
    assert example["position_plan"]["action"] == "추가매수" and example["position_plan"]["add_orders"]  # 추가 매수 주문이 있는 예시
    block = "```json\n" + json.dumps(example) + "\n```"
    assert not stock_calc.check_position_plan(block, {"quantity": 20, "buy_price": 70000}).problems
    content = stock_data.run("", {"stock": "005930", "quantity": 10.0, "buy_price": 300.0}, models).content
    assert "보수적" not in content and "비율로만" in content  # 비중을 모를 때 — 금액 한도를 지어내지 않게


def test_stock_definite_wording_has_a_boundary(ranks, models):
    team = load_team("stock")
    line = skill_line(agents_of("stock", ranks, models)["lead"], "- '확정적 표현'은")
    assert "조건 없이" in line and "매수·보유·매도 결론" in line and "틀릴 수 있는" in line
    assert any("확정적 표현" in c and "조건 없" in c for c in team.review_criteria)  # 부장 비평도 같은 경계를 받는다
    assert "투자 점수" in team.verification["critic"]["rules"][0]  # 비평 프롬프트의 '검토자 점수(1~10)'와 섞이지 않게
    assert "[면책]" not in task_of("stock", "judge").output_format  # 화면이 붙이는 면책 문구와 두 번 나오지 않게


def test_ticker_lookup_does_not_guess_and_shows_what_was_found(models, fake_market, use_script):
    script = use_script(Script(TickerResolution=stock_data.TickerResolution(items=[
        stock_data.TickerItem(query="에코프로비엠", name="에코프로비엠", ticker="247540.KQ", quantity=0, buy_price=0),
        stock_data.TickerItem(query="모르는회사", name="모르는회사", ticker="", quantity=0, buy_price=0),
    ])))
    result = stock_data.run("", {"holdings": "에코프로비엠 5주, 모르는회사 1주"}, models)
    prompt = script.prompts("TickerResolution")[0]
    assert "지어내지" in prompt and "빈 문자열" in prompt
    assert "입력 '에코프로비엠' → 조회된 종목 회사 247540.KQ (247540.KQ)" in result.content  # 다른 회사가 조회됐는지 알아볼 수 있게
    assert any("모르는회사" in n and "종목 코드" in n for n in result.notes)  # 코드를 모르는 종목은 조회하지 않고 알린다
