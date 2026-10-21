"""웹 검색(자료 조사) 테스트 — 실제 네트워크 대신 conftest 의 모의 검색 엔진 사용."""

import pytest

from core import search
from core.agent import OrgAgent
from core.config import subtree
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_review
from tests.test_workflow import BRIEF, NO_CRITIC

SETTINGS = {"web_search": {"max_queries": 2, "max_total_results": 3}}
PLAN = search.SearchPlan(need_search=True, queries=["HBF High Bandwidth Flash", "HBF 메모리"])


def test_research_uses_planned_queries_and_dedupes(models, use_script, no_network_search):
    use_script(Script(SearchPlan=PLAN))
    ctx = search.research("HBF 알려줘", models, SETTINGS)
    assert sorted(no_network_search) == sorted(["HBF High Bandwidth Flash", "HBF 메모리"])  # 동시에 검색 → 순서 무관
    assert len(ctx.results) == 3  # 4개 중 중복 1개 제거 → 3개 (max_total_results=3)
    assert "[1] HBF High Bandwidth Flash 결과" in ctx.as_prompt()  # 합칠 때는 검색어 순서 → 번호가 항상 같다


def test_research_skips_when_not_needed(models, use_script, no_network_search):
    use_script(Script(SearchPlan=search.SearchPlan(need_search=False, queries=[])))
    assert search.research("시 한 편 써줘", models, SETTINGS).results == [] and no_network_search == []


def test_research_falls_back_to_question_when_planner_fails(models, use_script, no_network_search):
    use_script(Script())  # 검색어 생성 모델 실패
    ctx = search.research("HBF 알려줘", models, SETTINGS)
    assert no_network_search == ["HBF 알려줘"] and any("검색어 생성 실패" in n for n in ctx.notes)


def test_search_failure_does_not_block(models, use_script, monkeypatch):
    use_script(Script(SearchPlan=PLAN))

    def broken(query, max_results, region, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", broken)
    ctx = search.research("HBF", models, SETTINGS)
    assert ctx.results == [] and any("offline" in n for n in ctx.notes)


def test_web_context_reaches_draft_and_review_prompts(org, ranks, team, models):
    node = subtree(org, "staff_draft")
    agent = OrgAgent(node, next(r for r in ranks if r.id == node.rank), team, models)
    assert "[1] 자료" in agent.draft_messages("질문", "업무", web_context="[1] 자료")[-1].content


@pytest.mark.parametrize("use_web, expect_search", [(True, True), (False, False)])
def test_graph_research_step(org, ranks, team, models, use_script, no_network_search, use_web, expect_search):
    use_script(Script(text="초안", SearchPlan=PLAN, ReviewResult=make_review()))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    events = list(stream_office(graph, org, "HBF 알려줘", "am_action", use_web=use_web, brief=BRIEF))
    research = [p for k, p in events if k == "research"]
    assert len(research) == 1 and (research[0] is not None) == expect_search
    assert bool(no_network_search) == expect_search
