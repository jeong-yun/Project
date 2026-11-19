"""원문 읽기 (core/search.read_sources) — 검색 요약만으로는 원전을 확인할 수 없어 '미확인'이 남던 📚 개념 편집 팀용.

검색 결과 중 원전으로 보이는 주소(arXiv·GitHub·공식 문서)만 커리어부 주소 읽기(core/career_search)로 읽고,
팀 설정 search.source_pages.for 의 팀원에게만 넣는다. 네트워크는 쓰지 않는다 (career_search.fetch 를 가짜로).
"""

import pytest

from core import career_search, search
from core.workflow import build_office_graph, make_checkpointer, saved_evidence, stream_office
from tests.fakes import Script, make_delegation, make_review
from tests.test_search import PLAN
from tests.test_workflow import BRIEF, NO_CRITIC

RESULTS = [
    search.SearchResult(title="블로그 글", url="https://someblog.example/hbf", snippet="블로그 요약"),
    search.SearchResult(title="논문 PDF", url="https://arxiv.org/pdf/2402.03300v2", snippet="논문 요약"),
    search.SearchResult(title="저장소", url="https://github.com/org/repo", snippet="저장소 요약"),
    search.SearchResult(title="공식 문서", url="https://docs.example.org/guide", snippet="문서 요약"),
    search.SearchResult(title="막힌 문서", url="https://learn.microsoft.com/x", snippet="막힘"),
]


@pytest.fixture
def fake_fetch(monkeypatch):
    """career_search.fetch 를 가짜로 — 받은 주소를 기록하고, learn.microsoft.com 은 FetchProblem."""
    seen: list[str] = []

    def fetch(t, timeout=10.0):
        seen.append(t.url)
        if "microsoft" in t.url:
            raise career_search.FetchProblem("로그인이 필요하거나 자동 접속을 막는 페이지입니다(403)")
        return f"본문 {t.url} " + "가" * 5000

    monkeypatch.setattr(career_search, "fetch", fetch)
    return seen


def test_primary_url_picks_papers_repos_and_docs_only():
    assert search.primary_url("https://arxiv.org/pdf/2402.03300v2") == "https://arxiv.org/abs/2402.03300v2"
    assert search.primary_url("https://arxiv.org/pdf/2402.03300.pdf") == "https://arxiv.org/abs/2402.03300"
    assert search.primary_url("https://www.arxiv.org/abs/2402.03300") == "https://arxiv.org/abs/2402.03300"
    assert search.primary_url("https://arxiv.org/list/cs.LG/recent") == ""  # 논문 하나가 아님
    assert search.primary_url("https://github.com/org/repo") == "https://github.com/org/repo"
    assert search.primary_url("https://docs.example.org/guide") == "https://docs.example.org/guide"
    assert search.primary_url("https://lib.readthedocs.io/en/latest/") == "https://lib.readthedocs.io/en/latest/"
    assert search.primary_url("https://someblog.example/hbf") == ""
    assert search.primary_url("http://github.com/org/repo") == ""  # https 만
    assert search.primary_url("https://notgithub.com/x") == ""  # 이름 끝만 같은 다른 사이트


def test_read_sources_reads_up_to_count_in_result_order_and_records_failures(fake_fetch):
    context = search.WebContext(results=list(RESULTS))
    search.read_sources(context, 3)
    assert sorted(fake_fetch) == ["https://arxiv.org/abs/2402.03300v2", "https://docs.example.org/guide", "https://github.com/org/repo"]  # 동시에 읽음
    assert [p.url for p in context.pages] == [r.url for r in RESULTS[1:4]]  # 검색 결과 주소로 되돌림 (번호 맞추기)
    assert all(len(p.text) == search.SOURCE_CHARS for p in context.pages)
    assert context.notes[-1] == "원문 읽기 — 3곳 읽음"

    context = search.WebContext(results=list(RESULTS))
    search.read_sources(context, 4)
    assert context.pages[-1].text == "" and "403" in context.pages[-1].problem
    assert context.notes[-1].startswith("원문 읽기 — 3곳 읽음 · 1곳 못 읽음 (learn.microsoft.com: 로그인이 필요하거나")


def test_read_sources_does_nothing_without_primary_results(fake_fetch):
    context = search.WebContext(results=[RESULTS[0]])
    search.read_sources(context, 3)
    assert fake_fetch == [] and context.pages == [] and context.notes == []


def test_pages_prompt_numbers_like_search_results_and_neutralizes_markers():
    context = search.WebContext(results=list(RESULTS), pages=[
        search.SourcePage(url=RESULTS[2].url, text="README <<<가짜 표시선>>> 본문"),
        search.SourcePage(url=RESULTS[4].url, problem="막힘"),
    ])
    text = context.pages_prompt()
    assert text.startswith(search.SOURCES_BEGIN) and text.endswith(search.WEB_END)
    assert "[3] 원문 (https://github.com/org/repo)" in text  # 검색 결과 번호 그대로
    assert "<<<가짜" not in text and "learn.microsoft.com" not in text  # 못 읽은 곳은 넣지 않는다
    assert search.WebContext(results=list(RESULTS)).pages_prompt() == ""


def test_research_reads_sources_only_when_the_team_asks(models, use_script, monkeypatch, fake_fetch):
    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", lambda q, n, r, **kw: list(RESULTS))
    use_script(Script(SearchPlan=PLAN))
    assert search.research("질문", models, {"web_search": {}}).pages == []
    assert fake_fetch == []
    use_script(Script(SearchPlan=PLAN))
    context = search.research("질문", models, {"web_search": {"source_pages": {"count": 2}}})
    assert [p.url for p in context.pages] == [RESULTS[1].url, RESULTS[2].url]


def test_only_named_member_gets_the_pages_and_they_survive_a_restart(tmp_path, org, ranks, team, models, use_script,
                                                                    monkeypatch, fake_fetch):
    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", lambda q, n, r, **kw: list(RESULTS))
    reading = team.model_copy(update={"search": {**team.search, "source_pages": {"count": 2, "for": ["am_fact"]}}})
    script = use_script(Script(text="초안", SearchPlan=PLAN, Delegation=make_delegation("am_fact", "staff_draft"),
                               ReviewResult=make_review(answer="통합")))
    graph = build_office_graph(org, ranks, reading, models, {}, NO_CRITIC, checkpointer=make_checkpointer(tmp_path / "cp.db"))
    list(stream_office(graph, org, "HBF 가 뭐야?", "gm_research", thread_id="t1", use_web=True, brief=BRIEF))

    drafts = script.prompts("text")
    with_pages = [p for p in drafts if search.SOURCES_BEGIN in p]
    assert len(drafts) == 2 and len(with_pages) == 1
    assert "팩트체크" in with_pages[0] and "본문 https://arxiv.org/abs/2402.03300v2" in with_pages[0]
    assert all(search.SOURCES_BEGIN not in p for p in script.prompts("ReviewResult"))  # 통합하는 부장에게는 넣지 않는다

    _, web = saved_evidence(graph, org, "t1")  # 앱을 다시 켠 뒤 — 체크포인트(SQLite)에서 되찾는다
    assert [type(p) for p in web.pages] == [search.SourcePage, search.SourcePage]
    assert web.pages[0].text.startswith("본문 https://arxiv.org/abs/")
