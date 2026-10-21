"""웹 검색 (자료 조사) 단계.

1) 보조(utility) 모델이 질문을 보고 검색 필요 여부와 검색어를 정한다 (SearchPlan).
   - 약어·고유명사·최신 정보는 반드시 검색, 모호한 약어는 풀어 쓴 검색어를 함께 만든다.
     예) "HBF 알려줘" → "HBF" 만 검색하면 가구 회사가 나오므로 "HBF High Bandwidth Flash memory" 로 검색
2) 검색어들을 동시에 웹 검색하고(스레드), 검색어 순서대로 합쳐 중복을 뺀 결과에 번호를 붙여 모든 직급의 프롬프트에 넣는다.
   → 답변자는 근거로 쓰고 [번호]로 출처를 표시, 검토자는 사실관계 검증에 사용한다.

3) team.yaml 의 search.exclude_domains 에 적힌 사이트(블로그·커뮤니티 등)는 근거에서 뺀다.

검색 엔진: DuckDuckGo (ddgs 패키지, API 키 불필요)
새 검색 엔진을 추가하려면 SEARCH_BACKENDS 에 함수를 등록하고 settings.yaml 의 web_search.provider 를 바꾼다.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory
from core.config import date_label, today
from core.llm_factory import NoAvailableModelError
from core.schemas import ModelsConfig


class SearchPlan(BaseModel):
    """검색 계획 (LLM 구조화 출력)."""

    need_search: bool = Field(description="웹 검색이 답변 정확도에 도움이 되면 true")
    queries: list[str] = Field(description="검색어 1~3개. 모호한 약어는 풀어 쓴 형태(영어 정식 명칭 등)를 포함")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebContext(BaseModel):
    """조사 결과 — 그래프 상태에 저장되고 모든 직급 프롬프트에 주입된다."""

    queries: list[str] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)
    provider: str = ""
    notes: list[str] = Field(default_factory=list)

    def as_prompt(self) -> str:
        if not self.results:
            return ""
        lines = [f"[{i}] {r.title} ({r.url})\n{r.snippet}" for i, r in enumerate(self.results, 1)]
        return "\n\n".join(lines)


def _duckduckgo(
    query: str, max_results: int, region: str, text_timelimit: str | None = None, news_timelimit: str | None = None
) -> list[SearchResult]:
    """DuckDuckGo 웹 검색 + 뉴스 검색.

    뉴스를 섞는 이유: 'HBF' 처럼 뜻이 여러 개인 약어는 일반 검색에서 사전·가구 회사 등이 먼저 나오지만,
    뉴스에는 '요즘 화제가 되는 의미'(예: 반도체 High Bandwidth Flash)가 나온다.
    timelimit: d(하루)·w(1주)·m(1개월)·y(1년) — 주식처럼 '현 시점' 자료가 중요한 팀은 team.yaml search 에서 지정.
    """
    from ddgs import DDGS

    ddgs = DDGS()
    try:
        rows = ddgs.text(query, max_results=max_results, region=region, timelimit=text_timelimit) or []
    except Exception:  # noqa: BLE001 — 일부 지역 설정(예: wt-wt)에서 내부 엔진 오류가 나면 us-en 으로 한 번 더
        if region == "us-en":
            raise
        region = "us-en"
        rows = ddgs.text(query, max_results=max_results, region=region, timelimit=text_timelimit) or []
    results = [
        SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
        for r in rows
        if r.get("href")
    ]
    try:
        news = ddgs.news(query, max_results=max(2, max_results // 2), region=region, timelimit=news_timelimit) or []
    except Exception:  # noqa: BLE001 — 뉴스 검색 실패는 무시 (일반 검색 결과만 사용)
        news = []
    results += [
        SearchResult(title=f"[뉴스 {r.get('date', '')[:10]}] {r.get('title', '')}", url=r.get("url", ""), snippet=r.get("body", ""))
        for r in news
        if r.get("url")
    ]
    return results


SEARCH_BACKENDS: dict[str, Callable[[str, int, str], list[SearchResult]]] = {
    "duckduckgo": _duckduckgo,
}


def plan_queries(question: str, models: ModelsConfig, max_queries: int, hint: str = "") -> tuple[SearchPlan, list[str]]:
    """검색어 계획. 모델 호출이 실패하면 질문 자체를 검색어로 쓴다. hint = 팀별 검색 요령 (team.yaml search.hint)."""
    messages = [
        SystemMessage(
            "\n".join(
                [
                    "너는 리서치 보조다. 사용자의 질문에 정확히 답하기 위해 웹 검색이 필요한지 판단하고 검색어를 만든다.",
                    f"- 오늘 날짜: {date_label(today())}. 최신 자료가 필요하면 검색어에 연도·월을 넣는다.",
                    *([f"- 이 팀의 검색 요령: {hint}"] if hint else []),
                    "- 약어, 신조어, 제품·기술·회사 이름, 최신 동향, 수치·날짜가 걸린 질문은 반드시 검색한다(need_search=true).",
                    "- 일반 상식이나 의견·작문 요청처럼 검색이 도움 안 되면 need_search=false.",
                    f"- 검색어는 최대 {max_queries}개. 짧은 약어만 단독으로 쓰지 말고, 풀어 쓴 정식 명칭이나 분야 키워드를 붙인다.",
                    "- 기술 용어는 영어 검색어를 최소 1개 포함한다.",
                    "- 약어처럼 뜻이 여러 개일 수 있으면, 요즘 가장 많이 화제가 되는 의미를 찾도록",
                    "  '약어 + 최신 동향/뉴스' 검색어와 '약어 + 유력 분야 키워드(영어)' 검색어를 함께 만든다.",
                    "  예) 'HBF' → ['HBF 반도체 메모리 최신', 'HBF memory technology', 'HBF 뜻']",
                ]
            )
        ),
        HumanMessage(question),
    ]
    try:
        call = llm_factory.invoke_structured(models.utility, models, SearchPlan, messages, "검색어 생성")
        plan: SearchPlan = call.parsed
        plan.queries = [q.strip() for q in plan.queries if q.strip()][:max_queries]
        return plan, call.errors
    except NoAvailableModelError as e:
        return SearchPlan(need_search=True, queries=[question[:200]]), [f"검색어 생성 실패 → 질문으로 검색: {e}"]


def excluded(url: str, patterns: list[str]) -> bool:
    """URL 이 제외 목록에 걸리는지. 패턴은 도메인(blog.naver.com) 또는 도메인+경로(finance.naver.com/item/board)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    target = host + parsed.path.lower()
    for pattern in patterns:
        pattern = pattern.lower().strip()
        if "/" in pattern:
            if target.startswith(pattern) or target.startswith("www." + pattern):
                return True
        elif host == pattern or host.endswith("." + pattern):
            return True
    return False


def research(question: str, models: ModelsConfig, settings: dict) -> WebContext:
    """검색 계획 → 검색 → 중복 제거. 어떤 단계가 실패해도 예외 없이 notes 에 남긴다."""
    cfg = settings.get("web_search", {})
    provider = cfg.get("provider", "duckduckgo")
    max_results = cfg.get("max_results", 5)
    max_total = cfg.get("max_total_results", 8)
    region = cfg.get("region", "kr-kr")

    plan, notes = plan_queries(question, models, cfg.get("max_queries", 3), cfg.get("hint", ""))
    context = WebContext(provider=provider, notes=notes)
    if not plan.need_search or not plan.queries:
        context.notes.append("검색이 필요 없는 질문으로 판단")
        return context
    context.queries = plan.queries

    backend = SEARCH_BACKENDS.get(provider)
    if backend is None:
        context.notes.append(f"알 수 없는 검색 엔진: {provider}")
        return context

    def fetch(query: str) -> list[SearchResult] | Exception:
        try:
            return backend(
                query, max_results, region,
                text_timelimit=cfg.get("text_timelimit"), news_timelimit=cfg.get("news_timelimit"),
            )
        except Exception as e:  # noqa: BLE001 — 네트워크·검색 엔진 오류는 답변을 막지 않는다
            return e

    # 검색어들을 동시에 검색하고, 합치는 것은 검색어 순서대로 → 결과 번호([1], [2]…)가 실행마다 같다
    with ThreadPoolExecutor(max_workers=len(plan.queries)) as pool:
        fetched = list(pool.map(fetch, plan.queries))

    seen: set[str] = set()
    exclude = cfg.get("exclude_domains") or []
    dropped = 0
    for query, results in zip(plan.queries, fetched):
        if isinstance(results, Exception):
            context.notes.append(f"검색 실패 ({query}): {type(results).__name__}: {str(results)[:200]}")
            continue
        for r in results:
            if exclude and excluded(r.url, exclude):
                dropped += 1
                continue
            if r.url not in seen and len(context.results) < max_total:
                seen.add(r.url)
                context.results.append(r)
    if dropped:
        context.notes.append(f"근거로 쓰지 않는 사이트(블로그·커뮤니티 등) 결과 {dropped}건 제외")
    return context
