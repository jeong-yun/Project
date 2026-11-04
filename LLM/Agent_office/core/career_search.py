"""👔 포트폴리오 최신화의 외부 자료 — 개인 경력이 검색어로 나가지 않게 (커리어 팀은 web_search 툴이 없다).

흐름 (career_ui.web_step — 맡기기 전 화면)
  1) draft_queries : 목표 직무 + 문서에 나온 '일반 기술 용어'(core/career.TECH_TERMS 에 있는 공개 용어)만으로 검색어 초안을 만든다.
                     보조 모델(core/search.plan_queries)에는 문서 원문이 아니라 이 두 가지만 넘긴다. 실패하면 코드가 만든 검색어.
  2) query_problems : 검색어마다 검사 — 숫자(올해·작년·내년 연도와 표준 번호 빼고)·이메일·주소, 그리고 '내 문서에 나온 말 중
                     일반 용어가 아닌 것'(이름·회사·프로젝트일 수 있다)이 있으면 그 검색어는 보내지 않는다.
  3) 화면에서 사람이 검색어를 고치거나 지우고 '승인'을 눌러야 core/search.search_queries 로 보낸다. 기본은 꺼짐.
  4) GitHub 주소(선택) — 공개 저장소의 README·PROJECTS.md 같은 글 파일만, 승인 버튼을 누를 때 가져온다.
외부 자료는 용어·트렌드에만 쓴다. 내 경험·숫자는 내 문서에서만 (career_facts · portfolio_changes 검증기가 확인).
"""

import re
import urllib.request
from urllib.parse import urlparse

from core import career
from core.config import today

# 검색어에 써도 되는 일반 낱말 (직무·기술 분야 말, 검색용 말) — 이 밖의 낱말이 내 문서에 나오면 이름일 수 있어 막는다
GENERIC_KO = {
    # 검색용
    "채용", "공고", "트렌드", "동향", "최신", "요즘", "키워드", "용어", "직무", "요구", "역량", "기술", "스킬", "스택", "전망", "수요",
    "자격", "자격증", "우대", "필수", "경력", "신입", "주니어", "시니어", "포트폴리오", "이력서", "면접", "취업", "이직", "커리어",
    "로드맵", "가이드", "사례", "표준", "실무", "업무", "분야", "산업", "시장", "핵심", "주요", "필요", "변화", "방향", "준비", "현황",
    "년", "올해", "작년", "내년", "상반기", "하반기", "정리", "비교", "개념", "뜻", "의미", "방법", "도구", "프레임워크",
    # 직무 이름
    "보안", "정보보호", "정보보안", "개인정보", "개인정보보호", "관리자", "담당자", "엔지니어", "분석가", "개발자", "아키텍트",
    "컨설턴트", "연구원", "리드", "매니저", "전문가", "책임자", "운영자", "관제", "관제사", "인프라", "클라우드", "데이터", "인공지능",
    "머신러닝", "딥러닝", "생성형", "에이전트", "멀티", "거버넌스", "컴플라이언스", "감사", "위험", "리스크", "관리", "관리체계",
    "인증", "심사", "규제", "법", "법규", "정책", "통제", "접근", "권한", "계정", "인증서", "암호", "암호화", "네트워크", "시스템",
    "서버", "플랫폼", "서비스", "애플리케이션", "웹", "모바일", "컨테이너", "가상화", "자동화", "파이프라인", "모델", "학습", "평가",
    "추론", "배포", "운영", "모니터링", "관측", "관측성", "로그", "이벤트", "분석", "탐지", "이상", "이상탐지", "대응", "침해", "사고",
    "침해사고", "위협", "헌팅", "취약점", "진단", "점검", "모의해킹", "레드팀", "블루팀", "공급망", "제로", "트러스트", "프롬프트",
    "인젝션", "안전", "신뢰", "책임", "윤리", "설명", "가능성", "개발", "엔지니어링", "구축", "설계", "아키텍처", "보안관제",
    "적용", "활용", "도입", "개선", "고도화", "효율화", "정보보호관리체계", "데이터엔지니어", "데이터분석", "보안엔지니어", "클라우드보안", "인공지능보안", "거버넌스관리",
}
GENERIC_EN = {"job", "jobs", "skills", "skill", "trends", "trend", "latest", "keywords", "keyword", "role", "roles", "requirements",
              "career", "hiring", "engineer", "engineering", "analyst", "manager", "developer", "architect", "security", "data", "cloud",
              "ai", "ml", "governance", "compliance", "risk", "infrastructure", "infra", "platform", "operations", "ops", "devops",
              "analytics", "analysis", "detection", "response", "incident", "threat", "hunting", "vulnerability", "management",
              "privacy", "audit", "framework", "frameworks", "standard", "standards", "best", "practices", "practice", "in", "for",
              "and", "of", "the", "to", "with", "what", "is", "new", "top", "guide", "roadmap", "tools", "tool", "pipeline",
              "pipelines", "model", "models", "agent", "agents", "agentic", "generative", "llm", "safety", "trust", "responsible",
              "machine", "learning", "deep", "observability", "monitoring", "log", "logs", "siem", "soc", "team", "lead", "senior",
              "junior", "entry", "level", "certification", "certifications", "terms", "glossary", "market", "demand", "outlook"}
SEARCH_JOSA = ("에서", "으로", "에게", "까지", "부터", "의", "을", "를", "이", "가", "은", "는", "과", "와", "로", "및", "에", "용")


def _strip(word: str) -> str:
    for suffix in SEARCH_JOSA:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _generic_ko(word: str, extra: set[str]) -> bool:
    """한글 낱말이 일반 낱말(또는 일반 낱말을 이어 붙인 것, 예: '보안관제분석')인가."""
    vocab = GENERIC_KO | extra
    if word in vocab:
        return True
    n = len(word)
    ok = [True] + [False] * n  # 앞에서부터 일반 낱말로 잘라 갈 수 있는지
    for i in range(n):
        if ok[i]:
            for j in range(i + 1, n + 1):
                if word[i:j] in vocab:
                    ok[j] = True
    return ok[n]


def _role_words(role: str) -> set[str]:
    return {w for w in re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9+#./\-]*", role or "")}


def query_problems(query: str, private_text: str, role: str = "") -> list[str]:
    """검색어가 내 문서의 개인 정보를 담고 있을 수 있는 이유들 (비어 있으면 보내도 되는 검색어).

    막는 것: 숫자(올해 ±1 연도 · 일반 표준 번호 빼고) · 이메일 · 주소(URL) · 따옴표 인용 ·
             내 문서(private_text)에 나온 낱말 중 일반 직무·기술 용어가 아닌 것 (사람·회사·프로젝트 이름일 수 있다)
    """
    q = career.normalize_text(query)
    if not q:
        return []
    problems = []
    if re.search(r"\S+@\S+", q):
        problems.append("이메일·계정 같은 글 (@)")
    if re.search(r"https?://|www\.|\.(com|kr|net|io|org)\b", q, re.I):
        problems.append("주소(URL)")
    if re.search(r"[\"'“”‘’「」『』]", q):
        problems.append("따옴표 인용 — 문서 문장을 그대로 보내지 않습니다")
    rest = q
    for term in sorted(career.TECH_TERMS, key=len, reverse=True):  # 일반 기술 용어(숫자가 든 표준 이름 포함)는 먼저 지운다
        rest = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", " ", rest, flags=re.I)
    year = today().year
    numbers = [n for n in re.findall(r"\d+(?:[.,]\d+)*", rest) if n not in {str(year - 1), str(year), str(year + 1)}]
    if numbers:
        problems.append(f"숫자 {', '.join(numbers[:3])} — 문서의 숫자일 수 있어 보내지 않습니다 (올해·작년·내년 연도만 가능)")
    private = career.normalize_text(private_text).lower()
    private_flat = re.sub(r"\s+", "", private)
    role_words = {w.lower() for w in _role_words(role)}
    flagged = []
    for tok in re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9+#./&\-]*", rest):
        low = tok.lower()
        if low in role_words or low in GENERIC_EN:
            continue
        if re.match(r"[가-힣]", tok):
            word = _strip(tok)
            if _generic_ko(word, role_words) or len(word) < 2:
                continue
            if word in private_flat:
                flagged.append(tok)
        elif re.search(rf"(?<![a-z0-9]){re.escape(low)}(?![a-z0-9])", private):
            flagged.append(tok)
    if flagged:
        problems.append(f"내 문서에 나온 말 {', '.join(dict.fromkeys(flagged))} — 이름·회사·프로젝트일 수 있어 보내지 않습니다")
    return problems


def fallback_queries(role: str, terms: list[str], max_queries: int = 3) -> list[str]:
    """보조 모델 없이 코드가 만드는 검색어 — 직무 이름 + 올해 + 일반 기술 용어."""
    year = today().year
    role = role.strip() or "IT 직무"
    queries = [f"{role} 채용 요구 역량 {year}", f"{role} 최신 기술 트렌드 {year}"]
    if terms:
        queries.append(f"{role} {' '.join(terms[:3])} 최신 동향")
    return queries[:max_queries]


def planning_question(role: str, terms: list[str]) -> str:
    """보조 모델에 넘기는 글 — 목표 직무와 일반 기술 용어만 (문서 원문·이름·숫자 없음)."""
    return (f"목표 직무 '{role or 'IT 직무'}' 의 요즘 채용 공고에서 많이 쓰는 용어·기술 트렌드·직무 키워드를 찾고 싶다.\n"
            f"관련 일반 기술 용어: {', '.join(terms[:8]) or '(없음)'}\n"
            "검색어에는 직무 이름·일반 기술 용어·연도만 쓴다. 사람·회사·프로젝트 이름이나 경력 숫자는 쓰지 않는다.")


def draft_queries(role: str, private_text: str, models, settings: dict) -> tuple[list[str], list[str]]:
    """검색어 초안 (사람이 화면에서 고친 뒤 승인한다) → (검색어, 안내). 보조 모델이 안 되면 코드가 만든 검색어."""
    from core import search

    cfg = settings.get("web_search", {})
    max_queries = int(cfg.get("max_queries", 3))
    terms = career.tech_terms(private_text)
    notes = []
    try:
        plan, errors = search.plan_queries(planning_question(role, terms), models, max_queries,
                                           "직무 채용 트렌드·용어만 찾는다. 개인·회사·프로젝트 이름은 쓰지 않는다.")
        queries = plan.queries if plan.need_search else []
        if any(str(e).startswith("검색어 생성 실패") for e in errors):  # plan_queries 는 모델이 없으면 질문 글을 검색어로 돌려준다 → 쓰지 않는다
            queries, errors = [], ["보조 모델을 쓸 수 없어 코드가 만든 검색어를 씁니다 (직무 이름 + 일반 기술 용어 + 연도)."]
        notes += errors
    except Exception as e:  # noqa: BLE001 — 모델이 없어도 코드가 만든 검색어로
        queries, notes = [], [f"검색어 초안 모델 호출 실패 → 코드가 만든 검색어: {type(e).__name__}"]
    safe = [q for q in queries if not query_problems(q, private_text, role)]
    if len(safe) < len(queries):
        notes.append(f"모델이 만든 검색어 {len(queries) - len(safe)}개는 개인 정보 검사에 걸려 뺐습니다.")
    if not safe:
        safe = [q for q in fallback_queries(role, terms, max_queries) if not query_problems(q, private_text, role)]
    return safe[:max_queries], notes


# ─── GitHub 공개 저장소의 글 (README · PROJECTS.md) ─────────────────────────

GITHUB_TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst")
MAX_FETCH = 200_000


def github_raw_url(url: str) -> str | None:
    """공개 GitHub 주소 → 내려받을 글 파일 주소 (raw). 저장소 주소면 README.md. 그 밖의 사이트·글이 아닌 파일은 None.

      https://github.com/o/r                     → https://raw.githubusercontent.com/o/r/HEAD/README.md
      https://github.com/o/r/blob/main/PROJECTS.md → https://raw.githubusercontent.com/o/r/main/PROJECTS.md
      https://raw.githubusercontent.com/o/r/main/x.md → 그대로
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https" or parsed.query or parsed.fragment or parsed.port or parsed.username:
        return None
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        path = "/".join(parts)
    elif host in ("github.com", "www.github.com") and len(parts) == 2:
        path = f"{parts[0]}/{parts[1]}/HEAD/README.md"
    elif host in ("github.com", "www.github.com") and len(parts) >= 5 and parts[2] in ("blob", "raw"):
        path = "/".join([parts[0], parts[1], *parts[3:]])
    else:
        return None
    if not path.lower().endswith(GITHUB_TEXT_SUFFIXES):
        return None
    if not all(re.fullmatch(r"[A-Za-z0-9._\-가-힣%]+", p) and p not in (".", "..") for p in path.split("/")):
        return None
    return f"https://raw.githubusercontent.com/{path}"


def fetch_github(url: str, timeout: float = 10.0) -> tuple[str, str]:
    """GitHub 글 파일을 가져온다 → (글, 가져온 주소). 허용하지 않는 주소면 ValueError."""
    raw = github_raw_url(url)
    if raw is None:
        raise ValueError("공개 GitHub 저장소 주소 또는 그 안의 글 파일(.md·.txt) 주소만 가져올 수 있습니다.")
    req = urllib.request.Request(raw, headers={"User-Agent": "agent-office-career/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 위에서 https raw.githubusercontent.com 만 허용
        data = resp.read(MAX_FETCH + 1)
    text = data[:MAX_FETCH].decode("utf-8", errors="replace")
    return text, raw
