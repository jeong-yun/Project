"""👔 커리어 팀의 외부 자료 — '읽기'와 '보내기'를 나눈다 (커리어 팀은 web_search 툴이 없다. 사람이 승인한 것만).

보내기 — 검색어 (내 문서의 낱말이 밖으로 나갈 수 있다 → 검사 + 승인. career_ui.web_step, 기본 꺼짐)
  1) draft_queries : 목표 직무 + 문서에 나온 '일반 기술 용어'(core/career.TECH_TERMS 에 있는 공개 용어)만으로 검색어 초안을 만든다.
                     보조 모델(core/search.plan_queries)에는 문서 원문이 아니라 이 두 가지만 넘긴다. 실패하면 코드가 만든 검색어.
  2) query_problems : 검색어마다 검사 — 숫자(올해·작년·내년 연도와 표준 번호 빼고)·이메일·주소, 그리고 '내 문서에 나온 말 중
                     일반 용어가 아닌 것'(이름·회사·프로젝트일 수 있다)이 있으면 그 검색어는 보내지 않는다.
  3) 화면에서 사람이 검색어를 고치거나 지우고 '승인'을 눌러야 core/search.search_queries 로 보낸다.
읽기 — 사용자가 직접 넣은 주소의 글 (주소만 나간다 → 가져올 주소 목록을 확인하고 승인. 네 양식 + 자유 질문)
  target : 주소 하나 → 가져올 곳 (네트워크 없음). 받지 않는 주소는 FetchProblem — '이유 — 다음에 할 일' 글이 그대로 화면에 나간다.
           GitHub : 공개 저장소·폴더(→ README.md)·글 파일. raw.githubusercontent.com 의 https 로만 가져온다 (로그인·토큰 없음).
           웹     : https 공개 사이트의 그 페이지 하나 (링크를 따라가지 않는다). 내부망으로 가는 주소 — localhost·IP 직접 입력·
                    기본 밖 포트·로그인 정보 — 와 '?' 뒤에 내 문서의 낱말이 든 주소는 받지 않는다.
  fetch  : 승인 뒤에 가져온다. 웹 주소는 이름을 풀어 나온 IP 가 모두 공개 주소인지 먼저 보고, 리다이렉트는 같은 검사를 다시 거친다
           (GitHub 은 raw 호스트 밖으로 넘어가지 않는다). 크기·시간 상한, 글이 아닌 것·글자가 거의 없는 페이지는 이유와 함께 뺀다.
           한계: 이름 검사와 실제 연결 사이에 DNS 답이 바뀌는 경우(DNS 리바인딩)까지는 막지 못한다 — 사용자가 직접 넣은 주소만 읽는다.
외부 자료는 용어·트렌드·프로젝트 이름 참고에만 쓴다. 내 경험·숫자는 내 문서에서만 (career_facts · portfolio_changes 검증기가 확인).
'내 저장소'로 표시한 GitHub 글만 내 문서(사실의 출처)가 된다 (core/tools/career_docs.py).
"""

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, quote, unquote, urlparse

from core import career
from core.config import today
from core.doc_text import MAX_CHARS, PLAIN_SUFFIXES

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


def _without_terms(text: str) -> str:
    """일반 기술 용어(숫자가 든 표준 이름 포함)를 지운 글 — 남은 낱말·숫자만 검사한다."""
    for term in sorted(career.TECH_TERMS, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", " ", text, flags=re.I)
    return text


def _private_words(rest: str, private_text: str, role: str = "") -> list[str]:
    """글(rest)의 낱말 중 내 문서(private_text)에 나오고 일반 직무·기술 용어가 아닌 것 — 사람·회사·프로젝트 이름일 수 있다."""
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
    return list(dict.fromkeys(flagged))


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
    rest = _without_terms(q)  # 일반 기술 용어는 먼저 지운다
    year = today().year
    numbers = [n for n in re.findall(r"\d+(?:[.,]\d+)*", rest) if n not in {str(year - 1), str(year), str(year + 1)}]
    if numbers:
        problems.append(f"숫자 {', '.join(numbers[:3])} — 문서의 숫자일 수 있어 보내지 않습니다 (올해·작년·내년 연도만 가능)")
    flagged = _private_words(rest, private_text, role)
    if flagged:
        problems.append(f"내 문서에 나온 말 {', '.join(flagged)} — 이름·회사·프로젝트일 수 있어 보내지 않습니다")
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


# ─── 읽기 — 사용자가 넣은 GitHub·웹 주소의 글 ──────────────────────────────

GITHUB_HOSTS = ("github.com", "www.github.com")
RAW_HOST = "raw.githubusercontent.com"
TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst", *(f".{s}" for s in PLAIN_SUFFIXES))  # 글 그대로 읽는 파일 (첨부 파일과 같은 범위)
FORMATS = "글 파일(.md·.txt·.rst)과 자료·코드(.py·.yaml·.json·.csv·.sql·.tf·.log)"
MAX_URLS = 5  # 참고 주소 수 — 외부 글 예산(core/tools/career_docs.MAX_EXTERNAL_TOTAL)과 맞춘다
MAX_FETCH = 200_000  # 주소 하나에서 받는 최대 바이트
MIN_PAGE_CHARS = 200  # 웹 페이지에서 읽은 글자가 이보다 적으면 '읽지 못함' (화면을 스크립트로 그리는 사이트)
USER_AGENT = "agent-office-career/1.0"
LOCAL_NAMES = re.compile(r"(^|\.)(localhost|local|localdomain|internal|intranet|lan|home|corp|arpa)$")
PUBLIC_NAME = re.compile(r"(?:[a-z0-9-]+\.)+(?:[a-z]{2,}|xn--[a-z0-9-]+)")
URL_IN_TEXT = re.compile(r"(?:https?://|(?:www\.)?github\.com/)[^\s<>\"'`]+", re.I)
JOSA_AFTER_URL = re.compile(r"(?<=[\x21-\x7e])(?:으로|에서|이랑|하고|[를을은는이가의에와과로도만랑])$")


class FetchProblem(ValueError):
    """주소를 받지 않거나 가져오지 못한 이유 — '왜 안 됐는지 — 무엇을 하면 되는지' 글이 그대로 화면에 나간다."""


@dataclass
class Target:
    url: str  # 사람이 넣은 주소에서 '?…'·'#…' 을 뗀 것 (화면·출처에 보이는 주소. 웹 주소는 '?…' 을 둔다)
    fetch: str  # 실제로 가져올 주소
    kind: str  # "github" / "web"
    folder: bool = False  # GitHub 저장소·폴더 주소 → 그 폴더의 README.md


def split_urls(text: str) -> list[str]:
    """주소 칸의 글 → 주소 목록 (한 줄에 하나, 같은 주소는 한 번). 줄 안의 공백은 주소의 일부다 (폴더 이름에 공백이 있을 수 있다)."""
    return list(dict.fromkeys(line.strip() for line in str(text or "").splitlines() if line.strip()))


def find_urls(text: str) -> list[str]:
    """글(자유 질문) 속의 주소 — 괄호 안에 쓴 주소의 닫는 괄호, 끝의 문장 부호, 주소에 바로 붙은 조사는 뗀다."""
    out = []
    for match in URL_IN_TEXT.finditer(text or ""):
        url, depth = match.group(0), 0
        for i, ch in enumerate(url):
            depth += (ch == "(") - (ch == ")")
            if depth < 0:  # 짝 없는 ')' — 여기부터는 주소가 아니다
                url = url[:i]
                break
        out.append(JOSA_AFTER_URL.sub("", url.rstrip(".,;:!?…")))
    return list(dict.fromkeys(u for u in out if u))


def _parse(url: str):
    """주소 모양 검사 (GitHub·웹 공통) — https · 로그인 정보 없음 · 기본 포트 → (나눈 주소, 호스트, GitHub 인가)."""
    try:
        parsed = urlparse(url)
        host, port = (parsed.hostname or "").lower(), parsed.port
    except ValueError:
        raise FetchProblem("주소 모양이 올바르지 않습니다 — 브라우저 주소창의 주소를 그대로 복사해 넣어 주세요.") from None
    if parsed.username or parsed.password:
        raise FetchProblem("주소에 로그인 정보(아이디·비밀번호)가 들어 있어 받지 않습니다 — 로그인 정보를 뺀 주소를 넣어 주세요.")
    github = host in GITHUB_HOSTS or host == RAW_HOST
    if parsed.scheme != "https" and not (github and parsed.scheme == "http"):  # GitHub 의 http 는 같은 곳이므로 https 로 바꿔 받는다
        raise FetchProblem("https 주소만 읽습니다 — https:// 로 시작하는 주소를 넣어 주세요.")
    if port not in (None, 443):
        raise FetchProblem("기본 포트가 아닌 주소는 받지 않습니다 — 포트 번호(:숫자)를 뺀 주소를 넣어 주세요.")
    return parsed, host, github


def target(url: str, private_text: str = "", role: str = "") -> Target:
    """주소 하나 → 가져올 곳. 네트워크를 쓰지 않는다 (승인 화면이 승인 전에 부른다). 받지 않는 주소면 FetchProblem."""
    text = (url or "").strip()
    if re.match(r"(?:www\.)?github\.com/", text, re.I):  # 'github.com/계정/저장소' 처럼 앞을 빼고 쓴 것
        text = "https://" + text
    parsed, host, github = _parse(text)
    return _github_target(parsed, host) if github else _web_target(parsed, host, private_text, role)


def _segments(path: str) -> list[str]:
    """경로 구간들 (URL 인코딩을 한 번 푼 것). '..'·빈 구간·역슬래시·인코딩한 '/' 같은 경로 속임은 받지 않는다."""
    trimmed = path.strip("/")
    parts = [unquote(p) for p in trimmed.split("/")] if trimmed else []
    if any(not p.strip() or p in (".", "..") or re.search(r"[\\/\x00-\x1f\x7f�]", p) for p in parts):
        raise FetchProblem("경로에 '..'·빈 구간·역슬래시가 있어 받지 않습니다 — GitHub 화면의 주소를 그대로 복사해 넣어 주세요.")
    return parts


def _github_target(parsed, host: str) -> Target:
    """공개 GitHub 주소 → raw 주소. 구간마다 인코딩을 한 번 풀었다가 한 번만 다시 한다 (괄호·쉼표·공백·한글, 이중 인코딩 없음).

      github.com/o/r                          → raw…/o/r/HEAD/README.md          (저장소 → README)
      github.com/o/r/tree/main/docs           → raw…/o/r/main/docs/README.md     (폴더 → 그 폴더의 README)
      github.com/o/r/blob/main/a (b)/x.py     → raw…/o/r/main/a%20%28b%29/x.py   (글 파일·코드)
      raw.githubusercontent.com/o/r/main/x.md → 그대로
    브랜치 이름에 '/' 가 있으면(feature/x) 폴더와 구분하지 못한다 → 404 안내에 '브랜치 이름 확인'을 적는다.
    """
    parts = _segments(parsed.path)
    shown = f"https://{RAW_HOST if host == RAW_HOST else 'github.com'}/" + "/".join(parts)
    bad_format = f"이 형식은 읽지 않습니다 — {FORMATS}만 읽습니다. 그림·PDF·Word 는 내려받아 파일 칸에 올려 주세요."
    if len(parts) < 2 or (host == RAW_HOST and len(parts) < 4):
        raise FetchProblem("프로필·계정 주소는 읽지 않습니다 — 저장소 주소(https://github.com/계정/저장소)나 그 안의 폴더·파일 주소를 넣어 주세요.")
    if not all(re.fullmatch(r"[A-Za-z0-9._-]+", p) for p in parts[:2]):
        raise FetchProblem("계정·저장소 이름이 올바르지 않습니다 — 저장소 주소(https://github.com/계정/저장소)를 확인해 주세요.")
    repo = [parts[0], re.sub(r"\.git$", "", parts[1])]
    suffix = PurePosixPath(parts[-1]).suffix.lower()
    if host == RAW_HOST:
        if suffix not in TEXT_SUFFIXES:
            raise FetchProblem(bad_format)
        rest, folder = parts[2:], False
    elif len(parts) == 2:
        rest, folder = ["HEAD", "README.md"], True
    elif parts[2] in ("blob", "raw", "tree") and len(parts) >= 4:
        suffix = suffix if len(parts) > 4 else ""  # 구간이 브랜치 이름뿐이면 (예: tree/v1.2) 그 브랜치의 README
        if suffix in TEXT_SUFFIXES:
            rest, folder = parts[3:], False
        elif not suffix or parts[2] == "tree":  # 폴더 (이름에 점이 있어도 tree 면 폴더)
            rest, folder = [*parts[3:], "README.md"], True
        else:
            raise FetchProblem(bad_format)
    else:
        raise FetchProblem("이슈·PR·위키 같은 화면은 읽지 않습니다 — 저장소·폴더·파일 주소(예: …/blob/main/README.md)를 넣어 주세요.")
    path = "/".join(quote(p, safe="") for p in [*repo, *rest])
    return Target(shown, f"https://{RAW_HOST}/{path}", "github", folder)


def _public_name(host: str) -> str:
    """공개 사이트 이름인지 (이름 모양만 — 실제 IP 는 fetch 가 이름을 풀어 다시 본다) → ASCII 호스트 이름. 아니면 FetchProblem."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise FetchProblem("IP 주소를 직접 넣은 주소는 읽지 않습니다 — 사이트 이름(예: example.com)으로 된 주소를 넣어 주세요.")
    try:
        host = host.encode("idna").decode("ascii")  # 한글 도메인
    except UnicodeError:
        host = ""
    if LOCAL_NAMES.search(host) or not PUBLIC_NAME.fullmatch(host):
        raise FetchProblem("내부망·내 PC 주소(localhost·.local·점 없는 이름·숫자 주소)는 읽지 않습니다 — 공개된 사이트 주소를 넣어 주세요.")
    return host


def _web_target(parsed, host: str, private_text: str, role: str) -> Target:
    """공개 웹 페이지 주소 하나. '?' 뒤의 값에 내 문서의 낱말이 있으면 보내지 않는다 (검색어와 같은 검사 — 숫자 ID 는 둔다)."""
    host = _public_name(host)
    values = " ".join(v or k for k, v in parse_qsl(parsed.query, keep_blank_values=True))
    words = _private_words(_without_terms(career.normalize_text(values)), private_text, role)
    if words:
        raise FetchProblem(f"주소의 '?' 뒤에 내 문서에 나온 말({', '.join(words)})이 있어 보내지 않습니다 — "
                           "'?' 뒤를 지운 주소를 넣거나 내용을 붙여 넣어 주세요.")
    query = f"?{parsed.query}" if parsed.query else ""
    path = parsed.path or "/"
    encoded = quote(path, safe="/%:@!$&'()*+,;=~") + quote(query, safe="?=&%+/:;,@!$'()*~")  # 이미 인코딩된 것은 그대로
    return Target(f"https://{host}{path}{query}", f"https://{host}{encoded}", "web")


def _public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    addr = getattr(addr, "ipv4_mapped", None) or addr  # ::ffff:192.168.0.1 → 192.168.0.1
    return addr.is_global  # 사설·루프백·링크로컬·예약 주소는 모두 False


def _resolve(host: str) -> list[str]:
    """사이트 이름 → IP 들 (못 찾으면 빈 목록). 테스트는 conftest 가 막는다."""
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    except OSError:
        return []


def _require_public(host: str) -> None:
    """이름을 풀어 나온 IP 가 하나라도 사설·루프백·링크로컬이면 요청을 보내지 않는다."""
    ips = _resolve(host)
    if not ips:
        raise FetchProblem("사이트 이름을 찾지 못했습니다 — 주소에 오타가 없는지, 인터넷이 연결돼 있는지 확인해 주세요.")
    if not all(_public_ip(ip) for ip in ips):
        raise FetchProblem("내부망(사설·로컬) 주소로 연결되는 이름이라 읽지 않습니다 — 공개된 사이트 주소를 넣어 주세요.")


def _checker(t: Target):
    """리다이렉트로 넘어갈 주소를 다시 검사하는 함수 — GitHub 은 raw 호스트 안에서만, 웹은 처음 주소와 같은 검사(공개 https + 이름 풀기)."""

    def check(new_url: str) -> None:
        try:
            parsed, host, _ = _parse(new_url)
            if parsed.scheme != "https":
                raise FetchProblem("https 가 아닌 주소")
            if t.kind == "github":
                if host != RAW_HOST:
                    raise FetchProblem("GitHub 밖의 주소")
            else:
                _require_public(_public_name(host))
        except FetchProblem as e:
            where = (urlparse(new_url).netloc if "://" in new_url else new_url)[:60]
            raise FetchProblem(f"다른 곳({where})으로 넘어가는 주소라 따라가지 않았습니다 — 넘어간 뒤의 주소를 직접 넣거나 "
                               f"내용을 붙여 넣어 주세요 ({str(e).split(' — ')[0]}).") from None

    return check


class _Redirects(urllib.request.HTTPRedirectHandler):
    """리다이렉트마다 넘어갈 주소를 다시 검사한다 (통과하지 못하면 따라가지 않는다)."""

    def __init__(self, check):
        self.check = check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _send(url: str, check, timeout: float) -> tuple[bytes, str]:
    """실제 네트워크 호출은 여기 한 곳 — 주소만 나간다 (본문·쿠키·토큰 없음). 테스트는 conftest 가 막는다. → (앞 MAX_FETCH 바이트, Content-Type)"""
    opener = urllib.request.build_opener(_Redirects(check))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*;q=0.5"})
    with opener.open(request, timeout=timeout) as response:  # noqa: S310 — target() 이 https 공개 주소만 허용
        return response.read(MAX_FETCH + 1)[:MAX_FETCH], str(response.headers.get("Content-Type") or "")


class _PageText(HTMLParser):
    """HTML → 글자 (표준 라이브러리). 스크립트·스타일은 빼고, 목록은 '- ' 줄로 (JD 요구사항을 코드가 뽑을 수 있게)."""

    SKIP = {"script", "style", "noscript", "template", "svg", "iframe"}
    BLOCK = {"p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "footer", "ul", "ol",
             "table", "title", "dt", "dd", "blockquote", "pre", "main", "nav"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK or tag == "li":
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def page_text(html: str) -> str:
    parser = _PageText()
    parser.feed(html)
    parser.close()
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in "".join(parser.out).splitlines())
    return "\n".join(line for line in lines if line and line != "-")


def _decode(data: bytes, content_type: str) -> str:
    named = re.search(r"charset=([\w.-]+)", content_type, re.I)
    for encoding in ([named.group(1)] if named else []) + ["utf-8", "cp949"]:  # 한글 사이트는 charset 없이 EUC-KR 인 곳이 있다
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _http_reason(code: int, t: Target) -> str:
    """HTTP 오류 → 왜 안 됐고 무엇을 하면 되는지."""
    private = "비공개 저장소는 읽을 수 없습니다 — 파일을 내려받아 파일 칸에 올려 주세요."
    if t.kind == "github" and code == 404 and t.folder:
        return f"이 폴더(저장소)에는 README.md 가 없습니다(404) — 읽을 파일 주소를 넣어 주세요 (브랜치·폴더 이름도 확인). {private}"
    if t.kind == "github" and code == 404:
        return f"찾지 못했습니다(404) — 브랜치 이름·파일 이름을 확인하세요. {private}"
    if t.kind == "github":
        return f"GitHub 이 요청을 받지 않았습니다({code}) — 잠시 뒤 다시 맡겨 주세요."
    if code in (404, 410):
        return f"페이지를 찾지 못했습니다({code}) — 주소를 확인해 주세요."
    if code in (401, 403, 429):
        return f"로그인이 필요하거나 자동 접속을 막는 페이지입니다({code}) — 내용을 붙여 넣거나 화면을 캡처해 올려 주세요."
    return f"사이트가 오류를 돌려줬습니다({code}) — 잠시 뒤 다시 맡기거나 내용을 붙여 넣어 주세요."


def fetch(t: Target, timeout: float = 10.0) -> str:
    """승인한 주소 하나의 글 (앞 MAX_CHARS 자, 표시선 위조는 무력화). 가져오지 못하면 FetchProblem — '이유 — 다음에 할 일'."""
    from core import search

    if t.kind == "web":
        _require_public(urlparse(t.fetch).hostname or "")
    try:
        data, content_type = _send(t.fetch, _checker(t), timeout)
    except FetchProblem:
        raise
    except urllib.error.HTTPError as e:
        raise FetchProblem(_http_reason(e.code, t)) from None
    except (OSError, ValueError) as e:  # 연결 실패·시간 초과·인증서 오류
        raise FetchProblem(f"연결하지 못했습니다 — 인터넷 연결을 확인하고 다시 맡겨 주세요 ({type(e).__name__}).") from None
    if t.kind == "web" and content_type and not re.search(r"html|^text/", content_type, re.I):
        raise FetchProblem(f"이 주소는 글(HTML)이 아닙니다 ({content_type.split(';')[0]}) — 파일을 내려받아 파일 칸에 올려 주세요.")
    if b"\x00" in data[:4000]:
        raise FetchProblem("글 파일이 아닙니다 — 그림·PDF·Word 는 내려받아 파일 칸에 올려 주세요.")
    text = _decode(data, content_type)
    if t.kind == "web" and ("html" in content_type.lower() or re.match(r"\s*<(?:!doctype|html)", text, re.I)):
        text = page_text(text)
    text = search.neutralize(text).strip()
    if t.kind == "web" and len(re.sub(r"\s", "", text)) < MIN_PAGE_CHARS:
        raise FetchProblem("이 페이지는 글자를 읽지 못했습니다 (화면을 스크립트로 그리는 사이트일 수 있습니다) — "
                           "공고 내용을 붙여 넣거나 화면을 캡처해 올려 주세요.")
    if not text:
        raise FetchProblem("파일이 비어 있습니다 — 다른 파일 주소를 넣어 주세요.")
    return text[:MAX_CHARS]


