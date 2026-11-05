"""👔 포트폴리오·자소서 팀의 코드 계산·검사 (LLM 이 아니라 코드가 한다 — 같은 입력이면 늘 같은 결과).

  parse_questions   : 자소서 문항 '한 줄에 하나' → 번호·질문·글자 수 제한(공백 포함/제외)
  count_chars       : 글자 수 (공백 포함 · 공백 제외)
  split_answers     : 최종 답변에서 '## 문항 N.' 절마다 답변 본문만 꺼낸다
  extract_requirements / coverage : JD 요구사항(R1…) 뽑기 → 올린 문서에 근거가 있는지 키워드로 1차 판정
  check_facts       : 지어낸 사실 막기 — JSON 인용(quote)이 원문에 글자 그대로 있는지 + 본문 숫자가 원문·JD·문항에 있는지
  check_char_limits / check_coverage_ids / check_portfolio : 하네스 검증기(core/harness/validators.py)가 부른다
  render_document   : 내려받기용 마크다운 (본문 + 코드가 센 글자 수 + JD 대응표 + 확인 필요 + 근거 인용)

최종 답변 끝의 ```json 블록 형식 (팀 업무 지침과 같다)
  {"evidence":   [{"claim": "본문 주장 요약", "quote": "올린 문서 원문 그대로"}],
   "jd_coverage": [{"id": "R1", "requirement": "...", "status": "근거 있음|부분|공백", "quote": "원문 그대로 (공백이면 빈칸)"}],
   "needs_check": ["원문에 없어 [확인 필요]로 표시한 내용"]}
"""

import json
import re
from dataclasses import dataclass, field

from core.governance import normalize_text, quote_snippets
from core.harness.base import extract_code_blocks

NEEDS_CHECK = "[확인 필요]"
COVER_STATUSES = ("근거 있음", "부분", "공백")
PORTFOLIO_PARTS = ("문제", "역할", "기술", "결과", "배운 점")


# ─── 자소서 문항 · 글자 수 ──────────────────────────────────────────────


@dataclass
class Question:
    no: int
    text: str  # 질문 (제한 표시를 뺀 것)
    limit: int | None = None  # 글자 수 제한
    no_spaces: bool = False  # True = 공백 제외로 센다

    @property
    def basis(self) -> str:
        return "공백 제외" if self.no_spaces else "공백 포함"

    def label(self) -> str:
        return f"{self.limit:,}자, {self.basis}" if self.limit else "제한 없음"


LIMIT = re.compile(r"[(\[（]?\s*(?:최대\s*)?(?:공백\s*(?:포함|제외)\s*)?(\d{1,2},?\d{3}|\d{2,4})\s*자(?:\s*(?:이내|내외|이하))?"
                   r"(?:\s*[,/]?\s*(?:공백|띄어쓰기)\s*(?:포함|제외))?\s*[)\]）]?")
NUMBERING = re.compile(r"^\s*(?:Q\s*\d+\s*[.):]?|문항\s*\d+\s*[.):]?|\d{1,2}\s*[.)]|[①-⑳])\s*", re.I)


def parse_questions(text: str) -> list[Question]:
    """한 줄에 문항 하나. 줄 안의 '(700자)' '(1,000자 이내, 공백 제외)' '[500자]' 를 제한으로 읽는다."""
    out: list[Question] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = None
        for m in LIMIT.finditer(line):  # 마지막 표시를 제한으로 (질문 안의 '3자' 같은 말과 헷갈리지 않게 끝쪽)
            pass
        limit, no_spaces, body = None, False, line
        if m:
            limit = int(m.group(1).replace(",", ""))
            no_spaces = bool(re.search(r"(공백|띄어쓰기)\s*제외", m.group(0)))
            body = (line[:m.start()] + line[m.end():]).strip()
        body = NUMBERING.sub("", body).strip(" -:·")
        out.append(Question(no=len(out) + 1, text=body or line, limit=limit, no_spaces=no_spaces))
    return out


def questions_from_request(question: str) -> list[Question]:
    """업무 질문 글(teams/career/team.yaml 템플릿)의 '[문항] …' 부분 → 문항. 서버를 다시 켜 툴 결과가 없을 때 쓴다."""
    m = re.search(r"^\[문항\]\s?(.*?)(?=^말투·강조점:|^추가 요청:|\Z)", question or "", re.S | re.M)
    return parse_questions(m.group(1)) if m else []


def count_chars(text: str) -> tuple[int, int]:
    """(공백 포함, 공백 제외). 줄바꿈은 공백 포함에서 1자로 센다."""
    text = (text or "").replace("\r\n", "\n").strip()
    return len(text), len(re.sub(r"\s", "", text))


ANSWER_HEAD = re.compile(r"^#{2,4}\s*(?:문항\s*)?(\d{1,2})\s*[.):]", re.M)


def split_answers(answer: str) -> dict[int, str]:
    """'## 문항 N. 질문' 절 → {N: 답변 본문}. 본문 = 제목 다음 줄부터 다음 제목(#) 앞까지, 코드 블록·인용(>)·'글자 수' 줄은 뺀다."""
    body = re.sub(r"```.*?```", "", answer, flags=re.S)
    heads = list(ANSWER_HEAD.finditer(body))
    out: dict[int, str] = {}
    for i, m in enumerate(heads):
        start = body.index("\n", m.end()) + 1 if "\n" in body[m.end():] else len(body)
        end = len(body)
        nxt = re.search(r"^#{1,4}\s", body[start:], re.M)
        if nxt:
            end = start + nxt.start()
        lines = [ln for ln in body[start:end].splitlines()
                 if not ln.lstrip().startswith(">") and not re.match(r"^\s*[*_(]*\s*글자\s*수", ln)
                 and not re.fullmatch(r"\s*(-{3,}|\*{3,})\s*", ln)]
        out.setdefault(int(m.group(1)), "\n".join(lines).strip())
    return out


@dataclass
class LimitRow:
    no: int
    question: str
    limit: int | None
    no_spaces: bool
    with_spaces: int
    without_spaces: int

    @property
    def counted(self) -> int:
        return self.without_spaces if self.no_spaces else self.with_spaces

    @property
    def ok(self) -> bool:
        return self.limit is None or self.counted <= self.limit


def limit_rows(answer: str, questions: list[Question]) -> list[LimitRow]:
    answers = split_answers(answer)
    rows = []
    for q in questions:
        w, wo = count_chars(answers.get(q.no, ""))
        rows.append(LimitRow(q.no, q.text, q.limit, q.no_spaces, w, wo))
    return rows


def check_char_limits(answer: str, questions: list[Question], min_ratio: float = 0.5) -> list[str]:
    """문항마다 답변이 있는지 · 제한을 넘지 않는지 · 너무 짧지 않은지(제한의 min_ratio 미만)."""
    answers = split_answers(answer)
    problems = []
    for row in limit_rows(answer, questions):
        if not answers.get(row.no):
            problems.append(f"문항 {row.no} 답변이 없다 ('## 문항 {row.no}. 질문' 제목 아래에 답변을 쓴다)")
            continue
        if row.limit is None:
            continue
        basis = "공백 제외" if row.no_spaces else "공백 포함"
        if row.counted > row.limit:
            problems.append(f"문항 {row.no} 이 {basis} {row.counted:,}자로 제한 {row.limit:,}자를 {row.counted - row.limit:,}자 넘는다 "
                            f"— 사례·문장을 줄여 {row.limit:,}자 이하로")
        elif row.counted < row.limit * min_ratio:
            problems.append(f"문항 {row.no} 이 {basis} {row.counted:,}자로 제한 {row.limit:,}자의 {int(min_ratio * 100)}%에 못 미친다 "
                            "— 근거 문서의 사례를 더 구체적으로 (근거가 모자라면 [확인 필요]로 표시)")
    extra = sorted(set(answers) - {q.no for q in questions})
    if questions and extra:
        problems.append(f"없는 문항 번호의 답변: {extra} — 문항은 {len(questions)}개다")
    return problems


# ─── JD 요구사항 · 근거 대응 ────────────────────────────────────────────


@dataclass
class Requirement:
    id: str
    text: str
    section: str = ""  # 업무 / 자격 / 우대 / (빈칸)
    keywords: list[str] = field(default_factory=list)


@dataclass
class Coverage:
    req: Requirement
    status: str  # 근거 있음 / 부분 / 공백
    hits: list[str]
    snippet: str = ""


SECTION_HINTS = (
    ("우대", re.compile(r"우대|preferred|nice to have|plus", re.I)),
    ("자격", re.compile(r"자격|요건|필수|requirement|qualification|지원\s*조건", re.I)),
    ("업무", re.compile(r"업무|담당|하는 일|직무\s*내용|responsibilit|what you", re.I)),
)
BULLET = re.compile(r"^\s*(?:[-•·*▪◦○●□■▶►※✔✓➢]|\d{1,2}[.)]|[가-하][.)])\s*")
JOSA = ("에서의", "으로서", "으로써", "에서는", "에게서", "에서", "으로", "까지", "부터", "이나", "이며", "하고", "하는", "하여",
        "해야", "하며", "하신", "있는", "있으신", "가능한", "가능자", "에", "의", "을", "를", "이", "가", "은", "는", "과", "와",
        "로", "및", "등", "한", "할", "된", "적", "분", "자")
STOP_KO = {"경험", "이상", "능력", "보유", "우대", "관련", "업무", "가능", "분야", "이해", "지식", "활용", "역량", "수행", "담당",
           "기반", "대한", "통한", "위한", "대해", "사항", "자격", "요건", "필수", "학력", "무관", "경력", "신입", "지원", "모집",
           "근무", "주요", "내용", "관리", "운영", "있으", "하신", "갖춘", "가진", "통해", "소지", "또는", "그리고", "이해도",
           "기본", "실무", "다양", "원활", "커뮤니케이션", "적극", "필요", "년", "개월", "대응", "개발", "구축", "수립",
           "경험자", "능숙", "능통", "익숙", "가능하신", "보유자", "우대합니다", "환영"}
STOP_EN = {"and", "or", "the", "of", "in", "with", "for", "to", "a", "an", "on", "experience", "years", "year", "plus", "etc",
           "skills", "skill", "knowledge", "ability", "using", "based", "related"}


def _strip_josa(word: str) -> str:
    for suffix in JOSA:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def keywords(text: str) -> list[str]:
    """요구사항 문장 → 키워드 (영문 용어·한글 명사 비슷한 말). 형태소 분석 없이 조사만 떼는 근사."""
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9+#./\-]*[A-Za-z0-9+#]|[A-Za-z]|[가-힣]{2,}", text):
        if re.match(r"[A-Za-z]", tok):
            if tok.lower() not in STOP_EN and len(tok) > 1:
                out.append(tok)
        else:
            word = _strip_josa(tok)
            if len(word) >= 2 and word not in STOP_KO:
                out.append(word)
    return list(dict.fromkeys(out))


def extract_requirements(jd: str, limit: int = 20) -> list[Requirement]:
    """JD → 요구사항 목록 (R1…). 섹션 제목(주요 업무·자격 요건·우대 사항)이 있으면 그 아래 줄만, 없으면 글머리표 줄, 그것도 없으면 문장."""
    lines = [ln.rstrip() for ln in (jd or "").splitlines()]
    items: list[tuple[str, str]] = []
    section, saw_section = "", False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        bare = BULLET.sub("", s).strip("[]【】<>()#*: ").strip()
        is_bullet = bool(BULLET.match(s))
        hint = next((name for name, rx in SECTION_HINTS if rx.search(bare)), "")
        if hint and len(bare) <= 20 and not is_bullet:
            section, saw_section = hint, True
            continue
        if (saw_section and section) or (not saw_section and is_bullet):
            if len(bare) >= 4:
                items.append((section, bare))
    if not items:  # 글머리표·섹션이 없는 JD — 문장 단위
        for sent in re.split(r"(?<=[.!?。])\s+|\n", jd or ""):
            sent = sent.strip()
            if len(sent) >= 8:
                items.append(("", sent))
    reqs = []
    for i, (sec, text) in enumerate(items[:limit], 1):
        reqs.append(Requirement(f"R{i}", text, sec, keywords(text)))
    return reqs


def coverage(reqs: list[Requirement], source: str, window: int = 3) -> list[Coverage]:
    """요구사항마다 1차 판정 — 키워드가 문서의 가까운 줄(연속 window 줄) 안에 함께 2/3 이상 있으면 근거 있음,
    문서 어딘가에 하나라도 있으면 부분, 없으면 공백. 키워드가 가장 많이 모인 원문 줄을 함께 준다.
    (흩어진 낱말이 우연히 맞는 것을 '근거 있음'으로 보지 않게 가까운 줄만 본다)"""
    low = source.lower()
    src_lines = [ln.strip() for ln in source.splitlines() if ln.strip()]
    out = []
    for req in reqs:
        words = [k.lower() for k in req.keywords]
        hits = [k for k in req.keywords if k.lower() in low]
        best_hits, best_line = 0, ""
        for i in range(len(src_lines)):
            chunk = " ".join(src_lines[i:i + window]).lower()
            n = sum(w in chunk for w in words)
            if n > best_hits:
                best_hits = n
                best_line = max(src_lines[i:i + window], key=lambda ln: sum(w in ln.lower() for w in words))
        status = "근거 있음" if words and best_hits / len(words) >= 2 / 3 else ("부분" if hits else "공백")
        out.append(Coverage(req, status, hits, best_line[:160]))
    return out


def coverage_markdown(rows: list[Coverage]) -> str:
    lines = ["| ID | 구분 | 요구사항 | 1차 판정(키워드) | 문서에서 찾은 키워드 | 가장 가까운 원문 |", "|---|---|---|---|---|---|"]
    for c in rows:
        lines.append(f"| {c.req.id} | {c.req.section or '-'} | {_cell(c.req.text)} | {c.status} | {', '.join(c.hits) or '-'} | "
                     f"{_cell(c.snippet) or '-'} |")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


# ─── 답변 JSON ──────────────────────────────────────────────────────────


def answer_json(answer: str) -> dict | None:
    """최종 답변의 마지막 ```json 블록 (없거나 깨졌으면 None)."""
    for src in reversed(extract_code_blocks(answer, "json")):
        try:
            data = json.loads(src)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _items(data: dict | None, key: str) -> list[dict]:
    value = (data or {}).get(key) or []
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


CODE_BLOCK = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n?(.*?)```", re.S)
KEEP_CODE = ("mermaid",)  # 본문에 남기는 코드 블록 (흐름도) — 결과 .md 에 그대로, Word 에는 코드 상자 + 안내


def body_text(answer: str) -> str:
    """JSON 요약 블록(```json, 언어 표시 없는 끝 JSON)과 그 밖의 코드 블록을 뺀 본문. ```mermaid 흐름도는 남긴다.

    (2026-11-04: 예전에는 모든 코드 블록을 지워 모델이 쓴 mermaid 흐름도가 결과 파일에서 사라졌다)
    mermaid 노드 글의 숫자도 본문이므로 사실 검사(check_facts)를 받는다.
    """
    def keep(m: re.Match) -> str:
        return m.group(0) if m.group(1).lower() in KEEP_CODE else ""

    return CODE_BLOCK.sub(keep, answer).strip()


def figure_marks(text: str) -> set[int]:
    """글 속 '[그림 N]' 번호들."""
    return {int(n) for n in re.findall(r"\[그림\s*(\d{1,3})\]", text or "")}


def missing_figures(answer: str, count: int) -> list[int]:
    """원본 그림 1~count 중 결과 본문에 '[그림 N]' 표시가 없는 번호."""
    have = figure_marks(body_text(answer))
    return [n for n in range(1, count + 1) if n not in have]


# ─── 지어낸 사실 막기 ──────────────────────────────────────────────────

# 숫자 앞의 영문 이름은 표준·제품·버전 번호다 (ISO 27001, Python 3.11, OWASP Top 10, Windows 11 …) — 사실 숫자로 보지 않는다
NAME_PREFIX = re.compile(
    r"(?:ISO|IEC|ISO/IEC|NIST|SP|CSF|CVE|CWE|RFC|Top|PCI|DSS|SOC|ISMS|ISMS-P|Python|Java|JDK|Windows|Ubuntu|CentOS|RHEL|Rocky|"
    r"IPv|HTTP|HTTPS|TLS|SSL|SHA|AES|RSA|Log4j|Spring|Vue|React|Angular|Node|PHP|Oracle|MySQL|PostgreSQL|Postgres|Kubernetes|"
    r"k8s|Office|Server|Level|Lv|Tier|Phase|Stage|Gen|OAuth|SAML|HTML|CSS|ES|Elasticsearch|Splunk|Kafka|Spark|Hadoop|CIS|OWASP|"
    r"GDPR|Web|v|Ver|Version|Win|Excel|Word|Hadoop|Airflow|Terraform|Ansible|Docker|SQL|GPT|Claude|Gemini|Llama|BERT|YOLO|"
    r"Unity|Unreal|Cisco|Juniper|Fortinet|FortiGate|Palo|Layer|L|IEEE|Wi-Fi|WiFi|USB|PCIe|DDR|Ryzen|Core|Xeon|Grade|Lv\.)"
    r"[\s\-]?$", re.I)
NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-/])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![A-Za-z0-9_])")
# 단위가 붙은 수는 단위까지 같아야 원문과 같은 사실이다 ('40개' 가 원문에 있어도 '40%' 는 다른 사실).
# 날짜·기간(년·월·일·개월)은 쓰는 방식이 여러 가지라('2023.03' = '2023년 3월') 단위를 보지 않는다.
UNIT = re.compile(r"^\s?(%|퍼센트|배|조|억|백만|만|천|원|명|건|시간|분|초|회|개사|개|대|곳|위|점|TB|GB|MB|ms)")
SAME_UNIT = {"퍼센트": "%"}


def _norm_number(whole: str, frac: str | None) -> str:
    value = whole.replace(",", "")
    try:
        value = str(int(value))
    except ValueError:
        pass
    if frac and frac.strip("0"):
        return f"{value}.{frac.rstrip('0')}"
    return value


def _unit(after: str) -> str:
    m = UNIT.match(after)
    return SAME_UNIT.get(m.group(1), m.group(1)) if m else ""


def numbers_in(text: str) -> set[str]:
    """비교용 숫자 모음 — '1,200' → '1200', 단위가 있으면 '1200분' 도. '2023.03' → '2023.3' · '2023' · '3' (날짜 조각도 따로)."""
    found: set[str] = set()
    norm = normalize_text(text)
    for m in NUMBER.finditer(norm):
        value = _norm_number(m.group(1), m.group(2))
        found |= {value, _norm_number(m.group(1), None)}
        unit = _unit(norm[m.end():])
        if unit:
            found.add(value + unit)
        if m.group(2):
            found.add(_norm_number(m.group(2), None))
    for m in re.finditer(r"\d+", text):  # 날짜·기간 '2021~2024', '2023-03' 의 조각
        found.add(str(int(m.group())))
    found |= date_variants(norm)
    return found


# ─── 날짜 표기 바꾸기 (같은 날짜 · 다른 표기) ──────────────────────────────
# 2026-11-04 실측: 수정 요청으로 '2026-06-01' → '26.06' 처럼 표기만 바꾼 것을 '문서에 없는 숫자 20개'로 실패 처리 → 재작성.
# 원문의 날짜(연·월)마다 같은 날짜의 다른 표기를 허용 숫자에 넣는다. 원문에 없는 다른 달(26.07)은 여전히 실패.
#   원문 2026-06-01 · 2026.06 · 2026/06 · 2026년 6월 → 26.06 · 26.6 · 2026.06 · 2026.6 · 26
#   원문 26.06 · 26년 6월 (연 두 자리, 20xx 로 본다)   → 2026 · 2026.06 · 2026.6
# '26.06%' · '26.06배' 처럼 단위가 붙은 수, 월이 1~12 가 아닌 수는 날짜로 보지 않는다.
DATE_LONG = re.compile(r"(?<![\d.])((?:19|20)\d{2})\s*(?:[-./]\s*|년\s*)(\d{1,2})(?!\d)(?:\s*월)?")
DATE_SHORT = re.compile(r"(?<![\d.,])(\d{2})(?:\.(\d{1,2})(?![\d%])|\s*년\s*(\d{1,2})\s*월)")


def _date_tokens(year: int, month: int) -> set[str]:
    yy = year % 100
    out = {str(year), str(yy)}
    for y in (str(year), f"{yy:02d}"):
        for mm in (f"{month:02d}", str(month)):
            out.add(_norm_number(y, mm))
    return out


def date_variants(text: str) -> set[str]:
    """글 속 날짜(연·월)마다 같은 날짜의 다른 표기 (비교용 숫자 모양 — numbers_in 과 같은 정규화)."""
    found: set[str] = set()
    for m in DATE_LONG.finditer(text):
        month = int(m.group(2))
        if 1 <= month <= 12:
            found |= _date_tokens(int(m.group(1)), month)
    for m in DATE_SHORT.finditer(text):
        month = int(m.group(2) or m.group(3))
        if not 1 <= month <= 12 or (m.group(2) and _unit(text[m.end():])):
            continue
        found |= _date_tokens(2000 + int(m.group(1)), month)
    return found


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?。])\s+|\n+", text) if s.strip()]


@dataclass
class FactCheck:
    quotes_checked: int = 0
    missing_quotes: list[str] = field(default_factory=list)  # 원문에 없는 인용
    unsourced_numbers: list[str] = field(default_factory=list)  # 원문·JD·문항에 없는 숫자 (문장 앞부분 포함)
    no_evidence: bool = False  # 근거 인용(evidence)이 하나도 없음

    @property
    def ok(self) -> bool:
        return not (self.missing_quotes or self.unsourced_numbers or self.no_evidence)


def _derived_percent(value: str, others: list[float]) -> bool:
    """'1,200분 → 720분 (40% 단축)' — 같은 문장의 원문 숫자 두 개로 계산되는 비율이면 지어낸 수가 아니다."""
    try:
        target = float(value)
    except ValueError:
        return False
    for a in others:
        for b in others:
            if a > 0 and a != b and (abs(round(abs(a - b) / a * 100, 1) - target) <= 0.5 or abs(round(b / a * 100, 1) - target) <= 0.5):
                return True
    return False


# 순서 번호 — 사실이 아니라 줄 번호다. 마크다운 표의 첫 칸이 숫자뿐인 줄(`| 13 | IPS… |`)과
# '13.' '13)' 로 시작하는 목록 번호. 2026-11-04 실측: 프로젝트를 11개 넘게 번호 매긴 표를 매번 '문서에 없는 숫자'로 잡아 재작성.
TABLE_ROW_NO = re.compile(r"^(\s*\|\s*)\d{1,3}(\s*\|)", re.M)
LIST_NO = re.compile(r"^(\s*(?:[-*+]\s+)?)\d{1,3}[.)](?=\s)", re.M)


def strip_ordinals(body: str) -> str:
    """표 첫 칸의 순서 번호와 목록 번호를 지운다 (표의 다른 칸·문장 안 숫자는 그대로 검사)."""
    return LIST_NO.sub(r"\1", TABLE_ROW_NO.sub(r"\1\2", body))


def unsourced_numbers(body: str, allowed: set[str]) -> list[str]:
    """본문 숫자 중 원문·JD·문항에 없는 것 ('[확인 필요]'가 붙은 문장, 표·목록의 순서 번호는 건너뛴다)."""
    out: list[str] = []
    for sent in _sentences(strip_ordinals(body)):
        if NEEDS_CHECK in sent or "확인 필요" in sent:
            continue
        norm = normalize_text(sent)
        tokens = []
        for m in NUMBER.finditer(norm):
            if NAME_PREFIX.search(norm[: m.start()][-14:]):
                continue
            tokens.append((m, _norm_number(m.group(1), m.group(2)), _unit(norm[m.end():])))
        sourced = [float(v) for _, v, u in tokens if (v + u if u else v) in allowed and re.fullmatch(r"\d+(\.\d+)?", v)]
        for m, value, unit in tokens:
            small = value.isdigit() and int(value) <= 10
            if small and unit not in ("%", "배", "조", "억", "만", "천", "백만"):
                continue  # '3가지', '1차', '2명' 같은 작은 수는 문장 표현으로 본다
            if (value + unit if unit else value) in allowed:
                continue
            if unit == "%" and _derived_percent(value, sourced):
                continue
            ctx = norm[max(0, m.start() - 12): m.end() + 6].strip()
            out.append(f"{m.group(0)}{unit} (…{ctx}…)")
    return list(dict.fromkeys(out))


def check_facts(answer: str, source: str, reference: str = "") -> FactCheck:
    """지어낸 사실 검사 — source = 올린 문서 원문, reference = JD·문항·회사 메모 (숫자만 허용)."""
    result = FactCheck()
    data = answer_json(answer)
    src = normalize_text(source)
    fit_cover = [x for fit in _items(data, "jd_fit") for x in _items(fit, "coverage")]  # JD 추천의 JD 마다 판정
    for item in [*_items(data, "evidence"), *_items(data, "jd_coverage"), *fit_cover]:
        if str(item.get("status") or "").strip() == "공백":
            continue
        for snippet in quote_snippets(item.get("quote")):
            result.quotes_checked += 1
            if snippet not in src:
                label = item.get("claim") or item.get("id") or item.get("requirement") or "?"
                result.missing_quotes.append(f"{str(label)[:30]}: \"{snippet[:70]}\"")
    if source.strip() and not _items(data, "evidence"):
        result.no_evidence = True
    allowed = numbers_in(source) | numbers_in(reference)
    result.unsourced_numbers = unsourced_numbers(body_text(answer), allowed)
    return result


def check_coverage_ids(answer: str, req_ids: list[str]) -> list[str]:
    """JSON jd_coverage 가 코드가 뽑은 요구사항(R1…)을 모두 다루는지, 판정 값이 정해진 것인지, 근거 판정에 인용이 있는지."""
    items = _items(answer_json(answer), "jd_coverage")
    if req_ids and not items:
        return [f"JSON 에 jd_coverage 가 없다 — 근거 자료의 JD 요구사항 {', '.join(req_ids)} 를 모두 판정하라"]
    seen = {str(x.get("id") or "").strip() for x in items}
    problems = []
    missing = [r for r in req_ids if r not in seen]
    if missing:
        problems.append(f"jd_coverage 에서 빠진 요구사항: {', '.join(missing)}")
    bad = [f"{x.get('id')}({x.get('status')})" for x in items if str(x.get("status") or "").strip() not in COVER_STATUSES]
    if bad:
        problems.append(f"status 는 {' / '.join(COVER_STATUSES)} 중 하나여야 한다: {', '.join(bad[:8])}")
    unquoted = [str(x.get("id")) for x in items if str(x.get("status") or "").strip() in ("근거 있음", "부분")
                and not quote_snippets(x.get("quote"))]
    if unquoted:
        problems.append(f"근거 있음·부분 판정인데 quote(원문 인용)가 없다: {', '.join(unquoted[:8])} — 인용할 원문이 없으면 '공백'")
    return problems


# ─── 포트폴리오 형식 ────────────────────────────────────────────────────


def project_cards(answer: str) -> list[tuple[str, str]]:
    """'## 프로젝트' 아래 '### 제목' 카드들 → [(제목, 본문)]."""
    body = body_text(answer)
    m = re.search(r"^##\s+[^\n]*프로젝트[^\n]*$", body, re.M)
    if not m:
        return []
    rest = body[m.end():]
    end = re.search(r"^##\s", rest, re.M)
    rest = rest[: end.start()] if end else rest
    parts = re.split(r"^###\s+(.+)$", rest, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def check_portfolio(answer: str) -> list[str]:
    body = body_text(answer)
    problems = []
    for name, rx in (("요약", r"^##\s+[^\n]*요약"), ("핵심 역량(기술 매트릭스 표)", r"^##\s+[^\n]*(역량|기술)"),
                     ("프로젝트", r"^##\s+[^\n]*프로젝트")):
        if not re.search(rx, body, re.M):
            problems.append(f"'## {name}' 절이 없다")
    cards = project_cards(answer)
    if not cards and not any("프로젝트" in p for p in problems):
        problems.append("'## 프로젝트' 아래 '### 프로젝트 이름' 카드가 하나도 없다")
    for title, text in cards:
        missing = [p for p in PORTFOLIO_PARTS if not re.search(rf"\*\*\s*{p}\s*\*\*|^\s*[-*]?\s*{p}\s*[:：]", text, re.M)]
        if missing:
            problems.append(f"프로젝트 '{title[:30]}' 카드에 빠진 항목: {', '.join(missing)} (항목마다 '- **문제**: …' 형식)")
    if not re.search(r"^\|.*\|\s*$", body, re.M):
        problems.append("기술 매트릭스 표(| 영역 | 기술·도구 | 근거 |)가 없다")
    return problems


# ─── 요청 형식 (원본 형식 유지) ─────────────────────────────────────────
# 2026-11-04 실측: 추가 요청 'Word 는 기존 Word 형식, md 는 기존 md 형식과 유사하게'를 표준 형식 검사(check_portfolio)가 덮어써
# 원본 표·구분이 사라지고 요청 형식은 부록으로 밀렸다 → 형식 지정이 있으면 표준 절 대신 '원본 절 제목·표 머리글이 그대로 있는지'를 본다.

FORMAT_WORDS = re.compile(r"형식|포맷|포멧|양식|서식|레이아웃|틀\s*(?:을|를|은|는|대로|에|그대로)|템플릿|template|format|layout|"
                          r"(?:기존|원본|원래|지금)\s*(?:문서|파일|포트폴리오|경력서|경력\s*기술서|이력서)?\s*(?:과|와|처럼|대로|그대로)",
                          re.I)
OMIT_PHRASES = re.compile(r"나머지(?:는|도)?\s*(?:원문|기존|이전)?\s*(?:그대로|동일|생략)|이하\s*(?:동일|생략)|\((?:중략|생략|이하 생략)\)|"
                          r"(?:변경|바뀐)\s*(?:없는|없음|사항\s*없는)\s*(?:항목|부분|프로젝트)?(?:은|는)?\s*(?:생략|그대로\s*둡니다)")


def format_requested(text: str) -> bool:
    """추가 요청 글에 형식 지정이 있는가 ('기존 Word 형식과 유사하게', '경력서.md 처럼', '원본 표 그대로' …)."""
    return bool(FORMAT_WORDS.search(text or ""))


def _flat(text: str) -> str:
    """비교용 — 마크다운 표시·공백·문장 부호를 뺀 글 (원본 제목이 결과에 '그대로' 있는지 볼 때)."""
    text = normalize_text(text).lower()
    return re.sub(r"[\s*_#`|>\-·.,:：()\[\]【】<>「」'\"~]+", "", text)


def _bare_heading(text: str) -> str:
    """절 제목의 번호(1. · Ⅰ. · 가. · (1) · 1-1.)를 뗀 이름."""
    return re.sub(r"^\s*(?:[\dⅠ-Ⅻⅰ-ⅻ]+(?:[-.]\d+)*[.)]?|[가-하][.)]|\(\d+\)|[①-⑳])\s*", "", str(text or "")).strip()


def original_format_problems(answer: str, fmt: dict) -> list[str]:
    """요청 형식 검사 — 원본(지금 포트폴리오·형식 참고 파일)의 절 제목·표 머리글이 결과 본문에 그대로 있는지 + 생략 문구가 없는지.

    fmt = {"headings": [...], "table_heads": [[칸, …], …], "sources": [파일 이름]}  (core/tools/career_docs.py 가 만든다)
    """
    body = body_text(answer)
    flat = _flat(body)
    problems = []
    missing = [h for h in dict.fromkeys(fmt.get("headings") or []) if _flat(_bare_heading(h)) and _flat(_bare_heading(h)) not in flat]
    if missing:
        problems.append(f"원본 절 제목 {len(missing)}개가 결과에 없다: " + ", ".join(f"'{h[:30]}'" for h in missing[:8]))
    for head in fmt.get("table_heads") or []:
        cells = [c for c in head if _flat(c)]
        lost = [c for c in cells if _flat(c) not in flat]
        if lost:
            problems.append(f"원본 표 머리글 '{' | '.join(c[:15] for c in cells[:8])}' 중 결과에 없는 칸: " + ", ".join(f"'{c[:20]}'" for c in lost[:6]))
    omitted = OMIT_PHRASES.search(body)
    if omitted:
        problems.append(f"'{omitted.group(0)}' 처럼 항목을 생략했다 — 바뀌지 않은 표·항목도 빠짐없이 쓴다")
    return problems


# ─── 🎯 업무(JD) 추천 — 모델은 판정만, 점수·순위는 코드가 ───────────────────

STATUS_VALUE = {"근거 있음": 1.0, "부분": 0.5, "공백": 0.0}
# 요구사항 가중치 — 자격 요건(필수)이 맞아야 서류가 통과하므로 두 배. 주요 업무·우대·구분 없음은 1배.
REQ_WEIGHT = {"자격": 2.0, "업무": 1.0, "우대": 1.0, "": 1.0}
SECTION_LABEL = {"자격": "필수", "업무": "업무", "우대": "우대", "": "기타"}
SCORE_RULE = ("점수 = Σ(가중치 × 판정값) ÷ Σ가중치 × 100 — 판정값: 근거 있음 1 · 부분 0.5 · 공백 0, "
              "가중치: 필수(자격 요건) ×2 · 주요 업무 ×1 · 우대 ×1 · 구분 없음 ×1. "
              "근거 있음·부분인데 인용이 올린 문서에 글자 그대로 없으면 0 으로 센다.")
JD_SPLIT = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,}|_{3,})\s*$", re.M)


def split_jds(text: str) -> list[str]:
    """붙여 넣은 글 → JD 여러 개. '---' · '===' 만 있는 줄로 나눈다 (그런 줄이 없으면 JD 하나)."""
    return [p.strip() for p in JD_SPLIT.split(text or "") if p.strip()]


def jd_title(text: str, fallback: str = "") -> str:
    """JD 의 첫 줄(회사·직무 이름이 보통 여기 있다) — 표에 보일 짧은 이름."""
    for line in (text or "").splitlines():
        s = line.strip().strip("[]【】<>#*=:- ").strip()
        if len(s) >= 2:
            return s[:50]
    return fallback


@dataclass
class JobPost:
    id: str  # J1…
    name: str  # 파일 이름 또는 '붙여 넣은 JD 1'
    text: str
    reqs: list[Requirement]

    @property
    def title(self) -> str:
        return jd_title(self.text, self.name)

    def as_option(self) -> dict:
        """하네스·화면에 넘기는 모양 (JSON 으로 저장할 수 있게 dict)."""
        return {"id": self.id, "name": self.name, "title": self.title,
                "reqs": [{"id": r.id, "section": r.section, "text": r.text} for r in self.reqs]}


def job_posts(parts: list[tuple[str, str]], limit: int = 15) -> list[JobPost]:
    """[(이름, JD 글)] → J1, J2 … 요구사항 ID 는 'J1-R1' 처럼 JD 번호를 붙인다 (JD 끼리 섞이지 않게)."""
    out = []
    for i, (name, text) in enumerate(parts, 1):
        reqs = extract_requirements(text, limit)
        for r in reqs:
            r.id = f"J{i}-{r.id}"
        out.append(JobPost(f"J{i}", name, text, reqs))
    return out


def fit_items(answer: str) -> dict[str, dict]:
    """JSON jd_fit → {JD id: 그 JD 판정}."""
    return {str(x.get("jd") or "").strip(): x for x in _items(answer_json(answer), "jd_fit")}


@dataclass
class JdScore:
    id: str
    title: str
    score: float | None  # 0~100 (요구사항을 못 뽑은 JD 는 None)
    earned: float
    total: float
    rows: list[dict]  # 요구사항마다 {id, section, text, status, quote, counted, note}
    missing: list[str] = field(default_factory=list)  # 판정이 없는 요구사항 (공백으로 셈)
    unverified: list[str] = field(default_factory=list)  # 인용이 문서에 없어 0 으로 센 요구사항

    def counts(self, section: str) -> dict[str, int]:
        rows = [r for r in self.rows if r["section"] == section]
        return {s: sum(r["status"] == s for r in rows) for s in COVER_STATUSES}

    @property
    def basis(self) -> str:
        """점수 근거 한 줄 — '필수 2개(근거 1·부분 1·공백 0) ×2 · 우대 … → 5.0/7.0 × 100 = 71.4'."""
        if self.score is None:
            return "요구사항을 찾지 못해 점수를 매기지 않음 (JD 에 '자격 요건'·글머리표가 있으면 더 잘 뽑힙니다)"
        parts = []
        for sec in ("자격", "업무", "우대", ""):
            rows = [r for r in self.rows if r["section"] == sec]
            if rows:
                c = self.counts(sec)
                parts.append(f"{SECTION_LABEL[sec]} {len(rows)}개(근거 {c['근거 있음']}·부분 {c['부분']}·공백 {c['공백']}) "
                             f"×{REQ_WEIGHT[sec]:g}")
        tail = f" → {self.earned:g}/{self.total:g} × 100 = {self.score:g}"
        flags = ""
        if self.unverified:
            flags += f" · 인용 불일치로 0 처리: {', '.join(self.unverified)}"
        if self.missing:
            flags += f" · 판정 없음(공백 처리): {', '.join(self.missing)}"
        return " · ".join(parts) + tail + flags


def score_jds(answer: str, jds: list[dict], source: str | None = None) -> list[JdScore]:
    """모델의 판정(jd_fit.coverage) → JD 마다 점수 (JD 순서 그대로). 같은 입력이면 늘 같은 점수.

    source 를 주면 근거 있음·부분의 인용이 올린 문서에 글자 그대로 있는지 보고, 없으면 0 으로 센다
    (인용 없는 '근거 있음'으로 점수를 올리지 못하게).
    """
    fits = fit_items(answer)
    src = normalize_text(source) if source is not None else None
    out = []
    for jd in jds:
        cover = {str(x.get("id") or "").strip(): x for x in _items(fits.get(jd["id"]), "coverage")}
        rows, missing, unverified = [], [], []
        earned = total = 0.0
        for req in jd.get("reqs") or []:
            item = cover.get(req["id"])
            status = str((item or {}).get("status") or "").strip()
            if status not in STATUS_VALUE:
                missing.append(req["id"])
                status = "공백"
            quote = str((item or {}).get("quote") or "").strip()
            value, note = STATUS_VALUE[status], ""
            if value and src is not None:
                snippets = quote_snippets(quote)
                if not snippets or any(s not in src for s in snippets):
                    value, note = 0.0, "인용이 문서에 없음 → 0"
                    unverified.append(req["id"])
            weight = REQ_WEIGHT.get(req.get("section") or "", 1.0)
            earned += weight * value
            total += weight
            rows.append({"id": req["id"], "section": req.get("section") or "", "text": req.get("text") or "", "status": status,
                         "quote": quote, "counted": value, "note": note})
        score = round(earned / total * 100, 1) if total else None
        out.append(JdScore(jd["id"], jd.get("title") or jd.get("name") or jd["id"], score, earned, total, rows, missing, unverified))
    return out


def ranked(scores: list[JdScore]) -> list[JdScore]:
    """점수 높은 순 (같은 점수는 올린 순서, 점수 없는 JD 는 맨 뒤)."""
    order = {s.id: i for i, s in enumerate(scores)}
    return sorted(scores, key=lambda s: (s.score is None, -(s.score or 0), order[s.id]))


def ranking_text(scores: list[JdScore]) -> str:
    return " > ".join(f"{s.id} {s.score:g}점" if s.score is not None else f"{s.id} (점수 없음)" for s in ranked(scores))


def check_recommend(answer: str, jds: list[dict], source: str | None = None) -> list[str]:
    """JD 추천 검사 — 모든 JD·요구사항을 판정했는지, 순위·추천이 코드가 계산한 점수와 맞는지."""
    data = answer_json(answer)
    fits = fit_items(answer)
    if not fits:
        return [f"JSON 에 jd_fit 가 없다 — {', '.join(j['id'] for j in jds)} 마다 요구사항 판정(coverage)·한 줄 요약을 적는다"]
    problems = []
    absent = [j["id"] for j in jds if j["id"] not in fits]
    if absent:
        problems.append(f"판정하지 않은 JD: {', '.join(absent)}")
    for jd in jds:
        fit = fits.get(jd["id"])
        if fit is None:
            continue
        items = _items(fit, "coverage")
        seen = {str(x.get("id") or "").strip() for x in items}
        missing = [r["id"] for r in jd.get("reqs") or [] if r["id"] not in seen]
        if missing:
            problems.append(f"{jd['id']} 에서 판정하지 않은 요구사항: {', '.join(missing)}")
        bad = [f"{x.get('id')}({x.get('status')})" for x in items if str(x.get("status") or "").strip() not in COVER_STATUSES]
        if bad:
            problems.append(f"{jd['id']} status 는 {' / '.join(COVER_STATUSES)} 중 하나: {', '.join(bad[:6])}")
        unquoted = [str(x.get("id")) for x in items if str(x.get("status") or "").strip() in ("근거 있음", "부분")
                    and not quote_snippets(x.get("quote"))]
        if unquoted:
            problems.append(f"{jd['id']} 근거 있음·부분인데 quote(문서 원문)가 없다: {', '.join(unquoted[:6])} — 인용할 원문이 없으면 '공백'")
        if not str(fit.get("summary") or "").strip():
            problems.append(f"{jd['id']} 한 줄 요약(summary)이 없다")
        weak = any(str(x.get("status") or "").strip() in ("부분", "공백") for x in items)
        if weak and not [x for x in fit.get("fill_first") or [] if str(x).strip()]:
            problems.append(f"{jd['id']} 에 부분·공백 요구사항이 있는데 먼저 채울 것(fill_first)이 없다")

    scores = {s.id: s for s in score_jds(answer, jds, source)}
    order = [str(x).strip() for x in (data or {}).get("ranking") or []]
    expected = ranking_text(list(scores.values()))
    if sorted(order) != sorted(scores):
        problems.append(f"ranking 은 JD id 를 하나씩 모두 담아야 한다 (지금 {order or '없음'}) — 코드가 계산한 순위: {expected}")
    else:
        values = [scores[i].score for i in order if scores[i].score is not None]
        if any(a < b for a, b in zip(values, values[1:])):
            problems.append(f"ranking {' > '.join(order)} 이 코드가 계산한 점수 순서와 다르다 — 코드 계산: {expected} "
                            "(같은 점수는 순서 상관없음). 점수를 바꾸지 말고 판정을 다시 보거나 순위를 고친다.")
    best = str(((data or {}).get("best") or {}).get("jd") or "").strip() if isinstance((data or {}).get("best"), dict) else ""
    top = [s for s in scores.values() if s.score is not None]
    high = max((s.score for s in top), default=None)
    if not best:
        problems.append("best(가장 맞는 JD 와 이유)가 없다 — {\"jd\": \"J…\", \"why\": \"…\"}")
    elif high is not None and (best not in scores or scores[best].score != high):
        problems.append(f"best {best} 는 코드 계산 1위가 아니다 — 코드 계산: {expected}")
    return problems


def recommend_markdown(answer: str, jds: list[dict], source: str | None = None) -> str:
    """코드가 만든 추천 표 — 순위·점수·점수 근거, JD 마다 요구사항 대응표·강점·공백·먼저 채울 것."""
    fits = fit_items(answer)
    data = answer_json(answer) or {}
    scores = ranked(score_jds(answer, jds, source))
    lines = ["## 추천 순위 (점수는 코드가 계산)", "", f"> {SCORE_RULE}", "",
             "| 순위 | JD | 한 줄 요약 | 적합도 | 점수 근거 |", "|---|---|---|---|---|"]
    for i, s in enumerate(scores, 1):
        summary = (fits.get(s.id) or {}).get("summary") or "-"
        score = f"**{s.score:g}점**" if s.score is not None else "-"
        lines.append(f"| {i} | {s.id}. {_cell(s.title)} | {_cell(summary)} | {score} | {_cell(s.basis)} |")
    best = data.get("best") if isinstance(data.get("best"), dict) else {}
    if scores and scores[0].score is not None:
        top = scores[0]
        why = str(best.get("why") or "").strip()
        lines += ["", f"**가장 맞는 JD: {top.id}. {top.title}** ({top.score:g}점)" + (f" — {why}" if why and best.get("jd") == top.id else "")]
        if best.get("jd") and best.get("jd") != top.id:
            lines.append(f"> ⚠️ 모델이 고른 추천({best.get('jd')})이 코드 계산 1위({top.id})와 다릅니다 — 코드 점수를 따르세요.")
    for s in scores:
        fit = fits.get(s.id) or {}
        lines += ["", f"### {s.id}. {s.title} — 요구사항별 대응", "", "| ID | 구분 | 요구사항 | 판정 | 근거(문서 원문) |", "|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {SECTION_LABEL.get(r['section'], '기타')} | {_cell(r['text'])} | {r['status']}"
                  f"{' ⚠️ ' + r['note'] if r['note'] else ''} | {_cell(r['quote']) or '-'} |" for r in s.rows]
        for label, key in (("강점", "strengths"), ("공백", "gaps"), ("먼저 채울 것", "fill_first")):
            values = [str(x).strip() for x in fit.get(key) or [] if str(x).strip()]
            if values:
                lines += ["", f"**{label}**", *[f"- {v}" for v in values]]
    return "\n".join(lines).strip()


# ─── 🔄 포트폴리오 최신화 — 바뀐 점 찾기 · 변경 목록 검사 ─────────────────────

# 일반 기술·직무 용어 (사람·회사·프로젝트 이름이 아닌 공개 용어) — 바뀐 기술 찾기와 검색어 안전 검사(core/career_search.py)가 쓴다
TECH_TERMS = (
    "Python", "SQL", "Java", "Rust", "JavaScript", "TypeScript", "Bash", "PowerShell", "pandas", "NumPy", "scikit-learn",
    "PyTorch", "TensorFlow", "Keras", "XGBoost", "LightGBM", "Isolation Forest", "Spark", "Kafka", "Airflow", "dbt", "Hadoop",
    "Flink", "Elasticsearch", "OpenSearch", "Kibana", "Logstash", "Splunk", "Grafana", "Prometheus", "Datadog", "ClickHouse",
    "PostgreSQL", "MySQL", "Oracle", "MongoDB", "Redis", "BigQuery", "Snowflake", "Databricks", "AWS", "Azure", "GCP", "EC2", "S3",
    "Lambda", "CloudTrail", "GuardDuty", "IAM", "VPC", "EKS", "ECS", "Terraform", "Ansible", "Docker", "Kubernetes", "Helm",
    "GitHub Actions", "Jenkins", "GitLab", "CI/CD", "Linux", "Windows", "SIEM", "SOAR", "EDR", "XDR", "NDR", "IDS", "IPS", "WAF",
    "DLP", "PAM", "MFA", "SSO", "Zero Trust", "제로 트러스트", "IoC", "TTP", "MITRE ATT&CK", "ATT&CK", "Sigma", "YARA", "CVE",
    "OWASP", "OWASP Top 10", "ISMS-P", "ISMS", "ISO/IEC 27001", "ISO 27001", "ISO/IEC 27701", "ISO/IEC 42001", "NIST CSF",
    "NIST CSF 2.0", "NIST AI RMF", "AI RMF", "EU AI Act", "GDPR", "개인정보보호법", "SOC 2", "PCI DSS", "CSPM", "CNAPP", "SBOM",
    "DevSecOps", "MLOps", "LLMOps", "LLM", "RAG", "LangChain", "LangGraph", "Streamlit", "FastAPI", "Django", "Flask", "React",
    "Tableau", "Power BI", "Excel", "Jupyter", "Git", "API", "REST", "GraphQL", "ETL", "ELT", "데이터 파이프라인", "데이터 거버넌스",
    "AI 거버넌스", "AI 보안", "프롬프트 인젝션", "레드팀", "위협 헌팅", "침해사고 대응", "취약점 진단", "모의해킹", "보안 관제",
    "로그 분석", "이상 탐지", "이상탐지", "머신러닝", "딥러닝", "생성형 AI", "AI 에이전트", "멀티 에이전트", "클라우드 보안",
    "컨테이너 보안", "공급망 보안", "IaC", "관측성", "Observability",
)


def tech_terms(text: str) -> list[str]:
    """글에 나온 일반 기술·직무 용어 (TECH_TERMS 중) — 영어는 낱말 경계, 한글은 띄어쓰기를 무시하고 찾는다."""
    found = []
    flat = re.sub(r"\s+", "", text or "")
    for term in TECH_TERMS:
        if re.match(r"[A-Za-z]", term):
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text or "", re.I):
                found.append(term)
        elif re.sub(r"\s+", "", term) in flat:
            found.append(term)
    return list(dict.fromkeys(found))


def diff_candidates(old: str, new: str) -> dict[str, list[str]]:
    """지금 포트폴리오(old) ↔ 최신 요약 문서(new) — 코드가 찾은 '바뀐 점 후보' (모델이 놓치지 않게 참고로 준다).

      projects : 최신 문서의 제목 줄(#) 중 낱말 대부분이 포트폴리오에 없는 것 → 새 프로젝트 후보
      numbers  : 최신 문서에만 있는 숫자가 든 문장 → 새 숫자·성과 후보
      terms    : 최신 문서에만 있는 기술 용어
    """
    old_low = (old or "").lower()
    projects = []
    for line in (new or "").splitlines():
        m = re.match(r"^\s*#{2,4}\s+(.+)$", line)
        if not m:
            continue
        words = keywords(re.sub(r"[\d.~()\-]+", " ", m.group(1)))
        if words and sum(w.lower() in old_low for w in words) / len(words) < 0.5:
            projects.append(m.group(1).strip())
    old_numbers = numbers_in(old or "")
    numbers = []
    for sent in _sentences(new or ""):
        norm = normalize_text(sent)
        fresh = []
        for m in NUMBER.finditer(norm):
            value, unit = _norm_number(m.group(1), m.group(2)), _unit(norm[m.end():])
            if NAME_PREFIX.search(norm[: m.start()][-14:]) or (value.isdigit() and int(value) <= 10 and not unit):
                continue
            if (value + unit if unit else value) not in old_numbers:
                fresh.append(value + unit)
        if fresh:
            numbers.append(sent.strip()[:160])
    terms = [t for t in tech_terms(new) if t not in tech_terms(old)]
    return {"projects": projects[:12], "numbers": list(dict.fromkeys(numbers))[:15], "terms": terms[:20]}


CHANGE_KINDS = ("문서", "외부")
WEB_REF = re.compile(r"\[W(\d{1,2})\]")


def check_changes(answer: str, old: str, source: str, urls: list[str], web_count: int = 0, reference: str = "") -> list[str]:
    """변경 목록(JSON changes) 검사.

      before : 지금 포트폴리오에 글자 그대로 있어야 한다 (새로 더한 것은 빈칸)
      문서    : source 는 내 문서 원문 인용 (글자 그대로)
      외부    : source 는 승인한 검색 결과·GitHub 주소 중 하나 — 용어·트렌드에만 쓴다
      after  : 내 문서에 없는 숫자를 넣지 않는다 (외부 자료의 숫자는 내 경력이 아니다)
      본문 [W번호] 는 외부 자료 목록 안의 번호
    """
    data = answer_json(answer)
    changes = _items(data, "changes")
    if not changes:
        if str((data or {}).get("no_changes") or "").strip():
            return []
        return ["JSON 에 changes(바뀐 점 목록)가 없다 — 바뀐 곳마다 {part, before, after, why, source, kind}. "
                "정말 바뀐 것이 없으면 no_changes 에 이유를 적는다"]
    problems = []
    old_n, src_n = normalize_text(old), normalize_text(source)
    allowed_urls = {u.strip().rstrip("/") for u in urls if u}
    allowed_numbers = numbers_in(source) | numbers_in(reference)
    for i, c in enumerate(changes, 1):
        label = f"변경 {i}({str(c.get('part') or '?')[:20]})"
        kind = str(c.get("kind") or "").strip()
        if kind not in CHANGE_KINDS:
            problems.append(f"{label}: kind 는 '문서' 또는 '외부'")
        if not str(c.get("after") or "").strip():
            problems.append(f"{label}: after(바뀐 뒤 내용)가 비었다")
        if not str(c.get("why") or "").strip():
            problems.append(f"{label}: why(바꾼 이유)가 비었다")
        missing_before = [s for s in quote_snippets(c.get("before")) if s not in old_n]
        if missing_before:
            problems.append(f"{label}: before 가 지금 포트폴리오에 글자 그대로 없다 (\"{missing_before[0][:50]}\") — 그대로 복사하거나, 새로 더한 것이면 빈칸")
        src = str(c.get("source") or "").strip()
        if kind == "문서":
            snippets = quote_snippets(src)
            if not snippets:
                problems.append(f"{label}: source(내 문서 원문 인용)가 없다")
            elif any(s not in src_n for s in snippets):
                problems.append(f"{label}: source 인용이 내 문서에 글자 그대로 없다 (\"{src[:50]}\")")
        elif kind == "외부" and src.rstrip("/") not in allowed_urls:
            problems.append(f"{label}: source 는 승인한 외부 자료의 URL 이어야 한다 (지금 \"{src[:60]}\")"
                            + ("" if allowed_urls else " — 이번에는 외부 자료가 없으므로 '외부' 변경을 쓰지 않는다"))
        nums = unsourced_numbers(str(c.get("after") or ""), allowed_numbers)
        if nums:
            problems.append(f"{label}: after 에 내 문서에 없는 숫자 {', '.join(nums[:3])} — 외부 자료의 숫자를 내 경력처럼 쓰지 않는다")
    refs = sorted({int(n) for n in WEB_REF.findall(body_text(answer))})
    bad = [n for n in refs if n < 1 or n > web_count]
    if bad:
        problems.append(f"본문의 {', '.join(f'[W{n}]' for n in bad)} 은 외부 자료 목록에 없다 (외부 자료 {web_count}건)")
    return problems


# ─── 내려받기용 문서 ────────────────────────────────────────────────────


def render_document(answer: str, kind: str, questions: list[Question] | None = None, title: str = "",
                    requirements: dict[str, str] | None = None, jds: list[dict] | None = None, source: str | None = None,
                    web: list[dict] | None = None) -> str:
    """최종 답변 → 내려받기용 마크다운.

    본문(JSON 빼고) + 자소서면 문항마다 코드가 센 글자 수 + 부록(JD 대응표 · 확인 필요 · 근거 인용 — JSON 에서 코드가 만든 표).
    JD 추천(jd_recommend)은 맨 앞에 코드가 계산한 순위·점수·요구사항 대응표 (jds = 툴이 뽑은 JD·요구사항, source = 문서 원문).
    포트폴리오 최신화(portfolio_update)는 부록에 바뀐 점(이전 → 이후 · 이유 · 출처)과 외부 자료 목록(web).
    """
    body = body_text(answer)
    if kind == "jd_recommend" and jds:
        explained = "\n\n---\n\n## 설명 (작성: 팀)\n\n" + body if body else ""
        body = recommend_markdown(answer, jds, source) + explained
    if kind == "cover_letter" and questions:
        rows = {r.no: r for r in limit_rows(answer, questions)}
        out_lines, current = [], None
        for line in body.splitlines():
            m = ANSWER_HEAD.match(line)
            if m and current is not None:
                out_lines += _count_line(rows.get(current))
            if m:
                current = int(m.group(1))
            elif re.match(r"^#{1,2}\s", line) and current is not None:
                out_lines += _count_line(rows.get(current))
                current = None
            if not re.match(r"^\s*[*_(]*\s*글자\s*수", line):  # 모델이 직접 센 글자 수는 빼고 코드 값으로
                out_lines.append(line)
        if current is not None:
            out_lines += _count_line(rows.get(current))
        body = "\n".join(out_lines).strip()
    if title and not body.lstrip().startswith("# "):
        body = f"# {title}\n\n{body}"

    data = answer_json(answer) or {}
    appendix: list[str] = []
    cover = _items(data, "jd_coverage")
    if cover:
        appendix += ["## JD 요구사항 대응표", "", "| ID | 요구사항 | 판정 | 근거(문서 원문) |", "|---|---|---|---|"]
        appendix += [f"| {_cell(x.get('id'))} | {_cell(x.get('requirement') or (requirements or {}).get(str(x.get('id'))))} | {_cell(x.get('status'))} | {_cell(x.get('quote')) or '-'} |"
                     for x in cover]
        gaps = [x for x in cover if str(x.get("status") or "").strip() == "공백"]
        if gaps:
            appendix += ["", "공백(문서에 근거가 없는 요구사항)은 지어내지 않았습니다. 실제 경험이 있다면 요약 문서에 더해 다시 작성하세요."]
        appendix.append("")
    changes = _items(data, "changes")
    if kind == "portfolio_update" and (changes or data.get("no_changes")):
        appendix += ["## 바뀐 점 (이전 → 이후)", ""]
        if changes:
            appendix += ["| # | 부분 | 이전 | 이후 | 이유 | 출처 |", "|---|---|---|---|---|---|"]
            appendix += [f"| {i} | {_cell(c.get('part'))} | {_cell(c.get('before')) or '(새로 추가)'} | {_cell(c.get('after'))} | "
                         f"{_cell(c.get('why'))} | {'🌐 ' if str(c.get('kind')).strip() == '외부' else '📄 '}{_cell(c.get('source'))} |"
                         for i, c in enumerate(changes, 1)]
        else:
            appendix.append(f"바뀐 점 없음 — {_cell(data.get('no_changes'))}")
        appendix.append("")
    if kind == "portfolio_update" and web:
        appendix += ["## 외부 자료 (용어·트렌드에만 사용 — 내 경력·숫자의 근거 아님)", ""]
        appendix += [f"- [W{i}] {_cell(w.get('title'))} — {w.get('url')}" for i, w in enumerate(web, 1)]
        appendix.append("")
    checks = [str(x) for x in (data.get("needs_check") or []) if str(x).strip()]
    if checks:
        appendix += ["## 확인 필요", "", *[f"- {c}" for c in checks], ""]
    evidence = _items(data, "evidence")
    if evidence:
        appendix += ["## 근거 인용 (올린 문서 원문 — 코드가 대조함)", "", "| 본문 주장 | 문서 원문 |", "|---|---|"]
        appendix += [f"| {_cell(x.get('claim'))} | {_cell(x.get('quote'))} |" for x in evidence]
        appendix.append("")
    if appendix:
        body += "\n\n---\n\n# 부록 — 검토용 (제출할 때는 빼세요)\n\n" + "\n".join(appendix)
    return body.strip() + "\n"


def _count_line(row: LimitRow | None) -> list[str]:
    if row is None:
        return []
    limit = f" · 제한 {row.limit:,}자({'공백 제외' if row.no_spaces else '공백 포함'}){' ✅' if row.ok else ' ⚠️ 초과'}" if row.limit else ""
    return ["", f"> 글자 수: 공백 포함 {row.with_spaces:,}자 · 공백 제외 {row.without_spaces:,}자{limit}", ""]
