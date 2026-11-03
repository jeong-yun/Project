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


def body_text(answer: str) -> str:
    """JSON·코드 블록을 뺀 본문."""
    return re.sub(r"```.*?```", "", answer, flags=re.S).strip()


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


def unsourced_numbers(body: str, allowed: set[str]) -> list[str]:
    """본문 숫자 중 원문·JD·문항에 없는 것 ('[확인 필요]'가 붙은 문장은 건너뛴다)."""
    out: list[str] = []
    for sent in _sentences(body):
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
    for item in [*_items(data, "evidence"), *_items(data, "jd_coverage")]:
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


# ─── 내려받기용 문서 ────────────────────────────────────────────────────


def render_document(answer: str, kind: str, questions: list[Question] | None = None, title: str = "",
                    requirements: dict[str, str] | None = None) -> str:
    """최종 답변 → 내려받기용 마크다운.

    본문(JSON 빼고) + 자소서면 문항마다 코드가 센 글자 수 + 부록(JD 대응표 · 확인 필요 · 근거 인용 — JSON 에서 코드가 만든 표).
    """
    body = body_text(answer)
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
