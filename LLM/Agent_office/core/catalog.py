"""IT 개념 카탈로그 (GitHub jeong-yun/CS 저장소의 Term/Every_day) 읽기·쓰기.

카탈로그 구조 (Term/Every_day/)
  README.md                                          작성 조건 (분류·태그·형식 규칙)
  0. Summary.md                                      요약 (분류별 로드맵 + 한 줄 요약 표)
  Cloud & Infrastructure(클라우드 및 인프라)_Term.txt   분류 1
  Artificial Intelligence(AI 지능 및 시스템)_Term.txt    분류 2
  Security & Trust(보안 및 신뢰성)_Term.txt             분류 3
  basic_Term.txt                                     기초 용어 (`용어: 한 줄 정의`)
  old_term.txt                                       통합·삭제되어 빠진 항목 보관소

항목(Entry) = 제목 줄 `[태그]한글명(English) (작성일: YY.MM.DD)(수정일: …)` + 다음 제목 전까지의 본문.
파일 안에서는 최신 항목이 위에 온다 → 새 항목은 파일 맨 위에 넣는다.

개념 작성 페이지(pages/)는 여기 함수로
  관련 기존 항목 찾기(find_related) → LLM 팀 검토 → 사람 승인 → 반영(apply_plan) → 커밋·푸시(git_commit / git_push)
를 한다. 반영은 로컬 파일만 바꾸고, 푸시는 사람이 버튼을 눌러야 한다.
"""

import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

CATEGORY_FILES = {
    "Cloud": "Cloud & Infrastructure(클라우드 및 인프라)_Term.txt",
    "AI": "Artificial Intelligence(AI 지능 및 시스템)_Term.txt",
    "Security": "Security & Trust(보안 및 신뢰성)_Term.txt",
}
CATEGORY_TITLES = {  # 0. Summary.md 의 분류 제목
    "Cloud": "### 1. Cloud & Infrastructure(클라우드 및 인프라)",
    "AI": "### 2. Artificial Intelligence(AI 지능 및 시스템)",
    "Security": "### 3. Security & Trust(보안 및 신뢰성)",
}
BASIC_FILE, OLD_FILE, README_FILE, SUMMARY_FILE = "basic_Term.txt", "old_term.txt", "README.md", "0. Summary.md"
TAGS = ("Privacy", "AI-Sec", "Perf", "Gov")
TABLE_HEADER = "## 🛠️ 핵심 개념 한 줄 요약"

HEADER = re.compile(r"^\s*(?:\[(?P<tag>[^\]]+)\])?\s*(?P<name>.+?)\s*\((?:작성일|수정일)\s*:?\s*(?P<created>[^)]*)\)(?P<rest>.*)$")
UPDATED = re.compile(r"수정일\s*:?\s*([^)]*)\)")


@dataclass
class Entry:
    category: str  # Cloud | AI | Security | old
    header: str  # 제목 줄 원문
    body: str  # 제목 다음 줄부터 (끝 공백 줄 제외)
    tag: str = ""
    name: str = ""  # 한글명(English) — 태그·날짜 제외

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.body}".rstrip()

    @property
    def key(self) -> str:
        """비교용 이름 (공백·대소문자 무시)."""
        return normalize(self.name)

    def section(self, number: int) -> str:
        """본문의 'n. …' 절 내용 (없으면 빈 문자열)."""
        match = re.search(rf"^{number}\.\s.*?(?=^\d\.\s|\Z)", self.body, flags=re.MULTILINE | re.DOTALL)
        return match.group(0).strip() if match else ""


def normalize(text: str) -> str:
    return re.sub(r"[\s·\-_/]+", "", text).lower()


def is_header(line: str) -> bool:
    """항목 제목 줄인가 — 들여쓰지 않은(또는 공백 1칸) 줄에 '(작성일' 또는 '(수정일' 이 있음.

    초기 항목 중에는 작성일 없이 '(수정일: …)' 만 단 제목도 있다 (예: 자가 치유 보안 인프라).
    본문 불릿(' - ', '○')과 절 번호 줄('1. 개념 …')은 제외.
    """
    stripped = line.lstrip()
    return (
        ("(작성일" in line or "(수정일" in line)
        and not stripped.startswith(("-", "○", "①", "·", "*", "|"))
        and not re.match(r"\d+[.)]", stripped)
        and len(line) - len(stripped) <= 1
    )


def parse_header(line: str) -> tuple[str, str]:
    m = HEADER.match(line)
    if not m:
        return "", line.strip()
    return (m.group("tag") or "").strip(), m.group("name").strip()


def parse_entries(text: str, category: str) -> list[Entry]:
    entries: list[Entry] = []
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if is_header(line)]
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        tag, name = parse_header(lines[start])
        body = "\n".join(lines[start + 1 : end]).rstrip()
        entries.append(Entry(category=category, header=lines[start].strip(), body=body, tag=tag, name=name))
    return entries


@dataclass
class Catalog:
    root: Path  # .../CS/Term/Every_day
    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "Catalog":
        cat = cls(Path(root))
        for category, filename in CATEGORY_FILES.items():
            path = cat.root / filename
            if path.exists():
                cat.entries += parse_entries(path.read_text(encoding="utf-8"), category)
        return cat

    def by_category(self, category: str) -> list[Entry]:
        return [e for e in self.entries if e.category == category]

    def find(self, name: str) -> Entry | None:
        """이름(한글명(English) 또는 그 일부)으로 항목 찾기 — 정확히 같은 이름 우선."""
        key = normalize(name)
        exact = [e for e in self.entries if e.key == key]
        if exact:
            return exact[0]
        partial = [e for e in self.entries if key and (key in e.key or e.key in key)]
        return partial[0] if len(partial) == 1 else None

    def basic_terms(self) -> list[str]:
        path = self.root / BASIC_FILE
        return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


# ─── 관련 항목 찾기 (LLM 없이 — 비용·환각 없음) ────────────────────────────

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]+|[가-힣]{2,}")
STOP = {"기반", "및", "최적화", "아키텍처", "모델", "기술", "시스템", "the", "and", "based", "for", "of", "with", "optimization", "architecture", "large", "model", "models"}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text) if t.lower() not in STOP}


def acronyms(text: str) -> set[str]:
    """괄호 속 약어 (예: 'Process Reward Models, PRMs' → prms, prm)."""
    out = set()
    for a in re.findall(r"\b[A-Z][A-Za-z0-9\-]{1,9}\b", text):
        if sum(c.isupper() for c in a) >= 2:
            out |= {a.lower(), a.lower().rstrip("s")}
    return out


def find_related(catalog: Catalog, name: str, draft: str = "", limit: int = 6, min_score: float = 3.0) -> list[tuple[Entry, float]]:
    """새 개념과 관련 있어 보이는 기존 항목 (LLM 없이 — 비용·환각 없음).

    점수 = 이름 단어 겹침(드문 단어일수록 큰 가중치, IDF) + 약어 일치 + 본문에 약어 등장 + 초안의 핵심 단어.
    'dynamic·policy·sampling' 처럼 여러 항목 이름에 흔한 단어는 가중치가 작아 엉뚱한 추천이 줄어든다.
    """
    import math

    names = [tokens(e.name) for e in catalog.entries]
    df: dict[str, int] = {}
    for toks in names:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = max(len(names), 1)

    def idf(t: str) -> float:
        return math.log((n + 1) / (df.get(t, 0) + 1)) + 0.1

    name_tokens, name_acr = tokens(name), acronyms(name)
    draft_tokens, draft_acr = tokens(draft[:4000]), acronyms(draft[:4000]) - acronyms(name)
    scored = []
    for e, e_name_tokens in zip(catalog.entries, names):
        e_acr = acronyms(e.name)
        score = 2.0 * sum(idf(t) for t in name_tokens & e_name_tokens)
        score += 6.0 * len(name_acr & e_acr) + 2.0 * len((name_acr - e_acr) & tokens(e.body[:3000]))
        score += 0.5 * sum(idf(t) for t in draft_tokens & e_name_tokens) + 4.0 * len(draft_acr & e_acr)
        if normalize(name) and normalize(name) in e.key:
            score += 10
        if score >= min_score:
            scored.append((e, round(score, 2)))
    scored.sort(key=lambda x: -x[1])
    if scored:  # 1등과 점수 차이가 큰 약한 후보(흔한 단어 하나만 겹침)는 뺀다
        scored = [x for x in scored if x[1] >= 0.4 * scored[0][1]]
    return scored[:limit]


# ─── 기존 개념인가? (추가 / 수정 자동 판정) ─────────────────────────────


def name_keys(name: str) -> set[str]:
    """개념 이름의 비교용 열쇠 — 전체 이름, 한글명, 괄호 속 영문명·약어를 각각 (공백·기호 무시, 영문 복수형 s 제거).

    'RAG' 같은 약어는 괄호 안에 단독으로 있을 때만 열쇠가 된다 → '에이전틱 RAG' 가 'RAG' 로 오인되지 않음.
    """
    keys = {normalize(name), normalize(name.split("(")[0])}
    for part in re.findall(r"\(([^()]*)\)", name):
        keys |= {normalize(x) for x in re.split(r"[,/:]|\s-\s|·", part)}
    keys |= {k[:-1] for k in keys if re.fullmatch(r"[a-z0-9\-]{4,}s", k)}
    return {k for k in keys if len(k) >= 3}


@dataclass
class ExistingMatch:
    entry: Entry  # 수정할 항목 (하위 개념이면 그것을 품은 부모 항목)
    kind: str  # "항목" = 독립 항목으로 있음 / "하위" = 다른 항목의 7.변형·확장 상세 안에 있음
    detail: str = ""  # 하위 개념이면 그 소절 제목


SUBTITLE = re.compile(r"^\s*(?:-\s*)?(?:\d+\)\s*)?(.+?)(?:\s+—.*|:.*)?$")


def find_existing(catalog: Catalog, name: str) -> list[ExistingMatch]:
    """새로 쓰려는 개념이 카탈로그에 이미 있는지 — 있으면 '수정', 없으면 '추가'.

    1) 독립 항목의 이름(한글·영문·약어)이 같으면 그 항목
    2) 없으면 다른 항목의 '7. 변형·확장 상세' 소절 제목에 있는지 (통합으로 편입된 개념) → 그 부모 항목
    """
    keys = name_keys(name)
    if not keys:
        return []
    direct = [ExistingMatch(e, "항목") for e in catalog.entries if keys & name_keys(e.name)]
    if direct:
        return direct
    found = []
    for e in catalog.entries:
        section = re.search(r"^\s?\d\.\s*(?:변형|확장|상세).*?(?=^\s?(?:\d\.\s|참고)|\Z)", e.body, flags=re.MULTILINE | re.DOTALL)
        if not section:
            continue
        for line in section.group(0).splitlines()[1:]:
            m = SUBTITLE.match(line)
            if m and not line.lstrip().startswith(("○", "●", "·")) and keys & name_keys(m.group(1)):
                found.append(ExistingMatch(e, "하위", m.group(1).strip()))
                break
    return found


def keep_dates(new_header: str, old_header: str) -> str:
    """수정본 제목 줄의 날짜 부분을 기존 항목의 것(작성일·수정일 이력)으로 되돌린다 — 모델이 작성일을 오늘로 바꿔 써도 보존."""
    old = re.search(r"\((?:작성일|수정일)", old_header)
    if not old:
        return new_header
    new = re.search(r"\((?:작성일|수정일)", new_header)
    prefix = (new_header[: new.start()] if new else new_header).rstrip()
    return f"{prefix} {old_header[old.start():].strip()}"


def mentions(catalog: Catalog, entry: Entry) -> list[Entry]:
    """다른 항목 본문에서 이 항목을 언급하는 곳 (삭제·이관 시 같이 고쳐야 할 후보)."""
    keys = {k for k in acronyms(entry.name)} | {entry.name.split("(")[0].strip().lower()}
    keys = {k for k in keys if len(k) >= 3}
    return [e for e in catalog.entries if e is not entry and any(k in e.body.lower() for k in keys)]


# ─── 반영 (적용 계획 → 파일 수정) ────────────────────────────────────────


@dataclass
class Change:
    path: str
    action: str  # 추가 | 교체 | 삭제 | 이관 | 요약 | 기초 용어
    detail: str


def today_label(d: date | None = None) -> str:
    d = d or date.today()
    return f"{d:%y.%m.%d}"


def with_updated_date(header: str, d: date | None = None) -> str:
    """제목 줄에 수정일 추가 — '(수정일: a, b)' 가 있으면 끝에 덧붙이고 없으면 새로 단다."""
    label = today_label(d)
    if label in header:
        return header
    match = UPDATED.search(header)
    if match and re.search(rf"(^|[,\s]){re.escape(label[3:])}\s*$", match.group(1).strip()):
        return header  # 같은 해의 짧은 표기(10.20)로 이미 들어 있음
    if match:
        return header[: match.end(1)] + f", {label[3:] if match.group(1).strip()[:2] == label[:2] else label}" + header[match.end(1):]
    return f"{header}(수정일: {label})"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else ""


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def remove_entry(text: str, entry: Entry) -> str:
    """파일 텍스트에서 항목 하나(제목~다음 제목 전)를 지운다."""
    entries = parse_entries(text, entry.category)
    kept = [e.text for e in entries if not (e.header == entry.header)]
    return "\n\n".join(kept)


def replace_entry(text: str, entry: Entry, new_text: str) -> str:
    entries = parse_entries(text, entry.category)
    return "\n\n".join(new_text.strip() if e.header == entry.header else e.text for e in entries)


def prepend_entry(text: str, new_text: str) -> str:
    return new_text.strip() + ("\n\n" + text.strip() if text.strip() else "")


def summary_bullet(plan: dict) -> str:
    s = plan.get("summary") or {}
    tag = f"[{plan['tag']}]" if plan.get("tag") else ""
    return f"* **{tag}{s.get('name') or plan['title']}:** {s.get('one_line', '').strip()}"


def summary_row(plan: dict) -> str:
    s = plan.get("summary") or {}
    return f"| **{s.get('name') or plan['title']}** | {s.get('keywords', '')} | {s.get('one_line', '').strip()} |"


def _summary_name(entry_or_name) -> str:
    name = entry_or_name.name if isinstance(entry_or_name, Entry) else str(entry_or_name)
    m = re.search(r"\(([^()]*)\)\s*$", name)  # 영문명이 괄호 안에 있으면 그것으로 요약을 찾는다
    return (m.group(1) if m else name).split(",")[0].strip()


def update_summary(text: str, plan: dict, removed: list[Entry]) -> str:
    """0. Summary.md — 로드맵 불릿과 한 줄 요약 표 행을 추가·교체·삭제한다."""
    lines = text.split("\n")
    drop_keys = {normalize(_summary_name(e)) for e in removed}
    if plan.get("action") in ("update", "delete") and plan.get("target"):
        drop_keys.add(normalize(_summary_name(plan["target"])))

    def mentions_dropped(line: str) -> bool:
        m = re.match(r"^\* \*\*(?:\[[^\]]+\])?(.+?):\*\*", line) or re.match(r"^\| \*\*(.+?)\*\* \|", line)
        return bool(m) and normalize(m.group(1)) in drop_keys

    lines = [l for l in lines if not mentions_dropped(l)]
    if plan.get("action") in ("add", "update") and plan.get("summary"):
        heading = CATEGORY_TITLES[plan["category"]]
        if heading in lines:
            i = lines.index(heading) + 1
            while i < len(lines) and lines[i].startswith("* "):
                i += 1
            lines.insert(i, summary_bullet(plan))
        if TABLE_HEADER in lines:
            i = lines.index(TABLE_HEADER) + 1
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            lines.insert(i, summary_row(plan))
    return "\n".join(lines)


def add_basic_terms(text: str, terms: list[dict]) -> tuple[str, list[str]]:
    existing = {normalize(l.split(":")[0]) for l in text.splitlines() if ":" in l}
    added = []
    for t in terms:
        term, definition = str(t.get("term", "")).strip(), str(t.get("definition", "")).strip()
        if term and definition and normalize(term) not in existing:
            added.append(f"{term}: {definition}")
            existing.add(normalize(term))
    if not added:
        return text, []
    return text.rstrip("\n") + "\n" + "\n".join(added), added


def apply_plan(catalog: Catalog, plan: dict, final_entry: str, d: date | None = None) -> list[Change]:
    """승인된 결과를 카탈로그 파일에 반영한다 (로컬 파일만 — 커밋·푸시는 따로).

    plan = {"action": "add|update|delete", "category": "AI|Cloud|Security", "tag": "...", "title": "...",
            "target": "수정·삭제할 기존 항목 이름", "absorb": ["편입·병합되어 빠질 기존 항목 이름", ...],
            "summary": {"name", "keywords", "one_line"}, "basic_terms": [{"term", "definition"}]}
    - update: 대상 항목을 새 본문으로 교체 (분류가 바뀌면 옛 파일에서 빼고 새 파일 맨 위로) — 제목에 수정일 추가
    - delete: 대상 항목을 old_term.txt 맨 위로 옮김
    - absorb: 흡수된 항목을 old_term.txt 로 옮김 (원문 그대로 — 무엇을 통합했는지 추적용)
    """
    root, changes = catalog.root, []
    action, category = plan["action"], plan.get("category")
    old_path = root / OLD_FILE
    old_text = _read(old_path)
    removed: list[Entry] = []

    target = catalog.find(plan["target"]) if plan.get("target") else None
    if action in ("update", "delete") and target is None:
        raise ValueError(f"수정·삭제할 기존 항목을 찾지 못했습니다: {plan.get('target')!r}")

    if action == "delete":
        path = root / CATEGORY_FILES[target.category]
        _write(path, remove_entry(_read(path), target))
        old_text = prepend_entry(old_text, target.text)
        removed.append(target)
        changes.append(Change(path.name, "삭제", target.name))
        changes.append(Change(OLD_FILE, "이관", f"{target.name} → old_term"))
    else:
        if category not in CATEGORY_FILES:
            raise ValueError(f"분류는 {list(CATEGORY_FILES)} 중 하나여야 합니다: {category!r}")
        entry_text = final_entry.strip()
        new_path = root / CATEGORY_FILES[category]
        if action == "update":
            first, _, rest = entry_text.partition("\n")
            entry_text = f"{with_updated_date(keep_dates(first, target.header), d)}\n{rest}".rstrip()
            old_file = root / CATEGORY_FILES[target.category]
            if target.category == category:
                _write(new_path, replace_entry(_read(new_path), target, entry_text))
                changes.append(Change(new_path.name, "교체", target.name))
            else:  # 분류 이동
                _write(old_file, remove_entry(_read(old_file), target))
                _write(new_path, prepend_entry(_read(new_path), entry_text))
                changes.append(Change(old_file.name, "삭제", f"{target.name} (분류 이동)"))
                changes.append(Change(new_path.name, "추가", f"{target.name} (분류 이동)"))
        else:
            _write(new_path, prepend_entry(_read(new_path), entry_text))
            changes.append(Change(new_path.name, "추가", entry_text.partition("\n")[0]))

    for name in plan.get("absorb") or []:
        entry = Catalog.load(root).find(name)  # 위에서 파일이 바뀌었을 수 있어 다시 읽는다
        if entry is None:
            changes.append(Change(OLD_FILE, "건너뜀", f"흡수 대상 '{name}' 을 찾지 못함"))
            continue
        path = root / CATEGORY_FILES[entry.category]
        _write(path, remove_entry(_read(path), entry))
        old_text = prepend_entry(old_text, entry.text)
        removed.append(entry)
        changes.append(Change(path.name, "삭제", f"{entry.name} (통합)"))
        changes.append(Change(OLD_FILE, "이관", f"{entry.name} → old_term"))
    if removed:
        _write(old_path, old_text)

    summary_path = root / SUMMARY_FILE
    if summary_path.exists():
        plan_for_summary = {**plan, "target": target.name if target else None}
        new_summary = update_summary(_read(summary_path), plan_for_summary, removed)
        if new_summary != _read(summary_path):
            _write(summary_path, new_summary)
            changes.append(Change(SUMMARY_FILE, "요약", "로드맵·한 줄 요약 갱신"))

    if plan.get("basic_terms"):
        basic_path = root / BASIC_FILE
        new_text, added = add_basic_terms(_read(basic_path), plan["basic_terms"])
        if added:
            _write(basic_path, new_text)
            changes.append(Change(BASIC_FILE, "기초 용어", ", ".join(a.split(":")[0] for a in added)))
    return changes


# ─── git (CS 저장소) ────────────────────────────────────────────────────


def repo_root(path: Path) -> Path | None:
    """path 가 들어 있는 git 저장소의 최상위 폴더."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def git(repo: Path, *args: str, timeout: int = 60) -> tuple[bool, str]:
    out = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=repo, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
    return out.returncode == 0, (out.stdout + out.stderr).strip()


def git_status(repo: Path, subdir: str) -> str:
    return git(repo, "status", "--short", "--", subdir)[1]


def git_diff(repo: Path, subdir: str) -> str:
    return git(repo, "diff", "--", subdir)[1]


def git_commit(repo: Path, subdir: str, message: str) -> tuple[bool, str]:
    ok, log = git(repo, "add", "--", subdir)
    if not ok:
        return ok, log
    return git(repo, "commit", "-m", message)


def git_push(repo: Path) -> tuple[bool, str]:
    return git(repo, "push", timeout=120)


def git_pull(repo: Path) -> tuple[bool, str]:
    return git(repo, "pull", "--ff-only", timeout=120)
