"""개념 작성 — 요청 만들기(관련 기존 항목 포함) · 결과 나누기(검증 로그/최종본/요약/적용 계획) · 형식 검사.

결과 형식 (개념 편집 팀이 지켜야 하는 출력 계약 — 코드가 이 표시로 잘라서 파일에 반영한다)
  <review_log> 1단계 비교 표·이중 해리 + 2단계 검증 로그 </review_log>
  (원문 프롬프트의 thought 태그는 Claude 모델 안전 필터가 요청 자체를 막아(2026-10-27 실측) 이름만 바꿨다 — 예전 결과의 thought 태그도 읽는다)
  <final> 카탈로그에 그대로 들어갈 항목 텍스트 (제목 줄부터 참고문헌까지) </final>
  ## GitHub 요약 …
  ```json {"concept_plan": {...}} ```   ← 적용 계획 (분류·태그·추가/수정·통합 대상·요약·기초 용어·다른 항목 수정)
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache

from core.brief import DesiredState
from core.catalog import CATEGORY_FILES, TAGS, Catalog, Entry, is_header, today_label, with_updated_date
from core.config import TEAMS_DIR, load_skills
from core.harness.base import extract_code_blocks

ACTIONS = {"add": "추가", "update": "수정", "delete": "삭제"}
LOG_TAG, OLD_LOG_TAG = "review_log", "thought"  # 검증 로그 태그 (예전 결과는 thought)
ABSOLUTE = ("완벽", "100%", "무조건", "완전 차단", "원천 차단", "절대적으로", "무한한")
REF_LABELS = ("논문·핵심기법", "논문·서베이", "논문·한계·반박", "논문·벤치마크", "표준/거버넌스", "도구/프레임워크")
# 카탈로그에는 '논문·한계·분석'·'논문·응용' 같은 논문 세부 라벨도 있다 → '논문·' 으로 시작하면 받는다
REF_LABEL = re.compile(r"^-\s*\[(?:" + "|".join(map(re.escape, REF_LABELS)) + r"|논문·[^\]]+)\]")
MAX_NARRATIVE = 2  # 서술식 종결은 이 줄 수까지 봐준다 (기준 항목 「플래시어텐션」에도 2줄 있음)
SUB_ITEM_LABELS = ("개념", "작동원리", "장점", "단점", "예시")  # 7절 하위 항목마다 있어야 하는 줄
PLACEHOLDER = re.compile(r"arXiv:\s*\d{4}\.\d*x+|\bxxxx\b|example\.com|<URL>|\(URL\)", re.IGNORECASE)
# 서술식 종결 (개조식 위반) — 줄 끝이 '~다.' 로 끝나는 경우
NARRATIVE_END = re.compile(r"(다|니다|었다|였다|한다|된다|이다)\.?\s*$")

PLAN_EXAMPLE = {
    "concept_plan": {
        "action": "add",
        "category": "AI",
        "tag": "Perf",
        "title": "한글명(English, 약어)",
        "target": "",
        "absorb": [],
        "summary": {"emoji": "🧠", "name": "English Name(약어)", "keywords": "핵심 키워드", "one_line": "개조식 한 줄 정의(비교·한계 포함)"},
        "basic_terms": [{"term": "용어(English)", "definition": "한 줄 정의"}],
        "edits": [{"entry": "수정할 다른 기존 항목 이름", "find": "그 항목 본문에 있는 바꿀 문장 그대로", "replace": "교체문", "reason": "이유"}],
    }
}


SECTION_TITLES = {  # 절 제목 표준형 (기준 항목 「플래시어텐션」) — 모델이 '1.정의및개념'처럼 붙여 쓴 경우에만 이것으로 맞춘다
    1: "1. 정의 및 개념", 2: "2. 작동 원리", 3: "3. 좋아진 점", 4: "4. 문제점 및 한계",
    5: "5. 사용 예시", 6: "6. 관계 (기존 개념 비교)", 7: "7. 변형·확장 상세",
}
SECTION_KEYWORD = re.compile(r"정의|작동|좋아진|문제점|사용|관계|변형|확장")
# 절 제목 줄 — 들여쓰지 않은 'N. 제목' ('2.5배' 같은 숫자나 '1) 하위 항목'은 아님)
SECTION_LINE = re.compile(r"^#{0,4}\s*(\d)\s*\.(?!\d)\s*(\S.*)$")
REF_TITLE = re.compile(r"^\s*#{0,4}\s*(?:(\d)\s*[.)]\s*)?참고\s*문헌\s*$")
CATEGORY_PREFIX = re.compile(r"^\s*\[\s*\d\s*\.\s*(Cloud[^\]]*|AI[^\]]*|Security[^\]]*)\]\s*")


# 🔧수정 / ➕추가 표시는 '변경 요약'(사람이 확인하는 용도)이지 카탈로그 본문이 아니다 → 최종본에서 뺀다
NOTE_LINE = re.compile(r"^\s*(?:[-○●·*]\s*)?[🔧➕]\s*(?:수정|추가|삭제)\s*[:：]")
NOTE_PAREN = re.compile(r"\s*[🔧➕]\s*\*?\((?:초안|기존 독립 항목|원문)[^)]*\)\*?")  # 예: 🔧*(초안 서술 정반대 교정)*
NOTE_SENTENCE = re.compile(r"\s*[🔧➕]\s*초안[^.。]*[.。]")  # 예: 🔧 초안의 "완벽히 보완적"은 절대 표현 → 완화.
NOTE_MARK = re.compile(r"[🔧➕]\s*(\*?)")  # 내용 앞에 붙은 표시만 — 내용은 남긴다


def strip_change_notes(entry: str) -> tuple[str, list[str]]:
    """최종본에서 변경 표시(🔧수정·➕추가 + 이유)를 떼어 낸다. (깨끗한 본문, 떼어 낸 변경 요약 목록)

    - '🔧수정: … / 이유: …' 같은 변경 메모 줄 → 통째로 빼서 변경 요약으로
    - '🔧*(초안 서술 정반대 교정)*', '🔧 초안의 … → 완화.' 같은 메모 구절 → 빼서 변경 요약으로
    - '🔧 단, 효과는 …' 처럼 내용 앞에 붙은 표시 → 표시만 지우고 내용은 그대로
    """
    notes, out = [], []
    for line in entry.split("\n"):
        if "🔧" not in line and "➕" not in line:
            out.append(line)  # 표시가 없는 줄은 그대로 (공백 하나도 바꾸지 않음)
            continue
        if NOTE_LINE.match(line):
            notes.append(re.sub(r"^\s*(?:[-○●·*]\s*)?", "", line).strip())
            continue
        for pattern in (NOTE_PAREN, NOTE_SENTENCE):
            for m in pattern.finditer(line):
                notes.append(m.group(0).strip())
            line = pattern.sub("", line)
        line = NOTE_MARK.sub("", line)
        line = re.sub(r"\(([^()]*)\)\*", r"(\1)", line)  # '🔧*(…)*' 에서 남은 기울임 표시 정리
        out.append(line.rstrip())
    return "\n".join(out), notes


def normalize_entry(entry: str) -> str:
    """LLM 이 쓴 최종본을 카탈로그 파일 형식으로 맞춘다 (결정론적 — 내용은 바꾸지 않음).

    - 제목 앞의 분류 표시('[2.AI] ') 제거, 태그 뒤 공백 제거('[Perf] X' → '[Perf]X')
    - 붙여 쓴 절 제목 '1.정의및개념' → '1. 정의 및 개념' (띄어 쓴 제목은 그대로 — 기존 항목의 절 제목 유지)
    - 번호 없는 '참고 문헌' → 마지막 절 다음 번호 ('8. 참고 문헌' — 카탈로그 형식)
    - 마크다운 굵게(**) 제거, 맨 앞 불릿 '- '·'· '·'* ' → ' - ', 맨 앞 ⓐ·①·● → ' ⓐ', 맨 앞 '1)' → '   1)', 빈 줄 제거
    - 들여쓴 줄('   ① 문제:', '     - 개념:', '       ●')은 들여쓰기·표식 그대로 (출력 형식 견본)
    """
    lines = [l.rstrip() for l in entry.replace("\r\n", "\n").split("\n")]
    lines = [l for l in lines if l.strip()]
    if not lines:
        return ""
    header = CATEGORY_PREFIX.sub("", lines[0].replace("**", "").strip())
    header = re.sub(r"^(\[[^\]]+\])\s+", r"\1", header)
    out, last_section, in_refs = [header], 0, False
    for line in lines[1:]:
        text = line.replace("**", "")
        ref = REF_TITLE.match(text)
        if ref:
            out.append(f"{ref.group(1) or last_section + 1}. 참고 문헌")
            in_refs = True
            continue
        section = SECTION_LINE.match(text)
        if section:
            n, title = int(section.group(1)), section.group(2).strip()
            glued = " " not in title and len(title) > 3 and SECTION_KEYWORD.match(title)
            out.append(SECTION_TITLES[n] if glued and n in SECTION_TITLES else f"{n}. {title}")
            last_section = n
            continue
        bullet = re.match(r"^(\s*)[-*•·]\s+(.*)$", text)
        if bullet:
            indent, rest = bullet.groups()
            out.append(f"{indent or ' '}- {rest}")
        elif in_refs and text.startswith("["):
            out.append(" - " + text)  # 참고문헌 한 줄: '[라벨] 저자. 제목 …' → ' - [라벨] …'
        elif re.match(r"^[ⓐ-ⓩ①-⑳●○※]", text):
            out.append(" " + text)  # 작동 원리 'ⓐ …' 등 표식 줄은 한 칸 들여쓰기
        elif re.match(r"^\d+\)\s", text):
            out.append("   " + text)  # 7절 하위 항목 '1) 이름'
        elif re.match(r"^\S", text) and not text.startswith(("[", "(")):
            out.append(" - " + text)  # 들여쓰기 없는 일반 문장(예: '등장 배경')은 불릿으로
        else:
            out.append(text)
    return "\n".join(out)


@dataclass
class ConceptResult:
    thought: str = ""
    final_entry: str = ""
    summary_md: str = ""
    plan: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    change_notes: list[str] = field(default_factory=list)  # 최종본에서 떼어 낸 🔧수정·➕추가 표시와 이유 (화면에만)


def _between(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def replace_final(answer: str, new_entry: str) -> str:
    """답변의 <final>…</final> 내용을 (사람이 고친) 최종본으로 바꾼다 — 채택 기록에 실제 반영본이 남도록."""
    pattern = re.compile(r"(<final>).*?(</final>)", flags=re.DOTALL | re.IGNORECASE)
    if not pattern.search(answer):
        return answer
    return pattern.sub(lambda m: f"{m.group(1)}\n{new_entry.strip()}\n{m.group(2)}", answer, count=1)


def parse_result(answer: str) -> ConceptResult:
    entry, notes = strip_change_notes(normalize_entry(_between(answer, "final")))
    log = _between(answer, LOG_TAG) or _between(answer, OLD_LOG_TAG)
    res = ConceptResult(thought=log, final_entry=entry, change_notes=notes)
    after = re.split(r"</final>", answer, flags=re.IGNORECASE)[-1] if "</final>" in answer.lower() else answer
    res.summary_md = re.sub(r"```json.*?```", "", after, flags=re.DOTALL).strip()
    for src in extract_code_blocks(answer, "json"):
        try:
            data = json.loads(src)
        except json.JSONDecodeError as e:
            res.problems.append(f"적용 계획 JSON 파싱 오류: {e.msg}")
            continue
        if isinstance(data, dict) and "concept_plan" in data:
            res.plan, fixes = normalize_plan(data["concept_plan"])
            res.problems += fixes
    return res


NAME_KEYS = ("name", "title", "term", "entry", "target", "항목", "이름")  # 모델이 이름을 객체로 감쌀 때 쓰는 키


def _as_name(value) -> str:
    """항목 이름 하나 — 문자열·숫자는 그대로, {"name": …} 같은 객체는 이름 칸을 꺼낸다. 알 수 없으면 ""."""
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    if isinstance(value, dict):
        return next((n for n in (_as_name(value.get(k)) for k in NAME_KEYS) if n), "")
    return ""


def _as_list(value) -> list:
    return [] if value in (None, "") else value if isinstance(value, list) else [value]


def normalize_plan(plan) -> tuple[dict, list[str]]:
    """모델이 쓴 concept_plan 의 칸 모양을 코드가 쓰는 모양으로 맞춘다. (맞춘 계획, 고쳐 달라고 알릴 문제 목록)

    absorb 를 이름 목록이 아니라 문자열 하나·객체 목록·null 섞인 목록으로 쓰면 화면·반영이 멈췄다 (2026-10-27).
    알아볼 수 있는 값은 고쳐서 쓰고, 모양이 틀렸다는 것은 문제로 남겨 하네스가 다시 쓰게 한다.
    """
    if not isinstance(plan, dict):
        return {}, [f"concept_plan 은 {{...}} 객체여야 한다 (현재 {type(plan).__name__})"]
    plan, problems = dict(plan), []
    for key in ("action", "category", "tag", "title", "target"):
        if key in plan and not isinstance(plan[key], str) and plan[key] is not None:
            name = _as_name(plan[key]) or (_as_name(plan[key][0]) if isinstance(plan[key], list) and len(plan[key]) == 1 else "")
            problems.append(f"concept_plan.{key} 는 문자열 하나로 쓴다 (현재 {plan[key]!r})")
            plan[key] = name
    if "absorb" in plan:
        names, bad = [], False
        for item in _as_list(plan["absorb"]):
            name = _as_name(item)
            bad |= not isinstance(item, str) or not name
            if name and name not in names:
                names.append(name)
        if bad or not isinstance(plan["absorb"], list) and plan["absorb"] not in (None, ""):
            problems.append(f'concept_plan.absorb 는 흡수될 기존 항목 이름(문자열) 목록으로 쓴다 — 예: ["항목 이름"] (현재 {plan["absorb"]!r})')
        plan["absorb"] = names
    if "edits" in plan:
        edits = []
        for item in _as_list(plan["edits"]):
            if not isinstance(item, dict):
                problems.append(f"concept_plan.edits 의 각 칸은 {{entry, find, replace, reason}} 객체여야 한다 (현재 {item!r})")
                continue
            edit = {**item, "entry": _as_name(item.get("entry")) or _as_name(item)}
            edit.update({k: "" if item.get(k) is None else str(item[k]) for k in ("find", "replace", "reason")})
            if not edit["entry"] or not edit["find"]:
                problems.append(f"concept_plan.edits 에는 수정할 항목 이름(entry)과 바꿀 문장(find)이 있어야 한다 (현재 {item!r})")
            edits.append(edit)
        if not isinstance(plan["edits"], list) and plan["edits"] not in (None, ""):
            problems.append("concept_plan.edits 는 [{entry, find, replace, reason}, …] 목록으로 쓴다")
        plan["edits"] = edits
    if "summary" in plan and not isinstance(plan["summary"], dict):
        if plan["summary"] not in (None, ""):
            problems.append(f"concept_plan.summary 는 {{emoji, name, keywords, one_line}} 객체여야 한다 (현재 {plan['summary']!r})")
        plan["summary"] = {}
    elif isinstance(plan.get("summary"), dict):
        plan["summary"] = {k: ", ".join(map(str, v)) if isinstance(v, list) else "" if v is None else str(v)
                           for k, v in plan["summary"].items()}
    if "basic_terms" in plan:
        terms = []
        for item in _as_list(plan["basic_terms"]):
            if isinstance(item, dict):
                terms.append(item)
            elif isinstance(item, str) and ":" in item:
                term, _, definition = item.partition(":")
                terms.append({"term": term.strip(), "definition": definition.strip()})
            else:
                problems.append(f"concept_plan.basic_terms 의 각 칸은 {{term, definition}} 객체여야 한다 (현재 {item!r})")
        plan["basic_terms"] = terms
    return plan, problems


def check_answer(answer: str, options: dict | None = None) -> list[str]:
    """결과 형식 검사 (하네스). 틀린 곳 목록 — 비어 있으면 통과."""
    res = parse_result(answer)
    problems = list(res.problems)
    lower = answer.lower()
    if not res.thought:
        problems.append(f"<{LOG_TAG}> … </{LOG_TAG}> (1단계 비교·2단계 검증 로그)가 없다")
    elif max(lower.find(f"<{LOG_TAG}>"), lower.find(f"<{OLD_LOG_TAG}>")) > lower.find("<final>") >= 0:
        problems.append(f"<{LOG_TAG}> 는 <final> 보다 먼저 와야 한다 (검증 로그 먼저)")
    plan = res.plan
    if not plan:
        problems.append('```json {"concept_plan": {...}} 적용 계획 블록이 없다')
        return problems
    action = plan.get("action")
    if action not in ACTIONS:
        problems.append(f"concept_plan.action 은 add/update/delete 중 하나 (현재 {action!r})")
    if action != "delete":
        if plan.get("category") not in CATEGORY_FILES:
            problems.append(f"concept_plan.category 는 {list(CATEGORY_FILES)} 중 하나 (현재 {plan.get('category')!r})")
        if plan.get("tag") and plan["tag"] not in TAGS:
            problems.append(f"concept_plan.tag 는 {list(TAGS)} 중 하나 또는 빈 문자열 (현재 {plan.get('tag')!r})")
        summary = plan.get("summary") or {}
        if not all(str(summary.get(k, "")).strip() for k in ("name", "keywords", "one_line")):
            problems.append("concept_plan.summary 에 name·keywords·one_line 이 모두 있어야 한다")
        problems += check_entry(res.final_entry, (options or {}).get("concept_header", "") if action == "update" else "")
    if action in ("update", "delete") and not str(plan.get("target", "")).strip():
        problems.append("수정·삭제는 concept_plan.target 에 기존 항목 이름을 적어야 한다")
    return problems


def header_dates(header: str) -> str:
    """제목 줄의 날짜 부분 '(작성일: …)(수정일: …)'."""
    m = re.search(r"\((?:작성일|수정일).*$", header)
    return m.group(0).strip() if m else ""


def check_entry(entry: str, expected_header: str = "") -> list[str]:
    """최종본(<final>) 형식 검사 — 제목 줄·부제·절 구성·등장 배경·관계 짝·7절 하위 항목·개조식·절대 표현·참고문헌 형식.

    expected_header: 수정(update)일 때 코드가 정한 제목 줄 (기존 작성일·수정일 이력 + 오늘 수정일) — 날짜 부분이 같아야 한다.
    """
    if not entry:
        return ["<final> … </final> 최종본이 없다"]
    problems = []
    lines = entry.splitlines()
    if not is_header(lines[0]):
        problems.append("최종본 첫 줄은 '[태그]한글명(English) (작성일: YY.MM.DD)' 제목 줄이어야 한다")
    elif expected_header and header_dates(lines[0]) != header_dates(expected_header):
        problems.append(f"수정본 제목 줄의 날짜는 기존 이력을 그대로 두고 오늘 수정일만 덧붙인다 → '{header_dates(expected_header)}'"
                        f" (현재 '{header_dates(lines[0])}')")
    subtitle = [i for i, l in enumerate(lines) if re.match(r"^\s*-\s*부제\s*:", l)]
    if subtitle and subtitle[0] != 1:
        problems.append("' - 부제: …' 줄은 제목 줄 바로 다음 줄에 둔다")
    for n in range(1, 7):
        if not re.search(rf"^{n}\.\s*\S", entry, flags=re.MULTILINE):
            problems.append(f"{n}번 절이 없다 (1.정의및개념 ~ 6.관계 필수)")
    section1 = re.split(r"^2\.\s*\S", entry, maxsplit=1, flags=re.MULTILINE)[0]  # 'Qwen2.5' 같은 본문 숫자에 걸리지 않게 줄 첫머리만
    if "배경" not in section1:
        problems.append("1번(정의 및 개념)에 '등장 배경'(문제 → 원인 → 처방) 2~4줄이 없다")
    relation = re.search(r"^6\.\s.*?(?=^\d\.\s|^\S*참고\s*문헌|\Z)", entry, flags=re.MULTILINE | re.DOTALL)
    if relation and not ("유사" in relation.group(0) and "차이" in relation.group(0)):
        problems.append("6.관계는 대상마다 '유사: … / 차이: … / 경계: …' 짝으로 써야 한다")
    problems += check_sub_items(entry)
    found = [w for w in ABSOLUTE if w in entry]
    if found:
        problems.append(f"절대 표현 금지: {found} → '효과적으로 억제·최소화' 같은 객관적 표현으로")
    if PLACEHOLDER.search(entry):
        problems.append(f"플레이스홀더 인용이 있다: {PLACEHOLDER.search(entry).group(0)!r} — 실제 식별자로 고치거나 빼라")
    narrative = [l.strip()[:40] for l in lines[1:] if NARRATIVE_END.search(l.strip()) and not l.strip().startswith(("\"", "“", "'"))]
    if len(narrative) > MAX_NARRATIVE:
        problems.append(f"개조식(명사형 종결) 위반 {len(narrative)}줄: {narrative[:3]} … → '~함/~됨/~임/명사구'로")
    ref_start = next((i for i, l in enumerate(lines) if re.match(r"^\s*(\d\.\s*)?참고\s*문헌", l)), None)
    if ref_start is None:
        problems.append("참고문헌 절이 없다 (마지막 절 다음 번호로 '8. 참고 문헌')")
    else:
        refs = [l.strip() for l in lines[ref_start + 1 :] if l.strip().startswith("-")]
        bad = [r[:50] for r in refs if not (REF_LABEL.match(r) and "http" in r)]
        if not refs:
            problems.append("참고문헌 항목이 없다")
        elif bad:
            problems.append(f"참고문헌 형식 위반 {len(bad)}건 (라벨 {list(REF_LABELS)} 중 하나·URL 필수): {bad[:2]}")
    return problems


def check_sub_items(entry: str) -> list[str]:
    """7. 변형·확장 상세의 하위 항목('1) 이름')마다 개념·작동원리·장점·단점(고유)·예시 줄이 있는지."""
    section = re.search(r"^7\.\s*(?:변형|확장).*?(?=^\d\.\s|^\S*참고\s*문헌|\Z)", entry, flags=re.MULTILINE | re.DOTALL)
    if not section:
        return []
    problems = []
    for item in re.split(r"^(?=\s*\d+\)\s)", section.group(0), flags=re.MULTILINE)[1:]:
        title = item.strip().splitlines()[0][:30]
        labels = {re.sub(r"\s", "", m) for m in re.findall(r"^\s*[-○]\s*(개념|작동\s*원리|장점|단점|예시)", item, flags=re.MULTILINE)}
        missing = [label for label in SUB_ITEM_LABELS if label not in labels]
        if missing:
            problems.append(f"7절 하위 항목 '{title}' 에 빠진 줄: {missing} — '- 개념: / - 작동원리: / - 장점: / - 단점(고유): / - 예시:'")
    return problems


# ─── 적용된 프롬프트 판(버전) ────────────────────────────────────────────

TEAM_ID = "concept"
PROMPT_LINE = "- 적용 프롬프트: "  # 요청 [입력] 맨 끝 줄 — 질문과 함께 저장되므로 지난 결과도 어느 판으로 만들었는지 안다
PROMPT_VERSION = re.compile(r"^- 적용 프롬프트: .*?\bv([0-9a-f]{8})\b.*$", re.MULTILINE)


def _git(folder, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, encoding="utf-8", timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


@lru_cache(maxsize=8)
def _version(skills_dir: str, sig: tuple) -> dict:
    text = load_skills(TEAM_ID)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    updated = datetime.fromtimestamp(max((m for _, m in sig), default=0)).strftime("%Y-%m-%d %H:%M")
    commit = _git(skills_dir, "log", "-1", "--format=%h %cs", "--", ".")
    dirty = bool(_git(skills_dir, "status", "--porcelain", "--", "."))
    label = f"IT 개념 프롬프트 v{digest} (스킬 파일 수정 {updated}"
    label += (f", 커밋 {commit}" if commit else "") + (" + 커밋 안 된 변경" if dirty else "") + ")"
    return {"hash": digest, "updated": updated, "commit": commit, "dirty": dirty, "label": label, "text": text}


def prompt_version() -> dict:
    """지금 에이전트에게 들어가는 스킬(teams/concept/skills/*.md)의 판 — 내용 해시 8자리·파일 수정 시각·마지막 git 커밋.

    {"hash", "updated", "commit", "dirty", "label", "text"} — 파일이 그대로면 직전에 계산한 것을 쓴다.
    """
    skills_dir = TEAMS_DIR / TEAM_ID / "skills"
    sig = tuple((p.name, p.stat().st_mtime) for p in sorted(skills_dir.glob("*.md")))
    return _version(str(skills_dir), sig)


def used_prompt_version(question: str) -> str:
    """지난 요청(질문)에 적힌 프롬프트 판 해시 — 없으면 '' (이 기능 이전의 결과)."""
    m = PROMPT_VERSION.search(question or "")
    return m.group(1) if m else ""


def expected_header(target: Entry, d: date | None = None) -> str:
    """수정본 제목 줄 — 기존 작성일·수정일 이력 + 오늘 수정일 (카탈로그 반영 때 apply_plan 이 쓰는 것과 같은 규칙)."""
    return with_updated_date(target.header, d)


# ─── 요청 만들기 ────────────────────────────────────────────────────────


def entry_digest(entry: Entry, limit: int = 900) -> str:
    """관련 항목 요약 — 제목 + 1.정의 앞부분 + 6.관계 앞부분 (무료 모델 토큰 한도를 고려해 짧게)."""
    definition = entry.section(1)[: limit // 2]
    relation = entry.section(6)[: limit // 2]
    return f"[{entry.category}] {entry.header}\n{definition}\n{relation}".strip()


def build_concept_request(
    action: str, name: str, draft: str, related: list[Entry], target: Entry | None, d: date | None = None,
    sub_title: str = "",
) -> tuple[str, DesiredState]:
    """개념 작성 요청 → (질문, 최종 상태 브리프). 질문 = 프롬프트의 [입력] 절.

    action 은 코드가 판정한 것 (catalog.find_existing): 기존에 있으면 update, 없으면 add.
    sub_title 이 있으면 이 개념이 target 항목의 '7. 변형·확장 상세' 소절로 이미 들어 있다는 뜻 → 부모 항목 전체를 수정한다.
    """
    d = d or date.today()
    if action == "update" and sub_title:
        how = (f"수정 (update) — 이 개념은 「{target.name}」의 7.변형·확장 상세 소절 '{sub_title}' 로 이미 있음 "
               "→ 부모 항목 전체를 다시 쓰되 그 소절을 초안으로 보완")
    elif action == "update":
        how = "수정 (update) — 같은 개념이 카탈로그에 이미 있음 → 기존 항목을 초안으로 보완한 전체 수정본 작성 (작성일은 기존 그대로)"
    elif action == "add":
        how = "추가 (add) — 같은 이름의 항목이 없음 (단, 1단계에서 기존 항목에 편입·병합해야 하면 update/absorb 로 판정)"
    else:
        how = f"{ACTIONS[action]} ({action})"
    header = expected_header(target, d) if action == "update" and target else ""
    parts = [
        "[입력]",
        f"- 작업(코드 자동 판정): {how}",
        f"- 개념명(한/영): {name}",
    ]
    if header:
        parts.append(f"- 수정본 제목 줄(날짜 이력 유지 — 이대로 쓴다): {header}")
    parts.append(f"- 초안:\n{draft.strip() or '(없음 — 개념명만으로 작성)'}")
    if target:
        parts.append(f"- 기존 카탈로그 항목(수정 대상 전문, 분류 {target.category}):\n{target.text}")
    if related:
        parts.append("- 관련 기존 카탈로그 항목 (1단계 비교 대상 후보 — 제목·정의·관계 발췌):\n\n" + "\n\n".join(entry_digest(e) for e in related))
    parts.append(f"{PROMPT_LINE}{prompt_version()['label']}")  # 어느 판 프롬프트로 만든 결과인지 — 질문과 함께 저장된다
    question = "\n".join(parts)
    plan_json = json.dumps(PLAN_EXAMPLE, ensure_ascii=False)
    brief = DesiredState(
        goal=f"'{name}' 개념의 검증된 카탈로그 최종본과 GitHub 요약, 카탈로그 적용 계획",
        context=(
            f"카탈로그 분류: Cloud(1.Cloud & Infrastructure) / AI(2.Artificial Intelligence) / Security(3.Security & Trust). "
            f"오늘 날짜 {d.isoformat()} — 새 항목 제목의 작성일은 ({today_label(d)}).\n"
            "작성 규칙은 시스템 프롬프트의 'IT 개념 프롬프트'를 따른다."
        ),
        success_criteria=[
            "<review_log> 안에 1단계(관련 기존 항목 비교 표·이중 해리·판정)와 2단계(원전 식별·주장별 판정 표·체크리스트)를 먼저 쓴다",
            "<final> 안에 카탈로그 형식 최종본(제목 줄 ~ 참고문헌)을 개조식으로 쓴다",
            "GitHub 요약(이모지+개념명+태그 / 분류 / 핵심 키워드 / 한 줄 정의)을 쓴다",
            "맨 끝에 적용 계획 ```json 블록을 쓴다 (편입·병합이면 absorb 에 흡수될 기존 항목, 다른 항목 수정은 edits 에)",
        ],
        constraints=[
            "🔧수정·➕추가 표시와 이유는 최종본(<final>)에 넣지 않고 </final> 뒤 '## 변경 요약'에만 쓴다 — 최종본에는 고친 결과만",
            "최종본 모양은 스킬의 [출력 형식 견본]을 그대로 따른다: 첫 줄 '[태그]한글명(English, 약어) (작성일: YY.MM.DD)' "
            "(분류는 제목에 쓰지 않고 적용 계획 JSON 에만), 다음 줄 ' - 부제: …', 절 제목 '1. 정의 및 개념' ~ '6. 관계 (기존 개념 비교)'·"
            "'7. 변형·확장 상세', 마지막 절은 번호를 붙인 '8. 참고 문헌'(7절이 없으면 '7. 참고 문헌'), 불릿 ' - ', 등장 배경 ①②③, "
            "작동 원리·부적합 영역 ⓐⓑⓒ, 7절 하위 항목 '   1) 이름' 아래 '     - 개념: / 작동원리: / 장점: / 단점(고유): / 예시:', "
            "같은 항목은 §번호·다른 항목은 「항목명」으로 참조, 마크다운 굵게(**) 쓰지 않음",
            *(["기존 항목 수정: 기존 형식(절 제목·들여쓰기·표식)이 프롬프트와 어긋나지 않으면 그대로 두고 내용만 고친다. "
               f"제목 줄 날짜는 '{header_dates(header)}' 그대로"] if header else []),
            "존재를 확인할 수 없는 논문·arXiv ID·URL·수치를 만들지 않는다",
            "초안의 배치·계보 주장을 원전 확인 없이 수용하지 않는다",
            "최종본에 '완벽·100%·무조건·완전 차단' 같은 절대 표현을 쓰지 않는다",
        ],
        output_format=(
            "<review_log>1단계 표·이중 해리·판정 / 2단계 검증 로그 표</review_log> → <final>[태그]한글명(English) (작성일: …) / 부제 / "
            "1.~6.(7.) / 8. 참고 문헌</final> → ## 변경 요약 → ## GitHub 요약 → ```json 적용 계획``` 예시: " + plan_json
        ),
        assumptions=[],
    )
    return question, brief


def delete_plan(target: Entry) -> dict:
    """삭제는 LLM 없이 — 대상 항목을 old_term 으로 옮기고 요약에서 뺀다."""
    return {"action": "delete", "category": target.category, "tag": target.tag, "title": target.name, "target": target.name}


def apply_edits(catalog: Catalog, edits: list[dict]) -> list[str]:
    """다른 기존 항목의 문장 교체 (find 가 그 항목 본문에 정확히 한 번 있을 때만). 결과 메시지 목록."""
    from core.catalog import CATEGORY_FILES as FILES, replace_entry, _read, _write

    logs = []
    for edit in edits or []:
        entry = catalog.find(str(edit.get("entry", "")))
        find, replace = str(edit.get("find", "")), str(edit.get("replace", ""))
        if entry is None:
            logs.append(f"⚠️ 수동 수정 필요 — 항목을 찾지 못함: {edit.get('entry')}")
            continue
        if not find or entry.body.count(find) != 1:
            logs.append(f"⚠️ 수동 수정 필요 — '{entry.name[:30]}' 에서 바꿀 문장을 정확히 찾지 못함: {find[:40]!r}")
            continue
        path = catalog.root / FILES[entry.category]
        new_text = f"{entry.header}\n{entry.body.replace(find, replace)}"
        _write(path, replace_entry(_read(path), entry, new_text))
        logs.append(f"✅ {entry.name[:40]} — {edit.get('reason') or '문장 교체'}")
        catalog.entries = Catalog.load(catalog.root).entries  # 다음 수정이 최신 본문을 보도록
    return logs
