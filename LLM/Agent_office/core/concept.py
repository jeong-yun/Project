"""개념 작성 — 요청 만들기(관련 기존 항목 포함) · 결과 나누기(검증 로그/최종본/요약/적용 계획) · 형식 검사.

결과 형식 (개념 편집 팀이 지켜야 하는 출력 계약 — 코드가 이 표시로 잘라서 파일에 반영한다)
  <thought> 1단계 비교 표·이중 해리 + 2단계 검증 로그 </thought>
  <final> 카탈로그에 그대로 들어갈 항목 텍스트 (제목 줄부터 참고문헌까지) </final>
  ## GitHub 요약 …
  ```json {"concept_plan": {...}} ```   ← 적용 계획 (분류·태그·추가/수정·통합 대상·요약·기초 용어·다른 항목 수정)
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date

from core.brief import DesiredState
from core.catalog import CATEGORY_FILES, TAGS, Catalog, Entry, is_header, today_label
from core.harness.base import extract_code_blocks

ACTIONS = {"add": "추가", "update": "수정", "delete": "삭제"}
ABSOLUTE = ("완벽", "100%", "무조건", "완전 차단", "원천 차단", "절대적으로", "무한한")
REF_LABELS = ("논문·핵심기법", "논문·서베이", "논문·한계·반박", "논문·벤치마크", "표준/거버넌스", "도구/프레임워크")
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


SECTION_TITLES = {  # 절 제목 표준형 — 모델이 '1.정의및개념'처럼 붙여 써도 카탈로그 형식으로 맞춘다
    1: "1. 정의 및 개념", 2: "2. 작동 원리", 3: "3. 좋아진 점 (장점/개선점)", 4: "4. 문제점 (단점/한계)",
    5: "5. 사용 예시", 6: "6. 관계(기존 개념과 비교)", 7: "7. 변형·확장 상세",
}
SECTION_LINE = re.compile(r"^\s*#{0,4}\s*\**\s*(\d)\s*[.)]\s*\**\s*(정의|작동|좋아진|문제점|사용|관계|변형|확장)")
CATEGORY_PREFIX = re.compile(r"^\s*\[\s*\d\s*\.\s*(Cloud[^\]]*|AI[^\]]*|Security[^\]]*)\]\s*")


def normalize_entry(entry: str) -> str:
    """LLM 이 쓴 최종본을 카탈로그 파일 형식으로 맞춘다 (결정론적 — 내용은 바꾸지 않음).

    - 제목 앞의 분류 표시('[2.AI] ') 제거, 태그 뒤 공백 제거('[Perf] X' → '[Perf]X')
    - 절 제목 '1.정의및개념' → '1. 정의 및 개념' 등 표준형, '참고 문헌' 절 이름 통일
    - 마크다운 굵게(**) 제거, 불릿 '- ' → ' - ', 들여쓴 하위 불릿 → '  ○ ', 빈 줄 제거
    """
    lines = [l.rstrip() for l in entry.replace("\r\n", "\n").split("\n")]
    lines = [l for l in lines if l.strip()]
    if not lines:
        return ""
    header = CATEGORY_PREFIX.sub("", lines[0].replace("**", "").strip())
    header = re.sub(r"^(\[[^\]]+\])\s+", r"\1", header)
    out = [header]
    for line in lines[1:]:
        text = line.replace("**", "")
        m = SECTION_LINE.match(text)
        if m and int(m.group(1)) in SECTION_TITLES:
            out.append(SECTION_TITLES[int(m.group(1))])
            continue
        if re.match(r"^\s*#{0,4}\s*(\d\s*[.)]\s*)?참고\s*문헌", text):
            out.append("참고 문헌")
            continue
        bullet = re.match(r"^(\s*)[-*•·]\s+(.*)$", text)
        if bullet:
            indent, rest = bullet.groups()
            out.append((" - " if len(indent) < 2 else "  ○ ") + rest)
        elif text.startswith("[") and "참고 문헌" in out:
            out.append(" - " + text)  # 참고문헌 한 줄: '[라벨] 저자. 제목 …' → ' - [라벨] …'
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


def _between(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_result(answer: str) -> ConceptResult:
    res = ConceptResult(thought=_between(answer, "thought"), final_entry=normalize_entry(_between(answer, "final")))
    after = re.split(r"</final>", answer, flags=re.IGNORECASE)[-1] if "</final>" in answer.lower() else answer
    res.summary_md = re.sub(r"```json.*?```", "", after, flags=re.DOTALL).strip()
    for src in extract_code_blocks(answer, "json"):
        try:
            data = json.loads(src)
        except json.JSONDecodeError as e:
            res.problems.append(f"적용 계획 JSON 파싱 오류: {e.msg}")
            continue
        if isinstance(data, dict) and "concept_plan" in data:
            res.plan = data["concept_plan"]
    return res


def check_answer(answer: str, options: dict | None = None) -> list[str]:
    """결과 형식 검사 (하네스). 틀린 곳 목록 — 비어 있으면 통과."""
    res = parse_result(answer)
    problems = list(res.problems)
    lower = answer.lower()
    if not res.thought:
        problems.append("<thought> … </thought> (1단계 비교·2단계 검증 로그)가 없다")
    elif lower.find("<thought>") > lower.find("<final>") >= 0:
        problems.append("<thought> 는 <final> 보다 먼저 와야 한다 (검증 로그 먼저)")
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
        problems += check_entry(res.final_entry)
    if action in ("update", "delete") and not str(plan.get("target", "")).strip():
        problems.append("수정·삭제는 concept_plan.target 에 기존 항목 이름을 적어야 한다")
    return problems


def check_entry(entry: str) -> list[str]:
    """최종본(<final>) 형식 검사 — 제목 줄·절 구성·등장 배경·관계 짝·개조식·절대 표현·참고문헌 형식."""
    if not entry:
        return ["<final> … </final> 최종본이 없다"]
    problems = []
    lines = entry.splitlines()
    if not is_header(lines[0]):
        problems.append("최종본 첫 줄은 '[태그]한글명(English) (작성일: YY.MM.DD)' 제목 줄이어야 한다")
    for n in range(1, 7):
        if not re.search(rf"^{n}\.\s*\S", entry, flags=re.MULTILINE):
            problems.append(f"{n}번 절이 없다 (1.정의및개념 ~ 6.관계 필수)")
    section1 = re.split(r"^2\.\s*\S", entry, maxsplit=1, flags=re.MULTILINE)[0]  # 'Qwen2.5' 같은 본문 숫자에 걸리지 않게 줄 첫머리만
    if "배경" not in section1:
        problems.append("1번(정의 및 개념)에 '등장 배경'(문제 → 원인 → 처방) 2~4줄이 없다")
    relation = re.search(r"^6\.\s.*?(?=^\d\.\s|^\S*참고\s*문헌|\Z)", entry, flags=re.MULTILINE | re.DOTALL)
    if relation and not ("유사" in relation.group(0) and "차이" in relation.group(0)):
        problems.append("6.관계는 대상마다 '유사: … / 차이: … / 경계: …' 짝으로 써야 한다")
    found = [w for w in ABSOLUTE if w in entry]
    if found:
        problems.append(f"절대 표현 금지: {found} → '효과적으로 억제·최소화' 같은 객관적 표현으로")
    if PLACEHOLDER.search(entry):
        problems.append(f"플레이스홀더 인용이 있다: {PLACEHOLDER.search(entry).group(0)!r} — 실제 식별자로 고치거나 빼라")
    narrative = [l.strip()[:40] for l in lines[1:] if NARRATIVE_END.search(l.strip()) and not l.strip().startswith(("\"", "“", "'"))]
    if len(narrative) > 1:
        problems.append(f"개조식(명사형 종결) 위반 {len(narrative)}줄: {narrative[:3]} … → '~함/~됨/~임/명사구'로")
    ref_start = next((i for i, l in enumerate(lines) if re.match(r"^\s*(\d\.\s*)?참고\s*문헌", l)), None)
    if ref_start is None:
        problems.append("참고문헌 절이 없다")
    else:
        refs = [l.strip() for l in lines[ref_start + 1 :] if l.strip().startswith("-")]
        bad = [r[:50] for r in refs if not (re.match(r"^-\s*\[(" + "|".join(map(re.escape, REF_LABELS)) + r")\]", r) and "http" in r)]
        if not refs:
            problems.append("참고문헌 항목이 없다")
        elif bad:
            problems.append(f"참고문헌 형식 위반 {len(bad)}건 (라벨·URL 필수): {bad[:2]}")
    return problems


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
    parts = [
        "[입력]",
        f"- 작업(코드 자동 판정): {how}",
        f"- 개념명(한/영): {name}",
        f"- 초안:\n{draft.strip() or '(없음 — 개념명만으로 작성)'}",
    ]
    if target:
        parts.append(f"- 기존 카탈로그 항목(수정 대상 전문, 분류 {target.category}):\n{target.text}")
    if related:
        parts.append("- 관련 기존 카탈로그 항목 (1단계 비교 대상 후보 — 제목·정의·관계 발췌):\n\n" + "\n\n".join(entry_digest(e) for e in related))
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
            "<thought> 안에 1단계(관련 기존 항목 비교 표·이중 해리·판정)와 2단계(원전 식별·주장별 판정 표·체크리스트)를 먼저 쓴다",
            "<final> 안에 카탈로그 형식 최종본(제목 줄 ~ 참고문헌)을 개조식으로 쓴다",
            "GitHub 요약(이모지+개념명+태그 / 분류 / 핵심 키워드 / 한 줄 정의)을 쓴다",
            "맨 끝에 적용 계획 ```json 블록을 쓴다 (편입·병합이면 absorb 에 흡수될 기존 항목, 다른 항목 수정은 edits 에)",
        ],
        constraints=[
            "최종본 형식: 첫 줄 '[태그]한글명(English, 약어) (작성일: YY.MM.DD)' (분류는 제목에 쓰지 않고 적용 계획 JSON 에만), "
            "절 제목 '1. 정의 및 개념' ~ '6. 관계(기존 개념과 비교)'·'7. 변형·확장 상세'·'참고 문헌', 불릿은 ' - ' 와 '  ○ ', 마크다운 굵게(**) 쓰지 않음",
            "존재를 확인할 수 없는 논문·arXiv ID·URL·수치를 만들지 않는다",
            "초안의 배치·계보 주장을 원전 확인 없이 수용하지 않는다",
            "최종본에 '완벽·100%·무조건·완전 차단' 같은 절대 표현을 쓰지 않는다",
        ],
        output_format=(
            "<thought>1단계 표·이중 해리·판정 / 2단계 검증 로그 표</thought> → <final>[태그]한글명(English) (작성일: …) 1.~6.(7.) 참고문헌</final> → "
            "## GitHub 요약 → ```json 적용 계획``` 예시: " + plan_json
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
