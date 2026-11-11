"""🌍 취미·교양 교육팀(edu_hobby) 전용 — 코드 계산·검증기·근거 자료 글 (화면 그리기 없음, 화면은 edu_hobby_ui.py).

교육부 공용 틀(core/learning.py)의 '2·3단계 약속'대로 팀 기록은 모두 Course.extras 에 넣는다 (Course 칸은 늘리지 않는다).

  📅 커리큘럼   : curriculum_problems(주차·단원 형식, 주당 시간 합 — 0.1시간 단위 정수로 계산) · save_curriculum(남은 부분 교체)
  📖 단어·용어장 : extras["vocab"]   — vocab_problems · add_vocab(같은 단어는 합침) · vocab_csv(Anki) · vocab_markdown
  🗣 상황별 표현 : extras["phrases"] — phrase_problems · add_phrases
  🧭 추가 지식   : extras["topics"]  — topic_problems · add_topics · new_course_prefill('새 과목으로 만들기' 양식 값)
  🔁 복습       : due_items(외움 아님·복습일 지남·오래된 것) · flashcard_quiz(모델 없이 낱말 카드) · review_refs ·
                  schedule / apply_review (간단한 간격 반복 — 맞히면 상자 +1, 틀리면 처음부터)

extras 항목 모양 (learning.py 머리말의 기본 키 + 이 팀이 더 쓰는 키)
  vocab   : {term, meaning, reading, pos, example, example_ko, related, level, added_at,  known, box, due, last_review, reviews, wrong}
  phrases : {situation, phrase, meaning, reading, dialogue[], tip, level, added_at,          known, box, due, last_review, reviews, wrong}
  topics  : {title, why, first_unit, field, done, added_at}

날짜는 'YYYY-MM-DD' 글로 저장하고, 계산하는 함수는 모두 today 를 인자로 받는다 (테스트가 시간에 흔들리지 않게).
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, timedelta

from core import learning
from core.harness.base import Validator, register

KIND = "hobby"
SKILLS = ("듣기", "말하기", "읽기", "쓰기")  # 언어 과목 커리큘럼 균형
SITUATIONS = ("✈️ 공항", "🏨 호텔", "🍝 식당·카페", "🧭 길 묻기·교통", "🛍 쇼핑", "🚑 응급·병원", "💬 일상 대화", "🙋 인사·자기소개")
SRS_DAYS = (1, 2, 4, 7, 15, 30)  # 간격 반복 상자 0~5 — 맞힌 뒤 다음 복습까지 며칠
STALE_DAYS = 30  # '외움'이어도 이만큼 복습하지 않았으면 다시 복습 대상
HARD_SHARE_PCT = 30  # 입문·초급 단어장에서 수준보다 두 단계 이상 어려운 단어가 이 % 를 넘으면 경고

VOCAB_FIELDS_LANG = {"term": "단어", "meaning": "뜻", "reading": "발음(읽는 법)", "pos": "품사", "example": "예문", "example_ko": "예문 번역"}
VOCAB_FIELDS_CULTURE = {"term": "용어", "meaning": "풀이", "related": "연도·인물 연결"}
PHRASE_FIELDS = {"phrase": "표현", "meaning": "뜻", "situation": "상황"}

VOCAB_JSON_HELP = ('{"vocab": [{"term": "il conto", "meaning": "계산서", "reading": "일 꼰또", "pos": "명사", '
                   '"example": "Il conto, per favore.", "example_ko": "계산서 주세요.", "level": "beginner"}]}  '
                   '— 교양 과목은 {"term", "meaning", "related": "연도·인물", "level"}')
PHRASE_JSON_HELP = ('{"phrases": [{"situation": "🍝 식당·카페", "phrase": "Il conto, per favore.", "reading": "일 꼰또, 뻬르 파보레", '
                    '"meaning": "계산서 주세요.", "level": "beginner", "dialogue": ["A: …", "B: …"], "tip": "문화 팁"}]}')
TOPIC_JSON_HELP = ('{"topics": [{"title": "로마사 첫걸음", "why": "왜 도움이 되는지", "first_unit": "첫 단원", '
                   '"field": "역사"}]}')
REVIEW_JSON_HELP = ('{"quiz": {"kind": "review", "level": "beginner", "questions": [{"id": "Q1", "question": "grazie 의 뜻은?", '
                    '"choices": ["고마워요", "안녕", "미안해요", "계산서"], "answer": "고마워요", "explanation": "…", "ref": "grazie"}]}}')
CURRICULUM_JSON_HELP = ('{"curriculum": [{"week": 1, "title": "단원 제목", "goals": ["마치면 할 수 있는 것"], "hours": 1.5, '
                        '"skills": ["듣기", "말하기"], "theme": "시대·주제 (교양)"}]}')


# ─── 공통 도우미 ──────────────────────────────────────────────────────


def norm(text: object) -> str:
    """같은 단어인지 볼 때 쓰는 열쇠 — 대소문자·앞뒤 문장부호·띄어쓰기 차이를 없앤다 ('Grazie!' = 'grazie')."""
    s = " ".join(str(text or "").split()).casefold()
    return s.strip(" .,!?¡¿;:…'\"“”‘’()[]")


def _s(value: object) -> str:
    return " ".join(str(value or "").split())


def _today(today: date | None) -> date:
    return today or date.today()


def _date(text: object) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _tenths(value: object) -> int | None:
    """시간 → 0.1시간 단위 정수 (부동소수점 합을 피한다). 숫자가 아니면 None."""
    try:
        return int(round(float(str(value).replace("시간", "").strip()) * 10))
    except (TypeError, ValueError):
        return None


def _hours_text(tenths: int) -> str:
    return f"{tenths // 10}" if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10}"


def _list(data: object, key: str) -> list | None:
    if isinstance(data, dict):
        data = data.get(key)
    return data if isinstance(data, list) else None


def _level_id(value: object, default: str) -> str:
    s = str(value or "").strip()
    return learning.parse_level(s) if s else default


def course_from_options(options: dict) -> learning.Course | None:
    """검증기 옵션의 course_id(툴 learning_course 가 넣는다) → 과목. 없으면 None (그 검사는 건너뛴다)."""
    cid = str(options.get("course_id") or "")
    return learning.LearningStore().get(cid) if learning.valid_course_id(cid) else None


# ─── 📅 커리큘럼 ──────────────────────────────────────────────────────


def curriculum_problems(data: object, weekly_hours: object = None, weeks: object = None,
                        language: bool = False) -> tuple[list[str], list[str]]:
    """커리큘럼 JSON 의 (문제, 경고). 문제가 있으면 다시 쓰게 하고, 경고는 남기기만 한다.

    - 단원마다 week(1부터) · title · goals · hours(숫자)
    - 주차는 1부터 빈 주 없이 이어지고, weeks 를 주면 그 주 수와 같다
    - 주마다 hours 합 ≤ 주당 학습 시간 (0.1시간 단위 정수로 더한다). 절반도 안 쓰면 경고
    - 언어 과목: 전체에서 듣기·말하기·읽기·쓰기가 모두 한 번 이상 (한쪽으로 몰리면 경고)
    - 교양 과목: 단원마다 theme(시대·주제) — 없으면 경고"""
    units = _list(data, "curriculum")
    if not units:
        return ['커리큘럼 JSON 이 없다 — 답변 맨 끝에 ```json {"curriculum": [...]} 블록을 쓴다'], []
    problems, warnings = [], []
    per_week: dict[int, int] = {}
    skill_count = dict.fromkeys(SKILLS, 0)
    for i, u in enumerate(units, 1):
        if not isinstance(u, dict):
            problems.append(f"{i}번 단원이 객체가 아니다")
            continue
        tag = f"{i}번 단원"
        week = u.get("week")
        if isinstance(week, str) and week.strip().isdigit():
            week = int(week.strip())
        if not isinstance(week, int) or isinstance(week, bool) or week < 1:
            problems.append(f"{tag}: week 는 1 이상의 정수여야 한다 (지금 {week!r})")
            continue
        if not _s(u.get("title")):
            problems.append(f"{tag}: title(단원 제목)이 비어 있다")
        goals = u.get("goals")
        if not isinstance(goals, list) or not any(_s(g) for g in goals):
            problems.append(f"{tag}: goals(마치면 할 수 있는 것)가 비어 있다")
        hours = _tenths(u.get("hours"))
        if hours is None or hours <= 0:
            problems.append(f"{tag}: hours(이 단원에 쓸 시간)는 0보다 큰 숫자여야 한다")
            hours = 0
        per_week[week] = per_week.get(week, 0) + hours
        if language:
            skills = u.get("skills") if isinstance(u.get("skills"), list) else []
            bad = [s for s in skills if s not in SKILLS]
            if bad:
                problems.append(f"{tag}: skills 는 {'·'.join(SKILLS)} 중에서 고른다 (모르는 값: {', '.join(map(str, bad))})")
            for s in skills:
                if s in skill_count:
                    skill_count[s] += 1
        elif not _s(u.get("theme")):
            warnings.append(f"{tag}: theme(시대·주제)가 없다")
    if per_week:
        last = max(per_week)
        missing = [w for w in range(1, last + 1) if w not in per_week]
        if missing:
            problems.append(f"빈 주차가 있다: {', '.join(map(str, missing))}주차 — 1주차부터 빠짐없이 잇는다")
        try:
            want = int(str(weeks or "0").split()[0])
        except ValueError:
            want = 0
        if want and last != want:
            problems.append(f"커리큘럼이 {last}주다 — {want}주로 짠다")
    limit = _tenths(weekly_hours)
    if limit and limit > 0:
        for w, total in sorted(per_week.items()):
            if total > limit:
                problems.append(f"{w}주차 시간 합 {_hours_text(total)}시간 > 주당 {_hours_text(limit)}시간 — 단원을 줄이거나 나눈다")
            elif total * 2 < limit:
                warnings.append(f"{w}주차는 {_hours_text(total)}시간뿐이다 (주당 {_hours_text(limit)}시간)")
    if language and per_week:
        absent = [s for s, n in skill_count.items() if n == 0]
        if absent:
            problems.append(f"언어 과목인데 {'·'.join(absent)} 단원이 없다 — 듣기·말하기·읽기·쓰기를 고루 넣는다")
        else:
            most = max(skill_count.values())
            if most > 2 * max(1, min(skill_count.values())) + 2:
                warnings.append("영역이 한쪽으로 몰렸다: " + " · ".join(f"{s} {n}" for s, n in skill_count.items()))
    return problems, warnings


def curriculum_units(data: object) -> list[dict]:
    """커리큘럼 JSON → 저장할 단원 초안 (goals 끝에 '🕐 시간 · 영역/주제' 한 줄을 붙여 화면에서 보이게)."""
    drafts = []
    for u in _list(data, "curriculum") or []:
        if not isinstance(u, dict) or not _s(u.get("title")):
            continue
        goals = [_s(g) for g in (u.get("goals") or []) if _s(g)] if isinstance(u.get("goals"), list) else []
        tenths = _tenths(u.get("hours"))
        tag = "·".join(s for s in (u.get("skills") or []) if s in SKILLS) if isinstance(u.get("skills"), list) else ""
        tag = tag or _s(u.get("theme"))
        meta = " · ".join(x for x in [f"🕐 {_hours_text(tenths)}시간" if tenths else "", tag] if x)
        drafts.append({"week": u.get("week"), "title": _s(u.get("title")), "goals": goals + ([meta] if meta else [])})
    return drafts


def save_curriculum(course: learning.Course, data: object) -> list[str]:
    """채택한 커리큘럼으로 과목의 남은(안 끝낸) 단원을 바꾼다. 끝낸 단원은 그대로, 지워진 단원 제목을 돌려준다.

    모델은 주차를 1부터 쓰므로, 끝낸 단원 다음 주차부터 이어지도록 주차를 민다."""
    drafts = curriculum_units(data)
    if not drafts:
        raise ValueError("커리큘럼 단원이 없습니다")
    start = max((u.week for u in course.curriculum if u.done), default=0) + 1
    first = min(learning._int(d["week"]) for d in drafts)
    for d in drafts:
        d["week"] = learning._int(d["week"]) - first + start
    return learning.replace_remaining(course, drafts, course.level)


def week_hours(data: object) -> dict[int, str]:
    """주차 → 시간 합 글 (미리 보기 표)."""
    per_week: dict[int, int] = {}
    for u in _list(data, "curriculum") or []:
        if isinstance(u, dict) and str(u.get("week", "")).strip().isdigit():
            w = int(str(u["week"]).strip())
            per_week[w] = per_week.get(w, 0) + (_tenths(u.get("hours")) or 0)
    return {w: _hours_text(t) for w, t in sorted(per_week.items())}


# ─── 📖 단어·용어장 · 🗣 상황별 표현 (공통: 합치기) ─────────────────────────

SRS_KEYS = ("known", "box", "due", "last_review", "reviews", "wrong")


def _clean(raw: dict, fields: tuple[str, ...], level: str, now: str) -> dict:
    item = {k: _s(raw.get(k)) for k in fields}
    item["level"] = _level_id(raw.get("level"), level)
    item["added_at"] = now
    item.update(known=False, box=0, due="", last_review="", reviews=0, wrong=0)
    return item


def merge_items(existing: list[dict], new: list[dict], key: str) -> tuple[list[dict], int, int]:
    """같은 key(정규화)는 하나로 합친다 — 기존 항목의 외움·복습 기록은 지키고 빈 칸만 새 값으로 채운다.

    (합친 목록, 새로 더한 수, 합친 수). new 안의 중복도 합친다."""
    out = [dict(x) for x in existing]
    index = {norm(x.get(key)): i for i, x in enumerate(out) if norm(x.get(key))}
    added = merged = 0
    for item in new:
        k = norm(item.get(key))
        if not k:
            continue
        if k in index:
            old = out[index[k]]
            for f, v in item.items():
                if f in SRS_KEYS or f == "added_at":
                    continue
                if v not in ("", None, []) and old.get(f) in ("", None, []):
                    old[f] = v
            merged += 1
        else:
            index[k] = len(out)
            out.append(dict(item))
            added += 1
    return out, added, merged


def duplicates(items: list, key: str) -> list[str]:
    """목록 안에서 두 번 이상 나온 key (정규화 기준, 처음 쓴 모양으로)."""
    seen: dict[str, str] = {}
    dups: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        k = norm(it.get(key))
        if k and k in seen and seen[k] not in dups:
            dups.append(seen[k])
        seen.setdefault(k, _s(it.get(key)))
    return dups


def vocab_problems(data: object, language: bool = False, level: str = "beginner",
                   count: object = None) -> tuple[list[str], list[str]]:
    """단어장 JSON 의 (문제, 경고). 칸 형식 · 중복 = 문제, 수준보다 훨씬 어려운 단어가 많음 · 개수 차이 = 경고."""
    items = _list(data, "vocab")
    if not items:
        return ['단어장 JSON 이 없다 — 답변 맨 끝에 ```json {"vocab": [...]} 블록을 쓴다'], []
    fields = VOCAB_FIELDS_LANG if language else VOCAB_FIELDS_CULTURE
    problems, warnings = [], []
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            problems.append(f"{i}번 항목이 객체가 아니다")
            continue
        empty = [label for k, label in fields.items() if not _s(it.get(k))]
        if empty:
            problems.append(f"{i}번 '{_s(it.get('term')) or '?'}': {', '.join(empty)} 칸이 비어 있다")
        lv = it.get("level")
        if lv not in (None, "") and lv not in learning.LEVEL_IDS:
            problems.append(f"{i}번 '{_s(it.get('term'))}': level 은 {', '.join(learning.LEVEL_IDS)} 중 하나")
    dups = duplicates(items, "term")
    if dups:
        problems.append(f"같은 단어가 두 번 나온다: {', '.join(dups[:8])} — 하나로 합친다")
    if level in ("beginner", "elementary"):
        limit = learning.level_index(level) + 2  # 수준보다 두 단계 이상 위
        hard = [_s(it.get("term")) for it in items if isinstance(it, dict) and it.get("level") in learning.LEVEL_IDS
                and learning.level_index(it["level"]) >= limit]
        if hard and len(hard) * 100 > HARD_SHARE_PCT * len(items):
            warnings.append(f"⚠ {learning.level_of(level).name} 단어장에 어려운 단어가 많다 ({len(hard)}/{len(items)}개: "
                            f"{', '.join(hard[:6])}) — 쉬운 단어로 바꾸면 더 좋다")
    try:
        want = int(str(count or "0").split()[0])
    except ValueError:
        want = 0
    if want and abs(len(items) - want) * 5 > want:  # 20% 넘게 차이
        warnings.append(f"⚠ 단어 {len(items)}개 — 요청은 {want}개")
    return problems, warnings


def vocab_items(data: object, level: str, now: str | None = None) -> list[dict]:
    fields = ("term", "meaning", "reading", "pos", "example", "example_ko", "related")
    return [_clean(it, fields, level, now or learning.now_text()) for it in (_list(data, "vocab") or [])
            if isinstance(it, dict) and _s(it.get("term"))]


def add_vocab(course: learning.Course, data: object, now: str | None = None) -> tuple[int, int]:
    """채택한 단어장을 extras['vocab'] 에 쌓는다 — 같은 단어는 합친다. (새로 더한 수, 합친 수)"""
    items = vocab_items(data, course.level, now)
    if not items:
        raise ValueError("단어장에 단어가 없습니다")
    merged, added, joined = merge_items(course.extras.get("vocab") or [], items, "term")
    course.extras["vocab"] = merged
    return added, joined


def phrase_problems(data: object, language: bool = True, situation: str = "") -> tuple[list[str], list[str]]:
    """상황별 표현 JSON 의 (문제, 경고). 표현·뜻·상황은 꼭, 언어 과목은 읽는 법도. 대화 예시·문화 팁이 없으면 경고."""
    items = _list(data, "phrases")
    if not items:
        return ['표현 JSON 이 없다 — 답변 맨 끝에 ```json {"phrases": [...]} 블록을 쓴다'], []
    problems, warnings = [], []
    need = {**PHRASE_FIELDS, **({"reading": "발음(읽는 법)"} if language else {})}
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            problems.append(f"{i}번 항목이 객체가 아니다")
            continue
        empty = [label for k, label in need.items() if not _s(it.get(k))]
        if empty:
            problems.append(f"{i}번 '{_s(it.get('phrase')) or '?'}': {', '.join(empty)} 칸이 비어 있다")
    dups = duplicates(items, "phrase")
    if dups:
        problems.append(f"같은 표현이 두 번 나온다: {', '.join(dups[:6])}")
    if not any(isinstance(it, dict) and it.get("dialogue") for it in items):
        warnings.append("⚠ 대화 예시(dialogue)가 하나도 없다")
    if not any(isinstance(it, dict) and _s(it.get("tip")) for it in items):
        warnings.append("⚠ 문화 팁(tip)이 하나도 없다")
    want = situation_name(situation)
    if want and "직접" not in want and not any(isinstance(it, dict) and want in _s(it.get("situation")) for it in items):
        warnings.append(f"⚠ 고른 상황 '{want}' 표현이 없다")
    return problems, warnings


def situation_name(situation: object) -> str:
    """'🍝 식당·카페' → '식당·카페' (이모지 빼고 비교)."""
    return re.sub(r"^[^\w가-힣]+", "", _s(situation))


def phrase_items(data: object, level: str, now: str | None = None) -> list[dict]:
    out = []
    for it in _list(data, "phrases") or []:
        if not isinstance(it, dict) or not _s(it.get("phrase")):
            continue
        item = _clean(it, ("situation", "phrase", "meaning", "reading", "tip"), level, now or learning.now_text())
        dialogue = it.get("dialogue")
        item["dialogue"] = [_s(x) for x in dialogue if _s(x)] if isinstance(dialogue, list) else ([_s(dialogue)] if _s(dialogue) else [])
        out.append(item)
    return out


def add_phrases(course: learning.Course, data: object, now: str | None = None) -> tuple[int, int]:
    items = phrase_items(data, course.level, now)
    if not items:
        raise ValueError("표현이 없습니다")
    merged, added, joined = merge_items(course.extras.get("phrases") or [], items, "phrase")
    course.extras["phrases"] = merged
    return added, joined


# ─── 🧭 추가 지식 ─────────────────────────────────────────────────────


def topic_problems(data: object) -> tuple[list[str], list[str]]:
    """추가 주제 JSON — 3~5개, 주제마다 title · why · first_unit, 같은 주제 두 번 금지."""
    items = _list(data, "topics")
    if not items:
        return ['추가 주제 JSON 이 없다 — 답변 맨 끝에 ```json {"topics": [...]} 블록을 쓴다'], []
    problems = []
    if not 3 <= len(items) <= 5:
        problems.append(f"주제가 {len(items)}개다 — 3~5개를 제안한다")
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            problems.append(f"{i}번 주제가 객체가 아니다")
            continue
        empty = [label for k, label in {"title": "주제", "why": "왜 도움이 되는지", "first_unit": "첫 단원"}.items() if not _s(it.get(k))]
        if empty:
            problems.append(f"{i}번 '{_s(it.get('title')) or '?'}': {', '.join(empty)} 칸이 비어 있다")
    dups = duplicates(items, "title")
    if dups:
        problems.append(f"같은 주제가 두 번 나온다: {', '.join(dups)}")
    return problems, []


def add_topics(course: learning.Course, data: object, now: str | None = None) -> tuple[int, int]:
    stamp = now or learning.now_text()
    items = [{"title": _s(it.get("title")), "why": _s(it.get("why")), "first_unit": _s(it.get("first_unit")),
              "field": _s(it.get("field")), "done": False, "added_at": stamp}
             for it in (_list(data, "topics") or []) if isinstance(it, dict) and _s(it.get("title"))]
    if not items:
        raise ValueError("추가 주제가 없습니다")
    merged, added, joined = merge_items(course.extras.get("topics") or [], items, "title")
    course.extras["topics"] = merged
    return added, joined


def new_course_prefill(course: learning.Course, topic: dict) -> dict[str, object]:
    """'새 과목으로 만들기' — 공용 '🎓 선생님 만들기' 양식 칸 값 (수준은 입문, 언어 과목 표시는 끔: 대개 교양 주제)."""
    title = _s(topic.get("title"))
    return {
        "topic": title,
        "level": learning.LEVELS[0].name,
        "goal": f"{course.topic} 공부에 이어 {title} 교양 쌓기" + (f" — 첫 단원: {_s(topic.get('first_unit'))}" if topic.get("first_unit") else ""),
        "style": "재미있게, 이야기·사진 위주",
        "language": False,
    }


# ─── 🔁 복습 (간격 반복) ───────────────────────────────────────────────

REVIEW_KINDS = {"vocab": ("term", "meaning"), "phrases": ("phrase", "meaning")}  # extras 키 → (앞면, 뒷면)


def is_due(item: dict, today: date | None = None) -> bool:
    """복습할 때인가 — 외움이 아니면 복습일이 없거나 지났을 때, 외움이면 마지막 복습이 STALE_DAYS 넘게 지났을 때."""
    t = _today(today)
    if item.get("known"):
        last = _date(item.get("last_review")) or _date(item.get("added_at"))
        return last is None or (t - last).days >= STALE_DAYS
    due = _date(item.get("due"))
    return due is None or due <= t


def due_items(course: learning.Course, today: date | None = None, limit: int = 10) -> list[tuple[str, dict]]:
    """복습 대상 [(extras 키, 항목)] — 외움이 아닌 것 먼저, 오래된 것부터 (복습일 · 아직 안 본 것은 더한 날).

    같은 날이면 단어 → 표현, 그 안에서는 더한 순서."""
    t = _today(today)
    picked = []
    for rank, key in enumerate(REVIEW_KINDS):
        for order, item in enumerate(course.extras.get(key) or []):
            if is_due(item, t):
                when = str(item.get("due") or item.get("last_review") or item.get("added_at") or "")[:10]
                picked.append((bool(item.get("known")), when or "0000", rank, order, key, item))
    picked.sort(key=lambda x: x[:4])
    return [(key, item) for *_, key, item in picked[:limit]]


def schedule(item: dict, ok: bool, today: date | None = None) -> dict:
    """간단한 간격 반복 (라이트너 상자) — 맞히면 상자 +1 (최대 5), 틀리면 0. 다음 복습일 = 오늘 + SRS_DAYS[상자].

    '외움'은 사람이 표에서 체크한다. 다만 틀리면 '외움'을 풀어 다시 복습 대상에 넣는다."""
    t = _today(today)
    box = int(item.get("box") or 0)
    box = min(box + 1, len(SRS_DAYS) - 1) if ok else 0
    item.update(box=box, due=(t + timedelta(days=SRS_DAYS[box])).isoformat(), last_review=t.isoformat(),
                reviews=int(item.get("reviews") or 0) + 1, wrong=int(item.get("wrong") or 0) + (0 if ok else 1))
    if not ok:
        item["known"] = False
    return item


def find_item(course: learning.Course, ref: object) -> tuple[str, dict] | None:
    """복습 문제의 ref(단어·표현 글) → (extras 키, 항목). 정규화해서 찾는다."""
    k = norm(ref)
    for key, (front, _back) in REVIEW_KINDS.items():
        for item in course.extras.get(key) or []:
            if norm(item.get(front)) == k:
                return key, item
    return None


def apply_review(course: learning.Course, results: dict[str, bool], today: date | None = None) -> int:
    """채점 결과 {ref: 맞음?} → 항목마다 다음 복습일. 고친 항목 수."""
    n = 0
    for ref, ok in results.items():
        found = find_item(course, ref)
        if found:
            schedule(found[1], ok, today)
            n += 1
    return n


def set_known(course: learning.Course, key: str, refs: dict[str, bool]) -> int:
    """표의 '외움' 체크 → 항목에 known. refs = {앞면 글: 외움?}. 고친 수."""
    front = REVIEW_KINDS[key][0]
    want = {norm(r): v for r, v in refs.items()}
    n = 0
    for item in course.extras.get(key) or []:
        k = norm(item.get(front))
        if k in want and bool(item.get("known")) != want[k]:
            item["known"] = want[k]
            n += 1
    return n


def _spot(seed: str, n: int) -> int:
    """정답 보기 자리 — 글자 코드 합으로 정한다 (실행마다 같은 자리, hash() 는 프로세스마다 달라 쓰지 않는다)."""
    return sum(map(ord, seed)) % n if n else 0


def flashcard_quiz(course: learning.Course, today: date | None = None, count: int = 5) -> tuple[learning.Quiz, dict[str, str]] | None:
    """모델 없이 만드는 낱말 카드 복습 퀴즈 — 복습 대상마다 '뜻 고르기'(단어) / '말 고르기'(표현).

    오답 보기는 같은 종류의 다른 항목에서. 보기가 2개도 안 되면 그 항목은 건너뛴다. (퀴즈, {문제 id: ref}) | None"""
    picks = due_items(course, today, count)
    questions, refs = [], {}
    for key, item in picks:
        front, back = REVIEW_KINDS[key]
        pool = [x for x in (course.extras.get(key) or []) if norm(x.get(front)) != norm(item.get(front))]
        if key == "vocab":
            answer = _s(item.get("meaning"))
            reading = f" ({_s(item.get('reading'))})" if item.get("reading") else ""
            text = f"'{_s(item.get('term'))}'{reading} 의 뜻은?"
            wrong = [_s(x.get("meaning")) for x in pool]
        else:
            answer = _s(item.get("phrase"))
            text = f"[{situation_name(item.get('situation')) or '표현'}] '{_s(item.get('meaning'))}' 는 어떻게 말할까요?"
            wrong = [_s(x.get("phrase")) for x in pool]
        start = _spot(answer, len(wrong) or 1)
        wrong = list(dict.fromkeys(w for w in wrong[start:] + wrong[:start] if w and w != answer))[:3]
        if not answer or not wrong:
            continue
        choices = list(wrong)
        choices.insert(_spot(answer + text, len(choices) + 1), answer)
        qid = f"Q{len(questions) + 1}"
        questions.append(learning.QuizQuestion(id=qid, question=text, choices=choices, answer=answer, level=course.level,
                                               explanation=_explain(key, item)))
        refs[qid] = _s(item.get(front))
    if not questions:
        return None
    return learning.Quiz(kind="review", level=course.level, questions=questions), refs


def _explain(key: str, item: dict) -> str:
    if key == "vocab":
        ex = f" · 예문: {item['example']}" + (f" ({item['example_ko']})" if item.get("example_ko") else "") if item.get("example") else ""
        return f"{item.get('term')} = {item.get('meaning')}{ex}"
    return f"{item.get('phrase')} = {item.get('meaning')}" + (f" · 💡 {item['tip']}" if item.get("tip") else "")


def review_refs(text: str) -> dict[str, str]:
    """모델이 낸 복습 퀴즈 JSON → {문제 id: ref}."""
    data = learning.find_json(text, "quiz")
    qs = (data or {}).get("quiz", {}).get("questions") if isinstance((data or {}).get("quiz"), dict) else None
    return {str(q.get("id") or f"Q{i}"): _s(q.get("ref")) for i, q in enumerate(qs or [], 1) if isinstance(q, dict)}


def review_problems(text: str, course: learning.Course | None, count: object = None) -> list[str]:
    """복습 퀴즈 — 공용 퀴즈 형식(quiz_problems) + 문제마다 ref, ref 는 과목 단어장·표현에 있는 것."""
    try:
        want = int(str(count or "0").split()[0])
    except ValueError:
        want = 0
    problems = learning.quiz_problems(learning.find_json(text, "quiz"), expected_count=want or None, kind="review")
    if problems:
        return problems
    for qid, ref in review_refs(text).items():
        if not ref:
            problems.append(f"{qid}: ref(복습한 단어·표현)가 없다 — 근거 자료 '복습 대상' 줄에서 '=' 앞의 단어·표현만 글자 그대로 쓴다")
        elif course is not None and find_item(course, ref) is None:
            problems.append(f"{qid}: ref '{ref}' 가 과목 단어장·표현에 없다 — '복습 대상' 줄에서 '=' 앞의 단어·표현만 글자 그대로 쓴다")
    return problems


# ─── 내려받기 ─────────────────────────────────────────────────────────

CSV_COLUMNS = ("term", "meaning", "reading", "pos", "example", "example_ko", "related", "level", "known")
CSV_HEAD = {"term": "단어", "meaning": "뜻", "reading": "읽는 법", "pos": "품사", "example": "예문", "example_ko": "예문 번역",
            "related": "연도·인물", "level": "수준", "known": "외움"}


def vocab_csv(items: list[dict]) -> str:
    """Anki 로 바로 가져올 수 있는 CSV 글 — 첫 줄들은 Anki 머리말(#separator·#html·#columns), 1열 = 앞면 · 2열 = 뒷면.

    BOM 없는 UTF-8 로 내려받게 한다 (Anki 가 머리말을 읽도록). 엑셀은 '데이터 > 텍스트에서' UTF-8 로 연다."""
    buf = io.StringIO()
    buf.write("#separator:Comma\n#html:false\n")
    buf.write("#columns:" + ",".join(CSV_HEAD[c] for c in CSV_COLUMNS) + "\n")
    writer = csv.writer(buf, lineterminator="\n")
    for it in items:
        row = []
        for c in CSV_COLUMNS:
            v = it.get(c, "")
            row.append(("예" if v else "") if c == "known" else (learning.level_of(v).name if c == "level" and v else _s(v)))
        writer.writerow(row)
    return buf.getvalue()


def parse_vocab_csv(text: str) -> list[dict]:
    """vocab_csv 글을 다시 읽는다 (테스트·되가져오기용) — '#' 머리말은 건너뛴다."""
    rows = [r for r in csv.reader(io.StringIO(text.lstrip("﻿"))) if r and not r[0].startswith("#")]
    return [dict(zip(CSV_COLUMNS, r)) for r in rows]


def _md(text: object) -> str:
    return _s(text).replace("|", "\\|")


def vocab_markdown(course: learning.Course, items: list[dict]) -> str:
    lines = [f"# 📖 {course.topic} 단어장 — {course.level_name}", "",
             f"{course.tutor.title} · {len(items)}개 · 외움 {sum(bool(i.get('known')) for i in items)}개", ""]
    if course.language:
        lines += ["| 단어 | 읽는 법 | 품사 | 뜻 | 예문 | 번역 | 수준 | 외움 |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {_md(i.get('term'))} | {_md(i.get('reading'))} | {_md(i.get('pos'))} | {_md(i.get('meaning'))} | "
                  f"{_md(i.get('example'))} | {_md(i.get('example_ko'))} | {learning.level_of(i.get('level', '')).name} | "
                  f"{'✅' if i.get('known') else ''} |" for i in items]
    else:
        lines += ["| 용어 | 풀이 | 연도·인물 | 수준 | 외움 |", "|---|---|---|---|---|"]
        lines += [f"| {_md(i.get('term'))} | {_md(i.get('meaning'))} | {_md(i.get('related'))} | "
                  f"{learning.level_of(i.get('level', '')).name} | {'✅' if i.get('known') else ''} |" for i in items]
    return "\n".join(lines) + "\n"


def phrases_markdown(course: learning.Course, items: list[dict]) -> str:
    lines = [f"# 🗣 {course.topic} 상황별 표현 — {course.level_name}", ""]
    for sit in dict.fromkeys(_s(i.get("situation")) or "기타" for i in items):
        lines += [f"## {sit}", ""]
        for i in items:
            if (_s(i.get("situation")) or "기타") != sit:
                continue
            lines.append(f"- **{_md(i.get('phrase'))}**" + (f" ({_md(i.get('reading'))})" if i.get("reading") else "")
                         + f" — {_md(i.get('meaning'))}")
            lines += [f"  - {_md(d)}" for d in i.get("dialogue") or []]
            if i.get("tip"):
                lines.append(f"  - 💡 {_md(i['tip'])}")
        lines.append("")
    return "\n".join(lines)


def search(items: list[dict], query: str, fields: tuple[str, ...]) -> list[dict]:
    q = norm(query)
    return [i for i in items if not q or any(q in norm(i.get(f)) for f in fields)]


# ─── 근거 자료 글 (툴 learning_course 가 과목 문맥 아래에 붙인다) ──────────────


def _review_lines(course: learning.Course, key: str, limit: int = 15) -> list[str]:
    front = REVIEW_KINDS[key][0]
    due = [it for k, it in due_items(course, None, 100) if k == key][:limit]
    if not due:
        return []
    return ["", "**🔁 복습 대상 (먼저 낼 것부터 — 복습 퀴즈의 ref 는 아래 줄에서 '=' 앞의 단어·표현만 글자 그대로)**",
            *(f"- {it.get(front)} = {it.get('meaning')}" + (" (외움 — 오래됨)" if it.get("known") else "") for it in due)]


def vocab_text(course: learning.Course, limit: int = 40) -> str:
    items = course.extras.get("vocab") or []
    if not items:
        return ""
    known = sum(bool(i.get("known")) for i in items)
    lines = [f"- 모두 {len(items)}개 · 외움 {known}개 — 새 단어장을 만들 때 아래 단어와 겹치지 않게 한다",
             "| 단어 | 읽는 법 | 뜻 | 수준 | 외움 |", "|---|---|---|---|---|"]
    lines += [f"| {_md(i.get('term'))} | {_md(i.get('reading'))} | {_md(i.get('meaning'))} | {learning.level_of(i.get('level', '')).name} | "
              f"{'✅' if i.get('known') else ''} |" for i in items[-limit:]]
    if len(items) > limit:
        lines.append(f"- (앞의 {len(items) - limit}개는 줄임)")
    return "\n".join(lines + _review_lines(course, "vocab"))


def phrases_text(course: learning.Course, limit: int = 30) -> str:
    items = course.extras.get("phrases") or []
    if not items:
        return ""
    lines = [f"- 모두 {len(items)}개 — 같은 표현을 다시 내지 않는다"]
    lines += [f"- [{situation_name(i.get('situation'))}] {i.get('phrase')} = {i.get('meaning')}" + (" ✅" if i.get("known") else "")
              for i in items[-limit:]]
    return "\n".join(lines + _review_lines(course, "phrases"))


def topics_text(course: learning.Course) -> str:
    items = course.extras.get("topics") or []
    if not items:
        return ""
    return "\n".join(["- 이미 제안한 주제 (다시 제안하지 않는다):"]
                     + [f"  - {t.get('title')}" + (" ✅ 마침" if t.get("done") else "") for t in items])


learning.register_context("vocab", vocab_text)
learning.register_context("phrases", phrases_text)
learning.register_context("topics", topics_text)


# ─── 검증기 (team.yaml 업무의 validators 에 적는다) ──────────────────────


def _language(options: dict, course: learning.Course | None) -> bool:
    return bool(options.get("language")) if "language" in options else bool(course and course.language)


def _report(v: Validator, problems: list[str], warnings: list[str], ok: str, help_text: str):
    if problems:
        return v.fail(" / ".join(problems[:8]) + f"\n형식 예: ```json\n{help_text}\n```")
    return v.ok(ok + (" — " + " / ".join(warnings[:6]) if warnings else ""))


@register
class HobbyCurriculumValidator(Validator):
    name = "hobby_curriculum"
    description = "취미 커리큘럼 JSON — 주차·단원 형식, 주마다 시간 합 ≤ 주당 학습 시간(코드 계산), 언어는 네 영역 균형 (교육부 취미)"

    def check(self, answer, options):
        course = course_from_options(options)
        hours = options.get("weekly_hours") or (course.weekly_hours if course else None)
        data = learning.find_json(answer, "curriculum")
        problems, warnings = curriculum_problems(data, hours, options.get("weeks"), _language(options, course))
        n = len((data or {}).get("curriculum") or [])
        return _report(self, problems, warnings, f"커리큘럼 {n}단원 형식·주당 시간 통과", CURRICULUM_JSON_HELP)


@register
class VocabListValidator(Validator):
    name = "vocab_list"
    description = "단어·용어장 JSON — 칸(언어: 뜻·발음·품사·예문·번역 / 교양: 풀이·연도·인물), 중복, 수준 범위는 경고만 (교육부 취미)"

    def check(self, answer, options):
        course = course_from_options(options)
        level = options.get("level") or (course.level if course else "beginner")
        data = learning.find_json(answer, "vocab")
        problems, warnings = vocab_problems(data, _language(options, course), level, options.get("vocab_count"))
        n = len((data or {}).get("vocab") or [])
        return _report(self, problems, warnings, f"단어 {n}개 형식 통과", VOCAB_JSON_HELP)


@register
class PhraseListValidator(Validator):
    name = "phrase_list"
    description = "상황별 표현 JSON — 표현·뜻·상황(언어는 읽는 법), 중복 금지, 대화 예시·문화 팁은 경고 (교육부 취미)"

    def check(self, answer, options):
        course = course_from_options(options)
        data = learning.find_json(answer, "phrases")
        problems, warnings = phrase_problems(data, _language(options, course), str(options.get("situation") or ""))
        n = len((data or {}).get("phrases") or [])
        return _report(self, problems, warnings, f"표현 {n}개 형식 통과", PHRASE_JSON_HELP)


@register
class TopicListValidator(Validator):
    name = "topic_list"
    description = "추가 지식 제안 JSON — 주제 3~5개, 주제마다 왜 도움이 되는지·첫 단원 (교육부 취미)"

    def check(self, answer, options):
        data = learning.find_json(answer, "topics")
        problems, warnings = topic_problems(data)
        n = len((data or {}).get("topics") or [])
        return _report(self, problems, warnings, f"추가 주제 {n}개 형식 통과", TOPIC_JSON_HELP)


@register
class ReviewQuizValidator(Validator):
    name = "review_quiz"
    description = "복습 퀴즈 — 공용 퀴즈 형식 + 문제마다 ref 가 과목 단어장·표현에 있는지 (교육부 취미, 간격 반복용)"

    def check(self, answer, options):
        problems = review_problems(answer, course_from_options(options), options.get("quiz_count"))
        if problems:
            return self.fail("복습 퀴즈 형식 오류: " + " / ".join(problems[:6])
                             + f"\n형식 예: ```json\n{REVIEW_JSON_HELP}\n```")
        return self.ok(f"복습 퀴즈 {len(review_refs(answer))}문제 형식 통과 (ref 모두 과목에 있음)")
