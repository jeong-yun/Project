"""🎓 교육부 공용 데이터 모델과 저장소 (화면 그리기 없음).

교육부 두 팀(🧭 직무 전환 교육팀 `edu_career` · 🌍 취미·교양 교육팀 `edu_hobby`)이 함께 쓰는 틀이다.
화면은 education_ui.py, 툴은 core/tools/learning_course.py, 검증기는 core/harness/learning_validators.py.

  수준       : LEVELS — 입문(신규) · 초급 · 중급 · 고급 · 전문 (언어 과목은 CEFR A1~C1 을 함께 표시)
  과목       : Course — 선생님(TutorProfile) · 커리큘럼(Unit 목록) · 수준 이력 · 퀴즈 결과 · 질문 기록 · extras
  저장소     : LearningStore — 데이터 폴더의 learning/<과목 id>.json (과목 하나 = 파일 하나, 잠금 + 원자적 교체)
  퀴즈       : parse_quiz(모델 답변 → Quiz) · grade_quiz(코드 채점) · can_level_up(80% 이상) · placement_level
  커리큘럼   : mark_unit_done · next_unit · progress_pct · change_level(남은 단원만 바꾸고 끝난 단원은 기록으로)
  문맥       : course_context(과목 → 근거 자료 글) — 모든 교육부 업무가 이 글로 선생님·수준·진도를 안다

════════════════════════════════════════════════════════════════════
2·3단계 세션과의 약속 (이 파일은 아래 '허용되는 변경'만 한다 — docs/PROPOSAL_교육부.md 맨 아래 안내)
════════════════════════════════════════════════════════════════════
① extras 키 이름 규칙 — 팀 전용 기록은 모두 Course.extras 에 넣는다 (Course 에 새 칸을 만들지 않는다)
   - 값은 JSON 으로 저장할 수 있는 것만 (dict · list · str · int · float · bool · None)
   - 직무 전환(edu_career): extras["projects"]  = [{"id", "title", "goal", "status", "milestones": [{"id", "title", "due", "done"}]}]
                            extras["roadmap"]   = {"target_role", "deadline", "steps": [...]}
                            extras["jd_links"]  = [{"title", "url", "note", "at"}]
   - 취미·교양(edu_hobby) : extras["vocab"]     = [{"term", "meaning", "example", "level", "added_at"}]
                            extras["phrases"]   = [{"situation", "phrase", "meaning", "level"}]
                            extras["topics"]    = [{"title", "why", "done"}]  (추가 지식 제안 — 역사·문화 등)
   - 새 키가 필요하면 위 목록(EXTRAS_KEYS)에 한 줄 추가만 한다. 다른 팀 키는 읽기만 한다.
② 과목 고치기는 LearningStore.update(course_id, fn) 로 — fn(course) 가 course 를 고치면 잠금 안에서 저장된다
     store.update(cid, lambda c: c.extras.setdefault("vocab", []).append({...}))
   단원 완료 표시는 mark_unit_done(course, unit_id, done=True), 수준 변경은 change_level(...) 만 쓴다 (이력이 남는다).
③ 과목 문맥 글 — course_context(course) 가 툴 근거 자료가 된다. 팀 전용 extras 를 근거 자료에 보이려면
   register_context("vocab", fn) 로 'fn(course) -> 마크다운 글'을 등록한다 (core/learning_career.py · core/learning_hobby.py 에서).
   이 두 모듈은 있으면 load_extensions() 가 자동으로 읽는다 (검증기 @register 도 그 모듈에 두면 같이 등록된다).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.config import BASE_DIR, load_settings

# ─── 수준 ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Level:
    id: str
    name: str  # 화면 이름 (예: 입문(신규))
    cefr: str  # 언어 과목에서 함께 보이는 CEFR 단계
    rule: str  # 이 수준의 기본 말투·난이도 규칙 (선생님 프로필에 따로 적은 규칙이 없을 때)


LEVELS: tuple[Level, ...] = (
    Level("beginner", "입문(신규)", "A1",
          "처음 배우는 사람이다. 짧은 문장과 일상 비유로 한 번에 개념 하나만 설명한다. 전문 용어·약어는 꼭 필요할 때만 쓰고, "
          "처음 나올 때 바로 괄호로 쉬운 말 풀이를 붙인다. 문제는 뜻 알기·고르기 수준."),
    Level("elementary", "초급", "A2",
          "기초 개념은 안다. 기본 용어는 써도 되지만 새 용어는 처음 나올 때 풀이한다. 예시를 따라 해 보는 연습 위주, 문제는 간단한 적용."),
    Level("intermediate", "중급", "B1",
          "기본기는 있다. 용어는 그대로 쓰고 왜 그런지(원리)와 비교·장단점을 설명한다. 작은 실습·사례 문제를 낸다."),
    Level("advanced", "고급", "B2",
          "실무·실전 수준이다. 설계 판단·예외·함정을 다루고 공식 문서·원문을 근거로 든다. 문제는 상황 판단·분석형."),
    Level("expert", "전문", "C1",
          "전문가 수준이다. 최신 동향·표준·논쟁점을 깊게 다루고, 학습자의 의견을 묻고 반박하며 토론하듯 가르친다."),
)
LEVEL_IDS: tuple[str, ...] = tuple(lv.id for lv in LEVELS)
LEVEL_UP_PCT = 80  # 레벨업 퀴즈 통과 기준 (정답률 %, 이상)


def level_of(level_id: str) -> Level:
    """수준 id → Level. 모르는 id 는 입문으로."""
    return next((lv for lv in LEVELS if lv.id == level_id), LEVELS[0])


def level_index(level_id: str) -> int:
    return LEVEL_IDS.index(level_id) if level_id in LEVEL_IDS else 0


def next_level(level_id: str) -> str | None:
    """다음 수준 id (이미 전문이면 None)."""
    i = level_index(level_id)
    return LEVEL_IDS[i + 1] if i + 1 < len(LEVEL_IDS) else None


def level_label(level_id: str, language: bool = False) -> str:
    """'초급' 또는 언어 과목이면 '초급 · A2'."""
    lv = level_of(level_id)
    return f"{lv.name} · {lv.cefr}" if language else lv.name


def parse_level(text: object) -> str:
    """화면 선택값·모델 답(수준 id · 이름 · CEFR · '신규')을 수준 id 로. 모르면 입문."""
    s = str(text or "").strip()
    for lv in LEVELS:
        if s == lv.id or s.startswith(lv.name) or s.startswith(lv.name.split("(")[0]) or s.upper().startswith(lv.cefr):
            return lv.id
    if "신규" in s or "처음" in s:
        return "beginner"
    return "beginner"


# ─── 과목 데이터 모델 ─────────────────────────────────────────────────

Kind = Literal["career", "hobby"]
KINDS: dict[str, str] = {"career": "🧭 직무 전환", "hobby": "🌍 취미·교양"}
TEAM_OF_KIND: dict[str, str] = {"career": "edu_career", "hobby": "edu_hobby"}
KIND_OF_TEAM: dict[str, str] = {v: k for k, v in TEAM_OF_KIND.items()}

# ① extras 키 이름 규칙 — 2·3단계가 새 키를 쓰려면 여기에 한 줄 추가만 한다 (키 → (주인 kind, 설명))
EXTRAS_KEYS: dict[str, tuple[str, str]] = {
    "projects": ("career", "개인 프로젝트·마일스톤"),
    "roadmap": ("career", "합격 로드맵"),
    "jd_links": ("career", "추천받은 JD 링크"),
    "vocab": ("hobby", "필수 단어·용어장"),
    "phrases": ("hobby", "상황별 표현"),
    "topics": ("hobby", "추가 지식(역사·문화) 제안"),
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TutorProfile(BaseModel):
    """선생님 프로필 — 이 과목 팀장의 페르소나 (근거 자료 '선생님 프로필'로 들어간다)."""

    name: str
    specialty: str = ""  # 전문 분야
    background: str = ""  # 경력 설정 (가상)
    teaching_style: str = ""  # 가르치는 방식
    level_rules: dict[str, str] = Field(default_factory=dict)  # 수준 id → 이 선생님의 말투·난이도 규칙

    @property
    def title(self) -> str:
        """화면 이름 — 이름이 이미 '선생님'으로 끝나면 그대로 (예: '한결 선생님'), 아니면 '마리아' → '마리아 선생님'."""
        return self.name if self.name.rstrip().endswith("선생님") else f"{self.name} 선생님"

    def rule_for(self, level_id: str) -> str:
        """이 수준에서 지킬 규칙 — 선생님이 따로 적은 규칙 + 공통 기본 규칙."""
        own = (self.level_rules.get(level_id) or "").strip()
        base = level_of(level_id).rule
        return f"{own} (공통 기준: {base})" if own else base


class Unit(BaseModel):
    """커리큘럼 단원 하나."""

    id: str  # "U1", "U2" … (과목 안에서 고유)
    week: int = 1  # 몇 주차
    title: str
    goals: list[str] = Field(default_factory=list)  # 이 단원을 마치면 할 수 있는 것
    level: str = "beginner"  # 이 단원을 만든 수준
    done: bool = False
    done_at: str = ""


class LevelChange(BaseModel):
    """수준 변화 이력 한 줄."""

    at: str
    from_level: str
    to_level: str
    how: Literal["start", "placement", "quiz", "manual"]  # 처음 · 수준 확인 퀴즈 · 레벨업 퀴즈 · 사람이 직접
    score_pct: int | None = None  # 퀴즈로 바뀌었으면 정답률
    note: str = ""
    replaced_units: list[str] = Field(default_factory=list)  # 이때 다시 만든(끝나지 않았던) 단원 제목 — 기록으로


class QuizResult(BaseModel):
    """퀴즈 채점 결과 (코드가 채점)."""

    at: str
    kind: Literal["placement", "levelup", "review"]
    level: str  # 퀴즈를 볼 때의 과목 수준
    total: int
    correct: int
    pct: int  # 정답률 % (정수, 내림)
    passed: bool
    wrong: list[str] = Field(default_factory=list)  # 틀린 문제 id


class QANote(BaseModel):
    """선생님에게 한 질문 기록 (요약)."""

    at: str
    question: str
    summary: str


class Course(BaseModel):
    """과목 하나 = 선생님 한 명 + 학습 기록. 개인 자료 — 데이터 폴더 learning/ 에만 저장한다."""

    id: str
    kind: Kind
    topic: str  # 배우는 분야 (예: 로봇 SW, 이탈리아어)
    goal: str = ""  # 목표 (예: 6개월 안에 로봇 SW 엔지니어 합격)
    level: str = "beginner"
    language: bool = False  # 언어 과목 — 수준 옆에 CEFR 표시
    weekly_hours: float = 0  # 주당 학습 시간
    tutor: TutorProfile
    curriculum: list[Unit] = Field(default_factory=list)
    level_history: list[LevelChange] = Field(default_factory=list)
    quiz_results: list[QuizResult] = Field(default_factory=list)
    qa_log: list[QANote] = Field(default_factory=list)
    extras: dict = Field(default_factory=dict)  # 2·3단계 팀 전용 기록 (키 규칙: EXTRAS_KEYS)
    created_at: str = ""
    updated_at: str = ""

    @property
    def team_id(self) -> str:
        return TEAM_OF_KIND[self.kind]

    @property
    def progress(self) -> int:
        return progress_pct(self)

    @property
    def level_name(self) -> str:
        return level_label(self.level, self.language)

    @property
    def title(self) -> str:
        return f"{KINDS[self.kind].split()[0]} {self.topic}"


COURSE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")  # 파일 이름이 되므로 경로 문자를 막는다


def new_course_id() -> str:
    return "c" + uuid.uuid4().hex[:8]


def valid_course_id(course_id: object) -> bool:
    return isinstance(course_id, str) and bool(COURSE_ID.match(course_id))


# ─── 커리큘럼 ────────────────────────────────────────────────────────


def progress_pct(course: Course) -> int:
    """진도 % = 끝낸 단원 수 / 전체 단원 수 (정수, 내림). 단원이 없으면 0."""
    total = len(course.curriculum)
    return sum(u.done for u in course.curriculum) * 100 // total if total else 0


def next_unit(course: Course) -> Unit | None:
    """다음에 할 단원 = 아직 안 끝낸 첫 단원."""
    return next((u for u in course.curriculum if not u.done), None)


def mark_unit_done(course: Course, unit_id: str, done: bool = True, now: str | None = None) -> Unit:
    """단원 완료 표시 (2·3단계도 이 함수만 쓴다). 없는 단원이면 KeyError."""
    unit = next((u for u in course.curriculum if u.id == unit_id), None)
    if unit is None:
        raise KeyError(f"과목 {course.id} 에 단원 {unit_id} 가 없습니다")
    unit.done, unit.done_at = done, (now or now_text()) if done else ""
    return unit


class UnitDraft(BaseModel):
    """모델이 만드는 단원 초안 (구조화 출력·JSON 공용)."""

    week: int = Field(description="몇 주차 (1부터, 끝난 단원 다음 주차부터 이어서)")
    title: str = Field(description="단원 제목 (짧게)")
    goals: list[str] = Field(description="이 단원을 마치면 할 수 있는 것 1~3개")


def _int(value: object, default: int = 1) -> int:
    try:
        return max(1, int(float(str(value))))
    except (TypeError, ValueError):
        return default


def units_from_drafts(drafts: list, level: str, start_index: int = 1, start_week: int = 1) -> list[Unit]:
    """단원 초안(UnitDraft · dict) → Unit. id 는 U{start_index}부터, 주차는 start_week 보다 앞서지 않게."""
    units, week = [], start_week
    for i, d in enumerate(drafts):
        d = d.model_dump() if isinstance(d, BaseModel) else dict(d)
        title = str(d.get("title") or "").strip()
        if not title:
            continue
        week = max(week, _int(d.get("week"), week))
        goals = d.get("goals") or []
        goals = [str(g).strip() for g in (goals if isinstance(goals, list) else [goals]) if str(g).strip()]
        units.append(Unit(id=f"U{start_index + len(units)}", week=week, title=title, goals=goals, level=level))
    return units


def replace_remaining(course: Course, new_units: list, level: str) -> list[str]:
    """끝난 단원은 그대로 두고, 안 끝난 단원을 new_units 로 바꾼다. 바뀐(지워진) 단원 제목을 돌려준다 (이력용)."""
    done = [u for u in course.curriculum if u.done]
    removed = [u.title for u in course.curriculum if not u.done]
    used = [int(u.id[1:]) for u in course.curriculum if u.id[1:].isdigit()]
    start_week = max((u.week for u in done), default=0) + 1
    fresh = units_from_drafts(new_units, level, start_index=max(used, default=0) + 1, start_week=start_week)
    course.curriculum = done + fresh
    return removed


def set_level(course: Course, to_level: str, how: str, *, score_pct: int | None = None, note: str = "",
              now: str | None = None) -> LevelChange:
    """수준만 바꾸고 이력을 남긴다 (남은 커리큘럼은 그대로 — 다시 만들려면 change_level)."""
    to_level = to_level if to_level in LEVEL_IDS else parse_level(to_level)
    change = LevelChange(at=now or now_text(), from_level=course.level, to_level=to_level, how=how, score_pct=score_pct, note=note)
    course.level = to_level
    course.level_history.append(change)
    return change


def change_level(course: Course, to_level: str, how: str, *, score_pct: int | None = None, note: str = "",
                 new_units: list | None = None, now: str | None = None) -> LevelChange:
    """수준 변경 + (new_units 가 있으면) 남은 커리큘럼 교체. 끝난 단원은 그대로, 바뀐 단원 제목은 이력에 남는다.

    new_units 는 plan_remaining(...) 이 만든 단원 초안 — 모델을 못 쓰면 None 으로 넘겨 수준만 바꾼다."""
    change = set_level(course, to_level, how, score_pct=score_pct, note=note, now=now)
    if new_units:
        change.replaced_units = replace_remaining(course, new_units, change.to_level)
    return change


# ─── 퀴즈 (모델이 문제를 내고, 채점은 코드가) ────────────────────────────

QuizKind = Literal["placement", "levelup", "review"]
QUIZ_KINDS: dict[str, str] = {
    "levelup": "레벨업 퀴즈 (다음 수준 문제 — 80% 이상이면 레벨업 제안)",
    "placement": "수준 확인 퀴즈 (여러 수준 섞어서 — 맞는 수준을 찾아 줌)",
    "review": "복습 퀴즈 (지금 수준 · 배운 단원)",
}


def parse_quiz_kind(text: object) -> str:
    s = str(text or "")
    for kid, label in QUIZ_KINDS.items():
        if s == kid or s.startswith(label.split(" (")[0]):
            return kid
    return "levelup"


class QuizQuestion(BaseModel):
    id: str
    question: str
    choices: list[str]
    answer: str  # choices 중 하나와 글자 그대로 같아야 한다
    explanation: str = ""
    level: str = ""  # 수준 확인 퀴즈에서 이 문제의 수준


class Quiz(BaseModel):
    kind: QuizKind = "levelup"
    level: str = ""  # 문제 수준 (레벨업 = 다음 수준)
    questions: list[QuizQuestion]


QUIZ_JSON_HELP = (
    '{"quiz": {"kind": "levelup", "level": "elementary", "questions": [{"id": "Q1", "question": "문제", '
    '"choices": ["보기1", "보기2", "보기3", "보기4"], "answer": "보기2", "explanation": "왜 정답인지", "level": "elementary"}]}}'
)


def json_blocks(text: str) -> list:
    """답변 속 ```json 블록을 읽는다 (깨진 블록은 건너뜀)."""
    from core.harness.base import extract_code_blocks  # 순환 import 방지

    out = []
    for src in extract_code_blocks(text or "", "json"):
        try:
            out.append(json.loads(src))
        except json.JSONDecodeError:
            continue
    return out


def find_json(text: str, key: str) -> dict | None:
    """key 를 가진 마지막 ```json 블록 (dict)."""
    found = [b for b in json_blocks(text) if isinstance(b, dict) and key in b]
    return found[-1] if found else None


def quiz_problems(data: object, expected_count: int | None = None, kind: str | None = None) -> list[str]:
    """퀴즈 JSON 의 형식 문제 목록 (빈 목록이면 통과) — 검증기 quiz_format 과 parse_quiz 가 함께 쓴다."""
    quiz = data.get("quiz") if isinstance(data, dict) and "quiz" in data else data
    if not isinstance(quiz, dict):
        return ['퀴즈 JSON 이 없다 — ```json 블록에 {"quiz": {...}} 형식으로 쓴다']
    questions = quiz.get("questions")
    if not isinstance(questions, list) or not questions:
        return ["quiz.questions 가 비어 있다 — 문제를 1개 이상 쓴다"]
    problems, ids = [], set()
    for i, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            problems.append(f"{i}번 문제가 객체가 아니다")
            continue
        qid = str(q.get("id") or f"Q{i}")
        if qid in ids:
            problems.append(f"문제 id {qid} 가 겹친다")
        ids.add(qid)
        if not str(q.get("question") or "").strip():
            problems.append(f"{qid}: question(문제 글)이 비어 있다")
        choices = q.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 6:
            problems.append(f"{qid}: choices(보기)는 2~6개 목록이어야 한다")
            continue
        texts = [str(c).strip() for c in choices]
        if any(not t for t in texts):
            problems.append(f"{qid}: 빈 보기가 있다")
        if len(set(texts)) != len(texts):
            problems.append(f"{qid}: 같은 보기가 두 번 있다")
        if str(q.get("answer") or "").strip() not in texts:
            problems.append(f"{qid}: answer(정답) '{q.get('answer')}' 가 보기 안에 글자 그대로 없다")
        if (kind or quiz.get("kind")) == "placement" and q.get("level") not in LEVEL_IDS:
            problems.append(f"{qid}: 수준 확인 퀴즈는 문제마다 level({', '.join(LEVEL_IDS)} 중 하나)을 쓴다")
    if expected_count and len(questions) != expected_count:
        problems.append(f"문제 수가 {len(questions)}개다 — {expected_count}개를 내야 한다")
    return problems


def parse_quiz(text: str) -> Quiz:
    """모델 답변 → Quiz. 형식이 틀리면 ValueError (이유 목록)."""
    data = find_json(text, "quiz")
    problems = quiz_problems(data)
    if problems:
        raise ValueError(" / ".join(problems[:5]))
    quiz = data["quiz"]
    questions = [QuizQuestion(id=str(q.get("id") or f"Q{i}"), question=str(q["question"]).strip(),
                              choices=[str(c).strip() for c in q["choices"]], answer=str(q["answer"]).strip(),
                              explanation=str(q.get("explanation") or ""), level=str(q.get("level") or ""))
                 for i, q in enumerate(quiz["questions"], 1)]
    kind = quiz.get("kind") if quiz.get("kind") in QUIZ_KINDS else "levelup"
    return Quiz(kind=kind, level=str(quiz.get("level") or ""), questions=questions)


@dataclass
class QuizGrade:
    """채점 결과 — 모두 정수 (부동소수점 비교 없음)."""

    total: int
    correct: int
    wrong: list[str] = field(default_factory=list)  # 틀린(또는 안 푼) 문제 id
    details: list[dict] = field(default_factory=list)  # [{id, picked, answer, ok, explanation}]

    @property
    def pct(self) -> int:
        return self.correct * 100 // self.total if self.total else 0

    def passed(self, threshold: int = LEVEL_UP_PCT) -> bool:
        return self.total > 0 and self.correct * 100 >= threshold * self.total


def grade_quiz(quiz: Quiz, answers: dict[str, str | None]) -> QuizGrade:
    """객관식 정답 대조 — answers = {문제 id: 고른 보기 글}. 안 푼 문제는 틀린 것으로 센다."""
    grade = QuizGrade(total=len(quiz.questions), correct=0)
    for q in quiz.questions:
        picked = answers.get(q.id)
        ok = picked is not None and str(picked).strip() == q.answer
        grade.correct += ok
        if not ok:
            grade.wrong.append(q.id)
        grade.details.append({"id": q.id, "picked": picked, "answer": q.answer, "ok": ok, "explanation": q.explanation})
    return grade


def can_level_up(course: Course, grade: QuizGrade, kind: str = "levelup") -> bool:
    """레벨업 제안 여부 — 레벨업 퀴즈 · 80% 이상 · 아직 맨 위 수준이 아님."""
    return kind == "levelup" and grade.passed() and next_level(course.level) is not None


def placement_level(quiz: Quiz, answers: dict[str, str | None]) -> str:
    """수준 확인 퀴즈 → 공부할 수준. 아래 수준부터 차례로 80% 이상 맞힌 마지막 수준의 '다음 수준'.

    예) 입문 문제 통과 · 초급 문제 실패 → 초급에서 공부. 하나도 통과 못 하면 가장 낮은 문제 수준. 맨 위까지 통과하면 그 수준."""
    by_level: dict[str, list[QuizQuestion]] = {}
    for q in quiz.questions:
        by_level.setdefault(q.level if q.level in LEVEL_IDS else "beginner", []).append(q)
    asked = sorted(by_level, key=level_index)
    placed = asked[0]
    for lv in asked:
        if not grade_quiz(Quiz(kind="placement", questions=by_level[lv]), answers).passed():
            break
        nxt = next_level(lv)
        placed = nxt if nxt else lv
    return placed


def quiz_result(course: Course, quiz: Quiz, grade: QuizGrade, now: str | None = None) -> QuizResult:
    return QuizResult(at=now or now_text(), kind=quiz.kind, level=course.level, total=grade.total, correct=grade.correct,
                      pct=grade.pct, passed=grade.passed(), wrong=grade.wrong)


# ─── 선생님 만들기 결과 → 과목 ─────────────────────────────────────────

TUTOR_JSON_HELP = (
    '{"tutor": {"name": "선생님 이름", "specialty": "전문 분야", "background": "경력 설정 (가상)", '
    '"teaching_style": "가르치는 방식", "level_rules": {"beginner": "입문일 때 말투·난이도", "elementary": "…", '
    '"intermediate": "…", "advanced": "…", "expert": "…"}}, '
    '"curriculum": [{"week": 1, "title": "단원 제목", "goals": ["마치면 할 수 있는 것"]}]}'
)


def tutor_problems(data: object) -> list[str]:
    """선생님 만들기 JSON 의 형식 문제 목록."""
    if not isinstance(data, dict):
        return ['선생님 JSON 이 없다 — ```json 블록에 {"tutor": {...}, "curriculum": [...]} 형식으로 쓴다']
    problems = []
    tutor = data.get("tutor")
    if not isinstance(tutor, dict) or not str(tutor.get("name") or "").strip():
        problems.append("tutor.name(선생님 이름)이 없다")
    elif not isinstance(tutor.get("level_rules", {}), dict):
        problems.append("tutor.level_rules 는 {수준 id: 규칙} 객체여야 한다")
    units = data.get("curriculum")
    if not isinstance(units, list) or not any(isinstance(u, dict) and str(u.get("title") or "").strip() for u in units):
        problems.append("curriculum(첫 커리큘럼 뼈대)에 단원이 없다")
    return problems


def parse_tutor_answer(text: str, level: str = "beginner") -> tuple[TutorProfile, list[Unit]]:
    """선생님 만들기 답변 → (선생님 프로필, 첫 커리큘럼). 형식이 틀리면 ValueError."""
    data = find_json(text, "tutor")
    problems = tutor_problems(data)
    if problems:
        raise ValueError(" / ".join(problems))
    t = data["tutor"]
    rules = {parse_level(k) if k not in LEVEL_IDS else k: str(v) for k, v in (t.get("level_rules") or {}).items() if str(v).strip()}
    tutor = TutorProfile(name=str(t["name"]).strip(), specialty=str(t.get("specialty") or ""),
                         background=str(t.get("background") or ""), teaching_style=str(t.get("teaching_style") or ""),
                         level_rules=rules)
    units = units_from_drafts([u for u in data["curriculum"] if isinstance(u, dict)], level)
    return tutor, units


def _hours(value: object) -> float:
    try:
        return max(0.0, float(str(value)))
    except (TypeError, ValueError):
        return 0.0


def course_from_answer(text: str, inputs: dict, course_id: str | None = None, now: str | None = None) -> Course:
    """'🎓 선생님 만들기' 채택 결과(사람이 고친 글) + 업무 입력값 → 새 과목. inputs 키: edu_kind · topic · goal · level · weekly_hours · language."""
    level = parse_level(inputs.get("level"))
    tutor, units = parse_tutor_answer(text, level)
    kind = inputs.get("edu_kind") if inputs.get("edu_kind") in KINDS else "career"
    stamp = now or now_text()
    course = Course(id=course_id or new_course_id(), kind=kind, topic=str(inputs.get("topic") or "").strip() or tutor.specialty,
                    goal=str(inputs.get("goal") or "").strip(), level=level, language=bool(inputs.get("language")),
                    weekly_hours=_hours(inputs.get("weekly_hours")), tutor=tutor, curriculum=units,
                    created_at=stamp, updated_at=stamp)
    course.level_history.append(LevelChange(at=stamp, from_level=level, to_level=level, how="start", note="과목 만들기"))
    return course


def summarize_answer(text: str, limit: int = 160) -> str:
    """질문 기록용 짧은 요약 — 답변에서 제목·코드·JSON 을 빼고 앞부분만 (모델 호출 없음)."""
    body = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    lines = [ln.strip(" #>*-") for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith("|")]
    short = " ".join(" ".join(lines).split())
    return short if len(short) <= limit else short[:limit].rstrip() + "…"


def add_qa(course: Course, question: str, answer: str, now: str | None = None, keep: int = 50) -> QANote:
    note = QANote(at=now or now_text(), question=" ".join(question.split())[:200], summary=summarize_answer(answer))
    course.qa_log = [*course.qa_log, note][-keep:]
    return note


# ─── 과목 문맥 글 (툴 근거 자료) ──────────────────────────────────────

ContextFn = Callable[[Course], str]
CONTEXT_HOOKS: dict[str, ContextFn] = {}  # extras 키 → 그 기록을 근거 자료 글로 만드는 함수 (2·3단계가 등록)


def register_context(key: str, fn: ContextFn) -> ContextFn:
    """③ extras[key] 를 근거 자료에 보이는 함수 등록 (2·3단계 모듈에서). fn(course) 가 빈 글이면 건너뛴다."""
    CONTEXT_HOOKS[key] = fn
    return fn


def course_context(course: Course, recent: int = 5) -> str:
    """과목 → 근거 자료 글: [선생님 프로필 · 학습자 수준과 그 수준의 규칙 · 커리큘럼과 진도 · 최근 기록 · 팀 전용 기록]."""
    load_extensions()
    t = course.tutor
    lines = [
        "### 👩‍🏫 선생님 프로필 — 이 과목 팀장은 이 선생님으로서 가르친다",
        f"- 이름: **{t.name}**" + (f" · 전문 분야: {t.specialty}" if t.specialty else ""),
    ]
    if t.background:
        lines.append(f"- 경력 설정(가상 — 말투·관점용이다. 경험담·합격 사례·통계의 근거로 쓰지 않는다): {t.background}")
    if t.teaching_style:
        lines.append(f"- 가르치는 방식: {t.teaching_style}")
    lines += [
        "",
        f"### 🎯 학습자 — 과목 `{course.id}` · {KINDS[course.kind]} · {course.topic}",
        f"- 목표: {course.goal or '(없음)'}" + (f" · 주당 {course.weekly_hours:g}시간" if course.weekly_hours else ""),
        f"- 지금 수준: **{level_label(course.level, course.language)}** ({len(course.level_history)}번째 기록)",
        f"- 이 수준의 말투·난이도 규칙 (반드시 지킨다): {t.rule_for(course.level)}",
    ]
    nxt = next_level(course.level)
    if nxt:
        lines.append(f"- 다음 수준: {level_label(nxt, course.language)} — 레벨업 퀴즈는 이 수준 문제로 낸다 ({LEVEL_UP_PCT}% 이상이면 통과)")
    lines += ["", f"### 📚 커리큘럼 — 진도 {progress_pct(course)}%"]
    if course.curriculum:
        lines += ["| 단원 | 주차 | 제목 | 목표 | 수준 | 완료 |", "|---|---|---|---|---|---|"]
        lines += [f"| {u.id} | {u.week} | {u.title} | {' / '.join(u.goals)} | {level_of(u.level).name} | {'✅' if u.done else ''} |"
                  for u in course.curriculum]
        nu = next_unit(course)
        lines.append(f"- 다음 단원: {nu.id} {nu.title}" if nu else "- 모든 단원을 끝냈다")
    else:
        lines.append("- 아직 단원이 없다")
    history = []
    for q in course.quiz_results[-recent:]:
        history.append(f"- 퀴즈 {q.at[:10]} · {QUIZ_KINDS[q.kind].split(' (')[0]} · {q.correct}/{q.total} ({q.pct}%)"
                       + (" · 통과" if q.passed else ""))
    for c in course.level_history[-recent:]:
        if c.how != "start":
            history.append(f"- 수준 {c.at[:10]} · {level_of(c.from_level).name} → {level_of(c.to_level).name} ({c.how})")
    for qa in course.qa_log[-recent:]:
        history.append(f"- 질문 {qa.at[:10]} · {qa.question[:80]} → {qa.summary[:100]}")
    if history:
        lines += ["", "### 🕘 최근 기록", *history]
    for key, fn in CONTEXT_HOOKS.items():
        if key in course.extras:
            try:
                text = fn(course).strip()
            except Exception as e:  # noqa: BLE001 — 팀 전용 기록 글이 깨져도 과목 문맥은 준다
                text = f"(extras['{key}'] 글 만들기 실패: {type(e).__name__})"
            if text:
                lines += ["", f"### 🗂 {EXTRAS_KEYS.get(key, ('', key))[1]}", text]
    return "\n".join(lines)


def level_table() -> str:
    """수준 표 (선생님 만들기 근거 자료)."""
    rows = ["| 수준 id | 이름 | CEFR(언어) | 기본 말투·난이도 |", "|---|---|---|---|"]
    rows += [f"| {lv.id} | {lv.name} | {lv.cefr} | {lv.rule} |" for lv in LEVELS]
    return "\n".join(rows)


# ─── 남은 커리큘럼 다시 만들기 (모델 1회) ──────────────────────────────


class CurriculumPlan(BaseModel):
    """수준이 바뀐 뒤 남은 커리큘럼 (구조화 출력)."""

    analysis: str = Field(description="먼저 새 수준과 끝낸 단원을 보고 무엇을 바꿔야 하는지 1~2문장")
    units: list[UnitDraft] = Field(description="끝낸 단원 다음부터 이어질 새 단원들 (바꾸기 전 남은 단원 수와 비슷하게)")


def plan_messages(course: Course, to_level: str):
    from langchain_core.messages import HumanMessage, SystemMessage  # 무거운 import 는 실제로 부를 때만

    done = [u for u in course.curriculum if u.done]
    left = [u for u in course.curriculum if not u.done]
    start_week = max((u.week for u in done), default=0) + 1
    system = (
        f"당신은 {course.tutor.title}({course.tutor.specialty})입니다. 가르치는 방식: {course.tutor.teaching_style}\n"
        f"학습자의 수준이 {level_label(course.level, course.language)} → {level_label(to_level, course.language)} 로 바뀌었습니다.\n"
        f"새 수준의 규칙: {course.tutor.rule_for(to_level)}\n"
        "끝낸 단원은 그대로 두고, 그 다음부터 이어질 남은 커리큘럼을 새 수준에 맞게 다시 짜세요. "
        f"주차는 {start_week}주차부터 이어 갑니다. 끝낸 내용을 반복하지 마세요.\n"
        # 선생님 만들기(teams/_shared/education/teaching.md)와 같은 규칙 — 이 프롬프트에는 팀 스킬이 들어가지 않는다
        "한국어로 쓴다. 한 주 분량은 " + (f"주당 학습 시간({course.weekly_hours:g}시간)" if course.weekly_hours else "주당 학습 시간")
        + " 안에 끝낼 수 있게 짠다. 단원 제목·목표에 시험 일정·회사·제도 이름을 지어내 넣지 않는다. "
        + (f"남은 단원 수는 바꾸기 전({len(left)}개)과 비슷하게(±2개) 한다." if left else "새 단원은 4~8개로 한다.")
    )
    human = "\n".join([
        f"과목: {course.topic} ({KINDS[course.kind]}) · 목표: {course.goal} · 주당 {course.weekly_hours:g}시간",
        "[끝낸 단원]", *(f"- {u.week}주차 {u.title}" for u in done), *(["- (없음)"] if not done else []),
        "[바꿀 남은 단원 — 참고]", *(f"- {u.week}주차 {u.title}" for u in left), *(["- (없음)"] if not left else []),
    ])
    return [SystemMessage(content=system), HumanMessage(content=human)]


def plan_remaining(course: Course, to_level: str, models, chain=None) -> list[UnitDraft]:
    """새 수준에 맞는 남은 단원 초안 (모델 1회 — chain 없으면 보조 모델). 실패하면 예외 — 화면이 수준만 바꾸고 알린다."""
    from core import llm_factory, usage  # 무거운 import 방지

    with usage.scope(purpose="커리큘럼 다시 만들기"):
        call = llm_factory.invoke_structured(chain or models.utility, models, CurriculumPlan, plan_messages(course, to_level),
                                             f"{course.tutor.title} (커리큘럼)")
    plan: CurriculumPlan = call.parsed
    return list(plan.units)


# ─── 저장소 ─────────────────────────────────────────────────────────


def default_data_dir() -> Path:
    """app.py 와 같은 규칙 — AGENT_OFFICE_DATA_DIR 또는 settings.yaml storage.data_dir (툴은 app 을 모른다)."""
    env = os.getenv("AGENT_OFFICE_DATA_DIR")
    return Path(env) if env else BASE_DIR / load_settings().get("storage", {}).get("data_dir", "data")


class LearningStore:
    """과목 저장소 — <데이터 폴더>/learning/<과목 id>.json. 고칠 때는 update() 로 (잠금 안에서 읽고 고치고 원자적 교체)."""

    _lock = threading.RLock()  # 한 프로세스(Streamlit) 안의 여러 스레드·화면이 동시에 고쳐도 덮어쓰지 않게
    FOLDER = "learning"

    def __init__(self, data_dir: Path | str | None = None):
        self.dir = Path(data_dir if data_dir is not None else default_data_dir()) / self.FOLDER

    def path(self, course_id: str) -> Path:
        if not valid_course_id(course_id):
            raise ValueError(f"과목 id 형식이 아닙니다: {course_id!r}")
        return self.dir / f"{course_id}.json"

    def get(self, course_id: str) -> Course | None:
        """없거나 id 형식이 틀리거나 파일이 깨졌으면 None."""
        if not valid_course_id(course_id):
            return None
        try:
            return Course.model_validate_json(self.path(course_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list(self) -> list[Course]:
        """최근에 고친 과목부터."""
        if not self.dir.is_dir():
            return []
        courses = [c for p in self.dir.glob("*.json") if (c := self.get(p.stem))]
        return sorted(courses, key=lambda c: (c.updated_at, c.id), reverse=True)

    def save(self, course: Course) -> Course:
        with self._lock:
            course.updated_at = now_text()
            path = self.path(course.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(course.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, path)  # 쓰는 도중 꺼져도 예전 파일이 깨지지 않게
            return course

    def update(self, course_id: str, fn: Callable[[Course], object]):
        """잠금 안에서 읽기 → fn(course) 로 고치기 → 저장. fn 의 반환값을 돌려준다. 없는 과목이면 KeyError."""
        with self._lock:
            course = self.get(course_id)
            if course is None:
                raise KeyError(f"과목이 없습니다: {course_id}")
            result = fn(course)
            self.save(course)
            return result

    def delete(self, course_id: str) -> None:
        with self._lock:
            self.path(course_id).unlink(missing_ok=True)


# ─── 2·3단계 확장 모듈 ────────────────────────────────────────────────

EXTENSION_MODULES = ("core.learning_career", "core.learning_hobby")  # 있으면 읽는다 (없어도 된다)
_loaded = False


def load_extensions() -> None:
    """2·3단계 모듈을 한 번 읽는다 — register_context · @register(검증기)가 그 안에서 등록된다."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for name in EXTENSION_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:  # 모듈은 있는데 그 안의 import 가 깨졌으면 숨기지 않는다
                raise
