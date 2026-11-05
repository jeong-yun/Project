"""🎓 교육부 공용 검증기 — 결정론적, 외부 호출 없음 (core/learning.py 가 퀴즈 형식·수준 표를 준다).

  quiz_format : 답변의 ```json {"quiz": …} 가 문제·보기·정답 형식을 지키고, 정답이 보기 안에 글자 그대로 있는지
                (options: quiz_count = 문제 수, quiz_kind = 레벨업/수준 확인/복습 — 수준 확인이면 문제마다 level 필수)
  level_fit   : 입문·초급 과목에 풀이 없는 전문 용어(영문 약어·영문 기술어)가 너무 많지 않은지 (휴리스틱)
                기준을 조금 넘으면 '⚠ 경고'만 남기고 통과, 두 배 넘게 많으면 다시 쓰게 한다. 중급 이상은 검사하지 않는다.
                (options: level = 과목 수준 — 툴 learning_course 가 넣는다, language = 언어 과목이면 영문 단어는 세지 않는다)

2·3단계 검증기는 core/learning_career.py · core/learning_hobby.py 에 @register 로 두면 아래 load_extensions() 가 등록한다
(이 파일과 core/harness/validators.py 를 두 세션이 함께 고치지 않게).
"""

import re

from core import learning
from core.harness.base import Validator, register

# ─── quiz_format ────────────────────────────────────────────────────


@register
class QuizFormatValidator(Validator):
    name = "quiz_format"
    description = "퀴즈 JSON 형식 — 문제·보기·정답, 정답이 보기 안에 있는지 (교육부)"

    def check(self, answer: str, options: dict):
        data = learning.find_json(answer, "quiz")
        try:
            count = int(str(options.get("quiz_count") or "0").split()[0])
        except ValueError:
            count = 0
        kind = learning.parse_quiz_kind(options["quiz_kind"]) if options.get("quiz_kind") else None
        problems = learning.quiz_problems(data, expected_count=count or None, kind=kind)
        if problems:
            return self.fail("퀴즈 형식 오류: " + " / ".join(problems[:6])
                             + f"\n형식 예: ```json\n{learning.QUIZ_JSON_HELP}\n```")
        n = len(data["quiz"]["questions"])
        return self.ok(f"퀴즈 {n}문제 형식 통과 (정답이 모두 보기 안에 있음)")


# ─── level_fit ──────────────────────────────────────────────────────

# 쉬운 말로 이미 굳은 약어 — 풀이 없이 써도 세지 않는다
COMMON_TERMS = {"OK", "AI", "IT", "PC", "TV", "SNS", "PDF", "USB", "CEO", "FAQ", "QA", "Q", "A", "JSON", "URL", "GPS", "ID",
                "CEFR", "A1", "A2", "B1", "B2", "C1", "C2"}
SOFT_LIMIT = {"beginner": 3, "elementary": 6}  # 풀이 없는 전문 용어 종류 수 — 넘으면 경고, 두 배를 넘으면 다시 쓰기
ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,}[a-z]?(?![A-Za-z0-9])")  # API · TCP · HTTPs · ROS2
EN_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z][a-z]+(?:[A-Z][a-z]+)*(?![A-Za-z])")  # middleware · kinematics · camelCase
EXPLAIN_AFTER = re.compile(r"^\s*(\(|（|—|:|=|\[|이란|란\b|이라는|라는|은 뜻|는 뜻|의 뜻)")


def _strip_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)


def unexplained_terms(text: str, language: bool = False) -> list[str]:
    """풀이 없이 쓴 전문 용어 (처음 나온 순서, 종류마다 하나).

    '풀이 있음' = 그 용어가 한 번이라도 바로 뒤에 괄호·대시·콜론·'이란/란/이라는/라는' 이 붙어 나오거나, 괄호 안에 들어 있으면
    (예: 'API(프로그램끼리 대화하는 약속)', '응용 프로그램 연결(API)').
    언어 과목은 배우는 말 자체가 영문이라 영문 낱말은 세지 않고 약어만 센다."""
    body = _strip_code(text)
    found: dict[str, bool] = {}
    patterns = [ACRONYM] if language else [ACRONYM, EN_WORD]
    for pattern in patterns:
        for m in pattern.finditer(body):
            term = m.group(0)
            if term.upper() in COMMON_TERMS or (pattern is EN_WORD and len(term) < 4):
                continue
            after, before = body[m.end():m.end() + 12], body[max(0, m.start() - 2):m.start()]
            explained = bool(EXPLAIN_AFTER.match(after)) or before.endswith(("(", "（"))
            found[term] = found.get(term, False) or explained
    return [t for t, ok in found.items() if not ok]


@register
class LevelFitValidator(Validator):
    name = "level_fit"
    description = "입문·초급 과목에 풀이 없는 전문 용어가 너무 많은지 (교육부 · 휴리스틱, 조금 넘으면 경고만)"

    def check(self, answer: str, options: dict):
        level = options.get("level") or ""
        if level not in SOFT_LIMIT:
            return self.ok("중급 이상(또는 수준 정보 없음) — 용어 검사를 하지 않음")
        terms = unexplained_terms(answer, bool(options.get("language")))
        limit, name = SOFT_LIMIT[level], learning.level_of(level).name
        listed = ", ".join(terms[:12])
        if len(terms) > limit * 2:
            return self.fail(f"{name} 학습자에게 풀이 없는 전문 용어가 {len(terms)}개다: {listed}. 처음 나올 때 괄호로 쉬운 말 풀이를 붙이거나 "
                             f"쉬운 말로 바꿔 {limit}개 이하로 줄여라.")
        if len(terms) > limit:
            return self.ok(f"⚠ 경고: {name} 학습자에게 풀이 없는 용어 {len(terms)}개 ({listed}) — 처음 나올 때 풀이를 붙이면 더 좋다")
        return self.ok(f"{name} 수준 용어 확인 통과 (풀이 없는 용어 {len(terms)}개)")


learning.load_extensions()  # 2·3단계 모듈의 @register 검증기도 함께 등록
