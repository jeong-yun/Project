"""기본 제공 검증기 — 모두 결정론적(같은 입력이면 항상 같은 결과)이고 외부 시스템을 건드리지 않는다.

  python_syntax : ```python 블록을 compile() 로 문법 검사 (실행하지 않음 → 안전)
  sql_sandbox   : ```sql 블록을 메모리 SQLite 샌드박스에서 검사 (스키마 생성 → EXPLAIN 컴파일 → 선택적으로 시간 제한 실행)
  json_keys     : ```json 블록이 올바른 JSON 인지 + 필수 키(단계 분할 키)를 모두 가졌는지 검사
  required_terms: 업무에 꼭 들어가야 할 항목(예: 시나리오·손절가·예산)이 답변에 있는지 검사
  itinerary_days: 여행 일정이 1일차~마지막 날까지 모든 날을 다루는지 검사 (일수는 코드가 계산)

ClickHouse 같은 실제 분석 DB 로 검증하려면, 같은 방식으로 Validator 를 하나 더 만들어
테스트 DB 에 쿼리를 보내고 문법 에러·타임아웃을 fail(log) 로 돌려주면 된다.
"""

import json
import sqlite3
import time
from datetime import date, timedelta

from core.config import BASE_DIR
from core.harness.base import Validator, extract_code_blocks, register


@register
class PythonSyntaxValidator(Validator):
    name = "python_syntax"
    description = "답변 속 ```python 코드 블록 문법 검사"

    def check(self, answer: str, options: dict):
        blocks = extract_code_blocks(answer, "python", "py")
        if not blocks:
            return self.ok("검사할 파이썬 코드 없음")
        for i, src in enumerate(blocks, 1):
            try:
                compile(src, f"<코드 블록 {i}>", "exec")
            except SyntaxError as e:
                return self.fail(f"코드 블록 {i} 의 {e.lineno}번째 줄 문법 오류: {e.msg} → {(e.text or '').strip()}")
        return self.ok(f"코드 블록 {len(blocks)}개 문법 통과")


@register
class SQLSandboxValidator(Validator):
    name = "sql_sandbox"
    description = "답변 속 ```sql 쿼리를 메모리 SQLite 샌드박스에서 검사"

    def check(self, answer: str, options: dict):
        blocks = extract_code_blocks(answer, "sql")
        if not blocks:
            return self.ok("검사할 SQL 없음")
        timeout = float(options.get("sql_timeout_sec", 2.0))
        conn = sqlite3.connect(":memory:")
        try:
            schema = options.get("sql_schema")
            if schema:  # 테스트용 테이블(과 샘플 데이터)을 먼저 만든다
                conn.executescript((BASE_DIR / schema).read_text(encoding="utf-8"))
            deadline = [0.0]
            # 실행이 길어지면 SQLite 가 이 함수를 주기적으로 부르고, 1 을 반환하면 쿼리를 중단한다 (타임아웃)
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline[0] else 0, 10_000)
            for i, stmt in enumerate(_split_statements("\n".join(blocks)), 1):
                deadline[0] = time.monotonic() + timeout
                try:
                    conn.execute(f"EXPLAIN {stmt}")  # 실행하지 않고 컴파일만: 문법·테이블·컬럼 오류 검출
                    if options.get("sql_execute") and stmt.lstrip().upper().startswith(("SELECT", "WITH")):
                        conn.execute(stmt).fetchmany(100)  # 실제 실행 (시간 제한)
                except sqlite3.OperationalError as e:
                    reason = "시간 초과(타임아웃)" if "interrupted" in str(e) else str(e)
                    return self.fail(f"쿼리 {i} 실패: {reason}\n{stmt.strip()[:300]}")
            return self.ok("모든 쿼리 통과")
        finally:
            conn.close()


def _split_statements(sql: str) -> list[str]:
    """세미콜론 기준으로 완결된 SQL 문장만 나눈다 (sqlite3.complete_statement 로 문자열 안 ; 를 구분)."""
    statements, buffer = [], ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip().rstrip(";").strip():
                statements.append(buffer.strip().rstrip(";"))
            buffer = ""
    if buffer.strip():
        statements.append(buffer.strip().rstrip(";"))
    return statements


@register
class JsonKeysValidator(Validator):
    name = "json_keys"
    description = "답변 속 ```json 블록의 형식과 필수 키(단계 분할 키) 검사"

    def check(self, answer: str, options: dict):
        required = options.get("required_keys", [])
        blocks = extract_code_blocks(answer, "json")
        if not blocks:
            return self.fail(f"JSON 블록이 없다. 필수 키 {required} 를 가진 ```json 블록으로 답하라.") if required else self.ok(
                "검사할 JSON 없음"
            )
        for i, src in enumerate(blocks, 1):
            try:
                data = json.loads(src)
            except json.JSONDecodeError as e:
                return self.fail(f"JSON 블록 {i} 파싱 오류: {e.msg} (줄 {e.lineno}, 칸 {e.colno})")
            missing = [k for k in required if not (isinstance(data, dict) and k in data)]
            if missing:
                return self.fail(f"JSON 블록 {i} 에 필수 키가 없다: {missing}")
        return self.ok(f"JSON 블록 {len(blocks)}개 통과")


@register
class RequiredTermsValidator(Validator):
    name = "required_terms"
    description = "업무 필수 항목(예: 시나리오·손절가·예산)이 답변에 있는지 검사"

    def check(self, answer: str, options: dict):
        # 항목 하나에 "A|B" 처럼 대안을 적으면 그중 하나만 있어도 통과
        required = options.get("required_terms", [])
        low = answer.lower()
        missing = [term for term in required if not any(alt.strip().lower() in low for alt in term.split("|"))]
        if missing:
            return self.fail(f"반드시 들어가야 할 항목이 빠졌다: {missing}. 이 항목들을 다루는 내용을 추가하라.")
        return self.ok(f"필수 항목 {len(required)}개 포함" if required else "검사할 필수 항목 없음")


@register
class ItineraryDaysValidator(Validator):
    name = "itinerary_days"
    description = "여행 일정이 모든 날(1일차~마지막 날)을 다루는지 검사"

    def check(self, answer: str, options: dict):
        days = int(options.get("trip_days") or 0)
        if days <= 0:
            return self.ok("검사할 여행 일정 정보 없음")
        start = options.get("trip_start")
        start_d = date.fromisoformat(start) if start else None
        compact = answer.replace(" ", "").lower()
        missing = []
        for i in range(1, days + 1):
            labels = [f"{i}일차", f"day{i}", f"{i}일째", f"day-{i}"]
            if start_d:
                d = start_d + timedelta(days=i - 1)
                labels += [f"{d.month}/{d.day}", f"{d.month}월{d.day}일", d.isoformat(), f"{d.month:02d}-{d.day:02d}", f"{d.month}.{d.day}"]
            if not any(label.lower() in compact for label in labels):
                missing.append(f"{i}일차")
        if missing:
            return self.fail(f"여행 기간 {days}일 중 일정이 없는 날: {missing}. 모든 날의 일정을 'n일차(날짜)' 형식으로 작성하라.")
        return self.ok(f"{days}일 모두 일정 있음")
