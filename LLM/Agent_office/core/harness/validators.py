"""기본 제공 검증기 — 모두 결정론적(같은 입력이면 항상 같은 결과)이고 외부 시스템을 건드리지 않는다.

  python_syntax : ```python 블록을 compile() 로 문법 검사 (실행하지 않음 → 안전)
  sql_sandbox   : ```sql 블록을 메모리 SQLite 샌드박스에서 검사 (스키마 생성 → EXPLAIN 컴파일 → 선택적으로 시간 제한 실행)
  json_keys     : ```json 블록이 올바른 JSON 인지 + 필수 키(단계 분할 키)를 모두 가졌는지 검사
  required_terms: 업무에 꼭 들어가야 할 항목(예: 시나리오·손절가·예산)이 답변에 있는지 검사
  itinerary_days: 여행 일정이 1일차~마지막 날까지 모든 날을 다루는지 검사 (일수는 코드가 계산)
  stock_plan    : 주식 점수표·매매 계획 JSON 을 코드로 검산 (core/stock_calc.py)
  concept_entry : 개념 카탈로그 결과 형식 검사 (core/concept.py)
  music_prompt  : 음악 팀 가사·Lyria 프롬프트 형식 검사 (core/music.py)
  floorplan     : 설계 팀 레이아웃 JSON 을 코드로 완성·검증 (겹침·외곽·면적·동선·계단·최소 치수, core/floorplan.py)
  control_ids   : 통제 번호가 고른 프레임워크 공개 목록(config/controls)에 있는지 (core/governance.py)
  evidence_quotes: 인용한 증적 문구가 입력 증적에 글자 그대로 있는지 (core/governance.py)

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


@register
class StockPlanValidator(Validator):
    name = "stock_plan"
    description = "주식 업무의 점수표·매매 계획 JSON 을 코드로 다시 계산해 검산 (가중 점수·결정 밴드·평균가·손익비·평단가)"

    def check(self, answer: str, options: dict):
        from core import stock_calc  # stock_calc 가 harness.base 를 쓰므로 순환 import 를 피해 늦게 불러온다

        results = stock_calc.run_checks(answer, options)
        if not results:
            return self.ok("검산할 주식 계획 없음")
        problems = [p for r in results for p in r.problems]
        if problems:
            return self.fail(
                "코드 검산 결과 다음이 틀렸다. 코드가 계산한 값으로 본문과 JSON 을 함께 고쳐라: " + " / ".join(problems)
            )
        return self.ok("검산 통과: " + ", ".join(r.kind for r in results))


@register
class ConceptEntryValidator(Validator):
    name = "concept_entry"
    description = "개념 카탈로그 결과 형식 검사 (<review_log>·<final>·개조식·절대 표현·참고문헌 형식·적용 계획 JSON)"

    def check(self, answer: str, options: dict):
        from core.concept import check_answer  # concept 가 harness.base 를 쓰므로 늦게 불러온다

        problems = check_answer(answer, options)
        if problems:
            return self.fail("개념 결과 형식 오류: " + " / ".join(problems) + " — 이 부분을 고쳐 같은 형식으로 다시 작성하라.")
        return self.ok("개념 결과 형식 통과")


@register
class MusicPromptValidator(Validator):
    name = "music_prompt"
    description = "음악 결과 형식 검사 (가사·Lyria 블록, 반주 문구, 섹션 태그·분량, 아티스트 이름, 시간표 길이)"

    def check(self, answer: str, options: dict):
        from core.music import check_answer  # music 이 harness.base 를 쓰므로 늦게 불러온다

        problems = check_answer(answer, options)
        if problems:
            return self.fail("음악 결과 형식 오류: " + " / ".join(problems) + " — 이 부분을 고쳐 같은 블록 형식으로 다시 작성하라.")
        return self.ok("음악 결과 형식 통과")


@register
class FloorplanValidator(Validator):
    name = "floorplan"
    description = ("설계 팀 레이아웃 JSON 검증 (방 겹침·외곽 이탈·목표 면적·문으로 도달·층간 계단·최소 치수 "
                   "+ 설계 품질: 장단변비·복도 몫·필수 이웃·채광·물 쓰는 곳 모으기)")

    def check(self, answer: str, options: dict):
        from core import floorplan  # shapely 를 쓰는 팀에서만 불러온다

        try:
            layout = floorplan.layout_from_text(answer)
        except floorplan.LayoutParseError as e:
            return self.fail(f"{e} — 같은 형식의 ```json 블록으로 다시 써라.")
        if layout is None:
            # 하위 실무자(대리·방프로그램)는 JSON 을 쓰지 않는다 — JSON 유무는 json_keys(final_only + also_at)가 본다
            return self.ok("검사할 레이아웃 JSON 없음")
        tolerance = options.get("area_tolerance")
        _, report = floorplan.check(layout, float(tolerance) if tolerance is not None else None)
        metrics = [n for n in report.notes if n.startswith("설계 지표")]
        if not report.ok:
            hints = [n for n in report.notes if "zone" in n or "목표 면적 합" in n or "단순 분할" in n][:3]
            return self.fail(
                "레이아웃 검증 실패: " + " / ".join(report.errors[:8])
                + (" (참고: " + " / ".join(hints + metrics) + ")" if hints or metrics else "")
                + " — 좌표를 직접 고치기 어렵다면 그 층의 rect 를 모두 빼고 방 목록·목표 면적·구역(zone)·이웃(adjacent)·typology 만 써서 "
                "코드(배치 엔진)가 배치하게 하라."
            )
        floors = len(layout.floors)
        rooms = sum(len(f.rooms) for f in layout.floors)
        warn = f", 경고 {len(report.warnings)}건: " + " / ".join(report.warnings[:4]) if report.warnings else ""
        return self.ok(f"레이아웃 검증 통과: {floors}개 층, 방 {rooms}개{warn}" + (" · " + " · ".join(metrics) if metrics else ""))


@register
class ControlIdsValidator(Validator):
    name = "control_ids"
    description = "통제 번호 검증 — 답변의 통제 번호가 고른 프레임워크(ISMS-P·ISO 27001:2022·NIST CSF 2.0) 공개 목록에 모두 있는지 (config/controls)"

    def check(self, answer: str, options: dict):
        from core import governance  # governance 가 harness.base 를 쓰므로 늦게 불러온다

        framework = str(options.get("control_framework") or "").strip()
        if not framework:
            framework = str(((governance.answer_json(answer) or {}).get("summary") or {}).get("framework") or "").strip()
        if not framework:
            return self.ok("검사할 프레임워크 정보 없음")
        try:
            result = governance.check_control_ids(answer, framework)
        except KeyError as e:
            return self.fail(f"{e} — summary.framework 에 위 목록의 이름 하나를 정확히 써라.")
        problems = []
        if result.unknown:
            problems.append(f"{result.framework} 목록에 없는 통제 번호 {len(result.unknown)}개: {', '.join(result.unknown[:12])}")
        problems += result.hints[:6]
        if result.name_mismatch:
            problems.append("번호와 이름이 맞지 않음: " + " / ".join(result.name_mismatch[:6]))
        if problems:
            return self.fail(" / ".join(problems) + " — 근거 자료의 '통제 번호 목록'에 있는 번호·이름만 쓰고, 맞는 번호를 모르면 "
                             "control_id 를 비우고 이름만 써라 (번호를 지어내지 않는다).")
        return self.ok(f"{result.framework} 통제 번호 {result.checked}개 모두 목록에 있음")


@register
class EvidenceQuotesValidator(Validator):
    name = "evidence_quotes"
    description = "인용 문구 검증 — 답변 JSON 의 evidence(인용 증적)가 입력 증적에 글자 그대로 있는지 (공백은 하나로 맞춰 비교)"

    def check(self, answer: str, options: dict):
        from core import governance

        source = str(options.get("evidence_text") or "")
        if not source.strip():
            return self.ok("비교할 입력 증적 없음")
        result = governance.check_quotes(answer, source)
        problems = []
        if result.missing:
            problems.append(f"입력 증적에 없는 인용 {len(result.missing)}건: " + " / ".join(result.missing[:8]))
        if result.no_evidence:
            problems.append(f"인용 없이 판정한 항목 {len(result.no_evidence)}건: {', '.join(result.no_evidence[:8])}")
        if problems:
            return self.fail(" / ".join(problems) + " — evidence 에는 증적 원문을 한 글자도 바꾸지 말고 그대로 복사해 넣어라 "
                             "(요약·번역·고쳐 쓰기 금지, 떨어진 두 곳은 '…'로 잇는다). 인용할 증적이 없으면 status 를 '확인 불가'로 쓴다.")
        return self.ok(f"인용 {result.checked}건 모두 입력 증적에 있음")
