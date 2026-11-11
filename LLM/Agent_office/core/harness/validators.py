"""기본 제공 검증기 — 모두 결정론적(같은 입력이면 항상 같은 결과)이고 외부 시스템을 건드리지 않는다.

  python_syntax : ```python 블록을 compile() 로 문법 검사 (실행하지 않음 → 안전)
  sql_sandbox   : ```sql 블록을 메모리 SQLite 샌드박스에서 검사 (스키마 생성 → EXPLAIN 컴파일 → 선택적으로 시간 제한 실행)
  json_keys     : ```json 블록이 올바른 JSON 인지 + 필수 키(단계 분할 키)를 모두 가졌는지 검사
  required_terms: 업무에 꼭 들어가야 할 항목(예: 시나리오·손절가·예산)이 답변에 있는지 검사
  itinerary_days: 여행 일정이 1일차~마지막 날까지 모든 날을 다루는지 검사 (일수는 코드가 계산)
  budget_total  : 여행 계획서의 합계가 양식의 총예산을 넘는데 요약에서 밝히지 않았는지 (금액은 코드가 읽음, 확실할 때만)
  stock_plan    : 주식 점수표·매매 계획 JSON 을 코드로 검산 (core/stock_calc.py)
  concept_entry : 개념 카탈로그 결과 형식 검사 (core/concept.py)
  music_prompt  : 음악 팀 가사·Lyria 프롬프트 형식 검사 (core/music.py)
  floorplan     : 설계 팀 레이아웃 JSON 을 코드로 완성·검증 (겹침·외곽·면적·동선·계단·최소 치수, core/floorplan.py)
  control_ids   : 통제 번호가 고른 프레임워크 공개 목록(config/controls)에 있는지 (core/governance.py)
  evidence_quotes: 인용한 증적 문구가 입력 증적에 글자 그대로 있는지 (core/governance.py)
  iac_fix       : IaC 수정안 diff 를 원본 사본에 적용 → 내장 규칙 재검사로 정말 고쳤는지 (core/iac_patch.py)
  career_facts  : 자소서·포트폴리오의 인용이 올린 문서에 글자 그대로 있는지 + 본문 숫자가 문서·JD·문항에 있는지 (core/career.py)
  char_limits   : 자소서 문항마다 답변이 있고 글자 수 제한(공백 포함/제외)을 지키는지 (코드가 센다)
  jd_coverage   : JD 요구사항(R1…)을 하나도 빼지 않고 근거 있음/부분/공백으로 판정했는지
  portfolio_format: 포트폴리오 절(요약·역량 표·프로젝트)과 프로젝트 카드 다섯 항목 — 형식 지정이 있으면 원본 절 제목·표 머리글,
                  원본 그림이 있으면 [그림 N] 표시
  jd_recommend  : JD 추천 — 모든 JD·요구사항 판정 + 순위·추천이 코드가 계산한 점수 순서와 같은지 (점수는 코드가 셈)
  portfolio_changes: 포트폴리오 최신화 변경 목록 — 이전 글 그대로 · 문서 인용 그대로 · 외부 출처는 승인한 URL · 외부 숫자 금지

ClickHouse 같은 실제 분석 DB 로 검증하려면, 같은 방식으로 Validator 를 하나 더 만들어
테스트 DB 에 쿼리를 보내고 문법 에러·타임아웃을 fail(log) 로 돌려주면 된다.
"""

import json
import re
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


_WON = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*만\s*(?:(\d)\s*천)?\s*원|(\d[\d,]{3,})\s*원")
_NOT_A_TOTAL = re.compile(r"1인|인당|명당|1명|하루|1일|일당|1박|박당|제외|별도|빼고|각자|씩")  # 총액이 아닌 예산 표현
_DAY_ONE = re.compile(r"(?<!\d)1\s*일\s*차|day\s*-?\s*1(?!\d)", re.IGNORECASE)


def won(text: str) -> list[int]:
    """글 속 원화 금액을 나온 순서대로 — '780,000원', '78만 원', '78만 5천원'. 단위(원)가 없는 숫자·외화는 뺀다."""
    return [int(m.group(3).replace(",", "")) if m.group(3)
            else round(float(m.group(1).replace(",", "")) * 10000) + int(m.group(2) or 0) * 1000 for m in _WON.finditer(text)]


def _won_text(amount: int) -> str:
    return f"{amount // 10000:,}만 원" if amount % 10000 == 0 else f"{amount:,}원"


def _budget_limit(text: str) -> int | None:
    """예산 칸에서 총예산(원) — '총 80만원'처럼 총액 표시가 있고 금액이 하나뿐일 때만 (1인·하루 기준, 항목 제외는 모름)."""
    amounts = won(text)
    if len(amounts) != 1 or not any(w in text for w in ("총", "전체", "합계", "합쳐", "모두")) or _NOT_A_TOTAL.search(text):
        return None
    return amounts[0]


def _stated_total(answer: str) -> int | None:
    """계획서가 밝힌 합계(원) — '합계' 줄에서 '합계' 뒤 첫 금액. 확실할 때만 돌려준다.

    표의 합계 줄만 본다(표가 없으면 글의 합계 줄). 금액 앞에 다른 숫자가 있거나(외화·단위 없는 숫자) 합계 줄마다 금액이 다르면 모름.
    """
    lines = [line for line in answer.splitlines() if "합계" in line]
    totals = set()
    for line in [line for line in lines if line.lstrip().startswith("|")] or lines:
        tail = line.split("합계", 1)[1]
        m = _WON.search(tail)
        if not m:
            continue  # 금액 없는 합계 줄 (제목 등)
        if re.search(r"\d", tail[: m.start()]):
            return None
        totals.add(won(m.group())[0])
    return totals.pop() if len(totals) == 1 else None


@register
class BudgetTotalValidator(Validator):
    name = "budget_total"
    description = ("여행 예산 합계 — 양식의 예산이 '총 80만원'처럼 총액 하나이고 계획서 '합계' 줄의 원화 금액이 분명할 때, "
                   "합계가 총예산을 넘으면 한 줄 요약(첫 일정 앞)에서 넘는다고 밝혔는지 (확실하지 않으면 통과)")

    def check(self, answer: str, options: dict):
        limit = _budget_limit(str(options.get("trip_budget") or ""))
        if limit is None:
            return self.ok("검사할 총예산 없음 (예산 칸이 '총 80만원'처럼 총액 하나일 때만 검사)")
        total = _stated_total(answer)
        if total is None:
            return self.ok("합계 금액을 확실히 읽지 못해 검사 생략 ('합계' 줄의 원화 금액이 하나로 분명할 때만 검사)")
        if total <= limit:
            return self.ok(f"합계 {_won_text(total)} — 총예산 {_won_text(limit)} 안")
        day_one = _DAY_ONE.search(answer)
        if re.search("초과|넘", answer[: day_one.start()] if day_one else answer):
            return self.ok(f"합계({_won_text(total)})가 총예산({_won_text(limit)})을 넘지만 요약에서 넘는다고 밝힘")
        over = _won_text(total - limit)
        return self.fail(
            f"예산 표의 합계({_won_text(total)})가 총예산({_won_text(limit)})을 {over} 넘는데, 한 줄 요약(첫 일정 앞)에 넘는다는 말이 없다. "
            f"항목을 줄여 합계를 {_won_text(limit)} 이하로 맞추고 표의 금액도 함께 고쳐라. 줄일 수 없으면 한 줄 요약에 "
            f"'총예산 {_won_text(limit)}을 약 {over} 초과한다'고 밝히고, '예산 안에 든다'는 표현은 지워라."
        )


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


@register
class IacFixValidator(Validator):
    name = "iac_fix"
    description = ("IaC 수정안 재검사 — 답변의 diff 를 원본 사본에 적용해 내장 규칙으로 다시 검사 "
                   "(발견이 사라졌나 · 같거나 더 높은 새 문제가 없나 · 파일이 읽히나), 발견 번호(F01…)가 점검 결과에 있는지 (core/iac_patch.py)")

    def check(self, answer: str, options: dict):
        from core import iac_patch  # iac_patch 가 harness.base 를 쓰므로 늦게 불러온다

        files = options.get("iac_files") or {}
        if not files:
            return self.ok("검사할 IaC 없음")
        v = iac_patch.verify(files, answer, require_fix=bool(options.get("iac_require_fix")),
                             cover_all=bool(options.get("iac_cover_all")))
        problems = iac_patch.feedback(v)
        if problems:
            return self.fail("IaC 수정안 검증 실패: " + " / ".join(problems[:8])
                             + " — diff 의 ' '(문맥)·'-'(지울) 줄은 근거 자료의 원본 파일을 글자 그대로 복사하고, 리소스를 지우지 말고 "
                             "설정 값을 바꿔라. 고치지 못하는 발견은 fixes 에서 빼고 not_attempted 에 이유와 함께 적어라.")
        fixed = [fid for fid, s in v.status.items() if s == "fixed"]
        if v.diffs:
            return self.ok(f"수정안 재검사 통과: 발견 {len(v.before.findings)}건 중 고침 {len(fixed)} · 시도 안 함 "
                           f"{v.count('not_attempted')} ({', '.join(fixed)}) — 재검사 {len(v.before.findings)}건 → {len(v.after.findings)}건")
        return self.ok(f"발견 번호 확인 통과: 점검 결과 {len(v.before.findings)}건 (수정안 diff 없음)")


@register
class CareerFactsValidator(Validator):
    name = "career_facts"
    description = ("지어낸 사실 검사 — JSON 의 인용(quote)이 올린 문서에 글자 그대로 있는지, 본문 숫자가 문서·JD·문항에 있는지 "
                   "([확인 필요] 문장은 제외, core/career.py)")

    def check(self, answer: str, options: dict):
        from core import career  # career 가 harness.base 를 쓰므로 늦게 불러온다

        if "career_source" not in options:
            return self.ok("비교할 문서 정보 없음")
        result = career.check_facts(answer, str(options.get("career_source") or ""), str(options.get("career_reference") or ""))
        problems = []
        if result.missing_quotes:
            problems.append(f"올린 문서에 없는 인용 {len(result.missing_quotes)}건: " + " / ".join(result.missing_quotes[:6]))
        if result.unsourced_numbers:
            problems.append(f"문서·JD·문항에 없는 숫자 {len(result.unsourced_numbers)}개: " + " / ".join(result.unsourced_numbers[:8]))
        if result.no_evidence:
            problems.append("JSON evidence(본문 주장마다 문서 원문 인용)가 없다")
        if problems:
            return self.fail(" / ".join(problems) + " — quote 에는 근거 자료 '올린 문서 원문'을 한 글자도 바꾸지 말고 복사하고, 숫자는 문서에 "
                             "적힌 값·단위 그대로 쓴다. 문서에 없는 경험·숫자는 지우거나 그 문장 끝에 [확인 필요]를 붙이고 needs_check 에 적는다.")
        return self.ok(f"인용 {result.quotes_checked}건 모두 문서에 있음 · 본문 숫자 모두 문서·JD·문항에 있음")


@register
class CharLimitsValidator(Validator):
    name = "char_limits"
    description = "자소서 글자 수 — 문항마다 '## 문항 N.' 답변이 있고, 제한(공백 포함/제외)을 넘지 않고 너무 짧지 않은지 (코드가 셈)"

    def check(self, answer: str, options: dict):
        from core import career

        questions = [career.Question(**q) for q in options.get("career_questions") or []]
        if not questions:
            return self.ok("검사할 자소서 문항 없음")
        problems = career.check_char_limits(answer, questions, float(options.get("min_fill_ratio", 0.5)))
        if problems:
            return self.fail("자소서 글자 수·문항 오류: " + " / ".join(problems))
        rows = career.limit_rows(answer, questions)
        return self.ok("문항별 글자 수 통과: " + ", ".join(
            f"{r.no}번 {r.counted:,}/{r.limit:,}자" if r.limit else f"{r.no}번 {r.with_spaces:,}자" for r in rows))


@register
class JdCoverageValidator(Validator):
    name = "jd_coverage"
    description = "JD 대응 — 코드가 뽑은 JD 요구사항(R1…)을 JSON jd_coverage 에서 모두 근거 있음/부분/공백으로 판정했는지"

    def check(self, answer: str, options: dict):
        from core import career

        ids = list(options.get("career_requirements") or [])
        if not ids:
            return self.ok("검사할 JD 요구사항 없음")
        problems = career.check_coverage_ids(answer, ids)
        if problems:
            return self.fail("JD 대응표 오류: " + " / ".join(problems) + " — 키워드를 본문에 끼워 넣지 말고, 요구사항마다 문서 근거를 판정해 "
                             "jd_coverage 에 적는다 (근거가 없으면 '공백').")
        return self.ok(f"JD 요구사항 {len(ids)}개 모두 판정")


@register
class PortfolioFormatValidator(Validator):
    name = "portfolio_format"
    description = ("포트폴리오 형식 — 형식 지정이 없으면 요약·핵심 역량(표)·프로젝트 절과 카드 다섯 항목(문제·역할·기술·결과·배운 점), "
                   "형식 지정(추가 요청·형식 참고 파일)이 있으면 원본 절 제목·표 머리글이 결과에 그대로 있는지, "
                   "원본 그림이 있으면 '[그림 N]' 표시가 모두 남았는지")

    def check(self, answer: str, options: dict):
        from core import career

        figures = list(options.get("career_figures") or [])
        lost = career.missing_figures(answer, len(figures)) if figures else []
        figure_problem = ([f"원본 그림 {', '.join(f'[그림 {n}]' for n in lost)} 이 빠졌다 — 그 그림이 있던 항목(예: 흐름도 칸) 아래에 "
                           "'[그림 N]' 한 줄을 글자 그대로 남겨라 (Word 로 내보낼 때 코드가 그 자리에 원본 그림을 넣는다)"] if lost else [])
        figure_note = f" · 그림 표시 {len(figures)}개 모두 남음" if figures and not lost else ""
        figure_problem += career.section_problems(answer)  # 형식 구간(docx·md)을 썼으면 두 구간 모두
        fmt = options.get("career_format")
        if isinstance(fmt, dict) and fmt.get("mode") == "original":
            sources = ", ".join(fmt.get("sources") or []) or "원본"
            if not (fmt.get("headings") or fmt.get("table_heads")):
                problems = career.original_format_problems(answer, {}) + figure_problem
                if problems:
                    return self.fail("포트폴리오 형식 오류: " + " / ".join(problems))
                return self.ok(f"요청 형식 — 비교할 원본 절 제목·표가 없어({sources}) 생략 문구만 확인 (표준 형식 검사는 하지 않음){figure_note}")
            problems = career.original_format_problems(answer, fmt) + figure_problem
            if problems:
                return self.fail("요청 형식 오류: " + " / ".join(problems) + f" — 추가 요청대로 원본({sources})의 절 제목·표 머리글·구분·순서를 "
                                 "그대로 쓴다. 표준 형식(요약·핵심 역량·카드)으로 바꾸거나 요청 형식을 부록으로 미루지 않는다.")
            return self.ok(f"요청 형식 통과 — 원본({sources}) 절 제목 {len(fmt.get('headings') or [])}개 · "
                           f"표 {len(fmt.get('table_heads') or [])}개가 결과에 있음{figure_note}")
        problems = career.check_portfolio(answer) + figure_problem
        if problems:
            return self.fail("포트폴리오 형식 오류: " + " / ".join(problems) + " — 팀 업무 지침의 포트폴리오 형식대로 다시 써라.")
        return self.ok(f"포트폴리오 형식 통과 (프로젝트 카드 {len(career.project_cards(answer))}개){figure_note}")


@register
class JdRecommendValidator(Validator):
    name = "jd_recommend"
    description = ("JD 추천 — 모든 JD·요구사항(J1-R1…)을 근거 있음/부분/공백으로 판정했는지, ranking·best 가 코드가 판정으로 계산한 "
                   "점수 순서와 같은지 (점수는 코드가 셈 — core/career.py score_jds)")

    def check(self, answer: str, options: dict):
        from core import career

        jds = list(options.get("career_jds") or [])
        if not jds:
            return self.ok("비교할 JD 없음")
        source = options.get("career_source") if "career_source" in options else None
        problems = career.check_recommend(answer, jds, source)
        if problems:
            return self.fail("JD 추천 오류: " + " / ".join(problems[:10]) + " — 점수는 코드가 판정으로 계산한다. 판정(coverage)은 문서 원문 "
                             "인용으로만 하고, ranking·best 는 코드 계산 순서를 따른다.")
        return self.ok("JD 추천 통과 — 코드 계산 순위: " + career.ranking_text(career.score_jds(answer, jds, source)))


@register
class PortfolioChangesValidator(Validator):
    name = "portfolio_changes"
    description = ("포트폴리오 최신화 변경 목록 — before 는 지금 포트폴리오에 그대로, 문서 변경의 source 는 내 문서 원문 인용, 외부 변경의 "
                   "source 는 승인한 검색 결과 URL, after 에 내 문서에 없는 숫자 없음, 본문 [W번호]는 외부 자료 목록 안")

    def check(self, answer: str, options: dict):
        from core import career

        if "career_old_portfolio" not in options:
            return self.ok("비교할 포트폴리오 정보 없음")
        problems = career.check_changes(answer, str(options.get("career_old_portfolio") or ""), str(options.get("career_source") or ""),
                                        list(options.get("career_web_urls") or []), int(options.get("career_web_count") or 0),
                                        str(options.get("career_reference") or ""))
        if problems:
            return self.fail("변경 목록 오류: " + " / ".join(problems[:10]) + " — 내 경험·숫자는 내 문서에서만, 외부 자료는 용어·트렌드에만 "
                             "쓰고 출처 URL 을 단다.")
        changes = career._items(career.answer_json(answer), "changes")
        outside = sum(str(c.get("kind")).strip() == "외부" for c in changes)
        return self.ok(f"변경 목록 통과 — {len(changes)}건 (문서 {len(changes) - outside} · 외부 {outside})")
