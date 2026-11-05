"""👔 포트폴리오·자소서 팀 — 원본 형식 지키기 · 날짜 표기 · 그림(흐름도)·mermaid · 형식별 출력 · 질문 먼저.

할 일 문서: docs/TODO_커리어팀_형식_피드백.md · docs/TODO_커리어팀_그림_mermaid.md
모든 문서·그림·인물은 코드로 만든 가짜다 (실제 사람·회사·경력 아님 — 이름은 고객사A · 제품X v1.2 같은 가명).
API 키·네트워크·비용 없음 (conftest 가 모델·검색을 막는다).
"""

import json

from core import career
from core.harness import REGISTRY


def check(name: str, answer: str, **options):
    return REGISTRY[name].check(answer, options)


def summary(evidence: list[tuple[str, str]], **extra) -> str:
    data = {"evidence": [{"claim": c, "quote": q} for c, q in evidence], "jd_coverage": [], "needs_check": [], **extra}
    return "\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


DATED_SOURCE = """가상 인물 경력 요약 (가명)
고객사A 탐지 모델 고도화 (2026-06-01 ~ 2026-10-31)
제품X v1.2 로그 수집 개선 (2025.03 ~ 2025.12)
기관B 점검 지원 (2024년 3월 ~ 2024년 11월)
하루 1,200건을 자동 분류했다."""


# ─── 1단계: 날짜 표기만 바꾸기 · 표 행 번호 ────────────────────────────────


def test_same_date_in_other_notation_passes_fact_check():
    opts = {"career_source": DATED_SOURCE, "career_reference": ""}
    ev = summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    good = ("## 경력\n- 고객사A 탐지 모델 고도화 (26.06 ~ 26.10)\n- 제품X 로그 수집 개선 (2025.3 ~ 25.12)\n"
            "- 기관B 점검 지원 (24.03 ~ 24.11)\n- 하루 1,200건을 자동 분류했습니다." + ev)
    assert check("career_facts", good, **opts).passed
    other = good.replace("26.06 ~ 26.10", "26.07 ~ 26.10")  # 원문에 없는 다른 달 → 여전히 실패
    res = check("career_facts", other, **opts)
    assert not res.passed and "26.07" in res.log


def test_short_date_in_source_allows_long_notation():
    source = "가상 프로젝트 C (26.06 ~ 진행 중)\n처리 건수 1,200건"
    allowed = career.numbers_in(source)
    assert {"2026", "2026.06", "2026.6"} <= allowed
    assert career.unsourced_numbers("프로젝트 C (2026.06 ~ 진행 중).", allowed) == []
    assert career.unsourced_numbers("프로젝트 C (2026.08 ~ 진행 중).", allowed)  # 다른 달은 안 됨


def test_rates_and_units_are_not_read_as_dates():
    assert "2026.06" not in career.date_variants("탐지율 26.06% 개선")
    assert "2012.5" not in career.date_variants("처리 속도 12.5배")
    assert "2026.13" not in career.date_variants("26.13")  # 13월은 없다
    allowed = career.numbers_in("정확도 26.06%")
    assert career.unsourced_numbers("2026.06 에 시작했습니다.", allowed)  # 비율에서 날짜를 만들어 내지 않는다


def test_table_row_numbers_are_not_facts_but_other_cells_are():
    allowed = career.numbers_in(DATED_SOURCE)
    table = "| # | 프로젝트 | 건수 |\n|---|---|---|\n| 13 | 고객사A 모델 | 1,200건 |\n| 14 | 제품X 수집 | 3,400건 |"
    out = career.unsourced_numbers(table, allowed)
    assert len(out) == 1 and out[0].startswith("3,400")  # 행 번호 13·14 는 건너뛰고, 다른 칸의 지어낸 숫자는 잡는다


# ─── 3단계: 형식 지정이 있으면 원본 형식 검사 ────────────────────────────────

OLD_MD = """# 기술 경력서 (가명)
## 1. 회사 업무
| 업무 명 | 기간 | 고객사 | 역할 |
|---|---|---|---|
| 탐지 모델 고도화 | 2026-06-01 ~ 현재 | 고객사A | 모델 개발 |
## 2. 개인 프로젝트
- 로그 요약 도구 (2025.03 ~ 2025.12)
"""
NEW_DOC = "최신 요약 (가명)\n탐지 모델 고도화 (2026-06-01 ~ 2026-10-31) — 고객사A, 모델 개발\n하루 1,200건을 자동 분류했다."


def write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_format_request_words():
    assert career.format_requested("Word 는 기술경력서 포맷과 유사하게, md 는 경력서.md 와 유사하게")
    assert career.format_requested("기존 문서처럼 표 형식으로")
    assert career.format_requested("원본 그대로 두고 숫자만 고쳐줘")
    assert not career.format_requested("데이터 분석 경험을 앞에 두고 차분하게")
    assert not career.format_requested("")


def test_update_tool_reads_original_structure_only_when_format_requested(tmp_path):
    from core.tools import career_docs

    inputs = {"career_kind": "portfolio_update", "portfolio": write(tmp_path, "old.md", OLD_MD), "portfolio_name": "old.md",
              "docs": [write(tmp_path, "new.md", NEW_DOC)], "docs_names": ["new.md"]}
    plain = career_docs.run("q", inputs, None)
    assert "career_format" not in plain.harness_options  # 형식 지정 없음 → 표준 형식 검사
    asked = career_docs.run("q", {**inputs, "extra_request": "Word 는 기존 포트폴리오 형식과 유사하게"}, None)
    fmt = asked.harness_options["career_format"]
    assert fmt["mode"] == "original" and fmt["sources"] == ["old.md"]
    assert "1. 회사 업무" in fmt["headings"] and ["업무 명", "기간", "고객사", "역할"] in fmt["table_heads"]
    assert "📐 요청 형식" in asked.content and "| 업무 명 | 기간 | 고객사 | 역할 |" in asked.content
    assert "기존 포트폴리오 형식" in asked.harness_options["career_reference"]


def test_portfolio_format_checks_original_structure_instead_of_standard():
    fmt = {"mode": "original", "headings": ["기술 경력서 (가명)", "1. 회사 업무", "2. 개인 프로젝트"],
           "table_heads": [["업무 명", "기간", "고객사", "역할"]], "sources": ["old.md"]}
    ev = summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    kept = ("# 기술 경력서 (가명)\n## 1. 회사 업무\n| 업무 명 | 기간 | 고객사 | 역할 |\n|---|---|---|---|\n"
            "| 탐지 모델 고도화 | 26.06 ~ 26.10 | 고객사A | 모델 개발 |\n## 2. 개인 프로젝트\n- 로그 요약 도구 (25.03 ~ 25.12)" + ev)
    res = check("portfolio_format", kept, career_format=fmt)
    assert res.passed and "요청 형식 통과" in res.log
    standard = ("# 포트폴리오\n## 요약\n요약\n## 핵심 역량\n| 영역 | 기술 | 근거 |\n|---|---|---|\n| a | b | c |\n## 프로젝트\n"
                "### 1. 탐지 모델\n- **문제**: a\n- **역할**: b\n- **기술**: c\n- **결과**: d\n- **배운 점**: e" + ev)
    assert check("portfolio_format", standard).passed  # 형식 지정이 없으면 지금처럼 표준 형식
    res = check("portfolio_format", standard, career_format=fmt)
    assert not res.passed and "1. 회사 업무" in res.log and "고객사" in res.log
    omitted = kept.replace("- 로그 요약 도구 (25.03 ~ 25.12)", "나머지는 원문 그대로 둡니다.")
    res = check("portfolio_format", omitted, career_format=fmt)
    assert not res.passed and "생략" in res.log
    nothing = {"mode": "original", "headings": [], "table_heads": [], "sources": []}
    assert check("portfolio_format", "# 아무 형식\n본문" + ev, career_format=nothing).passed  # 원본을 못 읽으면 표준 검사도 안 함


def test_docx_headings_and_table_heads_in_document_order():
    import io

    import docx

    from core.doc_text import read_bytes

    d = docx.Document()
    d.add_heading("1. 회사 업무", level=1)
    t = d.add_table(rows=2, cols=3)
    for c, v in enumerate(["업무 명", "기간", "역할"]):
        t.cell(0, c).text = v
    t.cell(1, 0).merge(t.cell(1, 1)).text = "탐지 모델 (병합 칸)"
    t.cell(1, 2).text = "개발"
    d.add_heading("2. 개인 프로젝트", level=1)
    d.add_paragraph("로그 요약 도구")
    buf = io.BytesIO()
    d.save(buf)
    doc = read_bytes("경력서.docx", buf.getvalue())
    assert doc.headings == ["1. 회사 업무", "2. 개인 프로젝트"]
    assert doc.table_heads == [["업무 명", "기간", "역할"]]
    lines = doc.text.splitlines()
    assert lines.index("업무 명 | 기간 | 역할") < lines.index("2. 개인 프로젝트")  # 표가 문서 순서 자리에 (끝으로 밀리지 않음)
    assert "탐지 모델 (병합 칸) | 개발" in doc.text  # 병합 칸 글은 한 번만


# ─── 4단계: '형식 참고 파일' 칸 ────────────────────────────────────────────

REF_MD = """# 경력서 (가명)
## 회사
### 탐지 모델 고도화 (26.06 ~ 진행 중)
- 고객사A, 처리 2,400건
## 개인
"""


class Upload:
    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


def test_format_refs_field_is_optional_multiple_file_on_both_portfolio_tasks():
    from core.config import load_team

    team = load_team("career")
    for task_id in ("portfolio", "portfolio_update"):
        task = next(t for t in team.tasks if t.id == task_id)
        f = next(f for f in task.fields if f.key == "format_refs")
        assert f.type == "file" and f.multiple and not f.required and "md" in f.options and "docx" in f.options
        assert "{format_refs_name}" in task.question


def test_format_refs_are_format_only_not_quote_source(tmp_path):
    from core.config import load_team
    from core.tasks import build_request, store_uploads
    from core.tools import career_docs

    task = next(t for t in load_team("career").tasks if t.id == "portfolio_update")
    values = store_uploads({"portfolio": Upload("old.md", OLD_MD.encode()), "docs": [Upload("new.md", NEW_DOC.encode())],
                            "format_refs": [Upload("경력서.md", REF_MD.encode())]}, tmp_path)
    req = build_request(task, values, "md 는 경력서.md 와 유사하게")
    assert "형식 참고 파일: 경력서.md" in req.question and req.tool_inputs["extra_request"] == "md 는 경력서.md 와 유사하게"
    res = career_docs.run(req.question, req.tool_inputs, None)
    opts = res.harness_options
    assert "📐 형식 참고 파일" in res.content and "처리 2,400건" in res.content
    assert opts["career_format"]["sources"] == ["old.md", "경력서.md"]
    assert "회사" in opts["career_format"]["headings"] and "1. 회사 업무" in opts["career_format"]["headings"]
    assert "2,400" not in opts["career_source"]  # 인용 근거(원문)에는 넣지 않는다
    assert "2400" in career.numbers_in(opts["career_reference"])  # 숫자는 내 문서로 허용
    no_extra = career_docs.run("q", {k: v for k, v in req.tool_inputs.items() if k != "extra_request"}, None)
    assert no_extra.harness_options["career_format"]["mode"] == "original"  # 참고 파일을 올렸으면 그 자체가 형식 지정
