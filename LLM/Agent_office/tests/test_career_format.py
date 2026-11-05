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
