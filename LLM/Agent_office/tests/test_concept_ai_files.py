"""AI 두 파일(앱 적용 가능 AI-App / 불가 AI-Model) — 읽기·쓰기, 옛 category "AI" 처리, Agent Office 적용 현황표, 화면.

가짜 카탈로그만 쓴다 (실제 CS 폴더·모델 호출 없음).
"""

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

import concept_ui
from core import catalog as cat
from core import concept
from core.config import TEAMS_DIR

D = date(2026, 11, 9)
ROOT = Path(__file__).resolve().parent.parent
STATUS = TEAMS_DIR / "concept" / "ai_app_status.yaml"

APP_TEXT = """[Perf]사고 트리(Tree of Thoughts, ToT) (작성일: 26.01.01)
1. 정의 및 개념
 - 여러 갈래를 탐색.

표에 없는 앱 개념(Unlisted Thing, ULT) (작성일: 26.11.09)
1. 정의 및 개념
 - 정의."""

MODEL_TEXT = """[Perf]다중 헤드 잠재 어텐션 (Multi-Head Latent Attention, MLA) (작성일: 26.02.01)(수정일: 26.03.01)
1. 정의 및 개념
 - KV 압축."""

SUMMARY = f"""# 요약

{cat.CATEGORY_TITLES["Cloud"]}
* **CNAPP:** 통합.
{cat.CATEGORY_TITLES["AI-App"]}
* **[Perf]Tree of Thoughts(ToT):** 갈래 탐색.
{cat.CATEGORY_TITLES["AI-Model"]}
* **[Perf]Multi-Head Latent Attention(MLA):** KV 압축.
{cat.CATEGORY_TITLES["Security"]}
"""

NEW_ENTRY = """[Perf]자기 일관성(Self-Consistency, SC) (작성일: 26.11.09)
1. 정의 및 개념
 - 여러 번 답하게 해 다수결.
 - 등장 배경: 한 번의 추론 경로 오류 → 경로 하나에 의존 → 여러 경로 다수결.
2. 작동 원리
 - 여러 샘플.
3. 좋아진 점
 - 정확도 상승.
4. 문제점
 - 호출 수 증가.
5. 사용 예시
 - 수학 추론.
6. 관계
 - CoT: 유사: 추론 경로 / 차이: 경로 수 / 경계: 결합
7. 참고 문헌
 - [논문·핵심기법] Wang et al. Self-Consistency (arXiv:2203.11171, https://arxiv.org/abs/2203.11171): 원전"""


def plan(**kw) -> dict:
    return {"action": "add", "category": "AI-App", "tag": "Perf", "title": "SC", "target": "", "absorb": [],
            "summary": {"name": "Self-Consistency(SC)", "keywords": "다수결", "one_line": "여러 번 답해 다수결"}, **kw}


@pytest.fixture
def root(tmp_path):
    (tmp_path / cat.CATEGORY_FILES["AI-App"]).write_text(APP_TEXT, encoding="utf-8")
    (tmp_path / cat.CATEGORY_FILES["AI-Model"]).write_text(MODEL_TEXT, encoding="utf-8")
    (tmp_path / cat.SUMMARY_FILE).write_text(SUMMARY, encoding="utf-8")
    return tmp_path


def text_of(root: Path, category: str) -> str:
    return (root / cat.file_of(category)).read_text(encoding="utf-8")


# ─── 읽기 · 쓰기 ────────────────────────────────────────────────────────


def test_two_ai_files_are_two_categories(root):
    c = cat.Catalog.load(root)
    assert [e.name for e in c.by_category("AI-App")] == ["사고 트리(Tree of Thoughts, ToT)", "표에 없는 앱 개념(Unlisted Thing, ULT)"]
    assert [e.name for e in c.by_category("AI-Model")] == ["다중 헤드 잠재 어텐션 (Multi-Head Latent Attention, MLA)"]
    assert not c.legacy


def test_add_goes_to_chosen_ai_file_and_summary_section(root):
    cat.apply_plan(cat.Catalog.load(root), plan(), NEW_ENTRY, D)
    assert text_of(root, "AI-App").startswith("[Perf]자기 일관성") and "자기 일관성" not in text_of(root, "AI-Model")
    lines = (root / cat.SUMMARY_FILE).read_text(encoding="utf-8").splitlines()
    app_at, model_at = lines.index(cat.CATEGORY_TITLES["AI-App"]), lines.index(cat.CATEGORY_TITLES["AI-Model"])
    sc = next(i for i, l in enumerate(lines) if "Self-Consistency" in l)
    assert app_at < sc < model_at  # 2-1 절 안에


def test_move_between_ai_files(root):
    """수정하면서 category 를 바꾸면 다른 AI 파일로 옮긴다 (작성일은 그대로, 수정일 추가)."""
    new = MODEL_TEXT.replace("KV 압축.", "KV 압축 (앱 적용으로 재분류 시험).")
    cat.apply_plan(cat.Catalog.load(root), plan(action="update", target="MLA", title="MLA"), new, D)
    assert "MLA" not in text_of(root, "AI-Model") and "재분류 시험" in text_of(root, "AI-App")
    assert "(작성일: 26.02.01)(수정일: 26.03.01, 11.09)" in text_of(root, "AI-App")


def test_old_result_category_ai_update_stays_in_its_file(root):
    """나누기 전 결과(category "AI")로 수정을 채택해도 대상 항목이 있는 파일에서 고친다."""
    new = MODEL_TEXT.replace("KV 압축.", "KV 압축 고침.")
    changes = cat.apply_plan(cat.Catalog.load(root), plan(action="update", category="AI", target="MLA"), new, D)
    assert "KV 압축 고침" in text_of(root, "AI-Model") and "MLA" not in text_of(root, "AI-App")
    assert changes[0].action == "교체"


def test_old_result_category_ai_add_asks_which_file(root):
    with pytest.raises(ValueError, match="AI-App"):
        cat.apply_plan(cat.Catalog.load(root), plan(category="AI"), NEW_ENTRY, D)
    p, err = concept_ui.check_adoptable(NEW_ENTRY, json.dumps(plan(category="AI")))
    assert p is None and "AI-App" in err and "AI-Model" in err
    assert concept_ui.check_adoptable(NEW_ENTRY, json.dumps(plan(action="update", category="AI", target="MLA")))[1] == ""


def test_legacy_catalog_still_reads_and_writes(tmp_path):
    """AI 를 나누기 전 카탈로그(옛 AI 파일 하나)도 항목이 사라지지 않고 고칠 수 있다."""
    (tmp_path / cat.LEGACY_FILES["AI"]).write_text(f"{APP_TEXT}\n\n{MODEL_TEXT}", encoding="utf-8")
    c = cat.Catalog.load(tmp_path)
    assert c.legacy and len(c.by_category("AI")) == 3
    new = MODEL_TEXT.replace("KV 압축.", "KV 압축 고침.")
    cat.apply_plan(c, plan(action="update", category="AI", target="MLA"), new, D)
    assert "KV 압축 고침" in (tmp_path / cat.LEGACY_FILES["AI"]).read_text(encoding="utf-8")
    cat.apply_plan(cat.Catalog.load(tmp_path), plan(category="AI"), NEW_ENTRY, D)  # 나누기 전이면 옛 파일 맨 위
    assert (tmp_path / cat.LEGACY_FILES["AI"]).read_text(encoding="utf-8").startswith("[Perf]자기 일관성")


# ─── 형식 검사 · 요청 ───────────────────────────────────────────────────


def answer(p: dict) -> str:
    return (f"<review_log>1단계 표 / 2단계 로그</review_log>\n<final>\n{NEW_ENTRY}\n</final>\n## GitHub 요약\n- 요약\n"
            f"```json\n{json.dumps({'concept_plan': p}, ensure_ascii=False)}\n```")


def test_harness_asks_to_pick_ai_file():
    assert concept.check_answer(answer(plan())) == []
    assert concept.check_answer(answer(plan(category="AI-Model"))) == []
    problems = concept.check_answer(answer(plan(category="AI")))
    assert any("AI-App" in p and "AI-Model" in p for p in problems)


def test_request_explains_ai_file_choice():
    _, brief = concept.build_concept_request("add", "자기 일관성(Self-Consistency)", "", [], None)
    assert "AI-App" in brief.context and "AI-Model" in brief.context and "코드만으로" in brief.context
    assert concept.PLAN_EXAMPLE["concept_plan"]["category"] in cat.CATEGORY_FILES
    skill = (TEAMS_DIR / "concept" / "skills" / "it_concept_prompt.md").read_text(encoding="utf-8")
    assert "AI-App" in skill and "AI-Model" in skill


# ─── Agent Office 적용 현황표 · 화면 ────────────────────────────────────


def test_status_table_is_checked_against_code():
    rows = yaml.safe_load(STATUS.read_text(encoding="utf-8"))["entries"]
    assert len(rows) == 27
    for r in rows:
        assert r["status"] in concept_ui.STATUS_ICON and r["note"].strip(), r["name"]
        for f in r["files"]:  # 적힌 관련 파일이 실제로 있어야 한다
            assert (ROOT / f).exists(), f"{r['name']}: 없는 파일 {f}"


def test_status_names_do_not_overlap():
    """두 행이 같은 이름 열쇠를 가지면 항목이 엉뚱한 행에 붙는다 (예: RAG ↔ 에이전틱 RAG)."""
    rows = concept.status_rows()
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            assert not (a["_keys"] & b["_keys"]), (a["name"], b["name"])


def test_app_status_lookup(root):
    c = cat.Catalog.load(root)
    assert concept.app_status(c.find("ToT"))["status"] in concept_ui.STATUS_ICON
    assert concept.app_status(c.find("Unlisted Thing")) is None
    rows = concept_ui.app_rows(c.by_category("AI-App"))
    assert rows[0]["Agent Office"] == concept_ui.UNCHECKED  # 확인 안 한 항목이 맨 위 (그다음 없음·일부·적용)


def test_ai_panel_screen(root):
    """🧩 AI 앱 적용 보기 — 탭 2개, 가능 쪽에 Agent Office 현황 열. (한글은 args 로 — Windows AppTest 빈 화면 방지)"""
    from streamlit.testing.v1 import AppTest

    def page(folder):
        from pathlib import Path

        import concept_ui
        from core import catalog as cat

        c = cat.Catalog.load(Path(folder))
        concept_ui.ai_panel(concept_ui.CatalogContext(Path(folder), None, "", c))

    at = AppTest.from_function(page, args=(str(root),), default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 2 and not at.warning
    assert "Agent Office" in at.dataframe[0].value.columns and len(at.dataframe[0].value) == 2
    assert list(at.dataframe[1].value["항목"]) == ["다중 헤드 잠재 어텐션 (Multi-Head Latent Attention, MLA)"]
