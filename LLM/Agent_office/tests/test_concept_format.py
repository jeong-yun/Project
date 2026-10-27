"""개념 편집 팀 출력 형식 — 기준 항목 「플래시어텐션」(카탈로그의 26.10.16 판, 사용자가 '이렇게 나와야 한다'고 준 글)이
하네스를 그대로 통과하는지, 망가진 형식은 고칠 곳을 알려 주는지, 카탈로그 반영 왕복, 스킬·브리프의 형식 규칙,
프롬프트 판 기록, 📜 적용된 프롬프트 화면(AppTest)."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from core import catalog as cat
from core import concept
from core.config import TEAMS_DIR, load_skills
from core.harness import REGISTRY

TARGET = (Path(__file__).parent / "fixtures" / "concept_flashattention.txt").read_text(encoding="utf-8").strip()
SKILL = TEAMS_DIR / "concept" / "skills" / "it_concept_prompt.md"
D = date(2026, 10, 26)
NAME = "플래시어텐션(FlashAttention)"
PLAN = {"action": "update", "category": "AI", "tag": "Perf", "title": NAME, "target": NAME, "absorb": [],
        "summary": {"emoji": "⚡", "name": "FlashAttention", "keywords": "IO-aware, Tiling", "one_line": "정확 어텐션의 IO 최적화"}}
OTHER = "[Perf]라딕스 어텐션(RadixAttention) (작성일: 26.08.01)\n1. 정의 및 개념\n - 요청 간 KV 재사용."


def answer_with(entry: str, plan: dict = PLAN) -> str:
    return (f"<review_log>1단계 표 / 2단계 검증 로그</review_log>\n<final>\n{entry}\n</final>\n## 변경 요약\n- 🔧수정: 예\n"
            f"## GitHub 요약\n- 요약\n```json\n{json.dumps({'concept_plan': plan}, ensure_ascii=False)}\n```")


def skeleton() -> str:
    """스킬의 [출력 형식 견본] 코드 블록."""
    text = SKILL.read_text(encoding="utf-8")
    return re.search(r"\[출력 형식 견본\].*?```\n(.*?)```", text, flags=re.DOTALL).group(1).strip()


# ─── 기준 항목이 통과 · 모양 보존 ────────────────────────────────────────


def test_target_entry_passes_harness():
    assert concept.check_entry(TARGET) == []
    assert concept.check_answer(answer_with(TARGET)) == []
    assert REGISTRY["concept_entry"].check(answer_with(TARGET), {}).passed


def test_normalize_keeps_target_shape():
    """정규화는 기준 항목의 모양(①②③·ⓐ·'     - 개념:'·●·'8. 참고 문헌')을 바꾸지 않는다 — 태그 뒤 공백만 카탈로그 다수 형식으로."""
    norm = concept.normalize_entry(TARGET)
    original = [l.rstrip() for l in TARGET.splitlines() if l.strip()]
    lines = norm.splitlines()
    assert lines[0] == original[0].replace("[Perf] ", "[Perf]")
    assert lines[1:] == original[1:]
    assert concept.parse_result(answer_with(TARGET)).final_entry == norm


def test_skeleton_in_skill_passes_harness():
    """스킬에 넣은 견본 모양 그대로 쓰면 형식 검사를 통과한다 (가르치는 모양 = 검사하는 모양)."""
    assert concept.check_entry(concept.normalize_entry(skeleton())) == []


def test_normalize_numbers_references_after_last_section():
    base = "[Perf]X(Y) (작성일: 26.10.26)\n1.정의및개념\n- a\n6.관계\n- B: 유사: b / 차이: c\n"
    assert "7. 참고 문헌" in concept.normalize_entry(base + "참고문헌\n[논문·핵심기법] A (https://a)").splitlines()
    with_seven = base + "7.변형·확장상세\n1) Z\n  - 개념: z\n참고 문헌\n- [논문·핵심기법] A (https://a)"
    lines = concept.normalize_entry(with_seven).splitlines()
    assert "7. 변형·확장 상세" in lines and "   1) Z" in lines and "  - 개념: z" in lines and "8. 참고 문헌" in lines
    # 이미 번호가 있으면 그대로, 띄어 쓴 기존 절 제목도 그대로 (수정할 때 기존 형식 유지)
    kept = concept.normalize_entry("[Perf]X(Y) (작성일: 26.10.26)\n3. 좋아진 점(장점/개선점)\n9. 참고 문헌").splitlines()
    assert kept[1:] == ["3. 좋아진 점(장점/개선점)", "9. 참고 문헌"]


# ─── 망가진 형식 → 고칠 곳을 알려 줌 ────────────────────────────────────


def _move_subtitle(text: str) -> str:
    lines = text.splitlines()
    sub = lines.pop(1)
    lines.insert(lines.index("1. 정의 및 개념") + 1, sub)
    return "\n".join(lines)


@pytest.mark.parametrize("broken, expect", [
    (lambda t: t.replace("8. 참고 문헌\n", ""), "참고문헌 절이 없다"),
    (lambda t: t.replace("     - 예시: 장문맥 추론 서빙.\n", ""), "4) 디코딩 특화"),
    (lambda t: t.replace("     - 단점(고유): 최적화된 GEMM", "     - 한계: 최적화된 GEMM"), "1) FlashAttention(FA1)"),
    (_move_subtitle, "부제"),
    (lambda t: t.replace("[도구/프레임워크]", "[블로그]"), "참고문헌 형식"),
    (lambda t: t.replace("유사: ", "공통: "), "유사"),
    (lambda t: t.replace("정확성 보존: 근사가 아니라 표준 어텐션과 동일한 출력.", "정확성을 보존한다.\n - 속도가 빨라진다.\n - 메모리가 준다."), "개조식"),
    (lambda t: t.replace("   ① 문제", "   ① 처음").replace(" - 등장 배경", " - 출현"), "등장 배경"),
])
def test_malformed_entry_fails_with_feedback(broken, expect):
    problems = concept.check_entry(broken(TARGET))
    assert problems and any(expect in p for p in problems), problems
    result = REGISTRY["concept_entry"].check(answer_with(broken(TARGET)), {})
    assert not result.passed and expect in result.log


def test_update_header_must_keep_date_history():
    entry = cat.parse_entries(TARGET, "AI")[0]
    expected = concept.expected_header(entry, D)
    assert expected.endswith("(작성일: 26.09.23)(수정일: 26.10.16, 10.26)")
    options = {"concept_header": expected}
    overwritten = TARGET.replace("(수정일: 26.10.16)", "(수정일: 26.10.26)")  # 지난 실행에서 모델이 이력을 덮어쓴 모양
    problems = concept.check_answer(answer_with(overwritten), options)
    assert any("(수정일: 26.10.16, 10.26)" in p for p in problems), problems
    assert concept.check_answer(answer_with(TARGET.replace("(수정일: 26.10.16)", "(수정일: 26.10.16, 10.26)")), options) == []
    assert concept.check_answer(answer_with(overwritten, {**PLAN, "action": "add", "target": ""}), options) == []  # 추가는 날짜 검사 안 함


# ─── 카탈로그 반영 왕복 ─────────────────────────────────────────────────


@pytest.fixture
def root(tmp_path):
    (tmp_path / cat.CATEGORY_FILES["AI"]).write_text(f"{OTHER}\n\n{TARGET}\n", encoding="utf-8")
    for key in ("Cloud", "Security"):
        (tmp_path / cat.CATEGORY_FILES[key]).write_text("", encoding="utf-8")
    return tmp_path


def test_target_round_trip_through_catalog(root):
    c = cat.Catalog.load(root)
    entry = c.find("FlashAttention")
    assert entry.section(7).startswith("7. 변형·확장 상세") and "FlashAttention-3(FA3)" in entry.section(7)
    assert "8. 참고 문헌" not in entry.section(7)  # 참고 문헌은 7절에 섞이지 않음
    hit = cat.find_existing(c, "FlashAttention-3(FA3)")[0]  # 7절 소절로 편입된 개념 → 부모 항목 수정
    assert hit.kind == "하위" and hit.entry.name == entry.name and hit.detail.startswith("FlashAttention-3")

    q, brief = concept.build_concept_request("update", "FlashAttention-3(FA3)", "초안", [], entry, D, sub_title=hit.detail)
    assert "(작성일: 26.09.23)(수정일: 26.10.16, 10.26)" in q
    assert any("(작성일: 26.09.23)(수정일: 26.10.16, 10.26)" in x for x in brief.constraints)

    new = concept.parse_result(answer_with(TARGET.replace("ⓓ TMA의 역할", "ⓓ TMA(텐서 메모리 가속기)의 역할"))).final_entry
    cat.apply_plan(c, PLAN, new, D)
    again = cat.Catalog.load(root)
    updated = again.find("FlashAttention")
    assert updated.header == "[Perf]플래시어텐션(FlashAttention) (작성일: 26.09.23)(수정일: 26.10.16, 10.26)"
    assert "TMA(텐서 메모리 가속기)의 역할" in updated.body and "\n8. 참고 문헌\n" in updated.body
    assert updated.body.splitlines()[0] == " - 부제: IO 인지 정확 어텐션 — 타일링·온라인 소프트맥스에서 하드웨어 비동기화까지"
    assert [e.name for e in again.entries] == [e.name for e in c.entries]  # 다른 항목은 그대로
    assert cat.find_existing(again, "FlashAttention-3(FA3)")[0].kind == "하위"


# ─── 스킬 · 브리프 · 프롬프트 판 ────────────────────────────────────────


def test_skill_carries_full_original_rules():
    text = SKILL.read_text(encoding="utf-8")
    for must in [
        "[출력 형식 견본]", "실패가 잦았다 → 실패가 잦음", "성능이 크게 상승합니다 → 성능 상승", "degeneration",  # 4단계 예
        "arXiv:2404.19737", "arXiv:2503.20783", "grpo_trainer", "GRPO 손실 한정 분석",  # 5단계 참고문헌 예
        "[판정 선례]", "개입 시점(학습 ↔ 추론)", "AR의 존폐",  # 6.관계 예
        "같이 쓰이니 한 개념 취급", "6.관계에 차이만 나열",  # 나쁜 예 목록
        "Tülu 3", "3단 중첩 금지",  # 1단계 원문
        "<final>", "## 변경 요약", "concept_plan", "(수정일: 26.10.16, 10.26)",  # 앱 출력 계약·수정일
    ]:
        assert must in text, must
    assert "  ○ " not in skeleton()  # 견본은 카탈로그 기준 항목처럼 '○' 대신 ' - '·ⓐ·● 표식


def test_brief_follows_catalog_format():
    q, brief = concept.build_concept_request("add", "X(Y)", "", [], None, D)
    rules = " ".join(brief.constraints) + brief.output_format
    assert "8. 참고 문헌" in rules and "부제" in rules and "출력 형식 견본" in rules and "①②③" in rules
    assert "○" not in rules and "'참고 문헌'," not in rules  # 예전 규칙(○ 하위 불릿·번호 없는 참고 문헌)이 남지 않음


def test_prompt_avoids_words_claude_safeguards_block():
    """'<thought>' 태그·'CoT 강제'·'내부 사고 과정' 요청은 Claude(Opus·Sonnet 5.5) 안전 필터가 요청 자체를 막는다 (2026-10-27 실측).

    검증 로그 태그는 <review_log> — 예전 결과의 <thought> 도 계속 읽는다.
    """
    q, brief = concept.build_concept_request("add", "X(Y)", "", [], None, D)
    sent = SKILL.read_text(encoding="utf-8") + brief.as_prompt(q)
    for banned in ("<thought>", "</thought>", "CoT 강제", "사고 과정"):
        assert banned not in sent, banned
    assert "<review_log>" in sent
    old = answer_with(TARGET).replace("<review_log>", "<thought>").replace("</review_log>", "</thought>")
    assert concept.parse_result(old).thought == "1단계 표 / 2단계 검증 로그" and concept.check_answer(old) == []


def test_request_records_prompt_version():
    version = concept.prompt_version()
    assert re.fullmatch(r"[0-9a-f]{8}", version["hash"]) and version["text"] == load_skills("concept")
    assert version["hash"] in version["label"] and version["updated"][:2] == "20"
    q, _ = concept.build_concept_request("add", "X(Y)", "", [], None, D)
    assert q.splitlines()[-1].startswith(concept.PROMPT_LINE)
    assert concept.used_prompt_version(q) == version["hash"]
    assert concept.used_prompt_version("[입력]\n- 개념명(한/영): 옛 요청") == ""  # 판 기록 이전 결과


# ─── 화면: 📜 적용된 프롬프트 (AppTest) ───────────────────────────────────


@pytest.fixture
def concept_app(tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
    at.query_params["team"] = "concept"
    at.query_params["view"] = "team"
    return at


def test_input_shows_applied_prompt(concept_app, root):
    at = concept_app.run()
    assert not at.exception, at.exception
    at = at.text_input(key="concept_root").set_value(str(root)).run()
    at = at.text_input(key="cw_name_concept").set_value("FlashAttention-3(FA3)").run()
    assert not at.exception, at.exception
    version = concept.prompt_version()["hash"]
    assert any(e.label == f"📜 적용된 프롬프트 — IT 개념 프롬프트 v{version}" for e in at.expander)
    codes = [c.value for c in at.code]
    assert any("[출력 형식 견본]" in c for c in codes)  # 스킬 전문
    assert any("[제약]" in c and "(수정일: 26.10.16" in c and f"v{version}" in c for c in codes)  # 보낼 브리프 (수정 → 날짜 이력)
    assert any("수석 에디터" in c or "편집장" in c for c in codes)  # 에이전트 시스템 프롬프트


def test_result_shows_prompt_version_it_was_made_with(tmp_path):
    from streamlit.testing.v1 import AppTest

    def page(content, question):
        import concept_ui

        concept_ui.result_tabs(content, {"run_id": "r1", "question": question, "brief": None, "harness_options": {}})

    old = "[입력]\n- 개념명(한/영): X\n- 적용 프롬프트: IT 개념 프롬프트 v0123abcd (스킬 파일 수정 2026-10-01 10:00)"
    at = AppTest.from_function(page, kwargs={"content": answer_with(TARGET), "question": old}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any(e.label == "📜 적용된 프롬프트 — IT 개념 프롬프트 v0123abcd" for e in at.expander)
    assert any("v0123abcd" in w.value and "다를 수 있습니다" in w.value for w in at.warning)
    assert not any("형식 검사에서 남은 문제" in w.value for w in at.warning)  # 기준 항목은 형식 문제 없음
