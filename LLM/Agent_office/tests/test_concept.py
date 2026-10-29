"""개념 카탈로그 테스트 — 항목 읽기, 관련 항목 찾기, 추가/수정/삭제 반영(old_term 이관·요약 갱신), 결과 형식 검사."""

import json
from datetime import date

import pytest

from core import catalog as cat
from core import concept
from core.harness import REGISTRY

D = date(2026, 10, 20)

AI_TEXT = """[Perf]그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) (작성일: 26.08.19)(수정일: 26.09.11)
1. 정의 및 개념
 - 그룹 평균으로 이점을 추정하는 RL 알고리즘.
6. 관계
 - RLVR: 유사: RL 정렬 / 차이: 보상 출처 ↔ 이점 추정
7. 변형·확장 상세
   1) Dr.GRPO(GRPO Done Right) — 길이 정규화 제거
     - 개념: 편향 제거 변형

[Perf]보정된 테스트 타임 스케일링(Calibrated Test-Time Scaling, CaTS) (작성일: 26.07.08)
1. 정의 및 개념
 - GRPO 와 무관한 반복 샘플링 예산 배분.

자가 치유 보안 인프라(Self-Healing Security) (수정일: 26.02.10)
1. 개념: 작성일 없이 수정일만 있는 옛 항목."""

SUMMARY = """# 요약

### 1. Cloud & Infrastructure(클라우드 및 인프라)
* **CNAPP:** 통합 플랫폼.
### 2. Artificial Intelligence(AI 지능 및 시스템)
* **[Perf]Calibrated Test-Time Scaling:** 신뢰도 기반 예산.
### 3. Security & Trust(보안 및 신뢰성)

## 🛠️ 핵심 개념 한 줄 요약
| 주제 | 핵심 키워드 | 한 줄 정의 |
| :--- | :--- | :--- |
| **Calibrated Test-Time Scaling** | 보정 | 신뢰도 기반 예산 |
"""

GOOD_ENTRY = """[Perf]DAPO(Decoupled Clip and Dynamic sAmpling Policy Optimization) (작성일: 26.10.20)
1. 정의 및 개념
 - GRPO 계열의 RL 알고리즘임.
 - 등장 배경: 엔트로피 붕괴 문제 → 클리핑 비대칭 원인 진단 → 분리 클리핑 처방.
2. 작동 원리
 - 클립 상·하한 분리.
3. 좋아진 점
 - 학습 안정성 향상.
4. 문제점
 - 하이퍼파라미터 증가.
5. 사용 예시
 - 수학 추론 RL 학습.
6. 관계
 · **GRPO**: 유사: 그룹 이점 / 차이: 클리핑 축 / 경계: 편입 검토 대상
참고문헌
 - [논문·핵심기법] Yu et al. DAPO (arXiv:2503.14476, https://arxiv.org/abs/2503.14476): 분리 클리핑 원전"""


def answer_with(entry: str, plan: dict) -> str:
    return f"<review_log>1단계 표 / 2단계 검증 로그</review_log>\n<final>\n{entry}\n</final>\n## GitHub 요약\n- 요약\n```json\n{json.dumps({'concept_plan': plan}, ensure_ascii=False)}\n```"


PLAN = {"action": "add", "category": "AI", "tag": "Perf", "title": "DAPO", "target": "", "absorb": [],
        "summary": {"name": "DAPO", "keywords": "분리 클리핑", "one_line": "GRPO 의 클리핑을 분리한 RL 알고리즘"},
        "basic_terms": [{"term": "엔트로피 붕괴(Entropy Collapse)", "definition": "정책이 한 답만 내게 되는 현상"}]}


@pytest.fixture
def root(tmp_path):
    (tmp_path / cat.CATEGORY_FILES["AI"]).write_text(AI_TEXT, encoding="utf-8")
    (tmp_path / cat.CATEGORY_FILES["Cloud"]).write_text("", encoding="utf-8")
    (tmp_path / cat.CATEGORY_FILES["Security"]).write_text("", encoding="utf-8")
    (tmp_path / cat.OLD_FILE).write_text("[AI-Sec]옛 항목 (작성일: 26.01.01)\n1. 개념", encoding="utf-8")
    (tmp_path / cat.BASIC_FILE).write_text("Firebase: 플랫폼\n", encoding="utf-8")
    (tmp_path / cat.SUMMARY_FILE).write_text(SUMMARY, encoding="utf-8")
    return tmp_path


def test_parse_entries_including_old_style_headers(root):
    c = cat.Catalog.load(root)
    assert [e.name.split("(")[0] for e in c.entries] == ["그룹 상대적 정책 최적화", "보정된 테스트 타임 스케일링", "자가 치유 보안 인프라"]
    grpo = c.entries[0]
    assert grpo.tag == "Perf" and "보상 출처" in grpo.section(6)
    assert c.find("GRPO") is grpo  # 약어로도 찾는다


def test_find_related_prefers_acronym_and_name(root):
    c = cat.Catalog.load(root)
    related = cat.find_related(c, "DAPO — GRPO 변형", "GRPO 의 클리핑을 분리")
    assert related[0][0].name.startswith("그룹 상대적 정책")


def test_updated_date_is_appended():
    assert cat.with_updated_date("X (작성일: 26.08.19)(수정일: 26.09.11)", D) == "X (작성일: 26.08.19)(수정일: 26.09.11, 10.20)"
    assert cat.with_updated_date("X (작성일: 26.08.19)", D) == "X (작성일: 26.08.19)(수정일: 26.10.20)"


def test_add_with_absorb_moves_old_entry_and_updates_summary(root):
    plan = {**PLAN, "absorb": ["Calibrated Test-Time Scaling"]}
    changes = cat.apply_plan(cat.Catalog.load(root), plan, GOOD_ENTRY, D)
    ai = (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert ai.startswith("[Perf]DAPO")  # 새 항목은 맨 위
    assert "CaTS" not in ai  # 흡수된 항목은 빠지고
    old = (root / cat.OLD_FILE).read_text(encoding="utf-8")
    assert old.startswith("[Perf]보정된 테스트 타임 스케일링") and "옛 항목" in old  # old_term 맨 위로 이관
    summary = (root / cat.SUMMARY_FILE).read_text(encoding="utf-8")
    assert "* **[Perf]DAPO:** GRPO 의 클리핑을 분리한 RL 알고리즘" in summary
    assert "| **DAPO** | 분리 클리핑 |" in summary
    assert "Calibrated Test-Time Scaling" not in summary  # 흡수된 항목 요약 삭제
    basic = (root / cat.BASIC_FILE).read_text(encoding="utf-8")
    assert "엔트로피 붕괴(Entropy Collapse): 정책이" in basic
    assert {c.action for c in changes} >= {"추가", "삭제", "이관", "요약", "기초 용어"}


def test_update_replaces_in_place_and_category_move(root):
    new = GOOD_ENTRY.replace("DAPO(Decoupled Clip and Dynamic sAmpling Policy Optimization) (작성일: 26.10.20)",
                             "그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) (작성일: 26.08.19)(수정일: 26.09.11)")
    cat.apply_plan(cat.Catalog.load(root), {**PLAN, "action": "update", "target": "GRPO"}, new, D)
    ai = (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert ai.startswith("[Perf]그룹 상대적 정책") and "(수정일: 26.09.11, 10.20)" in ai  # 제자리 + 수정일 추가
    assert ai.count("그룹 상대적 정책 최적화") == 1  # 제자리 교체 (중복 없음)
    # 분류 이동: 자가 치유 → Security
    healing = "자가 치유 보안 인프라(Self-Healing Security) (수정일: 26.02.10)\n1. 개념: 이동"
    cat.apply_plan(cat.Catalog.load(root), {**PLAN, "action": "update", "category": "Security", "target": "자가 치유 보안 인프라"}, healing, D)
    assert "자가 치유" not in (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert (root / cat.CATEGORY_FILES["Security"]).read_text(encoding="utf-8").startswith("자가 치유 보안 인프라")


def test_delete_moves_to_old_term(root):
    c = cat.Catalog.load(root)
    cat.apply_plan(c, concept.delete_plan(c.find("CaTS")), "", D)
    assert "CaTS" not in (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert (root / cat.OLD_FILE).read_text(encoding="utf-8").startswith("[Perf]보정된")
    with pytest.raises(ValueError):
        cat.apply_plan(c, {**PLAN, "action": "update", "target": "없는 개념"}, GOOD_ENTRY, D)


def test_apply_edits_only_when_text_matches_once(root):
    logs = concept.apply_edits(cat.Catalog.load(root), [
        {"entry": "GRPO", "find": "보상 출처 ↔ 이점 추정", "replace": "보상 출처 ↔ 이점 추정 (DAPO 는 §7)", "reason": "상호참조"},
        {"entry": "GRPO", "find": "없는 문장", "replace": "x"},
    ])
    assert logs[0].startswith("✅") and logs[1].startswith("⚠️")
    assert "(DAPO 는 §7)" in (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")


# ─── 결과 형식 검사 (하네스 concept_entry) ─────────────────────────────


def test_good_answer_passes_and_parses():
    answer = answer_with(GOOD_ENTRY, PLAN)
    assert concept.check_answer(answer) == []
    res = concept.parse_result(answer)
    assert res.final_entry.startswith("[Perf]DAPO") and res.plan["category"] == "AI" and "GitHub 요약" in res.summary_md
    assert REGISTRY["concept_entry"].check(answer, {}).passed


def test_bad_answer_reports_each_rule():
    bad_entry = GOOD_ENTRY.replace("학습 안정성 향상.", "학습 안정성이 완벽하게 향상된다.\n - 성능이 크게 상승한다.\n - 메모리도 줄어든다.").replace(
        "https://arxiv.org/abs/2503.14476", "URL 없음").replace("유사: 그룹 이점 / ", "")
    problems = concept.check_answer(answer_with(bad_entry, {**PLAN, "category": "Etc"}))
    text = " ".join(problems)
    assert "절대 표현" in text and "개조식" in text and "참고문헌 형식" in text and "유사" in text and "category" in text
    no_thought = concept.check_answer(f"<final>{GOOD_ENTRY}</final>")
    assert any("<review_log>" in p for p in no_thought) and any("concept_plan" in p for p in no_thought)


# 모델이 absorb·edits 를 엉뚱한 모양으로 쓴 경우 — 화면이 멈추던 일 (2026-10-27): 고쳐 읽고, 하네스는 다시 쓰라고 알린다
BAD_SHAPES = [
    ({"absorb": "GRPO"}, ["GRPO"]),
    ({"absorb": [{"name": "GRPO", "reason": "편입"}, {"title": "CaTS"}]}, ["GRPO", "CaTS"]),
    ({"absorb": [None, " GRPO ", 3, "GRPO", {"reason": "이름 없음"}]}, ["GRPO", "3"]),
    ({"absorb": {"항목": "GRPO"}}, ["GRPO"]),
]


@pytest.mark.parametrize("bad,absorb", BAD_SHAPES)
def test_malformed_absorb_is_normalized_and_flagged(bad, absorb):
    answer = answer_with(GOOD_ENTRY, {**PLAN, **bad})
    res = concept.parse_result(answer)
    assert res.plan["absorb"] == absorb
    assert any("absorb" in p for p in concept.check_answer(answer))
    result = REGISTRY["concept_entry"].check(answer, {})
    assert not result.passed and "absorb" in result.log


def test_malformed_edits_and_other_fields_do_not_break(root):
    plan = {**PLAN, "absorb": None, "tag": ["Perf"], "target": {"name": "GRPO"},
            "summary": {**PLAN["summary"], "keywords": ["분리", "클리핑"]},
            "basic_terms": ["용어A: 정의A", 7],
            "edits": [{"entry": {"name": "GRPO"}, "find": None, "reason": 1}, "그냥 문장", None]}
    res = concept.parse_result(answer_with(GOOD_ENTRY, plan))
    p = res.plan
    assert p["absorb"] == [] and p["tag"] == "Perf" and p["target"] == "GRPO" and p["summary"]["keywords"] == "분리, 클리핑"
    assert p["basic_terms"] == [{"term": "용어A", "definition": "정의A"}]
    assert p["edits"] == [{"entry": "GRPO", "find": "", "replace": "", "reason": "1"}]
    text = " ".join(res.problems)
    assert "edits" in text and "basic_terms" in text and "tag" in text and "target" in text
    assert concept.apply_edits(cat.Catalog.load(root), p["edits"])[0].startswith("⚠️")  # 반영해도 멈추지 않고 수동 수정 안내
    edits_dict = concept.parse_result(answer_with(GOOD_ENTRY, {**PLAN, "edits": {"entry": "GRPO", "find": "a", "replace": "b"}}))
    assert edits_dict.plan["edits"][0]["entry"] == "GRPO" and any("edits" in x for x in edits_dict.problems)
    assert concept.normalize_plan(["not", "a", "dict"])[0] == {}
    assert concept.check_answer(answer_with(GOOD_ENTRY, PLAN)) == []  # 바른 계획은 그대로 통과


def test_malformed_absorb_still_adopts_into_catalog(root):
    import concept_ui

    plan, err = concept_ui.check_adoptable(GOOD_ENTRY, json.dumps({**PLAN, "absorb": [{"name": "Calibrated Test-Time Scaling"}, None]}))
    assert err == "" and plan["absorb"] == ["Calibrated Test-Time Scaling"]
    changes = cat.apply_plan(cat.Catalog.load(root), plan, GOOD_ENTRY, D)
    assert any("(통합)" in c.detail for c in changes)
    assert concept_ui.check_adoptable(GOOD_ENTRY, "[1, 2]")[0] is None


def test_result_screen_opens_with_malformed_plan():
    """이미 저장된 결과(absorb 가 객체 목록)도 결과 화면이 열리고 경고만 보인다."""
    from streamlit.testing.v1 import AppTest

    def page(content):
        import concept_ui

        concept_ui.result_tabs(content, {"run_id": "r1", "question": "", "brief": None, "harness_options": {}})

    plan = {**PLAN, "action": "update", "target": "GRPO", "absorb": [{"name": "CaTS", "reason": "편입"}, None],
            "edits": ["문장"]}
    at = AppTest.from_function(page, kwargs={"content": answer_with(GOOD_ENTRY, plan)}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("통합으로 빠질 항목: CaTS" in c.value for c in at.caption)
    assert any("absorb" in w.value for w in at.warning)


def test_normalize_entry_to_catalog_style():
    # 실제 실행에서 모델이 쓴 형식 (분류 표시·붙여 쓴 절 제목·마크다운 불릿·참고문헌 불릿 없음)
    raw = """[2.AI] [Perf] DAPO(Decoupled Clip) (작성일: 26.10.20)

1.정의및개념
- **PPO 기반** RL 알고리즘
등장 배경
- 문제: Qwen2.5 실험에서 엔트로피 급락
2.작동원리
- Clip-Higher
  - 상·하한 분리
3.좋아진점
- 안정성
4.문제점
- 튜닝 부담
5.사용예시
- 수학 RL
6.관계(기존개념비교)
- GRPO: 유사: 그룹 / 차이: 클리핑
참고문헌
[논문·핵심기법] Yu et al. DAPO (arXiv:2503.14476, https://arxiv.org/abs/2503.14476): 원전"""
    norm = concept.normalize_entry(raw)
    lines = norm.splitlines()
    assert lines[0] == "[Perf]DAPO(Decoupled Clip) (작성일: 26.10.20)"
    assert "1. 정의 및 개념" in lines and "2. 작동 원리" in lines and "6. 관계 (기존 개념 비교)" in lines
    # 들여쓴 하위 불릿은 '○' 로 바꾸지 않고 그대로, 참고 문헌은 마지막 절 다음 번호 (카탈로그 형식)
    assert " - PPO 기반 RL 알고리즘" in lines and "  - 상·하한 분리" in lines and " - 등장 배경" in lines
    assert "7. 참고 문헌" in lines
    assert lines[-1].startswith(" - [논문·핵심기법]") and "" not in lines
    assert concept.check_entry(norm) == []  # 'Qwen2.5' 의 '2.' 에 속지 않음


def test_request_includes_related_digest_and_target(root):
    c = cat.Catalog.load(root)
    q, brief = concept.build_concept_request("update", "GRPO", "초안", [c.entries[1]], c.entries[0], D)
    assert "수정 (update)" in q and "수정 대상 전문" in q and "보정된 테스트 타임" in q
    assert "(작성일: 26.10.20)" not in q and "26.10.20" in brief.context


# ─── 추가 / 수정 자동 판정 ─────────────────────────────────────────────


def test_find_existing_decides_update_or_add(root):
    c = cat.Catalog.load(root)
    # 독립 항목 — 한글명·영문명·약어 어느 것으로 써도 같은 항목
    for q in ["GRPO", "그룹 상대적 정책 최적화", "Group Relative Policy Optimization", "CaTS"]:
        hits = cat.find_existing(c, q)
        assert len(hits) == 1 and hits[0].kind == "항목", q
    # 다른 항목의 7.변형 소절로 편입된 개념 → 부모 항목을 수정
    sub = cat.find_existing(c, "Dr.GRPO(GRPO Done Right)")
    assert sub[0].kind == "하위" and sub[0].entry.name.startswith("그룹 상대적") and "Dr.GRPO" in sub[0].detail
    # 없는 개념 → 추가
    assert cat.find_existing(c, "DAPO(Decoupled Clip)") == []
    # 이름 일부만 겹치는 것은 같은 개념이 아님 ('RAG' ≠ '에이전틱 RAG')
    (root / cat.CATEGORY_FILES["AI"]).write_text(AI_TEXT + "\n\n에이전틱 RAG(Agentic RAG) (작성일: 26.03.16)\n1. 개념", encoding="utf-8")
    assert cat.find_existing(cat.Catalog.load(root), "RAG(Retrieval-Augmented Generation)") == []


def test_update_keeps_original_created_date(root):
    # 모델이 수정본 제목에 오늘 날짜를 작성일로 써도, 기존 작성일·수정일 이력은 보존하고 수정일만 추가
    new = GOOD_ENTRY.replace("DAPO(Decoupled Clip and Dynamic sAmpling Policy Optimization) (작성일: 26.10.20)",
                             "그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) (작성일: 26.10.20)")
    cat.apply_plan(cat.Catalog.load(root), {**PLAN, "action": "update", "target": "GRPO"}, new, D)
    header = (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8").splitlines()[0]
    assert header == "[Perf]그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) (작성일: 26.08.19)(수정일: 26.09.11, 10.20)"


def test_request_explains_auto_decision(root):
    c = cat.Catalog.load(root)
    grpo = c.find("GRPO")
    q_sub, _ = concept.build_concept_request("update", "Dr.GRPO", "초안", [], grpo, D, sub_title="Dr.GRPO(GRPO Done Right)")
    assert "자동 판정" in q_sub and "7.변형·확장 상세 소절" in q_sub and "수정 대상 전문" in q_sub
    q_add, _ = concept.build_concept_request("add", "DAPO", "초안", [], None, D)
    assert "추가 (add)" in q_add and "수정 대상" not in q_add


# ─── 메인 화면 채택 → 카탈로그 반영 (concept_ui) ─────────────────────────


def test_main_app_adoption_helpers(root):
    import concept_ui

    plan, err = concept_ui.check_adoptable(GOOD_ENTRY, "{not json")
    assert plan is None and "JSON" in err
    plan, err = concept_ui.check_adoptable("", json.dumps(PLAN))
    assert plan is None and "최종본" in err  # 하위 초안(<final> 없음)을 고르면 반영 불가
    # 팀이 수정으로 판정했지만 대상 이름을 엉뚱하게 쓴 경우 → 화면에서 판정한 기존 항목으로 반영
    ctx = concept_ui.CatalogContext(root, None, "", cat.Catalog.load(root))
    run = {"concept": {"action": "update", "title": "GRPO", "target": "그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO)"}}
    new = GOOD_ENTRY.replace("DAPO(Decoupled Clip and Dynamic sAmpling Policy Optimization)",
                             "그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO)")
    concept_ui.apply_adopted(ctx, run, new, {**PLAN, "action": "update", "target": "Dr.GRPO"})
    assert [c.action for c in run["concept_changes"]][0] == "교체" and run["concept_commit"].endswith("수정")


def test_update_replaces_summary_line_in_place():
    # 실제 사례: 요약 'Muon(MomentUm Orthogonalized…)' ↔ 항목 '뮤온 옵티마이저(Muon: MomentUm …)' — 수정 시 옛 요약이 남아 2번 나옴
    text = """### 2. Artificial Intelligence(AI 지능 및 시스템)
* **[Perf]FlashAttention:** a
* **[Perf]Muon(MomentUm Orthogonalized by Newton-Schulz):** 옛 정의
* **[Perf]KAN:** b
### 3. Security & Trust(보안 및 신뢰성)

## 🛠️ 핵심 개념 한 줄 요약
| 주제 | 핵심 키워드 | 한 줄 정의 |
| :--- | :--- | :--- |
| **Muon(MomentUm Orthogonalized by Newton-Schulz)** | 옛 | 옛 정의 |
| **KAN** | k | b |"""
    plan = {"action": "update", "category": "AI", "tag": "Perf", "title": "뮤온",
            "target": "뮤온 옵티마이저(Muon: MomentUm Orthogonalized by Newton-Schulz)",
            "summary": {"name": "Muon Optimizer(Muon)", "keywords": "새", "one_line": "새 정의"}}
    out = cat.update_summary(text, plan, []).splitlines()
    assert sum("Muon" in l for l in out) == 2  # 불릿 1 + 표 1 (중복 없음)
    assert out[2] == "* **[Perf]Muon Optimizer(Muon):** 새 정의"  # 옛 자리 그대로
    assert "| **Muon Optimizer(Muon)** | 새 | 새 정의 |" in out and "옛 정의" not in "\n".join(out)
    assert "* **[Perf]KAN:** b" in out and "| **KAN** | k | b |" in out  # 다른 항목은 그대로


def test_change_notes_are_kept_out_of_final_entry():
    # 실제 사례(Muon): 최종본 안에 '🔧수정: … / 이유: …' 메모가 섞여 카탈로그에 그대로 들어감
    entry = "\n".join([
        "[Perf]뮤온 옵티마이저(Muon) (작성일: 26.09.04)",
        "1. 정의 및 개념",
        " - Newton-Schulz 기반 근사 직교화.",
        '  ○ 🔧수정: "SVD로 직교화"류 표현 삭제 / 이유: 원형은 직접 SVD가 아님',
        " - ➕추가: Gram Newton-Schulz drop-in 가속 / 이유: 후속 구현 반영",
        " - 정확도 향상. 🔧 단, 효과는 규모 의존적.",
        "   3) LoRA-GA — 초기화 축 🔧*(기존 독립 항목에서 이관, 내용 보존)*",
        ' - 별도이며 보완적. 🔧 초안의 "완벽히 보완적"은 절대 표현 → 완화.',
        " - 전망 이론 기반. 🔧*(카너먼=2002 노벨 경제학상)* 쌍 없이 정렬.",
        "  끝 공백이 있는 줄도 그대로   ",
    ])
    clean, notes = concept.strip_change_notes(entry)
    lines = clean.splitlines()
    assert "🔧" not in clean and "➕" not in clean and "이유:" not in clean
    assert " - 정확도 향상. 단, 효과는 규모 의존적." in lines  # 내용 앞 표시만 지움
    assert "   3) LoRA-GA — 초기화 축" in lines and " - 별도이며 보완적." in lines  # 메모 구절은 뺌
    assert " - 전망 이론 기반. (카너먼=2002 노벨 경제학상) 쌍 없이 정렬." in lines  # 사실 괄호는 남김
    assert "  끝 공백이 있는 줄도 그대로   " in lines  # 표시 없는 줄은 한 글자도 안 바꿈
    assert len(notes) == 4 and notes[0].startswith("🔧수정:")
    # 결과 나누기에서도 자동으로 떼어 내고, 화면용 변경 요약으로 보관
    res = concept.parse_result(answer_with(GOOD_ENTRY.replace("학습 안정성 향상.", "학습 안정성 향상.\n - 🔧수정: '완벽' 삭제 / 이유: 절대 표현"), PLAN))
    assert "🔧" not in res.final_entry and res.change_notes == ["🔧수정: '완벽' 삭제 / 이유: 절대 표현"]


def test_replace_final_keeps_rest_of_answer():
    answer = answer_with(GOOD_ENTRY, PLAN)
    replaced = concept.replace_final(answer, "[Perf]새 제목 (작성일: 26.10.20)\n1. 정의 및 개념")
    assert "<final>\n[Perf]새 제목" in replaced and "<review_log>" in replaced and "concept_plan" in replaced


# ─── GitHub 정보 (사이드바 링크·올리지 않은 커밋) ──────────────────────────


@pytest.mark.parametrize("remote", [
    "https://github.com/jeong-yun/CS.git",
    "https://github.com/jeong-yun/CS",
    "git@github.com:jeong-yun/CS.git",
    "ssh://git@github.com/jeong-yun/CS.git",
    "https://token@github.com/jeong-yun/CS.git",
])
def test_web_url_from_remote(remote):
    assert cat.web_url(remote) == "https://github.com/jeong-yun/CS"


def test_repo_info_links_and_unpushed(tmp_path):
    import subprocess

    def run(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    origin, work = tmp_path / "origin.git", tmp_path / "work"
    run("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    run("clone", str(origin), str(work), cwd=tmp_path)
    run("config", "user.email", "t@example.com", cwd=work)
    run("config", "user.name", "t", cwd=work)
    (work / "Term").mkdir()
    (work / "Term" / "a.txt").write_text("a", encoding="utf-8")
    run("add", ".", cwd=work)
    run("commit", "-m", "첫 커밋", cwd=work)
    run("push", "-u", "origin", "main", cwd=work)
    (work / "Term" / "a.txt").write_text("b", encoding="utf-8")
    run("commit", "-am", "둘째", cwd=work)

    info = cat.repo_info(work, "Term")
    assert info.unpushed == 1 and info.branch == "main" and "둘째" in info.last_commit
    assert info.url is None  # 로컬 원격은 GitHub 주소가 아님
    info.url = "https://github.com/jeong-yun/CS"
    assert info.tree("Term") == "https://github.com/jeong-yun/CS/tree/main/Term"
    assert info.commit(cat.head_sha(work)).startswith("https://github.com/jeong-yun/CS/commit/")


# ─── 푸시 전 자동 rebase (GitHub 웹에서 고친 커밋이 있을 때) ──────────────────


def _git(*args, cwd):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def clones(tmp_path):
    """(GitHub 역할 bare 저장소, 이 PC 의 clone, 'GitHub 웹 편집' 역할 clone) — 카탈로그는 Term/Every_day."""
    origin, work, web = tmp_path / "origin.git", tmp_path / "work", tmp_path / "web"
    _git("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    for path in (work, web):
        _git("clone", str(origin), str(path), cwd=tmp_path)
        _git("config", "user.email", "t@example.com", cwd=path)
        _git("config", "user.name", "t", cwd=path)
    root = work / "Term" / "Every_day"
    root.mkdir(parents=True)
    (root / cat.CATEGORY_FILES["AI"]).write_text(AI_TEXT + "\n", encoding="utf-8")
    (root / cat.SUMMARY_FILE).write_text(SUMMARY, encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "첫 커밋", cwd=work)
    _git("push", "-u", "origin", "main", cwd=work)
    _git("pull", "origin", "main", cwd=web)
    _git("branch", "--set-upstream-to=origin/main", "main", cwd=web)
    return origin, work, web


def _edit(repo, old, new, message):
    path = repo / "Term" / "Every_day" / cat.CATEGORY_FILES["AI"]
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    _git("commit", "-am", message, cwd=repo)


def _log(repo, ref="HEAD"):
    import subprocess

    return subprocess.run(["git", "log", "--format=%s", ref], cwd=repo, capture_output=True, text=True,
                          encoding="utf-8").stdout.split("\n")[:-1]


def test_push_auto_rebases_when_github_is_ahead(clones):
    origin, work, web = clones
    root = work / "Term" / "Every_day"
    # GitHub 웹에서 파일 끝 쪽을 고침 → 이 PC 는 파일 맨 위에 새 항목 (오늘 실제 사례와 같은 모양)
    _edit(web, "1. 개념: 작성일 없이 수정일만 있는 옛 항목.", "1. 개념: 웹에서 다듬은 옛 항목.", "웹 편집 1")
    _edit(web, " - GRPO 와 무관한 반복 샘플링 예산 배분.", " - 웹에서 다듬은 예산 배분.", "웹 편집 2")
    _git("push", cwd=web)
    _edit(work, AI_TEXT.splitlines()[0], GOOD_ENTRY + "\n\n" + AI_TEXT.splitlines()[0], "Term: DAPO 추가")
    assert not cat.git_push(work)[0]  # 그냥 푸시하면 non-fast-forward 로 거절

    sync, ok, log = cat.git_sync_push(work, root)
    assert sync.ok and ok and sync.integrated == 2 and "2개" in sync.message and "항목 4개" in sync.message
    assert _log(origin, "main") == ["Term: DAPO 추가", "웹 편집 2", "웹 편집 1", "첫 커밋"]  # 웹 커밋 위에 내 커밋
    text = (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert "DAPO" in text and "웹에서 다듬은 옛 항목" in text and "웹에서 다듬은 예산 배분" in text
    assert cat.ahead_behind(work) == (0, 0) and not cat.git_status(work, "Term")


def test_push_conflict_aborts_and_does_not_push(clones):
    origin, work, web = clones
    root = work / "Term" / "Every_day"
    line = " - 그룹 평균으로 이점을 추정하는 RL 알고리즘."
    _edit(web, line, " - 웹: 그룹 평균 기반 RL.", "웹 편집")
    _git("push", cwd=web)
    _edit(work, line, " - PC: 그룹 평균 기반 정책 최적화.", "Term: GRPO 수정")
    before = _log(work)

    sync, ok, log = cat.git_sync_push(work, root)
    assert not sync.ok and not ok and log == ""
    assert sync.conflicts == ["Term/Every_day/" + cat.CATEGORY_FILES["AI"]]
    assert "충돌" in sync.message and "푸시하지 않았" in sync.message and "git pull --rebase" in sync.message
    assert _log(origin, "main") == ["웹 편집", "첫 커밋"]  # GitHub 에는 아무것도 올라가지 않음
    assert _log(work) == before  # 내 커밋 그대로, rebase 중 상태 아님
    assert not (work / ".git" / "rebase-merge").exists() and not (work / ".git" / "rebase-apply").exists()
    assert not cat.git_status(work, ".")  # 작업 폴더 깨끗 (충돌 표시 없음)
    assert "PC: 그룹 평균" in (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")


def test_behind_count_and_pull(clones):
    origin, work, web = clones
    root = work / "Term" / "Every_day"
    _edit(web, "1. 개념: 작성일 없이 수정일만 있는 옛 항목.", "1. 개념: 웹 편집.", "웹 편집")
    _git("push", cwd=web)
    assert cat.repo_info(work, "Term").behind == 0  # 아직 fetch 전 — 모름
    assert cat.git_fetch(work, quiet=True)[0]
    info = cat.repo_info(work, "Term")
    assert info.behind == 1 and info.unpushed == 0
    # 커밋 안 된 변경(겹치지 않는 파일)이 있어도 내 커밋이 없으면 앞으로 당기기만 해서 받는다
    (root / cat.SUMMARY_FILE).write_text(SUMMARY + "\n메모", encoding="utf-8")
    sync = cat.git_sync(work, root)
    assert sync.ok and sync.integrated == 1 and "웹 편집" in (root / cat.CATEGORY_FILES["AI"]).read_text(encoding="utf-8")
    assert cat.repo_info(work, "Term").behind == 0 and cat.git_sync(work, root).integrated == 0


def test_dirty_tree_blocks_rebase_but_keeps_changes(clones):
    origin, work, web = clones
    root = work / "Term" / "Every_day"
    _edit(web, "1. 개념: 작성일 없이 수정일만 있는 옛 항목.", "1. 개념: 웹 편집.", "웹 편집")
    _git("push", cwd=web)
    _edit(work, AI_TEXT.splitlines()[0], GOOD_ENTRY + "\n\n" + AI_TEXT.splitlines()[0], "Term: DAPO 추가")
    (root / cat.SUMMARY_FILE).write_text(SUMMARY + "\n메모", encoding="utf-8")
    sync, ok, _ = cat.git_sync_push(work, root)
    assert not sync.ok and not ok and "커밋하지 않은 변경" in sync.message
    assert _log(origin, "main") == ["웹 편집", "첫 커밋"] and "메모" in (root / cat.SUMMARY_FILE).read_text(encoding="utf-8")


def test_sidebar_fetch_is_cached(monkeypatch, tmp_path):
    import concept_ui

    calls = []
    monkeypatch.setattr(cat, "git_fetch", lambda repo, quiet=False: calls.append(quiet) or (True, ""))
    concept_ui._fetch_cache.clear()
    for _ in range(3):
        concept_ui.fetch_remote(tmp_path)
    concept_ui.refresh_git()  # 커밋·푸시 버튼으로는 다시 받지 않음
    concept_ui.fetch_remote(tmp_path)
    assert calls == [True]  # TTL 안에서는 한 번만, 터미널 질문 끔
