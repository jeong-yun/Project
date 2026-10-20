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
    return f"<thought>1단계 표 / 2단계 검증 로그</thought>\n<final>\n{entry}\n</final>\n## GitHub 요약\n- 요약\n```json\n{json.dumps({'concept_plan': plan}, ensure_ascii=False)}\n```"


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
    bad_entry = GOOD_ENTRY.replace("학습 안정성 향상.", "학습 안정성이 완벽하게 향상된다.\n - 성능이 크게 상승한다.").replace(
        "https://arxiv.org/abs/2503.14476", "URL 없음").replace("유사: 그룹 이점 / ", "")
    problems = concept.check_answer(answer_with(bad_entry, {**PLAN, "category": "Etc"}))
    text = " ".join(problems)
    assert "절대 표현" in text and "개조식" in text and "참고문헌 형식" in text and "유사" in text and "category" in text
    no_thought = concept.check_answer(f"<final>{GOOD_ENTRY}</final>")
    assert any("<thought>" in p for p in no_thought) and any("concept_plan" in p for p in no_thought)


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
    assert "1. 정의 및 개념" in lines and "2. 작동 원리" in lines and "6. 관계(기존 개념과 비교)" in lines
    assert " - PPO 기반 RL 알고리즘" in lines and "  ○ 상·하한 분리" in lines and " - 등장 배경" in lines
    assert lines[-1].startswith(" - [논문·핵심기법]") and "" not in lines
    assert concept.check_entry(norm) == []  # 'Qwen2.5' 의 '2.' 에 속지 않음


def test_request_includes_related_digest_and_target(root):
    c = cat.Catalog.load(root)
    q, brief = concept.build_concept_request("update", "GRPO", "초안", [c.entries[1]], c.entries[0], D)
    assert "- 작업: 수정 (update)" in q and "수정 대상 전문" in q and "보정된 테스트 타임" in q
    assert "(작성일: 26.10.20)" not in q and "26.10.20" in brief.context
