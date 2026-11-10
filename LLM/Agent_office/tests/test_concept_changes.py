"""개념 편집 팀 — 기존 항목 수정(update)은 '원본 기반 + 근거 있는 변경만'.

팀은 항목 전체를 다시 쓰지 않고 변경 제안 목록(적용 계획 JSON 의 changes)만 낸다. 최종본은 코드가 원본에 끼워 넣어 만든다.
  - 변경 목록 밖은 원본과 글자 그대로 같다 (정규화도 하지 않는다)
  - 근거 없는 변경은 기본으로 반영하지 않는다 / 원문을 못 찾은 변경은 적용하지 않고 알린다 / 사람이 뺀 변경은 반영하지 않는다
  - 근거 글은 화면에만 — 카탈로그 파일·커밋 메시지·GitHub 요약·knowledge/ 에 들어가지 않는다
  - 새 항목 추가(add)는 전과 같다
"""

import json
import subprocess
from datetime import date
from itertools import product

import pytest

from core import catalog as cat
from core import concept
from core.harness import REGISTRY
from tests.test_progress_followup import app_env  # noqa: F401 — pytest 준비 함수 (앱 전체 흐름 테스트)

D = date.today()  # 채택(apply_adopted)은 오늘 날짜로 수정일을 붙인다 — 시계에 기대지 않게 기대값도 같은 날짜로 만든다

# 옛 형식이 섞인 원본 — 끝 공백·빈 줄·굵게(**)·들여쓰지 않은 불릿이 있고 2~5번 절·참고 문헌이 없다 (정규화하면 바뀌는 글)
ORIGINAL = "\n".join([
    "[Perf]그룹 상대적 정책 최적화(Group Relative Policy Optimization, GRPO) (작성일: 26.08.19)(수정일: 26.09.11)",
    "1. 정의 및 개념",
    " - 그룹 평균으로 이점을 추정하는 RL 알고리즘.  ",
    " - 원전은 PPO 를 통째로 대체한다고 주장.",
    "",
    "- **옛 형식** 불릿",
    "6. 관계",
    " - RLVR: 유사: RL 정렬 / 차이: 보상 출처 ↔ 이점 추정",
    "7. 변형·확장 상세",
    "   1) Dr.GRPO(GRPO Done Right) — 길이 정규화 제거",
    "     - 개념: 편향 제거 변형",
])
OTHER = "[Perf]보정된 테스트 타임 스케일링(Calibrated Test-Time Scaling, CaTS) (작성일: 26.07.08)\n1. 정의 및 개념  \n - 반복 샘플링 예산 배분."
HEADER = cat.with_updated_date(ORIGINAL.split("\n")[0], D)
OPTIONS = {"concept_header": HEADER, "concept_original": ORIGINAL}

EVIDENCE = "원전 4.1절은 가치 함수만 그룹 평균 기준선으로 바꾼다고 적음 — '통째로 대체'는 과장(판정: 과장)"
SOURCE = "Shao et al. DeepSeekMath (arXiv:2402.03300, https://arxiv.org/abs/2402.03300)"
FORMAT_WHY = "절 제목 표준형 '6. 관계 (기존 개념 비교)' — 스킬 [출력 형식 견본]"
FACT = {"find": " - 원전은 PPO 를 통째로 대체한다고 주장.", "replace": " - 원전은 PPO 의 가치 함수를 그룹 평균 기준선으로 대체.",
        "kind": "근거", "source": SOURCE, "evidence": EVIDENCE}
FORMAT = {"find": "6. 관계", "replace": "6. 관계 (기존 개념 비교)", "kind": "형식", "source": "", "evidence": FORMAT_WHY}
NO_EVIDENCE = {"find": "RL 정렬", "replace": "강화학습 정렬", "kind": "근거", "source": "", "evidence": ""}
NOT_FOUND = {"find": "원본에 없는 문장", "replace": "x", "kind": "근거", "source": SOURCE, "evidence": "근거 글"}
TWICE = {"find": "GRPO", "replace": "지알피오", "kind": "형식", "source": "", "evidence": "표기 통일"}
ALL = [FACT, FORMAT, NO_EVIDENCE, NOT_FOUND, TWICE]

PLAN = {"action": "update", "category": "AI-Model", "tag": "Perf", "title": "GRPO", "target": "GRPO", "absorb": [],
        "summary": {"name": "GRPO", "keywords": "그룹 평균 기준선", "one_line": "그룹 평균으로 이점을 추정하는 RL 알고리즘"}}


def answer(changes: list[dict] | None, plan: dict = PLAN, extra: str = "") -> str:
    plan = dict(plan) if changes is None else {**plan, "changes": changes}
    return (f"<review_log>1단계 표 / 2단계 검증 로그</review_log>\n{extra}## GitHub 요약\n- 🧠 GRPO [Perf]\n"
            f"```json\n{json.dumps({'concept_plan': plan}, ensure_ascii=False)}\n```")


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "Term" / "Every_day"
    folder.mkdir(parents=True)
    (folder / cat.CATEGORY_FILES["AI-Model"]).write_text(f"{ORIGINAL}\n\n{OTHER}\n", encoding="utf-8")
    for key in ("Cloud", "Security", "AI-App"):
        (folder / cat.CATEGORY_FILES[key]).write_text("", encoding="utf-8")
    (folder / cat.OLD_FILE).write_text("", encoding="utf-8")
    return folder


# ─── 요청: 수정은 변경 제안 목록, 추가는 전과 같음 ───────────────────────


def test_update_request_asks_for_changes_on_top_of_the_original(root):
    grpo = cat.Catalog.load(root).find("GRPO")
    q, brief = concept.build_concept_request("update", "GRPO", "초안", [], grpo, D)
    assert concept.original_of(q) == grpo.text == ORIGINAL  # 요청 글에서 원본을 글자 그대로 되찾을 수 있다 (앱을 다시 켠 뒤에도)
    assert "형식 검사 결과" in q and "참고문헌 절이 없다" in q  # 코드가 확인한 원본의 형식 문제 = 형식 수정 후보
    text = brief.as_prompt(q)
    for word in ("changes", "find", "replace", "source", "evidence", "근거에 따른 수정", "형식 수정", "원본"):
        assert word in text, word
    assert "<final>" not in brief.output_format and concept.header_dates(HEADER) in text


def test_add_request_and_result_are_unchanged():
    q, brief = concept.build_concept_request("add", "DAPO", "초안", [], None, D)
    text = brief.as_prompt(q)
    assert "changes" not in text and "<original>" not in text
    assert "<final>" in brief.output_format and "## 변경 요약" in brief.output_format
    # 추가 결과는 수정용 옵션이 남아 있어도 예전처럼 <final> 을 최종본으로 읽는다
    from tests.test_concept import GOOD_ENTRY, PLAN as ADD_PLAN, answer_with

    plain, with_options = concept.parse_result(answer_with(GOOD_ENTRY, ADD_PLAN)), concept.parse_result(answer_with(GOOD_ENTRY, ADD_PLAN), OPTIONS)
    assert plain == with_options and plain.final_entry.startswith("[Perf]DAPO") and plain.changes is None
    assert concept.check_answer(answer_with(GOOD_ENTRY, ADD_PLAN), OPTIONS) == []


# ─── 코드가 조립: 변경 목록 밖은 원본 그대로 ─────────────────────────────


def test_final_entry_is_the_original_with_only_valid_evidenced_changes():
    res = concept.parse_result(answer(ALL), OPTIONS)
    assert [c.usable for c in res.changes] == [True, True, True, False, False]
    assert [c.default_on for c in res.changes] == [True, True, False, False, False]  # 근거 없는 변경은 기본으로 빼 둔다
    before, after = ORIGINAL.split("\n"), res.final_entry.split("\n")
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [0, 3, 6] and len(before) == len(after)
    assert after[0] == HEADER  # 제목 줄은 날짜 이력 + 오늘 수정일 (코드 규칙)
    assert after[3] == FACT["replace"] and after[6] == "6. 관계 (기존 개념 비교)"
    # 정규화하지 않는다 — 끝 공백·빈 줄·굵게·옛 불릿이 그대로
    assert after[2].endswith("알고리즘.  ") and after[4] == "" and after[5] == "- **옛 형식** 불릿"
    assert "RL 정렬" in res.final_entry and "강화학습" not in res.final_entry  # 근거 없음 → 원본 그대로


def test_any_choice_of_changes_leaves_the_rest_identical():
    changes = concept.parse_result(answer(ALL), OPTIONS).changes
    for picked in product([True, False], repeat=len(changes)):
        text = concept.assemble(ORIGINAL, changes, list(picked), HEADER)
        for change, on in zip(changes, picked):
            if change.usable:
                assert (change.replace if on else change.find) in text
            if on and change.usable:
                text = text.replace(change.replace, change.find, 1)  # 반영한 곳만 되돌리면
        assert text.split("\n", 1)[1] == ORIGINAL.split("\n", 1)[1]  # 원본 본문과 한 글자도 다르지 않다
    assert concept.assemble(ORIGINAL, changes, [False] * len(changes), HEADER).split("\n", 1)[1] == ORIGINAL.split("\n", 1)[1]


def test_unusable_changes_say_why():
    c = concept.parse_result(answer(ALL + [dict(FACT), {**FORMAT, "replace": "6. 관계"}]), OPTIONS).changes
    assert c[2].no_evidence and c[0].no_evidence == c[1].no_evidence == ""
    assert "찾지 못" in c[3].problem and "2곳" not in c[3].problem and "곳" in c[4].problem
    assert "같은 곳" in c[5].problem  # 앞 변경과 겹침
    assert "바뀌는 것" in c[6].problem
    draft_only = concept.review_changes(ORIGINAL, [{**FACT, "source": "사용자 초안"}, {**FACT, "source": "arXiv:2607.0xxxx"}])
    assert all(p.no_evidence for p in draft_only)  # 초안·플레이스홀더는 근거가 아니다


def test_change_text_gets_no_notes_or_bold():
    marked = {**FORMAT, "replace": "**6. 관계 (기존 개념 비교)** 🔧*(초안 대비 절 제목 교정)*"}
    res = concept.parse_result(answer([marked]), OPTIONS)
    assert res.changes[0].replace == "6. 관계 (기존 개념 비교)" and "🔧" not in res.final_entry


# ─── 형식 검사(하네스) ────────────────────────────────────────────────


def test_harness_passes_clean_changes_even_when_the_original_is_old_style():
    assert concept.check_entry(ORIGINAL)  # 원본은 지금 형식 검사에 여러 건 걸린다 (2~5번 절·참고 문헌 없음)
    assert concept.check_answer(answer([FACT, FORMAT]), OPTIONS) == []  # 그래도 변경이 새 문제를 만들지 않으면 통과
    assert REGISTRY["concept_entry"].check(answer([FACT, FORMAT]), OPTIONS).passed
    assert concept.check_answer(answer([]), OPTIONS) == []  # 고칠 곳이 없다는 결론도 결과다


def test_harness_reports_each_bad_change():
    text = " / ".join(concept.check_answer(answer(ALL), OPTIONS))
    assert "changes 3번째" in text and "근거" in text and "changes 4번째" in text and "찾지 못" in text and "changes 5번째" in text
    assert "changes 1번째" not in text and "changes 2번째" not in text
    result = REGISTRY["concept_entry"].check(answer(ALL), OPTIONS)
    assert not result.passed and "changes 4번째" in result.log


def test_harness_rejects_rewrites_and_new_format_problems():
    rewrite = answer([FACT], extra=f"<final>\n{ORIGINAL}\n</final>\n")
    assert any("<final>" in p for p in concept.check_answer(rewrite, OPTIONS))
    assert concept.parse_result(rewrite, OPTIONS).final_entry.split("\n")[3] == FACT["replace"]  # 그래도 최종본은 코드가 조립한 것
    assert any("changes" in p for p in concept.check_answer(answer(None), OPTIONS))  # 목록 자체가 없음
    absolute = {**FACT, "replace": " - 원전은 PPO 를 완벽하게 대체한다."}
    text = " / ".join(concept.check_answer(answer([absolute]), OPTIONS))
    assert "절대 표현" in text and "개조식" in text
    drop_section = {"find": "1. 정의 및 개념\n", "replace": "", "kind": "형식", "source": "", "evidence": "절 정리"}
    assert any("1번 절이 없다" in p for p in concept.check_answer(answer([drop_section]), OPTIONS))  # 변경이 새로 만든 형식 문제


def test_malformed_changes_do_not_break():
    res = concept.parse_result(answer([FACT, "그냥 문장", None, {"find": None, "replace": 3}]), OPTIONS)
    assert res.changes[0].usable and len(res.changes) == 2 and not res.changes[1].usable
    assert any("changes" in p for p in res.problems)
    as_dict = concept.parse_result(answer(FACT), OPTIONS)  # 목록이 아니라 객체 하나
    assert as_dict.changes[0].usable and any("changes" in p for p in as_dict.problems)


# ─── 근거는 GitHub 에 올리지 않는다 ───────────────────────────────────


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def test_evidence_stays_out_of_summary_tab():
    res = concept.parse_result(answer([FACT, FORMAT], extra=f"## 변경 요약\n- 🔧수정: {EVIDENCE}\n"), OPTIONS)
    assert res.summary_md.startswith("## GitHub 요약")
    for secret in (EVIDENCE, "arxiv.org/abs/2402.03300", FORMAT_WHY):
        assert secret not in res.summary_md and secret not in res.final_entry


def test_evidence_never_reaches_catalog_files_or_commit(root):
    import concept_ui

    repo = root.parent.parent
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "첫 커밋", cwd=repo)

    content = answer(ALL)
    run = {"run_id": "r1", "question": "", "harness_options": dict(OPTIONS),
           "concept": {"action": "update", "title": "GRPO", "target": cat.parse_header(ORIGINAL.split("\n")[0])[1]}}
    res = concept.parse_result(content, OPTIONS)
    plan, problem = concept_ui.check_adoptable(res.final_entry, json.dumps(res.plan, ensure_ascii=False))
    assert problem == ""
    ctx = concept_ui.CatalogContext(root, repo, "Term/Every_day", cat.Catalog.load(root))
    concept_ui.apply_adopted(ctx, run, res.final_entry, plan)

    diff = cat.git_diff(repo, "Term/Every_day")
    messages = [run["concept_commit"], cat.pending_commit_message(diff)]
    assert messages[0] == "Term: GRPO 수정" and "수정" in messages[1]
    ok, _ = cat.git_commit(repo, "Term/Every_day", messages[0])
    assert ok
    pushed = _git("show", "--format=%B", "HEAD", cwd=repo) + "".join(p.read_text(encoding="utf-8") for p in root.iterdir())
    for secret in (EVIDENCE, "2402.03300", FORMAT_WHY, "판정: 과장"):
        assert secret not in pushed and secret not in diff and not any(secret in m for m in messages)
    # 카탈로그에는 고친 결과만 — 바뀐 줄은 제목(수정일)·사실 수정·형식 수정 세 줄뿐이고 다른 항목은 그대로
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert added == [HEADER, FACT["replace"], "6. 관계 (기존 개념 비교)"]
    text = (root / cat.CATEGORY_FILES["AI-Model"]).read_text(encoding="utf-8")
    assert text == f"{res.final_entry}\n\n{OTHER}\n" and "알고리즘.  \n" in text and "개념  \n" in text


def test_knowledge_export_has_no_evidence(tmp_path):
    from core import feedback_sync
    from core.schemas import AgentAnswer, HumanDecision, ModelSpec
    from core.storage import Storage

    storage = Storage(tmp_path / "runs.db")
    storage.create_run("r1", "concept", "lead", "GRPO 수정")
    step = AgentAnswer(node_id="lead", rank_id="team_lead", kind="review", content=answer(ALL), model=ModelSpec(provider="openai", model="x"))
    storage.save_steps("r1", [step], "awaiting_human")
    storage.save_decision("r1", HumanDecision(action="adopt", adopted_node_id="lead", final_answer=answer(ALL), edited=False, score=5))
    feedback_sync.export(storage, tmp_path / "knowledge")
    everything = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "knowledge").iterdir())
    assert "decisions" in str(list((tmp_path / "knowledge").iterdir())) and EVIDENCE not in everything and "2402.03300" not in everything


# ─── 채택: 대상 고정 · 원본이 바뀌었으면 멈춤 · 채택 기록 ─────────────────


def test_adoption_always_targets_the_original_entry_and_stops_if_it_changed(root):
    import concept_ui

    run = {"run_id": "r1", "question": "", "harness_options": dict(OPTIONS), "concept": {"action": "update", "title": "GRPO", "target": ""}}
    entry = concept.parse_result(answer([FACT]), OPTIONS).final_entry
    ctx = concept_ui.CatalogContext(root, None, "", cat.Catalog.load(root))
    # 팀이 대상을 다른 기존 항목으로 잘못 적어도 원본 항목만 바뀐다
    concept_ui.apply_adopted(ctx, run, entry, {**PLAN, "target": "CaTS"})
    text = (root / cat.CATEGORY_FILES["AI-Model"]).read_text(encoding="utf-8")
    assert OTHER in text and text.startswith(HEADER) and FACT["replace"] in text
    # 같은 결과를 한 번 더 채택하려 하면(원본이 이미 바뀜) 덮어쓰지 않고 알린다
    with pytest.raises(ValueError, match="바뀌었"):
        concept_ui.apply_adopted(ctx, run, entry, dict(PLAN))


def test_adopted_record_keeps_what_the_person_confirmed():
    picked = concept.assemble(ORIGINAL, concept.parse_result(answer(ALL), OPTIONS).changes, [False, True, False, False, False], HEADER)
    record = concept.replace_final(answer(ALL), picked)
    assert record.count("<final>") == 1 and record.index("</review_log>") < record.index("<final>") < record.index("## GitHub 요약")
    shown = concept.parse_result(record, OPTIONS, adopted=True)
    assert shown.final_entry == picked and len(shown.changes) == 5  # 채택한 글 그대로 (정규화 없이), 제안 목록은 참고용으로 남는다


# ─── 화면: 변경마다 근거를 보고 고른다 ───────────────────────────────────


def _page(content, run):
    import streamlit as st

    import concept_ui

    concept_ui.result_tabs(content, run)
    entry, plan = concept_ui.decision_fields(content, "k", run)
    st.session_state["out"] = {"entry": entry, "plan": plan}


def test_screen_lets_the_person_pick_changes_by_evidence():
    from streamlit.testing.v1 import AppTest

    run = {"run_id": "r1", "question": "", "brief": None, "harness_options": dict(OPTIONS)}
    at = AppTest.from_function(_page, kwargs={"content": answer(ALL), "run": run}, default_timeout=30).run()
    assert not at.exception, at.exception
    boxes = at.checkbox
    assert [b.value for b in boxes] == [True, True, False, False, False]
    assert [b.disabled for b in boxes] == [False, False, False, True, True]  # 원문을 못 찾은 변경은 고를 수 없다
    shown = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert EVIDENCE in shown and "2402.03300" in shown and FORMAT_WHY in shown  # 근거는 화면에서 본다
    assert "근거 없음" in " ".join(w.value for w in at.warning) and "찾지 못" in " ".join(e.value for e in at.error)
    assert FACT["find"] in " ".join(c.value for c in at.code) and FACT["replace"] in " ".join(c.value for c in at.code)
    out = at.session_state["out"]
    assert FACT["replace"] in out["entry"] and "6. 관계 (기존 개념 비교)" in out["entry"]
    assert "changes" not in json.loads(out["plan"]) and EVIDENCE not in out["plan"]  # 적용 계획 칸에는 근거를 싣지 않는다

    at = boxes[0].uncheck().run()  # 사실 수정은 빼고, 근거 없는 변경은 사람이 켠다
    at = at.checkbox[2].check().run()
    assert not at.exception, at.exception
    entry = at.session_state["out"]["entry"]
    assert FACT["find"] in entry and FACT["replace"] not in entry and "강화학습 정렬" in entry
    assert entry.split("\n")[2].endswith("알고리즘.  ")


def test_screen_recovers_original_from_question_after_restart(root):
    from streamlit.testing.v1 import AppTest

    grpo = cat.Catalog.load(root).find("GRPO")
    question, _ = concept.build_concept_request("update", "GRPO", "초안", [], grpo, D)
    run = {"run_id": "r1", "question": question, "brief": None, "harness_options": {}, "restored": True}
    at = AppTest.from_function(_page, kwargs={"content": answer([FACT]), "run": run}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(at.checkbox) == 1 and FACT["replace"] in at.session_state["out"]["entry"]


def test_followup_gets_the_whole_original():
    import concept_ui

    block = concept_ui.followup_original({"harness_options": dict(OPTIONS), "question": ""})
    assert ORIGINAL in block and "find" in block
    assert concept_ui.followup_original({"harness_options": {}, "question": "DAPO"}) == ""  # 추가에는 붙이지 않는다


# ─── 앱 전체 흐름 (가짜 모델): 맡기기 → 변경 고르기 → 채택 → 카탈로그 반영 ──────────────


def _open_concept(root):
    from streamlit.testing.v1 import AppTest

    from tests.test_progress_followup import APP

    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["team"] = "concept"
    at.query_params["view"] = "team"
    at.session_state["concept_root"] = str(root)  # never the real catalog
    return at


def test_app_update_flow_applies_only_picked_changes(app_env, use_script, root):
    from core.schemas import ReviewEdits
    from tests.fakes import Script, make_critic, make_delegation, make_review
    from tests.test_progress_followup import wait_for

    gate, draft_gate = app_env
    gate.set()
    draft_gate.set()
    approve = ReviewEdits(analysis="ok", issues=[], score=9, verdict="approve", revised_answer="", comment="ok", edits=[])
    script = use_script(Script(
        text="draft",
        Delegation=lambda m: make_delegation("gm_editor") if "gm_editor" in m[-1].content else make_delegation("am_fact", "dgm_lineage"),
        ReviewResult=make_review(verdict="approve", score=9, answer=answer([FACT, FORMAT])), ReviewEdits=approve,
        CriticResult=make_critic(5)))
    at = _open_concept(root).run()
    assert not at.exception, at.exception
    at = at.selectbox(key="dept_task_research").set_value("concept:write").run()
    at = at.text_input(key="cw_name_concept").set_value("GRPO").run()
    assert any("원본 항목은 그대로" in i.value for i in at.info)
    at = at.button(key="cw_go_concept").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if len(a.checkbox) >= 2 and any(b.label == "✅ 이 안으로 채택" for b in a.button) else None)(
        _open_concept(root).run()), 90)
    assert not at.exception, at.exception
    asked = "\n".join(script.prompts("ReviewResult"))
    assert f"<original>\n{ORIGINAL}\n</original>" in asked and "changes" in asked  # 팀은 원본 전문과 변경 제안 형식을 받는다
    picks = [b for b in at.checkbox if "반영" in b.label]
    assert [b.value for b in picks] == [True, True]
    at = picks[1].uncheck().run()  # 형식 수정은 빼고 채택
    entry = next(t for t in at.text_area if (t.key or "").startswith("ce_")).value
    assert FACT["replace"] in entry and "6. 관계 (기존 개념 비교)" not in entry
    assert at.text_input(key="concept_root").value == str(root)
    at = next(b for b in at.button if b.label == "✅ 이 안으로 채택").click().run()
    assert not at.exception, at.exception
    text = (root / cat.CATEGORY_FILES["AI-Model"]).read_text(encoding="utf-8")
    expected = ORIGINAL.replace(FACT["find"], FACT["replace"]).replace(ORIGINAL.split("\n")[0], HEADER)
    assert text == f"{expected}\n\n{OTHER}\n"  # 고른 변경 하나 + 제목 줄 수정일만 바뀜
    assert EVIDENCE not in text and "2402.03300" not in text
    at = _open_concept(root).run()  # 채택 뒤 화면: 채택한 글이 그대로 보이고 체크 상자는 없다
    assert not at.exception, at.exception
    assert any("안 채택" in s.value for s in at.success) and not [b for b in at.checkbox if "반영" in b.label]
    assert any(t.value == expected for t in at.text)
    assert not any("형식 검사에서 남은 문제" in w.value for w in at.warning)  # 채택 기록의 <final> 을 형식 오류로 따지지 않는다
    assert any("직접 수정" in s.value for s in at.success)  # 변경을 하나 뺐으므로 '직접 수정'으로 기록된다
