"""교훈이 프롬프트에 빠짐없이 들어가는지 · 한도를 넘으면 알리는지 · 정리 제안은 승인 전에 안 바꾸는지 · 프롬프트 보기 = 실제 프롬프트.

모든 모델 호출은 가짜 모델(tests/fakes.py Script)로 — 실제 키·비용 없음.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from streamlit.testing.v1 import AppTest

from core.feedback import (
    active_lessons, apply_cleanup, fit_lessons, lesson_budget, propose_cleanup, team_lessons, team_prompts,
)
from core.schemas import CleanupProposal, HumanDecision, Lesson, LessonGroup, LessonGroupSet, LessonSet
from core.storage import Storage
from core.workflow import build_office_graph, resume_with_decision, stream_office
from tests.fakes import Script, make_review
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "runs.db")


def snapshot(storage, team_id):
    return [(r["id"], r["lesson"], r["active"]) for r in storage.list_lessons(team_id)]


def add(storage, team_id, agent, *texts, kind="improve"):
    return storage.add_lessons(team_id, None, [Lesson(agent_id=agent, kind=kind, lesson=t) for t in texts])


# ─── 글자 한도: 켜 둔 교훈은 모두 · 넘치면 빠진 것을 알려 줌 ─────────────────────


def test_budget_setting_replaces_lesson_count():
    assert lesson_budget({"feedback": {"lesson_chars_in_prompt": 1234}}) == 1234
    assert lesson_budget({}) == 20000  # 설정이 없을 때 기본값


def test_all_active_lessons_within_budget_reach_the_prompt(org, ranks, team, models, storage, use_script):
    """예전에는 최근 5개만 들어갔다 — 이제 한도 안이면 12개 모두, 오래된 것 → 최신 순서로 들어간다."""
    texts = [f"교훈{i:02d} 출처를 단다" for i in range(12)]
    add(storage, team.id, "staff_case", *texts)
    storage.update_lesson(storage.list_lessons(team.id, "staff_case")[0]["id"], active=False)  # 끈 교훈은 빠진다
    script = use_script(Script(text="초안", ReviewResult=make_review(answer="보완안")))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC, storage=storage)
    list(stream_office(graph, org, "질문", "staff_case", brief=BRIEF))

    system = script.calls[0][1][0].content
    order = [system.index(t) for t in texts[:11]]
    assert order == sorted(order)  # 11개 모두, 오래된 것 → 최신
    assert texts[11] not in system  # 꺼 둔 교훈


def test_over_budget_leaves_oldest_out_but_reports_it():
    rows = [{"id": i, "lesson": "가" * size} for i, size in ((5, 40), (4, 70), (3, 30), (2, 50))]  # 최신 먼저
    pick = fit_lessons(rows, 100)
    assert [r["id"] for r in pick.kept] == [3, 5]  # 긴 4번만 건너뛰고 짧은 3번은 넣는다 (오래된 것 → 최신)
    assert [r["id"] for r in pick.left_out] == [4, 2]  # 빠진 것은 조용히 버리지 않고 돌려준다
    assert pick.chars == 70


def test_over_budget_warning_on_screen(tmp_path):
    """한도를 넘으면 부서 화면 맨 위 경고 + 팀원 프롬프트 보기에 '빠진 교훈' 경고가 뜬다."""

    # 아래 page 소스에는 한글을 두지 않는다 — AppTest 가 임시 파일로 옮길 때 Windows 기본 인코딩(cp949)으로 깨져 빈 화면이 된다
    def page(db, where):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models(), budget=50))
        team = load_team("concept")
        prompt_ui.budget_warning([team], where)
        prompt_ui.member_prompts(team, key="t")

    db = tmp_path / "runs.db"
    add(Storage(db), "concept", "lead", "가" * 30, "나" * 30)  # 60자 > 한도 50자
    at = AppTest.from_function(page, args=(db, "아래"), default_timeout=30).run()
    assert not at.exception, at.exception
    warnings = [w.value for w in at.warning]
    assert any("한도(50자)" in w and "빠진 교훈 1개" in w for w in warnings)  # 부서 화면 경고
    assert any("1개가 글자 한도" in w for w in warnings)  # 팀장 프롬프트 보기 경고 (첫 팀원 = 팀장)
    assert "나" * 30 in at.code[0].value and "가" * 30 not in at.code[0].value  # 최신만 들어감


# ─── 프롬프트 보기 = 실행 때 실제 프롬프트 ─────────────────────────────────────


def test_viewer_prompt_equals_runtime_prompt(org, ranks, team, models, storage, use_script):
    add(storage, team.id, "staff_case", "사례를 두 개 이상 든다", "숫자에는 기준일을 단다")
    script = use_script(Script(text="초안", ReviewResult=make_review(answer="보완안")))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC, storage=storage)
    list(stream_office(graph, org, "질문", "staff_case", brief=BRIEF))
    runtime = script.calls[0][1][0].content  # 사원 초안 호출의 시스템 프롬프트

    shown = {p.node_id: p for p in team_prompts(team, ranks, models, storage, lesson_budget(SETTINGS))}
    assert shown["staff_case"].prompt == runtime
    assert [r["lesson"] for r in shown["staff_case"].lessons.kept] == ["사례를 두 개 이상 든다", "숫자에는 기준일을 단다"]


def test_concept_viewer_uses_the_shared_runtime_prompt(tmp_path):
    """개념 편집 팀 '📜 적용된 프롬프트'도 같은 보기 — 스킬 머리말('## 스킬:')과 교훈까지 실행과 같다."""

    def page(db):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        prompt_ui.member_prompts(load_team("concept"), key="c")

    db = tmp_path / "runs.db"
    add(Storage(db), "concept", "lead", "포맷 규칙을 지킨다")
    at = AppTest.from_function(page, args=(db,), default_timeout=30).run()
    assert not at.exception, at.exception
    from core.config import load_models, load_ranks, load_team

    expected = team_prompts(load_team("concept"), load_ranks(), load_models(), Storage(db), 20000)[0].prompt
    assert at.code[0].value == expected and "## 스킬:" in expected and "포맷 규칙을 지킨다" in expected


# ─── 정리 제안: 승인 전에는 아무것도 바꾸지 않는다 ────────────────────────────


def test_cleanup_proposal_checks_ids_and_changes_nothing(team, models, storage, use_script):
    old = add(storage, team.id, "staff_case", "출처를 단다", "근거 링크를 붙인다", "표로 정리한다")
    other = add(storage, team.id, "am_action", "대리 교훈")
    new = add(storage, team.id, "staff_case", "모든 주장에 출처 링크를 붙인다")[0]
    before = snapshot(storage, team.id)
    use_script(Script(LessonGroupSet=LessonGroupSet(groups=[
        LessonGroup(agent_id="staff_case", ids=[old[0], old[1], new], relation="overlap", merged="모든 주장에 출처 링크를 단다",
                    detail="링크는 원문 주소", reason="같은 지시"),
        LessonGroup(agent_id="staff_case", ids=[old[2], other[0]], relation="conflict", merged="x", detail="", reason="다른 사람 교훈"),
        LessonGroup(agent_id="staff_case", ids=[old[2], 999], relation="overlap", merged="x", detail="", reason="없는 번호"),
    ])))
    proposals, notes = propose_cleanup(team, active_lessons(storage, team.id), {}, models, {"staff_case": new})

    assert len(proposals) == 1  # 다른 에이전트 교훈·없는 번호가 낀 묶음은 코드가 버린다
    p = proposals[0]
    assert p.ids == sorted([old[0], old[1], new]) and p.new_id == new and p.by == "model"
    assert p.merged == "모든 주장에 출처 링크를 단다 (사람 코멘트: 링크는 원문 주소)"  # 구체 내용은 공개 파일에서 떼는 자리로
    assert snapshot(storage, team.id) == before  # 제안만 — 저장소 그대로

    assert apply_cleanup(storage, p, "keep") and snapshot(storage, team.id) == before
    apply_cleanup(storage, p, "off_old")
    on = {r["id"] for r in storage.list_lessons(team.id) if r["active"]}
    assert old[0] not in on and old[1] not in on and new in on and old[2] in on  # 옛 것만 끔, 지우지 않음
    assert len(storage.list_lessons(team.id)) == len(before)


def test_cleanup_merge_adds_one_and_turns_originals_off(team, storage):
    from core.schemas import CleanupProposal

    ids = add(storage, team.id, "staff_case", "출처를 단다", "링크를 단다")
    p = CleanupProposal(team_id=team.id, agent_id="staff_case", ids=ids, relation="overlap", kind="improve",
                        merged="출처 링크를 단다", reason="겹침", by="model")
    apply_cleanup(storage, p, "merge", "출처 링크를 단다 (사용자가 고침)")
    rows = storage.list_lessons(team.id, "staff_case")
    assert [(r["lesson"], r["active"]) for r in rows] == [("출처 링크를 단다 (사용자가 고침)", 1), ("링크를 단다", 0), ("출처를 단다", 0)]


def test_cleanup_falls_back_to_text_compare_when_model_fails(team, models, storage, use_script):
    use_script(Script())  # 정리 모델 출력 없음 → 호출 실패
    a = add(storage, team.id, "staff_case", "결과가 상위 검토에서 그대로 승인되었다. 이 방식을 유지할 것.", kind="praise")
    add(storage, team.id, "staff_case", "표로 정리한다")
    b = add(storage, team.id, "staff_case", "결과가 상위 검토에서 그대로 승인되었다. 이 방식을 유지할 것.", kind="praise")
    proposals, notes = propose_cleanup(team, active_lessons(storage, team.id), {}, models)
    assert [(p.ids, p.by) for p in proposals] == [(sorted(a + b), "code")]
    assert any("문장이 거의 같은" in n for n in notes)


def _decide(org, ranks, team, models, storage, use_script, groups, lesson="모든 주장에 출처를 단다", comment=""):
    """사원 1명 업무 → 채택 결정. groups(msgs) 가 교훈 정리 모델의 답. 결정 뒤 나온 이벤트를 돌려준다."""
    use_script(Script(
        text="초안", ReviewResult=make_review(answer="보완안"),
        LessonSet=LessonSet(lessons=[Lesson(agent_id="staff_case", kind="improve", lesson=lesson)]),
        LessonGroupSet=lambda msgs: LessonGroupSet(groups=groups(max(r["id"] for r in storage.list_lessons(team.id)))),
    ))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC, storage=storage, checkpointer=MemorySaver())
    list(stream_office(graph, org, "질문", "staff_case", thread_id="r1", brief=BRIEF))
    return dict(resume_with_decision(graph, org, HumanDecision(action="adopt", adopted_node_id="staff_case", final_answer="초안",
                                                               score=4, comment=comment), "r1"))


def test_decision_merges_overlap_automatically_and_emphasizes(org, ranks, team, models, storage, use_script):
    """결정 → 새 교훈이 옛 교훈과 겹치면 바로 1개로 합친다: 반복 횟수 합 · 프롬프트 맨 앞 강조 · 원래 교훈은 끔(지우지 않음)."""
    old = add(storage, team.id, "staff_case", "출처를 단다 (사람 코멘트: 링크는 원문 주소로)")[0]
    other = add(storage, team.id, "staff_case", "표로 정리한다")[0]
    events = _decide(org, ranks, team, models, storage, use_script, lambda new: [LessonGroup(
        agent_id="staff_case", ids=[old, new], relation="overlap", merged="모든 주장에 출처 링크를 단다", detail="", reason="같은 지시")])

    assert "cleanup" not in events  # 사용자에게 고르라고 하지 않는다
    [m] = events["merged"]
    rows = {r["id"]: r for r in storage.list_lessons(team.id)}
    assert len(rows) == 4 and not rows[old]["active"] and not rows[m.new_id]["active"]  # 원래 교훈은 꺼 둠 (지우지 않음)
    merged = rows[m.merged_id]
    assert merged["active"] and merged["hits"] == 2
    assert merged["lesson"] == "모든 주장에 출처 링크를 단다 (사람 코멘트: 링크는 원문 주소로)"  # 사람 코멘트 원문은 코드가 그대로
    pick = team_lessons(storage, team.id, 20000)["staff_case"]
    assert pick.texts[0].startswith("[중요 — 피드백 2회 반복] 모든 주장에")  # 반복된 지시가 맨 앞
    assert pick.texts[1] == "표로 정리한다" and rows[other]["active"]

    from core.feedback import undo_merge
    undo_merge(storage, m)  # 되돌리기 — 원래대로
    on = {r["id"] for r in storage.list_lessons(team.id) if r["active"]}
    assert on == {old, m.new_id, other}


def test_decision_conflict_waits_for_user(org, ranks, team, models, storage, use_script):
    """서로 반대 지시는 자동으로 고치지 않는다 — 충돌로 알리기만 하고, 사용자가 고른 하나만 남긴다."""
    old = add(storage, team.id, "staff_case", "결론을 맨 앞에 쓴다")[0]
    events = _decide(org, ranks, team, models, storage, use_script, lambda new: [LessonGroup(
        agent_id="staff_case", ids=[old, new], relation="conflict", merged="결론 위치를 질문에 맞춘다", detail="",
        reason="결론 위치가 반대")], lesson="결론은 맨 끝에 쓴다", comment="결론은 끝에")
    assert "merged" not in events
    [c] = events["cleanup"]
    assert c.relation == "conflict" and old in c.ids
    assert all(r["active"] for r in storage.list_lessons(team.id))  # 아무것도 끄지 않음
    apply_cleanup(storage, c, "only", keep=old)
    assert [r["id"] for r in storage.list_lessons(team.id) if r["active"]] == [old]


def test_rules_always_fit_and_details_follow_relevance():
    """한도를 넘으면 규칙 문장은 모두 넣고, 세부 코멘트는 질문과 관련 높은 것부터 — 못 넣은 것은 cut 으로 알린다."""
    rows = [  # 최신 먼저
        {"id": 3, "lesson": "형식을 지킨다 (사람 코멘트: " + "표 형식 " * 20 + ")", "created_at": "2026-11-03"},
        {"id": 2, "lesson": "원전을 단다 (사람 코멘트: FlashAttention 원전 논문 " + "가" * 100 + ")", "created_at": "2026-11-02"},
        {"id": 1, "lesson": "짧게 쓴다", "created_at": "2026-11-01"},
    ]
    pick = fit_lessons(rows, 200, query="FlashAttention 정리")
    assert [r["id"] for r in pick.kept] == [1, 2, 3] and not pick.left_out  # 규칙 문장은 모두
    assert [r["id"] for r in pick.cut] == [3]  # 질문과 관련 낮은 최신 교훈의 세부 코멘트가 빠짐
    assert "FlashAttention 원전" in pick.texts[1] and pick.texts[2] == "형식을 지킨다"
    assert pick.chars <= 200
    assert [r["id"] for r in fit_lessons(rows, 200).cut] == [2]  # 질문이 없으면(화면) 최신 세부 코멘트부터


def test_old_db_gets_hits_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # 반복 횟수 칸이 없던 예전 DB
        conn.execute("CREATE TABLE lessons (id INTEGER PRIMARY KEY AUTOINCREMENT, team_id TEXT NOT NULL, agent_id TEXT NOT NULL,"
                     " run_id TEXT, kind TEXT NOT NULL, lesson TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,"
                     " created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO lessons (team_id, agent_id, kind, lesson, created_at) VALUES ('concept','lead','improve','옛 교훈','2026-10-01')")
    assert Storage(db).list_lessons("concept")[0]["hits"] == 1  # 교훈은 그대로, 1회로


def test_cleanup_panel_waits_for_click(tmp_path):
    """화면: 제안을 보여 주기만 하고, '합친 문장으로 바꾸기'를 눌러야 바뀐다."""
    from core.schemas import CleanupProposal

    db = tmp_path / "runs.db"
    ids = add(Storage(db), "concept", "lead", "형식 A 를 지킨다", "형식 A 와 B 를 지킨다")
    proposal = CleanupProposal(team_id="concept", agent_id="lead", ids=ids, relation="overlap", kind="improve",
                               merged="형식 A·B 를 지킨다", reason="겹침", by="model")

    def page(db, proposal):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        prompt_ui.render_cleanup(load_team("concept"), [proposal], key="r")

    at = AppTest.from_function(page, args=(db, proposal), default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(at.button) == 2  # 새 교훈이 없는 정리 → [합치기 / 둘 다 유지]
    assert all(r["active"] for r in Storage(db).list_lessons("concept"))  # 보여 주기만
    at.text_area[0].set_value("형식 A·B 를 꼭 지킨다")
    at = at.button[0].click().run()
    rows = Storage(db).list_lessons("concept")
    assert [(r["lesson"], r["active"]) for r in rows][0] == ("형식 A·B 를 꼭 지킨다", 1)
    assert [r["active"] for r in rows[1:]] == [0, 0]
    assert any("1개로 합쳤습니다" in s.value for s in at.success)


def test_conflict_panel_shows_original_feedback_and_keeps_the_pick(tmp_path):
    """충돌: 교훈마다 원래 피드백(결정·만족도·코멘트)을 보여 주고, '#번호만 남기기'를 눌러야 나머지를 끈다."""
    db = tmp_path / "runs.db"
    storage = Storage(db)
    storage.create_run("r9", "concept", "lead", "FlashInfer 정리")
    storage.save_decision("r9", HumanDecision(action="adopt", adopted_node_id="lead", score=2, comment="절 번호는 붙이지 마"))
    a = add(storage, "concept", "lead", "절 번호를 붙인다")[0]
    b = storage.add_lessons("concept", "r9", [Lesson(agent_id="lead", kind="improve", lesson="절 번호를 붙이지 않는다")])[0]
    proposal = CleanupProposal(team_id="concept", agent_id="lead", ids=[a, b], new_id=b, relation="conflict", kind="improve",
                               merged="절 번호는 문서 형식을 따른다", reason="절 번호가 반대", by="model")

    def page(db, proposal):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        prompt_ui.render_cleanup(load_team("concept"), [proposal], key="r")

    at = AppTest.from_function(page, args=(db, proposal), default_timeout=30).run()
    assert not at.exception, at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "채택 (만족도 2/5)" in captions and "FlashInfer 정리" in captions and "r9" in captions  # 원래 피드백
    assert any("절 번호는 붙이지 마" in t.value for t in at.text)  # 그때 남긴 코멘트
    assert all(r["active"] for r in Storage(db).list_lessons("concept"))  # 보여 주기만
    at = at.button(key=f"cleanup_only_r_concept_lead_{a}-{b}_{b}").click().run()
    assert [r["id"] for r in Storage(db).list_lessons("concept") if r["active"]] == [b]
    assert any(f"#{b}만 남기고" in s.value for s in at.success)


def test_merged_notice_can_be_undone(tmp_path):
    """자동으로 합친 교훈 — 무엇을 합쳤는지 알리고 '되돌리기'를 누르면 원래대로 (지우지 않음)."""
    from core.feedback import merge_lessons

    db = tmp_path / "runs.db"
    ids = add(Storage(db), "concept", "lead", "형식 A 를 지킨다", "형식 A 를 꼭 지킨다")
    p = CleanupProposal(team_id="concept", agent_id="lead", ids=ids, new_id=ids[1], relation="overlap", kind="improve",
                        merged="형식 A 를 지킨다", reason="같은 지시", by="model", hits=2)
    p = p.model_copy(update={"merged_id": merge_lessons(Storage(db), p)})

    def page(db, merged):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        prompt_ui.render_merged(load_team("concept"), merged, key="r")

    at = AppTest.from_function(page, args=(db, [p]), default_timeout=30).run()
    assert not at.exception, at.exception
    assert any(f"#{p.merged_id}" in m.value and "2회" in m.value for m in at.markdown)
    at = at.button[0].click().run()
    assert {r["id"] for r in Storage(db).list_lessons("concept") if r["active"]} == set(ids)
    assert len(Storage(db).list_lessons("concept")) == 3 and len(at.button) == 0


# ─── 📚 교훈 관리 화면 ───────────────────────────────────────────────────


def test_lesson_manager_edit_toggle_delete_with_confirm(tmp_path):
    db = tmp_path / "runs.db"
    lid = add(Storage(db), "concept", "lead", "옛 문장")[0]

    def page(db):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        team = load_team("concept")
        prompt_ui.render_lessons(team, next(p for p in prompt_ui.team_prompts(
            team, prompt_ui._ctx.ranks, prompt_ui._ctx.models, prompt_ui._ctx.storage, 20000) if p.node_id == "lead"), "m")

    at = AppTest.from_function(page, args=(db,), default_timeout=30).run()
    assert not at.exception, at.exception
    at.text_area(key=f"lesson_text_{lid}").set_value("새 문장")
    at = at.button(key=f"lesson_save_{lid}").click().run()
    assert Storage(db).list_lessons("concept")[0]["lesson"] == "새 문장"

    at = at.button(key=f"lesson_toggle_{lid}").click().run()
    assert Storage(db).list_lessons("concept")[0]["active"] == 0
    assert team_lessons(Storage(db), "concept", 20000) == {}  # 꺼진 교훈은 프롬프트에 없다

    at = at.button(key=f"lesson_del_btn_{lid}").click().run()
    assert len(Storage(db).list_lessons("concept")) == 1  # 한 번 누르면 묻기만
    at = at.button(key=f"lesson_del_no_{lid}").click().run()
    at = at.button(key=f"lesson_del_btn_{lid}").click().run()
    at = at.button(key=f"lesson_del_ok_{lid}").click().run()
    assert Storage(db).list_lessons("concept") == []



# ─── 부서 화면 (app.py) ─────────────────────────────────────────────────


def test_dept_screen_shows_head_and_every_team_prompt(tmp_path, monkeypatch, set_keys):
    """리서치·지식부 화면에서 토글을 켜면 부장 비평 관점 + 두 팀(general, concept) 팀원 프롬프트가 교훈과 함께 보인다."""
    from pathlib import Path

    import streamlit as st

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    add(Storage(tmp_path / "data" / "runs.db"), "general", "lead", "출처를 꼭 단다")
    # 앱의 저장소(get_storage)는 st.cache_resource 라, 앞서 다른 데이터 폴더로 앱을 띄운 테스트의 것이 남아 있으면 이 DB 를 안 본다
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.query_params["dept"] = "research"
        at.query_params["view"] = "dept"
        at = at.run()
        assert not at.exception, at.exception
        at = at.toggle(key="prompts_research").set_value(True).run()
        assert not at.exception, at.exception
    finally:
        st.cache_resource.clear()
    codes = [c.value for c in at.code]
    assert any(c.startswith("너는 리서치·지식부 부장이다.") and "비평가(Critic)" in c for c in codes)  # 부장 비평 프롬프트
    assert any("출처를 꼭 단다" in c and "지난 업무에서 받은 피드백" in c for c in codes)  # 일반 팀 팀장 + 교훈
    assert any("## 스킬:" in c for c in codes)  # 개념 편집 팀 (스킬 포함)


def test_orphan_lessons_move_only_after_user_picks(tmp_path):
    """조직도에 없는 옛 id(직급 id)로 남은 교훈 — 알리고, 사용자가 '옮기기'를 눌러야 받을 팀원에게 간다."""
    from core.config import load_ranks, load_team
    from core.feedback import orphan_lessons, suggest_owner

    db = tmp_path / "runs.db"
    ids = add(Storage(db), "general", "team_lead", "옛 팀장 교훈")
    general, ranks = load_team("general"), load_ranks()
    assert list(orphan_lessons(Storage(db), general, ranks)) == ["team_lead"]
    assert suggest_owner("team_lead", general, ranks) == "lead"  # 같은 직급의 팀원
    assert "lead" not in team_lessons(Storage(db), "general", 20000)  # 지금은 어느 프롬프트에도 없다

    def page(db):
        import prompt_ui
        from core.config import load_models, load_ranks, load_team
        from core.storage import Storage

        prompt_ui.configure(prompt_ui.Ctx(storage=Storage(db), ranks=load_ranks(), models=load_models()))
        prompt_ui.render_orphans(load_team("general"), key="o")

    at = AppTest.from_function(page, args=(db,), default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="orphan_to_general_team_lead").value == "lead"
    assert Storage(db).list_lessons("general")[0]["agent_id"] == "team_lead"  # 보여 주기만
    at = at.button(key="orphan_move_general_team_lead").click().run()
    assert [r["id"] for r in Storage(db).list_lessons("general", "lead")] == ids
    assert team_lessons(Storage(db), "general", 20000)["lead"].texts == ["옛 팀장 교훈"]
