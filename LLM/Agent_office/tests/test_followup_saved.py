"""💬 수정 대화(수정본·부장 검토)를 실행 기록(runs.db)에 남겨, 앱을 다시 켠 뒤에도 보이게 (2026-11-12).

전에는 대화(FollowupTurn 목록)가 서버 메모리에만 있어, 앱을 다시 켜면 ① 결정 전이던 업무의 수정본(채택 후보)이 사라지고
② 채택한 뒤 '부장이 무엇을 고쳤는지'를 다시 볼 수 없고 ③ 문제가 생겼을 때 뒤에서 확인할 수 없었다.
가짜 모델·임시 DB 만 쓴다. 화면 준비는 tests/test_progress_followup.py 의 것을 그대로 쓴다.
"""

import json
import sqlite3
from datetime import datetime, timedelta

from core import feedback_sync, followup, live
from core.schemas import AgentAnswer, HarnessResult, HumanDecision, Lesson, ModelSpec
from core.storage import Storage
from tests.fakes import Script, make_critic, make_delegation, make_review
from tests.test_followup import edit, judge
from tests.test_progress_followup import _open, app_env, wait_for  # noqa: F401 — app_env 는 pytest 준비 함수
from tests.test_revision_reaches_final import _result_box

SPEC = ModelSpec(provider="openai", model="gpt-5.4")
SECRETS = ("비공개 수정본 전문", "부장 분석 문장", "표 제목에 기준일 넣기", "요청이 다르게 반영됨", "정의를 두 문장으로 줄여줘")


def revision(version: int = 1, content: str = "비공개 수정본 전문") -> followup.FollowupTurn:
    """부장 검토·직접 고친 점까지 모든 칸이 찬 수정본 한 번."""
    return followup.FollowupTurn(
        request="정의를 두 문장으로 줄여줘", reply="정의를 줄였습니다.", revised=True, content=content, changes=["정의를 줄임"],
        harness=[HarnessResult(validator="python_syntax", passed=True, score=1.0, log="ok")],
        log=["부장이 추가로 고칠 점 1개를 직접 고침 1회"], model=SPEC, elapsed_sec=3.2, version=version,
        critic=followup.RevisionCritic(analysis="부장 분석 문장", logical_leaps=[], rule_violations=["요청이 다르게 반영됨"],
                                       severity="minor", score=4, extra_fixes=["용어 통일"]),
        critic_by="🔬 연구부 부장", critic_model=ModelSpec(provider="anthropic", model="claude-opus-5-5"), critic_passed=True,
        auto_fixed=["표 제목에 기준일 넣기"], auto_fixed_earlier=["각주 번호 맞추기"])


def question() -> followup.FollowupTurn:
    return followup.FollowupTurn(request="3번 근거가 뭐야?", reply="[2] 자료가 근거입니다.", model=SPEC, elapsed_sec=1.0)


def new_storage(tmp_path) -> Storage:
    storage = Storage(tmp_path / "runs.db")
    storage.create_run("r1", "general", "lead", "양자컴퓨팅 요약")
    return storage


# ─── 저장·복원 (core/storage.py · core/followup.py) ───────────────────────────


def test_turns_come_back_after_reopening_the_storage(tmp_path):
    storage = new_storage(tmp_path)
    turns = [question(), revision(1), revision(2, "두 번째 수정본")]
    storage.save_followups("r1", followup.dump_turns(turns[:2]))
    storage.save_followups("r1", followup.dump_turns(turns))  # 대화가 늘 때마다 통째로 다시 저장한다 (덮어씀)
    again = followup.load_turns(Storage(tmp_path / "runs.db").get_followups("r1"))  # 앱을 다시 켠 것처럼 새로 연다
    assert again == turns  # 요청·답·수정본 전문·버전·바뀐 점·하네스·기록·모델·걸린 시간·부장 검토·직접 고친 점까지 그대로
    assert isinstance(again[1], followup.FollowupTurn) and isinstance(again[1].critic, followup.RevisionCritic)
    assert followup.head_review_markdown(again[1], "부장") == followup.head_review_markdown(turns[1], "부장")
    assert followup.next_request(again[2]) == followup.next_request(turns[2]) != ""
    assert followup.decision_note(again) == followup.decision_note(turns)


def test_run_without_saved_turns_and_old_db_still_open(tmp_path):
    storage = new_storage(tmp_path)
    assert storage.get_followups("r1") == "" and followup.load_turns(storage.get_followups("r1")) == []  # 대화가 없는 실행
    step = AgentAnswer(node_id="lead", rank_id="team_lead", kind="review", content="원래 결과", model=SPEC)
    storage.save_steps("r1", [step], "awaiting_human")
    with sqlite3.connect(tmp_path / "runs.db") as conn:  # 이 표가 생기기 전의 DB
        conn.execute("DROP TABLE run_followups")
    old = Storage(tmp_path / "runs.db")
    row = old.get_run("r1")
    assert row["question"] == "양자컴퓨팅 요약" and row["steps"][0].content == "원래 결과"  # 옛 기록은 그대로 열린다
    assert old.get_followups("r1") == ""
    old.save_followups("r1", followup.dump_turns([revision()]))  # 그 뒤로는 저장도 된다
    assert len(followup.load_turns(old.get_followups("r1"))) == 1


def test_saved_turns_go_away_with_the_run(tmp_path):
    storage = new_storage(tmp_path)
    storage.save_followups("r1", followup.dump_turns([revision()]))
    storage.delete_run("r1")  # 🗑 결과 버리기
    assert storage.get_followups("r1") == ""
    storage.create_run("r2", "general", "lead", "오래된 미결정 업무")
    storage.save_followups("r2", followup.dump_turns([revision()]))
    old = (datetime.now() - timedelta(hours=50)).isoformat(timespec="seconds")
    storage._execute("UPDATE runs SET created_at=? WHERE id=?", (old, "r2"))
    assert storage.purge_undecided(24) == ["r2"] and storage.get_followups("r2") == ""  # 오래된 미결정 실행 정리


def test_saved_turns_never_reach_the_public_knowledge_folder(tmp_path):
    """수정 대화는 이 PC 의 data/runs.db 에만 — knowledge/(공개 저장소에 올라가는 폴더) 내보내기에 들어가지 않는다."""
    storage = new_storage(tmp_path)
    turns = [revision()]
    storage.save_followups("r1", followup.dump_turns(turns))
    storage.add_lessons("general", "r1", [Lesson(agent_id="lead", kind="improve", lesson="정의는 두 문장 안으로 쓴다")])
    storage.save_decision("r1", HumanDecision(action="adopt", adopted_node_id="lead", final_answer="비공개 수정본 전문", score=4,
                                              comment=followup.decision_note(turns)))
    dest = tmp_path / "knowledge"
    feedback_sync.export(storage, dest)
    exported = "\n".join(p.read_text(encoding="utf-8") for p in dest.iterdir() if p.is_file())
    assert "정의는 두 문장 안으로 쓴다" in exported and "adopted" in exported  # 교훈·결정 요약은 나간다
    for secret in SECRETS:  # 수정본 전문 · 부장 분석·지적·직접 고친 점 · 사람의 수정 요청
        assert secret not in exported
    # 교훈 재료(결정 코멘트)에도 부장 의견은 없다 — 사람이 보낸 요청만 (decision_note 규칙 그대로)
    note = followup.decision_note(followup.load_turns(storage.get_followups("r1")))
    assert "정의를 두 문장으로 줄여줘" in note and "부장 분석 문장" not in note and "표 제목에 기준일 넣기" not in note


# ─── 다시 켠 뒤 화면용 run 으로 (core/live.py) ────────────────────────────────


def restored_row(status: str) -> dict:
    step = AgentAnswer(node_id="lead", rank_id="team_lead", kind="review", content="원래 결과", model=SPEC)
    return {"id": "r1", "team_id": "general", "root_id": "lead", "question": "질문", "steps": [step], "status": status,
            "decision": None, "created_at": "2026-11-12T09:00:00"}


def test_restored_run_gets_saved_turns():
    turns = [question(), revision()]
    run = live.restore_run(restored_row("awaiting_human"), turns)
    assert run["followups"] == turns and run["status"] == "awaiting_human" and not run.get("followup_error")
    assert live.restore_run(restored_row("awaiting_human")).get("followups") == []  # 대화가 없던 실행


def test_answer_cut_off_by_restart_keeps_saved_turns_without_a_waiting_mark():
    """답을 기다리던 중에 앱이 꺼졌으면 — 저장된 대화까지만 되살리고, '답하는 중' 표시는 남기지 않는다."""
    run = live.restore_run(restored_row(live.FOLLOWUP_RUNNING), [revision()])
    assert [t.version for t in run["followups"]] == [1]
    assert "followup_pending" not in run and run["status"] == "awaiting_human" and run["done"]
    assert run["followup_error"] == live.FOLLOWUP_INTERRUPTED  # 다시 보내 달라고 알린다


# ─── 화면: 다시 켠 뒤에도 대화·수정본·부장 검토가 보이고, 복원한 수정본을 채택할 수 있다 ──────────


def _texts(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_restores_conversation_after_restart_and_adopts_the_revision(app_env, use_script, tmp_path):
    import streamlit as st

    gate, draft_gate = app_env
    gate.set()
    draft_gate.set()

    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    use_script(Script(
        text="초안 [1]", Delegation=delegation, ReviewResult=make_review(verdict="approve", score=9, answer="보완안"),
        CriticResult=make_critic(5), FollowupResult=[edit("수정본 하나")],
        RevisionCritic=[judge(4, ["표 제목이 요청과 다름"], extra=["용어 통일"])], HeadFix=followup.HeadFix(revised_answer="")))
    at = _open().run()
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()
    at.text_area(key="q_text").input("양자컴퓨팅 요약")
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown) else None)(_open().run()), 60)
    at.text_area(key=next(t.key for t in at.text_area if t.placeholder and "3번 근거" in t.placeholder)).input("정의를 줄여줘")
    at = next(b for b in at.button if b.label == "💬 보내기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("수정본 v1" in m.value for m in a.markdown) else None)(_open().run()), 60)
    before = _texts(at)
    db = tmp_path / "data" / "runs.db"
    run_id = wait_for(lambda: next((r["id"] for r in Storage(db).list_runs() if Storage(db).get_followups(r["id"])), None))

    st.cache_resource.clear()  # ── 앱을 다시 켠다 (서버 메모리의 run 이 사라진다) ──
    at = _open().run()
    assert not at.exception, at.exception
    after = _texts(at)
    for part in ("정의를 줄여줘", "정의를 줄였습니다.", "**✏️ 수정본 v1**", "검토** · 4/5 · ✅ 합격", "표 제목이 요청과 다름", "용어 통일"):
        assert part in before and part in after, part  # 대화 · 수정본 vN · 부장 검토(지적·남은 추가로 고칠 점)가 전과 같이 보인다
    title, shown = _result_box(at)
    assert shown == "수정본 하나" and "수정본 v1" in title  # 지금 결과 = 최신 수정본
    assert not any("답하는 중" in m.value for m in at.markdown)
    assert any(b.key == f"send_fixes_{run_id}_1" for b in at.button)  # 최신 수정본에만 '다음 수정 요청' 버튼
    pick = at.radio(key=f"pick_{run_id}_1")
    assert pick.options[0] == "✏️ 수정본 v1 (최신)" and any("(원래 최종)" in o for o in pick.options)  # '채택할 안'에 수정본이 다시 나온다

    at = next(b for b in at.button if b.label == "✅ 이 안으로 채택").click().run()  # 기본 선택 = 복원한 최신 수정본
    assert not at.exception, at.exception
    decision = Storage(db).get_run(run_id)["decision"]
    assert decision.action == "adopt" and decision.final_answer == "수정본 하나"
    assert "정의를 줄여줘" in decision.comment and "표 제목이 요청과 다름" not in decision.comment  # 교훈 재료에는 사람의 요청만

    st.cache_resource.clear()  # ── 채택한 뒤 다시 켜고 🕘 지난 업무에서 연다 ──
    at = _open().run()
    at = at.button(key=f"hist_{run_id}").click().run()
    assert not at.exception, at.exception
    history = _texts(at)
    for part in ("정의를 줄여줘", "**✏️ 수정본 v1**", "검토** · 4/5 · ✅ 합격", "표 제목이 요청과 다름"):
        assert part in history, part
    assert not any((b.key or "").startswith("send_fixes_") for b in at.button)  # 지난 업무에서는 보기만 한다
    downloads = [d.proto.label for d in at.get("download_button")]
    assert any(label.startswith("⬇ 수정본 v1 받기") for label in downloads)  # 수정본 전문은 내려받아 확인한다


# ─── ✏️ 질문·브리프 고쳐 다시 — 앞선 시도의 수정 대화는 새 결과에 붙지 않는다 ─────────────────


def test_app_redo_moves_earlier_revision_out_of_the_new_result(app_env, use_script, tmp_path):
    """수정본 v1 을 받은 뒤 질문을 고쳐 다시 맡기면, v1 은 옛 결과를 고친 글이라 새 결과의 '지금 결과'·'채택할 안'·수정 대화에
    남으면 안 된다 (남으면 새 결과 대신 옛 수정본이 채택된다). 실행 기록도 비우고, 옛 수정본은 '🆚 이전 시도'에서 본다."""
    gate, draft_gate = app_env
    gate.set()
    draft_gate.set()

    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    def review(messages):  # 고친 질문으로 다시 하면 결과가 달라진다
        again = any("고친 질문" in str(m.content) for m in messages)
        return make_review(verdict="approve", score=9, answer="새 결과" if again else "옛 결과")

    use_script(Script(text="초안 [1]", Delegation=delegation, ReviewResult=review, CriticResult=make_critic(5),
                      FollowupResult=[edit("옛 결과를 고친 수정본")], RevisionCritic=[judge(5)]))
    at = _open().run()
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()
    at.text_area(key="q_text").input("양자컴퓨팅 요약")
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown) else None)(_open().run()), 60)
    assert _result_box(at)[1] == "옛 결과"
    at.text_area(key=next(t.key for t in at.text_area if t.placeholder and "3번 근거" in t.placeholder)).input("정의를 줄여줘")
    at = next(b for b in at.button if b.label == "💬 보내기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("수정본 v1" in m.value for m in a.markdown) else None)(_open().run()), 60)
    db = tmp_path / "data" / "runs.db"
    run_id = wait_for(lambda: next((r["id"] for r in Storage(db).list_runs() if Storage(db).get_followups(r["id"])), None))
    assert any("앞선 수정본" in c.value for c in at.caption)  # 고쳐 다시 하기 전에 알린다 — 수정본이 채택 후보에서 빠진다

    at.text_area(key=f"rs_q_{run_id}").input("고친 질문")  # ── ✏️ 질문 고쳐 다시 ──
    at = next(b for b in at.button if b.label == "✏️ 이대로 다시 하기").click().run()
    assert not at.exception, at.exception
    assert followup.load_turns(Storage(db).get_followups(run_id)) == []  # 다시 맡긴 순간 실행 기록도 비운다 (도중에 꺼도 옛 수정본이 안 붙게)
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown)
                           and any(e.label.startswith("🆚 이전 시도 1개") for e in a.expander) else None)(_open().run()), 60)
    assert not at.exception, at.exception
    title, shown = _result_box(at)
    assert shown == "새 결과" and "수정본" not in title  # 지금 결과 = 새 결과 (옛 수정본이 아니다)
    pick = next(r for r in at.radio if r.label == "채택할 안")
    assert not any("수정본" in o for o in pick.options) and pick.options[0].endswith("(최종)")
    assert not any("**✏️ 수정본 v1**" in m.value or m.value == "정의를 줄여줘" for m in at.markdown)  # 수정 대화도 새로 시작
    assert followup.load_turns(Storage(db).get_followups(run_id)) == []
    # 옛 수정본은 버리지 않는다 — '🆚 이전 시도'의 옛 쪽에 그 시도의 마지막 글(수정본)로 남는다
    assert any("1번째 시도" in m.value and "수정본 v1" in m.value for m in at.markdown)
    assert any(m.value == "옛 결과를 고친 수정본" for m in at.markdown)
    kept = wait_for(lambda: (Storage(db).get_monitor(run_id) or {}).get("attempts"))  # 디스크에는 ⏱ 기록의 이전 시도에 남는다
    assert [t.content for t in followup.load_turns(json.dumps(kept[0]["followups"]))] == ["옛 결과를 고친 수정본"]

    st_at = next(b for b in at.button if b.label == "✅ 이 안으로 채택").click().run()  # 기본 선택 = 새 결과
    assert not st_at.exception, st_at.exception
    decision = Storage(db).get_run(run_id)["decision"]
    assert decision.final_answer == "새 결과" and "정의를 줄여줘" not in decision.comment  # 옛 대화는 새 결과의 교훈 재료가 아니다
