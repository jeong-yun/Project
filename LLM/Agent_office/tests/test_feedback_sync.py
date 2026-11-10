"""피드백 → knowledge/ (git 에 올릴 교훈·결정 요약) — 개인 정보 제외, 자동 동기화 조건, 다른 PC 에서 가져오기."""

import json

import pytest

from core import feedback_sync
from core.schemas import AgentAnswer, HumanDecision, Lesson, ModelSpec
from core.storage import Storage

SECRET_Q = "삼성전자 120주 평단 71,500원, 다음 주 오사카 3박 4일"
SECRET_A = "보유 120주는 72,000원에 절반 매도"


def seeded(path) -> Storage:
    storage = Storage(path)
    storage.create_run("r1", "stock", "lead", SECRET_Q)
    step = AgentAnswer(node_id="lead", rank_id="team_lead", kind="integrate", content=SECRET_A, model=ModelSpec(provider="openai", model="x"))
    storage.save_steps("r1", [step], "awaiting_human")
    storage.save_decision("r1", HumanDecision(action="adopt", adopted_node_id="lead", final_answer=SECRET_A, edited=True,
                                              score=4, comment="내 계좌 기준으로 다시"))
    storage.add_lessons("stock", "r1", [Lesson(agent_id="lead", kind="improve", lesson="손절가를 기간별 기준으로 쓴다"),
                                        Lesson(agent_id="am_data", kind="praise", lesson="시세 출처와 기준일을 단다")])
    return storage


def test_export_has_lessons_and_decision_summary_but_no_private_text(tmp_path):
    storage = seeded(tmp_path / "runs.db")
    out = feedback_sync.export(storage, tmp_path / "knowledge")
    assert out["lessons"] == 2 and out["decisions"] == 1 and set(out["changed"]) == {"lessons.jsonl", "decisions.jsonl", "교훈.md"}
    files = {p.name: p.read_text(encoding="utf-8") for p in (tmp_path / "knowledge").iterdir()}
    everything = "".join(files.values())
    for private in ("삼성전자", "71,500", "오사카", "72,000", "내 계좌"):  # 질문·답변·코멘트 원문은 올리지 않는다
        assert private not in everything
    decision = json.loads(files["decisions.jsonl"])
    assert decision["adopted_agent"] == "lead" and decision["score"] == 4 and decision["edited"] and decision["commented"]
    assert "손절가를 기간별 기준으로 쓴다" in files["lessons.jsonl"] and "🔧 손절가" in files["교훈.md"]
    assert "결정 1건 (채택 1 · 반려 0) · 평균 만족도 4.0/5" in files["교훈.md"]


def test_export_is_stable_and_marks_disabled_lessons(tmp_path):
    storage = seeded(tmp_path / "runs.db")
    folder = tmp_path / "knowledge"
    feedback_sync.export(storage, folder)
    assert feedback_sync.export(storage, folder)["changed"] == []  # 그대로면 다시 쓰지 않는다 (git 변경 없음)
    lesson_id = storage.list_lessons("stock", "lead")[0]["id"]
    storage.update_lesson(lesson_id, active=False)
    assert feedback_sync.export(storage, folder)["changed"] == ["lessons.jsonl", "교훈.md"]
    assert "~~🔧 손절가를 기간별 기준으로 쓴다" in (folder / "교훈.md").read_text(encoding="utf-8")


def test_auto_sync_only_when_env_is_set(tmp_path, monkeypatch):
    folder = tmp_path / "knowledge"
    monkeypatch.delenv(feedback_sync.ENV, raising=False)
    seeded(tmp_path / "a.db")
    assert not folder.exists()  # 테스트·개발용 실행은 knowledge/ 를 건드리지 않는다
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    storage = seeded(tmp_path / "b.db")  # 결정·교훈 저장 때마다 자동으로 내보낸다
    assert len((folder / "lessons.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    storage.add_lessons("travel", None, [Lesson(agent_id="lead", kind="praise", lesson="일정표에 이동 시간을 넣는다")])
    assert "일정표에 이동 시간을 넣는다" in (folder / "교훈.md").read_text(encoding="utf-8")


def test_new_pc_imports_lessons_from_knowledge_without_duplicates(tmp_path, monkeypatch):
    folder = tmp_path / "knowledge"
    feedback_sync.export(seeded(tmp_path / "old.db"), folder)
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    fresh = Storage(tmp_path / "new.db")  # clone 한 PC — 시작할 때 knowledge/ 의 교훈을 가져온다
    lessons = fresh.list_lessons("stock")
    assert {l["lesson"] for l in lessons} == {"손절가를 기간별 기준으로 쓴다", "시세 출처와 기준일을 단다"}
    assert fresh.recent_lessons("stock", "lead", 5) == ["손절가를 기간별 기준으로 쓴다"]  # 프롬프트에 바로 쓰인다
    Storage(tmp_path / "new.db")  # 다시 켜도 중복으로 넣지 않는다
    assert len(Storage(tmp_path / "new.db").list_lessons("stock")) == 2


def test_sync_failure_never_blocks(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("파일", encoding="utf-8")
    monkeypatch.setenv(feedback_sync.ENV, str(blocker))  # 폴더를 만들 수 없는 경로
    storage = seeded(tmp_path / "runs.db")  # 예외 없이 결정·교훈은 저장된다
    assert storage.get_run("r1")["status"] == "adopted" and "error" in storage.last_feedback_sync


def test_human_comment_is_cut_from_public_lessons_without_duplicating_on_import(tmp_path, monkeypatch):
    folder = tmp_path / "knowledge"
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    storage = Storage(tmp_path / "runs.db")
    storage.add_lessons("stock", None, [
        Lesson(agent_id="lead", kind="improve", lesson="기준가를 먼저 확인한다 (사람 코멘트: 내 평단 71,500원 기준\n[결정 전 문의] 1) 매도?)"),
        Lesson(agent_id="staff", kind="improve", lesson="출처를 단다 사람 코멘트: 삼성전자 120주"),
    ])
    text = "".join(p.read_text(encoding="utf-8") for p in folder.iterdir())
    assert "71,500" not in text and "삼성전자" not in text and "사람 코멘트" not in text
    assert "기준가를 먼저 확인한다" in text and "출처를 단다" in text
    assert len(Storage(tmp_path / "runs.db").list_lessons("stock")) == 2  # 같은 PC 에서 다시 켜도 코멘트 뗀 교훈을 중복으로 넣지 않는다
    assert "71,500" in storage.list_lessons("stock", "lead")[0]["lesson"]  # DB(로컬)의 원문은 그대로 — 프롬프트에는 계속 쓰인다


def test_repeat_count_travels_with_knowledge_only_when_repeated(tmp_path, monkeypatch):
    """겹치는 교훈을 합친 '피드백 N회 반복'도 다른 PC 로 이어진다. 1회짜리 줄은 예전 형식 그대로 (git 변경이 생기지 않게)."""
    storage = seeded(tmp_path / "old.db")
    storage.add_lessons("stock", None, [Lesson(agent_id="lead", kind="improve", lesson="기준일을 단다")], hits=3)
    folder = tmp_path / "knowledge"
    feedback_sync.export(storage, folder)
    rows = [json.loads(line) for line in (folder / feedback_sync.LESSONS).read_text(encoding="utf-8").splitlines()]
    assert [r.get("hits") for r in rows] == [None, None, 3]
    assert "피드백 3회 반복" in (folder / feedback_sync.READABLE).read_text(encoding="utf-8")
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    fresh = Storage(tmp_path / "new.db")
    assert {r["lesson"]: r["hits"] for r in fresh.list_lessons("stock")}["기준일을 단다"] == 3


def test_other_pc_follows_merged_lessons_instead_of_keeping_both(tmp_path, monkeypatch):
    """한 PC 에서 겹치는 교훈을 합치면(원래 교훈 끔 + 합친 교훈 추가) 다른 PC 도 가져올 때 원래 교훈을 끈다.

    전에는 '없는 교훈만 넣기'라 다른 PC 에는 원래 교훈이 켜진 채 남아 합친 교훈과 함께 프롬프트에 들어갔고,
    그 PC 를 켜면 knowledge/lessons.jsonl 의 켜짐 값이 다시 뒤집혔다 (2026-11-10 점검).
    규칙: 켜짐은 더 나중에 바꾼 쪽을 따르고(합치기가 원래 교훈을 끈 시각이 남는다), 반복 횟수는 큰 쪽을 따른다.
    """
    from core.feedback import merge_lessons
    from core.schemas import CleanupProposal

    folder = tmp_path / "knowledge"
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    a = Storage(tmp_path / "a.db")
    ids = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="출처를 단다"),
                                          Lesson(agent_id="lead", kind="improve", lesson="출처 링크를 붙인다")])
    Storage(tmp_path / "b.db")  # 다른 PC — 켤 때 두 교훈을 가져온다
    merge_lessons(a, CleanupProposal(team_id="general", agent_id="lead", ids=ids, new_id=ids[-1], relation="overlap",
                                     kind="improve", merged="모든 주장에 출처 링크를 단다", reason="같은 지시", by="model", hits=2))
    written_by_a = (folder / feedback_sync.LESSONS).read_text(encoding="utf-8")

    b = Storage(tmp_path / "b.db")  # 다른 PC 를 다시 켠다 (git pull 로 knowledge/ 를 받은 뒤)
    assert {r["lesson"]: (bool(r["active"]), r["hits"]) for r in b.list_lessons("general")} == {
        "출처를 단다": (False, 1), "출처 링크를 붙인다": (False, 1), "모든 주장에 출처 링크를 단다": (True, 2)}
    assert b.recent_lessons("general", "lead", 5) == ["모든 주장에 출처 링크를 단다"]  # 프롬프트에는 합친 교훈만
    assert (folder / feedback_sync.LESSONS).read_text(encoding="utf-8") == written_by_a  # 파일이 뒤집히지 않는다
    assert b.last_feedback_sync["aligned"] == 2 and b.last_feedback_sync["changed"] == []


def test_import_keeps_a_later_local_off_and_never_lowers_the_count(tmp_path, monkeypatch):
    """이 PC 에서 나중에 끈 교훈을 그 전에 쓴 파일('켜짐')이 다시 켜지 않고, 반복 횟수도 줄이지 않는다."""
    folder = tmp_path / "knowledge"
    a = Storage(tmp_path / "a.db")
    [lid] = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="출처를 단다")], hits=3)
    feedback_sync.export(a, folder)  # 파일: 켜짐 · 3회
    a.update_lesson(lid, active=False)
    a._run_sql("UPDATE lessons SET hits=5 WHERE id=?", (lid,))
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    again = Storage(tmp_path / "a.db")
    [row] = again.list_lessons("general")
    assert not row["active"] and row["hits"] == 5 and again.last_feedback_sync["aligned"] == 0


@pytest.fixture
def clock(monkeypatch):
    """부를 때마다 1초씩 가는 가짜 시계 — 켜고 끄기가 같은 초에 일어나면 어느 쪽이 나중인지 가릴 수 없다."""
    import itertools
    from datetime import datetime, timedelta

    from core import storage

    ticks = itertools.count()
    monkeypatch.setattr(storage, "_now", lambda: (datetime(2026, 11, 11, 9) + timedelta(seconds=next(ticks))).isoformat(timespec="seconds"))


def _state(storage, team="general"):
    return {r["lesson"]: bool(r["active"]) for r in storage.list_lessons(team)}


def test_turning_a_lesson_back_on_reaches_the_other_pc_and_stays_on(tmp_path, monkeypatch, clock):
    """한 PC 에서 껐다가 다시 켠 교훈은 다른 PC 에서도 켜지고, PC 를 한 바퀴 돌아도 조용히 다시 꺼지지 않는다.

    '꺼진 쪽을 따른다'만으로는 다른 PC 가 꺼진 채 남아 파일을 다시 '꺼짐'으로 쓰고, 처음 PC 가 그것을 가져와 다시 꺼졌다 (2026-11-11 검토).
    """
    folder = tmp_path / "knowledge"
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    a = Storage(tmp_path / "a.db")
    [lid] = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="출처를 단다")])
    Storage(tmp_path / "b.db")  # PC B 가 가져온다
    a.update_lesson(lid, active=False)  # A 에서 끔
    assert _state(Storage(tmp_path / "b.db")) == {"출처를 단다": False}  # B 도 꺼진다
    a.update_lesson(lid, active=True)  # A 에서 다시 켬 (📚 교훈 관리)
    written_by_a = (folder / feedback_sync.LESSONS).read_text(encoding="utf-8")
    assert _state(Storage(tmp_path / "b.db")) == {"출처를 단다": True}  # B 도 다시 켜진다
    assert (folder / feedback_sync.LESSONS).read_text(encoding="utf-8") == written_by_a  # B 가 파일을 '꺼짐'으로 되돌려 쓰지 않는다
    assert _state(Storage(tmp_path / "a.db")) == {"출처를 단다": True}  # A 를 다시 켜도 그대로 켜져 있다


def test_undoing_a_merge_reaches_the_other_pc(tmp_path, monkeypatch, clock):
    """합치기 '↩ 되돌리기'도 다른 PC 로 전해진다 — 원래 교훈이 다시 켜지고 합친 교훈이 꺼진다 (둘 다 꺼진 채 남지 않는다)."""
    from core.feedback import merge_lessons, undo_merge
    from core.schemas import CleanupProposal

    folder = tmp_path / "knowledge"
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    a = Storage(tmp_path / "a.db")
    ids = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="출처를 단다"),
                                          Lesson(agent_id="lead", kind="improve", lesson="출처 링크를 붙인다")])
    p = CleanupProposal(team_id="general", agent_id="lead", ids=ids, new_id=ids[-1], relation="overlap", kind="improve",
                        merged="모든 주장에 출처 링크를 단다", reason="같은 지시", by="model", hits=2)
    p = p.model_copy(update={"merged_id": merge_lessons(a, p)})
    merged_on = {"출처를 단다": False, "출처 링크를 붙인다": False, "모든 주장에 출처 링크를 단다": True}
    assert _state(Storage(tmp_path / "b.db")) == merged_on  # B: 합친 상태를 따른다 (원래 고치려던 문제)
    undo_merge(a, p)
    undone = {"출처를 단다": True, "출처 링크를 붙인다": True, "모든 주장에 출처 링크를 단다": False}
    assert _state(Storage(tmp_path / "b.db")) == undone
    assert _state(Storage(tmp_path / "a.db")) == undone  # 한 바퀴 돌아도 A 의 되돌리기가 그대로


def test_older_knowledge_file_does_not_override_a_later_choice(tmp_path, monkeypatch, clock):
    """파일이 이 PC 의 마지막 선택보다 오래됐으면(git 으로 옛 파일을 받음 등) 이 PC 의 선택을 따른다 — 켠 것도, 끈 것도."""
    folder = tmp_path / "knowledge"
    a = Storage(tmp_path / "a.db")  # 동기화를 켜지 않은 채로
    on, off = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="켤 교훈"),
                                              Lesson(agent_id="lead", kind="improve", lesson="끌 교훈")])
    a.update_lesson(on, active=False)
    feedback_sync.export(a, folder)  # 옛 파일: '켤 교훈' 꺼짐 · '끌 교훈' 켜짐
    a.update_lesson(on, active=True)
    a.update_lesson(off, active=False)
    monkeypatch.setenv(feedback_sync.ENV, str(folder))
    again = Storage(tmp_path / "a.db")
    assert _state(again) == {"켤 교훈": True, "끌 교훈": False} and again.last_feedback_sync["aligned"] == 0


def test_records_without_change_time_fall_back_to_the_off_side(tmp_path, monkeypatch):
    """켜짐을 바꾼 시각이 양쪽 다 없는 옛 기록(이 칸이 생기기 전에 합친 것 등)은 꺼진 쪽을 따른다 — 끄기는 지우지 않아 되돌릴 수 있다."""
    folder = tmp_path / "knowledge"
    a = Storage(tmp_path / "a.db")
    [lid] = a.add_lessons("general", None, [Lesson(agent_id="lead", kind="improve", lesson="출처를 단다")])
    feedback_sync.export(a, folder)
    b = Storage(tmp_path / "b.db")
    assert feedback_sync.import_lessons(b, folder) == 1
    a._run_sql("UPDATE lessons SET active=0 WHERE id=?", (lid,))  # 옛 버전이 끈 것 — 시각이 없다
    feedback_sync.export(a, folder)
    assert "active_at" not in (folder / feedback_sync.LESSONS).read_text(encoding="utf-8")  # 바꾼 적 없는 줄은 예전 형식 그대로
    aligned: list = []
    assert feedback_sync.import_lessons(b, folder, aligned) == 0 and len(aligned) == 1
    assert _state(b) == {"출처를 단다": False}
