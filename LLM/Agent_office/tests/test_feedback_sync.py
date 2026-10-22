"""피드백 → knowledge/ (git 에 올릴 교훈·결정 요약) — 개인 정보 제외, 자동 동기화 조건, 다른 PC 에서 가져오기."""

import json

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
