"""사람 채택(interrupt/resume) · 교훈 생성·저장·주입 · 저장 정책(미결정 삭제) 테스트."""

import sqlite3
from datetime import datetime, timedelta

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core.config import parent_map
from core.feedback import generate_lessons, rule_based_lessons
from core.schemas import AgentAnswer, HumanDecision, Lesson, LessonSet, ModelSpec
from core.storage import Storage
from core.workflow import build_office_graph, make_checkpointer, resume_with_decision, stream_office
from tests.fakes import Script, make_review
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS

SPEC = ModelSpec(provider="openai", model="x")
NAMES = {"gm_action": "부장·실행", "am_action": "대리·실행안", "staff_case": "사원·사례"}


def step(node_id, kind, content, review=None):
    return AgentAnswer(node_id=node_id, rank_id="staff", kind=kind, content=content, review=review, model=SPEC)


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "runs.db")


CHAIN_STEPS = [
    step("staff_case", "draft", "초안"),
    step("am_action", "review", "보완안", make_review("revise", 6)),
    step("gm_action", "review", "과한 수정안", make_review("revise", 8)),
]


# ─── 규칙 기반 교훈 ───────────────────────────────────────────────────


def test_rule_based_lessons_follow_tree(org):
    decision = HumanDecision(action="adopt", adopted_node_id="am_action", final_answer="보완안", score=5)
    lessons = {l.agent_id: l for l in rule_based_lessons(CHAIN_STEPS, decision, NAMES, parent_map(org))}
    assert lessons["staff_case"].kind == "improve" and "근거 부족" in lessons["staff_case"].lesson  # 상위가 보완함
    assert lessons["am_action"].kind == "praise"  # 채택됨
    assert lessons["gm_action"].kind == "improve" and "대리·실행안" in lessons["gm_action"].lesson  # 과한 수정


def test_rule_based_lessons_on_reject_include_comment(org):
    lessons = rule_based_lessons(CHAIN_STEPS[:1], HumanDecision(action="reject", comment="출처 없음"), NAMES, parent_map(org))
    assert lessons[0].kind == "improve" and "출처 없음" in lessons[0].lesson


# ─── LLM 교훈 + 누락 보충 + 사람 코멘트 보장 ─────────────────────────────


def test_generate_lessons_fills_missing_and_keeps_comment(org, models, team, use_script):
    use_script(Script(LessonSet=LessonSet(lessons=[
        Lesson(agent_id="staff_case", kind="praise", lesson="잘했다"),
        Lesson(agent_id="unknown", kind="praise", lesson="참여하지 않은 에이전트는 무시"),
    ])))
    decision = HumanDecision(action="adopt", adopted_node_id="staff_case", final_answer="수정본", edited=True, comment="예산 빠짐")
    lessons, notes = generate_lessons("질문", CHAIN_STEPS, decision, NAMES, parent_map(org), team, models)

    assert [l.agent_id for l in lessons] == ["staff_case", "am_action", "gm_action"]
    assert lessons[0].kind == "improve" and "예산 빠짐" in lessons[0].lesson  # 사람 코멘트가 반드시 전달됨
    assert any("빠뜨린" in n for n in notes)


def test_generate_lessons_falls_back_when_model_fails(org, models, team, use_script):
    use_script(Script())  # LessonSet 출력 없음 → 호출 실패
    lessons, notes = generate_lessons("질문", CHAIN_STEPS[:1], HumanDecision(action="reject"), NAMES, parent_map(org), team, models)
    assert lessons[0].kind == "improve" and any("규칙 기반" in n for n in notes)


# ─── 저장소: 결정한 것만 보관, 미결정은 삭제 ───────────────────────────


def test_storage_recent_lessons_respects_limit_and_active(storage):
    storage.add_lessons("general", "r1", [Lesson(agent_id="lead", kind="improve", lesson=f"교훈{i}") for i in range(3)])
    assert storage.recent_lessons("general", "lead", 2) == ["교훈1", "교훈2"]
    newest = storage.list_lessons("general", "lead")[0]
    storage.update_lesson(newest["id"], active=False)
    assert storage.recent_lessons("general", "lead", 2) == ["교훈0", "교훈1"]
    assert storage.lesson_counts("general") == {"lead": 2}


def test_purge_deletes_only_old_undecided_runs(storage):
    for run_id in ("decided", "old_pending", "new_pending"):
        storage.create_run(run_id, "general", "lead", "질문")
    storage.save_decision("decided", HumanDecision(action="reject"))
    old = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
    storage._execute("UPDATE runs SET created_at=? WHERE id IN ('decided','old_pending')", (old,))

    assert storage.purge_undecided(older_than_hours=1) == ["old_pending"]
    assert storage.get_run("decided")["status"] == "rejected"  # 피드백 있는 결과는 보관
    assert storage.get_run("new_pending") is not None  # 방금 만든 것은 (화면에서 결정 중일 수 있어) 보관
    storage.delete_run("new_pending")
    assert storage.get_run("new_pending") is None


def test_migrates_old_rank_based_db(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE runs (id TEXT, team_id TEXT, start_rank TEXT, question TEXT, chain TEXT, steps TEXT, status TEXT,"
        " decision TEXT, created_at TEXT, decided_at TEXT);"
        "CREATE TABLE lessons (id INTEGER PRIMARY KEY, team_id TEXT, rank_id TEXT, run_id TEXT, kind TEXT, lesson TEXT,"
        " active INTEGER DEFAULT 1, created_at TEXT);"
        "INSERT INTO lessons (team_id, rank_id, kind, lesson, created_at) VALUES ('general','staff','praise','옛 교훈','x');"
    )
    conn.commit()
    conn.close()
    storage = Storage(path)
    assert storage.recent_lessons("general", "staff", 5) == ["옛 교훈"]  # 교훈 보존
    storage.create_run("new", "general", "lead", "질문")  # 새 runs 테이블 동작


# ─── 워크플로우: interrupt → resume → 교훈 저장 → 다음 실행에 주입 ─────────


@pytest.mark.parametrize("checkpointer_kind", ["memory", "sqlite"])
def test_human_decision_resume_saves_and_injects_lessons(org, ranks, team, models, storage, tmp_path, use_script, checkpointer_kind):
    script = use_script(Script(text="초안", ReviewResult=make_review(answer="보완안"), LessonSet=LessonSet(lessons=[
        Lesson(agent_id="staff_case", kind="improve", lesson="사원 교훈"),
        Lesson(agent_id="am_action", kind="praise", lesson="대리 교훈"),
    ])))
    checkpointer = MemorySaver() if checkpointer_kind == "memory" else make_checkpointer(tmp_path / "cp.db")
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC, storage=storage, checkpointer=checkpointer)

    events = list(stream_office(graph, org, "질문", "am_action", thread_id="run-1", brief=BRIEF))
    assert [k for k, _ in events][-1] == "interrupt"
    assert graph.get_state({"configurable": {"thread_id": "run-1"}}).next == ("human_decision",)

    decision = HumanDecision(action="adopt", adopted_node_id="am_action", final_answer="보완안", score=4)
    lessons = next(p for k, p in resume_with_decision(graph, org, decision, "run-1") if k == "lessons")
    assert [l.lesson for l in lessons] == ["사원 교훈", "대리 교훈"]
    assert storage.recent_lessons(team.id, "staff_case", 5) == ["사원 교훈"]

    # 다음 실행: 저장된 교훈이 해당 에이전트의 프롬프트에 주입된다
    script.calls.clear()
    list(stream_office(graph, org, "다음 질문", "am_action", thread_id="run-2", brief=BRIEF))
    assert any("사원 교훈" in p for p in script.prompts("text"))
    assert any("대리 교훈" in p for p in script.prompts("ReviewResult"))
