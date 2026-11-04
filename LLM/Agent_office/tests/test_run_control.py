"""실행 중 개입 — ✏️ 질문·브리프 고쳐 체크포인트에서 다시, 🔀 단계마다 확인(3곳에서 멈춤), ⏹ 중단(부분 결과), ⛔ 비평 건너뛰기.

가짜 모델(Script)과 메모리 체크포인터만 쓴다. 실행 id(thread_id)는 usage.scope(run_id) 로도 묶어야 ⏱ 호출 모니터가 실행을 안다
(화면에서는 core/live.start_run 이 연다).
"""

import threading
import time

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core import call_monitor, usage
from core.brief import DesiredState
from core.live import apply_event, archive_attempt, new_run
from core.schemas import CriticResult
from core.step_mode import DirectionOption, DirectionOptions
from core.workflow import build_office_graph, restart_from, resume_step, stream_office
from tests.fakes import Script, make_critic, make_delegation, make_review

SETTINGS = {"review": {"rewrite_answer": True}}
NO_CRITIC = {"critic": {"enabled": False}}
BRIEF = DesiredState(goal="원래 목표", context="", success_criteria=["기준A"], constraints=[], output_format="", assumptions=[])


@pytest.fixture(autouse=True)
def fresh_monitor():
    call_monitor.reset()
    yield
    call_monitor.reset()


def script(**extra):
    def delegation(messages):
        text = messages[-1].content
        if "gm_research" in text:  # 팀장 차례
            return make_delegation("gm_research", "gm_action")
        return make_delegation("am_fact", "staff_draft")

    directions = DirectionOptions(options=[DirectionOption(title="A: 기술 중심", summary="기술"),
                                           DirectionOption(title="B: 성과 중심", summary="성과")])
    outputs = {"text": "초안 [1]", "Delegation": delegation, "ReviewResult": make_review(answer="보완안"),
               "DirectionOptions": directions, **extra}
    return Script(**outputs)


def graph_for(org, ranks, team, models, verification=NO_CRITIC):
    return build_office_graph(org, ranks, team, models, SETTINGS, verification, checkpointer=MemorySaver())


def drain(events, run_id):
    with usage.scope(run_id=run_id):
        return list(events)


def interrupts(events):
    return [i.value for k, p in events if k == "interrupt" for i in p]


def steps(events):
    return [p for k, p in events if k == "step"]


# ─── 기본(꺼짐)은 그대로 ─────────────────────────────────────────────


def test_default_mode_only_stops_for_human_decision(org, ranks, team, models, use_script):
    s = use_script(script())
    graph = graph_for(org, ranks, team, models)
    events = drain(stream_office(graph, org, "질문", "lead", thread_id="t0", brief=BRIEF), "t0")
    assert [v["type"] for v in interrupts(events)] == ["human_decision"]
    assert not [k for k, _ in s.calls if k == "DirectionOptions"]  # 방향 제안 호출 없음
    assert steps(events)[-1].node_id == "lead" and steps(events)[-1].kind == "integrate"
    assert graph.get_state({"configurable": {"thread_id": "t0"}}).next == ("human_decision",)


# ─── ✏️ 고쳐 다시 ─────────────────────────────────────────────────────


def test_restart_from_plan_with_edited_brief_reuses_research_and_keeps_old_attempt(org, ranks, team, models, use_script):
    s = use_script(script())
    graph = graph_for(org, ranks, team, models)
    run = new_run("t1", "질문", "lead", "general")
    for kind, payload in drain(stream_office(graph, org, "질문", "lead", thread_id="t1", brief=BRIEF), "t1"):
        apply_event(run, kind, payload)
    old_final = run["steps"][-1]
    assert run["status"] == "awaiting_human"
    attempt = archive_attempt(run, "브리프 고침")  # 화면: 예전 시도를 보관
    assert attempt["steps"][-1] is old_final and run["steps"] == []

    edited = BRIEF.model_copy(update={"goal": "새로 고친 목표"})
    n_prompts = len(s.prompts("text"))
    events = drain(restart_from(graph, org, "t1", "plan", brief=edited), "t1")
    for kind, payload in events:
        apply_event(run, kind, payload)
    new_prompts = s.prompts("text")[n_prompts:]
    assert new_prompts and all("새로 고친 목표" in p and "원래 목표" not in p for p in new_prompts)
    assert run["status"] == "awaiting_human" and run["steps"][-1].kind == "integrate"
    assert run["attempts"][0]["steps"][-1].content == old_final.content  # 예전 시도는 그대로 남아 비교 가능
    state = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert state.values["brief"].goal == "새로 고친 목표"
    assert [(x.node_id, x.kind) for x in state.values["steps"]].count(("lead", "integrate")) == 1  # 예전 단계가 섞이지 않음


def test_restart_from_brief_with_new_question(org, ranks, team, models, use_script):
    s = use_script(script())
    graph = graph_for(org, ranks, team, models)
    drain(stream_office(graph, org, "처음 질문", "lead", thread_id="t2", brief=BRIEF), "t2")
    n = len(s.prompts("text"))
    events = drain(restart_from(graph, org, "t2", "brief", question="고친 질문"), "t2")
    assert [v["type"] for v in interrupts(events)] == ["human_decision"]
    assert all("고친 질문" in p for p in s.prompts("text")[n:])
    assert graph.get_state({"configurable": {"thread_id": "t2"}}).values["question"] == "고친 질문"


# ─── 🔀 단계마다 확인 ─────────────────────────────────────────────────


def test_step_mode_pauses_three_times_and_applies_choices(org, ranks, team, models, use_script):
    s = use_script(script())
    graph = graph_for(org, ranks, team, models)
    ev = drain(stream_office(graph, org, "질문", "lead", thread_id="t3", brief=BRIEF, step_mode=True), "t3")
    (first,) = interrupts(ev)
    assert first["type"] == "step_check" and first["point"] == "brief"
    assert [o["title"] for o in first["options"]] == ["A: 기술 중심", "B: 성과 중심"]
    assert not steps(ev)  # 아직 아무도 일하지 않음

    ev = drain(resume_step(graph, org, {"direction": "B: 성과 중심"}, "t3"), "t3")
    brief = next(p for k, p in ev if k == "brief")
    assert "진행 방향(사용자 선택): B: 성과 중심" in brief.constraints
    (second,) = interrupts(ev)
    assert second["point"] == "plan"
    assert [a["child_id"] for a in second["assignments"]] == ["gm_research", "gm_action"]
    assert [(x.node_id, x.kind) for x in steps(ev)] == [("lead", "plan")]

    # 사람이 실행 부장에게만, 업무도 고쳐서 맡긴다
    ev = drain(resume_step(graph, org, {"assignments": [{"child_id": "gm_action", "subtask": "실행 계획만 짧게"}]}, "t3"), "t3")
    (third,) = interrupts(ev)
    assert third["point"] == "drafts" and [d["node_id"] for d in third["drafts"]] == ["staff_case"]
    assert {x.kind for x in steps(ev)} == {"draft"}  # 통합은 아직
    assert any("실행 계획만 짧게" in p for p in s.prompts("text"))

    ev = drain(resume_step(graph, org, {"edits": {"staff_case": "사람이 고친 초안"}, "note": "표로 정리"}, "t3"), "t3")
    assert [v["type"] for v in interrupts(ev)] == ["human_decision"]
    assert [(x.node_id, x.kind) for x in steps(ev)] == [("am_action", "review"), ("gm_action", "review"), ("lead", "review")]
    reviews = s.prompts("ReviewResult")
    assert "사람이 고친 초안" in reviews[0] and all("통합 지시(사용자): 표로 정리" in r for r in reviews)
    assert "gm_research" not in {x.node_id for x in steps(ev)}
    state = graph.get_state({"configurable": {"thread_id": "t3"}}).values
    assert [(x.node_id, x.kind) for x in state["steps"]][0] == ("lead", "plan")
    log = [e["what"] for e in call_monitor.interventions("t3")]
    assert log == ["🔀 ① 브리프·방향", "🔀 ② 팀장 업무 분배", "🔀 ③ 초안 (통합 전)"]


def test_step_mode_approve_all_gives_same_shape_as_normal_run(org, ranks, team, models, use_script):
    use_script(script())
    graph = graph_for(org, ranks, team, models)
    ev = drain(stream_office(graph, org, "질문", "lead", thread_id="t4", brief=BRIEF, step_mode=True), "t4")
    for _ in range(3):
        assert interrupts(ev)[0]["type"] == "step_check"
        ev = drain(resume_step(graph, org, {}, "t4"), "t4")
    assert [v["type"] for v in interrupts(ev)] == ["human_decision"]
    state = graph.get_state({"configurable": {"thread_id": "t4"}}).values
    assert sorted((x.node_id, x.kind) for x in state["steps"]) == sorted([
        ("lead", "plan"), ("dgm_research", "plan"), ("am_fact", "draft"), ("staff_draft", "draft"),
        ("dgm_research", "integrate"), ("gm_research", "review"), ("staff_case", "draft"), ("am_action", "review"),
        ("gm_action", "review"), ("lead", "integrate"),
    ])


# ─── ⏹ 중단 · ⛔ 건너뛰기 ─────────────────────────────────────────────


def run_in_thread(events_fn, run_id):
    box: dict = {}

    def target():
        box["events"] = drain(events_fn(), run_id)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, box


def wait_for(fn, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("기다리던 상태가 오지 않음")


def test_stop_keeps_finished_drafts_as_partial_result(org, ranks, team, models, use_script):
    gate = threading.Event()

    def text(messages):
        if "이름: 사원·사례" in messages[0].content:
            gate.wait(10)  # 사례 담당만 멈춰 있다
        return "초안 [1]"

    use_script(script(text=text))
    graph = graph_for(org, ranks, team, models)
    t, box = run_in_thread(lambda: stream_office(graph, org, "질문", "lead", thread_id="t5", brief=BRIEF), "t5")
    stuck = wait_for(lambda: next((c for c in call_monitor.in_flight("t5") if c.kind == "draft" and c.agent_id == "staff_case"), None))
    wait_for(lambda: len([c for c in call_monitor.calls("t5") if c.status == "done" and c.kind == "draft"]) >= 2)
    n = call_monitor.stop("t5")
    t.join(10)
    gate.set()
    assert n >= 1 and stuck.status == "cancelled"
    events = box["events"]
    final = steps(events)[-1]
    assert final.node_id == "lead" and "⏹ 사용자가 중단" in final.verification_log[0]
    assert "staff_case" not in {x.node_id for x in steps(events)}
    assert [v["type"] for v in interrupts(events)] == ["human_decision"]  # 부분 결과로 결정할 수 있다


def test_skip_critic_cuts_the_running_critic_call(org, ranks, team, models, use_script):
    gate = threading.Event()

    def critic(messages):
        gate.wait(10)
        return make_critic(5)

    use_script(script(CriticResult=critic))
    graph = graph_for(org, ranks, team, models, verification={"critic": {"enabled": True, "scope": "final"}})
    t, box = run_in_thread(lambda: stream_office(graph, org, "질문", "lead", thread_id="t6", brief=BRIEF), "t6")
    rec = wait_for(lambda: next((c for c in call_monitor.in_flight("t6") if c.kind == "critic"), None))
    assert call_monitor.skip("t6", "critic") == 1
    t.join(10)
    gate.set()
    assert rec.status == "cancelled"
    final = steps(box["events"])[-1]
    assert final.node_id == "lead" and final.critic is None
    assert any("비평을 끊음" in line for line in final.verification_log)
    assert call_monitor.interventions("t6")[-1]["what"] == "⛔ 건너뛰기"
    assert isinstance(make_critic(), CriticResult)
