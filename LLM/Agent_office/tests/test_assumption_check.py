"""🧭 시작 전 확인 — 요청 정리가 스스로 정한 가정을 팀이 일하기 전에 사람에게 묻는다 (core/step_mode.py · core/workflow.py).

모두 가짜 모델(Script)로 돈다. 핵심 약속 두 가지를 확인한다.
  - 멈춤은 그래프의 interrupt·체크포인트를 그대로 쓰고, 질문은 이미 나온 assumptions 라 모델 호출 수가 늘지 않는다.
  - 묻지 않는 경우(가정 없음 · 사람이 브리프를 다 채움/업무 양식 · ⚡ 빠른 답 · 화면 없는 실행)는 지금과 똑같이 흐른다.
"""

import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from streamlit.testing.v1 import AppTest

from core import call_monitor, step_mode, usage
from core.brief import DesiredState
from core.live import apply_event, new_run
from core.search import SearchPlan
from core.step_mode import DirectionOption, DirectionOptions
from core.workflow import build_office_graph, resume_step, stream_office
from tests.fakes import Script, make_critic, make_delegation, make_review

SETTINGS = {"review": {"rewrite_answer": True}}
CRITIC_ON = {"critic": {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 0}}
GUESSES = ["HBF 는 고대역폭 플래시(High Bandwidth Flash) 메모리를 뜻한다고 봤습니다", "개인 공부용 설명으로 봤습니다"]


def drafted(assumptions=GUESSES) -> DesiredState:
    return DesiredState(goal="HBF 설명", context="", success_criteria=["정의를 쓴다"], constraints=[], output_format="",
                        assumptions=list(assumptions))


def script(brief: DesiredState) -> Script:
    def delegation(messages):
        return make_delegation("gm_research", "gm_action") if "gm_research" in messages[-1].content else make_delegation("am_fact", "staff_draft")

    directions = DirectionOptions(options=[DirectionOption(title="A: 넓게", summary="넓게")])
    return Script(text="초안 [1]", Delegation=delegation, ReviewResult=make_review(answer="보완안"), CriticResult=make_critic(5),
                  DesiredState=brief, DirectionOptions=directions)


@pytest.fixture(autouse=True)
def fresh_monitor():
    call_monitor.reset()
    yield
    call_monitor.reset()


def graph_for(org, ranks, team, models, human=True):
    return build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, checkpointer=MemorySaver() if human else None)


def drain(events, run_id="t"):
    with usage.scope(run_id=run_id):
        return list(events)


def interrupts(events):
    return [i.value for k, p in events if k == "interrupt" for i in p]


def calls(s: Script) -> list[str]:
    return [k for k, _ in s.calls]


# ─── 멈추고, 답을 브리프에 넣고, 호출 수는 그대로 ───────────────────────


def test_pauses_before_team_work_with_existing_assumptions_only(org, ranks, team, models, use_script):
    s = use_script(script(drafted()))
    graph = graph_for(org, ranks, team, models)
    ev = drain(stream_office(graph, org, "HBF 가 뭐야?", "lead", thread_id="t1"), "t1")

    (check,) = interrupts(ev)
    assert check["type"] == "step_check" and check["point"] == "assumptions"
    assert check["assumptions"] == GUESSES and check["more"] == 0
    assert calls(s) == ["DesiredState"]  # 멈출 때까지 모델은 요청 정리 1번뿐 — 질문을 만드는 호출이 없다
    assert not [k for k, _ in ev if k in ("step", "research", "tools")]  # 자료 조사·분배·초안 전에 멈춤
    assert graph.get_state({"configurable": {"thread_id": "t1"}}).next == ("check_assumptions",)


def test_keep_all_gives_same_calls_and_brief_as_a_run_without_assumptions(org, ranks, team, models, use_script):
    """'모두 이대로 진행' — 가정이 있어 한 번 멈춘 실행과, 가정이 없어 멈추지 않은 실행의 모델 호출이 똑같다."""
    s = use_script(script(drafted()))
    graph = graph_for(org, ranks, team, models)
    drain(stream_office(graph, org, "HBF 가 뭐야?", "lead", thread_id="t2"), "t2")
    ev = drain(resume_step(graph, org, {"fixes": {}}, "t2"), "t2")
    assert [v["type"] for v in interrupts(ev)] == ["human_decision"]
    paused_calls = calls(s)
    assert graph.get_state({"configurable": {"thread_id": "t2"}}).values["brief"] == drafted()  # 브리프는 그대로
    assert [e["detail"] for e in call_monitor.interventions("t2")] == ["모두 이대로 진행"]

    s2 = use_script(script(drafted(assumptions=[])))
    ev2 = drain(stream_office(graph_for(org, ranks, team, models), org, "HBF 가 뭐야?", "lead", thread_id="t3"), "t3")
    assert [v["type"] for v in interrupts(ev2)] == ["human_decision"]  # 가정이 없으면 묻지 않는다
    assert sorted(paused_calls) == sorted(calls(s2))
    # 요청 정리 1 + 분배 2 + 초안 3 + 검토·통합 5 + 비평 1 = 12 (test_quick_answer 의 '팀 전체' 값과 같다)
    assert len(paused_calls) == 12 and paused_calls.count("DesiredState") == 1


def test_fixed_assumption_reaches_every_agent_as_human_decision(org, ranks, team, models, use_script):
    s = use_script(script(drafted()))
    graph = graph_for(org, ranks, team, models)
    drain(stream_office(graph, org, "HBF 가 뭐야?", "lead", thread_id="t4"), "t4")
    fix = "HBF 는 HTTP 기반 프레임워크 이름입니다"
    ev = drain(resume_step(graph, org, {"fixes": {GUESSES[0]: fix, GUESSES[1]: ""}}, "t4"), "t4")

    brief = next(p for k, p in ev if k == "brief")  # 화면의 브리프도 고친 것으로
    assert brief.assumptions == [GUESSES[1]]  # 고친 가정은 가정 목록에서 빠지고
    assert f"{step_mode.DECIDED}: {fix}" in brief.constraints  # 사람이 정한 해석으로 들어간다
    prompts = s.prompts("text") + s.prompts("Delegation") + s.prompts("ReviewResult")
    assert prompts and all(fix in p and GUESSES[0] not in p for p in prompts)  # 분배·초안·검토 모두 받는다
    (log,) = call_monitor.interventions("t4")
    assert log["what"] == step_mode.POINTS["assumptions"] and "가정 1개 고침" in log["detail"]


def test_asks_at_most_three_and_says_how_many_are_left(org, ranks, team, models, use_script):
    many = [f"가정 {i}" for i in range(5)]
    use_script(script(drafted(many)))
    ev = drain(stream_office(graph_for(org, ranks, team, models), org, "질문", "lead", thread_id="t5"), "t5")
    (check,) = interrupts(ev)
    assert check["assumptions"] == many[:3] and check["more"] == 2


def test_screen_state_shows_start_check_not_step_mode(org, ranks, team, models, use_script):
    use_script(script(drafted()))
    run = new_run("t6", "HBF 가 뭐야?", "lead", "general")
    for kind, payload in drain(stream_office(graph_for(org, ranks, team, models), org, "HBF 가 뭐야?", "lead", thread_id="t6"), "t6"):
        apply_event(run, kind, payload)
    assert run["status"] == "step_check" and run["checkpoint"]["point"] == "assumptions"
    assert step_mode.wait_label(run) == "🧭 시작 전 확인 대기"
    assert step_mode.wait_label({"checkpoint": {"point": "brief"}}) == "🔀 확인 대기"


# ─── 묻지 않는 경우 ─────────────────────────────────────────────────────


def test_no_question_when_human_filled_the_brief(org, ranks, team, models, use_script):
    """목표·완료 기준을 사람이 채움(자유 질문의 브리프 입력 · 업무 양식 · ❓ 질문 먼저 팀) → 요청 정리 모델을 안 부르고 묻지 않는다."""
    s = use_script(script(drafted()))
    mine = drafted(assumptions=["사람이 직접 적은 가정"])
    for tid, auto in (("t7", True), ("t8", False)):  # auto_brief=False = 업무 양식 (app.py — ask_first 도 이 길)
        ev = drain(stream_office(graph_for(org, ranks, team, models), org, "질문", "lead", thread_id=tid, brief=mine,
                                 auto_brief=auto), tid)
        assert [v["type"] for v in interrupts(ev)] == ["human_decision"]
    assert "DesiredState" not in calls(s)


def test_no_question_for_quick_answer_or_headless_run(org, ranks, team, models, use_script):
    s = use_script(script(drafted()))
    ev = drain(stream_office(graph_for(org, ranks, team, models), org, "질문", "lead", thread_id="t9", quick=True), "t9")
    assert [v["type"] for v in interrupts(ev)] == ["human_decision"]  # ⚡ 빠른 답 — 요청 정리 모델도 부르지 않는다
    assert "DesiredState" not in calls(s)

    ev = drain(stream_office(graph_for(org, ranks, team, models, human=False), org, "질문", "lead"), "t10")
    assert not interrupts(ev)  # 화면 없는 실행(평가·자율 실행 — 체크포인터 없음)은 끝까지 간다
    assert [p for k, p in ev if k == "step"][-1].kind == "integrate"


def test_step_mode_asks_assumptions_on_the_brief_screen_instead(org, ranks, team, models, use_script):
    use_script(script(drafted()))
    graph = graph_for(org, ranks, team, models)
    ev = drain(stream_office(graph, org, "질문", "lead", thread_id="t11", step_mode=True), "t11")
    (first,) = interrupts(ev)
    assert first["point"] == "brief" and first["brief"]["assumptions"] == GUESSES  # 따로 멈추지 않고 ① 에서 함께
    ev = drain(resume_step(graph, org, {"brief": first["brief"], "fixes": {GUESSES[1]: "회사 보고용"}}, "t11"), "t11")
    brief = next(p for k, p in ev if k == "brief")
    assert f"{step_mode.DECIDED}: 회사 보고용" in brief.constraints and brief.assumptions == [GUESSES[0]]
    assert "가정 1개 고침" in call_monitor.interventions("t11")[0]["detail"]


# ─── 화면 (자유 질문 → 🧭 확인 → 고친 내용으로 진행) ───────────────────


def test_free_question_screen_asks_and_continues(tmp_path, monkeypatch, use_script):
    import streamlit as st

    def wait_for(fn, timeout=60.0):
        end = time.time() + timeout
        while time.time() < end:
            value = fn()
            if value:
                return value
            time.sleep(0.05)
        raise AssertionError("기다리던 상태가 오지 않음")

    s = script(drafted())
    s.outputs["SearchPlan"] = SearchPlan(need_search=False, queries=[])
    use_script(s)
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.query_params["team"] = "general"
        at.query_params["view"] = "team"
        at.session_state["office_view"] = "chart"
        at.run()
        assert not at.exception, at.exception
        at = at.selectbox(key="dept_task_research").set_value("general:free").run()
        at.text_area(key="q_text").input("HBF 가 뭐야?")
        at = next(b for b in at.button if b.label == "맡기기").click().run()
        assert not at.exception, at.exception
        at = wait_for(lambda: (lambda a: a if any("시작 전 확인" in m.value for m in a.markdown) else None)(at.run()))
        assert not at.exception, at.exception
        assert any(GUESSES[0] in m.value for m in at.markdown)
        assert any("모델을 더 부르지 않습니다" in c.value for c in at.caption)
        assert {b.label for b in at.button} >= {"▶ 모두 이대로 진행", "✏️ 고친 내용으로 진행"}
        at.text_input[[t.label for t in at.text_input].index("다르면 고쳐 쓰기 (비워 두면 이대로 진행)")].input("HTTP 프레임워크입니다")
        at = next(b for b in at.button if b.label == "✏️ 고친 내용으로 진행").click().run()
        assert not at.exception, at.exception
        rid = wait_for(lambda: next((r for r in list(call_monitor._calls) if r), None))
        wait_for(lambda: any("가정 1개 고침" in e["detail"] for e in call_monitor.interventions(rid)))
        wait_for(lambda: any("HTTP 프레임워크입니다" in p for p in s.prompts("text")))
        assert calls(s).count("DesiredState") == 1
    finally:
        st.cache_resource.clear()
        call_monitor.reset()

