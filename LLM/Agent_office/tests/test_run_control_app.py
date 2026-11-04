"""화면 연기 테스트 (AppTest) — ⏱ 호출 모니터·⏭ 다른 모델로·⏹ 중단·✏️ 고쳐 다시·🔀 단계마다 확인.

가짜 모델(Script)로 실제 백그라운드 실행을 돌린다. 사원의 첫 초안 호출만 '응답 없이 멈춘 모델'로 만들어 둔다.
"""

import threading
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import call_monitor
from core.step_mode import DirectionOption, DirectionOptions
from tests.fakes import Script, make_delegation, make_review

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def wait_for(fn, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("기다리던 상태가 오지 않음")


def delegation(messages):
    text = messages[-1].content
    if "gm_research" in text:
        return make_delegation("gm_research", "gm_action")
    return make_delegation("am_fact", "staff_draft")


@pytest.fixture
def app(tmp_path, monkeypatch, use_script):
    import streamlit as st

    gate = threading.Event()
    first = {"done": False}
    lock = threading.Lock()

    def text(messages):
        with lock:
            hang = not first["done"] and "이름: 사원·사례" in messages[0].content
            if hang:
                first["done"] = True
        if hang:
            gate.wait(20)  # 이 호출은 응답이 없다 (사람이 다른 모델로 바꿔야 함)
        return "초안 [1]"

    directions = DirectionOptions(options=[DirectionOption(title="A: 기술 중심", summary="기술"),
                                           DirectionOption(title="B: 성과 중심", summary="성과")])
    use_script(Script(text=text, Delegation=delegation, ReviewResult=make_review(answer="보완안"), DirectionOptions=directions))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    call_monitor.reset()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    at.session_state["office_view"] = "chart"
    yield at, gate
    gate.set()
    st.cache_resource.clear()
    call_monitor.reset()


def reopen() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    at.session_state["office_view"] = "chart"
    return at


def submit(at, question="테스트 질문", step=False):
    at.run()
    assert not at.exception, at.exception
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()  # 부서 업무 목록 → 범용 팀 자유 질문
    if step:
        at.toggle(key="step_mode_general").set_value(True)
    at.text_area(key="q_text").input(question)
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    return at


def current_run_id():
    return wait_for(lambda: next((rid for rid in list(call_monitor._calls) if rid), None))


def test_monitor_shows_hung_call_and_switch_and_stop_work(app):
    at, gate = app
    at = submit(at)
    rid = current_run_id()
    hung = wait_for(lambda: next((c for c in call_monitor.in_flight(rid) if c.agent_id == "staff_case"), None))
    at.toggle(key="mon_open_general").set_value(True)
    at = at.run()
    assert not at.exception, at.exception
    assert any(t.label == "▶ 진행 중·조작" for t in at.tabs)
    assert any("모델 호출" in c.value for c in at.caption)  # ⏱ 한 줄 요약
    at = at.button(key=f"sw_pick_{hung.id}").click().run()  # 그 호출 줄의 ⏭ 바꾸기
    assert not at.exception, at.exception
    go = at.button(key=f"sw_go_{hung.id}")
    model = at.selectbox(key=f"sw_model_{hung.id}")
    assert model.options  # 규칙을 통과한 모델이 있다
    assert any(e.label.startswith("🚫 고를 수 없는 모델") for e in at.expander)
    at = go.click().run()
    assert not at.exception, at.exception
    wait_for(lambda: hung.status == "switched")
    assert any(e["what"] == "⏭ 모델 바꿈" for e in call_monitor.interventions(rid))
    # 다른 모델로 바꾼 뒤 남은 일은 끝까지 → 결정 대기
    wait_for(lambda: not call_monitor.in_flight(rid) and any(c.kind == "review" for c in call_monitor.calls(rid)))
    at = wait_for(lambda: (lambda a: a if any("최종 결과" in m.value for m in a.markdown) else None)(at.run()))
    assert any(e.label.startswith("✏️ 질문·브리프 고쳐 다시") for e in at.expander)
    # 📜 전체 진행에 개입 기록이 남는다
    at.toggle(key="show_all_general").set_value(True)
    at = at.run()
    assert any("⏭ 모델 바꿈" in c.value for c in at.caption)


def test_stop_then_restart_with_edited_question(app):
    at, gate = app
    at = submit(at)
    rid = current_run_id()
    wait_for(lambda: next((c for c in call_monitor.in_flight(rid) if c.agent_id == "staff_case"), None))
    at.toggle(key="mon_open_general").set_value(True)
    at = at.run()
    at = at.button(key=f"stop_{rid}").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("최종 결과" in m.value for m in a.markdown) else None)(at.run()))
    assert any("⏹ 사용자가 중단" in str(m.value) for m in at.markdown) or call_monitor.interventions(rid)[-1]["what"] == "⏹ 중단"
    # ✏️ 질문 고쳐 처음부터 다시
    at.text_area(key=f"rs_q_{rid}").input("고친 질문")
    at = next(b for b in at.button if b.label == "✏️ 이대로 다시 하기").click().run()
    assert not at.exception, at.exception
    wait_for(lambda: any(c.round == 2 for c in call_monitor.calls(rid)))
    fresh = reopen()  # 새 화면 (진행 중인 업무는 서버에 하나뿐인 저장소에 있어 그대로 보인다)
    at = wait_for(lambda: (lambda a: a if any(e.label.startswith("🆚 이전 시도 1개") for e in a.expander) else None)(fresh.run()))
    assert any(e["what"] == "✏️ 고쳐 다시" for e in call_monitor.interventions(rid))


def test_step_mode_pauses_and_direction_choice_continues(app):
    at, gate = app
    gate.set()  # 멈춘 호출 없이
    at = submit(at, step=True)
    at = wait_for(lambda: (lambda a: a if any("🔀 단계 확인" in m.value for m in a.markdown) else None)(at.run()))
    radio = next(r for r in at.radio if r.label == "진행 방향")
    assert radio.options == ["A: 기술 중심 — 기술", "B: 성과 중심 — 성과", "✍️ 직접 쓰기", "방향 지정 안 함"]
    radio.set_value("B: 성과 중심")
    at = next(b for b in at.button if b.label == "▶ 이대로 진행").click().run()
    assert not at.exception, at.exception
    rid = current_run_id()
    wait_for(lambda: any("방향 선택 — B: 성과 중심" in e["detail"] for e in call_monitor.interventions(rid)))
    fresh = reopen()
    at = wait_for(lambda: (lambda a: a if any("② 팀장 업무 분배" in m.value for m in a.markdown) else None)(fresh.run()))
    assert not at.exception, at.exception
    assert [c.label for c in at.checkbox if "에게 맡기기" in c.label] == ["**부장·리서치** 에게 맡기기", "**부장·실행** 에게 맡기기"]
