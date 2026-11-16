"""⚡ 빠른 답 (팀장 혼자) — 팀원 분배·초안·중간 검토·부서장 비평 없이 팀장이 근거 자료를 받아 한 번 답한다.

하네스(코드 검사)는 그대로 돌고, 결정·교훈은 지금처럼 남는다. '👥 팀 전체로 다시'는 같은 체크포인트에서 지금 방식으로.
모델 호출 수는 가짜 모델(Script)의 호출 기록으로 센다.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core import live, progress
from core.config import load_verification
from core.schemas import HumanDecision, Lesson, LessonSet, Team
from core.storage import Storage
from core.workflow import build_office_graph, restart_from, resume_with_decision, stream_office
from tests.fakes import Script, make_critic, make_delegation, make_review

SETTINGS = {"review": {"rewrite_answer": True}}
CRITIC_ON = {"critic": {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 1}}


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "runs.db")


def full_script(**extra):
    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    return Script(text="답변 [1]", Delegation=delegation, ReviewResult=make_review(answer="보완안"),
                  CriticResult=make_critic(5), **extra)


def model_calls(script) -> list[str]:
    return [k for k, _ in script.calls]


def test_quick_answer_is_one_lead_call(org, ranks, team, models, use_script):
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    events = list(stream_office(graph, org, "HBF 가 뭐야?", "lead", quick=True))
    steps = [p for k, p in events if k == "step"]

    assert model_calls(script) == ["text"]  # 요청 정리·분배·초안·검토·비평 없이 팀장 1번
    assert [(s.node_id, s.kind) for s in steps] == [("lead", "draft")]
    assert steps[0].critic is None
    assert any("부서장 검토 없이" in line for line in steps[0].verification_log)
    starts = {p["node"] for k, p in events if k == "start"}
    assert starts == {"lead"}  # 팀원은 일하지 않는다 (픽셀 사무실·조직도)
    prompt = script.prompts("text")[0]
    assert "HBF 가 뭐야?" in prompt and "혼자" in prompt  # 혼자 최종 답변을 쓴다는 안내
    assert "상사가 맡긴 업무" not in prompt


def test_full_team_call_count_for_comparison(org, ranks, team, models, use_script):
    """같은 질문을 지금 방식으로 — 호출 수 비교의 '전' 값 (범용 팀·팀장 문의, 비평 켬, 브리프 AI 정리)."""
    from core.brief import DesiredState
    drafted = DesiredState(goal="목표", context="", success_criteria=["기준"], constraints=[], output_format="", assumptions=[])
    script = use_script(full_script(DesiredState=drafted))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    list(stream_office(graph, org, "HBF 가 뭐야?", "lead"))
    calls = model_calls(script)
    # 요청 정리 1 + 분배 2(팀장·차장) + 초안 3 + 검토·통합 5 + 비평 1 = 12
    assert len(calls) == 12 and calls.count("CriticResult") == 1


def test_quick_runs_harness_including_final_only_and_retries(org, ranks, team, models, use_script):
    bad, good = "```python\ndef f(:\n```", "```python\ndef f():\n    return 1\n```"
    script = use_script(full_script())
    script.outputs["text"] = [bad, good]
    harness_team = Team(**{**team.model_dump(), "harness": {"validators": ["python_syntax"], "final_only": ["python_syntax"],
                                                           "max_retries": 2}})
    graph = build_office_graph(org, ranks, harness_team, models, SETTINGS, CRITIC_ON)
    events = list(stream_office(graph, org, "코드", "lead", quick=True))
    answer = [p for k, p in events if k == "step"][-1]

    assert answer.content == good and answer.attempts == 2 and answer.harness[0].passed
    assert ("retry", "harness") in [(k, p["reason"]) for k, p in events if k == "retry"]
    assert "문법 오류" in script.prompts("text")[-1]
    assert model_calls(script) == ["text", "text"]  # 하네스 재작성 1번 외에는 모델을 부르지 않는다


def test_quick_web_search_uses_the_question_without_a_model(org, ranks, team, models, use_script, no_network_search):
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    events = list(stream_office(graph, org, "HBF 가 뭐야?", "lead", quick=True, use_web=True))

    assert "SearchPlan" not in model_calls(script)  # 검색어를 모델로 만들지 않는다
    assert no_network_search == ["HBF 가 뭐야?"]
    web = next(p for k, p in events if k == "research")
    assert web.results and "HBF 가 뭐야? 결과" in script.prompts("text")[0]  # 웹 자료가 팀장 프롬프트에 들어간다


def test_quick_ignores_step_mode(org, ranks, team, models, use_script):
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, checkpointer=MemorySaver())
    events = list(stream_office(graph, org, "질문", "lead", thread_id="t", quick=True, step_mode=True))
    assert model_calls(script) == ["text"]
    assert graph.get_state({"configurable": {"thread_id": "t"}}).next == ("human_decision",)
    assert [k for k, _ in events][-1] == "interrupt"


def test_full_team_restart_from_a_quick_answer(org, ranks, team, models, use_script):
    """👥 팀 전체로 다시 — 같은 체크포인트에서 quick=False 로 처음부터 (요청 정리·분배·초안·검토·비평)."""
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, checkpointer=MemorySaver())
    list(stream_office(graph, org, "질문", "lead", thread_id="t", quick=True, auto_brief=False))
    script.calls.clear()

    again = list(restart_from(graph, org, "t", "brief", quick=False))
    steps = [p for k, p in again if k == "step"]
    assert steps[-1].node_id == "lead" and steps[-1].kind == "integrate" and steps[-1].critic is not None
    assert {"Delegation", "ReviewResult", "CriticResult"} <= set(model_calls(script))
    state = graph.get_state({"configurable": {"thread_id": "t"}})
    assert state.next == ("human_decision",) and state.values["quick"] is False and state.values["question"] == "질문"


def _decide(graph, org, decision):
    return list(resume_with_decision(graph, org, decision, "t"))


def test_quick_plain_adopt_makes_the_lesson_without_a_model(org, ranks, team, models, storage, use_script):
    script = use_script(full_script(LessonSet=LessonSet(lessons=[Lesson(agent_id="lead", kind="praise", lesson="모델 교훈")])))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, storage=storage, checkpointer=MemorySaver())
    list(stream_office(graph, org, "질문", "lead", thread_id="t", quick=True))
    script.calls.clear()

    events = _decide(graph, org, HumanDecision(action="adopt", adopted_node_id="lead", final_answer="답변 [1]", score=5))
    lessons = next(p for k, p in events if k == "lessons")
    assert model_calls(script) == []  # 고친 곳·코멘트 없는 채택 → 교훈은 규칙으로 (모델 0번)
    assert [(l.agent_id, l.kind) for l in lessons] == [("lead", "praise")]
    assert storage.recent_lessons(team.id, "lead", 5)  # 교훈은 팀장에게 저장된다


def test_quick_adopt_with_comment_still_asks_the_lesson_model(org, ranks, team, models, storage, use_script):
    script = use_script(full_script(LessonSet=LessonSet(lessons=[Lesson(agent_id="lead", kind="improve", lesson="출처를 단다")])))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, storage=storage, checkpointer=MemorySaver())
    list(stream_office(graph, org, "질문", "lead", thread_id="t", quick=True))
    script.calls.clear()

    events = _decide(graph, org, HumanDecision(action="adopt", adopted_node_id="lead", final_answer="답변 [1]", score=3,
                                               comment="출처가 없다"))
    lessons = next(p for k, p in events if k == "lessons")
    assert model_calls(script) == ["LessonSet"]  # 사람 코멘트를 다음 업무에 통하는 교훈으로 바꾸는 일은 모델이 한다
    assert "출처를 단다" in lessons[0].lesson and "출처가 없다" in lessons[0].lesson


def test_quick_plain_adopt_merges_repeated_praise_by_code(org, ranks, team, models, storage, use_script):
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON, storage=storage, checkpointer=MemorySaver())
    for tid in ("t1", "t2"):
        list(stream_office(graph, org, "질문", "lead", thread_id=tid, quick=True))
        list(resume_with_decision(graph, org, HumanDecision(action="adopt", adopted_node_id="lead", final_answer="x", score=5), tid))
    assert "LessonGroupSet" not in model_calls(script)  # 겹친 교훈 찾기도 모델 없이 (문장 비교)
    assert len(storage.recent_lessons(team.id, "lead", 5)) == 1  # 같은 칭찬 교훈은 하나로 합쳐진다


# ─── 화면 상태: 진행률 · 참여자 · 진행 단계 ───────────────────────────


def test_progress_counts_one_lead_answer(org):
    plan = progress.Plan(org=org, root_id="lead", node_models={n.id: "openai:m" for n in org.walk()}, critic_model="openai:c",
                         critic_enabled=True, harness=True, max_retries=2, brief=False, quick=True)
    run = {"steps": [], "assigned": {}}
    units, worst = progress.build_units(plan, run, [])
    assert [(u.kind, u.node) for u in units] == [("draft", "lead")]
    assert worst == 2  # 하네스 재작성 최대 2번만 더해질 수 있다 (비평·교정 없음)
    est = progress.estimate(plan, run, [])
    assert est.total == 1 and est.remaining_sec == progress.DEFAULT_SEC["draft"]


def test_quick_run_members_are_only_the_lead(org):
    assert live.run_members(org, "lead", {"quick": True}) == {"lead"}
    assert live.run_members(org, "lead", None) == {n.id for n in org.walk()}
    assert live.run_members(org, "gm_action", {"quick": False}) == {"gm_action", "am_action", "staff_case"}


def test_quick_stages_skip_distribution_and_review():
    skipped = live.skipped_stages({"quick": True})
    assert {live.STAGES[i] for i in skipped} == {"📋 업무 분배", "🔍 검토·통합"}
    assert live.skipped_stages({"quick": False}) == set()


def test_quick_hint_for_short_free_questions():
    assert live.quick_hint("HBF 가 뭐야?")
    assert not live.quick_hint("")
    assert not live.quick_hint("가" * 200)  # 긴 질문은 팀 전체가 낫다
    assert not live.quick_hint("첫 줄\n둘째 줄\n셋째 줄\n넷째 줄")


def test_real_config_counts(org, ranks, team, models, use_script):
    """실제 설정(config/verification.yaml)으로도 빠른 답은 1번이다 — 비평 설정이 켜져 있어도."""
    script = use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, load_verification())
    list(stream_office(graph, org, "질문", "lead", quick=True))
    assert model_calls(script) == ["text"]


# ─── 화면 (AppTest) — 자유 질문에서 ⚡ 빠른 답 → 안내 → 👥 팀 전체로 다시 ─────────────


def test_app_quick_answer_then_full_team(tmp_path, monkeypatch, use_script):
    import time
    from pathlib import Path

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from core import call_monitor
    from core.brief import DesiredState
    from core.search import SearchPlan

    def wait_for(fn, timeout=90.0):
        end = time.time() + timeout
        while time.time() < end:
            value = fn()
            if value:
                return value
            time.sleep(0.05)
        raise AssertionError("기다리던 상태가 오지 않음")

    drafted = DesiredState(goal="목표", context="", success_criteria=["기준"], constraints=[], output_format="", assumptions=[])
    script = use_script(full_script(DesiredState=drafted, SearchPlan=SearchPlan(need_search=False, queries=[])))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    call_monitor.reset()
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.query_params["team"] = "general"
        at.query_params["view"] = "team"
        at.session_state["office_view"] = "chart"
        at.run()
        assert not at.exception, at.exception
        at = at.selectbox(key="dept_task_research").set_value("general:free").run()
        assert any("빠른 답을 권합니다" in c.value for c in at.caption)  # 자유 질문 — 권유 한 줄 (모델 호출 없음)
        at.toggle(key="quick_general").set_value(True)
        at.text_area(key="q_text").input("HBF 가 뭐야?")
        at = next(b for b in at.button if b.label == "맡기기").click().run()
        assert not at.exception, at.exception

        def finished(a):
            a = a.run()
            return a if any(b.label == "👥 팀 전체로 다시" for b in a.button) else None

        at = wait_for(lambda: finished(at))
        assert any("부서장 검토 없이 팀장이 바로 답했습니다" in i.value for i in at.info)
        assert [k for k, _ in script.calls] == ["text"]  # 팀장 1번
        assert any("(건너뜀)" in m.value for m in at.markdown)  # 진행 단계 줄 — 분배·검토·통합 건너뜀

        at = next(b for b in at.button if b.label == "👥 팀 전체로 다시").click().run()
        assert not at.exception, at.exception

        def full_done(a):
            a = a.run()
            return a if any(e.label.startswith("🆚 이전 시도") for e in a.expander) and not any(
                b.label == "👥 팀 전체로 다시" for b in a.button) and "CriticResult" in [k for k, _ in script.calls] else None

        at = wait_for(lambda: full_done(at))
        kinds = [k for k, _ in script.calls]
        assert {"Delegation", "ReviewResult", "CriticResult"} <= set(kinds)  # 지금 방식 — 분배·검토·부서장 비평
        assert not any("부서장 검토 없이" in i.value for i in at.info)
    finally:
        st.cache_resource.clear()
        call_monitor.reset()


def test_app_head_desk_hands_quick_to_the_assigned_team(tmp_path, monkeypatch, use_script):
    """👔 부서장 창구(🛎 접수 창구에서 넘어오는 곳)의 ⚡ 빠른 답 → 배정된 팀의 팀장만 일한다 (픽셀 사무실도 팀장만)."""
    import json
    from pathlib import Path

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from core import building

    use_script(Script(TeamPick=building.TeamPick(analysis="해외", team_id="travel_abroad", reason="비자가 필요한 해외 여행")))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    started = []
    monkeypatch.setattr(live, "start_run", lambda run, *a, **k: started.append((run["team_id"], run["quick"], run["step_mode"])))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=90)
        at.query_params["view"] = "dept"
        at.query_params["dept"] = "travel"
        at.session_state["office_view"] = "office"
        at.run()
        assert not at.exception, at.exception
        at.toggle(key="quick_dept_travel").set_value(True)
        at.toggle(key="step_mode_dept_travel").set_value(True)  # 같이 켜도 빠른 답이 이긴다
        at.text_area(key="dept_head_text").input("일본 오사카 3박 4일 비자 필요해?").run()
        at = at.button(key="dept_head_go").click().run()
        at = at.button(key="dept_head_start").click().run()
        assert not at.exception, at.exception
        assert started == [("travel_abroad", True, False)]
        comp = next(e for e in at.get("component_instance") if "pixel_office" in e.proto.component_name)
        agents = {a["id"]: a for a in json.loads(comp.proto.json_args)["snapshot"]["agents"] if a["team"] == "🌏 해외 여행팀"}
        working = [k for k, a in agents.items() if not a["dim"]]
        assert len(agents) > 1 and len(working) == 1 and agents[working[0]]["lead"]  # 팀장만 이번 업무
    finally:
        st.cache_resource.clear()
