"""📊 진행률·남은 시간 · 💬 백그라운드 추가 질문 · claude_cli 고정 목록 · 부장 기본 모델 (2026-11-05).

가짜 모델(Script)만 쓴다. 스레드 테스트는 시간 대신 Event 로 기다린다 (CI 부하에 민감하지 않게).
"""

import threading
import time
from pathlib import Path

import pytest

from core import call_monitor, dept_heads, live, member_models, model_catalog, progress
from core.schemas import AgentAnswer, ModelSpec, RankModels
from core.storage import Storage
from tests.fakes import Script

LEAD = "claude_cli:claude-opus-5-5"


def wait_for(fn, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("기다리던 상태가 오지 않음")


def plan_for(org, **kw) -> progress.Plan:
    base = dict(org=org, root_id=org.id, node_models={n.id: "openai:gpt-5.4" for n in org.walk()},
                critic_model=LEAD, utility_model="google:gemini-3.5-flash", critic_enabled=True, critic_scope="final",
                max_self_corrections=1, harness=True, max_retries=2, brief=True, web=False)
    base.update(kw)
    return progress.Plan(**base)


def rec(kind, agent="", status="done", start=0.0, end=10.0, model="openai:gpt-5.4", run_id="r1"):
    provider, _, name = model.partition(":")
    r = call_monitor.CallRecord(id=f"x{kind}{agent}{start}", run_id=run_id, who=agent, kind=kind, provider=provider,
                                model=name, agent_id=agent, queued_at=start, started_at=start)
    r.status = status
    r.ended_at = end if status == "done" else None
    return r


def step(node, kind):
    return AgentAnswer(node_id=node, rank_id="staff", kind=kind, content="x", model=ModelSpec(provider="openai", model="gpt-5.4"))


# ─── 📊 단계 세기 ─────────────────────────────────────────────────────


def test_expected_steps_follow_org_chart(org):
    """범용 팀 전체: 요청 정리 1 + 분배 2(팀장·차장) + 초안 3 + 검토·통합 5 + 비평 1 = 12."""
    run = live.new_run("r1", "질문", org.id, "general")
    units, extra = progress.build_units(plan_for(org), run, [])
    kinds = [u.kind for u in units]
    assert kinds.count("plan") == 2 and kinds.count("draft") == 3 and kinds.count("review") == 5
    assert kinds.count("critic") == 1 and kinds.count("brief") == 1 and len(units) == 12
    assert extra == 2 * 1 + 8 * 2  # 비평 반려 1번(교정+비평) + 하네스 재시도 최대 2번 × 8명


def test_delegation_recounts_to_people_who_got_work(org):
    run = live.new_run("r1", "질문", org.id, "general")
    live.apply_event(run, "brief", object())
    live.apply_event(run, "delegated", {"node": org.id, "children": ["gm_action"]})  # 팀장이 실행 쪽에만 맡김
    est = progress.estimate(plan_for(org), run, [])
    assert run["assigned"] == {org.id: ["gm_action"]}
    # 요청 정리 + 분배(끝남) + 실행 부장 검토 + 대리 검토 + 사원 초안 + 팀장 통합 + 비평 = 7
    assert est.total == 7 and est.done == 2 and est.percent == 28 and est.remaining == 5


def test_critic_rejection_adds_steps_and_progress_reaches_100(org):
    plan = plan_for(org)
    run = live.new_run("r1", "질문", org.id, "general")
    run["brief"] = object()
    before = progress.estimate(plan, run, []).total
    recs = [rec("critic", org.id, end=5), rec("self_correct", org.id, status="running", start=6)]
    est = progress.estimate(plan, run, recs, now=20)
    assert est.total == before + 2  # 자가 교정 1 + 다시 비평 1
    running = [u for u in est.units if u.running_sec is not None]
    assert [u.kind for u in running] == ["self_correct"]
    # 모든 단계가 끝나면 100%
    run["steps"] = [step(n.id, "draft" if not n.children else "review") for n in org.walk()]
    run["assigned"] = {n.id: [c.id for c in n.children] for n in org.walk() if n.children}
    assert progress.estimate(plan, run, recs).percent == 100


# ─── 📊 남은 시간 ────────────────────────────────────────────────────


def test_eta_uses_history_median_and_says_when_it_is_a_guess(org):
    plan = plan_for(org, critic_enabled=False, harness=False, brief=False)
    run = live.new_run("r1", "질문", org.id, "general")
    none = progress.estimate(plan, run, [])
    assert none.basis.startswith("기록 부족 — 대략") and "(기본값)" in none.basis
    past = [rec(k, end=s).to_dict() for k, s in (("draft", 30), ("draft", 50), ("draft", 40), ("plan", 10), ("plan", 10),
                                                 ("review", 20), ("review", 20))]
    est = progress.estimate(plan, run, [], past)
    assert est.basis.startswith("같은 팀 지난 기록 7개")
    draft = next(u for u in est.units if u.kind == "draft")
    assert draft.est_sec == 40 and draft.basis == "history"  # 중앙값
    # 형제는 동시에 — 가장 긴 줄기: 분배10 + [연구: 검토20 + (분배10 + 초안40 + 검토20) + …] 처럼 더한다 (합보다 짧다)
    assert est.remaining_sec < sum(u.est_sec for u in est.units)
    assert "분" in est.short() and est.short().endswith("(추정)")


def test_same_provider_siblings_queue_by_concurrency(org):
    run = live.new_run("r1", "질문", "dgm_research", "general")
    run["assigned"] = {"dgm_research": ["am_fact", "staff_draft"]}
    past = [rec("draft", end=60).to_dict(), rec("draft", end=60).to_dict()]
    free = progress.estimate(plan_for(org, root_id="dgm_research", critic_enabled=False, harness=False, brief=False,
                                      concurrency={"openai": 4}), run, [], past)
    one = progress.estimate(plan_for(org, root_id="dgm_research", critic_enabled=False, harness=False, brief=False,
                                     concurrency={"openai": 1}), run, [], past)
    assert one.remaining_sec - free.remaining_sec == pytest.approx(60, abs=1)  # 동시 1개면 초안 둘이 줄을 선다


def test_eta_text():
    assert progress.eta_text(0) == "곧 끝남"
    assert progress.eta_text(42) == "약 40초 남음"
    assert progress.eta_text(185) == "약 3분 남음"


def test_storage_monitor_history_and_status(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    storage.create_run("a", "general", "lead", "q")
    storage.save_monitor("a", {"calls": [rec("draft", end=12).to_dict()], "interventions": []})
    storage.create_run("b", "stock", "lead", "q")
    storage.save_monitor("b", {"calls": [rec("draft", end=99).to_dict()], "interventions": []})
    hist = storage.monitor_history("general")
    assert len(hist) == 1 and hist[0]["ended_at"] == 12
    storage.set_status("a", live.FOLLOWUP_RUNNING)
    assert storage.get_run("a")["status"] == live.FOLLOWUP_RUNNING


# ─── 💬 백그라운드 추가 질문 ────────────────────────────────────────────


def waiting_run(org):
    run = live.new_run("r-fu", "질문", org.id, "general")
    run.update(status="awaiting_human", done=True)
    run["statuses"][org.id] = ("wait", "👤 결정 대기")
    run["activity"].clear()
    return run


def test_followup_runs_in_background_and_returns_to_decision(org):
    run = waiting_run(org)
    gate, entered = threading.Event(), threading.Event()
    finished = []

    def work():
        entered.set()
        gate.wait(20)
        return "답변"

    thread = live.start_followup(run, "3번 근거?", work, on_finish=finished.append)
    assert entered.wait(20)
    # 진행 중: 결정 대기가 아니라 '추가 질문 답하는 중' — 답하는 사람(팀장)이 일하는 모습
    assert run["status"] == live.FOLLOWUP_RUNNING and not run["done"]
    assert run["statuses"][org.id] == ("working", live.FOLLOWUP_LABEL)
    assert run["activity"][org.id] == live.FOLLOWUP_LABEL
    assert live.stage_of(run) == len(live.STAGES) - 1
    assert run["followup_pending"]["request"] == "3번 근거?"
    gate.set()
    thread.join(20)
    assert run["done"] and run["status"] == "awaiting_human" and run["followups"] == ["답변"]
    assert run["statuses"][org.id] == ("wait", "👤 결정 대기") and org.id not in run["activity"]
    assert "followup_pending" not in run and finished == [run] and run["notice"] == "followup"


def test_followup_call_is_monitored_and_can_be_stopped(org, ranks, team, models, use_script):
    from core import followup
    from core.agent import OrgAgent

    call_monitor.reset()
    gate = threading.Event()

    def slow(messages):
        gate.wait(20)
        return followup.FollowupResult(analysis="질문", revised=False, reply="늦은 답")

    use_script(Script(FollowupResult=slow))
    rank = next(r for r in ranks if r.id == org.rank)
    agent = OrgAgent(org, rank, team, models)
    run = waiting_run(org)
    thread = live.start_followup(run, "근거?", lambda: followup.respond(agent, "q", "결과", "근거?", []))
    try:
        call = wait_for(lambda: next(iter(call_monitor.in_flight(run["run_id"])), None))
        assert call.kind == "followup" and call.agent_id == org.id  # ⏱ 호출 모니터에 실시간으로 뜬다
        # 📊 진행률도 추가 질문 하나로 계산
        est = progress.estimate(plan_for(org), run, call_monitor.calls(run["run_id"]))
        assert est.phase == "followup" and est.total == 1 and est.done == 0
        assert call_monitor.stop(run["run_id"], "💬 추가 질문 중단") == 1
        thread.join(20)
    finally:
        gate.set()
    assert run["done"] and run["status"] == "awaiting_human"
    assert run["followup_error"].startswith("⏹ 추가 질문을 중단했습니다") and not run.get("followups")
    assert not call_monitor.control(run["run_id"]).stopped  # 중단은 이번 질문에만 — 결정은 정상으로
    call_monitor.reset()


def test_head_review_call_is_monitored_and_can_be_cut(org, ranks, team, models, use_script):
    """👔 부장 검토도 ⏱ 호출 모니터·📊 진행률에 잡히고, 끊으면 수정본은 그대로 낸다."""
    from core import followup
    from core.agent import OrgAgent
    from tests.test_followup import edit, head_review, judge

    call_monitor.reset()
    gate = threading.Event()

    def slow(messages):
        gate.wait(20)
        return judge(5)

    use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=slow))
    agent = OrgAgent(org, next(r for r in ranks if r.id == org.rank), team, models)
    run = waiting_run(org)
    thread = live.start_followup(run, "줄여줘", lambda: followup.respond(agent, "q", "결과", "줄여줘", [],
                                                                      head=head_review(org, team, models)))
    try:
        call = wait_for(lambda: next((c for c in call_monitor.in_flight(run["run_id"]) if c.kind == "critic"), None))
        assert "연구부 부장" in call.who
        est = progress.estimate(plan_for(org), run, call_monitor.calls(run["run_id"]))
        assert est.phase == "followup" and est.total == 2 and est.done == 1  # 팀장 답 1 + 부장 검토 1
        assert call_monitor.skip(run["run_id"], "critic") == 1  # ⛔ 부장 검토 건너뛰기
        thread.join(20)
    finally:
        gate.set()
    turn = run["followups"][0]
    assert run["status"] == "awaiting_human" and not run["followup_error"]
    assert turn.revised and turn.content == "수정본" and turn.critic is None and any("끊음" in x for x in turn.log)
    assert [c.kind for c in call_monitor.calls(run["run_id"])] == ["followup", "critic"]  # ⏱ 호출 기록에 둘 다 남는다
    call_monitor.reset()


def test_interrupted_followup_is_recovered(org):
    run = waiting_run(org)
    live.begin_followup(run, "질문")
    assert live.recover_followup(run)  # 답하던 스레드가 없음 → 결정 대기로 되돌리고 알림
    assert run["status"] == "awaiting_human" and run["done"] and run["followup_error"] == live.FOLLOWUP_INTERRUPTED
    row = {"id": "r9", "question": "q", "root_id": org.id, "team_id": "general", "status": live.FOLLOWUP_RUNNING,
           "steps": [], "created_at": "2026-11-05T10:00:00"}
    restored = live.restore_run(row)  # 서버를 다시 켬
    assert restored["status"] == "awaiting_human" and restored["followup_error"] == live.FOLLOWUP_INTERRUPTED
    assert restored["statuses"][org.id] == ("wait", "👤 결정 대기")


# ─── claude_cli 고정 목록 · 부장 기본 모델 ─────────────────────────────────


def test_claude_cli_fixed_list_is_always_selectable(models, tmp_path):
    bare = models.model_copy(update={"model_tiers": [row for row in (
        [k for k in r if not k.startswith("claude_cli:") or k == LEAD] for r in models.model_tiers) if row]})
    merged = model_catalog.merge(bare, tmp_path)  # 목록을 한 번도 받지 않은 데이터 폴더
    keys = member_models.catalog(merged.models)
    for model_id, _ in model_catalog.CLAUDE_CLI_MODELS:
        assert f"claude_cli:{model_id}" in keys
    assert merged.info["claude_cli:claude-fable-5-1"].display == "Claude Fable 5.1"
    assert not (tmp_path / model_catalog.CATALOG_FILE).exists()  # 저장하지 않는다
    no_cli = bare.model_copy(update={"providers": {k: v for k, v in bare.providers.items() if k != "claude_cli"}})
    assert "claude_cli:sonnet" not in member_models.catalog(model_catalog.merge(no_cli, tmp_path).models)


def test_claude_cli_caps_cover_new_models():
    from core import model_caps

    for key in ("claude_cli:claude-fable-5-1", "claude_cli:claude-sonnet-5-5", "claude_cli:haiku"):
        caps = model_caps.caps_for(key, None)
        assert caps.vision is True and caps.structured_output is True


def test_default_head_is_at_least_as_strong_as_team_lead(models):
    chain, source = dept_heads.head_chain(models)
    lead = member_models.model_key(models.ranks["team_lead"].primary)
    assert source == "critic" and member_models.model_key(chain.primary) == lead
    assert not member_models.weaker(models, member_models.model_key(chain.primary), lead)
    assert any(s.provider == "freellm" for s in chain.candidates)  # '무료만'에서도 비평이 돈다
    # 🛎 인포메이션 센터는 부장 기본(구독)이 아니라 가벼운 utility
    assert dept_heads.assist_chain(models) == models.utility


def test_weaker_head_gets_warning_not_blocked(models, org, team):
    assert member_models.head_warning(models, "google:gemini-3.5-flash", LEAD).startswith("부장 모델 gemini-3.5-flash")
    assert member_models.head_warning(models, LEAD, LEAD) == ""
    assert member_models.head_warning(models, "claude_cli:claude-fable-5-1", LEAD) == ""
    chains = member_models.team_rank_models(org, models, team, {})
    plan = dept_heads.critic_plan(models, team, chains, org.id, choice="google:gemini-3.5-flash")
    assert plan.source == "chosen" and any("보다 약합니다" in w for w in plan.warnings)  # 고른 그대로 쓰되 경고
    # 팀장을 부장(기본 Opus)보다 강한 Fable 로 고르면 경고
    chains = member_models.team_rank_models(org, models, team, {org.id: "claude_cli:claude-fable-5-1"})
    assert any("보다 약합니다" in w for w in dept_heads.critic_plan(models, team, chains, org.id).warnings)


# ─── 화면 (AppTest) ─────────────────────────────────────────────────


APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app_env(tmp_path, monkeypatch, use_script):
    import streamlit as st
    from tests.fakes import make_delegation, make_review

    gate, draft_gate = threading.Event(), threading.Event()

    def draft_text(messages):
        draft_gate.wait(20)  # 초안이 끝나지 않은 동안 진행률을 본다
        return "초안 [1]"

    def followup_answer(messages):
        gate.wait(20)
        from core.followup import FollowupResult

        return FollowupResult(analysis="질문", revised=False, reply="3번은 [1] 자료가 근거입니다.")

    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    use_script(Script(text=draft_text, Delegation=delegation, ReviewResult=make_review(verdict="approve", score=9, answer="보완안"),
                      FollowupResult=followup_answer))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    call_monitor.reset()
    progress.clear_history_cache()
    st.cache_resource.clear()
    yield gate, draft_gate
    gate.set()
    draft_gate.set()
    st.cache_resource.clear()
    call_monitor.reset()


def _open():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["team"] = "general"
    at.query_params["view"] = "team"
    at.session_state["office_view"] = "chart"
    return at


def test_app_progress_bar_and_background_followup(app_env):
    gate, draft_gate = app_env
    at = _open().run()
    assert not at.exception, at.exception
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()
    at.text_area(key="q_text").input("양자컴퓨팅 요약")
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    rid = wait_for(lambda: next((r for r in list(call_monitor._calls) if r), None))
    wait_for(lambda: any(c.kind == "draft" for c in call_monitor.in_flight(rid)))
    # 진행 중: 위쪽 진행 막대 + 📊 시간표 탭의 진행률·남은 단계
    at = _open()
    at.session_state["mon_open_general"] = True
    at = at.run()
    assert not at.exception, at.exception
    bars = at.get("progress")
    assert bars and all("남은 단계" in b.proto.text and "추정:" in b.proto.text for b in bars)
    assert any("기록 부족" in b.proto.text for b in bars)  # 지난 기록이 없는 새 데이터 폴더 → 기본값이라고 밝힌다
    assert any(e.label.startswith("⏳ 남은 단계") for e in at.expander)
    assert any(c.value.startswith("📊 기본 팀 — ") and "% · " in c.value for c in at.caption)  # 사무실·조직도 위 한 줄
    draft_gate.set()
    # 결정 대기까지 (가짜 모델이라 금방 끝난다)
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown) else None)(_open().run()), 60)
    assert not at.exception, at.exception
    at.text_area(key=next(t.key for t in at.text_area if t.placeholder and "3번 근거" in t.placeholder)).input("3번 근거가 뭐야?")
    at = next(b for b in at.button if b.label == "💬 보내기").click().run()
    assert not at.exception, at.exception
    # 백그라운드에서 답하는 중 — 화면은 막히지 않고, 진행 막대·추가 질문 표시·중단 버튼이 보인다
    at = _open()
    at.session_state["mon_open_general"] = True
    at = at.run()
    assert not at.exception, at.exception
    assert any("💬 추가 질문 답하는 중" in m.value for m in at.markdown)
    assert any(c.value.startswith("📊 기본 팀 — 💬 추가 질문 ") and "(추정)" in c.value for c in at.caption)
    assert any("추가 질문에 답하는 중입니다" in i.value for i in at.info)  # 결정은 답이 온 뒤에
    assert any(b.label == "⏹ 추가 질문 중단" for b in at.button)
    gate.set()
    at = wait_for(lambda: (lambda a: a if any("[1] 자료가 근거" in m.value for m in a.markdown) else None)(_open().run()), 30)
    assert not at.exception, at.exception
    assert any(r.label == "채택할 안" for r in at.radio)  # 다시 결정 대기


def test_app_shows_head_review_and_sends_extra_fixes(app_env, use_script):
    """수정본 아래에 👔 부장 검토가 보이고, '추가로 고칠 점'을 버튼 하나로 다음 수정 요청으로 보낸다."""
    from tests.fakes import make_critic, make_delegation, make_review
    from tests.test_followup import edit, judge

    gate, draft_gate = app_env
    gate.set()
    draft_gate.set()

    def delegation(messages):
        text = messages[-1].content
        return make_delegation("gm_research", "gm_action") if "gm_research" in text else make_delegation("am_fact", "staff_draft")

    script = use_script(Script(
        text="초안 [1]", Delegation=delegation, ReviewResult=make_review(verdict="approve", score=9, answer="보완안"),
        CriticResult=make_critic(5), FollowupResult=[edit("수정본 하나"), edit("수정본 둘", ["기준일 추가"])],
        RevisionCritic=[judge(5, extra=["표 제목에 기준일 넣기"]), judge(5)]))
    at = _open().run()
    at = at.selectbox(key="dept_task_research").set_value("general:free").run()
    at.text_area(key="q_text").input("양자컴퓨팅 요약")
    at = next(b for b in at.button if b.label == "맡기기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("💬 결과에 대해 묻기" in m.value for m in a.markdown) else None)(_open().run()), 60)
    at.text_area(key=next(t.key for t in at.text_area if t.placeholder and "3번 근거" in t.placeholder)).input("정의를 줄여줘")
    at = next(b for b in at.button if b.label == "💬 보내기").click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("추가로 고칠 점" in m.value for m in a.markdown) else None)(_open().run()), 30)
    assert not at.exception, at.exception
    review = next(m.value for m in at.markdown if "추가로 고칠 점" in m.value)
    assert "부장" in review and "5/5" in review and "합격" in review and "표 제목에 기준일 넣기" in review
    at = next(b for b in at.button if b.label.startswith("🛠")).click().run()
    assert not at.exception, at.exception
    at = wait_for(lambda: (lambda a: a if any("수정본 v2" in m.value for m in a.markdown) else None)(_open().run()), 30)
    assert not at.exception, at.exception
    assert "표 제목에 기준일 넣기" in script.prompts("FollowupResult")[1]  # 부장이 찾은 점이 다음 수정 요청으로 갔다
    assert not any(b.label.startswith("🛠") for b in at.button)  # 고칠 점이 없으면 버튼도 없다


def test_saving_monitor_after_restart_keeps_older_calls():
    """서버를 다시 켠 뒤 💬 추가 질문만 메모리에 있을 때 저장해도 예전 호출 기록이 남는다."""
    old = {"calls": [rec("draft", "a", start=1).to_dict(), rec("review", "lead", start=2).to_dict()],
           "interventions": [{"ts": 1.0, "what": "⏭ 모델 바꿈"}]}
    now = {"calls": [rec("followup", "lead", start=9).to_dict()], "interventions": []}
    merged = call_monitor.merge_saved(old, now)
    assert [c["kind"] for c in merged["calls"]] == ["draft", "review", "followup"]
    assert merged["interventions"] == old["interventions"]
    again = call_monitor.merge_saved(merged, {"calls": [rec("draft", "a", start=1).to_dict()], "interventions": []})
    assert len(again["calls"]) == 3  # 같은 호출은 두 번 들어가지 않는다
