"""⏱ 호출 모니터 (core/call_monitor.py) — 호출 기록·응답 없음 표시·다른 모델로 바꾸기·claude_cli 프로세스 끊기·바꿀 수 있는 모델 규칙.

가짜 모델만 쓴다 (네트워크·키 없음). '느린 모델'은 threading.Event 를 기다리는 가짜 모델이다.
"""

import io
import threading
import time

import pytest
from langchain_core.messages import HumanMessage

from core import call_monitor, claude_cli, llm_factory, switch_rules, usage
from core.agent import OrgAgent
from core.config import load_ranks
from core.member_models import split_key
from core.schemas import Delegation, ModelSpec, RankModels
from tests.fakes import FakeOfficeModel, Script, make_delegation

SLOW = ModelSpec(provider="openai", model="slow")
FAST = ModelSpec(provider="openai", model="fast")
MSGS = [HumanMessage("업무")]


@pytest.fixture(autouse=True)
def fresh_monitor():
    call_monitor.reset()
    llm_factory.set_concurrency({})
    yield
    call_monitor.reset()
    llm_factory._concurrency = None  # 다음 테스트는 settings.yaml 값으로 (test_concurrency.reset_concurrency 와 같음)
    llm_factory._slots.clear()


def wait_for(fn, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("기다리던 상태가 오지 않음")


def install(monkeypatch, scripts: dict[str, Script]):
    """spec.model 이름마다 다른 가짜 모델."""
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, models: FakeOfficeModel(script=scripts[spec.model]))


def in_thread(fn, run_id="r1", agent_id="a1"):
    box: dict = {}

    def target():
        with usage.scope(run_id=run_id, agent_id=agent_id, team_id="general"):
            try:
                box["value"] = fn()
            except BaseException as e:  # noqa: BLE001
                box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, box


def slow_delegation(gate: threading.Event, child="late"):
    def make(messages):
        gate.wait(5)
        return make_delegation(child)

    return make


# ─── 기록 ─────────────────────────────────────────────────────────────


def test_registry_records_queued_running_done(set_keys, models, monkeypatch):
    set_keys("openai")
    gate = threading.Event()
    install(monkeypatch, {"slow": Script(Delegation=slow_delegation(gate, "done"))})
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"))
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "running"), None))
    assert (rec.who, rec.kind, rec.provider, rec.model, rec.agent_id, rec.team_id) == ("사원", "plan", "openai", "slow", "a1", "general")
    assert rec.started_at >= rec.queued_at and rec.ended_at is None
    assert call_monitor.in_flight("r1") == [rec]
    gate.set()
    t.join(5)
    assert box["value"].parsed.assignments[0].child_id == "done"
    assert rec.status == "done" and rec.ended_at >= rec.started_at
    assert call_monitor.in_flight("r1") == []
    row = call_monitor.rows([rec])[0]
    assert row["상태"] == "✅ 끝" and row["일"] == "📋 업무 분배" and row["모델"] == "openai:slow"


def test_waiting_for_concurrency_slot_is_shown(set_keys, models, monkeypatch):
    set_keys("openai")
    llm_factory.set_concurrency({"openai": 1})
    gate = threading.Event()
    install(monkeypatch, {"slow": Script(Delegation=slow_delegation(gate))})
    t1, _ = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "A"))
    wait_for(lambda: any(c.status == "running" for c in call_monitor.calls("r1")))
    t2, _ = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "B"))
    queued = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.who == "B"), None))
    time.sleep(0.1)
    assert queued.status == "queued"
    assert "⏳ 동시 호출 자리 대기" in call_monitor.flags(queued, now=queued.queued_at + 2)
    gate.set()
    t1.join(5)
    t2.join(5)
    assert queued.status == "done" and queued.waited() > 0


def test_no_response_flag_after_stall_seconds():
    now = time.time()
    rec = call_monitor.CallRecord(id="x", run_id="r", who="비평가", kind="critic", provider="google", model="gemini",
                                  queued_at=now - 130)
    rec.running()
    rec.started_at = rec.last_activity = now - 125
    assert "⚠️ 응답 없음 2분" in call_monitor.flags(rec, now=now)
    assert not any(f.startswith("⚠️") for f in call_monitor.flags(rec, now=rec.started_at + 30))
    assert "⚠️ 응답 없음 5초" in call_monitor.flags(rec, now=rec.started_at + 5, stall_sec=3)
    rec.token("조각")  # 토큰이 오면 '응답 없음' 시계가 다시 시작된다
    assert not any(f.startswith("⚠️") for f in call_monitor.flags(rec, now=rec.last_activity + 10))
    rec.attempt, rec.is_fallback = 2, True
    flags = call_monitor.flags(rec, now=rec.last_activity + 1)
    assert "🔁 재시도 2회" in flags and "↪️ 대안 모델로" in flags


def test_retry_wait_is_recorded_and_can_be_switched(set_keys, models, monkeypatch):
    """일시적 오류 뒤 '재시도 대기' 중에도 다른 모델로 바꾸면 기다리지 않고 바로 그 모델로."""
    set_keys("openai")

    def overloaded(messages):
        raise RuntimeError("503 overloaded")

    install(monkeypatch, {"slow": Script(Delegation=overloaded), "fast": Script(Delegation=make_delegation("fast"))})
    monkeypatch.setattr(llm_factory, "RETRY_DELAYS", (30.0,))  # 진짜로 30초 기다리는 대기 — 바꾸면 바로 깨야 한다
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"))
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "retry_wait"), None))
    assert "⏸ 30초 뒤 같은 모델로 다시" in call_monitor.flags(rec)
    started = time.time()
    assert call_monitor.switch(rec.id, FAST)
    t.join(5)
    assert time.time() - started < 3
    assert box["value"].parsed.assignments[0].child_id == "fast" and box["value"].spec == FAST
    assert rec.status == "switched"


# ─── 끊고 다른 모델로 ────────────────────────────────────────────────


def test_switch_cancels_hung_call_and_discards_its_late_output(set_keys, models, monkeypatch):
    set_keys("openai")
    gate = threading.Event()
    install(monkeypatch, {"slow": Script(Delegation=slow_delegation(gate, "late")),
                          "fast": Script(Delegation=make_delegation("fast"))})
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"))
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "running"), None))
    assert call_monitor.switch(rec.id, FAST)
    t.join(5)
    call = box["value"]
    assert call.spec == FAST and call.parsed.assignments[0].child_id == "fast"
    assert any("사용자가" in e and "바꿈" in e for e in call.errors)
    gate.set()  # 끊긴 호출이 뒤늦게 답해도
    time.sleep(0.1)
    assert rec.status == "switched" and call.parsed.assignments[0].child_id == "fast"  # 결과는 바뀌지 않는다
    statuses = [(c.model, c.status) for c in call_monitor.calls("r1")]
    assert statuses == [("slow", "switched"), ("fast", "done")]
    log = call_monitor.interventions("r1")
    assert log[-1]["who"] == "사용자" and log[-1]["what"] == "⏭ 모델 바꿈" and "openai:fast" in log[-1]["detail"]


def test_switch_for_rest_of_run_applies_to_later_calls(set_keys, models, monkeypatch):
    set_keys("openai")
    gate = threading.Event()
    install(monkeypatch, {"slow": Script(Delegation=slow_delegation(gate)), "fast": Script(Delegation=make_delegation("fast"))})
    t, _ = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"))
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "running"), None))
    call_monitor.switch(rec.id, FAST, for_rest=True)
    t.join(5)
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"))
    t.join(5)
    assert box["value"].spec == FAST and not box["value"].is_fallback  # 처음부터 고른 모델로
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=FAST), models, Delegation, MSGS, "다른 사람"),
                       run_id="r1")
    t.join(5)
    gate.set()
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=SLOW), models, Delegation, MSGS, "사원"),
                       run_id="other-run")
    t.join(5)
    assert box["value"].spec == SLOW  # 다른 실행에는 적용되지 않는다


def test_streaming_draft_switch_discards_half_written_text(set_keys, org, ranks, team, models, monkeypatch):
    set_keys("openai")
    gate = threading.Event()

    class HalfThenHang(FakeOfficeModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield from super()._stream(messages, stop, run_manager, **kwargs)
            gate.wait(5)

    scripts = {"slow": Script(text="반쯤 쓴 초안"), "fast": Script(text="새 모델 초안")}
    monkeypatch.setattr(llm_factory, "create_chat_model",
                        lambda spec, m: (HalfThenHang if spec.model == "slow" else FakeOfficeModel)(script=scripts[spec.model]))
    node = next(n for n in org.walk() if not n.children)
    rank = next(r for r in ranks if r.id == node.rank)
    agent = OrgAgent(node, rank, team, models, chain=RankModels(primary=SLOW))
    restarts = []
    t, box = in_thread(lambda: agent.draft(MSGS, "draft:x", lambda: restarts.append(1)))
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "streaming"), None))
    assert rec.kind == "draft" and rec.chars > 0 and rec.ttft() is not None
    call_monitor.switch(rec.id, FAST)
    t.join(5)
    gate.set()
    assert box["value"].content == "새 모델 초안" and box["value"].model == FAST
    assert restarts == [1]  # 화면의 '작성 중' 미리보기를 비우라는 알림


def test_claude_cli_process_is_killed_on_switch(set_keys, models, monkeypatch):
    set_keys("openai")
    procs = []

    class HangingProc:
        def __init__(self, cmd, **kwargs):
            self.killed = threading.Event()
            self.stdin, self.stderr, self.returncode = io.StringIO(), io.StringIO(""), None
            self.stdin.close = lambda: None
            self.stdout = self._lines()
            procs.append(self)

        def _lines(self):
            self.killed.wait(10)
            return
            yield  # noqa

        def kill(self):
            self.returncode = -9
            self.killed.set()

        def wait(self, timeout=None):
            self.killed.wait(10)
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(claude_cli, "find_executable", lambda: "C:/fake/claude.exe")
    monkeypatch.setattr(claude_cli, "cli_status", lambda: "ok")
    monkeypatch.setattr(claude_cli.subprocess, "Popen", HangingProc)
    fast = Script(Delegation=make_delegation("fast"))
    monkeypatch.setattr(llm_factory, "create_chat_model", lambda spec, m: claude_cli.ClaudeCLIChatModel(model=spec.model, timeout=60)
                        if spec.provider == "claude_cli" else FakeOfficeModel(script=fast))
    cli = ModelSpec(provider="claude_cli", model="claude-opus-5-5")
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=cli), models, Delegation, MSGS, "팀장"))
    wait_for(lambda: procs)
    rec = wait_for(lambda: next((c for c in call_monitor.calls("r1") if c.status == "running"), None))
    assert not procs[0].killed.is_set()
    call_monitor.switch(rec.id, FAST)
    assert procs[0].killed.wait(2)  # 구독 프로세스를 바로 강제 종료
    t.join(5)
    assert box["value"].spec == FAST


def test_stopped_run_refuses_new_calls(set_keys, models, monkeypatch):
    set_keys("openai")
    install(monkeypatch, {"fast": Script(Delegation=make_delegation("fast"))})
    call_monitor.stop("r1")
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=FAST), models, Delegation, MSGS, "사원"))
    t.join(5)
    assert isinstance(box["error"], call_monitor.CallCancelled) and box["error"].reason == "stop"
    assert llm_factory._cooldown == {}  # 사람이 끊은 것은 모델 실패가 아니다 (쉬게 하지 않음)
    call_monitor.release("r1")
    t, box = in_thread(lambda: llm_factory.invoke_structured(RankModels(primary=FAST), models, Delegation, MSGS, "사원"))
    t.join(5)
    assert box["value"].spec == FAST


# ─── 바꿀 수 있는 모델 (🎛 팀원 모델 선택과 같은 규칙) ─────────────────


def test_switch_options_follow_tier_and_free_only_rules(set_keys, org, team, models):
    set_keys("openai", "google", "anthropic", "upstage")
    leaf = next(n for n in org.walk() if not n.children)
    top_key = models.model_tiers[0][0]
    # 팀장이 가장 강한 모델을 직접 골랐다 → 그 아래 사원은 가장 강한 줄을 고를 수 없다
    choices = {org.id: top_key}
    rec = call_monitor.CallRecord(id="c", run_id="r", who="사원", kind="draft", provider="openai", model="nope", agent_id=leaf.id)
    opts = {o.key: o for o in switch_rules.options(rec, org, team, models, models, choices)}
    assert set(opts) == {k for tier in models.model_tiers for k in tier}
    for key in models.model_tiers[0]:
        assert not opts[key].allowed and opts[key].reason == "상사보다 강한 모델 (팀원 규칙)"
    # 공용 호출(비평가)은 팀원 규칙을 받지 않는다
    critic = call_monitor.CallRecord(id="d", run_id="r", who="비평가", kind="critic", provider="x", model="y", agent_id=leaf.id)
    copts = {o.key: o for o in switch_rules.options(critic, org, team, models, models, choices)}
    assert all(o.reason != "상사보다 강한 모델 (팀원 규칙)" for o in copts.values())
    # 🆓 무료 모델만 → 유료 provider 는 안 됨 (이유 표시)
    free = llm_factory.free_only(models)
    fopts = switch_rules.options(critic, org, team, models, free, choices)
    paid = [o for o in fopts if split_key(o.key)[0] not in ("freellm",)]
    assert paid and all(not o.allowed for o in paid)
    assert any(o.reason == "🆓 무료 모델만 사용 중" for o in paid)
    # 지금 쓰는 모델은 고를 수 없다
    cur = call_monitor.CallRecord(id="e", run_id="r", who="비평가", kind="critic", provider=split_key(top_key)[0],
                                  model=split_key(top_key)[1])
    assert {o.key: o for o in switch_rules.options(cur, org, team, models, models)}[top_key].reason == "지금 쓰는 모델"


def test_switch_options_respect_disabled_providers(set_keys, org, team, models):
    set_keys("openai", "google", "anthropic", "upstage")
    off = models.model_copy(update={"disabled_providers": ["google"]})
    rec = call_monitor.CallRecord(id="c", run_id="r", who="비평가", kind="critic", provider="x", model="y")
    for o in switch_rules.options(rec, org, team, off, off):
        if o.key.startswith("google:"):
            assert not o.allowed and "비활성" in o.reason


# ─── 요약·저장 ───────────────────────────────────────────────────────


def test_summary_and_dump_round_trip():
    now = time.time()
    recs = []
    for i, (kind, model, wait, run) in enumerate([("draft", "a", 0, 10), ("critic", "b", 5, 300), ("draft", "a", 0, 20)]):
        r = call_monitor.CallRecord(id=f"c{i}", run_id="r", who=f"w{i}", kind=kind, provider="p", model=model, queued_at=now)
        r.started_at = now + wait
        r.ended_at = now + wait + run
        r.status = "done"
        r.output_tokens = 100
        recs.append(r)
    s = call_monitor.summary(recs, now=now + 400)
    assert s["calls"] == 3 and s["model_sec"] == 330 and s["wait_sec"] == 5
    assert s["by_kind"]["✍️ 초안"]["호출"] == 2 and s["by_model"]["p:b"]["모델 시간(초)"] == 300
    assert s["slowest"][0]["누가"] == "w1" and s["slowest"][0]["모델 시간"] == "5분 00초"
    back, _ = call_monitor.load({"calls": [r.to_dict() for r in recs], "interventions": [{"what": "x"}]})
    assert [r.to_dict() for r in back] == [r.to_dict() for r in recs]
    assert "ao-gt-row" in call_monitor.gantt_html(back, now=now + 400)
