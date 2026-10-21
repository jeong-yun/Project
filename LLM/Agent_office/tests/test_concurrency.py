"""동시 진행 테스트 — 형제 서브트리·자료 조사·검색어가 동시에 돌고, 상사는 하위가 모두 끝난 뒤에만 통합한다.

가짜 모델에 sleep 을 넣어 '언제 시작해서 언제 끝났는지'를 기록하고, 시간 구간이 겹치는지로 판단한다.
"""

import re
import threading
import time

import pytest

from core import llm_factory, search
from core import tools as tools_pkg
from core.live import apply_event, consume, new_run
from core.schemas import HumanDecision, LessonSet, Team, ToolResult
from core.workflow import build_office_graph, make_checkpointer, resume_with_decision, stream_office
from tests.fakes import Script, make_delegation, make_review
from tests.test_live_avatars import step
from tests.test_search import PLAN
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS, TREE_ORDER

AGENT_NAME = re.compile(r"이름: (.+?) \(직급")  # 시스템 프롬프트의 에이전트 이름 → 누가 부른 호출인지


@pytest.fixture(autouse=True)
def reset_concurrency():
    """테스트가 바꾼 동시 호출 제한을 settings.yaml 값으로 되돌린다."""
    yield
    llm_factory._concurrency = None
    llm_factory._slots.clear()


class Timeline:
    """호출마다 (이름, 시작, 끝) 을 기록한다. 동시에 진행 중인 호출 수의 최댓값도 잰다."""

    def __init__(self):
        self.spans: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, name: str, seconds: float):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        start = time.perf_counter()
        time.sleep(seconds)
        with self.lock:
            self.active -= 1
            self.spans[name] = (start, time.perf_counter())

    def overlap(self, a: str, b: str) -> bool:
        (s1, e1), (s2, e2) = self.spans[a], self.spans[b]
        return max(s1, s2) < min(e1, e2)


def tree_script(org, timeline: Timeline, draft_sec: dict[str, float], fail_reviewer: str | None = None) -> Script:
    """일반 팀 조직도: 관리자는 하위 전원에게 분배, 실무자는 초안-<id> 를 쓴다 (draft_sec 만큼 걸림).

    기록 이름: 초안은 실무자 id, 검토·통합은 "review:<관리자 id>".
    """
    ids = {n.name: n.id for n in org.walk()}

    def who(messages) -> str:
        return ids[AGENT_NAME.search(messages[0].content).group(1)]

    def delegation(messages):
        text = messages[-1].content
        if "gm_research" in text:
            return make_delegation("gm_research", "gm_action")
        return make_delegation("am_fact", "staff_draft")

    def draft(messages):
        node = who(messages)
        timeline.run(node, draft_sec.get(node, 0.05))
        return f"초안-{node}"

    def review(messages):
        node = who(messages)
        if node == fail_reviewer:
            raise ValueError("검토 모델 고장")
        timeline.run(f"review:{node}", 0.05)
        # 하위 초안 표시를 그대로 이어 붙여 올린다 → 최종 답에 누구 결과가 들어갔는지 확인할 수 있다
        return make_review(answer="보완안 " + " ".join(re.findall(r"초안-\w+", messages[-1].content)))

    return Script(text=draft, Delegation=delegation, ReviewResult=review)


def run_events(org, ranks, team, models, root):
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    return stream_office(graph, org, "질문", root, brief=BRIEF)


# ─── 형제 서브트리 동시 진행 ───────────────────────────────────────────


def test_sibling_drafts_overlap_and_parent_waits_for_all(org, ranks, team, models, use_script):
    timeline = Timeline()
    use_script(tree_script(org, timeline, {"am_fact": 0.1, "staff_draft": 0.4}))
    steps = [p for k, p in run_events(org, ranks, team, models, "dgm_research") if k == "step"]

    assert timeline.overlap("am_fact", "staff_draft")  # 두 실무자가 동시에 썼다
    review_start = timeline.spans["review:dgm_research"][0]
    assert review_start >= max(end for _, end in (timeline.spans["am_fact"], timeline.spans["staff_draft"]))
    # 빨리 끝난 초안이 먼저 화면에 온다 (완료 순서), 통합은 맨 마지막
    assert [(s.node_id, s.kind) for s in steps] == [
        ("dgm_research", "plan"), ("am_fact", "draft"), ("staff_draft", "draft"), ("dgm_research", "integrate")
    ]
    assert "초안-am_fact" in steps[-1].content and "초안-staff_draft" in steps[-1].content


def test_independent_subtrees_do_not_wait_for_each_other(org, ranks, team, models, use_script):
    """실행 부장 쪽(staff_case)은 리서치 쪽 초안이 끝나기를 기다리지 않고, 그 위 검토도 먼저 끝낼 수 있다."""
    timeline = Timeline()
    # 실행 쪽 초안은 바로 시작하고, 리서치 쪽 초안은 차장의 업무 분배 뒤에 시작한다.
    # PC 가 바쁠 때(다른 테스트 동시 실행 등)도 둘이 겹치고 순서 판단에 여유가 있게 시간을 넉넉히 둔다.
    slow, fast = 1.0, 0.4
    use_script(tree_script(org, timeline, {"am_fact": slow, "staff_draft": slow, "staff_case": fast}))
    started = time.perf_counter()
    steps = [p for k, p in run_events(org, ranks, team, models, "lead") if k == "step"]
    elapsed = time.perf_counter() - started

    assert timeline.overlap("am_fact", "staff_case") and timeline.overlap("staff_draft", "staff_case")
    order = [(s.node_id, s.kind) for s in steps]
    # 실행 쪽 검토 두 단계가 리서치 쪽 통합보다 먼저 끝난다 (예전처럼 깊이 우선 한 명씩이면 반대)
    assert order.index(("gm_action", "review")) < order.index(("dgm_research", "integrate"))
    assert order[-1] == ("lead", "integrate")
    lead_start = timeline.spans["review:lead"][0]
    assert all(end <= lead_start for name, (_, end) in timeline.spans.items() if name != "review:lead")  # 팀장은 모두 끝난 뒤 통합
    assert elapsed < slow + slow + fast  # 초안 세 개를 차례로 썼다면 이것만으로도 최소 2.4초


# ─── 한 하위의 오류 ────────────────────────────────────────────────────


def test_failed_drafter_is_reported_and_sibling_still_finishes(org, ranks, team, models, use_script):
    timeline = Timeline()
    script = use_script(tree_script(org, timeline, {"staff_draft": 0.2}))
    base = script.outputs["text"]

    def flaky(messages):
        if "am_fact 업무" in messages[-1].content:
            raise ValueError("모델 고장")
        return base(messages)

    script.outputs["text"] = flaky
    steps = [p for k, p in run_events(org, ranks, team, models, "dgm_research") if k == "step"]
    by_node = {s.node_id: s for s in steps if s.kind != "plan"}
    assert "모델 오류로 초안을 작성하지 못했습니다" in by_node["am_fact"].content and by_node["am_fact"].errors
    assert by_node["staff_draft"].content == "초안-staff_draft"
    assert by_node["dgm_research"].kind == "integrate"  # 실패 표시를 포함해 상사가 통합했다


def test_manager_error_stops_run_but_reports_sibling_work(org, ranks, team, models, use_script):
    """관리자(am_action) 검토가 실패하면 실행은 오류로 끝나지만, 이미 진행 중이던 형제 서브트리 결과는 화면에 남는다."""
    timeline = Timeline()
    use_script(tree_script(org, timeline, {"am_fact": 0.2, "staff_draft": 0.2}, fail_reviewer="am_action"))
    run = new_run("r1", "질문", "lead")
    consume(run, run_events(org, ranks, team, models, "lead"))

    assert run["status"] == "error" and run["done"] and "모든 후보 모델 호출에 실패" in run["error"]
    done = {(s.node_id, s.kind) for s in run["steps"]}
    assert {("am_fact", "draft"), ("staff_draft", "draft"), ("staff_case", "draft")} <= done
    assert ("am_action", "review") not in done and ("lead", "integrate") not in done
    assert run["drafting"] == {} and run["activity"] == {}


# ─── provider 별 동시 호출 제한 ─────────────────────────────────────────


def test_provider_slot_caps_concurrent_calls():
    llm_factory.set_concurrency({"default": 2})
    timeline = Timeline()

    def call(i):
        with llm_factory.provider_slot("freellm"):
            timeline.run(f"c{i}", 0.05)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert timeline.max_active == 2


def test_concurrency_limit_reads_settings_and_falls_back_to_default():
    llm_factory._concurrency = None
    assert llm_factory.concurrency_limit("freellm") == 2  # config/settings.yaml
    llm_factory.set_concurrency({"default": 5, "google": 1})
    assert llm_factory.concurrency_limit("google") == 1 and llm_factory.concurrency_limit("upstage") == 5


def test_limit_one_serializes_sibling_drafts(org, ranks, team, models, use_script):
    llm_factory.set_concurrency({"default": 1})
    timeline = Timeline()
    use_script(tree_script(org, timeline, {"am_fact": 0.1, "staff_draft": 0.1}))
    list(run_events(org, ranks, team, models, "dgm_research"))
    assert not timeline.overlap("am_fact", "staff_draft") and timeline.max_active == 1


# ─── 자료 조사 동시 진행 ────────────────────────────────────────────────


def test_data_tools_and_web_search_run_together(org, ranks, team, models, use_script, monkeypatch):
    timeline = Timeline()

    def slow_tool(question, inputs, models):
        timeline.run("tool", 0.2)
        return ToolResult(tool="slow", title="📊 느린 툴", content="수치")

    def slow_backend(query, max_results, region, **kwargs):
        timeline.run(f"search:{query}", 0.2)
        return [search.SearchResult(title=query, url=f"https://example.com/{query}", snippet="요약")]

    monkeypatch.setitem(tools_pkg.TOOLS, "slow_tool", slow_tool)
    monkeypatch.setitem(search.SEARCH_BACKENDS, "duckduckgo", slow_backend)
    use_script(Script(text="초안", SearchPlan=PLAN, ReviewResult=make_review()))
    tool_team = Team(**{**team.model_dump(), "tools": ["slow_tool", "web_search"]})
    graph = build_office_graph(org, ranks, tool_team, models, {**SETTINGS, "web_search": {"max_queries": 2}}, NO_CRITIC)
    events = list(stream_office(graph, org, "질문", "am_action", use_web=True, brief=BRIEF))

    q1, q2 = PLAN.queries
    assert timeline.overlap("tool", f"search:{q1}")  # 데이터 툴과 웹 검색이 동시에
    assert timeline.overlap(f"search:{q1}", f"search:{q2}")  # 검색어끼리도 동시에
    web = next(p for k, p in events if k == "research")
    assert [r.title for r in web.results] == [q1, q2]  # 합치는 순서는 검색어 순서 그대로
    assert next(p for k, p in events if k == "tools")[0].content == "수치"


# ─── 화면: 여러 실무자가 동시에 쓰는 토큰 ─────────────────────────────


def test_interleaved_tokens_are_kept_per_agent():
    run = new_run("r1", "질문", "lead")
    for node, text in [("a", "가"), ("b", "A"), ("a", "나"), ("b", "B")]:
        apply_event(run, "token", (node, text))
    assert run["drafting"] == {"a": "가나", "b": "AB"}

    apply_event(run, "retry", {"node": "a", "reason": "model"})  # a 만 처음부터 다시
    assert run["drafting"] == {"b": "AB"}
    apply_event(run, "token", ("a", "다"))
    apply_event(run, "step", step("b"))
    assert run["drafting"] == {"a": "다"}


# ─── 체크포인트 · 사람 결정(interrupt) ─────────────────────────────────


def test_concurrent_tree_still_stops_for_human_decision(org, ranks, team, models, use_script, tmp_path):
    timeline = Timeline()
    script = use_script(tree_script(org, timeline, {"am_fact": 0.1, "staff_case": 0.1}))
    script.outputs["LessonSet"] = LessonSet(lessons=[])
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC, checkpointer=make_checkpointer(tmp_path / "cp.db"))
    events = list(stream_office(graph, org, "질문", "lead", thread_id="run-1", brief=BRIEF))

    assert [k for k, _ in events][-1] == "interrupt"
    saved = graph.get_state({"configurable": {"thread_id": "run-1"}})
    assert saved.next == ("human_decision",)
    assert [(s.node_id, s.kind) for s in saved.values["steps"]] == TREE_ORDER  # 저장 순서는 실행마다 같다
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=saved.values["steps"][-1].content, score=5)
    assert [k for k, _ in resume_with_decision(graph, org, decision, "run-1")][-1] == "lessons"
