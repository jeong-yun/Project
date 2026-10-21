"""조직도(트리) 워크플로우 테스트 — 위→아래 분배, 아래→위 검토·통합, 브리프, 비평가·자가 교정, 하네스."""

import pytest

from core.brief import DesiredState
from core.schemas import Team
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_critic, make_delegation, make_review

SETTINGS = {"review": {"rewrite_answer": True}}
NO_CRITIC = {"critic": {"enabled": False}}
BRIEF = DesiredState(goal="목표", context="", success_criteria=["기준A"], constraints=[], output_format="", assumptions=[])


def run(org, ranks, team, models, root="lead", verification=NO_CRITIC, **kw):
    graph = build_office_graph(org, ranks, team, models, SETTINGS, verification)
    events = list(stream_office(graph, org, "질문", root, brief=kw.pop("brief", BRIEF), **kw))
    return events, [p for k, p in events if k == "step"]


def full_script(**extra):
    """팀장·차장은 하위 전원에게 분배, 모두 보완(revise)."""
    def delegation(messages):
        text = messages[-1].content
        if "gm_research" in text:  # 팀장 차례
            return make_delegation("gm_research", "gm_action")
        return make_delegation("am_fact", "staff_draft")  # 차장 차례

    return Script(text="초안 [1]", Delegation=delegation, ReviewResult=make_review(answer="보완안"), **extra)


TREE_ORDER = [
    ("lead", "plan"),
    ("dgm_research", "plan"),  # 부장·리서치는 하위가 1명 → 분배 LLM 생략
    ("am_fact", "draft"),
    ("staff_draft", "draft"),
    ("dgm_research", "integrate"),
    ("gm_research", "review"),
    ("staff_case", "draft"),
    ("am_action", "review"),
    ("gm_action", "review"),
    ("lead", "integrate"),
]


def test_tree_runs_top_down_then_bottom_up(org, ranks, team, models, use_script):
    use_script(full_script())
    _, steps = run(org, ranks, team, models)
    order = [(s.node_id, s.kind) for s in steps]
    # 화면에는 끝난 순서대로 온다 (형제는 동시에 진행) → 순서는 '분배 → 하위 → 상사 통합' 인과 관계만 지키면 된다
    assert sorted(order) == sorted(TREE_ORDER)
    before = lambda a, b: order.index(a) < order.index(b)  # noqa: E731
    assert before(("lead", "plan"), ("dgm_research", "plan"))
    assert before(("dgm_research", "plan"), ("am_fact", "draft")) and before(("dgm_research", "plan"), ("staff_draft", "draft"))
    assert before(("am_fact", "draft"), ("dgm_research", "integrate")) and before(("staff_draft", "draft"), ("dgm_research", "integrate"))
    assert before(("dgm_research", "integrate"), ("gm_research", "review"))
    assert before(("staff_case", "draft"), ("am_action", "review")) and before(("am_action", "review"), ("gm_action", "review"))
    assert order[-1] == ("lead", "integrate") and steps[-1].content == "보완안"


def test_state_steps_keep_depth_first_order(org, ranks, team, models, use_script):
    """그래프 상태(저장·교훈 생성에 쓰임)의 steps 는 실행마다 같은 순서(깊이 우선)."""
    use_script(full_script())
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    state = graph.invoke({"question": "질문", "root_id": "lead", "brief_input": BRIEF, "steps": []}, {"recursion_limit": 25})
    assert [(s.node_id, s.kind) for s in state["steps"]] == TREE_ORDER


def test_supervisor_only_assigns_needed_children(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["Delegation"] = make_delegation("gm_action")  # 팀장이 실행 부장에게만 맡김
    _, steps = run(org, ranks, team, models)
    assert "gm_research" not in {s.node_id for s in steps}
    assert steps[-1].node_id == "lead" and steps[-1].kind == "review"  # 하위 1명 → 검토


def test_invalid_delegation_assigns_all_children(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["Delegation"] = make_delegation("nobody")
    _, steps = run(org, ranks, team, models)
    assert {"gm_research", "gm_action"} <= {s.node_id for s in steps}


def test_clicked_agent_becomes_supervisor_of_subtree(org, ranks, team, models, use_script):
    use_script(full_script())
    _, steps = run(org, ranks, team, models, root="gm_action")
    assert [(s.node_id, s.kind) for s in steps] == [
        ("staff_case", "draft"), ("am_action", "review"), ("gm_action", "review")
    ]


def test_draft_tokens_are_tagged_with_agent(org, ranks, team, models, use_script):
    use_script(full_script())
    events, _ = run(org, ranks, team, models, root="am_action")
    tokens = [p for k, p in events if k == "token"]
    assert {nid for nid, _ in tokens} == {"staff_case"}
    assert "".join(t for _, t in tokens) == "초안 [1]"
    starts = [(p["node"], p["kind"]) for k, p in events if k == "start"]
    assert ("staff_case", "draft") in starts and ("am_action", "review") in starts
    # 하위가 1명인 관리자도 업무를 넘길 때 '대기' 상태 이벤트를 보낸다
    assert {"type": "delegated", "node": "am_action", "children": ["staff_case"]} in [p for k, p in events if k == "delegated"]


# ─── 브리프 (최종 상태 정의) ─────────────────────────────────────────


def test_user_brief_is_used_without_llm_and_reaches_agents(org, ranks, team, models, use_script):
    script = use_script(full_script())
    events, _ = run(org, ranks, team, models, root="am_action")
    assert [p for k, p in events if k == "brief"][0].goal == "목표"
    assert script.prompts("DesiredState") == []  # 사용자가 채웠으므로 요청 정리 LLM 호출 없음
    assert all("기준A" in p for p in script.prompts("text"))  # 완료 기준이 실무자 프롬프트에 들어감


def test_empty_brief_is_drafted_by_utility_model(org, ranks, team, models, use_script):
    drafted = BRIEF.model_copy(update={"goal": "AI 가 정리한 목표"})
    script = use_script(full_script(DesiredState=drafted))
    events, _ = run(org, ranks, team, models, root="am_action", brief=None)
    assert [p for k, p in events if k == "brief"][0].goal == "AI 가 정리한 목표"
    assert len(script.prompts("DesiredState")) == 1


# ─── 비평가 · 자가 교정 ───────────────────────────────────────────────


def test_critic_failure_triggers_self_correction(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["CriticResult"] = [make_critic(2, leaps=["근거 없는 결론"]), make_critic(5)]
    script.outputs["ReviewResult"] = [make_review(answer="1차"), make_review(answer="교정본")]
    cfg = {"critic": {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 1}}
    _, steps = run(org, ranks, team, models, root="am_action", verification=cfg)

    final = steps[-1]
    assert final.content == "교정본" and final.attempts == 2 and final.critic.score == 5
    correction = script.prompts("ReviewResult")[-1]
    assert "근거 없는 결론" in correction and "규칙에 어긋난다" in correction  # 자가 교정 프롬프트


def test_critic_scope_final_only_checks_supervisor(org, ranks, team, models, use_script):
    script = use_script(full_script(CriticResult=make_critic(5)))
    cfg = {"critic": {"enabled": True, "scope": "final"}}
    run(org, ranks, team, models, root="gm_action", verification=cfg)
    assert len(script.prompts("CriticResult")) == 1  # gm_action(최종)만


# ─── 하네스 (결정론적 검증 + Backtracking) ─────────────────────────────


def test_harness_failure_feeds_error_back_and_retries(org, ranks, team, models, use_script):
    bad, good = "```python\ndef f(:\n```", "```python\ndef f():\n    return 1\n```"
    script = use_script(full_script())
    script.outputs["text"] = [bad, good]
    harness_team = Team(**{**team.model_dump(), "harness": {"validators": ["python_syntax"], "max_retries": 2}})
    graph = build_office_graph(org, ranks, harness_team, models, SETTINGS, NO_CRITIC)
    events = list(stream_office(graph, org, "질문", "am_action", brief=BRIEF))
    draft = next(p for k, p in events if k == "step" and p.kind == "draft")

    assert draft.content == good and draft.attempts == 2 and draft.harness[0].passed
    assert ("retry", "harness") in [(k, p["reason"]) for k, p in events if k == "retry"]
    assert "문법 오류" in script.prompts("text")[-1]  # 에러 로그가 다시 쓰기 프롬프트에 들어감


# ─── 정확도 개선: 최고 시도 유지 · 통과 규칙 · 근거 자료 · 빈 초안 ─────────────

CRITIC_ON = {"critic": {"enabled": True, "scope": "final", "min_score": 4, "max_self_corrections": 1}}


def test_self_correction_keeps_best_attempt_when_correction_is_worse(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["CriticResult"] = [make_critic(3, leaps=["비약"]), make_critic(2, leaps=["더 큰 비약"])]
    script.outputs["ReviewResult"] = [make_review(answer="원본"), make_review(answer="더 나쁜 교정본")]
    _, steps = run(org, ranks, team, models, root="am_action", verification=CRITIC_ON)

    final = steps[-1]
    assert final.content == "원본" and final.critic.score == 3  # 교정본(2/5) 대신 원본(3/5)
    assert final.attempts == 2
    assert any("최종 비평가 여전히 불합격" in log and "3/5 → 2/5" in log for log in final.verification_log)
    assert any("1번째 답변을 유지" in log for log in final.verification_log)


def test_self_correction_still_failing_is_logged_when_last_is_best(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["CriticResult"] = [make_critic(2, leaps=["비약"]), make_critic(3, leaps=["작은 비약"])]
    script.outputs["ReviewResult"] = [make_review(answer="원본"), make_review(answer="조금 나은 교정본")]
    _, steps = run(org, ranks, team, models, root="am_action", verification=CRITIC_ON)

    final = steps[-1]
    assert final.content == "조금 나은 교정본" and final.critic.score == 3
    assert any("최종 비평가 여전히 불합격" in log for log in final.verification_log)
    assert not any("번째 답변을 유지" in log for log in final.verification_log)


def test_minor_violation_with_good_score_passes_without_correction(org, ranks, team, models, use_script):
    script = use_script(full_script(CriticResult=make_critic(4, violations=["표현이 장황함"], severity="minor")))
    events, steps = run(org, ranks, team, models, root="am_action", verification=CRITIC_ON)
    assert len(script.prompts("ReviewResult")) == 1  # 자가 교정 없음
    assert ("retry", "critic") not in [(k, p["reason"]) for k, p in events if k == "retry"]
    assert steps[-1].attempts == 1


def test_severe_violation_fails_even_with_high_score(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["CriticResult"] = [make_critic(5, violations=["숫자가 근거와 다름"], severity="severe"), make_critic(5)]
    script.outputs["ReviewResult"] = [make_review(answer="1차"), make_review(answer="교정본")]
    _, steps = run(org, ranks, team, models, root="am_action", verification=CRITIC_ON)
    assert steps[-1].content == "교정본"
    assert any("자가 교정 후 비평가 통과(5/5)" in log for log in steps[-1].verification_log)


def test_fail_on_any_keeps_strict_rule(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["CriticResult"] = [make_critic(4, violations=["사소함"], severity="minor"), make_critic(5)]
    script.outputs["ReviewResult"] = [make_review(answer="1차"), make_review(answer="교정본")]
    cfg = {"critic": {**CRITIC_ON["critic"], "fail_on": "any"}}
    _, steps = run(org, ranks, team, models, root="am_action", verification=cfg)
    assert steps[-1].content == "교정본"


def _team_with_tool(team, monkeypatch, content):
    import core.workflow as workflow
    from core.schemas import ToolResult

    monkeypatch.setattr(
        workflow, "run_tools", lambda *a, **k: [ToolResult(tool="fake", title="📈 시장 데이터", content=content)]
    )
    return Team(**{**team.model_dump(), "tools": ["fake"]})


def test_critic_receives_tool_evidence(org, ranks, team, models, use_script, monkeypatch):
    script = use_script(full_script(CriticResult=make_critic(5)))
    tool_team = _team_with_tool(team, monkeypatch, "종가 71,200원 (기준일 2026-10-20)")
    run(org, ranks, tool_team, models, root="am_action", verification=CRITIC_ON)
    critic_prompt = script.prompts("CriticResult")[0]
    assert "[근거 자료" in critic_prompt and "종가 71,200원 (기준일 2026-10-20)" in critic_prompt
    assert "출처·기준일" in critic_prompt  # 대조 지시


def test_critic_evidence_is_trimmed_to_budget(org, ranks, team, models, use_script, monkeypatch):
    script = use_script(full_script(CriticResult=make_critic(5)))
    tool_team = _team_with_tool(team, monkeypatch, "가" * 5000)
    cfg = {"critic": {**CRITIC_ON["critic"], "evidence_chars": 500}}
    run(org, ranks, tool_team, models, root="am_action", verification=cfg)
    critic_prompt = script.prompts("CriticResult")[0]
    assert "가" * 400 in critic_prompt and "가" * 600 not in critic_prompt
    assert "생략" in critic_prompt
    assert "가" * 5000 in script.prompts("text")[0]  # 실무자에게는 그대로 간다


@pytest.mark.parametrize(
    "empty, error",
    [
        ("", "No generation chunks"),  # 조각 0개 — langchain 이 ValueError 로 알려 준다
        (" \n", "EmptyResponseError"),  # 조각은 왔지만 공백뿐 — 예전에는 content="" 로 그대로 통과했다
    ],
)
def test_empty_draft_falls_back_to_next_model(org, ranks, team, models, use_script, empty, error):
    script = use_script(full_script())
    script.outputs["text"] = [empty, "초안"]  # 첫 후보는 빈 응답 → 다음 후보
    _, steps = run(org, ranks, team, models, root="am_action")
    draft = next(s for s in steps if s.kind == "draft")
    assert draft.content == "초안" and draft.is_fallback
    assert any(error in e for e in draft.errors), draft.errors


def test_all_empty_drafts_mark_agent_failed(org, ranks, team, models, use_script):
    script = use_script(full_script())
    script.outputs["text"] = ["   "]  # 공백뿐인 응답도 실패
    _, steps = run(org, ranks, team, models, root="am_action")
    draft = next(s for s in steps if s.kind == "draft")
    assert "초안을 작성하지 못했습니다" in draft.content
