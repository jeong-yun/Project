"""조직도(트리) 워크플로우 테스트 — 위→아래 분배, 아래→위 검토·통합, 브리프, 비평가·자가 교정, 하네스."""

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


def test_tree_runs_top_down_then_bottom_up(org, ranks, team, models, use_script):
    use_script(full_script())
    _, steps = run(org, ranks, team, models)
    order = [(s.node_id, s.kind) for s in steps]
    assert order == [
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
    assert steps[-1].content == "보완안"


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
