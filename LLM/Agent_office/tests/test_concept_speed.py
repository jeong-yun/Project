"""개념 편집 팀 속도·구독 Claude 사용 (2026-10-27)

- 구독 Claude(claude_cli) 안전 필터 거부 → 재시도 없이 대안 모델, 그 팀에서만 잠시 건너뜀, 화면에 이유 표시
- claude_cli 성공 호출은 사용량에 팀·에이전트와 함께 '구독(비용 0)'으로 기록, 실패 호출은 기록하지 않음
- 실무자는 스킬 중 맡은 절만 받고(skill_sections) 산출물 범위(output_scope)만 쓴다
- 편집장 검토는 전문 대신 고칠 곳만(review_mode: edit)
"""

import pytest
from langchain_core.messages import HumanMessage

from core import claude_cli, llm_factory, usage
from core.agent import OrgAgent, apply_text_edits, skill_sections
from core.claude_cli import ClaudeCLIChatModel, ClaudeCLIRefused
from core.config import load_skills, load_team, team_org
from core.schemas import ReviewEdits, Team, TextEdit
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_delegation
from tests.test_claude_cli import CHAIN, Plan, _fallback_to_fake, cli_logged_in, fake_cli, result_event  # noqa: F401
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS

REFUSAL = ("API Error: Opus 5.5's safeguards flagged this message (https://www.anthropic.com/legal/aup). "
           "This sometimes happens with safe, normal conversations. Claude Code can't respond to this message with Opus 5.5.")


def refused() -> list[dict]:
    """실측(2026-10-27, 옛 스킬의 <thought> 태그)과 같은 결과 메시지."""
    return [{"type": "result", "subtype": "success", "is_error": True, "result": REFUSAL}]


# ─── 안전 필터 거부 ──────────────────────────────────────────────────


def test_refusal_is_its_own_error_and_not_retried(fake_cli):
    fake_cli(refused(), code=1)
    with pytest.raises(ClaudeCLIRefused) as err:
        ClaudeCLIChatModel().invoke("hi")
    e = err.value
    assert not llm_factory.is_transient(e) and llm_factory.retry_delay(e, 0) is None  # 다시 보내도 같은 거부
    assert not llm_factory.is_quota_exhausted(e)  # 구독 한도가 아니다 → 구독 전체를 쉬게 하지 않는다
    assert llm_factory.failure_reason(f"claude_cli · claude-opus-5-5: ClaudeCLIRefused: {e}") == "안전 필터가 요청을 거부"


def test_refusal_skips_claude_for_that_team_only(models, set_keys, fake_cli, cli_logged_in, monkeypatch):
    set_keys("openai")
    script = Script(Plan=Plan(analysis="대안", picks=["x"]))
    _fallback_to_fake(monkeypatch, script)
    calls = fake_cli(refused(), code=1)

    with usage.scope(team_id="concept"):
        first = llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "부장")
        assert first.spec.provider == "openai" and first.is_fallback and len(calls) == 1
        assert "ClaudeCLIRefused" in first.errors[0]
        # 같은 팀의 다음 단계(편집장 검토 등)는 거부를 또 기다리지 않고 바로 대안 모델 — 이유는 기록에 남는다
        second = llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "편집장")
        assert len(calls) == 1 and second.spec.provider == "openai"
        assert second.errors == ["claude_cli · claude-opus-5-5: 건너뜀 — 안전 필터가 요청을 거부 (잠시 쉬는 중)"]
        assert llm_factory.failure_reason(second.errors[0]) == "안전 필터가 요청을 거부"

    # 다른 팀은 프롬프트가 다르므로 구독 Claude 를 그대로 쓴다
    fake_cli([result_event("", structured_output={"analysis": "a", "picks": []})])
    with usage.scope(team_id="stock"):
        other = llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "팀장")
    assert other.spec.provider == "claude_cli" and not other.is_fallback and other.errors == []
    # 화면(팀 scope 밖)은 팀을 직접 넘겨 본다
    assert llm_factory.skip_reason(CHAIN.primary, "concept") == "안전 필터가 요청을 거부"
    assert llm_factory.skip_reason(CHAIN.primary, "stock") == ""
    assert llm_factory.available_candidates(CHAIN, models, "concept")[0][0].provider == "openai"


def test_failure_reasons_are_plain_words():
    reason = llm_factory.failure_reason
    assert reason("claude_cli · claude-opus-5-5: ClaudeCLIUsageLimit: You've hit your limit") == "구독 사용 한도"
    assert reason("claude_cli · x: ClaudeCLITimeout: 600초") == "시간 초과"
    assert reason("google · gemini-3.5-flash: ServiceUnavailable: 503 high demand") == "서버 과부하"
    assert reason("google · gemini-3.5-flash: ValueError: 구조화 출력 파싱 실패: None") == "형식에 맞지 않는 답"


# ─── 사용량 기록 ─────────────────────────────────────────────────────


def test_claude_cli_usage_recorded_with_team_and_agent_and_failures_skipped(models, set_keys, fake_cli, cli_logged_in,
                                                                           monkeypatch):
    set_keys("openai")
    events = []
    usage.set_sink(events.append)
    fake_cli([result_event("", structured_output={"analysis": "a", "picks": []})])
    with usage.scope(team_id="concept", run_id="r1"), usage.scope(agent_id="gm_editor"):
        call = llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "부장")
    assert call.spec.provider == "claude_cli"
    (e,) = events
    assert (e.provider, e.team_id, e.run_id, e.agent_id, e.cost_usd) == ("claude_cli", "concept", "r1", "gm_editor", 0.0)

    # 거부된 호출은 답이 없으므로 기록하지 않는다 (대안 모델 호출만 남는다 — 가짜 모델은 기록 콜백이 없음)
    events.clear()
    _fallback_to_fake(monkeypatch, Script(Plan=Plan(analysis="대안", picks=[])))
    fake_cli(refused(), code=1)
    with usage.scope(team_id="concept"):
        llm_factory.invoke_structured(CHAIN, models, Plan, [HumanMessage("q")], "부장")
    assert events == []


# ─── 실무자: 스킬 일부 · 산출물 범위 ─────────────────────────────────


SKILL = """## 스킬: demo
[역할]
에디터다.
[1단계·비교]
비교 표.
[출력 형식 견본]
```
[태그]한글명(English) (작성일: YY.MM.DD)
1. 정의 및 개념
```
[5단계·근거]
참고문헌 한 줄."""


def test_skill_sections_keep_only_listed_parts_and_ignore_code_blocks():
    out = skill_sections(SKILL, ["[역할]", "[5단계"])
    assert "에디터다." in out and "참고문헌 한 줄." in out and out.startswith("## 스킬: demo")
    assert "비교 표." not in out and "1. 정의 및 개념" not in out  # 견본 코드 블록 안의 '[태그]…' 는 절 제목이 아님
    assert skill_sections(SKILL, []) == SKILL  # 목록이 없으면 전문


@pytest.fixture
def concept_agents(ranks, models):
    team = load_team("concept")
    by_rank = {r.id: r for r in ranks}
    skills = load_skills("concept")
    return {n.id: OrgAgent(n, by_rank[n.rank], team, models, skills=skills) for n in team_org(team, ranks).walk()}


def test_concept_drafters_get_only_their_skill_parts(concept_agents):
    full = load_skills("concept")
    lead, gm, fact, lineage = (concept_agents[i] for i in ("lead", "gm_editor", "am_fact", "dgm_lineage"))
    assert lead.skills == full and gm.skills == full  # 최종본을 쓰고 점검하는 사람은 전문 (출력 형식 견본 포함)
    for agent, own, other in ((fact, "[2단계", "[1단계"), (lineage, "[1단계", "[2단계")):
        assert own in agent.skills and other not in agent.skills
        assert "[출력 형식 견본]" not in agent.skills and "[사전 제약 조건" in agent.skills
        assert len(agent.system_prompt()) < len(gm.system_prompt()) * 0.5
    # 스킬 절 제목이 바뀌어 아무것도 안 걸리면 실무자가 지침 없이 일하게 된다 — 적어 둔 절마다 실제로 있어야 한다
    for agent in (fact, lineage):
        for head in agent.node.skill_sections:
            assert any(line.startswith(head) for line in full.splitlines()), head


def test_concept_drafters_write_only_their_part(concept_agents):
    fact = concept_agents["am_fact"].draft_messages("[업무 브리프] 최종본 <final> 까지", "검증 로그")
    human = fact[-1].content
    assert "[내 산출물 범위 — 이것만 쓴다]" in human and "<final> 최종본·GitHub 요약·적용 계획 JSON 은 쓰지 않는다" in human
    assert "## 고칠 문장" in human
    assert "## 6.관계 초안" in concept_agents["dgm_lineage"].draft_messages("q", "t")[-1].content
    assert "[내 산출물 범위" not in concept_agents["gm_editor"].draft_messages("q", "t")[-1].content


def test_concept_team_speed_settings():
    team = load_team("concept")
    assert team.review_mode == "edit"
    assert team.verification["critic"]["scope"] == "final"  # 부장 통합 뒤 비평은 생략 (편집장 단계에서 1회)
    assert team.harness.final_only == ["concept_entry"]  # 형식 검사는 그대로 최종본에


# ─── 편집장 검토: 고칠 곳만 ──────────────────────────────────────────


def test_apply_text_edits_only_changes_unique_matches():
    text, skipped = apply_text_edits("가 나 다 나", [TextEdit(find="가", replace="A"), TextEdit(find="나", replace="B"),
                                                    TextEdit(find="없음", replace="C")])
    assert text == "A 나 다 나"
    assert skipped == ["2곳에 있음: '나'", "찾지 못함: '없음'"]


def edits(*pairs, verdict="revise", answer=""):
    return ReviewEdits(analysis="분석", issues=["오류"], score=7, verdict=verdict, revised_answer=answer, comment="c",
                       edits=[TextEdit(find=f, replace=r) for f, r in pairs])


def _edit_team(team, **extra) -> Team:
    return Team(**{**team.model_dump(), "review_mode": "edit", **extra})


def test_edit_mode_review_applies_edits_instead_of_rewriting(org, ranks, team, models, use_script):
    script = use_script(Script(text="초안 오타 있음", ReviewEdits=edits(("오타", "수정"), ("없는 문장", "x"))))
    graph = build_office_graph(org, ranks, _edit_team(team), models, SETTINGS, NO_CRITIC)
    steps = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"]

    review = steps[-1]
    assert review.kind == "review" and review.content == "초안 수정 있음" and review.review.revised_answer == "초안 수정 있음"
    assert "고칠 곳 1/2개 적용 (전문을 다시 쓰지 않음)" in review.verification_log
    assert any("찾지 못함" in log for log in review.verification_log)
    prompt = script.prompts("ReviewEdits")[0]
    assert "edits 에 고칠 곳마다" in prompt and "전면 재작성이 꼭 필요할 때만" in prompt
    assert script.prompts("ReviewResult") == []  # 검토는 고칠 곳 스키마로만


def test_edit_mode_full_rewrite_and_approve_still_work(org, ranks, team, models, use_script):
    use_script(Script(text="초안", ReviewEdits=[edits(answer="전면 재작성본"), edits(verdict="approve")]))
    graph = build_office_graph(org, ranks, _edit_team(team), models, SETTINGS, NO_CRITIC)
    first = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert first.content == "전면 재작성본" and "고칠 곳 대신 전문을 다시 씀" in first.verification_log
    second = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert second.content == "초안"  # 승인 → 하위 답변 그대로


def test_edit_mode_harness_retry_edits_the_previous_result(org, ranks, team, models, use_script):
    bad = "```python\ndef f(:\n    return 1\n```"
    script = use_script(Script(text=bad, ReviewEdits=[edits(("return 1", "return 2")), edits(("def f(:", "def f():"))]))
    harness = {"validators": ["python_syntax"], "final_only": ["python_syntax"], "max_retries": 2}
    graph = build_office_graph(org, ranks, _edit_team(team, harness=harness), models, SETTINGS, NO_CRITIC)
    final = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]

    # 두 번째 고칠 곳은 첫 번째 결과(return 2)에 적용된다 — 하위 원문으로 되돌아가지 않는다
    assert final.content == "```python\ndef f():\n    return 2\n```" and final.harness[0].passed and final.attempts == 2
    assert "문법 오류" in script.prompts("ReviewEdits")[-1]
    assert sum("고칠 곳 1/1개 적용" in log for log in final.verification_log) == 2


def test_integrate_ignores_edit_mode(org, ranks, team, models, use_script):
    script = use_script(Script(text="초안", Delegation=make_delegation("am_fact", "staff_draft"),
                               ReviewResult=edits(answer="통합본")))
    graph = build_office_graph(org, ranks, _edit_team(team), models, SETTINGS, NO_CRITIC)
    steps = [p for k, p in stream_office(graph, org, "질문", "dgm_research", brief=BRIEF) if k == "step"]
    assert steps[-1].kind == "integrate" and steps[-1].content == "통합본"  # 여러 명 통합은 전문 그대로
    assert script.prompts("ReviewEdits") == []


def test_claude_cli_status_cache_untouched_by_refusal(fake_cli, monkeypatch):
    """거부는 로그인·설치 문제가 아니다 — 구독 상태를 '사용 불가'로 바꾸지 않는다."""
    claude_cli._status_cache["C:/fake/claude.exe"] = (claude_cli.time.monotonic(), "ok")
    fake_cli(refused(), code=1)
    with pytest.raises(ClaudeCLIRefused):
        ClaudeCLIChatModel().invoke("hi")
    assert claude_cli.cli_status() == "ok"
