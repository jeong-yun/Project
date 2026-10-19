"""에이전트 한 명(OrgAgent)의 프롬프트·호출·대안 모델 전환 테스트."""

import pytest
from langchain_core.language_models import FakeListChatModel

from core import llm_factory
from core.agent import NoAvailableModelError, OrgAgent
from core.config import subtree
from core.schemas import AgentAnswer, ModelSpec
from tests.fakes import Script, make_delegation, make_review

SPEC = ModelSpec(provider="openai", model="x")


class BrokenModel:
    def stream(self, messages, config=None):
        raise RuntimeError("model not found")


def agent_for(node_id, org, ranks, team, models, **kw):
    node = subtree(org, node_id)
    names = {n.id: n.name or n.id for n in org.walk()}
    return OrgAgent(node, next(r for r in ranks if r.id == node.rank), team, models, names=names, **kw)


def child(node_id, content):
    return AgentAnswer(node_id=node_id, rank_id="staff", content=content, model=SPEC)


def test_system_prompt_has_persona_specialty_skills_lessons(org, ranks, team, models):
    agent = agent_for("staff_draft", org, ranks, team, models, skills="결론을 먼저 쓴다")
    prompt = agent.system_prompt(lessons=["출처를 함께 적을 것"])
    assert team.personas["staff"] in prompt
    assert "기본 개념 초안 작성" in prompt  # 전문 분야
    assert "결론을 먼저 쓴다" in prompt  # 스킬
    assert "출처를 함께 적을 것" in prompt  # 교훈


def test_draft_streams_and_falls_back(org, ranks, team, models, set_keys, monkeypatch):
    agent = agent_for("staff_draft", org, ranks, team, models)
    primary, backup = (s.provider for s in agent.rank_models.candidates[:2])
    set_keys(primary, backup)
    monkeypatch.setattr(
        llm_factory, "create_chat_model",
        lambda spec, m: BrokenModel() if spec.provider == primary else FakeListChatModel(responses=["대체 답변"]),
    )
    stream = agent.stream_draft(agent.draft_messages("질문", "업무"))
    assert "".join(stream) == "대체 답변"
    assert stream.result.model.provider == backup and stream.result.is_fallback
    assert stream.result.kind == "draft" and stream.result.node_id == "staff_draft"


def test_no_key_raises(org, ranks, team, models, set_keys):
    set_keys()
    agent = agent_for("staff_draft", org, ranks, team, models)
    with pytest.raises(NoAvailableModelError, match="API 키"):
        agent.draft(agent.draft_messages("질문", "업무"))


def test_plan_lists_children_and_returns_delegation(org, ranks, team, models, use_script):
    script = use_script(Script(Delegation=make_delegation("gm_research")))
    plan = agent_for("lead", org, ranks, team, models).plan("질문", "목표")
    assert plan.kind == "plan" and plan.delegation.assignments[0].child_id == "gm_research"
    assert "gm_action" in script.prompts("Delegation")[0]  # 하위 목록이 프롬프트에 있다


def test_review_approve_keeps_child_answer(org, ranks, team, models, use_script):
    use_script(Script(ReviewResult=make_review("approve", 9, answer="무시될 텍스트")))
    result = agent_for("gm_action", org, ranks, team, models).review([], [child("am_action", "원래 답변")])
    assert result.kind == "review" and result.content == "원래 답변"


def test_review_prompt_is_adversarial_with_rubric(org, ranks, team, models):
    rubric = {"scale": "1~10", "levels": ["9~10: 최상"], "examples": [{"question": "q", "answer": "a", "score": 2, "reason": "혼동"}]}
    agent = agent_for("gm_action", org, ranks, team, models, rubric=rubric)
    text = "\n".join(m.content for m in agent.review_messages("브리프", "업무", [child("am_action", "답")]))
    assert "적대적 리뷰어" in text and "채점 예시" in text and "혼동" in text


def test_integrate_falls_back_to_concatenation_when_empty(org, ranks, team, models, use_script):
    use_script(Script(ReviewResult=make_review("approve", 8, answer="")))
    result = agent_for("dgm_research", org, ranks, team, models).review(
        [], [child("am_fact", "팩트"), child("staff_draft", "초안")]
    )
    assert result.kind == "integrate"
    assert "팩트" in result.content and "초안" in result.content
