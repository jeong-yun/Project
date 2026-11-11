"""공통 조립 프롬프트 고침 (2026-11-10 프롬프트 점검 C3~C10) — 조립된 글에 새 문장이 들어가고 어긋나던 문장이 없어졌는지.

모델을 부르지 않는다 (실행과 같은 조립 함수만 부른다). 그래서 '결과가 좋아진다'는 여기서 확인하지 못한다.
"""

from core import brief as brief_mod
from core import feedback, followup, verification
from core.agent import OrgAgent
from core.brief import DesiredState
from core.config import load_verification
from core.schemas import AgentAnswer, HumanDecision, ModelSpec
from core.search import WEB_BEGIN, WEB_END
from tests.fakes import Script, make_critic, make_review
from tests.test_followup import ANSWER, edit, head_review, judge, lead_agent

SPEC = ModelSpec(provider="openai", model="x")


def critic_system(evidence: str = "") -> str:
    return verification.build_critic_messages("q", "하위 결과", make_review(), ["기준"], None, evidence, "너는 연구부 부장이다.")[0].content


def node_agent(org, ranks, team, models, node_id: str, **kw) -> OrgAgent:
    node = next(n for n in org.walk() if n.id == node_id)
    return OrgAgent(node, next(r for r in ranks if r.id == node.rank), team, models, **kw)


def child(node_id: str, content: str = "답") -> AgentAnswer:
    return AgentAnswer(node_id=node_id, rank_id="staff", kind="draft", content=content, model=SPEC)


# ─── C3 부장 비평: 무엇을 채점하는지가 한 프롬프트 안에서 어긋나지 않는다 ─────────────


def test_critic_prompt_grades_both_the_review_logic_and_the_answer():
    for system in (critic_system(), critic_system("[1] 자료")):
        assert "논리만" not in system  # 전: "'검토 과정'의 논리만 채점" ↔ "근거 자료와 숫자·사실을 대조"
        assert "비평가(Critic)" in system and "① 검토자의 '지적 → 수정'" in system
        assert "② 수정 답변이 [검토 기준]" in system and "[근거 자료]" in system


# ─── C10 부장 비평·수정본 검토: 한도로 줄인 곳을 '빠졌다·자료에 없다'로 보지 않는다 ────


def test_head_prompts_say_trimmed_parts_are_not_missing(org, team, models):
    assert "생략" in verification.TRIM_NOTE and "지적하지 않는다" in verification.TRIM_NOTE
    assert verification.TRIM_NOTE in critic_system()
    long = "가" * 30000  # 실제로 줄어든 글의 표시와 안내 문장이 같은 낱말('생략')을 쓴다
    human = verification.build_critic_messages("q", "x", make_review(answer=long), [], answer_chars=1000)[1].content
    assert "생략 — 비평 입력 한도" in human
    review = followup.revision_messages(head_review(org, team, models), "원 질문", ANSWER, "줄여줘", edit("수정본"))[0].content
    assert verification.TRIM_NOTE in review  # 💬 수정본 검토도 같은 한도로 줄여 넣는다


# ─── C8 채점 루브릭: 용어 설명용 예시가 다른 종류의 결과물에 그대로 적용되지 않는다 ────


def test_review_rubric_puts_criteria_first_and_examples_as_score_anchors(org, ranks, team, models):
    rubric = load_verification()["review_rubric"]
    agent = node_agent(org, ranks, team, models, "am_action", rubric=rubric)
    system = agent.review_messages("브리프", "업무", [child("staff_case")])[0].content
    assert "근거·출처 표시, 빠진" not in system  # 전: 9~10점 조건에 늘 '근거·출처 표시' (가사·도면·diff 에도)
    assert "[완료 기준]을 완전히 충족" in system
    head = next(line for line in system.splitlines() if line.startswith("[채점 예시"))
    assert "점수 간격" in head and "[검토 기준]" in head and "형식" in head  # 예시는 점수 간격만, 무엇을 볼지는 검토 기준
    assert "이 기준점에 맞춰 채점한다" not in system
    assert "HBM" in system  # 예시 자체는 그대로 (팀 전용 루브릭은 아직 없다)


# ─── C4 수정 요청: 부장이 검토하는 기준을 팀장도 받는다 ────────────────────────


def test_lead_gets_the_criteria_the_head_reviews_with(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=judge(5)))
    followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [], head=head_review(org, team, models))
    lead, head = script.prompts("FollowupResult")[0], script.prompts("RevisionCritic")[0]
    assert "숫자에는 출처를 붙인다" in head  # 부장의 [검토 기준]
    assert "[검토 기준" in lead and "숫자에는 출처를 붙인다" in lead  # 팀장도 같은 기준을 본다
    # 기준이 없으면(부장 검토 없이 부르는 경우) 묶음도 없다
    plain = followup.build_messages(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [])[-1].content
    assert "[검토 기준" not in plain


# ─── C5 수정 요청: 부장 불합격 뒤 팀장에게 가는 글이 팀장의 출력 칸과 맞는다 ──────────


def test_lead_refix_message_matches_followup_output_fields(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=[edit("수정본 A"), edit("수정본 B")],
                               RevisionCritic=[judge(2, ["요청한 표가 빠짐"], severity="severe"), judge(5)]))
    followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "표 추가", [], head=head_review(org, team, models))
    again = script.calls[2][1][-1].content  # 팀장 → 부장 → 팀장(다시 고침): 마지막 사람 메시지 = 되돌림 글
    assert "요청한 표가 빠짐" in again and "changes" in again and "revised_answer" in again
    assert "점수는 마지막에" not in again and "분석(analysis) 맨 앞" not in again  # 본 업무용 문구 (점수 칸이 없는 출력)
    assert "사람의 수정 요청은 그대로" in again


def test_main_correction_message_keeps_reasons_out_of_the_three_sentence_limit():
    text = verification.correction_message(make_critic(2, violations=["A 가 B 규칙에 어긋남"], severity="severe"))
    assert "A 가 B 규칙에 어긋남" in text and "점수는 마지막에 매긴다" in text  # 본 업무는 그대로
    assert "3문장 제한에 넣지 않는다" in text  # 지적이 많으면 'analysis 3문장 이내'와 부딪히던 것
