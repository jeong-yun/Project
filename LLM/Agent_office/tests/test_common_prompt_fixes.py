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
from tests.test_workflow import full_script, run

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


# ─── C7 요청 정리·통합: 웹 검색이 꺼져 있으면 출처·최신 뉴스 기준을 넣지 않는다 ──────


def brief_system(script, models, **kw) -> str:
    brief_mod.draft_brief("HBF 가 뭐야?", models, **kw)
    return script.calls[-1][1][0].content


def test_brief_does_not_ask_for_sources_when_web_search_is_off(models, use_script):
    script = use_script(Script(DesiredState=DesiredState(goal="g", context="", success_criteria=["c"], constraints=[],
                                                         output_format="", assumptions=[])))
    on = brief_system(script, models, use_web=True)
    assert "'출처를 제시한다'" in on and "최신 동향(뉴스)" in on  # 검색을 쓰면 전과 같다
    assert brief_system(script, models) == on  # 기본값 = 전과 같은 글
    off = brief_system(script, models, use_web=False)
    assert "'출처를 제시한다' 같은 기준을 포함한다" not in off and "최신 동향(뉴스)" not in off
    assert "웹 검색을 쓰지 않는다" in off and "확인 필요" in off


def test_integration_mentions_web_news_only_when_web_results_are_given(org, ranks, team, models):
    agent = node_agent(org, ranks, team, models, "gm_action")
    kids = [child("am_action"), child("staff_case")]
    web = f"## 🌐 웹 검색 결과\n{WEB_BEGIN}\n[1] 자료\n{WEB_END}"
    assert "웹 검색의 최신 뉴스·동향 기준으로" in agent.review_messages("브리프", "업무", kids, web)[0].content
    for no_web in ("", "## 시장 데이터\n현재가 100"):  # 검색을 안 썼거나 데이터 툴 결과만 있을 때
        system = agent.review_messages("브리프", "업무", kids, no_web)[0].content
        assert "웹 검색의 최신 뉴스" not in system and "근거가 가장 뚜렷한 해석을 앞에 둔다" in system


# ─── C9 실무자: 업무를 나눠 받았으면 [완료 기준]은 팀 전체 기준이라고 알린다 ─────────


def test_worker_with_a_delegated_task_knows_criteria_are_for_the_whole_team(org, ranks, team, models, use_script):
    agent = node_agent(org, ranks, team, models, "staff_case")
    assert not agent.node.output_scope  # 산출물 범위가 없는 팀 (기본 팀)
    human = agent.draft_messages("브리프 [완료 기준 — 모두 만족해야 함]", "사례 두 개 찾기", split=True)[-1].content
    assert "[상사가 맡긴 업무 — 이 범위에 집중]\n사례 두 개 찾기" in human
    assert "[완료 기준]은 팀 전체 최종 답변의 기준" in human and "맡은 업무만 쓰고" in human
    alone = agent.draft_messages("브리프", "목표")[-1].content  # 혼자 전부 맡으면 기준은 그대로 내 몫
    assert "팀 전체 최종 답변의 기준" not in alone
    # 실행에서: 팀장이 업무를 나눠 주면 실무자 초안에 붙고, 한 줄로 내려온 업무(대리 → 사원 1명)에는 붙지 않는다
    script = use_script(full_script())
    run(org, ranks, team, models)
    drafts = script.prompts("text")
    assert len(drafts) == 3 and all("팀 전체 최종 답변의 기준" in p for p in drafts)
    script = use_script(full_script())
    run(org, ranks, team, models, root="am_action")
    [only] = script.prompts("text")
    assert "[상사가 맡긴 업무 — 이 범위에 집중]\n목표" in only and "팀 전체 최종 답변의 기준" not in only


def test_worker_with_output_scope_gets_the_note_once(ranks, models):
    from core.config import load_skills, load_team, team_org

    concept = load_team("concept")
    org = team_org(concept, ranks)
    node = next(n for n in org.walk() if n.output_scope)
    agent = OrgAgent(node, next(r for r in ranks if r.id == node.rank), concept, models, skills=load_skills("concept"))
    human = agent.draft_messages("브리프", "맡은 일", split=True)[-1].content
    assert "[내 산출물 범위 — 이것만 쓴다]" in human and human.count("팀 전체 최종 답변의 기준") == 1


# ─── C6 교훈 만들기: 사람이 고친 곳을 코드가 뽑아 준다 (앞 1,200자만 주지 않는다) ───────


def lesson_prompt(final: str, original: str, edited: bool = True) -> list[str]:
    steps = [child("staff_case", original)]
    decision = HumanDecision(action="adopt", adopted_node_id="staff_case", final_answer=final, edited=edited, score=3)
    system, human = feedback.build_feedback_messages("질문", steps, decision, {"staff_case": "사원"}, load_team_general())
    return [system.content, human.content]


def load_team_general():
    from core.config import load_team

    return load_team("general")


def test_lesson_prompt_shows_the_lines_the_human_changed_even_far_down():
    lines = [f"{i}번 줄 — 본문 " + "가" * 40 for i in range(200)]  # 약 1만 자
    original = "\n".join(lines)
    changed = original.replace("180번 줄 — 본문", "180번 줄 — 사람이 고친 본문")
    system, human = lesson_prompt(changed, original)
    assert "사람이 고친 곳" in human
    assert "- 180번 줄 — 본문" in human and "+ 180번 줄 — 사람이 고친 본문" in human  # 뒤쪽을 고쳐도 보인다
    assert "3번 줄" not in human.split("사람이 고친 곳")[1]  # 바뀌지 않은 줄은 넣지 않는다
    assert "사람이 직접 수정한 최종 답변" not in human
    # 고친 곳이 아주 많으면 한도까지만 넣고 잘렸다고 알린다
    many = "\n".join(line + " (고침)" for line in lines)
    _, human = lesson_prompt(many, original)
    shown = human.split("사람이 고친 곳")[1]
    assert len(shown) < feedback.EDIT_DIFF_CHARS + 300 and "생략" in shown


def test_lesson_prompt_without_a_usable_diff_falls_back_to_the_answer():
    _, human = lesson_prompt("고친 답", "원래 답", edited=False)
    assert "사람이 고친 곳" not in human and "사람이 직접 수정한 최종 답변" not in human  # 고치지 않았으면 없다
    steps = [child("am_action", "다른 사람 결과")]  # 채택한 안의 원래 글을 찾지 못하면 전처럼 앞부분을 준다
    decision = HumanDecision(action="adopt", adopted_node_id="staff_case", final_answer="고친 답", edited=True, score=3)
    human = feedback.build_feedback_messages("질문", steps, decision, {}, load_team_general())[1].content
    assert "사람이 직접 수정한 최종 답변:\n고친 답" in human


def test_lesson_prompt_lets_the_model_skip_agents_with_nothing_to_learn():
    system, _ = lesson_prompt("답", "답", edited=False)
    assert "정확히 1개" not in system and "빼도 된다" in system
    # 빠진 에이전트는 전처럼 규칙 기반 교훈으로 채워진다 (tests/test_feedback.py test_generate_lessons_fills_missing_and_keeps_comment)
