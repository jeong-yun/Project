"""공통 조립 프롬프트 고침 (2026-11-10 프롬프트 점검 C3~C19) — 조립된 글에 새 문장이 들어가고 어긋나던 문장이 없어졌는지.

모델을 부르지 않는다 (실행과 같은 조립 함수만 부른다). 그래서 '결과가 좋아진다'는 여기서 확인하지 못한다.
"""

from core import brief as brief_mod
from core import building, feedback, followup, scope, step_mode, verification
from core.agent import OrgAgent
from core.brief import DesiredState
from core.config import date_label, load_verification, today
from core.schemas import AgentAnswer, HumanDecision, ModelSpec, ReviewEdits, ReviewResult
from core.search import WEB_BEGIN, WEB_END
from tests.fakes import Script, make_critic, make_review
from tests.test_followup import ANSWER, edit, fix, head_review, judge, lead_agent
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
        assert "마지막으로 확인하는 비평가" in system and "① 검토자의 '지적 → 수정'" in system
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


# ─── C11 부장 프롬프트 3종(비평·수정본 검토·직접 수정): 오늘 날짜와 '한국어로 쓴다' ──────


def test_head_prompts_carry_today_and_language(org, team, models, monkeypatch):
    monkeypatch.setenv("AGENT_OFFICE_TODAY", "2026-11-12")
    head = head_review(org, team, models)
    for system in (critic_system(),
                   followup.revision_messages(head, "원 질문", ANSWER, "줄여줘", edit("수정본"))[0].content,
                   followup.head_fix_messages(head, "원 질문", "줄여줘", "수정본", ["표 제목에 기준일 넣기"])[0].content):
        assert system.startswith("너는 연구부 부장이다.")  # 부장 관점이 맨 앞인 것은 그대로
        assert f"오늘 날짜: {date_label(today())}" in system and "2026-11-12" in system and "한국어로 쓴다" in system
    plain = verification.build_critic_messages("q", "x", make_review(), [])[0].content  # 부장 관점이 없어도 들어간다
    assert plain.startswith("오늘 날짜: 2026-11-12")


# ─── C12 스킬("항상 따른다")과 교훈("반드시 반영")이 어긋나면 어느 쪽이 먼저인지 ────────


def test_lessons_say_they_win_over_skills_when_they_disagree(org, ranks, team, models):
    agent = node_agent(org, ranks, team, models, "am_action", skills="[스킬] 표는 3열로")
    system = agent.system_prompt(["표는 2열로 줄인다"])
    head = next(line for line in system.splitlines() if line.startswith("지난 업무에서 받은 피드백"))
    assert "반드시 반영" in head and "스킬" in head and "어긋나면 이 피드백을 따른다" in head
    assert system.index("[팀 업무 지침(스킬) — 항상 따른다]") < system.index(head)  # 순서는 그대로 (스킬 → 교훈)
    bare = node_agent(org, ranks, team, models, "am_action").system_prompt(["표는 2열로 줄인다"])  # 스킬이 없으면 견줄 것도 없다
    assert "지난 업무에서 받은 피드백 (반드시 반영):" in bare
    assert "지난 업무에서 받은 피드백" not in agent.system_prompt()


# ─── C13 '고칠 곳만 바꾸기'(edit) 검토: 출력 칸 설명이 시스템 지시와 같은 말을 한다 ──────


def test_edit_mode_field_description_agrees_with_the_instruction():
    desc = ReviewEdits.model_json_schema()["properties"]["revised_answer"]["description"]
    assert "revise 이면 지적을 모두 반영한" not in desc  # 전: 칸 설명은 '완성 답변 전문', 지시는 '전문을 다시 쓰지 말라'
    assert "빈 문자열" in desc and "edits" in desc and "전면 재작성" in desc
    # 전문을 쓰는 방식(다른 팀)의 칸 설명은 그대로
    assert "완성 답변 전문" in ReviewResult.model_json_schema()["properties"]["revised_answer"]["description"]


# ─── C14 수정본 검토: '근거 없이 새로 넣은 주장'을 한 칸에만 쓰게 한다 ──────────────────


def test_revision_review_puts_unsupported_claims_in_one_field(org, team, models):
    system = followup.revision_messages(head_review(org, team, models), "원 질문", ANSWER, "줄여줘", edit("수정본"))[0].content
    line = next(x for x in system.splitlines() if x.startswith("위 세 가지에 어긋나는 것은 rule_violations"))
    assert "logical_leaps" in line and "다시 쓰지 않는다" in line
    assert "네가 바로 직접 고치게 되니" not in system  # 상한 0·마지막 회차에는 사실이 아니던 말
    assert "바로 고쳐질 수 있으니" in system


# ─── C15 부장 직접 수정: 팀장이 받는 팀 업무 지침(스킬)을 부장도 받는다 ─────────────────


def test_head_fix_gets_the_team_skills_the_lead_gets(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fix("부장이 고친 수정본"),
                               RevisionCritic=[judge(5, extra=["표 제목에 기준일 넣기"]), judge(5)]))
    followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [], head=head_review(org, team, models))
    lead, head_fix = script.prompts("FollowupResult")[0], script.prompts("HeadFix")[0]
    assert "[스킬] 형식 유지" in lead and "[스킬] 형식 유지" in head_fix  # 같은 스킬 글
    assert "[팀 업무 지침(스킬)" in head_fix and "작업 절차" in head_fix  # 형식 규칙으로만 쓰라는 안내
    assert all("[스킬] 형식 유지" not in p for p in script.prompts("RevisionCritic"))  # 검토 프롬프트는 그대로
    bare = " ".join(m.content for m in followup.head_fix_messages(head_review(org, team, models), "q", "r", "수정본", ["점"]))
    assert "스킬" not in bare  # 스킬이 없는 팀은 전과 같은 글


# ─── C16 진행 방향 제안: 예시가 한 분야(자소서)에 치우치지 않는다 ──────────────────────


def test_direction_examples_are_not_tied_to_one_kind_of_work():
    system = step_mode.direction_messages("시스템", "브리프")[0].content
    title = step_mode.DirectionOption.model_json_schema()["properties"]["title"]["description"]
    for text in (system, title):
        assert "기술 중심" not in text and "리더십" not in text
        assert "범위를 넓게" in text


# ─── C17 팀 배정: 부서 화면의 🧭 배정에도 '다루는/다루지 않는 질문'이 들어간다 ──────────


def test_dept_routing_prompt_shows_team_scope_once():
    from core.config import list_teams

    teams = list_teams()
    cands = [t for t in teams if t.id in ("travel_domestic", "travel_abroad")]
    dept = next(d for d in building.departments([t.id for t in teams], building.load_config()) if "travel_abroad" in d.teams)
    system = building.route_messages("오사카 3박 4일", dept, cands)[0].content
    assert "다루는 질문: 해외 여행" in system and "다루지 않음" in system
    # 범위 확인(scope.router_check)도 같은 소개를 쓴다 — 두 번 붙지 않는다
    seen = {}
    scope.router_check("오사카 3박 4일", cands[0], cands, None,
                       route=lambda q, d, c, m: seen.update(prompt=building.route_messages(q, d, c)[0].content)
                       or {"team_id": cands[0].id, "how": "llm"})
    assert seen["prompt"].count("다루는 질문: 해외 여행") == 1


# ─── C18 말투·용어: 부장 뒤에 '너는 비평가다'가 또 오지 않고, 영어 덧말이 없다 ──────────


def test_role_lines_do_not_repeat_identity_or_carry_english_glosses(org, ranks, team, models):
    critic = critic_system()
    assert critic.count("너는 ") == 1 and "(Critic)" not in critic  # '너는 ○○부 부장이다' 한 번뿐
    agent = node_agent(org, ranks, team, models, "am_action")
    review = agent.review_messages("브리프", "업무", [child("staff_case")])[0].content
    assert "적대적 리뷰어" in review and "Adversarial" not in review
    assert "너는 적대적 리뷰어" not in review  # 페르소나("당신은 …입니다") 뒤에 정체를 한 번 더 말하지 않는다
    assert "취약점·논리적 오류·근거 없는 주장·[완료 기준] 미충족 항목을 먼저 찾아낸 뒤 판단한다" in review
