"""결정 전 후속 대화(core/followup.py) — 질문/수정 자동 판단, 수정본 하네스 검사, 👔 부장 최종 검토, 결정 코멘트 요약."""

from core import call_monitor, dept_heads, followup, member_models, usage, verification
from core.agent import OrgAgent
from core.followup import FollowupResult, RevisionCritic
from tests.fakes import Script

ANSWER = "## 결론\n답변 본문\n\n```python\nprint(1)\n```"


def lead_agent(org, ranks, team, models) -> OrgAgent:
    rank = next(r for r in ranks if r.id == org.rank)
    return OrgAgent(org, rank, team, models, skills="[스킬] 형식 유지")


def ask(text: str) -> FollowupResult:
    return FollowupResult(analysis="질문", revised=False, reply=text)


def edit(answer: str, changes=("정의를 줄임",)) -> FollowupResult:
    return FollowupResult(analysis="수정 요청", revised=True, reply="정의를 줄였습니다.", changes=list(changes),
                          revised_answer=answer)


def test_question_answers_without_new_version(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=ask("3번은 [2] 자료가 근거입니다.")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "3번 근거가 뭐야?", [],
                            evidence="## 🌐 웹 검색 결과\n[2] 자료")
    assert not turn.revised and turn.version == 0 and turn.content == ""
    assert "근거" in turn.reply
    prompt = script.prompts("FollowupResult")[0]
    # 현재 결과·근거 자료·팀 스킬이 모두 들어가야 근거를 대고 형식을 지킬 수 있다
    assert "[현재 결과]" in prompt and "[2] 자료" in prompt and "[스킬] 형식 유지" in prompt


def test_revision_is_versioned_and_history_is_sent(org, ranks, team, models, use_script):
    agent = lead_agent(org, ranks, team, models)
    script = use_script(Script(FollowupResult=[edit("수정본 1"), edit("수정본 2")]))
    first = followup.respond(agent, "원 질문", ANSWER, "정의를 줄여줘", [])
    second = followup.respond(agent, "원 질문", first.content, "표 추가해줘", [first])
    assert (first.version, second.version) == (1, 2) and second.content == "수정본 2"
    prompt = script.prompts("FollowupResult")[1]
    assert "정의를 줄여줘" in prompt and "수정본 v1" in prompt and "수정본 1" in prompt  # 대화 이력 + 최신 수정본


def test_revision_failing_harness_is_retried_with_feedback(org, ranks, team, models, use_script):
    broken = "```python\nprint(\n```"
    script = use_script(Script(FollowupResult=[edit(broken), edit(ANSWER)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "코드 바꿔줘", [],
                            validators=["python_syntax"], options={}, max_retries=1)
    assert turn.revised and turn.content == ANSWER
    assert all(h.passed for h in turn.harness) and any("하네스 실패" in x for x in turn.log)
    assert len(script.prompts("FollowupResult")) == 2  # 실패 → 오류를 돌려주고 한 번 더


def test_revision_still_failing_is_marked(org, ranks, team, models, use_script):
    broken = "```python\nprint(\n```"
    use_script(Script(FollowupResult=edit(broken)))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "코드 바꿔줘", [],
                            validators=["python_syntax"], options={}, max_retries=1)
    assert turn.revised and not all(h.passed for h in turn.harness)
    assert any("검사 실패" in x for x in turn.log)


def test_empty_revision_counts_as_answer_only(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=FollowupResult(analysis="?", revised=True, reply="고칠 곳이 없습니다.")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "고쳐줘", [])
    assert not turn.revised and turn.version == 0


def test_decision_note_summarizes_requests():
    turns = [followup.FollowupTurn(request="3번 근거?", reply="..."),
             followup.FollowupTurn(request="정의 줄여줘", reply="...", revised=True, content="x", version=1)]
    note = followup.decision_note(turns)
    assert "1) 3번 근거? (질문)" in note and "2) 정의 줄여줘 → 수정본 v1" in note
    assert followup.decision_note([]) == ""


# ─── 👔 수정본은 부서 부장이 최종 검토한다 (2026-11-10) ─────────────────────────────


def head_review(org, team, models, **critic_cfg) -> followup.HeadReview:
    """본 업무와 같은 재료 — 부장 모델(critic_plan)·비평 설정(critic_settings)·검토 기준."""
    chains = member_models.team_rank_models(org, models, team, {})
    head = dept_heads.DeptHead("research", "연구부", "🔬", "연구부 부장", "근거와 출처를 먼저 본다.")
    return followup.HeadReview(plan=dept_heads.critic_plan(models, team, chains, org.id, head),
                               cfg=verification.critic_settings({"critic": critic_cfg}),
                               criteria=["숫자에는 출처를 붙인다"])


def judge(score=5, violations=(), extra=(), severity=None) -> RevisionCritic:
    return RevisionCritic(analysis="요청 반영 점검", logical_leaps=[], rule_violations=list(violations),
                          severity=severity or ("minor" if violations else "none"), score=score, extra_fixes=list(extra))


def test_head_reviews_revision_and_finds_extra_fixes(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본 전문"), RevisionCritic=judge(5, extra=["표 제목에 기준일 넣기"])))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "정의를 줄여줘", [],
                            evidence="## 🌐 웹 검색 결과\n[2] 자료", head=head_review(org, team, models))
    assert turn.revised and turn.content == "수정본 전문"
    assert turn.critic.score == 5 and turn.critic_passed and turn.critic.extra_fixes == ["표 제목에 기준일 넣기"]
    assert turn.critic_by == "🔬 연구부 부장" and turn.critic_model is not None  # 누가·어느 모델로 검토했는지
    assert len(script.prompts("FollowupResult")) == 1 and len(script.prompts("RevisionCritic")) == 1  # 평소 호출 +1
    prompt = script.prompts("RevisionCritic")[0]
    assert "너는 연구부 부장이다" in prompt  # 본 업무와 같은 부장 관점
    for part in ("[사용자의 수정 요청]\n정의를 줄여줘", "[수정 전 결과]", "답변 본문", "[수정본]\n수정본 전문",
                 "숫자에는 출처를 붙인다", "[2] 자료", "extra_fixes"):
        assert part in prompt


def test_question_only_is_not_reviewed(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=ask("근거는 [2] 입니다."), RevisionCritic=judge(1)))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "근거가 뭐야?", [],
                            head=head_review(org, team, models))
    assert not turn.revised and turn.critic is None and script.prompts("RevisionCritic") == []


def test_team_with_critic_off_is_not_reviewed(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=judge(1)))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                            head=head_review(org, team, models, enabled=False))
    assert turn.revised and turn.critic is None and script.prompts("RevisionCritic") == []


def test_failed_review_makes_lead_fix_again_and_recheck_harness(org, ranks, team, models, use_script):
    broken = "```python\nprint(\n```"
    script = use_script(Script(FollowupResult=[edit("표 없는 수정본"), edit(broken), edit(ANSWER)],
                               RevisionCritic=[judge(2, ["요청한 표가 빠짐"], ["용어 통일"], "severe"), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "표 추가해줘", [],
                            validators=["python_syntax"], options={}, max_retries=1, head=head_review(org, team, models))
    assert turn.content == ANSWER and turn.critic.score == 5 and turn.critic_passed
    assert all(h.passed for h in turn.harness)  # 다시 고친 수정본도 하네스를 통과해야 한다
    prompts = script.prompts("FollowupResult")
    assert len(prompts) == 3 and len(script.prompts("RevisionCritic")) == 2  # 수정 → (부장 불합격) 다시 고침 → 하네스 실패 → 다시
    assert "요청한 표가 빠짐" in prompts[1]  # 부장 지적을 팀장에게 돌려준다
    assert "용어 통일" not in prompts[1]  # 추가로 고칠 점은 자동으로 고치지 않는다 — 사용자가 고른다
    assert any("불합격(2/5)" in x and "다시 고침" in x for x in turn.log)


def test_still_failing_stops_at_limit_and_keeps_better_revision(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=[edit("수정본 A"), edit("수정본 B")],
                               RevisionCritic=[judge(3, ["표가 빠짐"], severity="severe"), judge(2, ["더 나빠짐"], severity="severe")]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "표 추가해줘", [],
                            head=head_review(org, team, models, max_self_corrections=1))
    assert len(script.prompts("FollowupResult")) == 2 and len(script.prompts("RevisionCritic")) == 2  # 상한 1회 — 끝없이 돌지 않는다
    assert turn.content == "수정본 A" and turn.critic.score == 3 and not turn.critic_passed  # 나아지지 않으면 처음 수정본
    assert any("불합격" in x for x in turn.log)
    # 상한 0 이면 검토만 하고 다시 고치지 않는다
    script = use_script(Script(FollowupResult=edit("수정본 A"), RevisionCritic=judge(2, ["표가 빠짐"], severity="severe")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "표 추가해줘", [],
                            head=head_review(org, team, models, max_self_corrections=0))
    assert len(script.prompts("FollowupResult")) == 1 and not turn.critic_passed


def test_head_model_failure_keeps_revision_and_says_why(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=edit("수정본")))  # 부장 모델이 모두 실패 (가짜 출력 없음)
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                            head=head_review(org, team, models))
    assert turn.revised and turn.content == "수정본" and turn.critic is None
    assert any(x.startswith("👔 부장 검토를 하지 못했습니다 — ") for x in turn.log)


def test_user_skip_goes_without_head_review(org, ranks, team, models, use_script):
    call_monitor.reset()
    script = use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=judge(1)))
    with usage.scope(run_id="r-skip"):
        call_monitor.skip("r-skip", "critic")
        turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                                head=head_review(org, team, models))
    call_monitor.reset()
    assert turn.revised and turn.critic is None and script.prompts("RevisionCritic") == []
    assert any("건너뜀" in x for x in turn.log)


def test_review_text_for_screen_and_next_request():
    turn = followup.FollowupTurn(request="줄여줘", reply="줄였습니다", revised=True, content="x", version=1,
                                 critic=judge(4, ["2장 숫자에 출처가 없음"], ["표 제목에 기준일 넣기", "용어 통일"]),
                                 critic_by="🔬 연구부 부장", critic_passed=True)
    text = followup.head_review_markdown(turn, "🔬 연구부 부장 (gpt-5.5)")
    for part in ("🔬 연구부 부장 (gpt-5.5)", "4/5", "합격", "2장 숫자에 출처가 없음", "추가로 고칠 점", "표 제목에 기준일 넣기"):
        assert part in text
    request = followup.next_request(turn)  # 합격 — 추가로 고칠 점만 다음 수정 요청으로
    assert "표 제목에 기준일 넣기" in request and "용어 통일" in request and "출처가 없음" not in request
    turn.critic_passed = False  # 불합격으로 남았으면 지적도 함께 보낸다
    assert "출처가 없음" in followup.next_request(turn) and "불합격" in followup.head_review_markdown(turn, "부장")
    turn.critic, turn.critic_passed = judge(5), True
    assert followup.next_request(turn) == ""  # 고칠 점이 없으면 버튼도 없다
    assert followup.head_review_markdown(followup.FollowupTurn(request="q", reply="a"), "") == ""


def test_decision_note_leaves_out_head_review():
    """교훈은 사람의 요구에서 만든다 — 부장(모델)의 검토 의견은 결정 코멘트에 넣지 않는다."""
    turn = followup.FollowupTurn(request="줄여줘", reply="...", revised=True, content="x", version=1,
                                 critic=judge(4, extra=["용어 통일"]), critic_by="부장")
    note = followup.decision_note([turn])
    assert "줄여줘 → 수정본 v1" in note and "용어 통일" not in note and "요청 반영 점검" not in note
