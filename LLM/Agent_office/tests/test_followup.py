"""결정 전 후속 대화(core/followup.py) — 질문/수정 자동 판단, 수정본 하네스 검사, 👔 부장 최종 검토, 결정 코멘트 요약."""

from core import call_monitor, dept_heads, followup, member_models, usage, verification
from core.agent import OrgAgent
from core.followup import FollowupResult, HeadFix, RevisionCritic
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
    """상한 0 = 검토만 한다 — 추가로 고칠 점은 부장이 고치지 않고 화면·버튼으로 남는다."""
    script = use_script(Script(FollowupResult=edit("수정본 전문"), RevisionCritic=judge(5, extra=["표 제목에 기준일 넣기"])))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "정의를 줄여줘", [],
                            evidence="## 🌐 웹 검색 결과\n[2] 자료", head=head_review(org, team, models, max_self_corrections=0))
    assert turn.revised and turn.content == "수정본 전문"
    assert turn.critic.score == 5 and turn.critic_passed and turn.critic.extra_fixes == ["표 제목에 기준일 넣기"]
    assert turn.auto_fixed == [] and "표 제목에 기준일 넣기" in followup.next_request(turn)
    assert turn.critic_by == "🔬 연구부 부장" and turn.critic_model is not None  # 누가·어느 모델로 검토했는지
    assert len(script.prompts("FollowupResult")) == 1 and len(script.prompts("RevisionCritic")) == 1  # 평소 호출 +1
    prompt = script.prompts("RevisionCritic")[0]
    assert "너는 연구부 부장이다" in prompt  # 본 업무와 같은 부장 관점
    for part in ("[사용자의 수정 요청]\n정의를 줄여줘", "[수정 전 결과]", "답변 본문", "[수정본]\n수정본 전문",
                 "숫자에는 출처를 붙인다", "[2] 자료", "extra_fixes"):
        assert part in prompt
    assert "[부장이 직접 고친 점]" not in prompt  # 첫 검토에는 없다


# ─── 🛠 '추가로 고칠 점'은 부장이 직접 고치고 다시 검토한다 (2026-11-10 — 버튼을 누르지 않아도) ───


def fix(text: str) -> HeadFix:
    return HeadFix(revised_answer=text)


def test_head_fixes_extra_points_itself_and_reviews_again(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fix("부장이 고친 수정본"),
                               RevisionCritic=[judge(5, extra=["표 제목에 기준일 넣기"]), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "정의를 줄여줘", [],
                            evidence="## 🌐 웹 검색 결과\n[2] 자료", head=head_review(org, team, models))
    assert turn.content == "부장이 고친 수정본" and turn.version == 1  # 부장이 고친 글이 같은 수정본(v1)이 된다
    assert turn.changes == ["정의를 줄임"]  # 팀장이 요청으로 바꾼 점은 그대로
    assert turn.critic_passed and turn.critic.extra_fixes == [] and turn.auto_fixed == ["표 제목에 기준일 넣기"]
    assert followup.next_request(turn) == ""  # 남은 점이 없으면 버튼도 없다 — 사용자에게 다시 묻지 않는다
    # 호출: 팀장 1 (다시 부르지 않는다) + 부장 검토 1 + 부장 직접 수정 1 + 부장 다시 검토 1
    assert len(script.prompts("FollowupResult")) == 1 and len(script.prompts("HeadFix")) == 1
    reviews = script.prompts("RevisionCritic")
    assert len(reviews) == 2
    prompt = script.prompts("HeadFix")[0]
    for part in ("너는 연구부 부장이다", "[사용자의 수정 요청]\n정의를 줄여줘", "[수정본]\n수정본 A",
                 "[추가로 고칠 점]\n- 표 제목에 기준일 넣기", "숫자에는 출처를 붙인다", "[2] 자료", "완성 전문"):
        assert part in prompt
    # 다시 검토는 처음과 같은 온전한 검토 — 부장이 직접 고친 점만 더 알려 준다 ('요청과 무관한 변경'으로 지적하지 않게)
    for part in ("[부장이 직접 고친 점]\n- 표 제목에 기준일 넣기", "[수정본]\n부장이 고친 수정본", "extra_fixes 에는", "[수정 전 결과]"):
        assert part in reviews[1]
    assert any("직접 고침" in x for x in turn.log)
    text = followup.head_review_markdown(turn, "🔬 연구부 부장 (gpt-5.5)")
    assert "🔬 연구부 부장 (gpt-5.5)이(가) 추가로 고칠 점 1개를 직접 고쳤습니다" in text and "표 제목에 기준일 넣기" in text
    assert "**추가로 고칠 점**" not in text  # 남은 점이 없다


def test_no_extra_points_means_no_extra_calls(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=judge(5), HeadFix=fix("쓰이면 안 되는 글")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                            head=head_review(org, team, models))
    assert turn.content == "수정본" and turn.auto_fixed == [] and script.prompts("HeadFix") == []
    assert len(script.prompts("FollowupResult")) == 1 and len(script.prompts("RevisionCritic")) == 1  # 전과 같은 +1


def test_head_fix_is_recorded_as_self_correct_call(org, ranks, team, models, use_script):
    """⏱ 호출 기록·진행 막대: 부장의 직접 수정은 '🔁 자가 교정'(self_correct)으로 남고, 누가 했는지에 부장 이름이 붙는다."""
    call_monitor.reset()
    use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fix("부장이 고친 수정본"),
                      RevisionCritic=[judge(5, extra=["용어 통일"]), judge(5)]))
    with usage.scope(run_id="r-kinds"):
        followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [], head=head_review(org, team, models))
    calls = call_monitor.calls("r-kinds")
    call_monitor.reset()
    assert [c.kind for c in calls] == ["followup", "critic", "self_correct", "critic"]
    assert "연구부 부장" in calls[2].who and "직접 수정" in calls[2].who


def test_head_fix_shares_the_limit_and_leaves_what_the_last_review_found(org, ranks, team, models, use_script):
    def run(limit):
        script = use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=[fix("부장 수정 1"), fix("부장 수정 2")],
                                   RevisionCritic=[judge(5, extra=["표 제목에 기준일 넣기"]), judge(5, extra=["용어 통일"]), judge(5)]))
        return script, followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "정의를 줄여줘", [],
                                        head=head_review(org, team, models, max_self_corrections=limit))

    script, turn = run(1)  # 상한 1 — 다시 검토에서 또 추가 점이 나왔지만 더 돌지 않는다
    assert len(script.prompts("HeadFix")) == 1 and len(script.prompts("RevisionCritic")) == 2
    assert turn.content == "부장 수정 1" and turn.critic_passed
    assert turn.auto_fixed == ["표 제목에 기준일 넣기"] and turn.critic.extra_fixes == ["용어 통일"]
    assert "용어 통일" in followup.next_request(turn)  # 상한에 닿고도 남은 점만 화면·버튼으로
    text = followup.head_review_markdown(turn, "부장")
    assert "직접 고쳤습니다" in text and "**추가로 고칠 점**" in text
    script, turn = run(2)  # 상한 2 — 한 번 더 돈다
    assert len(script.prompts("HeadFix")) == 2 and len(script.prompts("RevisionCritic")) == 3
    assert turn.content == "부장 수정 2" and turn.auto_fixed == ["표 제목에 기준일 넣기", "용어 통일"]
    assert turn.critic.extra_fixes == [] and followup.next_request(turn) == ""
    assert "[수정본]\n부장 수정 1" in script.prompts("HeadFix")[1]  # 두 번째는 자기가 고친 글 위에서 고친다


def test_head_fix_that_fails_review_keeps_earlier_revision(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fix("부장이 고친 수정본"),
                               RevisionCritic=[judge(5, extra=["용어 통일"]), judge(2, ["요청한 내용이 빠짐"], severity="severe")]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "정의를 줄여줘", [],
                            head=head_review(org, team, models))
    assert len(script.prompts("FollowupResult")) == 1 and len(script.prompts("HeadFix")) == 1
    assert turn.content == "수정본 A" and turn.critic.score == 5 and turn.critic_passed  # 더 나빠지면 앞 수정본
    assert turn.auto_fixed == [] and turn.critic.extra_fixes == ["용어 통일"]  # 고친 글을 버렸으니 그대로 남는다
    assert any("유지" in x for x in turn.log) and not any("불합격 상태로" in x for x in turn.log)
    assert "용어 통일" in followup.next_request(turn)


def test_head_cannot_fix_keeps_previous_revision_and_button(org, ranks, team, models, use_script):
    """부장 모델 실패 · 하네스 실패 · 잘린 글 · 빈 글 — 팀장 수정본을 그대로 내고 남은 점과 버튼을 남긴다 (다시 검토도 하지 않는다)."""
    broken = "```python\nprint(\n```"
    for outputs, why in (({}, ""), ({"HeadFix": fix(broken + " — 길이는 충분한 글")}, "하네스 검사 실패"),
                         ({"HeadFix": fix("짧은 글")}, "많이 짧음"), ({"HeadFix": fix("")}, "많이 짧음")):
        script = use_script(Script(FollowupResult=edit(ANSWER), RevisionCritic=judge(5, extra=["용어 통일"]), **outputs))
        turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                                validators=["python_syntax"], options={}, head=head_review(org, team, models))
        assert turn.content == ANSWER and turn.auto_fixed == [] and len(script.prompts("RevisionCritic")) == 1
        assert all(h.passed for h in turn.harness) and turn.critic_passed and "용어 통일" in followup.next_request(turn)
        assert any("직접 고치지 못해 직전 수정본을 유지" in x and why in x for x in turn.log), turn.log


def test_failed_review_goes_to_lead_first_and_head_fixes_extras_after(org, ranks, team, models, use_script):
    """한 검토에 불합격 지적과 추가 점이 함께 있으면 팀장이 지적부터 고친다 — 추가 점은 그다음 검토 결과를 따른다."""
    script = use_script(Script(FollowupResult=[edit("수정본 A"), edit("수정본 B")], HeadFix=fix("부장이 고친 수정본"),
                               RevisionCritic=[judge(2, ["요청한 표가 빠짐"], ["용어 통일"], "severe"),
                                               judge(5, extra=["용어 통일"]), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "표 추가해줘", [],
                            head=head_review(org, team, models, max_self_corrections=2))
    lead = script.prompts("FollowupResult")
    assert len(lead) == 2 and "요청한 표가 빠짐" in lead[1] and "용어 통일" not in lead[1]  # 팀장에게는 지적만
    assert len(script.prompts("HeadFix")) == 1 and "[수정본]\n수정본 B" in script.prompts("HeadFix")[0]
    assert len(script.prompts("RevisionCritic")) == 3  # 팀장 다시 고침 1 + 부장 직접 수정 1 = 상한 2 를 함께 쓴다
    assert turn.content == "부장이 고친 수정본" and turn.critic_passed and turn.auto_fixed == ["용어 통일"]


def test_skipped_recheck_gives_head_fixed_revision_without_review(org, ranks, team, models, use_script):
    """부장이 직접 고친 뒤의 다시 검토를 사용자가 끊으면 — 고친 글을 검토 없이 내고, 무엇을 고쳤는지는 남긴다."""
    call_monitor.reset()

    def recheck(messages):
        call_monitor.skip("r-skip2", "critic")  # 다시 검토하는 중에 ⛔ 부장 검토 건너뛰기
        return judge(5)

    use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fix("부장이 고친 수정본"),
                      RevisionCritic=[judge(5, extra=["용어 통일"]), recheck]))
    with usage.scope(run_id="r-skip2"):
        turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                                head=head_review(org, team, models))
    call_monitor.reset()
    assert turn.content == "부장이 고친 수정본" and turn.critic is None and turn.critic_passed
    assert turn.auto_fixed == ["용어 통일"] and any("끊음" in x for x in turn.log)
    assert "용어 통일" in followup.head_review_markdown(turn, "")  # 부장 검토가 없어도 직접 고친 점은 보인다


def test_skip_while_head_is_fixing_keeps_reviewed_revision(org, ranks, team, models, use_script):
    """부장이 직접 고치는 중에 건너뛰면 — 그 호출도 끊기고, 부장이 이미 본 팀장 수정본과 남은 점이 그대로 나온다."""
    call_monitor.reset()

    def fixing(messages):
        call_monitor.skip("r-skip3", "critic")
        return fix("부장이 고친 수정본")

    script = use_script(Script(FollowupResult=edit("수정본 A"), HeadFix=fixing, RevisionCritic=judge(5, extra=["용어 통일"])))
    with usage.scope(run_id="r-skip3"):
        turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                                head=head_review(org, team, models))
    call_monitor.reset()
    assert turn.content == "수정본 A" and turn.critic.score == 5 and len(script.prompts("RevisionCritic")) == 1
    assert turn.auto_fixed == [] and "용어 통일" in followup.next_request(turn)
    assert any("직접 고치지 못해" in x and "사용자가 끊음" in x for x in turn.log)


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
    assert "용어 통일" not in prompts[1]  # 추가로 고칠 점은 팀장에게 넘기지 않는다 — 다음 검토 결과를 따른다
    assert turn.auto_fixed == [] and script.prompts("HeadFix") == []  # 다시 검토에 추가 점이 없으니 부장이 고칠 것도 없다
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
    script = use_script(Script(FollowupResult=edit("수정본"), RevisionCritic=judge(1, extra=["용어 통일"])))
    with usage.scope(run_id="r-skip"):
        call_monitor.skip("r-skip", "critic")
        turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여줘", [],
                                head=head_review(org, team, models))
    call_monitor.reset()
    assert turn.revised and turn.critic is None and script.prompts("RevisionCritic") == []
    assert len(script.prompts("FollowupResult")) == 1 and turn.auto_fixed == []  # 검토가 없으니 부장의 직접 수정도 없다
    assert script.prompts("HeadFix") == []
    assert any("건너뜀" in x for x in turn.log)


def test_review_text_for_screen_and_next_request():
    turn = followup.FollowupTurn(request="줄여줘", reply="줄였습니다", revised=True, content="x", version=1,
                                 critic=judge(4, ["2장 숫자에 출처가 없음"], ["표 제목에 기준일 넣기", "용어 통일"]),
                                 critic_by="🔬 연구부 부장", critic_passed=True)
    text = followup.head_review_markdown(turn, "🔬 연구부 부장 (gpt-5.5)")
    for part in ("🔬 연구부 부장 (gpt-5.5)", "4/5", "합격", "2장 숫자에 출처가 없음", "추가로 고칠 점", "표 제목에 기준일 넣기"):
        assert part in text
    assert "자동으로 고치지 않았습니다" not in text and "직접 고쳤습니다" not in text  # 고친 것이 없으면 그 줄도 없다
    request = followup.next_request(turn)  # 부장이 직접 고친 뒤에도 남은 것(지적 + 추가로 고칠 점)을 다음 수정 요청으로
    assert "표 제목에 기준일 넣기" in request and "용어 통일" in request and "출처가 없음" in request
    turn.critic_passed = False
    assert "불합격" in followup.head_review_markdown(turn, "부장")
    turn.critic, turn.critic_passed = judge(5), True
    assert followup.next_request(turn) == ""  # 고칠 점이 없으면 버튼도 없다
    assert followup.head_review_markdown(followup.FollowupTurn(request="q", reply="a"), "") == ""


def test_decision_note_leaves_out_head_review():
    """교훈은 사람의 요구에서 만든다 — 부장(모델)의 검토 의견은 결정 코멘트에 넣지 않는다."""
    turn = followup.FollowupTurn(request="줄여줘", reply="...", revised=True, content="x", version=1,
                                 critic=judge(4, extra=["용어 통일"]), critic_by="부장")
    note = followup.decision_note([turn])
    assert "줄여줘 → 수정본 v1" in note and "용어 통일" not in note and "요청 반영 점검" not in note


def test_screen_note_follows_team_settings():
    """💬 수정 요청 안내 문구 — 비평을 끈 팀·상한 0 인 팀에서는 '부장이 직접 고친 뒤 다시 검토'라고 하지 않는다."""
    usual = verification.critic_settings({})
    assert "직접 고친 뒤 다시 검토" in followup.review_note("연구부 부장", usual)
    assert "직접 고치고 다시 검토" in followup.review_note("연구부 부장", usual, waiting=True)
    only_review = verification.critic_settings({"critic": {"max_self_corrections": 0}})
    note = followup.review_note("연구부 부장", only_review)
    assert "**연구부 부장**" in note and "검토만" in note and "직접 고친 뒤" not in note
    assert "직접 고치고" not in followup.review_note("연구부 부장", only_review, waiting=True)
    off = verification.critic_settings({"critic": {"enabled": False}})
    assert "검토 없이" in followup.review_note("연구부 부장", off) and "연구부 부장" not in followup.review_note("연구부 부장", off)
    assert "검토" not in followup.review_note("연구부 부장", off, waiting=True)
