"""💬 수정 요청 — 긴 결과는 '고칠 곳만' 받아 코드가 적용하고, 바뀐 곳을 코드가 계산해 보여 준다 (2026-11-16).

전에는 수정 요청 한 번마다 팀장이 완성 전문을 다시 쓰고, 부서장이 '추가로 고칠 점'을 고칠 때도 전문을 다시 썼다.
포트폴리오처럼 긴 글은 다른 PC 에서 수정 한 번에 20~30분 (사용자 보고, 실행 기록 없음) — 출력 글자가 글 전체만큼이었다.
전문을 다시 쓰면 요청과 무관한 곳이 조용히 바뀌어 부서장 검토에서 '올바른 내용을 버림'으로 불합격이 나기도 쉽다.
가짜 모델만 쓴다 (tests/conftest.py).
"""

from core import followup
from core.followup import FollowupResult, HeadFix
from core.schemas import TextEdit
from tests.fakes import Script
from tests.test_followup import ANSWER, edit, head_review, judge, lead_agent

# 포트폴리오 크기의 긴 결과 (가상 글) — 절마다 문단이 있고, 고칠 곳은 한 문단뿐이다
SECTIONS = [f"## {i}. 프로젝트 {i}\n프로젝트 {i} 는 가상 서비스의 {i}번째 기능을 만든 일이다. " + ("설명 문장. " * 60).strip()
            for i in range(1, 13)]
LONG = "# 포트폴리오\n\n" + "\n\n".join(SECTIONS)
OLD = "프로젝트 3 는 가상 서비스의 3번째 기능을 만든 일이다."
NEW = "프로젝트 3 은 가상 서비스의 결제 기능을 만든 일이다 (테스트 61건)."


def patch(*pairs, changes=("3번 프로젝트 설명을 고침",)) -> FollowupResult:
    return FollowupResult(analysis="수정 요청", revised=True, reply="고쳤습니다.", changes=list(changes),
                          edits=[TextEdit(find=f, replace=r) for f, r in pairs])


def test_long_result_takes_only_the_edits_and_code_builds_the_revision(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=patch((OLD, NEW))))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 프로젝트를 결제 기능으로, 테스트 61건", [])
    assert turn.revised and turn.version == 1
    assert turn.content == LONG.replace(OLD, NEW)  # 고친 곳 밖은 글자 그대로
    assert any("고칠 곳 1/1개 적용" in x for x in turn.log)
    prompt = script.prompts("FollowupResult")[0]
    assert "edits" in prompt and "글자 그대로" in prompt  # 긴 결과는 고칠 곳만 쓰라고 한다


def test_short_result_is_still_rewritten_in_full(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=edit("수정본 전문")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", ANSWER, "줄여 줘", [])
    assert turn.content == "수정본 전문"
    assert "edits 에" not in script.prompts("FollowupResult")[0]  # 짧은 글은 지금처럼 전문


def test_full_rewrite_is_still_accepted_for_a_structural_request(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=edit("# 새 구조\n전부 다시 씀")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "순서를 전부 바꿔 줘", [])
    assert turn.content == "# 새 구조\n전부 다시 씀" and any("전문을 다시 씀" in x for x in turn.log)


def test_edit_that_cannot_be_found_is_skipped_with_the_reason(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=patch((OLD, NEW), ("없는 문장", "바뀐 문장"), ("설명 문장.", "x"))))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "고쳐 줘", [])
    assert turn.content == LONG.replace(OLD, NEW)
    assert any("고칠 곳 1/3개 적용" in x for x in turn.log)
    assert any("적용 못 함" in x and "찾지 못함" in x for x in turn.log)
    assert any("적용 못 함" in x and "곳에 있음" in x for x in turn.log)  # 여러 곳이면 엉뚱한 곳을 바꾸지 않는다


def test_all_edits_failing_falls_back_to_a_full_rewrite(org, ranks, team, models, use_script):
    script = use_script(Script(FollowupResult=[patch(("없는 문장", "바뀐 문장")), edit("전문으로 다시 쓴 수정본")]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "고쳐 줘", [])
    assert turn.content == "전문으로 다시 쓴 수정본"
    prompts = script.prompts("FollowupResult")
    assert len(prompts) == 2 and "완성 전문" in prompts[1] and "없는 문장" in prompts[1]
    assert any("전문으로 다시" in x for x in turn.log)


def test_harness_retry_applies_new_edits_to_the_previous_attempt(org, ranks, team, models, use_script):
    bad = LONG + "\n\n```python\nprint(\n```"
    good_fix = ("```python\nprint(\n```", "```python\nprint(1)\n```")
    script = use_script(Script(FollowupResult=[patch((OLD, NEW)), patch(good_fix)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", bad, "3번 고쳐 줘", [],
                            validators=["python_syntax"], options={}, max_retries=1)
    assert turn.content == LONG.replace(OLD, NEW) + "\n\n```python\nprint(1)\n```"  # 두 번의 고칠 곳이 모두 남는다
    assert "앞서 낸 edits 는 이미 반영됐다" in script.prompts("FollowupResult")[1]


def test_head_fix_on_a_long_revision_uses_edits_too(org, ranks, team, models, use_script):
    extra = ("프로젝트 7 는", "프로젝트 7 은")
    script = use_script(Script(FollowupResult=patch((OLD, NEW)), HeadFix=HeadFix(edits=[TextEdit(find=extra[0], replace=extra[1])]),
                               RevisionCritic=[judge(5, extra=["7번 조사 고치기"]), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 고쳐 줘", [],
                            head=head_review(org, team, models))
    assert turn.content == LONG.replace(OLD, NEW).replace(*extra) and turn.auto_fixed == ["7번 조사 고치기"]
    assert "edits" in script.prompts("HeadFix")[0]
    assert NEW in script.prompts("RevisionCritic")[1]  # 다시 검토는 부서장이 고친 전문을 본다


def test_head_fix_edits_all_missing_fall_back_to_full_text(org, ranks, team, models, use_script):
    full = LONG.replace(OLD, NEW).replace("프로젝트 7 는", "프로젝트 7 은")
    script = use_script(Script(FollowupResult=patch((OLD, NEW)),
                               HeadFix=[HeadFix(edits=[TextEdit(find="없음", replace="x")]), HeadFix(revised_answer=full)],
                               RevisionCritic=[judge(5, extra=["7번 조사 고치기"]), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 고쳐 줘", [],
                            head=head_review(org, team, models))
    assert turn.content == full and len(script.prompts("HeadFix")) == 2


def test_lead_rewrite_after_a_failed_review_edits_the_previous_revision(org, ranks, team, models, use_script):
    more = ("프로젝트 5 는", "프로젝트 5 은")
    script = use_script(Script(FollowupResult=[patch((OLD, NEW)), patch(more)],
                               RevisionCritic=[judge(2, violations=["'5번' 부분이 조사 규칙에 어긋남"], severity="severe"), judge(5)]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 고쳐 줘", [],
                            head=head_review(org, team, models))
    assert turn.content == LONG.replace(OLD, NEW).replace(*more) and turn.critic_passed
    assert "앞서 낸 edits 는 이미 반영됐다" in script.prompts("FollowupResult")[1]


# ─── 바뀐 곳 — 코드가 앞 버전과 비교한다 ─────────────────────────────────────


def test_turn_keeps_the_changed_paragraphs_only(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=patch((OLD, NEW))))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 프로젝트를 결제 기능으로, 테스트 61건", [])
    assert len(turn.diff) == 1
    block = turn.diff[0]
    assert OLD in block.before and NEW in block.after and "프로젝트 4" not in block.before
    assert block.where == "## 3. 프로젝트 3" and not block.unrequested
    assert turn.diff_from == 0  # 원래 결과와 비교


def test_change_nobody_asked_for_is_marked(org, ranks, team, models, use_script):
    """팀장이 전문을 다시 쓰며 요청과 무관한 절(9번)도 바꿨다 — 요청·바뀐 점 설명 어디에도 그 절이 없다."""
    sneaky = LONG.replace(OLD, NEW).replace("프로젝트 9 는 가상 서비스의 9번째 기능을 만든 일이다.", "다른 회사 경력으로 바꾼 글이다.")
    use_script(Script(FollowupResult=edit(sneaky, changes=["3번 프로젝트를 결제 기능으로"])))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "3번 프로젝트를 결제 기능으로", [])
    flags = {b.where: b.unrequested for b in turn.diff}
    assert flags == {"## 3. 프로젝트 3": False, "## 9. 프로젝트 9": True}
    assert "⚠️" in followup.diff_summary(turn)


def test_diff_survives_saving_and_compares_with_the_previous_version(org, ranks, team, models, use_script):
    agent = lead_agent(org, ranks, team, models)
    use_script(Script(FollowupResult=[patch((OLD, NEW)), patch(("프로젝트 5 는", "프로젝트 5 은"), changes=["5번 조사"])]))
    first = followup.respond(agent, "원 질문", LONG, "3번 고쳐 줘", [])
    second = followup.respond(agent, "원 질문", first.content, "5번 조사 고쳐 줘", [first])
    assert second.diff_from == 1 and len(second.diff) == 1 and "프로젝트 5 은" in second.diff[0].after
    assert NEW not in second.diff[0].after  # v1 에서 이미 바뀐 곳은 v2 의 바뀐 곳이 아니다
    again = followup.load_turns(followup.dump_turns([first, second]))
    assert again == [first, second]
    assert followup.load_turns('[{"request": "옛 기록", "reply": "답"}]')[0].diff == []  # 이 칸이 없던 옛 기록도 열린다


def test_changed_blocks_keeps_only_changed_lines():
    before = "\n".join(f"줄 {i}" for i in range(50))
    after = before.replace("줄 10", "줄 열").replace("줄 40", "줄 사십")
    blocks = followup.changed_blocks(before, after)
    assert [(b.before, b.after) for b in blocks] == [("줄 10", "줄 열"), ("줄 40", "줄 사십")]
    assert followup.changed_blocks(before, before) == []


def test_question_has_no_diff(org, ranks, team, models, use_script):
    use_script(Script(FollowupResult=FollowupResult(analysis="질문", revised=False, reply="답")))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", LONG, "왜?", [])
    assert turn.diff == [] and followup.diff_summary(turn) == ""
