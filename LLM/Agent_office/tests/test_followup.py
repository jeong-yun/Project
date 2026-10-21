"""결정 전 후속 대화(core/followup.py) — 질문/수정 자동 판단, 수정본 하네스 검사, 결정 코멘트 요약."""

from core import followup
from core.agent import OrgAgent
from core.followup import FollowupResult
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
