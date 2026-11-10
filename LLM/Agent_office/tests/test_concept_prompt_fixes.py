"""개념 편집 팀 프롬프트 점검(2026-11-10)에서 나온 세 가지 — 가짜 모델만.

C1: '고칠 곳만 바꾸기'(edits) 검토를 다시 하는 회차(부장 불합격·형식 검사 실패)에, 앞서 낸 edits 는 이미 반영됐다는 것을 알려 준다
    (실행 기록 14건: 첫 검토 99% 적용, 다시 쓰는 회차 32% — 모델이 예전 edits 를 다시 적어 '찾지 못함'). 전문을 다시 쓰는 팀의 되돌림 글은 그대로.
R1: 스킬의 '능동적 지식 확장'이 기억에서 꺼낸 내용을 "반드시 추가"하라던 것 → 근거 자료에서 확인되는 것만.
R2: 관련 기존 항목은 발췌(1절·6절 앞부분)만 들어간다 → 발췌임을 알리고 잘린 곳을 표시한다.
"""

from core import harness, verification
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_critic, make_delegation, make_review
from tests.test_concept_speed import _edit_team, edits
from tests.test_workflow import BRIEF, CRITIC_ON, NO_CRITIC, SETTINGS

BAD = "```python\ndef f(:\n    return 1\n```"
HARNESS = {"validators": ["python_syntax"], "final_only": ["python_syntax"], "max_retries": 2}


def _last_feedback(script: Script, key: str) -> str:
    """그 종류의 마지막 호출에서 모델이 받은 마지막 말 (= 되돌림 글)."""
    return [msgs for k, msgs in script.calls if k == key][-1][-1].content


# ─── C1 ───────────────────────────────────────────────────────────────


def test_edit_retry_after_harness_says_earlier_edits_are_already_applied(org, ranks, team, models, use_script):
    script = use_script(Script(text=BAD, ReviewEdits=[edits(("return 1", "return 2")), edits(("def f(:", "def f():"))]))
    graph = build_office_graph(org, ranks, _edit_team(team, harness=HARNESS), models, SETTINGS, NO_CRITIC)
    final = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert final.harness[0].passed
    feedback = _last_feedback(script, "ReviewEdits")
    assert feedback.startswith("자동 검증(하네스) 결과")  # 원래 되돌림 글 뒤에 덧붙인다
    for words in ("이미 반영", "지금 글", "새로 고칠 곳만", "다시 적지 않", "전문"):
        assert words in feedback, words


def test_edit_retry_after_critic_says_earlier_edits_are_already_applied(org, ranks, team, models, use_script):
    script = use_script(Script(text="초안 오타 있음", ReviewEdits=[edits(("오타", "수정")), edits(("있음", "없음"))],
                               CriticResult=[make_critic(2, leaps=["비약"]), make_critic(5)]))
    graph = build_office_graph(org, ranks, _edit_team(team), models, SETTINGS, CRITIC_ON)
    final = [p for k, p in stream_office(graph, org, "질문", "am_action", brief=BRIEF) if k == "step"][-1]
    assert final.content == "초안 수정 없음"  # 두 번째 edits 는 첫 결과에 이어서 적용된다
    feedback = _last_feedback(script, "ReviewEdits")
    assert feedback.startswith("너의 이전 단계 논리에서") and "이미 반영" in feedback and "새로 고칠 곳만" in feedback


def test_rewrite_teams_get_the_same_feedback_as_before(org, ranks, team, models, use_script):
    # 형식 검사 실패 — 전문을 다시 쓰는 팀
    script = use_script(Script(text=BAD, ReviewResult=[make_review(answer=BAD), make_review(answer=BAD.replace("f(:", "f():"))]))
    graph = build_office_graph(org, ranks, team.model_copy(update={"harness": _edit_team(team, harness=HARNESS).harness}),
                               models, SETTINGS, NO_CRITIC)
    list(stream_office(graph, org, "질문", "am_action", brief=BRIEF))
    expected = harness.harness_feedback(harness.run_validators(["python_syntax"], BAD, {}))
    assert _last_feedback(script, "ReviewResult") == expected
    # 부장 불합격 — 전문을 다시 쓰는 팀
    critic = make_critic(2, leaps=["비약"])
    script = use_script(Script(text="초안", ReviewResult=make_review(answer="보완안"), CriticResult=[critic, make_critic(5)]))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, CRITIC_ON)
    list(stream_office(graph, org, "질문", "am_action", brief=BRIEF))
    assert _last_feedback(script, "ReviewResult") == verification.correction_message(critic)


def test_edit_team_integration_of_several_results_is_unchanged(org, ranks, team, models, use_script):
    # edit 팀이라도 여러 명 통합은 전문을 쓴다 → edits 안내를 붙이지 않는다
    critic = make_critic(2, leaps=["비약"])
    script = use_script(Script(text="초안", Delegation=make_delegation("am_fact", "staff_draft"),
                               ReviewResult=make_review(answer="통합본"), CriticResult=[critic, make_critic(5)]))
    graph = build_office_graph(org, ranks, _edit_team(team), models, SETTINGS, CRITIC_ON)
    list(stream_office(graph, org, "질문", "dgm_research", brief=BRIEF))
    assert _last_feedback(script, "ReviewResult") == verification.correction_message(critic)
