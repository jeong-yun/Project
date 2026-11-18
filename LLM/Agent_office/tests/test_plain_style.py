"""공용 문체 스킬(teams/_shared/style) — AI 가 쓴 티가 나는 글버릇을 피하게 하고, 최종 답에서 코드가 세어 '기록만' 남긴다.

커리어부의 피할 표현 세기(career.check_style)를 재사용하고, 다시 쓰게 하지는 않는다 (커리어부 career_style 은 그대로).
글은 모두 테스트용으로 직접 지어낸 가상의 글이다. 실제 모델은 부르지 않는다.
"""

import re

import pytest

from core import career
from core.agent import skill_sections
from core.config import load_skills, load_team
from core.harness import REGISTRY
from core.workflow import build_office_graph, stream_office
from tests.fakes import Script, make_review
from tests.test_workflow import BRIEF, NO_CRITIC, SETTINGS

STYLE_DIR = "_shared/style"
STYLE_HEAD = "[문체 — 사람이 쓴 글처럼"
WITH_STYLE = ["general", "edu_career", "edu_hobby", "travel_abroad", "travel_domestic"]
# 붙이지 않는 팀 — security_governance: 모든 최종 답이 표 + JSON + 증적 원문 그대로 인용 / career: 자기 문체 스킬(natural_style)이 따로 있다
WITHOUT_STYLE = ["security_governance", "career"]

HUMAN = ("결론: 4주 온보딩은 버디 1명이 신입 1명을 맡는 방식으로 한다. 기존 팀원 부담은 주 3시간 안쪽이다.\n"
         "1주차에는 개발 환경을 세우고 문서 오탈자 PR 을 하나 올린다. 2주차는 버그 티켓 2건을 짝과 함께 닫는다. "
         "3주차부터는 혼자 기능 티켓을 맡는다. 배포는 4주차 화요일에 지켜보기만 한다.")
AI_LIKE = ("좋은 질문입니다! 신입 온보딩에 대해 알아보겠습니다. 성공적인 온보딩은 매우 중요합니다.\n"
           "다양한 교육 자료를 통해 신입이 효율적으로 적응하도록 돕습니다. 이를 통해 팀 전체의 생산성을 극대화할 수 있습니다. "
           "뿐만 아니라 체계적으로 설계된 멘토링은 핵심적인 역할을 합니다. 전문가들은 버디 제도가 혁신적이라고 말합니다.\n"
           "결론적으로, 다양한 활동과 적극적으로 소통하는 문화가 완벽한 온보딩을 만듭니다. 도움이 되길 바랍니다!")


def flat(text: str) -> str:
    return " ".join(text.split())


def check(answer: str):
    return REGISTRY["plain_style"].check(answer, {})


# ─── 스킬 파일 ─────────────────────────────────────────────────────────


def test_shared_list_is_read_by_the_same_parser_as_career():
    phrases = career.avoid_phrases((career.TEAMS_DIR / STYLE_DIR / "plain_style.md").read_text(encoding="utf-8"))
    groups = {p.group for p in phrases}
    assert {"상투적 연결", "인사·예고·맺음", "뜻이 빈 꾸밈말", "출처 없는 일반화", "과장", "번역투"} <= groups
    assert len(phrases) >= 40 and all(p.fix for p in phrases)
    # 커리어부 목록은 따로 그대로 — 공용 목록이 커리어부 검사를 바꾸지 않는다
    assert career.avoid_phrases() and not any(p.text == "좋은 질문" for p in career.avoid_phrases())


# ─── 경고만 남기는 검증기 ──────────────────────────────────────────────


def test_ai_like_answer_passes_but_leaves_a_warning():
    result = check(AI_LIKE)
    assert result.passed and result.score == 1.0  # 다시 쓰게 하지 않는다
    assert "경고" in result.log and "좋은 질문" in result.log and "다양한" in result.log


def test_plain_answer_has_no_warning_and_code_blocks_are_not_counted():
    code = "\n```python\n# 다양한 다양한 다양한 효율적으로 이를 통해\nx = 1\n```\n"
    result = check(HUMAN + code)
    assert result.passed and "경고" not in result.log
    assert "상투 표현 0개" in result.log


def test_bold_and_dash_are_recorded_not_judged():
    result = check("**결론**: 버디 제도로 한다 — 부담은 주 3시간이다. **주의**: 배포는 4주차.")
    assert result.passed and "굵은 글씨·긴 줄표 3곳" in result.log


def test_empty_answer():
    assert check("").passed


# ─── 팀원별로 절 빼기 ─────────────────────────────────────────────────

SKILL = "## 스킬: a\n머리말\n[공통]\n공통 내용\n[문체 — 글]\n문체 내용\n## 스킬: b\n머리 없는 파일 내용"


def test_skill_sections_minus_drops_only_that_section():
    out = skill_sections(SKILL, ["-[문체"])
    assert "공통 내용" in out and "머리 없는 파일 내용" in out and "머리말" in out
    assert "문체 내용" not in out
    assert skill_sections(SKILL, ["[공통", "-[문체"]) == skill_sections(SKILL, ["[공통"])


@pytest.mark.parametrize("team_id", WITH_STYLE)
def test_style_reaches_writers_only_and_member_prompts_do_not_grow(team_id):
    team, skills = load_team(team_id), load_skills(team_id)
    # validators 에 있어야 돌고, final_only 에 있어야 팀원 초안에는 돌지 않는다
    assert STYLE_DIR in team.skills_from and {"plain_style"} <= set(team.harness.validators) & set(team.harness.final_only)
    without = re.sub(r"## 스킬: plain_style\n.*?(?=## 스킬: |\Z)", "", skills, flags=re.S)  # 붙이기 전 그 팀 스킬
    for node in team.org.walk():
        got = skill_sections(skills, node.skill_sections)
        if node.children:  # 최종 답·통합을 쓰는 팀원
            assert STYLE_HEAD in got, node.id
        else:  # 맡은 부분만 쓰는 실무자 — 프롬프트가 붙이기 전과 같다
            assert STYLE_HEAD not in got and "[문체 — 피할 표현" not in got, node.id
            # 빈 머리말 '## 스킬: plain_style' 한 줄(20자)만 남는다 — 맡은 절만 받는 실무자의 다른 스킬 파일도 같다
            assert flat(got.replace("## 스킬: plain_style", "")) == flat(skill_sections(without, node.skill_sections)), node.id


@pytest.mark.parametrize("team_id", WITHOUT_STYLE)
def test_teams_with_fixed_format_or_own_style_are_left_alone(team_id):
    team = load_team(team_id)
    assert STYLE_DIR not in team.skills_from and "plain_style" not in team.harness.final_only


# ─── 가짜 모델로 한 번 돌리기 ─────────────────────────────────────────


def test_fake_model_final_answer_is_logged_not_rewritten(org, ranks, team, models, use_script):
    """일반 팀 · 대리·실행안(am_action)에게 문의 → 사원 초안 → 대리가 AI 티 글로 최종 답. 경고만 남고 다시 쓰지 않는다."""
    script = use_script(Script(text="사례 초안", ReviewResult=[make_review("revise", 7, AI_LIKE)]))
    graph = build_office_graph(org, ranks, team, models, SETTINGS, NO_CRITIC)
    events = list(stream_office(graph, org, "신입 온보딩 계획", "am_action", brief=BRIEF))
    steps = {p.node_id: p for k, p in events if k == "step"}

    final = steps["am_action"]
    assert final.content == AI_LIKE and final.attempts == 1
    style = next(h for h in final.harness if h.validator == "plain_style")
    assert style.passed and "경고" in style.log
    assert "plain_style" not in {h.validator for h in steps["staff_case"].harness}
    assert len(script.prompts("ReviewResult")) == 1  # 문체 때문에 더 부르지 않는다
    assert STYLE_HEAD in script.prompts("ReviewResult")[0]  # 최종 답을 쓰는 대리는 문체 스킬을 받는다
    assert all(STYLE_HEAD not in p for p in script.prompts("text"))  # 사원 초안 프롬프트에는 없다
