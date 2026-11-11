"""프롬프트 점검(2026-11-10)의 남은 중간 항목 — 리서치·지식부(R3~R9) · 투자·여행부(T4~T12). 가짜 모델만.

글만 바꾼 항목은 실제 teams/ 파일로 조립한 프롬프트에 새 문장이 들어가고 어긋나던 문장이 없어졌는지 본다.
코드가 바뀐 항목(T7 요약 블록 고르기 · T11 종목 찾기)은 동작을 본다. 실제 모델 결과가 좋아지는지는 여기서 확인하지 못한다.
"""

import json

import pytest

from core import harness, stock_calc
from core.agent import OrgAgent
from core.config import load_skills, load_team, team_org
from core.tools import stock_data
from tests.fakes import Script, make_delegation
from tests.test_teams import fake_market, fixed_today  # noqa: F401 — 가짜 시세·고정 날짜(2026-10-19) fixture 재사용


def agents_of(team_id: str, ranks, models) -> dict[str, OrgAgent]:
    """그 팀의 실제 설정·스킬로 만든 에이전트들 (id → OrgAgent)."""
    team, by_rank = load_team(team_id), {r.id: r for r in ranks}
    nodes = list(team_org(team, ranks).walk())
    names = {n.id: n.name or n.id for n in nodes}
    return {n.id: OrgAgent(n, by_rank[n.rank], team, models, names=names, skills=load_skills(team_id)) for n in nodes}


def task_of(team_id: str, task_id: str):
    return next(t for t in load_team(team_id).tasks if t.id == task_id)


def skill_line(agent: OrgAgent, start: str) -> str:
    """스킬에서 start 로 시작하는 줄 (앞 공백 무시)."""
    return next(line.strip() for line in agent.skills.splitlines() if line.strip().startswith(start))


# ─── R4·R5 기본 팀 ─────────────────────────────────────────────────────


def test_general_personas_do_not_fight_the_integration_rule(ranks, models):
    agents = agents_of("general", ranks, models)
    action, research, staff = (agents[i].system_prompt() for i in ("gm_action", "gm_research", "staff_draft"))
    assert "핵심만 남깁니다" not in action and "확인된 사실은 빼지 않고" in action  # 통합 지시 '사실을 근거 없이 버리지 않는다'와 같은 방향
    assert "실행을 책임지는" not in research and "사실관계" in research.splitlines()[0]  # 부장·리서치는 전문 분야에 맞는 페르소나
    assert "빠르게 초안" not in staff and "맡은 범위의 초안을 근거와 함께" in staff


def test_general_answer_style_follows_brief_format_and_has_one_word_for_unknowns(ranks, models):
    agent = agents_of("general", ranks, models)["staff_case"]
    order, cite = skill_line(agent, "- 업무 브리프에 [결과 형식]"), skill_line(agent, "- [근거 자료]가 주어졌으면")
    assert "그 순서를 따른다" in order and "없을 때만" in order  # '결론을 맨 앞에'는 결과 형식이 없을 때만
    assert "[n]" in cite and "'확인 필요'" in cite and "고유명사" not in cite  # 근거 자료가 없는 업무(코드·이메일)에 출처 번호를 요구하지 않는다
    assert "추측" in cite  # 공통 규칙의 '모른다고 밝힌다'·'추측임을 밝힌다'도 이 말로


# ─── R3·R6~R9 개념 편집 팀 ───────────────────────────────────────────────


def test_concept_editor_always_delegates_to_both_members(ranks, models, use_script):
    script = use_script(Script(Delegation=make_delegation("am_fact", "dgm_lineage")))
    agents_of("concept", ranks, models)["gm_editor"].plan("질문", "질문")
    prompt = script.prompts("Delegation")[0]
    assert "항상 둘 다 맡긴다" in prompt and "대리·팩트체크" in prompt and "차장·계보분석" in prompt


def test_concept_citation_numbers_stay_out_of_the_final(ranks, models):
    agents = agents_of("concept", ranks, models)
    for who in ("lead", "gm_editor", "am_fact"):  # 최종본·참고 문헌 후보를 쓰는 사람
        line = skill_line(agents[who], "- 웹 검색 자료의 [번호]")
        assert "<review_log>" in line and "최종본" in line and "참고 문헌" in line, who
    prompt = agents["gm_editor"].system_prompt()
    assert "- markdown 형식으로 작성한다." not in prompt  # '굵게(**) 쓰지 않음'과 부딪히던 줄
    assert "[출력 형식 견본]" in next(line for line in prompt.splitlines() if line.endswith("형식으로 작성한다."))


def test_concept_critic_rules_judge_only_what_the_head_can_see():
    rules = load_team("concept").verification["critic"]["rules"]
    text = "\n".join(rules)
    assert "원전 확인 없이" not in text and "존재 확인이 안 된" not in text  # 부장은 원전을 볼 수 없다
    assert sum("판정 표" in r for r in rules) >= 2 and "출처 미확인" in text
    assert "생략" in text  # 검증 로그가 길어 줄었으면 이 규칙으로 지적하지 않는다
    assert "초안 구조 전제 수용" in text and "이중 해리 미검사" in text and "판정 불일치" in text  # 위반 이름은 그대로


def test_concept_stage1_matches_what_the_model_actually_receives(ranks, models):
    lineage = agents_of("concept", ranks, models)["dgm_lineage"]  # 1단계 담당
    listing, primary, mismatch = (skill_line(lineage, s) for s in ("(1)", "· 판정 근거는", "(4)"))
    assert "3~6개" not in listing and "[입력]" in listing and "비교 대상 없음" in listing  # 사람이 고른 0~6개만 받는다
    assert "따로 열어" not in primary and "[근거 자료]" in primary and "원전 미확인" in primary  # 문서를 여는 도구가 없다
    assert "같이 쓰이는 사례만 보면" in primary  # 뒤 문장은 그대로
    assert "결론을 표에 맞춰 수정" not in mismatch and "(2)(3)" in mismatch and "다시 판정" in mismatch
