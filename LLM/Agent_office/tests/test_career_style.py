"""👔 문체 검사(career_style) — AI 가 쓴 글 티(상투 표현 쌓임·되풀이, 자소서 줄글 속 꾸밈 기호)를 최종 답변에서 잡는다.

글은 모두 테스트용으로 직접 지어낸 가상의 글이다 (실제 사람·회사 아님). 실제 모델은 부르지 않는다.
"""

import pytest

from core import career, followup
from core.config import load_team
from core.followup import FollowupResult
from tests.fakes import Script, make_review
from tests.test_career import JD, SOURCE, _career_prompt, check, cover, run_task, summary
from tests.test_followup import lead_agent

QUESTIONS = [{"no": 1, "text": "지원 동기", "limit": None, "no_spaces": False}]

# 사람이 쓴 것 같은 글 — 구체적인 장면, 길이가 섞인 문장, 상투 표현 없음
HUMAN = ("보안 관제 일을 3년 했습니다. 처음 1년은 야간조였습니다. 새벽 3시에 울리는 경보 열 건 중 아홉 건이 오탐이었고, 저는 그 아홉 건을 "
         "손으로 닫는 일이 싫었습니다. 그래서 2022년 4월부터 퇴근 뒤에 파이썬으로 분류 스크립트를 짰습니다. 석 달 걸렸습니다. 팀장님이 "
         "운영에 붙여 보자고 했고, 일일 분류 시간이 180분에서 45분으로 줄었습니다.\n\n"
         "귀사 공고의 '탐지 규칙 운영' 업무를 보고 지원했습니다. 제가 만든 규칙 가운데 몇 개는 석 달 안에 다시 고쳐야 했습니다. "
         "왜 고쳐야 했는지를 적어 둔 문서가 있습니다. 입사하면 그 기록 방식부터 팀에 맞춰 보고 싶습니다.")

# AI 가 쓴 티가 짙은 글 — 같은 사실인데 상투적 연결·다짐·꾸밈말이 쌓였다
AI_LIKE = ("저는 보안 관제 업무를 수행하며 다양한 위협에 대응해 온 경험을 바탕으로 귀사의 보안 운영에 기여하고자 지원하였습니다. "
           "보안 로그 분석 자동화 프로젝트를 통해 일일 분류 시간을 180분에서 45분으로 단축할 수 있었습니다. 이 과정에서 문제의 원인을 "
           "체계적으로 분석하고 효율적으로 해결하는 역량을 키울 수 있었습니다.\n\n"
           "뿐만 아니라 ISMS-P 인증 심사 준비 과정에서 유관 부서와 적극적으로 소통하며 원활한 협업을 이끌어 냈습니다. "
           "이를 통해 협업의 중요성을 깨달았습니다.\n\n"
           "이러한 경험을 바탕으로 귀사의 탐지 규칙 운영 업무에서 끊임없이 성장하며 조직에 기여하는 인재가 되겠습니다. "
           "앞으로도 지속적으로 역량을 발전시켜 나가겠습니다.")

CARDS = ("# 포트폴리오 — 보안 관리자\n## 요약\n{summary}\n## 핵심 역량\n| 영역 | 기술·도구 | 근거 |\n|---|---|---|\n| 관제 | Python | 1번 |\n"
         "## 프로젝트\n### 1. 보안 로그 분석 자동화 (2022.04 ~ 2022.12)\n- **문제**: {problem}\n- **역할**: {role}\n- **기술**: Python, 정규식\n"
         "- **결과**: {result}\n- **배운 점**: {lesson}\n## 직무 강점\n- {strength}\n")
PORTFOLIO_HUMAN = CARDS.format(
    summary="보안 관제 3년, 그중 2년은 탐지 규칙을 만들고 고쳤다. 로그 분류 자동화로 일일 분류 시간을 180분에서 45분으로 줄였다.",
    problem="야간 경보 열 건 중 아홉 건이 오탐인데 사람이 손으로 닫고 있었다", role="분류 스크립트 개발과 운영 적용",
    result="일일 분류 시간 180분 → 45분, 하루 1,200건 자동 분류",
    lesson="시험 환경에 백업 서버 로그가 없어 첫 배포를 하루 만에 내렸다. 그 뒤로 규칙은 나눠서 올린다", strength="오탐을 숫자로 센다 (1번)")
PORTFOLIO_AI = CARDS.format(
    summary="저는 보안 관제 분야에서 다양한 경험을 쌓아 온 보안 전문가입니다. 로그 분석 자동화를 통해 업무 효율성을 극대화하였습니다. "
            "이러한 경험을 바탕으로 조직의 보안 수준 제고에 기여하겠습니다.",
    problem="다양한 보안 이벤트를 수작업으로 분류함에 있어 비효율이 지속적으로 발생", role="자동화 솔루션을 주도적으로 설계하고 성공적으로 구축",
    result="일일 분류 시간을 180분에서 45분으로 단축하여 운영 효율성을 극대화",
    lesson="자동화를 통해 반복 업무를 효율적으로 개선할 수 있음을 배울 수 있었습니다. 데이터 기반 접근의 중요성을 깨달았습니다",
    strength="유관 부서와의 원활한 협업을 이끄는 소통 능력")


def style(answer: str, **options):
    return check("career_style", answer, **options)


def cover_style(text: str, **options):
    return style(cover({1: text}), career_questions=QUESTIONS, **options)


# ─── 무엇이 걸리고 무엇이 안 걸리나 ─────────────────────────────────────────


def test_ai_like_cover_letter_fails_and_says_which_sentence_and_why():
    res = cover_style(AI_LIKE)
    assert not res.passed
    assert "1,000자당" in res.log and "기준 8개" in res.log  # 지표 값과 기준
    assert "이러한 경험을 바탕으로 귀사의 탐지 규칙" in res.log  # 어느 문장이
    assert "'이러한 경험을 바탕으로'" in res.log and "'되겠습니다'" in res.log  # 왜 (걸린 표현)
    assert "잇는 말을 빼고" in res.log and "꾸밈말을 빼고" in res.log  # 어떻게 고치나 (묶음별)
    assert "문서에 없는" in res.log and "글자 수" in res.log  # 고치면서 사실을 보태지 않는다 · 글자 수 제한은 그대로


def test_concrete_natural_text_passes():
    res = cover_style(HUMAN)
    assert res.passed and "문체 통과" in res.log


def test_one_or_two_phrases_do_not_fail():
    twice = HUMAN.replace("그래서 2022년 4월부터", "이러한 경험을 바탕으로 2022년 4월부터").replace("맞춰 보고 싶습니다.", "맞춰 기여하겠습니다.")
    res = cover_style(twice)
    assert res.passed, res.log
    assert "상투 표현 2개" in res.log  # 세기는 하되 실패는 아니다
    assert cover_style(HUMAN + " 다양한 로그를 봤습니다.").passed


def test_same_phrase_three_times_fails_even_when_the_total_is_small():
    text = HUMAN + ("\n\n야간조 교대 문서를 고쳐 쓰는 일을 통해 인수인계 시간을 줄일 수 있었습니다. 규칙 설명 칸을 채우는 일을 통해 "
                    "고치는 시간을 줄일 수 있었습니다. 백업 서버 로그를 시험 환경에 넣는 일을 통해 오탐을 미리 볼 수 있었습니다.")
    res = cover_style(text)
    assert not res.passed and "'통해 … 수 있었습니다' 3번" in res.log
    assert cover_style(text.replace("넣는 일을 통해 오탐을 미리 볼 수 있었습니다", "넣어 오탐을 미리 봤습니다")).passed  # 두 번은 통과


def test_bold_and_long_dashes_in_cover_letter_prose_fail_but_one_does_not():
    marked = HUMAN.replace("180분에서 45분으로", "**180분에서 45분으로**").replace("'탐지 규칙 운영'", "**탐지 규칙 운영**")
    marked = marked.replace("석 달 걸렸습니다.", "석 달 걸렸습니다 — 퇴근 뒤에만 했기 때문입니다.")
    res = cover_style(marked)
    assert not res.passed and "굵은 글씨" in res.log and "3곳" in res.log
    assert cover_style(HUMAN.replace("180분에서 45분으로", "**180분에서 45분으로**")).passed
    assert cover_style("[야간조의 오탐을 줄인 3년]\n" + HUMAN).passed  # 소제목 [핵심 한 문장] 은 꾸밈 기호가 아니다


def test_portfolio_cards_are_checked_as_text_and_their_format_marks_are_not_decoration():
    assert style(PORTFOLIO_HUMAN).passed  # 카드의 '- **문제**:' 는 정해진 형식이다
    res = style(PORTFOLIO_AI)
    assert not res.passed and "다양한" in res.log and "극대화" in res.log
    assert "굵은 글씨" not in res.log


def test_json_mermaid_headings_and_table_rules_are_not_counted():
    noise = ("\n\n```mermaid\nflowchart LR\n  A[다양한 로그] --> B[원활한 협업을 통해 성공적으로 극대화]\n```\n"
             + summary([("이러한 경험을 바탕으로 적극적으로 기여하겠습니다", "일일 분류 시간을 180분에서 45분으로 줄였다.")] * 4,
                       needs=["다양한 · 원활한 · 끊임없이 · 지속적으로 · 효율적으로"]))
    assert style(PORTFOLIO_HUMAN.replace("## 직무 강점", "## 다양한 직무 강점 — 끊임없이 성장") + noise).passed
    assert cover_style(HUMAN + noise).passed
    assert career.check_style(PORTFOLIO_HUMAN + noise).hits == []
    fact_cards = "## 사실 카드\n| # | 경험 | 원문 인용 |\n|---|---|---|\n| 1 | 로그 자동화 | 일일 분류 시간을 180분에서 45분으로 줄였다. |"
    assert style(fact_cards).passed


def test_sentences_copied_from_my_own_document_are_not_counted():
    """포트폴리오 최신화는 바뀌지 않은 항목을 원문 그대로 쓴다 — 내 문서에 있는 문장은 내 글이다."""
    mine = ("다양한 로그를 체계적으로 모아 지속적으로 살폈습니다.\n유관 부서와 적극적으로 소통하며 원활한 협업을 이끌었습니다.\n"
            "이러한 경험을 바탕으로 조직에 기여하겠습니다.")
    answer = PORTFOLIO_HUMAN + "\n## 그 밖의 경험\n" + "\n".join(f"- {line}" for line in mine.splitlines())
    assert not style(answer).passed
    res = style(answer, career_source=SOURCE + "\n" + mine)
    assert res.passed and "상투 표현 0개" in res.log


def test_thresholds_separate_made_up_human_and_ai_examples():
    """기준 숫자의 근거 — 지어낸 예시로 잰 값: 사람 글 0개, AI 티 글은 1,000자당 30개 이상 (기준 8개는 그 사이, 사람 쪽에 가깝게)."""
    human, ai = career.check_style(cover({1: HUMAN}), questions=[1]), career.check_style(cover({1: AI_LIKE}), questions=[1])
    assert len(human.hits) == 0 and human.per_1000 == 0
    assert len(ai.hits) >= career.STYLE_MIN_HITS * 2 and ai.per_1000 >= career.STYLE_PER_1000 * 3
    assert (career.STYLE_MIN_HITS, career.STYLE_PER_1000, career.STYLE_REPEAT, career.STYLE_MARKS) == (5, 8.0, 3, 3)


# ─── 프롬프트와 검사가 같은 목록을 쓴다 ───────────────────────────────────────


def test_prompt_and_checker_share_one_phrase_list(ranks, models):
    phrases = career.avoid_phrases()
    assert len(phrases) >= 30 and {p.group for p in phrases} == {"상투적 연결", "틀에 박힌 맺음", "뜻이 빈 꾸밈말", "번역투·한자어"}
    for node_id in ("am_writer", "gm_role", "lead"):
        prompt = _career_prompt(node_id, ranks, models)
        assert "[문체 — 본인 글처럼 쓰기" in prompt and "내 문서 원문" in prompt
        assert "문서에 없는 장면·감정·숫자" in prompt  # 자연스럽게 하려고 사실을 보태지 않는다
        missing = [p.text for p in phrases if p.text not in prompt]
        assert not missing, missing  # 검사하는 표현은 모두 프롬프트에 있다
    listed = _career_prompt("lead", ranks, models).split("[문체 — 피할 표현", 1)[1].split("\n[", 1)[0]
    in_prompt = {item.strip() for line in listed.splitlines() if line.startswith("- ")
                 for item in line[2:].split(": ", 1)[1].split(" → ")[0].split(" / ")}
    assert in_prompt == {p.text for p in phrases}  # 프롬프트의 목록과 검사 목록이 같다
    assert "[문체 —" not in _career_prompt("staff_facts", ranks, models)  # 사실 카드만 쓰는 사원에게는 넣지 않는다


def test_old_instructions_that_invite_formula_writing_are_gone(ranks, models):
    prompt = _career_prompt("am_writer", ranks, models)
    assert "60자 안팎" not in prompt  # 모든 문장을 같은 길이로 맞추게 하던 말
    assert "마지막 문단은 '이 경험을 지원 직무의 어떤 일에 쓰는지'" not in prompt  # 답변마다 다짐으로 끝내게 하던 말
    assert "문장 길이를 섞는다" in prompt and "다짐·교훈 한 줄로 끝내지 않는다" in prompt


def test_head_rules_and_review_criteria_cover_style_without_counting():
    team = load_team("career")
    rules = "\n".join(team.verification["critic"]["rules"])
    assert "상투 표현의 개수·되풀이는 코드가 세어" in rules and "빈 문단" in rules  # 개수는 코드가, 부장은 구체적인 사실이 없는 문단을
    assert "장면·감정" in rules  # 자연스럽게 고치면서 지어낸 것은 여전히 위반
    assert any("본인이 쓴 글처럼" in c for c in team.review_criteria)
    assert team.private  # 비공개 팀 규칙은 그대로 (부장 비평은 팀장 모델로만)


# ─── 어디서 도는가 ─────────────────────────────────────────────────────────


def test_runs_only_on_the_final_answer_of_writing_tasks():
    team = load_team("career")
    assert "career_style" in team.harness.final_only and "career_style" not in team.harness.also_at
    tasks = {t.id: t for t in team.tasks}
    for task_id in ("cover_letter", "portfolio", "portfolio_update"):
        assert "career_style" in tasks[task_id].validators  # 업무 양식의 검증기 → 본 실행과 💬 수정 요청의 수정본에 함께 쓰인다
    assert "career_style" not in tasks["jd_recommend"].validators  # 추천 결과는 제출하는 글이 아니다


def test_fake_model_final_answer_with_ai_style_is_rewritten(tmp_path, ranks, models, use_script):
    src = tmp_path / "ab12cd34_요약.md"
    src.write_text(SOURCE, encoding="utf-8")
    coverage = [{"id": "R1", "status": "근거 있음", "quote": "보안 로그 분석 자동화"},
                {"id": "R2", "status": "부분", "quote": "일일 분류 시간을 180분에서 45분으로 줄였다."}, {"id": "R3", "status": "공백", "quote": ""}]
    tail = summary([("분류 시간 단축", "일일 분류 시간을 180분에서 45분으로 줄였다.")], coverage)
    natural = ("보안 로그 분류를 자동화했습니다. 일일 분류 시간이 180분에서 45분으로 줄었습니다. "
               "오탐으로 닫던 이벤트는 하루 1,200건을 자동으로 분류했고, 남은 시간에는 탐지 규칙을 고쳤습니다.")
    ai, good = cover({1: AI_LIKE}, tail), cover({1: natural}, tail)
    # 사원(사실 카드) → 대리가 AI 티 글을 씀 → 부장 승인 → 팀장 승인 → 문체 검사 실패 → 팀장이 다시 씀
    reviews = [make_review("revise", 8, ai), make_review("approve", 9), make_review("approve", 9), make_review("revise", 9, good)]
    events, script = run_task("cover_letter", {"docs": [str(src)], "docs_names": ["요약.md"], "jd": JD, "questions": "지원 동기"},
                              reviews, ranks, models, use_script)
    steps = {p.node_id: p for k, p in events if k == "step"}
    for node_id in ("staff_facts", "am_writer", "gm_role"):  # 팀원 초안·중간 검토에는 돌지 않는다
        assert "career_style" not in {h.validator for h in steps[node_id].harness}
    assert steps["am_writer"].attempts == 1 and steps["am_writer"].content == ai
    final = steps["lead"]
    assert final.content == good and final.attempts == 2
    assert {h.validator for h in final.harness} == {"career_facts", "char_limits", "jd_coverage", "career_style"}
    assert all(h.passed for h in final.harness), [h.log for h in final.harness if not h.passed]
    feedback = next(p for p in script.prompts("ReviewResult") if "자동 검증(하네스)" in p)
    assert "[career_style]" in feedback and "이러한 경험을 바탕으로" in feedback
    assert len(script.prompts("ReviewResult")) == 4  # 문체 실패로 는 호출은 팀장 재작성 1번


def test_followup_revision_gets_the_same_style_check(org, ranks, team, models, use_script):
    def edit(answer: str) -> FollowupResult:
        return FollowupResult(analysis="수정 요청", revised=True, reply="고쳤습니다.", changes=["고침"], revised_answer=answer)

    script = use_script(Script(FollowupResult=[edit(cover({1: AI_LIKE})), edit(cover({1: HUMAN}))]))
    turn = followup.respond(lead_agent(org, ranks, team, models), "원 질문", cover({1: HUMAN}), "포부를 더 써줘", [],
                            validators=["career_style"], options={"career_questions": QUESTIONS}, max_retries=1)
    assert turn.revised and turn.content == cover({1: HUMAN}) and all(h.passed for h in turn.harness)
    assert "이러한 경험을 바탕으로" in script.prompts("FollowupResult")[1]


# ─── 사실 규칙은 그대로 ───────────────────────────────────────────────────


@pytest.mark.parametrize("invent", [False, True])
def test_fact_check_is_unchanged_for_naturally_written_text(invent):
    """자연스러운 글도 문서에 없는 숫자를 쓰면 사실 검사에서 여전히 실패한다 (문체 검사는 사실을 보지 않는다)."""
    text = HUMAN.replace("45분으로 줄었습니다", "30분으로 줄었습니다") if invent else HUMAN
    answer = cover({1: text}, summary([("분류 시간 단축", "일일 분류 시간을 180분에서 45분으로 줄였다.")]))
    reference = JD + "\n지원 동기 3년 2022년 4월"  # 문항·메모에 적힌 숫자
    assert style(answer, career_questions=QUESTIONS, career_source=SOURCE, career_reference=reference).passed
    facts = check("career_facts", answer, career_source=SOURCE, career_reference=reference)
    assert facts.passed is not invent, facts.log
