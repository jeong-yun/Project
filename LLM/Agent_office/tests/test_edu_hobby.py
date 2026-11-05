"""🌍 취미·교양 교육팀 (3단계) — 커리큘럼 · 단어·용어장 · 상황별 표현 · 추가 지식 · 복습(간격 반복) · 화면 흐름.

모델은 모두 가짜(use_script · 가짜 실행) — .env 의 실제 키로 진짜 모델을 부르지 않는다.
샘플 과목·선생님은 모두 가상 인물이다 (예: 이탈리아어 · 입문 · 여행 회화 / 유럽 역사 · 입문).
날짜 계산은 today 를 직접 넘겨 시간에 흔들리지 않게 한다. CSV 는 BOM 과 상관없이 읽는다.
"""

import importlib
import json
from datetime import date, timedelta

import pytest

from core import learning
from core import learning_hobby as hobby
from core.config import BASE_DIR, list_teams
from core.harness import REGISTRY
from core.schemas import HumanDecision
from core.tools import TOOLS
from tests.fakes import Script

TODAY = date(2026, 11, 5)


def block(data: dict) -> str:
    return "## 본문\n설명\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


def italian(course_id="hitaly01", level="beginner") -> learning.Course:
    """가상 학습자 — 이탈리아어 · 입문 · 여행 회화 (선생님도 가상 인물)."""
    tutor = learning.TutorProfile(name="루치아 선생님", specialty="여행 이탈리아어", background="가상 — 피렌체 어학원 강사 8년",
                                  teaching_style="상황극 위주, 짧은 문장부터")
    units = [learning.Unit(id="U1", week=1, title="인사와 숫자", goals=["인사할 수 있다"], done=True),
             learning.Unit(id="U2", week=2, title="카페에서 주문", goals=["커피를 주문한다"]),
             learning.Unit(id="U3", week=3, title="길 묻기", goals=["역이 어딘지 묻는다"])]
    return learning.Course(id=course_id, kind="hobby", topic="이탈리아어", goal="이탈리아 여행 회화", level=level, language=True,
                           weekly_hours=3, tutor=tutor, curriculum=units, created_at="2026-11-01T09:00:00", updated_at="2026-11-01T09:00:00")


def history(course_id="heuro01") -> learning.Course:
    """가상 학습자 — 유럽 역사 · 입문 · 교양."""
    tutor = learning.TutorProfile(name="한서 선생님", specialty="유럽사 입문", background="가상 — 박물관 해설사 10년")
    return learning.Course(id=course_id, kind="hobby", topic="유럽 역사", goal="박물관 관람이 즐거워지게", level="beginner",
                           weekly_hours=2, tutor=tutor, curriculum=[], created_at="2026-11-01T09:00:00")


VOCAB = {"vocab": [
    {"term": "il conto", "meaning": "계산서", "reading": "일 꼰또", "pos": "명사(남성)", "example": "Il conto, per favore.",
     "example_ko": "계산서 주세요.", "level": "beginner"},
    {"term": "grazie", "meaning": "고마워요", "reading": "그라찌에", "pos": "감탄사", "example": "Grazie mille!",
     "example_ko": "정말 고마워요!", "level": "beginner"},
    {"term": "la stazione", "meaning": "역", "reading": "라 스따찌오네", "pos": "명사(여성)", "example": "Dov'è la stazione?",
     "example_ko": "역이 어디예요?", "level": "beginner"},
    {"term": "il biglietto", "meaning": "표", "reading": "일 빌리에또", "pos": "명사(남성)", "example": "Un biglietto, per favore.",
     "example_ko": "표 한 장 주세요.", "level": "beginner"},
]}
PHRASES = {"phrases": [
    {"situation": "🍝 식당·카페", "phrase": "Un caffè, per favore.", "reading": "운 까페, 뻬르 파보레", "meaning": "커피 한 잔 주세요.",
     "level": "beginner", "dialogue": ["A: Un caffè, per favore.", "B: Subito!"], "tip": "이탈리아에서 caffè 는 에스프레소다."},
    {"situation": "🍝 식당·카페", "phrase": "Il conto, per favore.", "reading": "일 꼰또, 뻬르 파보레", "meaning": "계산서 주세요.",
     "level": "beginner", "dialogue": ["A: Il conto, per favore.", "B: Certo."], "tip": "자리에서 손을 들어 부른다."},
]}
TOPICS = {"topics": [
    {"title": "로마사 첫걸음", "why": "유적 이름과 라틴어 뿌리 낱말이 쉬워진다", "first_unit": "로마는 어떻게 커졌나", "field": "역사"},
    {"title": "르네상스 미술", "why": "피렌체 미술관 관람이 즐거워진다", "first_unit": "보티첼리와 메디치 가문", "field": "예술"},
    {"title": "이탈리아 음식 문화", "why": "식당 표현을 상황 속에서 익힌다", "first_unit": "지역별 대표 음식", "field": "문화"},
]}


def curriculum(weeks=4, hours=(1.5, 1.5), skills=None, theme=None):
    skills = skills or [["듣기", "말하기"], ["읽기", "쓰기"]]
    units = []
    for w in range(1, weeks + 1):
        for j, h in enumerate(hours):
            u = {"week": w, "title": f"{w}주 단원 {j + 1}", "goals": [f"{w}주 목표 {j + 1}"], "hours": h}
            if theme is None:
                u["skills"] = skills[j % len(skills)]
            else:
                u["theme"] = theme
            units.append(u)
    return {"curriculum": units}


# ─── 팀 설정 · 등록 ──────────────────────────────────────────────────


def test_hobby_tasks_are_after_base_tasks_and_registered():
    import education_ui
    importlib.import_module("edu_hobby_ui")  # 등록

    team = next(t for t in list_teams() if t.id == "edu_hobby")
    ids = [t.id for t in team.tasks]
    assert ids[:3] == ["create_tutor", "quiz", "ask"] and ids[3:] == ["hobby_curriculum", "vocab", "phrases", "topics", "review"]
    for task in team.tasks[3:]:
        assert task.fixed == {"edu_kind": "hobby", "edu_task": task.id}
        assert any(f.key == "course_id" and f.required for f in task.fields)
        assert set(task.validators) <= set(team.harness.final_only)  # 최종 답변에만 (팀원 초안마다 돌지 않게)
        assert all(v in REGISTRY for v in task.validators)
    assert [t for _, t in education_ui.QUICK_TASKS["hobby"]] == ["hobby_curriculum", "vocab", "phrases", "topics", "review"]
    assert {"hobby_curriculum", "vocab", "phrases", "topics", "review"} <= set(education_ui.PREVIEW_HOOKS)
    assert {"hobby_curriculum", "vocab", "phrases", "topics"} <= set(education_ui.ADOPT_HOOKS)
    assert len(education_ui.COURSE_SECTIONS["hobby"]) == 4 and "review" in education_ui.ANSWER_TASKS
    assert (BASE_DIR / "teams" / "edu_hobby" / "skills" / "hobby_teaching.md").is_file()


def test_review_quiz_answers_are_hidden_in_result_box():
    import education_ui

    text = "## 문제\n1. 문제\n\n```json\n{\"quiz\": {\"questions\": []}}\n```"
    shown = education_ui.result_text({"edu_inputs": {"edu_task": "review"}}, text)
    assert "```json" not in shown and "정답·해설은 감췄습니다" in shown
    assert education_ui.result_text({"edu_inputs": {"edu_task": "vocab"}}, text) == text


# ─── 📅 커리큘럼 ──────────────────────────────────────────────────────


def test_curriculum_validator_checks_format_hours_and_balance():
    v = REGISTRY["hobby_curriculum"]
    opts = {"weekly_hours": "3", "weeks": "4", "language": True}
    assert v.check(block(curriculum()), opts).passed
    over = v.check(block(curriculum(hours=(2, 1.5))), opts)  # 2 + 1.5 = 3.5 > 3 (0.1시간 정수로 더함)
    assert not over.passed and "시간 합 3.5시간 > 주당 3시간" in over.log and "형식 예" in over.log
    assert v.check(block(curriculum(hours=(1.2, 1.8))), opts).passed  # 1.2 + 1.8 = 3.0 — 부동소수점 오차 없이 통과
    one_sided = v.check(block(curriculum(skills=[["말하기"]])), opts)
    assert not one_sided.passed and "듣기·읽기·쓰기" in one_sided.log
    gap = curriculum()
    gap["curriculum"] = [u for u in gap["curriculum"] if u["week"] != 2]
    res = v.check(block(gap), opts)
    assert not res.passed and "빈 주차" in res.log
    assert "8주로" in v.check(block(curriculum()), {**opts, "weeks": "8"}).log
    bad = curriculum()
    bad["curriculum"][0].pop("goals")
    bad["curriculum"][1]["hours"] = "많이"
    log = v.check(block(bad), opts).log
    assert "goals" in log and "hours" in log
    assert not v.check("JSON 없음", opts).passed
    # 교양 과목: 영역 대신 시대·주제 — 없으면 경고만
    culture = v.check(block(curriculum(weeks=2, hours=(0.5,), theme="")), {"weekly_hours": "2", "weeks": "2", "language": False})
    assert culture.passed and "theme" in culture.log and "0.5시간뿐" in culture.log


def test_save_curriculum_keeps_done_units_and_continues_weeks():
    c = italian()
    removed = hobby.save_curriculum(c, curriculum(weeks=2))
    assert removed == ["카페에서 주문", "길 묻기"]
    assert [u.id for u in c.curriculum] == ["U1", "U4", "U5", "U6", "U7"]
    assert c.curriculum[0].done and [u.week for u in c.curriculum] == [1, 2, 2, 3, 3]  # 끝낸 1주 다음부터
    assert c.curriculum[1].goals[-1] == "🕐 1.5시간 · 듣기·말하기"
    with pytest.raises(ValueError):
        hobby.save_curriculum(c, {"curriculum": []})


# ─── 📖 단어·용어장 ───────────────────────────────────────────────────


def test_vocab_validator_fields_duplicates_and_level_warning():
    v = REGISTRY["vocab_list"]
    lang = {"language": True, "level": "beginner", "vocab_count": "4"}
    assert v.check(block(VOCAB), lang).passed
    missing = json.loads(json.dumps(VOCAB))
    missing["vocab"][0]["reading"] = ""
    missing["vocab"][1].pop("example_ko")
    res = v.check(block(missing), lang)
    assert not res.passed and "발음(읽는 법)" in res.log and "예문 번역" in res.log
    dup = {"vocab": VOCAB["vocab"] + [{**VOCAB["vocab"][1], "term": "Grazie!"}]}
    res = v.check(block(dup), lang)
    assert not res.passed and "같은 단어" in res.log and "grazie" in res.log
    hard = json.loads(json.dumps(VOCAB))
    for item in hard["vocab"][:2]:
        item["level"] = "advanced"
    res = v.check(block(hard), lang)
    assert res.passed and "⚠" in res.log and "어려운 단어" in res.log  # 수준 범위는 경고만
    assert "⚠" not in v.check(block(hard), {**lang, "level": "intermediate"}).log  # 중급 이상은 보지 않음
    # 교양 과목: 용어 · 풀이 · 연도·인물
    culture = {"vocab": [{"term": "르네상스", "meaning": "고대 문화를 되살린 움직임", "related": "14~16세기 · 피렌체", "level": "beginner"}]}
    assert v.check(block(culture), {"language": False, "level": "beginner"}).passed
    res = v.check(block({"vocab": [{"term": "십자군", "meaning": "풀이"}]}), {"language": False, "level": "beginner"})
    assert not res.passed and "연도·인물" in res.log


def test_add_vocab_merges_same_term_and_keeps_review_state():
    c = italian()
    added, merged = hobby.add_vocab(c, VOCAB, now="2026-11-05T10:00:00")
    assert (added, merged) == (4, 0) and c.extras["vocab"][0]["known"] is False and c.extras["vocab"][0]["due"] == ""
    c.extras["vocab"][1].update(known=True, box=3, due="2026-11-20", example="")
    again = {"vocab": [{"term": "Grazie!", "meaning": "감사합니다", "example": "Grazie a te.", "level": "beginner"},
                       {"term": "ciao", "meaning": "안녕", "reading": "차오", "level": "beginner"},
                       {"term": "CIAO", "meaning": "안녕(반말)", "level": "beginner"}]}
    added, merged = hobby.add_vocab(c, again)
    assert (added, merged) == (1, 2)  # ciao 새로 · grazie 합침 · 같은 답 안의 CIAO 합침
    g = c.extras["vocab"][1]
    assert g["term"] == "grazie" and g["meaning"] == "고마워요" and g["example"] == "Grazie a te."  # 빈 칸만 채움
    assert g["known"] is True and g["box"] == 3 and g["due"] == "2026-11-20"  # 외움·복습 기록은 지킴
    assert [v["term"] for v in c.extras["vocab"]][-1] == "ciao" and len(c.extras["vocab"]) == 5


def test_vocab_csv_for_anki_and_markdown():
    c = italian()
    hobby.add_vocab(c, {"vocab": VOCAB["vocab"] + [{"term": 'say "ciao", ok', "meaning": "따옴표, 쉼표", "level": "beginner"}]})
    c.extras["vocab"][0]["known"] = True
    text = hobby.vocab_csv(c.extras["vocab"])
    head = text.splitlines()[:3]
    assert head[0] == "#separator:Comma" and head[1] == "#html:false" and head[2].startswith("#columns:단어,뜻,읽는 법")
    rows = hobby.parse_vocab_csv("﻿" + text)  # BOM 이 있든 없든 같은 결과
    assert rows == hobby.parse_vocab_csv(text)
    assert rows[0]["term"] == "il conto" and rows[0]["meaning"] == "계산서" and rows[0]["known"] == "예" and rows[0]["level"] == "입문(신규)"
    assert rows[-1]["term"] == 'say "ciao", ok' and rows[-1]["meaning"] == "따옴표, 쉼표"  # 쉼표·따옴표도 칸이 깨지지 않음
    md = hobby.vocab_markdown(c, c.extras["vocab"])
    assert md.startswith("# 📖 이탈리아어 단어장 — 입문(신규) · A1") and "| il conto | 일 꼰또 | 명사(남성) | 계산서 |" in md
    assert "외움 1개" in md
    assert [i["term"] for i in hobby.search(c.extras["vocab"], "STAZ", ("term",))] == ["la stazione"]
    assert [i["term"] for i in hobby.search(c.extras["vocab"], "표 한 장", ("term", "example_ko"))] == ["il biglietto"]


# ─── 🗣 상황별 표현 · 🧭 추가 지식 ──────────────────────────────────────


def test_phrase_validator_and_add_phrases():
    v = REGISTRY["phrase_list"]
    opts = {"language": True, "situation": "🍝 식당·카페"}
    assert v.check(block(PHRASES), opts).passed
    res = v.check(block(PHRASES), {**opts, "situation": "🚑 응급·병원"})
    assert res.passed and "응급·병원" in res.log  # 고른 상황이 없으면 경고만
    bare = {"phrases": [{"situation": "🏨 호텔", "phrase": "Ho una prenotazione.", "meaning": "예약했어요."}]}
    res = v.check(block(bare), {"language": True, "situation": "🏨 호텔"})
    assert not res.passed and "발음" in res.log
    assert v.check(block(bare), {"language": False, "situation": "🏨 호텔"}).passed
    dup = {"phrases": PHRASES["phrases"] + [{**PHRASES["phrases"][0], "phrase": "un caffè, per favore"}]}
    assert "같은 표현" in v.check(block(dup), opts).log
    c = italian()
    assert hobby.add_phrases(c, PHRASES) == (2, 0)
    assert hobby.add_phrases(c, PHRASES) == (0, 2)
    p = c.extras["phrases"][0]
    assert p["dialogue"] == ["A: Un caffè, per favore.", "B: Subito!"] and p["tip"].startswith("이탈리아에서") and p["known"] is False
    md = hobby.phrases_markdown(c, c.extras["phrases"])
    assert "## 🍝 식당·카페" in md and "**Un caffè, per favore.** (운 까페, 뻬르 파보레) — 커피 한 잔 주세요." in md


def test_topic_validator_add_and_new_course_prefill():
    v = REGISTRY["topic_list"]
    assert v.check(block(TOPICS), {}).passed
    assert "3~5개" in v.check(block({"topics": TOPICS["topics"][:2]}), {}).log
    bad = {"topics": TOPICS["topics"] + [{"title": "로마사 첫걸음", "why": ""}]}
    log = v.check(block(bad), {}).log
    assert "같은 주제" in log and "첫 단원" in log
    c = italian()
    assert hobby.add_topics(c, TOPICS) == (3, 0) and hobby.add_topics(c, TOPICS) == (0, 3)
    pre = hobby.new_course_prefill(c, c.extras["topics"][1])
    assert pre["topic"] == "르네상스 미술" and pre["level"] == "입문(신규)" and pre["language"] is False
    assert "이탈리아어" in pre["goal"] and "보티첼리" in pre["goal"]


# ─── 🔁 복습 (간격 반복) ───────────────────────────────────────────────


def test_schedule_spaced_repetition_intervals():
    item = {"term": "grazie"}
    days = []
    for _ in range(7):  # 계속 맞히면 1 → 2 → 4 → 7 → 15 → 30 → 30일
        hobby.schedule(item, True, TODAY)
        days.append((date.fromisoformat(item["due"]) - TODAY).days)
    assert days == [2, 4, 7, 15, 30, 30, 30] and item["box"] == 5 and item["reviews"] == 7
    item["known"] = True
    hobby.schedule(item, False, TODAY)  # 틀리면 처음부터, 내일 다시 · 외움도 풀림
    assert item["box"] == 0 and item["due"] == (TODAY + timedelta(days=1)).isoformat() and item["wrong"] == 1 and not item["known"]
    assert item["last_review"] == TODAY.isoformat()


def test_due_items_skip_known_and_future_but_bring_back_stale():
    c = italian()
    hobby.add_vocab(c, VOCAB, now="2026-10-01T10:00:00")
    hobby.add_phrases(c, PHRASES, now="2026-10-01T10:00:00")
    v = c.extras["vocab"]
    v[0].update(due="2026-11-03", box=1)  # 지난 복습일 — 가장 먼저
    v[1].update(due="2026-11-10", box=2)  # 아직 — 빠짐
    v[2].update(known=True, last_review="2026-11-01")  # 외움 · 최근 복습 — 빠짐
    v[3].update(known=True, last_review="2026-09-01")  # 외움이지만 30일 넘음 — 맨 뒤에 다시
    picked = [(k, it.get("term") or it.get("phrase")) for k, it in hobby.due_items(c, TODAY)]
    # 안 본 표현(10-01에 더함) → 복습일 지난 단어(11-03) → 외웠지만 오래된 단어 · grazie(아직)·la stazione(최근 외움)는 빠짐
    assert picked == [("phrases", "Un caffè, per favore."), ("phrases", "Il conto, per favore."), ("vocab", "il conto"),
                      ("vocab", "il biglietto")]
    assert len(hobby.due_items(c, TODAY, limit=2)) == 2
    assert hobby.is_due({"due": "2026-11-05"}, TODAY) and not hobby.is_due({"due": "2026-11-06"}, TODAY)


def test_flashcard_quiz_uses_shared_quiz_format_and_grading():
    c = italian()
    hobby.add_vocab(c, VOCAB)
    hobby.add_phrases(c, PHRASES)
    quiz, refs = hobby.flashcard_quiz(c, TODAY, count=5)
    assert len(quiz.questions) == 5 and quiz.kind == "review"
    assert learning.quiz_problems(quiz.model_dump(), kind="review") == []  # 공용 quiz_format 규칙 그대로
    assert REGISTRY["quiz_format"].check(block({"quiz": quiz.model_dump()}), {"quiz_count": "5"}).passed
    q1 = quiz.questions[0]
    assert refs[q1.id] == "il conto" and q1.answer == "계산서" and len(q1.choices) == 4 and "il conto" in q1.question
    phrase_q = next(q for q in quiz.questions if refs[q.id] == "Un caffè, per favore.")
    assert phrase_q.answer == "Un caffè, per favore." and "식당·카페" in phrase_q.question
    assert phrase_q.choices and set(phrase_q.choices) == {"Un caffè, per favore.", "Il conto, per favore."}
    again, _ = hobby.flashcard_quiz(c, TODAY, count=5)
    assert again == quiz  # 같은 과목·날짜면 같은 퀴즈 (보기 자리도 코드로 정함)
    answers = {q.id: q.answer for q in quiz.questions}
    answers[q1.id] = next(ch for ch in q1.choices if ch != q1.answer)
    grade = learning.grade_quiz(quiz, answers)
    assert grade.correct == 4 and grade.wrong == [q1.id]
    results = {refs[q.id]: q.id not in grade.wrong for q in quiz.questions}
    assert hobby.apply_review(c, results, TODAY) == 5
    assert c.extras["vocab"][0]["due"] == "2026-11-06" and c.extras["vocab"][1]["due"] == "2026-11-07"
    assert hobby.flashcard_quiz(learning.Course.model_validate({**italian().model_dump(), "extras": {"vocab": VOCAB["vocab"][:1]}}), TODAY) is None


def test_set_known_from_table():
    c = italian()
    hobby.add_vocab(c, VOCAB)
    assert hobby.set_known(c, "vocab", {"Grazie": True, "il conto": False}) == 1
    assert [v["known"] for v in c.extras["vocab"]] == [False, True, False, False]


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    return learning.LearningStore(tmp_path / "data")


def review_answer(refs=("il conto", "Grazie")) -> str:
    qs = [{"id": f"Q{i}", "question": f"{r} 문제", "choices": ["가", "나", "다", "라"], "answer": "나", "explanation": "해설", "ref": r}
          for i, r in enumerate(refs, 1)]
    return block({"quiz": {"kind": "review", "level": "beginner", "questions": qs}})


def test_review_quiz_validator_checks_refs_against_course(data_env):
    c = italian()
    hobby.add_vocab(c, VOCAB)
    data_env.save(c)
    v = REGISTRY["review_quiz"]
    assert v.check(review_answer(), {"course_id": c.id, "quiz_count": "2"}).passed
    res = v.check(review_answer(("il conto", "arrivederci")), {"course_id": c.id})
    assert not res.passed and "arrivederci" in res.log
    no_ref = review_answer().replace('"ref": "Grazie"', '"ref": ""')
    assert "ref(복습한" in v.check(no_ref, {"course_id": c.id}).log
    assert not v.check(review_answer(), {"course_id": c.id, "quiz_count": "5"}).passed  # 공용 문제 수 규칙
    assert v.check(review_answer(("아무거나",)), {}).passed  # 과목을 모르면 ref 존재 확인은 건너뜀
    assert hobby.review_refs(review_answer()) == {"Q1": "il conto", "Q2": "Grazie"}


def test_tool_context_shows_vocab_phrases_topics_and_review_targets(data_env):
    c = italian()
    hobby.add_vocab(c, VOCAB)
    hobby.add_phrases(c, PHRASES)
    hobby.add_topics(c, TOPICS)
    c.extras["vocab"][1]["known"] = True
    c.extras["vocab"][1]["last_review"] = date.today().isoformat()
    data_env.save(c)
    res = TOOLS["learning_course"]("q", {"course_id": c.id, "edu_kind": "hobby", "edu_task": "vocab"}, None)
    text = res.content
    assert "### 🗂 필수 단어·용어장" in text and "모두 4개 · 외움 1개" in text and "| il conto | 일 꼰또 | 계산서 | 입문(신규) |  |" in text
    assert "🔁 복습 대상" in text and "- il conto = 계산서" in text and "- grazie = 고마워요" not in text
    assert "### 🗂 상황별 표현" in text and "[식당·카페] Un caffè, per favore. = 커피 한 잔 주세요." in text
    assert "### 🗂 추가 지식(역사·문화) 제안" in text and "  - 로마사 첫걸음" in text
    assert res.harness_options["language"] is True and res.harness_options["level"] == "beginner"


# ─── 채택 → 과목 저장 (화면 없이 hook 직접) ─────────────────────────────


def _ctx(data_env):
    import education_ui

    return education_ui.Ctx(data_env.dir.parent, None, {}, "task", "team")


def _adopt(ctx, task, course_id, answer):
    import education_ui
    importlib.import_module("edu_hobby_ui")

    run = {"edu_inputs": {"edu_task": task, "course_id": course_id}, "run_id": "r1"}
    education_ui.on_adopt(ctx, run, HumanDecision(action="adopt", final_answer=answer))
    return run


def test_adopt_each_task_saves_to_course(data_env):
    ctx = _ctx(data_env)
    data_env.save(italian())
    run = _adopt(ctx, "hobby_curriculum", "hitaly01", block(curriculum(weeks=2)))
    assert "커리큘럼을 과목에 저장" in run["edu_saved"]
    assert [u.title for u in data_env.get("hitaly01").curriculum][:2] == ["인사와 숫자", "1주 단원 1"]
    run = _adopt(ctx, "vocab", "hitaly01", block(VOCAB))
    assert run["edu_saved"] == "📖 단어장에 4개를 더했습니다."
    run = _adopt(ctx, "vocab", "hitaly01", block({"vocab": VOCAB["vocab"][:1]}))
    assert "같은 단어 1개는 합침" in run["edu_saved"] and len(data_env.get("hitaly01").extras["vocab"]) == 4
    assert "2개를 더했습니다" in _adopt(ctx, "phrases", "hitaly01", block(PHRASES))["edu_saved"]
    assert "3개를 과목에 저장" in _adopt(ctx, "topics", "hitaly01", block(TOPICS))["edu_saved"]
    saved = data_env.get("hitaly01")
    assert len(saved.extras["phrases"]) == 2 and len(saved.extras["topics"]) == 3
    bad = _adopt(ctx, "vocab", "hitaly01", "JSON 없음")
    assert "edu_saved" not in bad and "단어장 JSON 이 없습니다" in bad["edu_error"]
    missing = _adopt(ctx, "topics", "cnone", block(TOPICS))
    assert "과목이 없습니다" in missing["edu_error"]


# ─── 화면 (AppTest): 단어장 채택 → 표현 채택 → 복습 퀴즈(간격 반복) → ⚡ 바로 복습 → 새 과목으로 ─────


@pytest.fixture
def hobby_app(tmp_path, monkeypatch, use_script):
    """가짜 실행 — 질문에 맞는 가짜 최종 답 (진짜 모델을 부르지 않는다) · 임시 데이터 폴더 · 가상 이탈리아어 과목."""
    import streamlit as st

    from core import live, workflow
    from core.schemas import AgentAnswer, ModelSpec

    use_script(Script())
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    store = learning.LearningStore(tmp_path / "data")
    store.save(italian())
    answers = {"복습 퀴즈": review_answer(("il conto", "grazie", "Un caffè, per favore.")), "단어장": block(VOCAB),
               "추가 지식": block(TOPICS), "표현": block(PHRASES)}  # 앞에서부터 질문에 든 말로 고른다 (복습 질문에도 '표현'이 있다)
    started = []

    def fake_start(run, events_factory, on_step=None, on_finish=None):
        started.append(run["question"])
        content = next(a for k, a in answers.items() if k in run["question"])
        step = AgentAnswer(node_id=run["root_id"], rank_id="team_lead", kind="integrate", content=content,
                           model=ModelSpec(provider="openai", model="x"))
        run.update(steps=[step], done=True, status="awaiting_human", activity={}, statuses={run["root_id"]: ("wait", "👤 결정 대기")})
        if on_finish:
            on_finish(run)

    monkeypatch.setattr(live, "start_run", fake_start)
    monkeypatch.setattr(workflow, "resume_with_decision", lambda *a, **k: iter([("lessons", [])]))
    st.cache_resource.clear()
    yield {"store": store, "started": started}
    st.cache_resource.clear()


def _click(at, label=None, key=None):
    btn = next(b for b in at.button if (key and b.key == key) or (label and b.label.startswith(label)))
    at = btn.click().run()
    assert not at.exception, at.exception
    return at


def _run_task(at, quick_key, task_id):
    at = _click(at, key=quick_key)
    assert at.selectbox(key="dept_task_education").value == f"edu_hobby:{task_id}"
    assert at.text_input(key=f"tf_edu_hobby_{task_id}_course_id").value == "hitaly01"
    return _click(at, label="맡기기")


def test_app_vocab_phrases_review_and_new_course(hobby_app):
    from streamlit.testing.v1 import AppTest

    store = hobby_app["store"]
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.query_params["view"] = "dept"
    at.query_params["dept"] = "education"
    at.session_state["office_view"] = "office"
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button if b.key and b.key.endswith("_hitaly01")]
    for label in ["📅 커리큘럼", "📖 단어장", "🗣 표현", "🧭 더 배울 것", "🔁 복습"]:
        assert label in labels

    # ① 📖 단어장 → 미리 보기 → 채택 → extras["vocab"]
    at = _run_task(at, "edu_vocab_hitaly01", "vocab")
    assert any("📖 새 단어장 미리 보기" in m.value for m in at.markdown)
    at = _click(at, label="✅ 이 안으로 채택")
    assert any("단어장에 4개를 더했습니다" in s.value for s in at.success)
    assert [v["term"] for v in store.get("hitaly01").extras["vocab"]] == ["il conto", "grazie", "la stazione", "il biglietto"]
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")

    # ② 🗣 표현 → 채택 → extras["phrases"]
    at = _run_task(at, "edu_phrases_hitaly01", "phrases")
    assert any("🗣 상황별 표현 미리 보기" in m.value for m in at.markdown)
    at = _click(at, label="✅ 이 안으로 채택")
    assert len(store.get("hitaly01").extras["phrases"]) == 2
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")

    # ③ 🔁 복습 퀴즈 (모델) — 결과 상자에서 정답 감춤 → 풀기 → 채점 → 다음 복습일
    at = _run_task(at, "edu_review_hitaly01", "review")
    assert any("정답·해설은 감췄습니다" in m.value for m in at.markdown)
    assert any("🔁 복습 퀴즈 풀기 — 3문제" in m.value for m in at.markdown)
    prefix = next(r.key for r in at.radio if r.key and r.key.startswith("hb_review_")).rsplit("_", 1)[0]
    at.radio(key=f"{prefix}_Q1").set_value("나")
    at.radio(key=f"{prefix}_Q2").set_value("가")  # 틀림
    at.radio(key=f"{prefix}_Q3").set_value("나")
    at = _click(at, label="✅ 채점하기")
    assert any("점수 2/3 (66%)" in m.value for m in at.markdown)
    c = store.get("hitaly01")
    today = date.today()
    vocab = {v["term"]: v for v in c.extras["vocab"]}
    assert vocab["il conto"]["box"] == 1 and vocab["il conto"]["due"] == (today + timedelta(days=2)).isoformat()
    assert vocab["grazie"]["box"] == 0 and vocab["grazie"]["due"] == (today + timedelta(days=1)).isoformat()
    assert c.extras["phrases"][0]["box"] == 1 and c.quiz_results[-1].kind == "review" and c.quiz_results[-1].correct == 2
    at = _click(at, label="✅ 이 안으로 채택")  # 복습은 채점할 때 이미 저장 — 채택은 기록만 닫는다
    assert len(store.get("hitaly01").quiz_results) == 1
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")

    # ④ 📖 자세히 — 단어장 표 · 검색 · ⚡ 바로 복습 (모델 없이)
    at = _click(at, key="edu_open_hitaly01")
    assert any("**📖 단어장** — 4개" in m.value for m in at.markdown)
    assert any("**🗣 표현 모음** — 2개" in m.value for m in at.markdown)
    at = at.text_input(key="hb_vq_hitaly01").input("stazione").run()
    assert not at.exception, at.exception
    started = len(hobby_app["started"])
    at = _click(at, key="hb_flash_go_hitaly01")
    radios = [r for r in at.radio if r.key and r.key.startswith("hb_flash_hitaly01_")]
    assert radios and len(hobby_app["started"]) == started  # 모델을 부르지 않음
    for r in radios:
        r.set_value(r.options[0])
    at = _click(at, label="✅ 채점하기")
    assert any(m.value.startswith("**점수 ") for m in at.markdown)
    assert len(store.get("hitaly01").quiz_results) == 2
    at = _click(at, key="hb_flash_close_hitaly01")

    # ⑤ 🧭 더 배울 것 → 채택 → ➕ 새 과목으로 = 선생님 만들기 양식 미리 채움
    at = _run_task(at, "edu_topics_hitaly01", "topics")
    at = _click(at, label="✅ 이 안으로 채택")
    assert len(store.get("hitaly01").extras["topics"]) == 3
    assert any(b.key and b.key.startswith("hb_tp_") and "_new_" in b.key for b in at.button)  # 채택 직후 결과 아래에도 버튼
    # (AppTest 는 채택 뒤 다시 그리기에서 남은 결정 양식 칸 때문에 이 화면에서 더 누르지 못한다 — 새 화면에서 📖 자세히 › 🧭 로 확인)
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.query_params["view"] = "dept"
    at.query_params["dept"] = "education"
    at.session_state["office_view"] = "office"
    at.run()
    at = _click(at, key="edu_open_hitaly01")
    assert any("**🧭 더 배울 것** — 3개" in m.value for m in at.markdown)
    new_btn = next(b for b in at.button if b.key == "hb_tp_hitaly01_new_0")
    at = new_btn.click().run()
    assert not at.exception, at.exception
    assert at.selectbox(key="dept_task_education").value == "edu_hobby:create_tutor"
    assert at.text_input(key="tf_edu_hobby_create_tutor_topic").value == "로마사 첫걸음"
    assert "이탈리아어" in at.text_input(key="tf_edu_hobby_create_tutor_goal").value
