"""🎓 교육부 — 공용 데이터 모델·저장소·채점·레벨업·진도·툴 문맥·검증기·부서장 배정·화면 흐름.

모델은 모두 가짜(use_script) 또는 키 없음 — .env 의 실제 키로 진짜 모델을 부르지 않는다.
샘플 과목·선생님은 모두 가상 인물이다.
"""

import json
import threading

import pytest

from core import building, learning
from core.config import BASE_DIR, list_teams, load_models
from core.harness import REGISTRY
from core.tools import TOOLS
from tests.fakes import Script

TUTOR = {
    "tutor": {"name": "한결 선생님", "specialty": "보안 관리 입문", "background": "가상 — 보안 관제 10년, 사내 교육 5년",
              "teaching_style": "짧은 개념 → 사례 → 스스로 해 보기",
              "level_rules": {"beginner": "용어마다 쉬운 풀이", "elementary": "기본 용어는 그대로", "intermediate": "원리와 비교",
                              "advanced": "실무 판단", "expert": "토론"}},
    "curriculum": [{"week": 1, "title": "보안이란 무엇인가", "goals": ["기밀성·무결성·가용성을 말할 수 있다"]},
                   {"week": 2, "title": "계정과 비밀번호", "goals": ["좋은 비밀번호 규칙을 안다"]},
                   {"week": 3, "title": "네트워크 기초", "goals": ["방화벽이 하는 일을 안다"]},
                   {"week": 4, "title": "사고 대응 첫걸음", "goals": ["사고가 나면 누구에게 알릴지 안다"]}],
}
TUTOR_ANSWER = "## 선생님 소개\n한결 선생님입니다.\n\n## 커리큘럼\n| 주차 | 단원 |\n\n```json\n" + json.dumps(TUTOR, ensure_ascii=False) + "\n```"


def quiz_json(n=5, kind="levelup", level="elementary", answer_index=1, levels=None):
    qs = []
    for i in range(1, n + 1):
        choices = [f"Q{i} 보기{j}" for j in range(1, 5)]
        q = {"id": f"Q{i}", "question": f"{i}번 문제", "choices": choices, "answer": choices[answer_index], "explanation": f"{i}번 해설"}
        if levels:
            q["level"] = levels[i - 1]
        qs.append(q)
    return {"quiz": {"kind": kind, "level": level, "questions": qs}}


def quiz_answer(**kw) -> str:
    return "## 문제\n1. 1번 문제\n\n```json\n" + json.dumps(quiz_json(**kw), ensure_ascii=False) + "\n```"


def make_course(level="beginner", kind="career", course_id="ctest01") -> learning.Course:
    inputs = {"edu_kind": kind, "topic": "보안 관리자", "goal": "6개월 안에 보안 관리자 신입 합격", "level": learning.level_of(level).name,
              "weekly_hours": 5, "language": False}
    return learning.course_from_answer(TUTOR_ANSWER, inputs, course_id=course_id, now="2026-11-05T10:00:00")


# ─── 수준 ───────────────────────────────────────────────────────────


def test_levels_cefr_and_parsing():
    assert [lv.name for lv in learning.LEVELS] == ["입문(신규)", "초급", "중급", "고급", "전문"]
    assert [lv.cefr for lv in learning.LEVELS] == ["A1", "A2", "B1", "B2", "C1"]
    assert learning.level_label("elementary") == "초급" and learning.level_label("elementary", language=True) == "초급 · A2"
    for text, want in [("입문(신규)", "beginner"), ("신규", "beginner"), ("초급", "elementary"), ("중급", "intermediate"),
                       ("B2", "advanced"), ("expert", "expert"), ("", "beginner"), ("모름", "beginner")]:
        assert learning.parse_level(text) == want, text
    assert learning.next_level("advanced") == "expert" and learning.next_level("expert") is None


# ─── 과목 · 저장소 ────────────────────────────────────────────────────


def test_course_from_tutor_answer():
    c = make_course("elementary")
    assert c.kind == "career" and c.team_id == "edu_career" and c.level == "elementary" and c.weekly_hours == 5
    assert c.tutor.name == "한결 선생님" and c.tutor.rule_for("beginner").startswith("용어마다 쉬운 풀이")
    assert [u.id for u in c.curriculum] == ["U1", "U2", "U3", "U4"] and all(u.level == "elementary" for u in c.curriculum)
    assert c.level_history[0].how == "start" and c.progress == 0 and learning.next_unit(c).id == "U1"
    with pytest.raises(ValueError, match="tutor.name"):
        learning.course_from_answer('```json\n{"tutor": {}, "curriculum": []}\n```', {"edu_kind": "hobby"})
    with pytest.raises(ValueError, match="JSON"):
        learning.parse_tutor_answer("JSON 없음")


def test_store_roundtrip_list_update_delete(tmp_path):
    store = learning.LearningStore(tmp_path)
    a, b = make_course(course_id="ca"), make_course(kind="hobby", course_id="cb")
    store.save(a)
    store.save(b)
    assert (tmp_path / "learning" / "ca.json").exists()
    assert {c.id for c in store.list()} == {"ca", "cb"}
    got = store.get("cb")
    assert got.kind == "hobby" and got.tutor.name == "한결 선생님" and got.curriculum[0].title == "보안이란 무엇인가"
    store.update("ca", lambda c: learning.mark_unit_done(c, "U1"))
    assert store.get("ca").curriculum[0].done and store.get("ca").progress == 25
    store.update("cb", lambda c: c.extras.setdefault("vocab", []).append({"term": "ciao", "meaning": "안녕"}))
    assert store.get("cb").extras["vocab"][0]["term"] == "ciao"
    with pytest.raises(KeyError):
        store.update("nope", lambda c: None)
    store.delete("ca")
    assert store.get("ca") is None and [c.id for c in store.list()] == ["cb"]


def test_store_rejects_path_like_ids_and_broken_files(tmp_path):
    store = learning.LearningStore(tmp_path)
    for bad in ["../etc", "a/b", "A B", "", "x" * 41, None]:
        assert store.get(bad) is None
    with pytest.raises(ValueError):
        store.path("../x")
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "broken.json").write_text("{not json", encoding="utf-8")
    assert store.get("broken") is None and store.list() == []


def test_store_concurrent_updates_do_not_lose_writes(tmp_path):
    store = learning.LearningStore(tmp_path)
    store.save(make_course(course_id="cc"))

    def worker(n):
        for i in range(5):
            store.update("cc", lambda c: learning.add_qa(c, f"질문 {n}-{i}", "답", keep=1000))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.get("cc").qa_log) == 40


# ─── 진도 · 수준 변경 ─────────────────────────────────────────────────


def test_progress_and_unit_done():
    c = make_course()
    assert learning.progress_pct(c) == 0
    learning.mark_unit_done(c, "U1", now="2026-11-05T11:00:00")
    learning.mark_unit_done(c, "U2")
    assert learning.progress_pct(c) == 50 and learning.next_unit(c).id == "U3"
    learning.mark_unit_done(c, "U2", done=False)
    assert c.curriculum[1].done_at == "" and learning.progress_pct(c) == 25
    with pytest.raises(KeyError):
        learning.mark_unit_done(c, "U9")
    c.curriculum = c.curriculum[:3]  # 3단원 중 1개 → 33% (정수 내림)
    assert learning.progress_pct(c) == 33
    c.curriculum = []
    assert learning.progress_pct(c) == 0 and learning.next_unit(c) is None


def test_change_level_rebuilds_only_remaining_units_and_keeps_history():
    c = make_course("beginner")
    learning.mark_unit_done(c, "U1")
    learning.mark_unit_done(c, "U2")
    new = [learning.UnitDraft(week=1, title="방화벽 정책 읽기", goals=["규칙 순서를 안다"]), {"week": 4, "title": "로그 보기", "goals": "로그를 연다"},
           {"title": ""}]
    change = learning.change_level(c, "elementary", "quiz", score_pct=80, new_units=new, now="2026-11-05T12:00:00")
    assert c.level == "elementary" and change.from_level == "beginner" and change.score_pct == 80
    assert change.replaced_units == ["네트워크 기초", "사고 대응 첫걸음"]  # 끝나지 않았던 단원은 기록으로
    assert [(u.id, u.week, u.title, u.level, u.done) for u in c.curriculum] == [
        ("U1", 1, "보안이란 무엇인가", "beginner", True), ("U2", 2, "계정과 비밀번호", "beginner", True),
        ("U5", 3, "방화벽 정책 읽기", "elementary", False), ("U6", 4, "로그 보기", "elementary", False)]  # id 는 새로, 주차는 이어서
    assert c.curriculum[3].goals == ["로그를 연다"]
    assert [h.how for h in c.level_history] == ["start", "quiz"]
    plain = learning.change_level(c, "intermediate", "manual")  # 모델을 못 쓰면 수준만
    assert plain.replaced_units == [] and len(c.curriculum) == 4 and c.level == "intermediate"


def test_plan_remaining_uses_tutor_and_new_level(use_script):
    script = use_script(Script(CurriculumPlan=learning.CurriculumPlan(
        analysis="초급으로", units=[learning.UnitDraft(week=3, title="권한 관리", goals=["최소 권한을 설명한다"])])))
    c = make_course("beginner")
    learning.mark_unit_done(c, "U1")
    units = learning.plan_remaining(c, "elementary", load_models())
    assert [u.title for u in units] == ["권한 관리"]
    prompt = script.prompts("CurriculumPlan")[0]
    assert "한결 선생님" in prompt and "입문(신규) → 초급" in prompt and "2주차부터" in prompt and "보안이란 무엇인가" in prompt


# ─── 퀴즈 채점 · 레벨업 ───────────────────────────────────────────────


def test_parse_and_grade_quiz_with_integer_threshold():
    quiz = learning.parse_quiz(quiz_answer(n=5))
    assert quiz.kind == "levelup" and len(quiz.questions) == 5 and quiz.questions[0].answer == "Q1 보기2"
    right = {q.id: q.answer for q in quiz.questions}
    grade = learning.grade_quiz(quiz, {**right, "Q5": "Q5 보기1"})
    assert (grade.correct, grade.total, grade.pct, grade.passed(), grade.wrong) == (4, 5, 80, True, ["Q5"])  # 4/5 = 정확히 80% → 통과
    grade = learning.grade_quiz(quiz, {"Q1": "Q1 보기2", "Q2": None})  # 안 푼 문제는 틀림
    assert (grade.correct, grade.pct, grade.passed()) == (1, 20, False)
    q4 = learning.parse_quiz(quiz_answer(n=4))
    assert not learning.grade_quiz(q4, {"Q1": "Q1 보기2", "Q2": "Q2 보기2", "Q3": "Q3 보기2"}).passed()  # 75%
    with pytest.raises(ValueError, match="보기 안에"):
        bad = quiz_json(n=2)
        bad["quiz"]["questions"][1]["answer"] = "없는 보기"
        learning.parse_quiz("```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```")


def test_can_level_up_rules():
    quiz = learning.parse_quiz(quiz_answer(n=5))
    full = learning.grade_quiz(quiz, {q.id: q.answer for q in quiz.questions})
    low = learning.grade_quiz(quiz, {})
    c = make_course("beginner")
    assert learning.can_level_up(c, full, "levelup")
    assert not learning.can_level_up(c, low, "levelup")
    assert not learning.can_level_up(c, full, "review")  # 복습 퀴즈는 레벨업 아님
    c.level = "expert"
    assert not learning.can_level_up(c, full, "levelup")  # 맨 위 수준


def test_placement_level_picks_next_after_last_passed_level():
    levels = ["beginner", "beginner", "elementary", "elementary", "intermediate"]
    quiz = learning.parse_quiz(quiz_answer(n=5, kind="placement", levels=levels))
    right = {q.id: q.answer for q in quiz.questions}
    assert learning.placement_level(quiz, {"Q1": right["Q1"], "Q2": right["Q2"]}) == "elementary"  # 입문 통과 → 초급에서
    assert learning.placement_level(quiz, {}) == "beginner"
    assert learning.placement_level(quiz, {k: v for k, v in right.items() if k != "Q3"}) == "elementary"  # 초급 50% → 멈춤
    assert learning.placement_level(quiz, right) == "advanced"  # 중급까지 통과 → 고급


# ─── 툴 · 문맥 ──────────────────────────────────────────────────────


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    return learning.LearningStore(tmp_path / "data")


def test_tool_gives_tutor_level_rule_curriculum_and_recent_records(data_env):
    c = make_course("elementary", course_id="ctool")
    learning.mark_unit_done(c, "U1")
    quiz = learning.parse_quiz(quiz_answer(n=5))
    c.quiz_results.append(learning.quiz_result(c, quiz, learning.grade_quiz(quiz, {}), now="2026-11-05T13:00:00"))
    learning.add_qa(c, "방화벽이 뭐예요?", "## 답\n방화벽은 들어오고 나가는 통신을 규칙으로 거르는 문지기입니다.")
    data_env.save(c)
    assert "learning_course" in TOOLS
    result = TOOLS["learning_course"]("질문", {"course_id": "ctool", "edu_kind": "career", "edu_task": "ask"}, load_models())
    assert result.harness_options == {"level": "elementary", "kind": "career", "course_id": "ctool", "language": False}
    text = result.content
    for want in ["한결 선생님", "가상 — 보안 관제 10년", "지금 수준: **초급**", "기본 용어는 그대로", "다음 수준: 중급",
                 "진도 25%", "| U1 | 1 | 보안이란 무엇인가", "다음 단원: U2 계정과 비밀번호", "퀴즈 2026-11-05 · 레벨업 퀴즈 · 0/5",
                 "방화벽이 뭐예요?", "문지기"]:
        assert want in text, want
    assert "한결 선생님" in result.title and "진도 25%" in result.title and not result.notes
    other = TOOLS["learning_course"]("q", {"course_id": "ctool", "edu_kind": "hobby"}, load_models())
    assert other.notes  # 다른 팀 과목이면 알린다


def test_tool_missing_course_create_and_nothing(data_env):
    missing = TOOLS["learning_course"]("q", {"course_id": "cnone", "edu_kind": "career"}, load_models())
    assert "찾지 못했다" in missing.content and missing.notes
    create = TOOLS["learning_course"]("q", {"edu_task": "create", "edu_kind": "hobby", "topic": "이탈리아어", "level": "초급",
                                            "language": True}, load_models())
    assert create.harness_options == {"level": "elementary", "kind": "hobby", "language": True, "topic": "이탈리아어"}
    assert "초급 · A2" in create.content and "| beginner | 입문(신규) | A1 |" in create.content
    assert TOOLS["learning_course"]("q", {}, load_models()) is None


def test_context_hook_for_extras():
    c = make_course(kind="hobby")
    c.extras["vocab"] = [{"term": "grazie", "meaning": "고마워요"}]
    learning.register_context("vocab", lambda course: "\n".join(f"- {v['term']} = {v['meaning']}" for v in course.extras["vocab"]))
    try:
        text = learning.course_context(c)
        assert "### 🗂 필수 단어·용어장" in text and "- grazie = 고마워요" in text
    finally:
        learning.CONTEXT_HOOKS.pop("vocab", None)


def test_ui_extension_points_protect_base_tasks():
    import education_ui

    with pytest.raises(ValueError):
        education_ui.register_preview("quiz", lambda *a: None)
    with pytest.raises(ValueError):
        education_ui.register_adopt("create", lambda *a: None)
    # 2·3단계 모듈(edu_career_ui …)이 이미 등록한 것은 그대로 두고 되돌린다 (비우면 그 팀 버튼·채택이 사라진다)
    quick_before, adopt_before = list(education_ui.QUICK_TASKS["career"]), dict(education_ui.ADOPT_HOOKS)
    try:
        education_ui.register_quick("career", "🧪 시험 버튼", "x_test_task")
        education_ui.register_quick("career", "🧪 시험 버튼", "x_test_task")  # 두 번 불러도 한 번만
        assert education_ui.QUICK_TASKS["career"].count(("🧪 시험 버튼", "x_test_task")) == 1
        fn = education_ui.register_adopt("x_test_task", lambda ctx, run, decision: "저장")
        assert education_ui.ADOPT_HOOKS["x_test_task"] is fn
    finally:
        education_ui.QUICK_TASKS["career"][:] = quick_before
        education_ui.ADOPT_HOOKS.clear()
        education_ui.ADOPT_HOOKS.update(adopt_before)
    assert set(learning.EXTENSION_MODULES) == {"core.learning_career", "core.learning_hobby"}
    assert set(education_ui.UI_EXTENSIONS) == {"edu_career_ui", "edu_hobby_ui"}


# ─── 검증기 ─────────────────────────────────────────────────────────


def test_quiz_format_validator():
    v = REGISTRY["quiz_format"]
    assert v.check(quiz_answer(n=5), {"quiz_count": "5", "quiz_kind": learning.QUIZ_KINDS["levelup"]}).passed
    assert not v.check(quiz_answer(n=4), {"quiz_count": "5"}).passed  # 문제 수
    assert not v.check("JSON 없음", {}).passed
    bad = quiz_json(n=2)
    bad["quiz"]["questions"][0]["choices"] = ["가", "가", "나"]
    res = v.check("```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```", {})
    assert not res.passed and "같은 보기" in res.log and "형식 예" in res.log
    placement = quiz_answer(n=3, kind="placement")  # 수준 확인인데 문제마다 level 없음
    assert not v.check(placement, {"quiz_kind": learning.QUIZ_KINDS["placement"]}).passed
    assert v.check(quiz_answer(n=3, kind="placement", levels=["beginner", "elementary", "expert"]),
                   {"quiz_kind": learning.QUIZ_KINDS["placement"]}).passed


def test_level_fit_validator_warns_then_fails_and_skips_upper_levels():
    v = REGISTRY["level_fit"]
    many = "TCP UDP DNS HTTP TLS SIEM EDR WAF 를 배운다."
    res = v.check(many, {"level": "beginner"})
    assert not res.passed and "8개" in res.log
    few = "TCP 와 UDP, DNS, HTTP 를 배운다."  # 4개 > 3 → 경고만
    res = v.check(few, {"level": "beginner"})
    assert res.passed and res.log.startswith("⚠ 경고")
    explained = "TCP(믿을 수 있는 전달 약속)와 UDP(빠른 전달), DNS(주소록), HTTP(웹 약속), 방화벽(firewall)."
    assert "통과" in v.check(explained, {"level": "beginner"}).log
    assert v.check(many, {"level": "intermediate"}).passed  # 중급 이상은 검사 안 함
    assert v.check(many, {}).passed
    code = "```python\nimport socket, requests, numpy, pandas, scipy, flask, django, fastapi\n```\n코드만 봐요."
    assert "0개" in v.check(code, {"level": "beginner"}).log  # 코드 블록은 세지 않음
    italian = "Buongiorno, grazie, prego, scusi, arrivederci 를 연습해요."
    assert "0개" in v.check(italian, {"level": "beginner", "language": True}).log  # 언어 과목은 배우는 말을 세지 않음
    assert not v.check(italian, {"level": "beginner"}).log.endswith("0개)")


# ─── 팀 설정 · 부서장 배정 ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def teams():
    return {t.id: t for t in list_teams()}


def test_education_department_and_team_config(teams):
    cfg = building.load_config()
    dept = next(d for d in building.departments(list(teams), cfg) if d.id == "education")
    assert dept.emoji == "🎓" and dept.teams == ["edu_career", "edu_hobby"]
    for tid in dept.teams:
        t = teams[tid]
        assert t.private and "learning_course" in t.tools and "edu_course" in t.features
        assert [x.id for x in t.tasks][:3] == ["create_tutor", "quiz", "ask"]  # 공용 업무 3개가 맨 앞 (2·3단계는 뒤에 추가)
        assert 3 <= len(t.org.walk()) <= 6 and t.org.name.startswith("선생님")
        assert "선생님 프로필" in t.personas["team_lead"]
        assert t.harness.validators == ["level_fit"] and {"quiz_format", "json_keys"} <= set(t.harness.final_only)
        assert all(any(f.key == "course_id" for f in x.fields) for x in t.tasks[1:3])
        assert {x.fixed.get("edu_kind") for x in t.tasks} == {learning.KIND_OF_TEAM[tid]}  # 모든 업무가 팀 목적을 fixed 로
        assert t.skills_from == ["_shared/education"] and len(t.free_examples) >= 2


def test_education_data_is_git_ignored():
    import subprocess

    out = subprocess.run(["git", "check-ignore", "-q", str(BASE_DIR / "data" / "learning" / "c1.json")], cwd=BASE_DIR)
    assert out.returncode == 0


ROUTES = [("보안 관리자로 이직하고 싶어", "edu_career"), ("이탈리아 여행 전에 회화 배우고 싶어", "edu_hobby"),
          ("로봇 SW 엔지니어 신입으로 취업 준비하려고 공부 순서가 궁금해", "edu_career"), ("유럽 역사를 교양으로 배우고 싶어", "edu_hobby")]


@pytest.mark.parametrize("text,want", ROUTES)
def test_head_assigns_career_vs_hobby_without_model(teams, set_keys, text, want):
    """키 없음 → 부서 안 낱말 비교 + 범위 확인(낱말). 다른 부서(커리어부·여행부)로 새지 않는다."""
    import building_ui

    set_keys()
    cfg = building.load_config()
    dept = next(d for d in building.departments(list(teams), cfg) if d.id == "education")
    pick = building_ui.assign(text, dept, teams, list(teams.values()), load_models(), None, cfg)
    assert pick["team_id"] == want and pick["outside"] is None, pick


def test_head_assigns_with_model_pick(teams, use_script):
    import building_ui

    script = use_script(Script(TeamPick=building.TeamPick(analysis="합격 목표", team_id="edu_career", reason="새 직무 합격이 목표")))
    cfg = building.load_config()
    dept = next(d for d in building.departments(list(teams), cfg) if d.id == "education")
    pick = building_ui.assign(ROUTES[0][0], dept, teams, list(teams.values()), load_models(), None, cfg)
    assert pick["team_id"] == "edu_career" and pick["how"] == "llm" and pick["outside"] is None
    prompt = script.prompts("TeamPick")[0]
    assert "교육부 부장" in prompt and "직무 전환 교육팀" in prompt and "새 직무 합격" in prompt


# ─── 화면 (AppTest): 과목 만들기 → 퀴즈 → 레벨업 → 커리큘럼 재생성 ─────────────


@pytest.fixture
def edu_app(tmp_path, monkeypatch, use_script):
    """가짜 모델(커리큘럼 다시 만들기) · 임시 데이터 폴더 · 실행은 곧바로 끝난 것처럼 (질문에 맞는 가짜 최종 답)."""
    import streamlit as st

    from core import live, workflow
    from core.schemas import AgentAnswer, ModelSpec

    use_script(Script(CurriculumPlan=learning.CurriculumPlan(analysis="초급으로", units=[
        learning.UnitDraft(week=2, title="접근 통제 연습", goals=["권한을 나눠 줄 수 있다"]),
        learning.UnitDraft(week=3, title="로그 읽기", goals=["로그인 실패 기록을 찾는다"])])))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    started = []

    def fake_start(run, events_factory, on_step=None, on_finish=None):
        started.append((run["team_id"], run["question"]))
        content = TUTOR_ANSWER if "선생님" in run["question"] else quiz_answer(n=5, level="elementary")
        step = AgentAnswer(node_id=run["root_id"], rank_id="team_lead", kind="integrate", content=content,
                           model=ModelSpec(provider="openai", model="x"))
        run.update(steps=[step], done=True, status="awaiting_human", activity={}, statuses={run["root_id"]: ("wait", "👤 결정 대기")})
        if on_finish:
            on_finish(run)

    monkeypatch.setattr(live, "start_run", fake_start)
    monkeypatch.setattr(workflow, "resume_with_decision", lambda *a, **k: iter([("lessons", [])]))
    st.cache_resource.clear()
    yield {"store": learning.LearningStore(tmp_path / "data"), "started": started}
    st.cache_resource.clear()


def _click(at, label=None, key=None):
    btn = next(b for b in at.button if (key and b.key == key) or (label and b.label.startswith(label)))
    at = btn.click().run()
    assert not at.exception, at.exception
    return at


def test_app_create_course_quiz_level_up_and_rebuild_curriculum(edu_app):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.query_params["view"] = "dept"
    at.query_params["dept"] = "education"
    at.session_state["office_view"] = "office"
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "🏢 🎓 교육부"
    assert any("📚 내 과목 (0)" in m.value for m in at.markdown)

    # ① 🎓 선생님 만들기 → 채택 → 과목 저장
    at = _click(at, key="edu_new_career")
    assert at.selectbox(key="dept_task_education").value == "edu_career:create_tutor"
    at.text_input(key="tf_edu_career_create_tutor_topic").input("보안 관리자")
    at.text_input(key="tf_edu_career_create_tutor_goal").input("6개월 안에 보안 관리자 신입 합격")
    at = _click(at, label="맡기기")
    assert edu_app["started"] and edu_app["started"][0][0] == "edu_career"
    assert any("새 선생님 미리 보기 — 한결 선생님" in m.value for m in at.markdown)
    at = _click(at, label="✅ 이 안으로 채택")
    courses = edu_app["store"].list()
    assert len(courses) == 1
    course = courses[0]
    assert course.tutor.name == "한결 선생님" and course.level == "beginner" and len(course.curriculum) == 4
    assert any("과목을 저장했습니다" in s.value for s in at.success)
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")
    assert any("📚 내 과목 (1)" in m.value for m in at.markdown)
    assert any("한결 선생님" in m.value for m in at.markdown)

    # 커리큘럼 단원 완료 체크 (📖 자세히)
    at = _click(at, key=f"edu_open_{course.id}")
    at = at.checkbox(key=f"edu_unit_{course.id}_U1").check().run()
    assert not at.exception, at.exception
    assert edu_app["store"].get(course.id).progress == 25

    # ② 📝 퀴즈 — 카드 버튼이 업무를 고르고 과목 칸을 채운다
    at = _click(at, key=f"edu_quiz_{course.id}")
    assert at.selectbox(key="dept_task_education").value == "edu_career:quiz"
    assert at.text_input(key="tf_edu_career_quiz_course_id").value == course.id
    at = _click(at, label="맡기기")
    assert "Q1 보기2" not in "".join(m.value for m in at.markdown if "정답·해설은 감췄습니다" in m.value)
    assert any("📝 퀴즈 풀기" in m.value for m in at.markdown)
    run_key = next(r.key for r in at.radio if r.key and r.key.startswith("edu_quiz_"))
    prefix = run_key.rsplit("_", 1)[0]
    for i in range(1, 6):
        at.radio(key=f"{prefix}_Q{i}").set_value(f"Q{i} 보기2")
    at = _click(at, label="✅ 채점하기")
    assert any("점수 5/5 (100%)" in m.value for m in at.markdown)
    saved = edu_app["store"].get(course.id)
    assert saved.quiz_results[-1].pct == 100 and saved.quiz_results[-1].passed

    # ③ 레벨업 → 남은 커리큘럼 다시 만들기 (가짜 모델)
    at = _click(at, label="⬆ 초급")
    after = edu_app["store"].get(course.id)
    assert after.level == "elementary"
    assert [u.title for u in after.curriculum] == ["보안이란 무엇인가", "접근 통제 연습", "로그 읽기"]  # 끝낸 단원은 그대로
    assert after.curriculum[0].done and after.curriculum[1].level == "elementary"
    change = after.level_history[-1]
    assert change.how == "quiz" and change.score_pct == 100 and change.replaced_units == ["계정과 비밀번호", "네트워크 기초", "사고 대응 첫걸음"]
    assert any("입문(신규) → **초급**" in s.value for s in at.success)
