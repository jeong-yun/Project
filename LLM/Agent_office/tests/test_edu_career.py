"""🧭 직무 전환 교육팀 (교육부 2단계) — 로드맵 · 개인 프로젝트 · 진행 점검 · JD 추천: 검증기 · 채택 저장 · 마일스톤 체크 · 화면 흐름.

모델은 모두 가짜(use_script) 또는 실행을 곧바로 끝내는 가짜 start_run — .env 의 실제 키로 진짜 모델을 부르지 않는다.
샘플 과목·선생님·회사는 모두 가상이다 ('로봇 SW 엔지니어로 전환, 입문').
날짜는 'N주차'·고정 날짜 글만 쓴다 (오늘 날짜·부동소수점·경로 모양에 기대지 않는다).
"""

import json

import pytest

from core import learning
from core import learning_career as lc
from core.config import BASE_DIR, list_teams, load_models
from core.harness import REGISTRY
from core.tools import TOOLS
from tests.fakes import Script

TUTOR = {
    "tutor": {"name": "다온 선생님", "specialty": "로봇 소프트웨어 입문", "background": "가상 — 협동 로봇 제어 SW 8년, 부트캠프 강사 3년",
              "teaching_style": "작게 움직여 보기 → 원리 → 다시 만들어 보기",
              "level_rules": {"beginner": "용어마다 쉬운 풀이", "elementary": "기본 용어 그대로", "intermediate": "원리 비교",
                              "advanced": "설계 판단", "expert": "토론"}},
    "curriculum": [{"week": 1, "title": "로봇이 움직이는 원리", "goals": ["Python 프로그래밍으로 모터를 켜 본다", "센서의 역할을 말할 수 있다"]},
                   {"week": 2, "title": "파이썬 기초", "goals": ["반복문을 쓴다"]},
                   {"week": 3, "title": "센서 값 읽기", "goals": ["거리 센서 값을 출력한다"]}],
}
TUTOR_ANSWER = "## 선생님 소개\n다온 선생님입니다.\n\n```json\n" + json.dumps(TUTOR, ensure_ascii=False) + "\n```"
GOAL = "누리로보틱스 같은 로봇 회사에 8개월 안에 신입 합격"  # 가상 회사 — 검색어 검사가 막아야 하는 '내 글의 이름'


def make_course(course_id="crobot1", level="beginner") -> learning.Course:
    inputs = {"edu_kind": "career", "topic": "로봇 SW 엔지니어", "goal": GOAL, "level": learning.level_of(level).name, "weekly_hours": 6}
    return learning.course_from_answer(TUTOR_ANSWER, inputs, course_id=course_id, now="2026-11-05T10:00:00")


def answer(key: str, value) -> str:
    return f"## 본문\n설명\n\n```json\n{json.dumps({key: value}, ensure_ascii=False)}\n```"


ROADMAP = {"target_role": "로봇 SW 엔지니어 신입", "deadline": "12주", "weekly_hours": 6,
           "competencies": [{"name": "프로그래밍", "need": "ROS2 노드 작성", "now": "파이썬 첫걸음", "gap": "ROS2"}],
           "steps": [{"week": 2, "title": "리눅스 명령어 기초", "competency": "개발 환경", "hours": 2.5, "goals": ["터미널로 파일을 다룬다"], "link": ""},
                     {"week": 2, "title": "파이썬 함수", "competency": "프로그래밍", "hours": 3.5, "goals": ["함수를 만든다"], "link": ""},
                     {"week": 3, "title": "ROS2 첫 노드", "competency": "프로그래밍", "hours": 5, "goals": ["토픽을 주고받는다"],
                      "link": "개인 프로젝트 첫 마일스톤"}],
           "certs": [], "portfolio": ["ROS2 노드 저장소"]}

PROJECTS = [{"title": "거리 센서로 멈추는 시뮬레이션 로봇", "problem": "벽에 부딪히지 않게 멈추기", "deliverable": "공개 저장소 + 데모 영상",
             "tech": ["Python", "ROS2"], "difficulty": "elementary", "goal": "센서 → 판단 → 제어 흐름을 보여 준다",
             "portfolio_line": "시뮬레이터에서 거리 센서로 [결과 숫자] cm 앞에서 멈추는 로봇을 만들었다",
             "milestones": [{"id": "M1", "title": "시뮬레이터 설치·실행 화면", "due": "1주차"},
                            {"id": "M2", "title": "센서 값 출력 노드", "due": "2주차"},
                            {"id": "M3", "title": "멈춤 제어 + 데모 영상", "due": "4주차"}]}]

JD_TEXT = """[가상 로봇 회사 — 로봇 SW 엔지니어 (신입)]
[자격 요건]
- Python 프로그래밍 가능자
- ROS2 기반 개발 경험
[우대 사항]
- 센서 데이터 처리 경험"""

ADVICE = {"roles": [{"title": "로봇 SW 엔지니어 (신입)", "why": "센서 프로젝트를 끝내면 보여 줄 것이 생긴다", "keywords": ["ROS2", "Python"]}],
          "seniority": "신입", "keywords": ["ROS2", "Python", "센서"],
          "gaps": [{"need": "ROS2 기반 개발", "how": "ROS2 튜토리얼 노드 3개"}],
          "add_units": [{"title": "ROS2 토픽·서비스 연습", "goals": ["노드 두 개가 대화한다"]}],
          "queries": ["로봇 SW 엔지니어 신입 채용 요구 역량"]}


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    return learning.LearningStore(tmp_path / "data")


# ─── 설정 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("first", ["core.governance", "core.career", "core.harness", "core.learning_career"])
def test_no_circular_import_whatever_loads_first(first):
    """core.governance → core.harness → (확장 모듈) → core.career → core.governance 순환이 생기지 않는지 — 새 프로세스에서."""
    import subprocess
    import sys

    code = f"import {first}; from core.harness import REGISTRY; assert 'roadmap_plan' in REGISTRY"
    done = subprocess.run([sys.executable, "-c", code], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-800:]


def test_team_tasks_validators_and_skills():
    team = next(t for t in list_teams() if t.id == "edu_career")
    ids = [t.id for t in team.tasks]
    assert ids[:3] == ["create_tutor", "quiz", "ask"]  # 공용 업무가 맨 앞 그대로
    assert ids[3:] == ["roadmap", "project_plan", "project_check", "jd_match"]
    for task in team.tasks[3:]:
        assert task.fixed == {"edu_kind": "career", "edu_task": task.id}  # register_adopt 키 = 업무 id
        assert task.fields[0].key == "course_id" and task.fields[0].required
        assert "{course_id}" not in task.question  # 질문은 검색어 만들기에 쓰인다 — 과목 id 를 넣지 않는다
        for v in task.validators:
            assert v in REGISTRY and v in team.harness.final_only, v
    assert "{jd}" in next(t for t in team.tasks if t.id == "jd_match").instructions  # JD 원문은 질문이 아니라 지침으로
    assert team.private
    from core.agent import skill_sections
    from core.config import load_skills

    skills = load_skills("edu_career")
    for head in ("[로드맵", "[개인 프로젝트", "[진행 점검", "[JD 추천", "[질문 답변 — 직무 전환"):
        assert head in skills, head
    staff = skill_sections(skills, ["[공통", "[질문 답변"])  # 자료 조사 사원은 질문 답변 절만 — 면접·이력서 절도 받는다
    assert "면접 질문이면" in staff and "[로드맵" not in staff


def test_ui_registers_quick_buttons_sections_and_hooks():
    import edu_career_ui
    import education_ui

    assert [x for x in education_ui.QUICK_TASKS["career"] if x[1] in dict(map(reversed, edu_career_ui.QUICK))] == list(edu_career_ui.QUICK)
    assert {"roadmap", "project_plan", "project_check", "jd_match"} <= set(education_ui.ADOPT_HOOKS) & set(education_ui.PREVIEW_HOOKS)
    names = [f.__name__ for f in education_ui.COURSE_SECTIONS["career"]]
    assert names.count("projects_section") == 1 and {"roadmap_section", "jd_section"} <= set(names)
    assert not education_ui.QUICK_TASKS["hobby"] or all(t not in ("roadmap", "jd_match") for _, t in education_ui.QUICK_TASKS["hobby"])


# ─── 🗺 로드맵 ──────────────────────────────────────────────────────


def test_roadmap_validator_week_hours_sum_by_code(data_env):
    v = REGISTRY["roadmap_plan"]
    ok = v.check(answer("roadmap", ROADMAP), {"weekly_hours": "6"})
    assert ok.passed and "6시간 ≤ 주당 6시간" in ok.log  # 2.5 + 3.5 = 6 (0.1시간 단위 정수로 더함)
    over = json.loads(json.dumps(ROADMAP))
    over["steps"][1]["hours"] = 4
    bad = v.check(answer("roadmap", over), {"weekly_hours": "6"})
    assert not bad.passed and "2주차 6.5시간" in bad.log
    # 업무 양식에 시간이 없으면 과목에 적은 주당 시간(6)으로
    data_env.save(make_course())
    assert not v.check(answer("roadmap", over), {"weekly_hours": "", "course_id": "crobot1"}).passed
    assert v.check(answer("roadmap", over), {"weekly_hours": "8"}).passed
    broken = {**ROADMAP, "steps": [{"week": 3, "title": "a", "hours": 1, "goals": ["g"]}, {"week": 2, "title": "", "hours": "많이"}]}
    log = v.check(answer("roadmap", broken), {}).log
    assert "주차가 앞 단계" in log and "title" in log and "hours" in log and "goals" in log
    assert not v.check("JSON 없음", {}).passed


def test_roadmap_adopt_keeps_done_units_and_replaces_rest():
    c = make_course()
    learning.mark_unit_done(c, "U1", now="2026-11-06T09:00:00")
    road = lc.roadmap_from_answer(answer("roadmap", ROADMAP))
    removed = lc.apply_roadmap(c, road, now="2026-11-07T09:00:00")
    assert removed == ["파이썬 기초", "센서 값 읽기"]
    assert [u.title for u in c.curriculum] == ["로봇이 움직이는 원리", "리눅스 명령어 기초", "파이썬 함수", "ROS2 첫 노드"]
    assert c.curriculum[0].done and [u.id for u in c.curriculum] == ["U1", "U4", "U5", "U6"]
    assert c.curriculum[3].goals[-1] == "🔗 개인 프로젝트 첫 마일스톤"  # 포트폴리오 연결이 단원 목표에
    assert c.extras["roadmap"]["target_role"] == "로봇 SW 엔지니어 신입" and c.extras["roadmap"]["at"] == "2026-11-07T09:00:00"
    assert c.extras["roadmap"]["steps"][0]["hours"] == "2.5"
    text = learning.course_context(c)
    assert "### 🗂 합격 로드맵" in text and "역량 프로그래밍" in text and "ROS2 노드 저장소" in text


# ─── 🛠 개인 프로젝트 ───────────────────────────────────────────────────


def test_project_plan_and_milestone_order_validators():
    plan, order = REGISTRY["project_plan"], REGISTRY["milestone_order"]
    good = answer("projects", PROJECTS)
    assert plan.check(good, {"level": "beginner"}).passed and order.check(good, {}).passed
    hard = [{**PROJECTS[0], "difficulty": "advanced"}]
    assert "한 단계 위까지" in plan.check(answer("projects", hard), {"level": "beginner"}).log
    assert plan.check(answer("projects", hard), {}).passed  # 수준 정보가 없으면 난이도 범위는 보지 않는다
    many = answer("projects", PROJECTS * 4)
    assert "1~3개" in plan.check(many, {"level": "beginner"}).log
    missing = [{"title": "빈 계획", "milestones": [{"title": "", "due": "1주차"}]}]
    log = plan.check(answer("projects", missing), {"level": "beginner"}).log
    assert "problem" in log and "tech" in log and "difficulty" in log and "portfolio_line" in log
    backwards = [{**PROJECTS[0], "milestones": [{"title": "a", "due": "3주차"}, {"title": "b", "due": "2주차"}]}]
    assert "보다 빠르다" in order.check(answer("projects", backwards), {}).log
    mixed = [{**PROJECTS[0], "milestones": [{"title": "a", "due": "1주차"}, {"title": "b", "due": "2026-12-01"}]}]
    assert "섞여" in order.check(answer("projects", mixed), {}).log
    dates = [{**PROJECTS[0], "milestones": [{"title": "a", "due": "2026-12-01"}, {"title": "b", "due": "2026-12-15"}]}]
    assert order.check(answer("projects", dates), {}).passed
    assert "'N주차'" in order.check(answer("projects", [{**PROJECTS[0], "milestones": [{"title": "a", "due": "곧"}]}]), {}).log


def test_project_adopt_milestone_check_status_and_context(data_env):
    c = make_course()
    added = lc.add_projects(c, lc.projects_from_answer(answer("projects", PROJECTS), c.level), now="2026-11-07T09:00:00")
    again = lc.add_projects(c, lc.projects_from_answer(answer("projects", PROJECTS), c.level))
    assert [p["id"] for p in added + again] == ["P1", "P2"]  # 다시 채택하면 새 번호로 더한다
    p = c.extras["projects"][0]
    assert p["status"] == "계획" and [m["id"] for m in p["milestones"]] == ["M1", "M2", "M3"] and p["milestones"][0]["done"] is False
    lc.set_milestone(c, "P1", "M1", done=True, now="2026-11-08T09:00:00")
    lc.set_milestone(c, "P1", "M2", blocked=True, memo="  시뮬레이터에서   토픽이 안 보임 ")
    assert p["status"] == "진행 중" and p["milestones"][0]["done_at"] == "2026-11-08T09:00:00"
    assert p["milestones"][1]["memo"] == "시뮬레이터에서 토픽이 안 보임" and lc.project_pct(p) == 33
    lc.set_milestone(c, "P1", "M2", done=True)
    assert p["milestones"][1]["blocked"] is False  # 끝내면 막힘 표시는 풀린다
    lc.set_milestone(c, "P1", "M3", done=True)
    assert p["status"] == "완료" and lc.next_milestone(p) is None
    lc.set_project_status(c, "P2", "보류")
    lc.set_milestone(c, "P2", "M1", done=True)
    assert c.extras["projects"][1]["status"] == "보류"  # 사람이 고른 보류는 그대로
    with pytest.raises(KeyError):
        lc.set_milestone(c, "P9", "M1", done=True)
    with pytest.raises(ValueError):
        lc.set_project_status(c, "P1", "끝남")
    lc.remove_project(c, "P2")
    assert [x["id"] for x in c.extras["projects"]] == ["P1"]
    data_env.save(c)
    result = TOOLS["learning_course"]("q", {"course_id": "crobot1", "edu_kind": "career", "edu_task": "project_check"}, load_models())
    for want in ["### 🗂 개인 프로젝트·마일스톤", "P1 · 거리 센서로 멈추는 시뮬레이션 로봇 — 완료 · 진행 100%",
                 "| M2 센서 값 출력 노드 | 2주차 | ✅ 완료 | 시뮬레이터에서 토픽이 안 보임 |"]:
        assert want in result.content, want
    assert "완료" in lc.learning_record(c) and PROJECTS[0]["portfolio_line"] in lc.learning_record(c)  # 끝낸 프로젝트만 포트폴리오 문장


# ─── 📌 진행 점검 ──────────────────────────────────────────────────────


def _course_with_project(store=None) -> learning.Course:
    c = make_course()
    lc.add_projects(c, lc.projects_from_answer(answer("projects", PROJECTS), c.level))
    lc.set_milestone(c, "P1", "M1", done=True)
    lc.set_milestone(c, "P1", "M2", blocked=True, memo="토픽이 안 보임")
    if store is not None:
        store.save(c)
    return c


def test_progress_check_validators_apply_to_course(data_env):
    _course_with_project(data_env)
    check, order = REGISTRY["progress_check"], REGISTRY["milestone_order"]
    opts = {"course_id": "crobot1"}
    good = {"project": "P1", "summary": "M2 에서 막힘", "next": ["ros2 topic list 로 토픽 이름 확인"],
            "unblock": [{"milestone": "M2", "how": "노드 이름과 토픽 이름을 맞춘다"}],
            "schedule": [{"id": "M2", "due": "3주차"}, {"id": "M3", "due": "5주차"}], "add": [{"title": "README 정리", "due": "6주차"}]}
    assert check.check(answer("check", good), opts).passed and order.check(answer("check", good), opts).passed
    assert "P7" in check.check(answer("check", {**good, "project": "P7"}), opts).log
    assert "M9" in check.check(answer("check", {**good, "schedule": [{"id": "M9", "due": "3주차"}]}), opts).log
    assert "next" in check.check(answer("check", {**good, "next": []}), opts).log
    # 과목에 적용해 보면 M2 가 M3(4주차)보다 늦어진다 → 순서 오류
    late = {**good, "schedule": [{"id": "M2", "due": "5주차"}], "add": []}
    assert "보다 빠르다" in order.check(answer("check", late), opts).log
    early_add = {**good, "add": [{"title": "정리", "due": "2주차"}]}  # 새 마일스톤은 맨 뒤 — 앞보다 빠르면 오류
    assert not order.check(answer("check", early_add), opts).passed

    c = data_env.get("crobot1")
    result = lc.apply_check(c, good)
    p = c.extras["projects"][0]
    assert result == {"project": "P1", "changed": ["M2 2주차 → 3주차", "M3 4주차 → 5주차"], "added": ["M4"]}
    assert [(m["id"], m["due"], m["done"]) for m in p["milestones"]] == [("M1", "1주차", True), ("M2", "3주차", False),
                                                                        ("M3", "5주차", False), ("M4", "6주차", False)]
    assert p["last_check"]["next"] == ["ros2 topic list 로 토픽 이름 확인"]
    no_projects = make_course("cempty")
    data_env.save(no_projects)
    assert "프로젝트가 없다" in check.check(answer("check", good), {"course_id": "cempty"}).log


# ─── 🎯 JD 추천 ─────────────────────────────────────────────────────


def test_jd_entries_reuse_career_requirements_and_score():
    c = make_course()
    learning.mark_unit_done(c, "U1")  # 'Python 프로그래밍으로 모터를 켜 본다' · '센서의 역할'
    [entry] = lc.jd_entries([("붙여 넣은 JD", JD_TEXT)], c, now="2026-11-07T09:00:00")
    assert entry["kind"] == "jd" and entry["title"].startswith("가상 로봇 회사") and entry["at"] == "2026-11-07T09:00:00"
    status = {r["text"]: (r["section"], r["status"]) for r in entry["reqs"]}
    assert status == {"Python 프로그래밍 가능자": ("필수", "근거 있음"), "ROS2 기반 개발 경험": ("필수", "공백"),
                      "센서 데이터 처리 경험": ("우대", "부분")}
    # 커리어부 점수 규칙 그대로 — 필수 ×2: (1×2 + 0×2 + 0.5×1) ÷ 5 × 100 = 50
    assert entry["score"] == 50 and "50점" in entry["note"] and "ROS2 기반 개발 경험" in entry["gaps"]
    assert "×2" in entry["basis"]
    text = learning.course_context(c.model_copy(update={"extras": {"jd_links": [entry]}}))
    assert "### 🗂 추천받은 JD 링크" in text and "| ROS2 기반 개발 경험 | 필수 | 공백 |" in text
    # 계획만 한(진행 중) 프로젝트의 기술(ROS2)은 근거가 아니다 — 프로젝트를 완료해야 학습 기록에 들어간다
    lc.add_projects(c, lc.projects_from_answer(answer("projects", PROJECTS), c.level))
    lc.set_milestone(c, "P1", "M1", done=True)
    ros = lambda: next(r for r in lc.jd_entries([("JD", JD_TEXT)], c)[0]["reqs"] if r["text"].startswith("ROS2"))["status"]  # noqa: E731
    assert ros() == "공백" and "ROS2" not in lc.learning_record(c) and "끝낸 마일스톤: 시뮬레이터 설치·실행 화면" in lc.learning_record(c)
    for mid in ("M2", "M3"):
        lc.set_milestone(c, "P1", mid, done=True)
    assert ros() == "근거 있음"


def test_jd_advice_validator_blocks_private_words_in_queries(data_env):
    data_env.save(make_course())
    v = REGISTRY["jd_advice"]
    opts = {"course_id": "crobot1"}
    assert v.check(answer("jd_advice", ADVICE), opts).passed
    leak = {**ADVICE, "queries": ["누리로보틱스 로봇 SW 엔지니어 채용"]}  # 목표에 적은 (가상) 회사 이름
    assert "누리로보틱스" in v.check(answer("jd_advice", leak), opts).log
    number = {**ADVICE, "queries": ["로봇 SW 엔지니어 8개월 합격"]}
    assert "숫자" in v.check(answer("jd_advice", number), opts).log
    bad = {**ADVICE, "roles": [], "seniority": "인턴", "add_units": [{"goals": ["x"]}]}
    log = v.check(answer("jd_advice", bad), opts).log
    assert "roles" in log and "seniority" in log and "add_units" in log


def test_jd_adopt_saves_links_adds_units_and_career_handoff():
    c = make_course()
    learning.mark_unit_done(c, "U1")
    adv = lc.advice_from_answer(answer("jd_advice", ADVICE), c)
    result = lc.apply_advice(c, adv, [("붙여 넣은 JD", JD_TEXT)], now="2026-11-07T09:00:00")
    assert result["units"] == ["U4"] and c.curriculum[-1].title == "ROS2 토픽·서비스 연습" and c.curriculum[-1].week == 4
    kinds = [x["kind"] for x in c.extras["jd_links"]]
    assert kinds == ["jd", "advice"] and c.extras["jd_links"][1]["title"] == "로봇 SW 엔지니어 (신입)"
    assert all({"title", "url", "note", "at"} <= set(x) for x in c.extras["jd_links"])  # extras 키 규칙
    assert lc.draft_queries(c) == ["로봇 SW 엔지니어 신입 채용 요구 역량"]  # 지난 추천의 검색어 (검사 통과)
    hand = lc.career_handoff(c)
    assert hand["jd"].startswith("[가상 로봇 회사") and hand["direction"].startswith("로봇 SW 엔지니어로 직무 전환")
    assert "끝낸 단원: 로봇이 움직이는 원리" in hand["profile_text"] and "파이썬 기초" not in hand["profile_text"]  # 끝낸 것만
    lc.add_links(c, [{"kind": "search", "title": f"r{i}", "url": "", "note": "", "at": ""} for i in range(lc.MAX_JD_LINKS + 5)])
    assert len(c.extras["jd_links"]) == lc.MAX_JD_LINKS


def test_draft_queries_fallback_passes_private_word_check():
    c = make_course()
    queries = lc.draft_queries(c)
    assert queries and all("로봇 SW 엔지니어" in q for q in queries)
    assert all(not lc.query_problems(q, c) for q in queries)
    assert lc.query_problems("누리로보틱스 연봉", c)


# ─── 화면 (AppTest): 카드 버튼 → 업무 → 채택 저장 → 마일스톤 체크 → 점검 → JD → 커리어부로 ─────────


def _final(question: str) -> str:
    if "합격 로드맵" in question:
        return answer("roadmap", ROADMAP)
    if "개인 프로젝트" in question and "설계" in question:
        return answer("projects", PROJECTS)
    if "진행을 점검" in question:
        return answer("check", {"project": "P1", "summary": "M2 막힘", "next": ["토픽 이름 확인"],
                                "unblock": [{"milestone": "M2", "how": "이름 맞추기"}],
                                "schedule": [{"id": "M3", "due": "5주차"}], "add": [{"title": "README 정리", "due": "6주차"}]})
    if "JD 방향" in question:
        return answer("jd_advice", ADVICE)
    return "답"


@pytest.fixture
def career_app(tmp_path, monkeypatch, use_script):
    import streamlit as st

    from core import live, workflow
    from core.schemas import AgentAnswer, ModelSpec

    use_script(Script(text="가짜 모델"))
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    store = learning.LearningStore(tmp_path / "data")
    course = make_course()
    learning.mark_unit_done(course, "U1")
    store.save(course)
    started = []

    def fake_start(run, events_factory, on_step=None, on_finish=None):
        started.append((run["team_id"], run["question"], dict(run.get("edu_inputs") or {})))
        step = AgentAnswer(node_id=run["root_id"], rank_id="team_lead", kind="integrate", content=_final(run["question"]),
                           model=ModelSpec(provider="openai", model="x"))
        run.update(steps=[step], done=True, status="awaiting_human", activity={}, statuses={run["root_id"]: ("wait", "👤 결정 대기")})
        if on_finish:
            on_finish(run)

    monkeypatch.setattr(live, "start_run", fake_start)
    monkeypatch.setattr(workflow, "resume_with_decision", lambda *a, **k: iter([("lessons", [])]))
    st.cache_resource.clear()
    yield {"store": store, "started": started, "cid": course.id}
    st.cache_resource.clear()


def _click(at, label=None, key=None):
    btn = next(b for b in at.button if (key and b.key == key) or (label and b.label.startswith(label)))
    at = btn.click().run()
    assert not at.exception, at.exception
    return at


def _run_task(at, card_key, task_id, cid, fill=None):
    """카드 버튼 → 업무 고름·과목 칸 자동 → 맡기기 → 채택 → 닫기."""
    at = _click(at, key=card_key)
    assert at.selectbox(key="dept_task_education").value == f"edu_career:{task_id}"
    assert at.text_input(key=f"tf_edu_career_{task_id}_course_id").value == cid
    for kind, key, value in fill or []:
        getattr(at, kind)(key=key).input(value)
    at = _click(at, label="맡기기")
    at = _click(at, label="✅ 이 안으로 채택")
    saved = [s.value for s in at.success]
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")
    return at, saved


def test_app_roadmap_project_check_jd_flow(career_app):
    from streamlit.testing.v1 import AppTest

    store, cid = career_app["store"], career_app["cid"]
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=120)
    at.query_params["view"] = "dept"
    at.query_params["dept"] = "education"
    at.session_state["office_view"] = "office"
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button if b.key and b.key.endswith(cid)]
    assert {"🗺 로드맵", "🛠 프로젝트", "📌 점검", "🎯 JD"} <= set(labels)

    # ① 🗺 로드맵 → 채택 → 남은 커리큘럼 교체 + extras['roadmap']
    at, saved = _run_task(at, f"edu_roadmap_{cid}", "roadmap", cid)
    assert any("로드맵을 과목에 저장했습니다" in s for s in saved)
    c = store.get(cid)
    assert c.extras["roadmap"]["target_role"] == "로봇 SW 엔지니어 신입"
    assert [u.title for u in c.curriculum] == ["로봇이 움직이는 원리", "리눅스 명령어 기초", "파이썬 함수", "ROS2 첫 노드"]
    assert "{course_id}" not in career_app["started"][-1][1] and cid not in career_app["started"][-1][1]  # 질문에 과목 id 없음

    # ② 🛠 프로젝트 → 채택 → extras['projects']
    at, saved = _run_task(at, f"edu_project_plan_{cid}", "project_plan", cid)
    assert any("프로젝트 1개(P1)" in s for s in saved)
    assert store.get(cid).extras["projects"][0]["title"] == PROJECTS[0]["title"]

    # ③ 📖 자세히 — 로드맵 요약 · 마일스톤 체크 (완료 · 막힘 · 메모)
    at = _click(at, key=f"edu_open_{cid}")
    assert any("목표 직무 **로봇 SW 엔지니어 신입**" in c.value for c in at.caption)
    at = at.checkbox(key=f"edc_{cid}_P1_M1_done").check().run()
    at = at.checkbox(key=f"edc_{cid}_P1_M2_blocked").check().run()
    at = at.text_input(key=f"edc_{cid}_P1_M2_memo").input("토픽이 안 보임").run()
    assert not at.exception, at.exception
    p = store.get(cid).extras["projects"][0]
    assert p["milestones"][0]["done"] and p["milestones"][1]["blocked"] and p["milestones"][1]["memo"] == "토픽이 안 보임"
    assert p["status"] == "진행 중" and at.selectbox(key=f"edc_{cid}_P1_status").value == "진행 중"

    # ④ 📌 점검 (자세히 안의 버튼 — 프로젝트 칸도 채운다) → 채택 → 기한·새 마일스톤
    at = _click(at, key=f"edc_{cid}_P1_check")
    assert at.text_input(key="tf_edu_career_project_check_project_id").value == "P1"
    at = _click(at, label="맡기기")
    assert any("채택하면 P1 에 반영될 것" in m.value for m in at.markdown)
    at = _click(at, label="✅ 이 안으로 채택")
    assert any("P1 일정을 고쳤습니다" in s.value for s in at.success)
    at = _click(at, label="✖ 이 결과 닫고 새 업무 시작")
    ms = store.get(cid).extras["projects"][0]["milestones"]
    assert [(m["id"], m["due"]) for m in ms] == [("M1", "1주차"), ("M2", "2주차"), ("M3", "5주차"), ("M4", "6주차")] and ms[0]["done"]

    # ⑤ 🎯 JD — 붙여 넣은 JD 는 커리어부 코드로 요구사항·1차 대응, 부족한 요건 단원 추가
    at, saved = _run_task(at, f"edu_jd_match_{cid}", "jd_match", cid,
                          fill=[("text_area", "tf_edu_career_jd_match_jd", JD_TEXT)])
    assert any("JD 1개 · 커리큘럼에 단원 1개 추가" in s for s in saved)
    assert JD_TEXT.splitlines()[2] not in career_app["started"][-1][1]  # JD 원문은 질문(검색어 재료)에 넣지 않는다
    c = store.get(cid)
    assert [x["kind"] for x in c.extras["jd_links"]] == ["jd", "advice"] and c.curriculum[-1].title == "ROS2 토픽·서비스 연습"

    # ⑥ 커리어부 '🎯 업무(JD) 추천'으로 가져가기 — 양식이 채워진다 (커리어부 파일은 그대로)
    if not any(b.key == f"edc_{cid}_to_career" for b in at.button):
        at = _click(at, key=f"edu_open_{cid}")
    at = _click(at, key=f"edc_{cid}_to_career")
    assert at.session_state["dept_pick"] == "career"
    assert at.selectbox(key="dept_task_career").value == "career:jd_recommend"
    assert at.text_area(key="tf_career_jd_recommend_jd").value.startswith("[가상 로봇 회사")
    assert "끝낸 단원: 로봇이 움직이는 원리" in at.text_area(key="tf_career_jd_recommend_profile_text").value
