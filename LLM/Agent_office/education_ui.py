"""🎓 교육부 화면 — app.py 가 부르는 교육부 전용 부분만 모았다 (두 팀 features: [edu_course]).

  courses_panel : 부서 업무 맡기기 위 '📚 내 과목' — 과목 카드(선생님 · 수준 · 진도 막대 · 다음 단원),
                  💬 질문 / 📝 퀴즈 를 누르면 그 팀 업무를 고르고 양식의 course_id 를 자동으로 채운다.
                  📖 자세히: 커리큘럼 단원 완료 체크 · 수준 직접 바꾸기(남은 커리큘럼 다시 만들기) · 수준 이력 · 퀴즈 · 질문 기록
  result_text   : 최종 결과 상자에 보일 글 — 퀴즈면 정답 JSON 을 감춘다
  preview       : 최종 결과 아래 — 선생님 만들기: 프로필·커리큘럼 미리 보기 / 퀴즈: 풀기 → 코드 채점 → 기록 → 80% 이상이면 레벨업 제안
  on_adopt      : 채택할 때 — 선생님 만들기 = 과목으로 저장, 질문 = 질문 기록(요약)

데이터·채점·수준 규칙은 core/learning.py (화면 없음). 모델 호출은 수준이 바뀔 때 남은 커리큘럼 다시 만들기 1회뿐.

════════════════════════════════════════════════════════════════════
2·3단계 확장 지점 — 이 파일은 고치지 않고 edu_career_ui.py / edu_hobby_ui.py 에서 등록한다 (있으면 맨 아래에서 자동으로 읽음)
════════════════════════════════════════════════════════════════════
  register_section(kind, fn)   : 과목 '📖 자세히' 안에 팀 전용 칸 — fn(ctx, course)  (예: 프로젝트·마일스톤, 단어장)
  register_quick(kind, label, task_id) : 과목 카드에 업무 버튼 추가 — 누르면 그 업무를 고르고 course_id 를 채운다
  register_preview(edu_task, fn): 그 업무 결과 아래 상자 — fn(ctx, run, final_text)
  register_adopt(edu_task, fn)  : 그 업무를 채택할 때 과목에 기록 — fn(ctx, run, decision) -> 화면에 보일 한 줄 | None
  register_answer_hidden(edu_task) : 그 업무 결과의 ```json(정답) 을 결과 상자에서 감춘다 (퀴즈형 업무 — 3단계 🔁 복습)
  edu_task = team.yaml 업무의 fixed.edu_task (공용: create · quiz · ask — 이 셋은 다시 등록하지 않는다)
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from core import learning
from core.config import rank_models
from core.schemas import HumanDecision, ModelsConfig, Team

FEATURE = "edu_course"
DEPT_ID = "education"
COURSE_FIELD = "course_id"  # 업무 양식에서 과목 id 를 받는 칸 이름 (2·3단계 업무도 이 이름을 쓴다)
PICK_KEY = "edu_course_pick"  # 지금 고른 과목 id
OPEN_KEY = "edu_course_open"  # '📖 자세히'를 펼친 과목 id
MSG_KEY = "edu_msg"  # 한 번 보일 알림 (수준 변경 등)
NOTICE = ("🔒 과목 기록(선생님·커리큘럼·퀴즈·질문)은 이 PC 의 `data/learning/` 에만 저장되고 GitHub 에 올라가지 않습니다 "
          "(교훈 공유에서도 교육부 두 팀은 빠집니다). 다만 업무 내용은 고른 AI 모델 회사로 전송됩니다.")


@dataclass
class Ctx:
    """app.py 가 넘기는 것 — 저장 폴더 · 지금 쓰는 모델 설정 · 팀 · 부서 업무 목록 키 · 지금 보는 팀 키."""

    data_dir: Path
    models: ModelsConfig
    teams_by_id: dict[str, Team]
    task_key: str  # 부서 업무 목록 selectbox 키 (dept_task_education)
    team_key: str  # 지금 보는 팀 (team_pick)

    @property
    def store(self) -> learning.LearningStore:
        return learning.LearningStore(self.data_dir)

    def tutor_chain(self, course: learning.Course):
        """남은 커리큘럼 다시 만들기 모델 = 그 팀 선생님(팀장) 직급 모델."""
        team = self.teams_by_id.get(course.team_id)
        return rank_models("team_lead", self.models, team) if team else None


# ─── 2·3단계 확장 지점 ─────────────────────────────────────────────────

SectionFn = Callable[[Ctx, learning.Course], None]
PreviewFn = Callable[[Ctx, dict, str], None]
AdoptFn = Callable[[Ctx, dict, HumanDecision], "str | None"]
COURSE_SECTIONS: dict[str, list[SectionFn]] = {"career": [], "hobby": []}
QUICK_TASKS: dict[str, list[tuple[str, str]]] = {"career": [], "hobby": []}
PREVIEW_HOOKS: dict[str, PreviewFn] = {}
ADOPT_HOOKS: dict[str, AdoptFn] = {}
BASE_TASKS = ("create", "quiz", "ask")
UI_EXTENSIONS = ("edu_career_ui", "edu_hobby_ui")


def register_section(kind: str, fn: SectionFn) -> SectionFn:
    COURSE_SECTIONS[kind].append(fn)
    return fn


def register_quick(kind: str, label: str, task_id: str) -> None:
    if (label, task_id) not in QUICK_TASKS[kind]:
        QUICK_TASKS[kind].append((label, task_id))


def register_preview(edu_task: str, fn: PreviewFn) -> PreviewFn:
    if edu_task in BASE_TASKS:
        raise ValueError(f"공용 업무 {edu_task} 는 다시 등록하지 않습니다")
    PREVIEW_HOOKS[edu_task] = fn
    return fn


def register_adopt(edu_task: str, fn: AdoptFn) -> AdoptFn:
    if edu_task in BASE_TASKS:
        raise ValueError(f"공용 업무 {edu_task} 는 다시 등록하지 않습니다")
    ADOPT_HOOKS[edu_task] = fn
    return fn


# ─── 과목 고르기 → 업무 양식 자동 채움 ────────────────────────────────────


def is_education(dept_id: str) -> bool:
    return dept_id == DEPT_ID


def field_key(team_id: str, task_id: str, field: str = COURSE_FIELD) -> str:
    """app.render_task_input 이 그리는 업무 양식 칸의 위젯 키."""
    return f"tf_{team_id}_{task_id}_{field}"


def pick_course(ctx: Ctx, course_id: str, team_id: str, task_id: str) -> None:
    """콜백 — 그 팀 업무를 고르고, 교육부 모든 업무 양식의 course_id 칸을 이 과목으로 채운다 (위젯이 그려지기 전)."""
    st.session_state[PICK_KEY] = course_id
    st.session_state[ctx.team_key] = team_id
    st.session_state[ctx.task_key] = f"{team_id}:{task_id}"
    for tid in learning.TEAM_OF_KIND.values():
        team = ctx.teams_by_id.get(tid)
        for task in team.tasks if team else []:
            if any(f.key == COURSE_FIELD for f in task.fields):
                st.session_state[field_key(tid, task.id)] = course_id


def open_task(ctx: Ctx, team_id: str, task_id: str) -> None:
    """콜백 — 과목 없이 그 팀 업무만 고른다 (선생님 만들기)."""
    st.session_state[ctx.team_key] = team_id
    st.session_state[ctx.task_key] = f"{team_id}:{task_id}"


def _toggle_open(course_id: str) -> None:
    st.session_state[OPEN_KEY] = None if st.session_state.get(OPEN_KEY) == course_id else course_id


def _table(rows: list[dict]) -> None:
    """번호 칸 없는 표."""
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _short(text: str, n: int = 40) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


# ─── 📚 내 과목 ─────────────────────────────────────────────────────


def courses_panel(ctx: Ctx) -> None:
    """부서 업무 맡기기 위 — 과목 카드 목록과 새 과목 만들기."""
    courses = ctx.store.list()
    st.markdown(f"##### 📚 내 과목 ({len(courses)})")
    msg = st.session_state.pop(MSG_KEY, None)
    if msg:
        st.success(msg)
    if not courses:
        st.info("아직 과목이 없습니다. 먼저 담당 선생님을 만드세요 — 목적에 맞는 팀을 고르면 '🎓 선생님 만들기' 양식이 열립니다.")
    for course in courses:  # 한 줄에 한 과목 — 왼쪽 정보, 오른쪽 버튼 (좁은 화면에서도 버튼이 겹치지 않게)
        course_card(ctx, course)
    n1, n2 = st.columns(2)
    for col, kind in ((n1, "career"), (n2, "hobby")):
        team_id = learning.TEAM_OF_KIND[kind]
        if team_id in ctx.teams_by_id:
            col.button(f"➕ {learning.KINDS[kind]} 선생님 만들기", key=f"edu_new_{kind}", use_container_width=True,
                       on_click=open_task, args=(ctx, team_id, "create_tutor"))
    st.caption(NOTICE)


def course_card(ctx: Ctx, course: learning.Course) -> None:
    """과목 카드 — 선생님 · 수준 · 진도 막대 · 다음 단원 + 💬 질문 / 📝 퀴즈 / (팀 전용 버튼) / 📖 자세히."""
    picked = st.session_state.get(PICK_KEY) == course.id
    is_open = st.session_state.get(OPEN_KEY) == course.id
    with st.container(border=True):
        info, acts = st.columns([3, 1])
        info.markdown(f"**{course.title}**" + (" · ✔ 선택됨" if picked else "") + f"  \n👩‍🏫 {course.tutor.title}"
                      + (f" · {_short(course.tutor.specialty, 30)}" if course.tutor.specialty else ""))
        info.caption(f"수준 **{course.level_name}** · 목표: {_short(course.goal or '-', 48)} · `{course.id}`")
        info.progress(course.progress, text=f"진도 {course.progress}% ({sum(u.done for u in course.curriculum)}/{len(course.curriculum)} 단원)")
        nu = learning.next_unit(course)
        info.caption(f"다음 단원: {nu.id} · {nu.week}주차 · {nu.title}" if nu else "🎉 모든 단원을 끝냈습니다")
        for label, task_id in [("💬 질문", "ask"), ("📝 퀴즈", "quiz"), *QUICK_TASKS.get(course.kind, [])]:
            acts.button(label, key=f"edu_{task_id}_{course.id}", use_container_width=True,
                        on_click=pick_course, args=(ctx, course.id, course.team_id, task_id))
        acts.button("접기" if is_open else "📖 자세히", key=f"edu_open_{course.id}", use_container_width=True,
                    on_click=_toggle_open, args=(course.id,))
        if is_open:  # 📖 자세히 — 카드 안 전체 너비로 (표가 좁아지지 않게)
            st.divider()
            course_detail(ctx, course)


def _set_unit(ctx: Ctx, course_id: str, unit_id: str, key: str) -> None:
    done = bool(st.session_state.get(key))
    ctx.store.update(course_id, lambda c: learning.mark_unit_done(c, unit_id, done))


def change_level_now(ctx: Ctx, course_id: str, to_level: str, how: str, score_pct: int | None = None) -> str:
    """수준 변경 + 남은 커리큘럼 다시 만들기 (모델 1회). 모델을 못 쓰면 수준만 바꾸고 알린다. 화면에 보일 한 줄을 돌려준다."""
    course = ctx.store.get(course_id)
    if course is None:
        return "과목을 찾지 못했습니다."
    units, problem = None, ""
    if any(not u.done for u in course.curriculum) or not course.curriculum:
        with st.spinner(f"{course.tutor.title}이 남은 커리큘럼을 {learning.level_label(to_level, course.language)} 수준에 맞게 다시 짜는 중…"):
            try:
                units = learning.plan_remaining(course, to_level, ctx.models, ctx.tutor_chain(course))
            except Exception as e:  # noqa: BLE001 — 키 없음·한도 초과: 수준 변경은 그대로 한다
                problem = f" (남은 커리큘럼은 다시 만들지 못했습니다: {type(e).__name__} — 나중에 수준을 다시 골라 시도하세요)"
    change = ctx.store.update(course_id, lambda c: learning.change_level(c, to_level, how, score_pct=score_pct, new_units=units))
    msg = (f"🎓 {course.topic}: {learning.level_label(change.from_level, course.language)} → "
           f"**{learning.level_label(change.to_level, course.language)}**")
    if change.replaced_units:
        msg += f" · 남은 단원 {len(change.replaced_units)}개를 새 수준에 맞게 다시 만들었습니다 (끝낸 단원은 그대로)"
    return msg + problem


def _manual_level(ctx: Ctx, course_id: str, key: str) -> None:
    to = st.session_state.get(key)
    course = ctx.store.get(course_id)
    if course is None or not to or to == course.level:
        st.session_state[MSG_KEY] = "지금과 같은 수준입니다."
        return
    st.session_state[MSG_KEY] = change_level_now(ctx, course_id, to, "manual")


def _delete(ctx: Ctx, course_id: str) -> None:
    ctx.store.delete(course_id)
    st.session_state[OPEN_KEY] = None
    if st.session_state.get(PICK_KEY) == course_id:
        st.session_state[PICK_KEY] = None
    st.session_state[MSG_KEY] = "🗑 과목을 지웠습니다."


HOW_TEXT = {"start": "처음", "placement": "수준 확인 퀴즈", "quiz": "레벨업 퀴즈", "manual": "직접 바꿈"}


def course_detail(ctx: Ctx, course: learning.Course) -> None:
    """📖 자세히 — 선생님 · 커리큘럼 완료 체크 · 수준 바꾸기 · 이력 · (팀 전용 칸)."""
    t = course.tutor
    st.markdown(f"**👩‍🏫 {t.name}** — {t.background or ''}  \n가르치는 방식: {t.teaching_style or '-'}")
    st.caption(f"지금 수준 규칙: {t.rule_for(course.level)}")
    st.markdown("**📚 커리큘럼** (끝낸 단원을 체크하세요)")
    for u in course.curriculum:
        key = f"edu_unit_{course.id}_{u.id}"
        st.checkbox(f"{u.id} · {u.week}주차 · {u.title}" + (f" — {' / '.join(u.goals)}" if u.goals else ""),
                    value=u.done, key=key, on_change=_set_unit, args=(ctx, course.id, u.id, key))
    lv_key = f"edu_level_{course.id}"
    c1, c2 = st.columns([2, 1])
    c1.selectbox("수준 직접 바꾸기", learning.LEVEL_IDS, index=learning.level_index(course.level), key=lv_key,
                 format_func=lambda lv: learning.level_label(lv, course.language))
    c2.button("수준 바꾸기", key=f"edu_level_go_{course.id}", use_container_width=True,
              help="수준을 바꾸면 선생님 말투·난이도와 남은(안 끝낸) 커리큘럼을 그 수준에 맞게 다시 만듭니다 (모델 1회). 끝낸 단원은 그대로 남습니다.",
              on_click=_manual_level, args=(ctx, course.id, lv_key))
    if course.level_history:
        st.markdown("**📈 수준 이력**")
        _table([{"때": c.at[:16].replace("T", " "), "변화": f"{learning.level_of(c.from_level).name} → {learning.level_of(c.to_level).name}",
                   "방법": HOW_TEXT.get(c.how, c.how), "점수": f"{c.score_pct}%" if c.score_pct is not None else "",
                   "다시 만든 단원": f"{len(c.replaced_units)}개" if c.replaced_units else ""} for c in course.level_history])
    if course.quiz_results:
        st.markdown("**📝 퀴즈 기록**")
        _table([{"때": q.at[:16].replace("T", " "), "종류": learning.QUIZ_KINDS[q.kind].split(" (")[0],
                   "수준": learning.level_of(q.level).name, "점수": f"{q.correct}/{q.total} ({q.pct}%)",
                   "통과": "✅" if q.passed else ""} for q in course.quiz_results[-10:]])
    if course.qa_log:
        st.markdown("**💬 질문 기록**")
        for qa in course.qa_log[-5:]:
            st.caption(f"{qa.at[:10]} · {qa.question} → {qa.summary}")
    for fn in COURSE_SECTIONS.get(course.kind, []):
        fn(ctx, course)
    with st.popover("🗑 과목 지우기"):
        st.caption("과목 기록(선생님·커리큘럼·퀴즈·질문)이 이 PC 에서 지워집니다. 되돌릴 수 없습니다.")
        st.button("지우기", key=f"edu_del_{course.id}", type="primary", on_click=_delete, args=(ctx, course.id))


# ─── 결과 상자 ──────────────────────────────────────────────────────


def edu_task(run: dict) -> str:
    return str((run.get("edu_inputs") or {}).get("edu_task") or "")


ANSWER_TASKS = {"quiz"}  # 결과 상자에서 정답 JSON 을 감출 업무 — 2·3단계 퀴즈형 업무는 register_answer_hidden 으로 더한다


def register_answer_hidden(edu_task_id: str) -> None:
    """그 업무 결과의 ```json 블록(정답·해설)을 결과 상자에서 감춘다 (예: 3단계 🔁 복습 퀴즈)."""
    ANSWER_TASKS.add(edu_task_id)


def result_text(run: dict, text: str) -> str:
    """최종 결과 상자에 보일 글 — 퀴즈는 정답·해설이 든 JSON 을 감춘다 (아래 '📝 퀴즈 풀기'에서 풀고 채점 뒤에 보인다)."""
    if edu_task(run) not in ANSWER_TASKS:
        return text
    return re.sub(r"```[ \t]*json[ \t]*\n.*?```", "*(정답·해설은 감췄습니다 — 아래 '📝 퀴즈 풀기'에서 풀면 채점 뒤에 보입니다)*",
                  text, flags=re.DOTALL | re.IGNORECASE)


def preview(ctx: Ctx, run: dict, final_text: str) -> None:
    """최종 결과 아래 — 업무 종류(edu_task)별 상자."""
    task = edu_task(run)
    if run.get("edu_saved"):
        st.success(run["edu_saved"])
    if run.get("edu_error"):
        st.warning(run["edu_error"])
    if task in PREVIEW_HOOKS:
        PREVIEW_HOOKS[task](ctx, run, final_text)
    elif task == "create" and not run.get("edu_saved"):
        tutor_preview(run, final_text)
    elif task == "quiz":
        quiz_box(ctx, run, final_text)
    elif task == "ask" and run.get("status") == "awaiting_human":
        st.caption("💬 채택하면 이 질문과 답의 요약이 과목의 질문 기록에 남습니다.")


def tutor_preview(run: dict, text: str) -> None:
    inputs = run.get("edu_inputs") or {}
    level = learning.parse_level(inputs.get("level"))
    try:
        tutor, units = learning.parse_tutor_answer(text, level)
    except ValueError as e:
        st.warning(f"🎓 선생님 JSON 형식이 맞지 않아 채택해도 과목으로 저장되지 않습니다 — {e}. "
                   "아래 '최종 답변' 칸의 JSON 을 고치거나 다시 맡기세요.")
        return
    with st.container(border=True):
        st.markdown(f"#### 🎓 새 선생님 미리 보기 — {tutor.name}")
        st.caption(f"{tutor.specialty} · {tutor.background}")
        st.markdown(f"가르치는 방식: {tutor.teaching_style or '-'}")
        _table([{"수준": learning.level_label(lv.id, bool(inputs.get("language"))), "말투·난이도": tutor.level_rules.get(lv.id, "(공통 기준)")}
                  for lv in learning.LEVELS])
        _table([{"단원": u.id, "주차": u.week, "제목": u.title, "마치면 할 수 있는 것": " / ".join(u.goals)} for u in units])
        st.caption("✅ 채택하면 '📚 내 과목'에 저장됩니다. 고치려면 아래 '최종 답변' 칸의 JSON 블록을 고친 뒤 채택하세요 (저장은 그 내용으로).")


def _grade_key(run: dict) -> str:
    return f"edu_quiz_{run['run_id']}"


def quiz_box(ctx: Ctx, run: dict, text: str) -> None:
    """📝 퀴즈 풀기 — 답 고르기 → 코드 채점 → 과목에 기록 → 레벨업(80% 이상) / 수준 정하기(수준 확인) 제안."""
    inputs = run.get("edu_inputs") or {}
    course_id = str(inputs.get(COURSE_FIELD) or "")
    course = ctx.store.get(course_id)
    try:
        quiz = learning.parse_quiz(text)
    except ValueError as e:
        st.warning(f"📝 퀴즈 형식이 맞지 않아 풀 수 없습니다 — {e}")
        return
    if inputs.get("quiz_kind"):
        quiz.kind = learning.parse_quiz_kind(inputs["quiz_kind"])
    st.markdown(f"#### 📝 퀴즈 풀기 — {learning.QUIZ_KINDS[quiz.kind].split(' (')[0]} · {len(quiz.questions)}문제 (채점은 코드가)")
    if course is None:
        st.warning(f"과목 `{course_id}` 를 찾지 못해 기록은 남기지 않습니다 (풀어 볼 수는 있습니다).")
    graded = run.get("edu_grade")
    if not graded:
        st.caption("정답은 채점 뒤에 보입니다 — 아래 '최종 결정'의 답변 칸에는 정답 JSON 이 있으니 먼저 풀어 보세요.")
        with st.form(_grade_key(run)):
            for i, q in enumerate(quiz.questions, 1):
                st.radio(f"**{i}.** {q.question}", q.choices, index=None, key=f"{_grade_key(run)}_{q.id}")
            go = st.form_submit_button("✅ 채점하기", type="primary")
        if go:
            answers = {q.id: st.session_state.get(f"{_grade_key(run)}_{q.id}") for q in quiz.questions}
            grade = learning.grade_quiz(quiz, answers)
            placed = learning.placement_level(quiz, answers) if quiz.kind == "placement" else None
            up = bool(course) and learning.can_level_up(course, grade, quiz.kind)
            if course is not None:
                ctx.store.update(course_id, lambda c: c.quiz_results.append(learning.quiz_result(c, quiz, grade)))
            run["edu_grade"] = {"correct": grade.correct, "total": grade.total, "pct": grade.pct, "passed": grade.passed(),
                                "details": grade.details, "placed": placed, "can_up": up,
                                "to": learning.next_level(course.level) if course else None, "kind": quiz.kind}
            st.rerun()
        return
    _graded_box(ctx, run, graded, course)


def _graded_box(ctx: Ctx, run: dict, g: dict, course: learning.Course | None) -> None:
    st.markdown(f"**점수 {g['correct']}/{g['total']} ({g['pct']}%)**" + (" — ✅ 통과" if g["passed"] else ""))
    for i, d in enumerate(g["details"], 1):
        mark = "✅" if d["ok"] else "❌"
        picked = d["picked"] if d["picked"] is not None else "(안 고름)"
        st.caption(f"{mark} {i}. 고른 답: {picked} · 정답: {d['answer']}" + (f" — {d['explanation']}" if d["explanation"] else ""))
    done = run.get("edu_level_done")
    if done:
        st.success(done)
        return
    language = bool(course and course.language)
    if g["can_up"] and g["to"] and course is not None:
        st.success(f"🎉 {learning.LEVEL_UP_PCT}% 이상 — **{learning.level_label(g['to'], language)}**(으)로 올릴 수 있어요. "
                   "올리면 선생님 말투·난이도와 남은 커리큘럼이 새 수준에 맞게 바뀝니다 (끝낸 단원은 그대로).")
        if st.button(f"⬆ {learning.level_label(g['to'], language)}(으)로 올리고 남은 커리큘럼 다시 만들기", type="primary",
                     key=f"edu_up_{run['run_id']}"):
            run["edu_level_done"] = change_level_now(ctx, course.id, g["to"], "quiz", g["pct"])
            st.rerun()
    elif g["kind"] == "levelup" and not g["passed"]:
        st.info(f"{learning.LEVEL_UP_PCT}% 에 못 미쳤습니다 — 틀린 문제를 선생님에게 질문해 보고 다시 도전하세요.")
    placed = g.get("placed")
    if placed and course is not None:
        if placed == course.level:
            st.info(f"🎯 지금 수준({learning.level_label(placed, language)})이 알맞습니다.")
        elif st.button(f"🎯 {learning.level_label(placed, language)} 수준으로 정하기", type="primary", key=f"edu_place_{run['run_id']}"):
            run["edu_level_done"] = change_level_now(ctx, course.id, placed, "placement", g["pct"])
            st.rerun()


# ─── 채택 ───────────────────────────────────────────────────────────


def on_adopt(ctx: Ctx, run: dict, decision: HumanDecision) -> None:
    """채택할 때 과목에 기록한다 — 실패해도 결정은 그대로 (이유는 결과 아래에 보인다)."""
    inputs = run.get("edu_inputs") or {}
    task = edu_task(run)
    try:
        if task in ADOPT_HOOKS:
            msg = ADOPT_HOOKS[task](ctx, run, decision)
        elif task == "create":
            course = ctx.store.save(learning.course_from_answer(decision.final_answer, inputs))
            st.session_state[PICK_KEY] = course.id
            msg = (f"📚 과목을 저장했습니다 — {course.title} · 👩‍🏫 {course.tutor.title} · {course.level_name} · "
                   f"단원 {len(course.curriculum)}개 (id `{course.id}`). 위 '📚 내 과목'에서 질문·퀴즈를 맡길 수 있습니다.")
        elif task == "ask":
            question = str(inputs.get("ask") or run.get("question") or "")
            ctx.store.update(str(inputs.get(COURSE_FIELD) or ""), lambda c: learning.add_qa(c, question, decision.final_answer))
            msg = "💬 질문 기록(요약)을 과목에 남겼습니다."
        else:
            msg = None
    except (ValueError, KeyError) as e:
        run["edu_error"] = f"과목에 기록하지 못했습니다 — {e}"
        return
    if msg:
        run["edu_saved"] = msg


# 2·3단계 화면 모듈 (있으면 읽는다 — 그 안에서 register_* 를 부른다)
for _name in UI_EXTENSIONS:
    try:
        importlib.import_module(_name)
    except ModuleNotFoundError as _e:
        if _e.name != _name:
            raise
