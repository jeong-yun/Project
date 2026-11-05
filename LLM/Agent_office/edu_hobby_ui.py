"""🌍 취미·교양 교육팀 화면 — education_ui.py 의 확장 지점에 등록만 한다 (공용 화면 구조는 고치지 않는다).

  과목 카드 버튼 (register_quick) : 📅 커리큘럼 · 📖 단어장 · 🗣 표현 · 🧭 더 배울 것 · 🔁 복습
  결과 아래 미리 보기 (register_preview) : 커리큘럼 표·주당 시간(코드 계산) / 단어장 표 / 표현·대화 / 추가 주제 / 복습 퀴즈 풀기·채점·다음 복습일
  채택 (register_adopt) : 커리큘럼 → 과목의 남은 단원 교체 · 단어장·표현·주제 → extras 에 쌓기(같은 것은 합침)
  📖 자세히 (register_section) : 단어장 표('외움' 체크·검색·CSV(Anki)·.md) · 표현 모음 · 추가 주제(➕ 새 과목으로) · ⚡ 바로 복습(모델 없이)

계산·검증·간격 반복은 core/learning_hobby.py (화면 없음).
"""

from __future__ import annotations

import zlib

import pandas as pd
import streamlit as st

import education_ui
from core import learning
from core import learning_hobby as hobby
from education_ui import Ctx

TEAM = "edu_hobby"
COURSE_FIELD = education_ui.COURSE_FIELD


def _cid(run: dict) -> str:
    return str((run.get("edu_inputs") or {}).get(COURSE_FIELD) or "")


def _course(ctx: Ctx, run: dict) -> learning.Course | None:
    return ctx.store.get(_cid(run))


def _waiting(run: dict) -> bool:
    """아직 채택 전(결정 대기)이고 저장하지 않았을 때만 미리 보기를 보인다."""
    return not run.get("edu_saved") and run.get("status") in ("awaiting_human", None)


def _table(rows: list[dict], **kw) -> None:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, **kw)


def _problems_box(problems: list[str], warnings: list[str], what: str) -> None:
    if problems:
        st.warning(f"{what} 형식이 맞지 않아 채택해도 저장되지 않을 수 있습니다 — " + " / ".join(problems[:5])
                   + "  \n아래 '최종 답변' 칸의 JSON 을 고치거나 다시 맡기세요.")
    for w in warnings[:5]:
        st.caption(w if w.startswith("⚠") else f"⚠ {w}")


def _level(lv: object) -> str:
    return learning.level_of(str(lv or "")).name if lv else ""


# ─── 📅 커리큘럼 ──────────────────────────────────────────────────────


def curriculum_preview(ctx: Ctx, run: dict, text: str) -> None:
    if not _waiting(run):
        return
    course = _course(ctx, run)
    inputs = run.get("edu_inputs") or {}
    data = learning.find_json(text, "curriculum")
    hours = inputs.get("weekly_hours") or (course.weekly_hours if course else None)
    language = bool(course and course.language)
    problems, warnings = hobby.curriculum_problems(data, hours, inputs.get("weeks"), language)
    with st.container(border=True):
        st.markdown("#### 📅 새 커리큘럼 미리 보기")
        _problems_box(problems, warnings, "커리큘럼")
        units = [u for u in (data or {}).get("curriculum") or [] if isinstance(u, dict)]
        if units:
            _table([{"주차": u.get("week"), "단원": u.get("title"),
                     "영역·주제": " · ".join(u.get("skills") or []) if language else (u.get("theme") or ""),
                     "시간": u.get("hours"), "마치면 할 수 있는 것": " / ".join(map(str, u.get("goals") or []))} for u in units])
            limit = hobby._tenths(hours)
            per_week = hobby.week_hours(data)
            st.caption("🧮 주마다 시간 합 (코드 계산): " + " · ".join(
                f"{w}주 {h}h" + (" ⛔" if limit and hobby._tenths(h) > limit else "") for w, h in per_week.items())
                + (f" — 주당 {hobby._hours_text(limit)}시간" if limit else ""))
        if course:
            done = sum(u.done for u in course.curriculum)
            left = len(course.curriculum) - done
            st.caption(f"✅ 채택하면 남은 단원 {left}개가 이 {len(units)}단원으로 바뀝니다 (끝낸 단원 {done}개는 그대로, 끝낸 주 다음부터 이어 붙임).")


def on_curriculum_adopt(ctx: Ctx, run: dict, decision) -> str | None:
    data = learning.find_json(decision.final_answer, "curriculum")
    if not data:
        raise ValueError("커리큘럼 JSON 이 없습니다")
    course = ctx.store.get(_cid(run))
    removed = ctx.store.update(_cid(run), lambda c: hobby.save_curriculum(c, data))
    n = len(hobby.curriculum_units(data))
    return (f"📅 커리큘럼을 과목에 저장했습니다 — 새 단원 {n}개" + (f" (바꾼 남은 단원 {len(removed)}개)" if removed else "")
            + (f" · {course.topic}" if course else ""))


# ─── 📖 단어·용어장 ───────────────────────────────────────────────────


def _vocab_rows(items: list[dict], language: bool, with_srs: bool = True) -> list[dict]:
    rows = []
    for i in items:
        row = {"외움": bool(i.get("known"))} if with_srs else {}
        if language:
            row |= {"단어": i.get("term", ""), "읽는 법": i.get("reading", ""), "품사": i.get("pos", ""), "뜻": i.get("meaning", ""),
                    "예문": i.get("example", ""), "번역": i.get("example_ko", "")}
        else:
            row |= {"용어": i.get("term", ""), "풀이": i.get("meaning", ""), "연도·인물": i.get("related", "")}
        row["수준"] = _level(i.get("level"))
        if with_srs:
            row["다음 복습"] = i.get("due") or "-"
        rows.append(row)
    return rows


def vocab_preview(ctx: Ctx, run: dict, text: str) -> None:
    if not _waiting(run):
        return
    course = _course(ctx, run)
    language = bool(course and course.language)
    data = learning.find_json(text, "vocab")
    problems, warnings = hobby.vocab_problems(data, language, course.level if course else "beginner",
                                              (run.get("edu_inputs") or {}).get("count"))
    with st.container(border=True):
        st.markdown("#### 📖 새 단어장 미리 보기")
        _problems_box(problems, warnings, "단어장")
        items = hobby.vocab_items(data, course.level if course else "beginner")
        if items:
            _table(_vocab_rows(items, language, with_srs=False))
        if course:
            have = {hobby.norm(v.get("term")) for v in course.extras.get("vocab") or []}
            same = [i["term"] for i in items if hobby.norm(i["term"]) in have]
            st.caption(f"✅ 채택하면 과목 단어장({len(have)}개)에 쌓입니다" + (f" — 이미 있는 {len(same)}개({', '.join(same[:5])})는 합칩니다" if same else "")
                       + ". '📖 자세히'에서 외움 체크·검색·내려받기를 할 수 있습니다.")


def on_vocab_adopt(ctx: Ctx, run: dict, decision) -> str | None:
    data = learning.find_json(decision.final_answer, "vocab")
    if not data:
        raise ValueError("단어장 JSON 이 없습니다")
    added, merged = ctx.store.update(_cid(run), lambda c: hobby.add_vocab(c, data))
    return f"📖 단어장에 {added}개를 더했습니다" + (f" (같은 단어 {merged}개는 합침)" if merged else "") + "."


def _set_known(ctx: Ctx, course_id: str, key: str, fronts: list[str], editor_key: str) -> None:
    """콜백 — 표의 '외움' 체크를 과목에 저장 (표에 보인 줄 순서 = fronts)."""
    edited = (st.session_state.get(editor_key) or {}).get("edited_rows") or {}
    refs = {fronts[int(r)]: bool(v["외움"]) for r, v in edited.items() if "외움" in v and int(r) < len(fronts)}
    if refs:
        ctx.store.update(course_id, lambda c: hobby.set_known(c, key, refs))


def _editor(ctx: Ctx, course: learning.Course, key: str, rows: list[dict], fronts: list[str], salt: str) -> None:
    """'외움'만 고칠 수 있는 표. 키에 과목 저장 시각·검색어를 넣어 저장할 때마다 새 표로 (줄 순서가 어긋나지 않게)."""
    editor_key = f"hb_{key}_ed_{course.id}_{zlib.crc32((salt + course.updated_at).encode()):x}"
    df = pd.DataFrame(rows)
    st.data_editor(df, key=editor_key, hide_index=True, use_container_width=True,
                   disabled=[c for c in df.columns if c != "외움"],
                   column_config={"외움": st.column_config.CheckboxColumn("외움", help="외운 것은 체크 — 복습에서 빠집니다 (30일 지나면 한 번 다시)")},
                   on_change=_set_known, args=(ctx, course.id, key, fronts, editor_key))


def vocab_section(ctx: Ctx, course: learning.Course) -> None:
    items = course.extras.get("vocab") or []
    if not items:
        return
    known = sum(bool(i.get("known")) for i in items)
    due = sum(hobby.is_due(i) for i in items)
    st.markdown(f"**📖 단어장** — {len(items)}개 · 외움 {known}개 · 오늘 복습 {due}개")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input("🔎 단어 찾기", key=f"hb_vq_{course.id}", placeholder="단어·뜻·읽는 법·예문")
    hide = c2.checkbox("외운 것 숨기기", key=f"hb_vh_{course.id}")
    shown = [i for i in hobby.search(items, query, ("term", "meaning", "reading", "example", "related")) if not (hide and i.get("known"))]
    if shown:
        _editor(ctx, course, "vocab", _vocab_rows(shown, course.language), [i["term"] for i in shown], f"{query}|{hide}")
    else:
        st.caption("찾는 단어가 없습니다.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ CSV (Anki 가져오기용)", data=hobby.vocab_csv(items).encode("utf-8"), file_name=f"vocab_{course.id}.csv",
                       mime="text/csv", key=f"hb_vcsv_{course.id}", use_container_width=True,
                       help="Anki: 파일 > 가져오기 — 1열 단어(앞면) · 2열 뜻(뒷면). 엑셀은 '데이터 > 텍스트/CSV' 에서 UTF-8 로 여세요.")
    d2.download_button("⬇ 단어장 (.md)", data=hobby.vocab_markdown(course, items), file_name=f"vocab_{course.id}.md",
                       mime="text/markdown", key=f"hb_vmd_{course.id}", use_container_width=True)


# ─── 🗣 상황별 표현 ───────────────────────────────────────────────────


def _phrase_rows(items: list[dict], with_srs: bool = True) -> list[dict]:
    rows = []
    for i in items:
        row = {"외움": bool(i.get("known"))} if with_srs else {}
        row |= {"상황": i.get("situation", ""), "표현": i.get("phrase", ""), "읽는 법": i.get("reading", ""), "뜻": i.get("meaning", "")}
        if with_srs:
            row["다음 복습"] = i.get("due") or "-"
        rows.append(row)
    return rows


def _dialogues(items: list[dict]) -> None:
    for i in items:
        parts = [f"**{i.get('phrase')}** — {i.get('meaning')}"]
        if i.get("dialogue"):
            parts.append("  \n".join(f"> {d}" if n == 0 else d for n, d in enumerate(i["dialogue"])))  # 인용 한 덩어리
        if i.get("tip"):
            parts.append(f"💡 {i['tip']}")
        st.markdown("\n\n".join(parts))  # 빈 줄로 나눠야 팁이 인용 안으로 들어가지 않는다


def phrases_preview(ctx: Ctx, run: dict, text: str) -> None:
    if not _waiting(run):
        return
    course = _course(ctx, run)
    data = learning.find_json(text, "phrases")
    problems, warnings = hobby.phrase_problems(data, bool(course.language) if course else True,
                                               str((run.get("edu_inputs") or {}).get("situation") or ""))
    with st.container(border=True):
        st.markdown("#### 🗣 상황별 표현 미리 보기")
        _problems_box(problems, warnings, "표현")
        items = hobby.phrase_items(data, course.level if course else "beginner")
        if items:
            _table(_phrase_rows(items, with_srs=False))
            with st.expander("💬 대화 예시 · 문화 팁"):
                _dialogues(items)
        st.caption("✅ 채택하면 과목의 표현 모음에 저장되고 🔁 복습에 쓰입니다. "
                   "여행 일정·숙소·교통편은 ✈️ 여행부에 맡기세요.")


def on_phrases_adopt(ctx: Ctx, run: dict, decision) -> str | None:
    data = learning.find_json(decision.final_answer, "phrases")
    if not data:
        raise ValueError("표현 JSON 이 없습니다")
    added, merged = ctx.store.update(_cid(run), lambda c: hobby.add_phrases(c, data))
    return f"🗣 표현 모음에 {added}개를 더했습니다" + (f" (같은 표현 {merged}개는 합침)" if merged else "") + "."


def phrases_section(ctx: Ctx, course: learning.Course) -> None:
    items = course.extras.get("phrases") or []
    if not items:
        return
    situations = list(dict.fromkeys(i.get("situation") or "기타" for i in items))
    st.markdown(f"**🗣 표현 모음** — {len(items)}개 · 상황 {len(situations)}개")
    pick = st.selectbox("상황", ["전체", *situations], key=f"hb_ps_{course.id}", label_visibility="collapsed")
    shown = [i for i in items if pick == "전체" or (i.get("situation") or "기타") == pick]
    _editor(ctx, course, "phrases", _phrase_rows(shown), [i["phrase"] for i in shown], pick)
    with st.expander("💬 대화 예시 · 문화 팁"):
        _dialogues(shown)
    st.download_button("⬇ 표현 모음 (.md)", data=hobby.phrases_markdown(course, items), file_name=f"phrases_{course.id}.md",
                       mime="text/markdown", key=f"hb_pmd_{course.id}")


# ─── 🧭 추가 지식 ─────────────────────────────────────────────────────


def new_course_from_topic(ctx: Ctx, course_id: str, title: str) -> None:
    """콜백 — 공용 '🎓 선생님 만들기' 양식을 이 주제로 채워 연다 (맡기기는 사람이 누른다)."""
    course = ctx.store.get(course_id)
    topic = next((t for t in (course.extras.get("topics") or []) if t.get("title") == title), None) if course else None
    if course is None or topic is None:
        return
    education_ui.open_task(ctx, TEAM, "create_tutor")
    for key, value in hobby.new_course_prefill(course, topic).items():
        st.session_state[education_ui.field_key(TEAM, "create_tutor", key)] = value
    st.session_state[education_ui.MSG_KEY] = (f"🧭 '{title}' 새 과목 — 아래 '🎓 선생님 만들기' 양식을 채워 두었습니다. "
                                              "수준·목표를 확인하고 맡기세요.")


def _topic_done(ctx: Ctx, course_id: str, title: str, key: str) -> None:
    done = bool(st.session_state.get(key))

    def mark(c: learning.Course) -> None:
        for t in c.extras.get("topics") or []:
            if t.get("title") == title:
                t["done"] = done

    ctx.store.update(course_id, mark)


def _topic_cards(ctx: Ctx, course: learning.Course | None, topics: list[dict], key_prefix: str, editable: bool) -> None:
    for i, t in enumerate(topics):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            head = f"**{t.get('title')}**" + (f" · {t['field']}" if t.get("field") else "")
            if editable and course is not None:
                key = f"{key_prefix}_done_{i}"
                left.checkbox(head + " — 마침", value=bool(t.get("done")), key=key,
                              on_change=_topic_done, args=(ctx, course.id, t.get("title"), key))
            else:
                left.markdown(head)
            left.caption(f"왜: {t.get('why') or '-'}  \n첫 단원: {t.get('first_unit') or '-'}")
            if course is not None:
                right.button("➕ 새 과목으로", key=f"{key_prefix}_new_{i}", use_container_width=True,
                             help="이 주제로 '🎓 선생님 만들기' 양식을 채워 엽니다",
                             on_click=new_course_from_topic, args=(ctx, course.id, t.get("title")))


def topics_preview(ctx: Ctx, run: dict, text: str) -> None:
    course = _course(ctx, run)
    if run.get("edu_saved") and course is not None:  # 채택 뒤 — 바로 새 과목으로 만들 수 있게
        titles = {hobby.norm(t.get("title")) for t in (learning.find_json(text, "topics") or {}).get("topics") or [] if isinstance(t, dict)}
        saved = [t for t in course.extras.get("topics") or [] if hobby.norm(t.get("title")) in titles]
        if saved:
            st.caption("🧭 마음에 드는 주제를 새 과목으로 만들 수 있습니다.")
            _topic_cards(ctx, course, saved, f"hb_tp_{run['run_id']}", editable=False)
        return
    if not _waiting(run):
        return
    data = learning.find_json(text, "topics")
    problems, warnings = hobby.topic_problems(data)
    with st.container(border=True):
        st.markdown("#### 🧭 더 배울 것 미리 보기")
        _problems_box(problems, warnings, "추가 주제")
        topics = [t for t in (data or {}).get("topics") or [] if isinstance(t, dict)]
        _topic_cards(ctx, None, topics, f"hb_tpv_{run['run_id']}", editable=False)
        st.caption("✅ 채택하면 과목에 저장되고, '➕ 새 과목으로'로 그 주제 선생님을 만들 수 있습니다.")


def on_topics_adopt(ctx: Ctx, run: dict, decision) -> str | None:
    data = learning.find_json(decision.final_answer, "topics")
    if not data:
        raise ValueError("추가 주제 JSON 이 없습니다")
    added, merged = ctx.store.update(_cid(run), lambda c: hobby.add_topics(c, data))
    return f"🧭 더 배울 주제 {added}개를 과목에 저장했습니다" + (f" (같은 주제 {merged}개는 합침)" if merged else "") + "."


def topics_section(ctx: Ctx, course: learning.Course) -> None:
    topics = course.extras.get("topics") or []
    if not topics:
        return
    st.markdown(f"**🧭 더 배울 것** — {len(topics)}개 · 마침 {sum(bool(t.get('done')) for t in topics)}개")
    _topic_cards(ctx, course, topics, f"hb_tp_{course.id}", editable=True)


# ─── 🔁 복습 (모델 퀴즈 · ⚡ 바로 복습) ───────────────────────────────────


def _ref_results(quiz: learning.Quiz, refs: dict[str, str], grade: learning.QuizGrade) -> dict[str, bool]:
    """문제별 정답 → 항목별 결과 (한 항목이 여러 문제에 나오면 모두 맞혀야 맞음)."""
    out: dict[str, bool] = {}
    for q in quiz.questions:
        ref = refs.get(q.id)
        if ref:
            out[ref] = out.get(ref, True) and q.id not in grade.wrong
    return out


def review_box(ctx: Ctx, course_id: str, quiz: learning.Quiz, refs: dict[str, str], state: dict, key: str) -> None:
    """복습 퀴즈 풀기 → 공용 채점(grade_quiz) → 퀴즈 기록 + 항목마다 다음 복습일 (간격 반복)."""
    graded = state.get("grade")
    if not graded:
        with st.form(key):
            for i, q in enumerate(quiz.questions, 1):
                st.radio(f"**{i}.** {q.question}", q.choices, index=None, key=f"{key}_{q.id}")
            go = st.form_submit_button("✅ 채점하기", type="primary")
        if go:
            answers = {q.id: st.session_state.get(f"{key}_{q.id}") for q in quiz.questions}
            grade = learning.grade_quiz(quiz, answers)
            results = _ref_results(quiz, refs, grade)
            course = ctx.store.get(course_id)
            if course is not None:
                def record(c: learning.Course) -> dict[str, str]:
                    c.quiz_results.append(learning.quiz_result(c, quiz, grade))
                    hobby.apply_review(c, results)
                    return {ref: (hobby.find_item(c, ref) or ("", {}))[1].get("due", "") for ref in results}

                dues = ctx.store.update(course_id, record)
            else:
                dues = {}
            state["grade"] = {"correct": grade.correct, "total": grade.total, "pct": grade.pct, "details": grade.details,
                              "refs": refs, "dues": dues, "saved": course is not None}
            st.rerun()
        return
    g = graded
    st.markdown(f"**점수 {g['correct']}/{g['total']} ({g['pct']}%)**")
    for i, d in enumerate(g["details"], 1):
        ref = g["refs"].get(d["id"], "")
        due = g["dues"].get(ref, "")
        picked = d["picked"] if d["picked"] is not None else "(안 고름)"
        st.caption(f"{'✅' if d['ok'] else '❌'} {i}. 고른 답: {picked} · 정답: {d['answer']}"
                   + (f" — {d['explanation']}" if d["explanation"] else "") + (f" · 🔁 다음 복습 {due}" if due else ""))
    if g["saved"]:
        st.success("🔁 채점 결과로 다음 복습일을 정했습니다 — 맞힌 것은 더 뒤로, 틀린 것은 내일 다시.")
    else:
        st.warning("과목을 찾지 못해 복습일은 저장하지 않았습니다.")


def review_preview(ctx: Ctx, run: dict, text: str) -> None:
    try:
        quiz = learning.parse_quiz(text)
    except ValueError as e:
        st.warning(f"🔁 복습 퀴즈 형식이 맞지 않아 풀 수 없습니다 — {e}")
        return
    quiz.kind = "review"
    st.markdown(f"#### 🔁 복습 퀴즈 풀기 — {len(quiz.questions)}문제 (채점은 코드가)")
    if not run.get("hb_review", {}).get("grade"):
        st.caption("정답은 채점 뒤에 보입니다 — 아래 '최종 결정'의 답변 칸에는 정답 JSON 이 있으니 먼저 풀어 보세요. 채점하면 바로 저장되므로 채택하지 않아도 됩니다.")
    review_box(ctx, _cid(run), quiz, hobby.review_refs(text), run.setdefault("hb_review", {}), f"hb_review_{run['run_id']}")


FLASH_KEY = "hb_flash_{}"


def _start_flash(ctx: Ctx, course_id: str, count: int) -> None:
    course = ctx.store.get(course_id)
    made = hobby.flashcard_quiz(course, count=count) if course else None
    if made is None:
        st.session_state[education_ui.MSG_KEY] = "⚡ 지금 복습할 단어·표현이 없거나(보기를 만들려면 2개 이상 필요) 모두 외웠습니다."
        return
    quiz, refs = made
    st.session_state[FLASH_KEY.format(course_id)] = {"quiz": quiz.model_dump(), "refs": refs, "n": st.session_state.get("hb_flash_n", 0) + 1}
    st.session_state["hb_flash_n"] = st.session_state[FLASH_KEY.format(course_id)]["n"]


def _stop_flash(course_id: str) -> None:
    st.session_state.pop(FLASH_KEY.format(course_id), None)


def review_section(ctx: Ctx, course: learning.Course) -> None:
    """⚡ 바로 복습 — 모델 없이 단어장·표현으로 낱말 카드 퀴즈 (복습 대상만)."""
    if not (course.extras.get("vocab") or course.extras.get("phrases")):
        return
    state = st.session_state.get(FLASH_KEY.format(course.id))
    due = len(hobby.due_items(course, limit=1000))
    if state is None:
        st.button(f"⚡ 바로 복습 — 모델 없이 낱말 카드 (오늘 복습 {due}개)", key=f"hb_flash_go_{course.id}",
                  disabled=due == 0, on_click=_start_flash, args=(ctx, course.id, 5),
                  help="외움이 아니고 복습일이 지난 단어·표현으로 5문제. 채점하면 다음 복습일이 정해집니다.")
        return
    with st.container(border=True):
        st.markdown("**⚡ 바로 복습** (모델 없이 · 채점은 코드가)")
        quiz = learning.Quiz.model_validate(state["quiz"])
        review_box(ctx, course.id, quiz, state["refs"], state, f"hb_flash_{course.id}_{state['n']}")
        st.button("닫기", key=f"hb_flash_close_{course.id}", on_click=_stop_flash, args=(course.id,))


# ─── 등록 ───────────────────────────────────────────────────────────

for _label, _task in [("📅 커리큘럼", "hobby_curriculum"), ("📖 단어장", "vocab"), ("🗣 표현", "phrases"),
                      ("🧭 더 배울 것", "topics"), ("🔁 복습", "review")]:
    education_ui.register_quick(hobby.KIND, _label, _task)

education_ui.register_preview("hobby_curriculum", curriculum_preview)
education_ui.register_preview("vocab", vocab_preview)
education_ui.register_preview("phrases", phrases_preview)
education_ui.register_preview("topics", topics_preview)
education_ui.register_preview("review", review_preview)

education_ui.register_adopt("hobby_curriculum", on_curriculum_adopt)
education_ui.register_adopt("vocab", on_vocab_adopt)
education_ui.register_adopt("phrases", on_phrases_adopt)
education_ui.register_adopt("topics", on_topics_adopt)

education_ui.register_answer_hidden("review")  # 복습 퀴즈 정답 JSON 은 결과 상자에서 감춘다

for _fn in (vocab_section, phrases_section, review_section, topics_section):
    education_ui.register_section(hobby.KIND, _fn)
