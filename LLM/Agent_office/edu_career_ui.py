"""🧭 직무 전환 교육팀 화면 (2단계) — education_ui 의 확장 지점에 등록만 한다 (education_ui.py 맨 아래가 이 모듈을 자동으로 읽는다).

  과목 카드 버튼 (register_quick) : 🗺 로드맵 · 🛠 프로젝트 · 📌 점검 · 🎯 JD — 누르면 그 업무를 고르고 과목 칸을 채운다
  📖 자세히 (register_section)     : 🗺 로드맵 요약 · 🛠 프로젝트·마일스톤 체크(완료·막힘·메모) · 🎯 JD 기록(올리기·1차 대응·커리어부로 가져가기·검색 승인)
  결과 아래 (register_preview)     : 로드맵 주차별 시간 합 · 프로젝트 미리 보기 · 점검 적용 미리 보기 · JD 1차 대응
  채택 (register_adopt)            : 로드맵 → extras["roadmap"] + 남은 커리큘럼 / 프로젝트 → extras["projects"] / 점검 → 기한·새 마일스톤 /
                                     JD → extras["jd_links"] + 부족한 요건 단원
계산·검증은 core/learning_career.py (화면 없음). 커리어부 파일은 고치지 않고 core/career.py 함수·화면 키만 가져다 쓴다.
"""

from __future__ import annotations

import streamlit as st

import education_ui
from core import learning
from core import learning_career as lc
from core.config import load_settings
from core.schemas import HumanDecision

KIND = "career"
TEAM_ID = "edu_career"
QUICK = (("🗺 로드맵", "roadmap"), ("🛠 프로젝트", "project_plan"), ("📌 점검", "project_check"), ("🎯 JD", "jd_match"))
JD_FILE_TYPES = ["docx", "pdf", "hwpx", "md", "txt"]  # 글 파일만 — 그림·스캔 JD 는 커리어부 '🎯 업무(JD) 추천'(비전 모델 확인 단계)에서

# 커리어부 '🎯 업무(JD) 추천'으로 가져가기 — app.py 의 화면 키 이름 (app.py 를 import 하면 앱이 실행되므로 이름만 같게 둔다)
VIEW_KEY, DEPT_VIEW, DEPT_KEY = "view", "dept", "dept_pick"
CAREER_DEPT, CAREER_TEAM, CAREER_TASK = "career", "career", "jd_recommend"


def _key(course: learning.Course, *parts: str) -> str:
    return "_".join(["edc", course.id, *parts])


def _msg(text: str) -> None:
    st.session_state[education_ui.MSG_KEY] = text


def _course_of(ctx: education_ui.Ctx, run: dict) -> learning.Course | None:
    return ctx.store.get(str((run.get("edu_inputs") or {}).get(education_ui.COURSE_FIELD) or ""))


def _hours_limit(run: dict, course: learning.Course | None) -> int | None:
    """업무 양식의 주당 학습 시간 → 없으면 과목에 적은 시간 (0.1시간 단위 정수)."""
    return lc.tenths((run.get("edu_inputs") or {}).get("weekly_hours")) or (lc.tenths(course.weekly_hours) if course else None)


# ─── 📖 자세히 — 🗺 로드맵 ──────────────────────────────────────────────


def roadmap_section(ctx: education_ui.Ctx, course: learning.Course) -> None:
    road = course.extras.get("roadmap") or {}
    st.markdown("**🗺 합격 로드맵**")
    if not road:
        st.caption("아직 로드맵이 없습니다 — 카드의 '🗺 로드맵'으로 목표 직무에 맞춘 주차별 로드맵을 짜 보세요.")
        return
    st.caption(f"목표 직무 **{road.get('target_role') or '-'}** · 기간 {road.get('deadline') or '-'}"
               + (f" · 주당 {road['weekly_hours']}시간" if road.get("weekly_hours") else "") + f" · 채택 {str(road.get('at', ''))[:10]}")
    if road.get("competencies"):
        education_ui._table([{"역량": c.get("name"), "합격에 필요한 수준": c.get("need"), "지금": c.get("now"), "채울 것": c.get("gap")}
                             for c in road["competencies"]])
    hours = lc.week_hours(road.get("steps") or [])
    if hours:
        st.caption("주차별 학습 시간 (코드 계산): " + " · ".join(f"{w}주 {lc.hours_text(h)}h" for w, h in list(hours.items())[:24]))
    extra = [f"🎫 자격증: {', '.join(road['certs'])}" if road.get("certs") else "",
             f"🗂 포트폴리오로 남길 것: {', '.join(road['portfolio'])}" if road.get("portfolio") else ""]
    if any(extra):
        st.caption("  \n".join(x for x in extra if x))


# ─── 📖 자세히 — 🛠 프로젝트·마일스톤 체크 ─────────────────────────────────


def _set_ms(ctx: education_ui.Ctx, cid: str, pid: str, mid: str, what: str, key: str) -> None:
    value = st.session_state.get(key)
    kwargs = {"done": bool(value)} if what == "done" else {"blocked": bool(value)} if what == "blocked" else {"memo": str(value or "")}
    ctx.store.update(cid, lambda c: lc.set_milestone(c, pid, mid, **kwargs))


def _set_status(ctx: education_ui.Ctx, cid: str, pid: str, key: str) -> None:
    ctx.store.update(cid, lambda c: lc.set_project_status(c, pid, st.session_state.get(key)))


def _remove(ctx: education_ui.Ctx, cid: str, pid: str) -> None:
    ctx.store.update(cid, lambda c: lc.remove_project(c, pid))
    _msg(f"🗑 프로젝트 {pid} 를 지웠습니다.")


def _check_task(ctx: education_ui.Ctx, cid: str, pid: str) -> None:
    """콜백 — '📌 진행 점검' 업무를 고르고 과목 칸과 프로젝트 칸을 채운다."""
    education_ui.pick_course(ctx, cid, TEAM_ID, "project_check")
    st.session_state[education_ui.field_key(TEAM_ID, "project_check", "project_id")] = pid


def projects_section(ctx: education_ui.Ctx, course: learning.Course) -> None:
    projects = course.extras.get("projects") or []
    st.markdown(f"**🛠 개인 프로젝트 ({len(projects)})** — 마일스톤을 체크하세요 (✅ 완료 · ⛔ 막힘 · 메모)")
    if not projects:
        st.caption("아직 프로젝트가 없습니다 — 카드의 '🛠 프로젝트'로 포트폴리오용 프로젝트를 설계해 보세요.")
        return
    for p in projects:
        pid = p["id"]
        with st.container(border=True):
            head, stat = st.columns([3, 1])
            head.markdown(f"**{pid} · {p.get('title')}**  \n{p.get('problem') or ''}")
            head.caption(f"결과물: {p.get('deliverable') or '-'} · 기술: {', '.join(p.get('tech') or []) or '-'}"
                         + (f" · 난이도 {learning.level_of(p['difficulty']).name}" if p.get("difficulty") in learning.LEVEL_IDS else ""))
            skey = _key(course, pid, "status")
            status = p.get("status") if p.get("status") in lc.PROJECT_STATUSES else "계획"
            st.session_state[skey] = status  # 체크로 상태가 바뀌면 고르기 칸도 따라간다
            stat.selectbox("상태", lc.PROJECT_STATUSES, key=skey, on_change=_set_status, args=(ctx, course.id, pid, skey))
            st.progress(lc.project_pct(p), text=f"진행 {lc.project_pct(p)}%")
            for m in p.get("milestones") or []:
                mid = m["id"]
                c1, c2, c3 = st.columns([5, 1, 3])
                dk, bk, mk = (_key(course, pid, mid, x) for x in ("done", "blocked", "memo"))
                c1.checkbox(f"{mid} · {m.get('due') or '-'} · {m.get('title')}", value=bool(m.get("done")), key=dk,
                            on_change=_set_ms, args=(ctx, course.id, pid, mid, "done", dk))
                c2.checkbox("⛔ 막힘", value=bool(m.get("blocked")), key=bk, disabled=bool(m.get("done")),
                            on_change=_set_ms, args=(ctx, course.id, pid, mid, "blocked", bk))
                c3.text_input("메모", value=m.get("memo") or "", key=mk, label_visibility="collapsed", placeholder="메모 (Enter 로 저장)",
                              on_change=_set_ms, args=(ctx, course.id, pid, mid, "memo", mk))
            if p.get("portfolio_line"):
                st.caption(f"🗂 포트폴리오 문장 초안: {p['portfolio_line']}")
            last = p.get("last_check") or {}
            if last.get("next"):
                st.caption(f"📌 지난 점검 ({str(last.get('at', ''))[:10]}): 다음 할 일 — " + " / ".join(last["next"][:3]))
            b1, b2 = st.columns([3, 1])
            b1.button("📌 이 프로젝트 진행 점검 맡기기", key=_key(course, pid, "check"), use_container_width=True,
                      on_click=_check_task, args=(ctx, course.id, pid))
            with b2.popover("🗑 지우기", use_container_width=True):
                st.caption("이 프로젝트와 마일스톤 기록이 지워집니다.")
                st.button("지우기", key=_key(course, pid, "del"), type="primary", on_click=_remove, args=(ctx, course.id, pid))


# ─── 📖 자세히 — 🎯 JD 기록 ─────────────────────────────────────────────

KIND_TEXT = {"jd": "📋 JD", "advice": "🎯 추천", "search": "🔎 검색"}


def _save_jds(ctx: education_ui.Ctx, cid: str, text_key: str, file_key: str) -> None:
    """콜백 — 붙여 넣은 JD('---' 로 나눔) + 올린 글 파일 → 커리어부 코드로 요구사항·1차 대응 → 과목 JD 기록."""
    from core import career
    from core.doc_text import DocReadError, read_bytes

    pairs = [(f"붙여 넣은 JD {i}", t) for i, t in enumerate(career.split_jds(st.session_state.get(text_key) or ""), 1)]
    problems = []
    for f in st.session_state.get(file_key) or []:
        try:
            text = read_bytes(f.name, f.getvalue()).text.strip()
        except (DocReadError, OSError, ValueError) as e:
            problems.append(f"'{f.name}' 읽기 실패: {e}")
            continue
        if text:
            pairs.append((f.name, text))
    if not pairs:
        _msg("읽은 JD 가 없습니다 — 붙여 넣거나 글 파일(docx·pdf·hwpx·md·txt)을 올리세요." + (" " + " / ".join(problems) if problems else ""))
        return
    entries = ctx.store.update(cid, lambda c: _add_jds(c, pairs))
    st.session_state[text_key] = ""
    _msg(f"🎯 JD {len(entries)}개를 과목에 남겼습니다 (요구사항·1차 대응은 커리어부 코드가 학습 기록으로 계산)."
         + (" " + " / ".join(problems) if problems else ""))


def _add_jds(course: learning.Course, pairs: list[tuple[str, str]]) -> list[dict]:
    entries = lc.jd_entries(pairs, course)
    lc.add_links(course, entries)
    return entries


def to_career(ctx: education_ui.Ctx, cid: str) -> None:
    """콜백 — 커리어부 '🎯 업무(JD) 추천' 양식을 열고 JD · 희망 방향 · 학습 기록 요약을 채운다 (커리어부 파일은 고치지 않는다)."""
    course = ctx.store.get(cid)
    if course is None:
        return
    values = lc.career_handoff(course)
    st.session_state[VIEW_KEY] = DEPT_VIEW
    st.session_state[DEPT_KEY] = CAREER_DEPT
    st.session_state[ctx.team_key] = CAREER_TEAM
    st.session_state[f"dept_task_{CAREER_DEPT}"] = f"{CAREER_TEAM}:{CAREER_TASK}"
    for field, value in values.items():
        st.session_state[f"tf_{CAREER_TEAM}_{CAREER_TASK}_{field}"] = value


def _search(ctx: education_ui.Ctx, cid: str, key: str) -> None:
    """콜백 — 사람이 승인한 검색어로만 검색 (모델이 검색어를 다시 만들지 않는다) → 결과 제목·주소를 JD 기록에."""
    from core import search

    course = ctx.store.get(cid)
    queries = [q.strip() for q in str(st.session_state.get(key) or "").splitlines() if q.strip()]
    if course is None or not queries or any(lc.query_problems(q, course) for q in queries):
        _msg("검색어 검사를 통과하지 못해 검색하지 않았습니다.")
        return
    web = search.search_queries(queries, load_settings())
    stamp = learning.now_text()
    found = [{"kind": "search", "title": r.title[:120], "url": r.url, "note": r.snippet[:160], "at": stamp,
              "query": " / ".join(queries)} for r in web.results]
    ctx.store.update(cid, lambda c: lc.add_links(c, found))
    _msg(f"🔎 승인한 검색어 {len(queries)}개로 {len(found)}건을 찾아 JD 기록에 남겼습니다." + (" " + " / ".join(web.notes) if web.notes else ""))


def jd_section(ctx: education_ui.Ctx, course: learning.Course) -> None:
    links = course.extras.get("jd_links") or []
    st.markdown(f"**🎯 JD 기록 ({len(links)})**")
    if links:
        education_ui._table([{"때": str(x.get("at", ""))[:10], "종류": KIND_TEXT.get(x.get("kind"), x.get("kind") or ""),
                              "제목": x.get("title"), "메모": education_ui._short(x.get("note") or "", 60), "주소": x.get("url") or ""}
                             for x in reversed(links[-12:])])
    last = next((x for x in reversed(links) if x.get("kind") == "jd"), None)
    if last and last.get("reqs"):
        st.caption(f"📋 최근 JD '{last['title']}' 요구사항 1차 대응 — 학습 기록(끝낸 단원·프로젝트)에 키워드가 있는지 코드가 본 것: {last.get('basis')}")
        education_ui._table([{"ID": r["id"], "구분": r["section"], "요구사항": r["text"], "1차 대응": r["status"]} for r in last["reqs"]])
    with st.expander("📥 JD 붙여 넣기·파일 올리기 (코드가 요구사항을 뽑아 저장)"):
        tk, fk = _key(course, "jd_text"), _key(course, "jd_files")
        st.text_area("JD 붙여 넣기 (여러 개는 '---' 만 있는 줄로 나눔)", key=tk, height=110)
        st.file_uploader("또는 JD 글 파일", type=JD_FILE_TYPES, accept_multiple_files=True, key=fk,
                         help="글 파일만 읽습니다. 그림·스캔 JD 는 커리어부 '🎯 업무(JD) 추천'의 확인 단계(비전 모델)를 쓰세요.")
        st.button("📥 읽고 과목에 남기기", key=_key(course, "jd_save"), on_click=_save_jds, args=(ctx, course.id, tk, fk))
    c1, c2 = st.columns(2)
    c1.button("🎯 JD 방향 추천 맡기기", key=_key(course, "jd_task"), use_container_width=True,
              on_click=education_ui.pick_course, args=(ctx, course.id, TEAM_ID, "jd_match"))
    c2.button("👔 커리어부 JD 추천으로 가져가기", key=_key(course, "to_career"), use_container_width=True,
              disabled=CAREER_TEAM not in ctx.teams_by_id, on_click=to_career, args=(ctx, course.id),
              help="커리어부 '🎯 업무(JD) 추천' 양식을 열고 최근 JD(최대 3개)·희망 방향·학습 기록 요약을 채웁니다. "
                   "그곳에서 경력 요약 문서를 함께 올리면 정식 적합도 점수(코드 계산)를 받습니다.")
    qk = _key(course, "queries")
    if qk not in st.session_state:
        st.session_state[qk] = "\n".join(lc.draft_queries(course))
    pending = [q.strip() for q in str(st.session_state.get(qk) or "").splitlines() if q.strip()]
    blocked = any(lc.query_problems(q, course) for q in pending)  # 걸린 검색어가 있으면 펼쳐 둔다 (다시 그려도 경고가 보이게)
    with st.expander("🔎 채용 트렌드 웹 검색 (검색어를 확인하고 승인해야 검색)", expanded=blocked):
        st.text_area("검색어 (한 줄에 하나 — 고치거나 지운 뒤 승인)", key=qk, height=90,
                     help="직무명·일반 기술 용어·연도만 씁니다. 목표 회사·사람 이름·숫자처럼 내 글에 나온 말은 코드가 막습니다 (커리어부 검사 그대로).")
        queries = [q.strip() for q in str(st.session_state.get(qk) or "").splitlines() if q.strip()]
        bad = {q: why for q in queries if (why := lc.query_problems(q, course))}
        for q, why in bad.items():
            st.warning(f"'{q}' — {' / '.join(why)}")
        st.button("✅ 이 검색어로 검색 승인", key=_key(course, "search"), disabled=bool(bad) or not queries,
                  on_click=_search, args=(ctx, course.id, qk))


# ─── 결과 아래 미리 보기 ────────────────────────────────────────────────


def _saved(run: dict) -> bool:
    return bool(run.get("edu_saved"))


def roadmap_preview(ctx: education_ui.Ctx, run: dict, text: str) -> None:
    course = _course_of(ctx, run)
    try:
        road = lc.roadmap_from_answer(text)
    except ValueError as e:
        st.warning(f"🗺 로드맵 JSON 형식이 맞지 않아 채택해도 과목에 저장되지 않습니다 — {e}")
        return
    limit = _hours_limit(run, course)
    hours = lc.week_hours(road["steps"])
    with st.container(border=True):
        st.markdown(f"#### 🗺 로드맵 미리 보기 — {road['target_role']} · {len(road['steps'])}단계")
        education_ui._table([{"주차": w, "시간 합 (코드)": f"{lc.hours_text(h)}h", "주당 한도": f"{lc.hours_text(limit)}h" if limit else "-",
                              "": "⚠ 넘침" if limit and h > limit else "✅"} for w, h in hours.items()])
        if course is not None and not _saved(run):
            left = sum(not u.done for u in course.curriculum)
            st.caption(f"✅ 채택하면 남은(안 끝낸) 단원 {left}개가 로드맵 단계 {len(road['steps'])}개로 바뀝니다 (끝낸 단원 "
                       f"{len(course.curriculum) - left}개는 그대로). 로드맵은 '📖 자세히 → 🗺 합격 로드맵'에 남습니다.")


def project_preview(ctx: education_ui.Ctx, run: dict, text: str) -> None:
    course = _course_of(ctx, run)
    try:
        projects = lc.projects_from_answer(text, course.level if course else None)
    except ValueError as e:
        st.warning(f"🛠 프로젝트 JSON 형식이 맞지 않아 채택해도 과목에 저장되지 않습니다 — {e}")
        return
    with st.container(border=True):
        st.markdown(f"#### 🛠 프로젝트 미리 보기 — {len(projects)}개")
        education_ui._table([{"프로젝트": p["title"], "난이도": learning.level_of(p["difficulty"]).name, "기술": ", ".join(p["tech"]),
                              "마일스톤": " → ".join(f"{m['title']}({m['due']})" for m in p["milestones"])} for p in projects])
        if not _saved(run):
            st.caption("✅ 채택하면 과목에 저장되고, '📖 자세히 → 🛠 개인 프로젝트'에서 마일스톤을 체크할 수 있습니다.")


def check_preview(ctx: education_ui.Ctx, run: dict, text: str) -> None:
    course = _course_of(ctx, run)
    data = learning.find_json(text, "check")
    problems = lc.check_problems(data, course)
    if problems:
        st.warning("📌 진행 점검 JSON 이 과목과 맞지 않아 채택해도 일정이 바뀌지 않습니다 — " + " / ".join(problems[:3]))
        return
    if _saved(run) or course is None:
        return
    trial = course.model_copy(deep=True)
    result = lc.apply_check(trial, data["check"])
    with st.container(border=True):
        st.markdown(f"#### 📌 채택하면 {result['project']} 에 반영될 것")
        st.caption("기한 변경: " + (" · ".join(result["changed"]) or "없음") + " / 새 마일스톤: " + (", ".join(result["added"]) or "없음"))
        education_ui._table([{"마일스톤": f"{m['id']} {m['title']}", "기한": m.get("due"), "완료": "✅" if m.get("done") else ""}
                             for m in lc.find_project(trial, result["project"])["milestones"]])


def jd_preview(ctx: education_ui.Ctx, run: dict, text: str) -> None:
    from core import career

    course = _course_of(ctx, run)
    try:
        adv = lc.advice_from_answer(text, course)
    except ValueError as e:
        st.warning(f"🎯 JD 추천 JSON 형식이 맞지 않아 채택해도 과목에 저장되지 않습니다 — {e}")
        return
    pairs = [(f"붙여 넣은 JD {i}", t) for i, t in enumerate(career.split_jds(str((run.get("edu_inputs") or {}).get("jd") or "")), 1)]
    with st.container(border=True):
        st.markdown("#### 🎯 JD 추천 미리 보기")
        st.caption(f"지원 구분: {adv.get('seniority')} · 키워드: {', '.join(lc._strs(adv.get('keywords')))}")
        if course is not None and pairs:
            entries = lc.jd_entries(pairs, course)
            education_ui._table([{"JD": e["title"], "1차 적합도 (학습 기록 · 코드)": f"{e['score']:g}" if e["score"] is not None else "-",
                                  "공백 요구사항": " / ".join(e["gaps"][:4])} for e in entries])
        if not _saved(run):
            n = len([u for u in lc._list(adv.get("add_units")) if isinstance(u, dict)])
            st.caption(f"✅ 채택하면 추천·JD {len(pairs)}개가 '🎯 JD 기록'에 남고, 부족한 요건 단원 {n}개가 커리큘럼 맨 뒤에 더해집니다.")


# ─── 채택 ───────────────────────────────────────────────────────────


def _cid(run: dict) -> str:
    return str((run.get("edu_inputs") or {}).get(education_ui.COURSE_FIELD) or "")


def on_roadmap_adopt(ctx: education_ui.Ctx, run: dict, decision: HumanDecision) -> str | None:
    road = lc.roadmap_from_answer(decision.final_answer)
    removed = ctx.store.update(_cid(run), lambda c: lc.apply_roadmap(c, road))
    return (f"🗺 로드맵을 과목에 저장했습니다 — {road['target_role']} · {len(road['steps'])}단계. "
            f"남은 단원 {len(removed)}개를 로드맵 단계로 바꿨습니다 (끝낸 단원은 그대로).")


def on_project_adopt(ctx: education_ui.Ctx, run: dict, decision: HumanDecision) -> str | None:
    course = ctx.store.get(_cid(run))
    projects = lc.projects_from_answer(decision.final_answer, course.level if course else None)
    added = ctx.store.update(_cid(run), lambda c: lc.add_projects(c, projects))
    return f"🛠 프로젝트 {len(added)}개({', '.join(p['id'] for p in added)})를 과목에 저장했습니다 — '📖 자세히'에서 마일스톤을 체크하세요."


def on_check_adopt(ctx: education_ui.Ctx, run: dict, decision: HumanDecision) -> str | None:
    course = ctx.store.get(_cid(run))
    data = learning.find_json(decision.final_answer, "check")
    problems = lc.check_problems(data, course)
    if problems:
        raise ValueError(" / ".join(problems[:3]))
    result = ctx.store.update(_cid(run), lambda c: lc.apply_check(c, data["check"]))
    return (f"📌 {result['project']} 일정을 고쳤습니다 — 기한 변경 {len(result['changed'])}개 · 새 마일스톤 {len(result['added'])}개 "
            "(끝낸 마일스톤은 그대로).")


def on_jd_adopt(ctx: education_ui.Ctx, run: dict, decision: HumanDecision) -> str | None:
    from core import career

    course = ctx.store.get(_cid(run))
    adv = lc.advice_from_answer(decision.final_answer, course)
    pairs = [(f"붙여 넣은 JD {i}", t) for i, t in enumerate(career.split_jds(str((run.get("edu_inputs") or {}).get("jd") or "")), 1)]
    result = ctx.store.update(_cid(run), lambda c: lc.apply_advice(c, adv, pairs))
    return (f"🎯 JD 추천을 과목에 남겼습니다 — JD {len(result['jds'])}개 · 커리큘럼에 단원 {len(result['units'])}개 추가"
            + (f" ({', '.join(result['units'])})" if result["units"] else "") + ".")


# ─── 등록 ───────────────────────────────────────────────────────────

for _label, _task in QUICK:
    education_ui.register_quick(KIND, _label, _task)
# 칸은 한 번씩만 — 이 모듈을 다시 읽어도(Streamlit 개발 모드 다시 불러오기) 같은 칸이 두 번 그려지지 않게 예전 등록을 바꾼다
_sections = education_ui.COURSE_SECTIONS[KIND]
_sections[:] = [f for f in _sections if getattr(f, "__module__", "") != __name__]
for _fn in (roadmap_section, projects_section, jd_section):
    education_ui.register_section(KIND, _fn)
education_ui.register_preview("roadmap", roadmap_preview)
education_ui.register_preview("project_plan", project_preview)
education_ui.register_preview("project_check", check_preview)
education_ui.register_preview("jd_match", jd_preview)
education_ui.register_adopt("roadmap", on_roadmap_adopt)
education_ui.register_adopt("project_plan", on_project_adopt)
education_ui.register_adopt("project_check", on_check_adopt)
education_ui.register_adopt("jd_match", on_jd_adopt)
