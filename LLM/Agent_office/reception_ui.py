"""🛎 인포메이션 센터 — 1층 로비 접수 창구 (core/reception.py · core/dept_builder.py).

  질문 → 👔 부서장 회의(모델 1회로 모든 부서 판단) → 부서마다 ✅/❌·이유 표
    맡을 부서가 있으면 → [그 부서장에게 넘기기] → 부서장 창구가 열리고 곧바로 팀 배정 (building_ui 2번 흐름)
    없으면 → "이 일을 할 부서가 없습니다. 새 부서를 만들까요?" + 제안(부서·팀 이름, 하는 일, 팀 구성, 필요한 도구)
      [만들기] → 파일 미리 보기(고칠 수 있음) → [✅ 확인하고 저장] 을 눌러야 저장 → "원래 질문을 새 부서로 넘길까요?"
      [나중에] → 🗂 추후 진행 목록(data/new_dept_later.json) — 나중에 다시 꺼내 [만들기]
  만들기·나중에는 자율 실행과 같은 해시 체인 감사 로그에 남는다 (dept_created / dept_deferred).
"""

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

import building_ui
from core import building, dept_builder, reception
from core.dept_builder import DeptBuildError

TEXT_KEY = "info_text"
RESULT_KEY = "info_result"  # {question, meeting, proposal?}
BUILD_KEY = "info_build"  # [만들기] 를 누른 뒤의 초안 {question, proposal, later_id}
CREATED_KEY = "info_created"  # 저장한 뒤 {dept_id, dept_name, team_name, question}
FLASH_KEY = "info_flash"
DEPT_YAML_KEY = "info_dept_yaml"
TEAM_YAML_KEY = "info_team_yaml"


@dataclass
class Ctx:
    teams: list
    ranks: list
    models: object
    data_dir: Path
    settings: dict


def _later(ctx: Ctx) -> reception.LaterStore:
    return reception.LaterStore(ctx.data_dir)


def _taken(ctx: Ctx, depts) -> set[str]:
    return {t.id for t in ctx.teams} | {d.id for d in depts} | {"etc"}


def render(ctx: Ctx, depts: list) -> None:
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)
    if st.session_state.get(CREATED_KEY):
        _created(ctx, depts)
        return
    if st.session_state.get(BUILD_KEY):
        _build(ctx, depts)
        return
    st.caption("무엇이든 물어보세요. 👔 부서장들이 '우리 부서 일인지'를 판단해 맡을 부서로 넘기고, 그 부서장이 팀을 배정합니다. "
               "맡을 부서가 없으면 새 부서를 제안합니다.")
    text = st.text_area("🛎 질문", key=TEXT_KEY, height=80, placeholder="예: 다음 달 오사카 3박 4일 / 이 테라폼 코드 보안 점검 / 부가세 신고서 써줘")
    if st.button("🛎 부서장 회의에 묻기", key="info_go", type="primary", disabled=not text.strip(),
                 help="보조(부장) 모델 1회로 모든 부서를 한꺼번에 판단합니다 (부서마다 따로 부르지 않음). 모델을 못 쓰면 낱말 비교."):
        with st.spinner("👔 부서장 회의 중…"):
            result = reception.meeting(text.strip(), depts, ctx.teams, ctx.models)
        st.session_state[RESULT_KEY] = {"question": text.strip(), "meeting": result}
    res = st.session_state.get(RESULT_KEY)
    if res:
        _result(ctx, depts, res)
    _later_list(ctx, depts)


def _verdicts(meeting: dict, depts) -> None:
    by_id = {d.id: d for d in depts}
    rows = []
    for v in meeting["verdicts"]:
        d = by_id.get(v["dept_id"])
        if d is None:
            continue
        mark = "✅" if v["can_handle"] else "❌"
        rows.append(f"| {mark} | {d.emoji} {d.name} | {v['fit']} | {v['reason']} |")
    st.markdown("| | 부서 | 맞는 정도 (0~10) | 부서장 판단 |\n|---|---|---|---|\n" + "\n".join(rows))
    how = (f"부서장 회의 — 모델 1회 ({meeting['model']})" if meeting["how"] == "llm" else "낱말 비교 (모델 없이)")
    st.caption(f"판단: {how}" + (f" · {meeting['analysis']}" if meeting.get("analysis") else "")
               + (f"  \n{meeting['note']}" if meeting.get("note") else ""))


def _result(ctx: Ctx, depts, res: dict) -> None:
    meeting, question = res["meeting"], res["question"]
    by_id = {d.id: d for d in depts}
    _verdicts(meeting, depts)
    if meeting["dept_id"] in by_id:
        d = by_id[meeting["dept_id"]]
        st.success(f"➡️ **{d.emoji} {d.name}** 이(가) 맡을 수 있습니다. 부서장이 알맞은 팀을 배정합니다.")
        c1, c2 = st.columns([2, 1])
        c1.button(f"👔 {d.name} 부서장에게 넘기기", key="info_handoff", type="primary", use_container_width=True,
                  on_click=_handoff, args=(d.id, question))
        other = c2.selectbox("다른 부서로", [x.id for x in depts if x.id != d.id], key="info_other",
                             format_func=lambda i: f"{by_id[i].emoji} {by_id[i].name}", label_visibility="collapsed")
        c2.button("이 부서로 보내기", key="info_handoff_other", on_click=_handoff, args=(other, question), use_container_width=True)
        return
    # 맡을 부서가 없음 → 새 부서 제안 (보조 모델 1회 — 처음 한 번만)
    if "proposal" not in res:
        with st.spinner("🆕 새 부서 초안을 만드는 중…"):
            res["proposal"] = reception.propose(question, depts, ctx.teams, ctx.models, [r.id for r in ctx.ranks], _taken(ctx, depts))
    st.warning("**이 일을 할 부서가 없습니다. 새 부서를 만들까요?**")
    prop = res["proposal"]
    _proposal_card(prop["proposal"])
    st.caption(("제안: 모델 " + prop["model"]) if prop["how"] == "llm" else prop.get("note") or "기본 초안")
    b1, b2 = st.columns(2)
    if b1.button("🆕 만들기", key="info_make", type="primary", use_container_width=True):
        _start_build(question, prop["proposal"], None)
        st.rerun()
    if b2.button("🗂 나중에", key="info_later", use_container_width=True):
        item = _later(ctx).add(question, prop["proposal"])
        dept_builder.audit(ctx.data_dir, ctx.settings, "dept_deferred", later_id=item["id"], question=question[:300],
                           dept_id=prop["proposal"]["dept_id"], team_id=prop["proposal"]["team_id"])
        st.session_state.pop(RESULT_KEY, None)
        st.session_state[FLASH_KEY] = "🗂 추후 진행 목록에 남겼습니다 — 아래 '추후 진행'에서 언제든 다시 만들 수 있습니다."
        st.rerun()


def _proposal_card(p: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{p['emoji']} {p['dept_name']}** (`{p['dept_id']}`) — {p['dept_description']}  \n"
                    f"첫 팀: **{p['team_name']}** (`{p['team_id']}`) — {p['team_description']}")
        st.markdown("**하는 일**: " + " · ".join(p.get("accepts") or [p["scope_summary"]]))
        st.markdown("**팀 구성**: " + " → ".join(f"{m['name']}" + (f"({m['specialty']})" if m.get("specialty") else "")
                                             for m in p["members"]))
        st.markdown("**필요한 도구**: " + (" · ".join(p.get("tools_needed") or []) or "없음")
                    + "  \n<small>⚠️ 웹 검색처럼 밖으로 나가는 도구는 기본으로 꺼 둡니다 — 저장 뒤 team.yaml 의 tools 에서 직접 켭니다.</small>",
                    unsafe_allow_html=True)


def _handoff(dept_id: str, question: str) -> None:
    """그 부서 업무 화면의 부서장 창구로 넘기기 (콜백). 화면이 열리면 부서장이 곧바로 팀을 배정한다."""
    st.session_state.pop(RESULT_KEY, None)
    building_ui.open_desk(dept_id, question, auto=True)


def _start_build(question: str, proposal: dict, later_id: str | None) -> None:
    st.session_state[BUILD_KEY] = {"question": question, "proposal": proposal, "later_id": later_id}
    st.session_state.pop(DEPT_YAML_KEY, None)
    st.session_state.pop(TEAM_YAML_KEY, None)


# ─── [만들기] → 미리 보기 → 확인하고 저장 ───────────────────────────────


def _build(ctx: Ctx, depts) -> None:
    b = st.session_state[BUILD_KEY]
    used = {d.color for d in depts}
    if DEPT_YAML_KEY not in st.session_state:
        dept_text, team_text = dept_builder.draft(b["proposal"], b["question"], [r.id for r in ctx.ranks], used)
        st.session_state[DEPT_YAML_KEY], st.session_state[TEAM_YAML_KEY] = dept_text, team_text
    p = b["proposal"]
    st.markdown(f"##### 🆕 새 부서 미리 보기 — {p['emoji']} {p['dept_name']} · {p['team_name']}")
    st.caption("아래 내용이 그대로 저장됩니다. 고칠 곳이 있으면 고친 뒤 **✅ 확인하고 저장**을 누르세요. "
               "저장 전에 형식·직급·도구·id 겹침을 검사합니다. 도구(tools)는 비워 두었습니다 — 웹 검색 등은 필요할 때 직접 켭니다.")
    c1, c2 = st.columns([2, 3])
    dept_text = c1.text_area("config/departments.yaml 에 넣을 부서", key=DEPT_YAML_KEY, height=320)
    tid = (p.get("team_id") or "new_team")
    team_text = c2.text_area(f"teams/{tid}/team.yaml (teams/_template 바탕)", key=TEAM_YAML_KEY, height=320)
    s1, s2 = st.columns(2)
    if s1.button("✅ 확인하고 저장", key="info_save", type="primary", use_container_width=True):
        try:
            dept, team = dept_builder.validate(dept_text, team_text, team_ids={t.id for t in ctx.teams},
                                               dept_ids={d.id for d in depts}, ranks=ctx.ranks, models=ctx.models)
            path = dept_builder.save(dept, team_text)
        except DeptBuildError as e:
            st.error(f"저장하지 않았습니다 — {e}")
            return
        dept_builder.audit(ctx.data_dir, ctx.settings, "dept_created", dept_id=dept["id"], team_id=dept["teams"][0],
                           path=str(path.name), question=b["question"][:300], later_id=b.get("later_id") or "")
        if b.get("later_id"):
            _later(ctx).remove(b["later_id"])
        st.session_state.pop(BUILD_KEY, None)
        st.session_state.pop(RESULT_KEY, None)
        st.session_state[CREATED_KEY] = {"dept_id": dept["id"], "dept_name": f"{dept.get('emoji', '')} {dept['name']}".strip(),
                                         "team_name": team.get("name", dept["teams"][0]), "question": b["question"]}
        st.rerun()  # 새 팀·층이 빌딩에 보이도록 (app.py 가 teams/ 를 다시 읽는다)
    if s2.button("취소", key="info_cancel", use_container_width=True):
        st.session_state.pop(BUILD_KEY, None)
        st.rerun()


def _created(ctx: Ctx, depts) -> None:
    c = st.session_state[CREATED_KEY]
    floor = ""
    try:
        floor = f" ({building.floor_of(next(d for d in depts if d.id == c['dept_id']).teams[0], ctx.teams, building.load_config())}층)"
    except (KeyError, StopIteration):
        pass
    st.success(f"🎉 새 부서 **{c['dept_name']}** · **{c['team_name']}** 을 만들었습니다{floor}. 빌딩에 새 방이 생겼습니다.")
    st.markdown(f"**원래 질문을 새 부서로 넘길까요?** — {c['question']}")
    b1, b2 = st.columns(2)
    if b1.button("👔 새 부서 부서장에게 넘기기", key="info_forward", type="primary", use_container_width=True,
                 disabled=c["dept_id"] not in {d.id for d in depts}):
        st.session_state.pop(CREATED_KEY, None)
        building_ui.navigate({"kind": "dept", "id": c["dept_id"], "question": c["question"]})
    if b2.button("아니요", key="info_forward_no", use_container_width=True):
        st.session_state.pop(CREATED_KEY, None)
        st.rerun()


# ─── 🗂 추후 진행 ─────────────────────────────────────────────────────


def _later_list(ctx: Ctx, depts) -> None:
    items = _later(ctx).items()
    if not items:
        return
    with st.expander(f"🗂 추후 진행 — 새 부서 후보 ({len(items)})"):
        for item in items:
            p = item["proposal"]
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.markdown(f"**{p['emoji']} {p['dept_name']}** · {p['team_name']} — {item['question'][:60]}  \n"
                        f"<small>{item['created_at']}</small>", unsafe_allow_html=True)
            if c2.button("만들기", key=f"later_make_{item['id']}", use_container_width=True):
                _start_build(item["question"], p, item["id"])
                st.rerun()
            if c3.button("지우기", key=f"later_del_{item['id']}", use_container_width=True):
                _later(ctx).remove(item["id"])
                st.rerun()
