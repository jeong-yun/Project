"""🏙 메인 화면의 픽셀 빌딩 — 층·부서·팀 방 (core/building.py + office_component/building.js).

  - 파이썬은 배치와 방마다 상태·요약 카드만 보낸다. 그림·깜빡임·엘리베이터·낮밤은 브라우저가 그린다.
  - 일하는 팀이 있을 때만 3초마다 상태를 다시 보낸다 (조각 fragment — 화면 전체를 다시 그리지 않음).
  - 방을 누르면 그 팀 화면, 부서 띠를 누르면 🧭 부서장, 로비를 누르면 아래 섹션.
    누른 값은 다음 실행 맨 앞(apply_nav)에서 처리한다 — 사이드바 위젯이 그려진 뒤에는 그 값을 바꿀 수 없어서.
  - '📋 목록으로 보기'를 켜면 그림 없이 부서·팀 목록 (좁은 화면·접근성용).
"""

import time
from collections.abc import Callable
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import building
from core.config import team_org
from core.schemas import Rank, Team

_building = components.declare_component("ao_pixel_building", path=str(Path(__file__).resolve().parent / "office_component"))

NAV_KEY = "building_nav"  # 다음 실행 맨 앞에서 처리할 이동 {kind, id, question}
CLICK_KEY = "building_click_t"
LIST_KEY = "building_list"
DEPT_KEY = "dept_head_pick"
DEPT_TEXT_KEY = "dept_head_text"
ROUTE_KEY = "dept_head_route"
SECTION_KEY = "home_section"
AUTONOMY_OPEN_KEY = "home_autonomy_open"
SECTIONS = {"brief": "📰 AI 브리핑·아이디어", "usage": "💰 사용량·API 키·가격표"}
COST_DAYS = 7
LIVE_SEC = 3


def apply_nav(go_to_team: Callable[..., None]) -> None:
    """app.py 가 위젯을 그리기 전에 부른다 — 빌딩·부서장에서 고른 이동을 처리."""
    nav = st.session_state.pop(NAV_KEY, None)
    if not nav:
        return
    if nav["kind"] == "team":
        go_to_team(nav["id"], nav.get("question") or "")
    elif nav["kind"] == "dept":
        st.session_state[DEPT_KEY] = nav["id"]
        st.session_state.pop(ROUTE_KEY, None)
    elif nav["kind"] == "section":
        if nav["id"] in SECTIONS:
            st.session_state[SECTION_KEY] = nav["id"]
        elif nav["id"] == "autonomy":
            st.session_state[AUTONOMY_OPEN_KEY] = True


def _navigate(nav: dict) -> None:
    st.session_state[NAV_KEY] = nav
    st.rerun()


def snapshot(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], cfg: dict) -> dict:
    """지금 빌딩 상태 — 최근 업무(팀마다 1건)·최근 7일 비용을 저장소에서 읽는다 (DB 가 그대로면 캐시)."""
    recent = building.recent_by_team(storage.list_runs(limit=120), storage.team_for)
    costs, legacy = building.team_costs(storage.usage_by_team(time.time() - COST_DAYS * 86400), storage.team_for)
    orgs = {t.id: team_org(t, ranks) for t in teams}
    return building.snapshot(teams, orgs, ranks, cfg, open_runs=open_runs, recent=recent, costs=costs, cost_days=COST_DAYS,
                             legacy_costs=legacy)


def render(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], go_to_team: Callable[..., None],
           models=None) -> None:
    cfg = building.load_config()
    head, toggle = st.columns([4, 1])
    head.caption("층마다 부서, 방마다 팀입니다. 방에 마우스를 올리면 팀 소개, 누르면 그 팀 화면 · 부서 띠를 누르면 🧭 부서장 · "
                 "로비(1층)를 누르면 아래 내용이 바뀝니다. 팀을 추가하면 층이 자동으로 늘어납니다 (config/departments.yaml).")
    as_list = toggle.toggle("📋 목록으로 보기", key=LIST_KEY, help="그림 대신 부서·팀 목록 — 좁은 화면이나 화면 읽기 프로그램용")
    live = any(not r["done"] for r in open_runs)
    if as_list:
        _list_view(snapshot(storage, teams, ranks, open_runs, cfg), go_to_team)
    else:
        # 일하는 팀이 있을 때만 몇 초마다 이 조각만 다시 그려 상태를 보낸다 (그림은 iframe 안에서 계속 움직인다).
        # 실행 dict 는 진행 스레드가 그 자리에서 고치므로 조각만 다시 그려도 최신 상태다. 쉬는 중이면 조각을 쓰지 않는다.
        if live:
            st.fragment(run_every=LIVE_SEC)(_draw)(storage, teams, ranks, open_runs, cfg)
        else:
            _draw(storage, teams, ranks, open_runs, cfg)
    if models is not None:
        _dept_head(teams, cfg, models)


def _draw(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], cfg: dict) -> None:
    snap = snapshot(storage, teams, ranks, open_runs, cfg)
    value = _building(snapshot=snap, key="pixel_building", default=None)
    # 컴포넌트 값은 다음 실행에도 남으므로, 누른 시각(t)이 새로울 때만 '방금 누름'으로 본다
    if isinstance(value, dict) and value.get("t") != st.session_state.get(CLICK_KEY):
        st.session_state[CLICK_KEY] = value.get("t")
        if value.get("kind") in ("team", "dept", "section") and value.get("id"):
            _navigate({"kind": value["kind"], "id": value["id"]})


def _list_view(snap: dict, go_to_team: Callable[..., None]) -> None:
    """📋 목록으로 보기 — 층(위에서부터) · 부서 · 팀 카드."""
    layout = snap["layout"]
    for floor in reversed(layout["floors"]):  # 맨 위층부터 (빌딩 그림과 같은 순서)
        rooms = [r for r in floor["rooms"] if r["team"]]
        depts = " · ".join(f"{snap['depts'][b['dept']]['emoji']} {snap['depts'][b['dept']]['name']}" for b in floor["banners"])
        cost = snap["floorCosts"][layout["floors"].index(floor)]
        st.markdown(f"##### {floor['n']}층 — {depts}" + (f" · 💰 최근 {COST_DAYS}일 {cost}" if cost else ""))
        cols = st.columns(2)
        for i, r in enumerate(rooms):
            room = snap["rooms"][r["team"]]
            card = room["card"]
            with cols[i % 2].container(border=True):
                st.button(f"{room['name']}  ·  {room['statusText']}", key=f"home_team_{room['id']}", on_click=go_to_team,
                          args=(room["id"],), use_container_width=True)
                st.caption(f"{room['deptName']} · {card['description']}")
                st.caption(f"👤 {card['lead']} · 팀원 {card['headcount']}명 ({card['members']})  \n"
                           f"📋 {' / '.join(card['tasks'])}" + (f"  \n🧰 {' · '.join(card['tools'])}" if card["tools"] else "")
                           + (f"  \n🕘 {room['recent']['when']} {room['recent']['status']} — {room['recent']['text']}"
                              if room["recent"] else ""))
    st.caption("1층 로비 — 📰 안내 데스크(브리핑) · 🤖 승인 데스크(자율 실행) · 💰 관리실(사용량)은 아래에 있습니다.")


def _dept_head(teams: list[Team], cfg: dict, models) -> None:
    """🧭 부서장 — 부서에 요청을 쓰면 보조 모델(1회)이 알맞은 팀을 골라 준다. 고른 뒤 그 팀 화면에 질문을 채워 이동."""
    depts = building.departments([t.id for t in teams], cfg)
    by_id = {t.id: t for t in teams}
    multi = [d for d in depts if len(d.teams) > 1] or depts
    with st.expander("🧭 부서장에게 요청 — 어느 팀에 맡길지 모를 때", expanded=DEPT_KEY in st.session_state):
        ids = [d.id for d in depts]
        if st.session_state.get(DEPT_KEY) not in ids:
            st.session_state[DEPT_KEY] = multi[0].id
        dept_id = st.selectbox("부서", ids, key=DEPT_KEY,
                               format_func=lambda i: next(f"{d.emoji} {d.name} ({len(d.teams)}팀)" for d in depts if d.id == i))
        dept = next(d for d in depts if d.id == dept_id)
        st.caption(f"{dept.description}  \n팀: " + " · ".join(by_id[t].name for t in dept.teams))
        text = st.text_area("요청", key=DEPT_TEXT_KEY, height=80, placeholder="예: 다음 달 아이랑 2박 3일 강릉 가고 싶어")
        if st.button("🧭 부서장에게 맡기기", key="dept_head_go", disabled=not text.strip(),
                     help="팀이 둘 이상이면 보조 모델 1회(아주 적은 비용)로 고릅니다. 모델을 못 쓰면 낱말 비교로 고릅니다."):
            with st.spinner("부서장이 알맞은 팀을 고르는 중…"):
                pick = building.route_request(text.strip(), dept, [by_id[t] for t in dept.teams], models)
            st.session_state[ROUTE_KEY] = {**pick, "dept": dept.id, "text": text.strip()}
            st.session_state.pop("dept_head_team", None)  # 아래 '맡길 팀'을 새 추천으로
        pick = st.session_state.get(ROUTE_KEY)
        if pick and pick["dept"] == dept.id:
            how = {"llm": f"보조 모델 {pick['model']}", "keyword": "낱말 비교", "one": "팀이 하나"}[pick["how"]]
            st.success(f"➡️ **{by_id[pick['team_id']].name}** 에 맡기면 좋겠습니다 — {pick['reason']} ({how})")
            if pick.get("note"):
                st.caption(pick["note"])
            options = dept.teams
            chosen = st.radio("맡길 팀", options, index=options.index(pick["team_id"]), horizontal=True,
                              format_func=lambda t: by_id[t].name, key="dept_head_team")
            if st.button("이 팀 화면으로 가서 질문 채우기", type="primary", key="dept_head_open"):
                st.session_state.pop(ROUTE_KEY, None)
                _navigate({"kind": "team", "id": chosen, "question": pick["text"]})
