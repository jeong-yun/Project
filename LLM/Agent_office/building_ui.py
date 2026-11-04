"""🏙 메인 화면 — 📮 접수 창구(🛎 인포메이션 센터 · 부서장) + 픽셀 빌딩 + 🗂 층별 부서 안내 (core/building.py + office_component/building.js).

  흐름 (부서장 중심)
    🛎 인포메이션 센터 (reception_ui.py) : 무엇이든 물으면 부서장 회의로 맡을 부서를 찾고 → 그 부서장 창구로 넘긴다.
                                          맡을 부서가 없으면 새 부서를 제안한다 ([만들기] / [나중에]).
    👔 부서장 창구 (_dept_desk)          : 부서를 고르면 부서장 입력창. 부서장(부장 모델 1회, 실패하면 낱말 비교 —
                                          core/building.route_request)이 하위 팀 하나를 고르고 이유·모델을 보여 준다.
                                          다른 팀으로 바꿔 보낼 수 있고, [▶ 바로 맡기기]면 그 팀 화면에서 팀장이 곧바로
                                          팀원에게 일을 나눈다 (기존 workflow — app.py 가 pop_assigned 로 받는다).
  - 빌딩: 파이썬은 배치와 방마다 상태·요약 카드만 보낸다. 그림·깜빡임·엘리베이터·낮밤은 브라우저가 그린다.
    일하는 팀이 있을 때만 3초마다 상태를 다시 보낸다 (조각 fragment). 방 = 팀 화면, 부서 띠 = 부서장 창구,
    로비 🛎 = 인포메이션 센터, 그 밖의 로비 = 아래 섹션. 누른 값은 다음 실행 맨 앞(apply_nav)에서 처리한다.
  - 🗂 층별 부서 안내: 층마다 부서 소개·부서장·소속 팀·팀별 대표 업무·인원·지금 상태 + 팀 바로 들어가기.
  - '📋 그림 없이 보기'를 켜면 빌딩 그림을 감추고 안내만 (좁은 화면·접근성용).
"""

import time
from collections.abc import Callable
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import building, dept_heads
from core.config import team_org
from core.schemas import Rank, Team

_building = components.declare_component("ao_pixel_building", path=str(Path(__file__).resolve().parent / "office_component"))

NAV_KEY = "building_nav"  # 다음 실행 맨 앞에서 처리할 이동 {kind, id, question, start, assigned}
CLICK_KEY = "building_click_t"
LIST_KEY = "building_list"
DESK_KEY = "desk_pick"  # 📮 접수 창구 — "info" 또는 부서 id
INFO = "info"
DEPT_TEXT_KEY = "dept_head_text"
ROUTE_KEY = "dept_head_route"
AUTO_ROUTE_KEY = "dept_head_auto"  # 🛎 인포메이션 센터에서 넘어온 질문 — 창구를 열자마자 부서장이 배정
MODELS_PICK_KEY = "dept_models_pick"  # 🎛 부서별 모델 — 고른 부서
MODELS_OPEN_KEY = "dept_models_open"  # 부서 띠를 누르면 펼친다
SECTION_KEY = "home_section"
AUTONOMY_OPEN_KEY = "home_autonomy_open"
SECTIONS = {"brief": "📰 AI 브리핑·아이디어", "usage": "💰 사용량·API 키·가격표"}
ASSIGNED_PREFIX = "dept_assigned_"  # 팀 화면에서 바로 시작할 요청 (부서장 배정)
NOTE_PREFIX = "dept_assigned_note_"  # 팀 화면 위에 보이는 '부서장이 배정' 안내
COST_DAYS = 7
LIVE_SEC = 3
HOW_TEXT = {"llm": "부장 모델", "keyword": "낱말 비교 (모델 없이)", "one": "팀이 하나"}


# ─── 이동 (다음 실행 맨 앞) ─────────────────────────────────────────────


def apply_nav(go_to_team: Callable[..., None]) -> None:
    """app.py 가 위젯을 그리기 전에 부른다 — 빌딩·창구에서 고른 이동을 처리."""
    nav = st.session_state.pop(NAV_KEY, None)
    if not nav:
        return
    if nav["kind"] == "team":
        if nav.get("start") and nav.get("question"):
            # ▶ 바로 맡기기 — 팀 화면이 열리면 팀장이 곧바로 팀원에게 나눈다 (app.py → pop_assigned)
            st.session_state[ASSIGNED_PREFIX + nav["id"]] = nav["question"]
        # 질문 칸에도 채워 둔다 — 그 팀이 다른 업무로 바빠 바로 시작하지 못해도 질문이 사라지지 않게
        go_to_team(nav["id"], nav.get("question") or "")
        if nav.get("assigned"):
            st.session_state[NOTE_PREFIX + nav["id"]] = {**nav["assigned"], "question": nav.get("question") or ""}
    elif nav["kind"] == "dept":
        if not nav.get("question"):  # 빌딩의 부서 띠를 누름 → 🎛 부서별 모델도 그 부서로 펼친다
            st.session_state[MODELS_OPEN_KEY] = True
        open_desk(nav["id"], nav.get("question") or "", auto=bool(nav.get("question")))
    elif nav["kind"] == "section":
        if nav["id"] == INFO:
            st.session_state[DESK_KEY] = INFO
        elif nav["id"] in SECTIONS:
            st.session_state[SECTION_KEY] = nav["id"]
        elif nav["id"] == "autonomy":
            st.session_state[AUTONOMY_OPEN_KEY] = True


def open_desk(dept_id: str, question: str = "", auto: bool = False) -> None:
    """부서장 창구 열기 (위젯을 그리기 전에만 — apply_nav · 버튼 콜백). auto = 열자마자 부서장이 배정."""
    st.session_state[DESK_KEY] = dept_id
    st.session_state[MODELS_PICK_KEY] = dept_id
    st.session_state.pop(ROUTE_KEY, None)
    if question:
        st.session_state[DEPT_TEXT_KEY] = question
    if auto and question:
        st.session_state[AUTO_ROUTE_KEY] = dept_id


def navigate(nav: dict) -> None:
    st.session_state[NAV_KEY] = nav
    st.rerun()


def pop_assigned(team_id: str, use_web: bool) -> dict:
    """부서장이 배정하고 사람이 '▶ 바로 맡기기'를 누른 요청 → 워크플로우 요청 (자유 질문, 최종 상태는 AI 가 정리). 없으면 {}."""
    text = st.session_state.pop(ASSIGNED_PREFIX + team_id, None)
    if not text:
        return {}
    return {"question": text, "brief": None, "auto_brief": True, "use_web": use_web, "tool_inputs": {},
            "extra_validators": [], "harness_options": {}}


def assignment_banner(team_id: str, run: dict | None) -> None:
    """팀 화면 위 — '👔 여행부 부장이 이 팀에 배정: 이유 (모델)'. 그 질문의 업무가 끝나 결정되면 사라진다."""
    note = st.session_state.get(NOTE_PREFIX + team_id)
    if not note:
        return
    if run is not None and run["question"] != note["question"]:
        st.session_state.pop(NOTE_PREFIX + team_id, None)
        return
    st.info(f"👔 **{note['head']}** 이(가) 이 팀에 배정했습니다 — {note['reason']}  \n"
            f"판단: {note['how']}" + (f" · 모델 {note['model']}" if note.get("model") else "")
            + (" · 사람이 다른 팀으로 바꿔 보냄" if note.get("changed") else ""))


# ─── 화면 ───────────────────────────────────────────────────────────


def snapshot(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], cfg: dict) -> dict:
    """지금 빌딩 상태 — 최근 업무(팀마다 1건)·최근 7일 비용을 저장소에서 읽는다 (DB 가 그대로면 캐시)."""
    recent = building.recent_by_team(storage.list_runs(limit=120), storage.team_for)
    costs, legacy = building.team_costs(storage.usage_by_team(time.time() - COST_DAYS * 86400), storage.team_for)
    orgs = {t.id: team_org(t, ranks) for t in teams}
    return building.snapshot(teams, orgs, ranks, cfg, open_runs=open_runs, recent=recent, costs=costs, cost_days=COST_DAYS,
                             legacy_costs=legacy)


def render(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], go_to_team: Callable[..., None],
           models=None, dept_models=None, reception=None) -> None:
    """reception = reception_ui.Ctx (없으면 🛎 인포메이션 센터를 그리지 않는다)."""
    cfg = building.load_config()
    depts = building.departments([t.id for t in teams], cfg)
    dept_store = dept_models.dept_store if dept_models is not None else None
    if models is not None:
        _desk(depts, teams, models, dept_store, reception)
    head, toggle = st.columns([4, 1])
    head.caption("층마다 부서, 방마다 팀입니다. 방 = 그 팀 화면 · 부서 띠 = 👔 부서장 창구 · 1층 🛎 = 인포메이션 센터 · "
                 "그 밖의 로비 = 아래 내용. 팀을 추가하면 층이 자동으로 늘어납니다 (config/departments.yaml).")
    as_list = toggle.toggle("📋 그림 없이 보기", key=LIST_KEY, help="빌딩 그림을 감추고 층별 부서 안내만 — 좁은 화면이나 화면 읽기 프로그램용")
    live = any(not r["done"] for r in open_runs)
    if not as_list:
        # 일하는 팀이 있을 때만 몇 초마다 이 조각만 다시 그려 상태를 보낸다 (그림은 iframe 안에서 계속 움직인다).
        if live:
            st.fragment(run_every=LIVE_SEC)(_draw)(storage, teams, ranks, open_runs, cfg)
        else:
            _draw(storage, teams, ranks, open_runs, cfg)
    _floor_guide(snapshot(storage, teams, ranks, open_runs, cfg), go_to_team, expanded=as_list)
    if dept_models is not None:
        _dept_models(dept_models)


def _dept_models(ctx) -> None:
    """🎛 부서별 모델 선택 — 부서 → 👔 부장(비평가·부서장) → 팀 → 팀원 (dept_models_ui.py)."""
    import dept_models_ui  # 메인 화면을 처음 그릴 때만

    with st.expander("🎛 부서별 모델 선택 — 👔 부장(비평가) · 팀원", expanded=bool(st.session_state.pop(MODELS_OPEN_KEY, False))):
        dept_models_ui.render_home(ctx, MODELS_PICK_KEY)


def _draw(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], cfg: dict) -> None:
    snap = snapshot(storage, teams, ranks, open_runs, cfg)
    value = _building(snapshot=snap, key="pixel_building", default=None)
    # 컴포넌트 값은 다음 실행에도 남으므로, 누른 시각(t)이 새로울 때만 '방금 누름'으로 본다
    if isinstance(value, dict) and value.get("t") != st.session_state.get(CLICK_KEY):
        st.session_state[CLICK_KEY] = value.get("t")
        if value.get("kind") in ("team", "dept", "section") and value.get("id"):
            navigate({"kind": value["kind"], "id": value["id"]})


# ─── 🗂 층별 부서 안내 ─────────────────────────────────────────────────


def _floor_guide(snap: dict, go_to_team: Callable[..., None], expanded: bool = False) -> None:
    """층(위에서부터) · 부서 카드(소개·부서장·인원·상태) · 팀 줄(대표 업무·인원·상태 + 바로 들어가기)."""
    layout = snap["layout"]
    st.markdown("#### 🗂 층별 부서 안내 — 부서마다 할 수 있는 일")
    for floor in reversed(layout["floors"]):  # 맨 위층부터 (빌딩 그림과 같은 순서)
        depts = [b["dept"] for b in floor["banners"]]
        names = " · ".join(f"{snap['depts'][d]['emoji']} {snap['depts'][d]['name']}" for d in depts)
        cost = snap["floorCosts"][layout["floors"].index(floor)]
        st.markdown(f"##### {floor['n']}층 — {names}" + (f" · 💰 최근 {COST_DAYS}일 {cost}" if cost else ""))
        cols = st.columns(min(2, len(depts)))  # 한 줄에 부서 2개까지 (좁으면 팀 줄이 찌그러진다)
        for i, dept_id in enumerate(depts):
            col = cols[i % len(cols)]
            d = snap["depts"][dept_id]
            with col.container(border=True):
                st.markdown(f"**{d['emoji']} {d['name']}** — {d['description']}")
                st.caption(f"👔 {d['head']} · {len(d['teams'])}팀 · 팀원 {d['headcount']}명 · {d['state']}")
                for t in d["teamCards"]:
                    room = snap["rooms"][t["id"]]
                    c_name, c_go = st.columns([3, 1])
                    c_name.markdown(f"**{t['name']}** · 👥 {t['headcount']}명 · {t['statusText']}  \n"
                                    f"<small>📋 {' / '.join(t['tasks'])}</small>", unsafe_allow_html=True)
                    c_go.button("들어가기", key=f"home_team_{t['id']}", on_click=go_to_team, args=(t["id"],),
                                use_container_width=True, help=f"{t['name']} 화면으로 바로 갑니다")
                    if expanded:
                        card = room["card"]
                        st.caption(f"👤 {card['lead']} · {card['members']}"
                                   + (f"  \n🧭 질문 범위: {card['scope']}" if card.get("scope") else "")
                                   + (f"  \n🧰 {' · '.join(card['tools'])}" if card["tools"] else "")
                                   + (f"  \n🕘 {room['recent']['when']} {room['recent']['status']} — {room['recent']['text']}"
                                      if room["recent"] else ""))
                st.button(f"👔 {d['name']} 부서장에게 맡기기", key=f"guide_dept_{dept_id}", on_click=open_desk, args=(dept_id,),
                          use_container_width=True)
    st.caption("1층 로비 — 🛎 인포메이션 센터(위 접수 창구) · 📰 신문 가판대(브리핑) · 🤖 승인 데스크(자율 실행) · 💰 관리실(사용량)")


# ─── 📮 접수 창구: 🛎 인포메이션 센터 · 👔 부서장 ───────────────────────────


def _desk(depts: list[building.Department], teams: list[Team], models, dept_store, reception) -> None:
    ids = [INFO, *[d.id for d in depts]] if reception is not None else [d.id for d in depts]
    if st.session_state.get(DESK_KEY) not in ids:
        st.session_state[DESK_KEY] = ids[0]
    by_dept = {d.id: d for d in depts}
    with st.container(border=True):
        st.markdown("#### 📮 접수 창구 — 어디에 맡길지 모르면 🛎 인포메이션, 부서를 알면 그 부서장에게")
        pick = st.radio("창구", ids, key=DESK_KEY, horizontal=True, label_visibility="collapsed",
                        format_func=lambda i: "🛎 인포메이션 센터" if i == INFO else f"{by_dept[i].emoji} {by_dept[i].name}")
        if pick == INFO:
            import reception_ui  # 메인 화면에서만

            reception_ui.render(reception, depts)
        else:
            _dept_desk(by_dept[pick], {t.id: t for t in teams}, models, dept_store)


def head_model(models, dept_store, dept_id: str) -> tuple[str, str]:
    """(지금 부서장이 쓸 모델, 출처 설명) — '🎛 부서별 모델 선택'에서 고른 것 → 없으면 models.yaml critic → utility."""
    choice = dept_store.get(dept_id) if dept_store is not None else ""
    chain, source = dept_heads.head_chain(models, choice)
    return chain.primary.label, dept_heads.SOURCE_LABELS.get(source, source)


def _dept_desk(dept: building.Department, by_id: dict[str, Team], models, dept_store) -> None:
    """👔 부서장 창구 — 요청을 읽고 하위 팀 하나를 고른다 → 이유·모델 → 다른 팀으로 바꾸기 → 바로 맡기기/다듬기."""
    head = dept_heads.head_of(dept)
    model, source = head_model(models, dept_store, dept.id)
    st.markdown(f"**👔 {head.name}** · {dept.emoji} {dept.name} — {dept.description}")
    st.caption("팀: " + " · ".join(f"{by_id[t].name}" + (f" ({by_id[t].scope.summary})" if by_id[t].scope else "")
                                  for t in dept.teams)
               + f"  \n🎛 부서장 모델: {model} — {source} (아래 '🎛 부서별 모델 선택'에서 바꿉니다)")
    text = st.text_area(f"{head.name}에게 요청", key=DEPT_TEXT_KEY, height=80,
                        placeholder={"travel": "예: 다음 달 아이랑 2박 3일 강릉 가고 싶어"}.get(dept.id, "무엇을 원하는지 한두 문장으로"))
    auto = st.session_state.pop(AUTO_ROUTE_KEY, None) == dept.id and text.strip()
    clicked = st.button("👔 부서장에게 맡기기", key="dept_head_go", disabled=not text.strip(),
                        help="팀이 둘 이상이면 부장 모델 1회로 고릅니다. 모델을 못 쓰면 낱말 비교로 고릅니다.")
    if clicked or auto:
        with st.spinner(f"{head.name}이(가) 요청을 읽고 알맞은 팀을 고르는 중…"):
            pick = building.route_request(text.strip(), dept, [by_id[t] for t in dept.teams], models,
                                          dept_store.get(dept.id) if dept_store is not None else "")
        st.session_state[ROUTE_KEY] = {**pick, "dept": dept.id, "text": text.strip()}
        st.session_state.pop("dept_head_team", None)  # 아래 '맡길 팀'을 새 추천으로
    pick = st.session_state.get(ROUTE_KEY)
    if not (pick and pick["dept"] == dept.id):
        return
    how = HOW_TEXT[pick["how"]]
    st.success(f"➡️ **{by_id[pick['team_id']].name}** 에 배정합니다 — {pick['reason']}")
    st.caption(f"판단: {how}" + (f" · 사용 모델 {pick['model']}" if pick.get("model") else "") + (f"  \n{pick['note']}" if pick.get("note") else ""))
    options = dept.teams
    chosen = st.radio("맡길 팀 (바꿔 보낼 수 있습니다)", options, index=options.index(pick["team_id"]), horizontal=True,
                      format_func=lambda t: by_id[t].name, key="dept_head_team")
    changed = chosen != pick["team_id"]
    if changed:
        st.caption(f"부서장 추천({by_id[pick['team_id']].name}) 대신 **{by_id[chosen].name}** 으로 보냅니다.")
    assigned = {"head": head.name, "reason": pick["reason"], "how": how, "model": pick.get("model", ""), "changed": changed}
    b1, b2 = st.columns(2)
    if b1.button(f"▶ {by_id[chosen].name}에 바로 맡기기", type="primary", key="dept_head_start", use_container_width=True,
                 help="팀 화면으로 가서 곧바로 시작합니다 — 팀장이 팀원에게 일을 나누고 결과를 모아 올립니다."):
        st.session_state.pop(ROUTE_KEY, None)
        navigate({"kind": "team", "id": chosen, "question": pick["text"], "start": True, "assigned": assigned})
    if b2.button("✏️ 팀 화면에서 질문 다듬고 맡기기", key="dept_head_open", use_container_width=True):
        st.session_state.pop(ROUTE_KEY, None)
        navigate({"kind": "team", "id": chosen, "question": pick["text"], "assigned": assigned})
