"""🏙 메인 화면 — 📮 접수 창구(🛎 인포메이션 센터) + 픽셀 빌딩 + 🗂 층별 부서 안내 (core/building.py + office_component/building.js),
그리고 🏢 부서 업무 화면의 👔 부서장 창구 (head_desk — app.py 가 부른다).

  흐름 (부서장 중심 — 입력창은 메인에 하나, 부서 화면에 하나)
    🛎 인포메이션 센터 (메인, reception_ui.py) : 어느 부서인지 모를 때. 부서장 회의로 맡을 부서를 찾고 → 그 부서 업무 화면의
                                                 부서장 창구로 넘긴다. 맡을 부서가 없으면 새 부서를 제안한다 ([만들기] / [나중에]).
    👔 부서장 창구 (부서 화면, head_desk)     : 부서장(부장 모델 1회, 실패하면 낱말 비교 — core/building.route_request)이 하위 팀 하나를
                                                 고르고, 범위 확인(core/scope — 낱말, 애매하면 부서장 회의 1회)으로 다른 부서 일인지 본다.
                                                 이유·모델을 보여 주고, 다른 팀으로 바꿔 보낼 수 있다. [▶ 바로 맡기기]면 그 팀의 팀장이
                                                 곧바로 팀원에게 일을 나눈다 (기존 workflow — app.py 가 pop_assigned 로 받는다).
  - 빌딩: 파이썬은 배치와 방마다 상태·요약 카드만 보낸다. 그림·깜빡임·엘리베이터·낮밤은 브라우저가 그린다.
    일하는 팀이 있을 때만 3초마다 상태를 다시 보낸다 (조각 fragment). 방 = 그 팀 업무가 골라진 부서 화면,
    부서 띠 = 부서 화면(부서장 창구), 로비 🛎 = 인포메이션 센터, 그 밖의 로비 = 아래 섹션.
    누른 값은 다음 실행 맨 앞(apply_nav)에서 처리한다.
  - 🗂 층별 부서 안내: 층마다 부서 소개·부서장·소속 팀·팀별 대표 업무·인원·지금 상태 + 부서 사무실로 들어가기.
  - '📋 그림 없이 보기'를 켜면 빌딩 그림을 감추고 안내만 (좁은 화면·접근성용).
"""

import time
from collections.abc import Callable
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import building, dept_heads
from core import scope as team_scope
from core.config import team_org
from core.schemas import Rank, Team

_building = components.declare_component("ao_pixel_building", path=str(Path(__file__).resolve().parent / "office_component"))

NAV_KEY = "building_nav"  # 다음 실행 맨 앞에서 처리할 이동 {kind, id, question, start, assigned}
CLICK_KEY = "building_click_t"
LIST_KEY = "building_list"
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
SCOPE_TEXT = {"accept_all": "범용 팀", "keyword": "분야 낱말", "followup": "후속 질문", "router": "부서장 회의(보조 모델)",
              "router_keyword": "낱말 (보조 모델 없이)"}


# ─── 이동 (다음 실행 맨 앞) ─────────────────────────────────────────────


def apply_nav(go_to_team: Callable[..., None], go_to_dept: Callable[..., None]) -> None:
    """app.py 가 위젯을 그리기 전에 부른다 — 빌딩·창구에서 고른 이동을 처리.

    go_to_team(팀 id, 질문, start) · go_to_dept(부서 id, 질문, auto) — 둘 다 🏢 부서 업무 화면을 연다."""
    nav = st.session_state.pop(NAV_KEY, None)
    if not nav:
        return
    if nav["kind"] == "team":
        start = bool(nav.get("start") and nav.get("question"))
        if start:
            # ▶ 바로 맡기기 — 부서 화면이 그 팀으로 열리면 팀장이 곧바로 팀원에게 나눈다 (app.py → pop_assigned)
            st.session_state[ASSIGNED_PREFIX + nav["id"]] = nav["question"]
        go_to_team(nav["id"], nav.get("question") or "", start)
        if nav.get("assigned"):
            st.session_state[NOTE_PREFIX + nav["id"]] = {**nav["assigned"], "question": nav.get("question") or ""}
    elif nav["kind"] == "dept":
        go_to_dept(nav["id"], nav.get("question") or "", bool(nav.get("question")))
    elif nav["kind"] == "section":
        if nav["id"] in SECTIONS:
            st.session_state[SECTION_KEY] = nav["id"]
        elif nav["id"] == "autonomy":
            st.session_state[AUTONOMY_OPEN_KEY] = True


def prepare_desk(dept_id: str, question: str = "", auto: bool = False) -> None:
    """부서장 입력칸 준비 (위젯을 그리기 전에만). auto = 부서 화면이 열리자마자 부서장이 팀을 배정."""
    st.session_state.pop(ROUTE_KEY, None)
    if question:
        st.session_state[DEPT_TEXT_KEY] = question
    if auto and question:
        st.session_state[AUTO_ROUTE_KEY] = dept_id


def open_desk(dept_id: str, question: str = "", auto: bool = False) -> None:
    """버튼 콜백 — 그 부서 업무 화면의 부서장 창구로 (다음 실행 맨 앞 apply_nav 가 화면을 바꾼다)."""
    st.session_state[NAV_KEY] = {"kind": "dept", "id": dept_id, "question": question if auto else ""}
    if question and not auto:
        st.session_state[DEPT_TEXT_KEY] = question


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


def assignment_note(team_id: str, run: dict | None) -> dict | None:
    """부서장이 이 팀에 배정한 기록 {head, reason, how, model, changed, question} — 다른 질문의 업무가 시작되면 지운다."""
    note = st.session_state.get(NOTE_PREFIX + team_id)
    if note and run is not None and run["question"] != note["question"]:
        st.session_state.pop(NOTE_PREFIX + team_id, None)
        return None
    return note


def assignment_banner(team_id: str, team_name: str, run: dict | None) -> bool:
    """부서 화면 위 — '👔 여행부 부장 → 🌏 해외 여행팀 배정: 이유 (모델)'. 그린 것이 있으면 True."""
    note = assignment_note(team_id, run)
    if not note:
        return False
    st.info(f"👔 **{note['head']}** → **{team_name}** 배정 — {note['reason']}  \n"
            f"판단: {note['how']}" + (f" · 모델 {note['model']}" if note.get("model") else "")
            + (f" · {note['scope']}" if note.get("scope") else "")
            + (" · 사람이 다른 팀으로 바꿔 보냄" if note.get("changed") else ""))
    return True


# ─── 화면 ───────────────────────────────────────────────────────────


def snapshot(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], cfg: dict) -> dict:
    """지금 빌딩 상태 — 최근 업무(팀마다 1건)·최근 7일 비용을 저장소에서 읽는다 (DB 가 그대로면 캐시)."""
    recent = building.recent_by_team(storage.list_runs(limit=120), storage.team_for)
    costs, legacy = building.team_costs(storage.usage_by_team(time.time() - COST_DAYS * 86400), storage.team_for)
    orgs = {t.id: team_org(t, ranks) for t in teams}
    return building.snapshot(teams, orgs, ranks, cfg, open_runs=open_runs, recent=recent, costs=costs, cost_days=COST_DAYS,
                             legacy_costs=legacy)


def render(storage, teams: list[Team], ranks: list[Rank], open_runs: list[dict], go_to_team: Callable[..., None],
           go_to_dept: Callable[..., None] | None = None, models=None, dept_models=None, reception=None) -> None:
    """reception = reception_ui.Ctx (없으면 🛎 인포메이션 센터를 그리지 않는다). go_to_dept = 부서 업무 화면 열기 (콜백)."""
    cfg = building.load_config()
    depts = building.departments([t.id for t in teams], cfg)
    go_to_dept = go_to_dept or (lambda dept_id, *_: open_desk(dept_id))
    if models is not None:
        _info_desk(depts, reception, go_to_dept)
    head, toggle = st.columns([4, 1])
    head.caption("층마다 부서, 방마다 팀입니다. 방·부서 띠 = 그 부서 업무 화면(부서장 창구 · 부서원 전원이 있는 사무실) · "
                 "1층 🛎 = 인포메이션 센터 · 그 밖의 로비 = 아래 내용. 팀을 추가하면 층이 자동으로 늘어납니다 (config/departments.yaml).")
    as_list = toggle.toggle("📋 그림 없이 보기", key=LIST_KEY, help="빌딩 그림을 감추고 층별 부서 안내만 — 좁은 화면이나 화면 읽기 프로그램용")
    live = any(not r["done"] for r in open_runs)
    if not as_list:
        # 일하는 팀이 있을 때만 몇 초마다 이 조각만 다시 그려 상태를 보낸다 (그림은 iframe 안에서 계속 움직인다).
        if live:
            st.fragment(run_every=LIVE_SEC)(_draw)(storage, teams, ranks, open_runs, cfg)
        else:
            _draw(storage, teams, ranks, open_runs, cfg)
    _floor_guide(snapshot(storage, teams, ranks, open_runs, cfg), go_to_team, go_to_dept, expanded=as_list)
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


def _floor_guide(snap: dict, go_to_team: Callable[..., None], go_to_dept: Callable[..., None], expanded: bool = False) -> None:
    """층(위에서부터) · 부서 카드(소개·부서장·인원·상태) · 팀 줄(대표 업무·인원·상태 + 그 팀 업무 보기) · 부서 사무실로."""
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
                st.caption(f"👔 {d['head']} · {len(d['teams'])}팀 · 부서원 {d['headcount']}명 · {d['state']}")
                for t in d["teamCards"]:
                    room = snap["rooms"][t["id"]]
                    c_name, c_go = st.columns([3, 1])
                    c_name.markdown(f"**{t['name']}** · 👥 {t['headcount']}명 · {t['statusText']}  \n"
                                    f"<small>📋 {' / '.join(t['tasks'])}</small>", unsafe_allow_html=True)
                    c_go.button("업무 보기", key=f"home_team_{t['id']}", on_click=go_to_team, args=(t["id"],),
                                use_container_width=True, help=f"{d['name']} 업무 화면에서 {t['name']} 업무를 바로 고릅니다")
                    if expanded:
                        card = room["card"]
                        st.caption(f"👤 {card['lead']} · {card['members']}"
                                   + (f"  \n🧭 질문 범위: {card['scope']}" if card.get("scope") else "")
                                   + (f"  \n🧰 {' · '.join(card['tools'])}" if card["tools"] else "")
                                   + (f"  \n🕘 {room['recent']['when']} {room['recent']['status']} — {room['recent']['text']}"
                                      if room["recent"] else ""))
                st.button(f"🏢 {d['name']} 사무실로 — 👔 부서장에게 맡기기", key=f"guide_dept_{dept_id}", on_click=go_to_dept,
                          args=(dept_id,), use_container_width=True)
    st.caption("1층 로비 — 🛎 인포메이션 센터(위 접수 창구) · 📰 신문 가판대(브리핑) · 🤖 승인 데스크(자율 실행) · 💰 관리실(사용량)")


# ─── 📮 접수 창구 (메인): 🛎 인포메이션 센터 + 부서 바로 가기 ───────────────────


def _info_desk(depts: list[building.Department], reception, go_to_dept: Callable[..., None]) -> None:
    """메인의 입력창은 🛎 인포메이션 센터 하나 — 부서를 알면 부서 버튼으로 그 부서 업무 화면(부서장 창구)에 간다."""
    with st.container(border=True):
        st.markdown("#### 📮 접수 창구 — 🛎 인포메이션 센터")
        st.caption("어느 부서에 맡길지 모르면 여기에 물어보세요 — 부서장 회의가 맡을 부서를 찾아 그 부서의 부서장에게 넘깁니다. "
                   "부서를 알면 아래 부서 버튼으로 바로 부서 사무실에 가서 부서장에게 맡기세요.")
        cols = st.columns(min(len(depts), 6) or 1)
        for i, d in enumerate(depts):
            cols[i % len(cols)].button(f"{d.emoji} {d.name}", key=f"desk_dept_{d.id}", on_click=go_to_dept, args=(d.id,),
                                       use_container_width=True, help=f"{d.name} 업무 화면 — 👔 {dept_heads.head_of(d).name}에게 맡기기")
        if reception is not None:
            import reception_ui  # 메인 화면에서만

            reception_ui.render(reception, depts)


# ─── 👔 부서장 창구 (🏢 부서 업무 화면) ───────────────────────────────────


def head_model(models, dept_store, dept_id: str) -> tuple[str, str]:
    """(지금 부서장이 쓸 모델, 출처 설명) — '🎛 부서별 모델 선택'에서 고른 것 → 없으면 models.yaml critic → utility."""
    choice = dept_store.get(dept_id) if dept_store is not None else ""
    chain, source = dept_heads.head_chain(models, choice)
    return chain.primary.label, dept_heads.SOURCE_LABELS.get(source, source)


def assign(text: str, dept: building.Department, by_id: dict[str, Team], all_teams: list[Team], models, dept_store,
           cfg: dict | None = None) -> dict:
    """부서장 배정 = ① 부서 안 팀 고르기 (building.route_request) ② 범위 확인 (core/scope — 다른 부서 일이면 알려 준다).

    {team_id, reason, how, model, note, scope, outside: {team_id, dept_id, dept_name, team_name, reason} | None}
    범위 확인은 낱말로 분명하면 모델 없이, 애매하면 부서장 회의(보조 모델) 1회 — core/scope.check."""
    pick = building.route_request(text, dept, [by_id[t] for t in dept.teams], models,
                                  dept_store.get(dept.id) if dept_store is not None else "")
    check = team_scope.check(text, by_id[pick["team_id"]], all_teams, models)
    pick = {**pick, "scope": f"범위 확인: {SCOPE_TEXT.get(check['stage'], check['stage'])}", "outside": None}
    if not check["in_scope"]:
        if check["team_id"] in dept.teams:  # 같은 부서의 다른 팀이 맞다 → 그 팀으로
            pick.update(team_id=check["team_id"], reason=f"{check['reason']} (범위 확인으로 바꿈)")
        else:
            depts = building.departments([t.id for t in all_teams], cfg if cfg is not None else building.load_config())
            other = building.dept_of(check["team_id"], depts)
            pick["outside"] = {"team_id": check["team_id"], "dept_id": other.id if other else "", "reason": check["reason"],
                               "dept_name": f"{other.emoji} {other.name}" if other else "다른 부서",
                               "team_name": next((t.name for t in all_teams if t.id == check["team_id"]), check["team_id"])}
    return pick


def head_desk(dept: building.Department, by_id: dict[str, Team], models, dept_store, all_teams: list[Team],
              go_to_dept: Callable[..., None]) -> None:
    """👔 부서장 창구 — 요청을 읽고 하위 팀 하나를 고른다 → 이유·모델 → 다른 팀으로 바꾸기 → 바로 맡기기/다듬기."""
    head = dept_heads.head_of(dept)
    model, source = head_model(models, dept_store, dept.id)
    st.caption("👔 부서장이 요청을 읽고 이 부서의 팀 하나를 골라 넘깁니다 — "
               + " · ".join(f"{by_id[t].name}" + (f" ({by_id[t].scope.summary})" if by_id[t].scope else "") for t in dept.teams)
               + f"  \n🎛 부서장 모델: {model} — {source} (아래 '🎛 부서별 모델 선택'에서 바꿉니다)")
    text = st.text_area(f"{head.name}에게 요청", key=DEPT_TEXT_KEY, height=80,
                        placeholder={"travel": "예: 다음 달 아이랑 2박 3일 강릉 가고 싶어"}.get(dept.id, "무엇을 원하는지 한두 문장으로"))
    auto = st.session_state.pop(AUTO_ROUTE_KEY, None) == dept.id and text.strip()
    clicked = st.button("👔 부서장에게 맡기기", key="dept_head_go", disabled=not text.strip(), type="primary",
                        help="팀이 둘 이상이면 부장 모델 1회로 고릅니다. 모델을 못 쓰면 낱말 비교로 고릅니다. "
                             "그다음 다른 부서 일인지 범위를 확인합니다.")
    if clicked or auto:
        with st.spinner(f"{head.name}이(가) 요청을 읽고 알맞은 팀을 고르는 중…"):
            pick = assign(text.strip(), dept, by_id, all_teams, models, dept_store)
        st.session_state[ROUTE_KEY] = {**pick, "dept": dept.id, "text": text.strip()}
        st.session_state.pop("dept_head_team", None)  # 아래 '맡길 팀'을 새 추천으로
    pick = st.session_state.get(ROUTE_KEY)
    if not (pick and pick["dept"] == dept.id):
        return
    how = HOW_TEXT[pick["how"]]
    outside = pick.get("outside")
    if outside:
        st.warning(f"🧭 이 요청은 **{outside['team_name']}**({outside['dept_name']}) 일로 보입니다 — {outside['reason']}")
        o1, o2 = st.columns(2)
        if outside["dept_id"]:
            o1.button(f"➡️ {outside['dept_name']} 부서장에게 넘기기", key="dept_head_outside", type="primary",
                      on_click=go_to_dept, args=(outside["dept_id"], pick["text"], True), use_container_width=True)
        o2.button("✏️ 요청 고치기", key="dept_head_outside_edit", on_click=lambda: st.session_state.pop(ROUTE_KEY, None),
                  use_container_width=True)
        return
    st.success(f"👔 {head.name} → **{by_id[pick['team_id']].name}** 에 배정합니다 — {pick['reason']}")
    st.caption(f"판단: {how}" + (f" · 사용 모델 {pick['model']}" if pick.get("model") else "")
               + (f" · {pick['scope']}" if pick.get("scope") else "") + (f"  \n{pick['note']}" if pick.get("note") else ""))
    options = dept.teams
    chosen = st.radio("맡길 팀 (바꿔 보낼 수 있습니다)", options, index=options.index(pick["team_id"]), horizontal=True,
                      format_func=lambda t: by_id[t].name, key="dept_head_team")
    changed = chosen != pick["team_id"]
    if changed:
        st.caption(f"부서장 추천({by_id[pick['team_id']].name}) 대신 **{by_id[chosen].name}** 으로 보냅니다.")
    assigned = {"head": head.name, "reason": pick["reason"], "how": how, "model": pick.get("model", ""), "changed": changed,
                "scope": pick.get("scope", "")}
    b1, b2 = st.columns(2)
    if b1.button(f"▶ {by_id[chosen].name}에 바로 맡기기", type="primary", key="dept_head_start", use_container_width=True,
                 help="곧바로 시작합니다 — 팀장이 팀원에게 일을 나누고 결과를 모아 올립니다. 진행은 이 부서 사무실에서 보입니다."):
        st.session_state.pop(ROUTE_KEY, None)
        navigate({"kind": "team", "id": chosen, "question": pick["text"], "start": True, "assigned": assigned})
    if b2.button("✏️ 그 팀 자유 질문으로 다듬고 맡기기", key="dept_head_open", use_container_width=True,
                 help="원하는 최종 상태(목표·완료 기준)를 직접 적고 맡깁니다."):
        st.session_state.pop(ROUTE_KEY, None)
        navigate({"kind": "team", "id": chosen, "question": pick["text"], "assigned": assigned})
