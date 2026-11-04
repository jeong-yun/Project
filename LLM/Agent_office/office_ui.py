"""🏢 픽셀 사무실 보기 — 조직도 대신 고를 수 있는 '돌아다니는 사무실' 화면.

  - 🏢 부서 업무 화면은 render_dept_office: 부서 하나 = 사무실 하나. 부서장실·회의실·자료실은 부서에 하나씩,
    팀마다 색 구역(팀장 자리 + 책상 묶음)에 그 부서의 모든 팀원이 앉는다 (core/office.build_dept_layout).
  - render_office(팀 하나만 그리는 옛 사무실)는 그대로 둔다 — 스냅샷·테스트에서 쓴다.

  - 파이썬은 지금 상태 스냅샷만 만든다 (core/office.py). 그림·걷기·✉️ 쪽지는 office_component/index.html 이
    브라우저에서 직접 그린다 → 1초마다 다시 그려도 iframe 을 새로 만들지 않아 움직임이 끊기지 않는다.
  - 캐릭터를 누르면 그 에이전트 id 를 돌려준다 → app.py 가 조직도 클릭과 똑같이 에이전트 패널을 연다.
  - 그림은 모두 코드로 그린 것이다 (외부 이미지·라이브러리 없음) — office_component/ASSETS.md
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core import office
from core.schemas import OrgNode

CHART, OFFICE = "chart", "office"
VIEW_KEY = "office_view"  # 부서·팀을 바꿔도 고른 보기를 그대로 쓴다
VIEW_LABELS = {CHART: "🗂 부서 조직도", OFFICE: "🏢 부서 사무실"}

_office = components.declare_component("ao_pixel_office", path=str(Path(__file__).resolve().parent / "office_component"))


@dataclass
class OfficeCtx:
    """사무실·커맨드 센터가 쓰는 팀 정보 (app.py 가 한 번 만들어 넘긴다)."""

    team_id: str
    team_name: str
    org: OrgNode
    names: dict[str, str]
    ranks: dict[str, tuple[str, str]]  # {에이전트 id: (직급 id, 직급 이름)}
    specialties: dict[str, str]
    parents: dict[str, str]
    blocked: dict[str, str]  # {에이전트 id: 고른 모델이 꺼진 이유}
    models: dict[str, str]  # {에이전트 id: 지금 쓸 모델 이름}
    storage: object
    runs: dict
    settings: dict
    data_dir: Path
    go_to_run: Callable[[dict], None]
    web: bool
    scope_check: Callable[[str], dict] | None = None  # 🖥️ 터미널 한 줄 업무의 범위 확인 (app.cc_scope_check)
    redirect: Callable[[str, str, dict], None] | None = None  # 범위 밖 → 맞는 팀에 맡기기 (팀 id, 질문, 판단)


def view_picker() -> str:
    st.session_state.setdefault(VIEW_KEY, OFFICE)  # 처음에는 픽셀 사무실 (조직도는 토글로)
    return st.radio("보기", [CHART, OFFICE], format_func=VIEW_LABELS.get, horizontal=True, key=VIEW_KEY,
                    label_visibility="collapsed")


def show_chart() -> None:
    """'조직도로 보기' 버튼 콜백 (위젯이 그려지기 전에 실행된다)."""
    st.session_state[VIEW_KEY] = CHART


ZONE_COLORS = ["#3B82F6", "#F59E0B", "#10B981", "#A855F7", "#EF4444", "#0EA5E9", "#EC4899", "#84CC16"]


def zone_colors(dept_color: str, team_ids: list[str]) -> dict[str, str]:
    """팀 구역 색 — 첫 팀은 부서 색, 다음 팀부터는 부서 색과 겹치지 않는 색."""
    rest = [c for c in ZONE_COLORS if c.lower() != dept_color.lower()]
    return {tid: dept_color if i == 0 else rest[(i - 1) % len(rest)] for i, tid in enumerate(team_ids)}


def render_dept_office(dept, views: list[office.TeamView], *, head_name: str, head_about: str, focus: str,
                       active_team: str) -> str | None:
    """🏢 부서 사무실 — 부서장 + 모든 팀의 팀원. 방금 누른 캐릭터가 있으면 그 키('팀 id/팀원 id' 또는 office.HEAD)."""
    colors = zone_colors(dept.color, [v.team_id for v in views])
    layout = office.build_dept_layout(dept.id, f"{dept.emoji} {dept.name}",
                                      [(v.team_id, v.team_name, colors[v.team_id], v.org) for v in views])
    snap = office.dept_snapshot(layout, views, head_name=head_name, head_about=head_about, focus=focus,
                                active_team=active_team)
    value = _office(snapshot=snap, key=f"pixel_office_dept_{dept.id}", default=None)
    last_key = f"office_click_dept_{dept.id}"
    if isinstance(value, dict) and value.get("id") in layout["seats"] and value.get("t") != st.session_state.get(last_key):
        st.session_state[last_key] = value.get("t")
        return value["id"]
    return None


def render_office(ctx: OfficeCtx, run: dict | None, active_ids: set[str] | None, focus: str) -> str | None:
    """픽셀 사무실을 그리고, 방금 누른 캐릭터가 있으면 그 에이전트 id 를 돌려준다."""
    snap = office.snapshot(
        office.build_layout(ctx.org), ctx.org, run, names=ctx.names, ranks=ctx.ranks, specialties=ctx.specialties,
        parents=ctx.parents, active_ids=active_ids, blocked=ctx.blocked, focus=focus,
    )
    value = _office(snapshot=snap, key=f"pixel_office_{ctx.team_id}", default=None)
    # 컴포넌트 값은 다음 실행에도 남으므로, 누른 시각(t)이 새로울 때만 '방금 누름'으로 본다
    last_key = f"office_click_{ctx.team_id}"
    if isinstance(value, dict) and value.get("id") in ctx.names and value.get("t") != st.session_state.get(last_key):
        st.session_state[last_key] = value.get("t")
        return value["id"]
    return None
