"""🏢 픽셀 사무실 보기 — 조직도 대신 고를 수 있는 '돌아다니는 사무실' 화면.

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
VIEW_KEY = "office_view"  # 팀을 바꿔도 고른 보기를 그대로 쓴다
VIEW_LABELS = {CHART: "🗂 조직도", OFFICE: "🏢 픽셀 사무실"}

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


def view_picker() -> str:
    st.session_state.setdefault(VIEW_KEY, OFFICE)  # 처음에는 픽셀 사무실 (조직도는 토글로)
    return st.radio("보기", [CHART, OFFICE], format_func=VIEW_LABELS.get, horizontal=True, key=VIEW_KEY,
                    label_visibility="collapsed")


def show_chart() -> None:
    """'조직도로 보기' 버튼 콜백 (위젯이 그려지기 전에 실행된다)."""
    st.session_state[VIEW_KEY] = CHART


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
