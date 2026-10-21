"""📐 설계 팀 화면 — app.py 가 부르는 설계 팀 전용 부분만 모았다 (features: [floorplan_render]).

  post_process  : 워크플로우가 끝나면(백그라운드) 최종 답변의 레이아웃 JSON → 검증 → DXF·PNG·SVG (+요청 시 3D)
  preview       : 최종 결과 아래 '📐 도면' 상자 — 층별 미리보기, 검증 결과, 3D 뷰어, 내려받기
  adopted_files : 채택 시 results/design/ 에 함께 저장할 파일 (레이아웃 JSON·DXF·PNG·SVG·3D)

그리는 일은 모두 core/floorplan*.py 가 한다 (화면 로직만 여기).
"""

import re

import streamlit as st

from core import floorplan_3d, floorplan_render
from core.floorplan import LayoutParseError, check, dump_layout, layout_from_text, report_markdown
from core.schemas import HumanDecision

FEATURE = "floorplan_render"
DEFAULT_SCALE = 100
MIME = {
    ".json": "application/json", ".dxf": "image/vnd.dxf", ".png": "image/png", ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary", ".obj": "text/plain", ".stl": "model/stl", ".ifc": "application/x-step",
}


def options_from_inputs(tool_inputs: dict) -> dict:
    """업무 양식 입력값 → 후처리 옵션 (3D 여부, 프린팅 축척 1:N)."""
    match = re.search(r"1\s*:\s*(\d+)", str(tool_inputs.get("print_scale") or ""))
    return {"want_3d": bool(tool_inputs.get("want_3d")), "print_scale": int(match.group(1)) if match else DEFAULT_SCALE}


def build_outputs(text: str, want_3d: bool = False, print_scale: int = DEFAULT_SCALE) -> dict:
    """답변 → {layout, report, pngs, files, has_3d, notes} 또는 {error}."""
    try:
        layout = layout_from_text(text)
    except LayoutParseError as e:
        return {"error": str(e)}
    if layout is None:
        return {"error": "최종 답변에 레이아웃 JSON(```json 블록, floors 포함)이 없어 도면을 그릴 수 없습니다."}
    resolved, report = check(layout)
    if any(r.shape() is None for f in resolved.floors for r in f.rooms):
        return {"error": "레이아웃을 완성하지 못해 도면을 그릴 수 없습니다: " + " / ".join(report.errors[:3]), "report": report}
    files: dict[str, bytes] = {"layout.json": dump_layout(resolved).encode("utf-8"), "평면도.dxf": floorplan_render.to_dxf(resolved)}
    pngs = []
    for i, floor in enumerate(resolved.floors):
        png = floorplan_render.to_png(resolved, i)
        pngs.append((floor.label, png))
        files[f"{floor.label}.png"] = png
        files[f"{floor.label}.svg"] = floorplan_render.to_svg(resolved, i)
    out = {"layout": resolved, "report": report, "pngs": pngs, "files": files, "has_3d": False, "notes": [],
           "print_scale": print_scale}
    if want_3d or resolved.three_d:
        add_3d(out)
    return out


def add_3d(out: dict) -> None:
    """3D 파일을 out 에 더한다 (GLB·OBJ·프린팅용 STL·IFC)."""
    layout, scale = out["layout"], out.get("print_scale") or DEFAULT_SCALE
    out["files"]["3d.glb"] = floorplan_3d.glb_bytes(layout)
    out["files"]["3d.obj"] = floorplan_3d.obj_bytes(layout)
    stl, notes, _ = floorplan_3d.stl_bytes(layout, scale)
    out["files"][f"3d_1-{scale}.stl"] = stl
    ifc = floorplan_3d.ifc_bytes(layout)
    if ifc:
        out["files"]["3d.ifc"] = ifc
    else:
        notes.append("IFC 는 ifcopenshell 이 설치돼 있을 때만 만든다 (pip install ifcopenshell)")
    out["notes"] = notes
    out["has_3d"] = True


def post_process(run: dict, final_text: str) -> None:
    """워크플로우가 끝난 뒤 (백그라운드 스레드) — 실패해도 결과·결정은 막지 않는다."""
    opts = run.get("design_opts") or {}
    try:
        run["design"] = build_outputs(final_text, opts.get("want_3d", False), opts.get("print_scale", DEFAULT_SCALE))
    except Exception as e:  # noqa: BLE001
        run["design"] = {"error": f"도면 생성 실패: {type(e).__name__}: {e}"}


def preview(run: dict, final_text: str) -> None:
    """최종 결과 아래 📐 도면 상자."""
    rid = run["run_id"]
    if run.get("post_busy"):
        st.info("📐 도면 그리는 중… (레이아웃 JSON → 검증 → DXF·PNG, 3D 요청 시 모델까지)")
        return
    out = run.get("design")
    if out is None:
        if st.button("📐 도면 그리기", key=f"mkplan_{rid}"):
            with st.spinner("도면 그리는 중…"):
                post_process(run, final_text)
            st.rerun()
        return
    with st.container(border=True):
        st.markdown("##### 📐 도면")
        if out.get("error"):
            st.warning(out["error"])
            if out.get("report"):
                st.markdown(report_markdown(out["report"]))
            return
        report = out["report"]
        layout = out["layout"]
        (st.success if report.ok else st.error)(report_markdown(report).splitlines()[0])
        if report.errors or report.warnings:
            with st.expander(f"검증 상세 (오류 {len(report.errors)} · 경고 {len(report.warnings)})", expanded=not report.ok):
                st.markdown(report_markdown(report))
        if layout.needs_check:
            st.warning("⚠️ 확인 필요: " + " / ".join(layout.needs_check))
        if layout.assumptions:
            st.caption("가정: " + " / ".join(layout.assumptions))
        labels = [label for label, _ in out["pngs"]] + (["🧊 3D"] if out["has_3d"] else [])
        tabs = st.tabs(labels)
        for tab, (label, png) in zip(tabs, out["pngs"]):
            with tab:
                st.image(png, caption=f"{label} 평면도 (치수 m, 벽 중심선 기준)", use_container_width=True)
        if out["has_3d"]:
            with tabs[-1]:
                if "figure" not in out:
                    out["figure"] = floorplan_3d.plotly_figure(layout)
                st.plotly_chart(out["figure"], use_container_width=True, key=f"plan3d_{rid}")
                st.caption("마우스로 돌리기·확대 · 지붕 없는 모형(위에서 안을 봄) · 회색 = 벽, 베이지 = 바닥판")
                for note in out["notes"]:
                    st.caption(f"🖨 {note}")
        elif st.button("🧊 3D 만들기", key=f"mk3d_{rid}", help="벽을 세운 3D 모델(GLB·OBJ·프린팅용 STL·IFC)을 만듭니다"):
            with st.spinner("3D 모델 만드는 중…"):
                add_3d(out)
            st.rerun()
        cols = st.columns(4)
        for i, (name, data) in enumerate(out["files"].items()):
            suffix = "." + name.rsplit(".", 1)[-1]
            cols[i % 4].download_button(f"⬇ {name}", data=data, file_name=f"design_{rid[:8]}_{name}",
                                        mime=MIME.get(suffix, "application/octet-stream"), key=f"dlplan_{rid}_{name}",
                                        use_container_width=True)
        st.caption("DXF 는 레이어(A-WALL·A-DOOR·A-GLAZ·A-ROOM·A-TEXT·A-DIMS)와 층별 시트로 나뉘어 있습니다. "
                   "다음에 고칠 때는 layout.json(또는 이 DXF)을 '✏️ 도면 수정·층 추가'에 첨부하세요.")


def adopted_files(run: dict, decision: HumanDecision) -> dict[str, bytes]:
    """채택 시 함께 저장할 파일. 사람이 고쳤거나 다른 안을 채택했으면 채택한 내용으로 다시 그린다."""
    out = run.get("design")
    if decision.edited or decision.adopted_node_id != run["root_id"] or not out or out.get("error"):
        opts = run.get("design_opts") or {}
        had_3d = bool(out and out.get("has_3d"))
        out = build_outputs(decision.final_answer, opts.get("want_3d", False) or had_3d, opts.get("print_scale", DEFAULT_SCALE))
        run["design"] = out
    return dict(out.get("files") or {})
