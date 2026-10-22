"""📐 설계 팀 화면 — app.py 가 부르는 설계 팀 전용 부분만 모았다 (features: [floorplan_render]).

  post_process  : 워크플로우가 끝나면(백그라운드) 최종 답변의 레이아웃 JSON → 후보안 A/B/C 배치·검증 → DXF·PNG·SVG (+요청 시 3D)
  preview       : 최종 결과 아래 '📐 도면' 상자 — 후보안 비교·채택, 층별 미리보기, 검증 결과, 3D 뷰어, 내려받기
  adopted_files : 채택 시 results/design/ 에 함께 저장할 파일 (레이아웃 JSON·DXF·PNG·SVG·3D)

그리는 일은 모두 core/floorplan*.py 가 한다 (화면 로직만 여기).
"""

import re

import streamlit as st

from core import floorplan_3d, floorplan_render
from core.floorplan import LayoutParseError, best_candidate, candidates, dump_layout, layout_from_text, metrics_line, report_markdown
from core.schemas import HumanDecision

FEATURE = "floorplan_render"
DEFAULT_SCALE = 100
CANDIDATES = 3  # 후보안 수 (좌표 없는 프로그램일 때 외곽 유형·현관 방향을 바꿔 배치)
MIME = {
    ".json": "application/json", ".dxf": "image/vnd.dxf", ".png": "image/png", ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary", ".obj": "text/plain", ".stl": "model/stl", ".ifc": "application/x-step",
}


def options_from_inputs(tool_inputs: dict) -> dict:
    """업무 양식 입력값 → 후처리 옵션 (3D 여부, 프린팅 축척 1:N)."""
    match = re.search(r"1\s*:\s*(\d+)", str(tool_inputs.get("print_scale") or ""))
    return {"want_3d": bool(tool_inputs.get("want_3d")), "print_scale": int(match.group(1)) if match else DEFAULT_SCALE}


def build_outputs(text: str, want_3d: bool = False, print_scale: int = DEFAULT_SCALE, n_candidates: int = CANDIDATES) -> dict:
    """답변 → {layout, report, pngs, files, has_3d, notes, candidates, adopted} 또는 {error}.

    candidates = 후보안 A/B/C ({label, title, layout, report, score, pngs}) — layout·report·pngs·files 는 채택한 안(adopted)의 것.
    처음에는 검증을 통과한 안 중 점수(복도 몫·장단변비·이웃 관계)가 가장 높은 안을 채택하고, 화면에서 바꿀 수 있다.
    """
    try:
        layout = layout_from_text(text)
    except LayoutParseError as e:
        return {"error": str(e)}
    if layout is None:
        return {"error": "최종 답변에 레이아웃 JSON(```json 블록, floors 포함)이 없어 도면을 그릴 수 없습니다."}
    cands = [c for c in candidates(layout, n_candidates) if not any(r.shape() is None for f in c.layout.floors for r in f.rooms)]
    if not cands:
        report = candidates(layout, 1)[0].report
        return {"error": "레이아웃을 완성하지 못해 도면을 그릴 수 없습니다: " + " / ".join(report.errors[:3]), "report": report}
    out = {"candidates": [], "has_3d": False, "notes": [], "print_scale": print_scale}
    for c in cands:
        pngs = [(floor.label, floorplan_render.to_png(c.layout, i)) for i, floor in enumerate(c.layout.floors)]
        out["candidates"].append({"label": c.label, "title": c.title, "layout": c.layout, "report": c.report,
                                  "score": c.score, "pngs": pngs})
    adopt_candidate(out, best_candidate(cands).label)
    if want_3d or out["layout"].three_d:
        add_3d(out)
    return out


def adopt_candidate(out: dict, label: str) -> None:
    """후보안 하나를 채택해 도면 파일(JSON·DXF·PNG·SVG, 3D 를 만들었으면 3D 도)을 그 안으로 다시 만든다."""
    cand = next(c for c in out["candidates"] if c["label"] == label)
    resolved = cand["layout"]
    files: dict[str, bytes] = {"layout.json": dump_layout(resolved).encode("utf-8"), "평면도.dxf": floorplan_render.to_dxf(resolved)}
    for i, (floor_label, png) in enumerate(cand["pngs"]):
        files[f"{floor_label}.png"] = png
        files[f"{floor_label}.svg"] = floorplan_render.to_svg(resolved, i)
    had_3d = out.get("has_3d")
    out.update({"adopted": label, "layout": resolved, "report": cand["report"], "pngs": cand["pngs"], "files": files})
    out.pop("figure", None)
    if had_3d:
        add_3d(out)


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
        if len(out.get("candidates") or []) > 1:
            candidate_tabs(out, rid)
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


def candidate_tabs(out: dict, rid: str) -> None:
    """후보안 A/B/C 비교 — 탭마다 평면도·설계 지표, '이 안으로 채택' 버튼."""
    cands = out["candidates"]
    st.markdown("**후보안** — 외곽 유형(일자·ㄱ자·ㄷ자·중정형)과 현관 방향을 바꿔 배치 엔진이 만든 안입니다. "
                "채택한 안으로 아래 도면과 내려받기 파일(JSON·DXF·PNG)을 만듭니다.")
    tabs = st.tabs([f"{'✅ ' if c['label'] == out['adopted'] else ''}후보안 {c['label']} · {c['title']}" for c in cands])
    for tab, c in zip(tabs, cands):
        with tab:
            rep = c["report"]
            status = "검증 통과" if rep.ok else f"검증 오류 {len(rep.errors)}건"
            st.caption(f"점수 {c['score']:g} · {status} · 경고 {len(rep.warnings)}건")
            cols = st.columns(min(len(c["pngs"]), 2))
            for i, (label, png) in enumerate(c["pngs"]):
                cols[i % len(cols)].image(png, caption=f"후보안 {c['label']} · {label}", use_container_width=True)
            for floor_label, m in rep.metrics.items():
                st.caption(metrics_line(floor_label, m))
            if c["label"] == out["adopted"]:
                st.caption("✅ 채택한 안 — 아래 도면·내려받기 파일이 이 안입니다")
            elif st.button(f"후보안 {c['label']} 채택", key=f"adopt_cand_{rid}_{c['label']}"):
                with st.spinner("채택한 안으로 도면 파일을 다시 만드는 중…"):
                    adopt_candidate(out, c["label"])
                st.rerun()


def adopted_files(run: dict, decision: HumanDecision) -> dict[str, bytes]:
    """채택 시 함께 저장할 파일. 사람이 고쳤거나 다른 안을 채택했으면 채택한 내용으로 다시 그린다."""
    out = run.get("design")
    if decision.edited or decision.adopted_node_id != run["root_id"] or not out or out.get("error"):
        opts = run.get("design_opts") or {}
        had_3d = bool(out and out.get("has_3d"))
        out = build_outputs(decision.final_answer, opts.get("want_3d", False) or had_3d, opts.get("print_scale", DEFAULT_SCALE),
                            n_candidates=1)  # 사람이 고친 안은 후보를 다시 만들지 않고 그대로 그린다
        run["design"] = out
    return dict(out.get("files") or {})
