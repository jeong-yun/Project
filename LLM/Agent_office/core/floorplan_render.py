"""평면도 그리기 — 레이아웃 JSON(core/floorplan.py) → DXF · PNG · SVG.

  DXF : ezdxf. 레이어 A-WALL(벽)·A-DOOR(문)·A-GLAZ(창)·A-ROOM(방 경계)·A-TEXT(방 이름·면적)·A-DIMS(치수).
        모델 공간에 층을 옆으로 나란히 두고, 층마다 종이 공간 시트(레이아웃 '1층', '2층' …)를 만든다.
        레이아웃 JSON 을 DXF 안(XRECORD)에 함께 넣어 두어, 다시 읽을 때 정확히 되살린다 (core/floorplan_import.py).
  PNG/SVG : matplotlib (pyplot 을 쓰지 않음 — 백그라운드 스레드에서도 안전). 층마다 한 장.
"""

import io
import json
import math
from contextlib import contextmanager

import ezdxf
from ezdxf.enums import TextEntityAlignment
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point

from core.floorplan import OUTSIDE, FloorGeometry, Layout, Opening, dump_layout, floor_geometry

XRECORD_KEY = "AGENT_OFFICE_LAYOUT"  # DXF 안에 레이아웃 JSON 을 넣어 두는 사전 이름
LAYERS = {  # 이름: (AutoCAD 색 번호, 설명)
    "A-WALL": (7, "벽"),
    "A-DOOR": (3, "문"),
    "A-GLAZ": (5, "창"),
    "A-ROOM": (8, "방 경계"),
    "A-TEXT": (2, "방 이름·면적"),
    "A-DIMS": (1, "치수"),
}
FLOOR_GAP = 6.0  # 모델 공간에서 층 사이 간격 (m)
ROOM_COLORS = {
    "living": "#FFF4D6", "dining": "#FFEBD1", "kitchen": "#FFE0C2", "entrance": "#EDEDED", "office": "#E8F0FF",
    "bedroom": "#E3F2E1", "bathroom": "#DDF0F7", "utility": "#E6E6F2", "storage": "#F0F0F0", "dressroom": "#EEE6F5",
    "balcony": "#F4F9E8", "corridor": "#F7F7F7", "stair": "#E9E2D8", "other": "#F5F5F5",
}


# ════════════════════════════════════════════════════════════════════
# DXF
# ════════════════════════════════════════════════════════════════════


def _rings(geom):
    """(Multi)Polygon → [(바깥 고리, [구멍 고리…])…]"""
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return [(list(p.exterior.coords), [list(i.coords) for i in p.interiors]) for p in polys if not p.is_empty and p.area > 0]


def _shift(points, dx: float):
    return [(x + dx, y) for x, y in points]


def _door_parts(o: Opening):
    """문 = 열린 문짝 선 + 여닫이 호. (경첩점, 문짝 끝, 시작각, 끝각, 반지름)"""
    (x1, y1), (x2, y2) = o.segment
    ang = math.atan2(y2 - y1, x2 - x1)
    # 문짝은 문이 달린 방(room) 안쪽으로 열린다고 본다 — 벽에 수직, 왼쪽(+90°) 방향
    nx, ny = -math.sin(ang), math.cos(ang)
    return (x1, y1), (x1 + nx * o.width, y1 + ny * o.width), math.degrees(ang), math.degrees(ang) + 90, o.width


def _inward(o: Opening, geom: FloorGeometry) -> Opening:
    """문짝이 문이 달린 방 쪽으로 열리도록 선분 방향을 맞춘다."""
    (x1, y1), (x2, y2) = o.segment
    ang = math.atan2(y2 - y1, x2 - x1)
    mx, my = (x1 + x2) / 2 - math.sin(ang) * 0.3, (y1 + y2) / 2 + math.cos(ang) * 0.3
    room = geom.rooms.get(o.room)
    if room is not None and not room.contains(Point(mx, my)):
        return o.model_copy(update={"segment": ((x2, y2), (x1, y1))})
    return o


def to_dxf(layout: Layout) -> bytes:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.styles.new("KOREAN", dxfattribs={"font": "malgun.ttf"})
    for name, (color, desc) in LAYERS.items():
        doc.layers.add(name, color=color).description = desc
    msp = doc.modelspace()
    width = layout.outline_polygon().bounds
    step = (width[2] - width[0]) + FLOOR_GAP
    text_h = 0.25

    sheets = []
    for i, floor in enumerate(layout.floors):
        geom = floor_geometry(layout, floor)
        dx = i * step
        # 벽: 채움(HATCH) + 윤곽선
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": "A-WALL"})
        for outer, holes in _rings(geom.walls):
            hatch.paths.add_polyline_path(_shift(outer, dx), is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL)
            msp.add_lwpolyline(_shift(outer, dx), close=True, dxfattribs={"layer": "A-WALL"})
            for hole in holes:
                hatch.paths.add_polyline_path(_shift(hole, dx), is_closed=True)
                msp.add_lwpolyline(_shift(hole, dx), close=True, dxfattribs={"layer": "A-WALL"})
        # 방 경계 + 이름·면적
        for room in floor.rooms:
            poly = geom.rooms.get(room.id)
            if poly is None:
                continue
            msp.add_lwpolyline(_shift(list(poly.exterior.coords)[:-1], dx), close=True, dxfattribs={"layer": "A-ROOM"})
            p = poly.representative_point()
            attrs = {"layer": "A-TEXT", "height": text_h, "style": "KOREAN"}
            msp.add_text(room.label, dxfattribs=attrs).set_placement((p.x + dx, p.y + 0.15), align=TextEntityAlignment.MIDDLE_CENTER)
            msp.add_text(f"{poly.area:.1f}m2", dxfattribs={**attrs, "height": text_h * 0.8}).set_placement(
                (p.x + dx, p.y - 0.25), align=TextEntityAlignment.MIDDLE_CENTER
            )
        # 문·창
        for o in geom.openings:
            if o.kind == "door":
                hinge, leaf, a0, a1, r = _door_parts(_inward(o, geom))
                msp.add_line(_shift([hinge], dx)[0], _shift([leaf], dx)[0], dxfattribs={"layer": "A-DOOR"})
                msp.add_arc((hinge[0] + dx, hinge[1]), r, a0, a1, dxfattribs={"layer": "A-DOOR"})
            else:
                (x1, y1), (x2, y2) = o.segment
                ang = math.atan2(y2 - y1, x2 - x1)
                nx, ny = -math.sin(ang), math.cos(ang)
                for k in (-0.05, 0.05):
                    msp.add_line((x1 + nx * k + dx, y1 + ny * k), (x2 + nx * k + dx, y2 + ny * k), dxfattribs={"layer": "A-GLAZ"})
        # 치수: 전체 가로·세로 + 남쪽 방 경계 연속 치수
        x0, y0, x1, y1 = geom.outline.bounds
        dim_attrs = {"layer": "A-DIMS"}
        override = {"dimtxt": text_h, "dimasz": 0.15, "dimexe": 0.1, "dimexo": 0.1, "dimtxsty": "KOREAN"}
        msp.add_linear_dim(base=(x0 + dx, y0 - 1.6), p1=(x0 + dx, y0), p2=(x1 + dx, y0), dimstyle="EZDXF",
                           override=override, dxfattribs=dim_attrs).render()
        msp.add_linear_dim(base=(x0 - 1.6 + dx, y0), p1=(x0 + dx, y0), p2=(x0 + dx, y1), angle=90, dimstyle="EZDXF",
                           override=override, dxfattribs=dim_attrs).render()
        xs = sorted({round(v, 2) for poly in geom.rooms.values() for v in (poly.bounds[0], poly.bounds[2])
                     if poly.bounds[1] <= y0 + 0.01})
        for a, b in zip(xs, xs[1:]):
            if b - a > 0.3:
                msp.add_linear_dim(base=(a + dx, y0 - 0.8), p1=(a + dx, y0), p2=(b + dx, y0), dimstyle="EZDXF",
                                   override=override, dxfattribs=dim_attrs).render()
        msp.add_text(f"{floor.label} 평면도  (단위 m, 축척 없음)", dxfattribs={"layer": "A-TEXT", "height": 0.4, "style": "KOREAN"}
                     ).set_placement((x0 + dx, y0 - 2.6), align=TextEntityAlignment.LEFT)
        sheets.append((floor.label, ((x0 + x1) / 2 + dx, (y0 + y1) / 2 - 0.8), max(y1 - y0, (x1 - x0) * 297 / 420) + 4.5))

    # 층마다 종이 공간 시트 (A3 가로)
    for label, center, view_h in sheets:
        sheet = doc.layouts.new(label)
        sheet.page_setup(size=(420, 297), margins=(10, 10, 10, 10), units="mm")
        sheet.add_viewport(center=(210, 148.5), size=(400, 277), view_center_point=center, view_height=view_h)
    if "Layout1" in doc.layouts and len(doc.layouts) > 2:
        doc.layouts.delete("Layout1")

    # 원본 레이아웃 JSON 을 DXF 안에 보관 (문자열 조각으로 나눠 XRECORD 에)
    xrec = doc.objects.add_xrecord(doc.rootdict.dxf.handle)
    text = json.dumps(json.loads(dump_layout(layout)), ensure_ascii=False, separators=(",", ":"))
    xrec.extend([(1, text[i:i + 240]) for i in range(0, len(text), 240)])
    doc.rootdict[XRECORD_KEY] = xrec

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


# ════════════════════════════════════════════════════════════════════
# PNG / SVG 미리보기
# ════════════════════════════════════════════════════════════════════


def korean_font() -> str | None:
    """설치된 한글 글꼴 이름 (없으면 None → 기본 글꼴, 한글이 네모로 보일 수 있음)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "NanumSquare", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if name in names:
            return name
    return None


@contextmanager
def _style():
    font = korean_font()
    with rc_context({"font.family": [font, "DejaVu Sans"] if font else ["DejaVu Sans"], "axes.unicode_minus": False}):
        yield


def _draw_floor(fig: Figure, layout: Layout, index: int) -> None:
    floor = layout.floors[index]
    geom = floor_geometry(layout, floor)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    for room in floor.rooms:
        poly = geom.rooms.get(room.id)
        if poly is None:
            continue
        ax.add_patch(MplPolygon(list(poly.exterior.coords), closed=True, facecolor=ROOM_COLORS.get(room.type, "#F5F5F5"),
                                edgecolor="none", zorder=1))
        p = poly.representative_point()
        bx0, by0, bx1, by1 = poly.bounds
        small = min(bx1 - bx0, by1 - by0) < 1.6
        ax.text(p.x, p.y + 0.18, room.label, ha="center", va="center", fontsize=7 if small else 9, weight="bold", zorder=5)
        ax.text(p.x, p.y - 0.22, f"{poly.area:.1f}㎡  ({bx1 - bx0:.1f}×{by1 - by0:.1f})", ha="center", va="center",
                fontsize=5.5 if small else 7, color="#444", zorder=5)
    for outer, holes in _rings(geom.walls):
        ax.add_patch(MplPolygon(outer, closed=True, facecolor="#222", edgecolor="#222", linewidth=0.3, zorder=3))
        for hole in holes:
            ax.add_patch(MplPolygon(hole, closed=True, facecolor="none", edgecolor="#222", linewidth=0.3, zorder=3))
    for o in geom.openings:
        if o.kind == "door":
            hinge, leaf, a0, a1, r = _door_parts(_inward(o, geom))
            ax.plot([hinge[0], leaf[0]], [hinge[1], leaf[1]], color="#2E7D32", linewidth=1.0, zorder=4)
            ax.add_patch(Arc(hinge, 2 * r, 2 * r, theta1=a0, theta2=a1, color="#2E7D32", linewidth=0.6, linestyle="--", zorder=4))
            if o.to == OUTSIDE:
                (x1, y1), (x2, y2) = o.segment
                ax.annotate("출입", ((x1 + x2) / 2, (y1 + y2) / 2), fontsize=6, color="#2E7D32", ha="center",
                            xytext=(0, -10), textcoords="offset points", zorder=6)
        else:
            (x1, y1), (x2, y2) = o.segment
            ax.plot([x1, x2], [y1, y2], color="#1E88E5", linewidth=3.2, solid_capstyle="butt", zorder=4)
            ax.plot([x1, x2], [y1, y2], color="white", linewidth=1.0, solid_capstyle="butt", zorder=4)

    # 치수선 (전체 가로·세로)
    x0, y0, x1, y1 = geom.outline.bounds
    off = 1.0
    ax.annotate("", (x0, y0 - off), (x1, y0 - off), arrowprops={"arrowstyle": "<->", "color": "#C62828", "lw": 0.8})
    ax.text((x0 + x1) / 2, y0 - off - 0.35, f"{x1 - x0:.2f} m", ha="center", va="top", fontsize=8, color="#C62828")
    ax.annotate("", (x0 - off, y0), (x0 - off, y1), arrowprops={"arrowstyle": "<->", "color": "#C62828", "lw": 0.8})
    ax.text(x0 - off - 0.35, (y0 + y1) / 2, f"{y1 - y0:.2f} m", ha="right", va="center", fontsize=8, color="#C62828", rotation=90)
    # 방위 표시
    ax.annotate("N", (x1 + 0.8, y1 - 0.2), (x1 + 0.8, y1 - 1.2), ha="center", fontsize=9, weight="bold",
                arrowprops={"arrowstyle": "-|>", "color": "#333"})
    ax.set_xlim(x0 - 2.4, x1 + 1.6)
    ax.set_ylim(y0 - 2.2, y1 + 1.0)
    total = sum(r.area for r in geom.rooms.values())
    title = f"{layout.title + ' · ' if layout.title else ''}{floor.label} 평면도 — 바닥 {geom.outline.area:.1f}㎡ · 방 {len(geom.rooms)}개 (실면적 합 {total:.1f}㎡)"
    ax.set_title(title, fontsize=10)
    if layout.needs_check:
        fig.text(0.01, 0.01, "⚠ 확인 필요: " + " / ".join(layout.needs_check)[:160], fontsize=6, color="#C62828")


def _figure(layout: Layout, index: int) -> Figure:
    bx = layout.outline_polygon(layout.floors[index]).bounds
    w, h = bx[2] - bx[0] + 4, bx[3] - bx[1] + 3.2
    scale = 8.5 / max(w, h)
    fig = Figure(figsize=(max(w * scale, 4), max(h * scale, 3)))
    FigureCanvasAgg(fig)
    _draw_floor(fig, layout, index)
    return fig


def to_png(layout: Layout, index: int = 0, dpi: int = 160) -> bytes:
    with _style():
        fig = _figure(layout, index)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def to_svg(layout: Layout, index: int = 0) -> bytes:
    with _style():
        fig = _figure(layout, index)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    return buf.getvalue()

