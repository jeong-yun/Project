"""평면도 그리기 — 레이아웃 JSON(core/floorplan.py) → DXF · PNG · SVG.

실제 건축 평면도처럼 보이도록 '도면 모델'(plan_floor)을 한 번 만들고, PNG/SVG 와 DXF 가 같은 것을 그린다.
  벽   : 실제 두께(layout.wall 외벽·내벽)의 채운 포셰(poché), 모서리는 합쳐서 깔끔하게, 문·창 자리는 뚫는다
  문   : 문짝 + 여닫이 호 — 방 안쪽으로 열리고 경첩은 가까운 모서리 쪽. 현관문은 굵게 + 출입 표시,
         발코니·드레스룸은 미닫이, 폭 1.4m 이상은 양여닫이
  창   : 벽 두께 안의 겹선 (벽면 2줄 + 유리 2줄, 넓은 창은 가운데 문틀)
  가구 : 방 종류별 (침대·옷장 / 소파·탁자·TV / 식탁·의자 / 주방 조리대·개수대·가스레인지 / 변기·세면대·욕조·샤워 /
         세탁기 / 선반 / 신발장 / 계단 디딤판·UP 화살표 / 수영장 물결 / 사우나 벤치). 방 크기에 맞춰 줄이고, 안 들어가면 뺀다.
         문 열리는 자리는 비워 둔다
  라벨 : 방 이름 + 면적 — 글자 크기를 줄이고, 줄을 나누고, 그래도 안 되면 지시선(콜아웃)으로 밖에 쓴다. 서로 겹치지 않는다
  치수 : 전체 치수 2면(남·서) + 외벽을 따라 방 경계 연속 치수 4면 (단위 mm), 방위표·축척 막대·표제란('개념·참고용 도면').
         ㄱ·ㄷ자는 꺾인 외벽마다 따로 (외접 사각형의 빈 곳이 아니라 실제 외벽을 따라), 보조선은 실제 외벽에서 시작
  외부 : 마당·데크(layout.outdoor)·중정은 실내와 다르게 — 옅은 초록(마당·중정)/나무색 줄(데크) + 점선 + '(외부)' 라벨

  DXF : ezdxf. 레이어 A-WALL(벽 포셰)·A-DOOR(문)·A-GLAZ(창)·A-ROOM(방 경계)·A-TEXT(방 이름·면적)·A-DIMS(치수)·
        A-FURN(가구·설비 블록)·A-STRS(계단 블록)·A-ANNO(방위·축척·표제란)·A-SITE(마당·데크).
        모델 공간에 층을 옆으로 나란히 두고, 층마다 종이 공간 시트(레이아웃 '1층', '2층' …, A3 표준 축척)를 만든다.
        레이아웃 JSON 을 DXF 안(XRECORD)에 함께 넣어 두어, 다시 읽을 때 정확히 되살린다 (core/floorplan_import.py).
  PNG/SVG : matplotlib (pyplot 을 쓰지 않음 — 백그라운드 스레드에서도 안전). 층마다 한 장. 1m = 같은 길이로 그려서
            글자 크기도 m 단위로 맞춘다 (라벨이 방 안에 들어가는지 계산과 그림이 같다).
"""

import datetime
import io
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import ezdxf
import numpy as np
from ezdxf.enums import TextEntityAlignment
from matplotlib import font_manager, rc_context
from matplotlib.font_manager import get_font
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ft2font import LoadFlags
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import box as shp_box
from shapely.geometry.polygon import orient

from core.floorplan import OUTSIDE, FloorGeometry, Layout, Opening, Room, dump_layout, floor_geometry
from core.floorplan import room_kind as program_kind

XRECORD_KEY = "AGENT_OFFICE_LAYOUT"  # DXF 안에 레이아웃 JSON 을 넣어 두는 사전 이름
LAYERS = {  # 이름: (AutoCAD 색 번호, 설명)
    "A-WALL": (7, "벽"),
    "A-DOOR": (3, "문"),
    "A-GLAZ": (5, "창"),
    "A-ROOM": (8, "방 경계"),
    "A-TEXT": (2, "방 이름·면적"),
    "A-DIMS": (1, "치수"),
    "A-FURN": (9, "가구·설비 (블록)"),
    "A-STRS": (4, "계단 (블록)"),
    "A-ANNO": (7, "방위·축척·표제란"),
    "A-SITE": (82, "바깥 공간 (마당·데크·중정)"),
}
FLOOR_GAP = 1.5  # 모델 공간에서 층 도면 틀 사이 간격 (m)
ROOM_COLORS = {  # 구역별 옅은 바닥색
    "living": "#FFF8E6", "dining": "#FFF3E0", "kitchen": "#FFEEDD", "entrance": "#F0F0F0", "office": "#EEF3FF",
    "bedroom": "#EEF7EC", "bathroom": "#E6F4FA", "utility": "#ECECF5", "storage": "#F3F3F3", "dressroom": "#F3EEF8",
    "balcony": "#F5FAEC", "corridor": "#FAFAFA", "stair": "#F1ECE4", "other": "#F7F7F7",
    "pool": "#F2F0EA", "sauna": "#F7EEE2", "court": "#E8F3E1",
}
OUTDOOR_FILL = {"yard": "#E8F3E1", "court": "#E8F3E1", "deck": "#F3E8D6"}  # 바깥 공간 바닥 (마당·중정 옅은 초록, 데크 나무색)
OUTDOOR_INK = "#5E7D4F"
INK = "#1A1A1A"  # 벽·문 선
FURN_INK = "#6B6B6B"  # 가구 선
WATER = "#2B7BB9"
DIM_INK = "#333333"
DIM_TIERS = (0.9, 1.6)  # 외곽(벽 중심선)에서 치수선까지 거리 (방 연속 치수, 전체 치수)
DIM_TEXT = 0.2  # 치수 글자 높이 (m)
LABEL_SIZES = (0.34, 0.30, 0.27, 0.24, 0.21, 0.19, 0.17, 0.15)  # 방 이름 글자 높이 후보 (m, 큰 것부터)
LABEL_MIN_CLEAR = 0.21  # 가구를 피해서 놓을 때 허용하는 가장 작은 글자
CALLOUT_SIZE = 0.22
PAPER = (400.0, 277.0)  # A3 가로 도면 영역 (mm, 여백 제외)
SCALES = (50, 100, 150, 200, 250, 300, 400, 500, 1000)
SCALE_BAR = 5  # 축척 막대 길이 (m)
DISCLAIMER = "개념·참고용 도면 — 시공용 아님 (건축사 검토 필요)"


# ════════════════════════════════════════════════════════════════════
# 도형 조각 (PNG·DXF 공용) — 좌표는 m
# ════════════════════════════════════════════════════════════════════


@dataclass
class Prim:
    """그릴 조각 하나. kind: poly(다각형·꺾은선) / circle / arc / text."""

    kind: str
    pts: list = field(default_factory=list)
    closed: bool = True
    center: tuple = (0.0, 0.0)
    r: float = 0.0
    a0: float = 0.0
    a1: float = 360.0
    fill: str | None = None  # PNG 채움색 (None = 선만)
    color: str = FURN_INK
    lw: float = 0.5  # 선 굵기 (pt)
    dash: bool = False
    solid: bool = False  # DXF 에서도 SOLID 해치로 채움
    text: str = ""
    h: float = 0.2  # 글자 높이 (m)
    bold: bool = False
    ha: str = "center"
    rot: float = 0.0
    dxf_color: int | None = None


def _poly(*pts, fill="white", **kw) -> Prim:
    return Prim("poly", pts=list(pts), closed=True, fill=fill, **kw)


def _line(*pts, **kw) -> Prim:
    return Prim("poly", pts=list(pts), closed=False, **kw)


def _rect(x0, y0, x1, y1, **kw) -> Prim:
    return _poly((x0, y0), (x1, y0), (x1, y1), (x0, y1), **kw)


def _circle(cx, cy, r, fill="white", **kw) -> Prim:
    return Prim("circle", center=(cx, cy), r=r, fill=fill, **kw)


def _ellipse(cx, cy, rx, ry, n=28, **kw) -> Prim:
    return _poly(*[(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)], **kw)


def _arc(cx, cy, r, a0, a1, **kw) -> Prim:
    return Prim("arc", center=(cx, cy), r=r, a0=a0, a1=a1, fill=None, **kw)


def _text(x, y, s, h, **kw) -> Prim:
    return Prim("text", center=(x, y), text=s, h=h, **kw)


def _rotate(p, theta: float):
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _transform(prim: Prim, theta: float, origin) -> Prim:
    """가구 로컬 좌표 → 도면 좌표 (회전 theta° 후 origin 만큼 이동)."""
    ox, oy = origin

    def tr(p):
        x, y = _rotate(p, theta)
        return (x + ox, y + oy)

    if prim.kind == "poly":
        return Prim(**{**prim.__dict__, "pts": [tr(p) for p in prim.pts]})
    if prim.kind == "arc":
        return Prim(**{**prim.__dict__, "center": tr(prim.center), "a0": prim.a0 + theta, "a1": prim.a1 + theta})
    return Prim(**{**prim.__dict__, "center": tr(prim.center)})


def _sample(prim: Prim, n: int = 32) -> list:
    """원·호를 꺾은선 점으로 (PNG 용)."""
    cx, cy = prim.center
    if prim.kind == "circle":
        return [(cx + prim.r * math.cos(2 * math.pi * k / n), cy + prim.r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    a0, a1 = prim.a0, prim.a1
    while a1 <= a0:
        a1 += 360
    m = max(4, int(n * (a1 - a0) / 360) + 1)
    return [(cx + prim.r * math.cos(math.radians(a0 + (a1 - a0) * k / m)),
             cy + prim.r * math.sin(math.radians(a0 + (a1 - a0) * k / m))) for k in range(m + 1)]


Box = tuple[float, float, float, float]


def _hit(a: Box, b: Box, pad: float = 0.0) -> bool:
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def _inside(a: Box, b: Box, tol: float = 1e-6) -> bool:
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


# ════════════════════════════════════════════════════════════════════
# 가구·설비 도형 (로컬 좌표: 너비 w 는 벽을 따라 u, 깊이 d 는 벽에서 방 안쪽으로 v — 등이 v=0 벽에 붙는다)
# ════════════════════════════════════════════════════════════════════


def _bed(w, d):
    out = [_rect(0, 0, w, d), _rect(0, 0, w, 0.06, fill="#DDDDDD")]
    if w >= 1.3:
        pw = (w - 0.3) / 2
        out += [_rect(0.12, 0.13, 0.12 + pw, 0.48), _rect(w - 0.12 - pw, 0.13, w - 0.12, 0.48)]
    else:
        out.append(_rect(0.15, 0.13, w - 0.15, 0.48))
    out += [_line((0, 0.62), (w, 0.62)), _line((w * 0.5, 0.62), (w, 1.05))]
    return out


def _nightstand(w, d):
    return [_rect(0, 0, w, d), _circle(w / 2, d / 2, min(w, d) * 0.25)]


def _wardrobe(w, d):
    out = [_rect(0, 0, w, d), _line((0.05, d / 2), (w - 0.05, d / 2), dash=True)]
    x = 0.15
    while x < w - 0.1:
        out.append(_line((x - 0.05, d / 2 - d * 0.32), (x + 0.05, d / 2 + d * 0.32), lw=0.35))
        x += 0.2
    return out


def _sofa(w, d):
    arm = 0.2
    out = [_rect(0, 0, w, d), _rect(0, 0, w, 0.22), _rect(0, 0.22, arm, d), _rect(w - arm, 0.22, w, d)]
    n = max(1, round((w - 2 * arm) / 0.65))
    for k in range(1, n):
        x = arm + k * (w - 2 * arm) / n
        out.append(_line((x, 0.22), (x, d)))
    return out


def _coffee(w, d):
    return [_rect(0, 0, w, d), _rect(0.06, 0.06, w - 0.06, d - 0.06, fill=None, lw=0.35)]


def _tv(w, d):
    return [_rect(0, 0, w, d), _rect(w * 0.15, 0.07, w * 0.85, 0.15, fill="#555555")]


def _dining(w, d):
    """식탁 + 의자 (긴 쪽 양옆). d 에 의자 깊이가 들어 있다."""
    out = []
    n = max(1, int(w / 0.6))
    for k in range(n):
        cx = w * (k + 0.5) / n
        for v0, v1, back in ((0.04, 0.5, (0.04, 0.12)), (d - 0.5, d - 0.04, (d - 0.12, d - 0.04))):
            out.append(_rect(cx - 0.21, v0, cx + 0.21, v1))
            out.append(_rect(cx - 0.21, back[0], cx + 0.21, back[1], fill="#DDDDDD"))
    out.append(_rect(0, 0.45, w, d - 0.45))
    return out


def _counter(w, d, sink=True, cook=True):
    out = [_rect(0, 0, w, d), _line((0, d - 0.04), (w, d - 0.04), lw=0.3)]
    if sink:
        sw = 0.7 if w >= 1.8 else 0.5
        sx = w * 0.32 if cook else w / 2
        out += [_rect(sx - sw / 2, 0.08, sx + sw / 2, 0.5), _rect(sx - sw / 2 + 0.04, 0.12, sx + sw / 2 - 0.04, 0.46, fill=None, lw=0.35),
                _circle(sx, 0.3, 0.03), _circle(sx, 0.04, 0.025, fill="#999999")]
    if cook:
        cx = w - 0.42 if sink else w / 2
        out.append(_rect(cx - 0.3, 0.05, cx + 0.3, 0.55, fill="#F2F2F2"))
        for dx, dy, r in ((-0.14, 0.17, 0.08), (0.14, 0.17, 0.07), (-0.14, 0.42, 0.07), (0.14, 0.42, 0.09)):
            out += [_circle(cx + dx, dy, r, fill=None), _circle(cx + dx, dy, r * 0.5, fill=None, lw=0.3)]
    return out


def _fridge(w, d):
    return [_rect(0, 0, w, d), _line((0.04, d - 0.04), (w - 0.04, 0.04), lw=0.35), _line((w / 2, d - 0.1), (w / 2, d), lw=0.35)]


def _toilet(w, d):
    return [_rect(0.02, 0, w - 0.02, 0.2), _ellipse(w / 2, 0.2 + (d - 0.2) / 2, w * 0.45, (d - 0.2) / 2),
            _ellipse(w / 2, 0.22 + (d - 0.2) / 2, w * 0.3, (d - 0.2) * 0.34, fill=None, lw=0.35)]


def _basin(w, d):
    return [_rect(0, 0, w, d), _ellipse(w / 2, d * 0.56, w * 0.36, d * 0.32), _circle(w / 2, 0.07, 0.025, fill="#999999")]


def _bathtub(w, d):
    return [_rect(0, 0, w, d), _poly((0.12, 0.08), (w - 0.08, 0.08), (w - 0.08, d - 0.08), (0.12, d - 0.08), (0.08, d - 0.12),
                                     (0.08, 0.12), fill="#F4FAFD", lw=0.35), _circle(w - 0.2, d / 2, 0.03)]


def _shower(w, d):
    return [_rect(0, 0, w, d, fill="#F4FAFD"), _line((0, 0), (w, d), lw=0.3), _line((0, d), (w, 0), lw=0.3),
            _circle(w / 2, d / 2, 0.04)]


def _washer(w, d):
    r = min(w, d) * 0.32
    return [_rect(0, 0, w, d), _rect(0.04, 0.03, w - 0.04, 0.1, fill="#EEEEEE"), _circle(w / 2, d / 2 + 0.04, r),
            _circle(w / 2, d / 2 + 0.04, r * 0.7, fill=None, lw=0.35)]


def _shoe(w, d):
    out = [_rect(0, 0, w, d), _line((0, 0), (w, d), lw=0.35)]
    n = max(1, round(w / 0.45))
    out += [_line((w * k / n, 0), (w * k / n, d), lw=0.3) for k in range(1, n)]
    return out


def _desk(w, d):
    return [_rect(0, 0, w, 0.6), _rect(w / 2 - 0.23, 0.68, w / 2 + 0.23, d - 0.02),
            _rect(w / 2 - 0.23, d - 0.1, w / 2 + 0.23, d - 0.02, fill="#DDDDDD")]


def _bookshelf(w, d):
    n = max(1, round(w / 0.8))
    return [_rect(0, 0, w, d)] + [_line((w * k / n, 0), (w * k / n, d), lw=0.35) for k in range(1, n)]


def _bench(w, d):
    return [_rect(0, 0, w, d, fill="#F3EAD9"), _line((0, d / 3), (w, d / 3), lw=0.3), _line((0, 2 * d / 3), (w, 2 * d / 3), lw=0.3)]


def _heater(w, d):
    return [_rect(0, 0, w, d, fill="#E4E4E4"), _line((0, 0), (w, d), lw=0.35), _line((0, d), (w, 0), lw=0.35),
            _circle(w / 2, d / 2, min(w, d) * 0.25)]


def _pool(w, d):
    """수영장 = 데크 안의 수조 + 물결 해치 + 계단."""
    deck = 0.5 if min(w, d) >= 3.5 else 0.25
    x0, y0, x1, y1 = deck, deck, w - deck, d - deck
    out = [_rect(x0, y0, x1, y1, fill="#CFE9F7", color=WATER, lw=0.8, dxf_color=5),
           _rect(x0 + 0.1, y0 + 0.1, x1 - 0.1, y1 - 0.1, fill=None, color=WATER, lw=0.35, dxf_color=5)]
    row, y = 0, y0 + 0.45
    while y < y1 - 0.3:
        x = x0 + 0.3 + (row % 2) * 0.4
        while x + 0.45 < x1 - 0.2:
            out.append(_line(*[(x + 0.45 * k / 8, y + 0.05 * math.sin(2 * math.pi * k / 8)) for k in range(9)],
                             color=WATER, lw=0.4, dxf_color=5))
            x += 0.8
        y += 0.55
        row += 1
    for k in range(1, 4):  # 들어가는 계단
        if x0 + 0.1 + k * 0.28 < x1:
            x = x0 + 0.1 + k * 0.28
            out.append(_line((x, y1 - 0.1), (x, max(y1 - 1.0, y0 + 0.1)), color=WATER, lw=0.4, dxf_color=5))
    return out


def _stair(w, d, word="UP"):
    """곧은 계단 — u 는 계단 폭, v 는 오르는 방향. 디딤판 + 절단선 + 화살표."""
    out = [_rect(0, 0, w, d, fill="white", lw=0.6)]
    tread, v = 0.27, 0.27
    cut = d * 0.62
    while v < d - 0.05:
        out.append(_line((0, v), (w, v), lw=0.4, dash=v > cut + 0.3))
        v += tread
    out.append(_line((0, cut - 0.2), (w * 0.42, cut + 0.1), (w * 0.5, cut - 0.05), (w * 0.58, cut + 0.25), (w, cut + 0.5), lw=0.5))
    tip = d - 0.25
    out += [_line((w / 2, 0.12), (w / 2, tip - 0.2), color=INK, lw=0.6),
            _poly((w / 2, tip), (w / 2 - 0.09, tip - 0.25), (w / 2 + 0.09, tip - 0.25), fill=INK, color=INK, solid=True),
            _circle(w / 2, 0.12, 0.04, fill=INK, color=INK),
            _text(w / 2 + min(0.3, w * 0.3), 0.35, word, min(0.22, w * 0.18), bold=True, color=INK)]
    return out


PIECES = {
    "bed": _bed, "nightstand": _nightstand, "wardrobe": _wardrobe, "shelf": _wardrobe, "sofa": _sofa, "coffee": _coffee,
    "tv": _tv, "dining": _dining, "counter": lambda w, d: _counter(w, d, False, False),
    "counter_sc": _counter, "counter_s": lambda w, d: _counter(w, d, True, False), "counter_c": lambda w, d: _counter(w, d, False, True),
    "fridge": _fridge, "toilet": _toilet, "basin": _basin, "bathtub": _bathtub, "shower": _shower, "washer": _washer,
    "shoe": _shoe, "desk": _desk, "bookshelf": _bookshelf, "bench": _bench, "heater": _heater, "pool": _pool,
    "stair_up": lambda w, d: _stair(w, d, "UP"), "stair_dn": lambda w, d: _stair(w, d, "DN"),
}
SOFT = {"pool", "stair_up", "stair_dn"}  # 방 전체를 채우는 기호 — 라벨은 그 위에 흰 바탕으로 쓴다


@dataclass
class Furn:
    kind: str
    room: str
    w: float
    d: float
    theta: float  # 회전 (도)
    origin: tuple  # 로컬 (0,0) 이 놓이는 도면 좌표
    box: Box  # 도면 위 차지하는 사각형

    @property
    def block(self) -> str:
        name = f"{self.kind.upper()}_{round(self.w * 1000)}x{round(self.d * 1000)}"
        return f"{name}_R{round(self.theta) % 360}" if self.kind.startswith("stair") else name

    def prims(self) -> list[Prim]:
        return PIECES[self.kind](self.w, self.d)

    def world(self) -> list[Prim]:
        return [_transform(p, self.theta, self.origin) for p in self.prims()]


# 방 종류 판정 — 종류가 'other' 여도 이름으로 수영장·사우나 등을 알아본다
NAME_KINDS = [
    ("pool", ("수영장", "풀장", "pool")), ("sauna", ("사우나", "찜질", "sauna")),
    ("bathroom", ("욕실", "화장실", "샤워")), ("bedroom", ("침실", "안방", "방")), ("kitchen", ("주방", "부엌")),
    ("dining", ("식당", "다이닝")), ("living", ("거실", "가족실")), ("office", ("서재", "공부방", "작업실")),
    ("utility", ("다용도", "세탁")), ("dressroom", ("드레스", "옷방")), ("entrance", ("현관",)),
]


def _room_text(room: Room) -> str:
    return f"{room.name} {getattr(room, 'notes', '') or ''}".lower()


def room_kind(room: Room) -> str:
    """가구를 고를 방 종류 (스키마 종류 → 없으면 이름으로 추정)."""
    if room.type != "other":
        return room.type
    if program_kind(room) in ("court", "yard", "deck"):
        return program_kind(room)
    text = _room_text(room)
    return next((kind for kind, keys in NAME_KINDS if any(k in text for k in keys)), "other")


def _wants_tub(room: Room) -> bool:
    text = _room_text(room)
    return any(k in text for k in ("욕조", "bathtub", "tub"))


# ════════════════════════════════════════════════════════════════════
# 문·창 기호
# ════════════════════════════════════════════════════════════════════


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _into(o: Opening, poly: Polygon | None, toward: bool = True):
    """개구부 선분에 수직이고 poly(방) 쪽을 향한 단위 벡터."""
    (x1, y1), (x2, y2) = o.segment
    ux, uy = _unit((x1, y1), (x2, y2))
    nx, ny = -uy, ux
    if poly is not None:
        mx, my = (x1 + x2) / 2 + nx * 0.3, (y1 + y2) / 2 + ny * 0.3
        if not poly.contains(Point(mx, my)):
            nx, ny = -nx, -ny
    return (nx, ny) if toward else (-nx, -ny)


def door_style(o: Opening, kinds: dict[str, str]) -> str:
    """swing(여닫이) / double(양여닫이) / sliding(미닫이) / open(문짝 없는 개구부 — 계단실 입구)."""
    pair = {kinds.get(o.room), kinds.get(o.to)}
    if o.to != OUTSIDE and "stair" in pair:
        return "open"
    if o.to != OUTSIDE and pair & {"balcony", "dressroom"}:
        return "sliding"
    return "double" if o.width >= 1.4 else "swing"


def _hinge_first(o: Opening, poly: Polygon | None) -> bool:
    """경첩을 선분 첫 끝에 둘지 — 방 모서리(벽 끝)에 더 가까운 끝."""
    if poly is None:
        return True
    (x1, y1), (x2, y2) = o.segment
    bx0, by0, bx1, by1 = poly.bounds
    if abs(x2 - x1) >= abs(y2 - y1):
        d1, d2 = min(abs(x1 - bx0), abs(x1 - bx1)), min(abs(x2 - bx0), abs(x2 - bx1))
    else:
        d1, d2 = min(abs(y1 - by0), abs(y1 - by1)), min(abs(y2 - by0), abs(y2 - by1))
    return d1 <= d2 + 1e-6


def _leaf(hinge, other, n, width, face, lw, color=INK) -> list[Prim]:
    """경첩 hinge 에서 n 방향으로 90° 열린 문짝 + 닫히는 쪽(other)까지의 호."""
    ax, ay = _unit(hinge, other)
    hx, hy = hinge[0] + n[0] * face, hinge[1] + n[1] * face
    t = 0.04
    leaf = _poly((hx, hy), (hx + n[0] * width, hy + n[1] * width), (hx + n[0] * width + ax * t, hy + n[1] * width + ay * t),
                 (hx + ax * t, hy + ay * t), fill="white", color=color, lw=lw)
    an, aa = math.degrees(math.atan2(n[1], n[0])), math.degrees(math.atan2(ay, ax))
    a0, a1 = (an, aa) if (aa - an) % 360 <= 180 else (aa, an)
    return [leaf, _arc(hx, hy, width, a0, a1, color=color, lw=0.45)]


def _door_prims(o: Opening, geom: FloorGeometry, kinds: dict[str, str], layout: Layout) -> list[Prim]:
    poly = geom.rooms.get(o.room)
    n = _into(o, poly)
    p1, p2 = o.segment
    entrance = o.to == OUTSIDE
    t = layout.wall.exterior if entrance else layout.wall.interior
    style = door_style(o, kinds)
    lw = 1.3 if entrance else 0.7
    out: list[Prim] = []
    if style == "open":
        return out
    if style == "sliding":
        u = _unit(p1, p2)
        length = o.width * 0.56
        for start, sign, off in ((p1, 1, 0.025), (p2, -1, -0.025)):
            sx, sy = start[0] + n[0] * off, start[1] + n[1] * off
            ex, ey = sx + u[0] * length * sign, sy + u[1] * length * sign
            out.append(_poly((sx - n[0] * 0.018, sy - n[1] * 0.018), (ex - n[0] * 0.018, ey - n[1] * 0.018),
                             (ex + n[0] * 0.018, ey + n[1] * 0.018), (sx + n[0] * 0.018, sy + n[1] * 0.018), color=INK, lw=0.6))
        out.append(_line(p1, p2, color=INK, lw=0.3, dash=True))
    elif style == "double":
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        out += _leaf(p1, mid, n, o.width / 2, t / 2, lw) + _leaf(p2, mid, n, o.width / 2, t / 2, lw)
    else:
        hinge, other = (p1, p2) if _hinge_first(o, poly) else (p2, p1)
        out += _leaf(hinge, other, n, o.width, t / 2, lw)
    if entrance:  # 출입 표시: 밖에서 안으로 향한 삼각형
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        u = _unit(p1, p2)
        tip = (mx - n[0] * (t / 2 + 0.15), my - n[1] * (t / 2 + 0.15))
        base = (mx - n[0] * (t / 2 + 0.5), my - n[1] * (t / 2 + 0.5))
        out.append(_poly(tip, (base[0] + u[0] * 0.2, base[1] + u[1] * 0.2), (base[0] - u[0] * 0.2, base[1] - u[1] * 0.2),
                         fill=INK, color=INK, solid=True))
    return out


def _window_prims(o: Opening, layout: Layout) -> list[Prim]:
    """벽 두께 안의 창 — 벽면 선 2줄 + 유리 2줄, 넓은 창은 가운데 문틀."""
    p1, p2 = o.segment
    ux, uy = _unit(p1, p2)
    nx, ny = -uy, ux
    t = layout.wall.exterior

    def off(p, k):
        return (p[0] + nx * k, p[1] + ny * k)

    out = [_poly(off(p1, -t / 2), off(p2, -t / 2), off(p2, t / 2), off(p1, t / 2), fill="white", color=INK, lw=0.5)]
    for k in (-0.025, 0.025):
        out.append(_line(off(p1, k), off(p2, k), color=INK, lw=0.45))
    if o.width >= 1.8:
        m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        out.append(_line(off(m, -t / 2), off(m, t / 2), color=INK, lw=0.45))
    return out


# ════════════════════════════════════════════════════════════════════
# 가구 배치 — 방 안(벽 안쪽 면)에서, 문이 열리는 자리를 피해, 큰 것부터 줄여 가며 맞춘다
# ════════════════════════════════════════════════════════════════════

GAP = 0.03  # 가구와 벽 사이


class _Room:
    def __init__(self, rid: str, box: Box, keep: list[Box], door_sides: set[str], window_sides: set[str]):
        self.rid, self.box, self.keep = rid, box, keep
        self.door_sides, self.window_sides = door_sides, window_sides
        self.items: list[Furn] = []

    def span(self, side: str) -> float:
        x0, y0, x1, y1 = self.box
        return (x1 - x0) if side in "SN" else (y1 - y0)

    def depth(self, side: str) -> float:
        x0, y0, x1, y1 = self.box
        return (y1 - y0) if side in "SN" else (x1 - x0)

    def sides(self, avoid_window: bool = False) -> list[str]:
        """가구를 붙일 벽 순서: 문 없는 벽 → (창 없는 벽) → 긴 벽."""
        return sorted("SNWE", key=lambda s: (s in self.door_sides, avoid_window and s in self.window_sides, -self.span(s)))

    def free(self, b: Box, front: Box | None = None) -> bool:
        if not _inside(b, self.box) or (front is not None and not _inside(front, self.box)):
            return False
        return not any(_hit(b, k) for k in self.keep) and not any(_hit(b, f.box, -1e-6) for f in self.items)

    def wall(self, kind: str, w: float, d: float, side: str, at="center", front: float = 0.0) -> Furn | None:
        """벽 side 에 등을 붙여 놓기. at: center / start / end / 벽 시작에서의 거리(m)."""
        x0, y0, x1, y1 = self.box
        lo, hi = (x0, x1) if side in "SN" else (y0, y1)
        if at == "center":
            a = (lo + hi) / 2 - w / 2
        elif at == "start":
            a = lo + GAP
        elif at == "end":
            a = hi - GAP - w
        else:
            a = lo + at
        if side == "S":
            b, fr, theta, org = (a, y0 + GAP, a + w, y0 + GAP + d), (a, y0, a + w, y0 + GAP + d + front), 0, (a, y0 + GAP)
        elif side == "N":
            b, fr, theta, org = (a, y1 - GAP - d, a + w, y1 - GAP), (a, y1 - GAP - d - front, a + w, y1), 180, (a + w, y1 - GAP)
        elif side == "W":
            b, fr, theta, org = (x0 + GAP, a, x0 + GAP + d, a + w), (x0, a, x0 + GAP + d + front, a + w), -90, (x0 + GAP, a + w)
        else:
            b, fr, theta, org = (x1 - GAP - d, a, x1 - GAP, a + w), (x1 - GAP - d - front, a, x1, a + w), 90, (x1 - GAP, a)
        if not self.free(b, fr):
            return None
        f = Furn(kind, self.rid, round(w, 3), round(d, 3), theta, org, b)
        self.items.append(f)
        return f

    def at(self, kind: str, w: float, d: float, cx: float, cy: float, vertical: bool = False, check: bool = True) -> Furn | None:
        """가운데 (cx, cy) 에 놓기. vertical 이면 긴 변(w)이 세로."""
        bw, bh = (d, w) if vertical else (w, d)
        b = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if check and not self.free(b):
            return None
        theta, org = (90, (b[2], b[1])) if vertical else (0, (b[0], b[1]))
        f = Furn(kind, self.rid, round(w, 3), round(d, 3), theta, org, b)
        self.items.append(f)
        return f

    def first(self, kind: str, sizes, sides, ats=("center", "start", "end"), front: float = 0.0) -> Furn | None:
        """크기 후보(큰 것부터) × 벽 × 위치 중 처음 들어가는 자리. 정한 위치가 다 막히면 벽을 따라 0.1m 씩 훑는다."""
        for w, d in sizes:
            for side in sides:
                room = self.span(side) - 2 * GAP - w
                if room < -1e-6:
                    continue
                sweep = [GAP + room * k / max(1, int(room / 0.1)) for k in range(int(room / 0.1) + 1)]
                for at in (*ats, *sweep):
                    if f := self.wall(kind, w, d, side, at, front):
                        return f
        return None

    def runs(self, kind: str, side: str, d: float, longest: float, shortest: float, front: float = 0.0,
             ats=("start", "end", "center")) -> Furn | None:
        """벽을 따라 긴 가구(조리대·옷장·선반) — 가장 길게 들어가는 길이."""
        length = min(longest, self.span(side) - 2 * GAP)
        while length >= shortest - 1e-6:
            for at in ats:
                if f := self.wall(kind, round(length, 2), d, side, at, front):
                    return f
            length -= 0.3
        return None


def _opp(side: str) -> str:
    return {"S": "N", "N": "S", "W": "E", "E": "W"}[side]


def _furnish_bedroom(r: _Room) -> None:
    sizes = [(1.5, 2.05), (1.35, 2.0), (1.1, 2.0), (0.95, 1.95)]
    bed = r.first("bed", sizes, r.sides(avoid_window=True), ats=("center", "start", "end"), front=0.5)
    if not bed:  # 발치 공간이 없으면 싱글 침대를 모서리에 붙이고 옆으로 드나든다 (옆 0.55m 이상)
        for w, d in sizes[2:]:
            sides = [s for s in r.sides(avoid_window=True) if r.span(s) >= w + 0.55 + 2 * GAP]
            if bed := r.first("bed", [(w, d)], sides, ats=("start", "end")):
                break
    if bed:
        side = {0: "S", 180: "N", -90: "W", 90: "E"}[bed.theta]
        lo = r.box[0] if side in "SN" else r.box[1]
        a = (bed.box[0] if side in "SN" else bed.box[1]) - lo
        for at in (a - 0.05 - 0.45, a + bed.w + 0.05):
            if at >= GAP - 1e-6:
                r.wall("nightstand", 0.45, 0.4, side, at)
    for side in r.sides(avoid_window=True):
        if r.runs("wardrobe", side, 0.6, 2.4, 0.9, front=0.6):
            break


def _furnish_living(r: _Room) -> None:
    for w, d in [(2.6, 0.9), (2.2, 0.9), (1.8, 0.85), (1.5, 0.8)]:
        for side in r.sides(avoid_window=True):
            if r.depth(side) < d + 1.6 or not (sofa := r.wall("sofa", w, d, side, "center", front=1.4)):
                continue
            opp = _opp(side)
            center = (sofa.box[0] + sofa.box[2]) / 2 if side in "SN" else (sofa.box[1] + sofa.box[3]) / 2
            lo = r.box[0] if side in "SN" else r.box[1]
            for tw in (1.8, 1.4, 1.0):
                if r.wall("tv", tw, 0.45, opp, center - lo - tw / 2):
                    break
            b = sofa.box
            for cw, cd in ((1.1, 0.55), (0.9, 0.5)):
                gap = 0.45 + cd / 2
                cx, cy = {"S": ((b[0] + b[2]) / 2, b[3] + gap), "N": ((b[0] + b[2]) / 2, b[1] - gap),
                          "W": (b[2] + gap, (b[1] + b[3]) / 2), "E": (b[0] - gap, (b[1] + b[3]) / 2)}[side]
                if r.at("coffee", cw, cd, cx, cy, vertical=side in "WE"):
                    break
            return


def _dining_set(r: _Room, sizes=((1.8, 1.8), (1.4, 1.7), (0.9, 1.6))) -> None:
    x0, y0, x1, y1 = r.box
    vertical = (y1 - y0) > (x1 - x0)
    free_x = [f.box for f in r.items]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for w, d in sizes:
        cands = [(cx, cy)]
        if free_x:  # 조리대가 있으면 남은 공간 가운데 쪽으로
            ux0, uy0 = min(b[0] for b in free_x), min(b[1] for b in free_x)
            ux1, uy1 = max(b[2] for b in free_x), max(b[3] for b in free_x)
            cands += [((ux1 + x1) / 2, cy), ((x0 + ux0) / 2, cy), (cx, (uy1 + y1) / 2), (cx, (y0 + uy0) / 2)]
        for px, py in cands:
            for vert in (vertical, not vertical):
                bw, bh = (d, w) if vert else (w, d)
                b = (px - bw / 2 - 0.3, py - bh / 2 - 0.3, px + bw / 2 + 0.3, py + bh / 2 + 0.3)  # 의자 빼는 여유
                if _inside(b, r.box) and r.free((px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)):
                    r.at("dining", w, d, px, py, vertical=vert)
                    return


def _furnish_kitchen(r: _Room, room: Room) -> None:
    """ㅡ자 조리대(개수대·가스레인지) + 모서리에 닿으면 ㄱ자로 잇는 보조 조리대 + 냉장고 (+ '식당' 이 붙으면 식탁)."""
    main = None
    for side in r.sides():
        kind = "counter_sc" if r.span(side) >= 2.1 else "counter_s"
        if main := r.runs(kind, side, 0.6, 3.6, 2.1 if kind == "counter_sc" else 1.5, front=0.9):
            break
    if main:
        side = {0: "S", 180: "N", -90: "W", 90: "E"}[main.theta]
        x0, y0, x1, y1 = r.box
        b = main.box
        if side in "SN":  # 조리대 끝이 닿은 옆벽
            touch = {"W": b[0] - x0 <= GAP + 0.01, "E": x1 - b[2] <= GAP + 0.01}
        else:
            touch = {"S": b[1] - y0 <= GAP + 0.01, "N": y1 - b[3] <= GAP + 0.01}
        kind = "counter" if main.kind == "counter_sc" else "counter_c"
        for adj in (a for a, ok in touch.items() if ok):
            length = min(2.4, r.span(adj) - main.d - 1.0)
            placed = None
            while length >= 0.9 and not placed:
                at = GAP + main.d if side in "SW" else r.span(adj) - GAP - main.d - length
                placed = r.wall(kind, round(length, 2), 0.6, adj, at, front=0.9)
                length -= 0.3
            if placed:
                break
    r.first("fridge", [(0.9, 0.75), (0.7, 0.7)], r.sides(), ats=("end", "start"), front=0.6)
    if any(k in _room_text(room) for k in ("식당", "다이닝", "dining")):
        _dining_set(r, sizes=((1.4, 1.7), (0.9, 1.6)))


def _furnish_bathroom(r: _Room, room: Room) -> None:
    placed = False
    if _wants_tub(room):
        longest = sorted("SNWE", key=lambda s: -r.span(s))
        # 좁은 욕실(폭 1.5m 안팎 — 배치 엔진이 0.3m 격자로 자주 만드는 폭)은 짧은 벽에 1.2m 소형 욕조
        placed = bool(r.first("bathtub", [(1.7, 0.75), (1.5, 0.75), (1.4, 0.7), (1.2, 0.7)], longest, ats=("start", "end")))
    if not placed:
        r.first("shower", [(0.9, 0.9), (0.8, 0.8)], r.sides(), ats=("start", "end"))
    # 변기는 옆 폭 0.75m, 세면대는 0.6m 이상인 벽에만 (좁은 틈에 끼워 넣지 않는다)
    r.first("toilet", [(0.42, 0.72), (0.38, 0.65)], [s for s in r.sides() if r.span(s) >= 0.75],
            ats=("start", "end", "center"), front=0.3)
    r.first("basin", [(0.55, 0.45), (0.45, 0.4)], [s for s in r.sides() if r.span(s) >= 0.6],
            ats=("center", "start", "end"), front=0.3)


def _furnish_sauna(r: _Room) -> None:
    benches = 0
    for side in sorted("SNWE", key=lambda s: (s in r.door_sides, -r.span(s))):
        if r.runs("bench", side, 0.55, 3.0, 0.9, front=0.4):
            benches += 1
        if benches == 2:
            break
    r.first("heater", [(0.45, 0.45)], r.sides(), ats=("end", "start"))


def furnish(kind: str, r: _Room, room: Room) -> None:
    if kind == "bedroom":
        _furnish_bedroom(r)
    elif kind == "living":
        _furnish_living(r)
    elif kind == "dining":
        _dining_set(r)
    elif kind == "kitchen":
        _furnish_kitchen(r, room)
    elif kind == "bathroom":
        _furnish_bathroom(r, room)
    elif kind == "utility":  # 세탁기 + 건조기 (같은 벽에 나란히)
        if one := r.first("washer", [(0.65, 0.65), (0.6, 0.6)], r.sides(), ats=("start", "end"), front=0.5):
            side = {0: "S", 180: "N", -90: "W", 90: "E"}[one.theta]
            a = (one.box[0] - r.box[0]) if side in "SN" else (one.box[1] - r.box[1])
            if not any(at >= 0 and r.wall("washer", one.w, one.d, side, at, front=0.5) for at in (a + one.w + 0.05, a - 0.05 - one.w)):
                r.first("washer", [(one.w, one.d)], r.sides(), front=0.5)
    elif kind == "dressroom":
        for side in sorted("SNWE", key=lambda s: (s in r.door_sides, -r.span(s)))[:2]:
            r.runs("shelf", side, 0.5, 3.0, 0.8, front=0.6)
    elif kind == "storage":
        r.runs("shelf", r.sides()[0], 0.45, 2.4, 0.8, front=0.5)
    elif kind == "entrance":
        for side in r.sides():
            if r.runs("shoe", side, 0.38, 1.8, 0.6, front=0.6):
                break
    elif kind == "office":
        r.first("desk", [(1.4, 1.2), (1.1, 1.15)], r.sides(), ats=("center", "start", "end"), front=0.2)
        r.runs("bookshelf", r.sides()[0], 0.35, 2.4, 0.8, front=0.6)
    elif kind == "sauna":
        _furnish_sauna(r)
    elif kind == "pool":
        x0, y0, x1, y1 = r.box
        for k in r.keep:  # 문이 열리는 쪽은 데크를 넓혀 문짝이 물 위로 가지 않게
            if k[1] <= y0 + 1e-6 < k[3]:
                y0 = max(y0, k[3] - 0.4)
            if k[1] < y1 - 1e-6 <= k[3]:
                y1 = min(y1, k[1] + 0.4)
            if k[0] <= x0 + 1e-6 < k[2]:
                x0 = max(x0, k[2] - 0.4)
            if k[0] < x1 - 1e-6 <= k[2]:
                x1 = min(x1, k[0] + 0.4)
        if min(x1 - x0, y1 - y0) >= 1.5:
            r.at("pool", round(x1 - x0, 3), round(y1 - y0, 3), (x0 + x1) / 2, (y0 + y1) / 2, check=False)


def _place_stair(r: _Room, geom: FloorGeometry, word: str) -> None:
    """계단: 긴 쪽이 오르는 방향, 출발점은 계단실 문에 가까운 끝."""
    x0, y0, x1, y1 = r.box
    if min(x1 - x0, y1 - y0) < 0.6:
        return
    along_x = (x1 - x0) >= (y1 - y0)
    doors = [o for o in geom.openings if o.kind == "door" and r.rid in (o.room, o.to)]
    forward = True
    if doors:
        (ax, ay), (bx, by) = doors[0].segment
        mx, my = (ax + bx) / 2, (ay + by) / 2
        forward = (mx - x0 <= x1 - mx) if along_x else (my - y0 <= y1 - my)
    w, d = (y1 - y0, x1 - x0) if along_x else (x1 - x0, y1 - y0)
    theta = (-90 if forward else 90) if along_x else (0 if forward else 180)
    org = {0: (x0, y0), 180: (x1, y1), -90: (x0, y1), 90: (x1, y0)}[theta]
    r.items.append(Furn(f"stair_{word.lower()}", r.rid, round(w, 3), round(d, 3), theta, org, (x0, y0, x1, y1)))


# ════════════════════════════════════════════════════════════════════
# 라벨 — 방 이름 + 면적, 방 안에 들어가게 줄이고·나누고, 안 되면 지시선
# ════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def korean_font() -> str | None:
    """설치된 한글 글꼴 이름 (없으면 None → 기본 글꼴, 한글이 네모로 보일 수 있음)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "NanumSquare", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if name in names:
            return name
    return None


def _families() -> list[str]:
    font = korean_font()
    return [font, "DejaVu Sans"] if font else ["DejaVu Sans"]


# 글자는 글리프 윤곽을 한 번 만들어 두고(캐시) 이어 붙여 '경로'로 그린다.
# matplotlib 글자 렌더링은 글자마다 글꼴 배치를 다시 해서 느리다 (특히 한글 글꼴 + 대체 글꼴) — 경로는 빠르고,
# 폭 계산(라벨 맞춤)과 그림이 정확히 같다.


@lru_cache(maxsize=4)
def _fonts(bold: bool):
    out = []
    for family in _families():
        font = get_font(font_manager.findfont(FontProperties(family=family, weight="bold" if bold else "normal")))
        font.set_size(100, 72)
        out.append(font)
    return out


@lru_cache(maxsize=8192)
def _glyph(ch: str, bold: bool) -> tuple[Path | None, float]:
    """글자 하나의 (윤곽 경로 — 높이 1, 앞으로 나가는 폭)."""
    fonts = _fonts(bold)
    font = next((f for f in fonts if f.get_char_index(ord(ch))), fonts[0])
    advance = font.load_char(ord(ch), flags=LoadFlags.NO_HINTING).linearHoriAdvance / 65536 / 100
    if ch.isspace():
        return None, advance
    tp = TextPath((0, 0), ch, size=1, prop=FontProperties(family=_families(), weight="bold" if bold else "normal"))
    return (Path(tp.vertices.copy(), tp.codes.copy()) if len(tp.vertices) else None), advance


@lru_cache(maxsize=4096)
def text_width(s: str, bold: bool = False) -> float:
    """글자 높이 1m 일 때 s 의 폭 (m)."""
    return sum(_glyph(ch, bold)[1] for ch in s)


MID = 0.36  # 글자 높이 1 일 때 기준선에서 글자 가운데까지 (한글·숫자 평균)


def text_path(s: str, h: float, x: float, y: float, bold: bool = False, ha: str = "center", rot: float = 0.0) -> Path | None:
    """글자열 → (x, y) 가 가운데(ha=center)/왼쪽 끝(ha=left) 인 경로 (높이 h m, rot° 회전)."""
    verts, codes, pen = [], [], 0.0
    for ch in s:
        glyph, adv = _glyph(ch, bold)
        if glyph is not None:
            verts.append(glyph.vertices + (pen, 0.0))
            codes.append(glyph.codes)
        pen += adv
    if not verts:
        return None

    v = np.concatenate(verts)
    v[:, 0] -= pen / 2 if ha == "center" else (pen if ha == "right" else 0.0)
    v[:, 1] -= MID
    tr = Affine2D().scale(h).rotate_deg(rot).translate(x, y)
    return Path(tr.transform(v), np.concatenate(codes))


LINE_PITCH = 1.3  # 줄 간격 (글자 높이 배수)


@dataclass
class Label:
    room: str
    lines: list[tuple[str, float, bool, tuple]]  # (글자, 높이 m, 굵게, 가운데 좌표)
    box: Box
    leader: tuple | None = None  # ((라벨 쪽 끝), (방 쪽 끝)) — 지시선
    halo: bool = False  # 가구 위에 겹쳐 쓰면 흰 바탕


def _wrap(name: str) -> list[str] | None:
    for sep in ("(", "·", " ", "/"):
        i = name.find(sep, 1)
        if i > 0:
            head, tail = (name[:i], name[i:]) if sep == "(" else (name[:i + (sep != " ")].strip(), name[i + 1:].strip())
            if head and tail:
                return [head, tail]
    if len(name) >= 4:
        return [name[: (len(name) + 1) // 2], name[(len(name) + 1) // 2:]]
    return None


WRAP_BELOW = 0.27  # 이보다 작아져야 하면 줄을 나눠 본다 (큰 글자 두 줄보다 조금 작은 한 줄이 낫다)


def _variants(name: str, area: str, sizes):
    """(줄들) 후보 — 한 줄을 WRAP_BELOW 까지 줄여 보고, 그다음부터 같은 크기에서 한 줄 → 두 줄."""
    one = [[(name, h, True), (area, h * 0.72, False)] for h in sizes]
    parts = _wrap(name)
    two = [[(p, h, True) for p in parts] + [(area, h * 0.72, False)] for h in sizes] if parts else []
    big = [k for k, h in enumerate(sizes) if h >= WRAP_BELOW]
    for k in big:
        yield one[k]
    for k in range(len(sizes)):
        if k not in big:
            yield one[k]
        if two:
            yield two[k]


def _block(lines) -> tuple[float, float]:
    return max(text_width(s, b) * h for s, h, b in lines), sum(h * LINE_PITCH for _, h, _ in lines)


def _layout_lines(lines, cx, cy) -> tuple[list, Box]:
    bw, bh = _block(lines)
    out, top = [], cy + bh / 2
    for s, h, b in lines:
        pitch = h * LINE_PITCH
        out.append((s, h, b, (cx, top - pitch / 2)))
        top -= pitch
    return out, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def _candidates(area: Box, bw: float, bh: float, cx: float, cy: float, n: int = 9):
    lx0, lx1, ly0, ly1 = area[0] + bw / 2, area[2] - bw / 2, area[1] + bh / 2, area[3] - bh / 2
    if lx0 > lx1 + 1e-9 or ly0 > ly1 + 1e-9:
        return []
    xs = {min(max(cx, lx0), lx1)} | {lx0 + (lx1 - lx0) * k / (n - 1) for k in range(n)}
    ys = {min(max(cy, ly0), ly1)} | {ly0 + (ly1 - ly0) * k / (n - 1) for k in range(n)}
    return sorted(((x, y) for x in xs for y in ys), key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


def _fit_label(rid: str, name: str, area: str, inner: Polygon, obstacles: list[Box], soft: list[Box]) -> Label | None:
    """방 안에 들어가는 라벨 — 가구를 피해서 → 안 되면 가구 위에 흰 바탕으로. 다 안 되면 None (지시선)."""
    ib = inner.bounds
    rect_like = inner.area >= 0.97 * (ib[2] - ib[0]) * (ib[3] - ib[1])
    c = inner.centroid if rect_like else inner.representative_point()
    for avoid in (True, False):
        sizes = [h for h in LABEL_SIZES if h >= LABEL_MIN_CLEAR] if avoid else list(LABEL_SIZES)
        for lines in _variants(name, area, sizes):
            bw, bh = _block(lines)
            for px, py in _candidates(ib, bw, bh, c.x, c.y, n=9 if avoid else 3):
                b = (px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)
                if not rect_like and not inner.contains(shp_box(*b)):
                    continue
                if avoid and any(_hit(b, o) for o in obstacles):
                    continue
                placed, b = _layout_lines(lines, px, py)
                return Label(rid, placed, b, halo=any(_hit(b, o) for o in obstacles + soft))
    return None


def _callout(rid: str, name: str, area: str, target, spots: list[tuple[Polygon, Box]], taken: list[Box],
             outline: Polygon, wall: float) -> Label:
    """방 안에 안 들어가는 라벨 — 가까운 빈 곳(다른 방 안·건물 밖)에 쓰고 지시선으로 잇는다."""
    lines = [(name, CALLOUT_SIZE, True), (area, CALLOUT_SIZE * 0.72, False)]
    bw, bh = _block(lines)
    tx, ty = target
    best = None
    for inner, ib in spots:
        for px, py in _candidates(ib, bw, bh, tx, ty, n=12)[:60]:
            b = (px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)
            if any(_hit(b, t, 0.06) for t in taken) or not inner.contains(shp_box(*b)):
                continue
            dist = (px - tx) ** 2 + (py - ty) ** 2
            if best is None or dist < best[0]:
                best = (dist, px, py)
            break
    if best is None:  # 건물 북쪽 바깥 띠 (치수선 안쪽)
        _, _, _, oy1 = outline.bounds
        px, py = tx, oy1 + wall / 2 + 0.12 + bh / 2
        while any(_hit((px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2), t, 0.06) for t in taken):
            px += bw + 0.1
        best = (0, px, py)
    _, px, py = best
    placed, b = _layout_lines(lines, px, py)
    dx, dy = tx - px, ty - py
    k = min((bw / 2 + 0.04) / abs(dx) if dx else math.inf, (bh / 2 + 0.04) / abs(dy) if dy else math.inf, 1.0)
    return Label(rid, placed, b, leader=((px + dx * k, py + dy * k), (tx, ty)))


# ════════════════════════════════════════════════════════════════════
# 도면 모델 (한 층)
# ════════════════════════════════════════════════════════════════════


@dataclass
class Chain:
    side: str  # S/N/W/E
    tier: float  # 외벽(edge)에서 치수선까지 (m)
    coords: list[float]  # 치수점 (S/N 은 x, W/E 는 y)
    edge: float = 0.0  # 치수선을 띄우는 기준 외벽 좌표 (S/N 은 y, W/E 는 x) — ㄱ·ㄷ자는 꺾인 외벽마다 다르다
    faces: list[float] = field(default_factory=list)  # 치수점마다 보조선이 시작하는 실제 외벽 좌표


@dataclass
class FloorPlan:
    layout: Layout
    index: int
    geom: FloorGeometry
    interiors: dict[str, Polygon]
    kinds: dict[str, str]
    furniture: list[Furn]
    doors: list[list[Prim]]
    windows: list[list[Prim]]
    labels: list[Label]
    chains: list[Chain]
    anno: list[Prim]  # 방위표·축척 막대·표제란
    frame: Box  # 도면 전체 범위 (m)
    scale: int  # 1:scale (A3)
    outdoor: list[Prim] = field(default_factory=list)  # 마당·데크 (맨 아래층만)

    @property
    def floor(self):
        return self.geom.floor


def _interior(poly: Polygon, walls) -> Polygon:
    inner = poly.difference(walls)
    if hasattr(inner, "geoms"):
        inner = max(inner.geoms, key=lambda g: g.area, default=poly)
    return inner if not inner.is_empty and inner.area > 0.05 else poly


def _keepouts(rid: str, inner_box: Box, geom: FloorGeometry, kinds: dict[str, str]) -> tuple[list[Box], set, set]:
    """문이 열리는 자리(문짝 반경)·지나가는 자리(0.6m) + 문·창이 있는 벽."""
    x0, y0, x1, y1 = inner_box
    keep, door_sides, window_sides = [], set(), set()
    for o in geom.openings:
        if rid not in (o.room, o.to):
            continue
        (ax, ay), (bx, by) = o.segment
        horizontal = abs(bx - ax) >= abs(by - ay)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        side = ("S" if abs(my - y0) <= abs(my - y1) else "N") if horizontal else ("W" if abs(mx - x0) <= abs(mx - x1) else "E")
        if o.kind == "window":
            window_sides.add(side)
            continue
        door_sides.add(side)
        style = door_style(o, kinds)
        if o.room == rid and style in ("swing", "double"):  # 이 방으로 열리는 문짝 반경
            depth = (o.width / 2 if style == "double" else o.width) + 0.05
        else:  # 지나가는 자리
            depth = 0.6
        lo, hi = (min(ax, bx), max(ax, bx)) if horizontal else (min(ay, by), max(ay, by))
        if side == "S":
            keep.append((lo - 0.05, y0 - 1, hi + 0.05, y0 + depth))
        elif side == "N":
            keep.append((lo - 0.05, y1 - depth, hi + 0.05, y1 + 1))
        elif side == "W":
            keep.append((x0 - 1, lo - 0.05, x0 + depth, hi + 0.05))
        else:
            keep.append((x1 - depth, lo - 0.05, x1 + 1, hi + 0.05))
    return keep, door_sides, window_sides


def _facade_runs(outline: Polygon, side: str) -> dict[float, list[tuple[float, float]]]:
    """그 방향(S/N/W/E)을 바라보는 외벽 선분들 — {외벽 좌표: [(시작, 끝)…]} (ㄱ·ㄷ자는 좌표가 여럿)."""
    runs: dict[float, list[tuple[float, float]]] = {}
    coords = list(outline.exterior.coords)
    for (ax, ay), (bx, by) in zip(coords, coords[1:]):
        horizontal = abs(ay - by) < 1e-6
        if horizontal != (side in "SN") or (abs(ax - bx) < 1e-6 and abs(ay - by) < 1e-6):
            continue
        if horizontal:
            probe = Point((ax + bx) / 2, ay + (0.05 if side == "N" else -0.05))
            key, span = round(ay, 3), (min(ax, bx), max(ax, bx))
        else:
            probe = Point(ax + (0.05 if side == "E" else -0.05), (ay + by) / 2)
            key, span = round(ax, 3), (min(ay, by), max(ay, by))
        if not outline.contains(probe):  # 바깥을 바라보는 벽
            runs.setdefault(key, []).append(span)
    return runs


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _face(outline: Polygon, side: str, v: float) -> float:
    """치수점 v 에서 그 방향의 가장 바깥 외벽 좌표 (보조선 시작점)."""
    x0, y0, x1, y1 = outline.bounds
    best = None
    for d in (-0.01, 0.01):
        if side in "SN":
            c = _clamp(v + d, x0 + 0.005, x1 - 0.005)
            line = LineString([(c, y0 - 1), (c, y1 + 1)])
        else:
            c = _clamp(v + d, y0 + 0.005, y1 - 0.005)
            line = LineString([(x0 - 1, c), (x1 + 1, c)])
        cut = line.intersection(outline)
        if cut.is_empty:
            continue
        b = cut.bounds
        val = {"S": b[1], "N": b[3], "W": b[0], "E": b[2]}[side]
        if best is None or (val < best if side in "SW" else val > best):
            best = val
    return best if best is not None else {"S": y0, "N": y1, "W": x0, "E": x1}[side]


def _chains(geom: FloorGeometry) -> list[Chain]:
    """치수 연속선 — 꺾인 외벽(run)마다 그 외벽에서 띄워 방 경계를 잰다 + 남·서 전체 치수 (보조선은 실제 외벽에서)."""
    x0, y0, x1, y1 = geom.outline.bounds
    out = []
    for side in "SWNE":
        horizontal = side in "SN"
        lo, hi = (x0, x1) if horizontal else (y0, y1)
        for edge, spans in _facade_runs(geom.outline, side).items():
            pts = {v for sp in spans for v in sp}
            for poly in geom.rooms.values():
                bx0, by0, bx1, by1 = poly.bounds
                near = {"S": by0, "N": by1, "W": bx0, "E": bx1}[side]
                if abs(near - edge) < 0.05:
                    for v in ((bx0, bx1) if horizontal else (by0, by1)):
                        if any(a - 0.01 <= v <= b + 0.01 for a, b in spans):
                            pts.add(v)
            merged = []
            for v in sorted(pts):
                if not merged or v - merged[-1] > 0.05:
                    merged.append(v)
            if len(merged) > 2:
                out.append(Chain(side, DIM_TIERS[0], merged, edge, [edge] * len(merged)))
        if side in "SW":
            outer = {"S": y0, "W": x0}[side]
            out.append(Chain(side, DIM_TIERS[1], [lo, hi], outer, [_face(geom.outline, side, lo), _face(geom.outline, side, hi)]))
    return out


def _outdoor_prims(layout: Layout) -> list[Prim]:
    """마당·데크 (layout.outdoor) — 실내와 다른 바닥색 + 점선 테두리 + 데크 널 줄 + '이름 (외부)' 라벨."""
    out: list[Prim] = []
    for room in layout.outdoor or []:
        poly = room.shape()
        if poly is None or poly.area < 0.5:
            continue
        kind = program_kind(room)
        pts = list(poly.exterior.coords)[:-1]
        fill = OUTDOOR_FILL.get(kind, OUTDOOR_FILL["yard"])
        out.append(_poly(*pts, fill=fill, color=fill, lw=0.1, dxf_color=82))
        out.append(_line(*pts, pts[0], color=OUTDOOR_INK, lw=0.7, dash=True, dxf_color=82))  # 점선 테두리
        bx0, by0, bx1, by1 = poly.bounds
        if kind == "deck":  # 데크 널 (0.6m 간격 가는 선)
            k = 1
            while by0 + 0.6 * k < by1 - 0.1:
                seg = LineString([(bx0, by0 + 0.6 * k), (bx1, by0 + 0.6 * k)]).intersection(poly)
                for g in getattr(seg, "geoms", [seg]):
                    if not g.is_empty and g.length > 0.2:
                        out.append(_line(*list(g.coords), color="#C9B48F", lw=0.35, dxf_color=42))
                k += 1
        c = poly.representative_point()
        h = _clamp(min(bx1 - bx0, by1 - by0) / 6, 0.17, 0.34)
        out.append(_text(c.x, c.y + h * 0.6, f"{room.label} (외부)", h, bold=True, color=OUTDOOR_INK))
        out.append(_text(c.x, c.y - h * 0.6, f"{poly.area:.1f}㎡", h * 0.72, color=OUTDOOR_INK))
    return out


def paper_scale(width: float, height: float) -> int:
    """A3 가로에 들어가는 가장 큰 표준 축척 (1:scale)."""
    for s in SCALES:
        if width * 1000 / s <= PAPER[0] and height * 1000 / s <= PAPER[1]:
            return s
    return SCALES[-1]


def _annotations(layout: Layout, geom: FloorGeometry, title: str, sub: str, frame_core: Box) -> tuple[list[Prim], Box]:
    """방위표(오른쪽 위)·축척 막대(왼쪽 아래)·표제란(오른쪽 아래)과 그것까지 포함한 도면 범위."""
    x0, y0, x1, y1 = geom.outline.bounds
    t = layout.wall.exterior
    out: list[Prim] = []
    # 방위표
    nx, ny, r = x1 + 1.25, y1 + 1.35, 0.42
    out += [_circle(nx, ny, r, fill=None, color=INK, lw=0.6),
            _poly((nx, ny + r * 0.95), (nx - r * 0.45, ny - r * 0.7), (nx, ny - r * 0.3), fill=INK, color=INK, solid=True),
            _poly((nx, ny + r * 0.95), (nx + r * 0.45, ny - r * 0.7), (nx, ny - r * 0.3), fill="white", color=INK, lw=0.5),
            _text(nx, ny + r + 0.25, "N", 0.3, bold=True, color=INK)]
    # 표제란
    stamp = datetime.date.today().isoformat()
    rows = [(title, 0.34, True, INK), (sub, 0.22, False, INK), ("", 0.2, False, INK), (DISCLAIMER, 0.19, True, "#B71C1C")]
    tb_w = max(6.0, max(text_width(s, b) * h for s, h, b, _ in rows if s) + 0.6)
    row_h = [0.62, 0.44, 0.42, 0.42]
    tb_x1 = x1 + t / 2
    tb_x0 = tb_x1 - tb_w
    tb_y1 = y0 - DIM_TIERS[1] - 0.75
    # 축척 막대 — 표제란과 겹치면 표제란을 아래로
    sb_x0, sb_y = x0 - t / 2, tb_y1 - 0.25
    if sb_x0 + SCALE_BAR + 0.8 > tb_x0:
        tb_y1 = sb_y - 0.75
    tb_y0 = tb_y1 - sum(row_h)
    out.append(_rect(tb_x0, tb_y0, tb_x1, tb_y1, fill="white", color=INK, lw=0.8))
    y = tb_y1
    for (s, h, b, color), rh in zip(rows, row_h):
        if y < tb_y1:
            out.append(_line((tb_x0, y), (tb_x1, y), color=INK, lw=0.4))
        if s:
            out.append(_text(tb_x0 + 0.25, y - rh / 2, s, h, bold=b, color=color, ha="left"))
        y -= rh
    scale_row = tb_y1 - row_h[0] - row_h[1] - row_h[2] / 2
    out.append(_text(tb_x0 + 0.25, scale_row, "{SCALE}" + f" · 단위 mm · {stamp}", 0.2, color=INK, ha="left"))
    for k in range(SCALE_BAR):
        out.append(_rect(sb_x0 + k, sb_y - 0.14, sb_x0 + k + 1, sb_y, fill=INK if k % 2 == 0 else "white", color=INK, lw=0.5,
                         solid=k % 2 == 0))
    for v in (0, 1, 2, SCALE_BAR):
        out.append(_text(sb_x0 + v, sb_y - 0.34, f"{v}" + ("m" if v == SCALE_BAR else ""), 0.17, color=INK))
    fx0, fy0, fx1, fy1 = frame_core
    frame = (min(fx0, sb_x0 - 0.3, tb_x0 - 0.2), min(fy0, tb_y0 - 0.3), max(fx1, nx + r + 0.3, tb_x1 + 0.2), max(fy1, ny + r + 0.6))
    return out, frame


def plan_floor(layout: Layout, index: int = 0) -> FloorPlan:
    """한 층의 도면 모델 — 벽·문·창·가구·라벨·치수·주석 (PNG·SVG·DXF 가 같이 쓴다)."""
    floor = layout.floors[index]
    geom = floor_geometry(layout, floor)
    rooms = {r.id: r for r in floor.rooms}
    kinds = {rid: room_kind(r) for rid, r in rooms.items()}
    interiors = {rid: _interior(poly, geom.walls_full) for rid, poly in geom.rooms.items()}
    word = "DN" if len(layout.floors) > 1 and index == len(layout.floors) - 1 else "UP"

    furniture: list[Furn] = []
    ctxs: dict[str, _Room] = {}
    for rid, inner in interiors.items():
        ib = inner.bounds
        rect_like = inner.area >= 0.97 * (ib[2] - ib[0]) * (ib[3] - ib[1])
        keep, door_sides, window_sides = _keepouts(rid, ib, geom, kinds)
        ctx = ctxs[rid] = _Room(rid, ib, keep, door_sides, window_sides)
        if not rect_like:
            continue  # 직사각형이 아닌 방은 가구를 넣지 않는다 (라벨만)
        if kinds[rid] == "stair":
            _place_stair(ctx, geom, word)
        else:
            furnish(kinds[rid], ctx, rooms[rid])
        furniture += ctx.items

    doors = [_door_prims(o, geom, kinds, layout) for o in geom.openings if o.kind == "door"]
    windows = [_window_prims(o, layout) for o in geom.openings if o.kind == "window"]

    # 라벨: 방 안 → 안 되면 지시선 (방 안 라벨을 모두 놓은 뒤)
    labels: list[Label] = []
    pending = []
    for room in floor.rooms:
        inner = interiors.get(room.id)
        if inner is None:
            continue
        ctx = ctxs[room.id]
        area = f"{geom.rooms[room.id].area:.1f}㎡"
        solid = [f.box for f in ctx.items if f.kind not in SOFT] + ctx.keep
        soft = [f.box for f in ctx.items if f.kind in SOFT]
        shrunk = inner.buffer(-0.06, join_style="mitre")
        name = room.label + (" (외부)" if kinds.get(room.id) == "court" else "")
        lab = _fit_label(room.id, name, area, shrunk if not shrunk.is_empty else inner, solid, soft)
        if lab:
            labels.append(lab)
        else:
            pending.append((room, area))
    if pending:
        spots = []
        for rid, inner in interiors.items():
            shrunk = inner.buffer(-0.06, join_style="mitre")
            if not shrunk.is_empty:
                spots.append((shrunk, shrunk.bounds))
        blocked = [f.box for f in furniture if f.kind not in SOFT] + [k for c in ctxs.values() for k in c.keep]
        for room, area in pending:
            c = interiors[room.id].representative_point()
            taken = [lab.box for lab in labels] + blocked
            labels.append(_callout(room.id, room.label, area, (c.x, c.y), spots, taken, geom.outline, layout.wall.exterior))

    chains = _chains(geom)
    outdoor = _outdoor_prims(layout) if floor.level == min(f.level for f in layout.floors) else []
    x0, y0, x1, y1 = geom.outline.bounds
    for p in outdoor:  # 건물 앞 마당 등 외곽 밖 바깥 공간도 도면 틀 안에
        if p.kind == "poly" and p.closed:
            x0, y0 = min(x0, *(q[0] for q in p.pts)), min(y0, *(q[1] for q in p.pts))
            x1, y1 = max(x1, *(q[0] for q in p.pts)), max(y1, *(q[1] for q in p.pts))
    t = layout.wall.exterior
    core = (x0 - DIM_TIERS[1] - 0.75, y0 - DIM_TIERS[1] - 0.6, x1 + DIM_TIERS[0] + 0.7, y1 + DIM_TIERS[0] + 0.7)
    total = sum(p.area for p in geom.rooms.values())
    title = f"{layout.title + ' · ' if layout.title else ''}{floor.label} 평면도"
    sub = (f"바닥 {geom.outline.area:.1f}㎡ · 방 {len(geom.rooms)}개 (실면적 합 {total:.1f}㎡) · "
           f"벽 외 {t * 1000:.0f} / 내 {layout.wall.interior * 1000:.0f}mm")
    anno, frame = _annotations(layout, geom, title, sub, core)
    scale = paper_scale(frame[2] - frame[0], frame[3] - frame[1])
    for p in anno:
        if p.kind == "text" and "{SCALE}" in p.text:
            p.text = p.text.replace("{SCALE}", f"축척 1:{scale} (A3)")
    return FloorPlan(layout, index, geom, interiors, kinds, furniture, doors, windows, labels, chains, anno, frame, scale, outdoor)


def _dim_prims(plan: FloorPlan) -> list[Prim]:
    """치수 연속선 (PNG 용) — 보조선·45° 눈금·mm 글자. 좁은 칸은 글자를 위로 엇갈려 올린다."""
    t = plan.layout.wall.exterior
    out: list[Prim] = []
    for ch in plan.chains:
        horizontal = ch.side in "SN"
        sign = -1 if ch.side in "SW" else 1
        line = ch.edge + sign * ch.tier

        def pt(a, b):
            return (a, b) if horizontal else (b, a)

        out.append(_line(pt(ch.coords[0] - 0.15, line), pt(ch.coords[-1] + 0.15, line), color=DIM_INK, lw=0.45))
        for v, f in zip(ch.coords, ch.faces or [ch.edge] * len(ch.coords)):
            out.append(_line(pt(v, f + sign * (t / 2 + 0.12)), pt(v, line + sign * 0.12), color=DIM_INK, lw=0.3))
            out.append(_line(pt(v - 0.08, line - 0.08), pt(v + 0.08, line + 0.08), color=DIM_INK, lw=0.9))
        lift = False
        for a, b in zip(ch.coords, ch.coords[1:]):
            s = f"{(b - a) * 1000:.0f}"
            narrow = text_width(s) * DIM_TEXT > (b - a) - 0.12
            lift = narrow and not lift
            off = 0.06 + DIM_TEXT * 0.5 + (DIM_TEXT * 1.1 if lift else 0)
            # 글자는 치수선 '위'(가로선은 +y, 세로선은 -x 쪽)
            pos = ((a + b) / 2, line + off) if horizontal else (line - off, (a + b) / 2)
            out.append(_text(*pos, s, DIM_TEXT, color=DIM_INK, rot=0 if horizontal else 90))
    return out


# ════════════════════════════════════════════════════════════════════
# DXF
# ════════════════════════════════════════════════════════════════════


def _rings(geom):
    """(Multi)Polygon → [(바깥 고리, [구멍 고리…])…]"""
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return [(list(p.exterior.coords), [list(i.coords) for i in p.interiors]) for p in polys if not p.is_empty and p.area > 0]


def _shift(points, dx: float):
    return [(x + dx, y) for x, y in points]


_ALIGN = {"center": TextEntityAlignment.MIDDLE_CENTER, "left": TextEntityAlignment.MIDDLE_LEFT, "right": TextEntityAlignment.MIDDLE_RIGHT}


def _dxf_prim(space, p: Prim, layer: str, dx: float = 0.0, text_rot: float = 0.0) -> None:
    attrs = {"layer": layer}
    if p.dxf_color is not None:
        attrs["color"] = p.dxf_color
    if p.kind == "poly":
        pts = _shift(p.pts, dx)
        space.add_lwpolyline(pts, close=p.closed, dxfattribs=attrs)
        if p.solid and p.closed:
            hatch = space.add_hatch(color=p.dxf_color or 7, dxfattribs={"layer": layer})
            hatch.paths.add_polyline_path(pts, is_closed=True)
    elif p.kind == "circle":
        space.add_circle((p.center[0] + dx, p.center[1]), p.r, dxfattribs=attrs)
    elif p.kind == "arc":
        space.add_arc((p.center[0] + dx, p.center[1]), p.r, p.a0, p.a1, dxfattribs=attrs)
    elif p.kind == "text":
        space.add_text(p.text, dxfattribs={**attrs, "height": p.h * 0.75, "style": "KOREAN", "rotation": p.rot + text_rot}
                       ).set_placement((p.center[0] + dx, p.center[1]), align=_ALIGN.get(p.ha, TextEntityAlignment.MIDDLE_CENTER))


def _dxf_block(doc, f: Furn) -> str:
    name = f.block
    if name not in doc.blocks:
        blk = doc.blocks.new(name)
        for p in f.prims():
            _dxf_prim(blk, p, "0", text_rot=-f.theta)  # 글자는 끼워 넣은 뒤에도 똑바로 서게
    return name


def to_dxf(layout: Layout) -> bytes:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.styles.new("KOREAN", dxfattribs={"font": "malgun.ttf"})
    for name, (color, desc) in LAYERS.items():
        doc.layers.add(name, color=color).description = desc
    msp = doc.modelspace()
    plans = [plan_floor(layout, i) for i in range(len(layout.floors))]
    step = max(p.frame[2] - p.frame[0] for p in plans) + FLOOR_GAP

    sheets = []
    for i, plan in enumerate(plans):
        geom, floor = plan.geom, plan.floor
        dx = i * step
        for p in plan.outdoor:  # 마당·데크
            _dxf_prim(msp, p, "A-SITE", dx)
        # 벽: 포셰 채움(HATCH) + 윤곽선
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": "A-WALL"})
        for outer, holes in _rings(geom.walls):
            hatch.paths.add_polyline_path(_shift(outer, dx), is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL)
            msp.add_lwpolyline(_shift(outer, dx), close=True, dxfattribs={"layer": "A-WALL"})
            for hole in holes:
                hatch.paths.add_polyline_path(_shift(hole, dx), is_closed=True)
                msp.add_lwpolyline(_shift(hole, dx), close=True, dxfattribs={"layer": "A-WALL"})
        # 방 경계 + 이름·면적 (방 안 라벨을 먼저 — 가져오기가 방 안 첫 글자를 이름으로 읽는다)
        for room in floor.rooms:
            poly = geom.rooms.get(room.id)
            if poly is not None:
                msp.add_lwpolyline(_shift(list(poly.exterior.coords)[:-1], dx), close=True, dxfattribs={"layer": "A-ROOM"})
        for lab in sorted(plan.labels, key=lambda lb: lb.leader is not None):
            for s, h, bold, (cx, cy) in lab.lines:
                text = s.replace("㎡", "m2")
                msp.add_text(text, dxfattribs={"layer": "A-TEXT", "height": h * 0.75, "style": "KOREAN"}
                             ).set_placement((cx + dx, cy), align=TextEntityAlignment.MIDDLE_CENTER)
            if lab.leader:
                (ax, ay), (bx, by) = lab.leader
                msp.add_line((ax + dx, ay), (bx + dx, by), dxfattribs={"layer": "A-TEXT"})
                msp.add_circle((bx + dx, by), 0.04, dxfattribs={"layer": "A-TEXT"})
        # 문·창
        for prims in plan.doors:
            for p in prims:
                _dxf_prim(msp, p, "A-DOOR", dx)
        for prims in plan.windows:
            for p in prims:
                _dxf_prim(msp, p, "A-GLAZ", dx)
        # 가구·설비·계단 (블록)
        for f in plan.furniture:
            layer = "A-STRS" if f.kind.startswith("stair") else "A-FURN"
            msp.add_blockref(_dxf_block(doc, f), (f.origin[0] + dx, f.origin[1]), dxfattribs={"layer": layer, "rotation": f.theta})
        # 치수 (mm 로 표시, 건축 눈금)
        x0, y0, x1, y1 = geom.outline.bounds
        t = layout.wall.exterior
        override = {"dimtxt": DIM_TEXT * 0.75, "dimasz": 0.12, "dimblk": "ARCHTICK", "dimexe": 0.12, "dimexo": t / 2 + 0.12,
                    "dimlfac": 1000, "dimdec": 0, "dimtad": 1, "dimgap": 0.05, "dimtxsty": "KOREAN", "dimtih": 0, "dimtoh": 0}
        for ch in plan.chains:
            edge = ch.edge
            sign = -1 if ch.side in "SW" else 1
            faces = ch.faces or [edge] * len(ch.coords)
            if ch.side in "SN":
                pts = [(v + dx, f) for v, f in zip(ch.coords, faces)]
                base_pt = (ch.coords[0] + dx, edge + sign * ch.tier)
                angle = 0
            else:
                pts = [(f + dx, v) for v, f in zip(ch.coords, faces)]
                base_pt = (edge + sign * ch.tier + dx, ch.coords[0])
                angle = 90
            msp.add_multi_point_linear_dim(base=base_pt, points=pts, angle=angle, dimstyle="EZDXF", override=override,
                                           dxfattribs={"layer": "A-DIMS"})
        # 방위표·축척 막대·표제란
        for p in plan.anno:
            _dxf_prim(msp, p, "A-ANNO", dx)
        fx0, fy0, fx1, fy1 = plan.frame
        sheets.append((floor.label, ((fx0 + fx1) / 2 + dx, (fy0 + fy1) / 2), PAPER[1] * plan.scale / 1000))

    # 층마다 종이 공간 시트 (A3 가로, 표준 축척)
    for label, center, view_h in sheets:
        sheet = doc.layouts.new(label)
        sheet.page_setup(size=(420, 297), margins=(10, 10, 10, 10), units="mm")
        sheet.add_viewport(center=(210, 148.5), size=PAPER, view_center_point=center, view_height=view_h)
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
# PNG / SVG
# ════════════════════════════════════════════════════════════════════


@contextmanager
def _style():
    with rc_context({"font.family": _families(), "axes.unicode_minus": False, "svg.fonttype": "path"}):
        yield


def _path(geom) -> Path:
    """구멍 있는 (Multi)Polygon → matplotlib 복합 경로."""
    verts, codes = [], []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in polys:
        if poly.is_empty or not isinstance(poly, Polygon):
            continue
        poly = orient(poly, 1.0)
        for ring in [poly.exterior, *poly.interiors]:
            pts = list(ring.coords)
            verts += pts
            codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes) if verts else Path([(0, 0)], [Path.MOVETO])


def _draw_texts(ax, items, z: float, gid: str | None = None) -> None:
    """[(Path, 색)] → 색별 PathCollection 하나씩."""
    by_color: dict[str, list] = {}
    for path, color in items:
        if path is not None:
            by_color.setdefault(color, []).append(path)
    for color, paths in by_color.items():
        ax.add_collection(PathCollection(paths, facecolors=color, edgecolors="none", linewidths=0, zorder=z, gid=gid))


def _draw_prims(ax, prims: list[Prim], z: float) -> None:
    polys, pstyle, lines, lstyle, texts = [], [], [], [], []
    for p in prims:
        if p.kind == "text":
            texts.append((text_path(p.text, p.h, *p.center, bold=p.bold, ha=p.ha, rot=p.rot), p.color))
        elif p.kind == "arc" or (p.kind == "poly" and not p.closed):
            lines.append(_sample(p) if p.kind == "arc" else p.pts)
            lstyle.append((p.color, p.lw, (0, (3, 2)) if p.dash else "solid"))
        else:
            polys.append(_sample(p) if p.kind == "circle" else p.pts)
            pstyle.append((p.fill or "none", p.color, p.lw))
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors=[s[0] for s in pstyle], edgecolors=[s[1] for s in pstyle],
                                         linewidths=[s[2] for s in pstyle], zorder=z, joinstyle="miter"))
    if lines:
        ax.add_collection(LineCollection(lines, colors=[s[0] for s in lstyle], linewidths=[s[1] for s in lstyle],
                                         linestyles=[s[2] for s in lstyle], zorder=z + 0.2))
    _draw_texts(ax, texts, z + 0.5)


def label_paths(lab: Label) -> list[Path]:
    """라벨 글자의 실제 윤곽 경로 (그림과 같은 것 — 테스트에서 겹침 확인용)."""
    return [path for s, h, bold, (cx, cy) in lab.lines if (path := text_path(s, h, cx, cy, bold=bold))]


FIG_LONG_SIDE = 11.0  # 그림 긴 변 (inch) — 틀(frame)에 여백이 있어 bbox_inches="tight" 없이 한 번만 그린다


def _figure(layout: Layout, index: int) -> Figure:
    plan = plan_floor(layout, index)
    fx0, fy0, fx1, fy1 = plan.frame
    if layout.needs_check:
        fy0 -= 0.6
    w, h = fx1 - fx0, fy1 - fy0
    k = FIG_LONG_SIDE / max(w, h)  # inch / m
    fig = Figure(figsize=(w * k, h * k))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(fx0, fx1)
    ax.set_ylim(fy0, fy1)
    ax.set_aspect("equal")
    ax.axis("off")
    geom = plan.geom
    _draw_prims(ax, plan.outdoor, 0.5)  # 마당·데크 (바깥)
    # 바닥 (구역별 옅은 색)
    fills = [(list(poly.exterior.coords), ROOM_COLORS.get(plan.kinds.get(rid, "other"), "#F7F7F7")) for rid, poly in geom.rooms.items()]
    ax.add_collection(PolyCollection([f[0] for f in fills], facecolors=[f[1] for f in fills], edgecolors="none", zorder=1))
    # 가구
    _draw_prims(ax, [p for f in plan.furniture for p in f.world()], 2)
    # 벽 포셰
    ax.add_patch(PathPatch(_path(geom.walls), facecolor="#2A2A2A", edgecolor=INK, linewidth=0.8, joinstyle="miter", zorder=4))
    _draw_prims(ax, [p for prims in plan.windows + plan.doors for p in prims], 5)
    # 라벨 (가구 위에 겹치면 흰 바탕)
    halos = [lab.box for lab in plan.labels if lab.halo]
    if halos:
        ax.add_collection(PolyCollection([[(b[0] - 0.05, b[1] - 0.03), (b[2] + 0.05, b[1] - 0.03), (b[2] + 0.05, b[3] + 0.03),
                                           (b[0] - 0.05, b[3] + 0.03)] for b in halos],
                                         facecolors="white", edgecolors="none", alpha=0.85, zorder=7))
    for lab in plan.labels:
        _draw_texts(ax, [(text_path(s, hgt, cx, cy, bold=bold), INK if bold else "#444444") for s, hgt, bold, (cx, cy) in lab.lines],
                    8, gid=f"label:{lab.room}")
    leaders = [lab.leader for lab in plan.labels if lab.leader]
    if leaders:
        ax.add_collection(LineCollection(leaders, colors="#444444", linewidths=0.5, zorder=8))
        ax.add_collection(PolyCollection([_sample(_circle(*end, 0.05), 12) for _, end in leaders], facecolors="#444444",
                                         edgecolors="none", zorder=8))
    # 치수·방위·축척·표제란
    _draw_prims(ax, _dim_prims(plan), 6)
    _draw_prims(ax, plan.anno, 6)
    if layout.needs_check:
        _draw_texts(ax, [(text_path("※ 확인 필요: " + " / ".join(layout.needs_check)[:160], 0.2, fx0 + 0.2, fy0 + 0.3, ha="left"),
                          "#C62828")], 8)
    return fig


def to_png(layout: Layout, index: int = 0, dpi: int = 160) -> bytes:
    with _style():
        fig = _figure(layout, index)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return buf.getvalue()


def to_svg(layout: Layout, index: int = 0) -> bytes:
    with _style():
        fig = _figure(layout, index)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", facecolor="white")
    return buf.getvalue()
