"""평면 배치 엔진 — 방 프로그램(종류·목표 면적·구역·이웃 관계) + 외곽 유형 → 0.3m 격자 위의 직사각형 방 (OR-Tools CP-SAT, CPU).

예전 자동 분할(앞 띠 / 복도 / 뒤 띠, 너비 = 면적 비례)은 '28m 일자 복도 + 1.4×6m 욕실' 같은 비현실적인 평면을 만들었다.
여기서는 방마다 직사각형 변수(x, y, w, h)를 두고 제약 충족 문제로 푼다.

  외곽 유형  : rect 일자형 · L ㄱ자형 · U ㄷ자형 · courtyard 중정형. 수영장·차고 같은 큰 특수실은 ㄱ·ㄷ자의 별동 날개에 둔다
  하드 제약  : 겹침 없음 · 외곽을 빈틈없이 채움 · 최소 폭·면적 · 장단변비 · 꼭 이웃할 방은 0.9m 이상 벽 공유(문) ·
               채광이 필요한 방은 외벽(또는 중정)에 닿음 · 모든 방이 현관(위층은 계단)에서 문으로 이어짐(접근 순서 규칙)
  목표       : 목표 면적 오차 + 복도 면적 최소 + 이웃하면 좋은 방 + 네모난 방 + 현관에서 짧은 동선 + 남향 거실·침실 + 물 쓰는 곳 모으기

core/floorplan.py 의 resolve 가 부른다. 시간 안에 답을 못 찾으면 SolverFailed → resolve 가 예전 단순 분할로 대신한다.
"""

import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field

from ortools.sat.python import cp_model
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from core.floorplan import (
    CIRCULATION,
    KIND_RULES,
    OUTSIDE,
    ROOM_TYPES,
    TYPOLOGY_KO,
    Door,
    Floor,
    Layout,
    Program,
    Rect,
    Room,
    max_aspect,
    needs_daylight,
    opening_segment,
    program_links,
    room_side,
    room_kind,
    room_zone,
    shared_wall,
)

GRID = 0.3  # 격자 (m)
HALL_W = 1.2  # 복도 폭 (m)
DOOR_LEN = 1.2  # 문(0.9m)을 문틀과 함께 낼 공유 벽 길이 (m) — 검증은 0.9m 이상이면 통과
OPEN_LEN = 2.4  # 트인 방(LDK)은 이만큼 이상 맞닿게
FRONT_DOOR_LEN = 1.2  # 현관 바깥문 자리
STAIR_SIZE = (2.4, 3.0)  # 꺾인 계단 (m)
TIME_LIMIT = 10.0  # 후보 하나(층 하나)의 최대 풀이 시간 (초) — 방이 적으면 줄인다 (auto_time_limit)
WORKERS = 8
COARSE_GRID = 0.6  # 방이 많으면 먼저 0.6m 격자로 빠르게 풀어 0.3m 풀이의 출발점(hint)으로 쓴다
COARSE_FROM = 8  # 방(복도·계단 제외)이 이보다 많으면 굵은 격자부터
FILLER_MAX = 4.5  # 틈 메우기 수납의 최대 넓이 (㎡)
SPECIAL_WING_MIN = 20.0  # 이 면적(㎡) 이상인 특수실(수영장·차고)은 ㄱ·ㄷ자 날개에 따로 둔다


class SolverFailed(RuntimeError):
    """시간 안에 제약을 모두 만족하는 배치를 찾지 못함 (메시지는 한국어)."""


def _snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ════════════════════════════════════════════════════════════════════
# 외곽 유형 (일자·ㄱ자·ㄷ자·중정형) → 외곽 다각형 + 블록(직사각형 조각) + 중정
# ════════════════════════════════════════════════════════════════════


@dataclass
class Shape:
    typology: str
    polygon: Polygon
    blocks: list[tuple[Rect, str]]  # (블록, 역할 main / special)
    court: Rect | None = None
    notes: list[str] = field(default_factory=list)

    def outline_value(self) -> Rect | list[tuple[float, float]]:
        b = self.polygon.bounds
        if self.polygon.area >= (b[2] - b[0]) * (b[3] - b[1]) - 1e-6:
            return Rect(x=b[0], y=b[1], w=round(b[2] - b[0], 2), h=round(b[3] - b[1], 2))
        return [(round(x, 2), round(y, 2)) for x, y in list(self.polygon.exterior.coords)[:-1]]


def _poly(points: list[tuple[float, float]], ox: float, oy: float) -> Polygon:
    return Polygon([(round(x + ox, 2), round(y + oy, 2)) for x, y in points])


def make_shape(typology: str, gross: float, special: float = 0.0, origin: tuple[float, float] = (0.0, 0.0),
               keep: Rect | None = None) -> Shape | None:
    """유형별 외곽. gross = 바닥 면적(중정 제외), special = 날개에 따로 둘 특수실 면적. 만들 수 없으면 None."""
    ox, oy = origin
    if typology == "rect":
        if keep is not None:
            poly = keep.polygon()
            return Shape("rect", poly, [(keep, "main")])
        d = _snap(_clamp(math.sqrt(gross / 1.6), 6.0, 12.6))
        w = _snap(gross / d)
        r = Rect(x=ox, y=oy, w=w, h=d)
        return Shape("rect", r.polygon(), [(r, "main")])
    if typology == "L":
        if special >= SPECIAL_WING_MIN:
            main_a, wing_a = gross - special, special
        else:
            main_a, wing_a = gross * 0.6, gross * 0.4
        d1 = _snap(_clamp(math.sqrt(main_a / 2.2), 7.8, 11.4))
        w1 = _snap(main_a / d1)
        if special >= SPECIAL_WING_MIN:
            ww = _snap(_clamp(math.sqrt(wing_a / 1.4), 6.0, min(w1 * 0.6, 13.2)))
        else:
            ww = _snap(_clamp(d1 * 0.9, 6.6, min(9.6, w1 * 0.6)))
        lw = _snap(wing_a / ww)
        if lw < 3.6 or w1 < ww + 3.6:
            return None
        pts = [(0, 0), (w1, 0), (w1, d1 + lw), (w1 - ww, d1 + lw), (w1 - ww, d1), (0, d1)]
        blocks = [(Rect(x=ox, y=oy, w=w1, h=d1), "main"),
                  (Rect(x=round(ox + w1 - ww, 2), y=round(oy + d1, 2), w=ww, h=lw), "special" if special >= SPECIAL_WING_MIN else "main")]
        return Shape("L", _poly(pts, ox, oy), blocks)
    if typology == "U":
        if special >= SPECIAL_WING_MIN:
            west_a = special
            east_a = (gross - special) * 0.3
        else:
            west_a = east_a = gross * 0.22
        bar_a = gross - west_a - east_a
        d1 = _snap(_clamp(math.sqrt(bar_a / 2.6), 7.8, 10.8))
        w1 = _snap(bar_a / d1)
        sw = special >= SPECIAL_WING_MIN
        ww = _snap(_clamp(math.sqrt(west_a / 1.4), 6.0, 13.2)) if sw else _snap(_clamp(d1 * 0.85, 6.0, 9.0))
        we = _snap(_clamp(d1 * 0.85, 6.0, 9.0))
        lw, le = _snap(west_a / ww), _snap(east_a / we)
        if min(lw, le) < 3.6 or w1 < ww + we + 3.6:
            return None
        pts = [(0, 0), (w1, 0), (w1, d1 + le), (w1 - we, d1 + le), (w1 - we, d1), (ww, d1), (ww, d1 + lw), (0, d1 + lw)]
        blocks = [(Rect(x=ox, y=oy, w=w1, h=d1), "main"),
                  (Rect(x=ox, y=round(oy + d1, 2), w=ww, h=lw), "special" if sw else "main"),
                  (Rect(x=round(ox + w1 - we, 2), y=round(oy + d1, 2), w=we, h=le), "main")]
        return Shape("U", _poly(pts, ox, oy), blocks)
    if typology == "courtyard":
        # 둘레 띠(깊이 d)의 넓이 = gross: 2d(W+H) − 4d² = gross, W = 1.2H. 중정이 4.8m 이상이 되는 가장 깊은 띠를 고른다
        for d in (6.6, 6.3, 6.0, 5.7, 5.4):
            h = _snap((gross + 4 * d * d) / (2 * d) / 2.2)
            w = _snap(gross / (2 * d) + 2 * d - h)
            cw, ch = round(w - 2 * d, 2), round(h - 2 * d, 2)
            if min(cw, ch) >= 4.8:
                r = Rect(x=ox, y=oy, w=w, h=h)
                court = Rect(x=round(ox + d, 2), y=round(oy + d, 2), w=cw, h=ch)
                return Shape("courtyard", r.polygon(), [(r, "main")], court=court)
        return None
    return None


def shape_from_outline(poly: Polygon) -> Shape:
    """주어진 외곽 다각형(직각 다각형)을 직사각형 블록으로 쪼갠다."""
    xs = sorted({round(x, 3) for x, _ in poly.exterior.coords})
    ys = sorted({round(y, 3) for _, y in poly.exterior.coords})
    rows: list[list[tuple[float, float]]] = []
    for y0, y1 in zip(ys, ys[1:]):
        runs, start = [], None
        for x0, x1 in zip(xs, xs[1:]):
            inside = poly.covers(box(x0, y0, x1, y1).centroid) and poly.buffer(1e-6).covers(box(x0, y0, x1, y1))
            if inside and start is None:
                start = x0
            if not inside and start is not None:
                runs.append((start, x0))
                start = None
        if start is not None:
            runs.append((start, xs[-1]))
        rows.append(runs)
    blocks: list[Rect] = []
    open_runs: dict[tuple[float, float], float] = {}
    for (y0, y1), runs in zip(zip(ys, ys[1:]), rows + [[]]):
        for run in list(open_runs):
            if run not in runs:
                blocks.append(Rect(x=run[0], y=open_runs[run], w=round(run[1] - run[0], 3), h=round(y0 - open_runs[run], 3)))
                del open_runs[run]
        for run in runs:
            open_runs.setdefault(run, y0)
    for run, start in open_runs.items():
        blocks.append(Rect(x=run[0], y=start, w=round(run[1] - run[0], 3), h=round(ys[-1] - start, 3)))
    b = poly.bounds
    typology = "rect" if len(blocks) == 1 and abs(poly.area - (b[2] - b[0]) * (b[3] - b[1])) < 1e-6 else "custom"
    return Shape(typology, poly, [(r, "main") for r in blocks])


# ════════════════════════════════════════════════════════════════════
# 격자
# ════════════════════════════════════════════════════════════════════


@dataclass
class Axis:
    cuts: list[float]  # 칸 경계 좌표 (m). 외곽 꼭짓점·계단 경계는 반드시 칸 경계에 온다

    @property
    def n(self) -> int:
        return len(self.cuts) - 1

    @property
    def cell(self) -> float:
        return (self.cuts[-1] - self.cuts[0]) / self.n

    def idx(self, v: float) -> int:
        i = bisect_left(self.cuts, v - 1e-6)
        if i > 0 and (i >= len(self.cuts) or abs(self.cuts[i - 1] - v) < abs(self.cuts[i] - v)):
            i -= 1
        return min(i, self.n)

    def at(self, i: int) -> float:
        return round(self.cuts[i], 3)


def make_axis(breaks: list[float], grid: float = GRID) -> Axis:
    """꺾이는 좌표(breaks) 사이를 grid(0.3m)에 가까운 같은 칸으로 나눈다."""
    pts = sorted({round(b, 3) for b in breaks})
    cuts = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        k = max(1, round((b - a) / grid))
        cuts += [a + (b - a) * i / k for i in range(1, k + 1)]
    return Axis(cuts)


# ════════════════════════════════════════════════════════════════════
# 한 층 풀기
# ════════════════════════════════════════════════════════════════════


@dataclass
class Item:
    id: str
    room: Room | None  # None = 엔진이 만든 복도
    kind: str
    fixed: tuple[int, int, int, int] | None = None  # (x, y, w, h) 칸
    sizes: list[tuple[int, int]] | None = None  # 정해진 크기 중 하나 (계단)
    target: float = 0.0  # 목표 칸 수 (외곽에 맞춰 늘이거나 줄인 값)
    orig: float = 0.0  # LLM 이 쓴 목표 칸 수
    optional: bool = False


@dataclass
class FloorPlan:
    rooms: list[Room]
    doors: list[Door]
    notes: list[str]
    rects: dict[str, Rect] = field(default_factory=dict)  # 풀이가 놓은 직사각형 (틈 메우기 전, 다음 풀이의 출발점)


def _access_rules(items: dict[str, Item], prog: Program, root: str | None, halls: list[str]) -> dict[str, list[str]]:
    """방마다 '이 방들 중 하나와 맞닿아 문으로 드나든다' 목록 — 순서 규칙으로 모든 방이 출발점(현관·계단)에 이어진다.

    출발점 → (거실 · 첫 복도) → 다른 복도 → 식당 → 주방 → 나머지 방 → 묶음(suite)에 딸린 방.
    앞 순서의 방만 뒤 순서 방의 출입구가 될 수 있어서 서로만 이어진 외딴 무리가 생기지 않는다.
    """
    by_kind: dict[str, list[str]] = {}
    for iid, it in items.items():
        by_kind.setdefault(it.kind, []).append(iid)
    living = next((i for i in by_kind.get("living", []) if i != root), None)
    dining = by_kind.get("dining", [])
    kitchens = by_kind.get("kitchen", [])
    baths = by_kind.get("bathroom", [])
    suite_of = {m: ms for ms in prog.suites.values() for m in ms[1:] if m in items}
    hubs = [h for h in [living, *halls] if h]
    rules: dict[str, list[str]] = {}
    base = [h for h in [root, *hubs] if h]
    if halls:
        rules[halls[0]] = [h for h in [root, living] if h]
        for k, h in enumerate(halls[1:], 1):
            rules[h] = [x for x in [root, living, *halls[:k]] if x]
    if living and root:
        rules[living] = [x for x in [root, halls[0] if halls else None] if x]
    common_baths = [b for b in baths if b not in suite_of]
    for iid, it in items.items():
        if iid == root or iid in rules:
            continue
        if iid in suite_of:
            ms = suite_of[iid]
            rules[iid] = [m for m in ms[: ms.index(iid)] if m in items]
            continue
        k = it.kind
        if k == "dining":
            opts = [root, living, *halls]
        elif k == "kitchen":
            opts = [*dining, root, living, *halls]
        elif k in ("utility", "pantry"):
            opts = [*kitchens, *halls, *dining, living]
        elif k == "sauna":
            opts = [*common_baths, *by_kind.get("pool", []), *halls, living]
        elif k == "bathroom":
            opts = [*halls, living, *dining]
        elif k in ("pool", "garage"):
            opts = [root, *halls, living, *dining]
        elif k == "court":
            opts = [x for x in items if x != iid and items[x].kind not in ("bathroom", "utility", "sauna", "court", "stair")]
        elif k == "balcony":
            opts = [x for x in items if items[x].kind in ("living", "dining", "bedroom", "office")]
        elif k == "entrance":
            opts = base
        elif k == "stair":
            opts = [*halls, living, root]
        else:  # 침실·서재·드레스룸·수납·기타
            opts = [*halls, living] + ([root] if k in ("storage", "other") else [])
        rules[iid] = [o for o in dict.fromkeys(opts) if o and o != iid and o in items]
    return rules


@dataclass
class Block:
    """배치 블록 (칸 단위). 긴 변 방향으로 '줄(row)'을 깔고, 방은 이어진 줄 몇 개에 걸친다."""

    x0: int
    y0: int
    x1: int
    y1: int
    role: str  # main / special
    horizontal: bool  # 줄이 가로(x)로 놓임 = 긴 변이 x
    rows: int

    @property
    def length(self) -> int:
        return self.x1 - self.x0 if self.horizontal else self.y1 - self.y0

    @property
    def depth(self) -> int:
        return self.y1 - self.y0 if self.horizontal else self.x1 - self.x0

    @property
    def start(self) -> int:  # 줄이 쌓이기 시작하는 좌표 (가로 줄이면 y)
        return self.y0 if self.horizontal else self.x0

    def spans(self) -> list[tuple[int, int]]:
        return [(a, b) for a in range(self.rows) for b in range(a, self.rows)]


def solve_floor(
    floor: Floor,
    shape: Shape,
    *,
    lowest: bool,
    stair_rect: Rect | None = None,
    place_stair: bool = False,
    entry_side: str = "S",
    seed: int = 0,
    time_limit: float | None = None,
    relaxed: bool = False,
    grid: float = GRID,
    n_fillers: int | None = None,
    hint: dict[str, Rect] | None = None,
) -> FloorPlan:
    """한 층의 방 프로그램을 shape 안에 배치한다. 못 찾으면 SolverFailed.

    모형: 외곽을 직사각형 블록으로 쪼개고(계단·중정 자리는 뺌), 블록마다 긴 변을 따라 높이가 정해지지 않은 '줄'을 2~4개 깐다.
    방은 한 블록 안에서 이어진 줄 몇 개에 걸치는 직사각형이고, 줄마다 그 줄에 걸친 방들의 길이 합 = 블록 길이.
    → 빈틈없이 채우기가 구조로 보장된다 (자유 직사각형 완전 패킹은 CP-SAT 도 답을 잘 못 찾는다).
    거실이 두 줄에 걸치고 그 옆 줄 하나가 1.2m 복도가 되는 식으로 실제 평면의 대부분을 표현할 수 있다.
    """
    notes: list[str] = []
    prog = program_links(floor)
    notes += prog.notes

    # 방 목록 (복도·계단은 엔진이 만든다)
    explicit_corr = [r for r in floor.rooms if r.type == "corridor"]
    explicit_stair = next((r for r in floor.rooms if r.type == "stair"), None)
    program = [r for r in floor.rooms if r.type not in CIRCULATION]
    if not program:
        raise SolverFailed("나눌 방이 없다 (복도·계단 외의 방을 1개 이상 넣어라)")

    # 고정된 방: 중정(중정형) · 위층 계단(아래층과 같은 자리)
    fixed_rooms: dict[str, tuple[Room, Rect]] = {}
    if shape.court is not None:
        court = next((r for r in program if room_kind(r) == "court"), None)  # LLM 이 쓴 중정이 있으면 그 방을 쓴다
        if court is None:
            cid = "court"
            while any(r.id == cid for r in floor.rooms):
                cid += "_"
            court = Room(id=cid, type="other", kind="court", name="중정")
        fixed_rooms[court.id] = (court, shape.court)
    stair_id = explicit_stair.id if explicit_stair else "stair"
    stair_room = explicit_stair or Room(id="stair", type="stair")
    if stair_rect is not None:
        fixed_rooms[stair_id] = (stair_room, stair_rect)

    # 블록 = 외곽 블록 − 고정된 방
    pieces: list[tuple[Polygon, str]] = []
    for blk, role in shape.blocks:
        rest = blk.polygon()
        for _, fr in fixed_rooms.values():
            rest = rest.difference(fr.polygon())
        pieces += [(pc, role) for pc in _rect_pieces(rest)]
    bx = [x for x, _ in shape.polygon.exterior.coords]
    by = [y for _, y in shape.polygon.exterior.coords]
    for pc, _ in pieces:
        b = pc.bounds
        bx += [b[0], b[2]]
        by += [b[1], b[3]]
    for _, fr in fixed_rooms.values():
        bx += [fr.x, fr.x + fr.w]
        by += [fr.y, fr.y + fr.h]
    ax, ay = make_axis(bx, grid), make_axis(by, grid)
    cx, cy = ax.cell, ay.cell
    cell_area = cx * cy
    hall_cells = max(1, round(HALL_W / grid))
    ka, kl = max(1, round(cell_area / (GRID * GRID))), max(1, round(cx / GRID))  # 벌점을 0.3m 칸 기준으로 맞추는 배수

    def cells(length: float) -> int:
        return max(1, math.ceil(length / grid - 1e-6))

    blocks: list[Block] = []
    for pc, role in pieces:
        b = pc.bounds
        x0, y0, x1, y1 = ax.idx(b[0]), ay.idx(b[1]), ax.idx(b[2]), ay.idx(b[3])
        if x1 <= x0 or y1 <= y0:
            continue
        horizontal = (b[2] - b[0]) >= (b[3] - b[1])
        depth_m = min(b[2] - b[0], b[3] - b[1])
        depth_cells = (y1 - y0) if horizontal else (x1 - x0)
        rows = max(1, min(4, round((depth_m + HALL_W) / 3.3), depth_cells // hall_cells))
        blocks.append(Block(x0, y0, x1, y1, role, horizontal, rows))
    total_cells = sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in blocks)

    items: dict[str, Item] = {}
    for r in program:
        if r.id not in fixed_rooms:
            items[r.id] = Item(r.id, r, room_kind(r))
    for fid, (room, fr) in fixed_rooms.items():
        x0, y0 = ax.idx(fr.x), ay.idx(fr.y)
        items[fid] = Item(fid, room, room_kind(room), fixed=(x0, y0, ax.idx(fr.x + fr.w) - x0, ay.idx(fr.y + fr.h) - y0))
    if place_stair and stair_rect is None:
        sx, sy = (math.ceil(STAIR_SIZE[0] / cx - 0.05), math.ceil(STAIR_SIZE[1] / cy - 0.05))
        sx2, sy2 = (math.ceil(STAIR_SIZE[1] / cx - 0.05), math.ceil(STAIR_SIZE[0] / cy - 0.05))
        items[stair_id] = Item(stair_id, stair_room, "stair", sizes=[(sx, sy), (sx2, sy2)])
    n_prog = len(program)
    n_halls = 1 if n_prog <= 5 else (2 if n_prog <= 10 else 3)
    halls = []
    for k in range(n_halls):
        base = explicit_corr[k] if k < len(explicit_corr) else None
        hid = base.id if base else ("hall" if k == 0 else f"hall{k + 1}")
        while hid in items:
            hid += "_"
        items[hid] = Item(hid, base, "corridor", optional=True)
        halls.append(hid)
    # 틈 메우기용 수납(붙박이장·팬트리) — 있어도 되고 없어도 되는 작은 방. 줄 길이를 맞추기 쉽게 한다
    fillers = []
    for k in range(n_fillers if n_fillers is not None else (1 if n_prog <= 6 else 2)):
        fid = f"closet{k + 1}"
        while fid in items:
            fid += "_"
        items[fid] = Item(fid, Room(id=fid, type="storage", name="수납"), "storage", optional=True)
        fillers.append(fid)

    entry = next((i for i, it in items.items() if it.kind == "entrance"), None)
    if lowest and entry:
        root = entry
    elif not lowest and stair_id in items:
        root = stair_id
    else:
        root = next((i for i, it in items.items() if it.kind == "living"), None) or next(iter(items))

    # 목표 면적 → 칸 수. 블록 역할(main/special)별로 넓이에 맞춰 늘이거나 줄인다
    has_special = any(b.role == "special" for b in blocks)

    def in_special(it: Item) -> bool:
        return has_special and it.room is not None and room_zone(it.room) == "special" and it.kind != "court"

    def target_m2(r: Room) -> float:
        return r.area if r.area and r.area > 0 else max(ROOM_TYPES[r.type][1], 4.0)

    movable = [i for i, it in items.items() if it.fixed is None]
    groups = {"special": [i for i in movable if in_special(items[i])]}
    groups["main"] = [i for i in movable if items[i].sizes is None and not items[i].optional and i not in groups["special"]]
    special_cells = sum((b.x1 - b.x0) * (b.y1 - b.y0) for b in blocks if b.role == "special")
    stair_cells = sum(s[0] * s[1] for it in items.values() if it.sizes for s in it.sizes[:1])
    hall_share = 0.0 if n_prog <= 2 else (0.06 if n_prog <= 6 else 0.08)
    caps = {"special": special_cells, "main": (total_cells - special_cells - stair_cells) * (1 - hall_share)}
    for g, ids in groups.items():
        if not ids:
            continue
        want = sum(target_m2(items[i].room) for i in ids) / cell_area
        scale = caps[g] / want if want else 1.0
        if g == "main" and abs(scale - 1) > 0.1:
            notes.append(f"목표 면적 합 {want * cell_area:.1f}㎡ ↔ 복도·계단을 뺀 가용 면적 약 {caps[g] * cell_area:.1f}㎡ → 비례로 늘이거나 줄였다")
        for i in ids:
            items[i].target = target_m2(items[i].room) * scale / cell_area
            items[i].orig = target_m2(items[i].room) / cell_area

    m = cp_model.CpModel()
    X, Y = ax.n, ay.n
    V: dict[str, dict] = {}
    obj: list = []
    # 면적 범위: 목표 합이 외곽과 맞으면 LLM 목표의 ±18% (검증 허용 ±20% 안), 크게 다르면 늘이거나 줄인 목표 기준
    lo_area_f, hi_area_f = (0.65, 1.5) if relaxed else (0.82, 1.18)
    aspect_f = 1.3 if relaxed else 1.0

    # 블록의 줄 경계
    bounds: list[list] = []
    for bi, b in enumerate(blocks):
        min_row = hall_cells if b.rows > 1 else b.depth
        hs = [m.NewIntVar(min(min_row, b.depth), b.depth, f"rh_{bi}_{k}") for k in range(b.rows)]
        m.Add(sum(hs) == b.depth)
        edges = [b.start]
        for h in hs:
            edges.append(edges[-1] + h)
        bounds.append(edges)
    cover: dict[tuple[int, int], list] = {}  # (블록, 줄) → [(존재 리터럴, 길이 변수, 구간)]

    for iid, it in items.items():
        v: dict = {}
        if it.fixed is not None:
            fx, fy, fw, fh = it.fixed
            v["x"], v["y"], v["w"], v["h"] = (m.NewConstant(fx), m.NewConstant(fy), m.NewConstant(fw), m.NewConstant(fh))
            v["xe"], v["ye"] = m.NewConstant(fx + fw), m.NewConstant(fy + fh)
            v["p"] = None
            v["a"] = m.NewConstant(fw * fh)
            V[iid] = v
            continue
        if it.kind == "corridor":
            minw_x = minw_y = hall_cells
        elif it.sizes:
            minw_x = min(s[0] for s in it.sizes)
            minw_y = min(s[1] for s in it.sizes)
        else:
            dw = max(KIND_RULES[it.kind][3], ROOM_TYPES[it.room.type][2])
            minw_x, minw_y = math.ceil(dw / cx - 1e-6), math.ceil(dw / cy - 1e-6)
        v["p"] = m.NewBoolVar(f"p_{iid}") if it.optional else None
        v["x"] = m.NewIntVar(0, X, f"x_{iid}")
        v["y"] = m.NewIntVar(0, Y, f"y_{iid}")
        v["w"] = m.NewIntVar(min(minw_x, X), X, f"w_{iid}")
        v["h"] = m.NewIntVar(min(minw_y, Y), Y, f"h_{iid}")
        v["xe"] = m.NewIntVar(0, X, f"xe_{iid}")
        v["ye"] = m.NewIntVar(0, Y, f"ye_{iid}")
        m.Add(v["xe"] == v["x"] + v["w"])
        m.Add(v["ye"] == v["y"] + v["h"])
        v["a"] = m.NewIntVar(0, X * Y, f"a_{iid}")
        m.AddMultiplicationEquality(v["a"], [v["w"], v["h"]])
        V[iid] = v
        # 어느 블록의 어느 줄들에 걸칠지
        if in_special(it):
            allowed = [k for k, b in enumerate(blocks) if b.role == "special"]
        elif has_special:
            allowed = [k for k, b in enumerate(blocks) if b.role != "special"]
        else:
            allowed = list(range(len(blocks)))
        choice = []
        for bi in allowed:
            b = blocks[bi]
            edges = bounds[bi]
            span_lits = []
            for k0, k1 in b.spans():
                s = m.NewBoolVar(f"s_{iid}_{bi}_{k0}_{k1}")
                if b.horizontal:
                    m.Add(v["y"] == edges[k0]).OnlyEnforceIf(s)
                    m.Add(v["ye"] == edges[k1 + 1]).OnlyEnforceIf(s)
                    m.Add(v["x"] >= b.x0).OnlyEnforceIf(s)
                    m.Add(v["xe"] <= b.x1).OnlyEnforceIf(s)
                else:
                    m.Add(v["x"] == edges[k0]).OnlyEnforceIf(s)
                    m.Add(v["xe"] == edges[k1 + 1]).OnlyEnforceIf(s)
                    m.Add(v["y"] >= b.y0).OnlyEnforceIf(s)
                    m.Add(v["ye"] <= b.y1).OnlyEnforceIf(s)
                span_lits.append(((k0, k1), s))
                choice.append(s)
            for k in range(b.rows):
                lits = [s for (k0, k1), s in span_lits if k0 <= k <= k1]
                c = m.NewBoolVar(f"c_{iid}_{bi}_{k}")
                m.Add(c == sum(lits))
                length = v["w"] if b.horizontal else v["h"]
                start, end = (v["x"], v["xe"]) if b.horizontal else (v["y"], v["ye"])
                iv = m.NewOptionalIntervalVar(start, length, end, c, f"iv_{iid}_{bi}_{k}")
                z = m.NewIntVar(0, b.length, f"z_{iid}_{bi}_{k}")
                m.Add(z == length).OnlyEnforceIf(c)
                m.Add(z == 0).OnlyEnforceIf(c.Not())
                cover.setdefault((bi, k), []).append((iv, z))
        m.Add(sum(choice) == (v["p"] if v["p"] is not None else 1))
        if it.optional:  # 복도·틈 메우기 수납: 없으면 넓이 0
            v["ae"] = m.NewIntVar(0, X * Y, f"ae_{iid}")
            m.Add(v["ae"] == v["a"]).OnlyEnforceIf(v["p"])
            m.Add(v["ae"] == 0).OnlyEnforceIf(v["p"].Not())
        if it.kind == "corridor":
            o = m.NewBoolVar(f"o_{iid}")
            m.Add(v["h"] == hall_cells).OnlyEnforceIf(o)
            m.Add(v["w"] == hall_cells).OnlyEnforceIf(o.Not())
            obj.append(3 * ka * v["ae"])
            continue
        if iid in fillers:
            m.Add(v["a"] <= math.ceil(FILLER_MAX / cell_area))
            _aspect_limit(m, v, 3.0, cx, cy)
            obj.append(40 * v["p"] + 2 * ka * v["ae"])
            continue
        if it.sizes:
            m.AddAllowedAssignments([v["w"], v["h"]], it.sizes)
            continue
        # 면적
        t = it.target
        ref = it.orig if 0.88 <= t / max(it.orig, 1e-6) <= 1.12 else t
        lo = max(KIND_RULES[it.kind][4] / cell_area, lo_area_f * ref, minw_x * minw_y)
        hi = max(lo + 1, hi_area_f * ref)
        m.Add(v["a"] >= math.floor(lo))
        m.Add(v["a"] <= math.ceil(hi))
        dev = m.NewIntVar(0, X * Y, f"dev_{iid}")
        m.AddAbsEquality(dev, v["a"] - round(t))
        obj.append(2 * ka * dev)
        # 장단변비 (긴 변 ≤ R × 짧은 변)
        ratio = max_aspect(it.room)
        if ratio:
            _aspect_limit(m, v, ratio * aspect_f, cx, cy)
        # 네모난 방 (둘레 최소)
        obj.append(2 * kl * (v["w"] + v["h"]))

    # 줄마다: 겹치지 않고 길이 합 = 블록 길이 → 빈틈없이 채움
    for (bi, k), entries in cover.items():
        m.AddNoOverlap([iv for iv, _ in entries])
        m.Add(sum(z for _, z in entries) == blocks[bi].length)

    # ── 맞닿음 ──
    adj_cache: dict[tuple[str, str, int], cp_model.IntVar] = {}
    door_cells = cells(DOOR_LEN)

    def adj(a: str, b: str, length: int = door_cells):
        key = (min(a, b), max(a, b), length)
        if key in adj_cache:
            return adj_cache[key]
        va, vb = V[a], V[b]
        dirs = []
        for p, q in ((va, vb), (vb, va)):
            d = m.NewBoolVar("")
            m.Add(p["xe"] == q["x"]).OnlyEnforceIf(d)
            m.Add(p["ye"] >= q["y"] + length).OnlyEnforceIf(d)
            m.Add(q["ye"] >= p["y"] + length).OnlyEnforceIf(d)
            dirs.append(d)
            d = m.NewBoolVar("")
            m.Add(p["ye"] == q["y"]).OnlyEnforceIf(d)
            m.Add(p["xe"] >= q["x"] + length).OnlyEnforceIf(d)
            m.Add(q["xe"] >= p["x"] + length).OnlyEnforceIf(d)
            dirs.append(d)
        for d in dirs:
            for pv in (va["p"], vb["p"]):
                if pv is not None:
                    m.AddImplication(d, pv)
        lit = m.NewBoolVar(f"adj_{a}_{b}")
        m.AddBoolOr(dirs).OnlyEnforceIf(lit)
        for d in dirs:
            m.AddImplication(d, lit)
        adj_cache[key] = lit
        return lit

    # 외벽(·중정) 선분: (가로?, 좌표 칸, 시작 칸, 끝 칸, 방이 어느 쪽에 붙는지 +1 위/오른쪽 −1 아래/왼쪽, 방위)
    segs = []
    coords = list(shape.polygon.exterior.coords)
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        if abs(y1 - y2) < 1e-6:
            up = shape.polygon.contains(Point((x1 + x2) / 2, y1 + 0.05))
            segs.append((True, ay.idx(y1), ax.idx(min(x1, x2)), ax.idx(max(x1, x2)), 1 if up else -1, "S" if up else "N"))
        else:
            right = shape.polygon.contains(Point(x1 + 0.05, (y1 + y2) / 2))
            segs.append((False, ax.idx(x1), ay.idx(min(y1, y2)), ay.idx(max(y1, y2)), 1 if right else -1, "W" if right else "E"))
    if shape.court is not None:  # 중정을 둘러싼 방도 채광 가능
        c = shape.court
        x0, y0, x1, y1 = ax.idx(c.x), ay.idx(c.y), ax.idx(c.x + c.w), ay.idx(c.y + c.h)
        segs += [(True, y0, x0, x1, -1, "N"), (True, y1, x0, x1, 1, "S"), (False, x0, y0, y1, -1, "E"), (False, x1, y0, y1, 1, "W")]

    def touch(a: str, sides: set[str] | None = None, length: int = door_cells):
        va = V[a]
        lits = []
        for horiz, c, s0, s1, side, compass in segs:
            if (sides is not None and compass not in sides) or s1 - s0 < length:
                continue
            d = m.NewBoolVar("")
            if horiz:
                m.Add((va["y"] if side > 0 else va["ye"]) == c).OnlyEnforceIf(d)
                m.Add(va["x"] + length <= s1).OnlyEnforceIf(d)
                m.Add(va["xe"] >= s0 + length).OnlyEnforceIf(d)
            else:
                m.Add((va["x"] if side > 0 else va["xe"]) == c).OnlyEnforceIf(d)
                m.Add(va["y"] + length <= s1).OnlyEnforceIf(d)
                m.Add(va["ye"] >= s0 + length).OnlyEnforceIf(d)
            lits.append(d)
        lit = m.NewBoolVar(f"ext_{a}")
        if lits:
            m.AddBoolOr(lits).OnlyEnforceIf(lit)
        else:
            m.Add(lit == 0)
        return lit

    # 필수 이웃 · 트인 방 · LLM 이 쓴 문
    required = list(prog.required)
    for d in floor.doors:
        if d.to != OUTSIDE and d.room in items and d.to in items:
            required.append((d.room, d.to))
    open_set = {tuple(sorted(p)) for p in prog.open_pairs}
    for a, b in required:
        if a not in items or b not in items:
            continue
        length = cells(OPEN_LEN) if tuple(sorted((a, b))) in open_set else door_cells
        lit = adj(a, b, length)
        if relaxed and (a, b) not in prog.explicit:
            obj.append(80 * lit.Not())
        else:
            m.Add(lit == 1)
    for a, b in prog.preferred:
        if a in items and b in items:
            obj.append(60 * adj(a, b).Not())

    # 접근(문으로 이어짐)
    rules = _access_rules(items, prog, root, halls)
    for f in fillers:
        rules[f] = [i for i, it in items.items() if i not in fillers
                    and it.kind in ("corridor", "living", "dining", "kitchen", "bedroom", "office", "dressroom", "entrance")]
    for iid, opts in rules.items():
        if not opts:
            continue
        lits = [adj(iid, o) for o in opts]
        cond = [V[iid]["p"]] if V[iid]["p"] is not None else []
        m.AddBoolOr(lits).OnlyEnforceIf(cond)
        # 침실은 복도에서 들어가는 쪽을 조금 선호 (거실에서 바로 들어가는 것도 허용)
        if items[iid].kind in ("bedroom", "office") and halls:
            via_hall = m.NewBoolVar("")
            m.AddBoolOr([adj(iid, h) for h in halls]).OnlyEnforceIf(via_hall)
            obj.append(25 * via_hall.Not())
    living = next((i for i, it in items.items() if it.kind == "living" and i != root), None)
    if halls and root and (lowest and entry or not lowest):
        m.AddBoolOr([adj(root, x) for x in [living, halls[0]] if x])

    # 채광(외벽) · 현관 바깥문 · 남향
    for iid, it in items.items():
        if it.room is None or it.fixed is not None or it.kind in ("corridor", "stair") or iid in fillers:
            continue
        if it.kind == "entrance" and lowest:
            m.Add(touch(iid, length=cells(FRONT_DOOR_LEN)) == 1)
            obj.append(120 * touch(iid, {entry_side}, cells(FRONT_DOOR_LEN)).Not())
            continue
        if needs_daylight(it.room):
            lit = touch(iid)
            if relaxed and it.kind != "living":
                obj.append(150 * lit.Not())
            else:
                m.Add(lit == 1)
        elif it.kind in ("kitchen", "dining", "bathroom", "pool", "sauna"):
            obj.append(15 * touch(iid).Not())
        if it.kind == "living":
            obj.append(80 * touch(iid, {"S"}).Not())
        elif it.kind == "bedroom":
            obj.append(15 * touch(iid, {"S", "E"}).Not())

    # 짧은 동선 (현관·계단 → 거실·첫 복도), 물 쓰는 곳 모으기
    def dist(a: str, b: str):
        va, vb = V[a], V[b]
        dx = m.NewIntVar(0, 2 * X, "")
        dy = m.NewIntVar(0, 2 * Y, "")
        m.AddAbsEquality(dx, va["x"] + va["xe"] - vb["x"] - vb["xe"])
        m.AddAbsEquality(dy, va["y"] + va["ye"] - vb["y"] - vb["ye"])
        return kl * (dx + dy)

    if root:
        for hub in [living, *halls[:1]]:
            if hub:
                obj.append(dist(root, hub))
    wet = [i for i, it in items.items() if it.kind in ("bathroom", "utility", "sauna") and it.fixed is None]
    for i, a in enumerate(wet):
        for b in wet[i + 1:]:
            obj.append(dist(a, b))
    # 구역 모으기: 개인 공간(침실·서재)은 안방 가까이, 공용(식당·주방)은 거실 가까이 → 공용·개인이 섞이지 않게
    heads = [ms[0] for ms in prog.suites.values() if ms and ms[0] in items]
    private_anchor = heads[0] if heads else next((i for i, it in items.items() if it.kind == "bedroom"), None)
    for iid, it in items.items():
        if it.fixed is not None or it.optional or iid in (private_anchor, living):
            continue
        if it.kind in ("bedroom", "office") and private_anchor:
            obj.append(dist(iid, private_anchor))
        elif it.kind in ("dining", "kitchen") and living:
            obj.append(dist(iid, living))
    # 같은 종류·같은 목표의 방은 순서를 정해 대칭 탐색을 줄인다
    in_suite = {x for ms in prog.suites.values() for x in ms}
    sig: dict[tuple, list[str]] = {}
    for iid, it in items.items():
        if it.room is not None and it.fixed is None and it.sizes is None and not it.optional:
            r = it.room
            if not any([r.adjacent, r.near, r.open_to, r.suite]) and iid not in in_suite:
                sig.setdefault((it.kind, round(it.target)), []).append(iid)
    for ids in sig.values():
        for a, b in zip(ids, ids[1:]):
            m.Add(V[a]["x"] * (Y + 1) + V[a]["y"] <= V[b]["x"] * (Y + 1) + V[b]["y"])
    for seq in (halls, fillers):  # 빈 복도·수납은 뒤로
        for a, b in zip(seq, seq[1:]):
            m.AddImplication(V[b]["p"], V[a]["p"])

    if hint:  # 굵은 격자 풀이의 방 위치를 출발점으로
        for iid, r in hint.items():
            if iid in V and items[iid].fixed is None:
                v = V[iid]
                x0, y0, x1, y1 = ax.idx(r.x), ay.idx(r.y), ax.idx(r.x + r.w), ay.idx(r.y + r.h)
                for key, val in (("x", x0), ("y", y0), ("w", x1 - x0), ("h", y1 - y0)):
                    m.AddHint(v[key], val)
                if v["p"] is not None:
                    m.AddHint(v["p"], 1)
        for iid in [*halls, *fillers]:
            if iid not in hint:
                m.AddHint(V[iid]["p"], 0)
    m.Minimize(sum(obj))
    time_limit = time_limit or auto_time_limit(len(program))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = WORKERS
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        why = "제약을 모두 만족하는 배치가 없다" if status == cp_model.INFEASIBLE else f"{time_limit:g}초 안에 배치를 찾지 못했다"
        raise SolverFailed(f"{floor.label}: {why}")

    # 결과 → 방(m 좌표)
    placed: list[Room] = []
    rects: dict[str, Rect] = {}
    for iid, it in items.items():
        v = V[iid]
        if v["p"] is not None and not solver.Value(v["p"]):
            continue
        x, y = solver.Value(v["x"]), solver.Value(v["y"])
        xe, ye = solver.Value(v["xe"]), solver.Value(v["ye"])
        rect = Rect(x=ax.at(x), y=ay.at(y), w=round(ax.at(xe) - ax.at(x), 3), h=round(ay.at(ye) - ay.at(y), 3))
        rects[iid] = rect
        base = it.room or Room(id=iid, type="corridor")
        update: dict = {"rect": rect}
        if it.kind in ("corridor", "stair", "court"):
            update["area"] = None
        placed.append(base.model_copy(update=update))
    order = {r.id: i for i, r in enumerate(floor.rooms)}
    placed.sort(key=lambda r: order.get(r.id, len(order)))
    placed, fill_notes = fill_voids(placed, shape.polygon, rules)
    doors = floor.doors or _doors_from_access(placed, rules, open_set, root if lowest and entry else None, entry_side, shape.polygon)
    status_ko = "최적" if status == cp_model.OPTIMAL else "시간 제한 내 최선"
    hall_area = sum(rects[h].w * rects[h].h for h in halls if h in rects)
    notes.append(f"{floor.label}: 배치 엔진(CP-SAT) {status_ko} 해 — 복도 {hall_area:.1f}㎡" + (" (제약 일부 완화)" if relaxed else ""))
    notes += fill_notes
    return FloorPlan(placed, doors, notes, rects)


def _aspect_limit(m: cp_model.CpModel, v: dict, ratio: float, cx: float, cy: float) -> None:
    """긴 변 ≤ ratio × 짧은 변 (m 기준). 칸 크기 반올림 때문에 기준을 넘지 않게 정수 계수를 보수적으로 잡는다."""
    m.Add(math.ceil(1000 * cx) * v["w"] <= math.floor(1000 * ratio * cy) * v["h"])
    m.Add(math.ceil(1000 * cy) * v["h"] <= math.floor(1000 * ratio * cx) * v["w"])


def plan_floor(floor: Floor, shape: Shape, *, time_limit: float | None = None, **kw) -> FloorPlan:
    """한 층 풀기. 방이 많으면 먼저 0.6m 격자로 빠르게 풀고(시간의 절반), 그 답을 출발점으로 0.3m 격자에서 다듬는다.

    0.3m 풀이가 시간 안에 안 나오면 0.6m 답을 그대로 쓴다. 그래도 없으면 제약을 조금 풀어(relaxed) 다시 푼다.
    """
    n = len([r for r in floor.rooms if r.type not in CIRCULATION])
    limit = time_limit or auto_time_limit(n)
    if n <= COARSE_FROM:
        try:
            return solve_floor(floor, shape, time_limit=limit, **kw)
        except SolverFailed:
            return solve_floor(floor, shape, time_limit=limit, relaxed=True, **kw)
    for relaxed in (False, True):
        try:
            coarse = solve_floor(floor, shape, time_limit=limit * 0.5, grid=COARSE_GRID, relaxed=relaxed, **kw)
        except SolverFailed:
            continue
        try:
            fine = solve_floor(floor, shape, time_limit=limit * 0.5, relaxed=relaxed, hint=coarse.rects, **kw)
            return fine
        except SolverFailed:
            coarse.notes.append(f"{floor.label}: 0.3m 격자 다듬기가 시간 안에 끝나지 않아 0.6m 격자 배치를 썼다")
            return coarse
    raise SolverFailed(f"{floor.label}: {limit:g}초 안에 배치를 찾지 못했다 (제약을 풀어도 실패)")


def _rect_pieces(geom) -> list[Polygon]:
    """빈 곳(격자 위 직각 다각형들) → 직사각형 조각."""
    polys = [geom] if isinstance(geom, Polygon) else [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    out = []
    for poly in polys:
        if poly.area < 1e-3:
            continue
        for r in shape_from_outline(poly.buffer(0)).blocks:
            out.append(r[0].polygon())
    return [p for p in out if p.area > 1e-3]


def fill_voids(rooms: list[Room], outline: Polygon, rules: dict[str, list[str]]) -> tuple[list[Room], list[str]]:
    """풀이에서 남은 빈 칸을 메운다: ① 이웃 방에 붙여 직사각형이 되면 늘이고 ② 넉넉하면 수납으로 ③ 나머지는 가장 길게 맞닿은 방에 붙인다."""
    notes: list[str] = []
    rooms = list(rooms)
    fixed = {"stair", "court"}

    def shapes():
        return {r.id: r.shape() for r in rooms}

    void = outline.difference(unary_union([p for p in shapes().values()]))
    pieces = _rect_pieces(void)
    if not pieces:
        return rooms, notes
    total = sum(p.area for p in pieces)
    left: list[Polygon] = []
    for piece in sorted(pieces, key=lambda p: -p.area):
        done = False
        polys = shapes()
        # ① 직사각형 그대로 늘일 수 있는 이웃 (목표보다 작은 방 먼저, 계단·중정 제외)
        cands = []
        for i, r in enumerate(rooms):
            if room_kind(r) in fixed or r.rect is None:
                continue
            u = polys[r.id].union(piece)
            if isinstance(u, Polygon) and abs(u.area - box(*u.bounds).area) < 1e-3:
                deficit = (r.area or 0) - polys[r.id].area
                cands.append((r.type == "corridor", -deficit, i, u))
        if cands:
            _, _, i, u = min(cands, key=lambda c: c[:3])
            b = u.bounds
            rooms[i] = rooms[i].model_copy(update={"rect": Rect(x=round(b[0], 3), y=round(b[1], 3), w=round(b[2] - b[0], 3), h=round(b[3] - b[1], 3))})
            done = True
        if not done:
            left.append(piece)
    for piece in left:
        polys = shapes()
        b = piece.bounds
        short = min(b[2] - b[0], b[3] - b[1])
        touching = [(polys[r.id].intersection(piece.buffer(0.01)).length, i) for i, r in enumerate(rooms) if room_kind(r) not in fixed]
        touching = [t for t in touching if t[0] > 0.05]
        if piece.area >= 1.2 and short >= 0.9 and touching:
            sid = "store"
            while any(r.id == sid for r in rooms):
                sid += "_"
            rooms.append(Room(id=sid, type="storage", name="수납", rect=Rect(x=round(b[0], 3), y=round(b[1], 3),
                                                                           w=round(b[2] - b[0], 3), h=round(b[3] - b[1], 3))))
            rules[sid] = [rooms[i].id for _, i in sorted(touching, reverse=True) if room_kind(rooms[i]) not in ("bathroom", "sauna", "pool")]
            continue
        if touching:
            _, i = max(touching)
            u = polys[rooms[i].id].union(piece.buffer(0.001, join_style="mitre")).buffer(-0.001, join_style="mitre").simplify(0.001)
            if isinstance(u, Polygon):
                rooms[i] = rooms[i].model_copy(update={"rect": None, "polygon": [(round(x, 3), round(y, 3)) for x, y in list(u.exterior.coords)[:-1]]})
    if total > 0.05:
        notes.append(f"배치 뒤 남은 틈 {total:.1f}㎡ 를 이웃 방·수납으로 메웠다")
    return rooms, notes


def auto_time_limit(n_rooms: int) -> float:
    """방 수에 맞춘 풀이 시간 — 작은 집은 금방 좋은 답이 나온다."""
    return 3.0 if n_rooms <= 6 else (6.0 if n_rooms <= 10 else TIME_LIMIT)


def _place_door(a: str, b: str | None, polys: dict[str, Polygon], outline: Polygon, width: float,
                taken: list[LineString], prefer: str | None = None) -> Door | None:
    """a 의 벽 중 b(없으면 바깥)와 맞닿은 곳에 문 하나 — 이미 놓은 문과 겹치지 않는 자리(가운데 → 양 끝 쪽)를 찾는다."""
    pa = polys[a]
    segs = shared_wall(pa, polys[b] if b else None, outline)
    segs.sort(key=lambda s: (prefer is not None and _side(s, pa) != prefer, -_seg_len(s)))
    for seg in segs:
        length = _seg_len(seg)
        if length < width + 0.2:
            continue
        side = _side(seg, pa)
        wall = room_side(pa, side)
        if wall is None:
            continue
        start = Point(seg[0]).distance(Point(wall[0]))
        for off in ((length - width) / 2, 0.1, length - width - 0.1):
            offset = round(start + off, 2)
            cand = opening_segment(pa, side, offset, width)
            if cand is None:
                continue
            line = LineString(cand)
            if not LineString(seg).buffer(0.02).covers(line) or any(line.buffer(0.15).intersects(t) for t in taken):
                continue
            taken.append(line)
            return Door(room=a, to=b or OUTSIDE, side=side, offset=offset, width=width)
    return None


def _doors_from_access(rooms: list[Room], rules: dict[str, list[str]], open_set: set, entry: str | None,
                       entry_side: str, outline: Polygon) -> list[Door]:
    """접근 규칙에서 실제로 맞닿은 첫 출입구로 문을 낸다 (트인 방 사이는 문 없음, 문끼리 겹치지 않게). 현관은 바깥문."""
    polys = {r.id: r.shape() for r in rooms if r.shape() is not None}
    doors: list[Door] = []
    taken: list[LineString] = []
    if entry and entry in polys:
        door = _place_door(entry, None, polys, outline, 1.0, taken, prefer=entry_side)
        doors.append(door or Door(room=entry, to=OUTSIDE, width=1.0))
    kinds = {r.id: room_kind(r) for r in rooms}
    for room in rooms:
        rid = room.id
        if rid not in polys or rid not in rules:
            continue
        opts = [o for o in rules[rid] if o in polys]
        if any(tuple(sorted((rid, o))) in open_set for o in opts):
            continue  # 트인 방으로 이어짐
        width = 0.8 if kinds[rid] in ("bathroom", "storage", "dressroom", "sauna", "pantry") else 0.9
        # 규칙의 출입구 → (못 찾으면) 물 쓰는 곳이 아닌 아무 이웃
        others = [o for o in polys if o != rid and o not in opts and kinds.get(o) not in ("bathroom", "sauna", "pool", "court", "storage")]
        for o in opts + others:
            door = _place_door(rid, o, polys, outline, width, taken)
            if door:
                doors.append(door)
                break
    return doors


def _seg_len(s) -> float:
    return ((s[1][0] - s[0][0]) ** 2 + (s[1][1] - s[0][1]) ** 2) ** 0.5


def _side(seg, poly: Polygon) -> str:
    (x1, y1), (x2, y2) = seg
    c = poly.centroid
    if abs(x2 - x1) >= abs(y2 - y1):
        return "N" if (y1 + y2) / 2 > c.y else "S"
    return "E" if (x1 + x2) / 2 > c.x else "W"


# ════════════════════════════════════════════════════════════════════
# 레이아웃 전체 (유형 고르기 → 층마다 풀기 → 계단 맞추기)
# ════════════════════════════════════════════════════════════════════


def _target(r: Room) -> float:
    return r.area if r.area and r.area > 0 else max(ROOM_TYPES[r.type][1], 4.0)


def auto_typology(layout: Layout) -> str:
    """유형을 안 정했을 때: 여러 층·작은 집은 일자형, 큰 특수실이 있으면 ㄱ자, 아주 크면 ㄷ자."""
    if len(layout.floors) > 1:
        return "rect"
    rooms = layout.floors[0].rooms
    area = layout.outline_polygon().area
    special = sum(_target(r) for r in rooms if room_zone(r) == "special" and room_kind(r) != "court")
    if special >= SPECIAL_WING_MIN:
        return "L"
    if area > 260:
        return "U"
    return "rect"


def _big_special(layout: Layout) -> list[Room]:
    return [r for f in layout.floors for r in f.rooms
            if room_zone(r) == "special" and room_kind(r) != "court" and _target(r) >= SPECIAL_WING_MIN and r.placement != "main"]


def candidate_options(layout: Layout) -> list[dict]:
    """후보안 B·C… 에 쓸 풀이 옵션 (A = 기본 배치와 다른 것부터): 다른 외곽 유형 → 큰 특수실 별채 → 현관 방향 바꾸기."""
    first = layout.typology if layout.typology not in (None, "auto") else auto_typology(layout)
    single = len(layout.floors) == 1
    special = _big_special(layout) if single else []
    if not single:
        typs = ["L"]
    elif layout.outline_polygon().area > 200:
        typs = ["U", "L", "rect"] if special else ["U", "courtyard", "L", "rect"]  # 큰 수영장은 중정 둘레에 안 들어간다
    else:
        typs = ["L", "rect"]
    opts = [{"typology": t, "entry_side": "S", "seed": 0} for t in typs if t != first]
    if special:
        opts.insert(1, {"typology": "rect", "entry_side": "S", "seed": 0, "annex_special": True})
    opts += [{"typology": first, "entry_side": side, "seed": k + 1} for k, side in enumerate(("E", "W"))]
    return opts


def solve_layout(
    layout: Layout,
    typology: str | None = None,
    entry_side: str = "S",
    seed: int = 0,
    time_limit: float | None = None,
    annex_special: bool = False,
) -> tuple[Layout, list[str]]:
    """좌표가 없는 층을 모두 배치한 레이아웃. 실패하면 SolverFailed.

    annex_special=True 면 큰 특수실(수영장·차고, 20㎡ 이상)을 별채로 빼고 본채만 배치한다 (후보안 비교용).
    """
    out = layout.model_copy(deep=True)
    notes: list[str] = []
    todo = [f for f in out.floors if f.rooms and not any(r.shape() is not None for r in f.rooms)]
    if not todo:
        return out, notes
    all_program = len(todo) == len([f for f in out.floors if f.rooms])
    if annex_special:
        big = {r.id for r in _big_special(out)}
        for f in todo:
            f.rooms = [r.model_copy(update={"placement": "annex"}) if r.id in big else r for r in f.rooms]

    # 별채로 뺄 방
    annex = [r for f in todo for r in f.rooms if r.placement == "annex"]
    if annex:
        for f in todo:
            f.rooms = [r for r in f.rooms if r.placement != "annex"]
        out.annex = (out.annex or []) + annex
        notes.append("별채로 뺀 방(본채 도면에는 없음): " + ", ".join(f"{r.label} {_target(r):g}㎡" for r in annex))

    # 외곽 유형
    base = out.outline_polygon()
    if all_program and isinstance(out.outline, Rect):
        typ = typology or (out.typology if out.typology not in (None, "auto") else auto_typology(out))
        rooms = [r for f in todo for r in f.rooms]
        special = sum(_target(r) for r in rooms if room_zone(r) == "special" and room_kind(r) != "court"
                      and r.placement != "main") if len(todo) == 1 else 0.0
        gross = out.outline.w * out.outline.h - sum(_target(r) for r in annex)
        keep = out.outline if typ == "rect" and not annex else None
        shape = make_shape(typ, gross, special, (out.outline.x, out.outline.y), keep)
        if shape is None:
            notes.append(f"{TYPOLOGY_KO.get(typ, typ)} 외곽을 만들기엔 면적이 작아 일자형으로 바꿨다")
            shape = make_shape("rect", gross, 0.0, (out.outline.x, out.outline.y), out.outline if not annex else None)
        if out.site is not None and not out.site.polygon().buffer(0.01).covers(shape.polygon):
            notes.append(f"{TYPOLOGY_KO.get(shape.typology)} 외곽이 대지를 벗어나 원래 외곽을 썼다")
            shape = shape_from_outline(base)
        out.outline = shape.outline_value()
        out.typology = shape.typology if shape.typology in TYPOLOGY_KO else out.typology
        if shape.typology != "rect" or typ != "rect":
            b = shape.polygon.bounds
            notes.append(f"외곽 유형: {TYPOLOGY_KO.get(shape.typology, shape.typology)} — {b[2] - b[0]:.1f}×{b[3] - b[1]:.1f}m 안 바닥 {shape.polygon.area:.1f}㎡")
    else:
        if isinstance(out.outline, Rect):
            shape = make_shape("rect", 0, keep=out.outline)
        else:
            shape = shape_from_outline(base)

    multi = len([f for f in out.floors if f.rooms]) > 1
    stair = out.stair
    if stair is None:
        for f in out.floors:
            s = next((r for r in f.rooms if r.type == "stair" and r.rect), None)
            if s:
                stair = s.rect
                break
    lowest = min(f.level for f in out.floors)
    for floor in sorted(todo, key=lambda f: f.level):
        floor_shape = shape
        if floor.outline is not None:
            floor_shape = make_shape("rect", 0, keep=floor.outline) if isinstance(floor.outline, Rect) else shape_from_outline(Polygon(floor.outline))
        kw = dict(lowest=floor.level == lowest, stair_rect=stair if multi else None, place_stair=multi and stair is None,
                  entry_side=entry_side, seed=seed, time_limit=time_limit)
        plan = plan_floor(floor, floor_shape, **kw)
        floor.rooms = plan.rooms
        floor.doors = plan.doors
        notes += [n if n.startswith(floor.label) else f"{floor.label}: {n}" for n in plan.notes]
        if multi and stair is None:
            stair = next((r.rect for r in floor.rooms if r.type == "stair"), None)
    if multi and stair is not None:
        out.stair = stair
    return out, notes


_CACHE: dict[str, tuple[str, list[str]]] = {}
_CACHE_MAX = 64


def solve_layout_cached(layout: Layout, **kw) -> tuple[Layout, list[str]]:
    """같은 프로그램·옵션이면 저장해 둔 결과를 쓴다 (하네스가 단계마다 같은 JSON 을 다시 검사해도 한 번만 푼다).

    CP-SAT 는 시간 제한 안에서 병렬로 풀어 실행마다 답이 조금 다를 수 있다 → 캐시가 같은 입력 = 같은 결과를 보장한다.
    """
    from core.floorplan import dump_layout

    key = json.dumps([dump_layout(layout), sorted(kw.items())], ensure_ascii=False)
    hit = _CACHE.get(key)
    if hit is None:
        out, notes = solve_layout(layout, **kw)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        hit = _CACHE[key] = (out.model_dump_json(), notes)
    return Layout.model_validate_json(hit[0]), list(hit[1])
