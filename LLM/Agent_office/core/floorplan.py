"""평면 레이아웃 — 설계 팀의 '원본 데이터' (스키마 · 방 나누기 · 문/창 배치 · 벽 형상 · 검증).

LLM 은 좌표 계산을 자주 틀린다 (방이 겹치거나 외곽을 벗어나고, 면적 합이 맞지 않음).
그래서 연구(AI-Architect, Text-to-Layout, Text2BIM, ProCAD)의 공통 결론대로 역할을 나눈다.
  LLM  : 구조화된 레이아웃 JSON 만 쓴다 — 방 목록(종류·이름·목표 면적)·앞/뒤 구역·연결(문) = '위상(topology)'
  코드 : 좌표가 없으면 방을 직사각형으로 나누고(partition), 문·창을 공유 벽 위에 놓고, 벽을 만들고, 검증한다
         → 렌더링(DXF·PNG·SVG, core/floorplan_render.py)과 3D(core/floorplan_3d.py)도 모두 이 JSON 에서 나온다.
결과마다 레이아웃 JSON 을 함께 저장해, 나중에 수정할 때 DXF 를 다시 읽지 않고 원본 JSON 을 고친다.

좌표계 (단위 m)
  x → 동(오른쪽), y → 북(위). 원점 = 건물 외곽의 왼쪽 아래(남서) 모서리.
  y=0 쪽(S)이 앞(도로·현관 쪽), 방위 N/E/S/W 는 도면의 위/오른쪽/아래/왼쪽.

JSON 예 (좌표 없이 '프로그램'만 — 코드가 나눈다)
  {"title": "2층 단독주택", "outline": {"w": 12, "h": 9},
   "floors": [{"level": 1, "rooms": [
       {"id": "entry", "type": "entrance", "name": "현관", "area": 4},
       {"id": "living", "type": "living", "name": "거실", "area": 28}, ...]},
     {"level": 2, "rooms": [...]}]}
"""

import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union

Side = Literal["N", "E", "S", "W"]
EPS = 0.01  # 좌표 비교 허용 오차 (m)
SNAP = 0.05  # 두 방이 '맞닿았다'고 볼 간격 (m) — LLM 좌표의 작은 틈 허용

# 방 종류 → (한국어 이름, 최소 면적 ㎡, 최소 폭 m, 분할 구역 front/back/circulation)
# 최소 치수는 개념 설계용 경험값 (건축법 기준이 아님 — 실제 설계는 건축사 검토 필요)
ROOM_TYPES: dict[str, tuple[str, float, float, str]] = {
    "living": ("거실", 10.0, 3.0, "front"),
    "kitchen": ("주방", 4.0, 1.8, "front"),
    "dining": ("식당", 5.0, 2.1, "front"),
    "entrance": ("현관", 1.5, 1.0, "front"),
    "office": ("서재", 5.0, 2.1, "front"),
    "bedroom": ("침실", 7.0, 2.4, "back"),
    "bathroom": ("욕실", 2.5, 1.2, "back"),
    "utility": ("다용도실", 1.5, 1.0, "back"),
    "storage": ("수납", 0.8, 0.6, "back"),
    "dressroom": ("드레스룸", 2.0, 1.2, "back"),
    "balcony": ("발코니", 1.5, 0.9, "front"),
    "corridor": ("복도", 0.0, 0.9, "circulation"),
    "stair": ("계단", 3.0, 0.9, "circulation"),
    "other": ("기타", 1.0, 0.6, "front"),
}
RoomType = Literal[
    "living", "kitchen", "dining", "entrance", "office", "bedroom", "bathroom", "utility",
    "storage", "dressroom", "balcony", "corridor", "stair", "other",
]
CIRCULATION = {"corridor", "stair"}
HABITABLE = {"living", "kitchen", "dining", "office", "bedroom", "other"}  # 창이 있어야 하는 방
# 자동 문 배치 때 이웃 방을 고르는 우선순위 (앞일수록 먼저 연결)
CONNECT_PRIORITY = ["corridor", "stair", "living", "dining", "entrance", "kitchen", "other", "office", "utility",
                    "bedroom", "dressroom", "bathroom", "storage", "balcony"]
DEFAULT_CORRIDOR = 1.2  # 자동 분할 복도 폭 (m)
DEFAULT_STAIR = (3.6, 1.2)  # 자동 계단 크기 (길이 × 폭, 곧은 계단)
OUTSIDE = "outside"


# ════════════════════════════════════════════════════════════════════
# 스키마
# ════════════════════════════════════════════════════════════════════


class Rect(BaseModel):
    """축에 나란한 직사각형 (x, y = 왼쪽 아래 모서리)."""

    x: float = 0.0
    y: float = 0.0
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    def polygon(self) -> Polygon:
        return box(self.x, self.y, self.x + self.w, self.y + self.h)


Point2 = tuple[float, float]


def _to_polygon(shape: "Rect | list[Point2] | None") -> Polygon | None:
    if shape is None:
        return None
    if isinstance(shape, Rect):
        return shape.polygon()
    return Polygon(shape)


class Room(BaseModel):
    id: str
    type: RoomType = "other"
    name: str = ""  # 비우면 종류의 한국어 이름
    area: float | None = Field(default=None, description="목표 면적 ㎡ (없으면 면적 검사 생략)")
    rect: Rect | None = None
    polygon: list[Point2] | None = None  # 직사각형이 아닌 방 (가져온 도면 등)
    zone: Literal["front", "back"] | None = None  # 자동 분할 때 앞(도로 쪽)/뒤 구역 지정 (없으면 종류 기본값)

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, v):
        return v if v in ROOM_TYPES else "other"  # 모르는 종류는 '기타'로 (LLM 이 새 종류를 지어내도 깨지지 않게)

    @property
    def label(self) -> str:
        return self.name or ROOM_TYPES[self.type][0]

    def shape(self) -> Polygon | None:
        if self.polygon:
            return Polygon(self.polygon)
        return self.rect.polygon() if self.rect else None


class Door(BaseModel):
    """문 = 벽 참조(방 + 방위) + 오프셋 + 폭. to 는 연결되는 방 id 또는 'outside'(외부 출입문)."""

    room: str
    to: str = OUTSIDE
    side: Side | None = None  # 없으면 코드가 두 방이 맞닿은 벽을 찾는다
    offset: float | None = Field(default=None, description="그 벽의 왼쪽/아래 끝에서 문 시작점까지 m (없으면 가운데)")
    width: float = Field(default=0.9, gt=0)


class Window(BaseModel):
    room: str
    side: Side | None = None  # 없으면 가장 긴 외벽
    offset: float | None = None
    width: float = Field(default=1.2, gt=0)
    sill: float = 0.9  # 창대 높이 (3D 용)
    height: float = 1.2


class WallSpec(BaseModel):
    exterior: float = Field(default=0.2, gt=0)  # 외벽 두께 m
    interior: float = Field(default=0.12, gt=0)  # 내벽 두께 m


class Floor(BaseModel):
    level: int = 1  # 1 = 1층, 0 이하 = 지하
    name: str = ""
    height: float = Field(default=2.8, gt=0)  # 층고 m
    outline: Rect | list[Point2] | None = None  # 없으면 건물 외곽(layout.outline)
    rooms: list[Room] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)  # 비우면 코드가 자동 배치
    windows: list[Window] = Field(default_factory=list)  # 비우면 코드가 자동 배치

    @property
    def label(self) -> str:
        return self.name or (f"{self.level}층" if self.level > 0 else f"지하{1 - self.level}층")


class Layout(BaseModel):
    version: int = 1
    title: str = ""
    units: Literal["m"] = "m"
    site: Rect | None = None  # 대지 (선택)
    outline: Rect | list[Point2]  # 건물 외곽
    wall: WallSpec = Field(default_factory=WallSpec)
    stair: Rect | None = None  # 모든 층이 같이 쓰는 계단 위치 (없으면 자동)
    floors: list[Floor] = Field(min_length=1)
    three_d: bool = False  # 3D 모델도 만들지
    area_tolerance: float = 0.2  # 목표 면적 허용 오차 (0.2 = ±20%)
    assumptions: list[str] = Field(default_factory=list)  # 정보가 없어 가정한 것 (대지 크기 등)
    needs_check: list[str] = Field(default_factory=list)  # 확인 필요 (이미지에서 읽은 치수 등)

    @model_validator(mode="after")
    def _sort_floors(self):
        self.floors.sort(key=lambda f: f.level)
        return self

    def outline_polygon(self, floor: Floor | None = None) -> Polygon:
        return _to_polygon(floor.outline if floor and floor.outline is not None else self.outline)


# ════════════════════════════════════════════════════════════════════
# JSON 읽기 (LLM 답변 → Layout)
# ════════════════════════════════════════════════════════════════════

JSON_BLOCK = re.compile(r"```[ \t]*json[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class LayoutParseError(ValueError):
    """레이아웃 JSON 이 없거나 형식이 틀림 (메시지는 그대로 LLM 피드백으로 쓴다)."""


def find_layout_json(text: str) -> dict | None:
    """답변 속 ```json 블록 중 'floors' 를 가진 마지막 블록 (수정 과정에서 여러 번 썼으면 마지막이 최종)."""
    found = None
    for src in JSON_BLOCK.findall(text):
        try:
            data = json.loads(src)
        except json.JSONDecodeError as e:
            if '"floors"' in src:
                raise LayoutParseError(f"레이아웃 JSON 파싱 오류: {e.msg} (줄 {e.lineno}, 칸 {e.colno})") from e
            continue
        if isinstance(data, dict) and "floors" in data:
            found = data
    return found


def schema_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:8]:
        where = ".".join(str(x) for x in err["loc"]) or "(최상위)"
        parts.append(f"{where}: {err['msg']}")
    return "레이아웃 JSON 형식 오류 — " + " / ".join(parts)


def parse_layout(data: dict | str) -> Layout:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise LayoutParseError(f"레이아웃 JSON 파싱 오류: {e.msg} (줄 {e.lineno}, 칸 {e.colno})") from e
    try:
        return Layout.model_validate(data)
    except ValidationError as e:
        raise LayoutParseError(schema_errors(e)) from e


def layout_from_text(text: str) -> Layout | None:
    """답변에서 레이아웃을 꺼낸다. JSON 블록이 없으면 None, 있는데 틀리면 LayoutParseError."""
    data = find_layout_json(text)
    return parse_layout(data) if data is not None else None


def dump_layout(layout: Layout) -> str:
    return layout.model_dump_json(indent=2, exclude_none=True)


# ════════════════════════════════════════════════════════════════════
# 자동 분할 (프로그램 → 직사각형 방) — 같은 입력이면 항상 같은 결과
# ════════════════════════════════════════════════════════════════════


def _r(v: float) -> float:
    return round(v, 2)


def _zone(room: Room) -> str:
    return room.zone or ROOM_TYPES[room.type][3]


def _target(room: Room) -> float:
    return room.area if room.area and room.area > 0 else max(ROOM_TYPES[room.type][1], 4.0)


def _balance(rooms: list[Room], front_share: float = 0.5) -> tuple[list[Room], list[Room]]:
    """한쪽 구역이 비면 순서를 지키며 면적 비율(front_share)대로 앞/뒤로 나눈다 (복도 양쪽에 방)."""
    total, acc = sum(_target(r) for r in rooms), 0.0
    for i, room in enumerate(rooms):
        if acc + _target(room) / 2 > total * front_share and i > 0:
            return rooms[:i], rooms[i:]
        acc += _target(room)
    return rooms[:-1], rooms[-1:]


def _min_w(room: Room) -> float:
    return ROOM_TYPES[room.type][2]


def _fill_band(rooms: list[Room], x0: float, x1: float, y0: float, y1: float, corridor_side: str) -> list[Room]:
    """띠(band) 하나를 왼쪽부터 방 너비로 나눈다 (너비 = 목표 면적 비례).

    너비가 최소 폭보다 좁거나 너무 길쭉한 작은 방은 다음 작은 방과 위아래로 쌓는다
    (예: 욕실 + 다용도실 → 한 칸에 2개). 앞의 방이 복도 쪽에 온다.
    """
    if not rooms:
        return []
    depth, width = y1 - y0, x1 - x0
    scale = width * depth / sum(_target(r) for r in rooms)
    areas = [_target(r) * scale for r in rooms]

    def slender(k: int) -> bool:
        w = areas[k] / depth
        return w < _min_w(rooms[k]) or (areas[k] < 8 and depth / w > 2.2)

    columns: list[list[int]] = []
    i = 0
    while i < len(rooms):
        j = i + 1
        if j < len(rooms) and slender(i) and areas[j] < 12:
            col_w = (areas[i] + areas[j]) / depth
            heights = [depth * areas[k] / (areas[i] + areas[j]) for k in (i, j)]
            if all(min(col_w, h) >= _min_w(rooms[k]) * 0.95 for k, h in zip((i, j), heights)):
                columns.append([i, j])
                i += 2
                continue
        columns.append([i])
        i += 1

    placed: list[Room] = []
    acc = 0.0
    for col in columns:
        col_area = sum(areas[k] for k in col)
        left, acc = x0 + acc, acc + col_area / depth
        right = x1 if col is columns[-1] else x0 + acc
        left, right = _r(left), _r(right)
        if len(col) == 1:
            placed.append(rooms[col[0]].model_copy(update={"rect": Rect(x=left, y=_r(y0), w=_r(right - left), h=_r(depth))}))
            continue
        h_near = _r(depth * areas[col[0]] / col_area)
        near, far = rooms[col[0]], rooms[col[1]]
        if corridor_side == "top":
            near_rect = Rect(x=left, y=_r(y1 - h_near), w=_r(right - left), h=h_near)
            far_rect = Rect(x=left, y=_r(y0), w=_r(right - left), h=_r(y1 - h_near - y0))
        else:
            near_rect = Rect(x=left, y=_r(y0), w=_r(right - left), h=h_near)
            far_rect = Rect(x=left, y=_r(y0 + h_near), w=_r(right - left), h=_r(y1 - y0 - h_near))
        placed += [near.model_copy(update={"rect": near_rect}), far.model_copy(update={"rect": far_rect})]
    return placed


def _band_min(rooms: list[Room], extra: float = 0.0) -> float:
    return max([2.4, extra, *[_min_w(r) + 0.3 for r in rooms]])


def partition(
    outline: Rect,
    rooms: list[Room],
    stair: Rect | None = None,
    band_split: float | None = None,
    corridor_width: float = DEFAULT_CORRIDOR,
    need_stair: bool = False,
) -> tuple[list[Room], Rect | None, float | None, list[str]]:
    """방 프로그램(종류·목표 면적·순서) + 직사각형 외곽 → 직사각형 방 배치.

    배치 규칙 (단순하지만 항상 같은 결과)
      [계단][뒤 구역: 침실·욕실 …]   ← 북쪽
      [복도 ───────────][현관]      ← 복도 띠 (현관은 오른쪽 끝, 외벽에 닿음)
      [앞 구역: 거실·주방 …]        ← 남쪽 (도로·채광)
      - 구역 안에서는 목록 순서대로 왼쪽부터, 너비 = 목표 면적 비례 → 방 순서가 곧 '이웃 관계'
      - 모든 방이 복도에 닿는다 (작은 방 2개를 쌓은 칸은 복도 쪽 방을 거쳐 들어감)
      - 한쪽 구역이 비면 면적이 반반이 되게 나눈다. 구역 깊이는 방 종류별 최소 폭 이상으로 맞춘다
      - stair·band_split 을 주면 그 위치에 맞춘다 (위층이 아래층과 계단·복도 위치를 맞추도록)

    돌려주는 값 = (배치된 방, 계단 위치, 앞 구역 깊이(band_split), 메모)
    """
    notes: list[str] = []
    x0, y0, W, D = outline.x, outline.y, outline.w, outline.h
    explicit_corr = [r for r in rooms if r.type == "corridor"]
    explicit_stair = [r for r in rooms if r.type == "stair"]
    program = [r for r in rooms if r.type not in CIRCULATION]
    need_stair = need_stair or bool(explicit_stair) or stair is not None
    if not program:
        raise ValueError("나눌 방이 없다 (복도·계단 외의 방을 1개 이상 넣어라)")
    if len(program) == 1 and not need_stair:
        only = program[0].model_copy(update={"rect": Rect(x=x0, y=y0, w=W, h=D)})
        return [only], None, None, notes

    entry = next((r for r in program if r.type == "entrance" and r.zone is None), None)
    rest = [r for r in program if r is not entry]
    front = [r for r in rest if _zone(r) == "front"]
    back = [r for r in rest if _zone(r) == "back"]
    if not front or not back:
        share = 0.5
        if band_split is not None:  # 띠 위치가 정해져 있으면 (위층) 두 띠의 넓이 비율대로
            back_area = W * (D - band_split - corridor_width) - (stair.w * stair.h if stair else 0)
            share = W * band_split / max(W * band_split + back_area, 1.0)
        front, back = _balance(front or back, share) if len(rest) > 1 else (rest, [])

    if stair is not None:
        if band_split is None:
            band_split = stair.y - y0 - corridor_width
        corridor_width = stair.y - y0 - band_split
        if band_split < 1.0 or corridor_width < 0.8:
            raise ValueError(
                "자동 분할은 계단이 뒤쪽(북쪽) 구역에 있을 때만 맞출 수 있다 — layout.stair 를 고치거나 방 좌표(rect)를 직접 써라"
            )
    if band_split is None:
        stair_area = 6.0 if need_stair else 0.0
        tf = sum(_target(r) for r in front) or 1.0
        tb = sum(_target(r) for r in back) + stair_area
        band_split = (D - corridor_width) * tf / (tf + tb)
        lo = _band_min(front)
        hi = D - corridor_width - _band_min(back, 2.7 if need_stair else 0.0)
        band_split = _r(min(max(band_split, lo), hi) if lo <= hi else (D - corridor_width) / 2)
    yc0, yc1 = _r(y0 + band_split), _r(y0 + band_split + corridor_width)
    back_d = _r(y0 + D - yc1)
    if need_stair and stair is None:
        sw = 1.2 if back_d >= 3.8 else 2.4  # 깊으면 곧은 계단(폭 1.2), 얕으면 꺾인 계단(폭 2.4)
        stair = Rect(x=x0, y=yc1, w=sw, h=back_d)

    # 뒤 구역 = 띠 − 계단. 계단이 띠 깊이보다 짧으면 남는 부분은 수납으로
    back_x0, back_x1 = x0, x0 + W
    extra: list[Room] = []
    if stair is not None:
        if stair.x <= x0 + EPS:
            back_x0 = _r(stair.x + stair.w)
        elif stair.x + stair.w >= x0 + W - EPS:
            back_x1 = _r(stair.x)
        else:
            raise ValueError("자동 분할은 계단이 외곽 왼쪽 또는 오른쪽 끝에 있을 때만 맞출 수 있다 — 방 좌표(rect)를 직접 써라")
        left_over = y0 + D - (stair.y + stair.h)
        if left_over > 0.6:
            extra.append(Room(id="stair_store", type="storage", name="계단 위 수납",
                              rect=Rect(x=stair.x, y=_r(stair.y + stair.h), w=stair.w, h=_r(left_over))))

    # 복도 띠 오른쪽 끝 = 현관 (외벽에 닿아 바깥 출입문을 둔다)
    corr_x1 = x0 + W
    if entry is not None:
        length = _r(min(max(_target(entry) / corridor_width, 1.5), W / 3))
        corr_x1 = _r(x0 + W - length)
        extra.append(entry.model_copy(update={"rect": Rect(x=corr_x1, y=yc0, w=length, h=_r(corridor_width))}))

    total_target = sum(_target(r) for r in program)
    usable = W * D - (corr_x1 - x0) * corridor_width - (stair.w * stair.h if stair else 0)
    if abs(total_target - usable) / usable > 0.1:
        notes.append(f"목표 면적 합 {total_target:.1f}㎡ ↔ 복도·계단을 뺀 가용 면적 {usable:.1f}㎡ → 비례로 늘이거나 줄였다")

    for name, rooms_in, cap in (("앞(남)", front, W * band_split), ("뒤(북)", back, (back_x1 - back_x0) * back_d)):
        want = sum(_target(r) for r in rooms_in)
        if want and abs(cap - want) / want > 0.2:
            notes.append(f"{name} 구역 넓이 {cap:.1f}㎡ ↔ 그 구역 방 목표 합 {want:.1f}㎡ — 방의 zone(front/back)을 바꿔 균형을 맞춰라")
    placed = _fill_band(front, x0, x0 + W, y0, yc0, "top") + _fill_band(back, back_x0, back_x1, yc1, y0 + D, "bottom")
    placed += extra
    corr = explicit_corr[0] if explicit_corr else Room(id="hall", type="corridor")
    placed.append(corr.model_copy(update={"rect": Rect(x=x0, y=yc0, w=_r(corr_x1 - x0), h=_r(corridor_width)), "area": None}))
    if stair is not None:
        base = explicit_stair[0] if explicit_stair else Room(id="stair", type="stair")
        placed.append(base.model_copy(update={"rect": stair, "area": None}))
    return placed, stair, band_split, notes


# ════════════════════════════════════════════════════════════════════
# 벽 위의 선분 다루기 (문·창 배치)
# ════════════════════════════════════════════════════════════════════

Segment = tuple[Point2, Point2]


def _straight_segments(geom) -> list[Segment]:
    """선 형상을 꺾이지 않는 직선 조각으로 (같은 방향으로 이어진 조각은 합친다)."""
    if geom.is_empty:
        return []
    lines = []
    if isinstance(geom, LineString):
        lines = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        lines = [g for g in geom.geoms if isinstance(g, LineString)]
    if not lines:
        return []
    merged = linemerge(lines) if len(lines) > 1 else lines[0]
    merged_lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    segs: list[Segment] = []
    for line in merged_lines:
        coords = list(line.coords)
        start = coords[0]
        for a, b, c in zip(coords, coords[1:], coords[2:]):
            v1, v2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
            if abs(v1[0] * v2[1] - v1[1] * v2[0]) > 1e-6:  # 꺾임
                segs.append((start, b))
                start = b
        segs.append((start, coords[-1]))
    return [s for s in segs if _seg_len(s) > EPS]


def _seg_len(s: Segment) -> float:
    return ((s[1][0] - s[0][0]) ** 2 + (s[1][1] - s[0][1]) ** 2) ** 0.5


def _ordered(s: Segment) -> Segment:
    """왼쪽/아래 끝이 먼저 오도록."""
    return s if (s[0][0], s[0][1]) <= (s[1][0], s[1][1]) else (s[1], s[0])


def _side_of(seg: Segment, poly: Polygon) -> Side:
    (x1, y1), (x2, y2) = seg
    c = poly.centroid
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(x2 - x1) >= abs(y2 - y1):
        return "N" if my > c.y else "S"
    return "E" if mx > c.x else "W"


def room_side(poly: Polygon, side: Side) -> Segment | None:
    """방의 한쪽 벽 (그 방위를 향한 가장 긴 경계 직선)."""
    candidates = [s for s in _straight_segments(poly.exterior) if _side_of(s, poly) == side]
    horizontal = side in ("N", "S")
    candidates = [s for s in candidates if (abs(s[1][0] - s[0][0]) >= abs(s[1][1] - s[0][1])) == horizontal]
    return _ordered(max(candidates, key=_seg_len)) if candidates else None


def _point_along(seg: Segment, dist: float) -> Point2:
    (x1, y1), (x2, y2) = seg
    length = _seg_len(seg)
    t = dist / length if length else 0
    return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)


def shared_wall(a: Polygon, b: Polygon | None, outline: Polygon | None = None) -> list[Segment]:
    """두 방(또는 방과 외곽)이 맞닿은 벽의 직선 조각들 — 작은 틈(SNAP)은 맞닿은 것으로 본다."""
    other = b.buffer(SNAP, join_style="mitre") if b is not None else outline.exterior.buffer(SNAP)
    segs = _straight_segments(a.exterior.intersection(other))
    segs = [s for s in segs if _seg_len(s) > SNAP * 2 + EPS]  # 모서리에서 버퍼에 살짝 걸친 짧은 꼬리 제거
    return [_ordered(s) for s in segs]


def opening_segment(poly: Polygon, side: Side, offset: float | None, width: float, within: Segment | None = None) -> Segment | None:
    """방의 side 벽 위, 벽 왼쪽/아래 끝에서 offset 만큼 떨어진 곳의 폭 width 선분."""
    wall = room_side(poly, side)
    if wall is None:
        return None
    if offset is None:
        base = within or wall
        # 공유 구간 가운데 → 방 벽 기준 오프셋으로 바꾼다
        start_on_wall = Point(base[0]).distance(Point(wall[0]))
        offset = start_on_wall + (_seg_len(base) - width) / 2
    return (_point_along(wall, offset), _point_along(wall, offset + width))


def _seg_on(seg: Segment, target, tol: float = SNAP + EPS) -> bool:
    return target.buffer(tol).covers(LineString(seg))


# ════════════════════════════════════════════════════════════════════
# 레이아웃 완성 (자동 분할 → 문·창 자동 배치)
# ════════════════════════════════════════════════════════════════════


def _polys(floor: Floor) -> dict[str, Polygon]:
    return {r.id: r.shape() for r in floor.rooms if r.shape() is not None}


def _auto_doors(floor: Floor, outline: Polygon, lowest: bool) -> tuple[list[Door], list[str]]:
    """공유 벽을 따라 문을 놓는다: 복도·계단부터 이웃으로 퍼져 나가며 모든 방을 잇는다 (너비 우선)."""
    polys = _polys(floor)
    rooms = {r.id: r for r in floor.rooms}
    notes: list[str] = []
    doors: list[Door] = []

    def add(a: str, b: str, width: float) -> bool:
        segs = shared_wall(polys[a], polys[b])
        segs = [s for s in segs if _seg_len(s) >= width + 0.2]
        if not segs:
            return False
        seg = max(segs, key=_seg_len)
        side = _side_of(seg, polys[a])
        wall = room_side(polys[a], side)
        if wall is None:
            return False
        offset = _r(Point(seg[0]).distance(Point(wall[0])) + (_seg_len(seg) - width) / 2)
        doors.append(Door(room=a, to=b, side=side, offset=offset, width=width))
        return True

    def door_width(room_type: str) -> float:
        return 0.8 if room_type in ("bathroom", "storage", "dressroom") else 0.9

    circulation = [r.id for r in floor.rooms if r.type in CIRCULATION]
    connected: list[str] = []
    if circulation:
        connected.append(circulation[0])
        for cid in circulation[1:]:  # 복도 조각·계단끼리
            for other in list(connected):
                if add(cid, other, 1.0):
                    connected.append(cid)
                    break
            else:
                connected.append(cid)  # 이어지지 않으면 검증에서 '도달 불가'로 드러난다
    else:
        seed = next((r.id for t in ("entrance", "living") for r in floor.rooms if r.type == t), floor.rooms[0].id)
        connected.append(seed)

    # 아직 이어지지 않은 방 × 이어진 이웃 중 우선순위가 가장 좋은 쌍부터 하나씩 잇는다 (복도 > 계단 > 거실 > …)
    order = {t: i for i, t in enumerate(CONNECT_PRIORITY)}
    while True:
        pairs = [
            (order.get(rooms[nb].type, 99), i, room.id, nb)
            for i, room in enumerate(floor.rooms) if room.id not in connected and room.id in polys
            for nb in connected
        ]
        for _, _, rid, nb in sorted(pairs):
            if add(rid, nb, door_width(rooms[rid].type)):
                connected.append(rid)
                break
        else:
            break

    if lowest:  # 맨 아래층: 외부 출입문 (현관 → 거실 → 첫 방, 남쪽 우선)
        for t in ("entrance", "living", None):
            cand = [r for r in floor.rooms if (t is None or r.type == t) and r.id in polys]
            placed = False
            for room in cand:
                segs = [s for s in shared_wall(polys[room.id], None, outline) if _seg_len(s) >= 1.2]
                segs.sort(key=lambda s: (_side_of(s, polys[room.id]) != "S", -_seg_len(s)))
                if segs:
                    seg = segs[0]
                    side = _side_of(seg, polys[room.id])
                    wall = room_side(polys[room.id], side)
                    offset = _r(Point(seg[0]).distance(Point(wall[0])) + (_seg_len(seg) - 1.0) / 2)
                    doors.append(Door(room=room.id, to=OUTSIDE, side=side, offset=offset, width=1.0))
                    placed = True
                    break
            if placed:
                break
        else:
            notes.append(f"{floor.label}: 외벽에 닿은 방이 없어 외부 출입문을 놓지 못했다")
    return doors, notes


def _auto_windows(floor: Floor, outline: Polygon) -> list[Window]:
    """방마다 외벽 하나에 창 (거주 공간은 남 → 동·서 → 북 순서, 문과 겹치지 않는 벽)."""
    doors = [LineString(seg).buffer(0.3) for d in floor.doors if (seg := door_segment(floor, d))]
    windows: list[Window] = []
    pref = {"S": 0, "E": 1, "W": 1, "N": 2}
    for room in floor.rooms:
        poly = room.shape()
        if poly is None or room.type in CIRCULATION or room.type in ("storage", "entrance"):
            continue
        small = room.type in ("bathroom", "utility", "dressroom")
        segs = sorted(shared_wall(poly, None, outline),
                      key=lambda s: (pref[_side_of(s, poly)] if room.type in HABITABLE else 0, -_seg_len(s)))
        for seg in segs:
            width = min(0.6 if small else (2.4 if room.type == "living" else 1.5), _seg_len(seg) - 0.6)
            if width < 0.4:
                continue
            side = _side_of(seg, poly)
            wall = room_side(poly, side)
            if wall is None:
                continue
            offset = _r(Point(seg[0]).distance(Point(wall[0])) + (_seg_len(seg) - width) / 2)
            cand = opening_segment(poly, side, offset, width)
            if cand and any(LineString(cand).intersects(d) for d in doors):
                continue
            windows.append(Window(room=room.id, side=side, offset=offset, width=_r(width),
                                  sill=1.5 if small else 0.9, height=0.6 if small else 1.2))
            break
    return windows


def resolve(layout: Layout) -> tuple[Layout, list[str]]:
    """좌표가 없는 층은 자동 분할하고, 문·창이 비어 있으면 자동 배치한 '완성 레이아웃'을 만든다.

    좌표를 이미 가진 층·직접 쓴 문/창은 그대로 둔다 → 사람이·LLM 이 고친 부분이 보존된다.
    완성본을 저장해 두면 다음 수정은 이 좌표에서 시작한다 (DXF 를 다시 해석하지 않음).
    """
    out = layout.model_copy(deep=True)
    notes: list[str] = []
    multi = len(out.floors) > 1
    stair = out.stair
    if stair is None:  # 좌표가 있는 층에 계단이 있으면 그 위치를 공통 계단으로
        for f in out.floors:
            s = next((r for r in f.rooms if r.type == "stair" and r.rect), None)
            if s:
                stair = s.rect
                break
    band_split = None
    for floor in out.floors:
        has = [r.shape() is not None for r in floor.rooms]
        if floor.rooms and not any(has):
            outline = floor.outline if floor.outline is not None else out.outline
            if not isinstance(outline, Rect):
                b = Polygon(outline).bounds
                outline = Rect(x=b[0], y=b[1], w=b[2] - b[0], h=b[3] - b[1])
                notes.append(f"{floor.label}: 다각형 외곽이라 바운딩 박스로 나눴다 — 외곽 밖 부분은 검증에서 지적된다")
            rooms, stair, band_split, part_notes = partition(outline, floor.rooms, stair, band_split, need_stair=multi)
            floor.rooms = rooms
            notes += [f"{floor.label}: {n}" for n in part_notes]
        elif floor.rooms and not all(has):
            missing = [r.label for r, h in zip(floor.rooms, has) if not h]
            raise LayoutParseError(
                f"{floor.label}: 일부 방만 좌표(rect)가 있다 {missing}. 그 층은 모든 방에 rect 를 쓰거나, 모두 빼서 자동 분할하라"
            )
    if multi and stair is not None:
        out.stair = stair
    lowest = min(f.level for f in out.floors)
    for floor in out.floors:
        outline = out.outline_polygon(floor)
        if not floor.doors and floor.rooms:
            floor.doors, door_notes = _auto_doors(floor, outline, floor.level == lowest)
            notes += door_notes
        else:
            floor.doors = [_resolve_door(floor, d, outline) for d in floor.doors]
        if not floor.windows and floor.rooms:
            floor.windows = _auto_windows(floor, outline)
    return out, notes


def _resolve_door(floor: Floor, door: Door, outline: Polygon) -> Door:
    """방위·오프셋이 빠진 문은 공유 벽 가운데로 채운다 (찾지 못하면 그대로 두고 검증에서 지적)."""
    if door.side and door.offset is not None:
        return door
    polys = _polys(floor)
    a = polys.get(door.room)
    if a is None:
        return door
    b = polys.get(door.to) if door.to != OUTSIDE else None
    if door.to != OUTSIDE and b is None:
        return door
    segs = shared_wall(a, b, outline)
    if door.side:
        segs = [s for s in segs if _side_of(s, a) == door.side]
    segs = [s for s in segs if _seg_len(s) >= door.width]
    if not segs:
        return door
    seg = max(segs, key=_seg_len)
    side = door.side or _side_of(seg, a)
    wall = room_side(a, side)
    if wall is None:
        return door
    offset = door.offset if door.offset is not None else _r(Point(seg[0]).distance(Point(wall[0])) + (_seg_len(seg) - door.width) / 2)
    return door.model_copy(update={"side": side, "offset": offset})


# ════════════════════════════════════════════════════════════════════
# 형상 (렌더링·3D 공용)
# ════════════════════════════════════════════════════════════════════


class Opening(BaseModel):
    kind: Literal["door", "window"]
    room: str
    to: str = ""
    segment: Segment
    width: float
    sill: float = 0.0
    head: float = 2.1  # 문·창 윗면 높이


@dataclass
class FloorGeometry:
    floor: Floor
    outline: Polygon
    rooms: dict[str, Polygon]
    walls: BaseGeometry  # 개구부를 뺀 벽 (Multi)Polygon
    walls_full: BaseGeometry  # 개구부를 빼지 않은 벽
    wall_pieces: list[tuple[Segment, float]]  # (벽 중심선 직선, 두께) — IFC 용
    openings: list[Opening]


def door_segment(floor: Floor, door: Door) -> Segment | None:
    poly = _polys(floor).get(door.room)
    if poly is None or door.side is None:
        return None
    return opening_segment(poly, door.side, door.offset, door.width)


def window_segment(floor: Floor, window: Window, outline: Polygon) -> Segment | None:
    poly = _polys(floor).get(window.room)
    if poly is None:
        return None
    side = window.side
    if side is None:
        segs = shared_wall(poly, None, outline)
        if not segs:
            return None
        side = _side_of(max(segs, key=_seg_len), poly)
    return opening_segment(poly, side, window.offset, window.width)


def wall_lines(rooms: dict[str, Polygon], outline: Polygon) -> list[tuple[Segment, bool]]:
    """모든 방 경계 + 외곽을 겹치지 않는 직선 벽 중심선으로. (선분, 외벽 여부)"""
    lines = [outline.exterior] + [p.exterior for p in rooms.values()]
    noded = unary_union([LineString(l.coords) for l in lines])
    out = []
    for seg in _straight_segments(noded):
        out.append((_ordered(seg), _seg_on(seg, outline.exterior, 0.02)))
    return out


def floor_geometry(layout: Layout, floor: Floor) -> FloorGeometry:
    outline = layout.outline_polygon(floor)
    rooms = _polys(floor)
    pieces = []
    for seg, exterior in wall_lines(rooms, outline):
        t = layout.wall.exterior if exterior else layout.wall.interior
        pieces.append((seg, t))
    walls_full = unary_union([LineString(s).buffer(t / 2, cap_style="square", join_style="mitre") for s, t in pieces])
    openings: list[Opening] = []
    for d in floor.doors:
        seg = door_segment(floor, d)
        if seg:
            openings.append(Opening(kind="door", room=d.room, to=d.to, segment=seg, width=d.width, sill=0.0,
                                    head=2.1 if d.to != OUTSIDE else 2.2))
    for w in floor.windows:
        seg = window_segment(floor, w, outline)
        if seg:
            openings.append(Opening(kind="window", room=w.room, segment=seg, width=w.width, sill=w.sill,
                                    head=min(w.sill + w.height, floor.height - 0.2)))
    cut = layout.wall.exterior / 2 + 0.05
    holes = unary_union([LineString(o.segment).buffer(cut, cap_style="flat") for o in openings]) if openings else None
    walls = walls_full.difference(holes) if holes is not None else walls_full
    return FloorGeometry(floor=floor, outline=outline, rooms=rooms, walls=walls, walls_full=walls_full,
                         wall_pieces=pieces, openings=openings)


# ════════════════════════════════════════════════════════════════════
# 검증 — 에러 메시지는 그대로 하네스 피드백(LLM 에게 돌려줄 문장)이 된다
# ════════════════════════════════════════════════════════════════════


class Report(BaseModel):
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # 자동 분할·배치 기록

    @property
    def ok(self) -> bool:
        return not self.errors


def _min_width_ok(poly: Polygon, min_w: float) -> bool:
    """폭 검사: 안쪽으로 min_w/2 만큼 줄였을 때 남는 게 있어야 한다 (직사각형이 아닌 방에도 통함)."""
    return not poly.buffer(-(min_w / 2 - EPS), join_style="mitre").is_empty


def _short_side(poly: Polygon) -> float:
    rect = poly.minimum_rotated_rectangle
    xs = list(rect.exterior.coords)
    sides = [_seg_len((xs[i], xs[i + 1])) for i in range(2)]
    return min(sides)


def validate(layout: Layout, tolerance: float | None = None) -> Report:
    """완성 레이아웃(resolve 결과)을 검사한다.

    - 방이 외곽 안에 있고 서로 겹치지 않는가
    - 면적이 목표 ±허용 오차 안인가
    - 문을 따라 모든 방에 갈 수 있는가 (맨 아래층은 바깥에서, 위층은 계단에서 출발)
    - 여러 층이면 계단이 모든 층에 같은 자리에 있는가
    - 방 종류별 최소 면적·폭
    """
    rep = Report()
    tol = layout.area_tolerance if tolerance is None else tolerance
    levels = [f.level for f in layout.floors]
    if len(set(levels)) != len(levels):
        rep.errors.append(f"층 번호(level)가 겹친다: {levels}. 층마다 다른 level 을 써라")
    if layout.site is not None and not layout.site.polygon().buffer(EPS).covers(layout.outline_polygon()):
        rep.errors.append("건물 외곽(outline)이 대지(site)를 벗어난다. 외곽을 줄이거나 대지 크기를 확인하라")
    lowest = min(levels)
    stair_shapes: list[tuple[Floor, Polygon | None]] = []

    for floor in layout.floors:
        fl = floor.label
        outline = layout.outline_polygon(floor)
        if not outline.is_valid or outline.area <= 0:
            rep.errors.append(f"{fl}: 외곽 다각형이 올바르지 않다 (꼭짓점 순서·자기 교차 확인)")
            continue
        if not floor.rooms:
            rep.errors.append(f"{fl}: 방이 하나도 없다")
            continue
        ids = [r.id for r in floor.rooms]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            rep.errors.append(f"{fl}: 방 id 가 겹친다 {dup}. 층 안에서 id 는 서로 달라야 한다")
        polys: dict[str, Polygon] = {}
        area_errors: list[str] = []
        targets: list[tuple[float, float]] = []
        for room in floor.rooms:
            poly = room.shape()
            if poly is None:
                rep.errors.append(f"{fl}: '{room.label}'({room.id}) 에 좌표(rect/polygon)가 없다")
                continue
            if not poly.is_valid or poly.area <= 0:
                rep.errors.append(f"{fl}: '{room.label}'({room.id}) 다각형이 올바르지 않다")
                continue
            polys[room.id] = poly
            outside = poly.difference(outline).area
            if outside > 0.05:
                b = outline.bounds
                rep.errors.append(
                    f"{fl}: '{room.label}'({room.id}) 이 건물 외곽을 벗어난다 (벗어난 면적 {outside:.1f}㎡). "
                    f"외곽 x {b[0]:g}~{b[2]:g}m, y {b[1]:g}~{b[3]:g}m 안으로 옮겨라"
                )
            name, min_area, min_w, _ = ROOM_TYPES[room.type]
            if poly.area < min_area - EPS:
                rep.errors.append(f"{fl}: '{room.label}' 면적 {poly.area:.1f}㎡ — {name} 최소 {min_area:g}㎡ 보다 작다")
            if min_w and not _min_width_ok(poly, min_w):
                rep.errors.append(f"{fl}: '{room.label}' 폭 {_short_side(poly):.2f}m — {name} 최소 폭 {min_w:g}m 보다 좁다")
            if room.area and room.area > 0:
                diff = (poly.area - room.area) / room.area
                if abs(diff) > tol + 1e-9:
                    area_errors.append(
                        f"{fl}: '{room.label}' 면적 {poly.area:.1f}㎡ — 목표 {room.area:g}㎡ 와 {diff:+.0%} 차이 (허용 ±{tol:.0%}). "
                        "목표 면적을 고치거나 방 크기·순서·zone 을 조정하라"
                    )
                targets.append((room.area, poly.area))
        # 목표 면적 합 자체가 외곽과 안 맞으면 (자동 분할이 비례로 늘이거나 줄임) 그 원인 하나만 알려 준다
        want, got = sum(t for t, _ in targets), sum(a for _, a in targets)
        if targets and abs(got - want) / want > tol:
            rep.errors.append(
                f"{fl}: 목표 면적 합 {want:.1f}㎡ 인데 실제 방 면적 합은 {got:.1f}㎡ ({(got - want) / want:+.0%}). "
                f"외곽 {outline.area:.1f}㎡ 에서 복도·계단(약 15%)을 뺀 만큼이 되도록 목표 면적을 고치거나, 외곽 크기를 바꿔라"
            )
        else:
            rep.errors += area_errors
        items = list(polys.items())
        for i, (a, pa) in enumerate(items):
            for b, pb in items[i + 1:]:
                inter = pa.intersection(pb).area
                if inter > 0.05:
                    la = next(r.label for r in floor.rooms if r.id == a)
                    lb = next(r.label for r in floor.rooms if r.id == b)
                    rep.errors.append(f"{fl}: '{la}'와 '{lb}'가 {inter:.1f}㎡ 겹친다. 두 방이 겹치지 않게 좌표를 고쳐라")
        covered = unary_union(list(polys.values())).intersection(outline).area if polys else 0
        if covered < outline.area * 0.85:
            rep.warnings.append(f"{fl}: 방이 외곽 면적의 {covered / outline.area:.0%} 만 채운다 (빈 공간 {outline.area - covered:.1f}㎡)")

        # 문 검사 + 연결 그래프
        graph: dict[str, set[str]] = {rid: set() for rid in polys}
        graph[OUTSIDE] = set()
        labels = {r.id: r.label for r in floor.rooms}
        for d in floor.doors:
            if d.room not in polys or (d.to != OUTSIDE and d.to not in polys):
                rep.errors.append(f"{fl}: 문({d.room}→{d.to})이 없는 방 id 를 가리킨다. rooms 의 id 를 써라")
                continue
            seg = door_segment(floor, d)
            target = outline.exterior if d.to == OUTSIDE else polys[d.to].exterior
            where = "외벽" if d.to == OUTSIDE else f"'{labels[d.to]}'와 맞닿은 벽"
            if seg is None or not _seg_on(seg, target) or not _seg_on(seg, polys[d.room].exterior):
                rep.errors.append(
                    f"{fl}: '{labels[d.room]}'의 문이 {where} 위에 있지 않다 (side={d.side}, offset={d.offset}). "
                    "side/offset 을 빼면 코드가 맞닿은 벽 가운데에 놓는다"
                )
                continue
            graph[d.room].add(d.to)
            graph[d.to].add(d.room)
        for w in floor.windows:
            if w.room not in polys:
                rep.errors.append(f"{fl}: 창이 없는 방 id '{w.room}' 를 가리킨다")
                continue
            seg = window_segment(floor, w, outline)
            if seg is None or not _seg_on(seg, outline.exterior):
                rep.errors.append(f"{fl}: '{labels[w.room]}'의 창(side={w.side})이 외벽 위에 있지 않다. 외벽에 닿은 쪽으로 옮겨라")
        segs = [(f"'{labels[d.room]}'의 문", door_segment(floor, d)) for d in floor.doors if d.room in polys]
        segs += [(f"'{labels[w.room]}'의 창", window_segment(floor, w, outline)) for w in floor.windows if w.room in polys]
        segs = [(name, LineString(sg)) for name, sg in segs if sg]
        for i, (n1, l1) in enumerate(segs):
            for n2, l2 in segs[i + 1:]:
                if l1.intersection(l2).length > 0.05:
                    rep.errors.append(f"{fl}: {n1}과 {n2}이 같은 벽에서 겹친다. offset 을 바꿔 떼어 놓아라")
        stairs = [rid for rid in polys if next(r.type for r in floor.rooms if r.id == rid) == "stair"]
        stair_shapes.append((floor, polys[stairs[0]] if stairs else None))
        starts = [OUTSIDE] if floor.level == lowest else stairs
        if floor.level == lowest and not graph[OUTSIDE]:
            rep.errors.append(f"{fl}: 외부 출입문(to: \"outside\")이 없다. 현관이나 거실 외벽에 문을 추가하라")
        if starts and (floor.level != lowest or graph[OUTSIDE]):
            seen, queue = set(starts), deque(starts)
            while queue:
                for nb in graph[queue.popleft()]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            unreachable = [labels[r] for r in polys if r not in seen]
            if unreachable:
                rep.errors.append(
                    f"{fl}: 문으로 갈 수 없는 방이 있다 {unreachable}. 복도나 이웃 방과 맞닿게 배치하거나 doors 에 연결을 추가하라"
                )
        for room in floor.rooms:
            if room.type in HABITABLE and room.id in polys and not any(w.room == room.id for w in floor.windows):
                rep.warnings.append(f"{fl}: '{room.label}'에 창이 없다 (채광·환기 확인)")

    # 계단 — 여러 층이면 모든 층에, 같은 자리에
    if len(layout.floors) > 1:
        for floor, shape in stair_shapes:
            if shape is None:
                rep.errors.append(f"{floor.label}: 여러 층 건물인데 계단(type: stair)이 없다. 모든 층의 같은 자리에 계단을 넣어라")
        for (f1, s1), (f2, s2) in zip(stair_shapes, stair_shapes[1:]):
            if s1 is None or s2 is None:
                continue
            iou = s1.intersection(s2).area / s1.union(s2).area
            if iou < 0.8:
                rep.errors.append(
                    f"계단 위치가 {f1.label}과 {f2.label}에서 다르다 (겹침 {iou:.0%}). 모든 층의 계단 rect 를 같게 하라"
                )
    return rep


def check(layout: Layout, tolerance: float | None = None) -> tuple[Layout, Report]:
    """완성(resolve) + 검증을 한 번에. 완성 중 생긴 오류도 Report 로 돌려준다."""
    try:
        resolved, notes = resolve(layout)
    except (LayoutParseError, ValueError) as e:
        return layout, Report(errors=[str(e)])
    rep = validate(resolved, tolerance)
    rep.notes = notes + rep.notes
    return resolved, rep


# ════════════════════════════════════════════════════════════════════
# 요약 (근거 자료·화면용)
# ════════════════════════════════════════════════════════════════════


def area_table(layout: Layout) -> str:
    lines = ["| 층 | 방 | 종류 | 면적(㎡) | 목표(㎡) | 크기(m) |", "|---|---|---|---:|---:|---|"]
    for floor in layout.floors:
        for room in floor.rooms:
            poly = room.shape()
            if poly is None:
                lines.append(f"| {floor.label} | {room.label} | {ROOM_TYPES[room.type][0]} | - | {room.area or '-'} | 좌표 없음 |")
                continue
            x0, y0, x1, y1 = poly.bounds
            lines.append(
                f"| {floor.label} | {room.label} | {ROOM_TYPES[room.type][0]} | {poly.area:.1f} | "
                f"{room.area if room.area else '-'} | {x1 - x0:.2f}×{y1 - y0:.2f} |"
            )
    total = sum(layout.outline_polygon(f).area for f in layout.floors)
    lines.append(f"\n연면적(외곽 기준) {total:.1f}㎡ · {len(layout.floors)}개 층")
    return "\n".join(lines)


def report_markdown(rep: Report) -> str:
    lines = ["✅ 검증 통과" if rep.ok else f"❌ 검증 실패 {len(rep.errors)}건"]
    lines += [f"- ❌ {e}" for e in rep.errors]
    lines += [f"- ⚠️ {w}" for w in rep.warnings]
    lines += [f"- ℹ️ {n}" for n in rep.notes]
    return "\n".join(lines)


def standards_table() -> str:
    """방 종류별 최소 치수 (코드가 검사하는 기준) — 근거 자료로 에이전트에게 준다."""
    lines = ["| 종류(type) | 이름 | 최소 면적(㎡) | 최소 폭(m) | 기본 구역 |", "|---|---|---:|---:|---|"]
    zone_ko = {"front": "앞(남·도로)", "back": "뒤(북)", "circulation": "동선"}
    for key, (name, area, width, zone) in ROOM_TYPES.items():
        lines.append(f"| {key} | {name} | {area:g} | {width:g} | {zone_ko[zone]} |")
    return "\n".join(lines)
