"""평면 레이아웃 — 설계 팀의 '원본 데이터' (스키마 · 방 프로그램 모델 · 문/창 배치 · 벽 형상 · 검증 · 후보안).

LLM 은 좌표 계산을 자주 틀린다 (방이 겹치거나 외곽을 벗어나고, 면적 합이 맞지 않음).
그래서 연구(AI-Architect, Text-to-Layout, Text2BIM, ProCAD)의 공통 결론대로 역할을 나눈다.
  LLM  : 구조화된 레이아웃 JSON 만 쓴다 — 방 목록(종류·이름·목표 면적)·구역(zone)·이웃 관계(adjacent·near·open_to)·
         묶음(suite)·외곽 유형(typology) = '위상(topology)'
  코드 : 좌표가 없으면 배치 엔진(core/floorplan_solver.py, OR-Tools CP-SAT)이 0.3m 격자에 방을 놓는다
         (실패하면 예전 단순 분할 partition 으로 대신). 문·창을 공유 벽 위에 놓고, 벽을 만들고, 검증한다(설계 품질 포함)
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

# ── 방 프로그램 모델 (배치 엔진 core/floorplan_solver.py 와 설계 품질 검사가 같이 쓴다) ──
# zone(구역): public 공용(현관·거실·식당·주방) / private 개인(침실·드레스룸) / wet 물 쓰는 곳(욕실·다용도실·사우나) / special 특수(수영장·차고·중정)
ProgramZone = Literal["public", "private", "wet", "special"]
PROGRAM_ZONES = ("public", "private", "wet", "special")
# 세부 종류(kind) → (구역, 최대 장단변비(긴 변 ÷ 짧은 변, None = 검사 안 함), 외벽(채광) 필요, 배치 엔진이 노리는 폭 m, 최소 면적 ㎡)
# kind 는 type 과 같거나, type 이 other 인 방의 세부 종류(sauna·pool …) — 이름(사우나·수영장 …)으로도 알아낸다
KIND_RULES: dict[str, tuple[str, float | None, bool, float, float]] = {
    "living": ("public", 2.0, True, 3.6, 10.0),
    "dining": ("public", 2.0, False, 2.4, 5.0),
    "kitchen": ("public", 2.5, False, 2.1, 4.0),
    "entrance": ("public", 3.0, False, 1.5, 1.5),
    "office": ("private", 1.8, True, 2.4, 5.0),
    "bedroom": ("private", 1.6, True, 2.7, 7.0),
    "dressroom": ("private", 2.5, False, 1.5, 2.0),
    "storage": ("private", 3.0, False, 0.9, 0.8),
    "bathroom": ("wet", 2.0, False, 1.5, 2.5),
    "utility": ("wet", 2.5, False, 1.2, 1.5),
    "balcony": ("public", 6.0, True, 0.9, 1.5),
    "corridor": ("public", None, False, 0.9, 0.0),
    "stair": ("public", None, False, 0.9, 3.0),
    "other": ("public", 2.5, False, 1.2, 1.0),
    # type 은 other, 세부 종류만 다른 방
    "sauna": ("wet", 2.0, False, 1.5, 3.0),
    "pool": ("special", 3.0, False, 3.6, 10.0),
    "garage": ("special", 2.0, False, 3.0, 15.0),
    "pantry": ("public", 3.0, False, 0.9, 1.0),
    "court": ("special", None, False, 3.0, 9.0),
    # 바깥 공간 (실내 바닥이 아님) — 배치 엔진이 layout.outdoor 로 옮겨 ㄱ자 안쪽 모서리 등에 둔다
    "yard": ("special", None, False, 3.0, 4.0),
    "deck": ("special", None, False, 2.4, 4.0),
}
OUTDOOR_KINDS = {"yard", "deck"}  # 마당·데크(테라스) — 바닥 면적·외곽에 넣지 않고 도면에 '외부'로 그린다
OUTDOOR_NAME = {"yard": "마당", "deck": "데크", "court": "중정"}
# 이름 → 세부 종류 (LLM 이 type 을 other 로 쓰거나 모르는 종류를 지어내도 알아보게)
KIND_WORDS = [
    (re.compile(r"사우나|찜질|한증|sauna", re.I), "sauna"),
    (re.compile(r"수영장|풀장|실내\s*풀|pool", re.I), "pool"),
    (re.compile(r"차고|주차|garage", re.I), "garage"),
    (re.compile(r"중정|안마당|court", re.I), "court"),
    (re.compile(r"마당|정원|잔디|yard|garden", re.I), "yard"),
    (re.compile(r"데크|테라스|파티오|deck|terrace|patio", re.I), "deck"),
    (re.compile(r"팬트리|식품\s*창고|pantry", re.I), "pantry"),
]
MASTER_WORDS = re.compile(r"안방|부부|master|마스터", re.I)
# 장단변비가 기준의 이 배수를 넘으면 오류(그 아래는 경고)
ASPECT_ERROR_FACTOR = 1.25
# 복도 몫(복도 면적 ÷ 층 바닥 면적) 경고·오류 기준, 일자 복도 길이 경고·오류 기준 (m)
CORRIDOR_SHARE_WARN, CORRIDOR_SHARE_ERROR = 0.10, 0.15
CORRIDOR_LENGTH_WARN, CORRIDOR_LENGTH_ERROR = 12.0, 18.0
# 외곽 효율(실내 바닥 ÷ 외접 사각형) 경고 기준, 요청하지 않은 빈 마당(외곽 안쪽 모서리·둘러싸인 빈 곳) 오류 기준 ㎡
# 근거: teams/design/skills/plan_references.md (ㄱ자 ≈ 0.75, ㄷ자 ≈ 0.83 → 0.85 미만이면 마당이 요청됐거나 외부 공간으로 지정돼야 한다)
FOOTPRINT_EFF_WARN = 0.85
VOID_ERROR = 15.0
DOOR_MIN = 0.9  # 문을 낼 수 있는 공유 벽 길이 (m)
Typology = Literal["auto", "rect", "L", "U", "courtyard"]
TYPOLOGY_WORDS = {
    "auto": "auto", "자동": "auto", "rect": "rect", "rectangle": "rect", "일자": "rect", "일자형": "rect", "직사각형": "rect",
    "l": "L", "ㄱ": "L", "ㄱ자": "L", "ㄱ자형": "L", "u": "U", "ㄷ": "U", "ㄷ자": "U", "ㄷ자형": "U",
    "courtyard": "courtyard", "court": "courtyard", "중정": "courtyard", "중정형": "courtyard", "ㅁ자": "courtyard", "ㅁ자형": "courtyard",
}
TYPOLOGY_KO = {"rect": "일자형(직사각형)", "L": "ㄱ자형", "U": "ㄷ자형", "courtyard": "중정형"}


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
    # 구역: public/private/wet/special (프로그램 구역) 또는 front/back (예전 단순 분할의 앞/뒤 위치 지정). 없으면 종류 기본값
    zone: Literal["front", "back", "public", "private", "wet", "special"] | None = None
    # ── 아래는 모두 선택 — 비우면 코드가 방 종류로 기본값을 채운다 ──
    kind: str | None = None  # 세부 종류 (sauna·pool·garage·pantry·court). 모르는 type 을 쓰면 여기에 남는다
    adjacent: list[str] | None = None  # 꼭 이웃해야 할 방 id (문을 낼 수 있게 0.9m 이상 벽을 공유)
    near: list[str] | None = None  # 이웃하면 좋은 방 id
    open_to: list[str] | None = None  # 벽 없이 트인 방 id (예: 거실·식당·주방 LDK)
    exterior: bool | None = None  # 외벽(채광)에 닿아야 하는지 (없으면 거실·침실·서재는 필요)
    suite: str | None = None  # 묶음 이름 (예: "master" — 안방 + 드레스룸 + 욕실, 안방을 거쳐 들어간다)
    placement: Literal["main", "wing", "annex"] | None = None  # 특수실: 본채 / 별동 날개(ㄱ·ㄷ자) / 별채(도면에서 뺀다)

    @model_validator(mode="before")
    @classmethod
    def _keep_kind(cls, data):
        # 모르는 type(sauna 등)은 '기타'로 바꾸되 세부 종류(kind)로 남겨 배치·검사에 쓴다
        if isinstance(data, dict):
            t = data.get("type")
            if isinstance(t, str) and t not in ROOM_TYPES and not data.get("kind"):
                data = {**data, "kind": t.lower()}
        return data

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, v):
        return v if v in ROOM_TYPES else "other"  # 모르는 종류는 '기타'로 (LLM 이 새 종류를 지어내도 깨지지 않게)

    @field_validator("adjacent", "near", "open_to", mode="before")
    @classmethod
    def _id_list(cls, v):
        return [v] if isinstance(v, str) else v

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
    # 외곽 유형: rect 일자형 / L ㄱ자형 / U ㄷ자형 / courtyard 중정형 / auto (없으면 코드가 면적·프로그램으로 고른다)
    typology: Typology | None = None
    annex: list[Room] | None = None  # 별채로 뺀 방 (placement: annex — 본채 도면에는 그리지 않는다)
    # 바깥 공간 (마당·데크·테라스, kind yard/deck) — 실내 바닥이 아니다. 맨 아래층 도면에 '외부'로 그린다 (rect/polygon 필요)
    outdoor: list[Room] | None = None

    @field_validator("typology", mode="before")
    @classmethod
    def _typology_alias(cls, v):
        return TYPOLOGY_WORDS.get(str(v).strip().lower(), "auto") if v is not None else None

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
# 방 프로그램 모델 — 구역·이웃 관계·외벽·장단변비 (LLM 이 안 쓰면 종류별 기본값)
# ════════════════════════════════════════════════════════════════════


def room_kind(room: Room) -> str:
    """세부 종류: kind → 이름(사우나·수영장 …) → type 순서로 정한다."""
    if room.kind in KIND_RULES:
        return room.kind
    if room.type in ("other", "bathroom", "utility", "storage", "balcony"):
        for pattern, kind in KIND_WORDS:
            if kind in OUTDOOR_KINDS and room.type != "other":
                continue  # 발코니·욕실 이름의 '테라스' 등은 실내 방으로 둔다
            if pattern.search(room.name or "") or pattern.search(room.kind or ""):
                return kind
    return room.type


def room_zone(room: Room) -> str:
    return room.zone if room.zone in PROGRAM_ZONES else KIND_RULES[room_kind(room)][0]


def needs_daylight(room: Room) -> bool:
    return room.exterior if room.exterior is not None else KIND_RULES[room_kind(room)][2]


def max_aspect(room: Room) -> float | None:
    return KIND_RULES[room_kind(room)][1]


def is_court(room: Room) -> bool:
    return room_kind(room) == "court"


def is_outdoor(room: Room) -> bool:
    return room_kind(room) in OUTDOOR_KINDS


def is_wet(room: Room) -> bool:
    return room_kind(room) in ("bathroom", "utility", "sauna")


@dataclass
class Program:
    """한 층의 이웃 관계. (방 id, 방 id) 쌍 — explicit 은 LLM 이 직접 쓴 것(어기면 오류), 나머지는 기본값(어기면 경고)."""

    required: list[tuple[str, str]]
    preferred: list[tuple[str, str]]
    open_pairs: list[tuple[str, str]]
    suites: dict[str, list[str]]  # 묶음 이름 → [머리 방(안방), 딸린 방 …] — 딸린 방은 묶음 안에서만 드나든다
    explicit: set[tuple[str, str]]
    notes: list[str]


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def program_links(floor: Floor) -> Program:
    """방 목록 → 이웃 관계. LLM 이 쓴 adjacent·near·open_to·suite 를 쓰고, 빠진 것은 종류별 기본값으로 채운다."""
    rooms = [r for r in floor.rooms if r.type not in CIRCULATION]
    ids = {r.id for r in floor.rooms}
    by_kind: dict[str, list[Room]] = {}
    for r in rooms:
        by_kind.setdefault(room_kind(r), []).append(r)
    req: list[tuple[str, str]] = []
    pref: list[tuple[str, str]] = []
    opened: list[tuple[str, str]] = []
    explicit: set[tuple[str, str]] = set()
    notes: list[str] = []

    def add(bucket, a, b, mark=False):
        if a == b or a not in ids or b not in ids:
            if b not in ids and mark:
                notes.append(f"'{a}' 의 이웃으로 적은 '{b}' 는 이 층에 없는 방 id 라서 무시했다")
            return
        p = _pair(a, b)
        if p not in bucket:
            bucket.append(p)
        if mark:
            explicit.add(p)

    for r in rooms:
        for o in r.adjacent or []:
            add(req, r.id, o, True)
        for o in r.near or []:
            add(pref, r.id, o, True)
        for o in r.open_to or []:
            add(opened, r.id, o, True)
            add(req, r.id, o, True)

    # 묶음(suite): LLM 이 쓴 것, 없으면 '안방 + 드레스룸 + 안방 욕실' 을 기본으로
    suites: dict[str, list[str]] = {}
    for r in rooms:
        if r.suite:
            suites.setdefault(r.suite, []).append(r.id)
    beds = by_kind.get("bedroom", [])
    if not suites and beds:
        master = next((b for b in beds if MASTER_WORDS.search(b.name or "")), None)
        if master is None and len(beds) >= 2:
            master = max(beds, key=lambda b: (b.area or 0, -beds.index(b)))
        if master is not None:
            members = [master.id]
            dress = by_kind.get("dressroom", [])
            if len(dress) == 1 or any(MASTER_WORDS.search(d.name or "") for d in dress):
                members.append(next((d for d in dress if MASTER_WORDS.search(d.name or "")), dress[0]).id)
            baths = by_kind.get("bathroom", [])
            mb = next((b for b in baths if MASTER_WORDS.search(b.name or "")), None)
            if mb is None and len(baths) >= 2 and len(beds) >= 2:
                mb = baths[-1]
            if mb is not None:
                members.append(mb.id)
            if len(members) > 1:
                suites["master"] = members
                labels = [next(r.label for r in rooms if r.id == m) for m in members]
                notes.append(f"안방 묶음을 {' + '.join(labels)} 으로 잡았다 (딸린 방은 안방을 거쳐 들어간다)")
    for name, members in suites.items():
        head = max(members, key=lambda m: (room_kind(next(r for r in rooms if r.id == m)) == "bedroom", -members.index(m)))
        suites[name] = [head] + [m for m in members if m != head]

    def first(kind: str) -> Room | None:
        found = by_kind.get(kind)
        return found[0] if found else None

    living, dining, kitchen, entry = first("living"), first("dining"), first("kitchen"), first("entrance")
    if kitchen and dining:
        add(req, kitchen.id, dining.id)
    elif kitchen and living:
        add(pref, kitchen.id, living.id)
    if living and dining:
        add(pref, living.id, dining.id)
    if entry and living:
        add(pref, entry.id, living.id)
    for u in by_kind.get("utility", []) + by_kind.get("pantry", []):
        if kitchen:
            add(pref, u.id, kitchen.id)
    suite_members = {m for ms in suites.values() for m in ms[1:]}
    common_baths = [b for b in by_kind.get("bathroom", []) if b.id not in suite_members]
    for s in by_kind.get("sauna", []):
        if common_baths:
            add(pref, s.id, common_baths[0].id)
        for p in by_kind.get("pool", []):
            add(pref, s.id, p.id)
    return Program(required=req, preferred=pref, open_pairs=opened, suites=suites, explicit=explicit, notes=notes)


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


def _shared_len(a: Polygon, b: Polygon | None, outline: Polygon | None = None) -> float:
    """가장 긴 공유 벽 한 조각의 길이 (문을 낼 수 있는지 볼 때)."""
    segs = shared_wall(a, b, outline)
    return max((_seg_len(s) for s in segs), default=0.0)


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

    opened = open_pairs(floor)

    def add(a: str, b: str, width: float) -> bool:
        if _pair(a, b) in opened and _shared_len(polys[a], polys[b]) >= DOOR_MIN:
            return True  # 트인 방 (LDK) — 문 없이 이어진다
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


def light_walls(floor: Floor, room: Room, outline: Polygon) -> list[Segment]:
    """창을 낼 수 있는 벽 = 외벽 + 중정(court)에 맞닿은 벽."""
    poly = room.shape()
    if poly is None or is_court(room):
        return []
    segs = shared_wall(poly, None, outline)
    for court in floor.rooms:
        if is_court(court) and court.id != room.id and court.shape() is not None:
            segs += shared_wall(poly, court.shape())
    return segs


def _light_boundary(floor: Floor, outline: Polygon):
    return unary_union([outline.exterior] + [r.shape().exterior for r in floor.rooms if is_court(r) and r.shape() is not None])


def _auto_windows(floor: Floor, outline: Polygon) -> list[Window]:
    """방마다 외벽(또는 중정 쪽 벽) 하나에 창 (거주 공간은 남 → 동·서 → 북 순서, 문과 겹치지 않는 벽)."""
    doors = [LineString(seg).buffer(0.3) for d in floor.doors if (seg := door_segment(floor, d))]
    windows: list[Window] = []
    pref = {"S": 0, "E": 1, "W": 1, "N": 2}
    for room in floor.rooms:
        poly = room.shape()
        if poly is None or room.type in CIRCULATION or room.type in ("storage", "entrance") or is_court(room):
            continue
        small = room.type in ("bathroom", "utility", "dressroom")
        segs = sorted(light_walls(floor, room, outline),
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


def resolve(layout: Layout, options: dict | None = None) -> tuple[Layout, list[str]]:
    """좌표가 없는 층은 배치 엔진(CP-SAT, core/floorplan_solver.py)으로 배치하고, 문·창이 비어 있으면 자동 배치한 '완성 레이아웃'을 만든다.

    좌표를 이미 가진 층·직접 쓴 문/창은 그대로 둔다 → 사람이·LLM 이 고친 부분이 보존된다.
    완성본을 저장해 두면 다음 수정은 이 좌표에서 시작한다 (DXF 를 다시 해석하지 않음).
    배치 엔진이 답을 못 찾으면 예전 단순 분할(앞 띠 / 복도 / 뒤 띠)로 대신하고 그 사실을 메모에 남긴다.
    options: typology(rect·L·U·courtyard) · entry_side(S·E·W·N) · seed · time_limit · engine("solver"|"partition")
    """
    out = layout.model_copy(deep=True)
    notes: list[str] = []
    opts = dict(options or {})
    engine = opts.pop("engine", "solver")
    todo = [f for f in out.floors if f.rooms and not any(r.shape() is not None for r in f.rooms)]
    if todo and engine == "solver":
        try:
            from core import floorplan_solver

            out, solver_notes = floorplan_solver.solve_layout_cached(out, **opts)
            notes += solver_notes
        except ImportError:
            notes.append("배치 엔진(ortools)이 설치돼 있지 않아 예전 단순 분할로 나눴다 (pip install ortools)")
        except Exception as e:  # noqa: BLE001 — 엔진이 실패해도 단순 분할로 도면은 나온다
            notes.append(f"배치 엔진(CP-SAT)이 답을 찾지 못해 예전 단순 분할(앞 띠/복도/뒤 띠)로 대신했다 — {e}")
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


def open_pairs(floor: Floor) -> set[tuple[str, str]]:
    """벽 없이 트인 방 쌍 (open_to) — 문 없이 이어지고, 그 사이에는 벽을 세우지 않는다."""
    ids = {r.id for r in floor.rooms}
    return {_pair(r.id, o) for r in floor.rooms for o in (r.open_to or []) if o in ids and o != r.id}


def _open_lines(floor: Floor, rooms: dict[str, Polygon]):
    lines = []
    for a, b in open_pairs(floor):
        if a in rooms and b in rooms:
            lines += [LineString(s) for s in shared_wall(rooms[a], rooms[b])]
    return unary_union(lines).buffer(SNAP) if lines else None


def floor_geometry(layout: Layout, floor: Floor) -> FloorGeometry:
    outline = layout.outline_polygon(floor)
    rooms = _polys(floor)
    opened = _open_lines(floor, rooms)
    pieces = []
    for seg, exterior in wall_lines(rooms, outline):
        if opened is not None and not exterior and opened.covers(LineString(seg)):
            continue  # 트인 방 사이 (거실·식당·주방 LDK) — 벽 없음
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
    metrics: dict[str, dict] = Field(default_factory=dict)  # 층별 설계 지표 (복도 몫·장단변비·이웃·채광·물 쓰는 곳)

    @property
    def ok(self) -> bool:
        return not self.errors

    def score(self) -> float:
        """후보안 비교용 점수 (높을수록 좋음): 오류·경고·복도 몫·어긴 이웃 관계를 깎는다."""
        s = 100.0 - 15 * len(self.errors) - 3 * len(self.warnings)
        for m in self.metrics.values():
            s -= 100 * max(0.0, m.get("corridor_share", 0) - 0.05)
            s -= 2 * m.get("preferred_missed", 0)
            s -= 5 * max(0.0, m.get("worst_aspect", 1) - 2)
            s -= 40 * (1 - m.get("footprint_eff", 1))  # 같은 조건이면 꽉 찬(일자형) 외곽이 낫다 — ㄱ자+데크 0.8 → −8
            s -= 2 * max(0, m.get("corridor_pieces", 0) - 1)
        return round(s, 1)


def _min_width_ok(poly: Polygon, min_w: float) -> bool:
    """폭 검사: 안쪽으로 min_w/2 만큼 줄였을 때 남는 게 있어야 한다 (직사각형이 아닌 방에도 통함)."""
    return not poly.buffer(-(min_w / 2 - EPS), join_style="mitre").is_empty


def _short_side(poly: Polygon) -> float:
    rect = poly.minimum_rotated_rectangle
    xs = list(rect.exterior.coords)
    sides = [_seg_len((xs[i], xs[i + 1])) for i in range(2)]
    return min(sides)


def _aspect(poly: Polygon) -> tuple[float, float, float]:
    """(긴 변 ÷ 짧은 변, 가로, 세로) — 직사각형이 아니면 최소 외접 직사각형으로."""
    x0, y0, x1, y1 = poly.bounds
    if abs(poly.area - (x1 - x0) * (y1 - y0)) < 0.01:
        w, h = x1 - x0, y1 - y0
    else:
        xs = list(poly.minimum_rotated_rectangle.exterior.coords)
        w, h = _seg_len((xs[0], xs[1])), _seg_len((xs[1], xs[2]))
    return (max(w, h) / max(min(w, h), EPS), x1 - x0, y1 - y0)


def design_quality(floor: Floor, polys: dict[str, Polygon], outline: Polygon) -> tuple[list[str], list[str], dict]:
    """설계 품질 검사 (오류, 경고, 지표). 문장은 그대로 LLM 피드백이 된다.

    - 장단변비: 방 종류별 기준(욕실 1:2, 침실 1:1.6 …)을 넘으면 경고, 기준의 1.25배를 넘으면 오류 (복도·계단 제외)
    - 복도 몫: 복도 면적 ÷ 바닥 면적 10% 초과 경고, 15% 초과 오류 · 일자 복도 12m 초과 경고, 18m 초과 오류
    - 필수 이웃: LLM 이 adjacent·open_to·suite 로 쓴 관계는 오류, 코드 기본값(주방–식당 등)은 경고
    - 채광: 거실·침실·서재(또는 exterior: true)가 외벽·중정에 닿지 않으면 오류
    - 물 쓰는 곳(욕실·다용도실·사우나)이 세 곳 이상으로 흩어지면 경고 (배관이 길어짐)
    """
    fl = floor.label
    errors: list[str] = []
    warnings: list[str] = []
    rooms = {r.id: r for r in floor.rooms if r.id in polys}
    labels = {rid: r.label for rid, r in rooms.items()}
    area = outline.area or 1.0

    # 장단변비
    worst = (1.0, "")
    for rid, room in rooms.items():
        limit = None if room.type in CIRCULATION else max_aspect(room)
        if limit is None:
            continue
        ratio, w, h = _aspect(polys[rid])
        if ratio > worst[0]:
            worst = (ratio, room.label)
        kind_name = ROOM_TYPES[room.type][0] if room_kind(room) == room.type else room.label
        if ratio > limit * ASPECT_ERROR_FACTOR + 0.02:
            errors.append(f"{fl}: '{room.label}' {w:.2f}×{h:.2f}m — 장단변비 {ratio:.1f} 로 너무 길쭉하다 ({kind_name} 기준 1:{limit:g} 이하). "
                          "목표 면적·이웃 관계를 고치거나 typology 를 바꿔 코드가 다시 배치하게 하라")
        elif ratio > limit + 0.02:  # 격자 반올림 정도의 차이는 넘어가 준다
            warnings.append(f"{fl}: '{room.label}' {w:.2f}×{h:.2f}m 장단변비 {ratio:.1f} (기준 1:{limit:g}) — 조금 길쭉하다")

    # 복도
    corr = [rid for rid, r in rooms.items() if r.type == "corridor"]
    corr_area = sum(polys[c].area for c in corr)
    share = corr_area / area
    longest = max((max(polys[c].bounds[2] - polys[c].bounds[0], polys[c].bounds[3] - polys[c].bounds[1]) for c in corr), default=0.0)
    if share > CORRIDOR_SHARE_ERROR:
        errors.append(f"{fl}: 복도가 바닥의 {share:.0%}({corr_area:.1f}㎡) — 15% 를 넘는다. 방을 거실·홀 주위로 모으거나 "
                      "typology(ㄱ자·ㄷ자·중정형)를 바꿔 복도를 줄여라")
    elif share > CORRIDOR_SHARE_WARN:
        warnings.append(f"{fl}: 복도가 바닥의 {share:.0%}({corr_area:.1f}㎡) — 10% 이하가 좋다")
    if longest > CORRIDOR_LENGTH_ERROR:
        errors.append(f"{fl}: 일자 복도 길이 {longest:.1f}m — 18m 를 넘는 긴 복도는 비현실적이다. 건물을 ㄱ자·ㄷ자로 꺾거나 "
                      "공용·개인 구역을 나눠 복도를 짧게 하라")
    elif longest > CORRIDOR_LENGTH_WARN:
        warnings.append(f"{fl}: 일자 복도 길이 {longest:.1f}m — 12m 이하가 좋다")
    # 복도 한 줄기(spine): 나란히 붙은 복도 조각 · 서로 안 이어진 복도 토막
    parallel = [(a, b) for i, a in enumerate(corr) for b in corr[i + 1:] if _shared_len(polys[a], polys[b]) > 1.5]
    hubs = [rid for rid, r in rooms.items() if r.type in ("entrance", "stair")]
    groups = _corridor_groups(corr, hubs, polys)
    spine = max((sum(polys[c].area for c in g) for g in groups), default=0.0) / corr_area if corr_area else 1.0
    if parallel:
        warnings.append(f"{fl}: 복도 조각 {len(parallel)}쌍이 나란히 붙어 있다 — 짧은 복도 여러 개 대신 한 줄기 복도로 합쳐라")
    if len(groups) > 1:
        warnings.append(f"{fl}: 복도가 {len(groups)}토막으로 끊겨 있다 — 현관에서 이어지는 한 줄기 동선(spine)이 좋다")

    # 이웃 관계
    prog = program_links(floor)
    req_ok = req_n = 0
    for a, b in prog.required:
        if a not in polys or b not in polys:
            continue
        req_n += 1
        if _shared_len(polys[a], polys[b]) >= DOOR_MIN - EPS:
            req_ok += 1
            continue
        if (a, b) in prog.explicit:
            errors.append(f"{fl}: '{labels[a]}'와 '{labels[b]}'가 맞닿아야 하는데 떨어져 있다 (문을 낼 0.9m 이상 공유 벽 없음) — "
                          "이웃 관계를 지키게 rect 를 빼고 코드가 배치하게 하라")
        else:
            warnings.append(f"{fl}: '{labels[a]}'와 '{labels[b]}'가 떨어져 있다 (보통 이웃하게 둔다)")
    missed = [(a, b) for a, b in prog.preferred
              if a in polys and b in polys and _shared_len(polys[a], polys[b]) < DOOR_MIN - EPS]
    for name, members in prog.suites.items():
        ms = [m for m in members if m in polys]
        if len(ms) < 2:
            continue
        seen, queue = {ms[0]}, deque([ms[0]])
        while queue:
            cur = queue.popleft()
            for o in ms:
                if o not in seen and _shared_len(polys[cur], polys[o]) >= DOOR_MIN - EPS:
                    seen.add(o)
                    queue.append(o)
        apart = [labels[m] for m in ms if m not in seen]
        if apart:
            msg = f"{fl}: 묶음 '{name}'({' + '.join(labels[m] for m in ms)}) 중 {apart} 이(가) 묶음 안 방과 맞닿지 않는다"
            (errors if any(rooms[m].suite for m in ms) else warnings).append(msg)

    # 채광
    light_ok = light_need = 0
    for rid, room in rooms.items():
        if room.type in CIRCULATION or is_court(room) or not needs_daylight(room):
            continue
        light_need += 1
        if any(_seg_len(sg) >= DOOR_MIN - EPS for sg in light_walls(floor, room, outline)):
            light_ok += 1
        else:
            errors.append(f"{fl}: '{room.label}'이(가) 외벽(또는 중정)에 닿지 않아 창을 낼 수 없다 — 거실·침실은 외벽에 붙여라")

    # 물 쓰는 곳 모으기
    clusters: list[set[str]] = []
    for rid in [rid for rid, r in rooms.items() if is_wet(r)]:
        near = [c for c in clusters if any(polys[rid].distance(polys[o]) <= 1.5 for o in c)]
        clusters = [c for c in clusters if c not in near] + [set().union({rid}, *near)]
    if len(clusters) > 2:
        groups = " / ".join("·".join(labels[r] for r in c) for c in clusters)
        warnings.append(f"{fl}: 물 쓰는 곳이 {len(clusters)}곳으로 흩어져 있다 ({groups}) — 욕실·다용도실·사우나를 가까이 모으면 배관이 짧아진다")

    metrics = {
        "corridor_share": round(share, 3), "corridor_area": round(corr_area, 1), "corridor_longest": round(longest, 1),
        "corridor_pieces": len(corr), "corridor_groups": len(groups), "corridor_parallel": len(parallel),
        "corridor_spine": round(spine, 2),
        "worst_aspect": round(worst[0], 2), "worst_aspect_room": worst[1],
        "required_ok": req_ok, "required": req_n, "preferred_missed": len(missed), "preferred": len(prog.preferred),
        "daylight_ok": light_ok, "daylight": light_need, "wet_clusters": len(clusters),
    }
    return errors, warnings, metrics


def _corridor_groups(corr: list[str], hubs: list[str], polys: dict[str, Polygon]) -> list[set[str]]:
    """복도 덩어리 목록 (1개 = 한 줄기 동선). 복도끼리 또는 현관·계단을 사이에 두고 0.9m 이상 맞닿으면 한 덩어리."""
    groups: list[set[str]] = []
    for c in corr + hubs:
        near = [g for g in groups if any(_shared_len(polys[c], polys[o]) >= DOOR_MIN - EPS for o in g)]
        groups = [g for g in groups if g not in near] + [set().union({c}, *near)]
    return [g & set(corr) for g in groups if g & set(corr)]


def footprint_quality(layout: Layout, floor: Floor, polys: dict[str, Polygon], outline: Polygon,
                      judge: bool = True) -> tuple[list[str], list[str], dict]:
    """외곽 효율 검사 (오류, 경고, 지표) — 요청하지 않은 빈 마당을 잡는다.

    실내 바닥 = 외곽 − 중정. 외접 사각형 안의 빈 곳(ㄱ·ㄷ자 안쪽 모서리, 둘러싸인 빈 곳)은
    중정 방이나 바깥 공간(layout.outdoor — 마당·데크)으로 지정돼 있으면 '계획된 외부', 아니면 '요청하지 않은 빈 곳'이다.
    - 외곽 효율(실내 ÷ 외접 사각형)과 계획된 외부까지 친 효율이 0.85 미만이면 경고
    - 요청하지 않은 빈 곳이 15㎡ 를 넘으면 오류 (외곽 유형을 엔진·LLM 이 정한 경우. 가져온 도면의 자유 외곽은 경고)
    judge=False 면 지표만 (위층 — 같은 외곽을 두 번 지적하지 않게)
    """
    fl = floor.label
    courts = [polys[r.id] for r in floor.rooms if r.id in polys and is_court(r)]
    indoor = outline.difference(unary_union(courts)) if courts else outline
    bb = box(*outline.bounds)
    outdoor = [r.shape() for r in (layout.outdoor or []) if r.shape() is not None]
    planned = unary_union(courts + outdoor) if courts or outdoor else None
    empty = bb.difference(indoor)
    regions = [g for g in getattr(empty, "geoms", [empty]) if not g.is_empty and g.area > 0.05]
    unplanned = [g.difference(planned) if planned is not None else g for g in regions]
    void_max = max((g.area for g in unplanned), default=0.0)
    planned_in = planned.intersection(bb).area if planned is not None else 0.0
    eff = indoor.area / bb.area if bb.area else 1.0
    eff_planned = min(1.0, (indoor.area + planned_in) / bb.area) if bb.area else 1.0
    errors: list[str] = []
    warnings: list[str] = []
    if judge:
        w, h = bb.bounds[2] - bb.bounds[0], bb.bounds[3] - bb.bounds[1]
        if void_max > VOID_ERROR:
            msg = (f"{fl}: 외곽 {w:.1f}×{h:.1f}m 안에 요청하지 않은 빈 곳 {void_max:.1f}㎡ 가 있다 (외곽 효율 {eff:.0%}) — "
                   "마당·중정을 요청하지 않았으면 typology 를 rect(일자형)로 두고 큰 특수실은 본채 끝이나 별채(placement: annex)에 넣어라. "
                   "마당이 필요하면 방 목록에 {\"type\": \"other\", \"name\": \"마당\"} 을 넣어라")
            (errors if layout.typology in TYPOLOGY_KO else warnings).append(msg)
        elif eff_planned < FOOTPRINT_EFF_WARN:
            warnings.append(f"{fl}: 외곽 효율 {eff_planned:.0%} — 외접 사각형 {w:.1f}×{h:.1f}m 의 {1 - eff_planned:.0%} 가 비어 있다 (85% 이상이 좋다)")
    metrics = {"footprint_eff": round(eff, 3), "footprint_eff_planned": round(eff_planned, 3), "void_max": round(void_max, 1),
               "outdoor_area": round(planned_in, 1)}
    return errors, warnings, metrics


def metrics_line(label: str, m: dict) -> str:
    fp = (f"외곽 효율 {m['footprint_eff']:.0%}" + (f"(외부 공간 포함 {m['footprint_eff_planned']:.0%})" if m.get("outdoor_area") else "")
          + f" · 빈 곳 최대 {m['void_max']:g}㎡ · ") if "footprint_eff" in m else ""
    spine = f" · 복도 {m['corridor_pieces']}조각/{m['corridor_groups']}줄기" if m.get("corridor_pieces") else ""
    return (f"설계 지표 — {label}: {fp}복도 {m['corridor_share']:.0%}({m['corridor_area']:g}㎡, 최장 {m['corridor_longest']:g}m){spine} · "
            f"장단변비 최대 {m['worst_aspect']:g}({m['worst_aspect_room'] or '-'}) · 필수 이웃 {m['required_ok']}/{m['required']} · "
            f"좋은 이웃 {m['preferred'] - m['preferred_missed']}/{m['preferred']} · 채광 {m['daylight_ok']}/{m['daylight']} · "
            f"물 쓰는 곳 {m['wet_clusters']}묶음")


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
    base = layout.outline_polygon()
    for o in layout.outdoor or []:
        shape = o.shape()
        if shape is None:
            rep.errors.append(f"바깥 공간 '{o.label}'({o.id}) 에 좌표(rect/polygon)가 없다")
        elif base.is_valid and shape.intersection(base).area > 0.05:
            rep.warnings.append(f"바깥 공간 '{o.label}' 이 건물 외곽과 {shape.intersection(base).area:.1f}㎡ 겹친다 — 마당·데크는 외곽 밖에 둔다")

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
        for a, b in open_pairs(floor):
            if a in polys and b in polys and _shared_len(polys[a], polys[b]) >= DOOR_MIN:
                graph[a].add(b)
                graph[b].add(a)
        for w in floor.windows:
            if w.room not in polys:
                rep.errors.append(f"{fl}: 창이 없는 방 id '{w.room}' 를 가리킨다")
                continue
            seg = window_segment(floor, w, outline)
            if seg is None or not _seg_on(seg, _light_boundary(floor, outline)):
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
            if room.type in HABITABLE and not is_court(room) and room.id in polys and not any(w.room == room.id for w in floor.windows):
                rep.warnings.append(f"{fl}: '{room.label}'에 창이 없다 (채광·환기 확인)")
        q_errors, q_warnings, metrics = design_quality(floor, polys, outline)
        f_errors, f_warnings, f_metrics = footprint_quality(layout, floor, polys, outline,
                                                            judge=floor.level == lowest or floor.outline is not None)
        rep.errors += q_errors + f_errors
        rep.warnings += q_warnings + f_warnings
        metrics = {**f_metrics, **metrics}
        rep.metrics[fl] = metrics
        rep.notes.append(metrics_line(fl, metrics))

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


def check(layout: Layout, tolerance: float | None = None, options: dict | None = None) -> tuple[Layout, Report]:
    """완성(resolve) + 검증을 한 번에. 완성 중 생긴 오류도 Report 로 돌려준다."""
    try:
        resolved, notes = resolve(layout, options)
    except (LayoutParseError, ValueError) as e:
        return layout, Report(errors=[str(e)])
    rep = validate(resolved, tolerance)
    rep.notes = notes + rep.notes
    return resolved, rep


# ════════════════════════════════════════════════════════════════════
# 후보안 (서로 다른 외곽 유형·현관 방향) — 화면에서 '후보안 A/B/C' 로 비교해 고른다
# ════════════════════════════════════════════════════════════════════


@dataclass
class Candidate:
    label: str  # A, B, C
    title: str  # 예: "ㄱ자형 · 현관 남쪽"
    options: dict
    layout: Layout
    report: Report

    @property
    def score(self) -> float:
        return self.report.score()


SIDE_KO = {"S": "남쪽", "E": "동쪽", "W": "서쪽", "N": "북쪽"}


def _signature(layout: Layout) -> str:
    return json.dumps([[(r.id, r.rect.model_dump() if r.rect else r.polygon) for r in f.rooms] for f in layout.floors],
                      sort_keys=True, default=str)


def candidates(layout: Layout, n: int = 3, tolerance: float | None = None, time_limit: float | None = None) -> list[Candidate]:
    """후보안을 최대 n개 만든다. 첫 번째(A)는 check(layout) 와 같은 기본 배치(하네스가 검증한 그 안).

    좌표가 이미 있는 레이아웃(수정 업무 등)은 후보가 하나뿐이다. 같은 배치가 나오면 다른 현관 방향·시드로 다시 푼다.
    B·C 는 검증을 통과한 안을 먼저 고르고, 모자랄 때만 통과 못 한 안으로 채운다 (모든 후보가 '말이 되게').
    """
    from core import floorplan_solver  # 순환 import 를 피해 늦게 불러온다

    base_resolved, base_report = check(layout, tolerance, {"time_limit": time_limit} if time_limit else None)
    typ = base_resolved.typology or "rect"
    extra = "".join(f" · {'·'.join(dict.fromkeys(r.label for r in items))}{' 별채' if items is base_resolved.annex else ''}"
                    for items in (base_resolved.annex, base_resolved.outdoor) if items)
    first = Candidate("A", f"{TYPOLOGY_KO.get(typ, typ)} · 기본{extra}", {}, base_resolved, base_report)
    out = [first]
    program_only = any(f.rooms and not any(r.shape() is not None for r in f.rooms) for f in layout.floors)
    if n <= 1 or not program_only:
        return out
    seen = {_signature(base_resolved)}
    backlog: list[Candidate] = []
    for opts in floorplan_solver.candidate_options(layout):
        if len(out) >= n:
            break
        if time_limit:
            opts = {**opts, "time_limit": time_limit}
        resolved, report = check(layout, tolerance, opts)
        sig = _signature(resolved)
        if sig in seen or any("단순 분할" in note for note in report.notes):
            continue
        seen.add(sig)
        got = resolved.typology or opts["typology"]
        if opts.get("annex_special") and resolved.annex:
            title = f"{TYPOLOGY_KO.get(got, got)} · {'·'.join(r.label for r in resolved.annex)} 별채"
        elif opts.get("aspect"):
            title = f"{TYPOLOGY_KO.get(got, got)} · {'길쭉한 남향' if opts['aspect'] > 1.8 else '깊고 짧은'} 외곽"
        else:
            title = f"{TYPOLOGY_KO.get(got, got)} · 현관 {SIDE_KO.get(opts['entry_side'], opts['entry_side'])}"
        if resolved.outdoor and got != "rect":
            title += " · " + "·".join(dict.fromkeys(r.label for r in resolved.outdoor))
        (out if report.ok else backlog).append(Candidate("", title, opts, resolved, report))
    out += backlog[: max(0, n - len(out))]
    for k, c in enumerate(out):
        c.label = "ABCDE"[k]
    return out


def best_candidate(cands: list[Candidate]) -> Candidate:
    """검증을 통과한 안 중 점수가 가장 높은 안 (같으면 앞의 안)."""
    return max(cands, key=lambda c: (c.report.ok, c.score, -cands.index(c)))


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
