"""첨부 도면 → 레이아웃 JSON (설계 팀의 '도면 수정·층 추가', '3D 만들기' 업무).

워크플로우 전에 데이터 툴(core/tools/floorplan_input.py)로 돌려, 에이전트들은 파일이 아니라 JSON 텍스트를 받는다.

  .json        : 이전 결과의 레이아웃 JSON — 그대로 (정확)
  .dxf         : ezdxf
                 - 이 앱이 만든 DXF 면 안에 넣어 둔 원본 JSON 을 꺼낸다 (정확)
                 - 다른 프로그램의 DXF 면 닫힌 폴리선 = 방, 그 안의 글자 = 방 이름으로 읽는다 (근사 — 한 층만)
  .png .jpg .pdf : 비전 모델(OpenAI·Google)이 그림을 보고 방 배치를 추정한다 (근사 — 치수는 '확인 필요')
                 PDF 는 첫 쪽을 그림으로 바꿔 읽는다 (pypdfium2)
  .dwg         : 읽지 않는다. ODA File Converter(무료)로 DXF 로 바꿔 다시 올리도록 안내한다
"""

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from core import llm_factory
from core.floorplan import ROOM_TYPES, Floor, Layout, LayoutParseError, Rect, Room, parse_layout
from core.floorplan_render import XRECORD_KEY
from core.schemas import ModelsConfig, RankModels

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
VISION_PROVIDERS = ("openai", "google", "anthropic")  # 그림을 읽을 수 있는 provider
DWG_HELP = (
    "DWG 는 AutoCAD 전용 형식이라 직접 읽지 않습니다. 무료 'ODA File Converter'(Open Design Alliance)로 "
    "DXF(ASCII, 2010 이상)로 바꾼 뒤 다시 올려 주세요. 변환기는 직접 내려받아 설치해야 합니다."
)
# 방 이름 → 종류 (DXF·그림의 글자로 종류 추정)
NAME_TYPES = [
    (r"거실|리빙|living", "living"), (r"주방|부엌|키친|kitchen", "kitchen"), (r"식당|다이닝|dining", "dining"),
    (r"현관|entr", "entrance"), (r"욕실|화장실|샤워|bath|wc|toilet", "bathroom"), (r"다용도|세탁|보일러|utility|laundry", "utility"),
    (r"드레스|dress|walk-?in", "dressroom"), (r"창고|수납|팬트리|storage|closet", "storage"), (r"발코니|베란다|balcony|terrace", "balcony"),
    (r"복도|홀|hall|corridor", "corridor"), (r"계단|stair", "stair"), (r"서재|작업실|office|study", "office"),
    (r"침실|안방|방|bed|room", "bedroom"),
]


@dataclass
class ImportResult:
    layout: Layout | None
    method: str  # 화면·근거 자료에 쓸 읽은 방법 설명
    exact: bool = False
    notes: list[str] = field(default_factory=list)


def guess_type(name: str) -> str:
    low = name.lower()
    for pattern, room_type in NAME_TYPES:
        if re.search(pattern, low):
            return room_type
    return "other"


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_") or "room"
    name, k = base, 2
    while name in used:
        name, k = f"{base}{k}", k + 1
    used.add(name)
    return name


# ─── JSON ────────────────────────────────────────────────────────────


def from_json(data: bytes) -> ImportResult:
    text = data.decode("utf-8-sig")
    try:
        layout = parse_layout(text)
    except LayoutParseError as e:
        return ImportResult(None, "레이아웃 JSON", notes=[str(e)])
    return ImportResult(layout, "레이아웃 JSON (이전 결과 — 정확)", exact=True)


# ─── DXF ─────────────────────────────────────────────────────────────


def _embedded_layout(doc) -> Layout | None:
    if XRECORD_KEY not in doc.rootdict:
        return None
    xrec = doc.rootdict[XRECORD_KEY]
    text = "".join(str(tag.value) for tag in xrec.tags if tag.code == 1)
    return parse_layout(text)


def from_dxf(data: bytes) -> ImportResult:
    import ezdxf

    try:
        doc = ezdxf.read(io.StringIO(data.decode("utf-8", errors="replace")))
    except Exception as e:  # noqa: BLE001 — 깨진 DXF
        return ImportResult(None, "DXF", notes=[f"DXF 를 읽지 못했습니다: {type(e).__name__}: {str(e)[:200]}"])
    try:
        embedded = _embedded_layout(doc)
    except LayoutParseError as e:
        embedded, note = None, f"DXF 안의 레이아웃 JSON 이 깨져 도형으로 읽습니다: {e}"
    else:
        note = ""
    if embedded is not None:
        return ImportResult(embedded, "DXF (이 앱이 만든 도면 — 안에 저장된 원본 JSON, 정확)", exact=True)

    msp = doc.modelspace()
    room_layers = {"A-ROOM", "ROOM", "ROOMS", "실", "방"}
    polys = []
    for e in msp.query("LWPOLYLINE POLYLINE"):
        closed = e.closed if e.dxftype() == "LWPOLYLINE" else e.is_closed
        if not closed:
            continue
        pts = [(p[0], p[1]) for p in (e.get_points("xy") if e.dxftype() == "LWPOLYLINE" else e.points())]
        if len(pts) >= 3:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                polys.append((poly, e.dxf.layer.upper() in room_layers))
    texts = []
    for e in msp.query("TEXT MTEXT"):
        content = e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text
        pos = e.dxf.insert
        texts.append((content.strip(), Point(pos[0], pos[1])))
    if not polys:
        return ImportResult(None, "DXF (도형 읽기)", notes=[n for n in [note, "닫힌 폴리선(방 경계)을 찾지 못했습니다."] if n])

    # 단위 추정: 도면 크기가 수백 이상이면 mm 로 보고 m 로 바꾼다
    span = max(max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p, _ in polys)
    unit = 0.001 if span > 300 else (0.01 if span > 80 else 1.0)
    notes = [n for n in [note] if n]
    if unit != 1.0:
        notes.append(f"도면 크기로 단위를 {'mm' if unit == 0.001 else 'cm'} 로 추정해 m 로 바꿨습니다 (확인 필요)")
    if any(is_room for _, is_room in polys):
        polys = [(p, r) for p, r in polys if r]
    polys.sort(key=lambda t: -t[0].area)
    outline = polys[0][0]
    candidates = [p for p, _ in polys[1:] if outline.buffer(0.01 / unit).contains(p)]
    if not candidates:  # 바깥 외곽선 없이 방만 있는 도면
        candidates = [p for p, _ in polys]
        outline = unary_union(candidates).envelope
    rooms, used = [], set()
    for poly in candidates:
        label = next((t for t, pt in texts if poly.contains(pt) and not re.fullmatch(r"[\d.,\s]+(m2|㎡|m²)?", t)), "")
        label = label.splitlines()[0][:20] if label else ""
        room_type = guess_type(label) if label else "other"
        coords = [(round(x * unit, 3), round(y * unit, 3)) for x, y in list(poly.exterior.coords)[:-1]]
        rooms.append(_room_from_coords(_slug(label or room_type, used), room_type, label, coords))
    ox0, oy0, ox1, oy1 = outline.bounds
    outline_coords = [(round(x * unit, 3), round(y * unit, 3)) for x, y in list(outline.exterior.coords)[:-1]]
    layout = Layout(
        title="가져온 DXF 도면",
        outline=_rect_or_poly(outline_coords),
        floors=[Floor(level=1, rooms=rooms)],
        needs_check=["DXF 도형에서 읽은 방 경계·이름 (방 종류 추정, 문·창은 다시 배치됨)"],
    )
    notes.append(f"닫힌 폴리선 {len(candidates)}개를 방으로 읽었습니다 (외곽 {(ox1 - ox0) * unit:.2f}×{(oy1 - oy0) * unit:.2f}m)")
    return ImportResult(layout, "DXF (도형 읽기 — 근사)", notes=notes)


def _rect_or_poly(coords: list[tuple[float, float]]) -> Rect | list[tuple[float, float]]:
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    poly = Polygon(coords)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if abs(poly.area - (x1 - x0) * (y1 - y0)) < 1e-3:
        return Rect(x=x0, y=y0, w=round(x1 - x0, 3), h=round(y1 - y0, 3))
    return coords


def _room_from_coords(rid: str, room_type: str, name: str, coords) -> Room:
    shape = _rect_or_poly(coords)
    if isinstance(shape, Rect):
        return Room(id=rid, type=room_type, name=name, rect=shape)
    return Room(id=rid, type=room_type, name=name, polygon=shape)


# ─── 그림·PDF (비전 모델) ─────────────────────────────────────────────


class VisionRoom(BaseModel):
    name: str = Field(description="방 이름 (그림에 적힌 그대로, 한국어)")
    type: str = Field(description=f"방 종류: {', '.join(ROOM_TYPES)} 중 하나")
    x: float = Field(description="방 왼쪽 아래 모서리 x (m, 건물 왼쪽 아래 = 0)")
    y: float = Field(description="방 왼쪽 아래 모서리 y (m, 위쪽이 +)")
    w: float = Field(description="방 가로 m")
    h: float = Field(description="방 세로 m")


class VisionFloor(BaseModel):
    level: int = Field(description="층 (1 = 1층)")
    rooms: list[VisionRoom] = Field(description="이 층의 방 목록 (복도·계단 포함)")


class VisionLayout(BaseModel):
    """비전 모델 구조화 출력 — OpenAI strict 호환을 위해 기본값 없이 모두 필수."""

    analysis: str = Field(description="먼저 그림에 적힌 치수·축척·방 이름을 2~3문장으로 정리")
    width: float = Field(description="건물 외곽 가로 m (치수가 없으면 방 크기로 추정)")
    depth: float = Field(description="건물 외곽 세로 m")
    dimensions_read: bool = Field(description="그림에 적힌 치수를 실제로 읽었으면 true, 추정이면 false")
    floors: list[VisionFloor] = Field(description="층 목록")
    uncertain: list[str] = Field(description="확실하지 않은 부분 (예: '욕실 크기 추정'). 없으면 빈 목록")


VISION_PROMPT = "\n".join([
    "너는 건축 평면도를 읽어 방 배치를 숫자로 옮기는 담당이다.",
    "- 좌표계: 건물 외곽의 왼쪽 아래 = (0,0), x 는 오른쪽, y 는 위쪽, 단위 m.",
    "- 방은 축에 나란한 직사각형으로 근사한다 (ㄱ자 방은 큰 직사각형 하나로).",
    "- 그림에 치수(예: 3,600 / 3.6m)가 있으면 그 값을 쓰고, 없으면 일반 주택 크기(침실 3×3.3m, 욕실 1.8×2.4m)로 추정한다.",
    "- 방이 서로 겹치지 않고 외곽 안에 들어가게 한다. 벽 두께는 무시하고 벽 중심선 기준으로 쓴다.",
    "- 확실하지 않은 치수·이름은 uncertain 에 적는다.",
])


def vision_models(models: ModelsConfig) -> RankModels | None:
    """보조 모델 후보 중 그림을 읽을 수 있는 것만 (유료 모델 → 직급 모델 순으로 찾는다)."""
    pool = [*models.utility.candidates, *[s for r in models.ranks.values() for s in r.candidates]]
    seen, picked = set(), []
    for spec in pool:
        if spec.provider in VISION_PROVIDERS and spec.label not in seen:
            seen.add(spec.label)
            picked.append(spec)
    if not picked:
        return None
    return RankModels(primary=picked[0], fallback=picked[1:4])


def pdf_first_page_png(data: bytes) -> bytes:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    page = pdf[0]
    image = page.render(scale=2.0).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def from_image(data: bytes, mime: str, models: ModelsConfig) -> ImportResult:
    rank_models = vision_models(models)
    if rank_models is None or not llm_factory.available_candidates(rank_models, models):
        return ImportResult(None, "그림 읽기", notes=["그림을 읽으려면 OpenAI 또는 Google API 키가 필요합니다 (.env)."])
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    messages = [
        SystemMessage(VISION_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "이 평면도의 방 배치를 읽어 줘."},
            {"type": "image_url", "image_url": {"url": url}},
        ]),
    ]
    call = llm_factory.invoke_structured(rank_models, models, VisionLayout, messages, "도면 그림 읽기")
    vis: VisionLayout = call.parsed
    floors, check = [], [f"그림에서 추정한 치수 — 실제 치수 확인 필요 ({call.spec.label})"]
    if not vis.dimensions_read:
        check.append("그림에 치수가 없어 일반적인 방 크기로 추정함")
    check += vis.uncertain
    for vf in vis.floors:
        used: set[str] = set()
        rooms = [
            Room(id=_slug(r.name or r.type, used), type=r.type if r.type in ROOM_TYPES else guess_type(r.name), name=r.name,
                 rect=Rect(x=round(r.x, 2), y=round(r.y, 2), w=round(max(r.w, 0.3), 2), h=round(max(r.h, 0.3), 2)))
            for r in vf.rooms
        ]
        floors.append(Floor(level=vf.level, rooms=rooms))
    if not floors:
        return ImportResult(None, "그림 읽기 (비전 모델)", notes=["그림에서 방을 찾지 못했습니다."])
    layout = Layout(title="그림에서 읽은 도면", outline=Rect(w=round(vis.width, 2), h=round(vis.depth, 2)), floors=floors,
                    needs_check=check)
    return ImportResult(layout, f"그림 읽기 (비전 모델 {call.spec.label} — 근사, 치수 확인 필요)", notes=call.errors[-3:])


# ─── 입구 ────────────────────────────────────────────────────────────


def import_file(path: Path, models: ModelsConfig) -> ImportResult:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".json":
        return from_json(data)
    if suffix == ".dxf":
        return from_dxf(data)
    if suffix == ".dwg":
        return ImportResult(None, "DWG", notes=[DWG_HELP])
    if suffix == ".pdf":
        try:
            png = pdf_first_page_png(data)
        except Exception as e:  # noqa: BLE001
            return ImportResult(None, "PDF", notes=[f"PDF 를 그림으로 바꾸지 못했습니다: {type(e).__name__}: {str(e)[:200]}"])
        return from_image(png, "image/png", models)
    if suffix in IMAGE_TYPES:
        return from_image(data, IMAGE_TYPES[suffix], models)
    return ImportResult(None, suffix, notes=[f"지원하지 않는 파일 형식입니다: {suffix} (json·dxf·png·jpg·pdf 지원, dwg 는 dxf 로 변환)"])


def latest_saved_layout(results_dir: Path) -> Path | None:
    """results/design/ 에 저장된 가장 최근 레이아웃 JSON."""
    folder = results_dir / "design"
    if not folder.exists():
        return None
    files = sorted(folder.glob("*_layout.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def to_json_text(layout: Layout) -> str:
    return json.dumps(json.loads(layout.model_dump_json(exclude_none=True)), ensure_ascii=False, indent=1)
