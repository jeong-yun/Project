"""📐 평면도 그리기 테스트 — 벽 포셰·문/창 기호·가구·라벨 자동 맞춤·치수·DXF 블록."""

import io
import json
import math
from pathlib import Path

import ezdxf
import pytest
from PIL import Image
from shapely.geometry import Point, box

from core.floorplan import Opening
from tests.test_floorplan import SAMPLE  # 기존 2층 주택 샘플 (계단 포함)

from core import floorplan_render
from core.floorplan import check, parse_layout

# 좁은 방이 많은 단층 주택 — 자동 분할이 '침실5' 위에 '욕실1'(깊이 1.7m)을 쌓는다 (예전엔 라벨이 겹쳤다)
POOL_SAMPLE = {
    "title": "4인 가족 단층 주택",
    "outline": {"w": 20, "h": 13},
    "floors": [{"level": 1, "rooms": [
        {"id": "entry", "type": "entrance", "name": "현관", "area": 6},
        {"id": "living", "type": "living", "name": "거실", "area": 50},
        {"id": "kitchen", "type": "kitchen", "name": "주방", "area": 16},
        {"id": "dining", "type": "dining", "name": "식당", "area": 14},
        {"id": "pool", "type": "other", "name": "수영장", "area": 48, "zone": "front"},
        {"id": "bed1", "type": "bedroom", "name": "침실1", "area": 22},
        {"id": "bed2", "type": "bedroom", "name": "침실2", "area": 14},
        {"id": "bed3", "type": "bedroom", "name": "침실3", "area": 13},
        {"id": "bed4", "type": "bedroom", "name": "침실4", "area": 13},
        {"id": "bed5", "type": "bedroom", "name": "침실5", "area": 10},
        {"id": "bath1", "type": "bathroom", "name": "욕실1(욕조)", "area": 5},
        {"id": "bath2", "type": "bathroom", "name": "욕실2", "area": 4},
        {"id": "sauna", "type": "other", "name": "사우나", "area": 6, "zone": "back"},
        {"id": "util", "type": "utility", "name": "다용도실", "area": 5},
        {"id": "dress", "type": "dressroom", "name": "드레스룸", "area": 6},
    ]}],
}


@pytest.fixture(scope="module")
def pool():
    return _frozen("render_pool_layout.json")


@pytest.fixture(scope="module")
def house():
    return _frozen("render_house_layout.json")


def _frozen(name: str):
    """좌표가 고정된 레이아웃 — 렌더러 테스트는 '그리기'만 본다.

    POOL_SAMPLE·SAMPLE 을 그대로 쓰면 배치 엔진(CP-SAT)이 방 모양을 정해, 엔진이 바뀔 때마다
    '이 방에 TV·세면대가 들어가는지' 같은 그리기 검사가 흔들린다. 그래서 이 테스트를 만들 때의 배치를 파일로 고정했다.
    """
    data = json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8"))
    layout, _ = check(parse_layout(data))  # 좌표가 있으면 배치 엔진을 쓰지 않는다 (문·창만 채움)
    return layout


def _kinds(plan, room):
    return [f.kind for f in plan.furniture if f.room == room]


# ─── 라벨 ─────────────────────────────────────────────────────────────


def test_labels_fit_without_overlap_in_narrow_rooms(pool):
    """예전엔 '침실5'(깊이 3.4m) 라벨이 바로 위 '욕실1'(깊이 1.7m) 라벨에 겹쳤다."""
    plan = floorplan_render.plan_floor(pool, 0)
    labels = {lab.room: lab for lab in plan.labels}
    assert set(labels) == {r.id for r in pool.floors[0].rooms}
    boxes = list(labels.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not floorplan_render._hit(a.box, b.box), (a.room, b.room)
    assert not floorplan_render._hit(labels["bed5"].box, labels["bath1"].box)
    for lab in boxes:
        # 실제로 그리는 글자 윤곽이 라벨 상자 안에 있다 (계산과 그림이 같다)
        for path in floorplan_render.label_paths(lab):
            ext = path.get_extents()
            assert floorplan_render._inside((ext.x0, ext.y0, ext.x1, ext.y1), lab.box, tol=0.03), lab.room
        if lab.leader is None:  # 방 안 라벨은 방(벽 안쪽) 안에
            assert floorplan_render._inside(lab.box, plan.interiors[lab.room].bounds), lab.room
        assert any("㎡" in s for s, *_ in lab.lines)


def test_label_shrinks_wraps_then_uses_callout():
    # 좁은 '수납' 은 줄여서 방 안에, 0.7m 폭 '드레스룸(안방)' 은 지시선으로.
    # '수납' 방 폭은 설치된 한글 글꼴로 잰다 — 가장 큰 글자로는 안 들어가고 한 단계 작은 글자로는 들어가는 폭.
    # (0.9m 로 고정하면 맑은 고딕에서만 맞고, 글자가 조금 좁은 나눔고딕(CI)에서는 큰 글자 그대로 들어간다)
    big, small = floorplan_render.LABEL_SIZES[:2]
    usable = floorplan_render.text_width("수납", True) * (big + small) / 2
    w = round(usable + 0.12 + 2 * 0.06, 3)  # + 내벽 두께(양쪽 반씩) + 라벨 여백(양쪽 0.06)
    layout = parse_layout({"outline": {"w": 6, "h": 4}, "floors": [{"rooms": [
        {"id": "a", "type": "living", "name": "거실", "rect": {"x": 0, "y": 0, "w": 5.3 - w, "h": 4}},
        {"id": "s", "type": "storage", "name": "수납", "rect": {"x": 5.3 - w, "y": 0, "w": w, "h": 4}},
        {"id": "d", "type": "dressroom", "name": "드레스룸(안방)", "rect": {"x": 5.3, "y": 0, "w": 0.7, "h": 4}},
    ]}]})
    plan = floorplan_render.plan_floor(layout, 0)
    labels = {lab.room: lab for lab in plan.labels}
    assert labels["a"].lines[0][1] == floorplan_render.LABEL_SIZES[0]
    assert labels["s"].leader is None and labels["s"].lines[0][1] < floorplan_render.LABEL_SIZES[0]
    assert labels["d"].leader is not None
    assert not floorplan_render._hit(labels["d"].box, labels["a"].box)
    assert not floorplan_render._hit(labels["d"].box, labels["s"].box)


# ─── 가구·설비 ────────────────────────────────────────────────────────


def test_furniture_by_room_type(pool, house):
    plan = floorplan_render.plan_floor(pool, 0)
    assert {"bed", "wardrobe"} <= set(_kinds(plan, "bed1"))
    assert {"sofa", "tv", "coffee"} <= set(_kinds(plan, "living"))
    assert "dining" in _kinds(plan, "dining")
    assert any(k.startswith("counter") for k in _kinds(plan, "kitchen"))
    assert {"bathtub", "toilet", "basin"} <= set(_kinds(plan, "bath1"))  # 이름에 '욕조'
    assert "shower" in _kinds(plan, "bath2") and "bathtub" not in _kinds(plan, "bath2")
    assert _kinds(plan, "util").count("washer") == 2
    assert "shelf" in _kinds(plan, "dress") and "shoe" in _kinds(plan, "entry")
    assert _kinds(plan, "pool") == ["pool"] and "bench" in _kinds(plan, "sauna")  # 종류 'other' 도 이름으로
    assert _kinds(plan, "hall") == []
    p1, p2 = floorplan_render.plan_floor(house, 0), floorplan_render.plan_floor(house, 1)
    assert "stair_up" in _kinds(p1, "stair") and "stair_dn" in _kinds(p2, "stair")
    # 가구는 방(벽 안쪽)을 벗어나지 않고, 같은 방 가구끼리 겹치지 않는다
    for f in plan.furniture:
        assert floorplan_render._inside(f.box, plan.interiors[f.room].bounds), f
    overlaps = [(a.kind, b.kind) for a in plan.furniture for b in plan.furniture
                if a is not b and a.room == b.room and floorplan_render._hit(a.box, b.box, -1e-6)]
    assert not overlaps


def test_furniture_scaled_or_skipped_when_room_too_small():
    layout = parse_layout({"outline": {"w": 9, "h": 4}, "floors": [{"rooms": [
        {"id": "big", "type": "bedroom", "rect": {"x": 0, "y": 0, "w": 4, "h": 4}},
        {"id": "small", "type": "bedroom", "rect": {"x": 4, "y": 0, "w": 2.2, "h": 2.4}},
        {"id": "tiny", "type": "bedroom", "rect": {"x": 4, "y": 2.4, "w": 2.2, "h": 1.6}},
        {"id": "bath", "type": "bathroom", "name": "욕실(욕조)", "rect": {"x": 6.2, "y": 0, "w": 1.3, "h": 1.4}},
        {"id": "st", "type": "storage", "rect": {"x": 6.2, "y": 1.4, "w": 1.3, "h": 2.6}},
        {"id": "wc", "type": "bathroom", "rect": {"x": 7.5, "y": 0, "w": 1.5, "h": 0.6}},
        {"id": "wc2", "type": "bathroom", "rect": {"x": 7.5, "y": 0.6, "w": 1.5, "h": 3.4}},
    ]}]})
    plan = floorplan_render.plan_floor(layout, 0)
    beds = {f.room: f for f in plan.furniture if f.kind == "bed"}
    assert beds["big"].w == 1.5  # 퀸
    assert beds["small"].w < 1.5  # 작은 방은 작은 침대
    assert "tiny" not in beds  # 1.6m 깊이엔 침대가 안 들어간다 → 뺀다
    assert "bathtub" not in _kinds(plan, "bath") and "shower" in _kinds(plan, "bath")  # 1.3×1.4m: 욕조 대신 샤워
    assert _kinds(plan, "wc") == []  # 0.6m 깊이 — 아무것도 안 들어감
    # 문이 열리는 자리에는 가구를 두지 않는다
    for f in plan.furniture:
        keep, _, _ = floorplan_render._keepouts(f.room, plan.interiors[f.room].bounds, plan.geom, plan.kinds)
        assert not any(floorplan_render._hit(f.box, k) for k in keep), f


# ─── 문·창 ────────────────────────────────────────────────────────────


def test_door_styles_and_hinge_near_corner(pool, house):
    room = box(0, 0, 4, 3)
    near_right = Opening(kind="door", room="r", to="x", segment=((3.0, 0.0), (3.9, 0.0)), width=0.9)
    near_left = Opening(kind="door", room="r", to="x", segment=((0.1, 0.0), (1.0, 0.0)), width=0.9)
    assert not floorplan_render._hinge_first(near_right, room) and floorplan_render._hinge_first(near_left, room)

    plan = floorplan_render.plan_floor(pool, 0)
    doors = [o for o in plan.geom.openings if o.kind == "door"]
    assert any(floorplan_render.door_style(o, plan.kinds) == "sliding" for o in doors if "dress" in (o.room, o.to))
    entrance = next(i for i, o in enumerate(doors) if o.to == "outside")
    assert any(p.solid for p in plan.doors[entrance])  # 현관: 출입 표시
    others = [p.lw for i, prims in enumerate(plan.doors) if i != entrance for p in prims]
    assert max(p.lw for p in plan.doors[entrance]) > max(others)  # 현관문은 굵게
    # 문짝은 그 방 안쪽으로 열린다 (호 가운데가 방 안)
    for o, prims in zip(doors, plan.doors):
        for arc in (p for p in prims if p.kind == "arc"):
            mid = math.radians(arc.a0 + (arc.a1 - arc.a0) % 360 / 2)
            pt = Point(arc.center[0] + math.cos(mid) * arc.r * 0.5, arc.center[1] + math.sin(mid) * arc.r * 0.5)
            assert plan.geom.rooms[o.room].buffer(0.01).contains(pt), o
    hplan = floorplan_render.plan_floor(house, 0)
    hdoors = [o for o in hplan.geom.openings if o.kind == "door"]
    stair = [prims for o, prims in zip(hdoors, hplan.doors) if "stair" in (o.room, o.to)]
    assert stair and not any(stair)  # 계단실 입구는 문짝 없는 개구부
    assert sum(p.kind == "poly" and not p.closed for p in plan.windows[0]) >= 2  # 벽 두께 안의 유리 겹선


# ─── 출력 ────────────────────────────────────────────────────────────


def test_dxf_layers_blocks_and_title(pool, house):
    doc = ezdxf.read(io.StringIO(floorplan_render.to_dxf(pool).decode("utf-8")))
    assert set(floorplan_render.LAYERS) <= {layer.dxf.name for layer in doc.layers}
    msp = doc.modelspace()
    inserts = msp.query("INSERT[layer=='A-FURN']")
    names = {e.dxf.name.split("_")[0] for e in inserts}
    assert {"BED", "WARDROBE", "SOFA", "TV", "DINING", "TOILET", "BASIN", "BATHTUB", "SHOWER", "WASHER", "SHELF",
            "SHOE", "POOL", "BENCH"} <= names
    assert all(len(doc.blocks[e.dxf.name]) > 0 for e in inserts)
    assert len(msp.query("HATCH[layer=='A-WALL']")) == 1
    assert len(msp.query("DIMENSION[layer=='A-DIMS']")) >= 15  # 방 연속 치수 4면 + 전체 2면
    anno = " ".join(e.dxf.text for e in msp.query("TEXT[layer=='A-ANNO']"))
    assert "개념·참고용 도면" in anno and "축척 1:100" in anno and "N" in anno
    room_texts = [e.dxf.text for e in msp.query("TEXT[layer=='A-TEXT']")]
    assert "침실5" in room_texts and "욕실1(욕조)" in room_texts
    house_doc = ezdxf.read(io.StringIO(floorplan_render.to_dxf(house).decode("utf-8")))
    assert len(house_doc.modelspace().query("INSERT[layer=='A-STRS']")) == 2


def test_png_svg_and_paper_scale(pool):
    image = Image.open(io.BytesIO(floorplan_render.to_png(pool, 0, dpi=60)))
    assert image.format == "PNG" and image.width > image.height > 200
    svg = floorplan_render.to_svg(pool, 0).decode("utf-8")
    assert svg.startswith("<?xml") and "label:bed5" in svg
    assert [floorplan_render.paper_scale(*wh) for wh in ((30, 20), (60, 30), (61, 30))] == [100, 150, 200]


def test_renders_without_korean_font(pool, monkeypatch):
    """한글 글꼴이 없는 PC(예: 글꼴을 안 깐 Linux)에서도 멈추지 않고 그린다 — 한글은 '없는 글자' 네모 폭으로 잰다.
    예전엔 load_char 가 "No glyphs have been loaded" 로 멈춰 GitHub Actions 에서 평면도 테스트 16개가 실패했다."""
    caches = (floorplan_render._fonts, floorplan_render._glyph, floorplan_render.text_width)
    for cache in caches:
        cache.cache_clear()
    monkeypatch.setattr(floorplan_render, "korean_font", lambda: None)
    try:
        assert floorplan_render.text_width("거실") > 0
        image = Image.open(io.BytesIO(floorplan_render.to_png(pool, 0, dpi=40)))
        assert image.format == "PNG" and image.width > 200
    finally:
        for cache in caches:
            cache.cache_clear()  # 다음 테스트는 실제 글꼴로 다시 잰다


def test_narrow_tub_bathroom_gets_small_bathtub():
    """배치 엔진이 0.3m 격자로 자주 만드는 폭 1.5m 욕실에도 '욕조' 요청이면 욕조(1.2m 소형)를 그린다 — 두 작업 병합 후 찾은 문제."""
    data = {
        "title": "좁은 욕조 욕실", "outline": {"w": 6.0, "h": 2.7},
        "floors": [{"level": 1, "rooms": [
            {"id": "entry", "type": "entrance", "name": "현관", "rect": {"x": 0, "y": 0, "w": 1.5, "h": 2.7}},
            {"id": "living", "type": "living", "name": "거실", "rect": {"x": 1.5, "y": 0, "w": 3.0, "h": 2.7}},
            {"id": "bath1", "type": "bathroom", "name": "욕실(욕조)", "rect": {"x": 4.5, "y": 0, "w": 1.5, "h": 2.7}},
        ]}],
    }
    layout, _ = check(parse_layout(data))
    plan = floorplan_render.plan_floor(layout, 0)
    assert "bathtub" in _kinds(plan, "bath1") and "shower" not in _kinds(plan, "bath1")
