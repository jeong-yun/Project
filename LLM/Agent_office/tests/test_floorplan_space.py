"""📐 설계 팀 공간 효율 테스트 — 외곽 효율·요청하지 않은 빈 마당 · 바깥 공간(마당·데크·중정) · 복도 한 줄기 · 치수선 · 풀이 멈춤 기준.

실제 LLM 호출 없음. 좌표를 직접 쓴 레이아웃은 배치 엔진을 쓰지 않아 빠르고 결과가 고정이다.
배치 엔진을 쓰는 검사는 결과의 '성질'(빈 곳 없음·바깥 공간 라벨·복도 조각 수)만 본다 — 시간에 따라 달라지는 좌표는 보지 않는다.
"""

import io
import json

import ezdxf
import pytest
from ortools.sat.python import cp_model
from shapely.geometry import Polygon, box

from core import floorplan_render, floorplan_solver
from core.floorplan import Room, candidates, check, parse_layout
from tests.test_floorplan_solver import BIG

# 마당을 요청한 단층 (방 목록에 '마당')
YARD = {
    "title": "마당이 있는 단층 주택",
    "outline": {"w": 13, "h": 10},
    "floors": [{"level": 1, "rooms": [
        {"id": "entry", "type": "entrance", "name": "현관", "area": 5},
        {"id": "living", "type": "living", "name": "거실", "area": 30, "open_to": ["dining"]},
        {"id": "dining", "type": "dining", "name": "식당", "area": 12},
        {"id": "kitchen", "type": "kitchen", "name": "주방", "area": 10},
        {"id": "master", "type": "bedroom", "name": "안방", "area": 16},
        {"id": "bed2", "type": "bedroom", "name": "침실2", "area": 12},
        {"id": "bed3", "type": "bedroom", "name": "침실3", "area": 11},
        {"id": "bath", "type": "bathroom", "name": "욕실", "area": 6},
        {"id": "util", "type": "utility", "name": "다용도실", "area": 5},
        {"id": "yard", "type": "other", "name": "마당", "area": 40},
    ]}],
}

# ㄱ자 외곽 (12×12 외접 사각형에서 북동쪽 6×6 = 36㎡ 가 빈다), 방 좌표 고정
L_OUTLINE = [(0, 0), (12, 0), (12, 6), (6, 6), (6, 12), (0, 12)]
L_ROOMS = [
    {"id": "entry", "type": "entrance", "rect": {"x": 0, "y": 0, "w": 2, "h": 3}},
    {"id": "living", "type": "living", "rect": {"x": 2, "y": 0, "w": 6, "h": 6}},
    {"id": "kitchen", "type": "kitchen", "rect": {"x": 8, "y": 0, "w": 4, "h": 6}},
    {"id": "hall", "type": "corridor", "rect": {"x": 0, "y": 3, "w": 2, "h": 3}},
    {"id": "bed", "type": "bedroom", "rect": {"x": 0, "y": 6, "w": 6, "h": 6}},
]


def l_layout(**extra):
    return parse_layout({"outline": L_OUTLINE, "floors": [{"rooms": L_ROOMS}], **extra})


# ─── 외곽 효율 지표 ─────────────────────────────────────────────────


def test_footprint_metrics_and_unrequested_void_error():
    _, rep = check(l_layout(typology="L"))
    m = rep.metrics["1층"]
    assert m["footprint_eff"] == pytest.approx(108 / 144, abs=0.001)  # 실내 108㎡ ÷ 외접 144㎡
    assert m["void_max"] == pytest.approx(36.0, abs=0.1)
    assert any("요청하지 않은 빈 곳 36.0㎡" in e for e in rep.errors)  # 엔진·LLM 이 정한 외곽 유형 → 오류
    # 가져온 도면처럼 외곽 유형이 없으면(자유 외곽) 같은 빈 곳도 경고만
    _, rep = check(l_layout())
    assert not any("빈 곳" in e for e in rep.errors) and any("요청하지 않은 빈 곳" in w for w in rep.warnings)


def test_outdoor_space_makes_the_void_planned():
    deck = {"id": "deck", "type": "other", "kind": "deck", "name": "데크", "rect": {"x": 6, "y": 6, "w": 6, "h": 6}}
    _, rep = check(l_layout(typology="L", outdoor=[deck]))
    m = rep.metrics["1층"]
    assert m["void_max"] == 0 and m["footprint_eff_planned"] == pytest.approx(1.0)
    assert m["outdoor_area"] == pytest.approx(36.0)
    assert not any("빈 곳" in e for e in rep.errors) and not any("외곽 효율" in w for w in rep.warnings)
    assert rep.metrics["1층"]["footprint_eff"] < 0.85  # 날 것의 효율은 그대로 보고한다


def test_rect_is_fully_efficient_and_score_prefers_compact():
    _, rect_rep = check(parse_layout(BIG | {"floors": [{"level": 1, "rooms": [
        {"id": "living", "type": "living", "rect": {"x": 0, "y": 0, "w": 6, "h": 5}},
        {"id": "bed", "type": "bedroom", "rect": {"x": 6, "y": 0, "w": 4, "h": 5}}]}], "outline": {"w": 10, "h": 5}}))
    m = rect_rep.metrics["1층"]
    assert m["footprint_eff"] == 1.0 and m["void_max"] == 0


# ─── 배치 엔진: 빈 마당 없음 · 요청한 마당은 바깥 공간 ─────────────────────


@pytest.fixture(scope="module")
def big_cands():
    return candidates(parse_layout(BIG), 3)


def test_big_sample_has_no_unrequested_void(big_cands):
    a = big_cands[0]
    assert a.layout.typology == "rect" and a.report.ok, a.report.errors
    m = a.report.metrics["1층"]
    assert m["footprint_eff"] >= 0.99 and m["void_max"] == 0
    pool = next(r for r in a.layout.floors[0].rooms if r.id == "pool").shape()
    outline = a.layout.outline_polygon()
    assert pool.bounds[1] == pytest.approx(outline.bounds[1]) and pool.bounds[3] == pytest.approx(outline.bounds[3])  # 끝 블록 깊이 전체
    for c in big_cands:  # 어느 후보안에도 요청하지 않은 빈 곳이 없다 (ㄱ자안은 안쪽 모서리가 데크)
        assert c.report.metrics["1층"]["void_max"] <= 15.0, (c.title, c.report.metrics)
        assert not any("빈 곳" in e for e in c.report.errors)
        if c.layout.typology == "L":
            assert c.layout.outdoor and all(r.kind == "deck" for r in c.layout.outdoor)


def test_requested_yard_becomes_labeled_outdoor_space():
    layout, rep = check(parse_layout(YARD))
    assert layout.typology == "L"
    assert not any(r.id == "yard" for r in layout.floors[0].rooms)  # 실내 방이 아니다
    yard = next(r for r in layout.outdoor if r.id == "yard")
    assert yard.kind == "yard" and yard.label == "마당" and yard.shape().area == pytest.approx(40, rel=0.15)
    outline = layout.outline_polygon()
    assert yard.shape().intersection(outline).area < 0.01  # 외곽 밖
    assert box(*outline.bounds).difference(outline).difference(yard.shape()).area < 0.5  # 꺾인 외곽의 빈 곳 = 마당
    assert rep.metrics["1층"]["void_max"] == 0 and not any("빈 곳" in e for e in rep.errors)
    plan = floorplan_render.plan_floor(layout, 0)
    assert any(p.kind == "text" and p.text == "마당 (외부)" for p in plan.outdoor)
    doc = ezdxf.read(io.StringIO(floorplan_render.to_dxf(layout).decode("utf-8")))
    assert any(e.dxf.layer == "A-SITE" for e in doc.modelspace())
    assert floorplan_render.to_png(layout)[:4] == b"\x89PNG"


def test_requested_court_uses_courtyard_and_shape_blocks():
    poly = box(0, 0, 16, 15).difference(box(5, 5, 11, 10))
    assert len(floorplan_solver.shape_from_outline(poly).blocks) == 4  # 중정 구멍 둘레 4블록 (예전엔 0)
    shape = floorplan_solver.make_shape("courtyard", 190, void=30)
    assert shape is not None and min(shape.court.w, shape.court.h) >= floorplan_solver.COURT_MIN
    assert floorplan_solver.make_shape("courtyard", 80.0) is None
    data = json.loads(json.dumps(YARD))
    data["floors"][0]["rooms"][-1] = {"id": "court", "type": "other", "name": "중정", "area": 30}
    assert floorplan_solver.auto_typology(parse_layout(data)) == "courtyard"


# ─── 복도 한 줄기 ───────────────────────────────────────────────────


def test_corridor_fragments_merge_into_one_spine():
    rooms = [Room(id="hall", type="corridor", rect={"x": 0, "y": 3, "w": 4.2, "h": 1.2}),
             Room(id="hall2", type="corridor", rect={"x": 4.2, "y": 3, "w": 3.6, "h": 1.2}),
             Room(id="bed", type="bedroom", rect={"x": 0, "y": 0, "w": 7.8, "h": 3})]
    rules = {"hall2": ["hall"], "bed": ["hall2", "hall"]}
    merged, notes = floorplan_solver.merge_corridors(rooms, rules)
    halls = [r for r in merged if r.type == "corridor"]
    assert len(halls) == 1 and halls[0].rect.w == pytest.approx(7.8) and notes
    assert "hall2" not in rules and rules["bed"] == ["hall"]


def test_corridor_spine_metrics_flag_side_by_side_fragments():
    def layout(second_hall):
        rooms = [
            {"id": "entry", "type": "entrance", "rect": {"x": 0, "y": 0, "w": 2.4, "h": 3}},
            {"id": "living", "type": "living", "rect": {"x": 2.4, "y": 0, "w": 7.6, "h": 4.2}},
            {"id": "hall", "type": "corridor", "rect": {"x": 0, "y": 3, "w": 1.2, "h": 5}},
            second_hall,
            {"id": "bed", "type": "bedroom", "rect": {"x": 2.4, "y": 4.2, "w": 7.6, "h": 3.8}},
        ]
        return parse_layout({"outline": {"w": 10, "h": 8}, "floors": [{"rooms": rooms}]})

    _, rep = check(layout({"id": "hall2", "type": "corridor", "rect": {"x": 1.2, "y": 3, "w": 1.2, "h": 5}}))
    m = rep.metrics["1층"]
    assert m["corridor_pieces"] == 2 and m["corridor_parallel"] == 1
    assert any("나란히 붙어 있다" in w for w in rep.warnings)
    assert "복도 2조각" in next(n for n in rep.notes if n.startswith("설계 지표"))


def test_big_sample_corridors_stay_one_spine(big_cands):
    for c in big_cands:
        m = c.report.metrics["1층"]
        assert m["corridor_pieces"] <= 2 and m["corridor_parallel"] == 0, (c.title, m)


# ─── 치수선은 실제 외벽을 따라 ─────────────────────────────────────────


def test_dimension_chains_follow_the_outline_not_the_bounding_box():
    layout, _ = check(l_layout(typology="L"))
    plan = floorplan_render.plan_floor(layout, 0)
    outline = Polygon(L_OUTLINE)
    north = [ch for ch in plan.chains if ch.side == "N"]
    assert 6.0 in {ch.edge for ch in north} and {ch.edge for ch in north} <= {6.0, 12.0}  # 꺾인 북쪽 외벽마다 따로
    assert all(6.0 - 0.01 <= v <= 12.01 for ch in north if ch.edge == 6.0 for v in ch.coords)
    assert not any(v > 6.01 for ch in north if ch.edge == 12.0 for v in ch.coords)  # 빈 북동쪽 위에 치수가 없다
    for ch in plan.chains:
        if ch.tier != floorplan_render.DIM_TIERS[0]:
            continue
        for a, b in zip(ch.coords, ch.coords[1:]):  # 치수 칸마다 바로 안쪽이 건물이다 (빈 외접 사각형이 아님)
            mid = (a + b) / 2
            sign = -1 if ch.side in "SW" else 1
            probe = (mid, ch.edge - sign * 0.1) if ch.side in "SN" else (ch.edge - sign * 0.1, mid)
            assert outline.contains(box(*probe, *probe).centroid), (ch, a, b)
    west_total = next(ch for ch in plan.chains if ch.side == "W" and ch.tier == floorplan_render.DIM_TIERS[1])
    assert west_total.faces == [0.0, 0.0]  # 전체 치수 보조선은 실제 외벽에서
    assert floorplan_render._face(outline, "N", 9.0) == 6.0 and floorplan_render._face(outline, "N", 3.0) == 12.0


# ─── 풀이 멈춤 기준 (CPU 부하) ─────────────────────────────────────────


def test_solver_limits_wall_mode_and_work_budget(monkeypatch):
    s = cp_model.CpSolver()
    floorplan_solver._set_limits(s, 10.0, 3, "wall")
    assert s.parameters.max_time_in_seconds == pytest.approx(10.0) and s.parameters.random_seed == 3
    assert floorplan_solver.work_budget(10.0) == pytest.approx(10.0 * floorplan_solver.DET_PER_SEC)
