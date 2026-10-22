"""📐 설계 팀 배치 엔진 테스트 — 방 프로그램 모델 · 외곽 유형 · CP-SAT 배치 제약 · 후보안 · 실패 시 단순 분할 · 설계 품질 검사.

실제 LLM 호출 없음. CP-SAT 는 CPU 로 시간 제한(방 수에 따라 3~10초) 안에서 푼다 — 같은 입력은 캐시로 한 번만 푼다.
"""

import json

import pytest
from shapely.geometry import box

from core import floorplan_solver
from core.floorplan import (
    KIND_RULES,
    Layout,
    best_candidate,
    candidates,
    check,
    floor_geometry,
    light_walls,
    max_aspect,
    parse_layout,
    partition,
    program_links,
    room_kind,
    room_zone,
)
from core.harness import REGISTRY

# 실제 사용자 요청을 줄인 프로그램 — 4인 가족 · 100평(연면적 330㎡로 해석) · 1층 · 침실 5 · 욕실 2 · 사우나 · 수영장
BIG = {
    "title": "4인 가족 1층 주택 (100평)",
    "outline": {"w": 28.5, "h": 11.6},
    "floors": [{"level": 1, "rooms": [
        {"id": "entry", "type": "entrance", "name": "현관", "area": 6},
        {"id": "living", "type": "living", "name": "거실", "area": 40},
        {"id": "dining", "type": "dining", "name": "식당", "area": 16},
        {"id": "kitchen", "type": "kitchen", "name": "주방", "area": 14},
        {"id": "master", "type": "bedroom", "name": "안방", "area": 22},
        {"id": "bed2", "type": "bedroom", "name": "침실2", "area": 13},
        {"id": "bed3", "type": "bedroom", "name": "침실3", "area": 13},
        {"id": "bed4", "type": "bedroom", "name": "침실4", "area": 13},
        {"id": "bed5", "type": "bedroom", "name": "침실5", "area": 13},
        {"id": "dress", "type": "dressroom", "name": "드레스룸", "area": 7},
        {"id": "bath1", "type": "bathroom", "name": "욕실1", "area": 6},
        {"id": "bath2", "type": "bathroom", "name": "욕실2", "area": 5},
        {"id": "util", "type": "utility", "name": "다용도실", "area": 5},
        {"id": "sauna", "type": "sauna", "name": "사우나", "area": 6},
        {"id": "pool", "type": "pool", "name": "수영장", "area": 116},
    ]}],
    "assumptions": ["'100평'을 연면적 약 330㎡ 로 해석"],
}

# 작은 단층 (약 110㎡) — 빠르게 풀리는 기본 제약 검사용
SMALL = {
    "title": "3인 가족 단층",
    "outline": {"w": 12.0, "h": 9.0},
    "floors": [{"level": 1, "rooms": [
        {"id": "entry", "type": "entrance", "name": "현관", "area": 5},
        {"id": "living", "type": "living", "name": "거실", "area": 32, "open_to": ["dining"]},
        {"id": "dining", "type": "dining", "name": "식당", "area": 12},
        {"id": "kitchen", "type": "kitchen", "name": "주방", "area": 10},
        {"id": "master", "type": "bedroom", "name": "안방", "area": 16},
        {"id": "bed2", "type": "bedroom", "name": "침실2", "area": 12, "adjacent": ["bath"]},
        {"id": "bath", "type": "bathroom", "name": "욕실", "area": 6},
        {"id": "util", "type": "utility", "name": "다용도실", "area": 5},
    ]}],
}


def rect_room(rid, rtype, x, y, w, h, **kw):
    return {"id": rid, "type": rtype, "rect": {"x": x, "y": y, "w": w, "h": h}, **kw}


def answer_with(layout_dict) -> str:
    return "## 설계 요약\n…\n\n```json\n" + json.dumps(layout_dict, ensure_ascii=False) + "\n```"


def shared(a, b) -> float:
    return a.boundary.intersection(b.boundary).length


@pytest.fixture(scope="module")
def small():
    layout, report = check(parse_layout(SMALL))
    assert report.ok, report.errors
    return layout, report


@pytest.fixture(scope="module")
def big_candidates():
    return candidates(parse_layout(BIG), 3)


# ─── 방 프로그램 모델 ─────────────────────────────────────────────────


def test_program_model_kinds_zones_and_defaults():
    layout = parse_layout(BIG | {"typology": "ㄱ자"})
    rooms = {r.id: r for r in layout.floors[0].rooms}
    assert layout.typology == "L"
    assert rooms["sauna"].type == "other" and room_kind(rooms["sauna"]) == "sauna" and room_zone(rooms["sauna"]) == "wet"
    assert room_kind(rooms["pool"]) == "pool" and room_zone(rooms["pool"]) == "special"
    named = parse_layout({"outline": {"w": 8, "h": 6}, "floors": [{"rooms": [{"id": "x", "type": "other", "name": "실내 수영장"}]}]})
    assert room_kind(named.floors[0].rooms[0]) == "pool"
    assert max_aspect(rooms["bath1"]) == 2.0 and max_aspect(rooms["master"]) == 1.6
    prog = program_links(layout.floors[0])
    assert ("dining", "kitchen") in prog.required  # 주방–식당 기본 필수 이웃
    assert prog.suites["master"] == ["master", "dress", "bath2"]  # 안방 + 드레스룸 + 안방 욕실(마지막 욕실)
    assert ("pool", "sauna") in prog.preferred


def test_typology_shapes_and_outline_decomposition():
    for typ in ("L", "U"):
        shape = floorplan_solver.make_shape(typ, 330.0, 116.0)
        assert shape.polygon.is_valid and shape.polygon.area == pytest.approx(330.0, rel=0.03)
        special = [b for b, role in shape.blocks if role == "special"]
        assert special and special[0].w * special[0].h == pytest.approx(116.0, rel=0.05)  # 수영장 날개
    court = floorplan_solver.make_shape("courtyard", 260.0)
    assert court.court is not None and court.polygon.contains(court.court.polygon())
    assert floorplan_solver.make_shape("courtyard", 80.0) is None  # 작은 집엔 중정이 안 들어간다
    ell = floorplan_solver.make_shape("L", 200.0)
    assert len(floorplan_solver.shape_from_outline(ell.polygon).blocks) == 2


# ─── 배치 제약 ───────────────────────────────────────────────────────


def test_solver_places_without_overlap_with_aspect_adjacency_and_daylight(small):
    layout, report = small
    floor = layout.floors[0]
    outline = layout.outline_polygon()
    polys = {r.id: r.shape() for r in floor.rooms}
    assert sum(p.area for p in polys.values()) == pytest.approx(outline.area, abs=0.1)  # 빈틈 없이
    items = list(polys.items())
    for i, (_, a) in enumerate(items):
        assert outline.buffer(0.01).covers(a)
        for _, b in items[i + 1:]:
            assert a.intersection(b).area < 0.01  # 겹침 없음
    for r in floor.rooms:
        limit = max_aspect(r)
        if limit and r.type not in ("corridor", "stair"):
            w, h = r.rect.w, r.rect.h
            assert max(w, h) / min(w, h) <= limit + 0.02, r.id  # 장단변비
        if r.type in ("bedroom", "living"):
            assert any(s[1][0] - s[0][0] + s[1][1] - s[0][1] >= 0.9 for s in light_walls(floor, r, outline)), r.id  # 외벽
    assert shared(polys["bed2"], polys["bath"]) >= 0.9  # LLM 이 쓴 adjacent
    assert shared(polys["kitchen"], polys["dining"]) >= 0.9  # 기본 필수 이웃
    assert shared(polys["living"], polys["dining"]) >= 2.4  # 트인 방(LDK)은 넓게 맞닿음
    m = report.metrics["1층"]
    assert m["required_ok"] == m["required"] and m["daylight_ok"] == m["daylight"]


def test_open_ldk_has_no_wall_and_no_door(small):
    layout, report = small
    floor = layout.floors[0]
    polys = {r.id: r.shape() for r in floor.rooms}
    line = polys["living"].boundary.intersection(polys["dining"].boundary)
    geom = floor_geometry(layout, floor)
    walls_on_line = [seg for seg, _ in geom.wall_pieces if line.buffer(0.02).covers(box(*seg[0], *seg[0]).centroid)
                     and line.buffer(0.02).covers(box(*seg[1], *seg[1]).centroid)]
    assert not walls_on_line  # 거실–식당 사이 벽 없음
    assert not any({d.room, d.to} == {"living", "dining"} for d in floor.doors)
    assert report.ok  # 문 없이도 모든 방에 갈 수 있다


def test_multi_floor_stairs_align_and_suite_is_entered_through_master():
    from tests.test_floorplan import SAMPLE

    layout, report = check(parse_layout(SAMPLE))
    assert report.ok, report.errors
    stairs = [next(r.rect for r in f.rooms if r.type == "stair") for f in layout.floors]
    assert stairs[0] == stairs[1] == layout.stair
    upper = layout.floors[1]
    dress_doors = [d for d in upper.doors if "dress" in (d.room, d.to)]
    assert dress_doors and all({d.room, d.to} == {"dress", "master"} for d in dress_doors)  # 드레스룸은 안방을 거쳐


def test_big_program_puts_pool_in_end_block_and_short_corridor(big_candidates):
    a = big_candidates[0]
    assert a.layout.typology == "rect" and a.report.ok, a.report.errors  # 빈 마당 없는 꽉 찬 일자형이 기본
    floor = a.layout.floors[0]
    pool = next(r for r in floor.rooms if r.id == "pool").shape()
    shape = floorplan_solver.make_shape("rect", 330.6, 116.0, keep=a.layout.outline)
    end = next(b for b, role in shape.blocks if role == "special")
    assert end.polygon().buffer(0.05).covers(pool)  # 수영장은 동쪽 끝 블록(깊이 전체)에
    m = a.report.metrics["1층"]
    assert m["corridor_share"] <= 0.10 and m["corridor_longest"] <= 18.0
    assert m["worst_aspect"] <= 3.0 + 0.02
    ell = next((c for c in big_candidates if c.layout.typology == "L"), None)
    if ell is not None:  # ㄱ자안은 수영장 날개 + 안쪽 모서리 데크
        wing = next(b for b, role in floorplan_solver.make_shape("L", 330.6, 116.0).blocks if role == "special")
        pool = next(r for r in ell.layout.floors[0].rooms if r.id == "pool").shape()
        assert wing.polygon().buffer(0.05).covers(pool) and ell.layout.outdoor


def test_three_candidates_are_diverse(big_candidates):
    assert [c.label for c in big_candidates] == ["A", "B", "C"]
    sigs = {json.dumps([(r.id, r.rect.model_dump()) for r in c.layout.floors[0].rooms if r.rect], sort_keys=True)
            for c in big_candidates}
    assert len(sigs) == 3
    assert len({c.title for c in big_candidates}) == 3
    assert len({c.layout.typology for c in big_candidates} | {"annex" for c in big_candidates if c.layout.annex}) >= 2
    annexed = [c for c in big_candidates if c.layout.annex]
    assert all(not any(r.id == "pool" for r in c.layout.floors[0].rooms) for c in annexed)  # 별채안은 수영장을 도면에서 뺀다
    assert best_candidate(big_candidates).report.ok


def test_solver_failure_falls_back_to_partition(monkeypatch):
    def boom(*args, **kwargs):
        raise floorplan_solver.SolverFailed("1층: 10초 안에 배치를 찾지 못했다")

    monkeypatch.setattr(floorplan_solver, "solve_layout_cached", boom)
    layout, report = check(parse_layout(SMALL))
    assert all(r.rect for r in layout.floors[0].rooms)  # 예전 단순 분할로라도 도면은 나온다
    assert any("단순 분할" in n and "찾지 못했다" in n for n in report.notes)
    assert any(r.type == "corridor" and r.rect.w >= 6 for r in layout.floors[0].rooms)  # 앞 띠/복도/뒤 띠의 긴 복도
    # 엔진을 직접 끄고 단순 분할만 쓸 수도 있다
    old, _ = check(parse_layout(SMALL), options={"engine": "partition"})
    assert any(r.type == "corridor" for r in old.floors[0].rooms)


def test_annex_placement_removes_room_from_main_house():
    data = json.loads(json.dumps(SMALL))
    data["floors"][0]["rooms"].append({"id": "garage", "type": "other", "name": "차고", "area": 30, "placement": "annex"})
    layout, report = check(parse_layout(data))
    assert [r.id for r in layout.annex] == ["garage"]
    assert not any(r.id == "garage" for r in layout.floors[0].rooms)
    assert any("별채" in n and "차고" in n for n in report.notes)


# ─── 설계 품질 검사 (예전 일자형 막대 평면을 잡아낸다) ─────────────────────


OLD_BAR = {
    # 예전 단순 분할이 실제로 만든 평면을 줄인 것: 28.5×11.6m 막대 + 28.5m 일자 복도 + 1.4×6.0m 욕실 + 2.5×6m 침실
    "outline": {"w": 28.5, "h": 11.6},
    "floors": [{"level": 1, "rooms": [
        rect_room("entry", "entrance", 0, 0, 1.4, 4.4, area=6),
        rect_room("living", "living", 1.4, 0, 8.6, 4.4, area=38),
        rect_room("kitchen", "kitchen", 10.0, 0, 4.0, 4.4, area=18),
        rect_room("pool", "other", 14.0, 0, 14.5, 4.4, name="수영장", area=64),
        rect_room("hall", "corridor", 0, 4.4, 28.5, 1.2),
        rect_room("bath", "bathroom", 0, 5.6, 1.4, 6.0, name="욕실", area=8.4),
        rect_room("b1", "bedroom", 1.4, 5.6, 2.5, 6.0, name="침실1", area=15),
        rect_room("b2", "bedroom", 3.9, 5.6, 2.5, 6.0, name="침실2", area=15),
        rect_room("b3", "bedroom", 6.4, 5.6, 2.5, 6.0, name="침실3", area=15),
        rect_room("master", "bedroom", 8.9, 5.6, 5.6, 6.0, name="안방", area=33),
        rect_room("bath2", "bathroom", 14.5, 5.6, 2.0, 6.0, name="욕실2", area=12),
        rect_room("sauna", "other", 16.5, 5.6, 1.5, 6.0, name="사우나", area=9),
        rect_room("rest", "bedroom", 18.0, 5.6, 10.5, 6.0, name="침실4", area=63),
    ]}],
}


def test_quality_report_flags_old_bar_layout():
    layout, report = check(parse_layout(OLD_BAR))
    errors = " / ".join(report.errors)
    assert "'욕실' 1.40×6.00m — 장단변비 4.3" in errors  # 1.4×6m 욕실
    assert "일자 복도 길이 28.5m" in errors  # 28m 일자 복도
    assert "'침실1' 2.50×6.00m — 장단변비 2.4" in errors
    assert "'사우나'" in errors  # 1.5×6m 사우나
    assert report.metrics["1층"]["corridor_longest"] == pytest.approx(28.5)
    assert any(n.startswith("설계 지표 — 1층") for n in report.notes)


def test_corridor_share_severity():
    def layout_with_hall(depth: float) -> Layout:
        rest = 10 - depth
        return parse_layout({"outline": {"w": 10, "h": 10}, "floors": [{"rooms": [
            rect_room("living", "living", 0, 0, 10, rest / 2, area=10 * rest / 2),
            rect_room("hall", "corridor", 0, rest / 2, 10, depth),
            rect_room("bed", "bedroom", 0, rest / 2 + depth, 5, rest / 2),
            rect_room("bed2", "bedroom", 5, rest / 2 + depth, 5, rest / 2),
        ]}]})

    _, rep = check(layout_with_hall(1.2))  # 12% → 경고
    assert any("복도가 바닥의 12%" in w for w in rep.warnings) and not any("복도가" in e for e in rep.errors)
    _, rep = check(layout_with_hall(2.0))  # 20% → 오류
    assert any("복도가 바닥의 20%" in e and "15% 를 넘는다" in e for e in rep.errors)


def test_required_adjacency_and_daylight_errors_are_explicit_only():
    rooms = [
        rect_room("living", "living", 0, 0, 6, 8),
        rect_room("bed", "bedroom", 6, 0, 4, 3, adjacent=["bath"]),
        rect_room("inner", "bedroom", 6, 3, 2.5, 3),  # 사방이 다른 방 — 외벽 없음
        rect_room("s1", "storage", 8.5, 3, 1.5, 2),
        rect_room("bath", "bathroom", 8.5, 5, 1.5, 3),
        rect_room("s2", "storage", 6, 6, 2.5, 2),
    ]
    _, rep = check(parse_layout({"outline": {"w": 10, "h": 8}, "floors": [{"rooms": rooms}]}))
    errors = " / ".join(rep.errors)
    assert "'욕실'와 '침실'가 맞닿아야 하는데 떨어져 있다" in errors  # LLM 이 쓴 adjacent → 오류
    assert "외벽(또는 중정)에 닿지 않아" in errors and "'침실'" in errors  # 가운데 침실(inner)은 창을 못 낸다


def test_floorplan_validator_feedback_has_design_metrics():
    v = REGISTRY["floorplan"]
    bad = v.check(answer_with(OLD_BAR), {})
    assert not bad.passed and "장단변비" in bad.log and "일자 복도" in bad.log and "설계 지표" in bad.log
    good = v.check(answer_with(SMALL), {})
    assert good.passed and "설계 지표" in good.log


# ─── 화면 (후보안) ───────────────────────────────────────────────────


def test_design_outputs_expose_candidates_and_adoption():
    import design_ui

    out = design_ui.build_outputs(answer_with(SMALL))
    labels = [c["label"] for c in out["candidates"]]
    assert labels[0] == "A" and len(labels) >= 2 and out["adopted"] in labels
    assert all(c["pngs"] and c["report"] is not None for c in out["candidates"])
    other = next(c for c in out["candidates"] if c["label"] != out["adopted"])
    design_ui.adopt_candidate(out, other["label"])
    assert out["adopted"] == other["label"] and out["layout"] is other["layout"]
    saved = Layout.model_validate_json(out["files"]["layout.json"])
    assert [r.rect for r in saved.floors[0].rooms] == [r.rect for r in other["layout"].floors[0].rooms]
    # 좌표가 이미 있는 레이아웃(수정 업무)은 후보가 하나
    fixed = design_ui.build_outputs(answer_with(OLD_BAR))
    assert len(fixed["candidates"]) == 1


def test_partition_still_available_for_old_layouts():
    rooms = [r for r in parse_layout(BIG).floors[0].rooms]
    placed, *_ = partition(parse_layout(BIG).outline, rooms)
    hall = next(r for r in placed if r.type == "corridor")
    assert hall.rect.w > 20  # 예전 단순 분할 = 긴 일자 복도 (비교용 '이전' 그림)
    assert KIND_RULES["bathroom"][1] == 2.0
