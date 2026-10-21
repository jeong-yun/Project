"""📐 설계 팀 테스트 — 레이아웃 스키마·검증·자동 분할·DXF/PNG·3D·첨부 가져오기·툴·하네스·업무 양식.

네트워크·실제 LLM 호출 없음 (비전 모델은 가짜로 바꾼다).
"""

import copy
import io
import json

import ezdxf
import pytest
import trimesh
from PIL import Image

from core import floorplan_3d, floorplan_import, floorplan_render, llm_factory
from core.config import list_teams
from core.exporter import export_adopted
from core.floorplan import (
    Layout,
    LayoutParseError,
    Rect,
    Room,
    check,
    dump_layout,
    find_layout_json,
    floor_geometry,
    parse_layout,
    partition,
    validate,
)
from core.harness import REGISTRY
from core.live import new_run
from core.schemas import HumanDecision, TaskField
from core.tasks import build_request, render, store_uploads
from core.tools import floorplan_input

# 면적 합을 외곽에 맞춘 2층 주택 프로그램 (좌표 없음 → 코드가 나눔). 검증을 통과한다
SAMPLE = {
    "title": "4인 가족 2층 주택",
    "outline": {"w": 12, "h": 9},
    "floors": [
        {"level": 1, "rooms": [
            {"id": "living", "type": "living", "name": "거실", "area": 38},
            {"id": "kitchen", "type": "kitchen", "name": "주방·식당", "area": 20},
            {"id": "entry", "type": "entrance", "name": "현관", "area": 4},
            {"id": "bath1", "type": "bathroom", "name": "욕실", "area": 6},
            {"id": "guest", "type": "bedroom", "name": "손님방", "area": 14},
            {"id": "util", "type": "utility", "name": "다용도실", "area": 6},
        ]},
        {"level": 2, "rooms": [
            {"id": "family", "type": "living", "name": "가족실", "area": 26, "zone": "front"},
            {"id": "master", "type": "bedroom", "name": "안방", "area": 20, "zone": "front"},
            {"id": "bed3", "type": "bedroom", "name": "침실3", "area": 14, "zone": "front"},
            {"id": "bed2", "type": "bedroom", "name": "침실2", "area": 13},
            {"id": "bath2", "type": "bathroom", "name": "욕실2", "area": 7},
            {"id": "dress", "type": "dressroom", "name": "드레스룸", "area": 6},
        ]},
    ],
    "assumptions": ["대지 정보가 없어 건물 12×9m 로 가정"],
}


def rect_room(rid, rtype, x, y, w, h, **kw):
    return {"id": rid, "type": rtype, "rect": {"x": x, "y": y, "w": w, "h": h}, **kw}


def one_floor(rooms, doors=None, **kw):
    """좌표를 직접 쓴 1층짜리 레이아웃 (외곽 10×8)."""
    floor = {"level": 1, "rooms": rooms}
    if doors is not None:
        floor["doors"] = doors
    return parse_layout({"outline": {"w": 10, "h": 8}, "floors": [floor], **kw})


@pytest.fixture
def resolved():
    layout, report = check(parse_layout(SAMPLE))
    assert report.ok, report.errors
    return layout


def answer_with(layout_dict) -> str:
    return "## 설계 요약\n방 구성…\n\n```json\n" + json.dumps(layout_dict, ensure_ascii=False) + "\n```"


# ─── 스키마·검증 ──────────────────────────────────────────────────────


def test_schema_accepts_program_and_maps_unknown_type():
    layout = parse_layout({"outline": {"w": 8, "h": 6}, "floors": [{"rooms": [{"id": "a", "type": "sauna"}]}]})
    assert layout.floors[0].rooms[0].type == "other" and layout.floors[0].label == "1층"


def test_schema_errors_are_korean():
    with pytest.raises(LayoutParseError, match="형식 오류"):
        parse_layout({"outline": {"w": -1, "h": 6}, "floors": []})
    with pytest.raises(LayoutParseError, match="파싱 오류"):
        find_layout_json('```json\n{"floors": [}\n```')


def test_sample_program_passes_and_is_deterministic(resolved):
    again, _ = check(parse_layout(SAMPLE))
    assert dump_layout(again) == dump_layout(resolved)
    assert all(r.rect for f in resolved.floors for r in f.rooms)
    assert {r.type for r in resolved.floors[0].rooms} >= {"corridor", "stair", "entrance"}
    assert any(d.to == "outside" for d in resolved.floors[0].doors)


def test_overlap_is_reported():
    layout = one_floor(
        [rect_room("a", "living", 0, 0, 6, 8), rect_room("b", "bedroom", 5, 0, 5, 8)],
        doors=[{"room": "a", "to": "outside", "side": "S"}, {"room": "b", "to": "a"}],
    )
    errors = check(layout)[1].errors
    assert any("겹친다" in e for e in errors)


def test_room_outside_outline_is_reported():
    layout = one_floor([rect_room("a", "living", 0, 0, 6, 8), rect_room("b", "bedroom", 6, 0, 5, 8)])
    errors = check(layout)[1].errors
    assert any("외곽을 벗어난다" in e and "x 0~10m" in e for e in errors)


def test_unreachable_room_is_reported():
    rooms = [rect_room("a", "living", 0, 0, 5, 8), rect_room("b", "bedroom", 5, 0, 5, 4), rect_room("c", "bedroom", 5, 4, 5, 4)]
    layout = one_floor(rooms, doors=[{"room": "a", "to": "outside"}, {"room": "b", "to": "a"}])
    errors = check(layout)[1].errors
    assert any("갈 수 없는 방" in e and "침실" in e for e in errors)
    # 문을 비워 두면 코드가 이어 준다
    assert check(one_floor(rooms))[1].ok


def test_door_not_on_shared_wall_is_reported():
    rooms = [rect_room("a", "living", 0, 0, 5, 8), rect_room("b", "bedroom", 5, 0, 5, 8)]
    layout = one_floor(rooms, doors=[{"room": "a", "to": "outside", "side": "S"}, {"room": "b", "to": "a", "side": "E", "offset": 1}])
    assert any("맞닿은 벽 위에 있지 않다" in e for e in check(layout)[1].errors)


def test_stair_mismatch_between_floors_is_reported():
    def floor(level, sx):
        return {"level": level, "rooms": [
            rect_room("hall", "corridor", 0, 0, 10, 2),
            rect_room("st", "stair", sx, 2, 2.5, 3),
            rect_room("r1", "bedroom", 0 if sx else 2.5, 2, 7.5, 3),
            rect_room("r2", "living", 0, 5, 10, 3),
        ]}

    layout = parse_layout({"outline": {"w": 10, "h": 8}, "floors": [floor(1, 0), floor(2, 7.5)]})
    errors = check(layout)[1].errors
    assert any("계단 위치가 1층과 2층에서 다르다" in e for e in errors)
    no_stair = copy.deepcopy(SAMPLE)
    layout = parse_layout({**no_stair, "floors": [floor(1, 0), {"level": 2, "rooms": [rect_room("r", "bedroom", 0, 0, 10, 8)]}]})
    assert any("계단(type: stair)이 없다" in e for e in check(layout)[1].errors)


def test_minimum_size_and_area_tolerance():
    rooms = [rect_room("a", "living", 0, 0, 8, 8, area=60), rect_room("b", "bedroom", 8, 0, 2, 8, area=16)]
    errors = check(one_floor(rooms))[1].errors
    assert any("최소 폭 2.4m" in e for e in errors)
    rooms = [rect_room("a", "living", 0, 0, 5, 8, area=60), rect_room("b", "bedroom", 5, 0, 5, 8, area=20)]  # 합은 80 = 외곽
    errors = check(one_floor(rooms))[1].errors
    assert any("침실" in e and "+100%" in e for e in errors) and any("거실" in e and "-33%" in e for e in errors)


def test_target_sum_mismatch_gives_single_actionable_error():
    small = copy.deepcopy(SAMPLE)
    for room in small["floors"][0]["rooms"]:
        room["area"] = room["area"] / 2
    errors = [e for e in check(parse_layout(small))[1].errors if e.startswith("1층")]
    assert len(errors) == 1 and "목표 면적 합" in errors[0]


def test_partial_coordinates_are_rejected():
    layout = one_floor([rect_room("a", "living", 0, 0, 5, 8), {"id": "b", "type": "bedroom"}])
    assert any("일부 방만" in e for e in check(layout)[1].errors)


# ─── 자동 분할 ────────────────────────────────────────────────────────


def test_partition_fills_outline_without_overlap_and_aligns_stairs(resolved):
    for floor in resolved.floors:
        geom = floor_geometry(resolved, floor)
        polys = list(geom.rooms.values())
        assert sum(p.area for p in polys) == pytest.approx(geom.outline.area, abs=0.1)
        for i, a in enumerate(polys):
            for b in polys[i + 1:]:
                assert a.intersection(b).area < 0.01
    stairs = [next(r.rect for r in f.rooms if r.type == "stair") for f in resolved.floors]
    assert stairs[0] == stairs[1] == resolved.stair


def test_partition_stacks_small_rooms_and_single_room():
    # 뒤 구역 깊이 약 4.2m → 욕실(약 3.5㎡)은 폭 0.85m 로 너무 좁아 수납과 한 칸에 쌓인다
    rooms = [Room(id="b1", type="bedroom", area=40), Room(id="wc", type="bathroom", area=4), Room(id="st", type="storage", area=3),
             Room(id="lv", type="living", area=50)]
    placed, stair, split, _ = partition(Rect(w=10, h=10), rooms)
    assert stair is None and split is not None
    wc, st = (next(r.rect for r in placed if r.id == k) for k in ("wc", "st"))
    assert wc.x == st.x and wc.w == st.w  # 한 칸에 위아래로
    only, *_ = partition(Rect(w=5, h=4), [Room(id="studio", type="living")])
    assert only[0].rect == Rect(w=5, h=4)


def test_partition_needs_rooms():
    with pytest.raises(ValueError, match="나눌 방이 없다"):
        partition(Rect(w=5, h=4), [Room(id="h", type="corridor")])


# ─── 하네스 ───────────────────────────────────────────────────────────


def test_floorplan_validator():
    v = REGISTRY["floorplan"]
    assert v.check("치수 체크리스트만 있는 답변", {}).passed
    assert not v.check('```json\n{"floors": [\n```', {}).passed
    bad = {"outline": {"w": 10, "h": 8}, "floors": [{"rooms": [rect_room("a", "living", 0, 0, 12, 8)]}]}
    result = v.check(answer_with(bad), {})
    assert not result.passed and "외곽을 벗어난다" in result.log and "rect 를 모두 빼고" in result.log
    ok = v.check(answer_with(SAMPLE), {})
    assert ok.passed and "2개 층" in ok.log


def test_design_team_config():
    team = next(t for t in list_teams() if t.id == "design")
    assert team.features == ["floorplan_render"] and team.tools == ["floorplan_input"]
    # 직급이 높을수록 위 — 대리(방 프로그램) → 차장(JSON) → 부장(평면 검토) → 팀장(최종)
    assert [(n.id, n.rank) for n in team.org.walk()] == [
        ("lead", "team_lead"), ("gm_review", "gm"), ("dgm_layout", "deputy_gm"), ("am_program", "assistant_manager")
    ]
    assert "json_keys" in team.harness.final_only and team.harness.also_at == {"json_keys": ["dgm_layout"]}
    edit = next(t for t in team.tasks if t.id == "edit")
    assert next(f for f in edit.fields if f.key == "attachment").type == "file"


# ─── 그리기 ───────────────────────────────────────────────────────────


def test_dxf_has_layers_sheets_dims_and_embedded_json(resolved, tmp_path):
    path = tmp_path / "plan.dxf"
    path.write_bytes(floorplan_render.to_dxf(resolved))
    doc = ezdxf.readfile(path)
    assert {"A-WALL", "A-DOOR", "A-GLAZ", "A-ROOM", "A-TEXT", "A-DIMS"} <= {layer.dxf.name for layer in doc.layers}
    assert [lay.name for lay in doc.layouts if lay.name != "Model"] == ["1층", "2층"]
    msp = doc.modelspace()
    assert len(msp.query("DIMENSION")) >= 4 and len(msp.query("ARC[layer=='A-DOOR']")) >= 8
    rooms = msp.query("LWPOLYLINE[layer=='A-ROOM']")
    assert len(rooms) == sum(len(f.rooms) for f in resolved.floors)
    assert floorplan_render.XRECORD_KEY in doc.rootdict


def test_png_and_svg_preview(resolved):
    png = floorplan_render.to_png(resolved, 1, dpi=60)
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.width > 200
    svg = floorplan_render.to_svg(resolved, 0).decode("utf-8")
    assert svg.startswith("<?xml") and "<svg" in svg


# ─── 3D ──────────────────────────────────────────────────────────────


def test_stl_is_watertight_and_scaled(resolved):
    stl, notes, watertight = floorplan_3d.stl_bytes(resolved, 100)
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
    assert watertight and mesh.is_watertight
    size = mesh.bounds[1] - mesh.bounds[0]
    assert size[0] == pytest.approx(122, abs=1)  # 12m + 외벽 → 1:100 이면 약 122mm
    assert any("1:100" in n for n in notes)


def test_print_check_thickens_thin_walls(resolved):
    notes = floorplan_3d.print_check(resolved, 200)
    assert any("내벽" in n and "0.60mm" in n for n in notes)
    stl, notes, watertight = floorplan_3d.stl_bytes(resolved, 200)
    assert watertight and any("두껍게" in n for n in notes)


def test_walls_have_openings_and_glb_obj(resolved):
    parts = floorplan_3d.build_parts(resolved)
    walls = [p for p in parts if p.kind == "wall" and p.level == 1]
    full_height = floor_geometry(resolved, resolved.floors[0]).walls_full.area * resolved.floors[0].height
    assert sum(p.mesh.volume for p in walls) < full_height * 0.97  # 문·창 자리는 비어 있다
    glb = trimesh.load(io.BytesIO(floorplan_3d.glb_bytes(resolved)), file_type="glb")
    assert len(glb.geometry) >= 1
    assert floorplan_3d.obj_bytes(resolved).startswith(b"v ") or b"\nv " in floorplan_3d.obj_bytes(resolved)[:2000]


@pytest.mark.skipif(not floorplan_3d.ifc_available(), reason="ifcopenshell 없음")
def test_ifc_has_storeys_walls_spaces(resolved, tmp_path):
    import ifcopenshell

    path = tmp_path / "m.ifc"
    path.write_bytes(floorplan_3d.ifc_bytes(resolved))
    model = ifcopenshell.open(str(path))
    assert [s.Name for s in model.by_type("IfcBuildingStorey")] == ["1층", "2층"]
    assert len(model.by_type("IfcSpace")) == sum(len(f.rooms) for f in resolved.floors)
    assert len(model.by_type("IfcWall")) > 10


def test_plotly_viewer(resolved):
    fig = floorplan_3d.plotly_figure(resolved)
    assert fig.data[0].type == "mesh3d" and len(fig.data[0].i) > 100


# ─── 첨부 가져오기 ────────────────────────────────────────────────────


def test_dxf_round_trip_exact_and_by_shapes(resolved):
    data = floorplan_render.to_dxf(resolved)
    exact = floorplan_import.from_dxf(data)
    assert exact.exact and dump_layout(exact.layout) == dump_layout(resolved)

    # 다른 프로그램이 만든 DXF 처럼: 안에 넣어 둔 JSON 을 지우고 1층 도형만 남긴다
    doc = ezdxf.read(io.StringIO(data.decode("utf-8")))
    del doc.rootdict[floorplan_render.XRECORD_KEY]
    width = resolved.outline.w
    msp = doc.modelspace()
    for e in list(msp):
        if e.dxftype() in ("LWPOLYLINE", "TEXT") and min(p[0] for p in _points(e)) > width + 1:
            msp.delete_entity(e)
    msp.add_lwpolyline([(0, 0), (12, 0), (12, 9), (0, 9)], close=True, dxfattribs={"layer": "A-ROOM"})
    stream = io.StringIO()
    doc.write(stream)
    approx = floorplan_import.from_dxf(stream.getvalue().encode("utf-8"))
    assert not approx.exact and approx.layout.needs_check
    rooms = approx.layout.floors[0].rooms
    first = resolved.floors[0].rooms
    assert len(rooms) == len(first)
    assert sorted(round(r.shape().area, 1) for r in rooms) == sorted(round(r.shape().area, 1) for r in first)
    assert {r.type for r in rooms} >= {"living", "kitchen", "bathroom", "bedroom", "stair", "corridor"}
    assert check(approx.layout)[1].ok


def _points(e):
    if e.dxftype() == "LWPOLYLINE":
        return list(e.get_points("xy"))
    return [e.dxf.insert]


def test_dxf_in_millimetres_is_scaled():
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (8000, 0), (8000, 6000), (0, 6000)], close=True)
    msp.add_lwpolyline([(0, 0), (4000, 0), (4000, 6000), (0, 6000)], close=True)
    msp.add_lwpolyline([(4000, 0), (8000, 0), (8000, 6000), (4000, 6000)], close=True)
    msp.add_text("거실", dxfattribs={"insert": (2000, 3000)})
    msp.add_text("침실1", dxfattribs={"insert": (6000, 3000)})
    stream = io.StringIO()
    doc.write(stream)
    result = floorplan_import.from_dxf(stream.getvalue().encode())
    assert result.layout.outline == Rect(w=8, h=6)
    assert {r.type for r in result.layout.floors[0].rooms} == {"living", "bedroom"}
    assert any("mm" in n for n in result.notes)


@pytest.fixture
def fake_vision(monkeypatch, set_keys):
    set_keys("google")
    calls = []

    def fake(rank_models, models, schema, messages, who):
        calls.append(messages)
        parsed = floorplan_import.VisionLayout(
            analysis="치수 없음", width=8, depth=6, dimensions_read=False, uncertain=["욕실 크기 추정"],
            floors=[floorplan_import.VisionFloor(level=1, rooms=[
                floorplan_import.VisionRoom(name="거실", type="living", x=0, y=0, w=5, h=6),
                floorplan_import.VisionRoom(name="화장실", type="toilet", x=5, y=0, w=3, h=2),
                floorplan_import.VisionRoom(name="침실", type="bedroom", x=5, y=2, w=3, h=4),
            ])],
        )
        return llm_factory.StructuredCall(parsed=parsed, spec=rank_models.primary, is_fallback=False, elapsed_sec=0.1)

    monkeypatch.setattr(llm_factory, "invoke_structured", fake)
    return calls


def _image_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), "white").save(buf, format=fmt)
    return buf.getvalue()


def test_image_and_pdf_import_use_vision_model(fake_vision, models, tmp_path):
    png = tmp_path / "plan.png"
    png.write_bytes(_image_bytes("PNG"))
    result = floorplan_import.import_file(png, models)
    assert not result.exact and result.layout.outline == Rect(w=8, h=6)
    rooms = {r.name: r.type for r in result.layout.floors[0].rooms}
    assert rooms == {"거실": "living", "화장실": "bathroom", "침실": "bedroom"}  # 모르는 종류는 이름으로 추정
    assert any("확인 필요" in c for c in result.layout.needs_check) and "욕실 크기 추정" in result.layout.needs_check
    content = fake_vision[0][1].content
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    pdf = tmp_path / "plan.pdf"
    pdf.write_bytes(_image_bytes("PDF"))
    assert floorplan_import.import_file(pdf, models).layout is not None
    assert fake_vision[1][1].content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_import_without_vision_key(models, set_keys, tmp_path):
    set_keys()
    png = tmp_path / "plan.jpg"
    png.write_bytes(_image_bytes("JPEG"))
    result = floorplan_import.import_file(png, models)
    assert result.layout is None and "API 키" in result.notes[0]


def test_dwg_and_unknown_files_explain(models, tmp_path):
    dwg = tmp_path / "a.dwg"
    dwg.write_bytes(b"AC1032")
    assert "ODA File Converter" in floorplan_import.import_file(dwg, models).notes[0]
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    assert "지원하지 않는" in floorplan_import.import_file(txt, models).notes[0]


# ─── 데이터 툴 ────────────────────────────────────────────────────────


def test_tool_converts_attachment_to_json_text(resolved, models, tmp_path):
    path = tmp_path / "old_layout.json"
    path.write_text(dump_layout(resolved), encoding="utf-8")
    result = floorplan_input.run("층 추가", {"attachment": str(path), "attachment_name": "우리집.json"}, models)
    assert "우리집.json" in result.content and "정확" in result.title
    assert "✅ 검증 통과" in result.content and "| 1층 | 거실 |" in result.content
    data = find_layout_json(result.content)
    assert data["floors"][0]["rooms"][0]["rect"]  # 에이전트는 좌표가 채워진 JSON 을 받는다
    assert "최소 치수" in result.content


def test_tool_without_attachment_gives_standards_and_uses_previous(resolved, models, tmp_path, monkeypatch):
    plain = floorplan_input.run("집 설계", {}, models)
    assert plain.title == "📏 설계 기준표" and "bedroom" in plain.content
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path))
    missing = floorplan_input.run("3D", {"use_previous": True}, models)
    assert "저장된 설계안" in missing.notes[0]
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "20261021-120000_집_layout.json").write_text(dump_layout(resolved), encoding="utf-8")
    found = floorplan_input.run("3D", {"use_previous": True}, models)
    assert "20261021-120000_집_layout.json" in found.content and "```json" in found.content


def test_tool_reports_unreadable_attachment(models, tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{not json", encoding="utf-8")
    result = floorplan_input.run("수정", {"attachment": str(bad)}, models)
    assert "읽기 실패" in result.title and "새로 설계" in result.content


# ─── 업무 양식 (file·checkbox 칸) ──────────────────────────────────────


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


def test_store_uploads_saves_files_and_keeps_name(tmp_path):
    values = store_uploads({"attachment": FakeUpload("우리 집 도면.json", b"{}"), "change": "층 추가", "x": None}, tmp_path)
    saved = tmp_path / values["attachment"].split("\\")[-1].split("/")[-1]
    assert saved.read_bytes() == b"{}" and values["attachment_name"] == "우리 집 도면.json"
    assert values["change"] == "층 추가" and values["x"] is None


def test_checkbox_and_file_values_in_templates():
    assert render("3D: {want_3d}\n축척: {scale}", {"want_3d": True, "scale": "1:100"}) == "3D: 예\n축척: 1:100"
    assert render("3D: {want_3d}\n끝", {"want_3d": False}) == "끝"
    assert TaskField(key="f", label="첨부", type="file", options=["json"]).type == "file"
    team = next(t for t in list_teams() if t.id == "design")
    edit = next(t for t in team.tasks if t.id == "edit")
    built = build_request(edit, {"attachment": "/tmp/a.json", "attachment_name": "a.json", "change": "2층 추가", "want_3d": False})
    assert "첨부 파일: a.json" in built.question and "3D" not in built.question
    assert built.tool_inputs["attachment"] == "/tmp/a.json"


# ─── 후처리·저장 ──────────────────────────────────────────────────────


def test_design_outputs_and_export(tmp_path):
    import design_ui

    assert design_ui.options_from_inputs({"want_3d": True, "print_scale": "1:50"}) == {"want_3d": True, "print_scale": 50}
    out = design_ui.build_outputs(answer_with(SAMPLE))
    assert out["report"].ok and not out["has_3d"]
    assert {"layout.json", "평면도.dxf", "1층.png", "2층.svg"} <= out["files"].keys()
    design_ui.add_3d(out)
    assert {"3d.glb", "3d.obj", "3d_1-100.stl"} <= out["files"].keys()
    assert "error" in design_ui.build_outputs("JSON 없는 답변")

    run = new_run("r1", "2층 주택 설계", "lead")
    run["design"] = out
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=answer_with(SAMPLE), score=5)
    files = design_ui.adopted_files(run, decision)
    paths = export_adopted(run, decision, "design", "📐 설계 팀", {"lead": "팀장"}, tmp_path, extra_files=files)
    names = [p.name for p in paths]
    assert names[0].endswith(".md") and any(n.endswith("_layout.json") for n in names) and any(n.endswith("_평면도.dxf") for n in names)
    saved = next(p for p in paths if p.name.endswith("_layout.json"))
    assert Layout.model_validate_json(saved.read_text(encoding="utf-8")).floors[1].rooms


def test_adopted_files_redraws_edited_answer():
    import design_ui

    run = new_run("r1", "설계", "lead")
    run["design"] = design_ui.build_outputs(answer_with(SAMPLE))
    smaller = copy.deepcopy(SAMPLE)
    smaller["floors"] = smaller["floors"][:1]
    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=answer_with(smaller), edited=True, score=4)
    files = design_ui.adopted_files(run, decision)
    assert "2층.png" not in files and "1층.png" in files


# ─── 워크플로우 연결 (가짜 모델) ──────────────────────────────────────


def test_design_workflow_tool_harness_and_final_json(ranks, models, use_script, tmp_path):
    from core.config import team_org
    from core.workflow import build_office_graph, stream_office
    from tests.fakes import Script, make_review

    team = next(t for t in list_teams() if t.id == "design")
    old = tmp_path / "old_layout.json"
    old.write_text(dump_layout(check(parse_layout(SAMPLE))[0]), encoding="utf-8")
    bad = {"outline": {"w": 12, "h": 9}, "floors": [{"rooms": [rect_room("a", "living", 0, 0, 14, 9)]}]}
    # 대리: 방 프로그램표(JSON 없음) → 차장: 처음엔 외곽을 벗어난 JSON → 하네스 피드백 → 고친 JSON → 부장·팀장: 그대로
    program = "| 층 | 방 | 목표 면적 |\n|---|---|---|\n| 2 | 안방 | 24㎡ |"
    reviews = [make_review("revise", 8, answer_with(bad)), make_review("revise", 8, answer_with(SAMPLE)), make_review("approve", 9)]
    script = use_script(Script(text=program, ReviewResult=reviews))
    edit = next(t for t in team.tasks if t.id == "edit")
    req = build_request(edit, {"attachment": str(old), "attachment_name": "우리집.json", "change": "안방을 넓게"})
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    events = list(stream_office(graph, org, req.question, "lead", brief=req.brief, auto_brief=False,
                                tool_inputs=req.tool_inputs, extra_validators=req.validators, harness_options=req.harness_options))

    tools = next(p for k, p in events if k == "tools")
    assert tools[0].tool == "floorplan_input" and "정확" in tools[0].title
    first_prompt = script.prompts("text")[0]
    assert "첨부 도면의 레이아웃 JSON" in first_prompt and "레이아웃 JSON 쓰기" in first_prompt  # 툴 결과 + 스킬
    assert "외곽을 벗어난다" in script.prompts("ReviewResult")[1]  # 하네스 에러가 그대로 다시 쓰기 지시로
    steps = [p for k, p in events if k == "step"]
    program_step = next(s for s in steps if s.node_id == "am_program")
    assert {h.validator for h in program_step.harness} == {"floorplan"}  # 대리는 JSON 을 쓰지 않으므로 json_keys 없음
    layout = next(s for s in steps if s.node_id == "dgm_layout")
    assert layout.attempts == 2 and all(h.passed for h in layout.harness)
    assert {h.validator for h in layout.harness} == {"floorplan", "json_keys"}  # also_at: 차장 단계부터 JSON 필수
    final = next(s for s in steps if s.node_id == "lead")
    assert {h.validator for h in final.harness} == {"floorplan", "json_keys"} and all(h.passed for h in final.harness)


def test_layout_designer_cannot_pass_program_without_json(ranks, models, use_script):
    """차장이 대리의 방 프로그램을 '승인'만 하고 넘기면 JSON 이 없다 → also_at 의 json_keys 가 걸러 다시 쓰게 한다."""
    from core.config import team_org
    from core.workflow import build_office_graph, stream_office
    from tests.fakes import Script, make_review

    team = next(t for t in list_teams() if t.id == "design")
    reviews = [make_review("approve", 9), make_review("revise", 8, answer_with(SAMPLE)), make_review("approve", 9)]
    script = use_script(Script(text="| 층 | 방 | 목표 면적 |\n|---|---|---|\n| 1 | 거실 | 30㎡ |", ReviewResult=reviews))
    plan = next(t for t in team.tasks if t.id == "plan")
    req = build_request(plan, {"request": "4인 가족 2층 주택", "floors": "2층", "print_scale": "1:100"})
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    steps = [p for k, p in stream_office(graph, org, req.question, "lead", brief=req.brief, auto_brief=False,
                                         tool_inputs=req.tool_inputs, extra_validators=req.validators,
                                         harness_options=req.harness_options) if k == "step"]
    layout = next(s for s in steps if s.node_id == "dgm_layout")
    assert layout.attempts == 2 and "```json" in layout.content
    assert "JSON 블록이 없다" in script.prompts("ReviewResult")[1]  # json_keys 실패 문장이 다시 쓰기 지시로


def test_also_at_must_name_final_only_validator_and_known_agent():
    from core.config import ConfigError, load_ranks, load_models, validate_config

    team = next(t for t in list_teams() if t.id == "design")
    broken = team.model_copy(update={"harness": team.harness.model_copy(update={"also_at": {"json_keys": ["nobody"]}})})
    with pytest.raises(ConfigError, match="also_at"):
        validate_config(load_ranks(), load_models(), [broken])
