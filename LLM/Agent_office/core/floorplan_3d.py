"""3D 모델 — 레이아웃 JSON(core/floorplan.py) → 벽을 세운 3D (GLB·OBJ·STL·IFC) + 화면용 plotly 뷰어.

사람이 3D 를 요청했을 때만 만든다 (평면도만으로 충분한 경우가 많고, 파일이 크다). 모두 CPU 로 돈다 (GPU 불필요).

  벽     : 2D 벽(개구부 포함)을 높이 구간별로 세운다.
           [바닥~창대] 문만 뚫림 · [창대~문/창 윗면] 문·창 뚫림 · [윗면~층고] 막힘(인방)
  바닥판 : 층마다 외곽 모양으로 두께 0.15m. 지붕은 없다 (위에서 안을 들여다보는 '인형의 집' 모형)
  STL    : 3D 프린팅용. 1:N 축척(mm), 조각을 하나로 합쳐(manifold3d) 물이 새지 않는(watertight) 덩어리로 만든다.
           벽이 축척 후 최소 두께(기본 1mm)보다 얇으면 경고하고, 원하면 그 두께까지 두껍게 만든다.
  IFC    : ifcopenshell (설치돼 있을 때만). 층(IfcBuildingStorey)·벽(IfcWall)·바닥(IfcSlab)·방(IfcSpace).
           개념 설계용 — 문·창은 IFC 에 넣지 않는다 (벽 조각만).
"""

import io
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString
from shapely.ops import unary_union

from core.floorplan import Layout, WallSpec, floor_geometry

SLAB = 0.15  # 바닥판 두께 m
WALL_COLOR = (205, 205, 205, 255)
SLAB_COLOR = (222, 205, 170, 255)
DEFAULT_MIN_WALL_MM = 1.0  # FDM 3D 프린터에서 무난한 최소 벽 두께


@dataclass
class Part:
    mesh: trimesh.Trimesh
    color: tuple[int, int, int, int]
    kind: str  # wall / slab
    level: int


def _extrude(geom, height: float, z: float) -> list[trimesh.Trimesh]:
    """다각형(구멍 포함)을 높이만큼 세운다.

    manifold3d 가 있으면 그 압출을 쓴다 (항상 닫힌 덩어리). 없으면 trimesh(earcut) — 이때는 좌표를 1mm 격자에 맞추고
    한 직선 위의 꼭짓점을 지운다 (남아 있으면 삼각분할이 점을 빼먹어 옆면과 뚜껑이 안 맞아 watertight 가 깨진다).
    """
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.area < 1e-4 or height <= 1e-6:
            continue
        mesh = _manifold_extrude(poly, height)
        if mesh is None:
            mesh = trimesh.creation.extrude_polygon(shapely.set_precision(poly, 1e-3).simplify(0), height)
        mesh.apply_translation((0, 0, z))
        out.append(mesh)
    return out


def _manifold_extrude(poly, height: float) -> trimesh.Trimesh | None:
    try:
        import manifold3d
    except ImportError:
        return None
    rings = [np.asarray(poly.exterior.coords[:-1])] + [np.asarray(r.coords[:-1]) for r in poly.interiors]
    mesh = manifold3d.CrossSection(rings, manifold3d.FillRule.EvenOdd).extrude(height).to_mesh()
    return trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts)


def thickened(layout: Layout, min_thickness_m: float) -> Layout:
    """벽 두께를 min_thickness_m 이상으로 (3D 프린팅용)."""
    wall = WallSpec(exterior=max(layout.wall.exterior, min_thickness_m), interior=max(layout.wall.interior, min_thickness_m))
    return layout.model_copy(update={"wall": wall})


def build_parts(layout: Layout) -> list[Part]:
    parts: list[Part] = []
    z0 = 0.0
    for floor in layout.floors:
        geom = floor_geometry(layout, floor)
        cut = layout.wall.exterior / 2 + 0.05
        cuts = [(o.sill, o.head, LineString(o.segment).buffer(cut, cap_style="flat")) for o in geom.openings]
        levels = sorted({0.0, floor.height, *[c[0] for c in cuts], *[c[1] for c in cuts]})
        levels = [z for z in levels if 0 <= z <= floor.height]
        for za, zb in zip(levels, levels[1:]):
            holes = [c[2] for c in cuts if c[0] <= za + 1e-6 and c[1] >= zb - 1e-6]
            shape = geom.walls_full.difference(unary_union(holes)) if holes else geom.walls_full
            # 0.5mm 키워 모서리 한 점으로만 닿은 벽 조각을 붙인다 (그대로 두면 STL 에서 한 모서리를 4면이 공유 → watertight 깨짐)
            shape = shape.buffer(5e-4, join_style="mitre")
            parts += [Part(m, WALL_COLOR, "wall", floor.level) for m in _extrude(shape, zb - za, z0 + za)]
        slab = geom.outline.buffer(layout.wall.exterior / 2, join_style="mitre")
        parts += [Part(m, SLAB_COLOR, "slab", floor.level) for m in _extrude(slab, SLAB, z0 - SLAB)]
        z0 += floor.height
    return parts


def scene_mesh(parts: list[Part]) -> trimesh.Trimesh:
    """조각을 이어 붙인 화면·GLB 용 메시 (면마다 색)."""
    meshes = []
    for p in parts:
        m = p.mesh.copy()
        m.visual.face_colors = np.tile(np.array(p.color, dtype=np.uint8), (len(m.faces), 1))
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def solid_mesh(parts: list[Part]) -> trimesh.Trimesh:
    """조각을 하나로 합친 덩어리 (프린팅용). manifold3d 가 없으면 이어 붙이기만 한다."""
    meshes = [p.mesh for p in parts]
    try:
        merged = trimesh.boolean.union(meshes, engine="manifold")
        if merged is not None and len(merged.faces):
            return merged
    except Exception:  # noqa: BLE001 — 엔진이 없거나 실패하면 이어 붙인 메시로
        pass
    return trimesh.util.concatenate(meshes)


def print_check(layout: Layout, scale: int, min_wall_mm: float = DEFAULT_MIN_WALL_MM) -> list[str]:
    """1:scale 로 줄였을 때 벽 두께가 프린터 최소 두께보다 얇은지."""
    factor = 1000 / scale  # m → 모형 mm
    notes = []
    for name, t in (("외벽", layout.wall.exterior), ("내벽", layout.wall.interior)):
        mm = t * factor
        if mm < min_wall_mm:
            need = min_wall_mm / factor
            notes.append(f"{name} {t:g}m → 1:{scale} 에서 {mm:.2f}mm (최소 {min_wall_mm:g}mm 미만). "
                         f"프린팅용 STL 은 벽을 {need:.2f}m 로 두껍게 만들었다 — 축척을 1:{int(t * 1000 / min_wall_mm)} 이하로 키워도 된다")
    return notes


def stl_bytes(layout: Layout, scale: int = 100, min_wall_mm: float = DEFAULT_MIN_WALL_MM) -> tuple[bytes, list[str], bool]:
    """프린팅용 STL (mm, 1:scale). (파일, 안내, watertight 여부)"""
    notes = print_check(layout, scale, min_wall_mm)
    model = thickened(layout, min_wall_mm * scale / 1000) if notes else layout
    mesh = solid_mesh(build_parts(model))
    mesh.apply_scale(1000 / scale)
    mesh.merge_vertices()  # 슬라이서가 읽는 것과 같은 상태 (같은 위치의 점을 합침)에서 닫혔는지 본다
    bx = mesh.bounds[1] - mesh.bounds[0]
    notes.append(f"모형 크기 {bx[0]:.0f} × {bx[1]:.0f} × {bx[2]:.0f} mm (1:{scale})" + ("" if mesh.is_watertight else " — 닫히지 않은 면이 있어 슬라이서에서 복구가 필요할 수 있다"))
    return mesh.export(file_type="stl"), notes, bool(mesh.is_watertight)


def glb_bytes(layout: Layout) -> bytes:
    return scene_mesh(build_parts(layout)).export(file_type="glb")


def obj_bytes(layout: Layout) -> bytes:
    mesh = trimesh.util.concatenate([p.mesh for p in build_parts(layout)])
    text = trimesh.exchange.obj.export_obj(mesh, include_color=False, include_texture=False)
    return text.encode("utf-8")


def plotly_figure(layout: Layout):
    """화면 3D 뷰어 (마우스로 돌리기·확대). plotly Mesh3d 하나에 면마다 색."""
    import plotly.graph_objects as go

    parts = build_parts(layout)
    mesh = scene_mesh(parts)
    v, f = mesh.vertices, mesh.faces
    colors = [f"rgb({c[0]},{c[1]},{c[2]})" for c in mesh.visual.face_colors]
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], facecolor=colors,
                              flatshading=True, lighting={"ambient": 0.55, "diffuse": 0.8}, hoverinfo="skip"))
    fig.update_layout(scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)",
                             "camera": {"eye": {"x": 1.2, "y": -1.6, "z": 1.3}}},
                      margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=520)
    return fig


# ════════════════════════════════════════════════════════════════════
# IFC (선택 — ifcopenshell 이 설치돼 있을 때만)
# ════════════════════════════════════════════════════════════════════


def ifc_available() -> bool:
    try:
        import ifcopenshell  # noqa: F401
    except ImportError:
        return False
    return True


def ifc_bytes(layout: Layout) -> bytes | None:
    """IFC4 파일. ifcopenshell 이 없으면 None."""
    if not ifc_available():
        return None
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.project
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit
    from ifcopenshell import api

    f = api.project.create_file(version="IFC4")
    project = api.root.create_entity(f, ifc_class="IfcProject", name=layout.title or "Agent Office 설계안")
    api.unit.assign_unit(f)
    model = api.context.add_context(f, context_type="Model")
    body = api.context.add_context(f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)
    site = api.root.create_entity(f, ifc_class="IfcSite", name="대지")
    building = api.root.create_entity(f, ifc_class="IfcBuilding", name=layout.title or "건물")
    api.aggregate.assign_object(f, relating_object=project, products=[site])
    api.aggregate.assign_object(f, relating_object=site, products=[building])

    def place(product, z: float) -> None:
        m = np.eye(4)
        m[2][3] = z
        api.geometry.edit_object_placement(f, product=product, matrix=m)

    def prism(product, coords, depth: float, z: float) -> None:
        rep = api.geometry.add_slab_representation(f, context=body, depth=depth, polyline=[(x, y) for x, y in coords[:-1]])
        api.geometry.assign_representation(f, product=product, representation=rep)
        place(product, z)

    z0 = 0.0
    for floor in layout.floors:
        storey = api.root.create_entity(f, ifc_class="IfcBuildingStorey", name=floor.label)
        storey.Elevation = z0
        api.aggregate.assign_object(f, relating_object=building, products=[storey])
        geom = floor_geometry(layout, floor)
        products = []
        slab = api.root.create_entity(f, ifc_class="IfcSlab", name=f"{floor.label} 바닥")
        prism(slab, list(geom.outline.buffer(layout.wall.exterior / 2, join_style="mitre").exterior.coords), SLAB, z0 - SLAB)
        products.append(slab)
        for k, (seg, t) in enumerate(geom.wall_pieces, 1):
            wall = api.root.create_entity(f, ifc_class="IfcWall", name=f"{floor.label} 벽 {k}")
            prism(wall, list(LineString(seg).buffer(t / 2, cap_style="square").exterior.coords), floor.height, z0)
            products.append(wall)
        api.spatial.assign_container(f, relating_structure=storey, products=products)
        spaces = []
        for room in floor.rooms:
            poly = geom.rooms.get(room.id)
            if poly is None:
                continue
            space = api.root.create_entity(f, ifc_class="IfcSpace", name=room.id)
            space.LongName = room.label
            prism(space, list(poly.exterior.coords), floor.height, z0)
            spaces.append(space)
        if spaces:
            api.aggregate.assign_object(f, relating_object=storey, products=spaces)
        z0 += floor.height
    buf = io.StringIO()
    buf.write(f.to_string())
    return buf.getvalue().encode("utf-8")
