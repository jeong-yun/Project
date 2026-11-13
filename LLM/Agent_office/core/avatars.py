"""에이전트 이미지 만들기·바꾸기.

화면의 에이전트 패널에서 이미지를 올리면:
  1) 가운데를 정사각형으로 잘라 256×256 PNG 로 저장 → teams/<팀>/avatars/<에이전트id>.png
  2) '작업 중' 이미지를 따로 올리지 않았으면, 올린 그림 오른쪽 아래에 말풍선(…)을 그려 자동으로 만든다
     → teams/<팀>/avatars/<에이전트id>_working.png
  3) '기본 이미지로 되돌리기'를 누르면 위 두 파일을 지운다 (직급 공용 이미지로 돌아감)
  4) 올린 그림은 🏢 부서 사무실 캐릭터의 얼굴로도 쓴다 (office_faces — 작게 줄여 화면 스냅숏에 싣는다)

이미지 찾는 순서는 core/config.py 의 resolve_avatar 참고 (에이전트 전용 → 팀 직급 → 공용).
"""

import base64
import io
import os
import time
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.config import AVATAR_EXTS, TEAMS_DIR
from core.memo import RACY_SEC, SigCache

SIZE = 256
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
BUBBLE_COLOR = "#FFAB00"
FACE_SIZE = 48  # 🏢 부서 사무실 캐릭터 얼굴 한 변 (px) — 화면에서는 20~35px 로 보인다
HEAD_FOLDER = "_dept_heads"  # 👔 부서장은 팀 밖이라 얼굴 그림을 teams/_dept_heads/avatars/<부서 id>.png 에 둔다 ('_' 폴더는 팀 목록에서 빠진다)
FACE_COLORS = 64  # 색 수를 줄여 스냅숏을 가볍게 (얼굴 하나 1~3KB — 사무실은 1초마다 스냅숏을 다시 보낸다)


class AvatarError(ValueError):
    """잘못된 이미지 (형식·크기) — 화면에 그대로 보여 준다."""


def make_avatar(label: str, color: str, font: ImageFont.ImageFont) -> Image.Image:
    """기본(placeholder) 아바타: 색 원 + 글자."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, SIZE - 8, SIZE - 8), fill=color)
    draw.text((SIZE / 2, SIZE / 2), label, fill="white", font=font, anchor="mm")
    return img


def make_working(base: Image.Image, color: str = BUBBLE_COLOR) -> Image.Image:
    """평소 이미지 오른쪽 아래에 '말하는 중' 말풍선(…)을 붙인다."""
    img = base.convert("RGBA").resize((SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((130, 160, 250, 236), radius=30, fill="white", outline=color, width=6)
    draw.polygon([(150, 226), (138, 252), (176, 232)], fill="white", outline=color)  # 꼬리
    draw.rounded_rectangle((146, 222, 180, 232), radius=4, fill="white")  # 꼬리 이음새 덮기
    for cx in (162, 190, 218):
        draw.ellipse((cx - 9, 189, cx + 9, 207), fill=color)
    return img


def to_square(data: bytes) -> Image.Image:
    """올린 파일 → 가운데 정사각형으로 잘라 SIZE×SIZE RGBA 이미지."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise AvatarError(f"이미지가 너무 큽니다 ({len(data) / 1024 / 1024:.1f}MB). 5MB 이하로 올려 주세요.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001 — Pillow 가 읽지 못하는 파일
        raise AvatarError(f"이미지 파일을 읽을 수 없습니다: {type(e).__name__}") from e
    img = img.convert("RGBA")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)


def avatar_dir(team_id: str) -> Path:
    return TEAMS_DIR / team_id / "avatars"


def _remove_variants(team_id: str, stem: str) -> None:
    """같은 이름의 다른 확장자 파일을 지운다 (png/jpg 가 같이 있으면 어느 것이 쓰일지 헷갈리므로)."""
    for ext in AVATAR_EXTS:
        path = avatar_dir(team_id) / f"{stem}.{ext}"
        if path.exists():
            path.unlink()


def save_avatar(team_id: str, node_id: str, data: bytes, working_data: bytes | None = None) -> tuple[Path, Path]:
    """에이전트 전용 이미지를 저장한다. Returns: (평소 이미지 경로, 작업 중 이미지 경로)"""
    base = to_square(data)
    working = to_square(working_data) if working_data else make_working(base)
    folder = avatar_dir(team_id)
    folder.mkdir(parents=True, exist_ok=True)
    _remove_variants(team_id, node_id)
    _remove_variants(team_id, f"{node_id}_working")
    base_path, working_path = folder / f"{node_id}.png", folder / f"{node_id}_working.png"
    base.save(base_path)
    working.save(working_path)
    return base_path, working_path


def has_custom_avatar(team_id: str, node_id: str) -> bool:
    return any((avatar_dir(team_id) / f"{node_id}.{ext}").exists() for ext in AVATAR_EXTS)


def reset_avatar(team_id: str, node_id: str) -> None:
    """에이전트 전용 이미지를 지워 기본(직급 공용) 이미지로 되돌린다."""
    _remove_variants(team_id, node_id)
    _remove_variants(team_id, f"{node_id}_working")


# ─── 🏢 부서 사무실 캐릭터 얼굴 ─────────────────────────────────────────
_face_cache = SigCache()


def make_face(path: Path) -> str:
    """이미지 파일 → FACE_SIZE 정사각형 PNG 의 data URL. 읽지 못하는 파일이면 '' (사무실은 도트 머리로 그린다)."""
    try:
        face = to_square(path.read_bytes()).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS).quantize(FACE_COLORS)
    except (OSError, AvatarError):  # 손으로 넣은 깨진 파일이 사무실 화면을 막지 않게
        return ""
    buf = io.BytesIO()
    face.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def office_faces(team_id: str, node_ids: Iterable[str]) -> dict[str, str]:
    """부서 사무실 캐릭터의 얼굴 — {에이전트 id: 전용 이미지(화면에서 올린 그림)를 작게 줄인 data URL}. 그림이 없는 사람은 빠진다.

    직급 이미지(팀·공용)는 얼굴로 쓰지 않는다. 화면을 다시 그릴 때마다 불리므로:
      - 사람마다 파일을 묻지 않고 폴더를 한 번만 훑는다 (이 PC 에서 파일 하나 묻기 ≈ 0.5ms, 폴더 훑기도 ≈ 0.6ms)
      - 파일의 (수정 시각, 크기)가 그대로면 전에 줄여 둔 것을 쓴다 (방금 바뀐 파일은 core/memo.py 처럼 캐시하지 않는다)"""
    try:
        with os.scandir(avatar_dir(team_id)) as entries:
            found = {e.name.lower(): (e.path, e.stat()) for e in entries}
    except OSError:  # 폴더 없음 = 올린 그림 없음
        return {}
    now, faces = time.time_ns(), {}
    for node_id in node_ids:
        hit = next((found[name] for ext in AVATAR_EXTS if (name := f"{node_id}.{ext}".lower()) in found), None)
        if hit is None:
            continue
        path, st = hit
        sig = None if now - st.st_mtime_ns < RACY_SEC * 1e9 else (st.st_mtime_ns, st.st_size)
        face = _face_cache.get(path, sig, lambda: make_face(Path(path)))
        if face:
            faces[node_id] = face
    return faces
