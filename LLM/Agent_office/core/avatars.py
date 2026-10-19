"""에이전트 이미지 만들기·바꾸기.

화면의 에이전트 패널에서 이미지를 올리면:
  1) 가운데를 정사각형으로 잘라 256×256 PNG 로 저장 → teams/<팀>/avatars/<에이전트id>.png
  2) '작업 중' 이미지를 따로 올리지 않았으면, 올린 그림 오른쪽 아래에 말풍선(…)을 그려 자동으로 만든다
     → teams/<팀>/avatars/<에이전트id>_working.png
  3) '기본 이미지로 되돌리기'를 누르면 위 두 파일을 지운다 (직급 공용 이미지로 돌아감)

이미지 찾는 순서는 core/config.py 의 resolve_avatar 참고 (에이전트 전용 → 팀 직급 → 공용).
"""

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.config import AVATAR_EXTS, TEAMS_DIR

SIZE = 256
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
BUBBLE_COLOR = "#FFAB00"


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
