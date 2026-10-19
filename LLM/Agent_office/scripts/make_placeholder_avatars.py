"""직급별 기본(placeholder) 아바타 이미지를 생성한다.

사용법 (26-Agent-Office 폴더에서):
    python scripts/make_placeholder_avatars.py           # 없는 파일만 생성
    python scripts/make_placeholder_avatars.py --force   # 기존 파일 덮어쓰기

직급마다 두 장을 만든다.
    assets/avatars/<직급id>.png          — 평소 이미지
    assets/avatars/<직급id>_working.png  — 답변·검토 중일 때 이미지 (말풍선 "…" 배지)
원하는 이미지로 바꾸려면 같은 이름의 파일로 교체하면 된다 (png/jpg/jpeg/webp).
"""

import argparse
import sys
from pathlib import Path

from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.avatars import SIZE, make_avatar, make_working  # noqa: E402,F401
from core.config import BASE_DIR, load_ranks  # noqa: E402

COLORS = ["#4C9AFF", "#36B37E", "#FFAB00", "#FF5630", "#6554C0"]  # 직급 순서대로
FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]


def load_korean_font(size: int) -> ImageFont.FreeTypeFont | None:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 이미지 덮어쓰기")
    args = parser.parse_args()

    korean_font = load_korean_font(110)
    for i, rank in enumerate(load_ranks()):
        out = BASE_DIR / (rank.avatar or f"assets/avatars/{rank.id}.png")
        working_out = out.with_name(f"{out.stem}_working.png")
        # 한글 폰트가 없으면 직급 id 첫 글자(영문)로 대체
        if korean_font:
            label, font = rank.name[0], korean_font
        else:
            label, font = rank.id[0].upper(), ImageFont.load_default(size=110)
        color = COLORS[i % len(COLORS)]
        base = make_avatar(label, color, font)
        out.parent.mkdir(parents=True, exist_ok=True)
        for path, image in ((out, base), (working_out, make_working(base, color))):
            if path.exists() and not args.force:
                print(f"skip  {path.relative_to(BASE_DIR)} (이미 존재)")
                continue
            image.save(path)
            print(f"saved {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
