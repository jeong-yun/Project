"""피드백(교훈·결정 요약)을 knowledge/ 폴더와 맞춘다 — GitHub 에 올리기 전에 한 번 돌리거나, 다른 PC 에서 받은 뒤 가져올 때.

실행 (Agent_office 폴더에서):
    python scripts/sync_feedback.py              # DB → knowledge/ 내보내기 (기본)
    python scripts/sync_feedback.py import       # knowledge/ → DB 가져오기 (없는 교훈만) 후 다시 내보내기
    python scripts/sync_feedback.py --data D:/x/data --dir knowledge

DB 위치는 AGENT_OFFICE_DATA_DIR(없으면 data/), 내보낼 폴더는 AGENT_OFFICE_KNOWLEDGE_DIR(없으면 knowledge/).
질문·답변 원문, 사용량·API 키 정보는 내보내지 않는다 (core/feedback_sync.py 참고).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import feedback_sync  # noqa: E402
from core.storage import Storage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="교훈·결정 요약 ↔ knowledge/ 동기화")
    parser.add_argument("action", nargs="?", choices=["export", "import"], default="export")
    parser.add_argument("--data", default=os.getenv("AGENT_OFFICE_DATA_DIR") or str(ROOT / "data"), help="runs.db 가 있는 폴더")
    parser.add_argument("--dir", default=os.getenv(feedback_sync.ENV) or str(ROOT / "knowledge"), help="내보낼 폴더")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.environ.pop(feedback_sync.ENV, None)  # Storage 생성 때 자동 동기화가 다른 폴더를 건드리지 않게 — 아래에서 직접 한다
    storage = Storage(Path(args.data) / "runs.db")
    folder = Path(args.dir)
    imported = feedback_sync.import_lessons(storage, folder) if args.action == "import" else 0
    result = feedback_sync.export(storage, folder)
    print(f"교훈 {result['lessons']}개 · 결정 {result['decisions']}건 → {folder}"
          + (f" (가져온 교훈 {imported}개)" if args.action == "import" else ""))
    print("바뀐 파일: " + (", ".join(result["changed"]) if result["changed"] else "없음 (이미 최신)"))


if __name__ == "__main__":
    main()
