"""데이터 폴더의 JSON 파일 하나를 안전하게 읽고 쓴다 (부서장·팀원 모델, 모델 능력·목록, 뉴스, 시세, 추후 목록이 함께 쓴다).

  읽기 : 없거나 깨졌거나 모양(dict/list)이 기본값과 다르면 기본값 — 화면이 멈추지 않게
  쓰기 : 임시 파일에 다 쓴 뒤 os.replace 로 바꿔치기 — 쓰는 도중 꺼져도 예전 파일이 깨지지 않는다
  잠금 : 같은 경로는 한 프로세스(Streamlit) 안에서 같은 잠금. 읽고 고치고 쓰는 동안 `with f.lock:` 로 감싼다
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def write_atomic(path: Path, text: str) -> None:
    """임시 파일(스레드마다 다른 이름)에 쓴 뒤 바꿔치기."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class JsonFile:
    def __init__(self, path: Path | str, default=None, indent: int = 2):
        self.path = Path(path)
        self.default = default
        self.indent = indent
        with _locks_guard:
            self.lock = _locks.setdefault(str(self.path.absolute()), threading.RLock())

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return copy.deepcopy(self.default)
        if self.default is not None and not isinstance(data, type(self.default)):
            return copy.deepcopy(self.default)
        return data

    def save(self, data) -> None:
        with self.lock:
            write_atomic(self.path, json.dumps(data, ensure_ascii=False, indent=self.indent))
