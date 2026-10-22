"""파일이 바뀌었는지(수정 시각·크기)로 판단하는 작은 캐시 — 화면을 다시 그릴 때마다 같은 파일을 다시 읽지 않게.

Streamlit 은 클릭할 때마다 app.py 전체를 다시 실행한다. 그때마다 YAML 설정·SQLite 조회·아바타 폴더를 새로 읽으면
화면 전환이 느려지므로, 파일의 (수정 시각, 크기) '지문'이 같으면 이전 결과를 그대로 쓴다.

  - 파일이 바뀌면 지문이 달라져 자동으로 다시 읽는다 (앱을 다시 켤 필요 없음)
  - 방금(RACY_SEC 안에) 바뀐 파일은 캐시하지 않는다 — 같은 시각 안에 두 번 바뀌어 지문이 같아지는 경우를 피한다
    (git 의 'racy' 처리와 같은 방식). 그래서 작업 중에 계속 쓰이는 DB 는 사실상 매번 새로 읽는다.
"""

import os
import threading
import time
from collections.abc import Callable
from typing import Any

RACY_SEC = 2.0


def file_sig(*paths) -> tuple | None:
    """각 경로의 (수정 시각 ns, 크기) — 없는 경로는 None 자리. 최근 RACY_SEC 안에 바뀐 것이 있으면 None (캐시 쓰지 말 것)."""
    now = time.time_ns()
    sig = []
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            sig.append(None)
            continue
        if now - st.st_mtime_ns < RACY_SEC * 1e9:
            return None
        sig.append((st.st_mtime_ns, st.st_size))
    return tuple(sig)


class SigCache:
    """{키: (지문, 값)} — 지문이 같을 때만 저장한 값을 돌려준다. 여러 스레드(백그라운드 업무)에서 불러도 안전."""

    def __init__(self, maxsize: int = 256):
        self._data: dict[Any, tuple[tuple, Any]] = {}
        self._lock = threading.Lock()
        self.maxsize = maxsize

    def get(self, key, sig: tuple | None, compute: Callable[[], Any]) -> Any:
        if sig is None:  # 방금 바뀐 파일 — 캐시하지 않고 새로 계산
            return compute()
        with self._lock:
            hit = self._data.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        value = compute()  # 계산 중에 파일이 바뀌어도 다음 호출의 지문이 달라 다시 계산된다
        with self._lock:
            if len(self._data) >= self.maxsize:
                self._data.clear()
            self._data[key] = (sig, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
