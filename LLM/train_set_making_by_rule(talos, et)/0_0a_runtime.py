"""[0.0a 실행 관리] 실행 잠금·중지 요청·상태 파일.  (UI·1.0 수집·3.x 생성 공용 보조 모듈, 직접 실행하지 않음)

수집(에이전트 포함)과 학습데이터 생성을 동시에, 그리고 UI·명령줄 어디서 실행해도 서로 부딪히지 않게 한다.
모든 파일은 .run/ 폴더에 둔다.

  .run/<이름>.lock   ← 실행 중 표시 (PID·시작 시각·인자). 그 PID 가 끝났으면(비정상 종료) 다음 실행이 지우고 새로 잡는다
                       generation = 3_0 학습데이터 생성, collect = 1_0 수집 1회, agent = 1_0 --agent 수집 에이전트
  .run/<이름>.stop   ← 중지 요청 (UI 의 '중지' 버튼). 실행 중인 쪽이 지금 하던 일을 마치고 멈춘다
  .run/<이름>.json   ← 상태 (예: collect.json = 마지막 수집 시각·결과, agent.json = 에이전트 다음 실행 시각)
  .run/<이름>.out    ← UI 에서 시작한 작업의 화면 출력
"""

import json
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / ".run"


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _zombie(stat_text):
    """/proc/<pid>/stat 내용 → 좀비(끝났지만 부모가 아직 거두지 않음)인지. 이름에 ')' 가 있을 수 있어 마지막 ')' 뒤를 본다."""
    return stat_text.rsplit(")", 1)[-1].split()[:1] == ["Z"]


def pid_alive(pid):
    """프로세스가 살아 있는지 (Windows: OpenProcess + 종료 코드, 그 외: kill 0 + Linux 는 좀비 빼기).
    2026-11-25~: 화면이 띄운 작업(spawn 은 기다리지 않음)이 끝나도 화면이 거두기 전까지 좀비로 남아 kill 0 이 성공 →
    '학습 로그에 반영' 이 1시간 동안 '도는 중' 으로 보이고 다음 날짜를 막던 문제. /proc 을 못 읽으면 (macOS 등) kill 0 결과대로."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return not _zombie(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return True


def read_lock(name):
    """잠금 정보 dict. 잠금이 없거나 잡은 프로세스가 이미 끝났으면 None."""
    try:
        info = json.loads((RUN_DIR / f"{name}.lock").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if pid_alive(info.get("pid")) else None


def acquire_lock(name, **info):
    """잠금을 잡는다. 잡으면 None, 다른 (살아 있는) 프로세스가 잡고 있으면 그 잠금 정보를 돌려준다."""
    RUN_DIR.mkdir(exist_ok=True)
    path = RUN_DIR / f"{name}.lock"
    data = dict(info, pid=os.getpid(), started_at=now_iso())
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = read_lock(name)
            if holder:
                return holder
            try:                                   # 비정상 종료로 남은 잠금 → 지우고 다시
                path.unlink()
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return None
    return read_lock(name) or {"pid": "?"}


def release_lock(name):
    """내가 잡은 잠금이면 푼다."""
    path = RUN_DIR / f"{name}.lock"
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def update_lock(name, **info):
    """내가 잡은 잠금에 정보(예: 지금 단계)를 덧붙인다 (UI 표시용)."""
    path = RUN_DIR / f"{name}.lock"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("pid") == os.getpid():
            data.update(info)
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    except (OSError, ValueError):
        pass


def request_stop(name):
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / f"{name}.stop").write_text(now_iso(), encoding="utf-8")


def stop_requested(name):
    return (RUN_DIR / f"{name}.stop").exists()


def clear_stop(name):
    try:
        (RUN_DIR / f"{name}.stop").unlink()
    except OSError:
        pass


def read_status(name):
    try:
        return json.loads((RUN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_status(name, **info):
    """상태 파일에 값을 덧붙여 저장 (원자적 교체)."""
    RUN_DIR.mkdir(exist_ok=True)
    data = read_status(name)
    data.update(info)
    path = RUN_DIR / f"{name}.json"
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in range(50):     # Windows: UI 등이 그 순간 상태 파일을 읽고 있으면 교체가 막힌다(WinError 5) → 잠깐 뒤 다시 (최대 약 5초)
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if k == 49:
                raise
            time.sleep(0.1)
    return data
