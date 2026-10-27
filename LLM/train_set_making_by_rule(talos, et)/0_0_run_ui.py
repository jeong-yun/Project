"""[0.0 화면] 브라우저에서 수집(에이전트)·학습데이터 생성을 실행하고, 진행률·룰별 상태·만든 학습데이터를 보는 로컬 화면.

    python 0_0_run_ui.py              → http://127.0.0.1:8765 이 브라우저로 열린다 (이 PC 에서만 접속 가능)
    python 0_0_run_ui.py --port 9000 --no-browser

[페이지]
  1. 진행 현황 (첫 화면)  생성 대상 룰 중 완료 %, 현재 룰 중 학습데이터 보유 %, 상태 플래그별 수, 어디까지(배치),
                          출처·공격 유형별 %, 남은 요청·예상 시간·토큰 (3_5_progress.py 계산)
  2. 생성·수집            학습데이터 생성(건수 확인·파일럿·시작·중지·강제 중지, 3_0_run_generation.py)
                          수집 에이전트 시작·중지(매일 정해진 시각), 지금 1회 수집 (1_0_run_daily.py)
  3. 룰별 상태            상태 플래그·출처·검색으로 찾기, CSV 내려받기 (07_llm_output/_progress/rule_status.csv)
  4. 신규 학습 데이터     08_dataset 의 IPS·WAF 신규 학습데이터를 날짜·장비별 표(데이터 프레임)로 보기, CSV 내려받기
                          - index, payload, method, label, label 사유, 검증 결과, label 정합성, 탐지 피처
                            label 사유 = 어떤 룰의 어떤 조건을 만족/불일치했는지 + 생성 메모
                            검증 결과 = 생성 때 검증(코드 검사·검토 모델·LLM/증강) + 지금 현재 룰로 다시 확인한 결과
                                        (normal 은 다른 공격 룰·공격 흔적이 없는지도 확인)
                          - 로그 형식 파일은 여기서 받지 않는다 → 6. 에서 버전을 확정할 때 기존 + 신규를 합쳐 만든다
                          - 장비 FW·웹: IPS 줄을 방화벽·웹(nginx) 로그 형식으로 바꿔 보기·CSV·파일로 저장
                            (1_4a, 08_dataset/<날짜>/fw·web_<yymmdd>.log + _label.jsonl)
  5. 피처                 4.1 피처 설명과 켜진 줄 수
  6. 기존 학습데이터와 비교  신규 학습데이터를 기준 버전(기존 학습데이터)과 비교해 등급(제거·A·B·C)을 매기고(5_5),
                          선별해 버전(26 하반기 등)에 담고 확정하면 학습 파일·로그를 내보낸다 (5_7, 실행은 5_0_run_supervised.py)

[동작]
  - 작업은 별도 프로세스로 실행한다 → 화면(이 프로그램)을 닫아도 작업은 계속된다. 다시 열면 상태를 이어서 보여 준다.
  - 실행 중 여부는 .run/*.lock 으로 판단한다 → 명령줄에서 시작한 작업도 화면에 보이고 중지할 수 있다.
  - 수집 에이전트가 도는 중에도 학습데이터 생성을 시작할 수 있다 (1_0_run_daily.py·3_0_run_generation.py 머리 설명 참고).
  - 학습데이터 페이지는 로그 줄 위치를 색인해 두고 필요한 줄만 읽는다 (수십만 줄이어도 페이지 이동이 빠름).
  - 보안: 127.0.0.1 에만 열고, 다른 사이트가 이 화면에 요청을 보내지 못하게 Host·전용 헤더를 확인한다.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from array import array
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RT = _load("runtime", "0_0a_runtime.py")
JOBS = {   # 이름: (스크립트, 잠금 이름)
    "generation": ("3_0_run_generation.py", "generation"),
    "collect": ("1_0_run_daily.py", "collect"),
    "agent": ("1_0_run_daily.py", "agent"),
    "dataset": ("1_4_build_dataset.py", "dataset"),    # 신규 학습 데이터 화면의 '학습 로그에 반영' (수정·삭제 적용)
    "supervised": ("5_0_run_supervised.py", "supervised"),  # 6. 기존 학습데이터와 비교 (비교·버전 확정)
    "redo": ("3_3_review.py", "generation"),   # 2. 검토에서 뺀 샘플 '다시 만들기' (3_3 --redo-safety, 생성과 같은 잠금)
}
START_WAIT = {"redo": 600}      # 다시 만들기는 다시 검토할 요청을 찾은 뒤(1분쯤) 잠금을 잡는다 → 그동안도 '시작하는 중'
PROGRESS_FILE = BASE / "07_llm_output" / "_progress" / "progress.json"
RULES_FILE = BASE / "07_llm_output" / "_progress" / "rule_status.jsonl"
CSV_FILE = BASE / "07_llm_output" / "_progress" / "rule_status.csv"
DATASET_DIR = BASE / "08_dataset"
MODEL_RE = re.compile(r"^(none|mock|[a-z0-9_-]+:[\w.\-]+)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_lock = threading.Lock()


# ── 작업 실행·상태 ────────────────────────────────────────────────────────

def _ui_record(job):
    return RT.read_status(f"ui_{job}")          # 화면에서 시작한 마지막 작업: pid, 출력 파일, 시작 시각


def job_state(job):
    """잠금(실행 중) + 화면에서 막 시작해 아직 잠금을 잡기 전인 경우까지."""
    lock = RT.read_lock(JOBS[job][1])
    rec = _ui_record(job)
    starting = (lock is None and rec.get("pid") and RT.pid_alive(rec["pid"])
                and time.time() - rec.get("t", 0) < START_WAIT.get(job, 60))
    return {"running": bool(lock) or bool(starting), "starting": bool(starting), "lock": lock or {},
            "stop_requested": RT.stop_requested(JOBS[job][1]), "ui": rec}


def spawn(job, args):
    """작업을 별도 프로세스로 시작 (화면을 닫아도 계속). 출력은 .run/<작업>-<시각>.out"""
    RT.RUN_DIR.mkdir(exist_ok=True)
    for old in sorted(RT.RUN_DIR.glob(f"{job}-*.out"))[:-5]:      # 출력 파일은 최근 6개만 남김
        try:
            old.unlink()
        except OSError:
            pass
    out_path = RT.RUN_DIR / f"{job}-{time.strftime('%Y%m%d-%H%M%S')}.out"
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, str(BASE / JOBS[job][0])] + args
    kw = dict(cwd=BASE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    with open(out_path, "w", encoding="utf-8") as out:
        try:    # 화면을 띄운 터미널(VS Code 등)의 잡(job)에서 떼어 냄 → 그 터미널을 닫아도 작업은 계속
            p = subprocess.Popen(cmd, stdout=out, creationflags=flags | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0),
                                 **kw)
        except OSError:                 # 떼어 낼 수 없는 잡이면 지금처럼
            p = subprocess.Popen(cmd, stdout=out, creationflags=flags, **kw)
    RT.write_status(f"ui_{job}", pid=p.pid, out=out_path.name, args=args, t=time.time(), started_at=RT.now_iso())
    return p.pid


def force_kill(pid):
    if not pid:
        return False
    if os.name == "nt":
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return r.returncode == 0
    try:
        os.kill(int(pid), 9)
        return True
    except OSError:
        return False


def tail(path, lines=150):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def log_text(job, lines=150):
    """화면에서 시작한 작업이면 그 출력, 아니면(명령줄 실행) 로그 파일.
    생성은 '다시 만들기'(같은 생성 잠금, .run/redo-*.out - 명령줄에서 띄운 것 포함)와 견주어 더 최근 출력."""
    outs = [RT.RUN_DIR / r["out"] for r in map(_ui_record, [job, "redo"] if job == "generation" else [job]) if r.get("out")]
    if job == "generation":
        outs += sorted(RT.RUN_DIR.glob("redo-*.out"))[-1:]
    stamp = {}
    for p in outs:
        try:
            stamp[p] = p.stat().st_mtime
        except OSError:
            pass
    if stamp:
        return tail(max(stamp, key=stamp.get), lines)
    fallback = {"generation": "3_0_run_generation.log", "collect": "run_daily.log", "agent": "run_daily.log"}.get(job)
    return tail(BASE / "logs" / fallback, lines) if fallback else ""


# ── 지금 실행 중인 생성의 진행률 (로그만 읽음) ──────────────────────────

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} \[(\w+)\]\s?(.*)$")
STEPS = [("3.1a", "기존 데이터 확인"), ("3.1", "요청 준비"), ("3.2", "생성"), ("3.3", "검토"), ("3.4", "증강"),
         ("1.4", "학습 로그 변환"), ("3.5", "진행 현황 갱신")]
# 준비·마무리 단계 시간 - 3_0 로그에 지난 실행 기록이 없을 때만 (3.4 는 요청 1개당 초, FIRST = 새 입력 날짜의 첫 실행)
STEP_DEFAULT = {"3.1a": 2.0, "3.1": 60.0, "3.4": 0.6, "1.4": 30.0, "3.5": 10.0}
FIRST_DEFAULT = {"3.1a": 900.0, "3.1": 340.0}
_tails, _tail_lock = {}, threading.Lock()
_step_cache = {"key": None, "val": None}


def _log_since(name, since):
    """logs/<name> 에서 since('YYYY-MM-DD HH:MM:SS') 이후 줄 → [(시각, 수준, 내용)].
    읽은 위치를 기억해 두고 새로 쓰인 부분만 읽는다 (며칠 걸리는 실행에서도 가볍게)."""
    path = BASE / "logs" / name
    with _tail_lock:
        try:
            size = path.stat().st_size
        except OSError:
            return []
        t = _tails.get(name)
        if not t or t["since"] != since or size < t["pos"]:
            t = _tails[name] = {"since": since, "pos": max(0, size - 20_000_000), "rows": []}
        if size > t["pos"]:
            with open(path, "rb") as f:
                f.seek(t["pos"])
                data = f.read(size - t["pos"])
            cut = data.rfind(b"\n") + 1                 # 아직 쓰는 중인 마지막 줄은 다음에 읽음
            for line in data[:cut].decode("utf-8", errors="replace").splitlines():
                m = TS_RE.match(line)
                if m and m.group(1) >= since:
                    t["rows"].append(m.groups())
            t["pos"] += cut
        return list(t["rows"])


def _secs(a, b):
    fmt = "%Y-%m-%d %H:%M:%S"
    return time.mktime(time.strptime(b, fmt)) - time.mktime(time.strptime(a, fmt))


def _tokens(msg):
    i = re.search(r"'input_tokens': (\d+)", msg)
    o = re.search(r"'output_tokens': (\d+)", msg)
    return (int(i.group(1)) if i else 0), (int(o.group(1)) if o else 0)


def _last_run_start():
    """잠금이 없을 때: 3_0 로그에서 마지막 실행의 시작 시각 ('입력 날짜 …' 줄)."""
    last = None
    for line in tail(BASE / "logs" / "3_0_run_generation.log", 400).splitlines():
        m = TS_RE.match(line)
        if m and m.group(3).startswith("입력 날짜"):
            last = m.group(1)
    return last


def step_seconds():
    """준비(3.1a·3.1)·마무리(3.4·1.4·3.5) 단계가 걸리는 시간 = 3_0 로그의 지난 실행(정상 종료, dry-run 제외) 최근 3번 중앙값.
    3.4(증강)는 요청 수에 비례해 요청 1개당 초. first = 새 입력 날짜의 첫 실행: 3.1a 가 기존 로그를 새로 대조하고
    3.1 이 요청 파일을 만들어 20분 넘게 걸린다 (그 뒤 실행은 결과를 다시 써서 1~2분). 로그가 바뀌었을 때만 다시 읽는다."""
    path = BASE / "logs" / "3_0_run_generation.log"
    try:
        key = (path.stat().st_size, path.stat().st_mtime)
    except OSError:
        key = None
    if _step_cache["val"] is None or _step_cache["key"] != key:
        runs = []
        for line in tail(path, 100_000).splitlines():
            m = TS_RE.match(line)
            if not m:
                continue
            ts, msg = m.group(1), m.group(3)
            if msg.startswith("입력 날짜"):
                n = re.search(r"요청 (\d+)개", msg)              # '남은 전부' 는 수를 모름
                runs.append({"n": int(n.group(1)) if n else 0, "dry": "(dry-run)" in msg, "start": {}, "sec": {}})
            elif runs and (s := re.match(r"[▶■] \[(\S+)\] \S+ (시작|종료)(?: \(코드 (-?\d+))?", msg)):
                k, what, code = s.groups()
                if what == "시작":
                    runs[-1]["start"][k] = ts
                elif code == "0" and k in runs[-1]["start"]:
                    runs[-1]["sec"][k] = _secs(runs[-1]["start"][k], ts)
        normal, first = {k: [] for k in STEP_DEFAULT}, {k: [] for k in FIRST_DEFAULT}
        for r in (r for r in runs if not r["dry"]):
            sec = r["sec"]
            for k in ("3.1a", "3.1"):                   # 3.1a 가 1분 넘게 걸린 실행 = 기존 로그를 새로 대조한 첫 실행
                if k in sec:
                    (first if sec.get("3.1a", 0) > 60 else normal)[k].append(sec[k])
            if "3.4" in sec and r["n"]:
                normal["3.4"].append(sec["3.4"] / r["n"])
            for k in ("1.4", "3.5"):
                if k in sec:
                    normal[k].append(sec[k])
        med = lambda xs, d: sorted(xs[-3:])[len(xs[-3:]) // 2] if xs else d
        _step_cache.update(key=key, val={"normal": {k: med(normal[k], d) for k, d in STEP_DEFAULT.items()},
                                         "first": {k: med(first[k], d) for k, d in FIRST_DEFAULT.items()}})
    return _step_cache["val"]


def step_summary():
    """요청 수별 예상 시간(화면)에 더하는 준비·마무리 시간. first_dates = 아직 생성을 돌리지 않은 입력 날짜
    (다음 실행이 기존 로그 대조·요청 파일 만들기를 새로 해서 준비가 오래 걸림)."""
    ss = step_seconds()
    n, f = ss["normal"], ss["first"]
    has_logs = any((BASE / "06_기존 데이터").glob("*.log"))     # 없으면 3.1a 는 대조를 건너뜀
    out = BASE / "07_llm_output"
    first = [d for d in input_dates() if not (out / d / "_1_requests").is_dir()
             or (has_logs and not (out / d / "_0_existing" / "summary.json").is_file())]
    return {"prep": n["3.1a"] + n["3.1"], "prep_first": f["3.1a"] + f["3.1"], "augment_per_request": n["3.4"],
            "post": n["1.4"] + n["3.5"], "first_dates": first}


SETTINGS_FILE = BASE / "docs" / "코드참조" / "LLM_설정.md"
_settings_cache = {"mtime": None, "run": {}}


def run_settings():
    """LLM_설정.md '## 실행' 표 {항목: 값} (바뀌었을 때만 다시 읽음)."""
    try:
        mt = SETTINGS_FILE.stat().st_mtime
    except OSError:
        return {}
    if _settings_cache["mtime"] != mt:
        run, section = {}, None
        for line in SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("## "):
                section = s[3:].strip()
            elif section == "실행" and s.startswith("|"):
                cells = [c.strip() for c in s.strip("|").split("|")]
                if len(cells) >= 2:
                    run[cells[0]] = cells[1]
        _settings_cache.update(mtime=mt, run=run)
    return _settings_cache["run"]


def parallel():
    """(생성 동시 요청 수, 검토 동시 요청 수, 생성·검토 동시 진행) - 3_0a_common.parallel_settings 와 같은 기본값."""
    run = run_settings()

    def num(key, default, top):
        try:
            return max(1, min(top, int(run.get(key, default))))
        except ValueError:
            return default
    return (num("생성 동시 요청 수", 10, 16), num("검토 동시 요청 수", 12, 32),
            run.get("생성·검토 동시 진행", "예").strip().lower() in ("예", "yes", "y", "true", "1", "on"))


def _llm_counts(name, since, until, ok_re):
    """3_2·3_3 로그 → 이번 실행(since~until)의 대상 수, 성공·실패 수, 최근 경고(재시도 등), 토큰."""
    s = {"header": False, "target": None, "follow": False, "wait": 0, "cap": 0, "workers": None, "now": None, "auto": False, "redo": False, "chk_ok": 0, "chk_fail": 0, "fixed": 0, "dropped": 0, "ok": 0,
         "fail": 0, "pending": 0, "retry": None, "start": None, "last": None, "tin": 0, "tout": 0, "stopped": False,
         "done_ts": [], "by_name": {}}
    for ts, lv, msg in _log_since(name, since):
        if until and ts > until:
            break
        if re.match(r"\[3\.[23]\] (?:생성|검토) ", msg) and " | " in msg:      # 단계 시작 줄 (실행마다 한 번)
            if s["header"]:
                break                                   # 두 번째부터는 다음 실행
            s["header"], s["start"], s["last"] = True, ts, ts
            w = re.search(r"동시 (?:최대 )?(\d+)개씩", msg)     # 2026-10-20 전 실행은 하나씩
            s["workers"] = int(w.group(1)) if w else 1
            s["auto"] = "동시 최대" in msg                     # 속도 제한에 맞춰 자동 조절하는 실행 (4개부터 시작)
            s["redo"] = "다시 검토" in msg                    # 3_3 --redo-safety: 대체 검토 모델만 부름
            if (m := re.search(r"이번 (\d+)개", msg)):
                s["target"] = int(m.group(1))
            if (m := re.search(r"생성과 함께 진행: 대기 (\d+)개", msg)):   # 3.3 이 생성(3.2)과 함께 도는 경우
                s["follow"], s["wait"] = True, int(m.group(1))
                c = re.search(r"최대 (\d+)개", msg)
                s["cap"] = int(c.group(1)) if c else 0
        elif (m := re.search(r"(?:생성|검토)( 대체)? 동시 요청 \d+ → (\d+)", msg)):   # 속도 제한에 맞춘 자동 조절
            if bool(m.group(1)) == s["redo"]:           # 보통 실행에서 대체 검토(일부 룰만)의 동시 수는 따로 셈
                s["now"] = int(m.group(2))
        elif (m := re.search(r"\[3\.2\] 검토 대기 중인 생성 결과 (\d+)개", msg)):
            s["pending"] = int(m.group(1))
        elif re.match(ok_re, msg):
            s["ok"] += 1
            s["done_ts"].append(ts)
            s["by_name"][msg.strip().split(":", 1)[0]] = ts
            s["last"] = ts
            if (m := re.search(r"코드 검사 ok (\d+) / fail (\d+)", msg)):      # 생성: 샘플 코드 검사 결과
                s["chk_ok"] += int(m.group(1))
                s["chk_fail"] += int(m.group(2))
            if (m := re.search(r"수정 (\d+)\), 제외 (\d+)개", msg)):            # 검토: 고친 샘플·뺀 샘플
                s["fixed"] += int(m.group(1))
                s["dropped"] += int(m.group(2))
            ti, to = _tokens(msg)
            s["tin"] += ti
            s["tout"] += to
        elif lv == "ERROR" and re.match(r"\s*\S+ 실패", msg):
            s["fail"] += 1
            s["done_ts"].append(ts)
            s["last"] = ts
        elif lv == "WARNING":                           # 응답 해석 실패·반복 루프로 잘림·LLM 호출 실패·속도 제한(429)
            s["retry"] = (ts, msg.strip()[:110])
        elif "중지 요청" in msg:
            s["stopped"] = True
    return s


def run_progress():
    """지금(또는 마지막) 학습데이터 생성 1회의 진행률. 3_0·3_1a·3_2·3_3 로그만 읽는다 (실행 중인 작업에는 손대지 않음).

    남은 시간 = 준비(3.1a·3.1) + 생성·검토 + 마무리(3.4·1.4·3.5)
      생성·검토: 남은 요청 수 × 몇 초마다 1개씩 끝나는지 (최근 20개가 끝난 실제 간격이라 동시 처리가 반영됨.
                 끝난 것이 5개 미만이면 요청 1개 평균 시간 ÷ 지금 동시 수). 처리 중인 요청은 지난 시간만큼 뺀다.
                 생성·검토를 함께 돌리면(3_0 '생성·검토 동시 진행') 늦게 끝나는 쪽 기준, 검토 속도는 '생성 끝 → 검토 끝' 시간 ÷ 동시 수.
      준비·마무리: 지난 실행 기록의 단계별 시간 (step_seconds, 증강은 요청 수에 비례). 진행 중인 단계는 지난 시간만큼 뺀다.
                   새 입력 날짜의 첫 실행이면 3.1a 의 기존 로그 대조 진행 줄(… N / M payload 대조)로 잰다.
    % = 지난 시간 ÷ (지난 시간 + 남은 시간)."""
    lock = RT.read_lock("generation")
    since = str(lock.get("started_at", "")).replace("T", " ") if lock else _last_run_start()
    if not since:
        return None
    by = (lock or {}).get("by", "3_0")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    steps, dry, stop, last_ts, until = {}, bool((lock or {}).get("dry_run")), False, since, None
    overlap = (lock or {}).get("step") == "3.2+3.3"
    asked = int((lock or {}).get("max_requests") or 0)  # 요청한 수 (0 = 남은 전부)
    if by == "3_0":
        for ts, lv, msg in _log_since("3_0_run_generation.log", since):
            if msg.startswith("입력 날짜"):
                if steps:                               # 다음 실행이 시작됨 → 여기까지가 이 실행
                    until = last_ts
                    break
                dry = dry or "(dry-run)" in msg
                if lock is None and (m := re.search(r"요청 (\d+)개", msg)):
                    asked = int(m.group(1))
            m = re.match(r"[▶■] \[(\S+)\] \S+ (시작|종료)(?: \(코드 (-?\d+))?", msg)
            if m:
                if m.group(1) == "3.3" and m.group(2) == "시작" and steps.get("3.2", {}).get("status") == "running":
                    overlap = True                      # 생성과 검토를 함께 돌리는 실행
                ok = m.group(2) == "시작" or m.group(3) in (None, "0")
                steps[m.group(1)] = {"status": "running" if m.group(2) == "시작" else ("done" if ok else "fail"), "at": ts}
            stop = stop or "중지 요청" in msg
            last_ts = ts
            if m and m.group(1) == "3.5" and m.group(2) == "종료":
                until = ts                              # 이 실행이 끝난 시각 (이후 로그는 다른 실행)
                break
    g = _llm_counts("3_2_generate.log", since, until, r"\s*\S+: 룰 \d+/\d+, 샘플")
    r = _llm_counts("3_3_review.log", since, until, r"\s*\S+: 샘플 \d+개 남음")
    stop = stop or g["stopped"] or r["stopped"] or RT.stop_requested("generation")
    st = lambda k: steps.get(k, {}).get("status", "wait")
    if by != "3_0":                                     # 3_2·3_3 를 명령줄에서 따로 실행한 경우
        steps = {"3.2": {"status": "running"}} if "3_2" in by else {"3.3": {"status": "running"}} if "3_3" in by else {}
    finished = lock is None
    cg, cr, _ = parallel()                              # 단계가 시작됐으면 그 실행의 실제 값 (로그 시작 줄)
    cg, cr = g["workers"] or cg, r["workers"] or cr
    # 지금 실제로 동시에 보내는 수 (자동 조절: 4개부터 늘려 최대까지, 429 면 줄임)
    ng = g["now"] or (min(cg, 4) if g["auto"] else cg)
    nr = r["now"] or (min(cr, 4) if r["auto"] else cr)
    avg = ((read_progress()["data"] or {}).get("average_per_request") or {})
    lat_g, lat_r = float(avg.get("generate_seconds") or 80.0), float(avg.get("review_seconds") or 140.0)
    gen_done, rev_done = g["ok"] + g["fail"], r["ok"] + r["fail"]
    # 몇 초마다 1개씩 끝나는지 (동시 처리 반영)
    # (예전: 단계 시작부터 잰 평균 → 실행 초반에 검토가 생성을 기다린 시간·동시 처리가 빠져 남은 시간이 몇 배로 나옴)
    def interval(done_ts, lat, conc):
        n = len(done_ts)
        if n >= 5:                                      # 최근 20개가 끝난 실제 간격 (동시 처리·지금 속도 반영)
            k = min(n, 20)
            return max(1.0, _secs(done_ts[-k], done_ts[-1]) / (k - 1))
        return max(1.0, lat / max(1, conc))              # 아직 적으면 요청 1개 평균 시간 ÷ 지금 동시 수
    gs, rs = interval(g["done_ts"], lat_g, ng), interval(r["done_ts"], lat_r, nr)

    # 생성·검토 대상 수 (로그에 나오기 전에는 추정)
    if dry:
        n_gen = n_rev = 0
    else:
        n_gen = g["target"] if g["target"] is not None else asked
        expect = g["ok"] + max(0, n_gen - gen_done)     # 앞으로 검토할 이번 생성분
        if r["follow"]:
            n_rev = min(r["cap"], r["wait"] + expect) if r["cap"] else r["wait"] + expect
        elif r["target"] is not None:
            n_rev = r["target"]
        elif stop or (by != "3_0" and "3_2" in by):
            n_rev = 0                                   # 중지 요청이면 남은 검토는 다음 실행
        else:
            n_rev = expect + g["pending"]
            n_rev = min(asked, n_rev) if asked else n_rev

    def remain(n, done, per, s, running):
        left = max(0, n - done)
        t0 = s["last"] or s["start"]
        credit = min(_secs(t0, now), 0.9 * per) if (running and t0) else 0.0
        return max(0.0, left * per - credit) if left else 0.0

    gen_rem = 0.0 if st("3.2") in ("done", "fail") else remain(n_gen, gen_done, gs, g, st("3.2") == "running")
    rev_rem = 0.0 if st("3.3") in ("done", "fail") else remain(n_rev, rev_done, rs, r, st("3.3") == "running")
    if overlap:
        # 검토는 생성된 요청이 와야 시작하므로, 끝난 간격으로 재면 생성을 기다린 시간이 섞인다 (실행 초반에 몇 배로 나옴).
        # 이번 실행에서 '생성 → 검토' 에 걸린 시간(검토 1개 소요 시간)을 지금 동시 수로 나눠 검토 속도로 쓴다.
        lags = sorted(_secs(g["by_name"][k], t) for k, t in r["by_name"].items() if k in g["by_name"] and t >= g["by_name"][k])
        lat_rv = lags[len(lags) // 2] if len(lags) >= 3 else lat_r
        if st("3.3") not in ("done", "fail"):
            rev_rem = max(0, n_rev - rev_done) * lat_rv / max(1, nr)
        # 마지막 생성이 끝난 뒤 그 검토가 한 번 더 걸린다
        llm_rem = max(rev_rem, gen_rem + (lat_rv if gen_rem and n_rev > rev_done else 0.0))
    else:
        llm_rem = gen_rem + rev_rem
    # 준비·마무리: 지난 실행 기록의 단계별 시간. 새 입력 날짜의 첫 실행이면 3.1a 가 기존 로그를 새로 대조한다 (진행 줄로 잼)
    ss, first, chk = step_seconds(), False, None
    if by == "3_0" and st("3.1") not in ("done", "fail"):
        for ts, lv, msg in _log_since("3_1a_existing_coverage.log", since):
            if until and ts > until:
                break
            if msg.startswith("[3.1a] 기존 로그"):
                first = True
            elif (m := re.search(r"(\d+) / (\d+) payload 대조 \((\d+)초\)", msg)):
                chk = (int(m.group(1)), int(m.group(2)), int(m.group(3)), ts)

    def step_left(k):
        s = st(k)
        if by != "3_0" or s in ("done", "fail") or (dry and k in ("3.4", "1.4")):
            return 0.0
        if k == "3.1a" and chk:                         # 지금까지 대조한 속도로 남은 payload
            done, total, secs, at = chk
            return max(5.0, secs * (total - done) / max(1, done) - _secs(at, now))
        d = (ss["first"] if first and k in ss["first"] else ss["normal"])[k]
        if k == "3.4":
            d = max(3.0, d * max(1, n_gen))             # 증강은 요청 수에 비례
        return max(0.1 * d, d - _secs(steps[k]["at"], now)) if s == "running" else d
    prep_rem = step_left("3.1a") + step_left("3.1")
    remaining = prep_rem + llm_rem + sum(step_left(k) for k in ("3.4", "1.4", "3.5"))
    end = now if not finished else until or max([last_ts] + [s["last"] for s in (g, r) if s["last"]])
    elapsed = _secs(since, end)

    if not finished:
        state, pct = "running", round(min(99.9, 100.0 * elapsed / max(1.0, elapsed + remaining)), 1)
    elif by != "3_0" or st("3.5") == "done":
        state, pct = "done", 100.0
    else:                                               # 강제 중지 등으로 끝까지 못 감: 한 일 / 하려던 일
        state, pct = "stopped", round(min(99.9, 100.0 * (gen_done + rev_done) / max(1, n_gen + n_rev)), 1)

    def status(k):
        s = st(k)
        if s == "running" and finished:                 # 끝나지 못하고 멈춘 단계 (강제 중지 등)
            return "fail"
        if s == "wait" and ((dry and k in ("3.4", "1.4")) or (k == "3.3" and stop and st("3.2") in ("done", "fail"))
                            or finished):
            return "skip"
        return s

    running = [k for k, _ in STEPS if st(k) == "running"] if not finished else []
    if "3.2" in running and "3.3" in running:
        current = f"생성 {gen_done}/{n_gen} · 검토 {rev_done}/{n_rev} 완료 - 지금 동시에 생성 {ng}개·검토 {nr}개 (최대 {cg}·{cr})"
    elif "3.2" in running:
        current = ("생성 대상 확인 중" if not g["header"] else "생성 마무리 중" if gen_done >= n_gen else
                   f"생성 {gen_done}/{n_gen} 완료 - {min(ng, n_gen - gen_done)}개 처리 중")
    elif "3.3" in running:
        current = ("검토 대상 확인 중" if not r["header"] else "검토 마무리 중" if rev_done >= n_rev else
                   f"검토 {rev_done}/{n_rev} 완료 - {min(nr, n_rev - rev_done)}개 처리 중")
    else:
        current = f"{running[0]} {dict(STEPS)[running[0]]} 중" if running else ""
    warn = next((s["retry"][1] for s in (g, r) if s["retry"] and _secs(s["retry"][0], now) < 180), "")
    if current and warn:
        current += f" · 최근 경고: {warn}"
    detail = {"3.2": f"{gen_done}/{n_gen}" if g["header"] and not dry else "",
              "3.3": f"{rev_done}/{n_rev}" if (r["header"] or st("3.2") != "wait") and not dry else ""}
    return {"state": state, "running": not finished, "by": by, "dry_run": dry, "started_at": since, "asked": asked,
            "elapsed_s": elapsed, "pct": pct, "remaining_s": remaining if not finished else None,
            "parallel": {"generate": cg, "review": cr, "overlap": overlap, "generate_now": ng, "review_now": nr},
            "steps": [{"key": k, "name": n, "status": status(k), "detail": detail.get(k, "")} for k, n in STEPS
                      if by == "3_0" or k in steps],
            "current": current, "stop_requested": stop and not finished,
            "gen": {"target": n_gen, "ok": g["ok"], "fail": g["fail"], "pending": g["pending"], "sec_per_req": round(gs),
                    "tokens_in": g["tin"], "tokens_out": g["tout"], "check_ok": g["chk_ok"], "check_fail": g["chk_fail"]},
            "rev": {"target": n_rev, "ok": r["ok"], "fail": r["fail"], "sec_per_req": round(rs),
                    "tokens_in": r["tin"], "tokens_out": r["tout"], "fixed": r["fixed"], "dropped": r["dropped"]}}


# ── 진행 현황·룰별 상태 ───────────────────────────────────────────────────

_progress = {"mod": None, "running": False, "error": None}
_rules_cache = {"mtime": None, "rows": []}


def refresh_progress():
    """3_5_progress 계산을 배경에서 (이미 도는 중이면 무시)."""
    with _lock:
        if _progress["running"]:
            return False
        _progress["running"] = True

    def work():
        try:
            if _progress["mod"] is None:
                _progress["mod"] = _load("progress", "3_5_progress.py")
            _progress["mod"].compute()
            _progress["error"] = None
        except Exception as e:          # 화면에 그대로 보여 줌
            _progress["error"] = f"{type(e).__name__}: {e}"
        finally:
            _progress["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return True


def read_progress():
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    return {"data": data, "computing": _progress["running"], "error": _progress["error"]}


def _q(q, key, default=""):
    return q.get(key, [default])[0]


def page_args(q, max_limit=200):
    offset = max(0, int(_q(q, "offset", "0") or 0))
    limit = min(max_limit, max(1, int(_q(q, "limit", "50") or 50)))
    return offset, limit


def rules_page(q):
    try:
        mtime = RULES_FILE.stat().st_mtime
    except OSError:
        return {"total": 0, "rows": []}
    if _rules_cache["mtime"] != mtime:
        with open(RULES_FILE, encoding="utf-8") as f:
            _rules_cache["rows"] = [json.loads(line) for line in f if line.strip()]
        _rules_cache["mtime"] = mtime
    rows = _rules_cache["rows"]
    flag, source, text = _q(q, "flag"), _q(q, "source"), _q(q, "q").strip().lower()
    if flag:
        rows = [r for r in rows if r["flag"] == flag]
    if source:
        rows = [r for r in rows if r["source"] == source]
    if text:
        rows = [r for r in rows if text in r["id"].lower() or text in r["rule_name"].lower()
                or text in r["attack_type"].lower()]
    offset, limit = page_args(q)
    return {"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]}


# ── 학습데이터 (08_dataset) ──────────────────────────────────────────────

_files = {}          # 경로 → (mtime, size, 줄 시작 위치 array)
_labels = {}         # label 파일 경로 → (mtime, label bytearray(1=anomalies), 룰 번호 array, 룰 id 목록)
_filters = {}        # (파일, 조건) → 맞는 줄 번호 목록 (최근 것만)
_check = {"in_dir": None, "rules": {}, "cross": None, "C": None, "seen": 0.0}


def dataset_dates():
    """08_dataset 날짜 폴더와 장비별 줄 수·라벨 수 (summary.json)."""
    out = []
    if not DATASET_DIR.is_dir():
        return out
    for d in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir() and DATE_RE.match(p.name)):
        try:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        files = {dev: next(iter(sorted(d.glob(f"{dev}_*_snort.log"))), None) for dev in ("ips", "waf")}
        out.append({"date": d.name, "events": s.get("events", 0), "label": s.get("label", {}),
                    "rules": s.get("rules", 0), "rejected": s.get("rejected", 0),
                    "files": {k: v.name for k, v in files.items() if v}})
    return out


def _log_path(date, dev):
    if not DATE_RE.match(date) or dev not in ("ips", "waf"):
        raise ValueError("date/dev")
    found = sorted((DATASET_DIR / date).glob(f"{dev}_*_snort.log"))
    if not found:
        raise ValueError("해당 날짜·장비의 로그가 없습니다")
    return found[0]


def line_index(path):
    """파일의 줄 시작 위치 (수정 시각·크기가 같으면 다시 만들지 않음)."""
    st = path.stat()
    hit = _files.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    offs, pos = array("q"), 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                offs.append(pos)
            pos += len(line)
    _files[path] = (st.st_mtime, st.st_size, offs)
    return offs


def read_lines(path, idx_list):
    offs = line_index(path)
    out = []
    with open(path, "rb") as f:
        for i in idx_list:
            f.seek(offs[i])
            out.append(f.readline().decode("utf-8", errors="replace").rstrip("\r\n"))
    return out


def label_info(log_path):
    """*_label.jsonl → 줄별 label·룰 id (필터용, 작게 보관)."""
    lab = log_path.with_name(log_path.stem + "_label.jsonl")
    if not lab.exists():
        return None
    st = lab.stat()
    hit = _labels.get(lab)
    if hit and hit[0] == st.st_mtime:
        return hit
    labels, rule_no, ids, id_no = bytearray(), array("I"), [], {}
    with open(lab, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            labels.append(1 if j.get("label") == "anomalies" else 0)
            rid = j.get("id", "")
            if rid not in id_no:
                id_no[rid] = len(ids)
                ids.append(rid)
            rule_no.append(id_no[rid])
    _labels[lab] = (st.st_mtime, labels, rule_no, ids, lab)
    return _labels[lab]


def matching(date, dev, label, rule, text, cons="", hold=""):
    """조건에 맞는 줄 번호 목록 (0부터). 조건이 없으면 None (= 모든 줄). cons = label 정합성 (전체 검증 결과 필요).
    hold = 보류 대상 (6. 선별의 보류 옵션과 같은 기준): rule = '정합 (룰 기준)' 공격, pyua = python-requests UA 정상."""
    path = _log_path(date, dev)
    if hold == "rule":
        label, cons = "anomalies", "RULE"
    elif hold == "pyua":
        label = "normal"
    if not (label or rule or text or cons):
        return None
    key = (str(path), path.stat().st_mtime, label, rule, text, cons, hold)
    if key in _filters:
        return _filters[key]
    n = len(line_index(path))
    keep = list(range(n))
    if cons:
        v = verify_state(date, dev)
        if v.get("status") != "done":
            raise ValueError(f"{date} {dev.upper()} 는 '전체 검증'을 먼저 실행하세요")
        want = CODE_NUM[cons]
        keep = [i for i in keep if i < len(v["codes"]) and v["codes"][i] == want]
    li = label_info(path)
    if (label or rule) and li:
        _, labels, rule_no, ids, _ = li
        if label:
            want = 1 if label == "anomalies" else 0
            keep = [i for i in keep if i < len(labels) and labels[i] == want]
        if rule:
            r = rule.lower()
            ok = {k for k, rid in enumerate(ids) if r in rid.lower()}
            keep = [i for i in keep if i < len(rule_no) and rule_no[i] in ok]
    if text:
        t = text.lower().encode("utf-8")
        hits = set()
        with open(path, "rb") as f:
            for i, line in enumerate(x for x in f if x.strip()):
                if t in line.lower():
                    hits.add(i)
        keep = [i for i in keep if i in hits]
    if hold == "pyua":                                  # 헤더의 User-Agent 가 python-requests (로그의 줄바꿈은 글자 \n)
        with open(path, "rb") as f:
            hits = {i for i, line in enumerate(x for x in f if x.strip()) if PY_UA_RE.search(line)}
        keep = [i for i in keep if i in hits]
    if len(_filters) > 16:
        _filters.pop(next(iter(_filters)))
    _filters[key] = keep
    return keep


def select_rows(dates, dev, label, rule, text, offset, limit, cons="", hold=""):
    """여러 날짜를 이어 붙인 것처럼 페이지를 고른다 → (전체 수, [(날짜, 그 날짜의 줄 번호, 전체 index)])."""
    parts = []
    for d in dates:
        m = matching(d, dev, label, rule, text, cons, hold)
        parts.append((d, m, len(line_index(_log_path(d, dev))) if m is None else len(m)))
    total = sum(p[2] for p in parts)
    picked, start = [], 0
    for d, m, n in parts:
        lo, hi = max(offset, start), min(offset + limit, start + n)
        for g in range(lo, hi):
            local = g - start
            picked.append((d, local if m is None else m[local], g))
        start += n
        if start >= offset + limit:
            break
    return total, picked


def checker():
    """현재 룰(05_llm_input 최신 rules_all)과 normal 교차 검사 함수 (처음 한 번 불러 둠).
    새 입력 날짜는 5초에 한 번만 확인한다 (폴더 찾기가 한 번에 8ms 라 줄마다 하면 전체 검증이 3배 가까이 느려짐)."""
    with _lock:
        if _check["C"] is None:
            _check["C"] = _load("llm_common_ui", "3_0a_common.py")
        C = _check["C"]
        if time.monotonic() - _check["seen"] < 5:
            return C, _check["rules"], _check["cross"]
        _check["seen"] = time.monotonic()
        in_dir = C.latest_input_dir()
        if in_dir is not None and _check["in_dir"] != in_dir.name:
            _check["rules"] = {r["id"]: r for r in C.read_jsonl(in_dir / "rules_all.jsonl")}
            try:
                aug = _load("augment_ui", "3_4_augment.py")
                _, _, run = C.load_settings()
                _check["cross"] = aug.make_cross_check(in_dir, run, logging.getLogger("ui"))
            except Exception:
                _check["cross"] = None
            _check["in_dir"] = in_dir.name
    return C, _check["rules"], _check["cross"]


def _cond_text(c):
    v = c.get("value") if c["type"] == "content" else c.get("pattern")
    v = (v or "")
    return f"{'정규식' if c['type'] == 'regex' else 'content'} {c.get('in', '')} '{v[:40]}{'…' if len(v) > 40 else ''}'" + \
           (" (없어야 함)" if c.get("negated") else "")


TRACE_NAMES = {"sql": "SQL 인젝션 흔적", "xss": "XSS 흔적", "cmd": "명령 주입 흔적", "traversal": "경로 이동 흔적",
               "inclusion": "파일 포함 흔적", "template": "템플릿·EL 주입 흔적", "xxe": "XXE 흔적", "deserial": "역직렬화 흔적",
               "shellshock": "Shellshock 흔적", "sensitive": "민감 파일 접근", "double_enc": "기호 이중 URL 인코딩",
               "unicode_enc": "%u 인코딩", "ctrl_bytes": "제어문자·NULL 바이트", "scanner_ua": "스캐너·공격 도구 UA"}
CONS = {"OK": "정합", "RULE": "정합 (룰 기준)", "CHECK": "확인 필요", "BAD": "불일치"}
HOLD = {"rule": "'정합 (룰 기준)' 공격", "pyua": "python-requests UA 정상"}     # 6. 선별의 보류 옵션 대상
PY_UA_RE = re.compile(rb"\\nuser-agent:[ \t]*python-requests", re.I)          # 5.5 와 같은 기준 (헤더 줄 시작)


def split_log(line):
    """학습 로그 한 줄 → 필드 목록. 8필드면 앞 6필드는 앞에서, label 은 마지막 '|||' 뒤로 읽는다
    (payload 가 '|' 로 끝나면 '…|id||||anomalies' 처럼 구분자와 붙어, 앞에서만 나누면 label 이 '|anomalies' 가 됨)."""
    head = line.split("|||", 6)
    if len(head) == 7 and "|||" in head[6]:
        return head[:6] + head[6].rsplit("|||", 1)
    return line.split("|||")
_feat = {"in_dir": None, "F": None, "matcher": None, "types": {}, "note": ""}


def quality_keep(attack, C, in_dir):
    """4.1 과 같은 기준의 '신뢰할 수 있는 공격 룰' (기존 normal 에도 걸리는 모호한 룰 제외). 이미 계산된 기록만 쓴다."""
    try:
        hits = json.loads((BASE / "09_features" / "_rule_quality.json").read_text(encoding="utf-8")).get("normal_hits", {})
    except (OSError, ValueError):
        return {r["id"] for r in attack}, "룰 신뢰도 기록 없음 - 모호한 룰도 포함 (4_0 을 한 번 실행하면 생김)"
    base = C.output_dir(in_dir.name)
    src = next((b for b in [base] + C.output_dates()[::-1] if (b / "_0_existing" / "coverage.json").exists()), None)
    cov = json.loads((src / "_0_existing" / "coverage.json").read_text(encoding="utf-8")) if src else {}
    keep = set()
    for r in attack:
        n, a = hits.get(r["id"], 0), cov.get(r["id"], {}).get("count", 0)
        if not (n and a / (a + n) < 0.9):
            keep.add(r["id"])
    return keep, f"모호한 룰 {len(attack) - len(keep)}개 제외"


def feature_engine():
    """탐지 피처 계산 준비: 4.1 의 공격 흔적 함수와 '신뢰할 수 있는 공격 룰' 대조기 (처음 한 번)."""
    C, rules, _ = checker()
    with _lock:
        if _feat["F"] is None:
            _feat["F"] = _load("features_ui", "4_1_build_features.py")
        in_name = _check["in_dir"]
        if in_name and _feat["in_dir"] != in_name:
            F = _feat["F"]
            attack = [r for r in rules.values() if F.is_attack_rule(r)]
            keep, note = quality_keep(attack, C, C.latest_input_dir())
            attack = [r for r in attack if r["id"] in keep]
            _feat["matcher"] = C.RuleMatcher(attack)
            _feat["types"] = {r["id"]: (r.get("attack_type") if r.get("attack_type") in F.ATTACK_TYPES else "other")
                              for r in attack}
            _feat["note"], _feat["in_dir"] = note, in_name
    return _feat


def _lost_reading(C, raw, http, conditions):
    """anomalies 인데 룰 조건이 어긋날 때: 로그 형식이 잃는 정보를 되살려 읽으면 조건이 모두 맞는지 → 그 읽기 ('' = 그래도 불일치).
    1.4 는 payload 끝 줄바꿈을 지우고(헤더만 있는 요청은 끝 빈 줄까지), 줄바꿈을 두 글자 '\\n' 으로 적어 본문의 글자 역슬래시+n 과
    구별되지 않는다 (요청 줄만 %5C). 예 (2026-10-26): FortiOS·Nessus 룰은 끝 '\\r\\n\\r\\n', Netgear 룰은 끝 줄바꿈,
    Exiftool 룰은 본문의 글자 '\\n' 이 조건이라 로그로는 확인할 수 없다 (생성 원본은 일치)."""
    head, sep, body = raw.replace("\\r\\n", "\\n").partition("\\n\\n")
    lit = C.norm_payload(head) + "\n\n" + body if sep and "\\n" in body else None      # 본문의 '\n' 을 글자 그대로
    for name, h in (("끝 줄바꿈이 지워짐", http + "\n"), ("끝 빈 줄이 지워짐", http + "\n\n"),
                    ("본문의 글자 '\\n' 을 줄바꿈으로 읽음", lit), ("본문의 글자 '\\n'·끝 줄바꿈", lit and lit + "\n")):
        if h is not None:
            parts = C.split_http(h)
            if all(C.check_condition(c, parts) is True for c in conditions):
                return name
    return ""


def analyze(payload, label, meta, ctx=None):
    """데이터 프레임 한 줄 → label 사유, 검증 결과, label 정합성(+사유), 탐지 피처.
    ctx = (C, rules, cross, feature_engine 결과): 전체 검증처럼 많은 줄을 볼 때 한 번 정해 넘긴다 (도중에 새 입력 날짜가 와도 같은 룰로).

    label 정합성 (라벨이 증거와 맞는지):
      OK    정합           anomalies: 룰 조건 모두 일치 + 범용 공격 흔적이나 다른 공격 룰도 공격으로 봄
                           normal:    룰 조건 불일치 + 공격 흔적·공격 룰 없음
      RULE  정합(룰 기준)  anomalies: 룰 조건은 모두 일치하지만 범용 공격 흔적·다른 공격 룰은 없음 (룰 고유 패턴으로만 공격)
      CHECK 확인 필요      룰이 없거나 판단 불가 조건(정규식 해석 불가)이 있음 / anomalies 인데 로그 그대로는 룰 조건이 어긋나지만
                           로그 형식이 잃는 정보(끝 줄바꿈·본문의 글자 '\\n')를 되살려 읽으면 모두 일치 (_lost_reading)
      BAD   불일치         anomalies 인데 룰 조건 불일치 / normal 인데 룰에 탐지되거나 공격 흔적·공격 룰이 있음
    탐지 피처: 4.1 의 피처 중 이 payload 에서 켜지는 것 (공격 흔적 k_, 공격 룰 ra_, 이 룰의 개별 룰·정규식 r_)
    """
    C, rules, cross, fe = ctx or (*checker(), feature_engine())
    F = fe["F"]
    rid = meta.get("id", "")
    rule = rules.get(rid)
    note = meta.get("note", "")
    checks, level = [], "ok"
    origin = meta.get("origin", "")
    # 1.4 가 로그에 '\x00' 처럼 글자로 적은 제어문자(줄바꿈 제외)를 실제 바이트로 되돌려 확인 (NUL 바이트 룰 등)
    raw = re.sub(r"\\x(0[0-9b-f]|1[0-9a-f]|7f)", lambda m: chr(int(m.group(1), 16)), payload)
    http = C.norm_payload(raw)
    feats = []
    traces = {k[2:]: v for k, v in F.trace_features(http).items() if v}
    for k, v in traces.items():
        feats.append({"name": f"k_{k}", "value": v, "group": "회피·이상 신호" if k in F.EVASION else "공격 흔적",
                      "desc": TRACE_NAMES.get(k, k)})
    by_type = {}
    for h, kind in fe["matcher"].match(http):
        if kind == "match":
            by_type.setdefault(fe["types"].get(h, "other"), []).append(h)
    for t, ids in sorted(by_type.items()):
        feats.append({"name": f"ra_{t}", "value": len(ids), "group": "공격 룰",
                      "desc": ", ".join(sorted(ids)[:3]) + (" …" if len(ids) > 3 else "")})
    other_rules = sorted({h for ids in by_type.values() for h in ids if h != rid})
    trace_txt = ", ".join(TRACE_NAMES.get(k, k) for k in traces)
    if meta.get("local_check"):
        checks.append(("ok" if meta["local_check"] == "ok" else "warn", f"생성 시 코드 검사 {meta['local_check']}"))
    if origin == "llm":
        checks.append(("ok" if meta.get("review") in ("pass", "fix") else "warn",
                       f"검토 모델 {meta.get('review') or '없음'}" + (" (고쳐서 통과)" if meta.get("review") == "fix" else "")))
    elif origin == "augmented":
        checks.append(("ok", "증강 샘플 (씨앗은 검토 통과)"))
    if rule is None:
        reason = f"룰 {rid}: 현재 룰 목록에 없음"
        checks.append(("warn", "지금 다시 확인 못 함 (룰 없음)"))
        cons, cons_reason = "CHECK", f"룰 {rid} 가 현재 룰 목록에 없어 룰 기준으로 확인할 수 없음"
    else:
        parts = C.split_http(http)
        res = [(c, C.check_condition(c, parts)) for c in rule["conditions"]]
        n = len(res)
        ok_n = sum(1 for _, r in res if r is True)
        failed = [c for c, r in res if r is False]
        unknown = [c for c, r in res if r is None]
        lost = _lost_reading(C, raw, http, rule["conditions"]) if label == "anomalies" and failed else ""
        slug = F.rule_slug(rid)
        if not failed and not unknown:
            feats.append({"name": f"r_{slug}_all", "value": 1, "group": "개별 룰", "desc": f"이 룰 탐지 ({rid})"})
        for i, (c, r) in enumerate(res):
            if c["type"] == "regex" and r is True and not c.get("negated"):
                feats.append({"name": f"r_{slug}_re{i}", "value": 1, "group": "개별 룰",
                              "desc": f"정규식 조건 일치: {(c.get('pattern') or '')[:40]}"})
        # label 정합성
        if label == "anomalies":
            if lost:
                cons, cons_reason = "CHECK", (f"로그 형식({lost}) 때문에 룰 {rid} 조건 {_cond_text(failed[0])} 을 로그로 확인할 수 "
                                              f"없음 - 되살려 읽으면 {n}/{n} 일치"
                                              + (" (생성 시 코드 검사 ok)" if meta.get("local_check") == "ok" else ""))
            elif failed:
                cons, cons_reason = "BAD", (f"anomalies 인데 룰 {rid} 조건 {ok_n}/{n} 만 일치 (불일치: {_cond_text(failed[0])})"
                                            " → 룰 기준으로 공격이 아님")
            elif unknown:
                cons, cons_reason = "CHECK", f"룰 {rid} 조건 중 판단할 수 없는 조건(정규식 해석 불가) {len(unknown)}개"
            elif trace_txt or other_rules:
                ev = ([f"공격 흔적 {trace_txt}"] if trace_txt else []) + \
                     ([f"다른 공격 룰 {len(other_rules)}개 ({', '.join(other_rules[:2])})"] if other_rules else [])
                cons, cons_reason = "OK", f"룰 {rid} 조건 {n}/{n} 일치 + {' · '.join(ev)} → 공격이 맞음"
            else:
                cons, cons_reason = "RULE", (f"룰 {rid} 조건 {n}/{n} 일치. 범용 공격 흔적·다른 공격 룰은 없음 → "
                                             "이 룰의 고유 패턴(경로·파라미터·UA 등)으로만 공격으로 봄")
        else:
            if not failed and not unknown:
                cons, cons_reason = "BAD", f"normal 인데 룰 {rid} 조건 {n}/{n} 을 모두 만족 (이 룰에 탐지됨)"
            elif trace_txt or by_type:
                hit_rules = sorted({h for ids in by_type.values() for h in ids})
                ev = ([f"공격 흔적 {trace_txt}"] if trace_txt else []) + \
                     ([f"공격 룰 {len(hit_rules)}개 탐지 ({', '.join(hit_rules[:2])})"] if hit_rules else [])
                cons, cons_reason = "BAD", f"normal 인데 {' · '.join(ev)} → 보안 관점에서 정상이 아님"
            elif not failed:
                cons, cons_reason = "CHECK", f"룰 {rid} 조건 중 판단할 수 없는 조건이 있어 정상인지 확정 못 함"
            else:
                cons, cons_reason = "OK", (f"룰 {rid} 조건 불일치({_cond_text(failed[0])}) + 공격 흔적·공격 룰 없음"
                                           " → 정상이 맞음")
        if label == "anomalies":
            reason = f"룰 {rid} 탐지 조건 {n}개를 모두 만족 → 공격 ({meta.get('attack_type', '')})"
            if failed:
                checks.append(("warn" if lost else "bad", f"지금 다시 확인: 조건 {ok_n}/{n} 일치 - 불일치 {_cond_text(failed[0])}"
                               + (f" (로그 형식 때문: {lost} → 되살리면 {n}/{n})" if lost else "")))
            else:
                checks.append(("ok", f"지금 다시 확인: 조건 {ok_n}/{n} 일치"))
        else:
            if failed:
                reason = f"룰 {rid} 조건 불일치: {_cond_text(failed[0])}" + \
                         (f" 외 {len(failed) - 1}개" if len(failed) > 1 else "") + " → 정상 요청"
                checks.append(("ok", f"지금 다시 확인: 이 룰에 탐지 안 됨 ({len(failed)}/{n}개 조건 불일치)"))
            else:
                reason = f"룰 {rid}: 어긴 조건을 확인하지 못함"
                checks.append(("bad", "지금 다시 확인: 이 룰의 조건을 모두 만족 (탐지될 수 있음)"))
            if cross:
                hits = cross(http, rid)
                checks.append(("bad", f"다른 공격 룰·공격 흔적: {', '.join(hits[:3])}") if hits
                              else ("ok", "다른 공격 룰·공격 흔적 없음"))
    if note:
        reason += f" · 생성 메모: {note}"
    if any(s == "bad" for s, _ in checks) or cons == "BAD":
        level = "bad"
    elif any(s == "warn" for s, _ in checks) or cons == "CHECK":
        level = "warn"
    return {"reason": reason, "checks": checks, "level": level, "cons": cons, "cons_text": CONS[cons],
            "cons_reason": cons_reason, "features": feats, "lv": VERIFY_LOGIC}


# ── 피처 설명 (4.1 피처 목록 · 읽기만) ──────────────────────────────────

TRACE_DOC = {   # k_ 피처: (무엇을 보는지, 예)
    "sql": ("값 안에 SQL 문법이 들어간 흔적: UNION SELECT, 따옴표 뒤 OR 1=1, 시간 지연 함수, DB 구조 조회, "
            "따옴표 뒤 주석, ; 뒤 DROP·INSERT 등",
            "id=1' OR '1'='1 · UNION SELECT · sleep(5) · information_schema"),
    "xss": ("브라우저에서 실행될 스크립트를 넣은 흔적: <script>, on이벤트 속성(onerror= 등), javascript: 주소, "
            "alert( · document.cookie",
            "<script>alert(1)</script> · <img src=x onerror=…> · javascript:"),
    "cmd": ("서버 명령을 이어 붙인 흔적: ; | && 뒤의 명령어(id, cat, wget 등), $( ) · 백틱 실행, /bin/sh, cmd /c, "
            "powershell -",
            ";cat /etc/passwd · $(id) · |whoami · /bin/sh"),
    "traversal": ("상위 폴더로 거슬러 올라가 시스템 파일을 읽으려는 흔적: ../ 두 번 이상, 인코딩한 ../, /etc/passwd, "
                  "win.ini·boot.ini, /proc/self/",
                  "../../etc/passwd · %2e%2e%2f · ..\\..\\boot.ini"),
    "inclusion": ("파일을 코드로 불러오게 하는 흔적: php:// · file:// · data:…;base64 주소, 파라미터에 원격 스크립트 URL",
                  "php://filter/… · file:///etc/passwd · ?page=http://x/s.txt?"),
    "template": ("템플릿·표현식 주입 흔적: Log4Shell ${jndi:, ${env:…}, {{ }} 안의 코드, #{…( · %{…( · <%=",
                 "${jndi:ldap://…} · {{config}} · %{(#…)}"),
    "xxe": ("XML 외부 엔티티 흔적: <!ENTITY 선언, DOCTYPE 안의 엔티티, SYSTEM \"file:…\"",
            "<!DOCTYPE x [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>"),
    "deserial": ("역직렬화·프로토타입 오염 흔적: 자바 직렬화 시작 바이트(rO0AB · aced0005), PHP 객체 표기, __proto__, "
                 "ysoserial, \"$type\":, !!python/object",
                 "rO0ABXNy… · O:8:\"stdClass\":1:{ · __proto__"),
    "shellshock": ("Bash Shellshock 공격 문자열: () { :; };", "User-Agent: () { :; }; /bin/bash -c …"),
    "sensitive": ("민감 파일 직접 요청: /.git/config, /.env, wp-config.php, web.config, id_rsa, /.htpasswd, /.ssh/",
                  "GET /.git/config · GET /.env"),
    "double_enc": ("기호를 두 번 URL 인코딩해 필터를 피하려는 흔적: %25 뒤에 . / \\ < > ' \" NUL ; | 의 인코딩",
                   "%252e%252e%252f · %253cscript"),
    "unicode_enc": ("IIS 식 %u 유니코드 인코딩: 일반 웹 요청에서는 쓰지 않는 표기", "%u002e%u002e%u2215"),
    "ctrl_bytes": ("요청 줄·헤더·일반 본문의 제어문자·NULL 바이트 (줄바꿈·탭, 파일 업로드 본문의 바이너리는 제외)",
                   "file.php%00.jpg · \\x00 · \\x1b"),
    "scanner_ua": ("User-Agent 가 스캐너·공격 도구: sqlmap, Nikto, Nmap, Nuclei, Acunetix, WPScan, ffuf 등",
                   "User-Agent: sqlmap/1.7 · Nikto/2.5"),
}
ATTACK_NAMES = {"sql_injection": "SQL 인젝션", "xss": "XSS", "xxe": "XXE", "ssrf": "SSRF", "deserialization": "역직렬화",
                "code_injection": "코드 주입", "command_injection": "명령 주입", "file_inclusion": "파일 포함",
                "path_traversal": "경로 이동", "file_upload": "파일 업로드", "webshell": "웹셸", "auth_bypass": "인증 우회",
                "rce": "원격 코드 실행", "buffer_overflow": "버퍼 오버플로", "info_disclosure": "정보 노출",
                "scanner": "스캐너", "dos": "서비스 거부", "csrf": "CSRF", "open_redirect": "오픈 리다이렉트",
                "other": "기타 공격"}
GROUP_DOC = [
    ("공격 흔적", "k_", "payload 를 URL 디코딩(2번)하고 소문자로 바꾼 뒤 공격 유형별 키워드가 몇 번 나오는지. "
                       "정상 요청에도 나오는 형태는 뺐다 (예: 명령 패턴은 & 가 아니라 && 만)."),
    ("회피·이상 신호", "k_", "탐지를 피하려는 인코딩, 제어문자·NULL 바이트, 공격 도구 User-Agent. 원문 payload 에서 센다."),
    ("공격 룰", "ra_", "신뢰할 수 있는 공격 룰(정보성 룰, 기존 normal 에도 걸리는 모호한 룰 제외)과 대조해 "
                      "공격 유형별로 모든 조건이 일치한 룰 수. 조건 일부만 맞거나 확인 못 한 조건이 있으면 세지 않는다."),
    ("개별 룰", "r_", "특정 룰 하나의 탐지 여부와 그 룰의 정규식 조건별 일치 (4_0 --rules 로 지정). "
                     "신규 학습 데이터 화면에서는 그 줄을 만든 룰로 계산한다."),
]
R_ALL, R_RE = "r_<룰>_all", "r_<룰>_re<n>"


def feature_key(name):
    """개별 룰 피처(r_<룰>_…)는 룰마다 이름이 달라 설명·집계는 한 줄로 묶는다."""
    if name.startswith("r_"):
        return R_ALL if name.endswith("_all") else R_RE
    return name


def feature_catalog():
    """4.1 피처 전체와 설명. 패턴은 4.1 코드에서 그대로 가져온다 (설명과 실제 계산이 어긋나지 않게)."""
    fe = feature_engine()
    F = fe["F"]
    _, rules, _ = checker()
    items = []
    for k, rx in F.KEYWORDS.items():
        what, ex = TRACE_DOC.get(k, ("", ""))
        items.append({"name": f"k_{k}", "group": "공격 흔적", "title": TRACE_NAMES.get(k, k), "what": what,
                      "example": ex, "value": "나온 횟수", "pattern": rx})
    evasion_pat = {"double_enc": r"%25(?:2e|2f|5c|3c|3e|27|22|00|3b|7c)", "unicode_enc": r"%u[0-9a-f]{4}",
                   "ctrl_bytes": r"글자 코드 < 32 (\n \r \t 제외) + 글자로 적힌 \x00~\x1f + %00",
                   "scanner_ua": F.UA_SCANNER.pattern}
    for k in F.EVASION:
        what, ex = TRACE_DOC.get(k, (F.EVASION[k], ""))
        items.append({"name": f"k_{k}", "group": "회피·이상 신호", "title": TRACE_NAMES.get(k, k), "what": what,
                      "example": ex, "value": "해당하면 1" if k == "scanner_ua" else "나온 횟수",
                      "pattern": evasion_pat.get(k, "")})
    by_type = {}
    for rid, t in fe["types"].items():
        by_type.setdefault(t, []).append(rid)
    for t in F.ATTACK_TYPES:
        ids = sorted(by_type.get(t, []))
        items.append({"name": f"ra_{t}", "group": "공격 룰", "title": f"{ATTACK_NAMES.get(t, t)} 공격 룰 일치 수",
                      "what": f"신뢰할 수 있는 {ATTACK_NAMES.get(t, t)} 공격 룰 {len(ids):,}개 중 이 payload 에서 "
                              "모든 조건이 일치한 룰 수", "rules": len(ids),
                      "example": " · ".join(f"{i} {rules[i]['rule_name']}" for i in ids[:3] if i in rules),
                      "value": "일치한 룰 수", "pattern": ""})
    items.append({"name": "ra_total", "group": "공격 룰", "title": "공격 룰 일치 수 합계",
                  "what": f"위 공격 유형 {len(F.ATTACK_TYPES)}개의 합 (신뢰할 수 있는 공격 룰 {len(fe['types']):,}개 전체)",
                  "rules": len(fe["types"]), "example": "", "value": "일치한 룰 수", "pattern": ""})
    items.append({"name": R_ALL, "group": "개별 룰", "title": "이 룰 탐지",
                  "what": "그 룰의 모든 조건(content·정규식·위치·순서)을 만족하면 1. 학습데이터 라벨과 같은 기준이라 "
                          "평가할 때는 룰 단위로 나눠야 한다 (라벨 누수)",
                  "example": "r_et_1_2053468_all", "value": "해당하면 1", "pattern": ""})
    items.append({"name": R_RE, "group": "개별 룰", "title": "이 룰의 정규식 조건 일치",
                  "what": "그 룰의 정규식(pcre) 조건 하나가 일치하면 1 (n = 룰 안의 조건 번호). 룰 전체가 아니라 "
                          "핵심 패턴만 맞아도 켜진다",
                  "example": "r_et_1_2053468_re2 (et:1:2053468 의 정규식 UNION.+SELECT)", "value": "해당하면 1",
                  "pattern": ""})
    return {"items": items, "groups": [{"name": n, "prefix": p, "desc": d} for n, p, d in GROUP_DOC],
            "attack_rules": len(fe["types"]), "note": fe["note"]}


def feature_counts(dev):
    """전체 검증 결과에서 피처별로 켜진 줄 수 (anomalies / normal). 검증하지 않은 날짜는 따로 알려 준다."""
    out = {"labels": {"anomalies": 0, "normal": 0}, "counts": {}, "dates": [], "missing": [], "running": False}
    for d in [x["date"] for x in dataset_dates()]:
        v = verify_state(d, dev)
        if v.get("status") == "running":
            out["running"] = True
        if v.get("status") != "done" or "features" not in v:
            out["missing"].append(d)
            continue
        out["dates"].append(d)
        for lab, n in v.get("labels", {}).items():
            out["labels"][lab] = out["labels"].get(lab, 0) + n
        for name, (a, n) in v["features"].items():
            c = out["counts"].setdefault(name, [0, 0])
            c[0] += a
            c[1] += n
    return out


# ── 전체 검증 (파일의 모든 줄의 label 정합성) ────────────────────────────
# 검증 안 한 데이터만 (mode "new"): 이미 확인한 줄은 그 결과를 그대로 쓰고 나머지만 새로 확인한다.
#   이미 확인한 줄 = 이 장비의 줄별 결과(로그가 그대로면 줄 순서대로, 다시 만들어졌으면 내용이 같은 줄), 중지할 때 저장한 결과,
#   다른 장비의 결과(IPS·WAF 로그는 같은 샘플). 새 생성 실행으로 로그가 다시 만들어져도 새로 생긴 줄만 확인한다.
#   수집으로 룰 입력 날짜가 바뀌어도 예전 결과를 그대로 쓴다 (줄마다 확인한 룰 입력 날짜를 남기고 화면에 보여 줌).
# 전체 검증 (mode "all"): 모든 줄을 지금 룰로 새로 확인한다 (최신 룰로 다시 볼 때).
# 줄별 결과(<장비>_frame.jsonl)의 줄마다 내용 지문(k)·룰 입력 날짜(in)를 적는다. 중지하면 확인한 줄은 <장비>_frame.partial.jsonl.

_verify = {}
_stopped = {}        # (날짜, 장비) → 중지한 검증 (화면 표시)
_saved = {}          # verify.json → (수정 시각, 내용)
CODE_NUM = {"OK": "1", "RULE": "2", "CHECK": "3", "BAD": "4"}
# analyze 판정 규칙 판 (줄별 결과의 lv). 올리면 예전 판으로 확인한 줄 중 '불일치' 만 다시 확인한다 (나머지는 그대로 다시 씀).
# 2 = anomalies 의 룰 조건 불일치를 로그 형식이 잃는 정보(끝 줄바꿈·본문의 글자 '\n')를 되살려 다시 확인 (2026-10-26)
VERIFY_LOGIC = 2
MS_PER_LINE = 12.0   # 검증 기록이 없을 때 새로 확인하는 줄 1개의 시간 (데이터에 따라 8~30ms)
REUSE_SEC = 0.0003   # 이미 확인한 결과를 그대로 쓰는 줄 1개의 시간
ROW_HEAD = re.compile(rb'^\{"k": "([0-9a-f]+)", "in": "([^"]*)"')


class _Stop(Exception):
    """검증 중지: 중지 요청, 또는 검증 중에 로그가 다시 만들어짐."""


def _verify_file(date, dev):
    return DATASET_DIR / date / f"{dev}_verify.json"


def _frame_file(date, dev):
    return DATASET_DIR / date / f"{dev}_frame.jsonl"


def _partial_file(date, dev):
    return DATASET_DIR / date / f"{dev}_frame.partial.jsonl"


def _other(dev):
    return "waf" if dev == "ips" else "ips"


def verify_state(date, dev):
    """그 날짜·장비 로그의 전체 검증 결과 (메모리 → 저장 파일). 로그가 바뀌었으면 무효."""
    path = _log_path(date, dev)
    st = path.stat()
    v = _verify.get((date, dev))
    if v and (v["status"] == "running" or (v.get("mtime") == st.st_mtime and v.get("size") == st.st_size)):
        return v
    try:
        saved = json.loads(_verify_file(date, dev).read_text(encoding="utf-8"))
        if saved.get("mtime") == st.st_mtime and saved.get("size") == st.st_size:
            saved["status"] = "done"
            _verify[(date, dev)] = saved
            return saved
    except (OSError, ValueError):
        pass
    return {"status": "none"}


def _saved_verify(date, dev):
    """저장된 verify.json (로그가 바뀌어 무효여도): 지난 검증의 줄당 시간 → 예상 시간."""
    p = _verify_file(date, dev)
    try:
        mt = p.stat().st_mtime
        if _saved.get(p, (None,))[0] != mt:
            _saved[p] = (mt, json.loads(p.read_text(encoding="utf-8")))
        return _saved[p][1]
    except (OSError, ValueError):
        return {}


def _ms_per_line(date, dev):
    """새로 확인하는 줄 1개의 시간(ms): 같은 날짜의 지난 검증 기록, 없으면 기본값."""
    for d in (dev, _other(dev)):
        ms = _saved_verify(date, d).get("ms_per_line")
        if ms:
            return ms
    return MS_PER_LINE


def _meta_key(meta):
    """analyze 가 보는 줄 정보 (장비마다 다른 method 는 빼고)."""
    return tuple(str(meta.get(k) or "") for k in ("id", "note", "origin", "local_check", "review", "attack_type"))


def _row_key(payload, label, meta):
    """줄 내용 지문: payload·label·줄 정보가 같으면 확인 결과도 같다 (IPS·WAF 의 같은 샘플, 다시 만든 로그의 같은 줄)."""
    return hashlib.sha1("\x00".join((payload, label) + _meta_key(meta)).encode("utf-8", "replace")).hexdigest()[:20]


def _known_rows(date, dev, in_dir):
    """이미 확인한 줄 → ({내용 지문: (파일 번호, 위치, 룰 입력 날짜)}, [파일]). 같은 줄이 여러 번 있으면 지금 룰 입력의 결과를 쓴다.
    다른 장비가 검증 중이면 그쪽 파일은 쓰지 않는다 (끝날 때 그 파일을 바꿔야 하므로)."""
    known, files, want = {}, [], in_dir.encode()
    for d in (dev, _other(dev)):
        if d != dev and _verify.get((date, d), {}).get("status") == "running":
            continue
        for p in (_frame_file(date, d), _partial_file(date, d)):
            try:
                with open(p, "rb") as f:
                    no, pos = len(files), 0
                    for line in f:
                        m = ROW_HEAD.match(line)
                        if m:
                            k = m.group(1).decode()
                            if k not in known or (m.group(2) == want and known[k][2] != want):
                                known[k] = (no, pos, m.group(2))
                        pos += len(line)
            except OSError:
                continue
            files.append(p)
    return known, files


def _stale_bad(v):
    """예전 판정 규칙(VERIFY_LOGIC 전)으로 끝난 검증의 '불일치' 줄 수 - '검증 안 한 데이터만' 이 다시 확인한다."""
    return (v.get("summary") or {}).get("BAD", 0) if v.get("logic", 1) < VERIFY_LOGIC else 0


def _all_verified(date, dev):
    """새로 확인할 줄이 없음: 지금 로그·룰 입력으로 끝난 검증이 있고 줄별 결과에 내용 지문이 있음 (예전 판정 규칙의 '불일치' 없음)."""
    p = frame_cache(date, dev)
    if p is None or _stale_bad(verify_state(date, dev)):
        return False
    with open(p, "rb") as f:
        return bool(ROW_HEAD.match(f.readline()))


def start_verify(date, dev, mode="all"):
    """mode: new = 검증 안 한 데이터만, all = 전체. → "started" | "running"(이미 도는 중) | "done"(새로 확인할 줄 없음)."""
    path = _log_path(date, dev)
    with _lock:
        if _verify.get((date, dev), {}).get("status") == "running":
            return "running"
        if mode == "new" and _all_verified(date, dev):
            return "done"
        st = path.stat()
        v = _verify[(date, dev)] = {"status": "running", "mode": mode, "phase": "start", "done": 0, "logic": VERIFY_LOGIC,
                                    "total": len(line_index(path)), "todo": None, "analyzed": 0, "reused": 0,
                                    "eta_s": None, "ms": _ms_per_line(date, dev), "mtime": st.st_mtime,
                                    "size": st.st_size, "started_at": RT.now_iso()}
        _stopped.pop((date, dev), None)
    threading.Thread(target=_verify_work, args=(date, dev, mode, path, v), daemon=True).start()
    return "started"


def stop_verify(date, dev):
    """검증 중지 요청. 확인한 줄은 저장되어 '검증 안 한 데이터만' 으로 이어서 할 수 있다."""
    v = _verify.get((date, dev))
    if v and v.get("status") == "running":
        v["stop"] = True
        return True
    return False


def _verify_work(date, dev, mode, path, v):
    frame, part = _frame_file(date, dev), _partial_file(date, dev)
    tmp = frame.with_name(f"{frame.name}.{os.getpid()}.tmp")
    for old in frame.parent.glob(f"{frame.name}.*.tmp"):        # 화면을 끄면서 남은 임시 파일 (그 프로세스가 없으면)
        pid = old.name.split(".")[-2]
        if pid != str(os.getpid()) and not RT.pid_alive(pid):
            try:
                old.unlink()
            except OSError:
                pass
    srcs, out = [], None
    try:
        # WAF 는 IPS 검증 중이면 끝나기를 기다렸다 (같은 샘플이라) 그 결과를 쓴다. IPS 는 기다리지 않음 (서로 기다리지 않게)
        while mode == "new" and dev == "waf" and _verify.get((date, "ips"), {}).get("status") == "running":
            v["phase"] = "wait"
            if v.get("stop"):
                raise _Stop("중지 요청")
            time.sleep(2)
        v["phase"] = "prep"
        ctx = (*checker(), dict(feature_engine()))      # 이번 검증은 한 룰 입력으로
        in_dir = ctx[3]["in_dir"] or ""
        li, n = label_info(path), v["total"]
        cur = verify_state(date, dev)
        # 이 장비의 줄별 결과가 지금 로그 그대로면 줄 순서대로 쓴다 (예전 형식이면 내용 지문만 붙임)
        aligned = mode == "new" and cur.get("status") == "done" and frame.exists() and len(line_index(frame)) == n
        old_in = cur.get("frame_in_dir") or ""
        known, files = _known_rows(date, dev, in_dir) if mode == "new" and not aligned else ({}, [])
        if aligned or not known:
            v["todo"] = 0 if aligned else n
        else:                                           # 새로 확인할 줄 수 (남은 시간 계산)
            todo = 0
            for s in range(0, n, 2000):
                if v.get("stop"):
                    raise _Stop("중지 요청")
                idx = list(range(s, min(n, s + 2000)))
                metas = read_label_lines(li[4], idx) if li else [{}] * len(idx)
                for line, meta in zip(read_lines(path, idx), metas):
                    f = split_log(line)
                    todo += _row_key(*((f[6], f[7]) if len(f) == 8 else (line, "")), meta) not in known
            v["todo"] = todo
        srcs = [open(p, "rb") for p in files]
        codes, summ, feats, labels, ins = [], {}, {}, {"anomalies": 0, "normal": 0}, {}
        v["eta_s"] = round(v["todo"] * v["ms"] / 1000 + (n - v["todo"]) * REUSE_SEC)
        v["phase"], t_run = "run", time.monotonic()
        win = [(t_run, 0)]
        out = open(tmp, "w", encoding="utf-8", newline="\n")
        for s in range(0, n, 200):
            st = path.stat()
            if v.get("stop") or st.st_mtime != v["mtime"] or st.st_size != v["size"]:
                raise _Stop("중지 요청" if v.get("stop") else "검증 중에 로그가 다시 만들어짐 (새 생성 실행)")
            idx = list(range(s, min(n, s + 200)))
            metas = read_label_lines(li[4], idx) if li else [{}] * len(idx)
            olds = read_lines(frame, idx) if aligned else None
            for k, (line, meta) in enumerate(zip(read_lines(path, idx), metas)):
                f = split_log(line)
                payload, lab = (f[6], f[7]) if len(f) == 8 else (line, "")
                key = _row_key(payload, lab, meta)
                if olds is not None or key in known:    # 이미 확인한 줄: 이 장비 줄 순서대로, 아니면 내용이 같은 줄
                    if olds is not None:
                        a = json.loads(olds[k])
                    else:
                        no, pos, _ = known[key]
                        srcs[no].seek(pos)
                        a = json.loads(srcs[no].readline())
                    a.pop("k", None)
                    row_in = a.pop("in", None) or old_in
                    if a.get("cons") == "BAD" and a.get("lv", 1) < VERIFY_LOGIC:   # 판정 규칙이 바뀐 뒤 '불일치' 는 다시 확인
                        a, row_in = analyze(payload, lab, meta, ctx), in_dir
                        v["analyzed"] += 1
                    else:
                        v["reused"] += 1
                else:
                    a, row_in = analyze(payload, lab, meta, ctx), in_dir
                    v["analyzed"] += 1
                out.write(json.dumps({"k": key, "in": row_in, **a}, ensure_ascii=False) + "\n")   # 데이터 프레임·내려받기가 다시 씀
                ins[row_in] = ins.get(row_in, 0) + 1
                codes.append(CODE_NUM[a["cons"]])
                v["done"] += 1
                summ[a["cons"]] = summ.get(a["cons"], 0) + 1
                if lab in labels:                       # 피처별로 켜진 줄 수 (피처 페이지)
                    labels[lab] += 1
                    for name in {feature_key(x["name"]) for x in a["features"]}:
                        feats.setdefault(name, [0, 0])[0 if lab == "anomalies" else 1] += 1
            # 남은 시간: 최근 1분 동안 새로 확인한 속도 (20줄이 안 되면 지난 검증 기록)
            now = time.monotonic()
            win.append((now, v["analyzed"]))
            while len(win) > 2 and now - win[0][0] > 60:
                win.pop(0)
            got = win[-1][1] - win[0][1]
            per = (win[-1][0] - win[0][0]) / got if got >= 20 else v["ms"] / 1000
            left = max(0, v["todo"] - v["analyzed"])
            v["eta_s"] = round(left * per + max(0, n - v["done"] - left) * REUSE_SEC)
        out.close()
        for f in srcs:
            f.close()
        srcs = []
        for attempt in range(1000):                     # 내려받기·같은 날짜의 다른 검증이 이 파일을 열고 있으면 (Windows) 기다렸다 다시
            try:
                tmp.replace(frame)
                break
            except PermissionError:
                busy = any(x.get("status") == "running" for (d2, _), x in list(_verify.items()) if d2 == date and x is not v)
                if attempt >= 4 and not busy:
                    raise
                time.sleep(2 if busy else 1)
        try:
            part.unlink()                               # 중지할 때 저장한 결과는 이제 줄별 결과에 다 들어감
        except OSError:
            pass
        ms = 1000 * (time.monotonic() - t_run) / v["analyzed"] if v["analyzed"] >= 200 else v["ms"]
        v.update(status="done", codes="".join(codes), summary=summ, features=feats, labels=labels, frame_in_dir=in_dir,
                 rule_inputs=ins, ms_per_line=round(ms, 2), finished_at=RT.now_iso(), phase=None, eta_s=None)
        _verify_file(date, dev).write_text(json.dumps(
            {k: v[k] for k in ("mtime", "size", "total", "mode", "analyzed", "reused", "ms_per_line", "codes", "summary",
                               "features", "labels", "frame_in_dir", "rule_inputs", "started_at", "finished_at", "logic")},
            ensure_ascii=False), encoding="utf-8")
    except _Stop as e:
        saved = _keep_partial(out, srcs, tmp, part, v["done"])
        with _lock:
            if _verify.get((date, dev)) is v:
                del _verify[(date, dev)]
        _stopped[(date, dev)] = {"at": RT.now_iso(), "saved": saved, "why": str(e)}
    except Exception as e:
        _keep_partial(out, srcs, tmp, part, v["done"])
        v.update(status="error", error=f"{type(e).__name__}: {e}", phase=None, eta_s=None)


def _keep_partial(out, srcs, tmp, part, done):
    """멈춘 검증의 임시 파일 정리. 확인한 줄이 있으면 '검증 안 한 데이터만' 이 이어서 쓰도록 저장 (예전에 저장한 것도 함께)."""
    for f in [out] + srcs:
        if f:
            f.close()
    saved = 0
    try:
        if done and tmp.exists():
            if part.exists():
                with open(tmp, "ab") as w, open(part, "rb") as r:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        w.write(block)
            tmp.replace(part)
            saved = done
    except OSError:
        pass
    try:
        tmp.unlink()
    except OSError:
        pass
    return saved


def frame_cache(date, dev):
    """검증 때 저장한 줄별 확인 결과 파일 (로그가 그대로일 때만, 아니면 None). 줄마다 확인한 룰 입력 날짜(in)가 있다."""
    v = verify_state(date, dev)
    p = _frame_file(date, dev)
    if v.get("status") != "done" or not p.exists():
        return None
    return p if len(line_index(p)) == v.get("total") else None


def cached_rows(cache, idx):
    """줄 번호들의 저장된 확인 결과 (analyze 와 같은 모양)."""
    rows = [json.loads(x) for x in read_lines(cache, idx)]
    for a in rows:
        a.pop("k", None)
        a.pop("in", None)
    return rows


def _est_new(date, dev, v, n):
    """'검증 안 한 데이터만' 이 새로 확인할 줄 수 (대략 - 시작하면 정확히 셈) = 전체 - 이미 확인한 줄 (+ 예전 판정 규칙의 '불일치')."""
    if v.get("status") == "done":
        return _stale_bad(v)
    known = 0
    for d in (dev, _other(dev)):
        for p in (_frame_file(date, d), _partial_file(date, d)):
            try:
                with open(p, "rb") as f:
                    m = ROW_HEAD.match(f.readline())
                if m:
                    known = max(known, len(line_index(p)))
            except OSError:
                pass
    return max(0, n - known)


def verify_summary(date, dev):
    dates = [d["date"] for d in dataset_dates()] if date == "all" else [date]
    latest = (input_dates() or [""])[-1]
    out = {"status": "done", "done": 0, "total": 0, "summary": {}, "names": CONS, "finished_at": "", "reused": 0,
           "todo": 0, "analyzed": 0, "eta_s": None, "est_new": 0, "est_new_s": 0.0, "est_all_s": 0.0,
           "rule_inputs": {}, "latest_input": latest}
    run_etas, wait_etas, phases = [], [], set()
    for d in dates:
        v = verify_state(d, dev)
        st = v.get("status")
        n = v.get("total", 0) or len(line_index(_log_path(d, dev)))
        ms = _ms_per_line(d, dev)
        out["total"] += n
        out["est_all_s"] += n * ms / 1000
        if st == "running":
            out["status"], out["mode"] = "running", v.get("mode")
            out["stopping"] = out.get("stopping") or bool(v.get("stop"))
            phases.add(v.get("phase"))
            for k in ("done", "analyzed", "reused", "todo"):
                out[k] += v.get(k) or 0
            if v.get("phase") == "wait":                # IPS 가 끝난 뒤 그 결과를 이어받는 시간
                wait_etas.append((_verify.get((d, "ips"), {}).get("eta_s") or 0) + n * REUSE_SEC)
            elif v.get("eta_s") is not None:
                run_etas.append(v["eta_s"])
            continue
        if st == "done":
            out["done"] += n
            out["reused"] += v.get("reused", 0)
            for k, c in v.get("summary", {}).items():
                out["summary"][k] = out["summary"].get(k, 0) + c
            out["finished_at"] = max(out["finished_at"], v.get("finished_at", ""))
            ri = v.get("rule_inputs") or ({v["frame_in_dir"]: n} if v.get("frame_in_dir") else {})
            for k, c in ri.items():                     # 줄마다 확인한 룰 입력 날짜
                out["rule_inputs"][k] = out["rule_inputs"].get(k, 0) + c
        elif out["status"] != "running":
            out["status"] = "error" if st == "error" else "none"
            out["error"] = v.get("error")
        new = _est_new(d, dev, v, n)
        out["est_new"] += new
        out["est_new_s"] += new * ms / 1000
        if (d, dev) in _stopped:
            out["stopped"] = _stopped[(d, dev)]
    if run_etas or wait_etas:
        # 여러 날짜를 함께 검증하면 한 프로세스를 나눠 쓰므로 남은 시간의 합 ÷ 동시에 도는 수
        out["eta_s"] = round(max([sum(run_etas) / max(1, len(run_etas))] + wait_etas))
    out["old_rules"] = sum(c for k, c in out["rule_inputs"].items() if k != latest)
    out["phase"] = ("run" if "run" in phases else "prep" if phases & {"prep", "start"} else "wait" if phases else None)
    # 저장된 줄별 결과가 모두 있으면 데이터 프레임 내려받기가 바로 된다
    out["frame_ready"] = out["status"] == "done" and all(frame_cache(d, dev) is not None for d in dates)
    return out


# ── 학습데이터 수정·삭제 (화면에서 저장 → 1.4 가 학습 로그를 만들 때마다 적용) ──────────────
# 08_dataset/<날짜>/edits.json = {sk: {"action": "delete" | "edit", "label", "http", "id", "orig_label", "why", "at"}}
# sk = 1.4 가 label 파일에 적는 샘플 지문 (IPS·WAF 의 같은 샘플은 같은 sk → 두 로그에 함께 적용).
# LLM 출력(07)은 그대로 두므로 되돌리면 원래 샘플로 돌아온다.

_edit_lock = threading.Lock()
_ds = {"M": None}


def dataset_mod():
    """1.4 (학습 로그 변환) 모듈 - 고친 payload 를 학습 로그와 똑같이 한 줄로 만들려고."""
    if _ds["M"] is None:
        _ds["M"] = _load("dataset_ui", "1_4_build_dataset.py")
    return _ds["M"]


def _edits_file(date):
    return DATASET_DIR / date / "edits.json"


def read_edits(date):
    try:
        return json.loads(_edits_file(date).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_edits(date, edits):
    p = _edits_file(date)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(5):                            # 1.4 가 잠깐 읽고 있으면 (Windows) 조금 뒤 다시
        try:
            tmp.replace(p)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def _raw_http(text):
    """화면에서 고친 payload (줄바꿈은 실제 줄바꿈, 제어문자는 '\\x00' 표기) → LLM 출력과 같은 모양의 http."""
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return re.sub(r"\\x(0[0-9b-f]|1[0-9a-f]|7f)", lambda m: chr(int(m.group(1), 16)), s) + "\n"


def _line_meta(date, dev, line):
    """학습 로그의 한 줄과 그 줄의 label 정보."""
    path = _log_path(date, dev)
    li = label_info(path)
    if li is None or not 0 <= line < len(line_index(path)):
        raise ValueError("line")
    return read_lines(path, [line])[0], read_label_lines(li[4], [line])[0]


def check_edit(b):
    """고친 label·payload 를 학습 로그와 똑같이 한 줄로 만들어 다시 검증 (저장하지 않음)."""
    text, meta = _line_meta(b.get("date", ""), b.get("dev", "ips"), int(b.get("line", -1)))
    f = split_log(text)
    label = b.get("label") if b.get("label") in ("anomalies", "normal") else (f[7] if len(f) == 8 else "")
    payload = f[6] if len(f) == 8 else text
    if b.get("payload") is not None:
        M = dataset_mod()
        st, _ = M.load_mapping()
        payload = M.one_line(M.normalize_http(_raw_http(b["payload"])), st)
        if M.SEP in payload:
            raise ValueError("payload 에 구분자 '|||' 를 쓸 수 없습니다")
    return dict(analyze(payload, label, dict(meta, label=label)), payload=payload, label=label)


def save_edit(b):
    """한 줄(샘플)의 수정·삭제 저장. action: edit | delete | undo. sk 를 주면 줄 대신 그 샘플 (목록의 되돌리기)."""
    date, act, sk = b.get("date", ""), b.get("action"), b.get("sk")
    if not DATE_RE.match(date) or act not in ("edit", "delete", "undo"):
        raise ValueError("date/action")
    text, meta = ("", {}) if sk else _line_meta(date, b.get("dev", "ips"), int(b.get("line", -1)))
    sk = sk or meta.get("sk")
    if not sk:
        raise ValueError("이 날짜 학습 로그는 예전 형식이라 고칠 수 없습니다 - 먼저 '학습 로그에 반영'으로 다시 만드세요")
    with _edit_lock:
        edits = read_edits(date)
        if act == "undo":
            edits.pop(sk, None)
        else:
            e = {"action": act, "id": meta.get("id", ""), "orig_label": meta.get("label", ""),
                 "why": str(b.get("why") or "")[:300], "at": RT.now_iso()}
            if act == "edit":
                if b.get("label") in ("anomalies", "normal") and b["label"] != meta.get("label"):
                    e["label"] = b["label"]
                if b.get("payload") is not None:
                    if "|||" in b["payload"]:
                        raise ValueError("payload 에 구분자 '|||' 를 쓸 수 없습니다")
                    e["http"] = _raw_http(b["payload"])
                if "label" not in e and "http" not in e:
                    raise ValueError("바뀐 것이 없습니다")
            edits[sk] = e
        _write_edits(date, edits)
    return {"ok": True, "saved": len(edits)}


def delete_bad(date, dev):
    """검증에서 '불일치' 인 줄을 모두 삭제로 저장 (사유 = 불일치 사유)."""
    n = 0
    for d in ([x["date"] for x in dataset_dates()] if date == "all" else [date]):
        v = verify_state(d, dev)
        idx = [i for i, c in enumerate(v.get("codes", "")) if c == CODE_NUM["BAD"]] if v.get("status") == "done" else []
        if not idx:
            continue
        path = _log_path(d, dev)
        metas = read_label_lines(label_info(path)[4], idx)
        cache = frame_cache(d, dev)
        why = [r.get("cons_reason", "") for r in cached_rows(cache, idx)] if cache else [""] * len(idx)
        with _edit_lock:
            edits = read_edits(d)
            for meta, w in zip(metas, why):
                if meta.get("sk"):
                    edits[meta["sk"]] = {"action": "delete", "id": meta.get("id", ""), "orig_label": meta.get("label", ""),
                                         "why": ("불일치: " + w)[:300], "at": RT.now_iso()}
                    n += 1
            _write_edits(d, edits)
    return {"deleted": n}


def dataset_job():
    """화면에서 시작한 '학습 로그에 반영'(1.4) 상태."""
    rec = _ui_record("dataset")
    running = bool(rec.get("pid")) and RT.pid_alive(rec["pid"]) and time.time() - rec.get("t", 0) < 3600
    out = {"running": running, "started_at": rec.get("started_at", "")}
    if rec.get("out") and not running:
        txt = tail(RT.RUN_DIR / rec["out"], 40)
        out["result"] = "fail" if ("Traceback" in txt or "[ERROR]" in txt) else "done" if "[데이터셋]" in txt else ""
    return out


def edits_summary(date):
    """저장한 수정·삭제와 학습 로그 반영 여부 (화면)."""
    dates = [d["date"] for d in dataset_dates()] if date == "all" else [date]
    out = {"saved": 0, "deleted": 0, "edited": 0, "pending": False, "old_format": [], "items": [], "rebuild": dataset_job()}
    for d in dates:
        edits = read_edits(d)
        try:
            summ = json.loads((DATASET_DIR / d / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summ = {}
        try:
            mt = _edits_file(d).stat().st_mtime
        except OSError:
            mt = 0
        out["saved"] += len(edits)
        out["deleted"] += sum(1 for e in edits.values() if e.get("action") == "delete")
        out["edited"] += sum(1 for e in edits.values() if e.get("action") == "edit")
        out["pending"] |= summ.get("edits_mtime", 0) != mt        # 저장한 뒤 아직 학습 로그를 다시 만들지 않음
        if summ.get("format", 1) < 2:
            out["old_format"].append(d)
        for sk, e in edits.items():
            out["items"].append({"date": d, "sk": sk, "action": e.get("action"), "id": e.get("id", ""),
                                 "orig_label": e.get("orig_label", ""), "label": e.get("label", ""),
                                 "payload": bool(e.get("http")), "why": e.get("why", ""), "at": e.get("at", "")})
    out["items"] = sorted(out["items"], key=lambda e: e["at"], reverse=True)[:300]
    return out


def start_rebuild(date):
    """'학습 로그에 반영': 1.4 로 그 날짜 학습 로그를 다시 만든다 (API 비용 없음)."""
    if date != "all" and date not in [d["date"] for d in dataset_dates()]:
        raise ValueError("date")
    for job in ("generation", "collect"):
        if job_state(job)["running"]:
            raise ValueError("학습데이터 생성·수집이 도는 중입니다 - 끝날 때 그 작업의 1.4 가 수정·삭제를 함께 반영합니다")
    if dataset_job()["running"]:
        raise ValueError("이미 학습 로그를 다시 만드는 중입니다")
    if any(v.get("status") == "running" for (d, _), v in list(_verify.items()) if date in ("all", d)):
        raise ValueError("검증이 도는 중입니다 - 끝나거나 중지한 뒤에 반영하세요")
    return {"pid": spawn("dataset", [] if date == "all" else ["--date", date, "--force"])}


def data_query(q):
    """신규 학습 데이터 화면의 조건 → (날짜 목록, 장비, label, 룰 id, payload 검색어, label 정합성, 보류 대상)."""
    dev, date = _q(q, "dev", "ips"), _q(q, "date")
    dates = [d["date"] for d in dataset_dates()]
    if date and date != "all":
        if date not in dates:
            raise ValueError("date")
        dates = [date]
    label, rule, text, cons = _q(q, "label"), _q(q, "rule").strip(), _q(q, "q").strip(), _q(q, "cons")
    hold = _q(q, "hold")
    if (dev not in ("ips", "waf", *CONV) or label not in ("", "anomalies", "normal") or cons not in ("", *CONS)
            or hold not in ("", *HOLD)):
        raise ValueError("dev/label/cons/hold")
    return dates, dev, label, rule, text, cons, hold


FRAME_COLS = ["index", "payload", "method", "label", "label 사유", "검증 결과", "label 정합성", "label 정합성 사유",
              "탐지 피처", "rule_id", "date"]
CHECK_MARK = {"ok": "✔", "warn": "△", "bad": "✘"}


def _cell(v):
    """엑셀이 수식으로 읽지 않게 (=, +, -, @ 로 시작하는 값 앞에 ')."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def frame_csv(q):
    """데이터 프레임 내려받기: 조건에 맞는 모든 줄을 화면의 열 그대로 CSV(UTF-8 BOM, 엑셀용)로.
    → (줄 수, 파일 이름, 바이트 조각 생성기). '전체 검증'의 저장된 줄별 결과가 있으면 그것을 쓰고(빠름),
    없으면 줄마다 다시 확인한다(1천 줄에 약 15초). 200줄씩 만들어 바로 보낸다."""
    dates, dev, label, rule, text, cons, hold = data_query(q)
    src = "ips" if dev in CONV else dev                 # FW·웹: 원본 IPS 줄을 골라 바꿔서 (conv_csv)
    if dev in CONV:
        conv_verified(dates, cons, hold)
    parts = []
    for d in dates:                                     # 조건 오류(예: 전체 검증 전 정합성 필터)는 보내기 전에
        m = matching(d, src, label, rule, text, cons, hold)
        parts.append((d, m, len(line_index(_log_path(d, src))) if m is None else len(m)))
    total = sum(n for _, _, n in parts)
    name = "_".join([dev, dates[0] if len(dates) == 1 else "all", "frame"] +
                    [x for x in (label, cons, hold and f"hold-{hold}") if x]) + ".csv"
    if dev in CONV:
        return total, name, conv_csv(dev, parts)

    def chunks():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(FRAME_COLS)
        yield ("﻿" + buf.getvalue()).encode("utf-8")
        g0 = 0
        for d, m, n in parts:
            path = _log_path(d, dev)
            li = label_info(path)
            cache = frame_cache(d, dev)
            for s in range(0, n, 200):
                idx = list(range(s, min(n, s + 200))) if m is None else m[s:s + 200]
                metas = read_label_lines(li[4], idx) if li else [{}] * len(idx)
                saved = cached_rows(cache, idx) if cache else None
                buf.seek(0)
                buf.truncate()
                for k, (line, meta) in enumerate(zip(read_lines(path, idx), metas)):
                    f = split_log(line)
                    payload, method, lab = (f[6], f[5], f[7]) if len(f) == 8 else (line, "", "")
                    try:
                        a = saved[k] if saved else analyze(payload, lab, meta)
                    except Exception as e:              # 한 줄이 실패해도 파일은 끝까지
                        a = {"reason": "", "checks": [("bad", f"확인 실패: {type(e).__name__}: {e}")], "cons_text": "",
                             "cons_reason": "", "features": []}
                    w.writerow([g0 + s + k] + [_cell(v) for v in (
                        payload, method, lab, a["reason"],
                        " / ".join(f"{CHECK_MARK.get(st, '')} {t}" for st, t in a["checks"]),
                        a["cons_text"], a["cons_reason"],
                        "; ".join(f"{x['name']}={x['value']}" + (f" ({x['desc']})" if x.get("desc") else "")
                                  for x in a["features"]),
                        meta.get("id", ""), d)])
                yield buf.getvalue().encode("utf-8")
            g0 += n
    return total, name, chunks()


def data_rows(q):
    """신규 학습 데이터 표 한 페이지 (데이터 프레임 열 + 수정·삭제 상태)."""
    dates, dev, label, rule, text, cons, hold = data_query(q)
    offset, limit = page_args(q, 100)
    if dev in CONV:
        return conv_page(dates, dev, label, rule, text, cons, hold, offset, limit)
    total, picked = select_rows(dates, dev, label, rule, text, offset, limit, cons, hold)
    rows = []
    by_date = {}
    for d, i, g in picked:
        by_date.setdefault(d, []).append((i, g))
    for d, items in by_date.items():
        path = _log_path(d, dev)
        lines = read_lines(path, [i for i, _ in items])
        li = label_info(path)
        metas = read_label_lines(li[4], [i for i, _ in items]) if li else [{}] * len(items)
        cache = frame_cache(d, dev)
        saved = cached_rows(cache, [i for i, _ in items]) if cache else None    # '전체 검증' 결과가 있으면 그대로
        edits = read_edits(d)
        for k, ((i, g), line, meta) in enumerate(zip(items, lines, metas)):
            f = split_log(line)
            payload, method, lab = (f[6], f[5], f[7]) if len(f) == 8 else (line, "", "")
            a = saved[k] if saved else analyze(payload, lab, meta)
            e = edits.get(meta.get("sk")) or {}
            rows.append(dict(a, index=g, line=i, date=d, payload=payload, method=method, label=lab, rule_id=meta.get("id", ""),
                             sk=meta.get("sk", ""), edited=meta.get("edited", ""),
                             edit={"action": e["action"], "label": e.get("label", ""), "payload": bool(e.get("http"))} if e else None))
    rows.sort(key=lambda r: r["index"])
    return {"total": total, "offset": offset, "rows": rows}


# 장비 FW·웹 보기: IPS 신규 학습 데이터 줄을 1_4a 로 바꿔 보여 주고, '파일로 저장'은 날짜 전체를 1_4a.save 로 쓴다. 조건(label·룰·검색·정합성·보류)은
# 원본 IPS 줄에 적용하고 검증·수정·삭제도 원본(IPS) 기준. 형식·환경 값은 docs/코드참조/FW_웹로그_형식.md
CONV = {"fw": "FW", "web": "웹"}
_conv = {}


def conv_mod():
    with _lock:
        if "M" not in _conv:
            if not (BASE / "1_4a_fw_web_log.py").exists():
                raise ValueError("FW·웹 로그 변환 파일(1_4a_fw_web_log.py)이 없습니다")
            _conv["M"] = _load("fw_web_ui", "1_4a_fw_web_log.py")
    return _conv["M"]


def conv_lines(date, idx):
    """원본 IPS 로그의 줄과 label 메타."""
    path = _log_path(date, "ips")
    li = label_info(path)
    return read_lines(path, idx), (read_label_lines(li[4], idx) if li else [{}] * len(idx))


def conv_verified(dates, cons, hold):
    """정합성 조건은 원본 '전체 검증' 결과가 있어야 함 (matching 과 같은 확인, 화면 문구에 원본 장비 이름 없이)."""
    if cons or hold == "rule":
        for d in dates:
            if verify_state(d, "ips").get("status") != "done":
                raise ValueError(f"{d} 데이터는 '전체 검증'을 먼저 실행하세요")


def conv_page(dates, dev, label, rule, text, cons, hold, offset, limit):
    """FW·웹 보기 한 페이지: 줄마다 변환 로그·파싱 결과·공격 근거 (웹은 룰 조건으로 근거가 남는지 확인)."""
    M = conv_mod()
    C, rules, _ = checker()
    conv_verified(dates, cons, hold)
    total, picked = select_rows(dates, "ips", label, rule, text, offset, limit, cons, hold)
    rows = []
    for d in dict.fromkeys(p[0] for p in picked):
        items = [(i, g) for dd, i, g in picked if dd == d]
        for (i, g), line, meta in zip(items, *conv_lines(d, [i for i, _ in items])):
            rows.append(dict(M.convert(dev, line, meta, (C, rules)), index=g, line=i, date=d))
    return {"total": total, "offset": offset, "rows": rows, "source": M.settings()["source"]}


def conv_csv(dev, parts):
    """FW·웹 보기 CSV: 변환 로그 + 파싱 필드(이름순) + 공격 근거 + 원본 payload. 200줄씩 만들어 바로 보낸다
    (변환만 해서 1천 줄에 약 1초, 웹은 룰 조건 대조 포함)."""
    M = conv_mod()                                      # 변환 파일이 없으면 보내기 전에 오류
    C, rules, _ = checker()
    head, names = M.csv_columns(dev)

    def chunks():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(head)
        yield ("﻿" + buf.getvalue()).encode("utf-8")
        g0 = 0
        for d, m, n in parts:
            for s in range(0, n, 200):
                idx = list(range(s, min(n, s + 200))) if m is None else m[s:s + 200]
                buf.seek(0)
                buf.truncate()
                for k, (line, meta) in enumerate(zip(*conv_lines(d, idx))):
                    row = M.convert(dev, line, meta, (C, rules))     # '-' 하나(빈 값 표기)는 수식이 아니라 그대로
                    w.writerow([v if v == "-" else _cell(v) for v in M.csv_values(dev, row, g0 + s + k, d, names)])
                yield buf.getvalue().encode("utf-8")
            g0 += n
    return chunks()


_conv_save = {}      # 화면의 '파일로 저장' (한 번에 하나): status·dev·dates·date·done·total·saved·error


def conv_saved(date, dev):
    """저장 파일 이름 + 저장했으면 크기·시각과 stale (저장한 뒤 원본 학습 로그나 형식 md 가 바뀜)."""
    M = conv_mod()
    log, lab = M.out_files(date, dev)
    out = {"name": f"08_dataset/{date}/{log.name}", "label": lab.name}
    try:
        st = log.stat()
    except OSError:
        return out
    ips = M.ips_files(date)
    src = [p for p in ((ips[0] if ips else None), M.SETTINGS_FILE) if p and p.exists()]
    out.update(mb=round(st.st_size / 1e6, 1), saved_at=time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
               stale=any(p.stat().st_mtime > st.st_mtime for p in src))
    return out


def conv_save_state(date, dev):
    """'파일로 저장' 작업 상태 + 고른 날짜(전체면 모든 날짜)의 저장 파일."""
    dates = [d["date"] for d in dataset_dates()]
    if dev not in CONV or (date != "all" and date not in dates):
        raise ValueError("date/dev")
    return {"job": dict(_conv_save), "files": {d: conv_saved(d, dev) for d in (dates if date == "all" else [date])}}


def conv_save_start(b):
    """'파일로 저장': 고른 날짜(전체면 모든 날짜)의 데이터 전체를 1_4a.save 로 (화면 안 스레드, 하루 약 11만 줄에 FW 30초·웹 1분)."""
    dev, date = b.get("dev", ""), b.get("date", "")
    dates = [d["date"] for d in dataset_dates()]
    todo = dates if date == "all" else [date]
    if dev not in CONV or not todo or not set(todo) <= set(dates):
        raise ValueError("date/dev")
    if dataset_job()["running"]:
        raise ValueError("학습 로그를 다시 만드는 중입니다 - 끝난 뒤에 저장하세요")
    M = conv_mod()
    with _lock:
        if _conv_save.get("status") == "running":
            raise ValueError("이미 파일을 저장하는 중입니다")
        _conv_save.clear()
        _conv_save.update(status="running", dev=dev, dates=todo, date=todo[0], done=0, total=0, saved=[],
                          started_at=RT.now_iso())
    threading.Thread(target=_conv_save_work, args=(M, dev, todo), daemon=True).start()
    return {"started": todo}


def _conv_save_work(M, dev, todo):
    s = _conv_save
    try:
        C, rules, _ = checker()
        for d in todo:
            s.update(date=d, done=0, total=0)
            _, n, total = M.save(d, dev, (C, rules), lambda k, t: s.update(done=k, total=t))
            s["saved"].append({"date": d, "lines": n, "skipped": total - n})
        s.update(status="done", finished_at=RT.now_iso())
    except Exception as e:                              # 사유를 화면에
        s.update(status="error", error=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}",
                 finished_at=RT.now_iso())


_label_offsets = {}


def read_label_lines(lab_path, idx_list):
    st = lab_path.stat()
    hit = _label_offsets.get(lab_path)
    if not hit or hit[0] != st.st_mtime:
        offs, pos = array("q"), 0
        with open(lab_path, "rb") as f:
            for line in f:
                if line.strip():
                    offs.append(pos)
                pos += len(line)
        hit = _label_offsets[lab_path] = (st.st_mtime, offs)
    out = []
    with open(lab_path, "rb") as f:
        for i in idx_list:
            if i >= len(hit[1]):
                out.append({})
                continue
            f.seek(hit[1][i])
            try:
                out.append(json.loads(f.readline().decode("utf-8")))
            except ValueError:
                out.append({})
    return out


# ── 6. 기존 학습데이터와 비교 (5.x: 비교·확정은 5_0_run_supervised.py 별도 프로세스, 화면은 결과·선별만) ──
# 선별 미리 보기는 확정(5.7 finalize)과 같은 select() 를 쓴다 → 사람이 확인한 수 그대로 내보냄.

_sup = {"C": None, "V": None, "tables": {}}


def sup_mods():
    with _lock:
        if _sup["C"] is None:
            _sup["C"] = _load("sup_common_ui", "5_0a_common.py")
            _sup["V"] = _load("sup_versions_ui", "5_7_versions.py")
    return _sup["C"], _sup["V"]


def sup_table(base, dev):
    """비교 결과 표 (5.5 <장비>_table.json, 파일이 바뀌면 다시 읽음). 없으면 None."""
    C, _ = sup_mods()
    p = C.compared_files(base, dev)["table"]
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _sup["tables"].get((base, dev))
    if hit and hit[0] == mt:
        return hit[1]
    t = json.loads(p.read_text(encoding="utf-8"))
    with _lock:
        _sup["tables"] = {k: x for k, x in _sup["tables"].items() if k[0] == base}     # 지금 기준 버전 것만 메모리에
        _sup["tables"][(base, dev)] = (mt, t)
    return t


def sup_status():
    """비교·확정 상태 파일. 실행기가 2초마다 바꿔 쓰는 순간에 읽으면 비어 올 수 있어 잠깐 뒤 다시 읽는다."""
    for _ in range(3):
        st = RT.read_status("supervised")
        if st or not (RT.RUN_DIR / "supervised.json").exists():
            return st
        time.sleep(0.05)
    return st


def _sup_base(q_base):
    C, _ = sup_mods()
    v = C.load_versions()
    finals = [x["name"] for x in v["versions"] if x.get("status") == "final"]
    if q_base and q_base not in finals:
        raise ValueError("기준 버전")
    return v, finals, (q_base or (finals[-1] if finals else ""))


def sup_state(q_base=""):
    """화면 6: 버전 목록, 실행 상태, 넣을 것, 장비별 비교 결과 상태(그 뒤 바뀐 날짜), 새로 비교할 양 추정,
    비교 예상 시간 (plans: 실행기가 처음 만드는 실행 계획과 같은 계산 = 5.0a compare_plan)."""
    C, V = sup_mods()
    v, finals, base = _sup_base(q_base)
    timing = C.read_json(C.CACHE / "timing.json", {})
    rows_by_date = {d["date"]: d["events"] for d in dataset_dates()}
    out = {"versions": V.version_rows(), "finals": finals, "base": base, "job": job_state("supervised"),
           "status": sup_status(), "problems": C.setup_problems(v, base or None),
           "feature_code": v.get("feature_code"), "defaults": C.default_sources(), "dates": sorted(rows_by_date),
           "disk_free_gb": round(C.disk_free_gb(), 1), "timing": timing, "dev": C.compare_estimate(base, rows_by_date),
           "options": V.DEFAULT_OPTIONS, "option_text": V.OPTION_TEXT, "grades": C.GRADE_TEXT, "plans": {}}
    for dev in C.DEVICES:
        out["dev"][dev]["rate"] = timing.get(f"feat_rows_per_sec_{dev}")
    for mode in ("new", "all") if base else ():
        for inc in (False, True):                       # ' 이미 비교한 데이터도 다시 비교' 체크 여부
            plan = C.compare_plan(base, C.DEVICES, mode, inc, timing, out["dev"])
            out["plans"][mode + ("_inc" if inc else "")] = {
                "total_s": round(sum(p["expected_s"] for p in plan)),
                "dev": {d: {"s": round(sum(p["expected_s"] for p in plan if p["dev"] == d)),
                            "new": next((p.get("n", 0) for p in plan if p["key"] == f"{d}:feat"), 0)} for d in C.DEVICES}}
    return out


def sup_summary(q):
    C, _ = sup_mods()
    _, _, base = _sup_base(_q(q, "base"))
    dev = _q(q, "dev", "ips")
    if dev not in C.DEVICES:
        raise ValueError("dev")
    s = C.read_json(C.compared_files(base, dev)["summary"], None) if base else None
    return s or {}


def sup_combined(base):
    """IPS·WAF 비교 표를 샘플(sk)로 짝지은 공통 선별 표 (5.7 combine, 두 표가 바뀌면 다시). 하나라도 없으면 None."""
    C, V = sup_mods()
    ti, tw = sup_table(base, "ips"), sup_table(base, "waf")
    if ti is None or tw is None:
        return None
    key = (base, id(ti), id(tw))                        # sup_table 은 파일이 바뀔 때만 새 표를 돌려준다
    hit = _sup.get("comb")
    if hit and hit[0] == key:
        return hit[1]
    tc = V.combine(ti, tw)
    with _lock:
        _sup["comb"] = (key, tc)
    return tc


def sup_preview(b):
    """선별 미리 보기: IPS·WAF 공통 선별 (두 장비에 같은 샘플, 장비 선택과 상관없음)."""
    C, V = sup_mods()
    _, _, base = _sup_base(b.get("base", ""))
    tc = sup_combined(base)
    if tc is None:
        return {"none": True, "missing": [d.upper() for d in C.DEVICES if sup_table(base, d) is None]}
    return V.preview(tc, b.get("conditions"), b.get("options"), b.get("overrides"), b.get("view", ""))


def sup_save(b):
    C, V = sup_mods()
    _, _, base = _sup_base(b.get("base", ""))
    if job_state("supervised")["running"]:
        raise ValueError("비교·확정이 도는 중입니다 - 끝난 뒤에 담으세요")
    tables = {}
    for dev in C.DEVICES:
        t = sup_table(base, dev)
        if t is None:
            raise ValueError(f"{dev.upper()} 비교 결과가 없습니다 - IPS·WAF 둘 다 비교한 뒤 담으세요")
        tables[dev] = t
    return V.save_draft(b.get("name", ""), base, b.get("conditions"), b.get("options"), b.get("overrides"), tables)


def sup_start(kind, b):
    """비교 또는 확정을 별도 프로세스로 시작."""
    C, V = sup_mods()
    if job_state("supervised")["running"]:
        raise ValueError("비교·확정이 이미 도는 중입니다")
    if kind == "compare":
        v, _, base = _sup_base(b.get("base", ""))
        devs = [d for d in (b.get("devs") or list(C.DEVICES)) if d in C.DEVICES]
        problems = C.setup_problems(v, base or None, devs)
        if not base or problems:
            raise ValueError("; ".join(problems) or "기준 버전이 없습니다")
        args = ["compare", "--base", base, "--mode", "all" if b.get("mode") == "all" else "new"]
        for d in devs:
            args += ["--dev", d]
        if b.get("include_compared"):
            args.append("--include-compared")
    else:
        v = C.load_versions()
        name = b.get("name", "")
        if C.get_version(v, name).get("status") != "draft":
            raise ValueError("초안만 확정할 수 있습니다")
        args = ["finalize", "--version", name]
    RT.clear_stop("supervised")
    return {"pid": spawn("supervised", args), "args": args}


def sup_candidates_csv(q):
    """선별 후보 CSV (사람 검토용): IPS·WAF 공통 선별을 조건·옵션(spec JSON 또는 저장한 초안) 그대로 고른 샘플 + payload.
    두 장비에 같은 샘플이 들어가므로 한 파일 (줄 번호·등급·method 는 장비별 열).
    view = rule | pyua 면 고른 샘플 대신 그 보류 대상 전부 (지금 선별에 들어갔는지 '선별' 열로)."""
    C, V = sup_mods()
    v, _, base = _sup_base(_q(q, "base"))
    view = _q(q, "view")
    if view not in ("", *V.HOLD):
        raise ValueError("view")
    if _q(q, "name"):
        spec = C.get_version(v, _q(q, "name"))
        base = spec.get("base") or base
    else:
        spec = json.loads(_q(q, "spec", "{}") or "{}")
    t = sup_combined(base)
    if t is None:
        raise ValueError("IPS·WAF 비교 결과가 둘 다 있어야 합니다")
    sel, cand, _ = V.select(t, spec.get("conditions"), spec.get("options"), spec.get("overrides"))
    on = set(sel) if view else None
    rows = V.hold_rows(t, cand, view) if view else sel
    cons_txt = {"0": "검증 전", "1": "정합", "2": "정합 (룰 기준)", "3": "확인 필요", "4": "불일치"}
    name = (f"{base}_보류_{'정합룰기준' if view == 'rule' else 'python-requests'}_{len(rows)}.csv" if view
            else f"{base}_IPS·WAF_공통후보_{len(rows)}.csv").replace(" ", "_")

    def chunks():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "IPS index", "WAF index", "rule_id", "공격 유형", "출처", "label", "등급 (공통)", "IPS 등급",
                    "WAF 등급", "근거", "label 정합성", "IPS method", "WAF method", "payload"] + (["선별"] if view else []))
        yield ("﻿" + buf.getvalue()).encode("utf-8")
        for s in range(0, len(rows), 500):
            part = rows[s:s + 500]
            by_date = {}
            for i in part:
                by_date.setdefault(t["dates"][t["date"][i]], []).append(i)
            buf.seek(0)
            buf.truncate()
            for d, idx in by_date.items():
                li = read_lines(_log_path(d, "ips"), [t["line"][i] for i in idx])
                lw = read_lines(_log_path(d, "waf"), [t["wline"][i] for i in idx])
                for i, a, b in zip(idx, li, lw):
                    f, fw = split_log(a), split_log(b)
                    fl = t["fl"][i]
                    why = ", ".join(x for x, bit in (("새 패턴", 32), ("RF 틀림", 64), ("LGBM 틀림", 128),
                                                     ("기준엔 반대 라벨만", 4), ("python-requests", 1)) if fl & bit)
                    w.writerow([d, t["line"][i], t["wline"][i], t["rules"][t["rule"][i]], t["types"][t["type"][i]],
                                t["sources"][t["src"][i]], "anomalies" if t["y"][i] else "normal", C.GRADES[t["g"][i]],
                                C.GRADES[t["gi"][i]], C.GRADES[t["gw"][i]], why, cons_txt.get(t["cons"][i], ""),
                                _cell(f[5] if len(f) == 8 else ""), _cell(fw[5] if len(fw) == 8 else ""),
                                _cell(f[6] if len(f) == 8 else a)] + (["들어감" if i in on else "빠짐"] if view else []))
            yield buf.getvalue().encode("utf-8")
    return name, chunks()


# ── 2. 검토에서 뺀 샘플 (3.3 의 _3_reviewed/*.rejected.jsonl: 검토 모델 drop · 코드 검사 · 형식) ──────────
# 검토 모델(gpt-6-luna)이 공격 샘플을 '안전하게 보존·수정할 수 없다'며 빼는 일이 많다 (10-22~26 약 9천 개).
# 날짜·검토 모델·공격 유형별로 몇 개를 왜 뺐는지 보여 준다. 분류는 이유 문구로 (검토 모델이 쓴 이유는 그대로 보여 줌).
# 다시 만들기 (2026-10-26~): 3_3 --redo-safety 가 안전 사유로 뺀(또는 거절한) 룰을 '검토 대체 모델'로 다시 검토하면 옛 결과는
#   _3_reviewed/_replaced 로 옮겨지고 새 결과가 쓰인다 → 다시 만든 샘플은 이 목록에서 빠지고 다시 검토 뒤에도 빠진 것만 남는다
#   (전·후 수는 meta.json 의 redo.before). 대체 모델이 뺀 샘플은 검토 모델을 대체 모델로 보인다 (meta.json 의 fallback.answered).
EXCL_SAFETY = re.compile(
    r"안전|위험|익스플로잇|악용|무기화|무해|피해 없|재현 가능|실행 가능|운영 가능|보존(하거나|할 수 없|해 줄 수 없)|"
    r"재현(하거나|할 수 없|용)|재구성할 수 없|보정(하지 않|할 수 없)|보강할 수 없|검증·정제할 수 없|제공할 수 없|지원할 수 없|"
    r"도움(이 될 수|을 줄 수 없|을 드릴 수 없)|승인하거나|공격 (실행|코드|입력을 만들|요청이라|요청으로 (바꿔야|수정할 수 없))|"
    r"페이로드를 재작성할 수 없|실행 페이로드|명령 (실행|주입|치환) (요청이라|페이로드|기능|공격)|콜백을 유도|다운로드 후 실행|"
    r"서비스 중단|설정 손실|DoS 트리거|(노출|권한 상승)을 유발|열거 기능을 유지|하드코(딩|드)된 (자격|인증)|파괴적 명령|"
    r"exploit|weapon|safety|safely|unsafe|harm|can(not|'t) (assist|help|provide|preserve|keep|convert)|refus", re.I)
EXCL_RULE = re.compile(r"모순|응답|response|JSON|CRLF|multipart|바이너리|REPEAT|조건 불일치|만족(할|시킬|하도록|시키려면)? ?수 없|"
                       r"within|일치하지 않|불만족|정규식|부정 조건|HTTP 형식|유효한 HTTP", re.I)
EXCL_DUP = re.compile(r"중복|다양성|동일한 내용")
EXCL_CATS = ["안전상 이유", "응답 거절", "룰 조건상 불가", "중복·다양성", "기타 검토", "코드 검사", "형식"]
_excl = {}                                              # 날짜 → (요청별 파일 지문, 그 날짜 결과)
_excl_req = {}                                          # (날짜, 요청) → (파일 지문, 요청 결과) - 바뀐 요청만 다시 읽음
_rv = {}                                                # 3_3_review 의 SAFETY_RE (다시 검토 대상 문구)


def redo_model():
    """LLM_설정.md '검토 대체 모델' (비었거나 '-'·none 이면 '' = 다시 만들기 못 함). 검토 모델과 같아도 된다
    (3_3 --redo-safety 는 처음 판정이 예전 검토 모델의 원문이라 같은 모델로도 다시 검토함, 2026-10-26 16:20~)."""
    m = run_settings().get("검토 대체 모델", "").strip()
    return "" if m.lower() in ("", "-", "none") else m


def _redo_re():
    """다시 만들기 대상을 고르는 문구 = 3_3_review.SAFETY_RE (3_3 --redo-safety 와 같은 기준). 못 읽으면 화면의 '안전상 이유'."""
    if "re" not in _rv:
        try:
            _rv["re"] = _load("review_33", "3_3_review.py").SAFETY_RE
        except Exception:
            _rv["re"] = EXCL_SAFETY
    return _rv["re"]


def _excl_cat(stage, reason):
    """빠진 이유 분류: 검토 모델이 뺀 것은 이유 문구로, 나머지는 단계 그대로.
    '검토 거절' (2026-10-27~) = 검토 모델이 같은 모델로 '재시도 횟수' 만큼 물어도 그 룰을 판정하지 않음 (거절·응답 중단)."""
    if stage == "검토 거절":
        return "응답 거절"
    if stage != "검토":
        return "코드 검사" if stage == "코드 검사" else "형식"
    if EXCL_SAFETY.search(reason):
        return "안전상 이유"
    if EXCL_DUP.search(reason):
        return "중복·다양성"
    if EXCL_RULE.search(reason):
        return "룰 조건상 불가"
    return "기타 검토"


def _jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    except (OSError, ValueError):
        return []


def excluded_dates():
    d = BASE / "07_llm_output"
    return sorted(p.name for p in d.glob("*") if DATE_RE.match(p.name) and (p / "_3_reviewed").is_dir()) \
        if d.is_dir() else []


def _excl_request(date, rev, gen, name, has_rej, fb, safety):
    """요청 하나 → {recs: 빠진 샘플, att: 룰별 [생성 공격 수, 남은 공격 수], redo: 다시 검토 전·후 수, cand: 다시 만들기 대상 룰}."""
    try:
        meta = json.loads((rev / f"{name}.meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    model, fbk = meta.get("reviewer") or "?", meta.get("fallback") or {}
    fb_rules, fb_model = set(fbk.get("answered") or []), fbk.get("model") or ""
    out = {"recs": [], "att": {}, "redo": None, "cand": set()}
    if isinstance(meta.get("redo"), dict):
        b, u = meta["redo"].get("before") or {}, (meta.get("usage") or {}).get("reviewer_fallback") or {}
        out["redo"] = {"at": str(meta["redo"].get("at", "")), "model": fb_model,
                       "samples": [b.get("samples") or 0, meta.get("samples") or 0],
                       "rejected": [b.get("rejected") or 0, meta.get("rejected") or 0],
                       "tokens": (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)}
    rej = _jsonl(rev / f"{name}.rejected.jsonl") if has_rej else []
    if not rej:
        return out
    # 다시 만들기 대상: 처음 검토 모델이 판정하지 않았거나(응답 거절, '검토 거절' 단계) 안전 사유로 뺀 샘플이 있는 룰 중
    # 아직 대체 모델에 보내지 않은 룰 (3_3 redo_candidates 와 같은 기준 - 대체 검토를 한 요청도 그때 보내지 않은 룰은 대상.
    # 2026-10-27 전 실행에서 판정 없이 통과된 룰은 여기서 셀 수 없음)
    sent = set(fbk.get("rules") or {})
    if fb and model not in ("?", fb):
        out["cand"] = {r.get("id") for r in rej if r.get("id") not in sent and (r.get("stage") == "검토 거절"
                       or (r.get("stage") == "검토" and safety.search(str(r.get("reason", "")))))}
    g_by = {g.get("id"): g for g in _jsonl(gen / f"{name}.jsonl") if isinstance(g, dict)}
    for rid, g in g_by.items():
        out["att"][rid] = [sum(1 for s in g.get("samples", []) if isinstance(s, dict) and s.get("label") == "anomalies"), 0]
    for k in _jsonl(rev / f"{name}.jsonl"):
        if k.get("id") in out["att"]:
            out["att"][k["id"]][1] += sum(1 for s in k.get("samples", []) if s.get("label") == "anomalies")
    for r in rej:
        g = g_by.get(r.get("id"), {})
        smp, i = g.get("samples") or [], r.get("i")
        s = smp[i] if isinstance(i, int) and 0 <= i < len(smp) and isinstance(smp[i], dict) else {}
        reason, stage = str(r.get("reason", "")), r.get("stage", "")
        out["recs"].append({"date": date, "req": name, "id": r.get("id", ""), "i": i, "stage": stage, "reason": reason,
                            "cat": _excl_cat(stage, reason),
                            "model": fb_model if stage == "검토" and fb_model and r.get("id") in fb_rules else model,
                            "label": s.get("label", ""), "type": g.get("attack_type") or "other",
                            "redo": r.get("id") in out["cand"]})
    return out


def excluded_records(date):
    """그 입력 날짜의 3.3 에서 빠진 샘플 [{date, req, id, i, stage, cat, reason, model, label, type, redo}],
    룰별 {id: [생성된 공격 수, 검토 뒤 남은 공격 수]}, 다시 검토한 요청 {이름: 전·후 수}, 다시 만들기 대상 {이름: 룰 수}.
    요청마다 파일(meta·뺀 샘플·남은 샘플) 지문으로 캐시해 바뀐 요청만 다시 읽는다 (다시 만들기가 도는 중에도 가볍게)."""
    rev = BASE / "07_llm_output" / date / "_3_reviewed"
    gen = rev.parent / "_2_generated"
    try:
        ents = {e.name: e for e in os.scandir(rev) if e.is_file()}
    except OSError:
        ents = {}
    fb, safety = redo_model(), _redo_re()
    sigs = {}
    for n in sorted({x[:-len(s)] for x in ents for s in (".meta.json", ".rejected.jsonl") if x.endswith(s)}):
        st = [ents.get(n + s) for s in (".meta.json", ".rejected.jsonl", ".jsonl")]
        sigs[n] = tuple((e.stat().st_size, e.stat().st_mtime_ns) if e else None for e in st) + (fb,)
    hit = _excl.get(date)
    if hit and hit[0] == sigs:
        return hit[1]
    res = {"recs": [], "att": {}, "redone": {}, "cand": {}}
    for n, sig in sigs.items():
        h = _excl_req.get((date, n))
        if not h or h[0] != sig:
            h = _excl_req[(date, n)] = (sig, _excl_request(date, rev, gen, n, bool(sig[1] and sig[1][0]), fb, safety))
        o = h[1]
        res["recs"] += o["recs"]
        for k, v in o["att"].items():
            x = res["att"].setdefault(k, [0, 0])
            x[0] += v[0]
            x[1] += v[1]
        if o["redo"]:
            res["redone"][n] = o["redo"]
        if o["cand"]:
            res["cand"][n] = len(o["cand"])
    with _lock:
        for k in [k for k in list(_excl_req) if k[0] == date and k[1] not in sigs]:     # 옮겨진 요청
            _excl_req.pop(k, None)
        _excl[date] = (sigs, res)
    return res


def _excl_select(q):
    """화면 조건(날짜·검토 모델·분류·공격 유형) → (모든 날짜, 고른 날짜, 그 날짜 전체 기록, 조건에 맞는 기록, 룰별 공격 수, 조건,
    다시 만들기 {redone: {(날짜, 요청): 전·후 수}, cand: {(날짜, 요청): 대상 룰 수}} - 다시 만들기는 날짜 단위라 날짜 조건만)."""
    dates = excluded_dates()
    date = _q(q, "date") or (dates[-1] if dates else "")
    if date and date != "all" and date not in dates:
        raise ValueError("date")
    model, cat, typ = _q(q, "model"), _q(q, "cat"), _q(q, "type")
    if cat and cat not in EXCL_CATS:
        raise ValueError("cat")
    recs, att, redo = [], {}, {"redone": {}, "cand": {}}
    for d in (dates if date == "all" else [date] if date else []):
        x = excluded_records(d)
        recs += x["recs"]
        for k, v in x["att"].items():
            a = att.setdefault(k, [0, 0])
            a[0] += v[0]
            a[1] += v[1]
        for kind in redo:
            redo[kind].update({(d, n): v for n, v in x[kind].items()})
    sel = [r for r in recs if (not model or r["model"] == model) and (not cat or r["cat"] == cat)
           and (not typ or r["type"] == typ)]
    return dates, date, recs, sel, att, {"model": model, "cat": cat, "type": typ}, redo


def excluded_summary(q):
    """2. 화면 '검토에서 뺀 샘플': 분류별·공격 유형(method)별 수, 이유 상위, 공격 샘플이 모두 빠진 룰, 다시 만들기 (대상·결과)."""
    dates, date, recs, sel, att, cond, redo = _excl_select(q)
    try:
        mapping = dataset_mod().load_mapping()[1]
    except Exception:
        mapping = {}
    by_model = [r for r in recs if not cond["model"] or r["model"] == cond["model"]]
    cats = {c: 0 for c in EXCL_CATS}
    for r in by_model:
        cats[r["cat"]] += 1
    lost = lambda r: r["label"] == "anomalies" and att.get(r["id"], [0, 0])[0] > 0 and att[r["id"]][1] == 0
    rows = {}
    for r in sel:
        x = rows.setdefault(r["type"], {"n": 0, "cats": {c: 0 for c in EXCL_CATS}, "att": 0, "norm": 0, "rules": set(),
                                        "lost": set(), "redo": 0})
        x["n"] += 1
        x["redo"] += r["redo"]
        x["cats"][r["cat"]] += 1
        x["att" if r["label"] == "anomalies" else "norm"] += 1
        x["rules"].add(r["id"])
        if lost(r):
            x["lost"].add(r["id"])
    types = [{"type": t, "name": ATTACK_NAMES.get(t, t), "method": (mapping.get(t) or {}).get("ips", ""), "n": x["n"],
              "cats": x["cats"], "att": x["att"], "norm": x["norm"], "rules": len(x["rules"]), "lost": len(x["lost"]),
              "redo": x["redo"]}
             for t, x in sorted(rows.items(), key=lambda kv: -kv[1]["n"])]
    reasons, ex = Counter(), {}
    for r in sel:
        reasons[(r["cat"], r["reason"])] += 1
        ex.setdefault((r["cat"], r["reason"]), set()).add(r["id"])
    top = [{"cat": c, "reason": t, "n": n, "rules": sorted(ex[(c, t)])[:3], "rule_n": len(ex[(c, t)])}
           for (c, t), n in reasons.most_common(40)]
    lost_rules = sorted({r["id"] for r in sel if lost(r)})
    # 다시 만들기: 대상(아직 대체 검토 안 한 요청의 안전 사유 룰) · 결과(다시 검토한 요청의 전·후 수) - 날짜 조건만
    done, cand, model = list(redo["redone"].values()), [r for r in recs if r["redo"]], redo_model()
    same = [x for x in done if x["model"] == model]          # 예상 토큰은 지금 대체 모델로 다시 만든 요청의 평균
    lock = RT.read_lock("generation") or {}
    rd = {"model": model, "running": "redo" in str(lock.get("by", "")),
          "cand_req": len(redo["cand"]), "cand_rules": sum(redo["cand"].values()), "cand_n": len(cand),
          "cand_att": sum(r["label"] == "anomalies" for r in cand),
          "done_req": len(done), "last_at": max((x["at"] for x in done), default=""),
          # 요청마다 남은 샘플 + 뺀 샘플 = 생성 샘플이라, 목록에서 빠진 수 = 늘어난 학습 샘플 수 (증강 전)
          "removed": sum(x["rejected"][0] - x["rejected"][1] for x in done),
          "left": sum(x["rejected"][1] for x in done),
          "avg_tokens": round(sum(x["tokens"] for x in same) / len(same)) if same else 0, "avg_n": len(same)}
    return {"dates": dates, "date": date, "models": sorted({r["model"] for r in recs}), "cats": EXCL_CATS,
            "by_cat": cats, "total": len(by_model), "sel": len(sel),
            "sel_label": {"anomalies": sum(r["label"] == "anomalies" for r in sel),
                          "normal": sum(r["label"] == "normal" for r in sel)},
            "rules": len({r["id"] for r in sel}), "lost_rules": lost_rules[:200], "lost_n": len(lost_rules),
            "types": types, "reasons": top, "redo": rd, **cond}


def excluded_csv(q):
    """검토에서 뺀 샘플 CSV (조건 그대로, 생성된 원래 http 포함)."""
    dates, date, recs, sel, att, cond, _ = _excl_select(q)
    try:
        mapping = dataset_mod().load_mapping()[1]
    except Exception:
        mapping = {}
    name = "_".join(["excluded", date or "none"] + [x for x in (cond["cat"], cond["type"]) if x]).replace(" ", "_") + ".csv"

    def chunks():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "요청", "rule_id", "공격 유형", "method (IPS)", "label", "단계", "분류", "이유", "검토 모델",
                    "다시 만들기 대상", "http"])
        yield ("﻿" + buf.getvalue()).encode("utf-8")
        by_req = {}
        for r in sel:
            by_req.setdefault((r["date"], r["req"]), []).append(r)
        for (d, req), rs in by_req.items():
            g_by = {g.get("id"): g for g in _jsonl(BASE / "07_llm_output" / d / "_2_generated" / f"{req}.jsonl")
                    if isinstance(g, dict)}
            buf.seek(0)
            buf.truncate()
            for r in rs:
                smp, i = (g_by.get(r["id"]) or {}).get("samples") or [], r["i"]
                http = smp[i].get("http", "") if isinstance(i, int) and 0 <= i < len(smp) and isinstance(smp[i], dict) else ""
                w.writerow([d, req, r["id"], r["type"], (mapping.get(r["type"]) or {}).get("ips", ""), r["label"],
                            r["stage"], r["cat"], _cell(r["reason"]), r["model"], "예" if r["redo"] else "",
                            _cell(str(http).replace("\r\n", "\n").replace("\n", "\\n"))])
            yield buf.getvalue().encode("utf-8")
    return name, chunks()


# ── 공통 ────────────────────────────────────────────────────────────────

def input_dates():
    d = BASE / "05_llm_input"
    return sorted(p.name for p in d.glob("*") if DATE_RE.match(p.name) and (p / "summary.json").exists()) \
        if d.is_dir() else []


def settings_summary():
    """머리글·생성 카드의 모델 {생성, 검토: 'provider:model' (검토 생략이면 'none'), 대체: 검토 대체 모델}."""
    try:
        text = SETTINGS_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    out = {}
    for role in ("생성", "검토"):
        m = re.search(rf"^\|\s*{role}\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", text, re.M)
        if m:
            out[role] = "none" if m.group(1) in ("none", "-") else f"{m.group(1)}:{m.group(2)}"
    out["대체"] = redo_model()
    return out


# ── 2. 모델 설정 (LLM_설정.md: '모델' 표의 생성·검토 줄, '실행' 표의 검토 대체 모델·동시 요청 수) ─────────
# 파이프라인(3_0 → 3.x)이 실행할 때마다 이 파일을 읽으므로, 생성·다시 만들기가 도는 동안(generation 잠금)에는 저장하지 않는다.
# 바꾼 칸만 고치고(설명 칸·다른 줄·줄바꿈은 그대로) 임시 파일에 쓴 뒤 바꾼다. 바꾼 기록은 logs/llm_settings_changes.jsonl.
# 키 값은 화면으로 내보내지 않는다 (.env 에 있는지·형식만 본다).
MODEL_LOG = BASE / "logs" / "llm_settings_changes.jsonl"
MODEL_NAME_RE = re.compile(r"^[\w.\-]+$")                       # MODEL_RE 의 model 부분
KNOWN_MODELS = {       # 고르기 목록 (직접 입력도 됨) - 이 프로젝트에서 썼거나 LLM_설정.md 에 적힌 이름 (+ 지난 기록의 모델)
    "upstage": ["solar-pro2"],
    "openai": ["gpt-6-luna", "gpt-6.1-sol", "gpt-5"],
    "anthropic": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"],
    "claude_cli": ["sonnet", "opus", "haiku", "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"],
}
MODEL_FIELDS = ("provider", "model", "temperature", "max_tokens")      # '모델' 표의 2~5번째 칸
RUN_FIELDS = {"fb": "검토 대체 모델", "conc_gen": "생성 동시 요청 수", "conc_rev": "검토 동시 요청 수",
              "eff_gen": "생성 추론 강도", "eff_rev": "검토 추론 강도"}
# 추론 강도에 쓸 수 있는 값 (3_0a_common.set_effort 와 같음): openai 는 reasoning_effort (xhigh·max → high),
# claude_cli 는 --effort (minimal → low), 그 밖의 provider 는 쓰지 않음. 비우면 넘기지 않음 (모델 기본값)
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
CONC_TOP = {"conc_gen": 16, "conc_rev": 32}                     # 3_0a_common.parallel_settings 의 상한
# 연결 확인: 아주 짧은 요청 1개를 별도 프로세스에서 (API 키를 화면 프로세스에 올리지 않고, 파이프라인 로그에도 남기지 않음)
CHECK_CODE = r"""
import importlib.util, json, logging, sys, time
base, role, model, temp, mt, eff = sys.argv[1:7]
spec = importlib.util.spec_from_file_location("llm_common_check", base + "/3_0a_common.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
C.load_env()
roles, providers, run = C.load_settings()
cfg = C.role_config(roles, role, model)
cfg.update(temperature=float(temp) if temp else None, max_tokens=int(mt))
run = dict(run, **{"재시도 횟수": "1", "응답 대기 시간": "120"})
log = logging.getLogger("model_check"); log.propagate = False; log.addHandler(logging.NullHandler())
if eff and hasattr(C, "set_effort"):                     # 실제 실행처럼 추론 강도를 넘김
    C.set_effort(cfg, eff, log)
t = time.time()
try:
    text, u = C.call_llm(cfg, providers, "Reply with the single word OK.", "ping", run, log)
    out = {"ok": True, "reply": " ".join(text.split())[:80], "in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
           "cut": u.get("finish_reason") == "length"}
except Exception as e:
    out = {"ok": False, "error": " ".join(str(e).split())[:400], "fatal": C.is_fatal(e)}
out["sec"] = round(time.time() - t, 1)
print(json.dumps(out, ensure_ascii=False))
"""
_mstat, _cli, _c3 = {}, {"t": 0.0, "val": None}, {}     # 모델별 기록(meta.json 마다), claude 로그인(5분), 3_0a_common
_magg = {"t": 0.0, "val": None}                         # 모델별 기록 합계 (1분 - 파일 7천 개를 매번 훑지 않게)


def _md_tables(text):
    """md 의 '## 제목' 별 표 {제목: [[셀, …]]} (3_0a_common.md_tables 와 같은 규칙, 머리글·구분선 제외)."""
    tables, section, seen = {}, None, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section, seen = s[3:].strip(), False
            tables[section] = []
        elif section is not None and s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if seen:
                tables[section].append(cells)
            seen = True
    return tables


def _common():
    """3_0a_common (claude 실행 파일 찾기·CLI 환경) - 처음 쓸 때 한 번 읽는다."""
    if "m" not in _c3:
        _c3["m"] = _load("llm_common_ui", "3_0a_common.py")
    return _c3["m"]


def _dotenv():
    """.env 의 {이름: 값} (3_0a_common.load_env 와 같은 규칙). 값은 화면으로 내보내지 않는다."""
    out = {}
    try:
        for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.replace("\t", " ").split(" #")[0].strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def _key_state(name, env):
    """키가 있는지만: 있음 / 없음 / 형식 오류 (한글·공백이 섞임 - 3_0a_common.call_llm 이 거절)."""
    v = (os.environ.get(name) or env.get(name) or "").strip()
    return "없음" if not v else "있음" if v.isascii() and " " not in v else "형식 오류"


def claude_status(refresh=False, url=""):
    """claude 실행 파일·로그인 (`claude auth status --json`, 5분 캐시). 메일·조직 같은 계정 정보는 내보내지 않는다."""
    if not refresh and _cli["val"] is not None and time.time() - _cli["t"] < 300:
        return _cli["val"]
    val = {"exe": "", "version": "", "logged_in": None, "method": "", "plan": "", "error": ""}
    try:
        C = _common()
        name = getattr(C, "CLAUDE_CLI_ENV", "CLAUDE_CLI_PATH")       # 파이프라인처럼 .env 의 실행 파일 경로도 본다
        exe = C.claude_cli_path({"url": url if url not in ("", "-") else _dotenv().get(name, "")})
        env = C._claude_cli_env()
    except Exception as e:
        exe, val["error"] = None, f"실행 파일을 찾지 못함 ({type(e).__name__})"
    if exe:
        v = re.search(r"claude-code[\\/]([\d.]+)[\\/]", exe)
        val.update(exe=Path(exe).name, version=v.group(1) if v else "")
        try:
            r = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, timeout=60, env=env, cwd=str(BASE),
                               stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            d = json.loads((r.stdout or b"").decode("utf-8", "replace").strip() or "{}")
            val.update(logged_in=bool(d.get("loggedIn")), method=str(d.get("authMethod") or ""),
                       plan=str(d.get("subscriptionType") or ""))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            val["error"] = f"로그인 상태를 읽지 못함 ({type(e).__name__})"
    _cli.update(t=time.time(), val=val)
    return val


def _meta_contrib(path, kind):
    """meta.json 하나 → ((역할, 'provider:model', 초, 입력 토큰, 출력 토큰), …). 검토에 대체 모델을 함께 쓴 요청은
    토큰을 검토·대체로 나누고 시간은 세지 않는다 (시간이 두 모델 합이라)."""
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    u = m.get("usage") or {}
    tok = lambda x: ((x or {}).get("input_tokens") or 0, (x or {}).get("output_tokens") or 0)
    if kind == "생성":
        return (("생성", m["generator"], m.get("seconds") or 0, *tok(u.get("generator"))),) if m.get("generator") else ()
    if not m.get("reviewer"):
        return ()
    fb = m.get("fallback") or {}
    if fb.get("model"):
        return (("검토", m["reviewer"], 0, *tok(u.get("reviewer_primary"))), ("대체", fb["model"], 0, *tok(u.get("reviewer_fallback"))))
    return (("검토", m["reviewer"], m.get("seconds") or 0, *tok(u.get("reviewer"))),)


def model_stats(fresh=False):
    """모델별 지난 기록 {생성|검토|대체: {'provider:model': {n, sec_n, sec, tin, tout}}} - 모든 날짜의
    _2_generated·_3_reviewed meta.json (바뀐 파일만 다시 읽음, 합계는 1분). 화면의 예상 시간·토큰과 '기록 없음' 표시에 쓴다."""
    if not fresh and _magg["val"] is not None and time.time() - _magg["t"] < 60:
        return _magg["val"]
    out, seen = {"생성": {}, "검토": {}, "대체": {}}, set()
    root = BASE / "07_llm_output"
    for d in (sorted(p for p in root.iterdir() if DATE_RE.match(p.name)) if root.is_dir() else []):
        for sub, kind in (("_2_generated", "생성"), ("_3_reviewed", "검토")):
            try:
                ents = [e for e in os.scandir(d / sub) if e.name.endswith(".meta.json")]
            except OSError:
                continue
            for e in ents:
                st = e.stat()
                hit = _mstat.get(e.path)
                if not hit or hit[0] != (st.st_size, st.st_mtime_ns):
                    hit = _mstat[e.path] = ((st.st_size, st.st_mtime_ns), _meta_contrib(e.path, kind))
                seen.add(e.path)
                for role, model, sec, tin, tout in hit[1]:
                    x = out[role].setdefault(model, {"n": 0, "sec_n": 0, "sec": 0.0, "tin": 0, "tout": 0})
                    x["n"] += 1
                    x["tin"] += tin
                    x["tout"] += tout
                    if sec:
                        x["sec_n"] += 1
                        x["sec"] += sec
    with _lock:
        for k in [k for k in list(_mstat) if k not in seen]:
            _mstat.pop(k, None)
        _magg.update(t=time.time(), val=out)
    return out


def _changes_tail(n=10):
    """바꾼 기록 최근 n개 (최근 것부터)."""
    try:
        rows = MODEL_LOG.read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for x in reversed(rows):
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def _read_settings():
    raw = SETTINGS_FILE.read_bytes()
    return raw, _md_tables(raw.decode("utf-8"))


def model_settings(q):
    """화면 2 '모델 설정': 지금 값, provider 별 사용 가능 여부(키는 있는지만), 고르기 목록, 모델별 지난 기록, 바꾼 기록."""
    raw, t = _read_settings()
    roles = {r[0]: dict(zip(MODEL_FIELDS, r[1:5])) for r in t.get("모델", []) if len(r) >= 5 and r[0] in ("생성", "검토")}
    prov = {r[0]: {"url": r[1], "key": r[2], "style": r[3]} for r in t.get("provider", []) if len(r) >= 4}
    run = {r[0]: r[1] for r in t.get("실행", []) if len(r) >= 2}
    env, providers = _dotenv(), {}
    for name, p in prov.items():
        key = p["key"] if p["key"] not in ("", "-") else ""
        providers[name] = {"style": p["style"], "key": key,
                           "state": claude_status(bool(_q(q, "refresh")), p["url"]) if p["style"] == "claude_cli"
                           else _key_state(key, env) if key else "필요 없음"}
    stats, known = model_stats(bool(_q(q, "refresh"))), {k: list(v) for k, v in KNOWN_MODELS.items()}
    for role in stats.values():                                  # 지난 기록에 있는 모델도 고르기 목록에
        for pm in role:
            pv, _, m = pm.partition(":")
            if m and m not in known.setdefault(pv, []):
                known[pv].append(m)
    try:
        where = SETTINGS_FILE.relative_to(BASE).as_posix()
    except ValueError:
        where = str(SETTINGS_FILE)
    return {"sig": hashlib.sha1(raw).hexdigest(), "file": where, "roles": roles,
            "run": {k: run.get(v, "") for k, v in RUN_FIELDS.items()}, "providers": providers, "known": known,
            "stats": stats, "changes": _changes_tail(),
            "locked": job_state("generation")["running"] or job_state("redo")["running"]}


def _int_in(v, lo, hi, what):
    try:
        n = int(str(v).strip())
    except ValueError:
        raise ValueError(what) from None
    if not lo <= n <= hi:
        raise ValueError(f"{what} ({lo}~{hi})")
    return n


def _temperature(v, what):
    t = str(v or "").strip()
    if t:
        try:
            ok = 0 <= float(t) <= 2
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"{what} temperature (비우거나 0~2)")
    return t


def save_models(b):
    """모델 설정 저장 → (HTTP 코드, 응답). 생성·다시 만들기가 도는 중이거나, 화면을 연 뒤 파일이 바뀌었으면 저장하지 않는다."""
    if job_state("generation")["running"] or job_state("redo")["running"]:
        return 409, {"error": "학습데이터 생성(또는 다시 만들기)이 실행 중이라 저장하지 않았습니다 - 실행 중인 단계가 이 설정을 "
                              "읽습니다. 생성이 끝난 뒤 저장하세요"}
    raw, t = _read_settings()
    if b.get("sig") != hashlib.sha1(raw).hexdigest():
        return 409, {"error": "화면을 연 뒤 LLM_설정.md 가 바뀌었습니다 (다른 곳에서 고침) - '다시 불러오기' 뒤 저장하세요"}
    prov = {r[0] for r in t.get("provider", []) if len(r) >= 4}
    want = {}                                                   # (표, 첫 칸) → {칸 번호: 새 값}
    for role in ("생성", "검토"):
        x = (b.get("roles") or {}).get(role) or {}
        p, m = str(x.get("provider", "")).strip(), str(x.get("model", "")).strip()
        if not (p in prov or (role == "검토" and p == "none")) or not MODEL_NAME_RE.match(m):
            raise ValueError(f"{role} 모델 (provider 표의 이름 + 영문·숫자·'.-_' 모델 이름)")
        want[("모델", role)] = {2: p, 3: m, 4: _temperature(x.get("temperature"), role),
                               5: str(_int_in(x.get("max_tokens", ""), 256, 200000, f"{role} max_tokens"))}
    r = b.get("run") or {}
    fb = str(r.get("fb", "")).strip() or "-"
    if fb != "-" and not (MODEL_RE.match(fb) and fb.partition(":")[0] in prov and fb.partition(":")[2]):
        raise ValueError("검토 대체 모델 ('provider:model' 또는 '-')")
    want[("실행", RUN_FIELDS["fb"])] = {2: fb}
    for k in ("conc_gen", "conc_rev"):
        want[("실행", RUN_FIELDS[k])] = {2: str(_int_in(r.get(k, ""), 1, CONC_TOP[k], RUN_FIELDS[k]))}
    for k in ("eff_gen", "eff_rev"):                           # 추론 강도 (보내지 않은 칸은 그대로)
        if k in r:
            e = str(r.get(k) or "").strip().lower()
            if e not in ("",) + EFFORTS:
                raise ValueError(f"{RUN_FIELDS[k]} (비우거나 {'·'.join(EFFORTS)})")
            want[("실행", RUN_FIELDS[k])] = {2: e}
    empty_ok = {RUN_FIELDS["fb"], RUN_FIELDS["eff_gen"], RUN_FIELDS["eff_rev"]}    # 빈칸·'-' 는 같은 뜻
    lines, section, changes, found = raw.decode("utf-8").splitlines(keepends=True), None, [], set()
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        s = body.strip()
        if s.startswith("## "):
            section = s[3:].strip()
            continue
        parts = body.split("|")
        key = (section, parts[1].strip()) if s.startswith("|") and len(parts) > 3 else None
        cells = want.get(key)
        if not cells or max(cells) >= len(parts) - 1:
            continue
        found.add(key)
        for j, val in cells.items():
            old = parts[j].strip()
            if old == val or (key[1] in empty_ok and {old, val} <= {"", "-", "none"}):
                continue
            changes.append([f"{key[1]} {MODEL_FIELDS[j - 2]}" if section == "모델" else key[1], old, val])
            parts[j] = f" {val} "                               # 빈 값도 같은 모양 ('|  |', 실행 표 그대로)
        lines[i] = "|".join(parts) + line[len(body):]           # 줄바꿈(CRLF) 그대로
    missing = [k[1] for k in set(want) - found if any(want[k].values())]     # 없는 줄에 빈 값이면 그냥 둠
    if missing:
        raise ValueError("LLM_설정.md 에 없는 줄: " + ", ".join(missing))
    if not changes:
        return 200, {"changed": []}
    tmp = SETTINGS_FILE.with_name(f"{SETTINGS_FILE.name}.{os.getpid()}.tmp")
    tmp.write_bytes("".join(lines).encode("utf-8"))
    for k in range(10):                                         # 다른 프로그램이 잠깐 열어 두었으면 (Windows) 조금 뒤 다시
        try:
            os.replace(tmp, SETTINGS_FILE)
            break
        except PermissionError:
            if k == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)
    MODEL_LOG.parent.mkdir(exist_ok=True)
    with open(MODEL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": RT.now_iso(), "file": SETTINGS_FILE.name, "changes": changes}, ensure_ascii=False) + "\n")
    return 200, {"changed": changes}


def check_model(b):
    """연결 확인: 고른 provider:model 에 아주 짧은 요청 1개 → (200, {ok, sec, reply·토큰 | error})."""
    role, model = b.get("role"), str(b.get("model", "")).strip()
    if role not in ("생성", "검토") or not MODEL_RE.match(model) or ":" not in model:
        raise ValueError("model ('provider:model')")
    temp = _temperature(b.get("temperature"), role)
    mt = min(_int_in(b.get("max_tokens") or 2048, 1, 200000, "max_tokens"), 2048)
    eff = str(b.get("effort") or "").strip().lower()
    if eff not in ("",) + EFFORTS:
        raise ValueError(f"추론 강도 (비우거나 {'·'.join(EFFORTS)})")
    try:
        r = subprocess.run([sys.executable, "-c", CHECK_CODE, str(BASE), role, model, temp, str(mt), eff], capture_output=True,
                           timeout=240, cwd=str(BASE), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return 200, {"ok": False, "error": "4분 안에 끝나지 않음"}
    for line in reversed((r.stdout or b"").decode("utf-8", "replace").splitlines()):
        if line.startswith("{"):
            try:
                return 200, json.loads(line)
            except ValueError:
                break
    err = " ".join((r.stderr or b"").decode("utf-8", "replace").split())[-300:]
    return 200, {"ok": False, "error": err or f"종료 코드 {r.returncode}"}


def state():
    try:
        run = run_progress()
    except Exception as e:              # 진행률 계산이 실패해도 화면은 계속
        run = {"error": f"{type(e).__name__}: {e}"}
    sup = job_state("supervised")
    sup["pct"] = sup_status().get("overall_pct") if sup["running"] else None      # 메뉴 6. 옆 %
    return {"now": RT.now_iso(), "generation": dict(job_state("generation"), run=run), "collect": job_state("collect"),
            "agent": dict(job_state("agent"), status=RT.read_status("agent")),
            "last_collect": RT.read_status("collect"), "input_dates": input_dates(),
            "models": settings_summary(), "settings": dict(zip(("generate", "review", "overlap"), parallel())),
            "step_seconds": step_summary(), "supervised": sup, "redo": job_state("redo")}


# ── 요청 처리 ────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "RuleDataUI/1.1"

    def log_message(self, fmt, *args):          # 콘솔 조용히
        pass

    def _allowed_host(self):
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, chunks, ctype, extra=None):
        """만드는 대로 보낸다 (길이를 미리 모름 → HTTP/1.0 이라 연결을 닫으면 끝)."""
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            for chunk in chunks:
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                        # 브라우저에서 내려받기를 취소함
        self.close_connection = True

    def do_GET(self):
        if not self._allowed_host():
            return self._send(403, {"error": "host"})
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._send(200, state())
            if u.path == "/api/progress":
                return self._send(200, read_progress())
            if u.path == "/api/rules":
                return self._send(200, rules_page(q))
            if u.path == "/api/log":
                job = _q(q, "job", "generation")
                if job not in JOBS:
                    return self._send(400, {"error": "job"})
                return self._send(200, {"text": log_text(job, int(_q(q, "lines", "150")))})
            if u.path == "/api/data/dates":
                return self._send(200, {"dates": dataset_dates()})
            if u.path == "/api/data/rows":
                return self._send(200, data_rows(q))
            if u.path == "/api/data/verify":
                return self._send(200, verify_summary(_q(q, "date"), _q(q, "dev", "ips")))
            if u.path == "/api/data/edits":
                return self._send(200, edits_summary(_q(q, "date")))
            if u.path == "/api/data/conv/save":
                return self._send(200, conv_save_state(_q(q, "date"), _q(q, "dev")))
            if u.path == "/api/excluded":
                return self._send(200, excluded_summary(q))
            if u.path == "/api/models":
                return self._send(200, model_settings(q))
            if u.path == "/download/excluded.csv":
                from urllib.parse import quote
                name, chunks = excluded_csv(q)
                return self._stream(chunks, "text/csv; charset=utf-8",
                                    {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
            if u.path == "/api/sup/state":
                return self._send(200, sup_state(_q(q, "base")))
            if u.path == "/api/sup/summary":
                return self._send(200, sup_summary(q))
            if u.path == "/download/sup/report":
                C, _ = sup_mods()
                _, _, base = _sup_base(_q(q, "base"))
                dev = _q(q, "dev", "ips")
                p = C.compared_files(base, dev)["report"] if base and dev in C.DEVICES else None
                if not p or not p.exists():
                    return self._send(404, {"error": "엑셀이 없습니다"})
                from urllib.parse import quote
                return self._send(200, p.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(f'{dev}_{base}_비교.xlsx')}"})
            if u.path == "/download/sup/candidates":
                from urllib.parse import quote
                name, chunks = sup_candidates_csv(q)
                return self._stream(chunks, "text/csv; charset=utf-8",
                                    {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
            if u.path == "/api/features":
                dev = _q(q, "dev", "ips")
                if dev not in ("ips", "waf"):
                    raise ValueError("dev")
                return self._send(200, dict(feature_catalog(), **feature_counts(dev)))
            if u.path == "/download/rule_status.csv" and CSV_FILE.exists():
                return self._send(200, CSV_FILE.read_bytes(), "text/csv; charset=utf-8",
                                  {"Content-Disposition": "attachment; filename=rule_status.csv"})
            if u.path == "/download/frame":             # 데이터 프레임 (화면의 열 그대로, 조건에 맞는 모든 줄)
                total, name, chunks = frame_csv(q)
                return self._stream(chunks, "text/csv; charset=utf-8",
                                    {"Content-Disposition": f"attachment; filename={name}", "X-Rows": str(total)})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        # 다른 사이트(브라우저)가 이 화면으로 요청을 보내지 못하게: Host, 전용 헤더, JSON 만 허용
        if (not self._allowed_host() or self.headers.get("X-RuleData-UI") != "1"
                or not self.headers.get("Content-Type", "").startswith("application/json")):
            return self._send(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return self._send(400, {"error": "json"})
        path = urlparse(self.path).path
        try:
            return self._send(*self.route(path, body))
        except ValueError as e:
            return self._send(400, {"error": str(e)})

    def route(self, path, b):
        if path == "/api/progress/refresh":
            return 200, {"started": refresh_progress()}
        if path == "/api/data/edit":
            return 200, save_edit(b)
        if path == "/api/data/check":
            return 200, check_edit(b)
        if path == "/api/data/edit/bad":
            return 200, delete_bad(b.get("date", ""), b.get("dev", "ips"))
        if path == "/api/data/rebuild":
            return 200, start_rebuild(b.get("date", ""))
        if path == "/api/data/conv/save":
            return 200, conv_save_start(b)
        if path == "/api/sup/compare":
            return 200, sup_start("compare", b)
        if path == "/api/sup/finalize":
            return 200, sup_start("finalize", b)
        if path == "/api/sup/stop":
            if not job_state("supervised")["running"]:
                return 409, {"error": "실행 중인 비교·확정이 없습니다"}
            RT.request_stop("supervised")
            return 200, {"stop_requested": True}
        if path == "/api/sup/preview":
            return 200, sup_preview(b)
        if path == "/api/sup/save":
            return 200, sup_save(b)
        if path == "/api/sup/delete":
            sup_mods()[1].delete_draft(b.get("name", ""))
            return 200, {"deleted": True}
        if path == "/api/sup/register":
            sup_mods()[1].register_base(b.get("name", ""), b.get("tag", ""), b.get("files"), b.get("feature_code"))
            return 200, {"registered": True}
        if path in ("/api/data/verify", "/api/data/verify/stop"):
            dev, mode = b.get("dev", "ips"), b.get("mode", "all")
            dates = [d["date"] for d in dataset_dates()]
            todo = dates if b.get("date") == "all" else [b.get("date")]
            if not set(todo) <= set(dates) or mode not in ("new", "all"):
                raise ValueError("date/mode")
            if path.endswith("/stop"):
                return 200, {"stopping": [d for d in todo if stop_verify(d, dev)]}
            res = {d: start_verify(d, dev, mode) for d in todo}
            return 200, {"started": [d for d, r in res.items() if r == "started"],
                         "done": [d for d, r in res.items() if r == "done"]}
        if path == "/api/models/save":           # 2. 모델 설정 저장 (LLM_설정.md, 생성 중에는 막음)
            return save_models(b)
        if path == "/api/models/check":          # 2. 모델 설정 '연결 확인' (요청 1개)
            return check_model(b)
        if path == "/api/excluded/redo":         # 2. 검토에서 뺀 샘플 '다시 만들기' = 3_3 --redo-safety (API 비용, 생성 잠금)
            if job_state("generation")["running"] or job_state("redo")["running"]:
                return 409, {"error": "학습데이터 생성(또는 다시 만들기)이 실행 중입니다 - 끝난 뒤 다시 누르세요"}
            if not redo_model():
                return 409, {"error": "LLM_설정.md 의 '검토 대체 모델' 이 비어 있습니다 (예: claude_cli:sonnet)"}
            args, d = ["--redo-safety"], b.get("date") or "all"
            if d != "all":
                if d not in excluded_dates():
                    raise ValueError("date")
                args += ["--date", d]
            RT.clear_stop("generation")
            return 200, {"pid": spawn("redo", args), "args": args}
        if path == "/api/generation/start":
            if job_state("generation")["running"] or job_state("redo")["running"]:
                return 409, {"error": "학습데이터 생성(또는 다시 만들기)이 이미 실행 중입니다"}
            args = ["--set", b.get("set", "all")]
            if args[1] not in ("all", "delta"):
                raise ValueError("set")
            n = int(b.get("max_requests", 5))
            if not 0 <= n <= 100000:
                raise ValueError("max_requests")
            args += ["--max-requests", str(n)]
            if b.get("date"):
                if b["date"] not in input_dates():
                    raise ValueError("date")
                args += ["--date", b["date"]]
            for key in ("generator", "reviewer"):
                if b.get(key):
                    if not MODEL_RE.match(b[key]):
                        raise ValueError(key)
                    args += [f"--{key}", b[key]]
            if b.get("dry_run"):
                args.append("--dry-run")
            RT.clear_stop("generation")
            return 200, {"pid": spawn("generation", args), "args": args}
        if path == "/api/generation/stop":
            st = job_state("generation")
            if not st["running"]:
                return 409, {"error": "실행 중인 생성이 없습니다"}
            if b.get("force"):
                return 200, {"killed": force_kill(st["lock"].get("pid") or st["ui"].get("pid"))}
            if "redo" in str(st["lock"].get("by", "")):     # 3_3 --redo-safety 는 중지 요청을 보지 않는다
                return 409, {"error": "다시 만들기는 '강제 중지'로만 멈춥니다 (이미 다시 검토한 요청은 다음 생성 때 증강·학습 로그에 반영)"}
            RT.request_stop("generation")
            return 200, {"stop_requested": True}
        if path == "/api/collect/start":
            if job_state("collect")["running"]:
                return 409, {"error": "수집이 이미 실행 중입니다"}
            return 200, {"pid": spawn("collect", [])}
        if path == "/api/collect/stop":
            st = job_state("collect")
            if not st["running"]:
                return 409, {"error": "실행 중인 수집이 없습니다"}
            return 200, {"killed": force_kill(st["lock"].get("pid") or st["ui"].get("pid"))}
        if path == "/api/agent/start":
            if job_state("agent")["running"]:
                return 409, {"error": "수집 에이전트가 이미 실행 중입니다"}
            at = b.get("at", "16:00")
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", at):
                raise ValueError("at")
            RT.clear_stop("agent")
            return 200, {"pid": spawn("agent", ["--agent", "--at", at])}
        if path == "/api/agent/stop":
            st = job_state("agent")
            if not st["running"]:
                return 409, {"error": "실행 중인 에이전트가 없습니다"}
            if b.get("force"):
                return 200, {"killed": force_kill(st["lock"].get("pid") or st["ui"].get("pid"))}
            RT.request_stop("agent")
            return 200, {"stop_requested": True}
        return 404, {"error": "not found"}


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Snort 룰 학습데이터</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--fg:#1d2330;--muted:#6b7385;--line:#e3e6ec;--accent:#2f6fde;--ok:#1f9d55;--warn:#b7791f;--bad:#d64545;--chip:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1c2028;--fg:#e6e9ef;--muted:#99a1b3;--line:#2c323d;--accent:#5b8ff0;--chip:#262b35}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line)}
.top{display:flex;align-items:baseline;gap:12px;padding:12px 20px 0}.top h1{font-size:18px;margin:0}.top .sub{color:var(--muted);font-size:13px}
nav{display:flex;gap:4px;padding:8px 16px 0;overflow-x:auto}nav a{padding:8px 14px;border-radius:8px 8px 0 0;color:var(--muted);text-decoration:none;white-space:nowrap;border-bottom:2px solid transparent}
nav a.on{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}nav a .run[hidden]{display:none}nav a .run{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);margin-left:6px;vertical-align:middle}
main{padding:16px;max-width:1500px;margin:0 auto}.page{display:none}.page.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}.card h2{font-size:15px;margin:0 0 12px}
.big{font-size:44px;font-weight:700;line-height:1}.muted{color:var(--muted)}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
.bar{height:12px;background:var(--chip);border-radius:6px;overflow:hidden;margin:10px 0}.bar>i{display:block;height:100%;background:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--chip);font-size:12px;white-space:nowrap}
.f-DONE,.l-normal{background:#d7f3e3;color:#13633a}.f-REVIEWED,.f-GENERATED{background:#dde8fb;color:#1f4fa3}.f-QUEUED,.f-TODO,.f-EXCLUDED{background:var(--chip);color:var(--muted)}
.f-CHANGED,.f-MISSING,.f-EMPTY{background:#fff0d1;color:#8a5a00}.f-FAILED,.l-anomalies{background:#fde0e0;color:#a32424}.f-EXISTING{background:#e8e0fb;color:#5531a3}
button{font:inherit;padding:6px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.danger{color:var(--bad);border-color:var(--bad)}button:disabled{opacity:.45;cursor:default}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.seg button{border:0;border-radius:0}.seg button.on{background:var(--accent);color:#fff}
input,select{font:inherit;padding:5px 8px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--muted)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;max-height:300px;overflow:auto;font:12px/1.45 Consolas,monospace;white-space:pre-wrap;margin:10px 0 0}
.status{display:flex;align-items:center;gap:8px;margin-bottom:10px}.dot{width:10px;height:10px;border-radius:50%;background:var(--muted)}.dot.on{background:var(--ok);box-shadow:0 0 0 4px rgba(31,157,85,.15)}.dot.idle{background:var(--card);border:2px solid var(--accent)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px}.kv b{font-weight:600;color:var(--muted)}
details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:8px}.msg{font-size:13px;margin-top:8px;min-height:18px}
.payload{font:12px/1.45 Consolas,monospace;white-space:pre-wrap;word-break:break-all;max-height:10.5em;overflow:auto;background:var(--bg);border-radius:6px;padding:6px 8px;min-width:280px}
.chk{font-size:12px;line-height:1.5;min-width:220px}td.method{min-width:130px}.chk .ok{color:var(--ok)}.chk .warn{color:var(--warn)}.chk .bad{color:var(--bad);font-weight:600}
tr.lv-bad td{background:rgba(214,69,69,.06)}.reason{font-size:12.5px;min-width:240px}
.pager{display:flex;gap:8px;align-items:center;margin-top:10px}
.est{margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--bg);font-size:13px;line-height:1.6}.est b{font-weight:600}
.c-OK{background:#d7f3e3;color:#13633a}.c-RULE{background:#dde8fb;color:#1f4fa3}.c-CHECK{background:#fff0d1;color:#8a5a00}.c-BAD{background:#fde0e0;color:#a32424}
.vsum{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px;font-size:13px}button.sm{padding:2px 8px;font-size:12px;margin:0 2px 2px 0}td.act{min-width:130px}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:50;display:none;align-items:center;justify-content:center;padding:16px}.mbox{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;width:min(940px,100%);max-height:92vh;overflow:auto}.mbox h3{margin:0 0 10px;font-size:15px}.mbox textarea{width:100%;min-height:260px;font:12px/1.45 Consolas,monospace;padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);margin-top:8px}#edRes{margin:8px 0;font-size:13px}#dEditList{margin-top:8px;max-height:360px;overflow:auto}.vsum .chip{cursor:pointer;font-size:12.5px;padding:2px 10px}
.feat{font-size:12px;line-height:1.55;min-width:230px}.feat code{font:11.5px Consolas,monospace;background:var(--chip);border-radius:4px;padding:0 4px}
.cons{min-width:230px;font-size:12.5px}.cons .chip{margin-bottom:3px}
.pct{font-size:40px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums;min-width:118px}
.bar>i{transition:width .8s}.bar.live>i{background-image:linear-gradient(45deg,rgba(255,255,255,.28) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.28) 50%,rgba(255,255,255,.28) 75%,transparent 75%);background-size:16px 16px;animation:mv 1s linear infinite}
@keyframes mv{to{background-position:16px 0}}@media (prefers-reduced-motion:reduce){.bar.live>i{animation:none}}
.steps{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 6px}.step{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:12px;background:var(--chip);color:var(--muted);font-size:12.5px;white-space:nowrap}
.step.done{background:#d7f3e3;color:#13633a}.step.running{background:#dde8fb;color:#1f4fa3;font-weight:600}.step.fail{background:#fde0e0;color:#a32424}.step.skip{opacity:.55;text-decoration:line-through}
.navpct{font-size:12px;color:var(--ok);margin-left:5px;font-weight:600}.runnow{margin-top:4px;font-size:13.5px}
code.fn{cursor:help;border-bottom:1px dotted var(--muted)}
.tipbox{position:fixed;z-index:50;max-width:min(440px,calc(100vw - 16px));background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.18);padding:10px 12px;font-size:12.5px;line-height:1.55;pointer-events:none}
.tipbox[hidden]{display:none}.tipbox .t{font-weight:600;margin-bottom:2px}
tr.hl td{background:rgba(47,111,222,.12)}.fgroup td{background:var(--bg);font-weight:600}.nw{white-space:nowrap}
.pat{font:11.5px Consolas,monospace;word-break:break-all;white-space:pre-wrap;max-height:9em;overflow:auto;background:var(--bg);border-radius:6px;padding:4px 6px;min-width:220px}
label.inl{flex-direction:row;align-items:center;gap:5px;font-size:13px;color:var(--fg)}.g0{background:var(--chip);color:var(--muted)}.g1{background:#fde0e0;color:#a32424}.g2{background:#dde8fb;color:#1f4fa3}.g3{background:#d7f3e3;color:#13633a}
.hold{display:flex;gap:6px 18px;flex-wrap:wrap;margin-top:8px;font-size:12.5px}.hold a,a[data-hv]{cursor:pointer;color:var(--accent);text-decoration:underline}
.tw{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}.tw>div{min-width:260px;flex:1}.tw h4{margin:10px 0 4px;font-size:13px}table.mini td,table.mini th{padding:3px 6px}.ck{display:inline-flex;gap:3px;align-items:center;margin-right:8px}
@media (max-width:480px){main{padding:10px}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><div class="top"><h1>Snort 룰 학습데이터</h1><span class="sub" id="hdr"></span></div>
<nav><a href="#/" data-p="progress">1. 진행 현황</a><a href="#/run" data-p="run">2. 생성·수집<span class="run" id="navRun" hidden></span><span class="navpct" id="navPct"></span></a><a href="#/rules" data-p="rules">3. 룰별 상태</a><a href="#/data" data-p="data">4. 신규 학습 데이터</a><a href="#/features" data-p="features">5. 피처</a><a href="#/sup" data-p="sup">6. 기존 학습데이터와 비교<span class="run" id="navSup" hidden></span><span class="navpct" id="navSupPct"></span></a></nav></header>
<main>
<!-- 1. 진행 현황 -->
<section class="page" id="p-progress">
<div class="card" id="runCard" hidden><h2 id="runTitle">학습데이터 생성</h2><div id="runBody"></div></div>
<div class="card"><h2>진행 현황 (전체 룰 기준) <button id="refresh" style="float:right">다시 계산</button></h2><div id="progBody" class="muted">불러오는 중…</div></div></section>

<!-- 2. 생성·수집 -->
<section class="page" id="p-run"><div class="grid">
<div class="card"><h2>학습데이터 생성 (3_0)</h2>
 <div class="status"><span class="dot" id="genDot"></span><span id="genStatus"></span></div>
 <div id="genRun" style="margin-bottom:14px"></div>
 <div class="row">
  <label>세트<select id="gSet"><option value="all">전체 (all)</option><option value="delta">변경분 (delta)</option></select></label>
  <label>요청 수 (0 = 남은 전부)<input id="gN" type="number" min="0" value="200" style="width:110px"></label>
  <label>입력 날짜<select id="gDate"><option value="">가장 최근</option></select></label>
 </div>
 <div class="est" id="genEst"></div>
 <details id="gOvrBox"><summary>이번 실행만 다른 모델 <span class="muted" id="gOvr">(저장 안 됨 · 비우면 아래 '모델 설정'의 저장된 값)</span></summary><div class="row" style="margin-top:6px">
  <label>생성 모델<input id="gGen" placeholder="예: upstage:solar-pro2"></label>
  <label>검토 모델<input id="gRev" placeholder="예: claude_cli:opus"></label></div></details>
 <div class="row" style="margin-top:12px">
  <button id="bDry">건수 확인 (호출 없음)</button><button id="bPilot">파일럿 5개</button>
  <button class="primary" id="bGen">생성 시작</button><button id="bStop">중지 (지금 요청까지)</button><button class="danger" id="bKill">강제 중지</button></div>
 <div class="msg" id="genMsg"></div><pre id="genLog"></pre></div>
<div class="card"><h2>수집 (1_0)</h2>
 <div class="status"><span class="dot" id="agDot"></span><span id="agStatus"></span></div>
 <div class="kv" id="agKv"></div>
 <div class="row" style="margin-top:12px">
  <label>매일 수집 시각<input id="agAt" value="16:00" style="width:80px"></label>
  <button class="primary" id="bAgStart">에이전트 시작</button><button id="bAgStop">에이전트 중지</button>
  <button id="bCollect">지금 1회 수집</button><button class="danger" id="bCollectKill">수집 강제 중지</button></div>
 <div class="msg muted">에이전트는 <b>켜 두는 것</b>입니다: 매일 정해진 시각에 수집하고 그 사이에는 대기합니다 (대기 = 정상, 끄려면 '에이전트 중지', 한 번만 수집하려면 '지금 1회 수집'). 시작할 때 놓친 수집이 있으면 바로 1회 수집합니다. 수집 중에도 학습데이터 생성을 함께 돌릴 수 있습니다.</div>
 <div class="msg" id="agMsg"></div><pre id="agLog"></pre></div>
</div>
<div class="card" id="mCard"><h2>모델 설정 <span class="muted" style="font-weight:400;font-size:13px" id="mHdr"></span><a id="mReload" style="float:right;font-size:13px;font-weight:400;cursor:pointer">다시 불러오기</a></h2>
 <div class="muted" style="font-size:12.5px">학습데이터 생성(3.2)·검토(3.3)에 쓰는 모델입니다. 저장하면 LLM_설정.md 의 해당 칸만 바뀌고 <b>다음 실행부터</b> 적용됩니다. 생성·다시 만들기가 도는 동안은 저장할 수 없습니다 (실행 중인 단계가 이 파일을 읽음). 키 값은 보이지 않고 .env 에 있는지만 봅니다. '연결 확인'은 아주 짧은 요청 1개를 보냅니다 (API 비용 또는 Claude 구독 사용량이 조금 듦). <b>추론 강도</b>는 openai(reasoning_effort)·claude_cli(--effort)에만 쓰고 비우면 모델 기본값이며, 검토 대체 모델도 검토 추론 강도를 씁니다 (낮출수록 빠르고 토큰이 적지만 판단이 얕아질 수 있음). 한 번만 다른 모델로 돌리려면 위 '이번 실행만 다른 모델'을 쓰세요 (저장 안 됨).</div>
 <div id="mBody" style="margin-top:8px;overflow-x:auto" class="muted">불러오는 중…</div>
 <div class="row" style="margin-top:8px;align-items:center">
  <label>생성 동시 요청 수<input id="mConcGen" type="number" min="1" max="16" style="width:70px"></label>
  <label>검토 동시 요청 수<input id="mConcRev" type="number" min="1" max="32" style="width:70px"></label>
  <button class="primary" id="mSave">저장</button><button id="mReset">지금 값으로 되돌리기</button><span class="msg" id="mMsg"></span></div>
 <div id="mHint" style="font-size:12.5px;margin-top:4px"></div>
 <details style="margin-top:6px"><summary>바꾼 기록 (최근 10)</summary><div id="mLog" style="font-size:12.5px"></div></details></div>
<div class="card"><h2>검토에서 뺀 샘플 <span class="muted" style="font-weight:400" id="exHdr"></span><a id="exCsv" style="float:right;font-size:13px;font-weight:400">CSV 받기</a></h2>
 <div class="muted" style="font-size:12.5px">생성한 샘플 중 검토(3.3)에서 빠진 것입니다. <b>검토 모델이 뺀 것</b>은 이유 문구로 나눕니다: <b>안전상 이유</b> = 공격 요청을 '안전하게 보존·수정할 수 없다'며 뺀 것 (gpt-6-luna 에 많음) · 룰 조건상 불가 = 룰 조건끼리 모순이거나 응답 방향 조건 등 요청으로 만들 수 없음 · 중복·다양성 · 기타. <b>응답 거절</b> = 검토 모델이 같은 모델로 '재시도 횟수'(3번) 물어도 그 룰을 판정하지 않음 (거절·응답 중단). <b>코드 검사</b> = 검토 뒤 룰 조건 대조에서 라벨과 어긋나 뺀 것. 공격 유형을 누르면 그 유형의 이유를 봅니다.<br><b>다시 만들기</b> = 응답 거절이거나 안전상 이유로 뺀 샘플이 있는 룰을 '검토 대체 모델'(LLM_설정.md)로 다시 검토하고 증강·학습 로그·진행 현황까지 다시 만듭니다 (API 비용 또는 Claude 구독 사용량). <b>검토 대체 모델은 이 버튼을 누를 때만 씁니다</b> (보통 생성 실행에서는 쓰지 않음). 다시 만든 샘플은 신규 학습데이터에 들어가 이 목록에서 빠지고, 다시 검토 뒤에도 빠진 샘플만 남습니다. 룰 조건상 불가·코드 검사 등으로만 빠진 샘플은 다시 만들지 않습니다.</div>
 <div class="row" style="margin-top:8px;align-items:center">
  <label>입력 날짜<select id="exDate"></select></label>
  <label>검토 모델<select id="exModel"><option value="">전체</option></select></label>
  <label>분류<select id="exCat"><option value="">전체</option><option>안전상 이유</option><option>응답 거절</option><option>룰 조건상 불가</option><option>중복·다양성</option><option>기타 검토</option><option>코드 검사</option><option>형식</option></select></label>
  <button id="exRedo" disabled>다시 만들기</button><span class="msg" id="exRedoMsg"></span>
 </div>
 <div id="exRedoInfo" style="font-size:12.5px;margin-top:6px"></div>
 <div class="vsum" id="exSum"></div>
 <div id="exBody" style="margin-top:8px;overflow-x:auto"></div>
 <div id="exReasons" style="margin-top:10px"></div></div>
</section>

<!-- 3. 룰별 상태 -->
<section class="page" id="p-rules"><div class="card"><h2>룰별 상태 <a href="/download/rule_status.csv" style="float:right;font-size:13px;font-weight:400">CSV 내려받기</a></h2>
 <div class="row"><label>상태<select id="rFlag"><option value="">전체</option></select></label>
  <label>출처<select id="rSrc"><option value="">전체</option><option value="talos">talos</option><option value="et">et</option><option value="custom">custom</option></select></label>
  <label>검색 (룰 id·이름·유형)<input id="rQ" style="width:260px"></label><span class="muted" id="rCount"></span></div>
 <div style="overflow-x:auto"><table style="margin-top:10px"><thead><tr><th>룰 id</th><th>상태</th><th>공격 유형</th><th>룰 이름</th><th>날짜 폴더 / 요청</th><th class="n">anomalies</th><th class="n">normal</th><th class="n">기존</th></tr></thead><tbody id="rBody"></tbody></table></div>
 <div class="pager"><button id="rPrev">이전</button><button id="rNext">다음</button></div></div></section>

<!-- 4. 신규 학습 데이터 -->
<section class="page" id="p-data"><div class="card"><h2>신규 학습 데이터 <span class="muted" style="font-weight:400" id="dSum"></span></h2>
 <div class="row">
  <label>날짜<select id="dDate"></select></label>
  <label>장비<select id="dDev"><option value="ips">IPS</option><option value="waf">WAF</option><option value="fw">방화벽(FW)</option><option value="web">웹 로그</option></select></label>
  <label>label<select id="dLabel"><option value="">전체</option><option>anomalies</option><option>normal</option></select></label>
  <label>룰 id<input id="dRule" placeholder="예: et:1:2001044" style="width:150px"></label>
  <label>payload 검색<input id="dQ" style="width:200px"></label>
  <label>label 정합성<select id="dCons"><option value="">전체</option><option value="OK">정합</option><option value="RULE">정합 (룰 기준)</option><option value="CHECK">확인 필요</option><option value="BAD">불일치</option></select></label>
  <label>보류 대상<select id="dHold"><option value="">전체</option><option value="rule">'정합 (룰 기준)' 공격</option><option value="pyua">python-requests UA 정상</option></select></label>
  <button id="dGo">조회</button><button id="dVerifyNew">검증 안 한 데이터만 검증</button><button id="dVerify">전체 검증</button><button class="danger" id="dVerifyStop" style="display:none">검증 중지</button><button id="dSave" style="display:none" title="고른 날짜(전체면 모든 날짜)의 데이터 전체를 08_dataset/&lt;날짜&gt;/ 에 로그 + label 파일로 저장">파일로 저장</button><a id="dDown" style="font-size:13px;align-self:center">CSV 내려받기</a>
 </div>
 <div class="vsum" id="dVsum"></div><div class="vsum" id="dSaveSum" style="display:none"></div>
 <div class="vsum" id="dEdits"></div><div id="dEditList" style="display:none"></div>
 <div class="muted" style="margin-top:8px;font-size:12.5px" id="dHelp"></div>
 <div id="dBody" style="margin-top:10px;overflow-x:auto"></div>
 <div class="pager"><button id="dPrev">이전</button><span class="muted" id="dCount"></span><button id="dNext">다음</button></div></div></section>
<div class="modal" id="edBox"><div class="mbox"><h3>학습데이터 수정 <span class="muted" id="edWhere" style="font-weight:400;font-size:13px"></span></h3>
 <div class="row"><label>label<select id="edLabel"><option>anomalies</option><option>normal</option></select></label></div>
 <textarea id="edPay" spellcheck="false"></textarea>
 <div class="muted" style="font-size:12.5px">payload 는 줄바꿈을 그대로 입력합니다 (로그에는 두 글자 \n 으로 저장). '다시 확인' 으로 지금 룰로 검증해 본 뒤 저장하세요. 저장한 수정은 '학습 로그에 반영' 때 학습 로그에 들어가고, 목록에서 되돌릴 수 있습니다.</div>
 <div id="edRes"></div>
 <div class="row"><button id="edCheck">다시 확인</button><button class="primary" id="edSave">저장</button><button id="edCancel">취소</button></div></div></div>

<!-- 5. 피처 -->
<section class="page" id="p-features"><div class="card"><h2>피처 <span class="muted" style="font-weight:400" id="fSum"></span></h2>
 <div style="font-size:13px" id="fIntro" class="muted">불러오는 중… (화면을 켜고 처음에는 공격 룰 1.6만 개를 읽느라 10초쯤 걸립니다)</div>
 <div class="row" style="margin-top:12px">
  <label>묶음<select id="fGroup"><option value="">전체</option></select></label>
  <label>검색 (이름·설명·패턴)<input id="fQ" style="width:240px"></label>
  <label>켜진 줄 수 기준<select id="fDev"><option value="ips">IPS 신규 학습 데이터</option><option value="waf">WAF 신규 학습 데이터</option></select></label>
  <button id="fCount">켜진 줄 수 다시 계산</button><span class="muted" style="align-self:center;font-size:12.5px" id="fCountMsg"></span>
 </div>
 <div style="margin-top:10px;overflow-x:auto"><table><thead><tr><th>피처</th><th>이름</th><th>무엇을 보는지</th><th>예</th><th>값</th><th class="n">켜진 줄<br>anomalies</th><th class="n">켜진 줄<br>normal</th><th class="n">켜졌을 때<br>anomalies 비율</th><th>패턴 (4.1 코드 그대로)</th></tr></thead><tbody id="fBody"></tbody></table></div>
</div></section>

<!-- 6. 기존 학습데이터와 비교 -->
<section class="page" id="p-sup">
<div class="card"><h2>기존 학습데이터와 비교 <span class="muted" style="font-weight:400" id="sHdr"></span></h2>
 <div class="muted" style="font-size:13px">신규 학습 데이터(4.)를 <b>기준 버전</b>(기존 학습데이터)과 비교해 샘플마다 등급을 매기고 → 학습에 넣을 샘플을 골라 <b>버전</b>(예: 26 하반기)에 담은 뒤 → 사람이 확인하고 <b>확정</b>하면 기존 + 신규 학습 파일(CSV)·로그를 만듭니다. 다음 버전(27 상반기)은 확정한 버전을 기준으로 다시 비교하고, 이미 비교한 샘플은 빼고 새로 들어온 샘플만 봅니다.<br>
 <b>등급</b> (앞이 우선) <span class="chip g0">제거</span> 중복(숫자만 다른 것 포함) 또는 기준에 같은 payload · <span class="chip g1">A</span> 기준 모델(기준 데이터로 학습한 RF·LGBM) 중 하나라도 틀림 = 기준 데이터에 부족한 데이터 · <span class="chip g2">B</span> 기준에 없던 피처 패턴(모델은 맞힘) · <span class="chip g3">C</span> 이미 학습된 패턴</div>
 <div id="sSetup"></div>
 <div class="row" style="margin-top:10px">
  <label>기준 버전<select id="sBase"></select></label>
  <label class="inl"><input type="checkbox" id="sInc"> 확정된 버전에서 이미 비교한 데이터도 다시 비교</label>
  <button class="primary" id="sRunNew">새로 생긴 데이터만 비교</button><button id="sRunAll">전체 다시 비교</button><button class="danger" id="sStop" style="display:none">중지</button>
 </div>
 <div class="est" id="sEst"></div>
 <div id="sJob" style="margin-top:8px"></div>
 <details><summary>실행 로그</summary><pre id="sLog"></pre></details></div>
<div class="card"><h2>비교 결과 <span class="seg" id="sDevSeg" style="vertical-align:middle;margin-left:6px"><button data-d="ips" class="on">IPS</button><button data-d="waf">WAF</button></span> <span class="muted" style="font-weight:400" id="sResHdr"></span><a id="sXlsx" style="float:right;font-size:13px;font-weight:400">엑셀 받기</a></h2>
 <div id="sRes" class="muted">비교 결과가 없습니다.</div></div>
<div class="card"><h2>선별 → 버전 후보 <span class="muted" style="font-weight:400" id="sSelHdr"></span></h2>
 <div class="muted" style="font-size:12.5px"><b>IPS·WAF 에 같은 샘플</b>을 넣습니다: 두 장비 비교 결과를 샘플로 짝지어 <b>합친 등급</b>으로 고릅니다 (A = 어느 장비든 A · B = A 가 아니고 어느 장비든 B · C = 그 밖 · 제거 = 두 장비 모두 중복, 한 장비만 기준과 중복이면 다른 장비 등급). 패턴당 상한은 장비마다 그 장비의 피처 패턴으로 세어 고른 뒤 합칩니다. 기본값: <b>A 전부 + B·C 패턴당 20개</b> (패턴이 넘치면 룰을 돌아가며 고름). 바꾸면 바로 다시 셉니다. 룰별 표에서 룰 단위로 빼거나 넣고, '후보 CSV 받기'로 payload 를 확인한 뒤 담으세요. 확정은 아래 버전 표에서 합니다.</div>
 <div class="row" style="margin-top:8px">
  <label>A 부족한 데이터<select id="oA"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option></select></label>
  <label>B 새 패턴<select id="oB"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option></select></label>
  <label>C 이미 학습됨<select id="oC"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option></select></label>
  <label class="inl"><input type="checkbox" id="oBad"> '불일치' 빼기</label>
  <label class="inl"><input type="checkbox" id="oRule"> '정합 (룰 기준)' 공격 빼기 <span class="muted">(보류)</span></label>
  <label class="inl"><input type="checkbox" id="oPy"> python-requests UA 정상 빼기 <span class="muted">(보류)</span></label>
 </div>
 <div class="hold" id="sHold"></div>
 <div class="row" style="margin-top:6px">
  <label>보기<select id="sView"><option value="">선별 전체</option><option value="rule">보류 데이터만: '정합 (룰 기준)' 공격</option><option value="pyua">보류 데이터만: python-requests UA 정상</option></select></label>
 </div>
 <div class="row" style="margin-top:6px;align-items:center;font-size:13px"><span class="muted">날짜</span><span id="cDates"></span><span class="muted" style="margin-left:10px">출처</span><span id="cSrc"></span></div>
 <div id="sPrev" style="margin-top:8px"></div>
 <div class="row" style="margin-top:10px"><label>룰 검색 (id·유형)<input id="sRuleQ" style="width:200px"></label><label>룰 표<select id="sRuleF"><option value="">A 많은 순</option><option value="ov">조정한 룰</option><option value="sel0">하나도 안 고른 룰</option></select></label><span class="muted" id="sRuleCnt" style="align-self:center;font-size:12.5px"></span></div>
 <div id="sRules" style="margin-top:6px;max-height:520px;overflow:auto"></div>
 <div class="row" style="margin-top:12px"><label>버전 이름<input id="sVName" placeholder="예: 26 하반기" style="width:160px"></label><button class="primary" id="sSave">버전 후보에 담기 (IPS·WAF)</button><a id="sCand" style="font-size:13px;align-self:center">후보 CSV 받기 (지금 장비)</a><span class="muted" id="sSaveMsg" style="align-self:center;font-size:13px"></span></div></div>
<div class="card"><h2>버전 <span class="muted" style="font-weight:400" id="sVerHdr"></span></h2><div id="sVers" style="overflow-x:auto"></div>
 <details style="margin-top:10px"><summary>기준 버전 등록 (기존 학습데이터 · 넣을 것)</summary><div id="sReg"></div></details></div>
</section>
</main>
<script>
const H={"Content-Type":"application/json","X-RuleData-UI":"1"};
const $=id=>document.getElementById(id), fmt=n=>(n??0).toLocaleString("ko-KR");
async function post(u,b){const r=await fetch(u,{method:"POST",headers:H,body:JSON.stringify(b||{})});const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
async function get(u){const r=await fetch(u);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
let S=null,P=null,M=null,flagNames={},rOff=0,page="progress";
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function chip(f){return `<span class="chip f-${f}">${esc(flagNames[f]||f)}</span>`}
/* ── 페이지 전환 ── */
function route(){const parts=location.hash.replace(/^#\/?/,"").split("/"),h=parts[0];page={"":"progress",run:"run",rules:"rules",data:"data",features:"features",sup:"sup"}[h]||"progress";
 document.querySelectorAll(".page").forEach(s=>s.classList.toggle("on",s.id==="p-"+page));
 document.querySelectorAll("nav a").forEach(a=>a.classList.toggle("on",a.dataset.p===page));
 if(page==="rules")loadRules();if(page==="data")initData();if(page==="run"||page==="progress")loadState();if(page==="run"){loadExcluded();loadModels()}if(page==="progress")loadProgress();
 if(page==="features")loadFeatures(parts[1]?decodeURIComponent(parts[1]):"");if(page==="sup")loadSup()}
window.addEventListener("hashchange",route);
/* ── 2. 생성·수집 ── */
async function loadState(){S=await get("/api/state");const g=S.generation,c=S.collect,a=S.agent,run=g.run,rdo=S.redo||{},redo=String((g.lock||{}).by||"").includes("redo");
 const pct=g.running&&run&&run.state==="running"?run.pct+"%":"";
 $("hdr").textContent=`입력 ${S.input_dates.at(-1)||"-"} · 생성 ${S.models["생성"]||"?"} / 검토 ${S.models["검토"]||"?"} · ${S.now.replace("T"," ")}`+(g.running?` · ${redo?"다시 만들기":"생성"} 실행 중 ${pct}`:"")+(a.running?((a.status||{}).state==="수집 중"?" · 수집 중":" · 수집 에이전트 켜짐 (대기)"):"");
 $("navRun").hidden=!(g.running||c.running||(a.running&&(a.status||{}).state==="수집 중"));$("navPct").textContent=pct;   // 대기 중인 에이전트는 표시 안 함
 document.title=(pct?`(${pct}) `:"")+"Snort 룰 학습데이터";const sv=S.supervised||{};$("navSup").hidden=!sv.running;$("navSupPct").textContent=sv.running&&sv.pct!=null?Math.floor(sv.pct)+"%":"";
 if(page==="progress"){const show=g.running||(run&&!run.error);$("runCard").hidden=!show;
  if(show){$("runTitle").textContent=(g.running?"지금 실행 중인 학습데이터 생성":"마지막 학습데이터 생성")+(run&&run.gen&&!run.dry_run?` (이번 실행: ${run.asked?`요청 ${fmt(run.asked)}개`:"남은 요청 전부"})`:"");$("runBody").innerHTML=runHtml(g)}}
 if(page!=="run")return;
 $("genRun").innerHTML=runHtml(g);
 const sel=$("gDate");if(sel.options.length-1!==S.input_dates.length){sel.innerHTML='<option value="">가장 최근</option>'+S.input_dates.slice().reverse().map(d=>`<option>${d}</option>`).join("")}
 $("genDot").className="dot"+(g.running||rdo.starting?" on":"");
 $("genStatus").textContent=rdo.starting&&!g.running?"다시 만들기 시작하는 중… (다시 검토할 요청을 찾는 중, 1분쯤)":g.running&&redo?`다시 만들기 실행 중 (검토에서 뺀 샘플 다시 검토) · PID ${g.lock.pid} · ${String(g.lock.started_at||"").replace("T"," ")} 시작 · 멈추려면 '강제 중지'`:g.running?(g.starting?"시작하는 중…":`실행 중 · 단계 ${g.lock.step||"-"} · 입력 ${g.lock.input_date||"-"} · PID ${g.lock.pid} · ${String(g.lock.started_at||"").replace("T"," ")} 시작`+(g.stop_requested?" · 중지 요청됨":"")):(c.running||(a.running&&(a.status||{}).state==="수집 중")?"대기 · 지금 수집 중이지만 생성을 함께 시작할 수 있습니다 (시작할 때의 입력 날짜를 끝까지 씀)":"대기");
 for(const id of["bDry","bPilot","bGen"])$(id).disabled=g.running||rdo.starting;$("bStop").disabled=!g.running||redo;$("bKill").disabled=!g.running;
 if(EX)exRedoUi();if(M)mHints();
 if(document.activeElement!==$("gN"))estimate();
 const ast=a.status||{},busy=c.running||(a.running&&ast.state==="수집 중");$("agDot").className="dot"+(busy?" on":a.running?" idle":"");
 $("agStatus").innerHTML=!a.running?(c.running?"<b>지금 1회 수집 중</b>":"에이전트 꺼짐"):busy?`<b>에이전트 켜짐 · 지금 수집 중</b> (매일 ${esc(a.lock.at||ast.at||"?")})`+(a.stop_requested?" · 중지 요청됨 (이번 수집을 마치고 꺼짐)":""):
  `에이전트 켜짐 · 매일 ${esc(a.lock.at||ast.at||"?")} 자동 수집 · <b>지금은 다음 수집까지 대기 중</b> <span class="muted">(정상 - 수집 작업은 끝났고, 다음 수집 시각까지 이 상태로 켜져 있음)</span>`+(a.stop_requested?" · 중지 요청됨 (곧 꺼짐)":"");
 const lc=S.last_collect||{};
 $("agKv").innerHTML=`<b>다음 수집</b><span>${a.running&&ast.next_run_at?ast.next_run_at.replace("T"," "):"-"}</span><b>마지막 수집</b><span>${lc.last_run_at?lc.last_run_at.replace("T"," ")+(lc.last_code===0?" (정상)":" (실패 — 로그 확인)"):"기록 없음"}</span><b>지금 수집 중</b><span>${c.running?`예 (PID ${c.lock.pid||c.ui.pid}, ${c.lock.by||""})`:"아니오"}</span>`;
 $("bAgStart").disabled=a.running;$("bAgStop").disabled=!a.running;$("bCollect").disabled=c.running;$("bCollectKill").disabled=!c.running;
 $("genLog").textContent=(await get("/api/log?job=generation")).text;
 $("agLog").textContent=(await get("/api/log?job="+(c.running?"collect":"agent"))).text;
 for(const id of["genLog","agLog"])$(id).scrollTop=1e9;}
/* 이번 실행의 진행률 (서버가 3_0·3_2·3_3 로그만 읽어 계산) */
function hms(s){s=Math.max(0,Math.round(s||0));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}시간 ${m}분`:m?`${m}분 ${s%60}초`:`${s}초`}
function clock(sec){const d=new Date(Date.now()+sec*1000),t=d.toTimeString().slice(0,5);return d.toDateString()===new Date().toDateString()?t:`${d.getMonth()+1}/${d.getDate()} ${t}`}
function runHtml(g){const r=g.run;
 if(g.starting)return '<div class="muted">시작하는 중…</div>';
 if(!r)return '<div class="muted">아직 실행 기록이 없습니다.</div>';
 if(r.error)return `<div class="muted">진행률을 계산하지 못했습니다: ${esc(r.error)}</div>`;
 const live=r.state==="running",G=r.gen,R=r.rev,icon={done:"✔",running:"●",fail:"✘",skip:"–",wait:"○"},word={done:"끝남",running:"진행 중",fail:"실패·중단 (로그 확인)",skip:"건너뜀",wait:"대기"};
 const s33=r.steps.find(s=>s.key==="3.3"),revWait=!s33||s33.status==="wait";
 const head=live?`남은 시간 <b>약 ${hms(r.remaining_s)}</b> · <b>${clock(r.remaining_s)}</b> 쯤 끝날 예상 · 경과 ${hms(r.elapsed_s)}`:`<b>${r.state==="done"?"끝남":"중단됨 (끝까지 못 감)"}</b> · 걸린 시간 ${hms(r.elapsed_s)}`;
 return `<div class="row" style="align-items:center;gap:16px;flex-wrap:nowrap"><div class="pct">${r.pct}%</div><div style="flex:1;min-width:0">
  <div class="bar${live?" live":""}" style="margin:4px 0 6px"><i style="width:${r.pct}%"></i></div>
  <div style="font-size:13px">${head}</div><div class="muted" style="font-size:12.5px">${esc(r.started_at)} 시작${r.dry_run?" · 건수 확인 (dry-run, 호출 없음)":""}${r.by!=="3_0"?(r.by.includes("redo")?" · 다시 만들기 (검토에서 뺀 샘플 다시 검토)":` · ${esc(r.by)} 단독 실행`):""}${r.stop_requested?" · <b>중지 요청됨</b> (지금 요청까지 하고 멈춤)":""}</div></div></div>
  <div class="steps">${r.steps.map(s=>`<span class="step ${s.status}" title="${word[s.status]}">${icon[s.status]} ${s.key} ${esc(s.name)}${s.detail?` <b>${s.detail}</b>`:""}</span>`).join("")}</div>
  ${live&&r.current?`<div class="runnow">지금: <b>${esc(r.current)}</b></div>`:""}
  ${r.dry_run?"":`<div class="kv" style="margin-top:6px"><b>생성</b><span>${fmt(G.ok)}개 성공 · ${fmt(G.fail)}개 실패 / 이번 ${fmt(G.target)}개${G.ok+G.fail?` · 약 ${hms(G.sec_per_req)}마다 1개 완료 · 토큰 ${fmt(G.tokens_in+G.tokens_out)} (입력 ${fmt(G.tokens_in)} / 출력 ${fmt(G.tokens_out)})`:""}</span>
  <b>검토</b><span>${revWait?`예정 ${fmt(R.target)}개 (생성 성공분 + 이전 대기분)`:`${fmt(R.ok)}개 성공 · ${fmt(R.fail)}개 실패 / ${fmt(R.target)}개${R.ok+R.fail?` · 약 ${hms(R.sec_per_req)}마다 1개 완료 · 토큰 ${fmt(R.tokens_in+R.tokens_out)}`:""}`}</span>
  ${G.check_ok+G.check_fail?`<b>생성 품질</b><span>코드 검사 통과 <b>${(100*G.check_ok/(G.check_ok+G.check_fail)).toFixed(1)}%</b> (샘플 ${fmt(G.check_ok)} / ${fmt(G.check_ok+G.check_fail)})${R.ok?` · 검토가 고친 샘플 요청당 <b>${(R.fixed/R.ok).toFixed(1)}개</b>, 뺀 샘플 ${(R.dropped/R.ok).toFixed(1)}개`:""} <span class="muted">(정규식 예시·요청 줄 자동 수정 전 100개 실행: 통과 69.5% · 고친 샘플 요청당 11.6개)</span></span>`:""}
  ${r.parallel?`<b>동시 처리</b><span>생성 ${live?`지금 ${r.parallel.generate_now}개 (최대 ${r.parallel.generate})`:`최대 ${r.parallel.generate}개`} · 검토 ${live?`지금 ${r.parallel.review_now}개 (최대 ${r.parallel.review})`:`최대 ${r.parallel.review}개`}${r.parallel.overlap?" · 생성된 요청부터 바로 검토":""}${live?" · 속도 제한에 맞춰 자동 조절":""}</span>`:""}</div>
  <div class="muted" style="font-size:12px;margin-top:4px">%·남은 시간은 이번 실행에서 실제로 요청이 끝나는 간격(동시 처리 반영)과 지난 실행의 준비·마무리 단계 시간으로 계산합니다 (응답 해석 실패·재시도가 많으면 늘어남). 토큰은 성공한 요청 기준.</div>`}`}
function genBody(extra){return Object.assign({set:$("gSet").value,max_requests:+$("gN").value,date:$("gDate").value,generator:$("gGen").value.trim(),reviewer:$("gRev").value.trim()},extra)}
function dur(sec){if(sec<3600)return `약 ${Math.max(1,Math.round(sec/60))}분`;const h=sec/3600;return h<48?`약 ${h.toFixed(1)}시간`:`약 ${(h/24).toFixed(1)}일 (${fmt(Math.round(h))}시간)`}
function estimate(){const el=$("genEst");if(!P){el.textContent="예상: 진행 현황을 불러오는 중…";return}
 const av=P.average_per_request||{},rq=P.requests||{},set=$("gSet").value;let left=((rq.pending_by_set||{})[set])??rq.pending??0;
 if(!left&&set==="all")left=(P.estimate_remaining||{}).requests||0;   // 새 입력 날짜라 요청 파일이 아직 없으면 (다음 실행의 3.1 이 만듦) 남은 룰 수로 추정
 const asked=+$("gN").value||0,all=asked===0,n=all?left:Math.min(asked,left);
 if(!av.generate_seconds){el.innerHTML="<b>예상</b> 생성 기록이 아직 없어 계산할 수 없습니다 — '파일럿 5개'를 먼저 돌리면 실제 평균으로 계산합니다.";return}
 // 동시 처리: 생성은 pg 개씩, 검토는 pr 개씩. 함께 진행하면 생성된 요청부터 바로 검토 → 늦게 끝나는 쪽 기준
 // 모델: '이번 실행만' 입력 > 저장된 설정. 그 모델의 지난 기록(5건 이상)이 있으면 그 평균, 없으면 전체 평균(다른 모델 포함)
 const ov={g:$("gGen").value.trim(),r:$("gRev").value.trim()},gm=ov.g||(S&&S.models["생성"])||"생성 모델",rm=ov.r||(S&&S.models["검토"])||"검토 모델",noRev=rm==="none";
 const raw=(role,m)=>(((M&&M.stats)||{})[role]||{})[m],ok5=h=>h&&h.n>=5?h:null,hg=ok5(raw("생성",gm)),hr=ok5(raw("검토",rm));
 const par=(S&&S.settings)||{generate:1,review:1,overlap:false},g1=hg&&hg.sec_n>=5?hg.sec/hg.sec_n:av.generate_seconds||0,r1=noRev?0:hr&&hr.sec_n>=5?hr.sec/hr.sec_n:av.review_seconds||0;
 const warn=[];if(M)for(const [role,m,h] of [["생성",gm,hg],...(noRev?[]:[["검토",rm,hr]])]){const n0=(raw(role,m)||{}).n||0;
  if(!h)warn.push(`${role} ${esc(m)}: 지난 기록 ${n0?fmt(n0)+"건뿐":"없음"} → 시간·토큰을 다른 모델을 포함한 전체 평균으로 계산 (실제와 다를 수 있음)`);
  else if(h.sec_n<5)warn.push(`${role} ${esc(m)}: 걸린 시간 기록이 없어 시간은 전체 평균으로 계산`)}
 $("gOvr").textContent=ov.g||ov.r?`(지금 입력: ${[ov.g&&"생성 "+ov.g,ov.r&&"검토 "+ov.r].filter(Boolean).join(" · ")} - 이번 실행만, 저장 안 됨)`:`(저장 안 됨 · 비우면 저장된 값: 생성 ${(S&&S.models["생성"])||"?"} / 검토 ${(S&&S.models["검토"])||"?"})`;
 if(ov.g||ov.r)$("gOvrBox").open=true;
 const tg=Math.ceil(n/par.generate)*g1,tr=Math.ceil(n/par.review)*r1,sec=!n?0:par.overlap?Math.max(tg+r1,g1+tr):tg+tr;
 // 준비·마무리: 지난 실행 기록 (증강은 요청 수에 비례). 아직 생성을 돌리지 않은 입력 날짜면 기존 로그 대조·요청 파일 만들기로 준비가 길다
 const ss=(S&&S.step_seconds)||{},day=$("gDate").value||((S&&S.input_dates)||[]).slice(-1)[0],first=(ss.first_dates||[]).includes(day);
 const prep=!n?0:first?(ss.prep_first??1240):(ss.prep??60),fin=!n?0:(ss.augment_per_request??0.6)*n+(ss.post??40);
 const gi=Math.round(n*(hg?hg.tin/hg.n:av.generator_input_tokens||0)),go=Math.round(n*(hg?hg.tout/hg.n:av.generator_output_tokens||0)),ri=noRev?0:Math.round(n*(hr?hr.tin/hr.n:av.reviewer_input_tokens||0)),ro=noRev?0:Math.round(n*(hr?hr.tout/hr.n:av.reviewer_output_tokens||0));
 el.innerHTML=`<b>예상</b>${ov.g||ov.r?' <span class="warn">(이번 실행만 바꾼 모델)</span>':""} 요청 ${fmt(n)}개${all?" (남은 전부)":""} → 시간 <b>${dur(sec+prep+fin)}</b> <span class="muted">(생성·검토 ${dur(sec)} + 준비 ${dur(prep)}${first?" (이 입력 날짜의 첫 실행 - 기존 데이터 대조·요청 파일 만들기)":""} + 마무리 ${dur(fin)} · 동시 최대 생성 ${par.generate}개·검토 ${par.review}개${par.overlap?" · 생성·검토 함께":""}, 하나씩이면 ${dur(n*(g1+r1))} · API 속도 제한에 걸리면 더 걸림)</span><br>토큰 합계 <b>${fmt(gi+go+ri+ro)}</b> = 생성 ${esc(gm)} <b>${fmt(gi+go)}</b> (입력 ${fmt(gi)} / 출력 ${fmt(go)}) + 검토 ${esc(rm)} <b>${fmt(ri+ro)}</b> (입력 ${fmt(ri)} / 출력 ${fmt(ro)})<div class="muted">기준: 생성 ${hg?`${esc(gm)} ${fmt(hg.n)}건`:`전체 ${av.generated_requests}건`} · 검토 ${noRev?"생략":hr?`${esc(rm)} ${fmt(hr.n)}건`:`전체 ${av.reviewed_requests}건`}의 실제 평균 (재시도 포함, 요청 1개 생성 ${Math.round(g1)}초·검토 ${Math.round(r1)}초) · 이 세트의 남은 요청 ${fmt(left)}개${!all&&asked>left?" (입력한 수가 남은 요청보다 많아 남은 수로 계산)":""}</div>${warn.length?`<div class="warn">${warn.join("<br>")}</div>`:""}`}
$("gN").oninput=estimate;$("gSet").onchange=estimate;$("gGen").oninput=$("gRev").oninput=estimate;
async function act(msgId,fn){try{const r=await fn();$(msgId).textContent="요청함 "+(r.pid?`(PID ${r.pid})`:"");setTimeout(loadState,800)}catch(e){$(msgId).textContent="실패: "+e.message}}
$("bDry").onclick=()=>act("genMsg",()=>post("/api/generation/start",genBody({dry_run:true})));
$("bPilot").onclick=()=>act("genMsg",()=>post("/api/generation/start",genBody({max_requests:5})));
$("bGen").onclick=()=>{const n=+$("gN").value;if(!confirm(n===0?"남은 요청 전부를 생성합니다 (API 비용·수일 소요). 진행할까요?":`요청 ${n}개를 생성합니다 (API 비용 발생). 진행할까요?`))return;act("genMsg",()=>post("/api/generation/start",genBody()))};
$("bStop").onclick=()=>act("genMsg",()=>post("/api/generation/stop",{}));
$("bKill").onclick=()=>{if(confirm("지금 하던 요청을 버리고 바로 멈춥니다. 진행할까요?"))act("genMsg",()=>post("/api/generation/stop",{force:true}))};
$("bAgStart").onclick=()=>act("agMsg",()=>post("/api/agent/start",{at:$("agAt").value.trim()}));
$("bAgStop").onclick=()=>act("agMsg",()=>post("/api/agent/stop",{}));
$("bCollect").onclick=()=>act("agMsg",()=>post("/api/collect/start",{}));
$("bCollectKill").onclick=()=>{if(confirm("수집을 바로 멈춥니다. 진행할까요?"))act("agMsg",()=>post("/api/collect/stop",{}))};
/* ── 2. 검토에서 뺀 샘플 (3.3 의 rejected: 검토 모델 drop · 코드 검사) ── */
let EX=null,exType="";
async function loadExcluded(){const qs=new URLSearchParams({date:$("exDate").value||"",model:$("exModel").value,cat:$("exCat").value,type:exType});
 $("exBody").style.opacity=.5;let r;try{r=await get("/api/excluded?"+qs)}catch(e){$("exBody").innerHTML=`<div class="muted">불러오지 못했습니다: ${esc(e.message)}</div>`;return}finally{$("exBody").style.opacity=1}
 EX=r;const cs=r.cats;
 const dopt=["all",...r.dates.slice().reverse()].map(d=>`<option value="${d}"${d===r.date?" selected":""}>${d==="all"?"전체 날짜":d}</option>`).join("");if($("exDate").innerHTML!==dopt)$("exDate").innerHTML=dopt;
 const mopt='<option value="">전체</option>'+r.models.map(m=>`<option${m===r.model?" selected":""}>${esc(m)}</option>`).join("");if($("exModel").innerHTML!==mopt)$("exModel").innerHTML=mopt;
 $("exHdr").textContent=`· ${r.date==="all"?"전체 날짜":r.date} · ${fmt(r.total)}개${r.model?" · "+r.model:""}`;
 $("exSum").innerHTML=cs.map(c=>`<span class="chip${r.cat===c?" c-RULE":""}" data-xc="${esc(c)}" style="cursor:pointer">${esc(c)} ${fmt(r.by_cat[c]||0)}</span>`).join("")+
  `<span class="muted">조건에 맞는 ${fmt(r.sel)}개 (공격 ${fmt(r.sel_label.anomalies)} / 정상 ${fmt(r.sel_label.normal)}) · 룰 ${fmt(r.rules)}개 · <b>공격 샘플이 모두 빠진 룰 ${fmt(r.lost_n)}개</b></span>`+
  (exType?` <span class="chip c-CHECK" data-xt="" style="cursor:pointer">공격 유형: ${esc(exType)} ✕</span>`:"");
 $("exBody").innerHTML=r.types.length?`<table class="mini"><thead><tr><th>공격 유형</th><th>method (IPS)</th><th class="n">뺀 샘플</th>${cs.map(c=>`<th class="n">${esc(c)}</th>`).join("")}<th class="n">공격</th><th class="n">정상</th><th class="n">룰</th><th class="n">공격이 모두<br>빠진 룰</th><th class="n">다시 만들기<br>대상</th></tr></thead><tbody>${r.types.map(t=>`<tr class="${t.type===exType?"hl":""}" data-xt="${esc(t.type)}" style="cursor:pointer"><td class="nw">${esc(t.name)} <span class="muted">${esc(t.type)}</span></td><td>${esc(t.method)}</td><td class="n"><b>${fmt(t.n)}</b></td>${cs.map(c=>`<td class="n">${t.cats[c]?fmt(t.cats[c]):""}</td>`).join("")}<td class="n">${fmt(t.att)}</td><td class="n">${fmt(t.norm)}</td><td class="n">${fmt(t.rules)}</td><td class="n">${t.lost?fmt(t.lost):""}</td><td class="n">${t.redo?fmt(t.redo):""}</td></tr>`).join("")}</tbody></table>`:'<div class="muted">빠진 샘플이 없습니다.</div>';
 exRedoUi();
 $("exReasons").innerHTML=(r.reasons.length?`<h4 style="margin:6px 0">${exType?esc(exType)+" · ":""}이유 상위 ${r.reasons.length}개 <span class="muted" style="font-weight:400">(검토 모델·코드 검사가 쓴 문구 그대로)</span></h4><table class="mini"><thead><tr><th class="n">수</th><th>분류</th><th>이유</th><th>룰 (예)</th></tr></thead><tbody>${r.reasons.map(x=>`<tr><td class="n">${fmt(x.n)}</td><td class="nw">${esc(x.cat)}</td><td class="reason">${esc(x.reason)}</td><td class="nw muted">${x.rules.map(esc).join(", ")}${x.rule_n>3?` 외 ${x.rule_n-3}`:""}</td></tr>`).join("")}</tbody></table>`:"")+
  (r.lost_n?`<details style="margin-top:6px"><summary>공격 샘플이 모두 빠진 룰 ${fmt(r.lost_n)}개 (검토 뒤 공격 씨앗이 없어 증강도 못 함)</summary><div class="muted" style="font-size:12.5px;word-break:break-all">${r.lost_rules.map(esc).join(", ")}${r.lost_n>r.lost_rules.length?" …":""}</div></details>`:"");
 $("exCsv").href="/download/excluded.csv?"+new URLSearchParams({date:r.date,model:r.model,cat:r.cat,type:exType})}
$("exDate").onchange=$("exModel").onchange=$("exCat").onchange=()=>loadExcluded();
document.getElementById("p-run").addEventListener("click",e=>{const c=e.target.closest("[data-xc]");if(c){$("exCat").value=$("exCat").value===c.dataset.xc?"":c.dataset.xc;loadExcluded();return}
 const t=e.target.closest("[data-xt]");if(t){exType=exType===t.dataset.xt?"":t.dataset.xt;loadExcluded()}});
/* 다시 만들기 (3_3 --redo-safety): 대상·결과·버튼. 생성과 같은 잠금이라 생성이 돌면 누를 수 없음 */
function exRedoUi(){const rd=EX.redo||{},g=(S&&S.generation)||{},st=((S&&S.redo)||{}).starting,mine=st||String((g.lock||{}).by||"").includes("redo"),busy=g.running||st;
 const b=$("exRedo"),scope=EX.date==="all"?"전체 날짜":EX.date,parts=[];
 b.textContent=rd.cand_req?`다시 만들기 (요청 ${fmt(rd.cand_req)}개)`:"다시 만들기";b.disabled=!rd.model||!rd.cand_req||busy;
 if(mine)parts.push(`<b>다시 만들기 실행 중</b> - 다시 검토한 요청부터 이 목록에서 빠집니다 (1분마다 새로). 학습 로그(08_dataset)에는 끝날 때 한 번에 반영 · 진행은 위 '학습데이터 생성'`);
 else if(busy)parts.push("학습데이터 생성이 실행 중이라 끝난 뒤 다시 만들 수 있습니다");
 parts.push(!rd.model?"LLM_설정.md 의 '검토 대체 모델' 이 비어 있어 다시 만들 수 없습니다":rd.cand_req?`<b>다시 만들 수 있음</b> (${esc(scope)}): 요청 ${fmt(rd.cand_req)}개 · 룰 ${fmt(rd.cand_rules)}개 · 그 룰에서 뺀 샘플 ${fmt(rd.cand_n)}개 (공격 ${fmt(rd.cand_att)}) → ${esc(rd.model)} 로 다시 검토`:`다시 만들 것이 없습니다 (${esc(scope)})`);
 if(rd.done_req)parts.push(`<b>다시 만든 요청 ${fmt(rd.done_req)}개</b> (마지막 ${esc(rd.last_at.replace("T"," "))}): 학습 샘플 <b>${rd.removed<0?"":"+"}${fmt(rd.removed)}개</b> (검토 통과, 증강 전 · 이 목록에서 빠짐) · 다시 검토 뒤에도 빠진 ${fmt(rd.left)}개는 목록에 남음`);
 $("exRedoInfo").innerHTML=parts.join("<br>")}
$("exRedo").onclick=()=>{const rd=(EX&&EX.redo)||{},d=EX?EX.date:"all",cli=String(rd.model).startsWith("claude_cli:");
 const cost=(cli?"Claude 구독 사용량을 씁니다 (로그인이 안 됐거나 사용 한도에 걸리면 그 실행은 멈춤)":"API 비용이 듭니다")+(rd.avg_tokens?` · 요청당 약 ${fmt(rd.avg_tokens)} 토큰 → 약 ${fmt(rd.avg_tokens*rd.cand_req)} 토큰 (이 모델로 다시 만든 ${fmt(rd.avg_n)}개 평균)`:"");
 if(!confirm(`${d==="all"?"전체 날짜":d}: 요청 ${fmt(rd.cand_req)}개 (룰 ${fmt(rd.cand_rules)}개)를 ${rd.model} 로 다시 검토해 학습데이터를 다시 만듭니다.\n안전상 이유로 뺀 샘플이 있는 룰만 다시 검토하고, 끝나면 증강·학습 로그·진행 현황까지 다시 만듭니다.\n${cost}. 진행할까요?`))return;
 act("exRedoMsg",()=>post("/api/excluded/redo",{date:d}))};
/* ── 2. 모델 설정 (LLM_설정.md 의 생성·검토·검토 대체 모델, 동시 요청 수) ── */
const MROWS=[["gen","생성"],["rev","검토"],["fb","검토 대체"]];
function mVal(k){return {provider:$("mP_"+k).value,model:$("mM_"+k).value.trim()}}
async function loadModels(refresh){let r;try{r=await get("/api/models"+(refresh?"?refresh=1":""))}catch(e){$("mBody").innerHTML=`<div class="muted">불러오지 못했습니다: ${esc(e.message)}</div>`;return}
 M=r;renderModels();estimate()}
function mOpts(sel,k){const names=Object.keys(M.providers).filter(p=>p!=="mock"||p===sel);if(sel&&!names.includes(sel)&&sel!=="none"&&sel!=="-")names.push(sel);
 return names.map(p=>`<option${p===sel?" selected":""}>${esc(p)}</option>`).join("")+(k==="rev"?`<option value="none"${sel==="none"?" selected":""}>none (검토 생략)</option>`:"")+(k==="fb"?`<option value="-"${sel==="-"?" selected":""}>사용 안 함</option>`:"")}
const MEFF=["minimal","low","medium","high","xhigh","max"];   // 추론 강도 (openai: xhigh·max → high, claude_cli: minimal → low)
function mEffOpts(sel){sel=sel==="-"?"":sel||"";return ["",...MEFF,...(sel&&!MEFF.includes(sel)?[sel]:[])].map(x=>`<option value="${esc(x)}"${x===sel?" selected":""}>${x?esc(x):"기본값 (비움)"}</option>`).join("")}
function renderModels(){const R=M.roles,Z={provider:"",model:"",temperature:"",max_tokens:""},fb=String(M.run.fb||"").trim(),on=fb.includes(":");
 const row=(k,name,v,eff)=>`<tr><td class="nw"><b>${name}</b></td><td><select id="mP_${k}" data-mk="${k}">${mOpts(v.provider,k)}</select></td><td><input id="mM_${k}" list="mL_${k}" value="${esc(v.model)}" style="width:160px"><datalist id="mL_${k}"></datalist></td>`+
  (k==="fb"?'<td colspan="3" class="muted">검토와 같음</td>':`<td><input id="mT_${k}" value="${esc(v.temperature)}" placeholder="비움" style="width:56px"></td><td><input id="mX_${k}" value="${esc(v.max_tokens)}" style="width:74px"></td><td><select id="mE_${k}">${mEffOpts(eff)}</select></td>`)+
  `<td id="mS_${k}"></td><td id="mH_${k}"></td><td class="nw"><button data-mc="${k}">연결 확인</button><div id="mC_${k}" style="font-size:12px;max-width:340px;white-space:normal"></div></td></tr>`;
 $("mBody").className="";
 $("mBody").innerHTML=`<table class="mini"><thead><tr><th>역할</th><th>provider</th><th>model</th><th>temperature</th><th>max_tokens</th><th>추론 강도</th><th>사용 가능</th><th>지난 기록 (요청당)</th><th></th></tr></thead><tbody>`+
  row("gen","생성",R["생성"]||Z,M.run.eff_gen)+row("rev","검토",R["검토"]||Z,M.run.eff_rev)+row("fb","검토 대체",on?{provider:fb.slice(0,fb.indexOf(":")),model:fb.slice(fb.indexOf(":")+1)}:{provider:"-",model:""})+"</tbody></table>";
 $("mConcGen").value=M.run.conc_gen;$("mConcRev").value=M.run.conc_rev;$("mHdr").textContent="· "+M.file;
 for(const [k] of MROWS)mRow(k);
 $("mLog").innerHTML=M.changes.length?`<table class="mini"><tbody>${M.changes.map(c=>`<tr><td class="nw muted">${esc(String(c.at).replace("T"," "))}</td><td>${(c.changes||[]).map(x=>`${esc(x[0])}: ${esc(x[1]||"(빈칸)")} → <b>${esc(x[2]||"(빈칸)")}</b>`).join("<br>")}</td></tr>`).join("")}</tbody></table>`:'<div class="muted">아직 없음</div>';
 mHints()}
function mAvail(p){const v=M.providers[p];if(!v)return '<span class="bad">provider 표에 없음</span>';
 if(v.style==="claude_cli"){const c=v.state||{};return !c.exe?`<span class="bad">claude 실행 파일 없음</span>${c.error?` <span class="muted">${esc(c.error)}</span>`:""}`:c.logged_in?`<span class="ok">로그인됨</span> <span class="muted">(${esc([c.method,c.plan,c.version&&"v"+c.version].filter(Boolean).join(" · "))})</span>`:c.logged_in===false?'<span class="bad">로그인 필요</span> <span class="muted">(터미널에서 claude → /login)</span>':`<span class="warn">${esc(c.error||"확인 못 함")}</span>`}
 if(v.style==="mock")return '<span class="muted">가짜 응답 (흐름 시험용)</span>';
 return v.state==="있음"?`<span class="ok">.env ${esc(v.key)} 있음</span>`:v.state==="형식 오류"?`<span class="bad">.env ${esc(v.key)} 값에 한글·공백</span>`:`<span class="bad">.env ${esc(v.key)} 없음</span>`}
function mRow(k){const v=mVal(k),off=v.provider==="none"||v.provider==="-",cli=(M.providers[v.provider]||{}).style==="claude_cli";
 $("mL_"+k).innerHTML=((M.known||{})[v.provider]||[]).map(x=>`<option value="${esc(x)}">`).join("");$("mM_"+k).disabled=off;
 if(k!=="fb")for(const f of["mT_","mX_"]){$(f+k).disabled=off||cli;$(f+k).title=cli?"claude_cli 는 쓰지 않음 (CLI 기본값)":""}
 if(k!=="fb"){const e=$("mE_"+k);e.disabled=off;e.title=v.provider==="openai"?"OpenAI reasoning_effort (minimal·low·medium·high, xhigh·max 는 high 로 보냄)":
  v.provider==="claude_cli"?"claude --effort (low·medium·high·xhigh·max, minimal 은 low 로 보냄)":"이 provider 는 추론 강도를 쓰지 않음 (openai·claude_cli 만)"}
 const h=(((M.stats||{})[k==="gen"?"생성":k==="rev"?"검토":"대체"])||{})[`${v.provider}:${v.model}`];
 $("mS_"+k).innerHTML=off?'<span class="muted">-</span>':mAvail(v.provider);
 $("mH_"+k).innerHTML=off?"":h?`${fmt(h.n)}건${h.sec_n?` · ${Math.round(h.sec/h.sec_n)}초`:""} · 토큰 ${fmt(Math.round(h.tin/h.n))} / ${fmt(Math.round(h.tout/h.n))}`+(k==="fb"?' <span class="muted">(일부 룰만)</span>':""):'<span class="warn">기록 없음</span>'}
function mHints(){if(!M||!$("mP_gen"))return;const h=[],g=mVal("gen"),r=mVal("rev"),f=mVal("fb"),lock=M.locked||(S&&(S.generation.running||(S.redo||{}).starting));
 if(lock)h.push("<b>학습데이터 생성(또는 다시 만들기)이 실행 중이라 저장할 수 없습니다</b> - 생성이 끝난 뒤 저장하세요 (실행 중인 단계가 이 파일을 읽음)");
 for(const [k,name] of MROWS){const v=mVal(k),pv=M.providers[v.provider];if(!pv||v.provider==="-"||v.provider==="none")continue;
  if(pv.key&&pv.state!=="있음")h.push(`${name} ${esc(v.provider)}: .env 에 ${esc(pv.key)} 가 ${pv.state==="형식 오류"?"올바르지 않아":"없어"} 실행이 실패합니다`);
  if(pv.style==="claude_cli"&&!(pv.state||{}).logged_in)h.push(`${name} claude_cli: ${(pv.state||{}).exe?"로그인이 필요합니다 (터미널에서 claude → /login)":"claude 실행 파일을 찾지 못했습니다"}`)}
 for(const [k,name] of [["gen","생성"],["rev","검토"]]){const v=mVal(k);if(v.provider==="openai"&&$("mT_"+k).value.trim())h.push(`${name} openai: 추론 모델(gpt-5 등)은 temperature 를 받지 않을 수 있습니다 - 비워 두기를 권장`)}
 for(const [k,name] of [["gen","생성"],["rev","검토"]]){const v=mVal(k),e=$("mE_"+k).value;if(!e||v.provider==="none")continue;
  if(v.provider!=="openai"&&v.provider!=="claude_cli")h.push(`${name} ${esc(v.provider)}: 추론 강도는 openai·claude_cli 에만 씁니다 (이 provider 에는 넘기지 않음)`);
  else if(v.provider==="openai"&&(e==="xhigh"||e==="max"))h.push(`${name} openai: '${e}' 는 OpenAI 에 없어 high 로 보냅니다`);
  else if(v.provider==="claude_cli"&&e==="minimal")h.push(`${name} claude_cli: 'minimal' 은 claude 에 없어 low 로 보냅니다`)}
 if($("mE_rev").value&&f.provider!=="-"&&f.provider!==r.provider&&(f.provider==="openai"||f.provider==="claude_cli"))h.push(`검토 대체 ${esc(f.provider)}: 검토 추론 강도(${esc($("mE_rev").value)})를 대체 검토에도 씁니다`);
 if(r.provider==="claude_cli"&&+$("mConcRev").value>8)h.push("검토 claude_cli: 호출마다 claude 프로세스가 뜨고 구독 사용 한도를 사람의 Claude 사용과 함께 써서 검토 동시 요청 수는 4~8 을 권장");
 if(r.provider==="none")h.push("검토 none: 검토 모델 없이 코드 검사만 합니다");
 if(f.provider!=="-"&&r.provider!=="none"&&`${f.provider}:${f.model}`===`${r.provider}:${r.model}`)h.push("검토 대체 모델이 검토 모델과 같습니다 - '다시 만들기' 는 처음 검토와 같은 모델로 다시 묻게 됩니다");
 $("mHint").innerHTML=h.map(x=>`<div class="warn">${x}</div>`).join("");$("mSave").disabled=!!lock}
function mBody(){const b={sig:M.sig,roles:{},run:{}};
 for(const [k,name] of [["gen","생성"],["rev","검토"]]){const v=mVal(k);b.roles[name]={provider:v.provider,model:v.model,temperature:$("mT_"+k).value.trim(),max_tokens:$("mX_"+k).value.trim()}}
 const f=mVal("fb");b.run={fb:f.provider==="-"?"-":`${f.provider}:${f.model}`,conc_gen:$("mConcGen").value.trim(),conc_rev:$("mConcRev").value.trim(),
  eff_gen:$("mE_gen").value,eff_rev:$("mE_rev").value};return b}
$("mSave").onclick=async()=>{if(!M||!$("mP_gen"))return;const b=mBody(),R=M.roles,diff=[],blank=x=>x||"(빈칸)";
 for(const name of["생성","검토"])for(const f of["provider","model","temperature","max_tokens"]){const o=String((R[name]||{})[f]||""),n=b.roles[name][f];if(o!==n)diff.push(`${name} ${f}: ${blank(o)} → ${blank(n)}`)}
 const fo=String(M.run.fb||"").trim();if(fo!==b.run.fb&&!(["","-","none"].includes(fo)&&b.run.fb==="-"))diff.push(`검토 대체 모델: ${blank(fo)} → ${b.run.fb==="-"?"사용 안 함 (-)":b.run.fb}`);
 for(const [k,name] of [["conc_gen","생성 동시 요청 수"],["conc_rev","검토 동시 요청 수"]])if(String(M.run[k])!==b.run[k])diff.push(`${name}: ${M.run[k]} → ${b.run[k]}`);
 for(const [k,name] of [["eff_gen","생성 추론 강도"],["eff_rev","검토 추론 강도"]]){const o=String(M.run[k]||"").replace(/^-$/,""),n=b.run[k];if(o!==n)diff.push(`${name}: ${o||"기본값 (비움)"} → ${n||"기본값 (비움)"}`)}
 if(!diff.length){$("mMsg").textContent="바뀐 것이 없습니다";return}
 if(!confirm(`LLM_설정.md 를 바꿉니다 (다음 생성·검토 실행부터 적용):\n\n${diff.join("\n")}\n\n저장할까요?`))return;
 try{const r=await post("/api/models/save",b);$("mMsg").textContent=r.changed.length?`저장함 (${r.changed.length}개 칸) - 다음 실행부터 적용`:"바뀐 것이 없습니다";await loadModels();loadState()}catch(e){$("mMsg").textContent="저장 못 함: "+e.message}};
$("mReset").onclick=()=>{if(M){renderModels();$("mMsg").textContent="지금 값으로 되돌림"}};
$("mReload").onclick=()=>{$("mMsg").textContent="다시 불러오는 중… (claude 로그인 확인 포함)";loadModels(true).then(()=>$("mMsg").textContent="")};
$("mCard").addEventListener("change",e=>{const k=e.target.dataset.mk;if(k){const l=(M.known||{})[e.target.value]||[];if(l.length&&!l.includes($("mM_"+k).value.trim()))$("mM_"+k).value=l[0];mRow(k)}mHints()});
$("mCard").addEventListener("input",e=>{const id=e.target.id||"";if(id.startsWith("mM_"))mRow(id.slice(3));mHints()});
$("mCard").addEventListener("click",async e=>{const btn=e.target.closest("[data-mc]");if(!btn||!M)return;const k=btn.dataset.mc,v=mVal(k),pv=M.providers[v.provider]||{},out=$("mC_"+k);
 if(v.provider==="-"||v.provider==="none"||!v.model){out.textContent="모델을 고르세요";return}
 const t=k==="gen"?"gen":"rev",cost=pv.style==="claude_cli"?"Claude 구독 사용량을 조금 씁니다 (claude 프로세스 1개, 10~30초)":pv.style==="mock"?"가짜 응답이라 비용이 없습니다":"API 요청 1개라 비용이 아주 조금 듭니다";
 if(!confirm(`${v.provider}:${v.model} 에 아주 짧은 요청 1개('OK' 라고 답하기)를 보내 연결을 확인합니다.\n${cost}. 진행할까요?`))return;
 out.textContent="확인 중… (최대 2분)";btn.disabled=true;
 try{const r=await post("/api/models/check",{role:k==="gen"?"생성":"검토",model:`${v.provider}:${v.model}`,temperature:$("mT_"+t).value.trim(),max_tokens:$("mX_"+t).value.trim(),effort:$("mE_"+t).value});
  out.innerHTML=r.ok?`<span class="ok">연결됨</span> ${r.sec}초 · ${r.reply?`응답 '${esc(r.reply)}'`:"빈 응답"}${r.cut?" (출력 한도에서 잘림)":""} · 토큰 ${fmt(r.in||0)} / ${fmt(r.out||0)}`:`<span class="bad">실패</span> ${r.sec!=null?r.sec+"초 · ":""}${esc(r.error||"")}`}
 catch(err){out.textContent="실패: "+err.message}finally{btn.disabled=false}});
/* ── 1. 진행 현황 ── */
function bar(p){return `<div class="bar"><i style="width:${Math.min(100,p)}%"></i></div>`}
async function loadProgress(){const r=await get("/api/progress");P=r.data;
 if(!P){$("progBody").innerHTML=r.computing?"계산 중…":`아직 계산 결과가 없습니다. '다시 계산'을 누르세요.${r.error?"<br>"+esc(r.error):""}`;return}
 flagNames=P.flag_names;const fl=P.flags,fr=P.frontier,rq=P.requests,est=P.estimate_remaining,av=P.average_per_request;
 if(page==="run")estimate();
 const fsel=$("rFlag");if(fsel.options.length===1)fsel.innerHTML+=Object.keys(flagNames).map(k=>`<option value="${k}">${esc(flagNames[k])} (${k})</option>`).join("");
 if(page!=="progress")return;
 const srcRows=Object.entries(P.by_source).map(([k,v])=>`<tr><td>${k}</td><td class="n">${fmt(v.target)}</td><td class="n">${fmt(v.done)}</td><td class="n">${v.pct}%</td></tr>`).join("");
 const typeRows=Object.entries(P.by_attack_type).filter(([k,v])=>v.target>0).slice(0,12).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${fmt(v.target)}</td><td class="n">${fmt(v.done)}</td><td class="n">${v.pct}%</td></tr>`).join("");
 $("progBody").innerHTML=`
 <div class="row" style="gap:40px;align-items:flex-start">
  <div style="min-width:300px;flex:1"><div class="muted">생성 대상 룰 중 완료</div><div class="big">${P.pct_done_of_target}%</div>${bar(P.pct_done_of_target)}
   <div>${fmt(P.target)}개 중 <b>${fmt(P.done)}</b>개 완료 · 샘플 anomalies ${fmt(P.samples.anomalies)} / normal ${fmt(P.samples.normal)}</div>
   <div class="muted" style="margin-top:6px">현재 룰 ${fmt(P.rules)}개 중 학습데이터 보유 <b>${P.pct_with_data_of_all}%</b> (완료 ${fmt(fl.DONE)} + 기존 충분 ${fmt(fl.EXISTING)})</div>
   <div style="margin-top:12px" class="kv"><b>어디까지</b><span>배치 ${fr.contiguous_done_batch} / ${fr.batches_total} 까지 빠짐없이 생성 (가장 뒤 배치 ${fr.last_touched_batch})</span>
    <b>요청</b><span>${fmt(rq.total)}개 중 생성 ${fmt(rq.generated)} · 검토 ${fmt(rq.reviewed)} · 실패 ${fmt(rq.failed)} · 대기 ${fmt(rq.pending)}</span>
    <b>마지막 완료</b><span>${P.last_done?esc(P.last_done.request)+" ("+P.last_done.at.replace("T"," ")+")":"-"}</span>
    <b>남은 작업 예상</b><span>${est.hours!=null?`요청 약 ${fmt(est.requests)}개 · 약 ${fmt(est.hours)}시간 · 토큰 생성 ${fmt(est.generator_tokens)} / 검토 ${fmt(est.reviewer_tokens)}`:"생성 기록이 쌓이면 계산"}</span>
    <b>요청 1개 평균</b><span>${av.generate_seconds!=null?`생성 ${av.generate_seconds}초 · 검토 ${av.review_seconds??"-"}초 · 토큰 생성 ${fmt(av.generator_tokens)} / 검토 ${fmt(av.reviewer_tokens)}`:"-"}</span>
    <b>계산 시각</b><span>${P.computed_at.replace("T"," ")} · 입력 ${P.input_date}${r.computing?" · 다시 계산 중…":""}</span></div></div>
  <div style="min-width:240px"><table><thead><tr><th>상태 플래그</th><th class="n">룰 수</th></tr></thead><tbody>${Object.keys(flagNames).map(k=>`<tr><td>${chip(k)} <span class="muted">${k}</span></td><td class="n">${fmt(fl[k])}</td></tr>`).join("")}</tbody></table></div>
  <div style="min-width:240px"><table><thead><tr><th>출처</th><th class="n">대상</th><th class="n">완료</th><th class="n">%</th></tr></thead><tbody>${srcRows}</tbody></table>
   <table style="margin-top:12px"><thead><tr><th>공격 유형</th><th class="n">대상</th><th class="n">완료</th><th class="n">%</th></tr></thead><tbody>${typeRows}</tbody></table></div>
 </div>`;}
$("refresh").onclick=async()=>{await post("/api/progress/refresh");$("progBody").insertAdjacentHTML("afterbegin",'<div class="muted">다시 계산 중…</div>');setTimeout(loadProgress,3000)};
/* ── 3. 룰별 상태 ── */
async function loadRules(){const qs=new URLSearchParams({flag:$("rFlag").value,source:$("rSrc").value,q:$("rQ").value,offset:rOff,limit:50});
 const r=await get("/api/rules?"+qs);$("rCount").textContent=`${fmt(r.total)}개 중 ${r.total?rOff+1:0}–${Math.min(rOff+50,r.total)}`;
 $("rBody").innerHTML=r.rows.map(x=>`<tr><td>${esc(x.id)}</td><td>${chip(x.flag)}</td><td>${esc(x.attack_type)}</td><td>${esc(x.rule_name)}</td><td class="muted">${esc(x.date)} ${esc(x.request)}</td><td class="n">${x.anomalies||""}</td><td class="n">${x.normal||""}</td><td class="n">${x.existing||""}</td></tr>`).join("");
 $("rPrev").disabled=rOff===0;$("rNext").disabled=rOff+50>=r.total;}
let tm;$("rQ").oninput=()=>{clearTimeout(tm);tm=setTimeout(()=>{rOff=0;loadRules()},300)};$("rFlag").onchange=$("rSrc").onchange=()=>{rOff=0;loadRules()};
$("rPrev").onclick=()=>{rOff=Math.max(0,rOff-50);loadRules()};$("rNext").onclick=()=>{rOff+=50;loadRules()};
/* ── 4. 신규 학습 데이터 (표만, 로그 형식 파일은 6. 에서 버전을 확정할 때 만듦) ── */
let dOff=0,dInit=false,D=[],dTot=0,dReady=false,dV=null,dVT=0,dRows=[],dE=null,dET=0,dERun=false,edX=null,dPending=null;
function dLimit(){return 50}
/* 장비 FW·웹: IPS 줄을 1_4a 로 바꿔 보여 줌 (조건·검증·수정·삭제는 원본 IPS 기준, 표는 보기만) */
function dSrc(){const v=$("dDev").value;return v==="fw"||v==="web"?"ips":v}
function dConv(){return dSrc()!==$("dDev").value}
const EVC={none:["c-CHECK","없음"],kept:["c-OK","남음"],lost:["c-BAD","사라짐"],check:["c-CHECK","확인 못 함"]};
function convTable(r,all){const nm=$("dDev").value==="fw"?"FW":"웹";
 return `<table><thead><tr><th class="n">index</th><th>${nm} 로그 (변환)</th><th>label <span class="muted" style="font-weight:400">(원본 샘플)</span></th><th>공격 근거 (${nm} 로그)</th><th>원본 payload</th></tr></thead><tbody>${r.rows.map(x=>{const e=EVC[x.evidence.state];
  return `<tr><td class="n">${x.index}${all?`<div class="muted">${x.date}</div>`:""}</td><td><div class="payload">${esc(x.raw)}</div><details><summary>파싱 결과 ${x.fields.length}개</summary><div class="reason" style="word-break:break-all">${x.fields.map(([k,v])=>`<code>${esc(k)}</code>=${esc(v)}`).join(", ")}</div></details></td><td><span class="chip l-${esc(x.label)}">${esc(x.label)}</span><div class="muted">${esc(x.rule_id)}</div></td><td class="reason">${e?`<span class="chip ${e[0]}">${e[1]}</span> `:""}${esc(x.evidence.text)}${x.lost_headers.length?`<div class="muted">사라지는 헤더: ${esc(x.lost_headers.join(", "))}</div>`:""}</td><td><details><summary>${esc(x.request.slice(0,80))}</summary><div class="payload">${esc(x.payload.replace(/\\n/g,"\n"))}</div></details></td></tr>`}).join("")}</tbody></table>`}
function convHelp(){const dv=$("dDev").value,fw=dv==="fw";
 return (fw?`<b>방화벽(FW) 로그 보기</b>: 신규 학습 데이터를 FW 세션 로그 형식(syslog + CSV 34열)으로 바꿔 보여 줍니다. <b>FW 로그에는 payload 가 없어</b>(출발·목적 IP·포트, protocol, 바이트·패킷, 룰, 동작만) <b>공격 근거가 로그에 없습니다</b> - label 은 원본 샘플의 것이고 FW 줄 내용으로는 판단할 수 없습니다. 보낸 바이트는 요청 길이로, 세션 시간·응답 크기는 샘플 지문으로 만든 값입니다 (label 과 무관).`
  :`<b>웹 로그 보기</b>: nginx 접근 로그 형식(syslog + combined + X-Forwarded-For·본문)으로 바꿔 보여 줍니다. 남는 것: 요청 줄·User-Agent·Referer·X-Forwarded-For·본문 (+ Basic 인증 사용자 이름). Host·Cookie·Content-Type·Authorization·사용자 정의 헤더는 사라집니다 → anomalies 인데 룰 조건이 그런 헤더에만 있으면 <b>공격 근거 '사라짐'</b> (남는 부분만으로 룰 조건을 다시 대조). 응답 코드·크기는 원본에 없어 샘플 지문으로 만든 값입니다 (label 과 무관). 값은 nginx 처럼 " \\ 와 줄바꿈·한글 등을 \\xHH 로 적습니다.`)
  +` label 정합성·보류 대상 필터와 검증 버튼은 <b>원본 payload 기준</b>이고, 수정·삭제는 이 보기에서 하지 않습니다. <b>CSV 내려받기</b>: 지금 조건에 맞는 줄의 변환 로그 + 파싱 필드 + 공격 근거 + 원본 payload (하루 약 11만 줄에 1~2분). <b>파일로 저장</b>: 고른 날짜(전체면 모든 날짜)의 데이터 전체를 조건과 상관없이 08_dataset/&lt;날짜&gt;/ 에 ${dv}_&lt;yymmdd&gt;.log (변환 로그 한 줄씩) + ${dv}_&lt;yymmdd&gt;_label.jsonl (같은 줄 순서의 label·룰 id·공격 근거) 로 저장합니다 (하루 약 11만 줄에 FW 30초·웹 1분). 학습 로그를 다시 만들면 다시 저장하세요. 형식·장비 값: docs/코드참조/FW_웹로그_형식.md (확인 필요 값 표시)`}
/* FW·웹 '파일로 저장': 고른 날짜(전체면 모든 날짜)의 데이터 전체 → 08_dataset/<날짜>/<fw|web>_<yymmdd>.log + _label.jsonl (1_4a.save) */
let dST=0;
async function loadSave(){clearTimeout(dST);const on=dConv();$("dSave").style.display=$("dSaveSum").style.display=on?"":"none";if(!on)return;
 let s;try{s=await get(`/api/data/conv/save?date=${$("dDate").value}&dev=${$("dDev").value}`)}catch(e){$("dSaveSum").textContent="";return}
 const j=s.job,run=j.status==="running",mine=j.dev===$("dDev").value,fs=Object.values(s.files),f=fs[0]||{},nm=d=>d==="fw"?"FW":"웹";$("dSave").disabled=run;
 let h=run?`<span>${nm(j.dev)} 로그 파일 저장 중… ${j.date}${j.dates.length>1?` (${j.dates.indexOf(j.date)+1} / ${j.dates.length}일)`:""} · ${fmt(j.done)} / ${fmt(j.total)}줄</span>`
  :fs.length>1?`<span>파일로 저장한 날짜 ${fs.filter(x=>x.saved_at).length} / ${fs.length}${fs.some(x=>x.stale)?` · 저장한 뒤 학습 로그나 형식 md 가 바뀐 날짜 ${fs.filter(x=>x.stale).length}`:""}</span>`
  :f.saved_at?`<span>저장한 파일: <code>${esc(f.name)}</code> + <code>${esc(f.label)}</code> · ${f.mb}MB · ${f.saved_at}${f.stale?" · <b>저장한 뒤 학습 로그나 형식 md 가 바뀜 → 다시 저장</b>":""}</span>`
  :`<span class="muted">저장한 파일 없음 - '파일로 저장' 은 <code>${esc(f.name||"")}</code> + <code>${esc(f.label||"")}</code> 로 저장합니다</span>`;
 if(!run&&mine&&j.status==="error")h+=`<span class="muted">저장 실패: ${esc(j.error)}</span>`;
 if(!run&&mine&&j.status==="done")h+=`<span class="muted">${j.finished_at.slice(11,16)} 저장 끝: ${j.saved.map(x=>`${x.date} ${fmt(x.lines)}줄`).join(", ")}</span>`;
 $("dSaveSum").innerHTML=h;if(run)dST=setTimeout(loadSave,1500)}
$("dSave").onclick=async()=>{const v=$("dDate").value,dev=$("dDev").value,ds=v==="all"?D:D.filter(d=>d.date===v),n=ds.reduce((s,d)=>s+(d.label.anomalies||0)+(d.label.normal||0),0);
 if(!confirm(`${v==="all"?"모든 날짜":v} 의 데이터 전체 ${fmt(n)}줄을 ${dev==="fw"?"FW":"웹"} 로그 파일로 저장합니다 (지금 조건과 상관없이, ${dur(n*(dev==="fw"?0.00025:0.0005))}).\n08_dataset/<날짜>/${dev}_<yymmdd>.log 와 _label.jsonl 이 이미 있으면 바꿉니다. 진행할까요?`))return;
 try{await post("/api/data/conv/save",{date:v,dev});loadSave()}catch(e){$("dSaveSum").style.display="";$("dSaveSum").textContent="실패: "+e.message}};
async function initData(){if(!dInit){const r=await get("/api/data/dates");D=r.dates;
  $("dDate").innerHTML=(D.length>1?'<option value="all">전체 (모든 날짜)</option>':"")+D.slice().reverse().map(d=>`<option value="${d.date}">${d.date} (${fmt(d.events)}건)</option>`).join("");
  if(D.length)$("dDate").value=D.at(-1).date;dInit=true}
 if(dPending){$("dRule").value=dPending.rule||"";$("dDev").value=dPending.dev;if(D.length>1)$("dDate").value="all";   // 6. 의 '4.에서 보기'
  $("dLabel").value=$("dCons").value=$("dQ").value="";$("dHold").value=dPending.hold||"";dHoldUi();dOff=0;dPending=null}
 loadData();loadVerify();loadEdits();loadSave();ensureFeat()}
function dHoldUi(){const h=$("dHold").value;for(const id of["dLabel","dCons"]){$(id).disabled=!!h;if(h)$(id).value=""}}   // 보류 대상이 label·정합성을 정함
function dSummary(){const v=$("dDate").value,ds=v==="all"?D:D.filter(d=>d.date===v);const a=ds.reduce((s,d)=>s+(d.label.anomalies||0),0),n=ds.reduce((s,d)=>s+(d.label.normal||0),0);
 $("dSum").textContent=`· ${v==="all"?"전체":v} · ${fmt(a+n)}건 (anomalies ${fmt(a)} / normal ${fmt(n)}) · 룰 ${fmt(ds.reduce((s,d)=>s+d.rules,0))}개`;
 // 지금 조건에 맞는 모든 줄을 표의 열 그대로 CSV
 $("dDown").href="/download/frame?"+new URLSearchParams({date:v,dev:$("dDev").value,label:$("dLabel").value,rule:$("dRule").value.trim(),q:$("dQ").value.trim(),cons:$("dCons").value,hold:$("dHold").value});
 $("dHelp").innerHTML="<b>label 사유</b>: 어떤 룰의 어떤 조건을 만족·불일치했는지 + 생성 메모. <b>검증 결과</b>: 생성 때 검증(코드 검사·검토 모델·증강) + 지금 현재 룰로 다시 확인. <b>label 정합성</b>: 라벨이 증거와 맞는지 — 정합(룰 + 범용 공격 흔적·다른 공격 룰도 일치) / 정합(룰 기준: 이 룰의 고유 패턴으로만 공격) / 확인 필요 / 불일치. <b>탐지 피처</b>: ML 피처(4.1) 중 이 payload 에서 켜지는 것 — k_ 공격 흔적, ra_ 신뢰할 수 있는 공격 룰, r_ 이 룰·정규식. 피처 이름에 마우스를 올리면 설명이 나오고, 누르면 <a href=\"#/features\">5. 피처</a> 페이지에서 봅니다. <b>보류 대상</b>: 6. 선별의 보류 옵션에 걸리는 샘플만 — '정합 (룰 기준)' 공격(그날 '전체 검증'이 있어야 함) / python-requests UA 정상. <b>CSV 내려받기</b>: 지금 조건(날짜·장비·label·룰·검색·정합성·보류 대상)에 맞는 모든 줄을 이 표의 열 그대로 CSV(엑셀용, payload 는 로그와 같은 한 줄)로 받습니다. '전체 검증'을 해 두면 바로 받고, 아니면 줄마다 다시 확인해 1천 줄에 약 15초 걸립니다. 로그 형식(|||) 파일은 <a href=\"#/sup\">6. 기존 학습데이터와 비교</a>에서 버전을 확정할 때 기존 학습데이터와 합쳐 만듭니다.";if(dConv())$("dHelp").innerHTML=convHelp();}
async function loadVerify(){const v=$("dDate").value;if(!v)return;let s;try{s=await get(`/api/data/verify?date=${v}&dev=${dSrc()}`)}catch(e){$("dVsum").textContent="";return}
 dV=s;dReady=s.status==="done"&&!!s.frame_ready;
 const run=s.status==="running";$("dVerifyStop").style.display=run?"":"none";$("dVerifyNew").disabled=$("dVerify").disabled=run;
 if(run){const what=s.mode==="new"?"검증 안 한 데이터만 검증":"전체 검증",left=s.eta_s!=null?` · 남은 시간 <b>약 ${hms(s.eta_s)}</b> · <b>${clock(s.eta_s)}</b> 쯤 끝날 예상`:"";
  const body=s.phase==="wait"?`${what} 대기 - IPS 검증이 끝나기를 기다리는 중 (같은 샘플이라 끝나면 그 결과를 그대로 씀)${left}`
   :s.phase==="run"?`${what} 중… ${fmt(s.done)} / ${fmt(s.total)}줄${s.todo<s.total?` (새로 확인 ${fmt(s.analyzed)} / ${fmt(s.todo)}줄 · 이미 확인한 결과 그대로 ${fmt(s.reused)}줄)`:""}${left}`
   :`${what} 준비 중 - 이미 확인한 줄과 새로 확인할 줄을 세는 중…`;
  $("dVsum").innerHTML=`<span>${body}${s.stopping?" · <b>중지하는 중…</b>":""}</span>`;clearTimeout(dVT);dVT=setTimeout(loadVerify,1500);return}
 const est=`<span class="muted">검증 안 한 데이터만: 새로 확인 약 ${fmt(s.est_new)}줄${s.est_new?` (약 ${hms(s.est_new_s)})`:""} · 전체 검증: ${fmt(s.total)}줄 (약 ${hms(s.est_all_s)})</span>`;
 const stop=s.stopped?`<span class="muted">${s.stopped.at.slice(11,16)} 에 중지 (${esc(s.stopped.why)}) - 확인한 ${fmt(s.stopped.saved)}줄은 저장됨 → '검증 안 한 데이터만 검증' 으로 이어서</span>`:"";
 if(s.status!=="done"){$("dVsum").innerHTML=(s.status==="error"?`<span class="muted">검증 오류: ${esc(s.error)}</span>`:`<span class="muted">검증 전 — 모든 줄의 label 정합성을 확인하면 정합성으로 걸러 볼 수 있습니다.</span>`)+stop+est;return}
 const tot=Object.values(s.summary).reduce((a,b)=>a+b,0);
 $("dVsum").innerHTML=`<b>label 정합성${dConv()?" (원본 payload 기준)":""} (${fmt(tot)}줄)</b>`+["OK","RULE","CHECK","BAD"].map(k=>`<span class="chip c-${k}" data-c="${k}">${esc(s.names[k])} ${fmt(s.summary[k]||0)} (${tot?(100*(s.summary[k]||0)/tot).toFixed(1):0}%)</span>`).join("")+`<span class="muted">${s.finished_at.replace("T"," ")} 검증${s.reused?` · 이미 확인한 결과 ${fmt(s.reused)}줄 그대로 씀`:""}${s.old_rules?` · ${fmt(s.old_rules)}줄은 이전 룰 입력(${esc(Object.keys(s.rule_inputs).filter(k=>k!==s.latest_input).sort().join(", "))}) 기준 - 최신 룰(${esc(s.latest_input)})로 다시 보려면 '전체 검증'`:""}</span>`+stop+(s.est_new?est:"");
 $("dVsum").querySelectorAll("[data-c]").forEach(c=>c.onclick=()=>{$("dHold").value="";dHoldUi();$("dCons").value=c.dataset.c;dOff=0;loadData()});renderEdits();}
async function startVerify(mode){try{const r=await post("/api/data/verify",{date:$("dDate").value,dev:dSrc(),mode});
  if(!r.started.length&&r.done.length){$("dVsum").innerHTML='<span class="muted">새로 확인할 줄이 없습니다 - 이미 모두 검증했습니다.</span>';setTimeout(loadVerify,2500)}else loadVerify()}catch(e){$("dVsum").textContent="실패: "+e.message}}
$("dVerifyNew").onclick=()=>startVerify("new");
$("dVerify").onclick=()=>{if(!dV||confirm(`모든 줄(${fmt(dV.total)}줄)을 지금 룰로 새로 확인합니다 (약 ${hms(dV.est_all_s)}). 진행할까요?`))startVerify("all")};
$("dVerifyStop").onclick=async()=>{try{await post("/api/data/verify/stop",{date:$("dDate").value,dev:dSrc()});loadVerify()}catch(e){$("dVsum").textContent="실패: "+e.message}};
async function loadData(){dSummary();if(!D.length){$("dBody").innerHTML='<div class="muted">08_dataset 에 학습데이터가 없습니다.</div>';return}
 const qs=new URLSearchParams({date:$("dDate").value,dev:$("dDev").value,label:$("dLabel").value,rule:$("dRule").value.trim(),q:$("dQ").value.trim(),cons:$("dCons").value,hold:$("dHold").value,offset:dOff,limit:dLimit()});
 $("dBody").style.opacity=.5;let r;try{r=await get("/api/data/rows?"+qs)}catch(e){$("dBody").innerHTML=`<div class="muted">불러오지 못했습니다: ${esc(e.message)}</div>`;return}finally{$("dBody").style.opacity=1}
 const all=$("dDate").value==="all";
 $("dBody").innerHTML=dConv()?convTable(r,all):`<table><thead><tr><th class="n">index</th><th>payload</th><th>method</th><th>label</th><th>label 사유</th><th>검증 결과</th><th>label 정합성</th><th>탐지 피처 <a href="#/features" style="font-weight:400">(설명)</a></th><th>수정·삭제</th></tr></thead><tbody>${r.rows.map((x,k)=>`<tr class="lv-${x.level}"><td class="n">${x.index}${all?`<div class="muted">${x.date}</div>`:""}</td><td><div class="payload">${esc(x.payload.replace(/\\n/g,"\n"))}</div></td><td class="method">${esc(x.method)}</td><td><span class="chip l-${esc(x.label)}">${esc(x.label)}</span></td><td class="reason">${esc(x.reason)}</td><td class="chk">${x.checks.map(([s,t])=>`<div class="${s}">${s==="ok"?"✔":s==="warn"?"△":"✘"} ${esc(t)}</div>`).join("")}</td><td class="cons"><span class="chip c-${x.cons}">${esc(x.cons_text)}</span><div>${esc(x.cons_reason)}</div></td><td class="feat">${x.features.length?x.features.map(f=>`<div><code class="fn" data-f="${esc(f.name)}" data-d="${f.name.startsWith("k_")?"":esc(f.desc)}">${esc(f.name)}</code> = ${f.value} <span class="muted">${esc(f.desc)}</span></div>`).join("")+`<div class="muted">${x.label==="normal"?"✘ 정상인데 켜진 피처가 있음":`켜진 피처 ${x.features.length}개 → 모델이 공격으로 볼 근거`}</div>`:`<div class="muted">${x.label==="normal"?"켜진 피처 없음 → 정상으로 볼 근거":"켜진 피처 없음 → 이 피처로는 탐지 못 함"}</div>`}</td><td class="act">${editCell(x,k)}</td></tr>`).join("")}</tbody></table>`;
 dRows=r.rows;dTot=r.total;$("dCount").textContent=`${fmt(r.total)}건 중 ${r.total?dOff+1:0}–${Math.min(dOff+dLimit(),r.total)}`;$("dPrev").disabled=dOff===0;$("dNext").disabled=dOff+dLimit()>=r.total;}
$("dDown").onclick=e=>{   // 저장된 '전체 검증' 결과가 없으면 줄마다 다시 확인하므로 오래 걸릴 수 있음
 if(dConv()){if(dTot>30000&&!confirm(`${fmt(dTot)}줄을 ${$("dDev").value==="fw"?"FW":"웹"} 로그로 바꿔 CSV 를 만듭니다 (${dur(dTot*0.001)}). 지금 받을까요?`))e.preventDefault();return}   // FW·웹: 변환만 (1천 줄에 약 1초)
 if(!dReady&&dTot>1000&&!confirm(`'전체 검증' 결과가 없어 ${fmt(dTot)}줄을 하나씩 다시 확인하며 CSV 를 만듭니다 (${dur(dTot*0.015)}).\n'전체 검증'을 먼저 하면 바로 받을 수 있습니다. 지금 받을까요?`))e.preventDefault()};
for(const id of["dDate","dDev"])$(id).onchange=()=>{dOff=0;$("dCons").value="";loadData();loadVerify();loadEdits();loadSave()};
/* 수정·삭제: 저장하면 '학습 로그에 반영'(1.4 다시 만들기) 때 학습 로그에 들어간다 */
function editCell(x,k){if(!x.sk)return"";
 if(x.edit)return x.edit.action==="delete"?`<span class="chip c-BAD">삭제 예정</span><div><button class="sm" data-a="undo" data-r="${k}">되돌리기</button></div>`
  :`<span class="chip c-RULE">${x.edited?"수정됨":"수정 예정"}${x.edit.label?" → "+esc(x.edit.label):""}${x.edit.payload?" (payload)":""}</span><div><button class="sm" data-a="edit" data-r="${k}">수정</button><button class="sm" data-a="undo" data-r="${k}">되돌리기</button></div>`;
 return(x.edited?`<span class="chip c-RULE">수정됨 (${esc(x.edited)})</span><br>`:"")+`<button class="sm" data-a="edit" data-r="${k}">수정</button><button class="sm danger" data-a="del" data-r="${k}">삭제</button>`}
async function editPost(body){try{await post("/api/data/edit",body);loadData();loadEdits()}catch(e){alert("실패: "+e.message)}}
$("dBody").addEventListener("click",e=>{const b=e.target.closest("button[data-a]");if(!b)return;const x=dRows[+b.dataset.r];if(!x)return;
 const at={date:x.date,dev:$("dDev").value,line:x.line};
 if(b.dataset.a==="edit")openEditor(x);
 else if(b.dataset.a==="undo")editPost(Object.assign(at,{action:"undo"}));
 else if(confirm(`index ${x.index} 줄 (룰 ${x.rule_id}, ${x.label}) 을 학습 로그에서 삭제합니다. '학습 로그에 반영' 때 적용되고 목록에서 되돌릴 수 있습니다. 진행할까요?`))
  editPost(Object.assign(at,{action:"delete",why:x.cons==="BAD"?"불일치: "+x.cons_reason:""}))});
function openEditor(x){edX=x;$("edWhere").textContent=`${x.date} · ${$("dDev").value.toUpperCase()} · index ${x.index} · 룰 ${x.rule_id}`;$("edLabel").value=x.label;
 $("edPay").value=x.payload.replace(/\\n/g,"\n");$("edRes").innerHTML=`지금: <span class="chip c-${x.cons}">${esc(x.cons_text)}</span> <span class="muted">${esc(x.cons_reason)}</span>`;$("edBox").style.display="flex"}
function edBody(){const b={date:edX.date,dev:$("dDev").value,line:edX.line,label:$("edLabel").value},p=$("edPay").value;if(p!==edX.payload.replace(/\\n/g,"\n"))b.payload=p;return b}
$("edCheck").onclick=async()=>{$("edRes").textContent="확인 중…";try{const a=await post("/api/data/check",edBody());$("edRes").innerHTML=`고친 뒤: <span class="chip c-${a.cons}">${esc(a.cons_text)}</span> ${esc(a.cons_reason)}<div class="muted">${esc(a.reason)}</div>`}catch(e){$("edRes").textContent="실패: "+e.message}};
$("edSave").onclick=async()=>{try{await post("/api/data/edit",Object.assign(edBody(),{action:"edit"}));$("edBox").style.display="none";loadData();loadEdits()}catch(e){$("edRes").textContent="실패: "+e.message}};
$("edCancel").onclick=()=>{$("edBox").style.display="none"};
async function loadEdits(){const v=$("dDate").value;if(!v)return;try{dE=await get(`/api/data/edits?date=${v}`)}catch(e){$("dEdits").textContent="";return}renderEdits();
 const run=!!(dE.rebuild&&dE.rebuild.running);clearTimeout(dET);if(run)dET=setTimeout(loadEdits,3000);
 if(dERun&&!run){D=(await get("/api/data/dates")).dates;loadData();loadVerify()}   // 다시 만든 뒤: 줄 수·검증 상태 새로
 dERun=run}
function renderEdits(){const s=dE;if(!s)return;const rb=s.rebuild||{},bad=(dV&&dV.status==="done"&&dV.summary&&dV.summary.BAD)||0;let h="";
 if(rb.running)h+=`<span><b>학습 로그 다시 만드는 중…</b> <span class="muted">(${esc((rb.started_at||"").slice(11,16))} 시작, 큰 날짜는 4~5분. 끝나면 '검증 안 한 데이터만 검증' 으로 바뀐 줄만 확인)</span></span>`;
 else if(rb.result==="fail")h+=`<span class="muted">학습 로그 다시 만들기 실패 - .run/dataset-*.out 을 확인하세요</span>`;
 if(s.old_format.length)h+=`<span class="muted">${esc(s.old_format.join(", "))} 학습 로그는 예전 형식이라 줄을 고치려면 먼저 다시 만들어야 합니다</span>`;
 if(s.saved)h+=`<span>수정·삭제 <b>${fmt(s.saved)}</b>건 (삭제 ${fmt(s.deleted)} · 수정 ${fmt(s.edited)})${s.pending?" - <b>학습 로그에 아직 반영 안 됨</b>":" - 학습 로그에 반영됨"}</span><button class="sm" data-e="list">목록</button>`;
 if(!s.saved&&s.pending)h+=`<span>되돌린 수정·삭제가 학습 로그에 아직 반영 안 됨</span>`;
 if((s.pending||s.old_format.length)&&!rb.running)h+=`<button class="sm primary" data-e="apply">학습 로그에 반영 (다시 만들기)</button>`;
 if(bad&&!rb.running)h+=`<button class="sm danger" data-e="bad">불일치 ${fmt(bad)}줄 모두 삭제</button>`;
 $("dEdits").innerHTML=h}
function editList(){const s=dE,box=$("dEditList");if(!s||box.style.display!=="none"){box.style.display="none";return}
 box.innerHTML=`<table><thead><tr><th>날짜</th><th>처리</th><th>룰 id</th><th>label</th><th>사유</th><th>저장 시각</th><th></th></tr></thead><tbody>${s.items.map((e,k)=>`<tr><td>${e.date}</td><td>${e.action==="delete"?"삭제":"수정"+(e.payload?" (payload)":"")}</td><td>${esc(e.id)}</td><td>${esc(e.orig_label)}${e.label?" → "+esc(e.label):""}</td><td class="reason">${esc(e.why)}</td><td>${esc(e.at.replace("T"," "))}</td><td><button class="sm" data-u="${k}">되돌리기</button></td></tr>`).join("")}</tbody></table>${s.saved>s.items.length?`<div class="muted">최근 ${s.items.length}건만 표시</div>`:""}`;box.style.display="block"}
$("dEdits").addEventListener("click",async e=>{const b=e.target.closest("button[data-e]");if(!b)return;const a=b.dataset.e,v=$("dDate").value;
 try{if(a==="list")return editList();
  if(a==="bad"){const n=dV.summary.BAD;if(!confirm(`검증에서 '불일치' 인 ${fmt(n)}줄을 모두 삭제로 저장합니다. '학습 로그에 반영' 때 적용되고 목록에서 되돌릴 수 있습니다. 진행할까요?`))return;await post("/api/data/edit/bad",{date:v,dev:dSrc()});loadData()}
  if(a==="apply"){if(!confirm("저장한 수정·삭제를 반영해 학습 로그를 다시 만듭니다 (1.4, API 비용 없음, 큰 날짜는 4~5분). 끝나면 '검증 안 한 데이터만 검증' 으로 바뀐 줄만 확인하세요. 진행할까요?"))return;await post("/api/data/rebuild",{date:v})}
  loadEdits()}catch(err){alert("실패: "+err.message)}});
$("dEditList").addEventListener("click",async e=>{const b=e.target.closest("button[data-u]");if(!b)return;const it=dE.items[+b.dataset.u];
 try{await post("/api/data/edit",{date:it.date,sk:it.sk,action:"undo"});$("dEditList").style.display="none";loadEdits();loadData()}catch(err){alert("실패: "+err.message)}});
for(const id of["dLabel","dCons"])$(id).onchange=()=>{dOff=0;loadData()};
$("dHold").onchange=()=>{dOff=0;dHoldUi();loadData()};
$("dGo").onclick=()=>{dOff=0;loadData()};for(const id of["dRule","dQ"])$(id).onkeydown=e=>{if(e.key==="Enter"){dOff=0;loadData()}};
$("dPrev").onclick=()=>{dOff=Math.max(0,dOff-dLimit());loadData()};$("dNext").onclick=()=>{dOff+=dLimit();loadData()};
/* ── 5. 피처 ── */
let FC=null,fFocus="",fLoading=null;
function fkey(n){return n.startsWith("r_")?(n.endsWith("_all")?"r_<룰>_all":"r_<룰>_re<n>"):n}
function featDoc(n){return FC?FC.items.find(i=>i.name===fkey(n))||null:null}
function ensureFeat(){if(FC)return Promise.resolve(FC);if(!fLoading)fLoading=get("/api/features?dev=ips").then(r=>FC=r).catch(()=>null).finally(()=>{fLoading=null});return fLoading}
async function loadFeatures(focus){if(focus!==undefined)fFocus=focus;
 try{FC=await get("/api/features?dev="+$("fDev").value)}catch(e){$("fBody").innerHTML=`<tr><td colspan="9" class="muted">불러오지 못했습니다: ${esc(e.message)}</td></tr>`;return}
 const r=FC,gs=$("fGroup");if(gs.options.length===1)gs.innerHTML+=r.groups.map(g=>`<option>${esc(g.name)}</option>`).join("");
 $("fIntro").innerHTML=`ML 탐지 모델(4.1)이 payload 에서 뽑는 피처입니다. 공격일 때만 켜지는 <b>확실한 신호</b>만 두었습니다 (길이·헤더 수 같은 모호한 값은 쓰지 않음). 4. 신규 학습 데이터의 '탐지 피처'에 나오는 이름이 모두 여기 있습니다.`+
  r.groups.map(g=>`<div style="margin-top:4px"><b>${esc(g.name)}</b> <code>${esc(g.prefix)}</code> ${esc(g.desc)}</div>`).join("")+`<div style="margin-top:4px">공격 룰 피처에 쓰는 룰: ${fmt(r.attack_rules)}개${r.note?` (${esc(r.note)})`:""}</div>`;
 $("fCountMsg").textContent=r.running?"켜진 줄 수 계산 중…":!r.dates.length?"켜진 줄 수가 아직 없습니다 - '켜진 줄 수 다시 계산'을 누르세요 (신규 학습 데이터 전체 검증과 같은 작업, 수 초~수 분)":r.missing.length?`검증 안 된 날짜 ${r.missing.length}개 (${r.missing.join(", ")}) - 다시 계산하면 포함`:"";
 if(r.running)setTimeout(()=>{if(page==="features")loadFeatures()},2000);
 renderFeatures()}
function renderFeatures(){const r=FC;if(!r)return;const g=$("fGroup").value,q=$("fQ").value.trim().toLowerCase(),la=r.labels.anomalies||0,ln=r.labels.normal||0,have=r.dates.length>0;
 $("fSum").textContent=`· ${r.items.length}개`+(have?` · 켜진 줄 수: ${$("fDev").value.toUpperCase()} 학습데이터 ${r.dates.length}개 날짜 (anomalies ${fmt(la)}줄 / normal ${fmt(ln)}줄)`:"");
 const pc=(x,t)=>t&&x?` <span class="muted">(${(100*x/t).toFixed(1)}%)</span>`:"";let last="";
 $("fBody").innerHTML=r.items.filter(i=>(!g||i.group===g)&&(!q||[i.name,i.title,i.what,i.example,i.pattern].join(" ").toLowerCase().includes(q))).map(i=>{
  const c=r.counts[i.name]||[0,0],head=i.group!==last?`<tr class="fgroup"><td colspan="9">${esc(i.group)}</td></tr>`:"";last=i.group;
  return head+`<tr id="f-${esc(i.name)}"${i.name===fFocus?' class="hl"':""}><td class="nw"><code>${esc(i.name)}</code></td><td style="min-width:120px">${esc(i.title)}</td><td class="reason">${esc(i.what)}</td><td class="reason muted">${esc(i.example)}</td><td class="muted nw">${esc(i.value)}</td>`+
   (have?`<td class="n nw">${fmt(c[0])}${pc(c[0],la)}</td><td class="n nw">${fmt(c[1])}${pc(c[1],ln)}</td><td class="n nw">${c[0]+c[1]?(100*c[0]/(c[0]+c[1])).toFixed(1)+"%":"-"}</td>`:'<td class="n muted">-</td><td class="n muted">-</td><td class="n muted">-</td>')+
   `<td>${i.pattern?`<details><summary>보기</summary><div class="pat">${esc(i.pattern)}</div></details>`:""}</td></tr>`}).join("")||'<tr><td colspan="9" class="muted">없음</td></tr>';
 if(fFocus){const el=document.getElementById("f-"+fFocus);if(el)el.scrollIntoView({block:"center"})}}
let fTm;$("fQ").oninput=()=>{clearTimeout(fTm);fTm=setTimeout(()=>{fFocus="";renderFeatures()},200)};$("fGroup").onchange=()=>{fFocus="";renderFeatures()};$("fDev").onchange=()=>loadFeatures();
$("fCount").onclick=async()=>{try{await post("/api/data/verify",{date:"all",dev:$("fDev").value,mode:"new"});$("fCountMsg").textContent="켜진 줄 수 계산 중…";setTimeout(()=>loadFeatures(),1500)}catch(e){$("fCountMsg").textContent="실패: "+e.message}};
/* 피처 이름 말풍선 (4. 신규 학습 데이터의 '탐지 피처') */
const tip=document.createElement("div");tip.className="tipbox";tip.hidden=true;document.body.appendChild(tip);let tipOn=null;
function showTip(t){tipOn=t;const d=featDoc(t.dataset.f);
 tip.innerHTML=`<div class="t"><code>${esc(t.dataset.f)}</code> ${d?esc(d.title):""}</div>`+(d?`<div class="muted">${esc(d.group)} · 값: ${esc(d.value)}</div><div>${esc(d.what)}</div>${d.example?`<div class="muted">예: ${esc(d.example)}</div>`:""}`:'<div class="muted">설명을 불러오는 중…</div>')+
  (t.dataset.d?`<div style="margin-top:4px">이 줄: ${esc(t.dataset.d)}</div>`:"")+'<div class="muted" style="margin-top:4px">누르면 5. 피처 페이지에서 봅니다</div>';
 tip.hidden=false;const rc=t.getBoundingClientRect(),w=tip.offsetWidth,h=tip.offsetHeight;
 tip.style.left=Math.max(8,Math.min(rc.left,innerWidth-w-8))+"px";tip.style.top=(rc.bottom+6+h>innerHeight-8?Math.max(8,rc.top-h-6):rc.bottom+6)+"px";
 if(!d)ensureFeat().then(()=>{if(tipOn===t&&featDoc(t.dataset.f))showTip(t)})}
document.addEventListener("mouseover",e=>{const t=e.target.closest&&e.target.closest("[data-f]");if(t){if(t!==tipOn)showTip(t)}else if(tipOn){tipOn=null;tip.hidden=true}});
document.addEventListener("scroll",()=>{tipOn=null;tip.hidden=true},true);
document.addEventListener("click",e=>{const t=e.target.closest&&e.target.closest("[data-f]");if(t){tipOn=null;tip.hidden=true;location.hash="#/features/"+encodeURIComponent(fkey(t.dataset.f))}});
/* ── 6. 기존 학습데이터와 비교 (비교·확정은 별도 프로세스, 선별 미리 보기는 확정과 같은 계산) ── */
let SS=null,sDev="ips",sPv=null,sPvT=0,sJT=0,sRunWas=false,sFormInit=false,sResInit=false;
const sO={dates:null,sources:null,rules:{},types:{}};   // 날짜·출처 null = 전부 (뒤에 생긴 날짜도)
function gchip(g){return `<span class="chip g${["제거","A","B","C"].indexOf(g)}">${esc(g)}</span>`}
function sSpec(){const v=id=>{const x=$(id).value;return x==="all"?"all":+x};
 return {conditions:{dates:sO.dates||[],sources:sO.sources||[]},options:{A:v("oA"),B:v("oB"),C:v("oC"),exclude_bad:$("oBad").checked,exclude_rule_based:$("oRule").checked,exclude_python_ua_normal:$("oPy").checked},overrides:{rules:sO.rules,types:sO.types}}}
function sSetForm(o,c,ov){for(const k of["A","B","C"])$("o"+k).value=String(o[k]);$("oBad").checked=!!o.exclude_bad;$("oRule").checked=!!o.exclude_rule_based;$("oPy").checked=!!o.exclude_python_ua_normal;
 sO.dates=c&&c.dates&&c.dates.length?c.dates.slice():null;sO.sources=c&&c.sources&&c.sources.length?c.sources.slice():null;sO.rules=Object.assign({},(ov&&ov.rules)||{});sO.types=Object.assign({},(ov&&ov.types)||{})}
async function loadSup(){let s;try{s=await get("/api/sup/state?base="+encodeURIComponent($("sBase").value||""))}catch(e){$("sSetup").innerHTML=`<div class="msg">불러오지 못했습니다: ${esc(e.message)}</div>`;return}
 SS=s;const opts=s.finals.map(n=>`<option${n===s.base?" selected":""}>${esc(n)}</option>`).join("");if($("sBase").innerHTML!==opts)$("sBase").innerHTML=opts;
 if(!sFormInit){sSetForm(s.options,null,null);sFormInit=true}
 $("sHdr").textContent=s.base?`· 기준 ${s.base} · 디스크 여유 ${s.disk_free_gb}GB`:"";
 $("sSetup").innerHTML=s.problems.length?`<div class="est"><b>넣을 것</b><ul style="margin:4px 0 0 18px;padding:0">${s.problems.map(p=>`<li>${esc(p)}</li>`).join("")}</ul><div class="muted">아래 '기준 버전 등록'에서 기존 학습데이터(표준모델 학습 형식 CSV)·원본 로그·표준모델 피처 파일 위치를 넣으세요.</div></div>`:"";
 const run=s.job.running,st=s.status||{};
 for(const id of["sRunNew","sRunAll"])$(id).disabled=run||!s.base||s.problems.length>0;$("sStop").style.display=run?"":"none";$("sSave").disabled=run;
 sEstimate();sJobView(run,st);renderVers();renderReg();
 if(run){clearTimeout(sJT);sJT=setTimeout(()=>{if(page==="sup")loadSup()},2000)}
 if(!sResInit||(sRunWas&&!run)){sResInit=true;loadSupRes();sPreview()}   // 처음 열 때, 비교·확정이 막 끝났을 때
 sRunWas=run;
 try{$("sLog").textContent=(await get("/api/log?job=supervised&lines=80")).text;$("sLog").scrollTop=1e9}catch(e){}}
function sEstimate(){const s=SS;if(!s||!s.base){$("sEst").innerHTML="";return}
 const line=mode=>{const p=(s.plans||{})[mode+($("sInc").checked?"_inc":"")];if(!p)return "-";   // 실행기가 처음 만드는 실행 계획과 같은 계산 (5.0a compare_plan)
  return `피처 계산 ${["ips","waf"].map(d=>`${d.toUpperCase()} ${fmt(p.dev[d].new)}개`).join(" · ")} → <b>${dur(p.total_s)}</b>`};
 const ch=["ips","waf"].map(d=>{const x=s.dev[d];if(!x.has)return `${d.toUpperCase()} 아직 비교 안 함`;
   const bad=(x.changed||[]).concat(x.new_dates||[]);return `${d.toUpperCase()} ${esc((x.created_at||"").replace("T"," "))} 비교 (${fmt(x.rows)}행)`+(bad.length?` · <b>그 뒤 바뀌거나 새로 생긴 날짜 ${esc(bad.join(", "))}</b> → 다시 비교해야 이 데이터가 들어갑니다`:" · 그 뒤 변경 없음")}).join("<br>");
 $("sEst").innerHTML=`${ch}<div style="margin-top:4px"><b>예상</b> 새로 생긴 데이터만: ${line("new")} · 전체 다시: ${line("all")}<div class="muted">단계별 예상 = 지난 실행 시간 (실행 중에는 실제 속도로 고침). 피처는 샘플마다 한 번만 계산해 저장하고, 기준 데이터·기준 모델은 처음 한 번만 준비합니다. 생성 등 다른 계산이 돌면 CPU 를 나눠 써서 더 걸립니다.</div></div>`}
function isoClock(x){if(!x)return "";const d=new Date(x);return isNaN(d)||d.toDateString()===new Date().toDateString()?x.slice(11,16):`${d.getMonth()+1}/${d.getDate()} ${x.slice(11,16)}`}
function sJobView(run,st){const el=$("sJob");if(!st||!st.state){if(!run)el.innerHTML="";return}   // 실행 중에 비어 오면 지난 화면 그대로
 const what=st.cmd==="finalize"?`버전 확정 '${esc(st.version||"")}'`:`비교 (기준 ${esc(st.base||"")} · ${(st.devs||[]).map(d=>d.toUpperCase()).join("·")} · ${st.mode==="all"?"전체 다시":"새로 생긴 데이터만"})`;
 if(!st.plan){sJobOld(el,run,st,what);return}
 // 실행 계획 (5.0 RunCtx): 전체 % · 남은 시간 · 끝날 시각, 단계 = 끝남(✔ 실제 시간) / 진행 중(● 지금 진행 수) / 대기(○ 예상 시간)
 const live=run&&st.state==="running",pct=st.state==="done"?100:Math.floor(st.overall_pct||0),icon={done:"✔",running:"●",wait:"○",skip:"–"};
 const chip=p=>{const stop=p.state==="running"&&!live,tail=p.state==="done"?hms(p.sec):p.state==="running"?(live?(st.detail||hms(p.sec)+"째")+(st.eta_s!=null?` · 약 ${hms(st.eta_s)} 남음`:""):hms(p.sec)):p.state==="wait"?"약 "+hms(p.expected_s):"";
  return `<span class="step ${stop?"fail":p.state}">${stop?"✘":icon[p.state]||""} ${esc(p.text)}${tail?` <b>${esc(tail)}</b>`:""}</span>`};
 const grp=[];for(const p of st.plan.filter(p=>p.state==="running"||p.state==="wait")){const g=p.dev?p.dev.toUpperCase()+" ":"",l=grp[grp.length-1];if(l&&l[0]===g)l[1].push(p.short);else grp.push([g,[p.short]])}
 const left=grp.map(([g,a])=>g+a.join(" → ")).join(" · ");
 const head=live?`<b>${what} 중</b> · 전체 <b>${pct}%</b> · 남은 약 <b>${hms(st.eta_total_s)}</b> · <b>${isoClock(st.finish_at)}</b> 쯤 끝남 · 경과 ${hms(st.elapsed_s)}${SS.job.stop_requested?" · <b>중지 요청됨</b>":""}`
  :st.state==="done"?`<b>${what} 끝남</b> · ${esc((st.finished_at||"").replace("T"," "))} · 걸린 시간 <b>${hms(st.elapsed_s)}</b>${st.first_eta_s!=null?` <span class="muted">(시작할 때 예상 ${hms(st.first_eta_s)} · ${isoClock(st.first_finish_at)} 끝날 예상)</span>`:""}`
  :st.state==="stopped"?`<b>${what} 중지됨</b> · ${pct}% 에서 멈춤 · `+(st.cmd==="finalize"?"만들던 파일은 지움":"계산한 피처는 저장됨 → '새로 생긴 데이터만 비교'로 이어서 할 수 있음")
  :st.state==="fail"?`<b>${what} 실패</b> · ${esc(st.error||"")} <span class="muted">(실행 로그 확인)</span>`:`<b>${what}</b> 끝나지 않음 (화면 밖에서 멈춤)`;
 el.innerHTML=`<div class="row" style="align-items:center;gap:16px;flex-wrap:nowrap"><div class="pct">${pct}%</div><div style="flex:1;min-width:0">
  <div class="bar${live?" live":""}" style="margin:4px 0 6px"><i style="width:${pct}%"></i></div><div style="font-size:13px">${head}</div>
  ${live&&left?`<div class="runnow">남은 일: ${esc(left)}</div>`:""}</div></div><div class="steps">${st.plan.map(chip).join("")}</div>`}
function sJobOld(el,run,st,what){   // 예전 실행기가 쓴 상태 (실행 계획 없이 단계만)
 const steps=(st.steps||[]).map((x,i,a)=>{const live=run&&i===a.length-1;return `<span class="step ${x.sec!=null&&!live?"done":live?"running":"fail"}">${x.sec!=null&&!live?"✔":live?"●":"✘"} ${esc(x.text)}${x.sec!=null&&!live?` <b>${hms(x.sec)}</b>`:""}</span>`}).join("");
 let head="";
 if(run){const pr=st.total?`${fmt(st.done)} / ${fmt(st.total)}${st.unit||""}`:st.done!=null?`${fmt(st.done)}${st.unit||""}`:"";
  head=`<b>${what} 중</b> · ${esc(st.step||"")} ${pr}${st.eta_s!=null?` · 이 단계 남은 시간 약 <b>${hms(st.eta_s)}</b> (${clock(st.eta_s)} 쯤)`:""}${SS.job.stop_requested?" · <b>중지 요청됨</b>":""}`+
   (st.total?`<div class="bar live" style="margin:6px 0"><i style="width:${Math.min(100,100*st.done/st.total)}%"></i></div>`:"")}
 else head=st.state==="done"?`<b>${what} 끝남</b> · ${esc((st.finished_at||"").replace("T"," "))}`:st.state==="stopped"?`<b>${what} 중지됨</b> · 계산한 피처는 저장됨 → '새로 생긴 데이터만 비교'로 이어서 할 수 있음`:st.state==="fail"?`<b>${what} 실패</b> · ${esc(st.error||"")} <span class="muted">(실행 로그 확인)</span>`:`<b>${what}</b> 끝나지 않음 (화면 밖에서 멈춤)`;
 el.innerHTML=`<div style="font-size:13px">${head}</div><div class="steps">${steps}</div>`}
const PCT_T=new Set(["흔적: User-Agent"]);
function scell(v,col,tn){if(v==null||v==="")return"";if(typeof v==="number"){if(Number.isInteger(v))return fmt(v);if(/율/.test(col)||PCT_T.has(tn))return (100*v).toFixed(1)+"%";return v.toFixed(1)}return esc(v)}
function stable(r,name){const t=r.tables[name];if(!t)return"";return `<h4>${esc(name)}</h4>${t.note?`<div class="muted" style="font-size:12px">${esc(t.note)}</div>`:""}<table class="mini"><thead><tr>${t.cols.map((c,i)=>`<th${i?' class="n"':""}>${esc(c)}</th>`).join("")}</tr></thead><tbody>${t.rows.map(row=>`<tr>${row.map((v,i)=>`<td${i?' class="n"':""}>${i===0&&name==="등급별"&&typeof v==="string"&&v!=="합계"?gchip(v.split(" ")[0])+" "+esc(v.split(" ").slice(1).join(" ")):scell(v,t.cols[i],name)}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
async function loadSupRes(){if(!SS||!SS.base)return;let r;try{r=await get(`/api/sup/summary?base=${encodeURIComponent(SS.base)}&dev=${sDev}`)}catch(e){return}
 $("sXlsx").href=`/download/sup/report?base=${encodeURIComponent(SS.base)}&dev=${sDev}`;$("sXlsx").style.visibility=r.rows?"visible":"hidden";
 if(!r.rows){$("sRes").innerHTML='<div class="muted">이 장비는 아직 비교하지 않았습니다 - 위에서 비교를 실행하세요.</div>';$("sResHdr").textContent="";return}
 $("sResHdr").textContent=`· ${r.created_at.replace("T"," ")} · ${fmt(r.rows)}행 · 스냅숏 ${r.snapshot}`;
 $("sRes").innerHTML=`<div class="kv">${r.kv.map(([k,v,n])=>`<b>${esc(k)}</b><span>${esc(v)}${n?` <span class="muted">${esc(n)}</span>`:""}</span>`).join("")}</div>
  <div class="tw"><div>${stable(r,"등급별")}${stable(r,"A 근거")}</div><div>${stable(r,"기준 모델의 신규 탐지")}${stable(r,"품질")}</div></div>
  <details><summary>공격 유형·출처·날짜별, label 정합성, 데이터 흔적, 평가 분할</summary><div class="tw"><div>${stable(r,"공격 유형별")}</div><div>${stable(r,"출처별")}${stable(r,"날짜별")}${stable(r,"label 정합성")}</div><div>${stable(r,"흔적: User-Agent")}${stable(r,"흔적: 헤더·길이")}${stable(r,"표준모델 피처 1개만 켜지는 공격")}${stable(r,"평가 분할")}</div></div></details>`}
function sPreviewSoon(){clearTimeout(sPvT);sPvT=setTimeout(sPreview,250)}
async function sPreview(){clearTimeout(sPvT);if(!SS||!SS.base)return;let r;
 try{r=await post("/api/sup/preview",Object.assign({base:SS.base,dev:sDev,view:$("sView").value},sSpec()))}catch(e){$("sPrev").innerHTML=`<div class="muted">미리 보기 실패: ${esc(e.message)}</div>`;return}
 if(r.none){sPv=null;$("sPrev").innerHTML=`<div class="muted">IPS·WAF 비교 결과가 둘 다 있어야 공통 선별을 볼 수 있습니다${r.missing&&r.missing.length?` (없음: ${esc(r.missing.join("·"))})`:""}.</div>`;$("sRules").innerHTML="";$("sRuleCnt").textContent="";$("sSelHdr").textContent="";$("cDates").innerHTML=$("cSrc").innerHTML=$("sHold").innerHTML="";return}
 sPv=r;renderConds(r);renderHold(r);const vw=r.view;
 $("sSelHdr").textContent=vw?`· IPS·WAF 공통 · 보류 데이터만 (${HOLDT[vw]}) ${fmt(r.cand)}개 · 그중 지금 선별에 들어간 것 ${fmt(r.hold[vw][1])}개`
  :`· IPS·WAF 공통 후보 ${fmt(r.cand)}개 중 선택 ${fmt(r.sel)}개 (공격 ${fmt(r.sel_label["공격"])} / 정상 ${fmt(r.sel_label["정상"])}) → 두 장비에 같은 샘플`;
 const ovl=r.overlap,os=r.one_side||{},ovr=k=>{const o=ovl[k];return `<b>${k}</b> 둘 다 ${fmt(o.cand[0])} · IPS 만 ${fmt(o.cand[1])} · WAF 만 ${fmt(o.cand[2])} <span class="muted">(선택 ${fmt(o.sel[0])} · ${fmt(o.sel[1])} · ${fmt(o.sel[2])})</span>`};
 const ovHtml=ovl?`<div class="est" style="margin-bottom:8px">합친 등급이 어느 장비에서 왔는지 (후보 기준): ${ovr("A")} &nbsp;|&nbsp; ${ovr("B")}`+
  ((os.ips_only||os.waf_only||os.label_mismatch)?`<div class="muted">한쪽 비교 결과에만 있는 샘플 IPS ${fmt(os.ips_only)} · WAF ${fmt(os.waf_only)}${os.label_mismatch?` · label 이 다른 샘플 ${fmt(os.label_mismatch)}`:""} → 공통 선별에서 뺌 (두 장비 모두 '새로 생긴 데이터만 비교'를 하면 맞춰짐)</div>`:"")+"</div>":"";
 const bg=r.by_grade,f=(g,l)=>bg.find(x=>x[0]===g&&x[1]===l)||[g,l,0,0];
 const rowsG=["제거","A","B","C"].map(g=>{const a=f(g,"공격"),n=f(g,"정상");return `<tr><td>${gchip(g)}</td><td class="n">${fmt(a[2])}</td><td class="n"><b>${fmt(a[3])}</b></td><td class="n">${fmt(n[2])}</td><td class="n"><b>${fmt(n[3])}</b></td></tr>`}).join("");
 const why=Object.entries(r.why).filter(([k,v])=>v).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${fmt(v)}</td></tr>`).join("");
 const by=(rows,label,tcol)=>`<table class="mini"><thead><tr><th>${label}</th><th class="n">후보</th><th class="n">선택</th><th class="n">공격</th><th class="n">정상</th>${tcol?"<th>빼기</th>":""}</tr></thead><tbody>${rows.map(x=>`<tr><td>${esc(x[0])}</td><td class="n">${fmt(x[1])}</td><td class="n"><b>${fmt(x[2])}</b></td><td class="n">${fmt(x[3])}</td><td class="n">${fmt(x[4])}</td>${tcol?`<td><input type="checkbox" data-t="${esc(x[0])}"${sO.types[x[0]]?" checked":""}></td>`:""}</tr>`).join("")}</tbody></table>`;
 $("sPrev").innerHTML=ovHtml+`<div class="tw"><div><h4>등급별 (${vw?"보류 데이터 → 그중 지금 선별에 들어간 것":"합친 등급 · 후보 → 선택"})</h4><table class="mini"><thead><tr><th>등급</th><th class="n">공격 후보</th><th class="n">공격 선택</th><th class="n">정상 후보</th><th class="n">정상 선택</th></tr></thead><tbody>${rowsG}<tr><td><b>합계</b></td><td class="n" colspan="4">${vw?`<b>${fmt(r.hold[vw][1])}</b> / 보류 데이터 ${fmt(r.cand)}`:`<b>${fmt(r.sel)}</b> / 후보 ${fmt(r.cand)}`}</td></tr></tbody></table>
  ${vw?`<div class="est">보류 데이터만 보는 중입니다 (${HOLDT[vw]}). 표·룰별 표·CSV 가 이 데이터만 셉니다. 보류 옵션을 켜면 '지금 선별에 들어간 것'이 빠집니다. <a data-hv="">선별 전체로 돌아가기</a></div>`
   :`<h4>뺀 이유</h4><table class="mini"><tbody>${why||'<tr><td class="muted">없음</td></tr>'}</tbody></table>`}</div>
  <div><h4>공격 유형별 (빼기 = 그 유형 모두 뺌)</h4>${by(r.by_type,"공격 유형",true)}</div>
  <div><h4>출처별</h4>${by(r.by_source,"출처")}<h4>날짜별</h4>${by(r.by_date,"날짜")}</div></div>`;
 $("sPrev").querySelectorAll("[data-t]").forEach(c=>c.onchange=()=>{if(c.checked)sO.types[c.dataset.t]="exclude";else delete sO.types[c.dataset.t];sPreviewSoon()});
 renderRules();
 $("sCand").textContent=vw?"보류 데이터 CSV 받기 (IPS·WAF 공통)":"후보 CSV 받기 (IPS·WAF 공통)";
 $("sCand").href="/download/sup/candidates?"+new URLSearchParams({base:SS.base,dev:sDev,spec:JSON.stringify(sSpec()),view:vw||""})}
const HOLDT={rule:"'정합 (룰 기준)' 공격",pyua:"python-requests UA 정상"};
function renderHold(r){const h=r.hold||{},one=k=>{const x=h[k]||[0,0,0,0,0,0];
  const unv=k==="rule"&&r.unverified_att?` <span class="muted">(검증 전 공격 ${fmt(r.unverified_att)}개는 셀 수 없음 → 4.에서 '검증 안 한 데이터만 검증' 뒤 '새로 생긴 데이터만 비교')</span>`:"";
  return `<span><b>${HOLDT[k]}</b> ${fmt(x[0])}개`+(x[0]?` <span class="muted">(제거 ${fmt(x[2])} · A ${fmt(x[3])} · B ${fmt(x[4])} · C ${fmt(x[5])}) · 지금 선별에 ${fmt(x[1])}개 = 옵션을 켜면 빠지는 수</span> <a data-hv="${k}">6.에서 보기</a> · <a data-h4="${k}">4.에서 보기</a>`:"")+unv+"</span>"};
 $("sHold").innerHTML=`<span class="muted">보류 대상 (조건 안의 후보 중)</span>${one("rule")}${one("pyua")}`}
document.getElementById("p-sup").addEventListener("click",e=>{const a=e.target.closest("a[data-hv],a[data-h4]");if(!a)return;e.preventDefault();
 if(a.dataset.h4){dPending={hold:a.dataset.h4,dev:sDev};location.hash="#/data";return}
 $("sView").value=a.dataset.hv;sPreview()});
$("sView").onchange=sPreview;
function renderConds(r){const box=(id,list,key,attr)=>{const el=$(id);el.innerHTML=list.map(d=>`<label class="ck"><input type="checkbox" data-${attr}="${esc(d)}"${!sO[key]||sO[key].includes(d)?" checked":""}>${esc(d)}</label>`).join("");
  el.querySelectorAll("input").forEach(c=>c.onchange=()=>{const on=[...el.querySelectorAll("input:checked")].map(x=>x.dataset[attr]);sO[key]=on.length===list.length?null:on.length?on:["-"];sPreviewSoon()})};
 box("cDates",r.all_dates,"dates","cd");box("cSrc",r.all_sources,"sources","cs")}
function renderRules(){const r=sPv;if(!r){$("sRules").innerHTML="";return}const q=$("sRuleQ").value.trim().toLowerCase(),f=$("sRuleF").value;
 let rows=r.rules.filter(x=>(!q||x[0].toLowerCase().includes(q)||x[1].toLowerCase().includes(q))&&(f!=="ov"||x[8])&&(f!=="sel0"||!x[7]));
 const n=rows.length;rows=rows.slice(0,300);$("sRuleCnt").textContent=`${fmt(n)}개 룰${n>300?" 중 300개 표시 (검색으로 좁히세요)":""} · 조정 ${Object.keys(sO.rules).length}개`;
 $("sRules").innerHTML=`<table class="mini"><thead><tr><th>룰 id</th><th>공격 유형</th><th class="n">후보</th><th class="n">제거</th><th class="n">A</th><th class="n">B</th><th class="n">C</th><th class="n">선택</th><th>조정</th><th></th></tr></thead><tbody>${rows.map(x=>`<tr><td class="nw">${esc(x[0])}</td><td>${esc(x[1])}</td>${[2,3,4,5,6].map(i=>`<td class="n">${fmt(x[i])}</td>`).join("")}<td class="n"><b>${fmt(x[7])}</b></td><td><select data-r="${esc(x[0])}"><option value="">기본</option><option value="exclude"${x[8]==="exclude"?" selected":""}>빼기</option><option value="include"${x[8]==="include"?" selected":""}>넣기 (상한 없이)</option></select></td><td class="nw"><a href="#/data" data-open="${esc(x[0])}">4.에서 보기</a></td></tr>`).join("")}</tbody></table>`}
$("sRules").addEventListener("change",e=>{const s=e.target.closest("select[data-r]");if(!s)return;if(s.value)sO.rules[s.dataset.r]=s.value;else delete sO.rules[s.dataset.r];sPreviewSoon()});
$("sRules").addEventListener("click",e=>{const a=e.target.closest("a[data-open]");if(!a)return;e.preventDefault();dPending={rule:a.dataset.open,dev:sDev,hold:$("sView").value};location.hash="#/data"});
function vDesc(x){if(x.kind==="external")return esc(x.note||"");const o=x.options||{},c=x.conditions||{},ov=x.overrides||{},cap=k=>o[k]==="all"?"전부":o[k]===0?"빼기":`패턴당 ${o[k]}`;
 return `A ${cap("A")} · B ${cap("B")} · C ${cap("C")}`+(o.exclude_bad?" · 불일치 빼기":"")+(o.exclude_rule_based?" · 정합(룰 기준) 공격 빼기":"")+(o.exclude_python_ua_normal?" · python-requests 정상 빼기":"")+
  `<br>날짜 ${c.dates&&c.dates.length?esc(c.dates.join(", ")):"전부"} · 출처 ${c.sources&&c.sources.length?esc(c.sources.join(", ")):"전부"}`+(Object.keys(ov.rules||{}).length?` · 룰 조정 ${Object.keys(ov.rules).length}개`:"")+(Object.keys(ov.types||{}).length?` · 유형 빼기 ${esc(Object.keys(ov.types).join(", "))}`:"")}
function renderVers(){const vs=SS.versions;$("sVerHdr").textContent=vs.length?`· ${vs.length}개 (확정 ${vs.filter(x=>x.status==="final").length})`:"";
 if(!vs.length){$("sVers").innerHTML='<div class="muted">버전이 없습니다 - 아래에서 기준 버전(기존 학습데이터)을 등록하세요.</div>';return}
 const dv=(x,d)=>{const c=x.dev[d]||{};if(x.status==="draft"&&c.draft)return `<span class="muted">추가 후보</span> <b>${fmt(c.draft.sel)}</b><div class="muted">공격 ${fmt(c.draft.sel_label["공격"])} / 정상 ${fmt(c.draft.sel_label["정상"])}</div>`+(x.mode==="combined"?'<div class="muted">IPS·WAF 같은 샘플</div>':'<div class="muted">장비별 (예전 방식)</div>');
   if(c.rows==null)return '<span class="muted">-</span>';const l=c.label||{};return `${fmt(c.rows)}<div class="muted">공격 ${fmt(l.anomalies)} / 정상 ${fmt(l.normal)}</div>`+(c.added!=null?`<div class="muted">기준 + 추가 ${fmt(c.added)}</div>`:"")};
 $("sVers").innerHTML=`<table class="mini"><thead><tr><th>버전</th><th>기준</th><th>상태</th><th>IPS 행</th><th>WAF 행</th><th>조건·선별</th><th>시각</th><th></th></tr></thead><tbody>${vs.map(x=>`<tr><td class="nw"><b>${esc(x.name)}</b>${x.tag?` <span class="muted">${esc(x.tag)}</span>`:""}</td><td class="nw">${esc(x.base||"-")}</td><td>${x.status==="final"?'<span class="chip c-OK">확정</span>':'<span class="chip c-CHECK">초안</span>'}</td><td class="nw">${dv(x,"ips")}</td><td class="nw">${dv(x,"waf")}</td><td class="reason">${vDesc(x)}</td><td class="muted nw">${esc((x.finalized_at||x.updated_at||x.created_at||"").replace("T"," ").slice(0,16))}</td><td class="nw">${x.status==="draft"?`<button class="sm" data-v="load" data-n="${esc(x.name)}">불러오기</button>`+(x.mode==="combined"?`<button class="sm primary" data-v="final" data-n="${esc(x.name)}">확정</button>`:`<button class="sm primary" data-v="resave" data-n="${esc(x.name)}" title="옵션·조정은 그대로, IPS·WAF 에 같은 샘플을 넣는 방식으로 다시 계산">새 방식으로 다시 담기</button>`)+`<button class="sm danger" data-v="del" data-n="${esc(x.name)}">삭제</button>`+(x.mode==="combined"?"":'<div class="muted" style="font-size:12px;white-space:normal;max-width:240px">장비별로 따로 고른 예전 초안 - 다시 담아야 확정됩니다</div>'):x.kind==="external"?'<span class="muted">등록한 기준</span>':`<span class="muted">11_supervised/versions/${esc(x.name)}/</span>`}</td></tr>`).join("")}</tbody></table>`}
$("sVers").addEventListener("click",async e=>{const b=e.target.closest("button[data-v]");if(!b)return;const name=b.dataset.n,x=SS.versions.find(v=>v.name===name);
 try{if(b.dataset.v==="load"){sSetForm(Object.assign({},SS.options,x.options),x.conditions,x.overrides);$("sVName").value=name;sPreview();$("sSaveMsg").textContent=`'${name}' 초안을 불러왔습니다`;return}
  if(b.dataset.v==="del"){if(!confirm(`초안 '${name}' 을 지웁니다. 진행할까요?`))return;await post("/api/sup/delete",{name});loadSup();return}
  if(b.dataset.v==="resave"){   // 예전 장비별 초안 → 같은 옵션·조정으로 IPS·WAF 공통 선별 다시 계산
   const r=await post("/api/sup/save",{name,base:x.base,conditions:x.conditions,options:x.options,overrides:x.overrides});
   $("sSaveMsg").textContent=`'${name}' 을 새 방식으로 다시 담았습니다: IPS·WAF 같은 샘플 ${fmt(r.draft_counts.ips.sel)}개`;loadSup();return}
  const n=x.dev.ips&&x.dev.ips.draft?x.dev.ips.draft.sel:0;
  if(!confirm(`'${name}' 을 확정합니다 (IPS·WAF 에 같은 추가 샘플 ${fmt(n)}개).\n기준 '${x.base}' + 추가 샘플로 IPS·WAF 학습 파일(CSV)·로그를 각각 만듭니다 (장비마다 약 0.5GB, 디스크 여유 ${SS.disk_free_gb}GB).\n저장한 뒤 두 장비의 비교 결과·학습데이터가 바뀌었으면 확정하지 않습니다. 확정한 버전은 고칠 수 없고 다음 버전의 기준이 됩니다. 진행할까요?`))return;
  await post("/api/sup/finalize",{name});setTimeout(loadSup,800)}catch(err){alert("실패: "+err.message)}});
function renderReg(){if($("sReg").dataset.done)return;$("sReg").dataset.done="1";const d=SS.defaults,fc=SS.feature_code||{};
 $("sReg").innerHTML=`<div class="muted" style="font-size:12.5px;margin:6px 0">첫 기준 버전(예: 26 상반기 = v26.02.02)의 파일 위치를 넣습니다. 공개본을 받아 쓸 때 넣을 것: 사용자 룰(<code>00_custom_rules</code>), 기존 학습 로그(<code>06_기존 데이터/{ips,waf}_*_raw.log</code>), 기준 학습데이터(표준모델 학습 형식 CSV: method, payload, label, {장비}_payload_* 피처 → <code>06_기존 데이터/학습데이터/</code>), 표준모델 피처 파일(노트북을 변환한 .py, '# 1) 피처 생성' 앞에 정규식 사전 → <code>06_기존 데이터/피처 코드/</code>).</div>
 <div class="row"><label>이름<input id="rgName" placeholder="26 상반기" style="width:120px"></label><label>태그<input id="rgTag" placeholder="v26.02.02" style="width:110px"></label></div>
 ${["ips","waf"].map(dev=>`<div class="row" style="margin-top:6px"><b style="align-self:center;width:40px">${dev.toUpperCase()}</b><label>학습 파일 CSV<input id="rg_${dev}_csv" style="width:320px" value="${esc(d[dev].train_csv)}"></label><label>원본 로그<input id="rg_${dev}_log" style="width:300px" value="${esc(d[dev].raw_log)}"></label><label>표준모델 피처 파일<input id="rg_${dev}_code" style="width:300px" value="${esc(fc[dev]||d[dev].feature_code)}"></label></div>`).join("")}
 <div class="row" style="margin-top:8px"><button id="rgGo">등록</button><span class="muted" id="rgMsg" style="align-self:center"></span></div>`;
 $("rgGo").onclick=async()=>{const files={},code={};for(const dev of["ips","waf"]){files[dev]={train_csv:$(`rg_${dev}_csv`).value.trim(),raw_log:$(`rg_${dev}_log`).value.trim()};code[dev]=$(`rg_${dev}_code`).value.trim()}
  try{await post("/api/sup/register",{name:$("rgName").value.trim(),tag:$("rgTag").value.trim(),files,feature_code:code});$("rgMsg").textContent="등록했습니다";loadSup()}catch(e){$("rgMsg").textContent="실패: "+e.message}}}
async function sRun(mode){if(mode==="all"&&!confirm("피처·기준 모델을 모두 다시 계산합니다 (IPS·WAF, 수십 분). 진행할까요?"))return;
 try{await post("/api/sup/compare",{base:SS.base,mode,include_compared:$("sInc").checked});setTimeout(loadSup,800)}catch(e){alert("실패: "+e.message)}}
$("sRunNew").onclick=()=>sRun("new");$("sRunAll").onclick=()=>sRun("all");$("sInc").onchange=sEstimate;
$("sStop").onclick=async()=>{try{await post("/api/sup/stop",{});loadSup()}catch(e){alert("실패: "+e.message)}};
$("sBase").onchange=()=>{sPv=null;sResInit=false;loadSup()};
$("sDevSeg").addEventListener("click",e=>{const b=e.target.closest("button[data-d]");if(!b)return;sDev=b.dataset.d;$("sDevSeg").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));loadSupRes()});   // 선별은 IPS·WAF 공통이라 그대로
for(const id of["oA","oB","oC","oBad","oRule","oPy"])$(id).onchange=sPreviewSoon;
let sRq;$("sRuleQ").oninput=()=>{clearTimeout(sRq);sRq=setTimeout(renderRules,200)};$("sRuleF").onchange=renderRules;
$("sSave").onclick=async()=>{const name=$("sVName").value.trim();if(!name){$("sSaveMsg").textContent="버전 이름을 넣으세요 (예: 26 하반기)";return}
 if(SS.versions.some(v=>v.name===name&&v.status==="draft")&&!confirm(`초안 '${name}' 을 지금 선별로 덮어씁니다. 진행할까요?`))return;
 try{const r=await post("/api/sup/save",Object.assign({name,base:SS.base},sSpec()));$("sSaveMsg").textContent=`'${name}' 에 담았습니다: IPS·WAF 같은 샘플 ${fmt(r.draft_counts.ips.sel)}개 → 아래 버전 표에서 확인 후 확정`;loadSup()}catch(e){$("sSaveMsg").textContent="실패: "+e.message}};
/* ── 주기적 갱신 ── */
let lastRunning=false,ticks=0;
async function tick(){try{await loadState();const running=S.generation.running;
 if((lastRunning&&!running)||(running&&++ticks%20===0)){await post("/api/progress/refresh");setTimeout(()=>{loadProgress();if(page==="rules")loadRules()},4000)}
 if(lastRunning&&!running&&page==="run")setTimeout(()=>{loadExcluded();loadModels(true)},5000);   // 생성이 끝나면 검토에서 뺀 샘플·모델별 기록도 새로
 if(running&&ticks%20===0&&page==="run"&&String((S.generation.lock||{}).by||"").includes("redo"))loadExcluded();   // 다시 만들기 중에는 1분마다
 lastRunning=running}catch(e){$("hdr").textContent="화면 서버에 연결할 수 없습니다 ("+e.message+")"}}
(async()=>{route();await loadProgress();await tick();setInterval(tick,3000);setInterval(()=>{if(page==="progress")loadProgress()},15000);
 if(!P||(Date.now()-new Date(P.computed_at))>5*60e3){await post("/api/progress/refresh");setTimeout(loadProgress,4000)}})();
</script></body></html>
"""


class Server(ThreadingHTTPServer):
    """다른 프로그램이 같은 포트를 쓰고 있으면 확실히 실패하게 (Windows 는 0.0.0.0 과 127.0.0.1 을 겹쳐 열 수 있음)."""
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def port_busy(port):
    """누가 이미 그 포트에서 응답하는지 (예: 다른 Streamlit 앱)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description="수집·학습데이터 생성 화면 (로컬)")
    ap.add_argument("--port", type=int, default=8765, help="포트 (쓰고 있으면 다음 빈 포트로)")
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    args = ap.parse_args()
    if sys.stdout is not None:          # 윈도우 콘솔(cp949)에서 못 쓰는 글자가 있어도 멈추지 않게
        sys.stdout.reconfigure(errors="replace")
    server = None
    for port in range(args.port, args.port + 20):
        if port_busy(port):
            print(f"포트 {port} 는 다른 프로그램이 쓰는 중 - 다음 포트를 봅니다")
            continue
        try:
            server = Server(("127.0.0.1", port), Handler)
            break
        except OSError:
            print(f"포트 {port} 를 열 수 없음 - 다음 포트를 봅니다")
    if server is None:
        print(f"포트 {args.port}~{args.port + 19} 를 모두 쓸 수 없습니다. --port 로 다른 번호를 주세요.")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"화면: {url}  (끄려면 Ctrl+C. 실행 중인 수집·생성 작업은 계속됩니다)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
