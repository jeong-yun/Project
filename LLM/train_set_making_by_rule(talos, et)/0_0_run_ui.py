"""[0.0 화면] 브라우저에서 수집(에이전트)·학습데이터 생성을 실행하고, 진행률·룰별 상태·만든 학습데이터를 보는 로컬 화면.

    python 0_0_run_ui.py              → http://127.0.0.1:8765 이 브라우저로 열린다 (이 PC 에서만 접속 가능)
    python 0_0_run_ui.py --port 9000 --no-browser

[페이지]
  1. 진행 현황 (첫 화면)  생성 대상 룰 중 완료 %, 현재 룰 중 학습데이터 보유 %, 상태 플래그별 수, 어디까지(배치),
                          출처·공격 유형별 %, 남은 요청·예상 시간·토큰 (3_5_progress.py 계산)
  2. 생성·수집            학습데이터 생성(건수 확인·파일럿·시작·중지·강제 중지, 3_0_run_generation.py)
                          수집 에이전트 시작·중지(매일 정해진 시각), 지금 1회 수집 (1_0_run_daily.py)
  3. 룰별 상태            상태 플래그·출처·검색으로 찾기, '제외' 룰의 제외 사유, CSV 내려받기 (07_llm_output/_progress/rule_status.csv)
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
    "recheck": ("3_6_recheck_labels.py", "recheck"),   # 4. '불일치' 줄 LLM 재확인 (3_6, 별도 프로세스)
    "features": ("4_1_build_features.py", "features_view"),   # 5. 피처 '켜진 줄 수 다시 계산' (4_1 --view, 표준·신규 피처 표시 캐시)
    "evalver": ("5_4_feature_experiment.py", "eval_version"),  # 6. 후보 버전 '피처 테스트' (5_4 eval-version, 장비마다 한 번씩)
    "server": ("5_8_server.py", "server_check"),   # 6. 후보 '서버 반영' 로컬 사전 점검 (5_8 check, 95%·98% 미리 보기)
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
    cfirst = bool((lock or {}).get("custom_first"))     # 생성 순서: 사용자 룰 먼저
    if by == "3_0":
        for ts, lv, msg in _log_since("3_0_run_generation.log", since):
            if msg.startswith("입력 날짜"):
                if steps:                               # 다음 실행이 시작됨 → 여기까지가 이 실행
                    until = last_ts
                    break
                dry = dry or "(dry-run)" in msg
                cfirst = cfirst or "사용자 룰 먼저" in msg
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
            "custom_first": cfirst, "elapsed_s": elapsed, "pct": pct, "remaining_s": remaining if not finished else None,
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
    return {"data": data, "computing": _progress["running"], "error": _progress["error"],
            "change": input_change((data or {}).get("input_date"))}


_change = {}            # (앞 입력 날짜, 입력 날짜, 두 룰 파일 수정 시각) → 결과 (None = 세는 중)
_change_lock = threading.Lock()


def input_change(cur):
    """화면 1: 입력 날짜 cur 의 룰이 바로 앞 입력 날짜보다 어떻게 바뀌었는지 (새 룰·빠진 룰과 그중 중복 제거·내용 바뀐 룰).
    룰 파일 2개(각 30MB 안팎)를 읽어 4~5초 걸리므로 처음 보는 파일이면 배경에서 세고, 다음 불러오기부터 보인다."""
    dates = input_dates()
    if cur not in dates or dates.index(cur) == 0:
        return None
    prev = dates[dates.index(cur) - 1]
    try:
        key = (prev, cur) + tuple((BASE / "05_llm_input" / d / "rules_all.jsonl").stat().st_mtime_ns for d in (prev, cur))
    except OSError:
        return None
    with _change_lock:
        if key not in _change:
            _change[key] = None
            threading.Thread(target=_change_work, args=(key, prev, cur), daemon=True).start()
        return _change[key]


def _change_work(key, prev, cur):
    try:
        val = _count_change(prev, cur)
    except Exception as e:      # 덤 정보라 실패해도 진행 현황은 그대로 (같은 파일이면 다시 세지 않음)
        val = {"error": f"{type(e).__name__}: {e}"}
    with _change_lock:
        _change[key] = val


def _count_change(prev, cur):
    with _lock:
        if _check["C"] is None:
            _check["C"] = _load("llm_common_ui", "3_0a_common.py")
        C = _check["C"]
    old, new = ({r["id"]: C.rule_hash(r) for r in C.read_jsonl(BASE / "05_llm_input" / d / "rules_all.jsonl")}
                for d in (prev, cur))
    added, removed = new.keys() - old.keys(), old.keys() - new.keys()

    def dup_keys(date):         # 1.2 통합에서 다른 룰과 같아 뺀 룰 (04_integrated/<날짜>/duplicates.json)
        try:
            s = json.loads((BASE / "05_llm_input" / date / "summary.json").read_text(encoding="utf-8"))
            src = s.get("source_dir") or f"04_integrated/{date}"
            dup = json.loads((BASE / src.replace("\\", "/") / "duplicates.json").read_text(encoding="utf-8"))
            return {r["key"] for r in dup.get("removed", [])}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return set()

    def by_source(ids):
        return dict(Counter(i.split(":")[0] for i in sorted(ids)))

    return {"prev": prev, "cur": cur, "net": len(new) - len(old),
            "added": {"total": len(added), "by_source": by_source(added)},
            "removed": {"total": len(removed), "by_source": by_source(removed),
                        "duplicate": len(removed & (dup_keys(cur) - dup_keys(prev)))},
            "changed": sum(1 for i in new.keys() & old.keys() if new[i] != old[i])}


C_REQ, C_GEN, C_REV, C_FAIL_LIMIT = "_1_requests", "_2_generated", "_3_reviewed", 2   # 3_0a_common.*_DIR, 3_2_generate.FAIL_LIMIT
_creq = {"key": None, "val": None, "busy": False, "files": {}, "metas": {}}  # 결과, 요청 파일 → (수정 시각, 룰 id), 생성 meta → (수정 시각, 빠진 룰)
_creq_lock = threading.Lock()


def custom_requests():
    """화면 2 '생성 순서': 가장 최근 입력 날짜의 생성 대기 요청 중 사용자 룰(id 가 custom:)이 든 요청, 세트별
    {pending: 대기 요청, custom: 사용자 룰이 든 요청, rules: 그 안의 사용자 룰, ahead: 기본 순서(이름순)에서 그 앞의 대기 요청}
    + review_wait: 생성됐지만 검토 안 된 결과 (모든 날짜 폴더 - 3_0 은 요청 수에서 이만큼 덜 생성한다)
    + retry: 다음 실행의 3.1 이 새 요청으로 다시 넣을 사용자 룰 (_custom_retry)
    + flags·noauto: 사용자 룰의 상태별 수, 자동으로 다시 만들지 않는 룰(샘플 없음·생성 실패) id (화면 1·3 과 같은 rule_status).
    요청·생성·검토 폴더나 rule_status 가 바뀌었을 때만 배경에서 다시 센다 (요청 파일은 수정 시각이 같으면 다시 읽지 않음, 처음은 2초쯤).
    세는 동안은 지난 결과, 처음이면 None."""
    dates = input_dates()
    if not dates:
        return {}
    root = BASE / "07_llm_output"
    outs = sorted(p for p in root.glob("*") if DATE_RE.match(p.name)) if root.is_dir() else []

    def mtime(p):               # 파일이 생기거나 지워지면 폴더 수정 시각이 바뀐다
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return 0
    key = (dates[-1], mtime(root / dates[-1] / C_REQ), mtime(RULES_FILE)) + \
        tuple(mtime(d / s) for d in outs for s in (C_GEN, C_REV))
    with _creq_lock:
        if _creq["key"] != key and not _creq["busy"]:
            _creq.update(key=key, busy=True)
            threading.Thread(target=_creq_work, args=(root / dates[-1], outs), daemon=True).start()
        return _creq["val"]


def _creq_work(base, outs):
    try:
        wait = 0
        for d in outs:          # 3_2 --count-pending --pending-all 과 같은 셈
            if (d / C_GEN).is_dir():
                rev = {e.name for e in os.scandir(d / C_REV)} if (d / C_REV).is_dir() else set()
                wait += sum(e.name.endswith(".jsonl") and e.name not in rev for e in os.scandir(d / C_GEN))
        by_set, retry = _count_custom(base)
        flags, noauto = {}, {}                          # 화면 1·3 과 같은 기준 (3.5 rule_status): 사용자 룰 상태별 수
        for r in _rule_rows():
            if r.get("source") == "custom":
                flags[r["flag"]] = flags.get(r["flag"], 0) + 1
                if r["flag"] in ("EMPTY", "FAILED"):    # 3.1·3.2 가 자동으로 다시 만들지 않음
                    noauto.setdefault(r["flag"], []).append(r["id"])
        val = {"date": base.name, "by_set": by_set, "review_wait": wait, "retry": retry, "flags": flags, "noauto": noauto}
    except Exception as e:      # 덤 정보라 실패해도 생성 카드는 그대로 (폴더가 바뀌면 다시 셈)
        val = {"error": f"{type(e).__name__}: {e}"}
    with _creq_lock:
        _creq.update(val=val, busy=False)


def _count_custom(base):
    """→ (세트별 생성 대기 요청 {pending, custom, rules, ahead}, 다음 실행 때 다시 요청할 사용자 룰 _custom_retry).
    요청 파일마다 룰 id 를 읽어 둔다 (수정 시각이 같으면 다시 읽지 않음)."""
    req_dir, gen_dir = base / C_REQ, base / C_GEN
    if not req_dir.is_dir():
        return {}, None         # 새 입력 날짜: 요청 파일은 다음 생성의 준비 단계(3.1)가 만든다
    gen = {e.name for e in os.scandir(gen_dir)} if gen_dir.is_dir() else set()
    old, files, out, ids_of = _creq["files"], {}, {}, {}
    for e in sorted(os.scandir(req_dir), key=lambda e: e.name):     # 3_2 와 같은 이름순 (= 배치 순서)
        m = re.match(r"(all|delta)_b\d+_.*\.jsonl$", e.name)
        if not m:
            continue
        mt = e.stat().st_mtime_ns
        hit = old.get(e.path)
        if not hit or hit[0] != mt:
            try:
                with open(e.path, encoding="utf-8") as f:
                    hit = (mt, tuple(_rule_id(x) for x in f if x.strip()))
            except (OSError, ValueError):       # 3.1 이 쓰는 중 등 → 다음에 다시 읽음
                continue
        files[e.path] = hit
        name = e.name[:-len(".jsonl")]
        ids_of[name] = hit[1]
        s = out.setdefault(m.group(1), {"pending": 0, "custom": 0, "rules": 0, "ahead": 0})  # 요청 파일이 있는 세트 (대기 0 이어도)
        if e.name in gen:                       # 이미 생성한 요청
            continue
        if f"{name}.fail.json" in gen:          # 응답 해석 실패가 반복된 요청은 3_2 가 건너뜀
            try:
                if json.loads((gen_dir / f"{name}.fail.json").read_text(encoding="utf-8")).get("fails", 0) >= C_FAIL_LIMIT:
                    continue
            except (OSError, ValueError):
                pass
        n = sum(i.startswith("custom:") for i in hit[1])
        if n:
            if not s["custom"]:
                s["ahead"] = s["pending"]
            s["custom"] += 1
            s["rules"] += n
        s["pending"] += 1
    _creq["files"] = files
    return out, _custom_retry(base, gen, ids_of)


_ID_HEAD = re.compile(r'\{"id": "([^"\\]*)"')            # 요청 파일 한 줄 = 룰 JSON, id 가 맨 앞 (3.1 write_jsonl)


def _rule_id(line):
    m = _ID_HEAD.match(line)                            # 줄 전체(룰 원문·조건)를 해석하지 않고 id 만
    return m.group(1) if m else str(json.loads(line).get("id", ""))


def _custom_retry(base, gen, ids_of):
    """다음 생성의 준비 단계(3.1)가 새 요청으로 다시 넣을 사용자 룰 = 생성 응답에서 빠진 룰(meta.json rules_missing) 중
    다른 요청에 아직 배정되지 않은 것 (3_1 의 assigned·retry 와 같은 셈, 세트 all). 사용자 룰이 든 배치의 룰만 본다 (묶음 수를 세려고 배치 전체).
    → {rules: [룰 id], requests: 새 요청 중 사용자 룰이 든 수 (3.1 처럼 배치마다 배치 파일 순서로 '호출당 룰 수'씩 묶어 셈)}."""
    gen_dir = base / C_GEN
    cb = {n.split("_")[1] for n, ids in ids_of.items() if n.startswith("all_") and any(i.startswith("custom:") for i in ids)}
    rel = {i for n, ids in ids_of.items() if n.startswith("all_") and n.split("_")[1] in cb for i in ids}
    old, metas, assigned, retry = _creq["metas"], {}, set(), set()
    for name, ids in ids_of.items():
        if not (name.startswith("all_") or f"{name}.jsonl" in gen) or rel.isdisjoint(ids):
            continue
        miss = ()
        if f"{name}.meta.json" in gen:
            p = gen_dir / f"{name}.meta.json"
            try:
                mt = p.stat().st_mtime_ns
                hit = old.get(p)
                if not hit or hit[0] != mt:
                    hit = (mt, tuple(json.loads(p.read_text(encoding="utf-8")).get("rules_missing", [])))
                metas[p] = hit
                miss = hit[1]
            except (OSError, ValueError):
                pass
        assigned |= set(ids) - set(miss)
        retry |= set(miss)
    _creq["metas"] = metas
    redo = retry - assigned
    try:
        per = max(1, int(run_settings().get("호출당 룰 수", 3)))
    except ValueError:
        per = 3
    n_req = 0
    for b in sorted(cb):                                # 배치 파일 순서대로 다시 넣을 룰을 per 개씩 (3_1 과 같은 묶음)
        try:
            with open(BASE / "05_llm_input" / base.name / "batches" / "all" / f"batch_{b[1:]}.jsonl", encoding="utf-8") as f:
                order = [str(json.loads(x).get("id", "")) for x in f if x.strip()]
        except (OSError, ValueError):                   # 배치 파일을 못 읽으면 요청 파일에 나온 순서
            order = list(dict.fromkeys(r for n, ids in ids_of.items() if n.split("_")[1] == b for r in ids))
        chosen = [r for r in order if r in redo]
        n_req += sum(any(r.startswith("custom:") for r in chosen[k:k + per]) for k in range(0, len(chosen), per))
    return {"rules": sorted(r for r in redo if r.startswith("custom:")), "requests": n_req}


def _q(q, key, default=""):
    return q.get(key, [default])[0]


def page_args(q, max_limit=200):
    offset = max(0, int(_q(q, "offset", "0") or 0))
    limit = min(max_limit, max(1, int(_q(q, "limit", "50") or 50)))
    return offset, limit


def _rule_rows():
    """3.5 룰별 상태 (rule_status.jsonl, 바뀔 때만 다시 읽음). 없으면 []."""
    try:
        mtime = RULES_FILE.stat().st_mtime
    except OSError:
        return []
    if _rules_cache["mtime"] != mtime:
        with open(RULES_FILE, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        _rules_cache.update(rows=rows, mtime=mtime)
    return _rules_cache["rows"]


def rules_page(q):
    rows = _rule_rows()
    if not rows:
        return {"total": 0, "rows": []}
    flag, source, text = _q(q, "flag"), _q(q, "source"), _q(q, "q").strip().lower()
    if flag:
        rows = [r for r in rows if r["flag"] == flag]
    if source:
        rows = [r for r in rows if r["source"] == source]
    if text:                    # 제외 사유(3.5 reason, 예전 판에는 없음)로도 찾음
        rows = [r for r in rows if text in r["id"].lower() or text in r["rule_name"].lower()
                or text in r["attack_type"].lower() or text in (r.get("reason") or "").lower()]
    offset, limit = page_args(q)
    return {"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]}


def rule_status_csv():
    """화면 3 'CSV 내려받기': 3.5 가 쓴 rule_status.csv 그대로. 예전 3.5 판이 써서 '제외 사유' 열이 없으면 빈 열을 붙인다."""
    data = CSV_FILE.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
    if not rows or "제외 사유" in rows[0]:
        return data
    rows[0].append("제외 사유")
    for r in rows[1:]:
        r.append("")
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return out.getvalue().encode("utf-8-sig")


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


def matching(date, dev, label, rule, text, cons="", hold="", llm=""):
    """조건에 맞는 줄 번호 목록 (0부터). 조건이 없으면 None (= 모든 줄). cons = label 정합성 (전체 검증 결과 필요).
    hold = 보류 대상 (6. 선별의 보류 옵션과 같은 기준): rule = '정합 (룰 기준)' 공격, pyua = python-requests UA 정상.
    llm = LLM 재확인 (3_6): 재확인 대상('불일치'·'정합 (룰 기준)') 줄 중 그 판정, any = 재확인한 것 전부,
    todo = '불일치' 중 아직 안 함 ('정합 (룰 기준)' 은 룰당 표본만 묻으므로 '아직'을 세지 않음),
    c_keep·c_delete·c_review·c_flip = 합의가 확인됨·삭제·추가 분석·label 반대(사람)인 줄 (교차 확인, 2026-11-02~)."""
    path = _log_path(date, dev)
    if hold == "rule":
        label, cons = "anomalies", "RULE"
    elif hold == "pyua":
        label = "normal"
    if not (label or rule or text or cons or llm):
        return None
    key = (str(path), path.stat().st_mtime, label, rule, text, cons, hold, llm,
           (recheck_stamp(), _dec_stamp()) if llm else None, _edits_mtime(date) if cons in ("BAD", "VOTED") else None)
    if key in _filters:
        return _filters[key]
    n = len(line_index(path))
    keep = list(range(n))
    if cons or llm:
        v = verify_state(date, dev)
        if v.get("status") != "done":                   # 까닭(검증 안 함·낡음·검증 중)과 할 일 - '전체' 보기는 select_rows 가 그 날짜를 뺌
            raise NoVerify(verify_gap(date, dev))
    if cons == "VOTED":                                 # 합의로 label 바꿈 (검증 '불일치' 중 LLM 재확인으로 label 을 바꿔 반영한 줄)
        vl = voted_lines(date, dev)
        keep = [i for i in keep if i in vl]
    elif cons:
        want = CODE_NUM[cons]
        keep = [i for i in keep if i < len(v["codes"]) and v["codes"][i] == want]
        if cons == "BAD":                               # '불일치'에서 합의로 label 바꿈은 뺀다
            vl = voted_lines(date, dev)
            if vl:
                keep = [i for i in keep if i not in vl]
    if llm:                                             # 줄별 결과의 내용 지문(k)으로 그 줄의 가장 최근 재확인을 찾는다
        cand = (CODE_NUM["BAD"],) if llm == "todo" else (CODE_NUM["BAD"], CODE_NUM["RULE"])
        bad = [i for i in keep if i < len(v["codes"]) and v["codes"][i] in cand]
        cache, latest = frame_cache(date, dev), recheck_latest()
        heads = [ROW_KEY.match(x) for x in read_lines(cache, bad)] if cache and bad else [None] * len(bad)
        if llm.startswith("c_"):                        # 합의 (그 줄의 대상 = '불일치'·'정합 (룰 기준)')
            cm = {CODE_NUM["BAD"]: recheck_cons("bad"), CODE_NUM["RULE"]: recheck_cons("rule")}
            keep = [i for i, m in zip(bad, heads) if m and (cm[v["codes"][i]].get(m.group(1)) or ("",))[0] == llm[2:]]
        else:
            got = {i: latest.get(m.group(1)) if m else None for i, m in zip(bad, heads)}
            keep = [i for i in bad if (got[i] is None if llm == "todo" else
                                       got[i] is not None and llm in ("any", got[i].get("verdict")))]
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


def select_rows(dates, dev, label, rule, text, offset, limit, cons="", hold="", llm="", skipped=None):
    """여러 날짜를 이어 붙인 것처럼 페이지를 고른다 → (전체 수, [(날짜, 그 날짜의 줄 번호, 전체 index)]).
    skipped 를 주면 날짜가 여럿일 때 검증 결과를 쓸 수 없는 날짜(정합성·LLM 재확인 조건)는 빼고 그 까닭을 skipped 에 담는다."""
    parts = []
    for d in dates:
        try:
            m = matching(d, dev, label, rule, text, cons, hold, llm)
        except NoVerify as e:
            if skipped is None or len(dates) == 1:
                raise
            skipped.append(e.gap)
            continue
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
CONS = {"OK": "정합", "RULE": "정합 (룰 기준)", "CHECK": "확인 필요", "BAD": "불일치",
        "VOTED": "합의로 label 바꿈"}   # VOTED: 검증은 '불일치'지만 LLM 재확인 합의로 label 을 바꿔 반영한 줄 (voted_lines, 2026-11-04~)
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
        conds = C.with_anchors(rule)          # 기준 content(R 정규식·위치 있는 부정 content)가 없는 예전 입력이면 채운 사본
        res = [(c, C.check_condition(c, parts)) for c in conds]
        n = len(res)
        ok_n = sum(1 for _, r in res if r is True)
        failed = [c for c, r in res if r is False]
        unknown = [c for c, r in res if r is None]
        lost = _lost_reading(C, raw, http, conds) if label == "anomalies" and failed else ""
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
GROUP_DOC = [   # 화면 5 묶음 순서 = 표준모델 피처 → 신규 피처(표준이 못 잡는 것만) → 라벨 검증용 (2026-10-29)
    ("표준모델 피처", "<장비>_payload_", "표준모델(기준 26 상반기)이 쓰는 피처 34개. 이름·정규식은 실행할 때 표준모델 피처 파일에서 "
                                      "읽는다 (6. 비교와 같은 계산). 소문자로 바꾼 payload 에서 _comb = 정규식 중 하나라도 맞으면 1, "
                                      "_count = 맞은 횟수, _ratio = 정규식에 맞은 글자 비율(0~1). 한글이 10% 이상인 payload 는 모두 0. "
                                      "정상에서도 자주 켜지는 범용 피처(기준 정상의 10% 이상)는 4. 줄별 '탐지 피처'에서 뺀다."),
    ("신규 피처: 공격 흔적 보충 (기존 피처 결합 가능)", "add_",
     "표준 피처가 못 잡는 공격 흔적 중 같은 계열 표준 comb 피처가 있어 그 피처에 합칠 수 있는 것 (줄마다 '합쳐질 표준 피처'와 근거). "
     "합친 정의(결합판, 이름은 표준과 같음)는 로컬 파일 5_0d_combined_features.py 이고 5.4 피처 실험 E13 에서 시험한다 "
     "(6. 비교·'26 하반기' 선별에는 쓰지 않음). 지금 값(add_)은 표준에 없는 갈래는 URL 디코딩(2번)·소문자 payload 에서 모두 세고, "
     "인코딩되면 표준 피처가 못 보는 갈래는 디코딩해야만 보일 때만 센다 (디코딩 뒤 수 − 원문 수)."),
    ("신규 피처: 공격 흔적 보충 (신규 피처로 생성 필요)", "add_",
     "대응하는 표준 피처가 없거나 값의 뜻(개수)이 달라 신규 피처로 만들어야 하는 것: XXE, 회피 신호(이중 인코딩·%u 인코딩·제어문자 개수)."),
    ("신규 피처: 공격 흔적 보충", "add_", "표준 피처가 못 잡는 공격 흔적·회피 신호 (결합판 파일 5_0d_combined_features.py 가 없거나 "
                                         "4.1 과 맞지 않아 결합 가능 여부를 나누지 않음)."),
    ("라벨 검증용: 공격 흔적", "k_", "4. label 정합성 판정과 3.4 증강 normal 교차 검사가 쓰는 흔적 (모든 갈래, 표준 피처와 같은 갈래 포함). "
                                   "판정 사유의 '공격 흔적 …' 이 이것이다. ML 피처가 아니며, 판정 기준을 바꾸지 않으려고 그대로 둔다 "
                                   "(ML 피처는 위의 표준모델 피처 + 신규 피처)."),
    ("라벨 검증용: 공격 룰", "ra_", "신뢰할 수 있는 공격 룰(정보성 룰, 기존 normal 에도 걸리는 모호한 룰 제외)과 대조해 "
                                  "공격 유형별로 모든 조건이 일치한 룰 수. 조건 일부만 맞거나 확인 못 한 조건이 있으면 세지 않는다. "
                                  "4. label 정합성 판정에 쓰고 ML 피처로는 쓰지 않는다 (LLM 데이터의 라벨을 정한 룰과 같은 정보 = "
                                  "라벨 누수, 5.4 피처 실험에서도 뺌)."),
    ("라벨 검증용: 개별 룰", "r_", "특정 룰 하나의 탐지 여부와 그 룰의 정규식 조건별 일치 (4_0 --rules 로 지정). "
                                 "신규 학습 데이터 화면에서는 그 줄을 만든 룰로 계산한다. 분석·라벨 검증용 (ML 피처 아님)."),
]
STD_KIND = {"comb": ("정규식 중 하나라도 맞으면 1", "해당하면 1"), "count": ("정규식에 맞은 횟수", "맞은 횟수"),
            "ratio": ("정규식에 맞은 글자 수 ÷ payload 글자 수", "비율 0~1")}
ADD_DOC = {   # add_ 신규 피처: (표준 피처가 못 잡아 보충하는 것, 예). 갈래는 4.1 FEATURE_PARTS 그대로
    "sql": ("표준 피처에 없는 것: xp_cmdshell·sp_executesql, group_concat(·updatexml(·load_file(, INTO OUTFILE, 따옴표 뒤 # · /* 주석. "
            "인코딩돼 표준 피처가 못 보는 것: 따옴표 뒤 OR/AND 비교, OR 1=1, benchmark(, char(숫자), ; 뒤 DROP·INSERT 등",
            "group_concat( · xp_cmdshell · ' /* · %27%20or%201%3D1 (인코딩)"),
    "xss": ("표준 피처에 없는 것: <script>·</script>, vbscript:, onclick=, document.cookie·document.domain, String.fromCharCode. "
            "인코딩돼 못 보는 것: javascript:, on이벤트=, <img …on…=, <iframe, alert(·prompt(·confirm(",
            "<script>… · document.cookie · %3Ciframe%20src%3D… (인코딩)"),
    "cmd": ("표준 피처에 없는 것: ; | && 뒤의 id·ls·bash·sh·ncat·ifconfig, $(id)·$(ls)·$(sleep, cmd /c, busybox. "
            "인코딩돼 못 보는 것: /bin/sh, chmod +x, ; | && 뒤의 whoami·cat·wget·curl 등, $(whoami)·백틱 실행",
            ";id · |ls · $(sleep 5) · cmd /c dir · %3Bwhoami (인코딩)"),
    "traversal": ("인코딩돼 표준 피처가 못 보는 것만: 여러 번 인코딩한 ../, win.ini·boot.ini·system.ini, /proc/self/",
                  "%25252e%25252e%25252f (세 번 인코딩한 ../) · %2fproc%2fself%2f · %5cwin.ini"),
    "inclusion": ("표준 피처에 없는 것: zip:// · glob:// · php://(input·filter 외), data:…;base64, 파라미터 값의 원격 스크립트 URL. "
                  "인코딩돼 못 보는 것: file:/// · php://input · php://filter",
                  "zip://a.jpg%23x · data:text/plain;base64,… · ?page=http://x/s.txt? · php%3A%2F%2Ffilter (인코딩)"),
    "template": ("템플릿·EL 주입 전체 (표준 피처는 일부 모양만 봄): ${jndi:, ${env:…}, {{ }} 안의 코드, #{…( · %{…( · <%=",
                 "{{config}} · ${env:HOME} · %{(#…)}"),
    "xxe": ("XML 외부 엔티티 전체 (표준 피처에 없음): <!ENTITY 선언, DOCTYPE 안의 엔티티, SYSTEM \"file:…\"",
            "<!DOCTYPE x [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>"),
    "deserial": ("표준 피처에 없는 것: 자바 직렬화 시작 바이트(rO0AB · aced0005), __proto__·constructor[prototype], ysoserial, "
                 "\"$type\":, !!python/object. 인코딩돼 못 보는 것: PHP 객체 표기 O:숫자:\"클래스\":숫자:{",
                 "rO0ABXNy… · __proto__ · O%3A8%3A%22stdClass%22%3A1%3A%7B (인코딩)"),
    "shellshock": ("Bash Shellshock 공격 문자열 전체 (표준 피처는 일부만 봄): () { :; };", "User-Agent: () { :; }; /bin/bash -c …"),
    "sensitive": ("표준 피처에 없는 것: web.config. 인코딩돼 못 보는 것: /.ssh/", "GET /web.config · %2F.ssh%2Fid_rsa (인코딩)"),
}
R_ALL, R_RE = "r_<룰>_all", "r_<룰>_re<n>"


_cb = {}


def combined_ui():
    """5.0d 결합판 (로컬 전용 파일: add_ 의 결합 가능 갈래 → 표준 comb 피처). 없거나 4.1 과 맞지 않으면 (None, 사유)."""
    with _lock:
        if "M" not in _cb:
            try:
                if not (BASE / "5_0d_combined_features.py").exists():
                    raise FileNotFoundError("5_0d_combined_features.py 없음")
                M = _load("combined_ui", "5_0d_combined_features.py")
                M.check_parts()
                _cb.update(M=M, err="")
            except Exception as e:                      # 화면은 결합 여부를 나누지 않고 보여 줌
                _cb.update(M=None, err=f"{type(e).__name__}: {e}")
    return _cb["M"], _cb["err"]


def add_title(k):
    """add_ 신규 피처의 이름 (화면 4·5). 결합판이 있으면 (가) 표준 결합 가능 / (나) 신규 피처를 붙인다."""
    M = combined_ui()[0]
    base = "스캐너·공격 도구 UA (표준에 없는 도구" if k == "scanner_ua" else TRACE_NAMES.get(k, k) + " ("
    if M is None:
        return "스캐너·공격 도구 UA (표준에 없는 도구)" if k == "scanner_ua" else \
            TRACE_NAMES.get(k, k) + ("" if k in ("double_enc", "unicode_enc", "ctrl_bytes") else " (표준 보충)")
    return base + (", " if k == "scanner_ua" else "") + ("표준 결합 가능)" if f"add_{k}" in M.COMBINABLE else "신규 피처)")


def feature_key(name):
    """개별 룰 피처(r_<룰>_…)는 룰마다 이름이 달라 설명·집계는 한 줄로 묶는다."""
    if name.startswith("r_"):
        return R_ALL if name.endswith("_all") else R_RE
    return name


def feature_catalog(dev="ips"):
    """화면 5 피처 목록: 표준모델 피처 → 신규 피처 add_*(표준이 못 잡는 것만) → 라벨 검증용(k_·ra_·r_).
    표준 피처 이름·정규식은 표준모델 피처 파일에서, 신규·흔적 패턴은 4.1 코드에서 그대로 가져온다 (설명과 실제 계산이 어긋나지 않게).
    표준 피처 설명·기준 비율·대응(어느 표준 피처와 같은 내용인지)은 로컬 대응표 (overlap_notes, 표준 피처 이름은 코드에 적지 않음)."""
    fe = feature_engine()
    F = fe["F"]
    _, rules, _ = checker()
    items = []
    notes = F.overlap_notes()
    fam, generic, rates = notes["families"], set(notes["generic"].get(dev, [])), notes["base_rates"].get(dev, {})
    std = F.std_features(dev)
    full = {F.std_short(n, dev): n for n, _, _ in std}

    def names(v):           # 대응표의 짧은 이름 (a·b) → 이 장비의 표준 피처 이름
        return " · ".join(full.get(s, s) for s in v.split("·")) if v else "표준 피처"
    CB, cb_err = combined_ui()          # 결합판: add_ 갈래 → 합칠 표준 comb 피처 (로컬 전용 파일, 표준 피처 이름은 코드에 적지 않음)
    merged = {}                         # 표준 피처(짧은 이름) → 더하는 add_ 갈래 수
    for t, a, *_ in (CB.MERGE if CB else []):
        merged.setdefault(t, {}).setdefault(a, 0)
        merged[t][a] += 1
    for n, kind, rxs in std:
        s = F.std_short(n, dev)
        sup = [a for a in F.FEATURE_NAMES if s in fam.get(a, [])]
        r = rates.get(s)
        ml = [f"기준(26 상반기) 공격의 {r[0]}% · 정상의 {r[1]}% 에서 켜짐" if r else "",
              "범용 (정상에서도 자주 켜짐) → 4. 줄별 '탐지 피처'에서 뺌" if s in generic else ""]
        if s in merged:
            ml.append("결합판(5.0d)에 더하는 갈래: " + ", ".join(f"{a} {c}개" for a, c in merged[s].items()))
        elif sup and CB:
            ca, na = [a for a in sup if a in CB.COMBINABLE], [a for a in sup if a not in CB.COMBINABLE]
            if ca:
                ml.append(f"같은 계열 신규 피처 {', '.join(ca)} 는 " + ("비율·개수 피처라 여기에 합치지 않음 (같은 계열 comb 피처에 합침)"
                                                                   if kind != "comb" else "다른 comb 피처에 합침"))
            if na:
                ml.append(f"같은 계열 {', '.join(na)} 는 신규 피처로 둠")
        elif sup:
            ml.append(f"못 잡는 부분 보충: {', '.join(sup)}")
        items.append({"name": n, "std": s, "group": "표준모델 피처", "title": notes["titles"].get(s, s),
                      "what": f"표준모델 피처 파일의 정규식 {len(rxs)}개 - {STD_KIND[kind][0]}", "ml": " · ".join(x for x in ml if x),
                      "example": "", "value": STD_KIND[kind][1], "pattern": "\n".join(rxs), "src": "view"})
    adds = []

    def add_group(name):                # (가) 결합 가능 / (나) 신규 생성 필요 / 결합판이 없으면 나누지 않음
        if CB is None:
            return "신규 피처: 공격 흔적 보충"
        return "신규 피처: 공격 흔적 보충 (" + ("기존 피처 결합 가능" if name in CB.COMBINABLE else "신규 피처로 생성 필요") + ")"

    def merge_info(name):               # (가): 갈래 → 합쳐질 표준 피처와 근거 / (나): 신규로 두는 이유
        if CB is None:
            return [], ""
        m = [{"pattern": p, "how": h, "std": full.get(t, t), "why": why} for t, a, p, h, why in CB.MERGE if a == name]
        if m:
            tg = list(dict.fromkeys(x["std"] for x in m))
            return m, "합쳐질 표준 피처: " + ", ".join(f"{x} (갈래 {sum(y['std'] == x for y in m)})" for x in tg)
        return [], "신규로 두는 이유: " + CB.NEW_ONLY.get(name, "")
    for k, parts in F.FEATURE_PARTS.items():
        what, ex = ADD_DOC.get(k, ("", ""))
        dropped = F.FEATURE_DROPPED.get(k, [])
        same = [full[s] for s in fam.get(f"add_{k}", []) if s in full]
        mg, how = merge_info(f"add_{k}")
        to = {x["pattern"]: x["std"] for x in mg}
        ml = ((how + " · " if how else "")
              + f"표준에 없는 갈래 {sum(h == 'new' for _, h in parts)}개 · 인코딩됐을 때만 세는 갈래 {sum(h == 'enc' for _, h in parts)}개"
              + (f" · 표준과 같아 뺀 갈래 {len(dropped)}개" if dropped else "")
              + ("" if mg else f" · 같은 계열 표준 피처: {', '.join(same)}" if same else " · 같은 계열 표준 피처 없음"))
        adds.append({"name": f"add_{k}", "group": add_group(f"add_{k}"), "title": add_title(k), "what": what, "ml": ml,
                     "example": ex, "value": "나온 횟수", "src": "view", "merge": mg,
                     "pattern": "\n".join(("[인코딩됐을 때만] " if h == "enc" else "") + p + (f"  → {to[p]}" if p in to else "")
                                          for p, h in parts),
                     "dropped": [{"pattern": p, "std": names(notes["parts"].get(p))} for p in dropped]})
    evasion_pat = {"double_enc": r"%25(?:2e|2f|5c|3c|3e|27|22|00|3b|7c)", "unicode_enc": r"%u[0-9a-f]{4}",
                   "ctrl_bytes": r"글자 코드 < 32 (\n \r \t 제외) + 글자로 적힌 \x00~\x1f + %00"}
    for k in F.EVASION:
        what, ex = TRACE_DOC.get(k, (F.EVASION[k], ""))
        ua = k == "scanner_ua"
        same = [full[s] for s in fam.get(f"add_{k}", []) if s in full]
        mg, how = merge_info(f"add_{k}")
        if ua:
            what = f"User-Agent 가 표준 피처에 없는 스캐너·공격 도구: {', '.join(F.UA_SCANNER_FEATURE.pattern.split('|')[:8])} 등"
            ex = "User-Agent: Acunetix … · OpenVAS …"
        adds.append({"name": f"add_{k}", "group": add_group(f"add_{k}"), "title": add_title(k), "what": what,
                     "ml": (how + " · " if how else "") + (f"표준 피처에 있는 도구 {len(F.UA_DROPPED)}개 뺌" if ua else "")
                           + ("" if mg or ua else f"같은 계열 표준 피처: {', '.join(same)}" if same else "같은 계열 표준 피처 없음"),
                     "example": ex, "value": "해당하면 1" if ua else "나온 횟수", "src": "view", "merge": mg,
                     "pattern": F.UA_SCANNER_FEATURE.pattern + (f"  → {mg[0]['std']}" if mg else "") if ua else evasion_pat.get(k, ""),
                     "dropped": [{"pattern": t, "std": names(notes["ua"].get(t))} for t in F.UA_DROPPED] if ua else []})
    order = [g[0] for g in GROUP_DOC]
    items += sorted(adds, key=lambda i: order.index(i["group"]))     # 묶음 순서대로 (같은 묶음 안은 원래 순서)
    for k, rx in F.KEYWORDS.items():
        what, ex = TRACE_DOC.get(k, ("", ""))
        items.append({"name": f"k_{k}", "group": "라벨 검증용: 공격 흔적", "title": TRACE_NAMES.get(k, k), "what": what,
                      "example": ex, "value": "나온 횟수", "pattern": rx, "src": "verify"})
    for k in F.EVASION:
        what, ex = TRACE_DOC.get(k, (F.EVASION[k], ""))
        items.append({"name": f"k_{k}", "group": "라벨 검증용: 공격 흔적", "title": TRACE_NAMES.get(k, k), "what": what,
                      "example": ex, "value": "해당하면 1" if k == "scanner_ua" else "나온 횟수", "src": "verify",
                      "pattern": F.UA_SCANNER.pattern if k == "scanner_ua" else evasion_pat.get(k, "")})
    by_type = {}
    for rid, t in fe["types"].items():
        by_type.setdefault(t, []).append(rid)
    for t in F.ATTACK_TYPES:
        ids = sorted(by_type.get(t, []))
        items.append({"name": f"ra_{t}", "group": "라벨 검증용: 공격 룰", "title": f"{ATTACK_NAMES.get(t, t)} 공격 룰 일치 수",
                      "what": f"신뢰할 수 있는 {ATTACK_NAMES.get(t, t)} 공격 룰 {len(ids):,}개 중 이 payload 에서 "
                              "모든 조건이 일치한 룰 수", "rules": len(ids),
                      "example": " · ".join(f"{i} {rules[i]['rule_name']}" for i in ids[:3] if i in rules),
                      "value": "일치한 룰 수", "pattern": "", "src": "verify"})
    items.append({"name": "ra_total", "group": "라벨 검증용: 공격 룰", "title": "공격 룰 일치 수 합계",
                  "what": f"위 공격 유형 {len(F.ATTACK_TYPES)}개의 합 (신뢰할 수 있는 공격 룰 {len(fe['types']):,}개 전체)",
                  "rules": len(fe["types"]), "example": "", "value": "일치한 룰 수", "pattern": "", "src": "verify"})
    items.append({"name": R_ALL, "group": "라벨 검증용: 개별 룰", "title": "이 룰 탐지",
                  "what": "그 룰의 모든 조건(content·정규식·위치·순서)을 만족하면 1. 학습데이터 라벨과 같은 기준이라 "
                          "평가할 때는 룰 단위로 나눠야 한다 (라벨 누수)",
                  "example": "r_et_1_2053468_all", "value": "해당하면 1", "pattern": "", "src": "verify"})
    items.append({"name": R_RE, "group": "라벨 검증용: 개별 룰", "title": "이 룰의 정규식 조건 일치",
                  "what": "그 룰의 정규식(pcre) 조건 하나가 일치하면 1 (n = 룰 안의 조건 번호). 룰 전체가 아니라 "
                          "핵심 패턴만 맞아도 켜진다",
                  "example": "r_et_1_2053468_re2 (et:1:2053468 의 정규식 UNION.+SELECT)", "value": "해당하면 1",
                  "pattern": "", "src": "verify"})
    return {"items": items, "groups": [{"name": n, "prefix": p, "desc": d} for n, p, d in GROUP_DOC if any(i["group"] == n for i in items)],
            "attack_rules": len(fe["types"]), "note": fe["note"], "dev": dev, "notes_found": bool(notes["titles"]),
            "combined": {"bundle": f"{CB.BUNDLE} {CB.VERSION}", "targets": len(CB.TARGETS), "branches": len(CB.MERGE),
                         "combinable": len(CB.COMBINABLE), "new_only": len(CB.NEW_ONLY)} if CB else {"error": cb_err}}


def feature_counts(dev):
    """피처별로 켜진 줄 수 (anomalies / normal).
    표준·신규 피처(src view): 표시 캐시의 켜진 줄 수 (4_1 --view = 5. '켜진 줄 수 다시 계산', 전체 검증을 다시 돌리지 않음).
    라벨 검증용(src verify): 전체 검증 결과. 검증하지 않은 날짜는 따로 알려 준다."""
    out = {"labels": {"anomalies": 0, "normal": 0}, "counts": {}, "dates": [], "missing": [], "running": False}
    dates = [x["date"] for x in dataset_dates()]
    for d in dates:
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
    st = job_state("features")
    view = {"running": st["running"], "progress": {k: st["lock"].get(k) for k in ("dev", "step", "done", "total")},
            "dates": [], "labels": {}}
    p = feature_engine()["F"].VIEW_DIR / f"{dev}_counts.json"
    try:
        vc, t = json.loads(p.read_text(encoding="utf-8")), p.stat().st_mtime
    except (OSError, ValueError):
        vc = None
    if vc:
        now = {}                        # 지금 로그 (계산한 뒤 날짜가 늘었거나 로그가 다시 만들어졌으면 stale)
        for d in dates:
            g = sorted((DATASET_DIR / d).glob(f"{dev}_*_snort.log"))
            if g:
                now[d] = g[0]
        out["counts"].update(vc.get("counts", {}))
        view.update(dates=vc.get("dates", []), labels=vc.get("labels", {}), lines=vc.get("lines", 0),
                    computed_at=vc.get("computed_at", ""),
                    stale=set(now) != set(vc.get("dates", [])) or any(g.stat().st_mtime > t for g in now.values()))
    out["view"] = view
    return out


_viewc = {}      # 장비 → (파일 시각, 표준·신규 피처 표시 캐시)


def view_cache_ui(dev):
    """표시 캐시(4_1 --view: payload 지문 → 표준 34·add_* 값)를 파일이 바뀔 때만 다시 읽는다. 없으면 None."""
    F = feature_engine()["F"]
    p = F.VIEW_DIR / f"{dev}.parquet"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _viewc.get(dev)
    if not hit or hit[0] != mt:
        hit = _viewc[dev] = (mt, F.view_cache(dev))
    return hit[1]


def ml_features(payloads, dev, cache=None):
    """4. 줄별 '탐지 피처' (ML 피처): 켜진 표준모델 피처(범용 제외) + 켜진 신규 피처 add_* → [[{name, value, desc}]].
    표시만 바꾼다 (label 정합성 판정은 그대로 흔적 k_·공격 룰로). 표시 캐시에 없는 payload 는 바로 계산한다 (100줄 약 0.3초)."""
    F = feature_engine()["F"]
    notes = F.overlap_notes()
    hide = {f"{dev}_payload_{s}" for s in notes["generic"].get(dev, [])}
    vals = F.view_features(payloads, dev, view_cache_ui(dev) if cache is None else cache, hide)
    return [[{"name": c, "value": int(v) if v == int(v) else round(v, 3),
              "desc": add_title(c[4:]) if c.startswith("add_") else notes["titles"].get(F.std_short(c, dev), "")}
             for c, v in r] for r in vals]


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
# 3 = R(상대) 정규식을 바로 앞 content 뒤에서 검사 (3_0a_common.check_condition, 2026-10-30). 판 3 전 결과는 '불일치'
#     말고도 R 정규식 룰의 줄을 다시 확인한다 (정합이던 normal 이 룰에 걸리거나, 불일치이던 공격이 맞을 수 있음)
# 4 = 코드 검사 고침 (3_0a_common CHECK_FIXES, 2026-10-30 오후): 요청 방향 file_data → 요청 본문, base64_data 디코딩,
#     헤더 앞뒤 줄바꿈, 소문자 메서드, URI //, 글자 \xNN·%XX(80 이상) → 바이트. 판 4 전 결과는 '불일치' 말고도
#     이 고침이 닿는 줄(_v4_touch)을 다시 확인한다
# 5 = 코드 검사 고침 2 (3_0a_common NEG_WINDOW·REGEX_V1, 2026-11-03): 위치(within 등)가 있는 부정 content 는 기준 content 뒤 창에서만,
#     파이썬 re 가 못 읽는 정규식((?-i)·\h)은 regex 모듈 V1 로. 판 5 전 결과는 '불일치' 말고도 이 고침이 닿는 룰의 줄(_v5_touch)을
#     다시 확인한다. analyze 는 C.with_anchors(rule) 로 기준 content 를 채운 조건을 본다
# 6 = 코드 검사 고침 3 (3_0a_common NEG_PAYLOAD_ANCHOR·BYTE_WINDOW·UTF8_BYTES, 2026-11-04): payload 의 부정 content 창은 다른 버퍼의
#     기준 content 뒤에서도, 창 길이는 UTF-8 바이트로, 0x80 이상 글자는 UTF-8 바이트 변형으로도 본다. 판 6 전 결과는 '불일치' 말고도
#     이 고침이 닿는 줄(_v6_touch: 위치 있는 부정 content·0x80 이상 바이트 content 룰, 또는 0x80 이상 글자가 있는 payload)을 다시 확인한다
# 7 = 코드 검사 고침 4 (3_0a_common B64_ANCHOR·REGEX_POSIX·PCRE_COND·URI_REL_REGEX, 2026-11-04 오후): base64_data 를 디코딩 시작점
#     뒤 덩어리로, POSIX 묶음 정규식은 V1, PCRE 조건문 바꿔 읽기, http_uri 뒤 R 정규식은 URI 에서. 판 7 전 결과는 '불일치' 말고도
#     이 고침이 닿는 룰의 줄(_v7_touch, C.fix4_touch)을 다시 확인한다
VERIFY_LOGIC = 7


def _has_r(rule):
    """R(상대) 정규식이 있는 룰인지 (판 3 전 결과를 다시 확인할 줄)."""
    return bool(rule) and any(c.get("type") == "regex" and "R" in c.get("flags", "") for c in rule.get("conditions", []))


_V4_PAY = re.compile(r"\\x[0-9a-fA-F]{2}|%[89a-fA-F][0-9a-fA-F]|^[a-z]")


def _v4_touch(rule, payload):
    """판 4 의 코드 검사 고침이 결과를 바꿀 수 있는 줄: file_data·base64_data·헤더 조건이 있는 룰,
    또는 payload 에 글자 \\xNN·%80 이상·소문자 메서드·URI 의 // 가 있는 줄."""
    if rule and ("file_data" in rule.get("rule", "")
                 or any(c.get("in") in ("base64_data", "header", "raw_header") for c in rule.get("conditions", []))):
        return True
    if _V4_PAY.search(payload or ""):
        return True
    head = (payload or "").split(" ", 2)
    return len(head) > 1 and "//" in head[1]


def _v5_touch(C, rule):
    """판 5 의 코드 검사 고침 2 가 결과를 바꿀 수 있는 룰: 위치가 있는 부정 content, 또는 re 로 못 읽는 정규식 (C.fix2_touch)."""
    f = getattr(C, "fix2_touch", None)
    return bool(rule) and f is not None and f(rule)


def _v6_touch(C, rule, payload):
    """판 6 의 코드 검사 고침 3 이 결과를 바꿀 수 있는 줄: 위치가 있는 부정 content·0x80 이상 바이트 content 가 있는 룰
    (C.fix3_touch), 또는 payload 에 0x80 이상 글자가 있는 줄 (UTF-8 바이트 변형은 정규식 결과도 바꿀 수 있음)."""
    f = getattr(C, "fix3_touch", None)
    return bool(rule) and f is not None and (f(rule) or not (payload or "").isascii())


def _v7_touch(C, rule):
    """판 7 의 코드 검사 고침 4 가 결과를 바꿀 수 있는 룰: base64_data 조건, POSIX 묶음·PCRE 조건문 정규식, http_uri 뒤 R 정규식
    (C.fix4_touch)."""
    f = getattr(C, "fix4_touch", None)
    return bool(rule) and f is not None and f(rule)
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
    """예전 판정 규칙(VERIFY_LOGIC 전)으로 끝난 검증의 '불일치' 줄 수 - '검증 안 한 데이터만' 이 다시 확인한다.
    판 7 전 검증은 R 정규식 룰·코드 검사 고침(1~4)이 닿는 줄도 다시 확인하므로 '불일치' 가 0 이어도 1 이상."""
    if v.get("logic", 1) >= VERIFY_LOGIC:
        return 0
    bad = (v.get("summary") or {}).get("BAD", 0)
    return max(bad, 1) if v.get("logic", 1) < 7 else bad


def _all_verified(date, dev):
    """새로 확인할 줄이 없음: 지금 로그·룰 입력으로 끝난 검증이 있고 줄별 결과에 내용 지문이 있음 (예전 판정 규칙의 '불일치' 없음)."""
    p = frame_cache(date, dev)
    if p is None or _stale_bad(verify_state(date, dev)):
        return False
    with open(p, "rb") as f:
        return bool(ROW_HEAD.match(f.readline()))


# ── 낡은 검증 (2026-11-02~): 검증한 뒤 생성·다시 만들기·학습 로그에 반영·수집의 1.4 가 학습 로그를 다시 만들면 그 검증은 쓸 수 없다.
# 정합성·LLM 재확인 조건에서 '전체' 보기는 그 날짜를 빼고 알리고(날짜 하나면 까닭·할 일을 담은 오류), 화면 4 위쪽에 날짜·장비·
# 로그를 다시 만든 시각(그때 돌던 화면 작업)과 '바뀐 날짜만 다시 검증'(= 검증 안 한 데이터만 검증)을 보인다.
REBUILD_JOBS = {"generation": "생성 실행", "redo": "다시 만들기", "dataset": "학습 로그에 반영", "collect": "수집",
                "agent": "수집 (에이전트)"}
_agent_win = {}      # 에이전트 출력 → (수정 시각, 1.4 구간)
_rewatch = {"on": False, "pairs": [], "jobs": [], "armed_at": "", "busy_at": 0.0, "fired": None, "thread": None}


def _hm(x):
    """시각 (ISO 문자열 또는 epoch 초) → 오늘이면 'HH:MM', 아니면 'MM-DD HH:MM'."""
    try:
        t = time.localtime(x) if isinstance(x, (int, float)) else time.strptime(str(x)[:19], "%Y-%m-%dT%H:%M:%S")
    except (ValueError, OverflowError, OSError):
        return str(x)[11:16]
    return time.strftime("%H:%M" if time.strftime("%Y-%m-%d", t) == time.strftime("%Y-%m-%d") else "%m-%d %H:%M", t)


def _agent_14(p):
    """수집 에이전트 출력의 1.4 구간 [(시작, 끝)] (에이전트는 하루 종일 떠 있어 출력 파일 시각으로는 알 수 없음)."""
    mt = p.stat().st_mtime
    hit = _agent_win.get(p)
    if hit and hit[0] == mt:
        return hit[1]
    wins, start = [], None
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[1.4]" in line and ("▶" in line or "■" in line):
                try:
                    t = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if "▶" in line:
                    start = t
                elif start is not None:
                    wins.append((start, t))
                    start = None
    if start is not None:
        wins.append((start, mt))
    _agent_win[p] = (mt, wins)
    return wins


def _rebuilt_by(ts):
    """학습 로그가 ts 에 다시 만들어질 때 돌던 화면 작업 이름 (.run/<작업>-<시작 시각>.out 의 시작 ~ 마지막 출력,
    에이전트는 출력의 1.4 구간, 여럿이면 가장 늦게 시작한 것). 명령줄로 돌렸거나 출력이 지워졌으면 ''."""
    best = (0.0, "")
    for job, name in REBUILD_JOBS.items():
        for p in RT.RUN_DIR.glob(f"{job}-*.out"):
            try:
                start = time.mktime(time.strptime(p.stem[len(job) + 1:], "%Y%m%d-%H%M%S"))
                wins = _agent_14(p) if job == "agent" else [(start, p.stat().st_mtime)]
            except (ValueError, OSError):
                continue
            for a, b in wins:
                if a - 5 <= ts <= b + 120 and a > best[0]:
                    best = (a, name)
    return best[1]


def verify_gap(date, dev):
    """그 날짜·장비의 검증 결과를 쓸 수 없는 까닭 (쓸 수 있으면 None). why: none = 검증 안 함 (멈춘 검증이면 stopped),
    stale = 검증한 뒤 학습 로그가 다시 만들어짐 (검증 시각·로그 시각·그때 돌던 작업), running = 검증 중, error = 검증 실패."""
    v = verify_state(date, dev)
    st = v.get("status")
    if st == "done":
        return None
    g = {"date": date, "dev": dev, "why": st if st in ("running", "error") else "none"}
    if st == "error":
        g["error"] = str(v.get("error") or "")[:200]
    elif st != "running":
        saved = _saved_verify(date, dev)
        if saved.get("finished_at"):
            mt = _log_path(date, dev).stat().st_mtime
            g.update(why="stale", verified_at=saved["finished_at"], log_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt)),
                     by=_rebuilt_by(mt))
            vh, lh = _hm(g["verified_at"]), _hm(mt)
            if vh == lh:                                # 같은 분이면 초까지
                vh, lh = vh + g["verified_at"][16:19], lh + g["log_at"][16:19]
            g.update(verified_hm=vh, log_hm=lh)
        if (date, dev) in _stopped:
            g["stopped"] = _stopped[(date, dev)]
    return g


def gap_text(g, dev_name=True):
    """verify_gap → 까닭과 할 일 (FW·웹 보기는 장비 이름 없이)."""
    who = f"{g['date']} {g['dev'].upper()}" if dev_name else g["date"]
    if g["why"] == "running":
        return f"{who} 는 지금 검증 중입니다 - 끝나면 정합성·LLM 재확인 조건으로 볼 수 있습니다"
    if g["why"] == "error":
        return f"{who} 검증이 실패했습니다 ({g.get('error', '')}) - 다시 검증하세요"
    if g["why"] == "stale":
        by = f" ('{g['by']}')" if g.get("by") else ""
        when = (f"검증({g['verified_hm']}) 뒤 {g['log_hm']} 에 학습 로그가 다시 만들어졌습니다{by}" if g["log_at"] > g["verified_at"]
                else f"검증({g['verified_hm']})한 학습 로그와 지금 로그({g['log_hm']})가 다릅니다{by}")
        return f"{who} 검증 결과가 낡았습니다 - {when}. '검증 안 한 데이터만 검증'을 누르면 바뀐 줄만 다시 봅니다"
    s = g.get("stopped")
    if s:
        return f"{who} 검증이 {_hm(s.get('at', ''))} 에 멈췄습니다 ({s.get('why', '')}) - '검증 안 한 데이터만 검증'으로 이어서 하세요"
    return f"{who} 는 아직 검증하지 않았습니다 - '검증 안 한 데이터만 검증'(또는 '전체 검증')을 먼저 실행하세요"


class NoVerify(ValueError):
    """정합성·LLM 재확인 조건인데 그 날짜·장비의 검증 결과를 쓸 수 없음 ('전체' 보기는 그 날짜를 빼고 알림)."""

    def __init__(self, gap):
        self.gap = gap
        super().__init__(gap_text(gap))


def verify_stale(devs=("ips", "waf")):
    """검증한 뒤 학습 로그가 다시 만들어진 날짜·장비 [verify_gap] (화면 4 위쪽 안내·'바뀐 날짜만 다시 검증')."""
    out = []
    for d in (x["date"] for x in dataset_dates()):
        for dev in devs:
            try:
                g = verify_gap(d, dev)
            except (OSError, ValueError):
                continue
            if g and g["why"] == "stale":
                out.append(g)
    return out


def _rebuild_busy():
    """지금 도는, 끝날 때 학습 로그를 다시 만드는 작업 이름 (생성·다시 만들기는 같은 generation 잠금 - 화면에서 시작한 다시
    만들기면 그 이름). 수집 에이전트는 하루 종일 떠 있으므로 그 안의 수집(collect 잠금)만 본다."""
    out = []
    if job_state("generation")["running"] or job_state("redo")["running"]:
        rec = _ui_record("redo")
        out.append(REBUILD_JOBS["redo" if rec.get("pid") and RT.pid_alive(rec["pid"]) else "generation"])
    return out + [REBUILD_JOBS[j] for j in ("dataset", "collect") if job_state(j)["running"]]


def _watch_pairs(devs, extra=()):
    """작업이 끝난 뒤 다시 볼 날짜·장비: 지금 검증했거나 검증 중인 것 + extra (처음부터 검증 안 한 날짜는 빼고)."""
    out = {tuple(p) for p in extra}
    for d in (x["date"] for x in dataset_dates()):
        for dev in devs:
            try:
                if _verify.get((d, dev), {}).get("status") == "running" or verify_gap(d, dev) is None:
                    out.add((d, dev))
            except (OSError, ValueError):
                continue
    return out


def rewatch_state():
    w = _rewatch
    return {"on": w["on"], "jobs": w["jobs"], "n": len(w["pairs"]), "armed_at": w["armed_at"], "fired": w["fired"]}


def arm_rewatch(pairs):
    """작업이 끝나면 학습 로그가 다시 만들어져 검증이 낡은 날짜만 '검증 안 한 데이터만 검증' (이 화면 서버가 켜져 있는 동안)."""
    with _lock:
        old = {tuple(p) for p in _rewatch["pairs"]} if _rewatch["on"] else set()
        _rewatch.update(on=True, pairs=sorted(old | set(pairs)), jobs=_rebuild_busy(), armed_at=RT.now_iso(),
                        busy_at=time.time(), fired=None)
        if _rewatch["thread"] is None:
            _rewatch["thread"] = threading.Thread(target=_rewatch_loop, daemon=True)
            _rewatch["thread"].start()
    return rewatch_state()


def _rewatch_loop():
    w = _rewatch
    while True:
        time.sleep(10)
        if not w["on"]:
            continue
        try:
            if _rebuild_busy():
                w["busy_at"] = time.time()
                continue
            if time.time() - w["busy_at"] < 30:            # 생성 → 다시 만들기처럼 이어지는 작업 사이
                continue
            started = []
            for d, dev in w["pairs"]:
                try:
                    if verify_gap(d, dev) and _verify.get((d, dev), {}).get("status") != "running" \
                            and start_verify(d, dev, "new") == "started":
                        started.append([d, dev])
                except (OSError, ValueError):
                    continue
            w.update(on=False, fired={"at": RT.now_iso(), "started": started})
        except Exception as e:                              # 지켜보기만 멈추고 화면은 그대로
            w.update(on=False, fired={"at": RT.now_iso(), "started": [], "error": f"{type(e).__name__}: {e}"})


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
                    # 판정 규칙이 바뀐 뒤 '불일치' 는 다시 확인 (판 3 전이면 R 정규식 룰의 줄도, 판 4 전이면 코드 검사 고침이 닿는 줄도)
                    if a.get("lv", 1) < VERIFY_LOGIC and (a.get("cons") == "BAD"
                                                          or (a.get("lv", 1) < 3 and _has_r(ctx[1].get(meta.get("id"))))
                                                          or (a.get("lv", 1) < 4 and _v4_touch(ctx[1].get(meta.get("id")), payload))
                                                          or (a.get("lv", 1) < 5 and _v5_touch(ctx[0], ctx[1].get(meta.get("id"))))
                                                          or (a.get("lv", 1) < 6 and _v6_touch(ctx[0], ctx[1].get(meta.get("id")), payload))
                                                          or (a.get("lv", 1) < 7 and _v7_touch(ctx[0], ctx[1].get(meta.get("id"))))):
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


# 합의로 label 바꿈 (2026-11-04~): LLM 재확인(투표 합의·사람 판단)으로 label 을 바꿔 학습 로그에 반영한 줄은 룰 기준 label 과
# 달라져 코드 검증이 '불일치'로 보지만 의도한 것이라 '불일치'와 따로 센다 (VOTED - 요약·필터·표·CSV, '불일치 모두 삭제'에서 뺌).
# edits.json 의 'LLM 재확인' label 수정(RECHECK_WHY) + label 파일의 edited 에 label (1.4 가 반영) + 지금 label 이 바꾼 label.
_voted = {}


def _edits_mtime(date):
    try:
        return _edits_file(date).stat().st_mtime
    except OSError:
        return 0


def voted_lines(date, dev):
    """그 날짜·장비에서 검증이 '불일치'인데 LLM 재확인으로 바꾼 label 이 반영된 줄 번호 집합 (검증·수정이 그대로면 지난 결과)."""
    v = verify_state(date, dev)
    if v.get("status") != "done":
        return set()
    key = (v.get("mtime"), v.get("size"), _edits_mtime(date))
    hit = _voted.get((date, dev))
    if hit and hit[0] == key:
        return hit[1]
    flips = {sk: e.get("label") for sk, e in read_edits(date).items()
             if e.get("action") == "edit" and e.get("label") and _rc_target(e)}
    out = set()
    bad = [i for i, c in enumerate(v.get("codes", "")) if c == CODE_NUM["BAD"]] if flips else []
    li = label_info(_log_path(date, dev)) if bad else None
    if li:
        for i, m in zip(bad, read_label_lines(li[4], bad)):
            if flips.get(m.get("sk")) == m.get("label") and "label" in str(m.get("edited") or ""):
                out.add(i)
    _voted[(date, dev)] = (key, out)
    return out


def _voted_row(a):
    """줄 결과(analyze 모양)를 '합의로 label 바꿈'으로 (코드 판정은 사유에 남김)."""
    return dict(a, cons="VOTED", cons_text=CONS["VOTED"],
                cons_reason=f"LLM 재확인 합의로 label 을 바꿔 반영함 - 코드 검증은 {a.get('cons_text', '')}: {a.get('cons_reason', '')}")


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


def verify_summary(date, dev, conv=False):
    """검증 요약 (날짜 하나 또는 전체). 2026-11-02~: 검증 결과를 쓸 수 없는 날짜(gaps: 검증 안 함·낡음·실패)와 까닭,
    검증한 날짜 수, 화면 4 위쪽 '검증이 낡은 날짜'(stale, 모든 날짜·장비 - FW·웹 보기 conv 는 원본만·장비 이름 없이),
    지금 도는 학습 로그를 다시 만드는 작업(busy)·끝나면 다시 검증(watch)."""
    dates = [d["date"] for d in dataset_dates()] if date == "all" else [date]
    latest = (input_dates() or [""])[-1]
    out = {"status": "done", "done": 0, "total": 0, "summary": {}, "names": CONS, "finished_at": "", "reused": 0,
           "todo": 0, "analyzed": 0, "eta_s": None, "est_new": 0, "est_new_s": 0.0, "est_all_s": 0.0,
           "rule_inputs": {}, "latest_input": latest, "gaps": [], "done_dates": 0, "n_dates": len(dates)}
    run_etas, wait_etas, phases, done_dates = [], [], set(), []
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
            done_dates.append(d)
            out["reused"] += v.get("reused", 0)
            for k, c in v.get("summary", {}).items():
                out["summary"][k] = out["summary"].get(k, 0) + c
            vt = len(voted_lines(d, dev))               # '불일치' 중 합의로 label 바꿈은 따로
            if vt:
                out["summary"]["BAD"] = out["summary"].get("BAD", 0) - vt
                out["summary"]["VOTED"] = out["summary"].get("VOTED", 0) + vt
            out["finished_at"] = max(out["finished_at"], v.get("finished_at", ""))
            ri = v.get("rule_inputs") or ({v["frame_in_dir"]: n} if v.get("frame_in_dir") else {})
            for k, c in ri.items():                     # 줄마다 확인한 룰 입력 날짜
                out["rule_inputs"][k] = out["rule_inputs"].get(k, 0) + c
        elif out["status"] != "running":
            out["status"] = "error" if st == "error" else "none"
            out["error"] = v.get("error")
        if st != "done":
            g = verify_gap(d, dev)
            out["gaps"].append(dict(g, text=gap_text(g, not conv)))
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
    # 저장된 줄별 결과가 모두 있으면 데이터 프레임 내려받기가 바로 된다 (_done = 검증한 날짜만 - 정합성 조건이면 그 날짜만 받음)
    out["frame_ready"] = out["status"] == "done" and all(frame_cache(d, dev) is not None for d in dates)
    out["frame_ready_done"] = bool(done_dates) and all(frame_cache(d, dev) is not None for d in done_dates)
    out["done_dates"] = len(done_dates)
    out["stale"] = [dict(g, text=gap_text(g, not conv)) for g in verify_stale(("ips",) if conv else ("ips", "waf"))]
    out["busy"], out["watch"] = _rebuild_busy(), rewatch_state()
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
    """검증에서 '불일치' 인 줄을 모두 삭제로 저장 (사유 = 불일치 사유). 합의로 label 바꿈(voted_lines)은 뺀다."""
    n = 0
    for d in ([x["date"] for x in dataset_dates()] if date == "all" else [date]):
        v = verify_state(d, dev)
        vl = voted_lines(d, dev)
        idx = [i for i, c in enumerate(v.get("codes", "")) if c == CODE_NUM["BAD"] and i not in vl] if v.get("status") == "done" else []
        if not idx:
            continue
        path = _log_path(d, dev)
        metas = read_label_lines(label_info(path)[4], idx)
        cache = frame_cache(d, dev)
        why = [r.get("cons_reason", "") for r in cached_rows(cache, idx)] if cache else [""] * len(idx)
        with _edit_lock:
            edits = read_edits(d)
            for meta, w in zip(metas, why):
                e0 = edits.get(meta.get("sk")) or {}
                if e0.get("action") == "edit" and e0.get("label") and _rc_target(e0):
                    continue                            # LLM 재확인(투표·사람)으로 label 을 바꾸기로 한 샘플은 반영 전이어도 그대로
                if meta.get("sk"):
                    edits[meta["sk"]] = {"action": "delete", "id": meta.get("id", ""), "orig_label": meta.get("label", ""),
                                         "why": ("불일치: " + w)[:300], "at": RT.now_iso()}
                    n += 1
            _write_edits(d, edits)
    return {"deleted": n}


# ── 4. LLM 재확인 ('불일치' 줄 전부 또는 '정합 (룰 기준)' 줄의 룰당 표본, 3_6 별도 프로세스 - 화면을 다시 켜도 계속) ──────
# 판정은 07_llm_output/_recheck/results.jsonl (줄 내용 지문 k 마다, 코드 판정 cons 와 따로). 반영은 기존 수정·삭제(edits)로.
RECHECK_V = {"ok": "label 맞음", "flip": "label 반대", "unfit": "학습에 부적합", "unknown": "판단 불가"}   # 3_6 VERDICTS 와 같음
RECHECK_WHY = "LLM 재확인"                               # 반영한 수정·삭제의 사유 앞부분 (되돌리기가 이것만 찾음)
RECHECK_RULE_WHY = "LLM 재확인 [정합 (룰 기준)]"          # 그중 '정합 (룰 기준)' 줄에서 반영한 것 ('불일치' 것과 따로 되돌림)
ROW_KEY = re.compile(r'^\{"k": "([0-9a-f]+)"')          # 줄별 결과(frame) 한 줄의 내용 지문
_rc = {"M": None, "res": (None, {}), "cm": (None, {}), "mc": (None, {})}   # 3_6 모듈, results.jsonl (수정 시각, 내용 지문 → 가장 최근 판정), 합의, 모델별 줄 수
_dec_lock = threading.Lock()                            # decisions.jsonl (사람 판단·유지 표시) 덧붙이기


def recheck_mod():
    with _lock:
        if _rc["M"] is None:
            _rc["M"] = _load("recheck_ui", "3_6_recheck_labels.py")
    return _rc["M"]


def recheck_stamp():
    try:
        return (recheck_mod().OUT_DIR / "results.jsonl").stat().st_mtime_ns
    except OSError:
        return 0


def recheck_latest():
    """줄 내용 지문 → 가장 최근 LLM 재확인 (results.jsonl 이 바뀔 때만 다시 읽음, 없으면 {})."""
    mt = recheck_stamp()
    if not mt:
        return {}
    if _rc["res"][0] != mt:
        M = recheck_mod()
        _rc["res"] = (mt, M.latest_by_key(M.read_results()))
    return _rc["res"][1]


# 교차 확인·합의 (2026-11-02~, 3_6 consensus_state): 여러 모델의 판정 + 화면의 사람 판단·유지 표시(decisions.jsonl) → 줄마다
# 확인됨(유지)·삭제·추가 분석. 삭제·label 반대는 수정·삭제(edits)로 저장하고, 유지·사람 판단은 decisions.jsonl 에 덧붙인다.
RECHECK_X = {"keep": "확인됨 (유지)", "delete": "삭제", "flip": "label 반대", "review": "추가 분석"}   # 3_6 STATES 와 같음


def _dec_file():
    return recheck_mod().OUT_DIR / "decisions.jsonl"


def _dec_stamp():
    try:
        return _dec_file().stat().st_mtime_ns
    except OSError:
        return 0


def recheck_cons(target):
    """대상('bad'·'rule')의 합의 {k: (상태, 까닭, {모델: 기록})} - results.jsonl·decisions.jsonl 이 바뀔 때만 다시 센다.
    3_6 이 예전 판(합의 함수 없음)이면 {}."""
    M = recheck_mod()
    if not hasattr(M, "consensus_map"):
        return {}
    st = (recheck_stamp(), _dec_stamp())
    if _rc["cm"][0] != st:
        res, dec = M.read_results(), M.read_decisions()
        _rc["cm"] = (st, {t: M.consensus_map(res, dec, t) for t in ("bad", "rule")})
        _rc["mc"] = (st, M.model_counts(res))
    return _rc["cm"][1].get(target, {})


def _vname(m):
    return m.split(":", 1)[-1]                          # 화면 표시: provider 없이 (claude:claude-opus-5-5 → claude-opus-5-5)


def _votes_list(votes):
    """{모델: 기록} → [[모델, 판정, 근거, 시각]] (모델 이름순)."""
    return [[_vname(m), r.get("verdict", ""), r.get("reason", ""), r.get("at", "")] for m, r in sorted(votes.items())]


def _votes_text(votes):
    return " · ".join(f"{_vname(m)} {RECHECK_V.get(r.get('verdict'), r.get('verdict'))}" for m, r in sorted(votes.items()))


def _write_decisions(recs):
    """decisions.jsonl 에 덧붙임 (화면만 씀 - 3_6 은 읽기만)."""
    if not recs:
        return
    p = _dec_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _dec_lock, open(p, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)


def _rc_scope(date):
    dates = [d["date"] for d in dataset_dates()]
    if date not in ("", "all", *dates):
        raise ValueError("date")
    return dates if date in ("", "all") else [date]


def _rc_target(e):
    """수정·삭제 하나 → LLM 재확인으로 저장한 것이면 그 대상 ('bad'·'rule'), 아니면 None."""
    why = str(e.get("why", ""))
    return "rule" if why.startswith(RECHECK_RULE_WHY) else "bad" if why.startswith(RECHECK_WHY) else None


def _rc_new(M):
    """3_6 이 '정합 (룰 기준)' 대상을 아는 판인지 (화면만 먼저 다시 켰을 때 예전 3_6 이면 '불일치'만)."""
    return hasattr(M, "rule_counts")


def recheck_state(q):
    """화면 4 'LLM 재확인': 실행 상태(.run/recheck.json), 지금 날짜·장비의 '불일치' 줄과 판정 수, 반영한 수,
    지금 날짜(두 장비, 내용 기준)의 '정합 (룰 기준)' 줄·재확인한 줄의 판정 수·룰 수·반영한 수, 모델 목록·기본값."""
    M, dev = recheck_mod(), _q(q, "dev", "ips")
    if dev not in ("ips", "waf"):
        raise ValueError("dev")
    scope = _rc_scope(_q(q, "date"))
    latest, cnt, bad, applied, rapplied = recheck_latest(), {k: 0 for k in RECHECK_V}, 0, 0, 0
    bad_keys = set()
    for d in scope:
        for r in M.bad_rows(d, dev):
            bad += 1
            bad_keys.add(r["k"])
            x = latest.get(r["k"])
            if x:
                cnt[x["verdict"]] = cnt.get(x["verdict"], 0) + 1
        for e in read_edits(d).values():
            t = _rc_target(e)
            applied += t == "bad"
            rapplied += t == "rule"
    rule = dict(M.rule_counts(latest, scope), applied=rapplied) if _rc_new(M) else None
    cross = None
    if hasattr(M, "consensus_counts"):                  # 합의 수 (교차 확인, 2026-11-02~)
        cross = {"bad": M.consensus_counts(bad_keys, recheck_cons("bad")),
                 "rule": M.consensus_counts(M.rule_keys(scope), recheck_cons("rule")) if rule else None,
                 "models": _rc["mc"][1], "default": M.CROSS_DEFAULT}
    _, t = _read_settings()
    rev = next((r for r in t.get("모델", []) if len(r) >= 3 and r[0] == "검토"), None)
    run = {r[0]: r[1] for r in t.get("실행", []) if len(r) >= 2}
    known = [f"{p}:{m}" for p, ms in KNOWN_MODELS.items() for m in ms]
    for role in (_magg["val"] or {}).values():          # 화면 2 가 이미 센 지난 기록의 모델도 (처음부터 세면 10초 넘게 걸려 세지 않음)
        known += [pm for pm in role if pm not in known]
    return {"job": job_state("recheck"), "status": RT.read_status("recheck"), "bad": bad, "counts": cnt,
            "todo": bad - sum(cnt.values()), "applied": applied, "verdicts": RECHECK_V, "efforts": EFFORTS, "known": known,
            "model": f"{rev[1]}:{rev[2]}" if rev and rev[1] not in ("", "-", "none") else "",
            "effort": run.get("검토 추론 강도", ""), "rule": rule, "cross": cross}


def _rc_args(b):
    """화면의 대상·날짜·장비·표본·모델·추론 강도 → 3_6 인자 {model, effort, dates, target, devs, per_rule, llm_only}.
    model '' = LLM_설정.md 검토 모델, effort '' = '검토 추론 강도' 값, '-' = 넘기지 않음 (모델 기본값).
    이번 실행에만 쓰고 LLM_설정.md 는 바꾸지 않는다. 장비 all = 두 장비 (같은 내용은 한 번)."""
    model = str(b.get("model") or "").strip()
    if model and (not MODEL_RE.match(model) or model == "none"):
        raise ValueError("모델 ('provider:model')")
    eff = str(b.get("effort") or "").strip().lower()
    if eff not in ("", "-", *EFFORTS):
        raise ValueError(f"추론 강도 (비우거나 - 또는 {'·'.join(EFFORTS)})")
    dates = _rc_scope(b.get("date") or "all")
    target, devs = b.get("target") or "bad", b.get("devs") or "all"
    if target not in ("bad", "rule"):
        raise ValueError("대상 (bad·rule)")
    if devs not in ("all", "ips", "waf"):
        raise ValueError("장비 (all·ips·waf)")
    try:
        per_rule = int(b.get("per_rule") or 3)
    except (TypeError, ValueError):
        raise ValueError("룰당 개수 (1~1000)")
    if not 1 <= per_rule <= 1000:
        raise ValueError("룰당 개수 (1~1000)")
    if target == "rule" and not _rc_new(recheck_mod()):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 '정합 (룰 기준)' 대상을 모르는 예전 판입니다 - 새 판으로 바뀐 뒤 화면을 다시 켜세요")
    cross = str(b.get("cross") or "").strip()           # 교차 확인: 1번째 모델 (그 모델이 맞음·판단 불가라 한 줄만 2번째 모델로)
    if cross:
        M = recheck_mod()
        if not hasattr(M, "check_cross"):
            raise ValueError("지금 불러온 3_6_recheck_labels.py 가 교차 확인을 모르는 예전 판입니다 - 화면을 다시 켜세요")
        if not MODEL_RE.match(cross) or cross in ("none", "mock"):
            raise ValueError("1번째 모델 ('provider:model')")
        model = model or M.CROSS_DEFAULT                # 2번째 모델을 비우면 기본 제안 (검토 모델은 보통 1번째와 같음)
        M.check_cross(cross, model)
    pick = _rc_pick(b.get("pick")) if cross else None   # 애매한 룰만 (교차 확인일 때만)
    tiebreak = bool(b.get("tiebreak")) and not cross    # 3차 모델 투표 (합의가 '투표 갈림'인 줄만, 2026-11-04~)
    if tiebreak and not hasattr(recheck_mod(), "vote_text"):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 3차 모델 투표를 모르는 예전 판입니다 - 화면을 다시 켜세요")
    return {"model": model, "effort": eff or None, "dates": None if (b.get("date") or "all") == "all" else dates,
            "target": target, "devs": ("ips", "waf") if devs == "all" else (devs,), "per_rule": per_rule,
            "llm_only": b.get("llm_only", True) is not False, "cross": cross or None, "pick": pick, "tiebreak": tiebreak}


def _rc_pick(p):
    """화면의 '대상 룰 - 애매한 룰만' {lo, hi (%), min_rows, small: include|exclude} → 3_6 pick (lo·hi 는 0~1). 없으면 None (전부)."""
    if not p:
        return None
    try:
        lo, hi, n = float(p.get("lo")), float(p.get("hi")), int(p.get("min_rows") or 2)
    except (TypeError, ValueError):
        raise ValueError("애매한 룰: 하한·상한(%)·판정 줄 수")
    if not 0 <= lo < hi <= 100 or not 1 <= n <= 1000 or p.get("small", "include") not in ("include", "exclude"):
        raise ValueError("애매한 룰: 0 ≤ 하한 < 상한 ≤ 100 (%), 판정 줄 1~1000")
    if not hasattr(recheck_mod(), "rule_groups"):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 '애매한 룰만'을 모르는 예전 판입니다 - 화면을 다시 켜세요")
    return {"lo": round(lo / 100, 4), "hi": round(hi / 100, 4), "min_rows": n, "small": p.get("small", "include")}


def _rc_key_problem(model):
    """LLM 재확인 모델(비우면 LLM_설정.md 검토 모델)의 provider 키가 .env(또는 환경 변수)에 있는지 - 값은 보지 않고 있는지만
    (2026-11-02~: 키 없이 시작하면 3_6 이 요청마다 실패하고 멈춤). 없으면 안내 문구, 있거나 키가 필요 없으면 ''."""
    _, t = _read_settings()
    if not model:
        rev = next((r for r in t.get("모델", []) if len(r) >= 3 and r[0] == "검토"), None)
        model = f"{rev[1]}:{rev[2]}" if rev and rev[1] not in ("", "-", "none") else ""
    prov, _, name = model.partition(":")
    p = next((r for r in t.get("provider", []) if len(r) >= 4 and r[0] == prov), None)
    key = p[2] if p and p[3] != "claude_cli" and p[2] not in ("", "-") else ""
    st = _key_state(key, _dotenv()) if key else "있음"
    if st == "있음":
        return ""
    cli = f"claude_cli:{name}" if prov == "anthropic" else "claude_cli:sonnet"
    return (f"{model} 의 키 {key} 가 .env 에 {'없습니다' if st == '없음' else '있지만 형식이 맞지 않습니다 (한글·공백)'} - "
            f"Claude 구독으로 쓰려면 모델을 {cli} 로 고르고, API 로 쓰려면 .env 에 {key} 를 넣은 뒤 다시 시작하세요")


def recheck_plan(b):
    """시작 전 확인: 대상 줄·이미 한 줄·요청 수·예상 토큰·시간·비용 (호출 없음) + 모델 키가 없으면 그 안내(key_problem)."""
    a, M = _rc_args(b), recheck_mod()
    if not _rc_new(M):                                  # 예전 3_6 ('불일치'·두 장비만)
        out = M.estimate(a["model"], a["effort"], a["dates"])
    else:
        out = M.estimate(a["model"], a["effort"], a["dates"], target=a["target"], devs=a["devs"], per_rule=a["per_rule"],
                         llm_only=a["llm_only"], **({"cross": a["cross"]} if a["cross"] else {}),
                         **({"pick": a["pick"]} if a["pick"] else {}), **({"tiebreak": True} if a["tiebreak"] else {}))
    return dict(out, key_problem=_rc_key_problem(a["model"]))


def recheck_start(b):
    if job_state("recheck")["running"]:
        raise ValueError("LLM 재확인이 이미 도는 중입니다")
    a = _rc_args(b)
    prob = _rc_key_problem(a["model"])                  # 키가 없으면 시작하지 않음 (요청마다 실패하고 멈추므로)
    if prob:
        raise ValueError(prob)
    args = (["--model", a["model"]] if a["model"] else []) + (["--effort", a["effort"]] if a["effort"] else []) + \
           (["--dates", ",".join(a["dates"])] if a["dates"] else [])
    if a["cross"]:                                      # 교차 확인: 대상은 1번째가 판정한 줄 (표본 옵션은 쓰지 않음)
        args += ["--cross", a["cross"]] + (["--target", "rule"] if a["target"] == "rule" else [])
        if a["pick"]:                                   # 애매한 룰만 (1번째 모델의 룰별 삭제 비율)
            p = a["pick"]
            args += ["--rules-from-ratio", str(p["lo"]), str(p["hi"]), "--min-rows", str(p["min_rows"]),
                     "--small-rules", p["small"]]
    elif a["tiebreak"]:                                 # 3차 모델 투표: '투표 갈림' 줄을 그 줄을 아직 판정하지 않은 이 모델로
        args += ["--tiebreak"] + (["--target", "rule"] if a["target"] == "rule" else [])
    elif a["target"] == "rule":
        args += ["--target", "rule", "--per-rule", str(a["per_rule"])] + ([] if a["llm_only"] else ["--with-augmented"])
    if len(a["devs"]) == 1:
        args += ["--devs", a["devs"][0]]
    RT.clear_stop("recheck")
    return {"pid": spawn("recheck", args), "args": args}


def recheck_rules(q):
    """'정합 (룰 기준)' 룰별 요약 (지금 날짜, 두 장비·내용 기준) - 2026-11-02~ 합의로 셈
    → {rows: [[룰 id, 공격 유형, 정합 (룰 기준) 줄, 판정한 줄, 확인됨, 삭제, 추가 분석, label 반대(사람)]]} (삭제 비율·판정 수 큰 순).
    3_6 이 예전 판이면 가장 최근 판정으로 [.., 맞음, 반대·부적합, 판단 불가, 0]."""
    M = recheck_mod()
    if not _rc_new(M):
        return {"rows": []}
    scope = _rc_scope(_q(q, "date"))
    if hasattr(M, "rule_summary_x"):
        rows = [[s["rule_id"], s["attack_type"], s["rows"], s["n"], s["keep"], s["delete"], s["review"], s["flip"]]
                for s in M.rule_summary_x(recheck_cons("rule"), scope)]
    else:
        rows = [[s["rule_id"], s["attack_type"], s["rows"], s["n"], s["ok"], s["flip"] + s["unfit"], s["unknown"], 0]
                for s in M.rule_summary(recheck_latest(), scope)]
    rows.sort(key=lambda x: (-x[5] / x[3], -x[3], x[0]))
    # 1번째 모델의 룰별 삭제 비율 묶음 (2026-11-02~, 교차 확인 '애매한 룰만'과 같은 기준: first·lo·hi(%)·min) → 줄 끝에 묶음
    first = _q(q, "first")
    if first and hasattr(M, "rule_groups"):
        try:
            lo, hi, n = float(_q(q, "lo", "30")) / 100, float(_q(q, "hi", "70")) / 100, int(_q(q, "min", "2"))
        except ValueError:
            raise ValueError("lo/hi/min")
        votes = {k: c[2] for k, c in recheck_cons("rule").items()}     # 합의 계산에 쓴 모델별 판정 (다시 읽지 않음)
        group, _, _ = M.rule_groups(M.rule_rows(scope), votes, M.base_model(first), lo, hi, n)
        for x in rows:
            x.append(group.get(x[0], ""))
    return {"rows": rows, "groups": M.GROUPS if hasattr(M, "GROUPS") else {}}


def recheck_review(q):
    """'추가 분석' 목록 (합의가 review 인 줄, 2026-11-02~): why = split (투표 갈림 - 두 모델 이상인데 다수 없음, 3차 투표 뒤에도
    남으면 사람이 정함, 기본) · pending (한 모델만 판정 - 교차 확인 전) · all. 예전 3_6 의 mixed 는 split 으로 본다.
    '불일치'는 지금 장비, '정합 (룰 기준)'은 두 장비 (내용 기준 한 번).
    → {total, offset, rows: [{k, date, dev, line, rule_id, label, payload, reason, votes: [[모델, 판정, 근거, 시각]], why}], counts}"""
    M, target = recheck_mod(), _q(q, "target") or "bad"
    dev, why = _q(q, "dev", "ips"), _q(q, "why") or "split"
    why = "split" if why == "mixed" else why
    if target not in ("bad", "rule") or dev not in ("ips", "waf") or why not in ("split", "pending", "all"):
        raise ValueError("target/dev/why")
    if not hasattr(M, "consensus_map"):
        return {"total": 0, "offset": 0, "rows": [], "counts": {}}
    scope = _rc_scope(_q(q, "date"))
    offset, limit = page_args(q, 100)
    cm = recheck_cons(target)
    src = (r for d in scope for r in M.bad_rows(d, dev)) if target == "bad" else iter(M.rule_rows(scope).values())
    pick, seen, cnt = [], set(), {"split": 0, "pending": 0}
    for r in src:
        c = cm.get(r["k"])
        if not c or c[0] != "review" or r["k"] in seen:
            continue
        seen.add(r["k"])
        w = "split" if c[1] in ("split", "mixed") else c[1]
        cnt[w] = cnt.get(w, 0) + 1
        if why in ("all", w):
            pick.append(r)
    page = pick[offset:offset + limit]
    if target == "rule":
        page = M.rule_details(page)                     # 이 쪽의 payload·코드 판정 사유만 읽음
    rows = [{"k": r["k"], "date": r["date"], "dev": r["dev"], "line": r["line"], "rule_id": r["rule_id"], "label": r["label"],
             "payload": r["payload"][:3000], "reason": r.get("reason", ""), "votes": _votes_list(cm[r["k"]][2]),
             "why": cm[r["k"]][1]} for r in page]
    return {"total": len(pick), "offset": offset, "rows": rows, "counts": cnt}


OLD_DEL = " 합의: 삭제"     # LLM 재확인 합의로 저장한 삭제의 사유 (옛 규칙·새 규칙 모두 - 사람 판단은 '사람 (추가 분석)')
OLD_STATE = {"delete": "삭제 유지 (부적합)", "keep": "확인됨", "flip": "label 반대", "pending": "추가 분석 - 2번째 모델 대기",
             "split": "추가 분석 - 투표 갈림", "none": "판정 기록 없음"}


def _old_rule_rows():
    """LLM 재확인 합의로 저장한 삭제 (모든 날짜, 옛 규칙 '하나라도 반대·부적합 → 삭제'로 저장한 것 포함) → [(날짜, sk, 수정, 대상, 새 규칙 칸)].
    새 규칙 칸 = 결과의 sk → k 로 찾은 지금 합의 (OLD_STATE 의 키)."""
    M, rows = recheck_mod(), []
    for d in (x["date"] for x in dataset_dates()):
        for sk, e in read_edits(d).items():
            t = _rc_target(e)
            if t and e.get("action") == "delete" and OLD_DEL in str(e.get("why", "")):
                rows.append((d, sk, e, t))
    if not rows:
        return []
    sk2k = {}
    for r in M.read_results():
        if r.get("sk") and r.get("k"):
            sk2k.setdefault((r.get("target") or "bad", r["sk"]), r["k"])
    cms, out = {t: recheck_cons(t) for t in ("bad", "rule")}, []
    for d, sk, e, t in rows:
        c = cms[t].get(sk2k.get((t, sk)))
        out.append((d, sk, e, t, "none" if not c else c[1] if c[0] == "review" and c[1] in ("pending", "split")
                    else "split" if c[0] == "review" else c[0]))
    return out


def recheck_oldrule(q=None, b=None):
    """'옛 규칙으로 지운 줄 다시 보기' (2026-11-04~, 사용자가 누를 때만 바꿈): LLM 재확인 합의로 저장한 삭제를 새 규칙(부적합 삭제 +
    다수결 투표)으로 다시 센다. 보기(q): 저장 시각(분)·대상별 새 규칙 칸 수 + 되돌린 기록. action=revert: 새 규칙으로 삭제가 아닌 줄의
    삭제를 지운다 (부적합 삭제는 그대로, 지운 것은 _recheck/old_rule_revert_<시각>.json) → '학습 로그에 반영' 뒤 다시 검증하면 합의 줄에
    다시 보인다. action=restore: 가장 최근 되돌림을 다시 삭제로 (그 샘플에 그 뒤 다른 수정·삭제가 생겼으면 건너뜀)."""
    if not hasattr(recheck_mod(), "vote_text"):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 새 합의 규칙(투표)을 모르는 예전 판입니다 - 화면을 다시 켜세요")
    outdir, act = recheck_mod().OUT_DIR, (b or {}).get("action")
    backups = sorted(p for p in outdir.glob("old_rule_revert_*.json") if not p.name.endswith(".restored.json"))   # 취소한 것은 뺌
    if act == "restore":
        if not backups:
            raise ValueError("되돌린 기록이 없습니다")
        p, n, skipped = backups[-1], 0, 0
        saved = json.loads(p.read_text(encoding="utf-8"))
        for d, items in saved["edits"].items():
            with _edit_lock:
                edits = read_edits(d)
                for sk, e in items.items():
                    if sk in edits:
                        skipped += 1
                        continue
                    edits[sk] = e
                    n += 1
                _write_edits(d, edits)
        p.replace(p.with_name(p.stem + ".restored.json"))
        return {"restored": n, "skipped": skipped}
    rows = _old_rule_rows()
    if act == "revert":
        drop = {}
        for d, sk, e, t, s in rows:
            if s not in ("delete", "none"):
                drop.setdefault(d, {})[sk] = e
        if not drop:
            return {"reverted": 0}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"old_rule_revert_{stamp}.json").write_text(json.dumps(
            {"at": RT.now_iso(), "why": "옛 규칙으로 지운 줄 중 새 규칙으로 삭제가 아닌 줄의 삭제를 지움", "edits": drop},
            ensure_ascii=False), encoding="utf-8")
        n = 0
        for d, items in drop.items():
            with _edit_lock:
                edits = read_edits(d)
                for sk, e in items.items():
                    if edits.get(sk) == e:              # 그사이 바뀌지 않은 것만
                        edits.pop(sk)
                        n += 1
                _write_edits(d, edits)
        return {"reverted": n, "backup": f"old_rule_revert_{stamp}.json"}
    groups = {}
    for d, sk, e, t, s in rows:
        g = groups.setdefault((str(e.get("at", ""))[:16], t), {k: 0 for k in OLD_STATE})
        g[s] += 1
    last = None
    if backups:
        try:
            x = json.loads(backups[-1].read_text(encoding="utf-8"))
            last = {"file": backups[-1].name, "at": x.get("at", ""), "n": sum(len(v) for v in x.get("edits", {}).values())}
        except (OSError, ValueError):
            last = None
    return {"groups": [[at, t, c] for (at, t), c in sorted(groups.items())], "states": OLD_STATE, "last": last}


def recheck_stop(b):
    """중지 = 지금 보낸 요청까지 (판정은 저장됨), 강제 중지 = 바로 (보내던 요청의 판정은 버림)."""
    st = job_state("recheck")
    if not st["running"]:
        raise ValueError("도는 LLM 재확인이 없습니다")
    if b.get("force"):
        return {"killed": force_kill(st["lock"].get("pid") or st["ui"].get("pid"))}
    RT.request_stop("recheck")
    return {"stop": True}


def _rc_rule_rows(M, d):
    """그 날짜의 '정합 (룰 기준)' 줄 (두 장비, 샘플 지문 sk 기준으로 한 번) - 반영할 때 bad_rows 대신."""
    out, seen = [], set()
    for dev in ("ips", "waf"):
        for k, rid, _llm, _line, sk, label, _typ in M.rule_pool(d, dev):
            if sk and sk not in seen:
                seen.add(sk)
                out.append({"k": k, "sk": sk, "rule_id": rid, "label": label})
    return out


def recheck_apply(b):
    """LLM 재확인 판정을 수정·삭제로 저장 ('학습 로그에 반영' 때 적용·되돌릴 수 있음). 대상 target:
    bad = 지금 날짜·장비의 '불일치' 줄, rule = 지금 날짜(두 장비)의 '정합 (룰 기준)' 줄 중 재확인한 줄.
    delete = label 반대·학습에 부적합 → 삭제, flip = label 반대 → label 바꾸기 (가장 최근 판정 하나로 - 예전 방식),
    xdelete = 합의가 삭제인 줄 → 삭제 (2026-11-04~ 부적합이 하나라도 있는 줄), xflip = 합의가 label 반대(투표 다수)인 줄 → label 바꾸기
    (사유 = 투표 결과·모델 근거, 2026-11-04~), xkeep = 합의가 확인됨인 줄에 유지 표시,
    decide = 고른 줄(ks)에 사람 판단 decision (keep 유지·delete 삭제·flip label 반대 - 삭제·label 반대는 수정·삭제로도 저장,
    유지는 그 샘플에 LLM 재확인으로 저장한 수정·삭제를 지움), undo = 그 대상에서 LLM 재확인으로 저장한 수정·삭제를 되돌리고
    유지 표시·사람 판단도 지움. 유지 표시·사람 판단은 decisions.jsonl (합의가 그것을 따름).
    사람이 직접 저장한 수정·삭제가 있는 샘플은 건드리지 않는다 (건너뛴 수로 알림)."""
    act, dev, target = b.get("action"), b.get("dev", "ips"), b.get("target") or "bad"
    if act not in ("delete", "flip", "undo", "xdelete", "xflip", "xkeep", "decide") or dev not in ("ips", "waf") \
            or target not in ("bad", "rule"):
        raise ValueError("action/dev/target")
    decision, ks = b.get("decision"), set(b.get("ks") or [])
    if act == "decide" and (decision not in ("keep", "delete", "flip") or not ks):
        raise ValueError("decision (keep·delete·flip) / 고른 줄")
    M, latest, out = recheck_mod(), recheck_latest(), {"saved": 0, "skipped": 0, "undone": 0, "marked": 0}
    if target == "rule" and not _rc_new(M):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 '정합 (룰 기준)' 대상을 모르는 예전 판입니다")
    cons = act in ("xdelete", "xflip", "xkeep", "decide")
    if cons and not hasattr(M, "consensus_map"):
        raise ValueError("지금 불러온 3_6_recheck_labels.py 가 교차 확인·합의를 모르는 예전 판입니다 - 화면을 다시 켜세요")
    vtext = getattr(M, "vote_text", None) or _votes_text   # 사유: 투표 결과 (예: 반대 2 : 맞음 1)와 모델 근거
    cm, why0 = recheck_cons(target) if cons else {}, RECHECK_RULE_WHY if target == "rule" else RECHECK_WHY
    now, decs, marked = RT.now_iso(), [], set()
    for d in _rc_scope(b.get("date") or "all"):
        rows = [] if act == "undo" else _rc_rule_rows(M, d) if target == "rule" else M.bad_rows(d, dev)
        with _edit_lock:
            edits, n0 = read_edits(d), (out["saved"], out["undone"])
            if act == "undo":
                for sk in [s for s, e in edits.items() if _rc_target(e) == target]:
                    edits.pop(sk)
                    out["undone"] += 1
            for r in rows:
                if cons:
                    c = cm.get(r["k"])
                    if not c or not r["sk"]:
                        continue
                    if act == "xkeep":                  # 유지 표시 = 합의로 확인됨 (decisions, 그 뒤 판정이 더 와도 유지)
                        if c[0] == "keep" and c[1] != "decided" and r["k"] not in marked:
                            marked.add(r["k"])
                            decs.append({"k": r["k"], "ck": r.get("ck", ""), "sk": r["sk"], "target": target, "decision": "keep",
                                         "by": "합의", "at": now, "votes": {m: x.get("verdict") for m, x in c[2].items()}})
                        continue
                    if act in ("xdelete", "xflip") and (c[0] != act[1:] or c[1] == "decided"):
                        continue                        # 합의가 삭제(부적합)·label 반대(투표)인 줄만 (사람 판단은 이미 저장)
                    if act == "decide":
                        if r["k"] not in ks:
                            continue
                        if r["k"] not in marked:
                            marked.add(r["k"])
                            decs.append({"k": r["k"], "ck": r.get("ck", ""), "sk": r["sk"], "target": target, "decision": decision,
                                         "by": "사람", "at": now, "votes": {m: x.get("verdict") for m, x in c[2].items()}})
                    e0 = edits.get(r["sk"])
                    if e0 and _rc_target(e0) is None:   # 사람이 직접 저장한 수정·삭제는 그대로
                        out["skipped"] += 1
                        continue
                    if act == "decide" and decision == "keep":
                        if e0:                          # 유지로 정함 → LLM 재확인으로 저장한 삭제·label 바꾸기를 지움
                            edits.pop(r["sk"])
                            out["undone"] += 1
                        continue
                    if act in ("xdelete", "xflip") and e0:
                        out["skipped"] += 1
                        continue
                    dec = act[1:] if act in ("xdelete", "xflip") else decision
                    who = "합의" if act in ("xdelete", "xflip") else "사람 (추가 분석)"
                    e = {"action": "delete" if dec == "delete" else "edit", "id": r["rule_id"], "orig_label": r["label"],
                         "why": f"{why0} {who}: {RECHECK_X[dec]} - {vtext(c[2])}"[:300], "at": now}
                    if dec == "flip":
                        e["label"] = "normal" if r["label"] == "anomalies" else "anomalies"
                    edits[r["sk"]] = e
                    out["saved"] += 1
                    continue
                x = latest.get(r["k"])
                if not x or not r["sk"] or not (x["verdict"] == "flip" or (act == "delete" and x["verdict"] == "unfit")):
                    continue
                if r["sk"] in edits:
                    out["skipped"] += 1
                    continue
                e = {"action": "delete" if act == "delete" else "edit", "id": r["rule_id"], "orig_label": r["label"],
                     "why": f"{why0} ({x.get('model', '')}): {RECHECK_V[x['verdict']]} - {x.get('reason', '')}"[:300],
                     "at": RT.now_iso()}
                if act == "flip":
                    e["label"] = "normal" if r["label"] == "anomalies" else "anomalies"
                edits[r["sk"]] = e
                out["saved"] += 1
            if (out["saved"], out["undone"]) != n0:
                _write_edits(d, edits)
    if act == "undo" and hasattr(M, "read_decisions"):     # 유지 표시·사람 판단도 지움 (그 대상의 지난 것은 합의에 쓰지 않음)
        decs.append({"decision": "reset", "target": target, "at": now})
    _write_decisions(decs)
    out["marked"] = len(marked)
    return out


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
    """신규 학습 데이터 화면의 조건 → (날짜 목록, 장비, label, 룰 id, payload 검색어, label 정합성, 보류 대상, LLM 재확인)."""
    dev, date = _q(q, "dev", "ips"), _q(q, "date")
    dates = [d["date"] for d in dataset_dates()]
    if date and date != "all":
        if date not in dates:
            raise ValueError("date")
        dates = [date]
    label, rule, text, cons = _q(q, "label"), _q(q, "rule").strip(), _q(q, "q").strip(), _q(q, "cons")
    hold, llm = _q(q, "hold"), _q(q, "llm")
    if (dev not in ("ips", "waf", *CONV) or label not in ("", "anomalies", "normal") or cons not in ("", *CONS)
            or hold not in ("", *HOLD) or llm not in ("", "todo", "any", *RECHECK_V, *(f"c_{s}" for s in RECHECK_X))):
        raise ValueError("dev/label/cons/hold/llm")
    return dates, dev, label, rule, text, cons, hold, llm


FRAME_COLS = ["index", "payload", "method", "label", "label 사유", "검증 결과", "label 정합성", "label 정합성 사유",
              "탐지 피처", "라벨 검증용 피처", "rule_id", "date", "LLM 재확인", "LLM 재확인 근거", "LLM 재확인 모델",
              "LLM 합의", "LLM 판정 (모델별)"]
CHECK_MARK = {"ok": "✔", "warn": "△", "bad": "✘"}
REVIEW_WHY = {"pending": "2번째 모델 대기", "split": "투표 갈림", "mixed": "판단 불가가 섞임", "unfit": "부적합", "decided": "표시함",
              "vote": "투표"}   # 합의 까닭 (3_6 consensus_state, 2026-11-04~ split·unfit - mixed 는 예전 판)


def _cons_maps():
    """줄의 label 정합성(BAD·RULE) → 그 대상의 합의 {k: (상태, 까닭, {모델: 기록})} (그 밖의 줄은 합의 없음)."""
    return {"BAD": recheck_cons("bad"), "RULE": recheck_cons("rule")}


def _cons_text(c):
    """합의 한 칸: '확인됨 (유지)' · '추가 분석 (2번째 모델 대기)' 등."""
    return RECHECK_X[c[0]] + (f" ({REVIEW_WHY[c[1]]})" if REVIEW_WHY.get(c[1]) else "") if c else ""


def ml_rows(payloads, dev, cache=None):
    """줄별 ML 피처 (ml_features). 표준모델 피처 파일을 못 읽는 등으로 실패해도 표·CSV 는 그대로 보이게 → (목록, 오류 문구)."""
    try:
        return ml_features(payloads, dev, cache), ""
    except Exception as e:
        return [[] for _ in payloads], f"탐지 피처를 계산하지 못했습니다: {type(e).__name__}: {e}"


def feat_text(feats):
    return "; ".join(f"{x['name']}={x['value']}" + (f" ({x['desc']})" if x.get("desc") else "") for x in feats)


def _cell(v):
    """엑셀이 수식으로 읽지 않게 (=, +, -, @ 로 시작하는 값 앞에 ')."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def frame_csv(q):
    """데이터 프레임 내려받기: 조건에 맞는 모든 줄을 화면의 열 그대로 CSV(UTF-8 BOM, 엑셀용)로.
    → (줄 수, 파일 이름, 바이트 조각 생성기). '전체 검증'의 저장된 줄별 결과가 있으면 그것을 쓰고(빠름),
    없으면 줄마다 다시 확인한다(1천 줄에 약 15초). 200줄씩 만들어 바로 보낸다."""
    dates, dev, label, rule, text, cons, hold, llm = data_query(q)
    src = "ips" if dev in CONV else dev                 # FW·웹: 원본 IPS 줄을 골라 바꿔서 (conv_csv)
    if dev in CONV:
        conv_verified(dates, cons, hold, llm)
    parts = []
    for d in dates:                                     # 조건 오류(예: 검증 전 정합성 필터)는 보내기 전에
        try:
            m = matching(d, src, label, rule, text, cons, hold, llm)
        except NoVerify:                                # '전체': 검증 결과를 쓸 수 없는 날짜는 빼고 (표와 같음, 화면이 받기 전에 알림)
            if len(dates) == 1:
                raise
            continue
        parts.append((d, m, len(line_index(_log_path(d, src))) if m is None else len(m)))
    total = sum(n for _, _, n in parts)
    name = "_".join([dev, dates[0] if len(dates) == 1 else "all", "frame"] +
                    [x for x in (label, cons, hold and f"hold-{hold}", llm and f"llm-{llm}") if x]) + ".csv"
    if dev in CONV:
        return total, name, conv_csv(dev, parts)
    latest, cms = recheck_latest(), _cons_maps()
    vc = view_cache_ui(dev)                             # 탐지 피처 = 표준·신규 피처 표시 캐시 (없는 줄만 계산)

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
            vl = voted_lines(d, dev) if cache else set()   # 합의로 label 바꿈
            for s in range(0, n, 200):
                idx = list(range(s, min(n, s + 200))) if m is None else m[s:s + 200]
                metas = read_label_lines(li[4], idx) if li else [{}] * len(idx)
                saved = cached_rows(cache, idx) if cache else None
                lines = read_lines(path, idx)
                fields = [split_log(line) for line in lines]
                ml, err = ml_rows([f[6] if len(f) == 8 else line for f, line in zip(fields, lines)], dev, vc)
                buf.seek(0)
                buf.truncate()
                for k, (line, f, meta) in enumerate(zip(lines, fields, metas)):
                    payload, method, lab = (f[6], f[5], f[7]) if len(f) == 8 else (line, "", "")
                    try:
                        a = saved[k] if saved else analyze(payload, lab, meta)
                    except Exception as e:              # 한 줄이 실패해도 파일은 끝까지
                        a = {"reason": "", "checks": [("bad", f"확인 실패: {type(e).__name__}: {e}")], "cons_text": "",
                             "cons_reason": "", "features": []}
                    if idx[k] in vl:
                        a = _voted_row(a)
                    key = _row_key(payload, lab, meta)
                    rc = latest.get(key) if latest else None    # LLM 재확인 (줄 내용 지문)
                    cx = cms.get(a.get("cons"), {}).get(key)    # 합의 (교차 확인)
                    w.writerow([g0 + s + k] + [_cell(v) for v in (
                        payload, method, lab, a["reason"],
                        " / ".join(f"{CHECK_MARK.get(st, '')} {t}" for st, t in a["checks"]),
                        a["cons_text"], a["cons_reason"],
                        err or feat_text(ml[k]), feat_text(x for x in a["features"] if not x["name"].startswith("k_")),
                        meta.get("id", ""), d,
                        RECHECK_V.get(rc["verdict"], rc["verdict"]) if rc else "", rc["reason"] if rc else "",
                        rc["model"] if rc else "", _cons_text(cx),
                        " / ".join(f"{m} {RECHECK_V.get(v, v)}: {r}" for m, v, r, _ in _votes_list(cx[2])) if cx else "")])
                yield buf.getvalue().encode("utf-8")
            g0 += n
    return total, name, chunks()


def data_rows(q):
    """신규 학습 데이터 표 한 페이지 (데이터 프레임 열 + 수정·삭제 상태)."""
    dates, dev, label, rule, text, cons, hold, llm = data_query(q)
    offset, limit = page_args(q, 100)
    if dev in CONV:
        return conv_page(dates, dev, label, rule, text, cons, hold, offset, limit, llm)
    skipped = []                                        # '전체' + 정합성·LLM 재확인 조건: 검증 결과를 쓸 수 없는 날짜
    total, picked = select_rows(dates, dev, label, rule, text, offset, limit, cons, hold, llm, skipped)
    rows, latest, cms = [], recheck_latest(), _cons_maps()
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
        edits, vl = read_edits(d), (voted_lines(d, dev) if cache else set())   # vl = 합의로 label 바꿈
        for k, ((i, g), line, meta) in enumerate(zip(items, lines, metas)):
            f = split_log(line)
            payload, method, lab = (f[6], f[5], f[7]) if len(f) == 8 else (line, "", "")
            a = saved[k] if saved else analyze(payload, lab, meta)
            if i in vl:
                a = _voted_row(a)
            e = edits.get(meta.get("sk")) or {}
            key = _row_key(payload, lab, meta)
            rc = latest.get(key) if latest else None    # LLM 재확인 (3_6, 줄 내용 지문)
            cx = cms.get(a.get("cons"), {}).get(key)    # 합의: 상태·까닭·모델별 판정 ('불일치'·'정합 (룰 기준)' 줄)
            rows.append(dict(a, index=g, line=i, date=d, payload=payload, method=method, label=lab, rule_id=meta.get("id", ""),
                             sk=meta.get("sk", ""), edited=meta.get("edited", ""),
                             edit={"action": e["action"], "label": e.get("label", ""), "payload": bool(e.get("http"))} if e else None,
                             llm={"v": rc["verdict"], "r": rc.get("reason", ""), "m": rc.get("model", ""), "at": rc.get("at", "")}
                             if rc else None, llmc={"s": cx[0], "w": cx[1], "v": _votes_list(cx[2])} if cx else None))
    rows.sort(key=lambda r: r["index"])
    # 탐지 피처 = ML 피처(표준·신규) 표시. 검증 결과의 흔적 k_ 는 판정에만 쓰고 보이지 않고, ra_·r_ 는 '라벨 검증용'으로 따로
    ml, err = ml_rows([r["payload"] for r in rows], dev)
    for r, x in zip(rows, ml):
        r["vfeatures"] = [v for v in r["features"] if not v["name"].startswith("k_")]
        r["features"] = x
    return {"total": total, "offset": offset, "rows": rows, "ml_error": err,
            "skipped": [dict(g, text=gap_text(g)) for g in skipped]}


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


def conv_verified(dates, cons, hold, llm=""):
    """정합성 조건은 원본 검증 결과가 있어야 함 (matching 과 같은 확인, 화면 문구에 원본 장비 이름 없이).
    날짜 하나면 까닭·할 일을 담은 오류, '전체'면 select_rows 가 그 날짜를 빼고 알린다 (2026-11-02~)."""
    if (cons or hold == "rule" or llm) and len(dates) == 1:
        g = verify_gap(dates[0], "ips")
        if g:
            raise ValueError(gap_text(g, dev_name=False))


def conv_page(dates, dev, label, rule, text, cons, hold, offset, limit, llm=""):
    """FW·웹 보기 한 페이지: 줄마다 변환 로그·파싱 결과·공격 근거 (웹은 룰 조건으로 근거가 남는지 확인)."""
    M = conv_mod()
    C, rules, _ = checker()
    conv_verified(dates, cons, hold, llm)
    skipped = []
    total, picked = select_rows(dates, "ips", label, rule, text, offset, limit, cons, hold, llm, skipped)
    rows = []
    for d in dict.fromkeys(p[0] for p in picked):
        items = [(i, g) for dd, i, g in picked if dd == d]
        for (i, g), line, meta in zip(items, *conv_lines(d, [i for i, _ in items])):
            rows.append(dict(M.convert(dev, line, meta, (C, rules)), index=g, line=i, date=d))
    return {"total": total, "offset": offset, "rows": rows, "source": M.settings()["source"],
            "skipped": [dict(g, text=gap_text(g, dev_name=False)) for g in skipped]}


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
    else:                                               # 1차 확정 (후보): 초안 → rev 1, 후보 → rev+1 (데이터 추가 반영)
        v = C.load_versions()
        name = b.get("name", "")
        if C.get_version(v, name).get("status") not in ("draft", "candidate"):
            raise ValueError("초안이나 후보만 내보냅니다 (최종 승인한 버전은 고칠 수 없음)")
        if job_state("evalver")["running"]:
            raise ValueError("피처 테스트가 도는 중입니다 - 끝난 뒤 내보내세요")
        if job_state("server")["running"]:              # 서버 사전 점검이 학습 파일을 읽는 중
            raise ValueError("서버 반영 사전 점검이 도는 중입니다 - 끝난 뒤 내보내세요")
        args = ["export", "--version", name]
    RT.clear_stop("supervised")
    return {"pid": spawn("supervised", args), "args": args}


# 후보 버전 (2026-10-30~): 피처 테스트(5_4 eval-version, job evalver - 장비마다 한 번, 남은 장비는 차례로)·목표·데이터 추가·승인·버리기
_evq = {"name": None, "rev": None, "todo": [], "exps": [], "quick": False, "last": None}


def _sup_busy(what):
    if job_state("supervised")["running"]:
        raise ValueError(f"비교·1차 확정이 도는 중입니다 - 끝난 뒤 {what}")
    if job_state("evalver")["running"] or _evq["todo"]:
        raise ValueError(f"피처 테스트가 도는 중입니다 - 끝난 뒤 {what}")
    if job_state("server")["running"]:
        raise ValueError(f"서버 반영 사전 점검이 도는 중입니다 - 끝난 뒤 {what}")


def sup_experiments():
    """피처 테스트에서 고를 실험 [[id, 설명]] (5.4 configs.json, 기본 E0 = 표준 피처만)."""
    C, _ = sup_mods()
    cfg = C.read_json(C.SUP / "experiments" / "configs.json", {}) or {}
    return [[k, (x or {}).get("desc", "")] for k, x in (cfg.get("experiments") or {}).items()]


def _ev_next():
    """남은 장비가 있고 피처 테스트가 쉬고 있으면 다음 장비를 시작 (화면이 상태를 물을 때마다 확인)."""
    q = _evq
    if not q["todo"] or job_state("evalver")["running"]:
        return
    st = RT.read_status("eval_version") or {}
    if q["last"] and st.get("status") in ("stopped", "error"):   # 앞 장비가 멈추거나 실패하면 이어서 하지 않음
        q["todo"] = []
        return
    dev = q["todo"].pop(0)
    args = ["eval-version", "--version", q["name"], "--rev", str(q["rev"]), "--dev", dev, "--exp", *q["exps"]] + \
        (["--quick"] if q["quick"] else [])
    RT.clear_stop("eval_version")
    q["last"] = {"dev": dev, "pid": spawn("evalver", args), "args": args}


def sup_eval_start(b):
    C, V = sup_mods()
    v = C.load_versions()
    name = b.get("name", "")
    ver = C.get_version(v, name)
    if ver.get("status") != "candidate":
        raise ValueError("1차 확정(후보) 버전만 테스트합니다")
    _sup_busy("테스트하세요")
    devs = [d for d in (b.get("devs") or list(C.DEVICES)) if d in C.DEVICES]
    known = {k for k, _ in sup_experiments()}
    exps = list(dict.fromkeys([V.DEFAULT_EXP] + [str(x) for x in (b.get("exps") or []) if str(x) in known]))
    if not devs:
        raise ValueError("장비를 고르세요")
    _evq.update(name=name, rev=int(ver["rev"]), todo=devs, exps=exps, quick=bool(b.get("quick")), last=None)
    _ev_next()
    return {"rev": _evq["rev"], "devs": devs, "exps": exps, "started": _evq["last"]}


def sup_eval_state(q):
    """후보 하나의 피처 테스트 결과(rev 별)·승인 판단·진행 상태·실험 목록."""
    C, V = sup_mods()
    _ev_next()
    name = _q(q, "name")
    ver = C.get_version(C.load_versions(), name)
    res = V.eval_results(name)
    rev = ver.get("rev")
    try:                                                # 서버 반영 승인 조건 (5_8 gate, 로컬 전용 5_8a 가 있을 때만 켜짐)
        server = sup_srv().gate(name, rev) if rev else None
    except Exception as e:
        server = {"active": False, "error": str(e)[:300]}
    return {"results": {str(k): x for k, x in res.items()}, "rev": rev,
            "check": V.check_targets(ver.get("targets") or {}, res.get(rev) or {}) if rev else None,
            "job": job_state("evalver"), "status": RT.read_status("eval_version"),
            "queue": {"name": _evq["name"], "rev": _evq["rev"], "todo": list(_evq["todo"]), "last": _evq["last"]},
            "experiments": sup_experiments(), "default_exp": V.DEFAULT_EXP, "kinds": V.ADD_KINDS, "server": server}


def sup_cand_action(path, b):
    """후보: 목표·데이터 추가·외부 파일·승인·버리기 (도는 내보내기·테스트가 없을 때만)."""
    C, V = sup_mods()
    name = b.get("name", "")
    if path == "/api/sup/targets":
        return V.set_targets(name, b.get("targets"))
    if path == "/api/sup/approve":
        _sup_busy("승인하세요")
        return V.approve(name, int(b.get("rev") or 0), b.get("reason", ""), b.get("exp"))
    if path == "/api/sup/discard":
        _sup_busy("버리세요")
        return V.discard(name)
    if job_state("supervised")["running"]:
        raise ValueError("1차 확정(내보내기)이 도는 중입니다 - 끝난 뒤 바꾸세요")
    if path == "/api/sup/additions":
        return V.set_additions(name, b.get("kind"), b.get("values"), b.get("action", "add"))
    if path == "/api/sup/manual":
        import base64
        data = base64.b64decode(b.get("data") or "")
        if not data or len(data) > 50 * 2 ** 20:
            raise ValueError("파일이 비었거나 50MB 를 넘습니다")
        return V.add_manual(name, b.get("filename", ""), data)
    if path == "/api/sup/manual/remove":
        return V.remove_manual(name, b.get("file", ""))
    raise ValueError("path")


# 후보 '서버 반영' (2026-11-04~, 5_8_server.py): 올릴 파일·레시피 사본(로컬 전용 5_8a)·로컬 사전 점검(job server)·서버 결과·체크리스트
_srv = {"S": None}


def sup_srv():
    with _lock:
        if _srv["S"] is None:
            _srv["S"] = _load("sup_server_ui", "5_8_server.py")
    return _srv["S"]


def sup_server_state(q):
    out = sup_srv().state(_q(q, "name"))
    out["job"] = job_state("server")
    out["run"] = RT.read_status("server_check")
    return out


def sup_server_action(path, b):
    """서버 반영: 파일·사본 만들기, 사전 점검 시작·중지, 서버 결과·체크리스트 기록 (내보내기가 돌 때는 막음)."""
    S = sup_srv()
    name = b.get("name", "")
    if path == "/api/sup/server/check/stop":
        if job_state("server")["running"]:
            RT.request_stop("server_check")
        return {"stop_requested": True}
    if job_state("supervised")["running"]:
        raise ValueError("비교·1차 확정이 도는 중입니다 - 끝난 뒤 하세요")
    if path == "/api/sup/server/check":
        if job_state("server")["running"]:
            raise ValueError("서버 반영 사전 점검이 이미 도는 중입니다")
        _, _, rev, _ = S._ver(name)
        devs = [d for d in (b.get("devs") or ["ips", "waf"]) if d in ("ips", "waf")]
        if not devs:
            raise ValueError("장비를 고르세요")
        ver = S._check_ver(S.local(), str(b.get("ver") or S.STD))        # 피처 정의 판 (두 장비 모두 있어야)
        try:                                            # 그리드 (서버 Train 처럼 트리 수 × 깊이): 그리드 행 표본 (0 = 전체), 끄면 RF 100 트리
            grows = max(0, int(b.get("grid_rows", S.GRID_ROWS)))
        except (TypeError, ValueError):
            raise ValueError("그리드 행은 0 이상 정수")
        args = ["check", "--version", name, "--rev", str(rev), "--dev", *devs, "--ver", ver] + \
            (["--grid-rows", str(grows)] if b.get("grid", True) else ["--no-grid"])
        RT.clear_stop("server_check")
        return {"pid": spawn("server", args), "args": args}
    if path == "/api/sup/server/prepare":
        if job_state("server")["running"]:
            raise ValueError("서버 반영 사전 점검이 도는 중입니다 - 끝난 뒤 다시 만드세요")
        return S.prepare(name, b.get("tag", ""), b.get("nul", "drop"), b.get("labels", "keep"), str(b.get("ver") or S.STD))
    if path == "/api/sup/server/result":
        return S.record(name, b.get("dev", ""), b.get("kind", ""), b)
    if path == "/api/sup/server/step":
        return S.set_step(name, b.get("dev", ""), b.get("key", ""), b.get("done", True), b.get("memo", ""))
    raise ValueError("path")


def sup_server_file(q):
    """서버 반영 파일 내려받기 (최신 rev 폴더 바로 아래 파일만, 이름으로)."""
    d, files = sup_srv().files_of(_q(q, "name"))
    fn = _q(q, "file")
    if fn not in files:
        raise ValueError("파일이 없습니다")
    return d / fn


def sup_candidates():
    """4. 화면 '후보에 추가' 용: 승인 전 후보 [{name, rev, samples}]."""
    C, _ = sup_mods()
    return [{"name": x["name"], "rev": x.get("rev"), "samples": len((x.get("additions") or {}).get("samples") or [])}
            for x in C.load_versions()["versions"] if x.get("status") == "candidate"]


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
        rows = [x for x in MODEL_LOG.read_text(encoding="utf-8").split("\n") if x.strip()][-n:]   # '\n' 으로만
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
            "step_seconds": step_summary(), "supervised": sup, "redo": job_state("redo"),
            "custom_requests": custom_requests()}


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
            if u.path == "/api/data/verify":                # view=conv: FW·웹 보기 (원본 검증, 문구에 장비 이름 없이)
                return self._send(200, verify_summary(_q(q, "date"), _q(q, "dev", "ips"), _q(q, "view") == "conv"))
            if u.path == "/api/data/edits":
                return self._send(200, edits_summary(_q(q, "date")))
            if u.path == "/api/data/recheck":
                return self._send(200, recheck_state(q))
            if u.path == "/api/data/recheck/rules":
                return self._send(200, recheck_rules(q))
            if u.path == "/api/data/recheck/oldrule":        # '옛 규칙으로 지운 줄 다시 보기' (세기만)
                return self._send(200, recheck_oldrule(q))
            if u.path == "/api/data/recheck/review":         # 합의 '추가 분석' 목록 (교차 확인)
                return self._send(200, recheck_review(q))
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
            if u.path == "/api/sup/eval":
                return self._send(200, sup_eval_state(q))
            if u.path == "/api/sup/candidates":
                return self._send(200, {"candidates": sup_candidates()})
            if u.path == "/api/sup/server":                 # 후보 '서버 반영' 칸 (5_8)
                return self._send(200, sup_server_state(q))
            if u.path in ("/download/sup/server", "/download/sup/server/bundle"):   # bundle = 넘겨줄 묶음 (그때 만듦)
                from urllib.parse import quote
                if u.path.endswith("/bundle"):
                    if job_state("server")["running"]:
                        return self._send(409, {"error": "서버 반영 사전 점검이 도는 중입니다 - 끝난 뒤 받으세요"})
                    q = dict(q, file=[sup_srv().bundle(_q(q, "name"))])
                p = sup_server_file(q)
                ctype = {".zip": "application/zip", ".csv": "text/csv; charset=utf-8", ".json": "application/json"}.get(
                    p.suffix.lower(), "text/plain; charset=utf-8")
                return self._send(200, p.read_bytes(), ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(p.name)}"})
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
                return self._send(200, dict(feature_catalog(dev), **feature_counts(dev)))
            if u.path == "/download/rule_status.csv" and CSV_FILE.exists():
                return self._send(200, rule_status_csv(), "text/csv; charset=utf-8",
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
        if path == "/api/data/recheck/oldrule":          # 옛 규칙으로 지운 줄: revert(바뀌는 줄 삭제 되돌리기)·restore(되돌림 취소)
            if b.get("action") not in ("revert", "restore"):
                raise ValueError("action (revert·restore)")
            return 200, recheck_oldrule(b=b)
        if path == "/api/data/edit/bad":
            return 200, delete_bad(b.get("date", ""), b.get("dev", "ips"))
        if path == "/api/data/recheck/plan":
            return 200, recheck_plan(b)
        if path == "/api/data/recheck/start":
            return 200, recheck_start(b)
        if path == "/api/data/recheck/stop":
            return 200, recheck_stop(b)
        if path == "/api/data/recheck/apply":
            return 200, recheck_apply(b)
        if path == "/api/data/rebuild":
            return 200, start_rebuild(b.get("date", ""))
        if path == "/api/data/conv/save":
            return 200, conv_save_start(b)
        if path == "/api/features/count":        # 5. '켜진 줄 수 다시 계산' = 4_1 --view (표준·신규 피처, 별도 프로세스, 재검증 없음)
            dev = b.get("dev", "ips")
            if dev not in ("ips", "waf"):
                raise ValueError("dev")
            if job_state("features")["running"]:
                return 409, {"error": "켜진 줄 수를 계산하는 중입니다 - 끝난 뒤 다시 누르세요"}
            return 200, {"pid": spawn("features", ["--view", dev])}
        if path == "/api/sup/compare":
            return 200, sup_start("compare", b)
        if path in ("/api/sup/finalize", "/api/sup/export"):   # finalize = 예전 이름 (1차 확정 = 후보 내보내기)
            return 200, sup_start("export", b)
        if path == "/api/sup/eval/start":
            return 200, sup_eval_start(b)
        if path == "/api/sup/eval/stop":
            _evq["todo"] = []
            if job_state("evalver")["running"]:
                RT.request_stop("eval_version")
            return 200, {"stop_requested": True}
        if path in ("/api/sup/targets", "/api/sup/approve", "/api/sup/discard", "/api/sup/additions", "/api/sup/manual",
                    "/api/sup/manual/remove"):
            return 200, sup_cand_action(path, b)
        if path in ("/api/sup/server/prepare", "/api/sup/server/check", "/api/sup/server/check/stop", "/api/sup/server/result",
                    "/api/sup/server/step"):
            return 200, sup_server_action(path, b)
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
            out = {"started": [d for d, r in res.items() if r == "started"], "done": [d for d, r in res.items() if r == "done"]}
            if b.get("watch") and _rebuild_busy():      # 학습 로그를 다시 만드는 작업이 끝나면 바뀐 날짜만 다시 검증
                out["watch"] = arm_rewatch(_watch_pairs([dev], [(d, dev) for d in todo]))
            return 200, out
        if path == "/api/data/verify/stale":            # 화면 4 위쪽 '바뀐 날짜만 다시 검증' (검증 뒤 학습 로그가 다시 만들어진 날짜·장비)
            devs = [d for d in (b.get("devs") or ["ips", "waf"]) if d in ("ips", "waf")]
            gaps = verify_stale(devs)
            out = {"started": [[g["date"], g["dev"]] for g in gaps if start_verify(g["date"], g["dev"], "new") == "started"]}
            if b.get("watch") and _rebuild_busy():
                out["watch"] = arm_rewatch(_watch_pairs(devs, [(g["date"], g["dev"]) for g in gaps]))
            return 200, out
        if path == "/api/data/verify/watch":            # 끝나면 다시 검증 취소
            if not b.get("on"):
                _rewatch.update(on=False, pairs=[])
            return 200, rewatch_state()
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
            if b.get("order") not in (None, "", "custom"):         # 생성 순서: 기본(배치 순서) / 사용자 룰 먼저
                raise ValueError("order")
            if b.get("order") == "custom":
                args.append("--custom-first")
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
.prog{align-self:center;font-size:13px;white-space:nowrap}   /* 강제 중지 옆 진행률 */
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
.est{margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--bg);font-size:13px;line-height:1.6}.est b{font-weight:600}#gOrderInfo .warn,.est .warn{color:var(--warn);font-weight:600}
.c-OK{background:#d7f3e3;color:#13633a}.c-RULE{background:#dde8fb;color:#1f4fa3}.c-CHECK{background:#fff0d1;color:#8a5a00}.c-BAD{background:#fde0e0;color:#a32424}.c-VOTED{background:#ece0fb;color:#5b2a9a}
.vsum{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px;font-size:13px}.vwarn{flex-basis:100%;color:var(--warn);font-size:13px;line-height:1.6}
.xg{margin-top:6px}.xh{margin-top:6px}.xhr{display:flex;gap:6px;align-items:center;font-size:12px;line-height:1.6}.xhl{width:64px;text-align:right}
.hb{display:inline-block;height:9px;border-radius:2px;background:#9a9a9a;vertical-align:middle;margin-right:4px}.hb.ok{background:var(--ok)}.hb.mid{background:var(--warn)}.hb.bad{background:var(--bad)}
.vstale{margin-top:8px;padding:6px 10px;border:1px solid var(--warn);border-radius:8px;font-size:13px;line-height:1.7}.vstale b{color:var(--warn)}button.sm{padding:2px 8px;font-size:12px;margin:0 2px 2px 0}td.act{min-width:130px}
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
.sstate{margin-top:8px;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);font-size:13px;line-height:1.6}.sstate.dirty{border:2px solid var(--warn)}.sstate.clean{border:2px solid var(--ok)}.sstate .steps{margin:0 0 8px}
.sstate .head{display:flex;gap:6px 12px;flex-wrap:wrap;align-items:center;font-size:14px}.sstate .head .chip{font-size:14px;padding:2px 12px;font-weight:600}.sstate ul{margin:4px 0 2px 18px;padding:0}tr.adj td{background:rgba(183,121,31,.12)}.optc{display:flex;gap:4px;align-items:center}.optf{padding:5px 8px;border:1px dashed var(--warn);border-radius:7px;color:var(--fg);font-size:13px}.optsum{margin-top:6px;font-size:13px;line-height:1.6}.optsum .warn{color:var(--warn);font-weight:600}.rcbox{margin-top:6px;padding:8px 10px;border:1px dashed var(--line);border-radius:8px}td.cons .llm{margin-top:4px}#dRcRules{margin-top:8px;max-height:420px;overflow:auto;font-size:13px}#dRcRules .row{margin:4px 0}
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
  <label>생성 순서<select id="gOrder"><option value="">기본 (ET → Talos → 사용자 룰)</option><option value="custom">사용자 룰 먼저</option></select></label>
 </div>
 <div class="muted" id="gOrderInfo" style="font-size:12.5px;margin-top:6px"></div>
 <div class="est" id="genEst"></div>
 <details id="gOvrBox"><summary>이번 실행만 다른 모델 <span class="muted" id="gOvr">(저장 안 됨 · 비우면 아래 '모델 설정'의 저장된 값)</span></summary><div class="row" style="margin-top:6px">
  <label>생성 모델<input id="gGen" placeholder="예: upstage:solar-pro2"></label>
  <label>검토 모델<input id="gRev" placeholder="예: claude_cli:opus"></label></div></details>
 <div class="row" style="margin-top:12px">
  <button id="bDry">건수 확인 (호출 없음)</button><button id="bPilot">파일럿 5개</button>
  <button class="primary" id="bGen">생성 시작</button><button id="bStop">중지 (지금 요청까지)</button><button class="danger" id="bKill">강제 중지</button><span class="muted prog" id="genProg" style="display:none"></span></div>
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
  <label>검색 (룰 id·이름·유형·제외 사유)<input id="rQ" style="width:260px"></label><span class="muted" id="rCount"></span></div>
 <div style="overflow-x:auto"><table style="margin-top:10px"><thead><tr><th>룰 id</th><th>상태</th><th title="'제외' 룰만: 생성 제외 유형(정보성 등) · 생성 대상 방향 아님(HTTP 응답) · LLM_설정.md '생성 제외 룰' 의 메모 (3.5 가 남김)">제외 사유</th><th>공격 유형</th><th>룰 이름</th><th>날짜 폴더 / 요청</th><th class="n">anomalies</th><th class="n">normal</th><th class="n">기존</th></tr></thead><tbody id="rBody"></tbody></table></div>
 <div class="pager"><button id="rPrev">이전</button><button id="rNext">다음</button></div></div></section>

<!-- 4. 신규 학습 데이터 -->
<section class="page" id="p-data"><div class="card"><h2>신규 학습 데이터 <span class="muted" style="font-weight:400" id="dSum"></span></h2>
 <div class="vstale" id="dStale" style="display:none"></div>
 <div class="row">
  <label>날짜<select id="dDate"></select></label>
  <label>장비<select id="dDev"><option value="ips">IPS</option><option value="waf">WAF</option><option value="fw">방화벽(FW)</option><option value="web">웹 로그</option></select></label>
  <label>label<select id="dLabel"><option value="">전체</option><option>anomalies</option><option>normal</option></select></label>
  <label>룰 id<input id="dRule" placeholder="예: et:1:2001044" style="width:150px"></label>
  <label>payload 검색<input id="dQ" style="width:200px"></label>
  <label>label 정합성<select id="dCons"><option value="">전체</option><option value="OK">정합</option><option value="RULE">정합 (룰 기준)</option><option value="CHECK">확인 필요</option><option value="BAD">불일치</option><option value="VOTED">합의로 label 바꿈</option></select></label>
  <label>보류 대상<select id="dHold"><option value="">전체</option><option value="rule">'정합 (룰 기준)' 공격</option><option value="pyua">python-requests UA 정상</option></select></label>
  <label>LLM 재확인<select id="dLlm"><option value="">전체</option><option value="todo">불일치 중 아직 안 함</option><option value="ok">label 맞음</option><option value="flip">label 반대</option><option value="unfit">학습에 부적합</option><option value="unknown">판단 불가</option><option value="any">재확인한 것 전부</option><option value="c_keep">합의: 확인됨 (유지)</option><option value="c_delete">합의: 부적합 삭제</option><option value="c_review">합의: 추가 분석</option><option value="c_flip">합의: label 반대 (투표·사람)</option></select></label>
  <button id="dGo">조회</button><button id="dVerifyNew">검증 안 한 데이터만 검증</button><button id="dVerify">전체 검증</button><button class="danger" id="dVerifyStop" style="display:none">검증 중지</button><button id="dSave" style="display:none" title="고른 날짜(전체면 모든 날짜)의 데이터 전체를 08_dataset/&lt;날짜&gt;/ 에 로그 + label 파일로 저장">파일로 저장</button><a id="dDown" style="font-size:13px;align-self:center">CSV 내려받기</a>
 </div>
 <div class="vsum" id="dVsum"></div><div class="vsum" id="dSaveSum" style="display:none"></div>
 <div class="vsum" id="dEdits"></div><div id="dEditList" style="display:none"></div>
 <div class="vsum" id="dRc"></div><div class="vsum" id="dRcR"></div><div id="dRcRules" style="display:none"></div><div id="dRcX" style="display:none"></div><div id="dRcOld" style="display:none"></div>
 <div class="rcbox" id="dRcBox" style="display:none"><div class="row">
  <label>대상<select id="rcTarget"><option value="bad">불일치 (전부)</option><option value="rule">정합 (룰 기준) - 룰당 표본</option></select></label>
  <label>날짜<select id="rcDate"></select></label>
  <label>장비<select id="rcDev"><option value="all">두 장비 (같은 샘플은 한 번)</option><option value="cur" id="rcDevCur"></option></select></label>
  <span class="optc" id="rcSample" hidden><label class="inl"><input type="checkbox" id="rcLlmOnly" checked> LLM 원본만 (증강 제외)</label>
   <label>룰당<span class="optc"><select id="rcPer"><option>1</option><option selected>3</option><option>5</option><option value="custom">직접 입력</option></select><input id="rcPerN" type="number" min="1" max="1000" step="1" value="10" style="width:64px" hidden></span></label></span></div>
 <div class="row">
  <label>방식<select id="rcMode"><option value="">새로 재확인</option><option value="cross">교차 확인 (1번째 결과를 2번째 모델로)</option><option value="tiebreak">3차 모델 투표 (합의가 '투표 갈림'인 줄)</option></select></label>
  <label id="rcFirstL" hidden>1번째 결과 <span class="muted">(이 모델이 맞음·판단 불가라 한 줄만)</span><select id="rcFirst"></select></label>
  <label><span id="rcModelL">모델 <span class="muted">(이번 실행만 · 비우면 검토 모델)</span></span><input id="rcModel" list="rcModels" style="width:230px"><datalist id="rcModels"></datalist></label>
  <label>추론 강도<select id="rcEff"></select></label>
  <button id="rcPlan">대상·예상 보기 (호출 없음)</button><button class="primary" id="rcGo">재확인 시작</button>
  <button id="rcStop" style="display:none">중지 (지금 요청까지)</button><button class="danger" id="rcKill" style="display:none">강제 중지</button><span class="muted prog" id="rcProg" style="display:none"></span></div>
 <div class="row" id="rcPickRow" hidden>
  <label>대상 룰<select id="rcPickSel"><option value="mid">애매한 룰만 (1번째 모델의 룰별 삭제 비율로)</option><option value="all">전부</option></select></label>
  <span class="optc" id="rcPickOpt"><label>삭제 비율<span class="optc"><input id="rcLo" type="number" min="0" max="99" value="30" style="width:52px">% 이상 ~<input id="rcHi" type="number" min="1" max="100" value="70" style="width:52px">% 미만</span></label>
  <label>판정<input id="rcMinR" type="number" min="1" max="1000" value="2" style="width:48px">줄 이상</label>
  <label>판정이 적은 룰<select id="rcSmall"><option value="include">교차 확인에 넣음</option><option value="exclude">뺌</option></select></label></span></div>
  <div class="est" id="rcEst"><span class="muted">고른 대상의 줄을 고른 LLM 에 보내 label 이 맞는지 다시 묻습니다 (API 비용 또는 Claude 구독 사용량). '불일치'는 줄 전부, '정합 (룰 기준)'은 룰마다 고른 수만큼(표본, 같은 조건이면 같은 줄)입니다. '대상·예상 보기'는 호출하지 않고 대상 줄·요청 수·예상 토큰·시간·비용만 셉니다. 시작하면 별도 프로세스로 돌아 화면을 다시 켜도 계속되고, 판정은 표에 코드 판정과 따로 보입니다 (학습 로그·검증 결과는 바꾸지 않음).</span></div></div>
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
  <button id="fCount">켜진 줄 수 다시 계산 (표준·신규 피처)</button><span class="muted" style="align-self:center;font-size:12.5px" id="fCountMsg"></span>
 </div>
 <div style="margin-top:10px;overflow-x:auto"><table><thead><tr><th>피처</th><th>이름</th><th>무엇을 보는지</th><th>예</th><th>값</th><th class="n">켜진 줄<br>anomalies</th><th class="n">켜진 줄<br>normal</th><th class="n">켜졌을 때<br>anomalies 비율</th><th>패턴 (계산에 쓰는 그대로)</th></tr></thead><tbody id="fBody"></tbody></table></div>
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
 <div id="sState" class="sstate">불러오는 중…</div>
 <details><summary>선별 방법 설명</summary><div class="muted" style="font-size:12.5px"><b>IPS·WAF 에 같은 샘플</b>을 넣습니다: 두 장비 비교 결과를 샘플로 짝지어 <b>합친 등급</b>으로 고릅니다 (A = 어느 장비든 A · B = A 가 아니고 어느 장비든 B · C = 그 밖 · 제거 = 두 장비 모두 중복, 한 장비만 기준과 중복이면 다른 장비 등급). 패턴당 상한은 장비마다 그 장비의 피처 패턴으로 세어 고른 뒤 합칩니다. 기본값: <b>A 전부 + B·C 패턴당 20개</b> (패턴이 넘치면 룰을 돌아가며 고름). 바꾸면 바로 다시 셉니다. 룰별 표에서 룰 단위로 빼거나 넣고, '후보 CSV 받기'로 payload 를 확인한 뒤 담으세요. 확정은 아래 버전 표에서 합니다.<br>
 위 상자는 지금 화면의 선별을 <b>불러온·방금 담은 초안</b>(없으면 기본값)과 비교합니다. 담기 전에 바꾼 것은 이 브라우저에 보관돼 새로 고치거나 화면을 다시 켜도 이어서 보입니다.</div></details>
 <div class="row" style="margin-top:8px">
  <label>A 부족한 데이터<span class="optc"><select id="oA"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option><option value="custom">패턴당 직접 입력</option></select><input id="oAn" type="number" min="0" max="100000" step="1" style="width:72px" hidden></span></label>
  <label>B 새 패턴<span class="optc"><select id="oB"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option><option value="custom">패턴당 직접 입력</option></select><input id="oBn" type="number" min="0" max="100000" step="1" style="width:72px" hidden><span class="optf" id="oBf" hidden>목표 합계로 채움</span></span></label>
  <label>C 이미 학습됨<span class="optc"><select id="oC"><option value="all">전부</option><option value="50">패턴당 50</option><option value="20">패턴당 20</option><option value="0">빼기</option><option value="custom">패턴당 직접 입력</option></select><input id="oCn" type="number" min="0" max="100000" step="1" style="width:72px" hidden></span></label>
  <label>목표 합계 <span class="muted">(비우면 안 씀 · B 로 채움)</span><input id="oT" type="number" min="1" step="1" placeholder="예: 200000" style="width:130px"></label>
  <label class="inl"><input type="checkbox" id="oBad"> '불일치' 빼기</label>
  <label class="inl" title="4. 신규 학습 데이터의 'LLM 재확인' 합의가 확인됨(두 모델 이상 모두 맞음 또는 유지 표시·사람 판단)인 '불일치' 샘플은 빼지 않음 - 한 모델만 맞음이면 뺌 (2026-11-02~, 비교를 다시 해야 최근 재확인 결과가 들어감)"><input type="checkbox" id="oLlm"> └ LLM 합의가 확인됨인 것은 남기기 <span class="muted" id="oLlmN"></span></label>
  <label class="inl"><input type="checkbox" id="oRule"> '정합 (룰 기준)' 공격 빼기 <span class="muted">(보류)</span></label>
  <label class="inl"><input type="checkbox" id="oPy"> python-requests UA 정상 빼기 <span class="muted">(보류)</span></label>
 </div>
 <div class="optsum" id="oSum"></div>
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
 <div class="muted" style="font-size:12.5px;margin-top:6px">흐름: 초안(담기) → <b>1차 확정</b>(후보 rev 1, 학습 파일 내보내기 - 아직 기준이 될 수 없음) → <b>피처 테스트</b>(precision·recall·F1, 기준 버전만으로 학습한 결과와 같은 시험 세트에서) → 필요하면 <b>데이터 추가</b>(다시 내보내면 rev+1) → 목표를 넘으면 <b>최종 승인</b>(확정·잠금, 다음 버전의 기준)</div>
 <details style="margin-top:10px"><summary>기준 버전 등록 (기존 학습데이터 · 넣을 것)</summary><div id="sReg"></div></details></div>
<div class="card" id="sCandCard" style="display:none"><h2>후보 버전 <span class="muted" style="font-weight:400" id="sCandHdr"></span><button class="sm" id="sCandClose" style="float:right">닫기</button></h2><div id="sCandBody"></div><div id="sSrvBody"></div><div id="sCandFoot"></div></div>
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
 $("genStatus").textContent=rdo.starting&&!g.running?"다시 만들기 시작하는 중… (다시 검토할 요청을 찾는 중, 1분쯤)":g.running&&redo?`다시 만들기 실행 중 (검토에서 뺀 샘플 다시 검토) · PID ${g.lock.pid} · ${String(g.lock.started_at||"").replace("T"," ")} 시작 · 멈추려면 '강제 중지'`:g.running?(g.starting?"시작하는 중…":`실행 중 · 단계 ${g.lock.step||"-"} · 입력 ${g.lock.input_date||"-"}${g.lock.custom_first?" · 사용자 룰 먼저":""} · PID ${g.lock.pid} · ${String(g.lock.started_at||"").replace("T"," ")} 시작`+(g.stop_requested?" · 중지 요청됨":"")):(c.running||(a.running&&(a.status||{}).state==="수집 중")?"대기 · 지금 수집 중이지만 생성을 함께 시작할 수 있습니다 (시작할 때의 입력 날짜를 끝까지 씀)":"대기");
 for(const id of["bDry","bPilot","bGen"])$(id).disabled=g.running||rdo.starting;$("bStop").disabled=!g.running||redo;$("bKill").disabled=!g.running;
 {const r=g.run,G=(r&&r.gen)||{},R=(r&&r.rev)||{};   // 강제 중지 옆 진행률 (서버가 로그로 계산한 run, 실행 중일 때만)
  progSet("genProg",g.running&&r&&r.state==="running"&&!r.error?`진행 ${r.pct}%`+(r.dry_run?"":` · 생성 ${fmt((G.ok||0)+(G.fail||0))}/${fmt(G.target||0)}${G.fail?` (실패 ${fmt(G.fail)})`:""}`
   +(R.target?` · 검토 ${fmt((R.ok||0)+(R.fail||0))}/${fmt(R.target)}${R.fail?` (실패 ${fmt(R.fail)})`:""}`:""))+(r.remaining_s!=null?` · 남은 약 ${mins(r.remaining_s)}`:""):"")}
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
function mins(s){s=Math.max(0,Math.round(s||0));const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?`${h}시간 ${m}분`:s>=60?`${Math.round(s/60)}분`:`${s}초`}   // 강제 중지 옆 진행률의 남은 시간 (분 단위)
/* 강제 중지 버튼 옆 진행률 (2026-11-04~): 실행 중일 때만 보이고, 끝나면 숨김 (결과는 원래 자리에) */
function progSet(id,txt){const el=$(id);if(!el)return;el.textContent=txt||"";el.style.display=txt?"":"none"}
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
  <div style="font-size:13px">${head}</div><div class="muted" style="font-size:12.5px">${esc(r.started_at)} 시작${r.dry_run?" · 건수 확인 (dry-run, 호출 없음)":""}${r.custom_first?" · <b>사용자 룰 먼저</b>":""}${r.by!=="3_0"?(r.by.includes("redo")?" · 다시 만들기 (검토에서 뺀 샘플 다시 검토)":` · ${esc(r.by)} 단독 실행`):""}${r.stop_requested?" · <b>중지 요청됨</b> (지금 요청까지 하고 멈춤)":""}</div></div></div>
  <div class="steps">${r.steps.map(s=>`<span class="step ${s.status}" title="${word[s.status]}">${icon[s.status]} ${s.key} ${esc(s.name)}${s.detail?` <b>${s.detail}</b>`:""}</span>`).join("")}</div>
  ${live&&r.current?`<div class="runnow">지금: <b>${esc(r.current)}</b></div>`:""}
  ${r.dry_run?"":`<div class="kv" style="margin-top:6px"><b>생성</b><span>${fmt(G.ok)}개 성공 · ${fmt(G.fail)}개 실패 / 이번 ${fmt(G.target)}개${G.ok+G.fail?` · 약 ${hms(G.sec_per_req)}마다 1개 완료 · 토큰 ${fmt(G.tokens_in+G.tokens_out)} (입력 ${fmt(G.tokens_in)} / 출력 ${fmt(G.tokens_out)})`:""}</span>
  <b>검토</b><span>${revWait?`예정 ${fmt(R.target)}개 (생성 성공분 + 이전 대기분)`:`${fmt(R.ok)}개 성공 · ${fmt(R.fail)}개 실패 / ${fmt(R.target)}개${R.ok+R.fail?` · 약 ${hms(R.sec_per_req)}마다 1개 완료 · 토큰 ${fmt(R.tokens_in+R.tokens_out)}`:""}`}</span>
  ${G.check_ok+G.check_fail?`<b>생성 품질</b><span>코드 검사 통과 <b>${(100*G.check_ok/(G.check_ok+G.check_fail)).toFixed(1)}%</b> (샘플 ${fmt(G.check_ok)} / ${fmt(G.check_ok+G.check_fail)})${R.ok?` · 검토가 고친 샘플 요청당 <b>${(R.fixed/R.ok).toFixed(1)}개</b>, 뺀 샘플 ${(R.dropped/R.ok).toFixed(1)}개`:""} <span class="muted">(정규식 예시·요청 줄 자동 수정 전 100개 실행: 통과 69.5% · 고친 샘플 요청당 11.6개)</span></span>`:""}
  ${r.parallel?`<b>동시 처리</b><span>생성 ${live?`지금 ${r.parallel.generate_now}개 (최대 ${r.parallel.generate})`:`최대 ${r.parallel.generate}개`} · 검토 ${live?`지금 ${r.parallel.review_now}개 (최대 ${r.parallel.review})`:`최대 ${r.parallel.review}개`}${r.parallel.overlap?" · 생성된 요청부터 바로 검토":""}${live?" · 속도 제한에 맞춰 자동 조절":""}</span>`:""}</div>
  <div class="muted" style="font-size:12px;margin-top:4px">%·남은 시간은 이번 실행에서 실제로 요청이 끝나는 간격(동시 처리 반영)과 지난 실행의 준비·마무리 단계 시간으로 계산합니다 (응답 해석 실패·재시도가 많으면 늘어남). 토큰은 성공한 요청 기준.</div>`}`}
function genBody(extra){return Object.assign({set:$("gSet").value,max_requests:+$("gN").value,date:$("gDate").value,order:$("gOrder").value,generator:$("gGen").value.trim(),reviewer:$("gRev").value.trim()},extra)}
/* 생성 순서: 사용자 룰(custom)이 든 대기 요청 (서버가 가장 최근 입력의 요청 파일에서 셈). llm(k) = 요청 k개 생성·검토 시간(초), 기록이 없으면 null */
function orderHint(llm){const el=$("gOrderInfo"),cr=S&&S.custom_requests,first=$("gOrder").value==="custom";
 if(cr&&cr.error){el.textContent="사용자 룰 요청을 세지 못했습니다: "+cr.error;return}
 if(!cr||!cr.date){el.textContent=S&&cr===null?"사용자 룰 요청 세는 중…":"";return}     // 입력 날짜가 없으면 {}
 const set=$("gSet").value,c=cr.by_set[set],day=$("gDate").value,other=day&&day!==cr.date?` <span class="warn">(가장 최근 입력 ${esc(cr.date)} 기준이라 고른 입력 날짜와 다를 수 있음)</span>`:"";
 // 사용자 룰 = 완료 + (a) 대기 요청에 든 룰 + (b) 생성 응답에서 빠져 다음 실행의 3.1 이 새 요청으로 다시 넣을 룰
 //          + (c) 자동으로 다시 만들지 않는 룰 (샘플 없음·생성 실패) (+ 검토·증강 대기) - 화면 1·3 과 같은 기준 (3.5 rule_status)
 const fl=cr.flags||{},FN=(P&&P.flag_names)||{},tot=Object.values(fl).reduce((a,b)=>a+b,0),rt=(set==="all"&&cr.retry)||{rules:[],requests:0},na=cr.noauto||{};
 const nb=rt.rules.length,nc=Object.values(na).reduce((a,x)=>a+x.length,0);
 const head=tot?`사용자 룰(custom) <b>${fmt(tot)}개</b> <span class="muted">(화면 1·3 기준: ${Object.entries(fl).sort((a,b)=>b[1]-a[1]).map(([k,n])=>`${esc(FN[k]||k)} ${fmt(n)}`).join(" · ")})</span><br>`:"";
 if(!c){el.innerHTML=head+`입력 ${esc(cr.date)} 의 이 세트 요청 파일이 아직 없습니다 - 다음 생성의 준비 단계(3.1)가 만든 뒤 고른 순서대로 처리${other}`;return}
 const abc=`(a) 대기 요청에 든 룰 <b>${fmt(c.rules)}개</b>${c.custom?` (요청 ${fmt(c.custom)}개)`:""} · (b) 다음 실행 때 새 요청으로 다시 넣을 룰 <b>${fmt(nb)}개</b>`+
  (nb?` <span class="muted">(생성 응답에서 빠짐 - 준비 단계 3.1 이 요청 약 ${fmt(rt.requests)}개로 다시 만듦)</span>`:"")+` · (c) 자동으로 다시 만들지 않는 룰 <b>${fmt(nc)}개</b>`+
  (nc?` <span class="muted">(${Object.entries(na).map(([k,x])=>`${esc(FN[k]||k)} ${x.length}`).join(" · ")})</span> <a href="#/rules" data-r3="${esc(Object.keys(na)[0])}">화면 3 에서 보기</a>`:"");
 const nreq=c.custom+rt.requests;                 // (a)+(b) 를 만드는 요청 수
 if(!nreq){el.innerHTML=head+abc+` · 다음 생성에서 새로 만들 사용자 룰은 없음`+other;return}
 // 요청 수에는 이전 실행의 검토 대기분이 들어가 그만큼 덜 생성한다 (3_0 --max-requests) → (a)+(b) 전부 = 그 요청 + 검토 대기
 const asked=+$("gN").value||0,rw=cr.review_wait||0,need=nreq+rw,t=llm&&c.custom&&c.ahead?llm(c.ahead):0,why=rw?` (요청 수에 이전 검토 대기 ${fmt(rw)}개 포함)`:"";
 el.innerHTML=head+abc+"<br>"+(!first?
  (c.custom&&!c.ahead?"기본 순서로도 (a) 가 맨 앞":`기본 순서로는 ${c.custom?`(a) 가 앞의 요청 ${fmt(c.ahead)}개 다음${t?` (그 요청들 생성·검토 ${dur(t)})`:""}`:"배치 순서상 뒤쪽"}`)+` · '사용자 룰 먼저'로 요청 수 ${fmt(need)} 이상이면 (a)+(b) 를 한 번에${why}`
  :asked&&asked<need?`<span class="warn">요청 수 ${fmt(asked)}개면 이번에는 사용자 룰 요청 중 ${fmt(Math.max(0,asked-rw))}개만</span> · (a)+(b) 를 모두 하려면 ${fmt(need)} 이상${why}`
  :`<b>사용자 룰 요청부터</b> 만들고 (${nb?"(b) 는 준비 단계가 새로 만든 요청을 맨 앞으로 · ":""}${asked===need?"그다음 끝냄":asked?"남는 수만큼 기본 순서로 이어감":"이어서 남은 요청 전부"})`)+other}
function goRules(flag){$("rSrc").value="custom";const f=$("rFlag");if(![...f.options].some(o=>o.value===flag))f.add(new Option(flag,flag));   // 화면 3 을 사용자 룰·그 상태로 걸러 열기
 f.value=flag;$("rQ").value="";rOff=0;location.hash="#/rules"}
$("gOrderInfo").addEventListener("click",e=>{const a=e.target.closest("a[data-r3]");if(!a)return;e.preventDefault();goRules(a.dataset.r3)});
function dur(sec){if(sec<3600)return `약 ${Math.max(1,Math.round(sec/60))}분`;const h=sec/3600;return h<48?`약 ${h.toFixed(1)}시간`:`약 ${(h/24).toFixed(1)}일 (${fmt(Math.round(h))}시간)`}
function estimate(){const el=$("genEst");if(!P){el.textContent="예상: 진행 현황을 불러오는 중…";orderHint(null);return}
 const av=P.average_per_request||{},rq=P.requests||{},set=$("gSet").value;let left=((rq.pending_by_set||{})[set])??rq.pending??0;
 if(!left&&set==="all")left=(P.estimate_remaining||{}).requests||0;   // 새 입력 날짜라 요청 파일이 아직 없으면 (다음 실행의 3.1 이 만듦) 남은 룰 수로 추정
 const asked=+$("gN").value||0,all=asked===0,n=all?left:Math.min(asked,left);
 if(!av.generate_seconds){el.innerHTML="<b>예상</b> 생성 기록이 아직 없어 계산할 수 없습니다 — '파일럿 5개'를 먼저 돌리면 실제 평균으로 계산합니다.";orderHint(null);return}
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
 const llm=k=>{const tg=Math.ceil(k/par.generate)*g1,tr=Math.ceil(k/par.review)*r1;return !k?0:par.overlap?Math.max(tg+r1,g1+tr):tg+tr},sec=llm(n);orderHint(llm);
 // 준비·마무리: 지난 실행 기록 (증강은 요청 수에 비례). 아직 생성을 돌리지 않은 입력 날짜면 기존 로그 대조·요청 파일 만들기로 준비가 길다
 const ss=(S&&S.step_seconds)||{},day=$("gDate").value||((S&&S.input_dates)||[]).slice(-1)[0],first=(ss.first_dates||[]).includes(day);
 const prep=!n?0:first?(ss.prep_first??1240):(ss.prep??60),fin=!n?0:(ss.augment_per_request??0.6)*n+(ss.post??40);
 const gi=Math.round(n*(hg?hg.tin/hg.n:av.generator_input_tokens||0)),go=Math.round(n*(hg?hg.tout/hg.n:av.generator_output_tokens||0)),ri=noRev?0:Math.round(n*(hr?hr.tin/hr.n:av.reviewer_input_tokens||0)),ro=noRev?0:Math.round(n*(hr?hr.tout/hr.n:av.reviewer_output_tokens||0));
 el.innerHTML=`<b>예상</b>${ov.g||ov.r?' <span class="warn">(이번 실행만 바꾼 모델)</span>':""} 요청 ${fmt(n)}개${all?" (남은 전부)":""} → 시간 <b>${dur(sec+prep+fin)}</b> <span class="muted">(생성·검토 ${dur(sec)} + 준비 ${dur(prep)}${first?" (이 입력 날짜의 첫 실행 - 기존 데이터 대조·요청 파일 만들기)":""} + 마무리 ${dur(fin)} · 동시 최대 생성 ${par.generate}개·검토 ${par.review}개${par.overlap?" · 생성·검토 함께":""}, 하나씩이면 ${dur(n*(g1+r1))} · API 속도 제한에 걸리면 더 걸림)</span><br>토큰 합계 <b>${fmt(gi+go+ri+ro)}</b> = 생성 ${esc(gm)} <b>${fmt(gi+go)}</b> (입력 ${fmt(gi)} / 출력 ${fmt(go)}) + 검토 ${esc(rm)} <b>${fmt(ri+ro)}</b> (입력 ${fmt(ri)} / 출력 ${fmt(ro)})<div class="muted">기준: 생성 ${hg?`${esc(gm)} ${fmt(hg.n)}건`:`전체 ${av.generated_requests}건`} · 검토 ${noRev?"생략":hr?`${esc(rm)} ${fmt(hr.n)}건`:`전체 ${av.reviewed_requests}건`}의 실제 평균 (재시도 포함, 요청 1개 생성 ${Math.round(g1)}초·검토 ${Math.round(r1)}초) · 이 세트의 남은 요청 ${fmt(left)}개${!all&&asked>left?" (입력한 수가 남은 요청보다 많아 남은 수로 계산)":""}</div>${warn.length?`<div class="warn">${warn.join("<br>")}</div>`:""}`}
$("gN").oninput=estimate;$("gSet").onchange=$("gOrder").onchange=$("gDate").onchange=estimate;$("gGen").oninput=$("gRev").oninput=estimate;
async function act(msgId,fn){try{const r=await fn();$(msgId).textContent="요청함 "+(r.pid?`(PID ${r.pid})`:"");setTimeout(loadState,800)}catch(e){$(msgId).textContent="실패: "+e.message}}
$("bDry").onclick=()=>act("genMsg",()=>post("/api/generation/start",genBody({dry_run:true})));
$("bPilot").onclick=()=>act("genMsg",()=>post("/api/generation/start",genBody({max_requests:5})));
$("bGen").onclick=()=>{const n=+$("gN").value,o=$("gOrder").value==="custom"?" (사용자 룰 먼저)":"";if(!confirm(n===0?`남은 요청 전부를 생성합니다${o} (API 비용·수일 소요). 진행할까요?`:`요청 ${n}개를 생성합니다${o} (API 비용 발생). 진행할까요?`))return;act("genMsg",()=>post("/api/generation/start",genBody()))};
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
function chgHtml(c){    // 바로 앞 입력 날짜와 비교한 룰 변화 (수집으로 룰이 몇 개 바뀌었는지)
 if(!c||c.error)return"";const src=o=>Object.entries(o).map(([k,v])=>`${k} ${fmt(v)}`).join(" · ");
 return `<div class="muted" style="margin-top:6px">입력 ${c.cur} ← ${c.prev}: 새 룰 ${fmt(c.added.total)}${c.added.total?` (${src(c.added.by_source)})`:""} · 빠진 룰 ${fmt(c.removed.total)}${c.removed.total?` (${src(c.removed.by_source)}${c.removed.duplicate?`, 중복 제거 ${fmt(c.removed.duplicate)}`:""})`:""} · 내용 바뀐 룰 ${fmt(c.changed)} → 현재 룰 ${c.net>0?"+":""}${fmt(c.net)}</div>`}
async function refreshProgressWait(){    // 다시 계산을 시작하고 끝날 때까지(15초 안팎) 기다렸다가 새로 불러옴
 await post("/api/progress/refresh");
 for(let i=0;i<40;i++){await new Promise(ok=>setTimeout(ok,3000));if(!(await get("/api/progress")).computing)break}
 await loadProgress();if(page==="rules")loadRules()}
async function loadProgress(){const r=await get("/api/progress");P=r.data;
 if(!P){$("progBody").innerHTML=r.computing?"계산 중…":`아직 계산 결과가 없습니다. '다시 계산'을 누르세요.${r.error?"<br>"+esc(r.error):""}`;return}
 flagNames=P.flag_names;const fl=P.flags,fr=P.frontier,rq=P.requests,est=P.estimate_remaining,av=P.average_per_request;
 const notYet=`입력 ${P.input_date} 로는 아직 생성 전`;    // 새 입력 날짜: 요청 파일은 다음 생성 때 이 날짜 폴더에 새로 만들어짐
 if(page==="run")estimate();
 const fsel=$("rFlag");if(fsel.options.length===1)fsel.innerHTML+=Object.keys(flagNames).map(k=>`<option value="${k}">${esc(flagNames[k])} (${k})</option>`).join("");
 if(page!=="progress")return;
 const srcRows=Object.entries(P.by_source).map(([k,v])=>`<tr><td>${k}</td><td class="n">${fmt(v.target)}</td><td class="n">${fmt(v.done)}</td><td class="n">${v.pct}%</td></tr>`).join("");
 const typeRows=Object.entries(P.by_attack_type).filter(([k,v])=>v.target>0).slice(0,12).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${fmt(v.target)}</td><td class="n">${fmt(v.done)}</td><td class="n">${v.pct}%</td></tr>`).join("");
 $("progBody").innerHTML=`
 <div class="row" style="gap:40px;align-items:flex-start">
  <div style="min-width:300px;flex:1"><div class="muted">생성 대상 룰 중 완료</div><div class="big">${P.pct_done_of_target}%</div>${bar(P.pct_done_of_target)}
   <div>${fmt(P.target)}개 중 <b>${fmt(P.done)}</b>개 완료 · 샘플 anomalies ${fmt(P.samples.anomalies)} / normal ${fmt(P.samples.normal)}</div>
   <div class="muted" style="margin-top:6px">현재 룰 ${fmt(P.rules)}개 중 학습데이터 보유 <b>${P.pct_with_data_of_all}%</b> (완료 ${fmt(fl.DONE)} + 기존 충분 ${fmt(fl.EXISTING)})</div>${chgHtml(r.change)}
   <div style="margin-top:12px" class="kv"><b>어디까지</b><span>${!rq.total?notYet:fr.batches_total?`배치 ${fr.contiguous_done_batch} / ${fr.batches_total} 까지 빠짐없이 생성 (가장 뒤 배치 ${fr.last_touched_batch})`:"-"}</span>
    <b>요청</b><span>${rq.total?`${fmt(rq.total)}개 중 생성 ${fmt(rq.generated)} · 검토 ${fmt(rq.reviewed)} · 실패 ${fmt(rq.failed)} · 대기 ${fmt(rq.pending)}`:`${notYet} · 다음 생성 때 미생성 ${fmt(fl.TODO)}룰을 요청 약 ${fmt(est.requests)}개로 새로 만듦`}</span>
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
 $("rBody").innerHTML=r.rows.map(x=>`<tr><td>${esc(x.id)}</td><td>${chip(x.flag)}</td><td class="muted">${x.flag==="EXCLUDED"?esc(x.reason||""):""}</td><td>${esc(x.attack_type)}</td><td>${esc(x.rule_name)}</td><td class="muted">${esc(x.date)} ${esc(x.request)}${x.requests>1?` <span title="샘플 수는 이 룰의 모든 요청(샘플 있는 ${x.requests}개)을 합친 수">외 ${x.requests-1}개</span>`:""}</td><td class="n">${x.anomalies||""}</td><td class="n">${x.normal||""}</td><td class="n">${x.existing||""}</td></tr>`).join("");
 $("rPrev").disabled=rOff===0;$("rNext").disabled=rOff+50>=r.total;}
let tm;$("rQ").oninput=()=>{clearTimeout(tm);tm=setTimeout(()=>{rOff=0;loadRules()},300)};$("rFlag").onchange=$("rSrc").onchange=()=>{rOff=0;loadRules()};
$("rPrev").onclick=()=>{rOff=Math.max(0,rOff-50);loadRules()};$("rNext").onclick=()=>{rOff+=50;loadRules()};
/* ── 4. 신규 학습 데이터 (표만, 로그 형식 파일은 6. 에서 버전을 확정할 때 만듦) ── */
let dOff=0,dInit=false,D=[],dTot=0,dReady=false,dV=null,dVT=0,dRows=[],dE=null,dET=0,dERun=false,edX=null,dPending=null,dSkipped=[];
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
  $("dLabel").value=$("dCons").value=$("dQ").value=$("dLlm").value="";$("dHold").value=dPending.hold||"";dHoldUi();dOff=0;dPending=null}
 loadData();loadVerify();loadEdits();loadSave();loadRc();ensureFeat();loadDCands().then(()=>{if(dCands.length)loadData()})}   // 후보가 있으면 표에 '후보에' 버튼
function dHoldUi(){const h=$("dHold").value;for(const id of["dLabel","dCons","dLlm"]){$(id).disabled=!!h;if(h)$(id).value=""}}   // 보류 대상이 label·정합성을 정함
function dSummary(){const v=$("dDate").value,ds=v==="all"?D:D.filter(d=>d.date===v);const a=ds.reduce((s,d)=>s+(d.label.anomalies||0),0),n=ds.reduce((s,d)=>s+(d.label.normal||0),0);
 $("dSum").textContent=`· ${v==="all"?"전체":v} · ${fmt(a+n)}건 (anomalies ${fmt(a)} / normal ${fmt(n)}) · 룰 ${fmt(ds.reduce((s,d)=>s+d.rules,0))}개`;
 // 지금 조건에 맞는 모든 줄을 표의 열 그대로 CSV
 $("dDown").href="/download/frame?"+new URLSearchParams({date:v,dev:$("dDev").value,label:$("dLabel").value,rule:$("dRule").value.trim(),q:$("dQ").value.trim(),cons:$("dCons").value,hold:$("dHold").value,llm:$("dLlm").value});
 $("dHelp").innerHTML="<b>label 사유</b>: 어떤 룰의 어떤 조건을 만족·불일치했는지 + 생성 메모. <b>검증 결과</b>: 생성 때 검증(코드 검사·검토 모델·증강) + 지금 현재 룰로 다시 확인. <b>label 정합성</b>: 라벨이 증거와 맞는지 — 정합(룰 + 범용 공격 흔적·다른 공격 룰도 일치) / 정합(룰 기준: 이 룰의 고유 패턴으로만 공격) / 확인 필요 / 불일치. <b>탐지 피처</b>: ML 피처(4.1) 중 이 payload 에서 켜지는 것 — k_ 공격 흔적, ra_ 신뢰할 수 있는 공격 룰, r_ 이 룰·정규식. 피처 이름에 마우스를 올리면 설명이 나오고, 누르면 <a href=\"#/features\">5. 피처</a> 페이지에서 봅니다. <b>보류 대상</b>: 6. 선별의 보류 옵션에 걸리는 샘플만 — '정합 (룰 기준)' 공격(그날 '전체 검증'이 있어야 함) / python-requests UA 정상. <b>CSV 내려받기</b>: 지금 조건(날짜·장비·label·룰·검색·정합성·보류 대상)에 맞는 모든 줄을 이 표의 열 그대로 CSV(엑셀용, payload 는 로그와 같은 한 줄)로 받습니다. '전체 검증'을 해 두면 바로 받고, 아니면 줄마다 다시 확인해 1천 줄에 약 15초 걸립니다. 로그 형식(|||) 파일은 <a href=\"#/sup\">6. 기존 학습데이터와 비교</a>에서 버전을 확정할 때 기존 학습데이터와 합쳐 만듭니다. <b>LLM 재확인</b>: '불일치' 줄 전부 또는 '정합 (룰 기준)' 줄의 룰당 표본을 고른 LLM 에 다시 물은 판정(label 맞음·반대·학습에 부적합·판단 불가)과 근거 - 코드 판정과 따로 보이고, 반영은 위 버튼으로 수정·삭제를 저장합니다 (대상마다 따로 되돌림). '정합 (룰 기준)'의 룰별 요약에서 삭제가 많은 룰을 6. 선별의 '룰 조정 - 빼기'로 넘길 수 있습니다. <b>교차 확인·합의</b>: 재확인 상자의 '방식 - 교차 확인'은 1번째 모델이 부적합이 아니라고 한 줄을 다른 모델로 다시 묻고(부적합은 이미 삭제라 묻지 않음, 지시문은 같음), 줄마다 합의를 냅니다 (2026-11-04~) - 어느 모델이든 '학습에 부적합'이면 삭제, 그 밖에는 맞음·반대 다수결(두 모델 이상): 맞음 다수면 확인됨(유지), 반대 다수면 label 반대(label 을 바꾸고 투표 결과·근거를 사유로), 판정이 한 모델뿐이면 2번째 모델 대기, 갈리면 투표 갈림 → '3차 모델로 투표'(그 줄을 아직 판정하지 않은 모델로 다시 묻고 다수결, 그래도 갈리면 추가 분석에서 사람이 유지·삭제·label 반대). '정합 (룰 기준)' 줄의 label 반대는 룰에 걸리는 정상(또는 룰에 안 걸리는 공격)을 만듭니다. 합의로 label 을 바꿔 학습 로그에 반영한 줄은 검증에서 '불일치' 대신 '합의로 label 바꿈'으로 따로 셉니다. 6. 선별의 '불일치' 중 남기기 옵션은 합의가 확인됨인 것만 남깁니다.";if(dConv())$("dHelp").innerHTML=convHelp();}
function dVUrl(){return `/api/data/verify?date=${$("dDate").value}&dev=${dSrc()}${dConv()?"&view=conv":""}`}   // conv = FW·웹 (원본 검증, 장비 이름 없이)
async function loadVerify(){const v=$("dDate").value;if(!v)return;let s;try{s=await get(dVUrl())}catch(e){$("dVsum").textContent="";return}
 dV=s;dReady=s.status==="done"&&!!s.frame_ready;renderStale(s);
 const run=s.status==="running";$("dVerifyStop").style.display=run?"":"none";$("dVerifyNew").disabled=$("dVerify").disabled=run;
 if(run){const what=s.mode==="new"?"검증 안 한 데이터만 검증":"전체 검증",left=s.eta_s!=null?` · 남은 시간 <b>약 ${hms(s.eta_s)}</b> · <b>${clock(s.eta_s)}</b> 쯤 끝날 예상`:"";
  const body=s.phase==="wait"?`${what} 대기 - IPS 검증이 끝나기를 기다리는 중 (같은 샘플이라 끝나면 그 결과를 그대로 씀)${left}`
   :s.phase==="run"?`${what} 중… ${fmt(s.done)} / ${fmt(s.total)}줄${s.todo<s.total?` (새로 확인 ${fmt(s.analyzed)} / ${fmt(s.todo)}줄 · 이미 확인한 결과 그대로 ${fmt(s.reused)}줄)`:""}${left}`
   :`${what} 준비 중 - 이미 확인한 줄과 새로 확인할 줄을 세는 중…`;
  $("dVsum").innerHTML=`<span>${body}${s.stopping?" · <b>중지하는 중…</b>":""}</span>`;clearTimeout(dVT);dVT=setTimeout(loadVerify,1500);return}
 const est=`<span class="muted">검증 안 한 데이터만: 새로 확인 약 ${fmt(s.est_new)}줄${s.est_new?` (약 ${hms(s.est_new_s)})`:""} · 전체 검증: ${fmt(s.total)}줄 (약 ${hms(s.est_all_s)})</span>`;
 const stop=s.stopped?`<span class="muted">${s.stopped.at.slice(11,16)} 에 중지 (${esc(s.stopped.why)}) - 확인한 ${fmt(s.stopped.saved)}줄은 저장됨 → '검증 안 한 데이터만 검증' 으로 이어서</span>`:"";
 // 검증 결과를 쓸 수 없는 날짜 (검증 안 함·낡음·실패): 날짜 하나면 까닭과 할 일, '전체'면 날짜 목록 (2026-11-02~)
 const gaps=(s.gaps||[]).length?`<div class="vwarn">${v!=="all"?esc(s.gaps[0].text):`정합성·LLM 재확인 조건에서 빠지는 날짜 ${s.gaps.length}개 - ${s.gaps.map(g=>`${esc(g.date)} ${({stale:"낡음",none:"검증 안 함",error:"검증 실패"})[g.why]||esc(g.why)}`).join(" · ")} <span class="muted">(까닭은 위 안내 줄, 그 날짜를 고르면 자세히)</span>`}</div>`:"";
 if(s.status!=="done"&&!s.done_dates){$("dVsum").innerHTML=(s.status==="error"&&v==="all"?`<span class="muted">검증 오류: ${esc(s.error)}</span>`:gaps?"":`<span class="muted">검증 전 — 모든 줄의 label 정합성을 확인하면 정합성으로 걸러 볼 수 있습니다.</span>`)+gaps+stop+est;return}
 const tot=Object.values(s.summary).reduce((a,b)=>a+b,0);
 $("dVsum").innerHTML=`<b>label 정합성${dConv()?" (원본 payload 기준)":""} (${fmt(tot)}줄${s.status!=="done"?` · 검증한 날짜 ${s.done_dates}/${s.n_dates}`:""})</b>`+["OK","RULE","CHECK","BAD",...(s.summary.VOTED?["VOTED"]:[])].map(k=>`<span class="chip c-${k}" data-c="${k}">${esc(s.names[k])} ${fmt(s.summary[k]||0)} (${tot?(100*(s.summary[k]||0)/tot).toFixed(1):0}%)</span>`).join("")+`<span class="muted">${s.finished_at.replace("T"," ")} 검증${s.reused?` · 이미 확인한 결과 ${fmt(s.reused)}줄 그대로 씀`:""}${s.old_rules?` · ${fmt(s.old_rules)}줄은 이전 룰 입력(${esc(Object.keys(s.rule_inputs).filter(k=>k!==s.latest_input).sort().join(", "))}) 기준 - 최신 룰(${esc(s.latest_input)})로 다시 보려면 '전체 검증'`:""}</span>`+gaps+stop+(s.est_new?est:"");
 $("dVsum").querySelectorAll("[data-c]").forEach(c=>c.onclick=()=>{$("dHold").value="";dHoldUi();$("dCons").value=c.dataset.c;dOff=0;loadData()});renderEdits();}
/* 화면 4 위쪽 '검증이 낡은 날짜' (2026-11-02~): 검증한 뒤 학습 로그가 다시 만들어진 날짜·장비 + 그때 돌던 작업, '바뀐 날짜만 다시 검증',
   학습 로그를 다시 만드는 작업이 끝나면 자동으로 다시 검증 (이 화면 서버가 켜져 있는 동안) */
function renderStale(s){const el=$("dStale"),st=s.stale||[],w=s.watch||{},conv=dConv(),by={};
 for(const g of st){const k=g.date+"|"+g.log_hm+"|"+(g.by||"");(by[k]=by[k]||Object.assign({devs:[]},g)).devs.push(g.dev.toUpperCase())}
 let h=st.length?`<b>검증이 낡은 날짜</b> <span class="muted">(검증한 뒤 학습 로그가 다시 만들어져 정합성·LLM 재확인 조건에서 빠짐)</span>: `+Object.values(by).map(g=>`<span class="nw">${esc(g.date)}${conv?"":" "+g.devs.join("·")} - ${esc(g.log_hm)} 학습 로그 다시 만듦${g.by?` (${esc(g.by)})`:""}</span>`).join(" · ")+` <button class="sm" id="dStaleGo">바뀐 날짜만 다시 검증</button>`:"";
 if(w.on)h+=`<div>작업(${esc((w.jobs||[]).join("·")||"학습 로그 다시 만들기")})이 끝나면 바뀐 날짜만 자동으로 다시 검증합니다 <span class="muted">(${w.n}개 지켜보는 중 · 이 화면 서버가 켜져 있는 동안)</span> <button class="sm" id="dWatchOff">자동 다시 검증 취소</button></div>`;
 else if(w.fired)h+=`<div class="muted">${esc(isoClock(w.fired.at))} 작업이 끝나 ${(w.fired.started||[]).length?`바뀐 날짜를 자동으로 다시 검증: ${esc(w.fired.started.map(([d,v])=>d+(conv?"":" "+v.toUpperCase())).join(", "))}`:"자동으로 다시 검증할 날짜가 없었습니다"}${w.fired.error?" · 오류: "+esc(w.fired.error):""}</div>`;
 el.innerHTML=h;el.style.display=h?"":"none";
 if($("dStaleGo"))$("dStaleGo").onclick=verifyStale;
 if($("dWatchOff"))$("dWatchOff").onclick=async()=>{try{await post("/api/data/verify/watch",{on:false})}catch(e){}loadVerify()}}
async function dBusy(){try{dV=await get(dVUrl())}catch(e){}return (dV&&dV.busy)||[]}   // 지금 도는, 학습 로그를 다시 만드는 작업
function busyMsg(b){return `지금 학습 로그를 다시 만드는 작업(${b.join("·")})이 도는 중입니다. 끝날 때 학습 로그가 다시 만들어져 지금 검증한 결과 중 일부가 낡게 됩니다.`}
function watchMsg(b){return `작업(${b.join("·")})이 끝나면 학습 로그가 바뀐 날짜만 자동으로 다시 검증할까요? ('검증 안 한 데이터만 검증' - 이 화면 서버가 켜져 있는 동안)\n\n확인 = 끝나면 자동으로 다시 검증 · 취소 = 하지 않음 (나중에 위쪽 '검증이 낡은 날짜' 줄에서 직접)`}
async function verifyStale(){const b=await dBusy();let watch=false;
 if(b.length){if(!confirm(busyMsg(b)+"\n\n그래도 지금 바뀐 날짜를 다시 검증할까요?"))return;watch=confirm(watchMsg(b))}
 try{await post("/api/data/verify/stale",{devs:dConv()?["ips"]:["ips","waf"],watch});loadVerify()}catch(e){$("dVsum").textContent="실패: "+e.message}}
async function startVerify(mode,watch=false){try{const r=await post("/api/data/verify",{date:$("dDate").value,dev:dSrc(),mode,watch});
  if(!r.started.length&&r.done.length){$("dVsum").innerHTML='<span class="muted">새로 확인할 줄이 없습니다 - 이미 모두 검증했습니다.</span>';setTimeout(loadVerify,2500)}else loadVerify()}catch(e){$("dVsum").textContent="실패: "+e.message}}
$("dVerifyNew").onclick=async()=>{const b=await dBusy();if(b.length&&!confirm(busyMsg(b)+"\n\n그래도 지금 검증할까요?"))return;startVerify("new",b.length?confirm(watchMsg(b)):false)};
$("dVerify").onclick=async()=>{const b=await dBusy();if(dV&&!confirm(`모든 줄(${fmt(dV.total)}줄)을 지금 룰로 새로 확인합니다 (약 ${hms(dV.est_all_s)}).${b.length?"\n\n※ "+busyMsg(b):""}\n진행할까요?`))return;startVerify("all",b.length?confirm(watchMsg(b)):false)};
$("dVerifyStop").onclick=async()=>{try{await post("/api/data/verify/stop",{date:$("dDate").value,dev:dSrc()});loadVerify()}catch(e){$("dVsum").textContent="실패: "+e.message}};
let dSeq=0;   // 늦게 온 앞 조회(예: 모든 날짜 첫 조회는 몇 초)가 새 조건의 결과를 덮지 않게
async function loadData(){dSummary();if(!D.length){$("dBody").innerHTML='<div class="muted">08_dataset 에 학습데이터가 없습니다.</div>';return}
 const qs=new URLSearchParams({date:$("dDate").value,dev:$("dDev").value,label:$("dLabel").value,rule:$("dRule").value.trim(),q:$("dQ").value.trim(),cons:$("dCons").value,hold:$("dHold").value,llm:$("dLlm").value,offset:dOff,limit:dLimit()}),seq=++dSeq;
 $("dBody").style.opacity=.5;let r;try{r=await get("/api/data/rows?"+qs)}catch(e){if(seq===dSeq)$("dBody").innerHTML=`<div class="muted">불러오지 못했습니다: ${esc(e.message)}</div>`;return}finally{if(seq===dSeq)$("dBody").style.opacity=1}
 if(seq!==dSeq)return;
 const all=$("dDate").value==="all";
 dSkipped=r.skipped||[];   // '전체' + 정합성·LLM 재확인 조건에서 검증 결과를 쓸 수 없어 뺀 날짜 (2026-11-02~)
 $("dBody").innerHTML=(dSkipped.length?`<div class="vwarn" style="margin-bottom:6px">정합성·LLM 재확인 조건은 검증 결과를 쓸 수 있는 날짜만 봅니다 - 빠진 날짜 ${dSkipped.length}개:<br>${dSkipped.map(g=>esc(g.text)).join("<br>")}</div>`:"")+(dConv()?convTable(r,all):`<table><thead><tr><th class="n">index</th><th>payload</th><th>method</th><th>label</th><th>label 사유</th><th>검증 결과</th><th>label 정합성</th><th>탐지 피처 <a href="#/features" style="font-weight:400">(설명)</a></th><th>수정·삭제</th></tr></thead><tbody>${r.rows.map((x,k)=>`<tr class="lv-${x.level}"><td class="n">${x.index}${all?`<div class="muted">${x.date}</div>`:""}</td><td><div class="payload">${esc(x.payload.replace(/\\n/g,"\n"))}</div></td><td class="method">${esc(x.method)}</td><td><span class="chip l-${esc(x.label)}">${esc(x.label)}</span></td><td class="reason">${esc(x.reason)}</td><td class="chk">${x.checks.map(([s,t])=>`<div class="${s}">${s==="ok"?"✔":s==="warn"?"△":"✘"} ${esc(t)}</div>`).join("")}</td><td class="cons"><span class="chip c-${x.cons}">${esc(x.cons_text)}</span><div>${esc(x.cons_reason)}</div>${x.llmc?xCell(x.llmc):x.llm?`<div class="llm"><span class="chip ${RCC[x.llm.v]||""}">LLM ${esc(RCV[x.llm.v]||x.llm.v)}</span> ${esc(x.llm.r)} <span class="muted">(${esc(x.llm.m)} · ${esc((x.llm.at||"").slice(5,16).replace("T"," "))})</span></div>`:""}</td><td class="feat">${x.features.length?x.features.map(f=>`<div><code class="fn" data-f="${esc(f.name)}">${esc(f.name)}</code> = ${f.value} <span class="muted">${esc(f.desc)}</span></div>`).join("")+`<div class="muted">${x.label==="normal"?"정상인데 켜짐 → 모델이 공격 쪽으로 볼 수 있음 (판정은 label 정합성 열)":`켜진 피처 ${x.features.length}개 → 모델이 공격으로 볼 근거`}</div>`:`<div class="muted">${r.ml_error?esc(r.ml_error):x.label==="normal"?"켜진 피처 없음 → 정상으로 볼 근거":"켜진 피처 없음 → 이 피처로는 탐지 못 함"}</div>`}${(x.vfeatures||[]).length?`<div class="muted" style="margin-top:4px">라벨 검증용: ${x.vfeatures.map(f=>`<code class="fn" data-f="${esc(f.name)}" data-d="${esc(f.desc)}">${esc(f.name)}</code>=${f.value}`).join(" · ")}</div>`:""}</td><td class="act">${editCell(x,k)}</td></tr>`).join("")}</tbody></table>`);
 dRows=r.rows;dTot=r.total;$("dCount").textContent=`${fmt(r.total)}건 중 ${r.total?dOff+1:0}–${Math.min(dOff+dLimit(),r.total)}`;$("dPrev").disabled=dOff===0;$("dNext").disabled=dOff+dLimit()>=r.total;}
$("dDown").onclick=e=>{   // 저장된 '전체 검증' 결과가 없으면 줄마다 다시 확인하므로 오래 걸릴 수 있음
 if(dSkipped.length&&!confirm(`검증 결과를 쓸 수 없는 날짜 ${dSkipped.length}개(${dSkipped.map(g=>g.date+(dConv()?"":" "+g.dev.toUpperCase())).join(", ")})는 CSV 에서도 빠집니다 (정합성·LLM 재확인 조건 - 까닭은 표 위 안내). 받을까요?`)){e.preventDefault();return}
 if(dConv()){if(dTot>30000&&!confirm(`${fmt(dTot)}줄을 ${$("dDev").value==="fw"?"FW":"웹"} 로그로 바꿔 CSV 를 만듭니다 (${dur(dTot*0.001)}). 지금 받을까요?`))e.preventDefault();return}   // FW·웹: 변환만 (1천 줄에 약 1초)
 const flt=$("dCons").value||$("dLlm").value||$("dHold").value==="rule";   // 정합성 조건이면 검증한 날짜만 받으므로 그 날짜의 저장된 결과로
 if(!(flt?dV&&dV.frame_ready_done:dReady)&&dTot>1000&&!confirm(`'전체 검증' 결과가 없어 ${fmt(dTot)}줄을 하나씩 다시 확인하며 CSV 를 만듭니다 (${dur(dTot*0.015)}).\n'전체 검증'을 먼저 하면 바로 받을 수 있습니다. 지금 받을까요?`))e.preventDefault()};
for(const id of["dDate","dDev"])$(id).onchange=()=>{dOff=0;$("dCons").value=$("dLlm").value="";loadData();loadVerify();loadEdits();loadSave();loadRc()};
/* 수정·삭제: 저장하면 '학습 로그에 반영'(1.4 다시 만들기) 때 학습 로그에 들어간다 */
let dCands=[];   // 승인 전 후보 버전 [{name, rev}] - 표의 '후보에' 버튼 (6. 후보의 '데이터 추가 - 샘플 넣기')
async function loadDCands(){try{dCands=(await get("/api/sup/candidates")).candidates}catch(e){dCands=[]}}
async function candAdd(x){if(!dCands.length)return;let name=dCands[0].name;
 if(dCands.length>1){const pick=prompt(`넣을 후보 버전 이름 (${dCands.map(c=>`${c.name} rev ${c.rev}`).join(", ")})`,name);if(!pick)return;name=pick.trim()}
 if(!confirm(`index ${x.index} 줄 (룰 ${x.rule_id}, ${x.label}, sk ${x.sk}) 을 후보 '${name}' 의 '데이터 추가 - 샘플 넣기'에 넣습니다. 다시 내보내기(rev+1) 때 학습 파일에 들어갑니다 (기준·신규 중복이면 들어가지 않음). 진행할까요?`))return;
 try{const r=await post("/api/sup/additions",{name,kind:"samples",values:[x.sk],action:"add"});alert(`후보 '${name}' 에 넣었습니다 (샘플 넣기 ${fmt((r.samples||[]).length)}개)`)}catch(e){alert("실패: "+e.message)}}
function editCell(x,k){if(!x.sk)return"";
 if(x.edit)return x.edit.action==="delete"?`<span class="chip c-BAD">삭제 예정</span><div><button class="sm" data-a="undo" data-r="${k}">되돌리기</button></div>`
  :`<span class="chip c-RULE">${x.edited?"수정됨":"수정 예정"}${x.edit.label?" → "+esc(x.edit.label):""}${x.edit.payload?" (payload)":""}</span><div><button class="sm" data-a="edit" data-r="${k}">수정</button><button class="sm" data-a="undo" data-r="${k}">되돌리기</button></div>`;
 return(x.edited?`<span class="chip c-RULE">수정됨 (${esc(x.edited)})</span><br>`:"")+`<button class="sm" data-a="edit" data-r="${k}">수정</button><button class="sm danger" data-a="del" data-r="${k}">삭제</button>`+
  (dCands.length?`<button class="sm" data-a="cand" data-r="${k}" title="이 줄(샘플)을 후보 버전의 '데이터 추가 - 샘플 넣기'에 넣음 (6. 화면)">후보에</button>`:"")}
async function editPost(body){try{await post("/api/data/edit",body);loadData();loadEdits()}catch(e){alert("실패: "+e.message)}}
$("dBody").addEventListener("click",e=>{const b=e.target.closest("button[data-a]");if(!b)return;const x=dRows[+b.dataset.r];if(!x)return;
 const at={date:x.date,dev:$("dDev").value,line:x.line};
 if(b.dataset.a==="edit")openEditor(x);
 else if(b.dataset.a==="cand")candAdd(x);
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
/* LLM 재확인 (3_6 별도 프로세스): 대상 '불일치'(줄 전부) 또는 '정합 (룰 기준)'(룰당 표본). 모델·추론 강도는 이번 실행만 (LLM_설정.md 는 그대로),
   시작 전 대상·예상·확인 창. 줄 1: 실행 상태 + '불일치' 판정·반영, 줄 2: '정합 (룰 기준)' 판정·룰별 요약·반영 (대상마다 따로 되돌림).
   FW·웹 보기에서는 IPS 라고 쓰지 않는다 (rcDevName) */
const RCV={ok:"label 맞음",flip:"label 반대",unfit:"학습에 부적합",unknown:"판단 불가"},RCC={ok:"c-OK",flip:"c-BAD",unfit:"c-CHECK",unknown:""},RCT={bad:"불일치",rule:"정합 (룰 기준)"};
// 합의 (교차 확인 2026-11-02~, 2026-11-04~ 부적합 삭제 + 다수결 투표): 상태·색·까닭 (3_6 consensus_state)
const XS={keep:"확인됨 (유지)",delete:"부적합 삭제",flip:"투표로 label 반대",review:"추가 분석"},XC={keep:"c-OK",delete:"c-BAD",flip:"c-CHECK",review:""},
 XW={pending:"2번째 모델 대기",split:"투표 갈림",mixed:"판단 불가가 섞임",unfit:"",decided:"표시함",vote:""};
const XRULE="'정합 (룰 기준)' 줄의 label 반대는 룰에 걸리는 정상(또는 룰에 안 걸리는 공격)을 만듭니다";
function xCell(c){return `<div class="llm"><span class="chip ${XC[c.s]||""}">합의 ${esc(c.s==="flip"&&c.w==="decided"?"label 반대":XS[c.s]||c.s)}${XW[c.w]?" · "+XW[c.w]:""}</span>${c.v.map(([m,v,r,at])=>`<div><span class="chip ${RCC[v]||""}">${esc(m)} ${esc(RCV[v]||v)}</span> ${esc(r)} <span class="muted">(${esc((at||"").slice(5,16).replace("T"," "))})</span></div>`).join("")}</div>`}
function xLine(X,t){const cc=t==="rule"?"RULE":"BAD",sp=X?(X.split??X.mixed??0):0;if(!X||!(X.keep+X.delete+X.review+X.flip))return "";
 return `<span class="muted">합의</span>`+[["keep",X.keep],["delete",X.delete],["flip",X.flip],["review",X.review]].map(([k,v])=>v?`<span class="chip ${XC[k]}" data-l="c_${k}" data-c="${cc}" title="표에서 보기">${XS[k]} ${fmt(v)}${k==="keep"&&X.kept?` (유지 표시 ${fmt(X.kept)})`:""}${k==="review"?` (2번째 모델 대기 ${fmt(X.pending)} · 투표 갈림 ${fmt(sp)})`:""}</span>`:"").join("")}
function xButtons(X,t){const tt=t==="rule"?' data-t="rule"':"",kp=X.keep-X.kept,sp=X.split??X.mixed??0;
 return (X.delete?`<button class="sm danger" data-r="xdelete"${tt} title="어느 모델이든 '학습에 부적합'이라 한 줄">부적합 삭제 ${fmt(X.delete)}줄 → 삭제로 저장</button>`:"")+
  (X.flip?`<button class="sm" data-r="xflip"${tt} title="두 모델 이상이 판정하고 '반대'가 다수인 줄 - label 을 바꾸고 투표 결과·근거를 사유로${t==="rule"?" · "+XRULE:""}">투표로 label 반대 ${fmt(X.flip)}줄 → label 바꿔 저장</button>`:"")+
  (kp>0?`<button class="sm" data-r="xkeep"${tt}>확인됨 ${fmt(kp)}줄 유지 표시</button>`:"")+(X.review?`<button class="sm" data-r="xreview"${tt}>추가 분석 ${fmt(X.review)}줄</button>`:"")+
  (sp&&X.split!=null?`<button class="sm" data-r="xvote"${tt} title="투표 갈림 줄을 그 줄을 아직 판정하지 않은 다른 모델(3차)로 다시 묻고 다수결">3차 모델로 투표 (갈림 ${fmt(sp)}줄)…</button>`:"")+
  (t==="rule"&&X.flip?`<span class="muted" style="font-size:12px">※ ${XRULE}</span>`:"")}
function xUndo(n,X,t){const d=X?X.decided:0;return n||d?`<button class="sm" data-r="undo"${t==="rule"?' data-t="rule"':""}>LLM 반영 되돌리기 (수정·삭제 ${fmt(n)}건${d?` · 표시 ${fmt(d)}줄`:""})</button>`:""}
let dRcS=null,dRcT=0,rcInit=false,rcWas=false,rcMsg="",rcMsgR="",rcSyncKey="",rcRulesD=null,rcThr=50,rcMinN=2,rcRulesG=null,rcProb=true;
function rcDevName(d){return d==="ips"&&dConv()?"원본 payload":d.toUpperCase()}
function rcSampleUi(){const r=$("rcTarget").value==="rule",x=$("rcMode").value==="cross",tb=$("rcMode").value==="tiebreak";$("rcSample").hidden=!r||x||tb;$("rcPerN").hidden=!r||x||tb||$("rcPer").value!=="custom";
 $("rcFirstL").hidden=$("rcPickRow").hidden=!x;$("rcPickOpt").hidden=$("rcPickSel").value!=="mid";
 $("rcModelL").innerHTML=x?'2번째 모델 <span class="muted">(1번째와 다른 모델 · 이번 실행만)</span>':tb?'3차 모델 <span class="muted">(투표 갈림 줄을 아직 판정하지 않은 모델 · 이번 실행만)</span>':'모델 <span class="muted">(이번 실행만 · 비우면 검토 모델)</span>';
 if(x){rcFirstFill();if(!$("rcModel").value.trim()&&dRcS&&dRcS.cross)$("rcModel").value=dRcS.cross.default}}   // 교차 확인: 표본 대신 1번째가 판정한 줄
function rcFirstFill(){const sel=$("rcFirst"),cur=sel.value,ms=((dRcS&&dRcS.cross&&dRcS.cross.models)||{})[$("rcTarget").value]||[];
 sel.innerHTML=ms.length?ms.map(([m,n])=>`<option value="${esc(m)}">${esc(m)} (${fmt(n)}줄)</option>`).join(""):'<option value="">(이 대상을 판정한 모델 없음)</option>';if(ms.some(([m])=>m===cur))sel.value=cur}
function rcBoxSync(){const dd=$("dDate"),sel=$("rcDate");   // 날짜 목록 = 화면의 날짜 목록 (고른 값은 화면 날짜로), '지금 장비만' 이름
 sel.innerHTML=[...dd.options].map(o=>`<option value="${esc(o.value)}">${esc(o.text)}</option>`).join("");sel.value=dd.value;
 $("rcDevCur").textContent=rcDevName(dSrc())+" 만";rcSampleUi()}
function rcChips(c,cons){return Object.keys(RCV).map(k=>c[k]?`<span class="chip ${RCC[k]}" data-l="${k}" data-c="${cons}">${RCV[k]} ${fmt(c[k])}</span>`:"").join("")}
async function loadRc(){clearTimeout(dRcT);const v=$("dDate").value,dv=dSrc();if(!v)return;let s;
 try{s=await get(`/api/data/recheck?date=${v}&dev=${dv}`)}catch(e){$("dRc").textContent=$("dRcR").textContent="";return}
 if(v!==$("dDate").value||dv!==dSrc())return;     // 그사이 날짜·장비를 바꿨으면 (늦게 온 앞 응답) 버림
 const key=v+"|"+$("dDev").value;if(key!==rcSyncKey){rcSyncKey=key;rcBoxSync();$("dRcRules").style.display="none"}
 dRcS=s;const st=s.status||{},run=s.job.running,c=s.counts,del=(c.flip||0)+(c.unfit||0),R=s.rule,tn=RCT[st.target]||RCT.bad;
 const fresh=!!st.pid&&(st.pid===s.job.lock.pid||st.pid===(s.job.ui||{}).pid)&&st.req_total!=null;   // 이번 실행이 쓴 상태 (아니면 지난 실행의 것)
 if(!rcInit){rcInit=true;$("rcModels").innerHTML=s.known.map(m=>`<option value="${esc(m)}">`).join("");$("rcModel").placeholder=s.model||"provider:model";
  $("rcEff").innerHTML=`<option value="">설정값 (${esc(s.effort||"비움 = 모델 기본")})</option><option value="-">넘기지 않음 (모델 기본)</option>`+s.efforts.map(e=>`<option>${e}</option>`).join("");
  $("rcTarget").options[1].disabled=!R}             // 3_6 이 예전 판이면 '불일치'만
 let h="<b>LLM 재확인</b>";
 if(run&&!fresh)h+=`<span>시작하는 중… ('${tn}' 줄과 그 룰을 읽는 중)</span>`;
 else if(run){const fin=(st.req_done||0)+(st.req_fail||0);
  h+=`<span><b>도는 중</b> ('${tn}'${st.per_rule?` 룰당 ${st.per_rule}개`:""} · ${esc(st.model||"")}${st.cross?` · 교차 확인 ← ${esc(st.cross)}${st.pick?` (애매한 룰 ${Math.round(st.pick.lo*100)}~${Math.round(st.pick.hi*100)}%)`:""}`:""}${st.tiebreak?" · 3차 모델 투표 (투표 갈림 줄)":""}) · 요청 ${fmt(fin)} / ${fmt(st.req_total||0)} · 판정 ${fmt(st.rows_done||0)} / ${fmt(st.rows_total||0)}줄${st.req_fail?` · 실패 ${st.req_fail}`:""}${st.eta_s!=null?` · 남은 약 ${hms(st.eta_s)}`:""}${s.job.stop_requested?" · <b>중지 요청됨</b> (보낸 요청까지)":""}</span>`}
 else if(st.state==="running")h+=`<span class="muted">지난 재확인이 끝나지 않고 멈춤 (강제 중지 등) - 다시 시작하면 남은 줄만</span>`;
 else if(st.finished_at)h+=`<span class="muted">${esc(isoClock(st.finished_at))} '${tn}' · ${esc(st.model||"")}${st.cross?` (교차 확인 ← ${esc(st.cross)}${st.pick?`, 애매한 룰 ${Math.round(st.pick.lo*100)}~${Math.round(st.pick.hi*100)}%`:""})`:""}${st.tiebreak?" (3차 모델 투표)":""} ${({done:"끝남",stopped:"중지됨",fail:"실패"})[st.state]||""}${st.req_fail?` · 요청 ${st.req_fail}개 실패 (다시 시작하면 그 줄만)`:""}${st.error?` · ${esc(st.error)}`:""}</span>`;
 h+=`<button class="sm" data-r="box">LLM 재확인…</button>`+(s.cross&&s.cross.bad&&s.cross.bad.split!=null?'<button class="sm" data-r="oldrule" title="LLM 재확인 합의로 저장한 삭제를 새 규칙(부적합 삭제 + 다수결 투표)으로 다시 셈 - 누르기 전에는 바꾸지 않음">옛 규칙으로 지운 줄 다시 보기</button>':"");
 const XB=s.cross&&s.cross.bad,XR=s.cross&&s.cross.rule;   // 합의 수 (3_6 이 예전 판이면 없음 → 예전 버튼)
 if(s.bad||s.applied){h+=`<span class="muted">'불일치' ${fmt(s.bad)}줄</span>`+rcChips(c,"BAD")+(s.todo?`<span class="chip" data-l="todo" data-c="BAD">아직 ${fmt(s.todo)}</span>`:"")+(XB?xLine(XB,"bad"):"");
  if(!run)h+=(XB?xButtons(XB,"bad")+xUndo(s.applied,XB,"bad"):(del?`<button class="sm danger" data-r="delete">반대·부적합 ${fmt(del)}줄 → 삭제로 저장</button>`:"")+
   (c.flip?`<button class="sm" data-r="flip">반대 ${fmt(c.flip)}줄 → label 바꾸기</button>`:"")+(s.applied?`<button class="sm" data-r="undo">LLM 반영 ${fmt(s.applied)}건 되돌리기</button>`:""))+
   (rcMsg?`<span class="muted">${esc(rcMsg)}</span>`:"")}
 $("dRc").innerHTML=s.bad||s.applied||run||st.state||(R&&R.rows)?h:"";      // 대상 줄도 지난 재확인도 없으면 숨김
 let hr="";
 if(R&&(R.checked||R.applied)){const rc=R.counts,rdel=(rc.flip||0)+(rc.unfit||0);
  hr=`<b>정합 (룰 기준)</b> <span class="muted">${fmt(R.rows)}줄 중 재확인 ${fmt(R.checked)}줄 · 룰 ${fmt(R.rules)}개 (두 장비, 내용 기준)</span>`+rcChips(rc,"RULE")+(XR?xLine(XR,"rule"):"")+`<button class="sm" data-r="rules">룰별 요약</button>`+
   (run?"":XR?xButtons(XR,"rule")+xUndo(R.applied,XR,"rule"):(rdel?`<button class="sm danger" data-r="delete" data-t="rule">반대·부적합 ${fmt(rdel)}줄 → 삭제로 저장</button>`:"")+(rc.flip?`<button class="sm" data-r="flip" data-t="rule">반대 ${fmt(rc.flip)}줄 → label 바꾸기</button>`:"")+
   (R.applied?`<button class="sm" data-r="undo" data-t="rule">LLM 반영 ${fmt(R.applied)}건 되돌리기</button>`:""))+(rcMsgR?`<span class="muted">${esc(rcMsgR)}</span>`:"")}
 $("dRcR").innerHTML=hr;
 $("rcStop").style.display=$("rcKill").style.display=run?"":"none";$("rcGo").disabled=$("rcPlan").disabled=run;
 {const rt=st.rows_total||0,qt=st.req_total||0,qf=(st.req_done||0)+(st.req_fail||0);   // 강제 중지 옆 진행률 (줄 기준, 줄 수가 없으면 요청 기준)
  progSet("rcProg",run&&fresh&&(rt||qt)?`진행 ${Math.floor(100*(rt?(st.rows_done||0)/rt:qf/qt))}%`+(rt?` · 줄 ${fmt(st.rows_done||0)}/${fmt(rt)}`:"")
   +` · 요청 ${fmt(qf)}/${fmt(qt)} (실패 ${fmt(st.req_fail||0)})`+(st.eta_s!=null?` · 남은 약 ${mins(st.eta_s)}`:""):run&&!fresh?"시작하는 중…":"")}
 if(run)dRcT=setTimeout(loadRc,2000);
 if(rcWas&&!run){loadData();if($("dRcRules").style.display!=="none")loadRcRules();if($("dRcX").style.display!=="none")loadRcX()}   // 끝나면 표의 판정·룰별 요약·추가 분석도 새로
 if($("rcMode").value==="cross")rcFirstFill();
 rcWas=run}
function rcPer(){return $("rcPer").value==="custom"?Math.max(1,Math.min(1000,Math.round(+$("rcPerN").value||1))):+$("rcPer").value}
function rcBody(){return {model:$("rcModel").value.trim(),effort:$("rcEff").value,date:$("rcDate").value||$("dDate").value,target:$("rcTarget").value,
 devs:$("rcDev").value==="all"?"all":dSrc(),per_rule:rcPer(),llm_only:$("rcLlmOnly").checked,cross:$("rcMode").value==="cross"?$("rcFirst").value:"",
 pick:$("rcMode").value==="cross"&&$("rcPickSel").value==="mid"?{lo:+$("rcLo").value,hi:+$("rcHi").value,min_rows:+$("rcMinR").value,small:$("rcSmall").value}:null,   // 애매한 룰만
 tiebreak:$("rcMode").value==="tiebreak"}}   // 3차 모델 투표 ('투표 갈림' 줄)
/* 애매한 룰만 (2026-11-02~): 1번째 모델의 룰별 삭제 비율 묶음과 분포 (10% 구간, 판정 줄 수 막대) */
function xHist(h,lo,hi){const mx=Math.max(1,...h.map(x=>x[2]));
 return `<div class="xh"><div class="muted">1번째 모델의 룰별 삭제 비율 분포 (막대 = 판정 줄 수, 판정이 적은 룰 포함)</div>${h.map(([b,r,n])=>`<div class="xhr"><span class="xhl muted">${b}~${b+10}%</span><span class="hb ${b+10<=lo?"ok":b>=hi?"bad":"mid"}" style="width:${n?Math.max(2,Math.round(220*n/mx)):0}px"></span><span class="muted">룰 ${fmt(r)}개 · ${fmt(n)}줄</span></div>`).join("")}</div>`}
function xGroups(G){const lo=Math.round(G.pick.lo*100),hi=Math.round(G.pick.hi*100),inc=G.pick.small==="include",S=G.summary;
 return `<div class="xg">${[["ok",`대체로 괜찮음 (${lo}% 미만)`,"묻지 않음 (1번째가 반대·부적합이라 한 줄은 합의대로 삭제)"],["mid",`애매함 (${lo}~${hi}%)`,"교차 확인"],["bad",`대체로 문제 (${hi}% 이상)`,"묻지 않음 - 룰별 요약에서 화면 6 '빼기'로"],["small",`판정 ${G.pick.min_rows}줄 미만`,inc?"교차 확인에 넣음":"묻지 않음"]].map(([g,n,w])=>`<div><span class="hb ${g}" style="width:10px"></span><b>${n}</b> 룰 ${fmt(S[g].rules)}개 · 판정 ${fmt(S[g].rows)}줄 (맞음·판단 불가 ${fmt(S[g].ask)}줄) <span class="muted">→ ${w}</span></div>`).join("")}${xHist(G.hist,lo,hi)}</div>`}
async function rcEstimate(b){b=b||rcBody();const rule=b.target==="rule";
 $("rcEst").textContent=rule?"세는 중… ('정합 (룰 기준)' 줄에서 룰마다 표본을 고르고 그 줄과 룰만 읽음, 호출 없음 - 처음은 30초쯤)":"세는 중… ('불일치' 줄과 그 룰을 읽음, 호출 없음)";let p;
 try{p=await post("/api/data/recheck/plan",b)}catch(e){$("rcEst").textContent="실패: "+e.message;return null}
 const usd=x=>x==null?"단가 미등록":x<0.01?"$0.01 미만":"$"+x.toFixed(2),s=p.stats;
 const uv=Object.entries((p.unverified||[]).reduce((a,[d,v])=>((a[v]=a[v]||[]).push(d.slice(5)),a),{})).map(([v,ds])=>`${rcDevName(v)} ${ds.join(", ")}`).join(" · ");
 const what=s&&s.tiebreak?`3차 모델 투표 - '${RCT[p.target]||""}' 합의가 투표 갈림인 ${fmt(s.split)}줄 (판정한 모델: ${s.voters.map(([m,n])=>`${esc(m)} ${fmt(n)}`).join(", ")}) 중 이 모델이 이미 판정 ${fmt(s.voted)}줄 → 다수결 (그래도 갈리면 추가 분석에서 사람이 정함)`
  :s&&s.first?`교차 확인 - ${esc(s.first)} 가 판정한 '${RCT[p.target]||""}' ${fmt(s.first_rows)}줄 중 ${s.ask_all+s.first_delete===s.first_rows&&s.first_delete?`부적합 ${fmt(s.first_delete)}줄은 이미 삭제 (묻지 않음) → `:""}${s.groups?`${fmt(s.ask_all)}줄 중 아래 묶음의 룰 ${fmt(s.rules_asked)}개 · ${fmt(s.ask)}줄`:`${fmt(s.ask)}줄`}을 2번째 모델로`
  :s?`'정합 (룰 기준)' ${fmt(s.pool)}줄 (룰 ${fmt(s.rules)}개${s.llm_only?` · 그중 LLM 원본 ${fmt(s.llm)}줄`:""}) → 룰 ${fmt(s.rules_sampled)}개 × 최대 ${fmt(s.per_rule)}개 = 표본 ${fmt(p.rows)}줄`:`대상 '불일치' ${fmt(p.rows)}줄`;
 $("rcEst").innerHTML=(p.key_problem?`<div class="warn">시작할 수 없음: ${esc(p.key_problem)}</div>`:"")+`<b>${esc(p.model)}</b> · ${what} <span class="muted">(내용 기준 - 장비가 달라도 같은 샘플은 한 번)</span>${p.done?` · 이 모델로 이미 ${fmt(p.done)}줄`:""} → <b>이번 ${fmt(p.todo)}줄 · 요청 ${fmt(p.requests)}개</b> <span class="muted">(같은 룰끼리 이어서 10줄씩)</span><br>
  예상 토큰 입력 <b>${fmt(p.tokens_in)}</b> · 출력 <b>${fmt(p.tokens_out)}</b> · 시간 약 <b>${hms(p.seconds)}</b> <span class="muted">(동시 ${p.workers}개)</span> · 비용 <b>${p.subscription?"Claude 구독 사용량 (토큰 요금 없음)":usd(p.cost)}</b>
  ${p.subscription&&p.requests?`<div class="warn">Claude CLI 는 토큰 요금 대신 Claude 구독 사용량을 씁니다 - 요청 ${fmt(p.requests)}개면 사용량을 많이 써서 한도에 닿을 수 있고 (닿으면 남은 요청은 멈춤), 같은 구독으로 하는 다른 Claude 작업도 막힐 수 있습니다.</div>`:""}
  ${uv?`<div class="warn">검증이 끝나지 않았거나 그 뒤 로그가 바뀌어 빠진 날짜·장비: ${esc(uv)} <span class="muted" style="font-weight:400">(그 날짜·장비에서 '검증 안 한 데이터만 검증' 뒤 다시 세면 들어감)</span></div>`:""}
  ${s&&s.groups?xGroups(s.groups):""}
  <div class="muted">${p.history?"이 모델의 지난 재확인 기록(줄당 평균)으로 계산":"이 모델의 재확인 기록이 없어 글자 수 ÷ 3 + 줄당 답 80·요청당 추론 토큰으로 어림 (실제와 다를 수 있음)"} · 같은 대상을 다른 모델로 (참고 단가, USD/100만 토큰 2026-10-29): ${p.others.map(o=>`${esc(o.model)} ${usd(o.cost)}`).join(" · ")}</div>`;
 return p}
$("rcPlan").onclick=()=>rcEstimate();
for(const id of["rcTarget","rcPer","rcMode","rcPickSel"])$(id).onchange=rcSampleUi;
$("rcGo").onclick=async()=>{const b=rcBody(),p=await rcEstimate(b);if(!p)return;if(p.key_problem){alert(p.key_problem);return}if(!p.todo){alert(b.tiebreak?"3차 투표할 줄이 없습니다 (투표 갈림 줄이 없거나 이 모델이 이미 모두 판정함 - 다른 모델을 고르세요)":b.cross?"교차 확인할 줄이 없습니다 (2번째 모델이 이미 모두 판정함)":"새로 재확인할 줄이 없습니다 (이 모델로 모두 재확인함)");return}
 const what=b.tiebreak?`3차 모델 투표 '${RCT[b.target]}' ${fmt(p.todo)}줄 (합의가 투표 갈림인 줄 → 다수결)`:b.cross?`교차 확인 '${RCT[b.target]}' ${fmt(p.todo)}줄 (1번째 ${b.cross} 가 부적합이 아니라고 한 줄${b.pick?` 중 애매한 룰 ${b.pick.lo}~${b.pick.hi}%${b.pick.small==="include"?" + 판정 적은 룰":""}`:""})`:b.target==="rule"?`'정합 (룰 기준)' 표본 ${fmt(p.todo)}줄 (룰당 최대 ${b.per_rule}개${b.llm_only?", LLM 원본만":""})`:`'불일치' ${fmt(p.todo)}줄`;
 if(!confirm(`다시 확인합니다: ${what} · 모델 ${p.model}\n요청 ${fmt(p.requests)}개 · 예상 토큰 입력 ${fmt(p.tokens_in)} / 출력 ${fmt(p.tokens_out)} · ${p.subscription?"Claude 구독 사용량":"API 비용 "+(p.cost==null?"(단가 미등록)":"약 $"+p.cost.toFixed(2))} · 시간 약 ${hms(p.seconds)}${p.subscription?`\n\n※ Claude CLI: 토큰 요금은 없지만 요청 ${fmt(p.requests)}개만큼 Claude 구독 사용량을 많이 씁니다 (한도에 닿으면 남은 요청은 멈춤, 같은 구독의 다른 작업도 막힐 수 있음).`:""}\n별도 프로세스로 돌고, 판정은 표에 코드 판정과 따로 보입니다 (학습 로그·검증 결과는 바꾸지 않음). 진행할까요?`))return;
 try{await post("/api/data/recheck/start",b);rcMsg=rcMsgR="";setTimeout(loadRc,800)}catch(e){alert("실패: "+e.message)}};
$("rcStop").onclick=async()=>{try{await post("/api/data/recheck/stop",{});loadRc()}catch(e){alert("실패: "+e.message)}};
$("rcKill").onclick=async()=>{if(!confirm("보내던 요청을 버리고 바로 멈춥니다 (그 요청의 판정은 저장되지 않음). 진행할까요?"))return;try{await post("/api/data/recheck/stop",{force:true});setTimeout(loadRc,800)}catch(e){alert("실패: "+e.message)}};
async function rcClick(e){const ch=e.target.closest("[data-l]");
 if(ch){$("dHold").value="";dHoldUi();$("dCons").value=ch.dataset.c||"";$("dLlm").value=ch.dataset.l;dOff=0;loadData();return}
 const b=e.target.closest("button[data-r]");if(!b)return;const a=b.dataset.r,t=b.dataset.t||"bad",s=dRcS;
 if(a==="box"){const x=$("dRcBox");x.style.display=x.style.display==="none"?"":"none";if(!x.style.display)rcBoxSync();return}
 if(a==="rules"){const x=$("dRcRules");x.style.display=x.style.display==="none"?"":"none";if(!x.style.display)loadRcRules();return}
 if(a==="xreview"){rcX={t,why:"split",off:0};$("dRcX").style.display="";loadRcX();return}   // 합의 '추가 분석' 목록
 if(a==="oldrule"){const x=$("dRcOld");x.style.display=x.style.display==="none"?"":"none";if(!x.style.display)loadRcOld();return}   // 옛 규칙으로 지운 줄
 if(a==="xvote"){const x=$("dRcBox");x.style.display="";rcBoxSync();$("rcTarget").value=t;$("rcMode").value="tiebreak";$("rcModel").value="";rcSampleUi();   // 3차 모델 투표: 상자를 투표로 열고 모델 고르기
  $("rcEst").innerHTML=`<span class="muted">'${RCT[t]}' 합의가 투표 갈림인 줄을 그 줄을 아직 판정하지 않은 다른 모델(3차)로 묻고 다수결로 정합니다. 3차 모델을 고르고 '대상·예상 보기'로 대상 줄·요청 수·예상 토큰·비용을 본 뒤 시작하세요 (진행률은 '강제 중지' 옆).</span>`;
  $("rcModel").focus();x.scrollIntoView({block:"nearest"});return}
 const c=t==="rule"?s.rule.counts:s.counts,n=t==="rule"?s.rule.applied:s.applied,tn=t==="rule"?"'정합 (룰 기준)' ":"",v=$("dDate").value;
 const X=(s.cross&&s.cross[t])||{};
 const msg={delete:`LLM 이 'label 반대'·'학습에 부적합'이라 한 ${tn}${fmt((c.flip||0)+(c.unfit||0))}줄을 삭제로 저장합니다.`,flip:`LLM 이 'label 반대'라 한 ${tn}${fmt(c.flip)}줄의 label 을 바꿔 저장합니다.`,
  undo:`${tn}LLM 재확인으로 저장한 수정·삭제 ${fmt(n)}건을 되돌리고${X.decided?` 유지 표시·사람 판단 ${fmt(X.decided)}줄도 지웁니다`:"니다"}.`,
  xdelete:`합의가 '부적합 삭제'인 ${tn}${fmt(X.delete||0)}줄을 삭제로 저장합니다 (어느 모델이든 '학습에 부적합'이라 한 줄).`,
  xflip:`합의가 '투표로 label 반대'인 ${tn}${fmt(X.flip||0)}줄의 label 을 바꿔 저장합니다 (두 모델 이상이 판정하고 '반대'가 다수 - 사유에 투표 결과·모델 근거).${t==="rule"?`\n\n※ ${XRULE}.`:""}`,
  xkeep:`합의가 '확인됨'(두 모델 이상 모두 맞음)인 ${tn}${fmt((X.keep||0)-(X.kept||0))}줄에 유지 표시를 남깁니다 (그 뒤 다른 모델 판정이 와도 확인됨으로 둠${t==="bad"?" · 화면 6 '불일치' 중 남기기 옵션은 '새로 생긴 데이터만 비교' 뒤 반영":""}).`}[a];
 if(!confirm(msg+(t==="rule"?` (${v==="all"?"모든 날짜":v}, 두 장비)`:"")+(a==="undo"||a==="xkeep"?"":" 사람이 이미 수정·삭제를 저장한 샘플은 건너뜁니다. '학습 로그에 반영' 때 적용되고 목록에서 되돌릴 수 있습니다.")+" 진행할까요?"))return;
 try{const r=await post("/api/data/recheck/apply",{date:v,dev:dSrc(),action:a,target:t});
  const m=a==="undo"?`되돌림 ${r.undone}건`:a==="xkeep"?`유지 표시 ${fmt(r.marked)}줄`:`저장 ${r.saved}건${r.skipped?` · 이미 수정·삭제한 샘플이라 건너뜀 ${r.skipped}건`:""}`;if(t==="rule")rcMsgR=m;else rcMsg=m;loadEdits();loadData();loadRc();if($("dRcX").style.display!=="none")loadRcX()}catch(err){alert("실패: "+err.message)}}
/* '옛 규칙으로 지운 줄 다시 보기' (2026-11-04~): LLM 재확인 합의로 저장한 삭제를 새 규칙으로 다시 셈 - 되돌리기는 누를 때만 */
async function loadRcOld(){const el=$("dRcOld");el.innerHTML='<span class="muted">합의 삭제를 새 규칙으로 다시 세는 중…</span>';let r;
 try{r=await get("/api/data/recheck/oldrule")}catch(e){el.textContent="실패: "+e.message;return}
 const ks=Object.keys(r.states),sum=k=>r.groups.reduce((a,g)=>a+g[2][k],0),chg=["keep","flip","pending","split"].reduce((a,k)=>a+sum(k),0);
 el.innerHTML=`<div class="row"><b>옛 규칙으로 지운 줄 다시 보기</b> <span class="muted">LLM 재확인 합의로 저장한 삭제(옛 규칙 '하나라도 반대·부적합 → 삭제'로 저장한 것 포함)를 새 규칙으로 다시 셉니다. 누르기 전에는 아무것도 바꾸지 않습니다.</span><button class="sm" id="rcOldClose">닫기</button></div>`+
  (r.groups.length?`<table class="mini"><thead><tr><th>저장 시각</th><th>대상</th><th class="n">줄</th>${ks.map(k=>`<th class="n">${esc(r.states[k])}</th>`).join("")}</tr></thead><tbody>${r.groups.map(([at,t,c])=>`<tr><td class="nw">${esc(at.replace("T"," "))}</td><td>${esc(RCT[t])}</td><td class="n">${fmt(ks.reduce((a,k)=>a+c[k],0))}</td>${ks.map(k=>`<td class="n">${fmt(c[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>
  <div class="row"><span>새 규칙으로 바뀌는 줄 <b>${fmt(chg)}</b> (확인됨 ${fmt(sum("keep"))} · label 반대 ${fmt(sum("flip"))} · 추가 분석 ${fmt(sum("pending")+sum("split"))}) · 그대로 삭제 ${fmt(sum("delete"))}${sum("none")?` · 판정 기록 없음 ${fmt(sum("none"))} (그대로)`:""}</span>${chg?`<button class="sm danger" id="rcOldGo">바뀌는 ${fmt(chg)}줄 삭제 되돌리기</button>`:""}</div>
  <div class="muted" style="font-size:12.5px">되돌린 뒤 '학습 로그에 반영'(다시 만들기) → '검증 안 한 데이터만 검증'을 하면 그 줄이 합의 줄에 다시 보입니다: 확인됨은 학습에 남고, label 반대는 '투표로 label 반대 저장', 추가 분석은 교차 확인(2번째 모델 대기)·'3차 모델로 투표'(투표 갈림). 지운 삭제는 기록 파일에 남아 '되돌림 취소'로 다시 삭제할 수 있습니다.</div>`
  :'<div class="muted">LLM 재확인 합의로 저장한 삭제가 없습니다.</div>')+
  (r.last?`<div class="row"><span class="muted">${esc(isoClock(r.last.at))} 에 ${fmt(r.last.n)}줄 되돌림 (${esc(r.last.file)})</span><button class="sm" id="rcOldUndo">이 되돌림 취소 (${fmt(r.last.n)}줄 다시 삭제)</button></div>`:"")}
$("dRcOld").addEventListener("click",async e=>{const b=e.target.closest("button");if(!b)return;
 if(b.id==="rcOldClose"){$("dRcOld").style.display="none";return}
 const a=b.id==="rcOldGo"?"revert":b.id==="rcOldUndo"?"restore":"";if(!a)return;
 if(!confirm(a==="revert"?"새 규칙으로 삭제가 아닌 줄(확인됨·label 반대·추가 분석)의 합의 삭제를 지웁니다. 부적합 삭제는 그대로입니다.\n학습 로그에서 이미 빠진 줄은 '학습 로그에 반영'(다시 만들기)을 해야 돌아오고, 그 뒤 '검증 안 한 데이터만 검증'을 하면 합의 줄에 다시 보입니다. 지운 것은 기록 파일에 남아 취소할 수 있습니다. 진행할까요?"
  :"가장 최근 되돌림을 취소하고 그 줄들을 다시 삭제로 저장합니다 (그 뒤 다른 수정·삭제가 생긴 샘플은 건너뜀). 진행할까요?"))return;
 try{const r=await post("/api/data/recheck/oldrule",{action:a});
  alert(a==="revert"?`${fmt(r.reverted)}줄의 삭제를 지웠습니다 (기록: ${r.backup||"-"}). '학습 로그에 반영' 뒤 '검증 안 한 데이터만 검증'을 하세요.`:`${fmt(r.restored)}줄을 다시 삭제로 저장했습니다${r.skipped?` (그 뒤 다른 수정이 있어 건너뜀 ${fmt(r.skipped)}줄)`:""}.`);
  loadRcOld();loadEdits();loadRc()}catch(err){alert("실패: "+err.message)}});
/* 합의 '추가 분석' 목록 (2026-11-02~): 두 모델의 판정·근거를 나란히 보고 줄마다 또는 고른 줄을 묶어 유지·삭제·label 반대 (사람 판단) */
let rcX={t:"bad",why:"split",off:0},rcXD=null;
async function loadRcX(){const el=$("dRcX"),v=$("dDate").value,{t,why,off}=rcX;el.innerHTML='<span class="muted">추가 분석 줄을 읽는 중…</span>';let r;
 try{r=await get(`/api/data/recheck/review?target=${t}&date=${v}&dev=${dSrc()}&why=${why}&offset=${off}&limit=50`)}catch(e){el.textContent="실패: "+e.message;return}
 rcXD=r;const cn=r.counts||{};
 el.innerHTML=`<div class="row"><b>추가 분석</b> <span class="muted">('${RCT[t]}' · ${v==="all"?"모든 날짜":esc(v)} · ${t==="bad"?rcDevName(dSrc()):"두 장비"} · 내용 기준)</span>
  <select id="rcXWhy"><option value="split">투표 갈림 ${fmt(cn.split||0)}줄</option><option value="pending">2번째 모델 대기 ${fmt(cn.pending||0)}줄</option><option value="all">전체</option></select><button class="sm" id="rcXClose">닫기</button></div>
  <div class="row"><span class="muted">고른 <b id="rcXN">0</b>줄을</span><button class="sm" data-xd="keep">유지</button><button class="sm danger" data-xd="delete">삭제</button><button class="sm" data-xd="flip">label 반대</button>
  <span class="muted">유지 = 확인됨으로 표시 (학습에 남김) · 삭제·label 반대 = 수정·삭제로 저장 ('학습 로그에 반영' 때 적용) · 'LLM 반영 되돌리기'로 모두 되돌림</span></div>
  ${r.rows.length?`<table class="mini"><thead><tr><th><input type="checkbox" id="rcXAll"></th><th>날짜 · 줄</th><th>룰 · label</th><th>payload</th><th>모델별 판정 · 근거</th><th>정하기</th></tr></thead><tbody>${r.rows.map((x,i)=>`<tr><td><input type="checkbox" data-xi="${i}"></td><td class="nw">${esc(x.date.slice(5))} ${esc(x.dev.toUpperCase())}<div class="muted">${x.line}</div></td><td>${esc(x.rule_id)}<div><span class="chip l-${esc(x.label)}">${esc(x.label)}</span></div></td><td><div class="payload" style="max-height:140px">${esc(x.payload.replace(/\\n/g,"\n"))}</div></td><td>${x.votes.map(([m,vv,rr])=>`<div><span class="chip ${RCC[vv]||""}">${esc(m)} ${esc(RCV[vv]||vv)}</span> ${esc(rr)}</div>`).join("")}${x.why==="pending"?'<div class="muted">2번째 모델 판정 없음 - 교차 확인을 돌리면 합의가 정해질 수 있음</div>':x.why==="split"&&x.votes.length<3?'<div class="muted">투표 갈림 - 3차 모델 투표로 다수가 생기면 정해짐</div>':""}</td><td class="nw"><button class="sm" data-xr="${i}" data-xd="keep">유지</button><button class="sm danger" data-xr="${i}" data-xd="delete">삭제</button><button class="sm" data-xr="${i}" data-xd="flip">반대</button></td></tr>`).join("")}</tbody></table>`:'<div class="muted">이 조건의 추가 분석 줄이 없습니다.</div>'}
  <div class="row"><button class="sm" id="rcXPrev"${off?"":" disabled"}>이전</button><button class="sm" id="rcXNext"${off+50<r.total?"":" disabled"}>다음</button><span class="muted">${fmt(r.total)}줄 중 ${r.total?off+1:0}–${Math.min(off+50,r.total)}</span></div>`;
 $("rcXWhy").value=why}
function rcXCount(){$("rcXN").textContent=$("dRcX").querySelectorAll("[data-xi]:checked").length}
$("dRcX").addEventListener("change",e=>{if(e.target.id==="rcXWhy"){rcX.why=e.target.value;rcX.off=0;loadRcX()}
 else if(e.target.id==="rcXAll"){$("dRcX").querySelectorAll("[data-xi]").forEach(c=>c.checked=e.target.checked);rcXCount()}else if(e.target.dataset.xi!=null)rcXCount()});
$("dRcX").addEventListener("click",async e=>{const b=e.target.closest("button");if(!b)return;
 if(b.id==="rcXClose"){$("dRcX").style.display="none";return}
 if(b.id==="rcXPrev"||b.id==="rcXNext"){rcX.off=Math.max(0,rcX.off+(b.id==="rcXNext"?50:-50));loadRcX();return}
 const d=b.dataset.xd;if(!d)return;
 const idx=b.dataset.xr!=null?[+b.dataset.xr]:[...$("dRcX").querySelectorAll("[data-xi]:checked")].map(c=>+c.dataset.xi);
 if(!idx.length){alert("줄을 고르세요");return}
 const ks=idx.map(i=>rcXD.rows[i].k),what={keep:"유지 (확인됨으로 표시)",delete:"삭제",flip:"label 반대"}[d];
 if(!confirm(`'${RCT[rcX.t]}' ${ks.length}줄을 ${what}로 정합니다.${d==="keep"?"":" 수정·삭제로 저장하고 '학습 로그에 반영' 때 적용됩니다 (사람이 직접 저장한 수정·삭제가 있는 샘플은 건너뜀)."} 'LLM 반영 되돌리기'로 되돌릴 수 있습니다. 진행할까요?`))return;
 try{const r=await post("/api/data/recheck/apply",{date:$("dDate").value,dev:dSrc(),action:"decide",target:rcX.t,decision:d,ks});
  const m=`정함 ${fmt(r.marked)}줄 · 저장 ${fmt(r.saved)}건${r.undone?` · 지움 ${fmt(r.undone)}건`:""}${r.skipped?` · 사람이 저장한 수정·삭제라 건너뜀 ${fmt(r.skipped)}건`:""}`;if(rcX.t==="rule")rcMsgR=m;else rcMsg=m;
  loadRcX();loadRc();loadEdits();loadData()}catch(err){alert("실패: "+err.message)}});
for(const id of["dRc","dRcR"])$(id).addEventListener("click",rcClick);
/* '정합 (룰 기준)' 룰별 요약: 삭제 비율·판정 수 큰 순 (2026-11-02~ 합의로 셈: 확인됨·삭제·추가 분석). 기준(삭제 비율·판정 수)에 드는 룰 → 화면 6 선별의 '룰 조정 - 빼기' (담기 전) */
function rcRuleFirst(){   // 룰별 요약의 1번째 모델 묶음: 교차 확인 상자에서 고른 1번째 결과('정합 (룰 기준)'), 아니면 그 대상을 가장 많이 판정한 모델
 if($("rcMode").value==="cross"&&$("rcTarget").value==="rule"&&$("rcFirst").value)return $("rcFirst").value;
 const ms=((dRcS&&dRcS.cross&&dRcS.cross.models)||{}).rule||[];return ms.length?ms[0][0]:""}
async function loadRcRules(){const el=$("dRcRules"),v=$("dDate").value;el.innerHTML='<span class="muted">룰별로 세는 중…</span>';let r;
 const fm=rcRuleFirst(),g=fm?{first:fm,lo:+$("rcLo").value||30,hi:+$("rcHi").value||70,min:+$("rcMinR").value||2}:null;
 try{r=await get(`/api/data/recheck/rules?date=${v}`+(g?`&first=${encodeURIComponent(g.first)}&lo=${g.lo}&hi=${g.hi}&min=${g.min}`:""))}catch(e){el.textContent="실패: "+e.message;return}
 if(v!==$("dDate").value)return;rcRulesD=r.rows;rcRulesG=g&&r.rows.length&&r.rows[0].length>8?g:null;
 el.innerHTML=!rcRulesD.length?'<span class="muted">재확인한 \'정합 (룰 기준)\' 줄이 아직 없습니다.</span>':`<div class="row"><b>룰별 요약</b> <span class="muted">(${v==="all"?"모든 날짜":esc(v)} · 두 장비, 내용 기준 · 줄마다 합의: 하나라도 반대·부적합 = 삭제, 두 모델 이상 모두 맞음 = 확인됨, 그 밖 = 추가 분석 · 판정한 룰 ${fmt(rcRulesD.length)}개)</span></div>
  <div class="row">삭제 비율<input id="rcThr" type="number" min="1" max="100" value="${rcThr}" style="width:58px">% 이상 · 판정<input id="rcMinN" type="number" min="1" value="${rcMinN}" style="width:52px">줄 이상인 룰${rcRulesG?` <label class="inl" title="1번째 모델(${esc(rcRulesG.first)})의 룰별 삭제 비율이 ${rcRulesG.hi}% 이상 (판정 ${rcRulesG.min}줄 이상) - 교차 확인 '애매한 룰만'에서 묻지 않는 룰. 기준은 교차 확인 상자의 값"><input type="checkbox" id="rcProb"${rcProb?" checked":""}> + 1번째 모델 '대체로 문제' (${rcRulesG.hi}% 이상)</label>`:""} <b id="rcPick"></b>
   <button class="sm" id="rcToSup">화면 6 선별에서 빼기</button><span class="muted">화면 6 '룰 조정'에 '빼기'로 넣습니다 (담기 전이라 미리 보기에서 확인한 뒤 담기)</span></div><div id="rcTbl"></div>`;
 renderRcRules()}
function rcPicked(){return (rcRulesD||[]).filter(x=>(x[3]>=rcMinN&&x[5]*100>=rcThr*x[3])||(rcRulesG&&rcProb&&x[8]==="bad"))}   // x = [룰, 유형, 줄, 판정, 확인됨, 삭제, 추가 분석, label 반대, (1번째 모델 묶음)]
const XG={ok:["괜찮음","c-OK"],mid:["애매함","c-CHECK"],bad:["문제","c-BAD"],small:["판정 적음",""]};
function renderRcRules(){if(!$("rcTbl"))return;const pick=new Set(rcPicked().map(x=>x[0])),rows=rcRulesD.slice(0,300),pct=x=>Math.round(100*x[5]/x[3])+"%";
 $("rcPick").textContent=`${fmt(pick.size)}개`;$("rcToSup").disabled=!pick.size;
 $("rcTbl").innerHTML=`<table class="mini"><thead><tr><th>룰 id</th><th>공격 유형</th><th class="n">정합 (룰 기준) 줄</th><th class="n">판정</th><th class="n">확인됨</th><th class="n">삭제</th><th class="n">추가 분석</th><th class="n">label 반대 (투표·사람)</th><th class="n">삭제 비율</th>${rcRulesG?`<th title="1번째 모델(${esc(rcRulesG.first)})의 룰별 삭제 비율 묶음 (${rcRulesG.lo}%·${rcRulesG.hi}%, 판정 ${rcRulesG.min}줄 기준)">1번째 묶음</th>`:""}</tr></thead><tbody>${rows.map(x=>`<tr${pick.has(x[0])?' class="adj"':""}><td><a href="#" data-rule="${esc(x[0])}">${esc(x[0])}</a></td><td>${esc(x[1]||"")}</td>${x.slice(2,8).map(n=>`<td class="n">${fmt(n)}</td>`).join("")}<td class="n"><b>${pct(x)}</b></td>${rcRulesG?`<td>${x[8]?`<span class="chip ${XG[x[8]][1]}">${XG[x[8]][0]}</span>`:""}</td>`:""}</tr>`).join("")}</tbody></table>`+
  (rcRulesD.length>300?`<div class="muted">삭제 비율이 높은 300개까지 보임 (전체 ${fmt(rcRulesD.length)}개, 빼기 기준은 전체에 적용)</div>`:"")}
$("dRcRules").addEventListener("input",e=>{if(e.target.id==="rcThr")rcThr=Math.max(1,Math.min(100,+e.target.value||1));else if(e.target.id==="rcMinN")rcMinN=Math.max(1,+e.target.value||1);else if(e.target.id==="rcProb")rcProb=e.target.checked;else return;renderRcRules()});
$("dRcRules").addEventListener("click",e=>{const a=e.target.closest("a[data-rule]");
 if(a){e.preventDefault();$("dHold").value="";dHoldUi();$("dRule").value=a.dataset.rule;$("dCons").value="RULE";$("dLlm").value="any";dOff=0;loadData();return}
 if(e.target.id!=="rcToSup")return;const ids=rcPicked().map(x=>x[0]);if(!ids.length)return;
 if(!confirm(`부적합 비율 ${rcThr}% 이상 · 재확인 ${rcMinN}줄 이상인 룰 ${fmt(ids.length)}개를 화면 6 선별의 '룰 조정 - 빼기'로 넣습니다.\n아직 담기 전이라 화면 6 미리 보기에서 확인한 뒤 '담기'를 눌러야 저장됩니다. '넣기'로 조정해 둔 룰은 그대로 둡니다. 진행할까요?`))return;
 supExclude(ids)});
let sPendEx=null;   // 화면 4 → 화면 6 '룰 조정 - 빼기' (화면 6 폼이 아직 없으면 처음 열 때 넣음)
function supExclude(ids){sPendEx=ids.slice();if(sFormInit)sApplyEx();location.hash="#/sup"}
function sApplyEx(){if(!sPendEx||!SS)return;const ids=sPendEx;sPendEx=null;
 if(!SS.base){$("sSaveMsg").textContent="기준 버전이 없어 화면 4 에서 넘긴 룰 조정을 넣지 못했습니다";return}
 let add=0,same=0,keep=0;for(const id of ids){const v=sO.rules[id];if(v==="include")keep++;else if(v==="exclude")same++;else{sO.rules[id]="exclude";add++}}
 $("sSaveMsg").textContent=`화면 4 LLM 재확인에서 넘긴 룰 ${fmt(ids.length)}개 → '빼기' 조정 ${fmt(add)}개 더함${same?` · 이미 빼기 ${fmt(same)}개`:""}${keep?` · '넣기'로 조정해 둔 ${fmt(keep)}개는 그대로`:""} (아직 담기 전 - 미리 보기 확인 뒤 담기)`;
 sWorkSave();sPreview()}
for(const id of["dLabel","dCons","dLlm"])$(id).onchange=()=>{dOff=0;loadData()};
$("dHold").onchange=()=>{dOff=0;dHoldUi();loadData()};
$("dGo").onclick=()=>{dOff=0;loadData()};for(const id of["dRule","dQ"])$(id).onkeydown=e=>{if(e.key==="Enter"){dOff=0;loadData()}};
$("dPrev").onclick=()=>{dOff=Math.max(0,dOff-dLimit());loadData()};$("dNext").onclick=()=>{dOff+=dLimit();loadData()};
/* ── 5. 피처 ── */
let FC=null,fFocus="",fLoading=null;
function fkey(n){return n.startsWith("r_")?(n.endsWith("_all")?"r_<룰>_all":"r_<룰>_re<n>"):n}
const stdS=n=>n.replace(/^[a-z]+_payload_/,"");     // 표준 피처는 장비가 달라도 같은 피처 (IPS 이름 ↔ WAF 이름)
function fSame(i,n){return i.name===fkey(n)||(!!i.std&&/^[a-z]+_payload_/.test(n)&&i.std===stdS(n))}
function featDoc(n){return FC?FC.items.find(i=>fSame(i,n))||null:null}
function ensureFeat(){if(FC)return Promise.resolve(FC);if(!fLoading)fLoading=get("/api/features?dev=ips").then(r=>FC=r).catch(()=>null).finally(()=>{fLoading=null});return fLoading}
async function loadFeatures(focus){if(focus!==undefined)fFocus=focus;
 try{FC=await get("/api/features?dev="+$("fDev").value)}catch(e){$("fBody").innerHTML=`<tr><td colspan="9" class="muted">불러오지 못했습니다: ${esc(e.message)}</td></tr>`;return}
 const r=FC,gs=$("fGroup");if(gs.options.length===1)gs.innerHTML+=r.groups.map(g=>`<option>${esc(g.name)}</option>`).join("");
 $("fIntro").innerHTML=`ML 탐지 모델이 payload 에서 보는 피처입니다. <b>표준모델 피처(34개)를 그대로 쓰고, 표준 피처가 못 잡는 것만 신규 피처(add_)로 더합니다</b> (표준 피처와 같은 내용은 신규 피처에 없음, 줄마다 '표준과 같아 뺀 갈래'). 4. 신규 학습 데이터의 '탐지 피처'도 이 이름입니다 (켜진 표준 피처 + 켜진 신규 피처, 범용 표준 피처 제외). <b>라벨 검증용</b>(k_·ra_·r_)은 ML 피처가 아니라 4. label 정합성 판정에 쓰는 신호이고, 판정 기준은 그대로입니다.`+
  (r.combined&&r.combined.bundle?` 신규 피처는 <b>기존 피처 결합 가능</b>(같은 계열 표준 comb 피처에 합칠 수 있음, ${r.combined.combinable}개 - 줄마다 '합쳐질 표준 피처')과 <b>신규 피처로 생성 필요</b>(${r.combined.new_only}개)로 나눴습니다. 합친 정의는 '${esc(r.combined.bundle)}'(표준 피처 ${r.combined.targets}개에 갈래 ${r.combined.branches}개, 이름은 표준과 같음)이고 5.4 피처 실험 E13 에서 시험합니다 (6. 비교·'26 하반기' 선별에는 쓰지 않음).`:
   r.combined&&r.combined.error?`<div class="muted" style="margin-top:4px">결합판(5_0d)을 읽지 못해 신규 피처를 나누지 않았습니다: ${esc(r.combined.error)}</div>`:"")+
  (r.notes_found?"":`<div class="warn" style="margin-top:4px">로컬 대응표(09_features/_feature_overlap.json)가 없어 표준 피처 설명·기준 비율·대응을 보여 주지 못합니다 (11_supervised/experiments/03_피처겹침/notes.py 가 만듦).</div>`)+
  r.groups.map(g=>`<div style="margin-top:4px"><b>${esc(g.name)}</b> <code>${esc(g.prefix)}</code> ${esc(g.desc)}</div>`).join("")+`<div style="margin-top:4px">공격 룰 피처에 쓰는 룰: ${fmt(r.attack_rules)}개${r.note?` (${esc(r.note)})`:""}</div>`;
 const v=r.view||{},pg=v.progress||{};
 $("fCountMsg").textContent=(v.running?`표준·신규 피처 켜진 줄 수 계산 중… ${pg.step||""}${pg.total?` ${fmt(pg.done||0)}/${fmt(pg.total)}`:""}`:!(v.dates||[]).length?"표준·신규 피처의 켜진 줄 수가 아직 없습니다 - '켜진 줄 수 다시 계산'을 누르세요 (재검증 없이 피처만 계산, 처음 한 번 수 분)":v.stale?`표준·신규 피처 켜진 줄 수를 계산한 뒤(${(v.computed_at||"").replace("T"," ").slice(0,16)}) 학습 데이터가 바뀌었습니다 - 다시 계산하면 반영`:"")+
  (r.missing.length?` · 라벨 검증용 켜진 줄 수는 4. 전체 검증 결과로 셉니다 (검증 안 된 날짜 ${r.missing.length}개)`:"");
 if(v.running||r.running)setTimeout(()=>{if(page==="features")loadFeatures()},v.running?3000:6000);
 renderFeatures()}
function renderFeatures(){const r=FC;if(!r)return;const g=$("fGroup").value,q=$("fQ").value.trim().toLowerCase(),v=r.view||{},vl=v.labels||{},dv=$("fDev").value.toUpperCase();
 const L={view:[vl.anomalies||0,vl.normal||0,(v.dates||[]).length>0],verify:[r.labels.anomalies||0,r.labels.normal||0,r.dates.length>0]};
 $("fSum").textContent=`· ${r.items.length}개`+(L.view[2]?` · 켜진 줄 수: 표준·신규 피처 = ${dv} 학습데이터 ${v.dates.length}개 날짜 (anomalies ${fmt(L.view[0])}줄 / normal ${fmt(L.view[1])}줄)`:"")+
  (L.verify[2]?` · 라벨 검증용 = 전체 검증한 ${r.dates.length}개 날짜 (anomalies ${fmt(L.verify[0])}줄 / normal ${fmt(L.verify[1])}줄)`:"");
 const pc=(x,t)=>t&&x?` <span class="muted">(${(100*x/t).toFixed(1)}%)</span>`:"";let last="";
 $("fBody").innerHTML=r.items.filter(i=>(!g||i.group===g)&&(!q||[i.name,i.title,i.what,i.example,i.pattern,i.ml].join(" ").toLowerCase().includes(q))).map(i=>{
  const c=r.counts[i.name]||[0,0],[la,ln,have]=L[i.src]||L.verify,head=i.group!==last?`<tr class="fgroup"><td colspan="9">${esc(i.group)}</td></tr>`:"";last=i.group;
  return head+`<tr id="f-${esc(i.name)}"${fFocus&&fSame(i,fFocus)?' class="hl"':""}><td class="nw"><code>${esc(i.name)}</code></td><td style="min-width:120px">${esc(i.title)}</td><td class="reason">${esc(i.what)}${i.ml?`<div class="muted" style="margin-top:3px">${esc(i.ml)}</div>`:""}</td><td class="reason muted">${esc(i.example)}</td><td class="muted nw">${esc(i.value)}</td>`+
   (have?`<td class="n nw">${fmt(c[0])}${pc(c[0],la)}</td><td class="n nw">${fmt(c[1])}${pc(c[1],ln)}</td><td class="n nw">${c[0]+c[1]?(100*c[0]/(c[0]+c[1])).toFixed(1)+"%":"-"}</td>`:'<td class="n muted">-</td><td class="n muted">-</td><td class="n muted">-</td>')+
   `<td style="min-width:200px">${i.pattern?`<details><summary>보기</summary><div class="pat">${esc(i.pattern)}</div></details>`:""}`+
   `${i.merge&&i.merge.length?`<details><summary class="muted">합쳐질 표준 피처 ${new Set(i.merge.map(m=>m.std)).size}개 (갈래 ${i.merge.length}개)</summary><div class="pat">${i.merge.map(m=>"→ "+esc(m.std)+" ← "+(m.how==="enc"?"[인코딩됐을 때만] ":m.how==="ua"?"[User-Agent] ":"")+esc(m.pattern)+"<br>   "+esc(m.why)).join("<br>")}</div></details>`:""}`+
   `${i.dropped&&i.dropped.length?`<details><summary class="muted">표준과 같아 뺀 갈래 ${i.dropped.length}개</summary><div class="pat">${i.dropped.map(d=>esc(d.pattern)+" → "+esc(d.std)).join("<br>")}</div></details>`:""}</td></tr>`}).join("")||'<tr><td colspan="9" class="muted">없음</td></tr>';
 if(fFocus){const el=$("fBody").querySelector("tr.hl");if(el)el.scrollIntoView({block:"center"})}}
let fTm;$("fQ").oninput=()=>{clearTimeout(fTm);fTm=setTimeout(()=>{fFocus="";renderFeatures()},200)};$("fGroup").onchange=()=>{fFocus="";renderFeatures()};$("fDev").onchange=()=>loadFeatures();
$("fCount").onclick=async()=>{try{await post("/api/features/count",{dev:$("fDev").value});$("fCountMsg").textContent="표준·신규 피처 켜진 줄 수 계산 중…";setTimeout(()=>loadFeatures(),1500)}catch(e){$("fCountMsg").textContent="실패: "+e.message}};
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
function sSpec(){const v=k=>{const x=$("o"+k).value;return x==="all"?"all":x==="custom"?Math.max(0,Math.round(+$("o"+k+"n").value||0)):+x},t=Math.round(+$("oT").value||0);
 return {conditions:{dates:sO.dates||[],sources:sO.sources||[]},options:{A:v("A"),B:v("B"),C:v("C"),exclude_bad:$("oBad").checked,exclude_rule_based:$("oRule").checked,exclude_python_ua_normal:$("oPy").checked,keep_llm_ok:$("oLlm").checked,target:t>0?t:null},overrides:{rules:sO.rules,types:sO.types}}}
function sOptUi(){const t=+$("oT").value>0;   // 직접 입력 칸은 '패턴당 직접 입력' 일 때만, 목표 합계가 있으면 B 는 '목표 합계로 채움'
 for(const k of["A","B","C"]){const fill=k==="B"&&t;$("o"+k).hidden=fill;$("o"+k+"n").hidden=fill||$("o"+k).value!=="custom"}$("oBf").hidden=!t}
function sSetForm(o,c,ov){for(const k of["A","B","C"]){const std=["all","50","20","0"].includes(String(o[k]));$("o"+k).value=std?String(o[k]):"custom";if(!std)$("o"+k+"n").value=o[k]}
 $("oT").value=o.target||"";sOptUi();$("oBad").checked=!!o.exclude_bad;$("oLlm").checked=!!o.keep_llm_ok;$("oRule").checked=!!o.exclude_rule_based;$("oPy").checked=!!o.exclude_python_ua_normal;
 sO.dates=c&&c.dates&&c.dates.length?c.dates.slice():null;sO.sources=c&&c.sources&&c.sources.length?c.sources.slice():null;sO.rules=Object.assign({},(ov&&ov.rules)||{});sO.types=Object.assign({},(ov&&ov.types)||{})}
/* 선별 저장 상태: 지금 선별(미리 보기가 정리한 조건)을 기준(불러온·방금 담은 초안, 없으면 기본값)과 비교 → 바뀐 것·전후 선택 수·담기 버튼.
   담기 전에 바꾼 것은 이 브라우저(localStorage)에 보관 → 새로 고치거나 화면을 다시 켜도 이어서 보임 (보관이 안 돼도 화면은 그대로 동작) */
let sBase0={name:null,spec:null,at:"",sel:null,snap:""},sRestored="",sJobRun=false,sCnt={};
const OPT_L={A:"A 부족한 데이터",B:"B 새 패턴",C:"C 이미 학습됨"},SWORK="rd.sup.work";
function sNorm(o,c,ov){c=c||{};ov=ov||{};return {conditions:{dates:(c.dates||[]).slice().sort(),sources:(c.sources||[]).slice().sort()},
 options:Object.assign({},SS.options,o||{}),overrides:{rules:Object.assign({},ov.rules||{}),types:Object.assign({},ov.types||{})}}}
function sSetBase(x){const dc=x&&(((x.dev||{}).ips||{}).draft||(x.draft_counts||{}).ips)||{},sn=x&&x.snapshot;   // x = 초안 (버전 표의 줄 또는 담기 응답), null = 기본값
 sBase0=x?{name:x.name,spec:sNorm(x.options,x.conditions,x.overrides),at:x.updated_at||x.created_at||"",sel:dc.sel,snap:sn&&sn.ips&&sn.waf?`${sn.ips}+${sn.waf}`:""}
  :{name:null,spec:sNorm(),at:"",sel:null,snap:""}}
function sDiff(a,b){   // 기준 a → 지금 b: [바뀐 수, 줄]
 const out=[],cap=v=>v==="all"?"전부":v===0?"빼기":`패턴당 ${v}`,onoff=v=>v?"켬":"끔",eff=(o,k)=>k==="B"&&o.target?"목표 합계로 채움":cap(o[k]),tg=v=>v?fmt(v)+"개":"없음";let n=0;
 for(const k of["A","B","C"])if(eff(a.options,k)!==eff(b.options,k)){n++;out.push(`${OPT_L[k]}: ${eff(a.options,k)} → <b>${eff(b.options,k)}</b>`)}
 if((a.options.target||null)!==(b.options.target||null)){n++;out.push(`목표 합계: ${tg(a.options.target)} → <b>${tg(b.options.target)}</b>`)}
 for(const k of["exclude_bad","keep_llm_ok","exclude_rule_based","exclude_python_ua_normal"])if(!!a.options[k]!==!!b.options[k]){n++;out.push(`${esc(SS.option_text[k]||k)}: ${onoff(a.options[k])} → <b>${onoff(b.options[k])}</b>`)}
 for(const[key,lbl,all]of[["dates","날짜",sPv.all_dates],["sources","출처",sPv.all_sources]]){
  const eff=x=>new Set(x.length?x:all),ea=eff(a.conditions[key]),eb=eff(b.conditions[key]),add=[...eb].filter(x=>!ea.has(x)),del=[...ea].filter(x=>!eb.has(x));
  if(add.length){n+=add.length;out.push(`${lbl} 넣음: <b>${esc(add.join(", "))}</b>`)}
  if(del.length){n+=del.length;out.push(`${lbl} 뺌: <b>${esc(del.join(", "))}</b>`)}}
 const RL={exclude:"빼기",include:"넣기"},ra=a.overrides.rules,rb=b.overrides.rules,rs=[];
 for(const id of new Set([...Object.keys(ra),...Object.keys(rb)]))if(ra[id]!==rb[id])rs.push(`${id} ${rb[id]?(ra[id]?RL[ra[id]]+" → ":"")+RL[rb[id]]:"조정 해제"}`);
 if(rs.length){n+=rs.length;out.push(`룰 조정 ${rs.length}개: ${esc(rs.slice(0,4).join(", "))}${rs.length>4?` 외 ${rs.length-4}개`:""} <span class="muted">(아래 룰 표 '조정한 룰')</span>`)}
 const ta=new Set(Object.keys(a.overrides.types)),tb=new Set(Object.keys(b.overrides.types)),tadd=[...tb].filter(x=>!ta.has(x)),tdel=[...ta].filter(x=>!tb.has(x));
 if(tadd.length){n+=tadd.length;out.push(`공격 유형 빼기: <b>${esc(tadd.join(", "))}</b>`)}
 if(tdel.length){n+=tdel.length;out.push(`공격 유형 빼기 해제: <b>${esc(tdel.join(", "))}</b>`)}
 return [n,out]}
function sCount(spec){   // 다른 조건의 선택 수 (지금 비교 결과로 미리 보기와 같은 계산, 같은 조건이면 다시 세지 않음). 세는 중이면 null
 if(!sPv)return null;const key=JSON.stringify(spec)+"|"+sPv.snapshot;if(key in sCnt)return sCnt[key];
 if(Object.keys(sCnt).length>20)sCnt={};sCnt[key]=null;
 post("/api/sup/preview",Object.assign({base:SS.base,dev:sDev,view:""},spec)).then(r=>{sCnt[key]=r.none?{err:1}:{sel:r.sel,label:r.sel_label,rules:new Map(r.rules.map(x=>[x[0],x[7]]))};sState();renderRules()},()=>{sCnt[key]={err:1}});
 return null}
function sCur(){return sPv?sNorm(sPv.options,sPv.conditions,sPv.overrides):null}
function sNowIso(){const d=new Date(),p=x=>String(x).padStart(2,"0");return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`}
function sWorkSave(){if(!SS||!SS.base)return;try{localStorage.setItem(SWORK,JSON.stringify({base:SS.base,at:sNowIso(),name:$("sVName").value.trim(),baseName:sBase0.name,form:sSpec()}))}catch(e){}}
function sWorkDrop(){try{localStorage.removeItem(SWORK)}catch(e){}}
function sWorkLoad(base){try{const w=JSON.parse(localStorage.getItem(SWORK)||"null");return w&&w.base===base&&w.form?w:null}catch(e){return null}}
function sInitForm(s){sSetBase(null);const w=sWorkLoad(s.base);if(!w){sSetForm(s.options,null,null);return}   // 처음 열 때: 보관한 작업이 있으면 이어서
 const x=w.baseName&&s.versions.find(v=>v.name===w.baseName&&v.status==="draft");if(x)sSetBase(x);
 sSetForm(Object.assign({},s.options,w.form.options),w.form.conditions,w.form.overrides);$("sVName").value=w.name||"";sRestored=w.at||""}
function sDiscard(){const b=sBase0.spec;sSetForm(b.options,b.conditions,b.overrides);if(sBase0.name)$("sVName").value=sBase0.name;sRestored="";sWorkDrop();sPreview()}
function sState(){const el=$("sState");if(!SS||!SS.base){el.innerHTML="";el.className="sstate";return}
 const name=$("sVName").value.trim(),same=SS.versions.find(v=>v.name===name),cur=sCur(),[n,lines]=cur?sDiff(sBase0.spec,cur):[0,[]];
 const fin=!!(sBase0.name&&SS.versions.some(v=>v.name===sBase0.name&&v.status==="final")),stale=!!(cur&&!fin&&sBase0.snap&&sBase0.snap!==sPv.snapshot);
 const saved=!!(cur&&!n&&sBase0.name&&!fin&&!stale&&name===sBase0.name),locked=!!(same&&same.status==="final"),step=!cur?1:fin&&!n?5:saved?4:3;
 const kw=same&&same.status==="candidate"?"후보":"초안";   // 후보: 같은 이름으로 담으면 선별만 바뀜 (다시 내보내야 반영)
 const lbl=saved?"✔ 저장됨":!name?"버전 이름을 넣고 담기":locked?`'${name}' 은 확정된 버전 - 다른 이름을 쓰세요`:!same?`새 초안 '${name}' 으로 담기`
  :name!==sBase0.name?`${kw} '${name}' ${kw==="후보"?"의 선별 바꾸기":"덮어쓰기"}`:n?`변경 저장 (${kw} '${name}')`:`다시 담기 (${kw} '${name}')`;
 const L=x=>x?`공격 ${fmt(x["공격"])} / 정상 ${fmt(x["정상"])}`:"",bname=esc(sBase0.name||"");let head,body="";
 if(!cur)head='<span class="muted">미리 보기를 기다리는 중입니다 (IPS·WAF 비교 결과가 둘 다 있어야 보입니다)</span>';
 else if(n){const bp=sCount(sBase0.spec),ok=bp&&!bp.err;
  head=`<span class="chip c-CHECK">저장 안 된 변경 ${n}개</span><span>기준: ${sBase0.name?`${fin?"확정된 버전":"초안"} '${bname}' (${esc(isoClock(sBase0.at))} 저장)`:"기본값"}</span>`;
  body=`<ul>${lines.map(l=>`<li>${l}</li>`).join("")}</ul><div>선택 ${ok?fmt(bp.sel):bp?"-":"(세는 중…)"} → <b>${fmt(sPv.sel)}</b>개 <span class="muted">(${ok?L(bp.label)+" → ":""}${L(sPv.sel_label)})</span></div>`+
   (sRestored?`<div class="muted">새로 고치거나 화면을 다시 켜기 전(${esc(isoClock(sRestored))})에 바꾼 것을 이어서 보여 줍니다 (이 브라우저에 보관)</div>`:"")}
 else if(fin)head=`<span class="chip c-OK">'${bname}' 확정됨</span><span>다음 버전은 기준 버전을 '${bname}' 으로 바꿔 비교한 뒤 새 이름으로 담습니다</span>`;
 else if(stale)head=`<span class="chip c-CHECK">초안 '${bname}' 와 조건은 같지만 저장 뒤 비교 결과가 바뀜</span><span>저장 때 선택 ${fmt(sBase0.sel)} → 지금 <b>${fmt(sPv.sel)}</b>개 · 다시 담아야 확정할 수 있습니다</span>`;
 else if(sBase0.name)head=`<span class="chip c-OK">초안 '${bname}' 와 같음</span><span>${esc(isoClock(sBase0.at))} 저장 · 선택 <b>${fmt(sPv.sel)}</b>개 (${L(sPv.sel_label)})${saved?" → 아래 버전 표에서 확정":""}</span>`;
 else head=`<span class="chip">기본값 그대로</span><span>아직 초안에 담지 않음 · 선택 <b>${fmt(sPv.sel)}</b>개 (${L(sPv.sel_label)})</span>`;
 const dis=sJobRun||saved||locked;
 el.className="sstate"+(!cur?"":n||stale?" dirty":saved||fin?" clean":"");
 el.innerHTML=`<div class="steps">${["조건 고르기","미리 보기 확인","담기 (저장)","버전 표에서 1차 확정"].map((t,i)=>`<span class="step ${i+1<step?"done":i+1===step?"running":""}">${"①②③④"[i]} ${t}</span>`).join("")}</div>
  <div class="head">${head}<span style="flex:1"></span>${n?`<button class="sm" data-sdiscard>바꾼 것 버리기 (${sBase0.name?`'${bname}'`:"기본값"}으로)</button>`:""}<button data-ssave${dis?" disabled":""}${saved?"":' class="primary"'}>${esc(lbl)}</button></div>${body}${sJobRun?'<div class="muted">비교·확정이 도는 중이라 지금은 담을 수 없습니다 (끝나면 새 비교 결과로 다시 셉니다)</div>':""}`;
 const sb=$("sSave");sb.textContent=lbl;sb.disabled=dis;sb.classList.toggle("primary",!saved);
 if(cur){if(n)sWorkSave();else{sWorkDrop();sRestored=""}}}
async function loadSup(){let s;try{s=await get("/api/sup/state?base="+encodeURIComponent($("sBase").value||""))}catch(e){$("sSetup").innerHTML=`<div class="msg">불러오지 못했습니다: ${esc(e.message)}</div>`;return}
 SS=s;const opts=s.finals.map(n=>`<option${n===s.base?" selected":""}>${esc(n)}</option>`).join("");if($("sBase").innerHTML!==opts)$("sBase").innerHTML=opts;
 if(!sFormInit){sFormInit=true;sInitForm(s);if(sPendEx)sApplyEx()}   // 화면 4 에서 넘긴 '빼기' 룰 (처음 열 때)
 else if(sBase0.name){const x=s.versions.find(v=>v.name===sBase0.name);if(!x)sSetBase(null);else if(x.status==="draft"&&(x.updated_at||x.created_at||"")!==sBase0.at)sSetBase(x)}   // 기준 초안이 지워졌거나 다른 곳에서 다시 담김
 $("sHdr").textContent=s.base?`· 기준 ${s.base} · 디스크 여유 ${s.disk_free_gb}GB`:"";
 $("sSetup").innerHTML=s.problems.length?`<div class="est"><b>넣을 것</b><ul style="margin:4px 0 0 18px;padding:0">${s.problems.map(p=>`<li>${esc(p)}</li>`).join("")}</ul><div class="muted">아래 '기준 버전 등록'에서 기존 학습데이터(표준모델 학습 형식 CSV)·원본 로그·표준모델 피처 파일 위치를 넣으세요.</div></div>`:"";
 const run=s.job.running,st=s.status||{};
 for(const id of["sRunNew","sRunAll"])$(id).disabled=run||!s.base||s.problems.length>0;$("sStop").style.display=run?"":"none";$("sSave").disabled=run;sJobRun=run;
 sEstimate();sJobView(run,st);renderVers();renderReg();sState();
 const cands=s.versions.filter(v=>v.status==="candidate");dCands=cands.map(v=>({name:v.name,rev:v.rev}));   // 후보 패널 (없으면 가장 최근 후보를 엶)
 if(sCandName)loadCand();else if(!sCandClosed&&cands.length)sCandOpen(cands.sort((a,b)=>(b.exported_at||"").localeCompare(a.exported_at||""))[0].name,false);
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
 const what=["finalize","export"].includes(st.cmd)?`1차 확정 '${esc(st.version||"")}' (후보 내보내기)`:`비교 (기준 ${esc(st.base||"")} · ${(st.devs||[]).map(d=>d.toUpperCase()).join("·")} · ${st.mode==="all"?"전체 다시":"새로 생긴 데이터만"})`;
 if(!st.plan){sJobOld(el,run,st,what);return}
 // 실행 계획 (5.0 RunCtx): 전체 % · 남은 시간 · 끝날 시각, 단계 = 끝남(✔ 실제 시간) / 진행 중(● 지금 진행 수) / 대기(○ 예상 시간)
 const live=run&&st.state==="running",pct=st.state==="done"?100:Math.floor(st.overall_pct||0),icon={done:"✔",running:"●",wait:"○",skip:"–"};
 const chip=p=>{const stop=p.state==="running"&&!live,tail=p.state==="done"?hms(p.sec):p.state==="running"?(live?(st.detail||hms(p.sec)+"째")+(st.eta_s!=null?` · 약 ${hms(st.eta_s)} 남음`:""):hms(p.sec)):p.state==="wait"?"약 "+hms(p.expected_s):"";
  return `<span class="step ${stop?"fail":p.state}">${stop?"✘":icon[p.state]||""} ${esc(p.text)}${tail?` <b>${esc(tail)}</b>`:""}</span>`};
 const grp=[];for(const p of st.plan.filter(p=>p.state==="running"||p.state==="wait")){const g=p.dev?p.dev.toUpperCase()+" ":"",l=grp[grp.length-1];if(l&&l[0]===g)l[1].push(p.short);else grp.push([g,[p.short]])}
 const left=grp.map(([g,a])=>g+a.join(" → ")).join(" · ");
 const head=live?`<b>${what} 중</b> · 전체 <b>${pct}%</b> · 남은 약 <b>${hms(st.eta_total_s)}</b> · <b>${isoClock(st.finish_at)}</b> 쯤 끝남 · 경과 ${hms(st.elapsed_s)}${SS.job.stop_requested?" · <b>중지 요청됨</b>":""}`
  :st.state==="done"?`<b>${what} 끝남</b> · ${esc((st.finished_at||"").replace("T"," "))} · 걸린 시간 <b>${hms(st.elapsed_s)}</b>${st.first_eta_s!=null?` <span class="muted">(시작할 때 예상 ${hms(st.first_eta_s)} · ${isoClock(st.first_finish_at)} 끝날 예상)</span>`:""}`
  :st.state==="stopped"?`<b>${what} 중지됨</b> · ${pct}% 에서 멈춤 · `+(["finalize","export"].includes(st.cmd)?"만들던 파일은 지움":"계산한 피처는 저장됨 → '새로 생긴 데이터만 비교'로 이어서 할 수 있음")
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
function sPreviewSoon(ms){clearTimeout(sPvT);sPvT=setTimeout(sPreview,typeof ms==="number"?ms:250);sWorkSave()}   // 숫자 칸은 다 친 뒤에 (ms)
let sPvSeq=0;   // 늦게 온 앞 미리 보기가 새 결과를 덮지 않게 (목표 합계 첫 계산은 몇 초)
async function sPreview(){clearTimeout(sPvT);if(!SS||!SS.base)return;let r;const seq=++sPvSeq;$("oSum").classList.add("muted");
 try{r=await post("/api/sup/preview",Object.assign({base:SS.base,dev:sDev,view:$("sView").value},sSpec()))}catch(e){if(seq===sPvSeq)$("sPrev").innerHTML=`<div class="muted">미리 보기 실패: ${esc(e.message)}</div>`;return}
 if(seq!==sPvSeq)return;$("oSum").classList.remove("muted");
 if(r.none){sPv=null;$("sPrev").innerHTML=`<div class="muted">IPS·WAF 비교 결과가 둘 다 있어야 공통 선별을 볼 수 있습니다${r.missing&&r.missing.length?` (없음: ${esc(r.missing.join("·"))})`:""}.</div>`;$("sRules").innerHTML="";$("sRuleCnt").textContent="";$("sSelHdr").textContent="";$("cDates").innerHTML=$("cSrc").innerHTML=$("sHold").innerHTML=$("oSum").innerHTML="";sState();return}
 sPv=r;renderConds(r);renderHold(r);sOptSum(r);const vw=r.view;
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
 $("sCand").href="/download/sup/candidates?"+new URLSearchParams({base:SS.base,dev:sDev,spec:JSON.stringify(sSpec()),view:vw||""});sState()}
/* 등급별로 더해지는 수 → 합계 (선택 옆에 바로), 목표 합계면 B 에서 채운 수·패턴당 몇 개 */
function sOptSum(r){const o=r.options,f=r.fill,sg=r.sel_grade||{},cap=v=>v==="all"?"전부":v===0?"빼기":`패턴당 ${fmt(v)}`;
 $("oLlm").disabled=!o.exclude_bad;$("oLlmN").textContent=r.has_lk?`(후보의 '불일치' 중 ${fmt(r.llm_ok_bad)}개)`:"(비교 결과에 LLM 합의 '확인됨' 없음 - 4.에서 교차 확인 뒤 다시 비교)";
 const part=g=>`${g} ${g==="B"&&o.target?"목표로 채움":cap(o[g])} <b>+${fmt(sg[g])}</b>`+(g==="B"&&f&&f.added&&f.k!=null?` (패턴당 ${f.extra?`${f.k}~${f.k+1}`:f.k}개)`:"");
 const note=!f?"":f.state==="over"?` <span class="warn">목표 ${fmt(f.target)}개보다 B 없이 고른 수(${fmt(f.base)}개)가 많아 B 를 넣지 않음</span>`
  :f.state==="under"?` <span class="warn">B 를 모두 넣어도 최대 ${fmt(f.max)}개까지 (목표 ${fmt(f.target)}개)</span>`:` = 목표 ${fmt(f.target)}개`;
 $("oSum").innerHTML=`${["A","B","C"].map(part).join(" · ")} → 합계 <b>${fmt(r.sel)}</b>개 <span class="muted">(공격 ${fmt(r.sel_label["공격"])} / 정상 ${fmt(r.sel_label["정상"])})</span>${note}`+
  (f&&f.state==="exact"&&f.added&&f.k!=null?`<div class="muted">B 에서 +${fmt(f.added)}개: 패턴당 ${fmt(f.k)}개까지 넣고${f.extra?`, 남은 ${fmt(f.extra)}개는 ${fmt(f.k+1)}번째 샘플을 패턴을 돌아가며 1개씩`:""} · B 를 모두 넣으면 최대 ${fmt(f.max)}개</div>`:"")}
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
 const n=rows.length;rows=rows.slice(0,300);
 const cur=sCur(),na=cur&&Object.keys(cur.overrides.rules).length?sCount(Object.assign({},cur,{overrides:{rules:{},types:cur.overrides.types}})):null,nm=na&&!na.err?na.rules:null;   // 조정한 룰: 조정 없이 고를 수 → 지금
 $("sRuleCnt").textContent=`${fmt(n)}개 룰${n>300?" 중 300개 표시 (검색으로 좁히세요)":""} · 조정 ${Object.keys(sO.rules).length}개${Object.keys(sO.rules).length?" (노란 줄 · 선택 = 조정 없이 → 지금)":""}`;
 $("sRules").innerHTML=`<table class="mini"><thead><tr><th>룰 id</th><th>공격 유형</th><th class="n">후보</th><th class="n">제거</th><th class="n">A</th><th class="n">B</th><th class="n">C</th><th class="n">선택</th><th>조정</th><th></th></tr></thead><tbody>${rows.map(x=>`<tr${x[8]?' class="adj"':""}><td class="nw">${esc(x[0])}</td><td>${esc(x[1])}</td>${[2,3,4,5,6].map(i=>`<td class="n">${fmt(x[i])}</td>`).join("")}<td class="n nw">${x[8]&&nm?`<span class="muted">${fmt(nm.get(x[0])||0)} →</span> `:""}<b>${fmt(x[7])}</b></td><td><select data-r="${esc(x[0])}"><option value="">기본</option><option value="exclude"${x[8]==="exclude"?" selected":""}>빼기</option><option value="include"${x[8]==="include"?" selected":""}>넣기 (상한 없이)</option></select></td><td class="nw"><a href="#/data" data-open="${esc(x[0])}">4.에서 보기</a></td></tr>`).join("")}</tbody></table>`}
$("sRules").addEventListener("change",e=>{const s=e.target.closest("select[data-r]");if(!s)return;if(s.value)sO.rules[s.dataset.r]=s.value;else delete sO.rules[s.dataset.r];sPreviewSoon()});
$("sRules").addEventListener("click",e=>{const a=e.target.closest("a[data-open]");if(!a)return;e.preventDefault();dPending={rule:a.dataset.open,dev:sDev,hold:$("sView").value};location.hash="#/data"});
function vDesc(x){if(x.kind==="external")return esc(x.note||"");const o=x.options||{},c=x.conditions||{},ov=x.overrides||{},cap=k=>o[k]==="all"?"전부":o[k]===0?"빼기":`패턴당 ${o[k]}`;
 const f=(((x.dev||{}).ips||{}).draft||{}).fill,fb=f&&f.added&&f.k!=null?` (패턴당 ${f.extra?`${f.k}~${f.k+1}`:f.k}개)`:"";
 return (o.target?`<b>목표 합계 ${fmt(o.target)}</b> · `:"")+`A ${cap("A")} · B ${o.target?"목표로 채움"+fb:cap("B")} · C ${cap("C")}`+(o.exclude_bad?" · 불일치 빼기"+(o.keep_llm_ok?" (LLM 이 맞다고 한 것은 남김)":""):"")+(o.exclude_rule_based?" · 정합(룰 기준) 공격 빼기":"")+(o.exclude_python_ua_normal?" · python-requests 정상 빼기":"")+
  `<br>날짜 ${c.dates&&c.dates.length?esc(c.dates.join(", ")):"전부"} · 출처 ${c.sources&&c.sources.length?esc(c.sources.join(", ")):"전부"}`+(Object.keys(ov.rules||{}).length?` · 룰 조정 ${Object.keys(ov.rules).length}개`:"")+(Object.keys(ov.types||{}).length?` · 유형 빼기 ${esc(Object.keys(ov.types).join(", "))}`:"")}
function renderVers(){const vs=SS.versions;$("sVerHdr").textContent=vs.length?`· ${vs.length}개 (확정 ${vs.filter(x=>x.status==="final").length})`:"";
 if(!vs.length){$("sVers").innerHTML='<div class="muted">버전이 없습니다 - 아래에서 기준 버전(기존 학습데이터)을 등록하세요.</div>';return}
 const dv=(x,d)=>{const c=x.dev[d]||{};if(x.status==="draft"&&c.draft)return `<span class="muted">추가 후보</span> <b>${fmt(c.draft.sel)}</b><div class="muted">공격 ${fmt(c.draft.sel_label["공격"])} / 정상 ${fmt(c.draft.sel_label["정상"])}</div>`+(x.mode==="combined"?'<div class="muted">IPS·WAF 같은 샘플</div>':'<div class="muted">장비별 (예전 방식)</div>');
   if(c.rows==null)return '<span class="muted">-</span>';const l=c.label||{};return `${fmt(c.rows)}<div class="muted">공격 ${fmt(l.anomalies)} / 정상 ${fmt(l.normal)}</div>`+(c.added!=null?`<div class="muted">기준 + 추가 ${fmt(c.added)}</div>`:"")};
 $("sVers").innerHTML=`<table class="mini"><thead><tr><th>버전</th><th>기준</th><th>상태</th><th>IPS 행</th><th>WAF 행</th><th>조건·선별</th><th>시각</th><th></th></tr></thead><tbody>${vs.map(x=>`<tr${x.status==="draft"&&x.name===sBase0.name?' class="hl"':""}><td class="nw"><b>${esc(x.name)}</b>${x.status==="draft"&&x.name===sBase0.name?' <span class="chip c-RULE">지금 화면</span>':""}${x.tag?` <span class="muted">${esc(x.tag)}</span>`:""}</td><td class="nw">${esc(x.base||"-")}</td><td>${x.status==="final"?'<span class="chip c-OK">확정</span>'+(x.approval?`<div class="muted">승인 rev ${x.approval.rev}${x.approval.met?"":" (목표 미달·사유)"}</div>`:""):x.status==="candidate"?`<span class="chip c-RULE">1차 확정 rev ${x.rev}</span>`:'<span class="chip c-CHECK">초안</span>'}</td><td class="nw">${dv(x,"ips")}</td><td class="nw">${dv(x,"waf")}</td><td class="reason">${vDesc(x)}</td><td class="muted nw">${esc((x.finalized_at||x.updated_at||x.created_at||"").replace("T"," ").slice(0,16))}</td><td class="nw">${x.status==="draft"?`<button class="sm" data-v="load" data-n="${esc(x.name)}">불러오기</button>`+(x.mode==="combined"?`<button class="sm primary" data-v="final" data-n="${esc(x.name)}" title="후보 rev 1 로 학습 파일을 내보냄 (피처 테스트·최종 승인 전이라 아직 기준이 될 수 없음)">1차 확정</button>`:`<button class="sm primary" data-v="resave" data-n="${esc(x.name)}" title="옵션·조정은 그대로, IPS·WAF 에 같은 샘플을 넣는 방식으로 다시 계산">새 방식으로 다시 담기</button>`)+`<button class="sm danger" data-v="del" data-n="${esc(x.name)}">삭제</button>`+(x.mode==="combined"?"":'<div class="muted" style="font-size:12px;white-space:normal;max-width:240px">장비별로 따로 고른 예전 초안 - 다시 담아야 확정됩니다</div>'):x.status==="candidate"?`<button class="sm primary" data-v="open" data-n="${esc(x.name)}">열기 (테스트·추가·승인)</button><button class="sm" data-v="load" data-n="${esc(x.name)}" title="이 후보의 선별을 위 폼으로 불러와 고친 뒤 같은 이름으로 담으면 선별만 바뀜 (rev·데이터 추가·목표는 그대로, 다시 내보내야 반영)">선별 불러오기</button>`
  :x.kind==="external"?'<span class="muted">등록한 기준</span>':`<span class="muted">11_supervised/versions/${esc(x.name)}/</span>`+(x.approval?`<button class="sm" data-v="open" data-n="${esc(x.name)}">승인 기록</button>`:"")}</td></tr>`).join("")}</tbody></table>`}
$("sVers").addEventListener("click",async e=>{const b=e.target.closest("button[data-v]");if(!b)return;const name=b.dataset.n,x=SS.versions.find(v=>v.name===name);
 try{if(b.dataset.v==="load"){const cur=sCur(),n=cur?sDiff(sBase0.spec,cur)[0]:0;if(n&&!confirm(`저장 안 된 변경 ${n}개를 버리고 초안 '${name}' 을 불러옵니다. 진행할까요?`))return;
   sSetForm(Object.assign({},SS.options,x.options),x.conditions,x.overrides);$("sVName").value=name;sSetBase(x);sRestored="";sWorkDrop();renderVers();sPreview();$("sSaveMsg").textContent=`'${name}' 초안을 불러왔습니다`;return}
  if(b.dataset.v==="del"){if(!confirm(`초안 '${name}' 을 지웁니다. 진행할까요?`))return;await post("/api/sup/delete",{name});loadSup();return}
  if(b.dataset.v==="resave"){   // 예전 장비별 초안 → 같은 옵션·조정으로 IPS·WAF 공통 선별 다시 계산
   const r=await post("/api/sup/save",{name,base:x.base,conditions:x.conditions,options:x.options,overrides:x.overrides});
   $("sSaveMsg").textContent=`'${name}' 을 새 방식으로 다시 담았습니다: IPS·WAF 같은 샘플 ${fmt(r.draft_counts.ips.sel)}개`;loadSup();return}
  if(b.dataset.v==="open"){sCandClosed=false;sCandOpen(name);return}
  const n=x.dev.ips&&x.dev.ips.draft?x.dev.ips.draft.sel:0;
  if(!confirm(`'${name}' 을 1차 확정합니다 (후보 rev 1, IPS·WAF 에 같은 추가 샘플 ${fmt(n)}개).\n기준 '${x.base}' + 추가 샘플로 IPS·WAF 학습 파일(CSV)·로그를 각각 만듭니다 (장비마다 약 0.5GB, 디스크 여유 ${SS.disk_free_gb}GB).\n저장한 뒤 두 장비의 비교 결과·학습데이터가 바뀌었으면 내보내지 않습니다. 후보는 피처 테스트·최종 승인 전이라 아직 다음 버전의 기준이 될 수 없습니다. 진행할까요?`))return;
  await post("/api/sup/export",{name});sCandClosed=false;sCandName=name;setTimeout(loadSup,800)}catch(err){alert("실패: "+err.message)}});
/* ── 6. 후보 버전 (2026-10-30~): 1차 확정(rev) → 피처 테스트(5_4 eval-version, job evalver - 장비마다 차례로) → 데이터 추가(rev+1) → 최종 승인 ── */
let sCandName=null,sCandD=null,sCandT=0,sCandRun=false,sCandMsg="",sCandExps=null,sCandClosed=false;
const CM3=["precision","recall","f1"],CML={precision:"precision",recall:"recall",f1:"F1"};
const cPct=x=>x==null?"-":(100*x).toFixed(2)+"%",cN=x=>x==null?"-":fmt(x);
function sCandOpen(name,scroll=true){sCandName=name;sCandMsg="";$("sCandCard").style.display="";loadCand();if(scroll)setTimeout(()=>$("sCandCard").scrollIntoView({block:"start"}),60)}
async function loadCand(){clearTimeout(sCandT);if(!sCandName)return;const name=sCandName;let d;
 try{d=await get("/api/sup/eval?name="+encodeURIComponent(name))}catch(e){$("sCandBody").innerHTML=`<div class="msg">불러오지 못했습니다: ${esc(e.message)}</div>`;return}
 if(name!==sCandName)return;sCandD=d;renderCand();
 if(!sSrvD||sSrvD.name!==name||sSrvD.rev!==d.rev)loadSrv();       // 서버 반영 칸 (후보·rev 가 바뀌면 다시)
 const run=d.job.running||(d.queue.name===name&&d.queue.todo.length>0);if(run)sCandT=setTimeout(loadCand,3000);
 if(sCandRun&&!run)loadSup();sCandRun=run}
function renderCand(){const el=$("sCandBody"),d=sCandD,x=SS&&SS.versions.find(v=>v.name===sCandName);
 if(!x||!d||!["candidate","final"].includes(x.status)){$("sCandCard").style.display="none";sCandName=null;return}
 const fin=x.status==="final",rev=x.rev,t=x.targets||{},exp=t.exp||d.default_exp,res=d.results||{},cur=res[String(rev)]||{},chk=d.check||{dev:{}};
 const A=x.additions||{},stale=!!(x.spec_changed_at&&x.spec_changed_at>(x.exported_at||"")),last=(x.revs||[]).slice(-1)[0]||{};
 const run=d.job.running||(d.queue.name===x.name&&d.queue.todo.length>0),exporting=!!(SS.job&&SS.job.running),st=d.status||{};
 const sg=d.server||{},sOk=!sg.active||sg.met;   // 서버 반영 승인 조건 (5_8 gate, 로컬 전용 5_8a 가 있을 때만)
 const tested=["ips","waf"].every(dv=>(cur[dv]||{})[exp]),step=fin?7:tested&&sg.active&&sg.met?6:tested?5:3;
 $("sCandHdr").textContent=`· ${x.name} · ${fin?"최종 승인 (확정)":"1차 확정 rev "+rev}`;
 let top=null,h=`<div class="steps">${["초안",`1차 확정 (rev ${rev})`,"피처 테스트","데이터 추가 (필요하면)","서버 반영","최종 승인"].map((s,i)=>`<span class="step ${i+1<step?"done":i+1===step?"running":""}">${"①②③④⑤⑥"[i]} ${esc(s)}</span>`).join("")}</div>`;
 const cnt=dv=>{const c=(x.dev||{})[dv]||{},o=((last.counts||{})[dv]||{}).by_origin||{};return `${dv.toUpperCase()} ${cN(c.rows)}행 (기준 + 추가 ${cN(c.added)}: 신규 ${cN(o.new)} · 외부 파일 ${cN(o.manual)})`};
 h+=`<div class="muted" style="margin:6px 0">${esc(isoClock(x.exported_at||""))} 내보냄 · ${cnt("ips")} · ${cnt("waf")} · 11_supervised/versions/${esc(x.name)}/</div>`;
 if(stale&&!fin)h+=`<div class="msg">선별을 다시 담았습니다 (${esc(isoClock(x.spec_changed_at))}) - '다시 내보내기 (rev ${rev+1})' 를 해야 학습 파일·테스트·승인에 들어갑니다</div>`;
 const exps=d.experiments||[],chosen=sCandExps||new Set([d.default_exp]);
 h+=`<h3 style="margin:12px 0 4px">피처 테스트 <span class="muted" style="font-weight:400;font-size:12.5px">rev ${rev} 로 '후보 모델'(기준 + 추가)과 '기준만 모델'을 같은 시험 세트에서 (5.4 eval-version, 장비마다 차례로)</span></h3>`;
 if(!fin)h+=`<div class="row"><label class="inl"><input type="checkbox" id="cvIps" checked> IPS</label><label class="inl"><input type="checkbox" id="cvWaf" checked> WAF</label>
  <label class="inl" title="표본(기준 학습 6만·시험 2만, 추가 학습 4만·시험 2만 행)으로 빨리 보기 - 최종 승인 판단에는 전체 테스트를 권장"><input type="checkbox" id="cvQuick"> 빠른 확인 (표본)</label>
  <details style="align-self:center"><summary>실험 ${chosen.size}개: ${esc([...chosen].join(", "))}</summary><div style="max-height:220px;overflow:auto;padding:4px 0">${exps.map(([id,ds])=>`<label class="inl" style="display:block"><input type="checkbox" data-cexp="${esc(id)}"${chosen.has(id)?" checked":""}${id===d.default_exp?" disabled":""}> <b>${esc(id)}</b> <span class="muted">${esc(ds)}</span></label>`).join("")}</div></details>
  <button class="primary" id="cvGo"${run||exporting||stale?" disabled":""}>테스트 시작 (rev ${rev})</button>${run?'<button class="danger" id="cvStop">중지</button>':""}</div>`;
 if(run)h+=`<div><b>도는 중</b> ${esc(String(st.dev||"").toUpperCase())} · ${esc(st.step_text||"시작하는 중")} · ${Math.round(st.pct||0)}%${st.eta_s!=null?` · 남은 약 ${hms(st.eta_s)}`:""}${d.queue.todo.length?` · 다음 ${esc(d.queue.todo.map(v=>v.toUpperCase()).join("·"))}`:""}</div>`;
 else if(st.status&&st.version===x.name)h+=`<div class="muted">지난 테스트: rev ${esc(String(st.rev??""))} ${esc(String(st.dev||"").toUpperCase())} ${({done:"끝남",stopped:"중지됨",error:"실패"})[st.status]||esc(st.status)}${st.error?` · ${esc(st.error)}`:""}</div>`;
 const revs=Object.keys(res).map(Number).sort((a,b)=>b-a);
 h+=revs.length?`<table class="mini" style="margin-top:6px"><thead><tr><th>rev</th><th>장비</th><th>실험</th>${CM3.map(m=>`<th class="n">${CML[m]}</th>`).join("")}<th class="n">놓친 공격</th><th class="n">오탐</th><th>기준만 모델 (P · R · F1 · 놓침 · 오탐)</th><th>끝난 시각</th></tr></thead><tbody>${
  revs.flatMap(r=>["ips","waf"].flatMap(dv=>Object.entries((res[String(r)]||{})[dv]||{}).map(([e,z])=>{const b=z.base||{},ar=m=>z[m]!=null&&b[m]!=null?(z[m]>=b[m]?'<span style="color:var(--ok)">▲</span>':'<span style="color:var(--bad)">▼</span>'):"";
   return `<tr${r!==rev?' class="muted"':""}><td>${r}${r!==rev?" (이전 rev)":""}</td><td>${dv.toUpperCase()}</td><td title="${esc(z.desc||"")}">${esc(e)}${e===exp?" ★":""}${z.quick?' <span class="chip">빠른 확인</span>':""}</td>${CM3.map(m=>`<td class="n nw">${cPct(z[m])} ${ar(m)}</td>`).join("")}<td class="n">${cN(z.miss)}</td><td class="n">${cN(z.fp)}</td><td class="muted nw">${cPct(b.precision)} · ${cPct(b.recall)} · ${cPct(b.f1)} · ${cN(b.miss)} · ${cN(b.fp)}</td><td class="muted nw">${esc(isoClock(z.finished_at||""))}</td></tr>`})))
  .join("")}</tbody></table><div class="muted" style="font-size:12.5px">▲▼ = 기준만 모델보다 높음·낮음, ★ = 최종 승인 판단에 쓰는 실험</div>`:`<div class="muted">아직 테스트 결과가 없습니다.</div>`;
 const T=dv=>t[dv]||{},tin=(dv,m)=>`<input type="number" step="0.001" min="0" max="1" data-tg="${dv}:${m}" value="${T(dv)[m]??""}" style="width:76px"${fin?" disabled":""}>`;
 h+=`<h3 style="margin:14px 0 4px">목표 <span class="muted" style="font-weight:400;font-size:12.5px">장비별 최소값 (0~1, 비우면 그 지표는 보지 않음)</span></h3><table class="mini"><thead><tr><th>장비</th>${CM3.map(m=>`<th>${CML[m]} 최소</th>`).join("")}<th>기준 버전보다 낮으면 안 됨</th><th>rev ${rev} 판단 (${esc(chk.exp||exp)})</th></tr></thead><tbody>${["ips","waf"].map(dv=>{const c=(chk.dev||{})[dv]||{};
  return `<tr><td>${dv.toUpperCase()}</td>${CM3.map(m=>`<td>${tin(dv,m)}</td>`).join("")}<td><input type="checkbox" data-tg="${dv}:not_below_base"${T(dv).not_below_base?" checked":""}${fin?" disabled":""}></td><td>${!c.has?'<span class="muted">테스트 결과 없음</span>':c.met?'<span class="chip c-OK">목표 넘음</span>':`<span class="chip c-BAD">미달</span> <span class="muted">${esc((c.fails||[]).join(" · "))}</span>`}</td></tr>`}).join("")}</tbody></table>`;
 if(!fin){h+=`<div class="row"><label>판단에 쓰는 실험<select id="cvExp">${exps.map(([id])=>`<option${id===exp?" selected":""}>${esc(id)}</option>`).join("")}</select></label><button id="cvSaveT">목표 저장</button></div>`;
  const chips=k=>(A[k]||[]).map(v=>`<span class="chip">${esc(v)} <a href="#" data-arm="${k}" data-val="${esc(v)}" title="빼기">×</a></span>`).join(" ");
  h+=`<h3 style="margin:14px 0 4px">데이터 추가 <span class="muted" style="font-weight:400;font-size:12.5px">최종 승인 전에만. 넣은 것은 '다시 내보내기 (rev ${rev+1})' 뒤에 학습 파일·테스트에 들어감 (제거 = 기준·신규 중복은 넣지 않음)</span></h3>
  <div class="row"><label>룰 넣기 (쉼표)<input id="cvRule" placeholder="예: et:1:2001044" style="width:220px"></label><button class="sm" data-add="rules">넣기</button><span>${chips("rules")}</span></div>
  <div class="row"><label>공격 유형 넣기 (쉼표)<input id="cvType" placeholder="예: xxe, ssrf" style="width:180px"></label><button class="sm" data-add="types">넣기</button><span>${chips("types")}</span></div>
  <div class="row"><span style="align-self:center">샘플 넣기 <b>${fmt((A.samples||[]).length)}개</b> <span class="muted">(4. 신규 학습 데이터 표의 줄마다 '후보에' 버튼)</span></span>${(A.samples||[]).length?'<button class="sm" id="cvSampClr">샘플 모두 빼기</button>':""}</div>
  <div class="row"><label>외부 파일 (CSV: method, payload, label[, attack_type] · 또는 '|||' 로그 8필드)<input type="file" id="cvFile" accept=".csv,.log,.txt"></label><button class="sm" id="cvUp">올리기 (검사)</button></div>
  ${(A.manual||[]).length?`<table class="mini"><thead><tr><th>파일</th><th>형식</th><th class="n">줄</th><th class="n">넣을 줄</th><th class="n">공격 / 정상</th><th>중복 (기준 · 신규 · 파일 안)</th><th>올린 시각</th><th></th></tr></thead><tbody>${A.manual.map(m=>`<tr><td>${esc(m.orig||m.file)}</td><td>${m.format==="log"?"로그":"CSV"}</td><td class="n">${fmt(m.rows)}</td><td class="n"><b>${fmt(m.use)}</b></td><td class="n">${fmt((m.label||{}).anomalies)} / ${fmt((m.label||{}).normal)}</td><td class="muted">${fmt((m.dup||{})["기준"])} · ${fmt((m.dup||{})["신규"])} · ${fmt((m.dup||{})["파일 안"])}</td><td class="muted">${esc(isoClock(m.added_at||""))}</td><td><button class="sm" data-mrm="${esc(m.file)}">빼기</button></td></tr>`).join("")}</tbody></table>`:""}
  <div class="row" style="margin-top:6px"><button class="primary" id="cvExport"${run||exporting?" disabled":""}>다시 내보내기 (rev ${rev+1})</button><span class="muted" style="align-self:center">지금 선별 + 위의 데이터 추가로 학습 파일·로그를 다시 만든다 (지난 rev 의 테스트 결과는 '이전 rev' 로 남음)</span></div>`;
  top=h;h="";const ok=chk.complete&&chk.met&&sOk;     // 최종 승인은 서버 반영 칸 아래 (sCandFoot)
  h+=`<h3 style="margin:14px 0 4px">최종 승인</h3><div>${!chk.complete?`<span class="muted">rev ${rev} 의 IPS·WAF 테스트 결과(실험 ${esc(chk.exp||exp)})가 둘 다 있어야 승인합니다</span>`:chk.met?'<span class="chip c-OK">피처 테스트: 두 장비 모두 목표를 넘음</span>':'<span class="chip c-BAD">피처 테스트: 목표 미달</span>'}</div>${srvGate(sg,rev)}
  ${chk.complete&&!ok?'<div class="muted">그래도 승인하려면 사유를 적으세요 (manifest 에 남음)</div><textarea id="cvReason" placeholder="목표·서버 반영 기준에 못 미쳐도 승인하는 사유" style="width:100%;min-height:54px;margin-top:4px"></textarea>':""}
  <div class="row" style="margin-top:6px"><button class="primary" id="cvApprove"${!chk.complete||run||exporting||stale?" disabled":""}>최종 승인 (rev ${rev})</button><button class="danger" id="cvDiscard"${run||exporting?" disabled":""}>후보 버리기</button><span class="muted" style="align-self:center">승인하면 잠기고 다음 버전의 기준이 됩니다. 버리면 폴더를 _보관/ 으로 옮기고 이름을 다시 쓸 수 있습니다.</span></div>`}
 else{top=h;h="";const ap=x.approval||{};h+=`<h3 style="margin:14px 0 4px">최종 승인 기록</h3><div>${esc(isoClock(ap.at||""))} · rev ${esc(String(ap.rev??""))} · 실험 ${esc(ap.exp||"")} · ${ap.server?`${ap.targets_met?'<span class="chip c-OK">목표 넘음</span>':'<span class="chip c-BAD">목표 미달</span>'} · 서버 반영 ${ap.server.met?'<span class="chip c-OK">넘음</span>':'<span class="chip c-BAD">미달·결과 없음</span>'}${ap.met?"":` · 사유: ${esc(ap.reason||"")}`}`:ap.met?'<span class="chip c-OK">목표 넘음</span>':`<span class="chip c-BAD">목표 미달</span> 사유: ${esc(ap.reason||"")}`}</div>`}
 if(sCandMsg)h+=`<div class="muted" style="margin-top:6px">${esc(sCandMsg)}</div>`;
 el.innerHTML=top??h;$("sCandFoot").innerHTML=top==null?"":h}
function cB64(buf){let s="";const a=new Uint8Array(buf);for(let i=0;i<a.length;i+=0x8000)s+=String.fromCharCode.apply(null,a.subarray(i,i+0x8000));return btoa(s)}
async function cPost(url,body,msg){try{const r=await post(url,Object.assign({name:sCandName},body));sCandMsg=typeof msg==="function"?msg(r):(msg||"");await loadSup();loadCand();return r}catch(e){alert("실패: "+e.message);return null}}
$("sCandClose").onclick=()=>{sCandClosed=true;sCandName=null;clearTimeout(sCandT);clearTimeout(sSrvT);$("sCandCard").style.display="none"};
$("sCandBody").addEventListener("change",e=>{const c=e.target.closest("[data-cexp]");if(!c)return;const s=sCandExps||new Set([sCandD.default_exp]);if(c.checked)s.add(c.dataset.cexp);else s.delete(c.dataset.cexp);sCandExps=s});
const cvClick=async e=>{const el=e.target,x=SS.versions.find(v=>v.name===sCandName);if(!x)return;   // 후보 카드 본문·최종 승인(아래) 공통
 const rm=el.closest("a[data-arm]");if(rm){e.preventDefault();cPost("/api/sup/additions",{kind:rm.dataset.arm,values:[rm.dataset.val],action:"remove"},`'${rm.dataset.val}' 를 뺐습니다`);return}
 const add=el.closest("button[data-add]");if(add){const k=add.dataset.add,inp=$(k==="rules"?"cvRule":"cvType"),vals=inp.value.split(",").map(s=>s.trim()).filter(Boolean);if(!vals.length)return;
  cPost("/api/sup/additions",{kind:k,values:vals,action:"add"},`${k==="rules"?"룰":"공격 유형"} ${vals.length}개를 넣었습니다 - 다시 내보내면 들어갑니다`);return}
 const mr=el.closest("button[data-mrm]");if(mr){if(confirm("이 외부 파일을 뺍니다 (원본은 manual/_빠짐/ 으로 옮김). 진행할까요?"))cPost("/api/sup/manual/remove",{file:mr.dataset.mrm},"외부 파일을 뺐습니다");return}
 if(el.id==="cvSampClr"){if(confirm(`넣은 샘플 ${(x.additions.samples||[]).length}개를 모두 뺍니다. 진행할까요?`))cPost("/api/sup/additions",{kind:"samples",values:[],action:"set"},"샘플 넣기를 비웠습니다");return}
 if(el.id==="cvUp"){const f=$("cvFile").files[0];if(!f){alert("파일을 고르세요");return}const buf=await f.arrayBuffer();
  cPost("/api/sup/manual",{filename:f.name,data:cB64(buf)},r=>`'${f.name}' ${fmt(r.rows)}줄 중 넣을 줄 ${fmt(r.use)} (중복 기준 ${fmt(r.dup["기준"])} · 신규 ${fmt(r.dup["신규"])} · 파일 안 ${fmt(r.dup["파일 안"])}) - 다시 내보내면 들어갑니다`);return}
 if(el.id==="cvGo"){const devs=[["ips","cvIps"],["waf","cvWaf"]].filter(([,id])=>$(id).checked).map(([d])=>d);if(!devs.length){alert("장비를 고르세요");return}
  const exps=[...(sCandExps||new Set([sCandD.default_exp]))],quick=$("cvQuick").checked;
  if(!confirm(`rev ${x.rev} 피처 테스트를 시작합니다: ${devs.map(d=>d.toUpperCase()).join(" → ")} · 실험 ${exps.join(", ")}${quick?" · 빠른 확인 (표본)":""}.\n장비마다 기준 학습 파일 전체를 읽고 모델을 두 번(후보·기준만) 학습합니다 (CPU 를 많이 씀, 생성과 겹치면 느려짐). 진행할까요?`))return;
  cPost("/api/sup/eval/start",{devs,exps,quick},"테스트를 시작했습니다");return}
 if(el.id==="cvStop"){cPost("/api/sup/eval/stop",{},"중지를 요청했습니다 (지금 단계가 끝나면 멈춤, 끝난 실험 결과는 남음)");return}
 if(el.id==="cvSaveT"){const tg={exp:$("cvExp").value};for(const dv of["ips","waf"]){tg[dv]={};for(const m of CM3){const v=$("sCandBody").querySelector(`[data-tg="${dv}:${m}"]`).value.trim();tg[dv][m]=v===""?null:+v}
   tg[dv].not_below_base=$("sCandBody").querySelector(`[data-tg="${dv}:not_below_base"]`).checked}
  cPost("/api/sup/targets",{targets:tg},"목표를 저장했습니다");return}
 if(el.id==="cvExport"){const n=["rules","types","samples"].reduce((a,k)=>a+((x.additions||{})[k]||[]).length,0)+((x.additions||{}).manual||[]).length;
  if(!confirm(`'${x.name}' 을 다시 내보냅니다 (rev ${x.rev} → rev ${x.rev+1}, 데이터 추가 ${n}건 반영).\n학습 파일·로그를 새로 만들고 rev ${x.rev} 의 큰 파일은 남기지 않습니다 (목록·manifest 는 revs/rev${x.rev}/ 에, 테스트 결과는 '이전 rev' 로 남음). 진행할까요?`))return;
  try{await post("/api/sup/export",{name:x.name});sCandMsg=`rev ${x.rev+1} 내보내기를 시작했습니다`;setTimeout(loadSup,800)}catch(err){alert("실패: "+err.message)}return}
 if(el.id==="cvApprove"){const c=sCandD.check||{},why=($("cvReason")||{}).value||"",sg=sCandD.server||{},sBad=!!(sg.active&&!sg.met);
  if((!c.met||sBad)&&!why.trim()){alert((sBad?"서버 반영 기준(4지표 95%·예측 일치 98%)에 못 미치거나 서버 결과가 없습니다":"목표에 못 미칩니다")+" - 사유를 적어야 승인됩니다");return}
  const ms=["ips","waf"].map(dv=>{const z=((c.dev||{})[dv]||{}).metrics||{};return `${dv.toUpperCase()} P ${cPct(z.precision)} · R ${cPct(z.recall)} · F1 ${cPct(z.f1)}`}).join("\n");
  const sl=sg.active?"\n서버 반영: "+["ips","waf"].map(dv=>{const g=(sg.dev||{})[dv]||{};return `${dv.toUpperCase()} ${g.met?"넘음":"미달 ("+(g.fails||[]).join(", ")+")"}`}).join(" · "):"";
  if(!confirm(`'${x.name}' rev ${x.rev} 를 최종 승인합니다 (실험 ${c.exp}).\n${ms}${sl}${c.met&&!sBad?"":"\n미달 - 사유: "+why}\n승인하면 잠겨 고칠 수 없고, 다음 버전의 기준이 됩니다. 진행할까요?`))return;
  cPost("/api/sup/approve",{rev:x.rev,reason:why,exp:c.exp},"최종 승인했습니다");return}
 if(el.id==="cvDiscard"){if(!confirm(`후보 '${x.name}' 을 버립니다. 폴더(학습 파일·테스트 결과·외부 파일)를 _보관/ 으로 옮기고 목록에서 뺍니다 (이름을 다시 쓸 수 있음). 진행할까요?`))return;
  const r=await cPost("/api/sup/discard",{},"");if(r){sCandName=null;$("sCandCard").style.display="none";$("sSaveMsg").textContent=`후보 '${x.name}' 을 버렸습니다 (${r.moved_to||"폴더 없음"})`}}};
$("sCandBody").addEventListener("click",cvClick);$("sCandFoot").addEventListener("click",cvClick);
/* ── 6. 후보 '서버 반영' (2026-11-04~, 5_8_server.py · job server): 올릴 파일·레시피 사본(로컬 전용 5_8a)·로컬 사전 점검(95%·98% 미리 보기)·
   서버 결과(4지표·예측 일치)·체크리스트. 승인 조건은 5_8 gate (renderCand 의 srvGate) ── */
let sSrvD=null,sSrvT=0,sSrvRun=false,sSrvMsg="",sSrvDirty={};   // sSrvDirty: 입력 중인 값 (3초마다 새로 그려도 남김, 저장하면 지움)
const svChip=(ok,t)=>ok==null?"":ok?`<span class="chip c-OK">${t||"넘음"}</span>`:`<span class="chip c-BAD">${t||"미달"}</span>`;
const svTop=(o,k=6)=>Object.entries(o||{}).slice(0,k).map(([a,b])=>`${esc(a)} ${fmt(b)}`).join(", ")||"-";
const svM4=m=>m?`정확도 ${cPct(m.accuracy)} · P ${cPct(m.precision)} · R ${cPct(m.recall)} · F1 ${cPct(m.f1)}`:"-";
const svDl=f=>`/download/sup/server?name=${encodeURIComponent(sCandName||"")}&file=${encodeURIComponent(f||"")}`;
async function loadSrv(force=false){clearTimeout(sSrvT);const el=$("sSrvBody");if(!sCandName){el.innerHTML="";return}const name=sCandName;let d;
 try{d=await get("/api/sup/server?name="+encodeURIComponent(name))}catch(e){if(name===sCandName)el.innerHTML=`<h3 style="margin:14px 0 4px">서버 반영</h3><div class="muted">${esc(e.message)}</div>`;return}
 if(name!==sCandName)return;if(!sSrvD||sSrvD.name!==name)sSrvDirty={};sSrvD=d;const run=!!(d.job&&d.job.running),a=document.activeElement;
 if(force||!(run&&a&&el.contains(a)&&a.tagName==="INPUT"))renderSrv();      // 점검 중 3초마다 새로 그릴 때 입력 중인 칸은 두고
 if(run)sSrvT=setTimeout(loadSrv,3000);if(sSrvRun&&!run)loadCand();sSrvRun=run}
function srvGate(sg,rev){if(!sg||!sg.active)return sg&&sg.error?`<div class="muted">서버 반영 조건을 읽지 못함: ${esc(sg.error)}</div>`:"";
 return `<div>서버 반영 (rev ${rev} · 4지표 95% · ② 예측 일치 98%): ${["ips","waf"].map(dv=>{const g=(sg.dev||{})[dv]||{};return `${dv.toUpperCase()} ${svChip(g.met)}${g.met?"":` <span class="muted">${esc((g.fails||[]).join(" · "))}</span>`}${!g.met&&g.f1_ok?' <span class="muted">(f1 만 보면 95% 이상 - 지난번 통과 방식, 승인은 사유로)</span>':""}${g.impl!=null?` <span class="muted">(① 다른 칸 ${fmt(g.impl)})</span>`:""}`}).join(" · ")}</div>`}
function renderSrv(){const el=$("sSrvBody"),d=sSrvD;if(!d||d.name!==sCandName){el.innerHTML="";return}
 const L=d.local||{},run=!!(d.job&&d.job.running),rs=d.run||{},fin=d.status==="final";
 let h=`<h3 style="margin:14px 0 4px">서버 반영 <span class="muted" style="font-weight:400;font-size:12.5px">rev ${d.rev} · 서버에서 IPS·WAF 각각 직접 진행 · 학습 피처 = 디코딩 입력 · 기준: 혼동행렬 4지표(정확도·precision·recall·f1, 공격 기준) 모두 95% 이상 · ② 디코딩 입력·원본 입력 예측 일치 98% 이상 · ① 구현 일치(같은 원본에서 파이썬 피처 = SQL 피처)</span></h3>`;
 h+=`<div class="muted" style="font-size:12.5px">${L.ok?`서버 레시피 ${(L.files||[]).length}개 읽음 (12_server/recipes, 로컬 전용) · 사본 규칙 ${esc(L.rules_version||"")}`:L.present?`서버 레시피를 읽지 못함: ${esc(L.error||("없는 파일 "+(L.missing||[]).join(", ")))}`:`로컬 전용 ${esc(d.local_file)} 가 없어 레시피 사본·디코딩 점검·승인 조건은 꺼져 있습니다 (올릴 파일·원본 피처 미리 보기·결과 기록만)`}</div>`;
 const pv=d.prev_server;if(pv&&pv.dev)h+=`<div style="font-size:12.5px;margin-top:4px"><b>지난 서버 결과 (기준 '${esc(pv.version)}', 참고)</b>: ${["ips","waf"].filter(dv=>pv.dev[dv]).map(dv=>`${dv.toUpperCase()} ${svM4(pv.dev[dv])} ${svChip(!["accuracy","precision","recall","f1"].some(k=>(pv.dev[dv][k]??0)<.95),"4지표")} ${svChip((pv.dev[dv].f1??0)>=.95,"f1")}`).join(" · ")} <span class="muted">${esc(pv.note||"")}</span></div>`;
 const gr=d.grid||{};h+=`<div class="muted" style="font-size:12.5px">서버 Train: 열 = 피처 + payload (method 안 씀 - 이 PC 는 payload 를 학습 피처로 안 쓰는 것으로 봄) · RF 트리 수 ${esc((gr.trees||[]).join("·"))} × 최대 깊이 ${esc((gr.depth||[]).join("·"))} 그리드 · ② 는 Score 레시피 결과 둘 비교 (저장 모델로 직접 예측은 안 됨)</div>`;
 const nn=((((d.dev||{}).ips||{}).upload||{}).checks||{}).normal_rows;
 h+=`<div class="msg" style="border-left:4px solid var(--bad);margin:6px 0"><b>레시피 1 (decode) 주의</b>: 원래 레시피 1 은 label 을 모두 '1'(공격)로 덮어 <b>정상 행이 공격이 됩니다</b>${nn!=null?` (이번 추가 샘플의 정상 ${fmt(nn)}행)`:""}. 첨부 형식 열 빼기(열이 없어 오류)·4월 첨부 기준 행 번호 빼기(엉뚱한 행이 빠짐)도 우리 파일에 맞지 않습니다 → <b>레시피 사본 1</b> 을 쓰세요. 원래 레시피 1 을 꼭 써야 하면 아래에서 '공격 행만 올리기'를 고르세요 (정상 행은 서버 데이터에 안 들어감).</div>`;
 const vI=(d.versions||{}).ips||[],vW=(d.versions||{}).waf||[],vAll=[...new Map([...vI,...vW].map(x=>[x.ver,x])).values()],both=v=>vI.some(x=>x.ver===v)&&vW.some(x=>x.ver===v);
 h+=`<div class="row" style="margin-top:6px"><label>서버 데이터 이름 날짜<input id="svTag" value="${esc(d.tag)}" maxlength="6" style="width:80px"></label><span class="muted" style="align-self:center">기준 ${esc(d.base_tag||"-")}</span>
  <label>올릴 행<select id="svLab">${Object.entries(d.label_modes||{}).map(([k,t])=>`<option value="${k}"${k===d.labels?" selected":""}>${esc(t)}</option>`).join("")}</select></label>
  <label>NUL(0x00) 이 든 샘플<select id="svNul">${Object.entries(d.nul_modes||{}).map(([k,t])=>`<option value="${k}"${k===d.nul?" selected":""}>${esc(t)}</option>`).join("")}</select></label>
  <label title="피처 세션이 만드는 보강판(12_server/features/out)이 IPS·WAF 둘 다 있어야 고를 수 있음 - 레시피 사본 3·3r·4 와 로컬 사전 점검에 같이 씀">피처 정의 판<select id="svVer">${vAll.map(x=>`<option value="${esc(x.ver)}"${x.ver===d.ver?" selected":""}${both(x.ver)?"":" disabled"}>${esc(x.ver)} - ${esc(x.desc||"")}${both(x.ver)?"":` (${vI.some(y=>y.ver===x.ver)?"IPS":"WAF"} 만)`}</option>`).join("")}</select></label>
  <button class="primary" id="svPrep"${run?" disabled":""}>${d.prepared?"다시 만들기":"올릴 파일·레시피 사본 만들기"} (rev ${d.rev})</button>${d.prepared?`<span class="muted" style="align-self:center">${esc(isoClock(d.prepared_at||""))} 만듦 · 판 ${esc(d.ver)} · <a href="/download/sup/server/bundle?name=${encodeURIComponent(d.name)}" title="사본·올릴 파일·피처 사전·SQL·점검 결과·서버 기록을 설명(README_설명.md)과 함께 zip 하나로 - 회사 내부용">넘겨줄 묶음 받기 (zip)</a></span>`:""}</div>`;
 const stl=["ips","waf"].filter(dv=>(d.stale||{})[dv]);if(stl.length)h+=`<div class="msg">${stl.map(v=>v.toUpperCase()).join("·")} 추가 샘플 목록이 파일을 만든 뒤 바뀌었습니다 - 다시 만드세요</div>`;
 h+=`<div class="row"><label class="inl"><input type="checkbox" id="svIps" checked> IPS</label><label class="inl"><input type="checkbox" id="svWaf" checked> WAF</label>
  <label class="inl" title="서버 Train 처럼 RF 트리 수 × 최대 깊이 그리드에서 검증 f1 이 가장 좋은 조합을 고름 (끄면 RF 100 트리·깊이 제한 없음 - 빠른 확인)"><input type="checkbox" id="svGrid" checked> 그리드 (서버처럼)</label>
  <label title="그리드(조합 고르기)에 쓸 행 - 층화 표본의 80/20. 고른 조합으로 다시 학습할 때는 늘 전체 행">그리드 행<select id="svGrows"><option value="${gr.rows||150000}">${fmt(gr.rows||150000)} 표본 (기본)</option><option value="0">전체 (오래 걸림)</option></select></label>
  <button id="svChk"${run?" disabled":""}>로컬 사전 점검 시작 (판 위에서 고른 것)</button>${run?'<button class="danger" id="svStop">중지</button>':""}
  <span class="muted" style="align-self:center;font-size:12.5px">학습 파일 전체(올릴 파일에서 뺀 행 빼고)를 레시피 1 로 디코딩해 피처를 만들고, 그리드로 고른 RF 로 전체 학습 → 학습분 안 4지표·② 예측 일치 (처음은 오래 걸리고, 계산한 피처는 캐시에 남아 다음부터 빠름)</span></div>`;
 if(run)h+=`<div><b>점검 중</b>${rs.version&&rs.version!==d.name?` (다른 버전 '${esc(rs.version)}' - 끝나야 시작·다시 만들기 가능)`:""} ${esc(String(rs.dev||"").toUpperCase())} · ${esc(rs.step_text||"시작하는 중")} · ${Math.round(rs.pct||0)}%${rs.eta_s!=null?` · 남은 약 ${hms(rs.eta_s)}`:""}</div>`;
 else if(rs.status&&rs.version===d.name&&rs.status!=="done")h+=`<div class="muted">지난 점검: ${({stopped:"중지됨",error:"실패",running:"끊김"})[rs.status]||esc(rs.status)}${rs.error?` · ${esc(rs.error)}`:""}</div>`;
 for(const dv of["ips","waf"])h+=srvDev(d,dv);
 if(sSrvMsg)h+=`<div class="muted" style="margin-top:6px">${esc(sSrvMsg)}</div>`;
 el.innerHTML=h;
 for(const [k,v] of Object.entries(sSrvDirty)){const i=el.querySelector(`[data-sv="${k}"],[data-svmemo="${k}"]`)||(/^sv[A-Z]/.test(k)?$(k):null);if(i)i.value=v}}
function srvDev(d,dv){const x=(d.dev||{})[dv]||{},u=x.upload,cp=x.copies,ck=x.check,s=x.server||{},v=x.verdict||{},U=dv.toUpperCase();
 let h=`<div style="border-top:1px solid var(--line);margin-top:10px;padding-top:6px"><b>${U}</b> `;
 if(u){const c=u.checks||{},ex=Object.entries(u.excluded||{});h+=`올릴 파일 <a href="${svDl(u.file)}">${esc(u.file)}</a> ${fmt(u.rows)}행 (공격 ${fmt((u.label||{}).anomalies)} · 정상 ${fmt((u.label||{}).normal)})${ex.length?` <b>· 뺀 행 ${ex.map(([k,n])=>`${esc(k)} ${fmt(n)}`).join(", ")}</b> (추가 샘플 ${fmt(u.added_rows)} 중)`:""} <span class="muted">· ${fmt(Math.round((u.bytes||0)/1024))}KB · sha256 ${esc((u.sha256||"").slice(0,12))}…</span>
  <div class="muted" style="font-size:12.5px">검사: NUL ${fmt(c.nul_rows)}${u.nul_replaced?` (%00 으로 바꿈 ${fmt(u.nul_replaced)})`:""} · 빈 payload ${fmt(c.empty_payload)} · 빈 method ${fmt(c.empty_method)}${c.empty_method?" (서버가 '-' 로 바꿈)":""} · 값 안 줄바꿈 ${fmt(c.newline_rows)}행 · 큰따옴표 ${fmt(c.quote_rows)}행 · SQL 만 줄바꿈으로 보는 글자 ${fmt(c.sql_break_rows)}행</div>`;
  h+=cp?`<div>레시피 사본 <a href="${svDl(cp.zip)}">${esc(cp.zip)}</a> <span class="muted" style="font-size:12.5px">(판 ${esc(cp.ver||"std")} · ${(cp.files||[]).length}개 - 서버 Python 레시피에 붙여 넣기, 순서·데이터셋 이름은 README)</span></div>`:""}
 else h+=`<span class="muted">아직 올릴 파일을 만들지 않았습니다</span>`;
 if(ck){const ins=ck.insample||{},ho=ck.holdout,ag=ck.agree,im=ck.impl,f95=ins.fails||[];
  h+=`<div style="margin-top:4px"><b>로컬 사전 점검</b> <span class="muted" style="font-size:12.5px">${esc(isoClock(ck.at||""))} · ${fmt(Math.round(ck.seconds||0))}초 · 판 ${esc(ck.feature_ver||"std")} · 학습 피처 ${ck.basis==="decoded"?"디코딩 입력":"원본 입력"} · ${fmt(ck.rows_used??ck.rows)}행 (추가 ${fmt(ck.added_used??ck.added_rows)}${ck.excluded?`, 올릴 파일에서 뺀 ${fmt(ck.excluded)}행 뺌`:""}) · ${esc(ck.model||"")}</span></div>`;
  if(ck.basis==null)h+=`<div class="msg">예전 방식(원본 피처로 학습) 점검 결과입니다 - 지금 기준(디코딩 피처로 학습)으로 다시 점검하세요</div>`;
  const f1r=ins.all&&ins.all.f1!=null?ins.all.f1>=.95:null;
  h+=`<div>95% 미리 보기 (학습분 안 = 서버 레시피 6 방식, 전체): ${svM4(ins.all)} ${svChip(!f95.length,!f95.length?"4지표 넘음":"4지표 미달")}${f95.length?` <span class="muted">${esc(f95.join(" · "))}</span>`:""} ${svChip(f1r,f1r?"참고 f1 ≥95% 넘음":"참고 f1 미달")}</div>`;
  if(ins.base&&ins.added)h+=`<div class="muted" style="font-size:12.5px">참고 - 기준만 ${svM4(ins.base)} · 추가만 ${svM4(ins.added)}</div>`;
  const gd=ck.grid;if(gd&&gd.best)h+=`<div style="font-size:12.5px">그리드: 고른 조합 <b>트리 ${fmt(gd.best.trees)} · 깊이 ${fmt(gd.best.depth)}</b> <span class="muted">(검증 ${esc(gd.metric||"f1")} 가장 높음, 그리드 ${fmt(gd.rows)}행${gd.sampled?" 표본":""}의 80/20)</span> <details style="display:inline"><summary>12 조합 보기</summary><table class="mini"><thead><tr><th class="n">트리</th><th class="n">깊이</th><th class="n">정확도</th><th class="n">P</th><th class="n">R</th><th class="n">F1</th><th class="n">추가 행 F1</th></tr></thead><tbody>${(gd.table||[]).map(r=>`<tr${r.trees===gd.best.trees&&r.depth===gd.best.depth?' class="hl"':""}><td class="n">${r.trees}</td><td class="n">${r.depth}</td>${["accuracy","precision","recall","f1","added_f1"].map(k=>`<td class="n">${cPct(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></details></div>`;
  if(ho)h+=`<div class="muted" style="font-size:12.5px">참고 검증분 (${esc(ho.note||"5-fold 한 fold")}, ${fmt(ho.test_rows)}행): ${svM4(ho.all)}${(ho.fails||[]).length?` · 95% 아래: ${esc(ho.fails.join(" · "))}`:""} · 그중 추가 행 ${svM4(ho.added)}</div>`;
  if(ag&&ag.all){const ca=(ag.causes||{}).added||{},cb=(ag.causes||{}).base||{};
   h+=`<div>② 98% 미리 보기 (같은 모델의 디코딩 입력 vs 원본 입력 예측 일치, 전체 ${fmt(ag.all.rows)}행): <b>${cPct(ag.all.rate)}</b> ${svChip(ag.a98)} · 기준 ${cPct(ag.base.rate)} · 추가 ${fmt(ag.added.rows)}행 ${cPct(ag.added.rate)} <span class="muted">(추가만은 참고)</span></div>`;
   if(ag.added.disagree)h+=`<div class="muted" style="font-size:12.5px">② 가 다른 추가 샘플 ${fmt(ag.added.disagree)}개 (원본 공격→디코딩 정상 ${fmt(ag.added.raw_attack_dec_normal)} · 원본 정상→디코딩 공격 ${fmt(ag.added.raw_normal_dec_attack)}) - 원인: 피처 ${svTop(ca.features)} · 인코딩 ${svTop(ca.encodings)} · 룰 ${svTop(ca.rules,5)} · 유형 ${svTop(ca.types,5)} · <a href="${svDl(ag.diff_file)}">샘플 목록 CSV</a> <span title="디코딩/원본 예측이 다른 신규 샘플을 화면 6 에서 빼거나 표시할지는 사용자 결정 전 - 지금은 목록(sk)만">(빼기·표시는 결정 뒤)</span></div>`;
   if(ag.base.disagree)h+=`<div class="muted" style="font-size:12.5px">기준 행에서 다른 ${fmt(ag.base.disagree)}개: 피처 ${svTop(cb.features,4)} · 인코딩 ${svTop(cb.encodings,4)}</div>`}
  else if(ck.decode)h+=`<div class="muted" style="font-size:12.5px">예전 점검의 98% 미리 보기: 전체 추정 ${cPct(ck.decode.overall_est)}</div>`;
  if(im)h+=`<div class="muted" style="font-size:12.5px">① 구현 일치 미리 보기 (같은 원본, 파이썬 vs SQL 줄바꿈 처리 ${esc((im.chars||[]).join("·")||"차이 없음")}): 그 글자가 든 행 ${fmt(im.rows)} · 피처 다른 행 ${fmt(im.feat_diff_rows)} · 예측 다른 행 ${fmt(im.pred_diff_rows)} (Java 정규식 차이는 서버 5 에서)</div>`;
  if(ck.note)h+=`<div class="muted">${esc(ck.note)}</div>`;
  if((ck.upload_match||{}).diff)h+=`<div class="msg">추가 행 ${fmt(ck.upload_match.diff)}개가 학습 파일의 추가 행과 다릅니다 - 다시 내보내세요</div>`}
 const sm=s.metrics,sa=s.agree,si=s.impl;
 h+=`<div style="margin-top:6px"><b>서버 결과</b> ${sm?`${svM4(sm)} ${svChip(v.m95,v.m95?"4지표 넘음":"4지표 미달")} ${svChip(v.f1_ok,v.f1_ok?"참고 f1 ≥95% 넘음":"참고 f1 미달")} <span class="muted" style="font-size:12.5px">(TP ${fmt(sm.tp)} · FP ${fmt(sm.fp)} · TN ${fmt(sm.tn)} · FN ${fmt(sm.fn)} · ${esc(isoClock(sm.at||""))})</span> <a href="#" data-svclr="${dv}:metrics">지우기</a>`:'<span class="muted">4지표 없음</span>'} · ② 예측 일치 ${sa?`<b>${cPct(sa.value)}</b> ${svChip(v.a98)} <span class="muted" style="font-size:12.5px">${esc(isoClock(sa.at||""))}</span> <a href="#" data-svclr="${dv}:agree">지우기</a>`:'<span class="muted">없음</span>'} · ① 다른 칸 ${si?`<b>${fmt(si.diff_cells)}</b>${si.diff_rows!=null?` (행 ${fmt(si.diff_rows)}${si.rate!=null?` · 같은 행 ${cPct(si.rate)} ${svChip(v.impl_ok,v.impl_ok?"참고 99.8% 넘음":"참고 99.8% 미달")}`:""})`:""} <a href="#" data-svclr="${dv}:impl">지우기</a>`:'<span class="muted">없음</span>'}</div>`;
 h+=`<div class="row"><span style="align-self:center">6 출력</span>${["tp","fp","tn","fn"].map(k=>`<label>${k.toUpperCase()}<input data-sv="${dv}:${k}" style="width:84px" inputmode="numeric"></label>`).join("")}<button class="sm" data-svm="${dv}">4지표 저장</button>
  <label>5b 출력 '예측 일치 (전체)' %<input data-sv="${dv}:agree" style="width:70px"></label><button class="sm" data-sva="${dv}">② 저장</button></div>
  <div class="row"><span style="align-self:center">5 출력 (①)</span><label>다른 칸<input data-sv="${dv}:icells" style="width:70px" inputmode="numeric"></label><label>다른 행 (선택)<input data-sv="${dv}:irows" style="width:70px" inputmode="numeric"></label><label>전체 행 (선택)<input data-sv="${dv}:itotal" style="width:84px" inputmode="numeric"></label><button class="sm" data-svi="${dv}">① 저장</button></div>`;
 h+=`<div class="row"><label>서버 틀린 행 CSV (6 출력)<input type="file" data-svf="${dv}:wrong_csv" accept=".csv,.tsv,.txt"></label><label>예측 다른 행 CSV (5b 출력)<input type="file" data-svf="${dv}:diff_csv" accept=".csv,.tsv,.txt"></label><button class="sm" data-svup="${dv}">올리기 (우리 샘플과 짝짓기)</button></div>`;
 for(const [k,t] of [["wrong_csv","서버 틀린 행"],["diff_csv","서버 예측 다른 행"]]){const w=s[k];if(w)h+=`<div class="muted" style="font-size:12.5px">${t} ${fmt(w.rows)}행 (${esc(w.filename||"")}, ${esc(isoClock(w.at||""))}): 추가 샘플 ${fmt(w.added_rows)} · 기준 ${fmt(w.base_rows)} · 룰 ${svTop(w.by_rule,6)} · 유형 ${svTop(w.by_type,5)} <a href="#" data-svclr="${dv}:${k}">지우기</a></div>`}
 const st=x.steps||[],nd=st.filter(z=>z.done).length;
 h+=`<details${nd&&nd<st.length?" open":""}><summary>서버 단계 체크리스트 ${nd}/${st.length}</summary><table class="mini"><tbody>${st.map(z=>`<tr><td><input type="checkbox" data-svs="${dv}:${esc(z.key)}"${z.done?" checked":""}></td><td class="nw">${esc(z.title)}</td><td class="reason">${esc(z.detail||"")}</td><td class="muted nw">${esc(isoClock(z.at||""))}</td><td><input data-svmemo="${dv}:${esc(z.key)}" value="${esc(z.memo||"")}" placeholder="메모 (Enter 로 저장)" style="width:170px"></td></tr>`).join("")}</tbody></table></details>`;
 return h+`</div>`}
async function svPost(url,body,msg,done=[]){try{const r=await post(url,Object.assign({name:sCandName},body));for(const k of done)delete sSrvDirty[k];
  sSrvMsg=typeof msg==="function"?msg(r):(msg||"");await loadSrv(true);loadCand();return r}catch(e){alert("실패: "+e.message);return null}}
$("sSrvBody").addEventListener("input",e=>{const t=e.target,k=t.dataset.sv||t.dataset.svmemo||(/^sv[A-Z]/.test(t.id||"")?t.id:"");if(k)sSrvDirty[k]=t.value});
$("sSrvBody").addEventListener("click",async e=>{const el=e.target,d=sSrvD;if(!d)return;
 const clr=el.closest("a[data-svclr]");if(clr){e.preventDefault();const [dv,what]=clr.dataset.svclr.split(":");if(confirm(`${dv.toUpperCase()} 서버 결과(${what})를 지웁니다. 진행할까요?`))svPost("/api/sup/server/result",{dev:dv,kind:"clear",what},"지웠습니다");return}
 if(el.id==="svPrep"){const tag=$("svTag").value.trim(),nul=$("svNul").value,labels=$("svLab").value,ver=$("svVer").value;
  if(!confirm(`rev ${d.rev} 의 추가 샘플로 서버에 올릴 파일(IPS·WAF)${(d.local||{}).ok?"과 레시피 사본":""}을 만듭니다 (데이터 이름 날짜 ${tag}, 기준 ${d.base_tag}, 피처 정의 판 ${ver}).\n올릴 행: ${(d.label_modes||{})[labels]||labels} · NUL: ${(d.nul_modes||{})[nul]||nul}${d.prepared?"\n이미 만든 파일은 새로 바뀝니다 (서버 결과·체크리스트 기록은 남음).":""}\n진행할까요?`))return;
  svPost("/api/sup/server/prepare",{tag,nul,labels,ver},r=>"만들었습니다: "+["ips","waf"].map(v=>`${v.toUpperCase()} ${r.dev[v].upload} ${fmt(r.dev[v].rows)}행${Object.keys(r.dev[v].excluded||{}).length?" (뺀 행 "+Object.entries(r.dev[v].excluded).map(([k,n])=>k+" "+fmt(n)).join(", ")+")":""}${r.dev[v].copies?" + "+r.dev[v].copies:""}`).join(" · "),["svTag","svNul","svLab","svVer"]);return}
 if(el.id==="svChk"){const devs=[["ips","svIps"],["waf","svWaf"]].filter(([,id])=>$(id).checked).map(([v])=>v);if(!devs.length){alert("장비를 고르세요");return}
  const grid=$("svGrid").checked,grows=+$("svGrows").value,ver=$("svVer").value;
  if(!confirm(`rev ${d.rev} 로컬 사전 점검을 시작합니다: ${devs.map(v=>v.toUpperCase()).join(" → ")} · 피처 정의 판 ${ver} · ${grid?`그리드 (트리 수 × 깊이 12 조합, ${grows?fmt(grows)+"행 표본":"전체 행"})`:"그리드 없이 RF 100 트리"}.\n장비마다 학습 파일 전체(약 40만 행)를 디코딩해 피처를 만들고 RF 를 학습합니다 (CPU 를 많이 씀 - 처음은 장비마다 십수 분 이상, 피처 캐시 뒤에는 그리드·학습 시간만). 진행할까요?`))return;
  svPost("/api/sup/server/check",{devs,grid,grid_rows:grows,ver},"사전 점검을 시작했습니다");return}
 if(el.id==="svStop"){svPost("/api/sup/server/check/stop",{},"중지를 요청했습니다 (지금 단계가 끝나면 멈춤)");return}
 const g=(dv,k)=>($("sSrvBody").querySelector(`[data-sv="${dv}:${k}"]`)||{}).value||"";
 const bm=el.closest("button[data-svm]");if(bm){const dv=bm.dataset.svm;
  svPost("/api/sup/server/result",{dev:dv,kind:"metrics",tp:g(dv,"tp"),fp:g(dv,"fp"),tn:g(dv,"tn"),fn:g(dv,"fn")},r=>`${dv.toUpperCase()} 서버 4지표: ${svM4(r)}`,
   ["tp","fp","tn","fn"].map(k=>`${dv}:${k}`));return}
 const ba=el.closest("button[data-sva]");if(ba){const dv=ba.dataset.sva;
  svPost("/api/sup/server/result",{dev:dv,kind:"agree",value:g(dv,"agree")},r=>`${dv.toUpperCase()} ② 예측 일치 ${cPct(r.value)} 저장`,[`${dv}:agree`]);return}
 const bi=el.closest("button[data-svi]");if(bi){const dv=bi.dataset.svi;
  svPost("/api/sup/server/result",{dev:dv,kind:"impl",diff_cells:g(dv,"icells"),diff_rows:g(dv,"irows"),total_rows:g(dv,"itotal")},r=>`${dv.toUpperCase()} ① 다른 칸 ${fmt(r.diff_cells)} 저장`,
   ["icells","irows","itotal"].map(k=>`${dv}:${k}`));return}
 const bu=el.closest("button[data-svup]");if(bu){const dv=bu.dataset.svup;let n=0;
  for(const k of["wrong_csv","diff_csv"]){const inp=$("sSrvBody").querySelector(`[data-svf="${dv}:${k}"]`),f=inp&&inp.files[0];if(!f)continue;
   if(f.size>50*2**20){alert(`${f.name}: 50MB 를 넘습니다`);continue}n++;
   await svPost("/api/sup/server/result",{dev:dv,kind:k,filename:f.name,data:cB64(await f.arrayBuffer())},r=>`${f.name}: ${fmt(r.rows)}행 중 우리 추가 샘플 ${fmt(r.added_rows)}`)}
  if(!n)alert("파일을 고르세요");return}});
$("sSrvBody").addEventListener("change",e=>{const c=e.target.closest("input[data-svs]");if(!c)return;const [dv,key]=c.dataset.svs.split(":"),memo=($("sSrvBody").querySelector(`[data-svmemo="${dv}:${key}"]`)||{}).value||"";
 svPost("/api/sup/server/step",{dev:dv,key,done:c.checked,memo},"",[`${dv}:${key}`])});
$("sSrvBody").addEventListener("keydown",e=>{const m=e.target.closest("input[data-svmemo]");if(!m||e.key!=="Enter")return;const [dv,key]=m.dataset.svmemo.split(":"),c=$("sSrvBody").querySelector(`[data-svs="${dv}:${key}"]`);
 svPost("/api/sup/server/step",{dev:dv,key,done:!!(c&&c.checked),memo:m.value},"메모를 저장했습니다",[`${dv}:${key}`])});
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
for(const id of["oBad","oLlm","oRule","oPy"])$(id).onchange=()=>sPreviewSoon();
for(const k of["A","B","C"]){const s=$("o"+k),n=$("o"+k+"n");   // 패턴당 직접 입력: 칸을 보이고 (비었으면 10) 바로 다시 셈
 s.onchange=()=>{if(s.value==="custom"&&n.value==="")n.value=10;sOptUi();if(s.value==="custom")n.focus();sPreviewSoon()};n.oninput=()=>sPreviewSoon(600)}
$("oT").oninput=()=>{sOptUi();sPreviewSoon(600)};
let sRq;$("sRuleQ").oninput=()=>{clearTimeout(sRq);sRq=setTimeout(renderRules,200)};$("sRuleF").onchange=renderRules;
$("sSave").onclick=async()=>{const name=$("sVName").value.trim();if(!name){$("sSaveMsg").textContent="버전 이름을 넣으세요 (예: 26 하반기)";return}
 if(SS.versions.some(v=>v.name===name&&v.status==="draft")&&!confirm(`초안 '${name}' 을 지금 선별로 덮어씁니다. 진행할까요?`))return;
 if(SS.versions.some(v=>v.name===name&&v.status==="candidate")&&!confirm(`후보 '${name}' 의 선별을 지금 선별로 바꿉니다. rev·데이터 추가·목표는 그대로이고, '다시 내보내기'(rev+1)를 해야 학습 파일·테스트에 들어갑니다. 진행할까요?`))return;
 try{const r=await post("/api/sup/save",Object.assign({name,base:SS.base},sSpec()));sSetBase(r);sRestored="";sWorkDrop();
  $("sSaveMsg").innerHTML=`<b>${esc(isoClock(r.updated_at))} '${esc(name)}' 에 담았습니다</b> · IPS·WAF 같은 샘플 ${fmt(r.draft_counts.ips.sel)}개 → 아래 버전 표의 강조한 줄에서 확인 후 확정`;loadSup()}catch(e){$("sSaveMsg").textContent="실패: "+e.message}};
$("sState").addEventListener("click",e=>{if(e.target.closest("[data-ssave]"))$("sSave").click();else if(e.target.closest("[data-sdiscard]"))sDiscard()});
$("sVName").addEventListener("input",sState);
/* ── 주기적 갱신 ── */
let lastRunning=false,ticks=0,seenCollect=null,autoInput=null;
async function tick(){try{await loadState();const running=S.generation.running;
 if((lastRunning&&!running)||(running&&++ticks%20===0)){await post("/api/progress/refresh");setTimeout(()=>{loadProgress();if(page==="rules")loadRules()},4000)}
 // 수집(화면·에이전트)이 끝났거나 진행 현황이 최신 입력 날짜로 계산되지 않았으면 다시 계산 (입력 날짜마다 한 번)
 const lc=(S.last_collect||{}).last_run_at||"",newest=(S.input_dates||[]).slice(-1)[0];
 if((seenCollect!==null&&lc!==seenCollect)||(P&&newest&&P.input_date!==newest&&autoInput!==newest)){autoInput=newest;refreshProgressWait()}
 seenCollect=lc;
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
