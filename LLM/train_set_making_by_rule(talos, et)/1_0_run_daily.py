"""[1.0 진입점] 하루 1회 실행용: 아래 단계를 번호 순서대로 실행한다.  (--agent 면 매일 정해진 시각에 반복)

[실행 순서 - 파일 이름의 번호 = 순서]
  1.0  1_0_run_daily.py         ← 이 파일. 아래 1.1 ~ 1.4 를 차례로 실행
  1.1  1_1_fetch_rules.py       ← 수집: 룰 압축 파일 다운로드 (바뀐 것만)              → 01_raw/
  1.1a 1_1a_fetch_custom.py    ← (선택, 공개본에는 없음) 사용자 룰 수집 → 00_custom_rules/*.rules
  1.2  1_2_parse_rules.py       ← 파싱: 압축 해제·선별·HTTP 필터·변경분·통합·중복 제거 → 02_extracted/ ~ 04_integrated/
  1.2a 1_2a_rule_selection.py   ←   (1.2 보조) docs/코드참조/선별조건.md 를 읽는 로더. 직접 실행하지 않음
  1.3  1_3_build_llm_input.py   ← LLM 입력: 학습데이터 생성에 필요한 필드만 추출·배치 분할 → 05_llm_input/
       (LLM 데이터 생성은 3.0 을 따로 실행: 05_llm_input 배치 → 07_llm_output/<날짜>/*.jsonl)
  1.4  1_4_build_dataset.py     ← 데이터셋: LLM 응답 → IPS·WAF(8필드) 수집 형식 로그 + 줄별 label → 08_dataset/
  2.0  2_0_register_task.ps1    ← (선택) 1.0 을 매일 자동 실행하도록 작업 스케줄러에 등록 (에이전트 대신 쓸 수 있음)
  3.0  3_0_run_generation.py    ← (따로 실행, API 비용) LLM 데이터 생성 진입점. 자세한 순서는 3_0_run_generation.py 머리 설명
  (화면) 0_0_run_ui.py          ← 수집 에이전트 시작·중지, 학습데이터 생성, 진행률을 화면에서

- 각 단계를 별도 프로세스로 실행하므로, 한 단계의 오류가 다른 단계 코드에 영향을 주지 않는다.
- 앞 단계가 일부 실패해도 다음 단계는 실행한다. (이미 있는 결과 중 아직 처리 안 된 것을 처리)
- 각 단계는 바뀐 것이 없으면 스스로 건너뛴다.
- 수집은 한 번에 하나만 돈다 (.run/collect.lock). 에이전트·작업 스케줄러·명령줄이 겹치면 나중 것이 건너뛴다.
- 학습데이터 생성(3.0)과는 동시에 돌려도 된다: 1.3 은 입력을 임시 폴더에 다 쓴 뒤 한 번에 바꾸고,
  3.0 은 시작할 때 고른 입력 날짜를 끝까지 쓰며, 이미 만든 룰은 날짜가 바뀌어도 다시 만들지 않는다.
- 실행 기록: logs/run_daily.log  (각 단계의 상세 로그는 fetch.log / parse.log / build_llm_input.log)

[에이전트 모드]  python 1_0_run_daily.py --agent --at 16:00
  - 매일 --at 시각에 수집한다. 시작했을 때 놓친 수집(PC 꺼짐 등)이 있으면 바로 1회 수집한다.
  - 수집이 실패하면 1시간 뒤 다시 시도한다.
  - 중지: UI 의 '에이전트 중지' 또는 .run/agent.stop 파일 (수집 중이면 그 수집을 마친 뒤 멈춤)
  - 상태: .run/agent.json (다음 실행 시각 등), 마지막 수집: .run/collect.json

사용법:
    python 1_0_run_daily.py               # 수집 → 파싱·통합 → LLM 입력
    python 1_0_run_daily.py --skip-fetch  # 수집 빼고 (이미 받은 파일 다시 처리할 때)
    python 1_0_run_daily.py --force       # 모든 단계 강제 실행
    python 1_0_run_daily.py --agent --at 16:00   # 수집 에이전트 (매일 16시)
"""

import argparse
import datetime as dt
import importlib.util
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
_rt_spec = importlib.util.spec_from_file_location("runtime", BASE_DIR / "0_0a_runtime.py")
RT = importlib.util.module_from_spec(_rt_spec); _rt_spec.loader.exec_module(RT)

log = logging.getLogger("run_daily")
RETRY_MINUTES = 60          # 에이전트: 수집 실패 시 다시 시도할 간격


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "run_daily.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:  # pythonw(작업 스케줄러) 실행 시에는 콘솔이 없음
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def run_step(script, extra_args):
    """같은 파이썬으로 스크립트를 실행하고 종료 코드를 돌려준다. (0 = 정상)"""
    cmd = [sys.executable, str(BASE_DIR / script)] + extra_args
    step = script.split("_", 2)[:2]          # "1_1_fetch_rules.py" → ["1", "1"]
    label = f"[{step[0]}.{step[1]}] {script}"
    log.info("▶ %s 시작", label)
    # 작업 스케줄러(pythonw, 콘솔 없음)로 실행될 때만 CREATE_NO_WINDOW 로 콘솔 창이 뜨지 않게 한다.
    # 직접 실행할 때는 각 단계의 로그가 화면에도 그대로 보인다.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.stdout is None else 0
    code = subprocess.run(cmd, cwd=BASE_DIR, creationflags=flags).returncode
    log.info("■ %s 종료 (코드 %d%s)", label, code, "" if code == 0 else " - 상세 내용은 해당 로그 확인")
    return code


def collect_once(args, by):
    """수집 1회 (1.1 ~ 1.4). 다른 수집이 돌고 있으면 건너뛰고 None 을 돌려준다."""
    holder = RT.acquire_lock("collect", by=by, skip_fetch=args.skip_fetch)
    if holder:
        log.warning("다른 수집이 실행 중입니다 (PID %s, %s 시작, %s) - 이번 수집은 건너뜀",
                    holder.get("pid"), holder.get("started_at"), holder.get("by", ""))
        return None
    try:
        extra = ["--force"] if args.force else []
        codes = []
        if not args.skip_fetch:
            codes.append(run_step("1_1_fetch_rules.py", extra))       # 1.1 수집
            if (BASE_DIR / "1_1a_fetch_custom.py").exists():           # 사용자 룰 수집은 있을 때만 (선택 단계)
                codes.append(run_step("1_1a_fetch_custom.py", []))     # 1.1a 사용자 룰 (엑셀 변경 반영, API 는 매주 금요일 16시 이후 1회)
        codes.append(run_step("1_2_parse_rules.py", extra))           # 1.2 파싱·통합
        codes.append(run_step("1_3_build_llm_input.py", extra))       # 1.3 LLM 입력
        codes.append(run_step("1_4_build_dataset.py", extra))         # 1.4 데이터셋 (LLM 응답이 있을 때만)
        code = 0 if all(c == 0 for c in codes) else 1
    finally:
        RT.release_lock("collect")
    if not args.skip_fetch:      # 다운로드까지 한 수집만 '마지막 수집'으로 기록 (에이전트가 놓친 수집을 판단하는 기준)
        info = {"last_run_at": RT.now_iso(), "last_code": code, "by": by}
        if code == 0:
            info["last_ok_at"] = info["last_run_at"]
        RT.write_status("collect", **info)
    return code


def next_action(now, at):
    """에이전트가 지금 수집해야 하면 None, 아니면 다음에 깨어날 시각."""
    h, m = (int(x) for x in at.split(":"))
    today_at = now.replace(hour=h, minute=m, second=0, microsecond=0)
    due = today_at if now >= today_at else today_at - dt.timedelta(days=1)       # 가장 최근 예정 시각
    nxt = today_at if now < today_at else today_at + dt.timedelta(days=1)       # 다음 예정 시각
    st = RT.read_status("collect")
    last_ok = st.get("last_ok_at")
    if last_ok and dt.datetime.fromisoformat(last_ok) >= due:
        return nxt                                            # 이번 회차는 이미 수집함
    last_run = st.get("last_run_at")
    if last_run and st.get("last_code") not in (0, None):    # 실패했으면 1시간 뒤 다시
        retry = dt.datetime.fromisoformat(last_run) + dt.timedelta(minutes=RETRY_MINUTES)
        if retry > now:
            return min(retry, nxt)
    return None


def run_agent(args):
    holder = RT.acquire_lock("agent", at=args.at)
    if holder:
        log.error("수집 에이전트가 이미 실행 중입니다 (PID %s, %s 시작)", holder.get("pid"), holder.get("started_at"))
        return 1
    RT.clear_stop("agent")
    log.info("[에이전트] 시작: 매일 %s 에 수집 (놓친 수집은 바로 1회, 중지: UI 또는 .run/agent.stop)", args.at)
    try:
        while not RT.stop_requested("agent"):
            wake = next_action(dt.datetime.now(), args.at)
            if wake is None:
                RT.write_status("agent", state="수집 중", at=args.at, pid=os.getpid())
                if collect_once(args, by="agent") is None:              # 다른 수집 중 → 5분 뒤 다시
                    wake = dt.datetime.now() + dt.timedelta(minutes=5)
                else:
                    continue
            st = RT.read_status("collect")
            RT.write_status("agent", state="대기", at=args.at, pid=os.getpid(),
                            next_run_at=wake.isoformat(timespec="seconds"),
                            last_run_at=st.get("last_run_at"), last_code=st.get("last_code"))
            log.info("[에이전트] 다음 수집: %s", wake.strftime("%Y-%m-%d %H:%M"))
            while not RT.stop_requested("agent") and dt.datetime.now() < wake:
                time.sleep(15)
    finally:
        RT.release_lock("agent")
        RT.clear_stop("agent")
        RT.write_status("agent", state="중지", pid=None, stopped_at=RT.now_iso())
        log.info("[에이전트] 중지")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Snort 룰 수집 → 파싱·통합 → LLM 입력 일괄 실행")
    parser.add_argument("--skip-fetch", action="store_true", help="수집 단계 건너뛰기")
    parser.add_argument("--force", action="store_true", help="모든 단계 강제 실행")
    parser.add_argument("--agent", action="store_true", help="에이전트 모드: 매일 --at 시각에 수집을 반복")
    parser.add_argument("--at", default="16:00", help="에이전트 수집 시각 HH:MM (기본 16:00)")
    args = parser.parse_args()
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", args.at):
        parser.error("--at 은 HH:MM 형식 (예: 16:00)")
    if args.agent and args.skip_fetch:
        parser.error("에이전트는 매번 수집(다운로드)까지 한다 - --skip-fetch 와 함께 쓸 수 없음")

    setup_logging()
    if args.agent:
        return run_agent(args)
    code = collect_once(args, by="1_0")
    return 2 if code is None else code


if __name__ == "__main__":
    sys.exit(main())
