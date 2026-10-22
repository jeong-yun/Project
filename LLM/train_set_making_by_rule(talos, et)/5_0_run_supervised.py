"""[5.0 지도학습 실행기] 화면 6 의 '비교'·'확정'을 별도 프로세스로 실행한다 (잠금 supervised: 한 번에 하나).

    python 5_0_run_supervised.py compare --base "26 상반기" --dev ips --dev waf [--mode new|all] [--include-compared]
    python 5_0_run_supervised.py finalize --version "26 하반기"

  compare   5.5 신규 학습데이터(08_dataset)를 기준 버전과 비교 → 샘플별 등급·요약·엑셀 (11_supervised/compared/)
            --mode new  새로 생긴 샘플의 피처만 계산 (기본) / all  피처·기준 모델을 모두 다시 계산
            --include-compared  확정된 버전에서 이미 비교한 샘플도 다시 비교 (기본은 뺌)
  finalize  5.7 버전 확정 → 기준 버전 + 선별한 샘플의 학습 파일(CSV)·로그 내보내기 (11_supervised/versions/<이름>/)

진행 상태는 .run/supervised.json (단계·처리 수·남은 시간·결과), 중지 요청은 .run/supervised.stop (화면의 '중지').
중지해도 계산한 피처는 캐시에 남아 다음 '새로 생긴 데이터만 비교' 가 이어서 한다.
피처 계산은 프로세스 여러 개로 나눠 한다 (--workers, 기본 CPU 의 절반 - 1, 최대 3).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
_mods = {}
JOB = "supervised"


def _load(name, file):
    if name not in _mods:
        spec = importlib.util.spec_from_file_location(name, BASE / file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _mods[name] = mod
    return _mods[name]


def feature_task(dev, code, payloads):
    """피처 계산 프로세스에서 실행 (5.0b compute_chunk)."""
    return _load("sup_features", "5_0b_features.py").compute_chunk(dev, code, payloads)


class RunCtx:
    """5.5·5.7 의 진행 보고 → .run/supervised.json, 중지 요청 확인."""

    def __init__(self, RT, C, pool=None):
        self.RT, self.C, self.pool, self.task = RT, C, pool, feature_task
        self.steps, self.t_step, self.last = [], time.monotonic(), 0.0

    def _write(self, **kw):
        self.RT.write_status(JOB, steps=self.steps, **kw)

    def step(self, key, text):
        now = time.monotonic()
        if self.steps:
            self.steps[-1]["sec"] = round(now - self.t_step, 1)
        self.steps.append({"key": key, "text": text})
        self.t_step, self.last = now, now
        self.log(text)
        self._write(step=text, step_key=key, done=None, total=None, eta_s=None, unit="")

    def progress(self, done, total=None, eta_s=None, unit="행"):
        now = time.monotonic()
        if now - self.last < 1.0 and not (total and done >= total):
            return
        self.last = now
        self._write(done=done, total=total, eta_s=None if eta_s is None else round(eta_s), unit=unit)

    def check(self):
        if self.RT.stop_requested(JOB):
            raise self.C.Stopped("중지 요청")

    def log(self, msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    def finish(self, **kw):
        if self.steps and "sec" not in self.steps[-1]:
            self.steps[-1]["sec"] = round(time.monotonic() - self.t_step, 1)
        self._write(finished_at=self.C.now_iso(), done=None, total=None, eta_s=None, **kw)


def main():
    ap = argparse.ArgumentParser(description="화면 6 기존 학습데이터와 비교·버전 확정 실행기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="신규 학습데이터를 기준 버전과 비교")
    c.add_argument("--base", required=True, help="기준 버전 이름 (예: 26 상반기)")
    c.add_argument("--dev", action="append", choices=["ips", "waf"], help="장비 (여러 번, 기본 둘 다)")
    c.add_argument("--mode", choices=["new", "all"], default="new")
    c.add_argument("--include-compared", action="store_true")
    c.add_argument("--workers", type=int, default=0)
    f = sub.add_parser("finalize", help="버전 확정·내보내기")
    f.add_argument("--version", required=True)
    args = ap.parse_args()
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")

    RT = _load("runtime", "0_0a_runtime.py")
    C = _load("sup_common", "5_0a_common.py")
    info = {"cmd": args.cmd, "base": getattr(args, "base", None), "version": getattr(args, "version", None),
            "devs": getattr(args, "dev", None) or list(C.DEVICES), "mode": getattr(args, "mode", None)}
    holder = RT.acquire_lock(JOB, by="5_0", **info)
    if holder:
        print(f"이미 실행 중입니다 (PID {holder.get('pid')}, {holder.get('cmd')})")
        return 2
    RT.clear_stop(JOB)
    try:
        (RT.RUN_DIR / f"{JOB}.json").unlink()            # 지난 실행의 상태는 지우고 새로
    except OSError:
        pass
    RT.write_status(JOB, state="running", pid=os.getpid(), started_at=C.now_iso(), **info)
    ctx = RunCtx(RT, C)
    code = 0
    try:
        if args.cmd == "compare":
            workers = args.workers or min(3, max(1, (os.cpu_count() or 2) // 2 - 1))
            M = _load("sup_compare", "5_5_newdata_check.py")
            pool = None
            if workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                pool = ProcessPoolExecutor(max_workers=workers)
            ctx.pool = pool
            try:
                res = M.run(args.base, info["devs"], args.mode, args.include_compared, ctx)
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
            result = {dev: {"rows": s["rows"], "snapshot": s["snapshot"],
                            "grades": next((t["rows"] for k, t in s["tables"].items() if k == "등급별"), [])}
                      for dev, s in res.items()}
            ctx.finish(state="done", result=result)
        else:
            V = _load("sup_versions", "5_7_versions.py")
            out = V.finalize(args.version, ctx)
            ctx.finish(state="done", result=out)
        timing = C.read_json(C.CACHE / "timing.json", {})             # 다음 실행의 예상 시간
        for s in ctx.steps:
            if "sec" in s:
                timing[f"{args.cmd}:{s['key']}"] = s["sec"]
        C.write_json(C.CACHE / "timing.json", timing)
    except C.Stopped as e:
        ctx.log(f"중지: {e}")
        ctx.finish(state="stopped", error=str(e))
        code = 3
    except Exception as e:
        traceback.print_exc()
        ctx.finish(state="fail", error=f"{type(e).__name__}: {e}")
        code = 1
    finally:
        RT.clear_stop(JOB)
        RT.release_lock(JOB)
    return code


if __name__ == "__main__":
    sys.exit(main())
