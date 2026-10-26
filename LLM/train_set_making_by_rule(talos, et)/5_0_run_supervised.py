"""[5.0 지도학습 실행기] 화면 6 의 '비교'·'확정'을 별도 프로세스로 실행한다 (잠금 supervised: 한 번에 하나).

    python 5_0_run_supervised.py compare --base "26 상반기" --dev ips --dev waf [--mode new|all] [--include-compared]
    python 5_0_run_supervised.py finalize --version "26 하반기"

  compare   5.5 신규 학습데이터(08_dataset)를 기준 버전과 비교 → 샘플별 등급·요약·엑셀 (11_supervised/compared/)
            --mode new  새로 생긴 샘플의 피처만 계산 (기본) / all  피처·기준 모델을 모두 다시 계산
            --include-compared  확정된 버전에서 이미 비교한 샘플도 다시 비교 (기본은 뺌)
  finalize  5.7 버전 확정 → 기준 버전 + 선별한 샘플의 학습 파일(CSV)·로그 내보내기 (11_supervised/versions/<이름>/)

진행 상태는 .run/supervised.json (전체 %·남은 시간·끝날 시각·실행 계획, 지금 단계·처리 수, 결과),
중지 요청은 .run/supervised.stop (화면의 '중지').
중지해도 계산한 피처는 캐시에 남아 다음 '새로 생긴 데이터만 비교' 가 이어서 한다.
피처 계산은 프로세스 여러 개로 나눠 한다 (--workers, 기본 CPU 의 절반 - 1, 최대 3).
"""

import argparse
import importlib.util
import json
import os
import sys
import threading
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
    """5.5·5.7 의 진행 보고 → .run/supervised.json, 중지 요청 확인.

    전체 진행률: 시작할 때 실행 계획(plan, 단계마다 예상 시간 = 5.0a compare_plan / 5.7 finalize_plan)을 만든다.
      % = (끝난 단계 예상 합 + 지금 단계 예상 × 단계 안 진행 비율) ÷ 전체 예상 합
          단계 안 진행 비율 = 진행 보고가 있으면 그 값 (done/total, 하위 단계 k/K + 그 하위 단계 안의 지난 시간 ÷ 예상),
          없으면 지난 시간 ÷ 예상 (95% 까지). 하위 단계 예상은 지난 실행의 하위 단계 시간 비율(없으면 같게).
          실행 중에 계획을 고치면(읽은 뒤 행 수·새 payload 수를 앎) 그때 보인 % 에서 이어 가도록 맞춘다 → %는 줄지 않음.
      남은 시간 = 남은 예상 × 보정 (지금까지 실제 ÷ 예상, 0.5~3. 한 일이 적은 초반에는 1 쪽으로 약하게), 끝날 시각 = 지금 + 남은 시간.
      진행 보고가 없는 동안에도 2초마다 상태 파일을 다시 써서 %·남은 시간이 움직인다.
    상태 파일: overall_pct, eta_total_s, finish_at, elapsed_s, detail(지금 단계의 진행 수), first_eta_s·first_finish_at(시작할 때 예상),
               plan [{key, dev, short, text, expected_s, state: done|running|wait|skip, sec}]  (+ 예전 화면용 steps, done/total)."""

    def __init__(self, RT, C, pool=None):
        self.RT, self.C, self.pool, self.task = RT, C, pool, feature_task
        self.steps, self.t_step, self.last = [], time.monotonic(), 0.0
        self.t0, self.cmd, self.timing, self.make_plan, self.est = time.monotonic(), "", {}, None, {}
        self.plan, self.notes, self.hist_notes, self.first = [], {}, set(), {}
        self.subs = None                    # 지금 단계의 하위 단계 {k, n, w, t, text, own}
        self.sub_secs = []                  # 지금 단계에서 끝난 하위 단계 시간 (다음 실행의 하위 단계 비율)
        self.intra, self.prog = None, ""    # 진행 보고 비율 (하위 단계 안, 없으면 단계 전체)·그 글
        self.anchor, self.shown = (0.0, 0.0), 0.0
        self.ended = False
        self.lock = threading.RLock()
        threading.Thread(target=self._beat, daemon=True).start()

    # ── 실행 계획·전체 진행률 ──
    def set_plan(self, cmd, make_plan, est=None, timing=None):
        """make_plan(est) → [{key, dev, short, text, expected_s, tkey}], est = 장비별 추정 (또는 그것을 만드는 함수).
        계획을 못 만들면 전체 진행률 없이 실행한다."""
        with self.lock:
            self.cmd, self.timing = cmd, timing or {}
            try:
                self.make_plan, self.est = make_plan, (est() if callable(est) else est) or {}
                self.plan = [dict(p, state="wait") for p in make_plan(self.est)]
            except Exception as e:
                self.make_plan, self.plan = None, []
                self.log(f"  (실행 계획을 만들지 못해 전체 진행률 없이 실행: {type(e).__name__}: {e})")
            if self.plan and self.plan[0]["key"] == "prep":     # 준비 (모듈·피처 코드 읽기) = 첫 단계 전까지
                self.plan[0]["state"], self.t_step = "running", time.monotonic()
            st = self._status(time.monotonic())
            self.first = {"first_eta_s": st.get("eta_total_s"), "first_finish_at": st.get("finish_at"),
                          "first_plan_s": st.get("plan_total_s")}
        self._write()

    def estimate(self, dev, **facts):
        """실행 중에 안 값(n_rows 비교할 행 수, n_new 새 payload 수 등)으로 아직 안 끝난 단계의 예상 시간을 고친다."""
        with self.lock:
            if not self.make_plan:
                return
            try:
                self.est.setdefault(dev, {}).update(facts)
                new = {p["key"]: p for p in self.make_plan(self.est)}
                for p in self.plan:
                    q = new.get(p["key"])
                    if q and p["state"] in ("wait", "running"):
                        p["expected_s"] = q["expected_s"]
                        if "n" in q:
                            p["n"] = q["n"]
                self._rebase(time.monotonic())
            except Exception:                         # 예상을 못 고쳐도 실행은 계속
                return
        self._write()

    def note(self, key, value, hist=False):
        """다음 실행의 예상에 쓸 값 (끝까지 성공하면 cache/timing.json 에 저장, hist=True 면 최근 3번도 남김)."""
        self.notes[key] = value
        if hist:
            self.hist_notes.add(key)

    def save_timing(self, timing):
        """끝까지 성공한 실행 → timing (다음 실행의 예상): 단계별 실제 시간·note 의 마지막 값 + hist (최근 3번, 5.0a past_median).
        규모에 비례하는 단계(계획의 per)는 행·바이트당 초로 남긴다."""
        H = timing.setdefault("hist", {})

        def add(k, val):
            H[k] = list(H.get(k) or [])[-2:] + [val]

        for s in self.steps:                          # 예전 키 (단계 key 그대로)
            if "sec" in s:
                timing[f"{self.cmd}:{s['key']}"] = s["sec"]
        for p in self.plan:
            if p["state"] != "done" or not p.get("tkey") or p.get("sec") is None:
                continue
            timing[p["tkey"]] = p["sec"]
            size = self.notes.get(p["per"]) if p.get("per") else None
            if size:
                add(p["tkey"], p["sec"] / size)
            elif not p.get("per"):
                add(p["tkey"], p["sec"])
        for k, val in self.notes.items():
            timing[k] = val
            if k in self.hist_notes:
                add(k, val)
        return timing

    def _cur(self):
        return next((p for p in self.plan if p["state"] == "running"), None)

    def _frac(self, now, cur):
        """지금 단계 안 진행 비율."""
        E = max(0.1, cur["expected_s"])
        s = self.subs
        if s:
            W = sum(s["w"])
            g = self.intra if self.intra is not None else min(0.95, (now - s["t"]) / max(0.1, E * s["w"][s["k"]] / W))
            return min(1.0, (sum(s["w"][:s["k"]]) + s["w"][s["k"]] * g) / W)
        if self.intra is not None:
            return min(1.0, self.intra)
        return min(0.95, (now - self.t_step) / E)

    def _raw(self, now):
        """(지금 계획으로 잰 진행 비율, 한 일의 예상 초, 전체 예상 초)."""
        total = sum(p["expected_s"] for p in self.plan) or 1.0
        done = sum(p["expected_s"] for p in self.plan if p["state"] in ("done", "skip"))
        cur = self._cur()
        if cur:
            done += cur["expected_s"] * self._frac(now, cur)
        return min(1.0, done / total), done, total

    def _rebase(self, now):
        """계획을 고친 뒤: 지금 보인 % 에서 이어 가도록 (남은 % 를 새 계획의 남은 일로 다시 나눔)."""
        self.anchor = (self.shown, self._raw(now)[0])

    def _status(self, now):
        if not self.plan:
            return {}
        raw, done, total = self._raw(now)
        p0, r0 = self.anchor
        pct = 1.0 if r0 >= 1.0 else p0 + (1.0 - p0) * max(0.0, raw - r0) / (1.0 - r0)
        self.shown = pct = min(1.0, max(self.shown, pct))
        elapsed = now - self.t0
        ratio = min(3.0, max(0.5, elapsed / done)) if done > 0 else 1.0
        corr = 1.0 + (ratio - 1.0) * min(1.0, done / max(30.0, 0.15 * total))
        remain = max(0.0, total - done) * corr
        plan = []
        for p in self.plan:
            q = {k: p[k] for k in ("key", "dev", "short", "text", "expected_s", "state", "n") if k in p}
            if p["state"] == "running":
                q["sec"] = round(now - self.t_step, 1)
            elif p.get("sec") is not None:
                q["sec"] = p["sec"]
            plan.append(q)
        return {"overall_pct": round(100.0 * pct, 1), "eta_total_s": round(remain),
                "finish_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + remain)),
                "elapsed_s": round(elapsed), "plan_total_s": round(total), "plan": plan,
                "detail": " · ".join(x for x in ((self.subs or {}).get("text"), self.prog) if x)}

    def _write(self, final=False, beat=False, **kw):
        with self.lock:
            if (self.ended and not final) or (beat and self._cur() is None):
                return
            try:
                st = dict(self._status(time.monotonic()), **self.first)
            except Exception:                         # 진행률 계산이 실패해도 실행은 계속
                st = {}
            st.update(kw)
            for i in range(10 if final else 3):
                try:
                    self.RT.write_status(JOB, steps=self.steps, **st)
                    return
                except OSError:                       # 화면이 상태 파일을 읽는 순간이면 바꾸지 못함 (Windows) → 잠깐 뒤 다시
                    time.sleep(0.05 * (i + 1))

    def _beat(self):
        while not self.ended:
            time.sleep(2.0)
            self._write(beat=True)

    def _end_step(self, now):
        """지금 단계를 끝냄: 걸린 시간, 하위 단계 시간 (다음 실행의 하위 단계 비율)."""
        if self.steps and "sec" not in self.steps[-1]:
            self.steps[-1]["sec"] = round(now - self.t_step, 1)
        cur = self._cur()
        if cur:
            cur.update(state="done", sec=round(now - self.t_step, 1))
            s = self.subs
            if s and not s["own"] and cur.get("tkey"):
                secs = self.sub_secs + [now - s["t"]]
                if len(secs) == s["n"]:
                    self.notes[cur["tkey"] + ":sub"] = [round(x, 1) for x in secs]

    # ── 5.5·5.7 이 부르는 것 ──
    def step(self, key, text):
        now = time.monotonic()
        with self.lock:
            self._end_step(now)
            self.steps.append({"key": key, "text": text})
            self.t_step, self.last = now, now
            self.subs, self.sub_secs, self.intra, self.prog = None, [], None, ""
            if self.plan:
                i = next((i for i, p in enumerate(self.plan) if p["key"] == key), None)
                if i is None:                         # 계획에 없던 단계 → 다음 대기 단계 자리에 넣음
                    i = next((i for i, p in enumerate(self.plan) if p["state"] == "wait"), len(self.plan))
                    self.plan.insert(i, {"key": key, "dev": key.split(":")[0] if ":" in key else "", "short": text,
                                         "text": text, "expected_s": float(self.timing.get(f"{self.cmd}:{key}") or 5.0),
                                         "tkey": f"{self.cmd}:{key}", "state": "wait"})
                    self._rebase(now)
                for p in self.plan[:i]:
                    if p["state"] == "wait":
                        p["state"] = "skip"
                self.plan[i].update(state="running", text=text)
        self.log(text)
        self._write(step=text, step_key=key, done=None, total=None, eta_s=None, unit="")

    def sub(self, k, n, text, weights=None):
        """지금 단계의 하위 단계 k(0부터)/n 시작 (예: 날짜 2/4, 모델 예측 3/4). weights = 하위 단계별 예상 비율
        (없으면 지난 실행의 하위 단계 시간, 그것도 없으면 같게)."""
        if n <= 0:
            return
        now = time.monotonic()
        with self.lock:
            s = self.subs
            if s and s["n"] == n and k > s["k"]:
                self.sub_secs.append(now - s["t"])
            cur = self._cur()
            hist = self.timing.get(((cur or {}).get("tkey") or "") + ":sub")
            w = list(weights) if weights else hist if isinstance(hist, list) and len(hist) == n else [1.0] * n
            w = [max(0.0, float(x)) for x in w]
            if len(w) != n or sum(w) <= 0:
                w = [1.0] * n
            self.subs = {"k": min(k, n - 1), "n": n, "w": w, "t": now, "text": text, "own": bool(weights)}
            self.intra, self.prog = None, ""
        self._write()

    def progress(self, done, total=None, eta_s=None, unit="행", what="", text=None):
        """진행 수 (하위 단계가 있으면 그 안의 진행). text 가 없으면 '<what> done/total<unit>'."""
        now = time.monotonic()
        with self.lock:
            try:
                if total:
                    self.intra = min(1.0, done / total)
                self.prog = text if text is not None else (f"{what} " if what else "") + (
                    f"{done:,}/{total:,}{unit}" if total else f"{done:,}{unit}")
            except (TypeError, ValueError):
                pass
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
        now = time.monotonic()
        with self.lock:
            if kw.get("state") == "done":
                self._end_step(now)
                self.anchor, self.shown = (1.0, 1.0), 1.0
                for p in self.plan:
                    if p["state"] == "wait":
                        p["state"] = "skip"
            else:                                     # 중지·실패: 멈춘 단계는 running 그대로 (화면에 ✘)
                if self.steps and "sec" not in self.steps[-1]:
                    self.steps[-1]["sec"] = round(now - self.t_step, 1)
                cur = self._cur()
                if cur:
                    cur["sec"] = round(now - self.t_step, 1)
            self.ended = True
            st = {}
            try:
                st = dict(self._status(now), **self.first)
            except Exception:
                pass
            if self.plan:
                for p in st.get("plan", []):          # 멈춘 단계의 시간 (running 은 _status 가 지금 시각으로 씀)
                    q = next((x for x in self.plan if x["key"] == p["key"]), {})
                    if q.get("sec") is not None:
                        p["sec"] = q["sec"]
                if kw.get("state") == "done":
                    st.update(overall_pct=100.0, eta_total_s=0, finish_at=self.C.now_iso())
                else:
                    st.update(eta_total_s=None, finish_at=None)
            st.update(finished_at=self.C.now_iso(), done=None, total=None, eta_s=None, **kw)
        self._write(final=True, **st)


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
    timing = C.read_json(C.CACHE / "timing.json", {})                # 지난 실행의 단계별 시간 → 실행 계획
    code = 0
    try:
        if args.cmd == "compare":
            ctx.set_plan("compare", lambda est: C.compare_plan(args.base, info["devs"], args.mode, args.include_compared,
                                                               timing, est), lambda: C.compare_estimate(args.base), timing)
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
            ctx.set_plan("finalize", lambda est: V.finalize_plan(args.version, timing, est), {}, timing)
            out = V.finalize(args.version, ctx)
            ctx.finish(state="done", result=out)
        C.write_json(C.CACHE / "timing.json", ctx.save_timing(C.read_json(C.CACHE / "timing.json", {})))   # 다음 실행의 예상
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
