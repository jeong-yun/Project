"""화면 전환 속도 측정 (Streamlit AppTest) — 🏠 메인·팀별 화면을 다시 그리는 데 걸리는 시간.

실행 (Agent_office 폴더에서):
    python scripts/bench_ui.py                     # 빈 임시 데이터로
    python scripts/bench_ui.py --data ../data -n 5 # 기존 data/ 를 임시 폴더에 복사해서 (원본은 건드리지 않음)
    python scripts/bench_ui.py --profile stock     # 그 화면 한 번을 cProfile 로 — 비용 큰 함수 상위 목록

측정값
  script : 스크립트(app.py) 실행 시간 — Streamlit 서버가 화면 한 번 다시 그릴 때 쓰는 시간과 같다
  total  : AppTest 가 결과를 받아 해석하는 시간까지 포함 (참고용)
  첫 방문 : 서버를 켠 뒤 그 화면을 처음 볼 때 (캐시가 비어 있음) / 다시 : 같은 화면을 다시 그릴 때 (중앙값)
  전환   : 🏠 메인 → 팀 → 다른 팀 → 🏠 메인 … 을 차례로 고를 때 한 번의 중앙값

브라우저가 받은 화면을 그리는 시간·네트워크는 빠져 있다 (서버 쪽 시간만).
API 키는 가짜 값('bench-key')을 넣어 키가 있을 때의 경로(FreeLLMAPI 연결 확인 등)를 탄다 — 모델 호출은 하지 않는다.
"""

import argparse
import cProfile
import io
import os
import pstats
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

VIEWS = ["home", "general", "stock", "travel", "concept", "music", "design"]
KEY_ENVS = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "UPSTAGE_API_KEY", "FREELLM_API_KEY"]

_timing = {"script": 0.0, "profile": None}


def patch_script_runner() -> None:
    """스크립트 스레드 안에서 실행 시간을 재고(필요하면 cProfile) — AppTest 의 대기·해석 시간과 분리한다.

    AppTest 는 실행마다 app.py 를 새로 컴파일하지만(ScriptCache 를 새로 만듦), 실제 서버는 한 번 컴파일한 것을
    계속 쓴다 → 서버와 같게 하나를 같이 쓰게 한다.
    """
    from streamlit.runtime.scriptrunner.script_cache import ScriptCache
    from streamlit.runtime.scriptrunner.script_runner import ScriptRunner
    from streamlit.testing.v1 import local_script_runner

    shared_cache = ScriptCache()
    local_script_runner.ScriptCache = lambda: shared_cache
    original = ScriptRunner._run_script

    def timed(self, rerun_data):
        prof = _timing["profile"]
        t0 = time.perf_counter()
        if prof is not None:
            prof.enable()
        try:
            return original(self, rerun_data)
        finally:
            if prof is not None:
                prof.disable()
            _timing["script"] = time.perf_counter() - t0

    ScriptRunner._run_script = timed


def select(at, view: str) -> None:
    """사이드바에서 화면·팀을 고른 것과 같게 위젯 값을 바꾼다."""
    if view == "home":
        at.session_state["view"] = "home"
    else:
        at.session_state["view"] = "team"
        at.session_state["team_pick"] = view


def run(at) -> tuple[float, float]:
    t0 = time.perf_counter()
    at.run()
    total = time.perf_counter() - t0
    if at.exception:
        raise RuntimeError(f"화면 오류: {at.exception}")
    return _timing["script"] * 1000, total * 1000


def new_app(catalog_dir: str | None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(BASE / "app.py"), default_timeout=120)
    if catalog_dir:
        at.session_state["concept_root"] = catalog_dir
    return at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="복사해 쓸 data 폴더 (runs.db 등). 원본은 바꾸지 않는다")
    parser.add_argument("--catalog", help="개념 팀 카탈로그(CS/Term/Every_day) 경로 — 없으면 settings 기본값")
    parser.add_argument("-n", type=int, default=5, help="화면마다 다시 그리는 횟수 (기본 5)")
    parser.add_argument("--profile", choices=VIEWS, help="이 화면을 다시 그리는 한 번을 cProfile 로 측정")
    parser.add_argument("--top", type=int, default=25, help="cProfile 상위 몇 줄 (기본 25)")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ao_bench_"))
    data = tmp / "data"
    if args.data:
        shutil.copytree(args.data, data)
    os.environ["AGENT_OFFICE_DATA_DIR"] = str(data)
    os.environ["AGENT_OFFICE_RESULTS_DIR"] = str(tmp / "results")
    for env in KEY_ENVS:
        os.environ.setdefault(env, "bench-key")

    t0 = time.perf_counter()
    import streamlit  # noqa: F401 — 측정에서 streamlit 자체 import 는 뺀다
    t_streamlit = time.perf_counter() - t0
    patch_script_runner()

    try:
        at = new_app(args.catalog)
        select(at, "home")
        script, total = run(at)
        print(f"서버 시작 후 첫 실행 (app.py 모듈 import 포함): script {script:.0f} ms · total {total:.0f} ms"
              f"  (streamlit import {t_streamlit * 1000:.0f} ms 별도)")

        if args.profile:
            select(at, args.profile)
            run(at)  # 첫 방문 캐시를 채우고
            _timing["profile"] = prof = cProfile.Profile()
            script, _ = run(at)
            _timing["profile"] = None
            print(f"\n[{args.profile}] 다시 그리기 1회 script {script:.0f} ms — cProfile 상위 (cumulative)")
            out = io.StringIO()
            pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(args.top)
            print(out.getvalue())
            out = io.StringIO()
            pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(args.top)
            print("— tottime —\n" + out.getvalue())
            return

        print(f"\n{'화면':<8} {'첫 방문(ms)':>12} {'다시 script':>12} {'다시 total':>11}")
        rows = {}
        for view in VIEWS:
            select(at, view)
            first, _ = run(at)
            again = [run(at) for _ in range(args.n)]
            s = statistics.median(a[0] for a in again)
            t = statistics.median(a[1] for a in again)
            rows[view] = (first, s, t)
            print(f"{view:<8} {first:>12.0f} {s:>12.0f} {t:>11.0f}")

        order = ["home", "general", "stock", "home", "travel", "concept", "home", "music", "design", "general"]
        switches = []
        for _ in range(max(1, args.n // 2)):
            for view in order:
                select(at, view)
                switches.append(run(at))
        print(f"\n전환 {len(switches)}회 ({' → '.join(order)}) — script 중앙값 {statistics.median(s for s, _ in switches):.0f} ms"
              f" · 최대 {max(s for s, _ in switches):.0f} ms · total 중앙값 {statistics.median(t for _, t in switches):.0f} ms")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
