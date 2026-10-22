"""[3.0 데이터 생성 진입점] 아래 단계를 번호 순서대로 실행한다. (API 비용이 들어 1_0_run_daily.py 와 따로 실행)

[실행 순서 - 파일 이름의 번호 = 순서]
  3.0  3_0_run_generation.py     ← 이 파일. 3.1 ~ 3.4 를 차례로 실행하고 마지막에 1.4 로 IPS/WAF 로그, 3.5 로 진행 현황까지
  3.0a 3_0a_common.py            ←   (보조) 설정·LLM 호출·코드 검사 공통 기능. 직접 실행하지 않음
  3.1a 3_1a_existing_coverage.py← 기존 학습 로그(06)에 룰별 샘플이 몇 건 있는지 계산 (비용 0, 입력이 그대로면 건너뜀)
  3.1  3_1_prepare_requests.py   ← LLM 입력 준비: 생성할 룰만 골라 '호출 1회 = 요청 파일 1개' 로 나눔 (비용 0)
                                   기존 1000건 이상 룰 제외, 1~50건 룰은 기존 예시를 붙여 '비슷하지만 다른' 샘플 생성
  3.2  3_2_generate.py           ← 데이터 생성: 생성 모델(UpStage Solar) → 룰별 씨앗 샘플 + 코드 검사
  3.3  3_3_review.py             ← 데이터 검토: 검토 모델(ChatGPT gpt-6-luna) → pass / fix / drop, 코드 재검사
  3.4  3_4_augment.py            ← 데이터 증강: 조건과 무관한 부분만 바꿔 룰당 목표 수까지 (비용 0)
  1.4  1_4_build_dataset.py      ← IPS·WAF(8필드) 수집 로그 → 08_dataset/
  3.5  3_5_progress.py           ← 진행 현황: 현재 룰 중 학습데이터를 만든 비율, 룰별 상태 플래그 → 07_llm_output/_progress/

  속도 (LLM_설정.md '## 실행'): 3.2·3.3 은 요청 여러 개를 동시에 보내고('생성 동시 요청 수' 3, '검토 동시 요청 수' 4),
    '생성·검토 동시 진행' = 예 이면 3.2 와 3.3 을 함께 돌려 생성된 요청부터 바로 검토한다.
  --max-requests N: 이번 실행에서 생성·검토할 요청 수. 이전 실행의 검토 대기분이 있으면 그것부터 검토하고 그만큼 덜 생성한다.
  각 단계는 이미 끝난 요청은 건너뛰므로, 중간에 멈춰도 다시 실행하면 이어서 진행한다.
  입력 날짜가 바뀌어도(1_0 이 새 05_llm_input/<날짜> 를 만들어도) 이어서 진행한다:
    - 시작할 때 입력 날짜를 하나 정해 3.1a·3.1·3.2 가 끝까지 같은 입력을 쓴다 (도중에 수집 에이전트가 새 입력을 만들어도)
    - 3.1: 다른 날짜 폴더에서 이미 생성한 룰은 룰 내용이 같으면 다시 넣지 않는다 (내용이 바뀐 룰만 다시 생성)
    - 3.3·3.4: --date 를 주지 않으면 모든 날짜 폴더의 검토·증강 대기분을 처리한다
    - 1.4: 날짜 폴더마다 08_dataset/<날짜>/ 로그를 만든다
  한 번에 하나만 돈다 (.run/generation.lock). UI·명령줄에서 동시에 시작하면 나중 것은 바로 끝난다.
  중지 요청(.run/generation.stop, UI 의 '중지'): 지금 요청까지 마치고 생성·검토를 멈춘 뒤, 증강·로그 변환·진행 현황만 한다.
  설정: docs/코드참조/LLM_설정.md (모델·샘플 수·목표 수), API 키: .env

[사용법]
    python 3_0_run_generation.py --status                   # 생성하지 않고 진행 현황만 (현재 룰 중 몇 %, 룰별 상태)
    python 3_0_run_generation.py --dry-run                  # 호출 없이 준비·건수만 확인 (비용 0)
    python 3_0_run_generation.py --generator mock --reviewer mock   # 키 없이 전체 흐름 시험
    python 3_0_run_generation.py                            # 변경분(delta) 요청 1개 생성·검토·증강
    python 3_0_run_generation.py --set all --max-requests 30
    python 3_0_run_generation.py --set all --max-requests 0 # 남은 요청 전부 (비용 주의)
    (화면) python 0_0_run_ui.py                             # 위 작업을 화면에서
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
_spec = importlib.util.spec_from_file_location("llm_common", BASE_DIR / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
RT = C.RT
log = logging.getLogger("3_0_run_generation")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "3_0_run_generation.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def start_step(script, args):
    """단계를 별도 프로세스로 시작 → (표시 이름, 프로세스)."""
    num = script.split("_", 2)[:2]          # "3_1a_existing..." → ["3", "1a"]
    label = f"[{num[0]}.{num[1]}] {script}"
    log.info("▶ %s 시작", label)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.stdout is None else 0
    return label, subprocess.Popen([sys.executable, str(BASE_DIR / script)] + args, cwd=BASE_DIR, creationflags=flags)


def end_step(label, proc):
    code = proc.wait()
    log.info("■ %s 종료 (코드 %d%s)", label, code, "" if code == 0 else " - 상세 내용은 해당 로그 확인")
    return code


def run_step(script, args):
    num = script.split("_", 2)[:2]
    RT.update_lock("generation", step=f"{num[0]}.{num[1]}")
    return end_step(*start_step(script, args))


def run(args):
    # 입력 날짜를 시작할 때 한 번 정해 준비·생성이 끝까지 같은 입력을 쓰게 한다
    in_dir = C.latest_input_dir(args.date)
    if in_dir is None:
        log.error("05_llm_input 에 %s 입력이 없습니다 (먼저 1_0_run_daily.py)", args.date or "완성된")
        return 1
    pin = in_dir.name
    log.info("입력 날짜 %s%s, 세트 %s, 요청 %s개%s", pin, "" if args.date else " (가장 최근)", args.set,
             args.max_requests or "남은 전부", " (dry-run)" if args.dry_run else "")
    RT.update_lock("generation", input_date=pin)
    date = ["--date", pin]
    sweep = ["--date", args.date] if args.date else []      # 3.3·3.4: 날짜를 안 줬으면 모든 날짜 폴더의 대기분까지
    # --count-pending: 생성만 되고 검토 안 된 결과가 있으면 그만큼 새로 만들지 않는다 (다시 실행해도 호출 수가 늘지 않게)
    gen = date + ["--set", args.set, "--max-requests", str(args.max_requests), "--count-pending"]
    gen += ([] if args.date else ["--pending-all"])
    gen += (["--generator", args.generator] if args.generator else []) + (["--dry-run"] if args.dry_run else [])
    rev = sweep + (["--reviewer", args.reviewer] if args.reviewer else []) + (["--dry-run"] if args.dry_run else [])
    # 검토도 이번 요청 수만큼만 (이전 실행의 검토 대기분 포함, 오래된 것부터). 0 = 대기분 전부
    rev += ["--max-requests", str(args.max_requests)] if args.max_requests > 0 else []
    try:
        overlap = C.parallel_settings(C.load_settings()[2])[2] and not args.dry_run
    except (ValueError, FileNotFoundError):
        overlap = False

    if run_step("3_1a_existing_coverage.py", date) != 0:
        log.warning("기존 데이터 확인 실패 - 기존 데이터 기준 없이 진행합니다")
    if run_step("3_1_prepare_requests.py", date + ["--set", args.set]) != 0:
        return 1
    codes = []
    if overlap:      # 생성과 검토를 함께: 생성된 요청부터 바로 검토 (LLM_설정.md '생성·검토 동시 진행')
        RT.update_lock("generation", step="3.2+3.3")
        g_label, g_proc = start_step("3_2_generate.py", gen)
        r_label, r_proc = start_step("3_3_review.py", rev + ["--follow", str(g_proc.pid)])
        if end_step(g_label, g_proc) != 0:
            log.error("생성 단계 실패 - 검토는 이미 생성된 결과만 처리합니다")
        codes.append(end_step(r_label, r_proc))
        if RT.stop_requested("generation"):
            log.info("중지 요청 - 남은 생성·검토는 다음 실행으로 미루고 증강·로그 변환·진행 현황만 합니다")
    else:
        if run_step("3_2_generate.py", gen) != 0:
            log.error("생성 단계 실패 - 이후 단계는 이미 생성된 결과만 처리합니다")
        if RT.stop_requested("generation"):
            log.info("중지 요청 - 검토(3.3)는 다음 실행으로 미루고 증강·로그 변환·진행 현황만 합니다")
        else:
            codes.append(run_step("3_3_review.py", rev))
    if args.dry_run:
        log.info("dry-run 이라 증강·로그 변환은 건너뜁니다")
        return 0 if all(c == 0 for c in codes) else 1
    codes.append(run_step("3_4_augment.py", sweep))
    if not args.no_dataset:
        codes.append(run_step("1_4_build_dataset.py", []))
    return 0 if all(c == 0 for c in codes) else 1


def main():
    ap = argparse.ArgumentParser(description="LLM 데이터 생성 전체 실행 (3.1 → 3.2 → 3.3 → 3.4 → 1.4 → 3.5)")
    ap.add_argument("--set", choices=["delta", "all"], default="delta", help="배치 종류 (기본 delta = 변경분)")
    ap.add_argument("--date", help="05_llm_input 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--max-requests", type=int, default=1, help="이번에 생성할 요청 수 (기본 1, 0 = 남은 것 전부)")
    ap.add_argument("--generator", help="생성 모델 'provider:model'")
    ap.add_argument("--reviewer", help="검토 모델 'provider:model' 또는 'none'")
    ap.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 준비와 건수만 확인")
    ap.add_argument("--no-dataset", action="store_true", help="마지막 1.4 (IPS/WAF 로그 변환) 를 건너뛴다")
    ap.add_argument("--status", action="store_true",
                    help="생성하지 않고 진행 현황만: 현재 룰 중 학습데이터를 만든 비율, 룰별 상태 플래그 (3.5)")
    args = ap.parse_args()
    setup_logging()
    if args.status:
        return run_step("3_5_progress.py", [])

    holder = RT.acquire_lock("generation", by="3_0", set=args.set, max_requests=args.max_requests,
                             dry_run=args.dry_run)
    if holder:
        log.error("학습데이터 생성이 이미 실행 중입니다 (PID %s, %s 시작) - 끝난 뒤 다시 실행하거나 UI 에서 확인하세요",
                  holder.get("pid"), holder.get("started_at"))
        return 2
    RT.clear_stop("generation")
    os.environ[C.LOCK_OWNER_ENV] = "generation"      # 자식 단계(3.1~3.4)는 이 잠금 아래에서 돈다
    try:
        code = run(args)
        run_step("3_5_progress.py", [])               # 진행 현황 갱신 (비용 0)
        return code
    finally:
        RT.clear_stop("generation")
        RT.release_lock("generation")


if __name__ == "__main__":
    sys.exit(main())
