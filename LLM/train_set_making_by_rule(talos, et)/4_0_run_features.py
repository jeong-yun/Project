"""[4.0 ML 피처 진입점] 4.1 피처 생성 → 4.2 피처 점검을 차례로 실행한다.

1_0_run_daily.py(룰 수집)·3_0_run_generation.py(LLM 데이터 생성)와는 **따로** 실행한다.
LLM 을 부르지 않으므로 비용이 없다. 3.0 으로 데이터셋(08)을 만든 뒤, 또는 기존 로그로 피처를 볼 때 실행한다.

[실행 순서]
  4.0  4_0_run_features.py       ← 이 파일
  4.1  4_1_build_features.py     ← 학습 로그 payload → 피처 표 (09_features/<이름>/features.csv)
  4.2  4_2_evaluate_features.py  ← 피처 묶음별 간단 모델 성능 (09_features/<이름>/eval_*.md)

[사용법]
    python 4_0_run_features.py                                  # 08_dataset 최근 날짜 → 피처 → 7:3 분할 점검
    python 4_0_run_features.py --rules et:1:2053468             # 개별 룰 피처 추가
    python 4_0_run_features.py --input "06_기존 데이터/ips_260202_raw.log" --sample 30000 --name existing_ips
    python 4_0_run_features.py --name existing_ips --test 09_features/2026-10-16    # 다른 출처로 평가
    python 4_0_run_features.py --group rule_id                  # 룰 단위로 나눠 평가 (생성 데이터의 라벨 누수 방지)
    python 4_0_run_features.py --skip-build --name existing_ips # 피처는 그대로, 점검만 다시
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DATASET_DIR = BASE_DIR / "08_dataset"
log = logging.getLogger("4_0_run_features")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR / "4_0_run_features.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    if sys.stdout is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def run_step(script, args):
    num = script.split("_", 2)[:2]
    label = f"[{num[0]}.{num[1]}] {script}"
    log.info("▶ %s 시작", label)
    code = subprocess.run([sys.executable, str(BASE_DIR / script)] + args, cwd=BASE_DIR).returncode
    log.info("■ %s 종료 (코드 %d%s)", label, code, "" if code == 0 else " - 상세 내용은 해당 로그 확인")
    return code


def main():
    ap = argparse.ArgumentParser(description="ML 피처 생성·점검 (4.1 → 4.2)")
    ap.add_argument("--input", nargs="*", help="로그 파일들 (기본: 08_dataset 최근 날짜)")
    ap.add_argument("--date", help="08_dataset 날짜 폴더 (기본: 가장 최근)")
    ap.add_argument("--name", help="09_features 아래 폴더 이름")
    ap.add_argument("--rules", help="개별 룰 피처 룰 id (쉼표 구분)")
    ap.add_argument("--sample", type=int, help="무작위 줄 수")
    ap.add_argument("--no-rule-agg", action="store_true", help="공격 룰(ra_) 피처 빼기")
    ap.add_argument("--min-precision", type=float, help="ra_ 에 쓸 룰의 최소 정밀도 (기본 0.9)")
    ap.add_argument("--test", help="평가용 피처 폴더 (다른 출처 데이터로 평가)")
    ap.add_argument("--group", help="이 열 기준으로 학습/평가 분할 (예: rule_id)")
    ap.add_argument("--skip-build", action="store_true", help="4.1 을 건너뛰고 4.2 만")
    args = ap.parse_args()
    setup_logging()

    # 4.1 과 4.2 가 같은 폴더를 보도록 이름을 먼저 정한다 (4.1 의 기본 규칙과 같게)
    if args.name:
        name = args.name
    elif args.input:
        name = Path(args.input[0]).stem
    else:
        days = sorted(p.name for p in DATASET_DIR.iterdir()
                      if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)) if DATASET_DIR.is_dir() else []
        name = args.date or (days[-1] if days else None)
        if name is None:
            log.error("08_dataset 에 날짜 폴더가 없습니다 (먼저 3_0_run_generation.py)")
            return 1

    if not args.skip_build:
        build = ["--name", name]
        build += (["--input"] + args.input) if args.input else []
        build += (["--date", args.date] if args.date else []) + (["--rules", args.rules] if args.rules else [])
        build += (["--sample", str(args.sample)] if args.sample else []) + (["--no-rule-agg"] if args.no_rule_agg else [])
        build += ["--min-precision", str(args.min_precision)] if args.min_precision is not None else []
        if run_step("4_1_build_features.py", build) != 0:
            return 1
    ev = ["--features", f"09_features/{name}"] + (["--test", args.test] if args.test else [])
    ev += ["--group", args.group] if args.group else []
    return run_step("4_2_evaluate_features.py", ev)


if __name__ == "__main__":
    sys.exit(main())
