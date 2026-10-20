"""[1.1 수집] Snort 룰 원천 소스(ET Open, Cisco Talos) 정기 수집기.  (1_0_run_daily.py 가 실행)

[전체 흐름]
  1. .env 에서 설정(OINKCODE 등)을 읽는다.
  2. state.json 에서 "지난번에 받은 버전"을 읽는다.
  3. 각 사이트에 "지금 버전"만 가볍게 물어본다.
       - ET    : version.txt  (숫자 한 줄, 예: 11310)
       - Talos : .md5         (파일의 지문값, 예: 591c47da...)
  4. 지난번 버전 == 지금 버전  → "변경 없음 - 건너뜀" (다운로드 안 함)
     지난번 버전 != 지금 버전  → 룰 파일 다운로드 → state.json 갱신
  5. 모든 결과는 화면 + logs/fetch.log 에 기록한다.

[폴더 구조]
  rule_data/
   ├─ 1_1_fetch_rules.py            ← 이 파일
   ├─ .env                      ← OINKCODE 등 비밀 설정 (공유 금지)
   ├─ state.json                ← 마지막으로 받은 버전 기록 (자동 생성)
   ├─ logs/fetch.log            ← 실행 기록 (자동 생성)
   └─ 01_raw/<et|talos>/<날짜>/ ← 받은 룰 파일 (자동 생성, 번호 = 처리 순서)

[사용법]
    python 1_1_fetch_rules.py                    # ET + Talos 둘 다 확인
    python 1_1_fetch_rules.py --source et        # ET 만
    python 1_1_fetch_rules.py --source talos     # Talos 만
    python 1_1_fetch_rules.py --force            # 변경 여부와 상관없이 강제로 다시 받기

- 파이썬 기본 라이브러리만 사용하므로 pip install 이 필요 없다.
"""

# ── 기본 라이브러리 불러오기 ────────────────────────────────────────────
import argparse            # 명령줄 옵션(--source, --force) 처리
import datetime as dt      # 오늘 날짜(폴더 이름), 다운로드 시각 기록
import hashlib             # 받은 파일의 sha256 지문 계산
import json                # state.json 읽기/쓰기
import logging             # 화면/파일 로그 출력
import os                  # 환경변수(os.environ) 읽기
import re                  # 정규식 - 로그에서 OINKCODE 가리기
import sys                 # 종료 코드 반환, 콘솔 존재 여부 확인
import time                # 재시도 전 잠깐 대기
import urllib.error        # HTTP 오류 종류
import urllib.request      # 웹에서 파일 받기
from pathlib import Path   # 파일/폴더 경로 다루기

# ── 경로 설정 ──────────────────────────────────────────────────────────
# 이 파일이 있는 폴더를 기준으로 삼는다.
# → 작업 스케줄러처럼 다른 위치에서 실행해도 항상 같은 폴더에 저장된다.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "01_raw"         # 룰 원본 저장 폴더 (번호 = 처리 순서)
LOG_DIR = BASE_DIR / "logs"           # 로그 폴더
STATE_FILE = BASE_DIR / "state.json"  # 마지막 버전 기록 파일
ENV_FILE = BASE_DIR / ".env"          # 설정 파일

# ── 다운로드 관련 상수 ──────────────────────────────────────────────────
USER_AGENT = "rule-data-fetcher/1.0"  # 사이트에 알려줄 프로그램 이름
TIMEOUT = 60                          # 응답 대기 최대 시간(초)
RETRIES = 3                           # 실패 시 최대 시도 횟수

# 이 프로그램 전용 로거. setup_logging() 에서 출력 위치를 정한다.
log = logging.getLogger("fetch_rules")


def load_env(path):
    """.env 파일을 읽어 환경변수(os.environ)에 넣는다.

    .env 예시:
        # 주석은 무시
        SNORT_OINKCODE=abc123...
        TALOS_SNORT_VERSION=29200

    - 빈 줄, # 으로 시작하는 줄, = 이 없는 줄은 건너뛴다.
    - 값 양쪽의 따옴표("..." / '...')는 제거한다.
    - 값 뒤에 공백 + # 으로 붙인 메모(예: KEY=abc  #업무용)는 주석으로 보고 버린다.
    - setdefault 를 쓰므로, 윈도우 환경변수에 이미 같은 이름이 있으면 그 값을 우선한다.
    """
    if not path.exists():           # .env 가 없으면 아무것도 안 함 (ET 는 OINKCODE 없이도 동작)
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)   # 첫 번째 = 기준으로만 나눔 (값에 = 이 있어도 안전)
        os.environ.setdefault(key.strip(), value.replace("\t", " ").split(" #")[0].strip().strip('"').strip("'"))   # 값 뒤 ' #메모' 는 주석


def mask(text):
    """문자열 안의 'oinkcode=XXXX' 를 'oinkcode=****' 로 바꾼다.

    Talos 다운로드 주소에는 OINKCODE 가 그대로 들어가므로,
    오류 메시지나 로그에 주소를 찍을 때 반드시 이 함수를 거쳐서 비밀값 노출을 막는다.
    """
    return re.sub(r"(oinkcode=)[^&\s]+", r"\1****", str(text))


def setup_logging():
    """로그를 '파일(logs/fetch.log)'과 '화면' 두 곳에 출력하도록 설정한다.

    로그 형식: 2026-10-14 14:59:34,473 [INFO] [et] 변경 없음 ...
    """
    LOG_DIR.mkdir(exist_ok=True)    # logs 폴더가 없으면 생성
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 1) 파일 출력 - 기존 내용 뒤에 계속 이어서 기록된다.
    fh = logging.FileHandler(LOG_DIR / "fetch.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.setLevel(logging.INFO)      # INFO 이상(INFO, WARNING, ERROR)만 기록
    log.addHandler(fh)

    # 2) 화면 출력 - 작업 스케줄러는 콘솔 창 없는 pythonw 로 실행하므로
    #    그때는 sys.stdout 이 None 이다. 이 경우 화면 출력은 생략한다.
    if sys.stdout is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)


def load_state():
    """state.json 을 읽어 딕셔너리로 돌려준다. 파일이 없으면(첫 실행) 빈 딕셔너리.

    state.json 예시:
        {
          "et":    {"marker": "11310", "file": "raw\\et\\2026-10-14\\...", ...},
          "talos": {"marker": "591c47da...", ...}
        }
    여기서 "marker" 가 '지난번에 받은 버전'이며, 변경 여부 판단의 기준이 된다.

    파일이 비었거나 깨져 있으면 멈추지 않고 경고만 남긴 뒤 빈 상태로 시작한다.
    (이 경우 다음 실행에서 모든 소스를 다시 받는다. 깨진 파일은 .broken 으로 보관)
    """
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        STATE_FILE.replace(STATE_FILE.with_name(STATE_FILE.name + ".broken"))
        log.warning("state.json 이 비었거나 깨짐 (%s) - 빈 상태로 시작 (원본: state.json.broken)", e)
        return {}


def save_state(state):
    """현재 상태를 state.json 에 저장한다. (사람이 읽기 쉽게 들여쓰기 + 한글 그대로)

    임시 파일(.tmp)에 다 쓴 뒤 이름을 바꾸는 방식이라,
    저장 도중 프로그램이 끊겨도 state.json 이 빈 파일로 남지 않는다.
    """
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def http_get(url, dest=None):
    """URL 에서 데이터를 받는다.

    - dest 가 없으면: 내용을 bytes 로 돌려준다. (version.txt, .md5 같은 작은 파일용)
    - dest 가 있으면: 그 경로에 파일로 저장한다. (룰 zip / tar.gz 같은 큰 파일용)

    실패 처리:
    - 4xx 오류(잘못된 OINKCODE, 접근 거부, 요청 제한 등): 다시 해도 같으므로 즉시 실패
    - 5xx 오류 / 네트워크 끊김 / 타임아웃: 5초, 10초 쉬었다가 최대 3번까지 재시도
    """
    last_err = None
    for attempt in range(1, RETRIES + 1):     # attempt = 1, 2, 3
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                if dest is None:
                    return resp.read()        # 작은 파일: 내용 통째로 반환

                # 큰 파일: 먼저 '파일명.part' 로 받고, 다 받은 뒤에 진짜 이름으로 바꾼다.
                # → 도중에 끊겨도 반쯤 받은 파일이 정상 파일처럼 남지 않는다.
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 64)   # 64KB 씩 나눠 받아 메모리 절약
                        if not chunk:                  # 더 받을 게 없으면 종료
                            break
                        f.write(chunk)
                tmp.replace(dest)             # .part → 최종 파일명 (같은 이름 있으면 덮어씀)
                return None
        except urllib.error.HTTPError as e:
            # 4xx 는 재시도해도 소용없음 (잘못된 OINKCODE, 요청 제한 등)
            if 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code} {mask(url)}") from None
            last_err = e                      # 5xx 는 아래에서 재시도
        except (urllib.error.URLError, OSError) as e:
            last_err = e                      # 네트워크 오류, 타임아웃 등 → 재시도
        log.warning("다운로드 실패 (%d/%d): %s - %s", attempt, RETRIES, mask(url), mask(last_err))
        time.sleep(5 * attempt)               # 1회차 후 5초, 2회차 후 10초 ... 대기
    # 3번 모두 실패
    raise RuntimeError(f"다운로드 실패: {mask(url)} - {mask(last_err)}")


def sha256(path):
    """파일의 sha256 지문값을 계산한다. (나중에 파일이 바뀌었는지/깨졌는지 확인용)

    큰 파일(Talos 약 149MB)도 메모리를 적게 쓰도록 1MB 씩 읽어서 계산한다.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):   # 빈 bytes 가 나올 때까지 반복
            h.update(chunk)
    return h.hexdigest()


def save_file(source, filename, url, marker, state, today):
    """룰 파일을 01_raw/<source>/<오늘날짜>/<filename> 에 받고, state 에 기록한다.

    매개변수:
        source   : "et" 또는 "talos"
        filename : 저장할 파일 이름
        url      : 다운로드 주소
        marker   : 이번에 받은 버전 (ET=version 번호, Talos=md5)
        state    : state.json 내용 (딕셔너리) - 여기에 결과를 써 넣는다
        today    : 오늘 날짜 문자열 (예: "2026-10-14")
    """
    out_dir = RAW_DIR / source / today
    out_dir.mkdir(parents=True, exist_ok=True)   # 폴더가 없으면 상위 폴더까지 생성
    dest = out_dir / filename
    log.info("[%s] 다운로드 시작: %s", source, filename)
    http_get(url, dest)

    # 다운로드 성공 시에만 여기까지 온다. (실패하면 http_get 에서 예외 발생)
    # → 실패했을 때는 state 가 갱신되지 않으므로, 다음 실행 때 자동으로 다시 시도된다.
    state[source] = {
        "marker": marker,                                    # 다음 실행 때 비교할 버전
        "file": str(dest.relative_to(BASE_DIR)),             # 저장 위치
        "sha256": sha256(dest),                              # 파일 지문
        "size": dest.stat().st_size,                         # 파일 크기(bytes)
        "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),  # 받은 시각
    }
    log.info("[%s] 저장 완료: %s (%d bytes)", source, dest.relative_to(BASE_DIR), dest.stat().st_size)


def fetch_et(state, today, force):
    """Proofpoint ET Open 룰을 확인하고, 새 버전이면 받는다.

    ET 는 로그인/OINKCODE 없이 누구나 받을 수 있다.
    version.txt 에 숫자 하나(예: 11310)가 들어 있고, 룰이 갱신될 때마다 이 숫자가 바뀐다.

    반환값: 새로 받았으면 True, 건너뛰었으면 False
    """
    # .env 에 ET_BASE_URL 이 있으면 그 주소를, 없으면 기본 주소를 사용
    base = os.environ.get("ET_BASE_URL", "https://rules.emergingthreats.net/open/snort-2.9.0").rstrip("/")

    version = http_get(f"{base}/version.txt").decode().strip()   # 현재 버전 (작은 요청)
    prev = state.get("et", {}).get("marker")                       # 지난번 버전 (첫 실행이면 None)

    if version == prev and not force:
        log.info("[et] 변경 없음 (version %s) - 건너뜀", version)
        return False

    log.info("[et] 새 버전 감지: %s -> %s", prev, version)
    save_file("et", "emerging-all.rules.zip", f"{base}/emerging-all.rules.zip", version, state, today)
    return True


def fetch_talos(state, today, force):
    """Cisco Talos 룰을 확인하고, 새 스냅샷이면 받는다.

    Talos 는 snort.org 계정의 OINKCODE 가 필요하다. (.env 의 SNORT_OINKCODE)
    파일 버전 번호가 따로 없으므로, 사이트가 제공하는 .md5(파일 지문)를 비교해서
    지문이 바뀌었으면 새 파일로 판단한다.

    ※ snort.org 는 큰 파일을 너무 자주 받으면 차단할 수 있다.
       그래서 항상 작은 .md5 만 먼저 확인하고, 바뀌었을 때만 본 파일(약 149MB)을 받는다.

    반환값: 새로 받았으면 True, 건너뛰었으면 False
    """
    oinkcode = os.environ.get("SNORT_OINKCODE", "").strip()
    if not oinkcode:                       # OINKCODE 가 없으면 Talos 는 건너뜀 (오류 아님)
        log.info("[talos] SNORT_OINKCODE 미설정 - 건너뜀")
        return False

    ver = os.environ.get("TALOS_SNORT_VERSION", "29200")   # 29200 = Snort 2.9.20 용 룰
    filename = f"snortrules-snapshot-{ver}.tar.gz"
    base = f"https://www.snort.org/rules/{filename}"

    # .md5 응답은 "지문값" 또는 "지문값  파일명" 형태 → 첫 단어만 사용
    md5 = http_get(f"{base}.md5?oinkcode={oinkcode}").decode().strip().split()[0]
    prev = state.get("talos", {}).get("marker")

    if md5 == prev and not force:
        log.info("[talos] 변경 없음 (md5 %s) - 건너뜀", md5)
        return False

    log.info("[talos] 새 스냅샷 감지: md5 %s -> %s", prev, md5)
    save_file("talos", filename, f"{base}?oinkcode={oinkcode}", md5, state, today)
    return True


# 소스 이름 → 처리 함수 연결표.
# 나중에 새 룰 소스를 추가하려면 fetch_xxx() 함수를 만들고 여기에 한 줄 추가하면 된다.
SOURCES = {"et": fetch_et, "talos": fetch_talos}


def main():
    """프로그램 시작점. 옵션을 읽고 각 소스를 차례로 처리한다."""
    # 1) 명령줄 옵션 읽기
    parser = argparse.ArgumentParser(description="Snort 룰 정기 수집기")
    parser.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    parser.add_argument("--force", action="store_true", help="변경 여부와 무관하게 다시 다운로드")
    args = parser.parse_args()

    # 2) 준비: 설정 → 로그 → 지난 상태 → 오늘 날짜
    load_env(ENV_FILE)
    setup_logging()
    state = load_state()
    today = dt.date.today().isoformat()    # 예: "2026-10-14" → 저장 폴더 이름

    # 3) 처리할 소스 고르기 (--source 를 안 주면 전부)
    targets = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}

    # 4) 소스별로 실행. 하나가 실패해도 나머지는 계속 진행한다.
    failed = False
    for name, fn in targets.items():
        try:
            if fn(state, today, args.force):   # 새로 받았을 때만
                save_state(state)               # 바로 state.json 저장
        except Exception as e:
            failed = True
            log.error("[%s] 실패: %s", name, mask(e))

    # 5) 종료 코드: 0 = 모두 정상, 1 = 하나라도 실패
    #    (작업 스케줄러의 '마지막 실행 결과'에서 0x0 / 0x1 로 확인할 수 있다)
    return 1 if failed else 0


# 이 파일을 직접 실행했을 때만 main() 을 호출한다.
# (다른 파이썬 파일에서 import 할 때는 자동 실행되지 않음)
if __name__ == "__main__":
    sys.exit(main())
