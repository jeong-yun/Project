"""[5.0a 지도학습 공통] 경로·버전 목록·신규 학습데이터 읽기·모델 설정.  (5.x 공용 보조 모듈, 직접 실행하지 않음)

화면(0_0_run_ui.py)도 이 모듈을 쓰므로 맨 위에서는 표준 라이브러리만 불러온다 (pandas 등은 함수 안에서).

폴더 (모두 이 PC 에만 둔다, GitHub 에 올리지 않음)
  11_supervised/versions.json            버전 목록 + 표준모델 피처 파일 위치
  11_supervised/compared/<기준>/<장비>_*  비교 결과: 샘플별 등급(rows.parquet), 화면용 표(table.json), 요약, 엑셀
  11_supervised/versions/<이름>/          확정한 버전의 내보내기: 기존 + 신규 학습 파일(CSV)·로그, 추가한 샘플 목록
  11_supervised/cache/                   다시 만들 수 있는 캐시: 기준 데이터(parquet), 기준 모델, 샘플별 피처

버전
  - 기준 버전 = 비교의 기준이 되는 기존 학습데이터. 첫 기준(예: 26 상반기 = v26.02.02)은 파일 위치를 등록한다
    (학습 파일 CSV: method, payload, label, 표준모델 피처 열 / 원본 로그: |||, 8필드).
  - 그 뒤 버전(예: 26 하반기)은 기준 버전 + 선별한 신규 샘플. 확정하면 학습 파일·로그를 내보내고 다음 버전의 기준이 된다.
"""

import hashlib
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SUP = BASE / "11_supervised"
VERSIONS_FILE = SUP / "versions.json"
COMPARED = SUP / "compared"
VERDIR = SUP / "versions"
CACHE = SUP / "cache"
DATASET = BASE / "08_dataset"
OLD = BASE / "06_기존 데이터"
DEVICES = ("ips", "waf")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NAME_RE = re.compile(r"^[0-9A-Za-z가-힣][0-9A-Za-z가-힣 ._()\-]{0,39}$")
SEED = 42
GRADES = ["제거", "A", "B", "C"]
GRADE_TEXT = {"제거": "제거 (중복)", "A": "A 부족한 데이터 (기준 모델이 틀림)",
              "B": "B 새 패턴 (기준 모델은 맞힘)", "C": "C 이미 학습된 패턴"}
CONS_CODE = {"1": "OK", "2": "RULE", "3": "CHECK", "4": "BAD"}      # 4. 화면 '전체 검증' 결과 (label 정합성)
CONS_TEXT = {"OK": "정합", "RULE": "정합 (룰 기준)", "CHECK": "확인 필요", "BAD": "불일치", "": "검증 전"}


class Stopped(Exception):
    """화면의 '중지' 요청 (실행기가 단계 사이·묶음마다 확인)."""


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path, data, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def disk_free_gb(path=BASE):
    return shutil.disk_usage(path).free / 2 ** 30


# ── 버전 목록 ────────────────────────────────────────────────────────────

def load_versions():
    """{'feature_code': {장비: 경로}, 'versions': [...]} (파일이 없으면 빈 목록)."""
    v = read_json(VERSIONS_FILE, None) or {}
    v.setdefault("feature_code", {})
    v.setdefault("versions", [])
    return v


def save_versions(v):
    write_json(VERSIONS_FILE, v)


def get_version(v, name):
    for x in v["versions"]:
        if x["name"] == name:
            return x
    raise ValueError(f"버전 '{name}' 이 없습니다")


def check_name(name):
    if not isinstance(name, str) or not NAME_RE.match(name) or name.strip() != name or name.endswith("."):
        raise ValueError("버전 이름은 한글·영문·숫자·공백·._()- 로 40자까지입니다 (예: 26 하반기)")
    return name


def chain(v, name):
    """[기준 버전, 그 기준, …, 첫 기준]."""
    out, seen = [], set()
    while name and name not in seen:
        seen.add(name)
        out.append(name)
        name = get_version(v, name).get("base")
    return out


def version_files(v, ver, dev):
    """버전의 (학습 파일, 원본 로그). 등록한 기준은 등록한 경로, 확정한 버전은 내보내기 폴더."""
    if ver.get("kind") == "external":
        f = (ver.get("files") or {}).get(dev) or {}
        return Path(f.get("train_csv") or "-"), Path(f.get("raw_log") or "-")
    d = VERDIR / ver["name"]
    return d / f"{dev}_train.csv", d / f"{dev}_raw.log"


def feature_code(v, dev):
    p = (v.get("feature_code") or {}).get(dev)
    return Path(p) if p else None


def setup_problems(v, base_name=None, devs=DEVICES):
    """비교를 시작하기 전에 빠진 것 ('넣을 것' 안내)."""
    out = []
    for dev in devs:
        p = feature_code(v, dev)
        if not p or not p.is_file():
            out.append(f"{dev.upper()} 표준모델 피처 파일이 없습니다 ({p or '등록 안 됨'})")
    if not any(x.get("status") == "final" for x in v["versions"]):
        out.append("기준 버전(기존 학습데이터)이 없습니다 - 기준 버전을 등록하세요")
    if base_name:
        try:
            ver = get_version(v, base_name)
            if ver.get("status") != "final":
                out.append(f"'{base_name}' 은 확정 전 버전이라 기준으로 쓸 수 없습니다")
            for dev in devs:
                csv_p, log_p = version_files(v, ver, dev)
                if not csv_p.is_file():
                    out.append(f"{base_name} {dev.upper()} 학습 파일이 없습니다 ({csv_p})")
                if not log_p.is_file():
                    out.append(f"{base_name} {dev.upper()} 원본 로그가 없습니다 ({log_p})")
        except ValueError as e:
            out.append(str(e))
    return out


def default_sources():
    """기준 버전 등록 화면의 기본값: 06_기존 데이터 아래 표준 위치에 있는 파일."""
    def first(pattern):
        found = sorted(OLD.glob(pattern)) if OLD.is_dir() else []
        return str(found[0]) if found else ""
    return {dev: {"train_csv": first(f"학습데이터/{dev}_*.csv"), "raw_log": first(f"{dev}_*_raw.log"),
                  "feature_code": first(f"피처 코드/{dev}_*.py")} for dev in DEVICES}


# ── 비교 결과 위치 ───────────────────────────────────────────────────────

def compared_files(base_name, dev):
    d = COMPARED / base_name
    return {"rows": d / f"{dev}_rows.parquet", "table": d / f"{dev}_table.json", "summary": d / f"{dev}_summary.json",
            "report": d / f"{dev}_report.xlsx"}


def considered_keys(v, base_name, dev):
    """기준 버전까지 확정된 버전들이 이미 비교한 샘플 지문 (다음 비교에서 뺀다)."""
    keys = set()
    for name in chain(v, base_name):
        p = VERDIR / name / f"{dev}_considered.txt"
        if p.is_file():
            keys.update(x for x in p.read_text(encoding="utf-8").split() if x)
    return keys


# ── 신규 학습데이터 (08_dataset) ─────────────────────────────────────────

def dataset_dates():
    if not DATASET.is_dir():
        return []
    return sorted(p.name for p in DATASET.iterdir() if p.is_dir() and DATE_RE.match(p.name))


def log_path(date, dev):
    found = sorted((DATASET / date).glob(f"{dev}_*_snort.log"))
    return found[0] if found else None


def log_stat(date, dev):
    p = log_path(date, dev)
    if p is None:
        return None
    st = p.stat()
    return [p.name, st.st_size, st.st_mtime]


def verify_codes(date, dev, stat):
    """4. 화면 '전체 검증' 결과 (줄마다 1~4). 검증한 뒤 로그가 바뀌었으면 None."""
    v = read_json(DATASET / date / f"{dev}_verify.json", None) or {}
    if not stat or v.get("size") != stat[1] or v.get("mtime") != stat[2]:
        return None
    return v.get("codes") or None


def sample_key(label, rule_id, payload):
    """샘플 내용 지문 (로그를 다시 만들어도 같고, 화면에서 고치면 바뀐다)."""
    return hashlib.sha1(f"{label}\x00{rule_id}\x00{payload}".encode("utf-8", "replace")).hexdigest()[:20]


def payload_key(payload):
    """피처 캐시 지문 (피처는 payload 로만 정해진다)."""
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()[:16]


SAMPLE_COLS = ["date", "line", "payload_raw", "method", "label", "rule_id", "source", "sk", "attack_type", "origin",
               "edited", "cons"]


def split_log_line(s):
    """학습 로그 한 줄(|||, 8필드) → 필드 목록. 앞 6필드는 앞에서, label 은 마지막 '|||' 뒤로 읽는다
    (payload 가 '|' 로 끝나면 '…|id||||anomalies' 처럼 구분자와 붙어 앞에서만 나누면 label 이 '|anomalies' 가 됨).
    8필드가 아니면 그냥 나눈 결과."""
    head = s.split("|||", 6)
    if len(head) == 7 and "|||" in head[6]:
        return head[:6] + head[6].rsplit("|||", 1)
    return s.split("|||")


def read_dataset(date, dev):
    """그 날짜·장비의 신규 학습데이터 → (열 이름: 값 목록, 로그 상태). 줄 번호는 4. 화면과 같다 (빈 줄 빼고 0부터)."""
    cols = {c: [] for c in SAMPLE_COLS}
    p = log_path(date, dev)
    if p is None:
        return cols, None
    stat = log_stat(date, dev)
    lab = p.with_name(p.stem + "_label.jsonl")
    metas = []
    if lab.exists():
        with open(lab, encoding="utf-8") as f:
            metas = [x for x in f if x.strip()]
    codes = verify_codes(date, dev, stat)
    with open(p, "rb") as f:
        for i, line in enumerate(x for x in f if x.strip()):
            fs = split_log_line(line.rstrip(b"\r\n").decode("utf-8", errors="replace"))
            payload, method, label = (fs[6], fs[5], fs[7]) if len(fs) == 8 else ("|||".join(fs), "", "")
            m = json.loads(metas[i]) if i < len(metas) else {}
            rid = m.get("id", "")
            for k, val in (("date", date), ("line", i), ("payload_raw", payload), ("method", method), ("label", label),
                           ("rule_id", rid), ("source", rid.split(":")[0] if rid else ""), ("sk", m.get("sk", "")),
                           ("attack_type", "normal" if label == "normal" else (m.get("attack_type") or "other")),
                           ("origin", m.get("origin", "")), ("edited", m.get("edited", "")),
                           ("cons", CONS_CODE.get(codes[i], "") if codes and i < len(codes) else "")):
                cols[k].append(val)
    return cols, stat


def holdout(rule_id):
    """평가용 분할 (rule_id 단위 70/30). 지문으로 정해 새 룰이 들어와도 이미 있던 룰의 분할은 그대로."""
    return int(hashlib.sha1(f"{SEED}:{rule_id}".encode()).hexdigest()[:8], 16) % 100 < 30


# ── 모델 ────────────────────────────────────────────────────────────────

def make_model(name):
    """표준모델 v26.03.03 재현 설정 (rf / lgbm)."""
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1)


# ── 비교 실행 계획 (화면 6 의 '예상'과 실행기의 전체 진행률이 같은 계산을 쓴다) ─────────────

COMPARE_STEPS = (("base", "기준 데이터"), ("model", "기준 모델"), ("read", "읽기"), ("feat", "피처"), ("grade", "등급"),
                 ("save", "저장"))


def compare_step_text(dev, step, base_name="", mode="new"):
    """비교 단계 이름 (5.5 가 단계를 시작할 때와 실행 계획이 같은 글을 쓴다)."""
    D = dev.upper()
    return {"base": f"{D} 기준 데이터 읽기 ({base_name})", "model": f"{D} 기준 모델 (RF·LGBM, 처음 한 번 학습)",
            "read": f"{D} 신규 학습데이터 읽기", "feat": f"{D} 피처 계산 ({'전체' if mode == 'all' else '새 샘플만'})",
            "grade": f"{D} 등급 매기기", "save": f"{D} 결과 저장"}[step]


def base_cache_ok(ver, dev):
    """기준 데이터 캐시(5.5 load_base)가 지금 학습 파일 그대로인지 (위치·크기·수정 시각)."""
    csv_p, _ = version_files(None, ver, dev)
    meta = read_json(CACHE / "base" / f"{ver['name']}_{dev}.json", None) or {}
    try:
        st = csv_p.stat()
    except OSError:
        return False
    return ((CACHE / "base" / f"{ver['name']}_{dev}.parquet").exists() and meta.get("src") == str(csv_p)
            and meta.get("size") == st.st_size and meta.get("mtime") == st.st_mtime)


def models_cache_ok(ver, dev):
    """기준 모델 4개(5.5 base_models)를 지금 기준 데이터 캐시로 학습해 저장해 두었는지."""
    key = read_json(CACHE / "base" / f"{ver['name']}_{dev}.json", None)
    if not key:
        return False
    for cfg in ("34", "43"):
        for m in ("rf", "lgbm"):
            p = CACHE / "models" / ver["name"] / f"{dev}_{cfg}_{m}.joblib"
            meta = read_json(p.with_suffix(".json"), None) or {}
            meta.pop("cols", None)
            if not p.exists() or meta != key:
                return False
    return True


def dataset_rows():
    """08_dataset 날짜별 줄 수 (날짜 폴더 summary.json 의 events, 두 장비가 같음)."""
    return {d: int((read_json(DATASET / d / "summary.json", {}) or {}).get("events") or 0) for d in dataset_dates()}


def compare_estimate(base_name, rows_by_date=None):
    """비교 전 장비별 상태·추정 (화면 6 과 실행기의 처음 계획이 같이 씀): 지난 비교(시각·행 수·스냅숏), 그 뒤 바뀌거나
    새로 생긴 날짜, 새로 비교할 행 수 추정(est_new), 전체 행 수·로그 크기, 확정된 버전에서 이미 비교한 샘플 수(어림),
    기준 데이터·기준 모델 캐시가 있는지."""
    rows_by_date = dataset_rows() if rows_by_date is None else rows_by_date
    v = load_versions()
    try:
        ver = get_version(v, base_name) if base_name else None
    except ValueError:
        ver = None
    out = {}
    for dev in DEVICES:
        s = read_json(compared_files(base_name, dev)["summary"], None) if ver else None
        d = {"has": bool(s), "total": sum(rows_by_date.values())}
        if s:
            ls = s.get("log_stats") or {}
            d.update(created_at=s.get("created_at"), rows=s.get("rows"), snapshot=s.get("snapshot"), mode=s.get("mode"),
                     changed=[x for x, st in ls.items() if log_stat(x, dev) != st],
                     new_dates=[x for x in rows_by_date if x not in ls and log_path(x, dev)])
            t = (s.get("tables") or {}).get("날짜별") or {}
            k = (t.get("cols") or [""]).index("합계") if "합계" in (t.get("cols") or []) else None
            prev = {r[0]: r[k] for r in t.get("rows", [])} if k is not None else {}
            d["est_new"] = sum(n if x in d["new_dates"] else max(0, n - prev.get(x, 0)) for x, n in rows_by_date.items()
                               if x in d["new_dates"] or x in d["changed"])
        else:
            d["est_new"] = d["total"]
        d["bytes"] = sum(st[1] for st in (log_stat(x, dev) for x in rows_by_date) if st)
        try:                                                              # 지문 한 줄 = 20자 + 줄바꿈
            d["excluded"] = sum(p.stat().st_size // 21 for p in (VERDIR / n / f"{dev}_considered.txt"
                                                                 for n in chain(v, base_name)) if p.is_file()) if ver else 0
        except (OSError, ValueError):
            d["excluded"] = 0
        d["base_cached"] = bool(ver) and base_cache_ok(ver, dev)
        d["models_cached"] = bool(ver) and models_cache_ok(ver, dev)
        out[dev] = d
    return out


def _first(*xs):
    return next(x for x in xs if x is not None)


def _old(sec, cached, limit):
    """예전 timing.json 은 캐시를 썼는지 따로 적지 않았다 → limit 초보다 짧으면 캐시를 쓴 시간으로 본다."""
    return sec if sec is not None and (sec < limit) == cached else None


def past_median(timing, tkey):
    """지난 실행 시간 = 최근 3번의 중앙값 (timing.json 의 hist, 5.0 실행기가 끝까지 성공한 실행마다 남김).
    한 번 다른 계산과 겹쳐 느렸던 실행에 예상이 끌려가지 않게. 규모에 비례하는 단계는 행·바이트당 초."""
    h = (timing.get("hist") or {}).get(tkey)
    return statistics.median(h) if h else None


def compare_plan(base_name, devs, mode="new", include_compared=False, timing=None, est=None):
    """비교 실행 계획 [{key, dev, short, text, expected_s, tkey, per}] - 단계마다 지난 실행 시간(cache/timing.json 의 tkey,
    최근 3번 중앙값)으로 잰 예상 시간. 화면 6 의 '예상'과 실행기의 전체 진행률(5.0 RunCtx)이 같이 쓴다.
      기준 데이터·기준 모델  캐시가 있으면 지난 '캐시 읽기' 시간, 없으면 지난 '만들기·학습' 시간 (기록이 없으면 3·20초 / 200·240초)
      읽기       로그 바이트당 지난 시간 × 지금 로그 크기
      피처       고정 시간(캐시 읽기·합치기) + 새로 계산할 payload 수 ÷ 초당 처리 수
      등급·저장  행당 지난 시간 × 비교할 행 수        (per = 그 규모를 실행기가 note 로 남기는 키)
    est = compare_estimate() 의 장비별 값. 실행 중에 안 값(n_rows 비교할 행 수, n_new 새 payload 수)이 있으면 그 값을 쓰고,
    같은 08_dataset 을 읽는 다른 장비의 추정에도 쓴다 (IPS 를 읽고 나면 WAF 예상도 고쳐짐)."""
    timing = read_json(CACHE / "timing.json", {}) if timing is None else timing
    est = compare_estimate(base_name) if est is None else est

    def fact(dev, k, same):                                # 실행 중에 안 값 (이 장비, 없으면 같은 데이터의 다른 장비)
        e = est.get(dev) or {}
        if k in e:
            return e[k]
        return next((x[k] for d, x in est.items() if d != dev and k in x and x.get(same) == e.get(same)), None)

    plan = [{"key": "prep", "dev": "", "short": "준비", "text": "준비 (계산 모듈·피처 코드 읽기)",
             "expected_s": _first(past_median(timing, "compare:prep"), timing.get("compare:prep"), 10.0), "tkey": "compare:prep"}]
    for dev in devs:
        e = est.get(dev) or {}
        h = lambda k, dev=dev: past_median(timing, f"compare:{dev}:{k}")     # 최근 3번 중앙값
        t = lambda k, dev=dev: timing.get(f"compare:{dev}:{k}")              # 마지막 실행 (hist 가 없을 때)
        total, nbytes = e.get("total") or 0, e.get("bytes") or 0
        rows = _first(fact(dev, "n_rows", "total"), max(0, total - (0 if include_compared else e.get("excluded") or 0)))
        new = _first(fact(dev, "n_new", "total" if mode == "all" else "est_new"), total if mode == "all" else e.get("est_new") or 0)
        rate = timing.get(f"feat_rows_per_sec_{dev}") or 120.0
        bc, mc = bool(e.get("base_cached")), bool(e.get("models_cached")) and mode != "all"
        per = lambda k, now: None if h(k) is None else h(k) * now                         # 행·바이트당 × 지금 규모
        last = lambda k, now, then: None if t(k) is None else t(k) * now / t(then) if t(then) else t(k)
        exp = {"base": (_first(h("base:cached"), t("base:cached"), _old(t("base"), True, 60), 3.0) if bc else
                        _first(h("base:build"), t("base:build"), _old(t("base"), False, 60), 200.0)),
               "model": (_first(h("model:cached"), t("model:cached"), _old(t("model"), True, 120), 20.0) if mc else
                         _first(h("model:train"), t("model:train"), _old(t("model"), False, 120), 240.0)),
               "read": _first(per("read", nbytes), last("read", nbytes, "read_bytes"), 5 + nbytes / 3e6),
               "feat": _first(h("feat:fixed"), t("feat:fixed"), _old(t("feat"), True, 60), 15.0) + new / rate,
               "grade": _first(per("grade", rows), last("grade", rows, "rows"), 5 + rows * 3e-4),
               "save": _first(per("save", rows), last("save", rows, "rows"), 5 + rows * 1.2e-4)}
        tkey = {"base": "base:cached" if bc else "base:build", "model": "model:cached" if mc else "model:train"}
        scale_key = {"read": "read_bytes", "grade": "rows", "save": "rows"}
        for step, short in COMPARE_STEPS:
            p = {"key": f"{dev}:{step}", "dev": dev, "short": short, "text": compare_step_text(dev, step, base_name, mode),
                 "expected_s": round(max(0.5, exp[step]), 1), "tkey": f"compare:{dev}:{tkey.get(step, step)}"}
            if step in scale_key:
                p["per"] = f"compare:{dev}:{scale_key[step]}"
            if step == "feat":
                p["n"] = int(new)
            plan.append(p)
    return plan
