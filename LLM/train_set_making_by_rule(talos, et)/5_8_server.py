"""[5.8 서버 반영 지원] 화면 6 후보 버전(rev)을 서버(학습 플랫폼)에 반영하는 과정을 돕는다.  (화면 6 후보 카드 '서버 반영')

서버에서는 사용자가 IPS·WAF 각각 직접 진행한다: 추가 데이터 올리기 → decode → 기준 데이터와 결합 → py·sql 피처 → 학습(RF) → 점수 → 지표.
기준 (2026-11-04 사용자)
  - 학습 피처는 디코딩 입력 기준 (답 4: 레시피 3 자체를 decoded_payload 로). SQL 시스템은 원본 payload 로 피처를 만든다.
  - 95%: 혼동행렬 4지표(정확도·precision·recall·f1, 공격 = anomalies, 전체 학습 데이터 기준)가 모두 95% 이상 (학습 데이터 자체 점수).
  - 확인 둘 (답 5): ① 구현 일치 = 같은 원본 입력에서 파이썬 피처 vs SQL 피처, ② 입력 일치 = 같은 모델의 디코딩 입력 vs 원본 입력 예측 98% 이상.
  - 피처 안에서는 디코딩하지 않는다. 인코딩된 모양을 잡는 '피처 정의 판'(보강판)은 피처 세션이 만들고, 여기서는 판을 고른다 (std 기본).

이 모듈이 하는 것 (후보의 최신 rev 하나와 짝, 결과는 11_supervised/versions/<이름>/server/ - 버리면 후보 폴더와 함께 _보관)
  1. 올릴 파일   <장비>_<날짜>_attack_data.csv = 이번 rev 의 추가 샘플 method·payload·label (학습 파일과 같은 payload,
                 label 은 normal/anomalies 그대로, UTF-8·BOM 없음·첫 줄 열 이름·큰따옴표 안 줄바꿈).
                 NUL(0x00) 은 서버에 못 올림 → 기본은 그 행 빼기 (막기·%00 표기도 고를 수 있음). 빈 payload·다른 label 은 막고,
                 빈 method 는 수만 보임 (서버 전처리가 빈 값을 '-' 로). 원래 decode 레시피는 label 을 모두 공격으로 덮으므로
                 '공격 행만 올리기'도 고를 수 있다 (사본을 쓰면 정상 행도 그대로).
  2. 레시피 사본 서버 레시피 원본(로컬 전용 12_server/recipes)을 이번 버전 이름·피처 정의 판으로 고친 사본(.py·README, 장비별 zip).
                 레시피 내용·서버 데이터셋 이름 규칙은 로컬 전용 5_8a_server_local.py 에만 있다 - 그 파일이 없으면 사본·디코딩 점검·
                 승인 조건은 꺼지고, 올릴 파일·원본 피처 미리 보기·결과 기록만 된다.
  3. 로컬 사전 점검 (check, 화면 job 'server' · 잠금 server_check · 상태 .run/server_check.json)
       학습 파일 전체(올릴 파일에서 뺀 행은 빼고)의 payload 를 레시피 1 로 디코딩해 고른 판의 피처를 만들고(묶음 프로세스 4개, payload
       지문 캐시 11_supervised/cache/server_feat/), 서버 Train 처럼 RF 트리 수 100·200·300 × 최대 깊이 6·16·25·35 그리드(그리드 행 표본
       15만의 80/20, 검증 f1 로 고름)에서 고른 조합으로 전체를 학습한다 (--no-grid = RF 100 트리·깊이 제한 없음).
       95% 미리 보기  학습 데이터 자체를 예측한 4지표 (전체 · 기준 · 추가) + 참고 f1 만 95% (지난번 통과 방식). 참고로 그리드 검증분.
       ② 98% 미리 보기 같은 모델로 디코딩 입력 vs 원본 입력(학습 파일 피처 = SQL 쪽 값, 판이 std 가 아니면 계산) 예측 일치 - 전체 행.
                      일치하지 않는 추가 샘플의 원인(피처·인코딩·룰·유형)과 목록 CSV.
       ① 구현 일치    같은 원본에서 파이썬 vs SQL 의 줄바꿈 처리 차이(SQL 만 공백으로 바꾸는 글자)가 든 행과 예측 차이 (Java 정규식 차이는
                      피처 세션의 검사).
  4. 서버 결과   장비별 TP·FP·TN·FN → 4지표·95% 판정, ② 예측 일치 % → 98% 판정, ① 다른 칸 수 (기록·표시만). 서버의 틀린 행·예측 다른 행
                 CSV 를 올리면 우리 추가 샘플과 payload 로 짝지어 룰·유형별로 보인다.
  5. 체크리스트  서버 단계마다 완료·시각·메모.
  6. 넘겨줄 묶음 사본·올릴 파일·피처 사전·SQL·점검 결과·서버 기록을 설명(README_설명.md)과 함께 zip 하나로 (bundle).
  최종 승인(5_7 approve)은 5_8a 가 있으면 장비마다 서버 4지표 95% + ② 예측 일치 98% 를 보고, 못 넘거나 결과가 없으면 사유를 받는다 (gate).

    python 5_8_server.py prepare --version "26 하반기" [--tag 261104] [--nul drop|block|pct] [--labels keep|attack_only] [--ver std]
    python 5_8_server.py check --version "26 하반기" --dev ips waf [--ver std] [--grid-rows 150000 | --no-grid] [--workers 4]
    python 5_8_server.py bundle --version "26 하반기"
    python 5_8_server.py state --version "26 하반기"
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("sup_common_srv", "5_0a_common.py")
RT = _load("runtime_srv", "0_0a_runtime.py")
LOCAL_FILE = "5_8a_server_local.py"                 # 로컬 전용 (레시피 사본 규칙·디코딩·피처 정의 판) - 공개본에는 없음
LOCK = "server_check"
TARGET_METRIC, TARGET_AGREE = 0.95, 0.98
METRICS4 = ("accuracy", "precision", "recall", "f1")
NUL_MODES = {"drop": "그 행 빼기 (기본)", "block": "막기 (NUL 이 든 샘플이 있으면 파일을 만들지 않음)", "pct": "%00 으로 표기 바꾸기"}
LABEL_MODES = {"keep": "정상 행도 올리기 (사본 1 을 씀 - label 그대로)",
               "attack_only": "공격 행만 올리기 (원래 decode 레시피의 label 덮기를 피함 - 정상 행은 서버 데이터에 안 들어감)"}
STD = "std"
UPLOAD_MAX = 50 * 2 ** 20                            # 화면에서 올리는 서버 결과 CSV 하나의 최대 크기
WORKERS, CHUNK = 4, 2000                             # 디코딩·피처 계산 프로세스 수·묶음 크기
CACHE_VER = 1
# 서버 Train (사용자 2026-11-04): 열 = 피처 + payload (method 안 씀 - 로컬은 payload 를 학습 피처로 안 쓰는 것으로 봄),
# RF 트리 수 × 최대 깊이 그리드에서 가장 좋은 값. 로컬도 같은 그리드를 f1 로 고른다 (그리드는 표본 행의 80/20 나누기, 고른 조합으로 전체 다시 학습).
GRID_TREES, GRID_DEPTH, GRID_METRIC, GRID_ROWS = (100, 200, 300), (6, 16, 25, 35), "f1", 150000
IMPL_REF = 0.998                                     # ① 참고 기준 (실제 시스템 입력은 다른 업무자가 99.8% 이상 같은지 확인 뒤 적용)
PREV_FILE = BASE / "12_server" / "server_results.json"   # 로컬 전용: 지난 버전의 서버 결과 (예: 26 상반기 - 사용자가 알려 준 값)
GENERIC_STEPS = [("upload", "올리기", ""), ("r1", "1 decode", ""), ("r2", "2 결합", ""), ("r3", "3 py 피처 (디코딩, 학습용)", ""),
                 ("r3r", "3r py 피처 (원본, ① 확인용)", ""), ("r4", "4 sql 피처 (원본)", ""), ("train", "Train", ""),
                 ("score", "Score", ""), ("r6", "6 지표 (95%)", ""), ("score_sql", "Score (sql)", ""), ("r5", "5 ① 구현 일치", ""),
                 ("r5b", "5b ② 입력 일치 (98%)", "")]
STEP_KEYS = tuple(k for k, _, _ in GENERIC_STEPS)
# ② 미달 원인: 일치하지 않는 행의 원본 payload 에 든 인코딩 (소문자로 찾음)
ENC = {"%20 공백": r"%20", "+ 공백": r"\+", "%2f /": r"%2f", "%2e .": r"%2e", "%5c \\": r"%5c", "%27 '": r"%27", "%22 \"": r"%22",
       "%3c <": r"%3c", "%3e >": r"%3e", "%3b ;": r"%3b", "%28 (": r"%28", "%29 )": r"%29", "%3d =": r"%3d", "%7c |": r"%7c",
       "%60 `": r"%60", "%25 이중 인코딩": r"%25[0-9a-f]{2}", "%u 유니코드": r"%u[0-9a-f]{4}", "%0a/%0d 줄바꿈": r"%0[ad]",
       "%00 NUL": r"%00", "HTML 엔티티": r"&(?:#\d+|#x[0-9a-f]+|lt|gt|quot|amp|apos);", "\\uXXXX": r"\\u[0-9a-f]{4}"}
_ENC_RX = {k: re.compile(v) for k, v in ENC.items()}
_HTTP_END = re.compile(r" HTTP/\d(\.\d)?(?=\\n|$)")
_wlock = threading.Lock()
csv.field_size_limit(2 ** 31 - 1)


class Stopped(Exception):
    pass


# ── 공통 ────────────────────────────────────────────────────────────────

_LOCAL = {}


def local():
    """로컬 전용 5_8a (레시피 사본·디코딩·피처 정의 판). 없으면 None. 파일이 바뀌면 다시 읽는다."""
    p = BASE / LOCAL_FILE
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    if _LOCAL.get("mt") != mt:
        _LOCAL.update(mt=mt, mod=_load("server_local_srv", LOCAL_FILE))
    return _LOCAL["mod"]


def _features():
    if "F" not in _LOCAL:
        _LOCAL["F"] = _load("sup_features_srv", "5_0b_features.py")
    return _LOCAL["F"]


def train_payload(s):
    """학습 로그 payload → 학습 파일 payload (5_0b raw_to_payload 와 같음 - 화면 프로세스는 pandas 를 부르지 않으려고 여기 둠.
    점검(check)이 이 결과를 학습 파일의 추가 행과 맞춰 본다)."""
    if not isinstance(s, str):
        return ""
    m = _HTTP_END.search(s)
    if not m:
        return s
    head, tail = s[:m.end()], s[m.end():]
    return head + tail.replace("\\r\\n", "\n").replace("\\n", "\n")


def pkey(payload):
    """서버에서 받은 CSV 의 payload 와 우리 샘플을 짝짓는 지문 (줄바꿈 \\r\\n·\\r → \\n)."""
    p = str(payload or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha1(p.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def metrics_from_counts(tp, fp, tn, fn):
    tp, fp, tn, fn = (int(x) for x in (tp, fp, tn, fn))
    n = tp + fp + tn + fn
    pr = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = (2 * pr * rc / (pr + rc) if pr + rc else 0.0) if pr is not None and rc is not None else None
    r6 = lambda x: None if x is None else round(x, 6)
    return {"accuracy": r6((tp + tn) / n if n else None), "precision": r6(pr), "recall": r6(rc), "f1": r6(f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n}


def _metrics(y, p):
    """y, p = bool 배열 (True = anomalies)."""
    return metrics_from_counts(int((y & p).sum()), int((~y & p).sum()), int((~y & ~p).sum()), int((y & ~p).sum()))


def _pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


def m95_fails(m):
    if not m:
        return ["4지표 없음"]
    return [f"{k} {_pct(m.get(k))} < 95%" for k in METRICS4 if m.get(k) is None or m[k] < TARGET_METRIC]


def f1_ok(m):
    """참고 판정: f1 만 95% 이상 (지난 버전에서 recall 이 95% 미만이어도 f1 이 95% 이상이라 통과한 방식). 승인 조건은 4지표 그대로."""
    return None if not m or m.get("f1") is None else m["f1"] >= TARGET_METRIC


def prev_server(v, ver):
    """기준 버전의 지난 서버 결과 {version, dev: {장비: 4지표}, note}: 승인 기록의 server(5_8 로 넣은 것) → 없으면 로컬 기록 파일."""
    base = ver.get("base") or ""
    try:
        b = C.get_version(v, base)
    except ValueError:
        return None
    srv = (b.get("approval") or {}).get("server") or {}
    dev = {d: (x or {}).get("metrics") for d, x in (srv.get("dev") or {}).items() if (x or {}).get("metrics")}
    if dev:
        return {"version": base, "dev": dev, "note": f"'{base}' 최종 승인 때 넣은 서버 결과", "source": "승인 기록"}
    rec = (C.read_json(PREV_FILE, {}) or {}).get(base)
    if rec and rec.get("dev"):
        return {"version": base, "dev": rec["dev"], "note": rec.get("note", ""), "source": rec.get("source", PREV_FILE.name)}
    return None


def _ver(name):
    """→ (versions, 버전, 최신 rev, manifest). 1차 확정(후보) 또는 확정한 버전만."""
    v = C.load_versions()
    ver = C.get_version(v, name)
    if ver.get("status") not in ("candidate", "final") or not ver.get("rev"):
        raise ValueError("1차 확정(후보 rev)을 한 버전만 서버에 반영합니다")
    rev = int(ver["rev"])
    man = C.read_json(C.VERDIR / name / "manifest.json", None)
    if not man or man.get("rev") != rev:
        raise ValueError(f"{name}/manifest.json 이 rev {rev} 의 것이 아닙니다 - 다시 내보내세요")
    return v, ver, rev, man


def base_tag(v, ver):
    """기준 버전의 날짜 6자리 (태그 v26.02.02 → 260202, 없으면 기준 학습 파일 이름에서)."""
    b = C.get_version(v, ver["base"])
    d = re.sub(r"\D", "", b.get("tag") or "")
    if len(d) == 6:
        return d
    for dev in C.DEVICES:
        m = re.search(r"(?<!\d)(\d{6})(?!\d)", Path(((b.get("files") or {}).get(dev) or {}).get("train_csv") or "").name)
        if m:
            return m.group(1)
    return ""


def srv_dir(name):
    return C.VERDIR / name / "server"


def rev_dir(name, rev):
    return srv_dir(name) / f"rev{int(rev)}"


def load_rec(name):
    return C.read_json(srv_dir(name) / "server.json", {}) or {}


def _save_rec(name, rec):
    C.write_json(srv_dir(name) / "server.json", rec)


def _rev_rec(rec, rev):
    r = rec.setdefault("revs", {}).setdefault(str(int(rev)), {})
    r.setdefault("dev", {})
    return r


def _added_sha(man, dev):
    return (((man.get("dev") or {}).get(dev) or {}).get("files") or {}).get(f"{dev}_added.csv", {}).get("sha256")


def read_added(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def upload_rows(name, dev):
    """추가 행 = 이번 rev 의 <장비>_added.csv 순서 (학습 파일 끝 행과 같은 순서·같은 payload)."""
    out = []
    for r in read_added(C.VERDIR / name / f"{dev}_added.csv"):
        out.append({"method": r.get("method") or "", "payload": train_payload(r.get("payload") or ""), "label": r.get("label") or "",
                    "sk": r.get("sk") or "", "rule_id": r.get("rule_id") or "", "attack_type": r.get("attack_type") or "",
                    "origin": r.get("origin") or "", "added_by": r.get("added_by") or ""})
    return out


def _names(L, dev, tag, btag):
    if L and btag:
        return L.names(dev, tag, btag)
    return {"upload": f"{dev}_{tag}_attack_data"}


def versions_of(L):
    """장비별 고를 수 있는 피처 정의 판 [{ver, desc, at}] (5_8a 가 없으면 std 만)."""
    out = {}
    for dev in C.DEVICES:
        try:
            out[dev] = L.feature_versions(dev) if L else [{"ver": STD, "desc": "표준모델 피처 파일 그대로", "at": ""}]
        except Exception as e:
            out[dev] = [{"ver": STD, "desc": f"표준모델 피처 파일 그대로 (판 목록 오류: {str(e)[:120]})", "at": ""}]
    return out


def _check_ver(L, ver):
    ver = ver or STD
    if ver == STD:
        return ver
    if not L:
        raise ValueError(f"{LOCAL_FILE} 가 없어 피처 정의 판 '{ver}' 을 쓸 수 없습니다")
    vs = versions_of(L)
    miss = [d.upper() for d in C.DEVICES if ver not in [x["ver"] for x in vs[d]]]
    if miss:
        raise ValueError(f"피처 정의 판 '{ver}' 이 {', '.join(miss)} 에 없습니다 (12_server/features/out - 피처 세션이 만듦)")
    return ver


def prepared_excl(name, rev, dev):
    """올릴 파일을 만들 때 뺀 추가 행의 sk (NUL 빼기·공격 행만) - 점검도 서버 데이터와 같게 뺀다."""
    r = ((load_rec(name).get("revs") or {}).get(str(int(rev))) or {})
    up = (((r.get("dev") or {}).get(dev) or {}).get("upload") or {})
    return set(up.get("excluded_sks") or []), r.get("labels") or "keep", bool(up)


# ── 1·2. 올릴 파일·레시피 사본 ───────────────────────────────────────────

def _scan(rows, brk):
    c = Counter()
    ex = {"nul": [], "empty_payload": [], "bad_label": []}
    for r in rows:
        p, mth = r["payload"], r["method"]
        if "\x00" in p or "\x00" in mth:
            c["nul_rows"] += 1
            ex["nul"].append(r["sk"])
        if not p.strip():
            c["empty_payload"] += 1
            ex["empty_payload"].append(r["sk"])
        if not mth.strip():
            c["empty_method"] += 1
        if r["label"] not in ("normal", "anomalies"):
            c["bad_label"] += 1
            ex["bad_label"].append(r["sk"])
        if r["label"] == "normal":
            c["normal_rows"] += 1
        if "\n" in p:
            c["newline_rows"] += 1
        if "\r" in p:
            c["cr_rows"] += 1
        if '"' in p:
            c["quote_rows"] += 1
        if brk and any(ch in p for ch in brk):
            c["sql_break_rows"] += 1
    keys = ("nul_rows", "empty_payload", "empty_method", "bad_label", "normal_rows", "newline_rows", "cr_rows", "quote_rows", "sql_break_rows")
    return {k: int(c.get(k, 0)) for k in keys}, ex


def prepare(name, tag="", nul="drop", labels="keep", ver=STD):
    """최신 rev 의 올릴 파일(장비별 CSV)과 레시피 사본(5_8a 가 있을 때)을 server/rev<n>/ 에 만든다."""
    v, vv, rev, man = _ver(name)
    L = local()
    tag = str(tag or time.strftime("%y%m%d")).strip()
    btag = base_tag(v, vv)
    if not re.fullmatch(r"\d{6}", tag):
        raise ValueError("데이터 이름 날짜는 숫자 6자리 (yymmdd, 예: 261104)")
    if tag == btag:
        raise ValueError(f"데이터 이름 날짜 {tag} 가 기준 버전 날짜와 같습니다 - 서버의 기준 데이터를 덮어쓰지 않게 다른 날짜를 쓰세요")
    if nul not in NUL_MODES or labels not in LABEL_MODES:
        raise ValueError("nul·labels")
    ver = _check_ver(L, ver)
    brk = L.sql_break_chars() if L else ""
    rows, scans, errors = {}, {}, []
    for dev in C.DEVICES:
        rows[dev] = upload_rows(name, dev)
        scans[dev] = _scan(rows[dev], brk)
        s, ex = scans[dev]
        md = (man.get("dev") or {}).get(dev) or {}
        if len(rows[dev]) != int(md.get("added_rows") or 0):
            errors.append(f"{dev.upper()} 추가 샘플 목록 {len(rows[dev])}행이 manifest 의 추가 행 {md.get('added_rows')} 와 다릅니다 - 다시 내보내세요")
        if s["nul_rows"] and nul == "block":
            errors.append(f"{dev.upper()} NUL(0x00) 이 든 샘플 {s['nul_rows']}개 - 서버에 올릴 수 없습니다 "
                          f"('그 행 빼기'나 '%00 으로 표기 바꾸기'를 고르세요, sk {', '.join(ex['nul'][:5])})")
        if s["empty_payload"]:
            errors.append(f"{dev.upper()} payload 가 빈 샘플 {s['empty_payload']}개 (sk {', '.join(ex['empty_payload'][:5])})")
        if s["bad_label"]:
            errors.append(f"{dev.upper()} label 이 normal/anomalies 가 아닌 샘플 {s['bad_label']}개 (sk {', '.join(ex['bad_label'][:5])})")
    if errors:
        raise ValueError(" / ".join(errors))
    out = rev_dir(name, rev)
    out.mkdir(parents=True, exist_ok=True)
    now = C.now_iso()
    devs = {}
    for dev in C.DEVICES:
        nm = _names(L, dev, tag, btag)
        p = out / f"{nm['upload']}.csv"
        for old in out.glob(f"{dev}_*_attack_data.csv"):  # 날짜를 바꿔 다시 만들면 예전 파일은 지움 (같은 rev 폴더)
            if old.name != p.name:
                old.unlink()
        tmp = p.with_name(p.name + ".tmp")
        lab, excl, why = Counter(), [], Counter()
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["method", "payload", "label"])
            for r in rows[dev]:
                pl, mth = r["payload"], r["method"]
                has_nul = "\x00" in pl or "\x00" in mth
                if has_nul and nul == "drop":
                    excl.append(r["sk"])
                    why["NUL"] += 1
                    continue
                if labels == "attack_only" and r["label"] != "anomalies":
                    excl.append(r["sk"])
                    why["정상 행"] += 1
                    continue
                if has_nul and nul == "pct":
                    pl, mth = pl.replace("\x00", "%00"), mth.replace("\x00", "%00")
                w.writerow([mth, pl, r["label"]])
                lab[r["label"]] += 1
        tmp.replace(p)
        s, ex = scans[dev]
        up = {"file": p.name, "dataset": nm["upload"], "rows": int(sum(lab.values())), "added_rows": len(rows[dev]),
              "label": {k: int(lab.get(k, 0)) for k in ("anomalies", "normal")}, "methods": len({r["method"] for r in rows[dev]}),
              "bytes": p.stat().st_size, "sha256": _sha256(p), "checks": s, "examples": {k: x[:20] for k, x in ex.items()},
              "nul_replaced": s["nul_rows"] if nul == "pct" else 0, "excluded": dict(why), "excluded_sks": excl, "at": now}
        cp = None
        if L and btag:
            note = (LABEL_MODES["attack_only"] + f" - 정상 {why.get('정상 행', 0)}행 뺌" if labels == "attack_only"
                    else LABEL_MODES["keep"])
            cp = L.make_copies(dev, tag, btag, out / f"{dev}_recipes", name, rev, ver, note)
            for old in out.glob(f"{dev}_*_레시피_사본.zip"):
                if old.name != cp["zip"]:
                    old.unlink()
            (out / cp["zip"]).write_bytes((out / f"{dev}_recipes" / cp["zip"]).read_bytes())
            cp = {"zip": cp["zip"], "files": cp["files"], "names": cp["names"], "changes": cp["changes"],
                  "rules_version": cp.get("rules_version"), "ver": cp.get("ver"), "at": now}
        devs[dev] = {"upload": up, "copies": cp}
    with _wlock:
        rec = load_rec(name)
        r = _rev_rec(rec, rev)
        old_tag = r.get("tag")
        r.update(tag=tag, base_tag=btag, nul=nul, labels=labels, ver=ver, prepared_at=now,
                 added_sha256={d: _added_sha(man, d) for d in C.DEVICES})
        for dev, x in devs.items():
            d = r["dev"].setdefault(dev, {})
            d.update(x)
            if old_tag and old_tag != tag:
                d["renamed_from"] = old_tag
        _save_rec(name, rec)
    return {"rev": rev, "tag": tag, "base_tag": btag, "ver": ver, "labels": labels, "nul": nul,
            "dev": {d: {"upload": x["upload"]["file"], "rows": x["upload"]["rows"], "excluded": x["upload"]["excluded"],
                        "copies": (x["copies"] or {}).get("zip")} for d, x in devs.items()}}


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── 4·5. 서버 결과·체크리스트 ───────────────────────────────────────────

def _dev_rec(name, dev, rev=None):
    if dev not in C.DEVICES:
        raise ValueError("dev")
    v, ver, cur, man = _ver(name)
    rev = int(rev or cur)
    if rev != cur:
        raise ValueError(f"최신 rev {cur} 에만 기록합니다")
    return rev


def _int(b, k, need=True):
    s = str(b.get(k, "") if b.get(k) is not None else "").replace(",", "").strip()
    if not s:
        if need:
            raise ValueError(f"{k} 를 넣으세요")
        return None
    try:
        x = int(s)
    except ValueError:
        raise ValueError(f"{k} 는 0 이상 정수")
    if x < 0:
        raise ValueError(f"{k} 는 0 이상 정수")
    return x


def record(name, dev, kind, b):
    """서버 결과 기록. kind = metrics (tp·fp·tn·fn) | agree (② value % 또는 0~1) | impl (① diff_cells[, diff_rows, total_rows])
    | wrong_csv·diff_csv (data = base64 CSV) | clear (what)."""
    rev = _dev_rec(name, dev, b.get("rev"))
    memo = str(b.get("memo") or "")[:500]
    now = C.now_iso()
    if kind == "metrics":
        try:
            vals = [int(str(b.get(k, "")).replace(",", "").strip()) for k in ("tp", "fp", "tn", "fn")]
        except ValueError:
            raise ValueError("TP·FP·TN·FN 은 0 이상 정수")
        if min(vals) < 0 or sum(vals) == 0:
            raise ValueError("TP·FP·TN·FN 은 0 이상 정수 (모두 0 은 안 됨)")
        item = dict(metrics_from_counts(*vals), at=now, memo=memo, source="입력")
    elif kind == "agree":
        try:
            x = float(str(b.get("value", "")).replace("%", "").strip())
        except ValueError:
            raise ValueError("예측 일치는 % 수 (예: 98.7)")
        x = x / 100 if x > 1 else x
        if not 0 <= x <= 1:
            raise ValueError("예측 일치는 0~100%")
        item = {"value": round(x, 6), "at": now, "memo": memo, "source": "입력"}
        for k in ("agree_rows", "total_rows"):
            if str(b.get(k) or "").strip():
                item[k] = _int(b, k)
    elif kind == "impl":
        item = {"diff_cells": _int(b, "diff_cells"), "diff_rows": _int(b, "diff_rows", False), "total_rows": _int(b, "total_rows", False),
                "at": now, "memo": memo, "source": "입력"}
        if item["diff_rows"] is not None and item["total_rows"]:
            item["rate"] = round(1 - item["diff_rows"] / item["total_rows"], 6)
    elif kind in ("wrong_csv", "diff_csv"):
        import base64
        data = base64.b64decode(b.get("data") or "")
        if not data or len(data) > UPLOAD_MAX:
            raise ValueError("파일이 비었거나 50MB 를 넘습니다")
        item = dict(match_csv(name, dev, data.decode("utf-8-sig", "replace"), kind), at=now, memo=memo,
                    filename=str(b.get("filename") or "")[:200])
    elif kind == "clear":
        what = b.get("what")
        if what not in ("metrics", "agree", "impl", "wrong_csv", "diff_csv"):
            raise ValueError("what")
        with _wlock:
            rec = load_rec(name)
            ((_rev_rec(rec, rev)["dev"].setdefault(dev, {})).setdefault("server", {})).pop(what, None)
            _save_rec(name, rec)
        return {"cleared": what}
    else:
        raise ValueError("kind")
    with _wlock:
        rec = load_rec(name)
        _rev_rec(rec, rev)["dev"].setdefault(dev, {}).setdefault("server", {})[kind] = item
        _save_rec(name, rec)
    return item


def match_csv(name, dev, text, kind):
    """서버 CSV (6 의 틀린 행 또는 5b 의 예측 다른 행): payload 로 우리 추가 샘플과 짝지어 룰·유형·label 별로 센다."""
    head = text.split("\n", 1)[0]
    delim = "\t" if head.count("\t") > head.count(",") else ","
    rd = csv.DictReader(io.StringIO(text, newline=""), delimiter=delim)
    cols = [c.strip() for c in (rd.fieldnames or [])]
    if "payload" not in cols:
        raise ValueError(f"payload 열이 없습니다 (열: {', '.join(cols[:12])})")
    idx = {}
    for r in upload_rows(name, dev):
        idx.setdefault(pkey(r["payload"]), r)
        idx.setdefault(pkey(r["payload"].replace("\x00", "%00")), r)
    n, hit = 0, []
    lab = Counter()
    for r in rd:
        r = {(k or "").strip(): v for k, v in r.items()}
        n += 1
        lab[(r.get("label") or "").strip()] += 1
        m = idx.get(pkey(r.get("payload")))
        if m:
            hit.append((m, r))
    top = lambda key: dict(Counter(m[key] or "-" for m, _ in hit).most_common(20))
    return {"rows": n, "added_rows": len(hit), "base_rows": n - len(hit), "label": dict(lab.most_common(6)),
            "by_rule": top("rule_id"), "by_type": top("attack_type"), "by_origin": top("origin"),
            "added_label": dict(Counter(m["label"] for m, _ in hit)), "columns": cols[:30],
            "samples": [{"sk": m["sk"], "rule_id": m["rule_id"], "attack_type": m["attack_type"], "label": m["label"],
                         **{k: (r.get(k) or "")[:40] for k in ("prediction", "prediction_py_decoded", "prediction_sql_raw") if k in r}}
                        for m, r in hit[:300]]}


def set_step(name, dev, key, done=True, memo=""):
    rev = _dev_rec(name, dev)
    if key not in STEP_KEYS:
        raise ValueError("step")
    with _wlock:
        rec = load_rec(name)
        st = _rev_rec(rec, rev)["dev"].setdefault(dev, {}).setdefault("steps", {})
        st[key] = {"done": bool(done), "at": C.now_iso() if done else None, "memo": str(memo or "")[:500]}
        _save_rec(name, rec)
    return st[key]


def gate(name, rev=None):
    """최종 승인 조건 (5_8a 가 있을 때만 켜짐): 장비마다 서버 4지표 모두 95% 이상 + ② 예측 일치 98% 이상 (① 구현 일치는 표시만).
    → {active, rev, met, dev: {장비: {met, fails, metrics, agree, impl}}}"""
    if local() is None:
        return {"active": False, "met": True, "dev": {}, "note": f"{LOCAL_FILE} 가 없어 서버 반영 조건을 보지 않음"}
    v, ver, cur, man = _ver(name)
    rev = int(rev or cur)
    r = (load_rec(name).get("revs") or {}).get(str(rev)) or {}
    out = {"active": True, "rev": rev, "met": True, "dev": {}}
    for dev in C.DEVICES:
        s = (((r.get("dev") or {}).get(dev) or {}).get("server")) or {}
        m, a = s.get("metrics"), s.get("agree")
        fails = ["서버 4지표 없음"] if not m else m95_fails(m)
        if not a:
            fails.append("② 예측 일치(98%) 없음")
        elif a.get("value") is None or a["value"] < TARGET_AGREE:
            fails.append(f"② 예측 일치 {_pct(a.get('value'))} < 98%")
        out["dev"][dev] = {"met": not fails, "fails": fails, "metrics": {k: (m or {}).get(k) for k in METRICS4} if m else None,
                           "agree": (a or {}).get("value"), "impl": (s.get("impl") or {}).get("diff_cells"), "f1_ok": f1_ok(m)}
        out["met"] = out["met"] and not fails
    return out


def state(name):
    """화면 6 '서버 반영' 칸: 레시피·판·올릴 파일·사본·사전 점검·서버 결과·체크리스트·승인 조건 (최신 rev)."""
    v, ver, rev, man = _ver(name)
    L = local()
    if L:
        try:
            avail = dict(L.available(), present=True)
        except Exception as e:                          # 레시피 파일이 깨졌어도 화면은 열림
            avail = {"ok": False, "present": True, "error": str(e)[:300]}
    else:
        avail = {"ok": False, "present": False, "missing": [LOCAL_FILE]}
    rec = load_rec(name)
    r = (rec.get("revs") or {}).get(str(rev)) or {}
    btag = base_tag(v, ver)
    tag = r.get("tag") or time.strftime("%y%m%d")
    stale = {d: bool(r.get("added_sha256")) and (r.get("added_sha256") or {}).get(d) != _added_sha(man, d) for d in C.DEVICES}
    out = {"name": name, "rev": rev, "status": ver.get("status"), "tag": tag, "base_tag": btag, "prepared": bool(r.get("prepared_at")),
           "prepared_at": r.get("prepared_at"), "nul": r.get("nul") or "drop", "nul_modes": NUL_MODES,
           "labels": r.get("labels") or "keep", "label_modes": LABEL_MODES, "ver": r.get("ver") or STD, "versions": versions_of(L),
           "stale": stale, "targets": {"metric": TARGET_METRIC, "agree": TARGET_AGREE, "impl_ref": IMPL_REF}, "local": avail,
           "local_file": LOCAL_FILE, "prev_server": prev_server(v, ver),
           "grid": {"trees": list(GRID_TREES), "depth": list(GRID_DEPTH), "metric": GRID_METRIC, "rows": GRID_ROWS},
           "dev": {}, "older_revs": sorted((int(k) for k in (rec.get("revs") or {}) if k != str(rev)), reverse=True)}
    for dev in C.DEVICES:
        d = (r.get("dev") or {}).get(dev) or {}
        try:
            stp = L.steps(dev, tag, btag) if L and btag else GENERIC_STEPS
        except Exception:
            stp = GENERIC_STEPS
        done = d.get("steps") or {}
        chk = C.read_json(rev_dir(name, rev) / f"check_{dev}.json", None)
        srv = d.get("server") or {}
        up = dict(d.get("upload") or {})
        up.pop("excluded_sks", None)
        out["dev"][dev] = {
            "upload": up or None, "copies": d.get("copies"), "renamed_from": d.get("renamed_from"),
            "names": (L.names(dev, tag, btag) if L and btag else {}),
            "steps": [dict({"key": k, "title": t, "detail": x}, **(done.get(k) or {})) for k, t, x in stp],
            "check": _slim_check(chk), "server": srv,
            "verdict": {"m95": None if not srv.get("metrics") else not m95_fails(srv["metrics"]),
                        "m95_fails": m95_fails(srv.get("metrics")) if srv.get("metrics") else [],
                        "f1_ok": f1_ok(srv.get("metrics")),
                        "a98": None if not srv.get("agree") else (srv["agree"].get("value") or 0) >= TARGET_AGREE,
                        "impl_ok": None if (srv.get("impl") or {}).get("rate") is None else srv["impl"]["rate"] >= IMPL_REF}}
    out["gate"] = gate(name, rev)
    return out


def _slim_check(chk):
    """화면에 보낼 점검 결과 (sk 목록은 빼고 수만)."""
    if not chk:
        return chk
    chk = dict(chk)
    a = chk.get("agree")
    if isinstance(a, dict) and "diff_sks" in a:
        chk["agree"] = dict(a, diff_sks=len(a["diff_sks"]))
    return chk


def files_of(name):
    """내려받을 수 있는 파일 (최신 rev 폴더 바로 아래, 이름만)."""
    v, ver, rev, man = _ver(name)
    d = rev_dir(name, rev)
    return d, ({p.name for p in d.iterdir() if p.is_file() and not p.name.endswith(".tmp")} if d.is_dir() else set())


# ── 6. 넘겨줄 묶음 ───────────────────────────────────────────────────────

def _check_md(dev, chk):
    """점검 결과 JSON → 사람이 읽는 요약 (묶음의 설명)."""
    if not chk:
        return f"## {dev.upper()} 로컬 사전 점검\n\n아직 하지 않았습니다.\n"
    m4 = lambda m: " · ".join(f"{k} {_pct((m or {}).get(k))}" for k in METRICS4)
    ins, ho, ag, im = chk.get("insample") or {}, chk.get("holdout"), chk.get("agree"), chk.get("impl")
    lines = [f"## {dev.upper()} 로컬 사전 점검 ({chk.get('at', '')}, {chk.get('seconds')}초)", "",
             f"- 피처 정의 판 {chk.get('feature_ver')} · 학습 피처 {('디코딩 입력' if chk.get('basis') == 'decoded' else '원본 입력')} · "
             f"{chk.get('rows_used', chk.get('rows'))}행 (기준 {chk.get('base_rows')} + 추가 {chk.get('added_used', chk.get('added_rows'))}, "
             f"올릴 파일에서 뺀 행 {chk.get('excluded', 0)}) · {chk.get('model')}",
             f"- 95% 미리 보기 (학습분 안, 전체): {m4(ins.get('all'))} → 4지표 {'넘음' if not ins.get('fails') else '미달: ' + ', '.join(ins['fails'])}"
             f" · 참고 f1 ≥95% {'넘음' if f1_ok(ins.get('all')) else '미달'}",
             f"  - 기준만: {m4(ins.get('base'))} / 추가만: {m4(ins.get('added'))}"]
    g = chk.get("grid")
    if g:
        lines.append(f"- 그리드 (트리 수 × 최대 깊이, 검증 {g.get('metric')} 로 고름, {g.get('rows')}행{' 표본' if g.get('sampled') else ''}의 80/20): "
                     f"고른 조합 트리 {g['best']['trees']} · 깊이 {g['best']['depth']}")
        lines += [f"  - 트리 {r['trees']} · 깊이 {r['depth']}: {m4(r)}" for r in g.get("table") or []]
    if ho:
        lines.append(f"- 참고 검증분 ({ho.get('note') or '5-fold 한 fold'}, {ho.get('test_rows')}행): {m4(ho.get('all'))}")
    if ag:
        lines += [f"- ② 입력 일치 (디코딩 vs 원본 예측, 전체): {_pct(ag.get('all', {}).get('rate'))} → {'넘음' if ag.get('a98') else '미달'}"
                  f" · 기준 {_pct(ag.get('base', {}).get('rate'))} · 추가 {_pct(ag.get('added', {}).get('rate'))}",
                  f"  - 추가 행 중 다른 {ag.get('added', {}).get('disagree')}개 원인: 피처 {(ag.get('causes') or {}).get('added', {}).get('features')}"
                  f" · 인코딩 {(ag.get('causes') or {}).get('added', {}).get('encodings')}"]
    if im:
        lines.append(f"- ① 구현 일치 (같은 원본, 파이썬 vs SQL 줄바꿈 처리 {', '.join(im.get('chars') or []) or '차이 없음'}): "
                     f"그 글자가 든 행 {im.get('rows')} · 피처 다른 행 {im.get('feat_diff_rows')} · 예측 다른 행 {im.get('pred_diff_rows')}")
    return "\n".join(lines) + "\n"


BUNDLE_README = """# 서버 반영 묶음 - {name} rev {rev}

만든 시각 {at} · 데이터 이름 날짜 {tag} (기준 {btag}) · 피처 정의 판 {ver} · 올릴 파일: {labels} · NUL: {nul}

> 회사 내부용 (서버 레시피·표준 피처 정규식이 들어 있음) - 공개 저장소·외부로 보내지 말 것.

## 무엇인가
- 화면 6 의 후보 버전 '{name}' (rev {rev}) 의 추가 샘플을 서버(학습 플랫폼)에 반영하는 데 필요한 것을 한 번에 묶었다.
- 기준: ① 구현 일치 = 같은 원본 입력에서 파이썬 피처 = SQL 피처 (참고 99.8%), ② 입력 일치 = 같은 모델의 디코딩 입력(파이썬) vs 원본 입력(SQL)
  예측 98% 이상, 95% = 혼동행렬 4지표(정확도·precision·recall·f1, anomalies 기준) 모두 95% 이상 (학습 데이터 자체 점수, 참고로 f1 만 95%
  이상이면 지난번 통과 방식). 학습 피처는 디코딩 입력 기준.
- 서버 Train: 열 = 피처 + payload (method 안 씀), RF 트리 수 100·200·300 × 최대 깊이 6·16·25·35 그리드. 이 PC 의 점검도 같은 그리드를
  검증 f1 로 고른다 (payload 는 학습 피처로 안 쓰는 것으로 봄). ② 는 Score 레시피 결과 둘(3 표 = 6 에 쓰는 것, 4 표에 한 번 더)을 비교한다.

## 지난 서버 결과 (참고)
{prev}

## 파일 (장비별 폴더 ips/·waf/)
- `<장비>_{tag}_attack_data.csv` - 서버에 올릴 추가 샘플 (method, payload, label; UTF-8·BOM 없음·큰따옴표 안 줄바꿈).
- `recipes/` - 레시피 사본 .py (서버의 Python 레시피 편집기에 붙여 넣기)와 순서·데이터셋 이름 README.txt.
  1 decode · 2 결합 · 3 py 피처(디코딩, 학습용) · 3r py 피처(원본, ① 확인용) · 4 sql 피처(원본) · 5 ① 구현 일치(원래 레시피 5) ·
  5b ② 입력 일치 · 6 지표.
- `features/` - 고른 피처 정의 판의 정규식 사전(regex.json·레시피 3 셀)과 Spark SQL(레시피 4 셀). std 는 표준모델 피처 파일 그대로.
- `check_<장비>.json`·`check_<장비>_diff.csv` - 이 PC 의 로컬 사전 점검 결과와 ② 가 다른 추가 샘플 목록 (아래 요약).
- `server.json` - 화면에 넣은 서버 결과·체크리스트 기록.

## 레시피 1 (decode) 주의
- 원래 레시피 1 은 label 을 모두 '1'(공격)로 덮는다 → 정상 행이 공격이 된다. 첨부 형식 열 빼기(열이 없어 오류)와 4월 첨부 기준 행 번호
  빼기(엉뚱한 행이 빠짐)도 우리 파일에 맞지 않는다 → 사본 1 을 쓴다. 원래 레시피 1 을 꼭 써야 하면 '공격 행만 올리기'로 만든 파일을 쓴다.

## 로컬 사전 점검 요약
{checks}

## 서버 결과 (화면에 넣은 것)
{server}
"""


def bundle(name):
    """넘겨줄 묶음 zip (server/rev<n>/서버반영_<이름>_rev<n>_<날짜>.zip) → 파일 이름."""
    v, vv, rev, man = _ver(name)
    L = local()
    rec = load_rec(name)
    r = (rec.get("revs") or {}).get(str(rev)) or {}
    if not r.get("prepared_at"):
        raise ValueError("먼저 '올릴 파일·레시피 사본 만들기'를 하세요")
    out = rev_dir(name, rev)
    tag, ver = r.get("tag"), r.get("ver") or STD
    zp = out / f"서버반영_{re.sub(r'[^0-9A-Za-z가-힣._-]+', '_', name)}_rev{rev}_{tag}.zip"
    checks, srvs = [], []
    with zipfile.ZipFile(zp.with_name(zp.name + ".tmp"), "w", zipfile.ZIP_DEFLATED) as z:
        for dev in C.DEVICES:
            d = (r.get("dev") or {}).get(dev) or {}
            up, cp = d.get("upload") or {}, d.get("copies") or {}
            if up.get("file") and (out / up["file"]).exists():
                z.write(out / up["file"], f"{dev}/{up['file']}")
            rd = out / f"{dev}_recipes"
            for f in cp.get("files") or []:
                if (rd / f).exists():
                    z.write(rd / f, f"{dev}/recipes/{f}")
            if L and ver != STD:
                for k in ("regex.json", "recipe3.py", "recipe4.py"):
                    p = L.FEATURES_OUT / f"{dev}_{ver}_{k}"
                    if p.exists():
                        z.write(p, f"{dev}/features/{p.name}")
            else:
                F = _features()
                ns = F.load_legacy(C.feature_code(v, dev))
                z.writestr(f"{dev}/features/{dev}_std_regex.json", json.dumps(
                    {k: {n: list(x) for n, x in ns[f"regex_dict_{k}"].items()} for k in ("comb", "count", "ratio")}, ensure_ascii=False, indent=1))
            chk = C.read_json(out / f"check_{dev}.json", None)
            for f in (f"check_{dev}.json", f"check_{dev}_diff.csv"):
                if (out / f).exists():
                    z.write(out / f, f"{dev}/{f}")
            checks.append(_check_md(dev, chk))
            s = d.get("server") or {}
            m, a, im = s.get("metrics"), s.get("agree"), s.get("impl")
            srvs.append(f"- {dev.upper()}: 4지표 " + (" · ".join(f"{k} {_pct(m.get(k))}" for k in METRICS4) + f" (TP {m['tp']} · FP {m['fp']} · "
                        f"TN {m['tn']} · FN {m['fn']}) → 4지표 {'넘음' if not m95_fails(m) else '미달'} · 참고 f1 ≥95% "
                        f"{'넘음' if f1_ok(m) else '미달'}" if m else "없음") + f" · ② 예측 일치 {_pct((a or {}).get('value')) if a else '없음'}"
                        f" · ① 다른 칸 {(im or {}).get('diff_cells', '없음')}")
        pv = prev_server(v, vv)
        prev = ("\n".join([f"- 기준 버전 '{pv['version']}' ({pv.get('source', '')}): {pv.get('note', '')}"] +
                          [f"  - {d.upper()}: " + " · ".join(f"{k} {_pct((x or {}).get(k))}" for k in METRICS4) for d, x in pv["dev"].items()])
                if pv else "- 기록 없음")
        z.writestr("server.json", json.dumps(r, ensure_ascii=False, indent=1))
        z.writestr("README_설명.md", BUNDLE_README.format(
            name=name, rev=rev, at=C.now_iso(), tag=tag, btag=r.get("base_tag"), ver=ver,
            labels=LABEL_MODES.get(r.get("labels") or "keep"), nul=NUL_MODES.get(r.get("nul") or "drop"),
            checks="\n".join(checks), server="\n".join(srvs), prev=prev))
    for old in out.glob("서버반영_*.zip"):
        if old.name != zp.name:
            old.unlink()
    zp.with_name(zp.name + ".tmp").replace(zp)
    return zp.name


# ── 3. 로컬 사전 점검 ────────────────────────────────────────────────────

class _Prog:
    """진행률: 장비마다 같은 몫, 장비 안에서는 단계 몫 (읽기·원본 피처·디코딩 피처·그리드·학습·학습분 안·일치·구현·저장)."""

    def __init__(self, devs, grid, dec, raw_calc):
        w = {"read": 3, "raw": 12 if raw_calc else 0, "dec": 25 if dec else 0, "grid": 32 if grid else 0, "fit": 14, "insample": 3,
             "agree": 5 if dec else 0, "impl": 2 if dec else 0, "save": 1}
        tot = sum(w.values())
        self.start, acc = {}, 0.0
        for k, x in w.items():
            self.start[k] = acc / tot
            acc += x
        self.w = {k: x / tot for k, x in w.items()}
        self.devs, self.t0, self.last = devs, time.time(), 0.0

    def set(self, i, phase, frac, text, force=False):
        if RT.stop_requested(LOCK):
            raise Stopped()
        now = time.time()
        if not force and now - self.last < 1.0:
            return
        self.last = now
        pct = 100 * (i + self.start[phase] + self.w[phase] * min(1.0, max(0.0, frac))) / len(self.devs)
        el = now - self.t0
        RT.write_status(LOCK, dev=self.devs[i], step=phase, step_text=text, pct=round(pct, 1),
                        eta_s=round(el / pct * (100 - pct)) if pct >= 3 else None, elapsed_s=round(el))


_W = {}


def _w_init(code_path, defs_json, decode):
    """계산 프로세스 하나 준비: 판의 피처 정의 + (디코딩이면) 레시피 1 함수."""
    F = _features()
    ns = F.load_legacy(code_path)
    if defs_json:
        d = json.loads(defs_json)
        ns = dict(ns, regex_dict_comb=d["comb"], regex_dict_count=d["count"], regex_dict_ratio=d["ratio"])
    _W.update(F=F, ns=ns, names=F.legacy_names(ns), dec=local().decoder() if decode else None)


def _w_run(payloads):
    """payload 묶음 → (디코딩) → 피처 행렬 (열 = 피처 이름 순서), 디코딩 오류 수."""
    import pandas as pd
    F, ns, names, dec = _W["F"], _W["ns"], _W["names"], _W["dec"]
    if dec is not None:
        dec.errors = 0
        payloads = [dec(p) for p in payloads]
    X = F.generate_fast(pd.Series(list(payloads), dtype=object), ns)[names].to_numpy(dtype=float)
    return X, (dec.errors if dec is not None else 0)


def _cache_key(code, ns, decode):
    """피처 캐시 열쇠 = 판 정규식(장비 접두사 빼고) + 한글 비율 함수 + 디코딩 함수 + 5.0b 계산 판.
    IPS·WAF 의 정규식이 접두사 말고 같으면 열쇠가 같아, 두 장비에 같은 payload(정상 행 등)는 한 번만 계산한다."""
    F = _features()
    rx = {k: {n.split("_", 1)[1]: list(x) for n, x in ns[f"regex_dict_{k}"].items()} for k in ("comb", "count", "ratio")}
    ko = ns["ko_ratio"].__code__
    key = {"v": CACHE_VER, "rx": rx, "ko": hashlib.sha1(ko.co_code + repr(ko.co_consts).encode("utf-8")).hexdigest(),
           "dec": local().decoder_key() if decode else "", "gf": hashlib.sha1(F.generate_fast.__code__.co_code).hexdigest()}
    return hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _feature_matrix(P, ns, names, defs, code, decode, prog, i, phase, workers):
    """payload 배열 → 피처 행렬 (decode=True 면 레시피 1 디코딩 뒤). payload 지문 캐시(11_supervised/cache/server_feat/<열쇠>/)에서 없는 것만
    프로세스 여러 개로 계산해 묶음마다 저장 (멈춰도 계산한 것은 남음). → (행렬, 디코딩 오류 수, 새로 계산한 수)"""
    import numpy as np
    import pandas as pd
    F = _features()
    U = prog.devs[i].upper()
    what = "디코딩한 payload" if decode else "원본 payload"
    d = C.CACHE / "server_feat" / _cache_key(code, ns, decode)
    d.mkdir(parents=True, exist_ok=True)
    fcols = [f"f{j:02d}" for j in range(len(names))]
    if not (d / "info.json").exists():
        C.write_json(d / "info.json", {"decode": decode, "names_without_prefix": [n.split("_", 1)[1] for n in names], "at": C.now_iso()})
    pk = F.hash_series(pd.Series(P, dtype=object))

    def have():
        parts = sorted(d.glob("part-*.parquet"))
        if not parts:
            return pd.DataFrame(columns=fcols, index=pd.Index([], dtype="uint64", name="pk"))
        h = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        return h.drop_duplicates("pk", keep="last").set_index("pk")
    prog.set(i, phase, 0, f"{U} {what} 피처 캐시 읽기", True)
    H = have()
    miss = ~pd.Index(pk).isin(H.index)
    first = pd.Series(np.arange(len(pk)))[miss].groupby(pk[miss]).first().to_numpy() if miss.any() else np.zeros(0, int)
    todo, errs = list(first), 0
    if todo:
        chunks = [todo[k:k + CHUNK] for k in range(0, len(todo), CHUNK)]
        stamp = time.strftime("%Y%m%d%H%M%S")
        done = [0]
        t0 = time.time()

        def save(no, X):
            ix = chunks[no]
            df = pd.DataFrame(X, columns=fcols)
            df.insert(0, "pk", pk[ix])
            df.to_parquet(d / f"part-{stamp}-{no:05d}.parquet", index=False)
            done[0] += len(ix)
            rate = done[0] / max(1e-6, time.time() - t0)
            prog.set(i, phase, done[0] / len(todo), f"{U} {what} 피처 {done[0]:,}/{len(todo):,} (캐시에 없던 것, {rate:.0f}개/초)")
        left = list(range(len(chunks)))
        if workers > 1 and len(chunks) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = None
            try:
                pool = ProcessPoolExecutor(max_workers=min(workers, len(chunks)), initializer=_w_init,
                                           initargs=(str(code), json.dumps(defs) if defs else "", decode))
                futs = {pool.submit(_w_run, [P[j] for j in chunks[no]]): no for no in left}
                for fu in as_completed(futs):
                    X, e = fu.result()
                    errs += e
                    save(futs[fu], X)
                    left.remove(futs[fu])
            except Stopped:
                raise
            except Exception as e:                      # 프로세스를 못 쓰면(모듈 이름·풀 깨짐 등) 남은 묶음을 이 프로세스에서 - 같은 오류면 거기서 남
                print(f"계산 프로세스를 쓰지 못해 남은 {len(left)}묶음을 한 프로세스로 계산합니다: {type(e).__name__}: {e}", flush=True)
            finally:
                if pool:
                    pool.shutdown(wait=False, cancel_futures=True)
        if left:
            _w_init(str(code), json.dumps(defs) if defs else "", decode)
            for no in list(left):
                X, e = _w_run([P[j] for j in chunks[no]])
                errs += e
                save(no, X)
        H = have()
    X = H.reindex(pk)[fcols].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError(f"{U} {what} 피처 캐시에서 빠진 행이 있습니다 - 캐시 폴더 {d} 를 지우고 다시 하세요")
    return X, errs, len(todo)


def check(name, devs, grid=True, ver=None, workers=WORKERS, grid_rows=GRID_ROWS):
    held = RT.acquire_lock(LOCK, version=name, devs=list(devs))
    if held:
        raise SystemExit(f"서버 사전 점검이 이미 도는 중입니다 (pid {held.get('pid')})")
    RT.clear_stop(LOCK)
    RT.write_status(LOCK, version=name, rev=None, devs=list(devs), status="running", started_at=C.now_iso(), finished_at=None,
                    error=None, results={}, pct=0, eta_s=None, elapsed_s=0, dev=devs[0] if devs else "", step="",
                    step_text="시작", grid=grid, grid_rows=grid_rows, ver=ver)
    try:
        v, vv, rev, man = _ver(name)
        L = local()
        if ver is None:                                 # 판을 안 주면 올릴 파일을 만들 때 고른 판 (없으면 std)
            ver = ((load_rec(name).get("revs") or {}).get(str(rev)) or {}).get("ver") or STD
        ver = _check_ver(L, ver)
        RT.write_status(LOCK, rev=rev, ver=ver)
        prog = _Prog(list(devs), grid, L is not None, ver != STD)
        res = {}
        for i, dev in enumerate(devs):
            res[dev] = _check_dev(name, v, rev, man, dev, grid, prog, i, ver, workers, grid_rows)
            RT.write_status(LOCK, results=res)
        RT.write_status(LOCK, status="done", pct=100, eta_s=0, finished_at=C.now_iso(), step_text="끝남")
        return res
    except Stopped:
        RT.write_status(LOCK, status="stopped", finished_at=C.now_iso(), step_text="중지됨 (계산한 피처는 캐시에 남음)")
        print("중지됨")
    except BaseException as e:
        RT.write_status(LOCK, status="error", error=f"{type(e).__name__}: {e}"[:600], finished_at=C.now_iso())
        raise
    finally:
        RT.release_lock(LOCK)
        RT.clear_stop(LOCK)


def _causes(np, rows, Xr, Xd, P, names, meta=None):
    """일치하지 않는 행들의 원인: 값이 바뀐 피처, 원본 payload 의 인코딩, (추가 행) 룰·유형·label·출처."""
    feat, enc = Counter(), Counter()
    for k in rows:
        for j in np.where(np.abs(Xr[k] - Xd[k]) > 1e-9)[0]:
            feat[names[j]] += 1
        low = str(P[k]).lower()
        for e, rx in _ENC_RX.items():
            if rx.search(low):
                enc[e] += 1
    out = {"features": dict(feat.most_common(12)), "encodings": dict(enc.most_common(12))}
    if meta is not None:
        for key, col in (("rules", "rule_id"), ("types", "attack_type"), ("labels", "label"), ("origins", "origin")):
            out[key] = dict(Counter(meta[k][col] or "-" for k in rows).most_common(15))
    return out


def _grid(X, y, part, rows_cap, prog, i, U):
    """서버 Train 처럼 RF 트리 수 × 최대 깊이 그리드 (warm_start 로 깊이마다 트리를 100 → 200 → 300 으로 늘리며 평가).
    그리드 행(기본 표본 15만, 0 = 전체)을 층화 80/20 으로 나눠 검증 f1 로 고른다. → (고른 행, 표, 정보)"""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    idx = np.arange(n)
    if rows_cap and n > rows_cap:
        rng = np.random.default_rng(C.SEED)
        pos, neg = idx[y], idx[~y]
        kp = min(len(pos), int(round(rows_cap * len(pos) / n)))
        idx = np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, min(len(neg), rows_cap - kp), replace=False)]))
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(np.zeros(len(idx)), y[idx]))
    tr, va = idx[tr], idx[va]
    yv, pv = y[va], part[va]
    table, k, total = [], 0, len(GRID_TREES) * len(GRID_DEPTH)
    for d in GRID_DEPTH:
        m = RandomForestClassifier(n_estimators=GRID_TREES[0], max_depth=d, random_state=C.SEED, n_jobs=-1, warm_start=True)
        for t in GRID_TREES:
            prog.set(i, "grid", k / total, f"{U} 그리드 {k + 1}/{total}: 트리 {t} · 깊이 {d} ({len(tr):,}행 학습 → {len(va):,}행 검증)", True)
            m.set_params(n_estimators=t)
            m.fit(X[tr], y[tr])
            p = m.predict(X[va]).astype(bool)
            table.append({"trees": t, "depth": d, "all": _metrics(yv, p), "base": _metrics(yv[~pv], p[~pv]),
                          "added": _metrics(yv[pv], p[pv]) if pv.any() else None})
            k += 1
        del m
    best = max(table, key=lambda r: ((r["all"].get(GRID_METRIC) or 0), -r["trees"], -r["depth"]))   # 같으면 트리 적고 얕은 쪽
    info = {"rows": int(len(idx)), "sampled": bool(rows_cap and n > rows_cap), "train_rows": int(len(tr)), "valid_rows": int(len(va)),
            "seed": C.SEED, "metric": GRID_METRIC}
    return best, table, info


def _check_dev(name, v, rev, man, dev, grid, prog, i, fver, workers, grid_rows=GRID_ROWS):
    import numpy as np
    import pandas as pd
    t0 = time.time()
    F = _features()
    L = local()
    U = dev.upper()
    md = (man.get("dev") or {}).get(dev) or {}
    base_rows, added_rows = int(md.get("base_rows") or 0), int(md.get("added_rows") or 0)
    vdir = C.VERDIR / name
    code = C.feature_code(v, dev)
    ns = F.load_legacy(code)
    names = F.legacy_names(ns)
    defs = L.version_defs(dev, fver) if L and fver != STD else None
    if defs:
        ns = dict(ns, regex_dict_comb=defs["comb"], regex_dict_count=defs["count"], regex_dict_ratio=defs["ratio"])
        if F.legacy_names(ns) != names:
            raise ValueError(f"{U} 피처 정의 판 '{fver}' 의 피처 이름이 표준모델 피처 파일과 다릅니다")
    prog.set(i, "read", 0, f"{U} 학습 파일 읽기 ({base_rows + added_rows:,}행)", True)
    df = pd.read_csv(vdir / f"{dev}_train.csv", usecols=["method", "payload", "label"] + names,
                     dtype={"method": str, "payload": str, "label": str}, keep_default_na=False, low_memory=False)
    n = len(df)
    if n != base_rows + added_rows:
        raise ValueError(f"{U} 학습 파일 {n:,}행이 manifest 의 기준 {base_rows:,} + 추가 {added_rows:,} 와 다릅니다")
    ups = upload_rows(name, dev)
    tail = df.iloc[base_rows:]
    same = [(u["method"] == m and u["payload"] == p and u["label"] == lb)
            for u, m, p, lb in zip(ups, tail["method"], tail["payload"], tail["label"])]
    excl, labels_mode, prepared = prepared_excl(name, rev, dev)   # 올릴 파일에서 뺀 행은 서버 데이터에 없으므로 점검에서도 뺀다
    keep = np.ones(n, bool)
    for k, u in enumerate(ups):
        if u["sk"] in excl:
            keep[base_rows + k] = False
    idx = np.where(keep)[0]
    P = df["payload"].to_numpy(dtype=object)[idx]
    lab = df["label"].str.strip().str.lower().to_numpy()[idx]
    y = lab != "normal"
    part = idx >= base_rows                             # True = 추가 행
    res = {"version": name, "rev": rev, "dev": dev, "rows": n, "base_rows": base_rows, "added_rows": added_rows,
           "rows_used": int(len(idx)), "added_used": int(part.sum()), "excluded": int(n - len(idx)),
           "excluded_why": "올릴 파일에서 뺀 추가 행 (NUL 빼기·공격 행만)" if n - len(idx) else "",
           "prepared": prepared, "labels_mode": labels_mode, "feature_ver": fver,
           "bad_label": int((~pd.Series(lab).isin(["normal", "anomalies"])).sum()),
           "label": {"anomalies": int(y.sum()), "normal": int((~y).sum())},
           "upload_match": {"rows": len(ups), "same": int(sum(same)), "diff": int(len(same) - sum(same))},
           "model": "RF 100 트리 (표준모델 재현 설정 5_0a make_model('rf'), 서버 Train 설정과 다를 수 있음)"}
    # 원본 입력 피처 (SQL 쪽과 같은 값): std 는 학습 파일 값, 다른 판은 계산
    if defs is None:
        Xr = df[names].to_numpy(dtype=float)[idx]
        res["raw_source"] = "학습 파일 피처 (표준판, 서버 py·sql 피처와 같은 값)"
    else:
        Xr, _, nr = _feature_matrix(P, ns, names, defs, code, False, prog, i, "raw", workers)
        res["raw_source"] = f"판 '{fver}' 으로 계산 (새로 {nr:,})"
    del df
    if L:                                               # 학습 피처 = 디코딩 입력 (사용자 답 4)
        Xd, derr, nd = _feature_matrix(P, ns, names, defs, code, True, prog, i, "dec", workers)
        res.update(basis="decoded", decoder_errors=int(derr), dec_new=int(nd))
    else:
        Xd = Xr
        res.update(basis="raw", note=f"{LOCAL_FILE} 가 없어 디코딩하지 못해 원본 입력 피처로 학습했습니다 (② 입력 일치·① 구현 일치 없음)")
    trees, depth = 100, None                            # 그리드 없이: 표준모델 재현 설정 (5_0a make_model('rf'))
    if grid:
        best, table, gi = _grid(Xd, y, part, grid_rows, prog, i, U)
        trees, depth = best["trees"], best["depth"]
        res["grid"] = dict(gi, best={"trees": trees, "depth": depth},
                           table=[{"trees": r["trees"], "depth": r["depth"], **{k: r["all"][k] for k in METRICS4},
                                   "added_f1": (r["added"] or {}).get("f1")} for r in table])
        res["holdout"] = {"all": best["all"], "base": best["base"], "added": best["added"], "train_rows": gi["train_rows"],
                          "test_rows": gi["valid_rows"], "seed": C.SEED, "note": "그리드 검증 (고른 조합, 그리드 행의 20%)"}
        res["holdout"]["fails"] = m95_fails(res["holdout"]["all"])
    from sklearn.ensemble import RandomForestClassifier
    res["model"] = (f"RF 트리 {trees} · 최대 깊이 {depth or '제한 없음'} ("
                    + (f"그리드 {len(GRID_TREES)}×{len(GRID_DEPTH)} 에서 검증 {GRID_METRIC} 로 고름" if grid else "그리드 없이 표준모델 재현 설정")
                    + ", 학습 열 = 피처만 - 서버는 피처 + payload, method 안 씀)")
    prog.set(i, "fit", 0, f"{U} RF 학습 - 트리 {trees} · 깊이 {depth or '제한 없음'} · "
                          f"{'디코딩' if res['basis'] == 'decoded' else '원본'} 피처 전체 {len(idx):,}행 (서버 Train 처럼)", True)
    m = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=C.SEED, n_jobs=-1)
    m.fit(Xd, y)
    prog.set(i, "insample", 0, f"{U} 학습 데이터 자체 예측 (서버 레시피 6 방식)", True)
    p_in = m.predict(Xd).astype(bool)
    res["insample"] = {"all": _metrics(y, p_in), "base": _metrics(y[~part], p_in[~part]), "added": _metrics(y[part], p_in[part])}
    res["insample"]["fails"] = m95_fails(res["insample"]["all"])
    res["insample"]["f1_ok"] = f1_ok(res["insample"]["all"])
    diff_rows = []
    if L:
        prog.set(i, "agree", 0, f"{U} ② 같은 모델로 원본 입력 예측 (전체 {len(idx):,}행)", True)
        pr = m.predict(Xr).astype(bool)                 # 원본 입력 (SQL 쪽)
        ag = pr == p_in                                 # p_in = 디코딩 입력 (파이썬 쪽, 학습 피처)

        def part_sum(mask):
            k = int(mask.sum())
            a = int((ag & mask).sum())
            return {"rows": k, "agree": a, "rate": round(a / k, 6) if k else None, "disagree": k - a,
                    "raw_attack_dec_normal": int((mask & pr & ~p_in).sum()), "raw_normal_dec_attack": int((mask & ~pr & p_in).sum())}
        alls, bs, ads = part_sum(np.ones(len(idx), bool)), part_sum(~part), part_sum(part)
        meta = {}
        for pos in np.where(part)[0]:
            meta[pos] = ups[int(idx[pos]) - base_rows]
        dis_a = [int(k) for k in np.where(~ag & part)[0]]
        dis_b = [int(k) for k in np.where(~ag & ~part)[0]]
        res["agree"] = {"all": alls, "base": bs, "added": ads, "a98": alls["rate"] is not None and alls["rate"] >= TARGET_AGREE,
                        "a98_added": ads["rate"] is not None and ads["rate"] >= TARGET_AGREE,
                        "causes": {"added": _causes(np, dis_a, Xr, Xd, P, names, meta),
                                   "base": _causes(np, dis_b[:20000], Xr, Xd, P, names)}}
        for k in dis_a:
            u = meta[k]
            ch = np.where(np.abs(Xr[k] - Xd[k]) > 1e-9)[0]
            top = sorted(ch, key=lambda j: -abs(Xr[k, j] - Xd[k, j]))[:6]
            low = str(P[k]).lower()
            diff_rows.append({"sk": u.get("sk", ""), "rule_id": u.get("rule_id", ""), "attack_type": u.get("attack_type", ""),
                              "origin": u.get("origin", ""), "label": u.get("label", ""),
                              "원본_입력_예측": "anomalies" if pr[k] else "normal", "디코딩_입력_예측": "anomalies" if p_in[k] else "normal",
                              "바뀐_피처(원본→디코딩)": " ".join(f"{names[j]}:{Xr[k, j]:g}→{Xd[k, j]:g}" for j in top),
                              "인코딩": " ".join(e for e, rx in _ENC_RX.items() if rx.search(low)),
                              "payload_앞부분": str(P[k])[:300]})
        brk = L.sql_break_chars()
        prog.set(i, "impl", 0, f"{U} ① 구현 일치 (SQL 만 공백으로 바꾸는 글자 {len(brk)}개)", True)
        if brk:
            ir = np.where(pd.Series(P, dtype=object).str.contains("[" + re.escape(brk) + "]", regex=True).to_numpy())[0]
            if len(ir):
                tr_ = str.maketrans({ch: " " for ch in brk})
                Xs = F.generate_fast(pd.Series([P[k].translate(tr_) for k in ir], dtype=object), ns)[names].to_numpy(dtype=float)
                fdi = (np.abs(Xr[ir] - Xs) > 1e-9).any(axis=1)
                pdi = m.predict(Xr[ir]).astype(bool) != m.predict(Xs).astype(bool)
            else:
                fdi = pdi = np.zeros(0, bool)
            res["impl"] = {"chars": [f"U+{ord(ch):04X}" for ch in brk], "rows": int(len(ir)), "added_rows": int(part[ir].sum()) if len(ir) else 0,
                           "feat_diff_rows": int(fdi.sum()), "pred_diff_rows": int(pdi.sum()),
                           "rate": round(1 - int(fdi.sum()) / len(idx), 6) if len(idx) else None}
        else:
            res["impl"] = {"chars": [], "rows": 0, "feat_diff_rows": 0, "pred_diff_rows": 0, "rate": 1.0}
    else:
        res["agree"] = None
        res["impl"] = None
    prog.set(i, "save", 0, f"{U} 결과 저장", True)
    out = rev_dir(name, rev)
    out.mkdir(parents=True, exist_ok=True)
    if res.get("agree") is not None:
        fn = f"check_{dev}_diff.csv"
        cols = ["sk", "rule_id", "attack_type", "origin", "label", "원본_입력_예측", "디코딩_입력_예측", "바뀐_피처(원본→디코딩)", "인코딩",
                "payload_앞부분"]
        with open(out / fn, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(diff_rows)
        res["agree"]["diff_file"] = fn
        res["agree"]["diff_rows"] = len(diff_rows)
        res["agree"]["diff_sks"] = sorted({r["sk"] for r in diff_rows if r["sk"]})
    res["at"] = C.now_iso()
    res["seconds"] = round(time.time() - t0, 1)
    res["grid_on"] = bool(grid)
    C.write_json(out / f"check_{dev}.json", res)
    a = (res.get("agree") or {}).get("all") or {}
    print(f"{U}: 학습분 안 정확도 {_pct(res['insample']['all']['accuracy'])} · recall {_pct(res['insample']['all']['recall'])} · "
          f"② 일치 {_pct(a.get('rate'))} · {res['seconds']}초", flush=True)
    return str(out / f"check_{dev}.json")


def main():
    ap = argparse.ArgumentParser(description="서버 반영 지원 (화면 6 후보 rev)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check", help="로컬 사전 점검 (디코딩 피처로 학습: 95% 미리 보기, ② 입력 일치, ① 구현 일치)")
    a.add_argument("--version", required=True)
    a.add_argument("--rev", type=int, help="확인용 (최신 rev 와 다르면 멈춤)")
    a.add_argument("--dev", nargs="+", choices=list(C.DEVICES), default=list(C.DEVICES))
    a.add_argument("--ver", default=None, help="피처 정의 판 (기본: 올릴 파일을 만들 때 고른 판, 없으면 std)")
    a.add_argument("--no-grid", action="store_true", help="그리드 없이 RF 100 트리·깊이 제한 없음 (빠른 확인)")
    a.add_argument("--grid-rows", type=int, default=GRID_ROWS, help=f"그리드에 쓸 행 (층화 표본, 0 = 전체, 기본 {GRID_ROWS:,})")
    a.add_argument("--workers", type=int, default=WORKERS)
    b = sub.add_parser("prepare", help="올릴 파일·레시피 사본 만들기")
    b.add_argument("--version", required=True)
    b.add_argument("--tag", default="")
    b.add_argument("--nul", choices=list(NUL_MODES), default="drop")
    b.add_argument("--labels", choices=list(LABEL_MODES), default="keep")
    b.add_argument("--ver", default=STD)
    z = sub.add_parser("bundle", help="넘겨줄 묶음 zip")
    z.add_argument("--version", required=True)
    s = sub.add_parser("state", help="상태 JSON")
    s.add_argument("--version", required=True)
    g = sub.add_parser("gate", help="승인 조건")
    g.add_argument("--version", required=True)
    args = ap.parse_args()
    if args.cmd == "check":
        if args.rev:
            cur = int(C.get_version(C.load_versions(), args.version).get("rev") or 0)
            if cur != args.rev:
                raise SystemExit(f"최신 rev 는 {cur} 입니다 (요청 rev {args.rev})")
        check(args.version, list(dict.fromkeys(args.dev)), not args.no_grid, args.ver, max(1, args.workers), max(0, args.grid_rows))
    elif args.cmd == "prepare":
        print(json.dumps(prepare(args.version, args.tag, args.nul, args.labels, args.ver), ensure_ascii=False, indent=1))
    elif args.cmd == "bundle":
        print(bundle(args.version))
    elif args.cmd == "state":
        print(json.dumps(state(args.version), ensure_ascii=False, indent=1)[:20000])
    else:
        print(json.dumps(gate(args.version), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
