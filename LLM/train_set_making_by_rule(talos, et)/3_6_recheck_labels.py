"""[3.6 label 재확인] 화면 4 '전체 검증'의 label 정합성 결과 중 고른 줄을 LLM 으로 한 번 더 확인한다.  (화면 4 'LLM 재확인')

[대상]  끝난 검증(08_dataset/<날짜>/<장비>_verify.json 이 지금 로그 그대로)의 줄 (--target)
  - bad (기본): '불일치'(코드 4) 줄 전부
  - rule: '정합 (룰 기준)'(코드 2) 줄의 표본 - 룰마다 --per-rule 개 (기본 3, 기본은 LLM 원본만 - 증강 줄은 --with-augmented 일 때만).
    룰 안에서 내용 지문 순서로 고르므로 같은 조건이면 같은 줄, 룰당 수를 늘리면 앞에서 고른 줄 + 더 (이미 한 줄은 건너뜀)
  - 장비 --devs (기본 ips,waf). 줄 내용 지문(<장비>_frame.jsonl 의 k = payload·label·줄 정보)이 같으면 한 번만 보낸다
    (IPS·WAF 의 같은 샘플, 다시 만든 로그의 같은 줄)
  - 같은 모델(추론 강도 포함)로 이미 재확인한 내용은 건너뛴다 → 검증을 더 하면 새로 생긴 줄만 이어서
[호출]  같은 룰의 줄을 모아 요청 하나에 여러 줄 (--per-request, 기본 10). 지시문: docs/코드참조/LLM_재확인_프롬프트.md
  (대상별 절 '## [불일치]'·'## [정합 (룰 기준)]' 은 그 대상에만, 나머지 절은 모두에 보냄)
  입력: 원본 룰, label, payload(HTTP), 코드 검증의 판정 사유, 생성 메모
  → 판정 ok(label 맞음) · flip(label 반대) · unfit(학습에 부적합, 삭제 권장) · unknown(판단 불가, 거절 포함) + 짧은 근거
  provider 호출·재시도·동시 요청 자동 조절·응답 해석은 3_0a_common 그대로. 판정이 빠진 줄은 그 줄만 다시 묻고('재시도 횟수'까지),
  끝까지 판정이 없으면 판단 불가로 남긴다 (같은 모델로 다시 보내지 않음). 호출 자체가 실패한 요청은 남기지 않아 다음 실행 때 다시.
  모델: --model 'provider:model' (기본: LLM_설정.md 검토 모델), --effort (기본: '검토 추론 강도'). LLM_설정.md 는 바꾸지 않는다.
[결과]  07_llm_output/_recheck/ (요청이 끝날 때마다 덧붙임 → 중간에 멈춰도 이어서)
          results.jsonl   줄 내용 지문(k)·샘플 지문(ck·sk)마다 판정 하나 (모델·근거·시각·요청·대상)
          requests.jsonl  요청별 줄 수·토큰·시간 (화면의 예상 시간·토큰)
          raw/<요청>.tryN.txt  응답 원문
        코드 판정(cons)·학습 로그·검증 파일은 바꾸지 않는다. 화면 4 가 표·필터에 보여 주고,
        반영(삭제·label 바꾸기)은 화면에서 기존 수정·삭제(edits)로 저장한다 (되돌릴 수 있음).
[실행]  화면 4 의 'LLM 재확인' (별도 프로세스 - 화면을 다시 켜도 계속, 잠금 .run/recheck.lock, 중지 .run/recheck.stop) 또는
    python 3_6_recheck_labels.py --dry-run                        # 대상 줄·요청 수·예상 토큰만 (호출 없음)
    python 3_6_recheck_labels.py --model openai:gpt-6-luna --effort low
    python 3_6_recheck_labels.py --target rule --per-rule 3 --dry-run   # '정합 (룰 기준)' 룰당 3개 (LLM 원본만)
    python 3_6_recheck_labels.py --model mock --out <시험 폴더>    # 호출 없이 흐름 시험 (판정은 가짜)
  RULEDATA_RECHECK_DIR 환경변수로 결과 폴더를 바꿀 수 있다 (시험용 화면과 실행기가 같은 폴더를 보게).
"""

import argparse
import array
import hashlib
import importlib.util
import json
import math
import mmap
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("llm_common_recheck", BASE / "3_0a_common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
RT = C.RT

DATASET = BASE / "08_dataset"
OUT_DIR = Path(os.environ.get("RULEDATA_RECHECK_DIR") or C.LLM_OUTPUT_DIR / "_recheck")
PROMPT_FILE = BASE / "docs" / "코드참조" / "LLM_재확인_프롬프트.md"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VERDICTS = {"ok": "label 맞음", "flip": "label 반대", "unfit": "학습에 부적합", "unknown": "판단 불가"}
TARGETS = {"bad": "불일치", "rule": "정합 (룰 기준)"}  # --target → 화면 이름 (지시문의 '## [이름]' 절)
DEVS = ("ips", "waf")
PER_RULE = 3                    # rule: 룰당 표본 수 (기본)
PER_REQUEST = 10                # 요청 하나에 넣는 줄 수 (같은 룰끼리 모음)
REQUEST_CHARS = 60000           # 요청 하나의 payload 글자 합 상한 (넘으면 나눔)
HTTP_CHARS = 6000               # 줄 하나의 payload 최대 글자 (넘으면 앞뒤만 남기고 가운데를 줄임)
OUT_PER_ROW = 80                # 기록이 없을 때 예상: 줄 하나의 답(판정 + 근거 한 문장) 출력 토큰
THINK_PER_REQUEST = {"openai": 2000, "claude_cli": 1500, "anthropic": 1500}   # 기록이 없을 때 예상: 요청마다 추론 토큰
SEC_PER_REQUEST = 40            # 기록이 없을 때 예상: 요청 하나의 시간 (초)
# 참고 단가 (USD / 100만 토큰, 입력·출력). 2026-10-29 확인: Claude 는 Anthropic 가격표, gpt-6-luna 는 OpenAI 가격을 옮긴
# 집계 사이트, solar-pro2 는 집계 사이트마다 달라 높은 쪽. 바뀌면 고친다. claude_cli 는 토큰 요금 대신 구독 사용량.
PRICES = {"openai:gpt-6-luna": (0.10, 0.50), "openai:gpt-5": (1.25, 10.0), "upstage:solar-pro2": (0.15, 0.60),
          "anthropic:claude-opus-5-5": (4.0, 20.0), "anthropic:claude-sonnet-5-5": (2.0, 10.0),
          "anthropic:claude-haiku-5-5": (0.10, 0.50)}


# ── 대상 ('불일치' 줄) ─────────────────────────────────────────────────────

def split_log(line):
    """학습 로그 한 줄 → 필드 목록 (화면 4 split_log 와 같음: 앞 6필드는 앞에서, label 은 마지막 '|||' 뒤)."""
    head = line.split("|||", 6)
    if len(head) == 7 and "|||" in head[6]:
        return head[:6] + head[6].rsplit("|||", 1)
    return line.split("|||")


def _lines(path, want):
    """빈 줄을 뺀 줄 번호(화면 4 와 같은 번호) → 내용, want 의 줄만. 파일은 읽는 동안만 연다
    (화면의 검증이 끝날 때 frame 파일을 바꿔 쓰므로 오래 잡고 있지 않음)."""
    out, last = {}, max(want, default=-1)
    if last < 0:
        return out
    with open(path, "rb") as f:
        for i, line in enumerate(x for x in f if x.strip()):
            if i in want:
                out[i] = line.rstrip(b"\r\n").decode("utf-8", errors="replace")
            if i >= last:
                break
    return out


def _log_path(date, dev):
    found = sorted((DATASET / date).glob(f"{dev}_*_snort.log"))
    return found[0] if found else None


def verified(date, dev):
    """끝난 검증 (저장한 결과가 지금 로그 그대로) → (로그 경로, 줄별 코드) 또는 None."""
    p = _log_path(date, dev)
    if p is None:
        return None
    try:
        v = json.loads((DATASET / date / f"{dev}_verify.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    st = p.stat()
    if v.get("mtime") != st.st_mtime or v.get("size") != st.st_size or not v.get("codes"):
        return None
    return p, v["codes"]


def sample_key(label, rule_id, payload):
    """5_0a_common.sample_key 와 같은 샘플 내용 지문 (화면 6 비교 표의 ck)."""
    return hashlib.sha1(f"{label}\x00{rule_id}\x00{payload}".encode("utf-8", "replace")).hexdigest()[:20]


_cache, _cache_lock = {}, threading.Lock()     # (날짜, 장비) → (파일 상태, 줄 목록): 화면이 예상을 다시 셀 때 다시 읽지 않게


def _stamp(paths):
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def bad_rows(date, dev):
    """그 날짜·장비의 '불일치' 줄 [{k, ck, sk, rule_id, label, payload, note, attack_type, reason, in, date, dev, line}].
    파일(검증 결과·로그·줄별 결과·label)이 그대로면 지난 결과 (화면이 자주 물어도 다시 읽지 않음)."""
    path = _log_path(date, dev)
    if path is None:
        return []
    frame, lab = DATASET / date / f"{dev}_frame.jsonl", path.with_name(path.stem + "_label.jsonl")
    key = _stamp((DATASET / date / f"{dev}_verify.json", path, frame, lab))
    with _cache_lock:
        hit = _cache.get((date, dev))
    if hit and hit[0] == key:
        return hit[1]
    got = verified(date, dev)
    codes = got[1] if got else ""
    bad = {i for i, c in enumerate(codes) if c == "4"}
    rows = []
    if bad:
        fr = _lines(frame, bad) if frame.exists() else {}
        if len(fr) == len(bad):                     # 줄별 결과가 로그와 맞을 때만 (검증이 다시 쓰는 중이면 다음에)
            logs, metas = _lines(path, bad), (_lines(lab, bad) if lab.exists() else {})
            for i in sorted(bad):
                a = json.loads(fr[i])
                if a.get("cons") != "BAD" or not a.get("k"):
                    continue                        # 예전 형식 결과 (내용 지문 없음) → 다시 검증하면 대상
                f = split_log(logs.get(i, ""))
                payload, label = (f[6], f[7]) if len(f) == 8 else (logs.get(i, ""), "")
                m = json.loads(metas[i]) if i in metas else {}
                rid = m.get("id", "")
                rows.append({"k": a["k"], "ck": sample_key(label, rid, payload), "sk": m.get("sk", ""), "rule_id": rid,
                             "label": label, "payload": payload, "note": m.get("note", ""),
                             "attack_type": m.get("attack_type", ""), "reason": a.get("cons_reason", ""),
                             "in": a.get("in", ""), "date": date, "dev": dev, "line": i})
    with _cache_lock:
        _cache[(date, dev)] = (key, rows)
    return rows


def dataset_dates():
    return sorted(p.name for p in DATASET.iterdir() if p.is_dir() and DATE_RE.match(p.name)) if DATASET.is_dir() else []


def collect(dates=None, devs=DEVS):
    """'불일치' 대상 → {줄 내용 지문 k: 줄} (날짜·IPS 먼저 나온 것, 같은 내용이 또 있으면 at 에 위치만 더함)."""
    rows = {}
    for d in dates or dataset_dates():
        for dev in devs:
            for r in bad_rows(d, dev):
                if r["k"] in rows:
                    rows[r["k"]]["at"].append([d, dev, r["line"]])
                else:
                    rows[r["k"]] = dict(r, at=[[d, dev, r["line"]]])
    return rows


# ── '정합 (룰 기준)' 표본 ──────────────────────────────────────────────────────
# 줄이 많아(2026-10-30 기준 약 14만 줄) 먼저 가벼운 목록(내용 지문·룰·LLM 원본 여부)만 읽어 표본을 고르고, 고른 줄만 payload 를 읽는다.

_ROW_K = re.compile(r'^\{"k": "([0-9a-f]+)"')           # 줄별 결과(frame) 한 줄의 내용 지문 (k 가 맨 앞)
_LINE_START = re.compile(rb"^(?=[ \t\r\f\v]*[^ \t\r\f\v\n])", re.M)     # 빈 줄이 아닌 줄의 시작 (_lines 의 x.strip() 과 같은 기준)
_pool, _offs = {}, {}                                   # (날짜, 장비) → (파일 상태, 목록), 파일 → (파일 상태, 줄 위치)


def _offsets(path):
    """빈 줄을 뺀 줄 번호(_lines 와 같은 번호) → 그 줄의 시작 바이트 위치. 파일이 그대로면 지난 결과.
    흩어진 줄을 파일 전체를 줄마다 돌지 않고 바로 읽으려고 (줄 시작은 정규식이 한 번에 찾음, 파일은 찾는 동안만 엶)."""
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _offs.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    offs = array.array("q")
    if st.st_size:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            offs.extend(m.start() for m in _LINE_START.finditer(mm))
    with _cache_lock:
        _offs[str(path)] = (key, offs)
    return offs


def _lines_at(path, want):
    """_lines 와 같은 결과 (빈 줄을 뺀 줄 번호 → 내용, want 의 줄만)를 줄 위치로 바로 읽음."""
    offs, out = _offsets(path), {}
    with open(path, "rb") as f:
        for i in sorted(want):
            if 0 <= i < len(offs):
                f.seek(offs[i])
                out[i] = f.readline().rstrip(b"\r\n").decode("utf-8", errors="replace")
    return out


def rule_pool(date, dev):
    """그 날짜·장비의 '정합 (룰 기준)'(코드 2) 줄 [(k, 룰 id, LLM 원본 여부, 줄 번호, sk, label, 공격 유형)] (payload 없이).
    끝난 검증이 아니면 []. 파일이 그대로면 지난 결과."""
    path = _log_path(date, dev)
    if path is None:
        return []
    frame, lab = DATASET / date / f"{dev}_frame.jsonl", path.with_name(path.stem + "_label.jsonl")
    key = _stamp((DATASET / date / f"{dev}_verify.json", path, frame, lab))
    with _cache_lock:
        hit = _pool.get((date, dev))
    if hit and hit[0] == key:
        return hit[1]
    got = verified(date, dev)
    want = {i for i, c in enumerate(got[1]) if c == "2"} if got else set()
    out = []
    if want:
        fr = _lines_at(frame, want) if frame.exists() else {}
        if len(fr) == len(want):                    # 줄별 결과가 로그와 맞을 때만 (bad_rows 와 같음)
            metas = _lines_at(lab, want) if lab.exists() else {}
            for i in sorted(want):
                m = _ROW_K.match(fr[i])
                if not m:
                    continue                        # 예전 형식 결과 (내용 지문 없음)
                a = json.loads(metas[i]) if i in metas else {}
                out.append((m.group(1), a.get("id", ""), a.get("origin") == "llm", i, a.get("sk", ""), a.get("label", ""),
                            a.get("attack_type", "")))
    with _cache_lock:
        _pool[(date, dev)] = (key, out)
    return out


def unverified(dates=None, devs=DEVS):
    """로그는 있는데 끝난 검증이 없는 (검증 전·중지·그 뒤 로그가 바뀜) [날짜, 장비] - 대상에서 빠짐."""
    return [[d, dev] for d in dates or dataset_dates() for dev in devs if _log_path(d, dev) and verified(d, dev) is None]


def rule_rows(dates=None, devs=DEVS):
    """고른 날짜·장비의 '정합 (룰 기준)' 줄을 내용(k)으로 합침 → {k: 줄(가벼운)}. 같은 내용은 날짜·장비 순서로 처음 것, 다른 위치는 at 에."""
    rows = {}
    for d in dates or dataset_dates():
        for dev in devs:
            for k, rid, llm, line, sk, label, typ in rule_pool(d, dev):
                x = rows.get(k)
                if x is None:
                    rows[k] = {"k": k, "rule_id": rid, "llm": llm, "date": d, "dev": dev, "line": line, "sk": sk, "label": label,
                               "attack_type": typ, "at": [[d, dev, line]]}
                else:
                    x["at"].append([d, dev, line])
    return rows


def rule_sample(rows, per_rule=PER_RULE, llm_only=True):
    """룰마다 내용 지문 순서로 per_rule 개 (llm_only 면 LLM 원본 줄만) → ({k: 줄}, 표본을 고른 룰 수).
    순서가 고정이라 같은 조건이면 같은 줄, 룰당 수를 늘리면 앞의 줄 + 더."""
    by_rule = {}
    for x in rows.values():
        if x["llm"] or not llm_only:
            by_rule.setdefault(x["rule_id"], []).append(x)
    pick = {}
    for xs in by_rule.values():
        for x in sorted(xs, key=lambda x: x["k"])[:per_rule]:
            pick[x["k"]] = x
    return pick, len(by_rule)


def rule_details(rows):
    """표본 줄(가벼운) → 보낼 줄 (bad_rows 와 같은 모양: payload·label·생성 메모·코드 판정 사유·룰 입력). 파일마다 한 번씩 읽는다.
    그사이 검증을 다시 해 내용 지문이 달라진 줄은 뺀다 (다음 실행 때)."""
    by_file, out = {}, []
    for x in rows:
        by_file.setdefault((x["date"], x["dev"]), []).append(x)
    for (d, dev), xs in sorted(by_file.items()):
        path = _log_path(d, dev)
        if path is None:
            continue
        frame, lab = DATASET / d / f"{dev}_frame.jsonl", path.with_name(path.stem + "_label.jsonl")
        want = {x["line"] for x in xs}
        fr, logs = _lines_at(frame, want), _lines_at(path, want)
        metas = _lines_at(lab, want) if lab.exists() else {}
        for x in xs:
            i = x["line"]
            if i not in fr:
                continue
            a = json.loads(fr[i])
            if a.get("k") != x["k"]:
                continue
            f = split_log(logs.get(i, ""))
            payload, label = (f[6], f[7]) if len(f) == 8 else (logs.get(i, ""), "")
            m = json.loads(metas[i]) if i in metas else {}
            out.append(dict(x, ck=sample_key(label, x["rule_id"], payload), sk=m.get("sk", x["sk"]), label=label,
                            payload=payload, note=m.get("note", ""), attack_type=m.get("attack_type", x["attack_type"]),
                            reason=a.get("cons_reason", ""), **{"in": a.get("in", "")}))
    return out


def rule_counts(latest, dates=None, devs=DEVS):
    """'정합 (룰 기준)' 줄(내용 기준)과 그중 재확인한 줄의 판정 수(가장 최근, 어느 모델이든) → {rows, checked, counts, rules}
    (화면 4 가 자주 물으므로 줄 목록을 만들지 않고 셈)."""
    seen, cnt, rules = set(), {v: 0 for v in VERDICTS}, set()
    for d in dates or dataset_dates():
        for dev in devs:
            for k, rid, *_ in rule_pool(d, dev):
                if k in seen:
                    continue
                seen.add(k)
                x = latest.get(k)
                if x and x.get("verdict") in cnt:
                    cnt[x["verdict"]] += 1
                    rules.add(rid)
    return {"rows": len(seen), "checked": sum(cnt.values()), "counts": cnt, "rules": len(rules)}


def rule_summary(latest, dates=None, devs=DEVS):
    """'정합 (룰 기준)' 룰별 요약: 그 룰의 줄(내용 기준) 중 재확인한 줄을 가장 최근 판정(어느 모델이든)으로 셈
    → [{rule_id, attack_type, rows, n, ok, flip, unfit, unknown}] (재확인한 줄이 있는 룰만)."""
    rows = rule_rows(dates, devs)
    out = {}
    for x in rows.values():
        s = out.get(x["rule_id"])
        if s is None:
            s = out[x["rule_id"]] = {"rule_id": x["rule_id"], "attack_type": x["attack_type"], "rows": 0, "n": 0,
                                     **{v: 0 for v in VERDICTS}}
        s["rows"] += 1
        r = latest.get(x["k"])
        if r and r.get("verdict") in VERDICTS:
            s["n"] += 1
            s[r["verdict"]] += 1
    return [s for s in out.values() if s["n"]]


# ── 지난 결과 ─────────────────────────────────────────────────────────────

def read_results(out_dir=None):
    """results.jsonl → 줄 목록 (오래된 순)."""
    p = (out_dir or OUT_DIR) / "results.jsonl"
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue                        # 쓰는 중에 끊긴 마지막 줄
    except OSError:
        pass
    return out


def latest_by_key(results):
    """줄 내용 지문 → 가장 최근 판정 (어느 모델이든)."""
    out = {}
    for r in results:
        out[r.get("k")] = r
    return out


def history(out_dir=None):
    """모델별 지난 요청 {model: {n, sec, rows, tin, tout}} (requests.jsonl, 성공한 요청만) → 예상 시간·토큰."""
    out = {}
    try:
        with open((out_dir or OUT_DIR) / "requests.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("error"):
                    continue
                h = out.setdefault(r.get("model", ""), {"n": 0, "sec": 0.0, "rows": 0, "tin": 0, "tout": 0})
                u = r.get("usage") or {}
                h["n"] += 1
                h["sec"] += r.get("seconds") or 0
                h["rows"] += r.get("rows") or 0
                h["tin"] += u.get("input_tokens") or 0
                h["tout"] += u.get("output_tokens") or 0
    except OSError:
        pass
    return out


# ── 요청 ─────────────────────────────────────────────────────────────────

def model_id(cfg):
    """결과·건너뛰기에 쓰는 모델 이름: provider:model (+ @추론 강도, 실제로 넘길 때만)."""
    e = cfg.get("effort") or cfg.get("reasoning_effort")
    return f"{cfg['provider']}:{cfg['model']}" + (f"@{e}" if e else "")


def http_text(payload):
    """로그 payload(줄바꿈 = 두 글자 \\n) → 실제 줄바꿈의 HTTP 원문. 너무 길면 앞뒤만 남긴다."""
    h = C.norm_payload(payload)
    if len(h) > HTTP_CHARS:
        h = h[:4000] + f"\n…(가운데 {len(h) - 5500:,}자 생략)…\n" + h[-1500:]
    return h


_RULE_ID = re.compile(r'\{"id": "([^"\\]*)"')


def rules_for(rows):
    """대상 줄의 룰 {(룰 입력 날짜, 룰 id): 룰} - 검증할 때 쓴 룰 입력(줄의 in), 없으면 가장 최근 입력."""
    want = {}
    for r in rows:
        want.setdefault(r["in"] or "", set()).add(r["rule_id"])
    latest, out = C.latest_input_dir(), {}
    for name, ids in want.items():
        d = C.LLM_INPUT_DIR / name if name and (C.LLM_INPUT_DIR / name / "rules_all.jsonl").exists() else latest
        if d is None:
            continue
        with open(d / "rules_all.jsonl", encoding="utf-8") as f:
            for line in f:
                m = _RULE_ID.match(line)                # 룰 줄은 id 가 맨 앞 (1.3) → 필요한 룰만 해석
                if m and m.group(1) not in ids or not line.strip():
                    continue
                r = json.loads(line)
                if r.get("id") in ids:
                    out[(name, r["id"])] = r
    return out


def batches(todo, per=PER_REQUEST):
    """같은 룰의 줄끼리 per 줄씩 (payload 글자 합이 REQUEST_CHARS 를 넘으면 나눔)."""
    reqs, cur, size = [], [], 0
    for r in sorted(todo, key=lambda r: (r["rule_id"], r["date"], r["dev"], r["line"])):
        n = len(http_text(r["payload"]))
        if cur and (len(cur) >= per or size + n > REQUEST_CHARS):
            reqs.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        reqs.append(cur)
    return reqs


HEADS = {"bad": "코드 검증에서 label 정합성이 '불일치' 인 샘플",
         "rule": "코드 검증에서 룰 조건은 모두 맞지만 범용 공격 흔적이 없어 label 정합성이 '정합 (룰 기준)' 인 샘플"}


def prompt_text(target="bad"):
    """지시문 파일 → 이 대상에 보낼 글. '## [대상 이름]' 절은 그 대상일 때만 (제목의 [대상 이름] 은 떼고), 나머지 절은 모두."""
    out, keep = [], True
    for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("## "):
            m = re.match(r"## \[([^\]]+)\]\s*", line)
            keep = not m or m.group(1) == TARGETS[target]
            if m and keep:
                line = "## " + line[m.end():]
        if keep:
            out.append(line)
    return "".join(out)


def user_message(rows, rules, target="bad"):
    lines, seen = [], set()
    for r in rows:
        if (r["in"], r["rule_id"]) not in seen:
            seen.add((r["in"], r["rule_id"]))
            rule = rules.get((r["in"], r["rule_id"])) or {}
            lines.append(json.dumps({"rule_id": r["rule_id"], "attack_type": rule.get("attack_type") or r["attack_type"],
                                     "rule_name": rule.get("rule_name", ""), "rule": rule.get("rule") or "(룰을 찾지 못함)"},
                                    ensure_ascii=False))
    for n, r in enumerate(rows, 1):
        lines.append(json.dumps({"n": n, "rule_id": r["rule_id"], "label": r["label"], "http": http_text(r["payload"]),
                                 "code_check": r["reason"], "note": r["note"]}, ensure_ascii=False))
    return (f"다음은 {HEADS[target]} {len(rows)}개와 그 룰이다 (JSON Lines, 룰 줄이 먼저).\n"
            f"샘플마다 {{\"n\": 번호, \"verdict\": \"ok|flip|unfit|unknown\", \"reason\": \"근거 한 문장\"}} 를 한 줄씩, "
            f"n 1~{len(rows)} 모두 답한다.\n\n" + "\n".join(lines))


def parse_answer(text, n):
    """응답 → {번호: (판정, 근거)} (번호·판정이 맞는 것만, 같은 번호는 처음 것)."""
    got = {}
    for o in C.parse_json_lines(text):
        try:
            i = int(o.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        v = str(o.get("verdict", "")).strip().lower()
        if 1 <= i <= n and v in VERDICTS and i not in got:
            got[i] = (v, " ".join(str(o.get("reason") or "").split())[:300])
    return got


def mock_answer(rows):
    """--model mock: 호출 없이 흐름 시험. 판정은 내용 지문으로 정한 가짜."""
    names = list(VERDICTS)
    return "\n".join(json.dumps({"n": n, "verdict": names[int(r["k"][:4], 16) % 4], "reason": "mock (가짜 판정)"},
                                ensure_ascii=False) for n, r in enumerate(rows, 1))


def prepare(dates, mid, per=PER_REQUEST, out_dir=None, target="bad", devs=DEVS, per_rule=PER_RULE, llm_only=True):
    """대상·이번에 보낼 줄·요청 묶음·룰. rows = 대상 (rule 이면 표본), stats = rule 의 표본 정보
    (내용 기준 줄·LLM 원본 줄·룰 수·표본을 고른 룰 수)."""
    done = {(r.get("k"), r.get("model")) for r in read_results(out_dir)}
    stats = None
    if target == "rule":
        pool = rule_rows(dates, devs)
        rows, n_rules = rule_sample(pool, per_rule, llm_only)
        todo = rule_details([r for r in rows.values() if (r["k"], mid) not in done])
        stats = {"pool": len(pool), "llm": sum(x["llm"] for x in pool.values()), "rules": len({x["rule_id"] for x in pool.values()}),
                 "rules_sampled": n_rules, "per_rule": per_rule, "llm_only": llm_only}
    else:
        rows = collect(dates, devs)
        todo = [r for r in rows.values() if (r["k"], mid) not in done]
    rules = rules_for(todo)
    return {"rows": rows, "todo": todo, "reqs": batches(todo, per), "rules": rules,
            "done": sum((k, mid) in done for k in rows), "stats": stats}


def model_config(model=None, effort=None):
    """→ (cfg, providers, run, 형식). model 이 비면 LLM_설정.md 검토 모델, effort 가 None 이면 '검토 추론 강도'."""
    roles, providers, run = C.load_settings()
    cfg = C.role_config(roles, "검토", model or None)
    if cfg is None:
        raise ValueError("모델이 없습니다 (provider:model, none 은 쓸 수 없음)")
    p = providers.get(cfg["provider"])
    if p is None:
        raise ValueError(f"LLM_설정.md 의 provider 표에 '{cfg['provider']}' 가 없습니다")
    C.set_effort(cfg, run.get("검토 추론 강도", "") if effort is None else effort)
    return cfg, providers, run, p["style"]


def cost(mid, tin, tout):
    """참고 단가로 예상 비용 (USD). 단가가 없으면 None."""
    pr = PRICES.get(mid.split("@")[0])
    return None if pr is None else round(tin / 1e6 * pr[0] + tout / 1e6 * pr[1], 4)


def estimate(model=None, effort=None, dates=None, per=PER_REQUEST, target="bad", devs=DEVS, per_rule=PER_RULE, llm_only=True):
    """화면 4 의 '대상·예상': 대상 줄·이미 한 줄·요청 수·예상 토큰·시간·비용 (호출 없음).
    토큰: 그 모델의 지난 재확인 기록이 있으면 줄당 평균, 없으면 글자 ÷ 3 (3_1 과 같은 어림) + 줄당 답·요청당 추론 토큰."""
    cfg, _, run, style = model_config(model, effort)
    mid = model_id(cfg)
    p = prepare(dates, mid, per, target=target, devs=devs, per_rule=per_rule, llm_only=llm_only)
    reqs, todo = p["reqs"], p["todo"]
    prompt_chars = len(prompt_text(target))
    chars = sum(prompt_chars + len(user_message(b, p["rules"], target)) for b in reqs)
    hist = history()

    def tokens(m):                                  # 그 모델의 지난 기록 (줄당 평균), 없으면 어림
        h = hist.get(m)
        if h and h["rows"]:
            return round(h["tin"] / h["rows"] * len(todo)), round(h["tout"] / h["rows"] * len(todo))
        return chars // 3, len(todo) * OUT_PER_ROW + len(reqs) * THINK_PER_REQUEST.get(m.split(":")[0], 0)
    tin, tout = tokens(mid)
    h = hist.get(mid)
    workers = max(1, min(C.parallel_settings(run)[1], len(reqs) or 1))
    sec_req = h["sec"] / h["n"] if h and h["n"] else SEC_PER_REQUEST
    others = [{"model": m, "cost": cost(m, *tokens(m))} for m in PRICES]     # 같은 대상을 다른 모델로 (참고 단가)
    return {"model": mid, "style": style, "rows": len(p["rows"]), "done": p["done"], "todo": len(todo), "requests": len(reqs),
            "tokens_in": tin, "tokens_out": tout, "history": bool(h), "seconds": round(math.ceil(len(reqs) / workers) * sec_req),
            "workers": workers, "cost": cost(mid, tin, tout), "subscription": style == "claude_cli", "others": others,
            "by_date": _count_by_date(p["rows"].values(), todo), "target": target, "stats": p["stats"],
            "unverified": unverified(dates, devs)}


def _count_by_date(rows, todo):
    t = {r["k"] for r in todo}
    out = {}
    for r in rows:
        x = out.setdefault(r["date"], {"rows": 0, "todo": 0})
        x["rows"] += 1
        x["todo"] += r["k"] in t
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────

def ask(rows, rules, cfg, providers, run, prompt, style, req, out_dir, log, target="bad"):
    """요청 하나 → ({번호: (판정, 근거)}, 사용량, 시도 수). 판정이 빠진 줄은 그 줄만 다시 묻는다 ('재시도 횟수'까지).
    다시 묻기 전에 중지 요청이 있으면 멈추고, 그때 빠진 줄은 남기지 않는다 (다음 실행 때 다시)."""
    tries = C.run_int(run, "재시도 횟수", 3)
    usage, got, todo = {}, {}, list(range(1, len(rows) + 1))
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        if attempt > 1 and RT.stop_requested("recheck"):
            return got, usage.get("recheck", {}), attempt - 1
        sub = [rows[i - 1] for i in todo]
        if style == "mock":                         # 3_0a_common 의 mock 처럼 동시 요청 자리를 잡고 호출 시간만 흉내
            with cfg["throttle"]:
                time.sleep(float(os.environ.get("RULEDATA_MOCK_DELAY", "0") or 0))
            text, u = mock_answer(sub), {"input_tokens": 0, "output_tokens": 0, "finish_reason": "stop"}
        else:
            text, u = C.call_llm(cfg, providers, prompt, user_message(sub, rules, target), run, log)
        C.add_usage(usage, "recheck", u)
        (out_dir / "raw" / f"{req}.try{attempt}.txt").write_text(text, encoding="utf-8")
        for j, v in parse_answer(text, len(sub)).items():
            got[todo[j - 1]] = v
        todo = [i for i in todo if i not in got]
        if not todo:
            break
        if attempt < tries:
            log.warning("  %s: 판정이 빠진 줄 %d개 - 그 줄만 다시 묻습니다 (%d번째)", req, len(todo), attempt)
    for i in todo:                                  # 끝까지 판정이 없음 = 판단 불가 (거절 포함), 같은 모델로 다시 보내지 않음
        got[i] = ("unknown", f"응답에서 판정을 찾지 못함 (거절 또는 형식 오류, raw/{req} 확인)")
    return got, usage.get("recheck", {}), attempt


def _status(log, **info):
    """진행 상태 (.run/recheck.json, 화면의 진행률). 막 쓴 파일을 백신 검사 등이 잠깐 잡으면 Windows 가 바꾸기를 막으므로
    (요청이 연달아 끝날 때 자주) 잠깐 뒤 다시 쓰고, 끝내 못 쓰면 넘어간다 (판정 결과는 이미 저장됨)."""
    for k in range(8):
        try:
            return RT.write_status("recheck", **info)
        except OSError as e:
            if k == 7:
                log.warning("  진행 상태를 저장하지 못함 (판정 결과는 저장됨): %s", e)
                return None
            time.sleep(0.1 * (k + 1))


def run(args, log):
    C.load_env()
    try:
        cfg, providers, run_cfg, style = model_config(args.model, args.effort)
    except (ValueError, FileNotFoundError) as e:
        log.error("[3.6] %s", e)
        return 1
    mid, out_dir = model_id(cfg), Path(args.out) if args.out else OUT_DIR
    dates = None if args.dates in (None, "", "all") else [d for d in args.dates.split(",") if DATE_RE.match(d)]
    devs, target, llm_only = tuple(d for d in DEVS if d in args.devs.split(",")) or DEVS, args.target, not args.with_augmented
    st = {"req_done": 0, "req_fail": 0, "rows_done": 0, "verdicts": {k: 0 for k in VERDICTS}, "tokens_in": 0, "tokens_out": 0}
    if not args.dry_run:        # 잠금을 먼저 잡고 상태를 새로 쓴다 (대상을 세는 동안 화면이 지난 실행의 상태를 보지 않게)
        holder = RT.acquire_lock("recheck", by="3_6", model=mid, target=target)
        if holder:
            log.error("[3.6] LLM 재확인이 이미 실행 중입니다 (PID %s, %s 시작)", holder.get("pid"), holder.get("started_at"))
            return 2
        RT.clear_stop("recheck")
        _status(log, state="running", pid=os.getpid(), model=mid, style=style, dates=dates or "all", started_at=RT.now_iso(),
                finished_at=None, error="", rows_total=None, req_total=None, eta_s=None, target=target, devs=list(devs),
                per_rule=args.per_rule if target == "rule" else None, llm_only=llm_only if target == "rule" else None, **st)
    try:
        p = prepare(dates, mid, args.per_request, out_dir, target, devs, args.per_rule, llm_only)
    except BaseException as e:
        if not args.dry_run:
            _status(log, state="fail", error=f"{type(e).__name__}: {e}", finished_at=RT.now_iso())
            RT.release_lock("recheck")
        raise
    reqs = p["reqs"][:args.max_requests] if args.max_requests > 0 else p["reqs"]
    n_rows = sum(len(b) for b in reqs)
    s = p["stats"]
    what = (f"'{TARGETS[target]}' 줄 {len(p['rows'])}개" if s is None else
            f"'{TARGETS[target]}' 줄 {s['pool']}개 (룰 {s['rules']}개) 중 {'LLM 원본만 ' if llm_only else ''}"
            f"룰당 {args.per_rule}개 표본 {len(p['rows'])}개")
    log.info("[3.6] label 재확인 %s%s | %s (내용 기준, %s%s) 중 이 모델로 이미 %d개 → 이번 %d줄 · 요청 %d개%s",
             mid, " (호출 없이 시험)" if style == "mock" else "", what, ", ".join(dates) if dates else "모든 날짜",
             "" if devs == DEVS else f", {'·'.join(devs)} 만", p["done"], n_rows, len(reqs),
             " (dry-run: 호출 안 함)" if args.dry_run else "")
    if args.dry_run:
        return 0
    if not reqs:
        _status(log, state="done", finished_at=RT.now_iso(), rows_total=0, req_total=0)
        RT.release_lock("recheck")
        return 0
    RT.update_lock("recheck", rows=n_rows, requests=len(reqs))
    prompt = prompt_text(target)
    workers = max(1, min(C.parallel_settings(run_cfg)[1], len(reqs)))
    cfg["throttle"] = C.Throttle(workers, log, "재확인", start=min(4, workers))
    stamp, t0 = time.strftime("%Y%m%d-%H%M%S"), time.time()
    lock, fatal, stopped = threading.Lock(), threading.Event(), threading.Event()
    _status(log, rows_total=n_rows, req_total=len(reqs))
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(item):
        no, rows = item
        th = cfg["throttle"]
        while True:     # 동시 요청 자리가 날 때까지 기다리며 중지·중단을 본다 (요청이 적으면 모두 한꺼번에 시작하므로)
            if fatal.is_set():
                return
            if RT.stop_requested("recheck"):
                stopped.set()
                return
            with th.cv:
                if th.active < th.limit:
                    break
            time.sleep(0.2)
        req, t1 = f"{stamp}_{no:04d}", time.time()
        try:
            got, u, tries = ask(rows, p["rules"], cfg, providers, run_cfg, prompt, style, req, out_dir, log, target)
            err = ""
        except C.LLMError as e:
            got, u, tries, err = {}, {}, 0, str(e)
            log.error("  %s 실패: %s", req, e)
            if C.is_fatal(e) and not fatal.is_set():
                fatal.set()
                log.error("설정·키·모델 이름·크레딧·사용 한도 문제라 나머지도 중단합니다 (다음 실행 때 이어서)")
        except Exception as e:                      # 예상 못 한 오류도 다른 요청은 계속
            got, u, tries, err = {}, {}, 0, f"{type(e).__name__}: {e}"
            log.exception("  %s 실패 (예상 못 한 오류)", req)
        now = RT.now_iso()
        recs = [{"k": r["k"], "ck": r["ck"], "sk": r["sk"], "model": mid, "verdict": got[n][0], "reason": got[n][1],
                 "at": now, "req": req, "rule_id": r["rule_id"], "label": r["label"], "date": r["date"], "dev": r["dev"],
                 "line": r["line"], "target": target} for n, r in enumerate(rows, 1) if n in got]
        with lock:
            with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in recs)
            with open(out_dir / "requests.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"req": req, "model": mid, "target": target, "rows": len(rows), "answered": len(recs), "tries": tries,
                                    "seconds": round(time.time() - t1, 1), "usage": u, "at": now, "error": err[:300]},
                                   ensure_ascii=False) + "\n")
            st["req_fail" if err else "req_done"] += 1
            st["rows_done"] += len(recs)
            for x in recs:
                st["verdicts"][x["verdict"]] += 1
            st["tokens_in"] += u.get("input_tokens", 0)
            st["tokens_out"] += u.get("output_tokens", 0)
            fin = st["req_done"] + st["req_fail"]
            _status(log, eta_s=round((time.time() - t0) / fin * (len(reqs) - fin)), **st)
        if recs:
            c = {k: sum(x["verdict"] == k for x in recs) for k in VERDICTS}
            log.info("  %s: 줄 %d개 판정 (%s), 토큰 %s", req, len(recs),
                     " · ".join(f"{VERDICTS[k]} {n}" for k, n in c.items() if n), u)

    state, error = "done", ""
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rc") as ex:
            list(ex.map(one, enumerate(reqs, 1)))   # 순서대로 시작, 동시에 workers 개
        if RT.stop_requested("recheck"):            # 요청 도중에 중지 (다시 묻기 전에 멈춤)
            stopped.set()
        if fatal.is_set():
            state, error = "fail", "설정·키·모델·크레딧·사용 한도 문제로 중단 (로그 확인)"
        elif stopped.is_set():
            state = "stopped"
            log.info("[3.6] 중지 요청 - 여기서 멈춤 (남은 줄은 다음 실행 때 이어서)")
    except BaseException as e:
        state, error = "fail", f"{type(e).__name__}: {e}"
        raise
    finally:
        _status(log, state=state, error=error, finished_at=RT.now_iso(), eta_s=None, **st)
        RT.clear_stop("recheck")
        RT.release_lock("recheck")
    v = st["verdicts"]
    log.info("[3.6] 끝 (%s): 요청 %d개 성공 · %d개 실패, 줄 %d개 판정 - %s, 토큰 입력 %d · 출력 %d → %s", state, st["req_done"],
             st["req_fail"], st["rows_done"], " · ".join(f"{VERDICTS[k]} {v[k]}" for k in VERDICTS), st["tokens_in"],
             st["tokens_out"], out_dir)
    return 0 if state != "fail" and not st["req_fail"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="화면 4 '불일치'·'정합 (룰 기준)' 줄의 label 을 LLM 으로 다시 확인 (3.6)")
    ap.add_argument("--target", choices=list(TARGETS), default="bad",
                    help="bad = '불일치' 줄 전부 (기본), rule = '정합 (룰 기준)' 줄을 룰당 표본")
    ap.add_argument("--devs", default=",".join(DEVS), help="장비 (쉼표, 기본 ips,waf - 같은 내용은 한 번)")
    ap.add_argument("--per-rule", type=int, default=PER_RULE, help=f"rule: 룰당 표본 수 (기본 {PER_RULE})")
    ap.add_argument("--with-augmented", action="store_true", help="rule: 증강 줄도 표본에 넣음 (기본: LLM 원본만)")
    ap.add_argument("--model", help="'provider:model' (기본: LLM_설정.md 검토 모델, mock = 호출 없이 시험)")
    ap.add_argument("--effort", help="추론 강도 (minimal·low·medium·high·xhigh·max, '-' = 넘기지 않음. 기본: '검토 추론 강도')")
    ap.add_argument("--dates", help="날짜 (쉼표로 여러 개, 기본·all = 모든 날짜)")
    ap.add_argument("--per-request", type=int, default=PER_REQUEST, help=f"요청 하나의 줄 수 (기본 {PER_REQUEST})")
    ap.add_argument("--max-requests", type=int, default=0, help="이번에 보낼 요청 수 (기본 0 = 전부)")
    ap.add_argument("--out", help="결과 폴더 (기본 07_llm_output/_recheck, 시험용)")
    ap.add_argument("--dry-run", action="store_true", help="호출하지 않고 대상·요청 수만")
    args = ap.parse_args(argv)
    args.per_request = max(1, min(50, args.per_request))
    args.per_rule = max(1, min(1000, args.per_rule))
    if not set(args.devs.split(",")) <= set(DEVS):
        ap.error(f"--devs: {', '.join(DEVS)} 중에서")
    return run(args, C.setup_logging("3_6_recheck_labels"))


if __name__ == "__main__":
    sys.exit(main())
