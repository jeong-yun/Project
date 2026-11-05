"""[5.7 선별·버전] 비교 결과(5.5)에서 학습에 넣을 신규 샘플을 고르고(선별), 버전(26 하반기 등)으로 담아 확정·내보낸다.  (화면 6)

화면(0_0_run_ui.py)도 선별 미리 보기에 이 모듈을 쓰므로 맨 위에서는 표준 라이브러리만 불러온다 (확정은 실행기가 pandas 로).
→ 화면의 미리 보기와 확정이 같은 select() 를 써서, 사람이 본 수와 내보내는 샘플이 같다.

IPS·WAF 공통 선별 (2026-10-26~): 두 장비에 **같은 신규 샘플**을 넣는다 (각 장비의 부족한 부분·신규 부분을 합침).
  - 두 장비의 비교 결과(5.5 <장비>_table.json)를 샘플 지문 sk 로 짝짓는다 (combine). 한쪽 표에만 있는 샘플은 뺀다 (수는 화면에).
  - 합친 등급: 제거 = 신규 내부 중복이거나 두 장비 모두 기준과 중복 / A = 어느 장비든 A / B = A 가 아니고 어느 장비든 B / C.
    한쪽 장비만 기준과 중복이면(그 장비에서 제거) 다른 장비의 등급을 따른다. 장비별 등급은 gi·gw 로 남긴다.
  - 패턴당 상한(B·C): 피처 패턴이 장비마다 달라, 장비마다 그 장비 패턴으로 (합친 등급·label·패턴) 상한만큼 고른 뒤 합친다
    → 두 장비 모두 패턴마다 상한까지는 들어가고, 한 장비에서 상한을 넘는 것은 다른 장비가 필요로 한 샘플뿐이다.
  - 조건(날짜·출처)·보류 옵션·'불일치' 빼기·룰·유형 조정은 합친 목록에 한 번 적용한다.
선별 기본값 (화면에서 바꿀 수 있고, 확정 전에 사람이 룰·공격 유형 단위로 최종 판단)
  제거  항상 뺌 (중복)
  A     전부 (기준 모델이 틀림 = 기준 데이터에 부족한 데이터)
  B     패턴당 20개까지 (새 패턴) - 같은 패턴의 반복(예: SQLi 변형)으로 쏠리지 않게
  C     패턴당 20개까지 (이미 학습된 패턴, 분포 유지용)
  - 상한은 등급·label·패턴마다 적용하고, 넘치면 룰을 돌아가며 고른다 (같은 패턴 안에서도 여러 룰, 룰 안에서는 LLM 샘플 먼저).
  - '불일치'(4. 화면 검증) 빼기: 기본 켬.  보류 옵션(담당관 회신 전, 기본 끔): '정합 (룰 기준)' 공격 빼기, python-requests UA 정상 빼기.
  - 룰·공격 유형 조정: 빼기(그 샘플 모두 뺌) / 넣기(상한 없이 제거 외 모두 넣음, 룰만).
  - '불일치' 중 LLM 재확인 합의가 확인됨인 것은 남기기 (2026-10-29~, 기본 끔): 4. 화면 'LLM 재확인'(3.6) 결과를 5.5 가 비교 표의
    lk 로 남기고(샘플 내용 지문 ck), 이 옵션을 켜면 '불일치' 빼기에서 그 샘플은 빼지 않는다. 담은 뒤 lk 가 바뀌면 확정하지 않음.
    2026-11-02~: lk = '불일치' 합의가 확인됨(두 모델 이상 모두 맞음 또는 화면의 유지 표시·사람 판단) - 한 모델만 맞음이면 남기지 않음.
  - 목표 합계 (2026-10-29~, 비우면 안 씀): B 의 패턴당 상한 대신, 다른 선택(A·C·룰 '넣기')을 먼저 넣고 모자란 수를 B 에서
    정확히 채운다. B 패턴당 상한을 1, 2, 3 … 으로 올리며 목표를 넘지 않는 가장 큰 k 까지 넣고, 남은 수는 k+1 번째 샘플을
    패턴을 돌아가며 1개씩 (_fill). C 로는 채우지 않는다.
버전 (2026-10-30~: 초안 → 1차 확정(후보, rev) → 피처 테스트 → 데이터 추가 → 최종 승인(확정))
  초안   이름, 기준 버전, 조건(날짜·출처), 선별 옵션, 룰·유형 조정 + 저장할 때의 비교 스냅숏(장비별), mode = combined
         (예전 장비별 초안은 '새 방식으로 다시 담기' 를 해야 내보낸다 - 옵션·조정은 그대로)
  1차 확정 (후보, status candidate, rev 1·2·…)  export
         두 장비의 비교 스냅숏이 지금 비교 결과와 같고 그 뒤 학습데이터가 바뀌지 않았을 때만 내보낸다 (사람이 확인한 그대로).
         고른 샘플 하나의 목록(+ '데이터 추가')을 IPS·WAF 각각의 로그 줄·피처 코드로 내보낸다 (두 장비의 추가 샘플이 같음)
         11_supervised/versions/<이름>/        ← 늘 최신 rev 의 파일 (지난 rev 는 revs/rev<n>/ 에 작은 파일만)
           <장비>_train.csv       기준 학습 파일 + 추가 샘플 (method, payload, label, 표준모델 피처 = 표준모델 학습 형식)
                                  = 기준 base_rows 행 + <장비>_added.csv 순서의 추가 행 (manifest dev.<장비>.base_rows)
           <장비>_raw.log         기준 원본 로그 + 추가 샘플 로그 줄 (|||, 8필드)   ← 로그 형식 파일은 여기서만 만든다
           <장비>_added.csv       추가한 샘플 (날짜·줄·sk·룰·합친 등급 grade·이 장비 등급 grade_dev·근거·origin·added_by·payload)
                                  origin = new (신규 학습데이터) / manual (외부 파일) - 5.4 eval-version 이 by_origin 으로 나눔,
                                  gen_origin = llm·augmented, added_by = 선별·넣기:룰·넣기:공격 유형·넣기:샘플·manual:<파일>
           <장비>_considered.txt  이번 버전에서 비교한 샘플 지문 (확정 뒤 다음 비교에서 뺌)
           manifest.json          기준·조건·옵션·mode·rev·데이터 추가·수·파일 크기·sha256 + samples + approval (최종 승인 기록)
           manual/                외부 파일 원본 (데이터 추가), eval/rev<n>/<장비>_<실험>.json 피처 테스트 결과 (5.4 eval-version)
         후보는 아직 다음 버전의 기준이 될 수 없다. 승인 전에만 '데이터 추가'(룰·공격 유형·샘플 강제로 넣기, 외부 파일) →
         다시 내보내면 rev+1. 버리기(discard) = 폴더를 _보관/ 로 옮기고 이름을 다시 쓸 수 있게.
  최종 승인 (확정, status final)  approve
         최신 rev 의 피처 테스트(장비별, 기본 실험 E0 = 표준 피처만)가 목표(precision·recall·F1 최소값, '기준 버전보다 낮지
         않음')를 넘을 때. 못 넘으면 사유를 적어야 승인되고 manifest·versions.json 에 남는다. 승인한 버전은 잠기고
         다음 버전(예: 27 상반기)의 기준이 된다.
"""

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("sup_common_v", "5_0a_common.py")
DEFAULT_OPTIONS = {"A": "all", "B": 20, "C": 20,          # "all" = 전부, 0 = 빼기, n = 패턴당 n개까지
                   "exclude_bad": True, "exclude_rule_based": False, "exclude_python_ua_normal": False,
                   "keep_llm_ok": False,                  # '불일치' 빼기에서 LLM 재확인(4. 화면, 3.6) 합의가 확인됨인 것은 남김
                   "target": None}                        # 목표 합계 (None = 안 씀, 있으면 B 는 이 합계까지 채움)
OPTION_TEXT = {"exclude_bad": "'불일치' 빼기", "exclude_rule_based": "'정합 (룰 기준)' 공격 빼기 (보류)",
               "exclude_python_ua_normal": "python-requests UA 정상 빼기 (보류)",
               "keep_llm_ok": "'불일치' 중 LLM 재확인 합의가 확인됨인 것은 남기기"}
HOLD = {"rule": "'정합 (룰 기준)' 공격", "pyua": "python-requests UA 정상"}   # 보류 옵션의 대상 (화면에서 따로 봄)
PY_UA = 1                                                  # 5.5 FLAG_BITS
STEPS = ("초안", "1차 확정", "피처 테스트", "데이터 추가", "최종 승인")
DEFAULT_EXP = "E0"                                         # 피처 테스트 기본 실험 = 표준 피처만 (최종 승인 판단에 씀)
METRICS = ("precision", "recall", "f1")
ADD_KINDS = {"rules": "룰", "types": "공격 유형", "samples": "샘플"}   # 데이터 추가: 강제로 넣기 단위
SK_RE = re.compile(r"^[0-9a-f]{16}$")
MANUAL_MAX = 50000                                         # 외부 파일 하나의 최대 줄 수
ARCHIVE = BASE / "_보관"                                   # 버린 후보를 옮기는 곳


# ── 선별 ────────────────────────────────────────────────────────────────

def norm_options(o):
    out = dict(DEFAULT_OPTIONS)
    for k in ("A", "B", "C"):
        val = (o or {}).get(k, out[k])
        if val != "all":
            try:
                val = max(0, min(100000, int(val)))
            except (TypeError, ValueError):
                raise ValueError(f"선별 옵션 {k}: 'all' 또는 0 이상의 수")
        out[k] = val
    for k in ("exclude_bad", "exclude_rule_based", "exclude_python_ua_normal", "keep_llm_ok"):
        out[k] = bool((o or {}).get(k, out[k]))
    tg = (o or {}).get("target")
    if tg not in (None, "", 0, "0"):
        try:
            out["target"] = max(1, min(10_000_000, int(tg)))
        except (TypeError, ValueError):
            raise ValueError("선별 옵션 목표 합계: 비우거나 1 이상의 수")
    return out


def norm_conditions(c):
    c = c or {}
    dates = [d for d in (c.get("dates") or []) if isinstance(d, str) and C.DATE_RE.match(d)]
    sources = [s for s in (c.get("sources") or []) if isinstance(s, str) and s.isidentifier()]
    return {"dates": sorted(set(dates)), "sources": sorted(set(sources))}


def norm_overrides(o):
    o = o or {}
    rules = {str(k): v for k, v in (o.get("rules") or {}).items() if v in ("exclude", "include")}
    types = {str(k): v for k, v in (o.get("types") or {}).items() if v == "exclude"}
    return {"rules": rules, "types": types}


def combine(ti, tw):
    """IPS·WAF 비교 표 → 두 장비에 같은 샘플을 넣는 공통 선별용 표 (select·preview 가 그대로 쓰는 모양).
    샘플 지문 sk 로 짝짓고 줄 순서는 IPS 표(날짜·줄 순서). 한쪽 표에만 있거나 label 이 다른 샘플은 뺀다 (one_side).
    합친 등급: 두 장비 모두 제거면 제거 (신규 내부 중복은 payload 가 같아 두 장비가 같이 제거, 아니면 두 장비 모두 기준과 중복)
    / 한쪽만 제거(그 장비 기준과 중복)면 다른 장비 등급 / 아니면 어느 장비든 A → A, 어느 장비든 B → B, 그 밖 C.
    pats = [IPS 패턴, WAF 패턴] (상한을 장비마다 그 장비 패턴으로 셈), gi·gw = 장비별 등급, iidx·widx = 장비 표의 줄 번호."""
    if ti.get("base") != tw.get("base"):
        raise ValueError("IPS·WAF 비교 결과의 기준 버전이 다릅니다 - 둘 다 같은 기준으로 비교하세요")
    if "sk" not in ti or "sk" not in tw:
        raise ValueError("비교 결과에 샘플 지문(sk)이 없습니다 - '새로 생긴 데이터만 비교'를 한 번 다시 하세요")
    pos = {}
    for j, s in enumerate(tw["sk"]):
        pos.setdefault(s or f"{tw['dates'][tw['date'][j]]}:{tw['line'][j]}", j)
    keys = ("date", "line", "y", "rule", "type", "src", "llm")
    out = {k: [] for k in keys + ("g", "gi", "gw", "pat", "pat_w", "fl", "iidx", "widx", "wline", "sk")}
    cons, n_ips_only, n_mismatch, used = [], 0, 0, 0
    GI, GW, FI, FW, KI, KW = ti["g"], tw["g"], ti["fl"], tw["fl"], ti["cons"], tw["cons"]
    LI, LW = ti.get("lk") or "", tw.get("lk") or ""      # LLM 재확인 'label 맞음' (5.5, 예전 표에는 없음)
    lk = [] if LI or LW else None
    for i, s in enumerate(ti["sk"]):
        j = pos.get(s or f"{ti['dates'][ti['date'][i]]}:{ti['line'][i]}")
        if j is None:
            n_ips_only += 1
            continue
        if ti["y"][i] != tw["y"][j]:
            n_mismatch += 1
            continue
        used += 1
        gi, gw = GI[i], GW[j]
        if gi == 0 and gw == 0:
            g = 0
        else:
            gs = [x for x in (gi, gw) if x]                  # 제거인 장비는 빼고
            g = 1 if 1 in gs else 2 if 2 in gs else 3
        for k in keys:
            out[k].append(ti[k][i])
        out["g"].append(g)
        out["gi"].append(gi)
        out["gw"].append(gw)
        out["pat"].append(ti["pat"][i])
        out["pat_w"].append(tw["pat"][j])
        out["fl"].append(FI[i] | FW[j])
        out["iidx"].append(i)
        out["widx"].append(j)
        out["wline"].append(tw["line"][j])
        out["sk"].append(s)
        ci, cw = KI[i], KW[j]
        cons.append("4" if "4" in (ci, cw) else "2" if "2" in (ci, cw) else ci if ci != "0" else cw)
        if lk is not None:
            lk.append("1" if "1" in (LI[i:i + 1], LW[j:j + 1]) else "0")
    out.update({"base": ti["base"], "dev": "ips+waf", "combined": True, "n": used, "cons": "".join(cons),
                "lk": "".join(lk) if lk is not None else "",
                "dates": ti["dates"], "rules": ti["rules"], "types": ti["types"], "sources": ti["sources"],
                "snapshot": f"{ti.get('snapshot')}+{tw.get('snapshot')}", "snapshots": {"ips": ti.get("snapshot"),
                                                                                         "waf": tw.get("snapshot")},
                "one_side": {"ips_only": n_ips_only, "waf_only": tw["n"] - used - n_mismatch, "label_mismatch": n_mismatch}})
    out["pats"] = [out["pat"], out["pat_w"]]
    return out


def _round_robin(idx, cap, R, L):
    """같은 (등급·label·패턴) 묶음에서 cap 개: 룰을 돌아가며 (룰마다 LLM 먼저, 날짜·줄 순) 0번째, 1번째 … 순서로."""
    by_rule = {}
    for i in idx:
        lst = by_rule.get(R[i])
        if lst is None:
            by_rule[R[i]] = [i]
        else:
            lst.append(i)
    lists = [[i for i in x if L[i]] + [i for i in x if not L[i]] for x in by_rule.values()]
    picked, k = [], 0
    while len(picked) < cap:
        for lst in lists:
            if k < len(lst):
                picked.append(lst[k])
                if len(picked) == cap:
                    break
        k += 1
    return picked


def _rr_order(idx, R, L):
    """_round_robin(idx, len(idx)) 과 같은 순서를 정렬로 ((룰 안의 순서, 룰 순서) 순) → 앞의 cap 개 = _round_robin(idx, cap).
    룰 하나에 샘플이 몰린 묶음에서도 빠르다 (_round_robin 은 cap 이 작을 때를 위한 것)."""
    r0 = R[idx[0]]
    if len(idx) == 1 or all(R[i] == r0 for i in idx):   # 룰이 하나뿐인 묶음: LLM 먼저, 날짜·줄 순
        return [i for i in idx if L[i]] + [i for i in idx if not L[i]]
    by_rule = {}
    for i in idx:
        lst = by_rule.get(R[i])
        if lst is None:
            by_rule[R[i]] = [i]
        else:
            lst.append(i)
    keyed = []
    for r, x in enumerate(by_rule.values()):
        keyed.extend((p, r, i) for p, i in enumerate([i for i in x if L[i]] + [i for i in x if not L[i]]))
    keyed.sort()
    return [i for _, _, i in keyed]


_FILL_CACHE = {}    # (비교 스냅숏, B 후보) → 순위: 목표 합계만 바꿔 다시 셀 때 (화면) 다시 늘어놓지 않음
_FILL_LOCK = threading.Lock()                           # 화면의 미리 보기가 동시에 와도 한 번만 셈


def _fill_rank(t, idx, R, L):
    """B 후보 idx 의 순위 → (줄 번호 → 두 장비 중 앞선 순위, 장비별 묶음 순서 목록, 순위별 수)."""
    ck = (t.get("snapshot"), t["n"], len(idx), hash(tuple(idx)))
    with _FILL_LOCK:
        hit = _FILL_CACHE.get(ck)
        if hit is None:
            hit = _fill_rank_calc(t, idx, R, L)
            _FILL_CACHE.clear()                         # 하나만 둔다 (메모리)
            _FILL_CACHE[ck] = hit
    return hit


def _fill_rank_calc(t, idx, R, L):
    G, Y = t["g"], t["y"]
    rank, per_dev = [t["n"]] * t["n"], []
    for P in (t.get("pats") or [t["pat"]]):
        groups = {}
        for i in idx:
            key = (P[i] << 3) | (G[i] << 1) | Y[i]
            lst = groups.get(key)
            if lst is None:
                groups[key] = [i]
            else:
                lst.append(i)
        orders = [_rr_order(x, R, L) for x in groups.values()]
        for order in orders:
            for p, i in enumerate(order):
                if p < rank[i]:
                    rank[i] = p
        per_dev.append(orders)
    return rank, per_dev, Counter(rank[i] for i in idx)


def _fill(t, idx, need, R, L):
    """목표 합계: B 후보 idx 에서 정확히 need 개 (0 < need < len(idx)) → (고른 줄, k, 하나 더 넣은 수).
    장비마다 그 장비 패턴의 (등급·label·패턴) 묶음을 룰을 돌아가며 늘어놓고(_rr_order), 샘플의 순위 = 두 장비 중 앞선 순위.
    순위 k 미만 전부 = 패턴당 상한 k 와 같은 선택이라 need 를 넘지 않는 가장 큰 k 까지 넣고,
    남은 수는 순위 k 인 샘플(묶음마다 하나)을 패턴을 돌아가며 (장비를 번갈아, 앞에 나온 패턴부터) 1개씩 더한다."""
    rank, per_dev, cnt = _fill_rank(t, idx, R, L)
    k, total = 0, 0
    while total + cnt[k] <= need:                      # need < len(idx) 이라 어느 k 에서 넘는다
        total += cnt[k]
        k += 1
    picked = [i for i in idx if rank[i] < k]
    more = set()
    for orders in zip_longest(*per_dev):               # IPS 첫 패턴, WAF 첫 패턴, IPS 둘째 패턴 …
        for order in orders:
            if len(more) == need - total:
                break
            if order is not None and k < len(order) and rank[order[k]] == k:
                more.add(order[k])
        if len(more) == need - total:
            break
    picked += sorted(more)
    return picked, k, len(more)


def select(t, conditions=None, options=None, overrides=None, fill=None):
    """선별용 표(5.5 <장비>_table.json 또는 combine 한 공통 표) → (고른 줄 번호 목록, 조건 안의 줄 번호 목록, 뺀 이유별 수).
    상한(B·C): 표에 장비별 패턴(pats)이 있으면 장비마다 그 장비 패턴으로 묶어 상한만큼 고른 뒤 합친다.
    목표 합계(options target): B 는 상한 대신 다른 선택을 다 넣은 뒤 모자란 수만큼 채운다 (_fill).
    fill: dict 를 주면 목표 합계 결과 {target, base(B 없이 고른 수), max(B 전부), added, k, extra, state} 를 적는다."""
    cond, opts, ov = norm_conditions(conditions), norm_options(options), norm_overrides(overrides)
    every = not cond["dates"] and not cond["sources"]
    date_ok = [not cond["dates"] or d in cond["dates"] for d in t["dates"]]
    src_ok = [not cond["sources"] or s in cond["sources"] for s in t["sources"]]
    rule_ov = [{"exclude": 1, "include": 2}.get(ov["rules"].get(r), 0) for r in t["rules"]]
    type_ex = [ov["types"].get(x) == "exclude" for x in t["types"]]
    caps = [None, opts["A"], "fill" if opts["target"] else opts["B"], opts["C"]]
    ex_bad, ex_rule, ex_py = opts["exclude_bad"], opts["exclude_rule_based"], opts["exclude_python_ua_normal"]
    LK = (t.get("lk") or "") if opts["keep_llm_ok"] else ""      # '불일치' 중 LLM 재확인이 label 맞다고 한 줄은 남김
    n_dup = n_ex = n_bad = n_rule = n_py = n_grade = 0
    sel, cand, capped, fillb = [], [], [], []
    R, L, G, Y = t["rule"], t["llm"], t["g"], t["y"]
    # 표의 줄은 날짜·줄 순서 (5.5) → 줄 번호 순서가 곧 날짜·줄 순서
    for i, d, s, g, y, r, ty, k, fl in zip(range(t["n"]), t["date"], t["src"], G, Y, R, t["type"], t["cons"], t["fl"]):
        if not every and not (date_ok[d] and src_ok[s]):
            continue
        cand.append(i)
        if g == 0:
            n_dup += 1
            continue
        rc = rule_ov[r]
        if rc == 1 or type_ex[ty]:
            n_ex += 1
        elif ex_bad and k == "4" and not (LK and LK[i] == "1"):
            n_bad += 1
        elif ex_rule and k == "2" and y:
            n_rule += 1
        elif ex_py and not y and fl & PY_UA:
            n_py += 1
        elif rc == 2 or caps[g] == "all":
            sel.append(i)
        elif caps[g] == 0:
            n_grade += 1
        elif caps[g] == "fill":
            fillb.append(i)
        else:
            capped.append(i)
    chosen = set()
    for P in (t.get("pats") or [t["pat"]]):              # 장비별 패턴마다 (한 장비 표면 그 표의 패턴 하나)
        groups = {}
        for i in capped:
            key = (P[i] << 3) | (G[i] << 1) | Y[i]       # 등급·label·패턴
            lst = groups.get(key)
            if lst is None:
                groups[key] = [i]
            else:
                lst.append(i)
        for key, idx in groups.items():
            cap = caps[(key >> 1) & 3]
            chosen.update(idx if len(idx) <= cap else _round_robin(idx, cap, R, L))
    sel.extend(chosen)
    n_cap = len(capped) - len(chosen)
    if opts["target"]:                                  # 목표 합계: 지금까지 고른 것(A·C·룰 '넣기') 다음 B 로 모자란 수만큼
        need, base = opts["target"] - len(sel), len(sel)
        add, k, extra = ([], 0, 0) if need <= 0 else (fillb, None, 0) if need >= len(fillb) else _fill(t, fillb, need, R, L)
        sel.extend(add)
        if fill is not None:
            fill.update(target=opts["target"], base=base, max=base + len(fillb), added=len(add), k=k, extra=extra,
                        state="over" if need < 0 else "under" if need > len(fillb) else "exact")
    sel.sort()
    why = {"제거 (중복)": n_dup, "'불일치'": n_bad, "'정합 (룰 기준)' 공격 (보류)": n_rule,
           "python-requests UA 정상 (보류)": n_py, "룰·유형 빼기": n_ex, "등급 빼기": n_grade, "패턴당 상한 초과": n_cap}
    if opts["target"]:
        why["목표 합계 밖 (B)"] = len(fillb) - len(add)
    return sel, cand, why


def hold_rows(t, cand, kind):
    """조건 안의 후보 중 보류 옵션의 대상 줄 번호: rule = '정합 (룰 기준)' 공격, pyua = python-requests UA 정상."""
    Y, K, FL = t["y"], t["cons"], t["fl"]
    if kind == "rule":
        return [i for i in cand if Y[i] and K[i] == "2"]
    return [i for i in cand if not Y[i] and FL[i] & PY_UA]


def preview(t, conditions=None, options=None, overrides=None, view=""):
    """화면 미리 보기: 등급·label 별 후보/선택 수, 공격 유형·출처·날짜별, 뺀 이유, 룰별 표, 보류 대상 수.
    view = rule | pyua 면 표들을 그 보류 대상만으로 센다 (선별 자체는 그대로 - 지금 선별에 몇 개가 들어갔는지 보임)."""
    fill = {}
    sel, cand, why = select(t, conditions, options, overrides, fill)
    ov = norm_overrides(overrides)
    on = bytearray(t["n"])
    for i in sel:
        on[i] = 1
    sg = Counter(t["g"][i] for i in sel)                # 등급별 선택 수 (보기와 상관없이 지금 선별 전체)
    hold, rows_of, G, Y, K = {}, {}, t["g"], t["y"], t["cons"]
    LK = t.get("lk") or ""                              # '불일치' 후보 중 LLM 재확인이 label 맞다고 한 수 (옵션 옆에)
    llm_ok = sum(1 for i in cand if K[i] == "4" and LK[i] == "1") if LK else 0
    for kind in HOLD:                                   # [후보, 지금 선택(옵션을 켜면 빠질 수), 제거, A, B, C]
        idx = rows_of[kind] = hold_rows(t, cand, kind)
        h = hold[kind] = [len(idx), sum(on[i] for i in idx), 0, 0, 0, 0]
        for i in idx:
            h[2 + G[i]] += 1
    unverified = sum(1 for i in cand if Y[i] and K[i] == "0")    # 검증 전이라 정합성을 모르는 공격
    overlap = None
    if t.get("combined"):                                # 합친 A·B 가 어느 장비에서 온 것인지 [둘 다, IPS 만, WAF 만] (후보·선택)
        GI, GW = t["gi"], t["gw"]
        overlap = {x: {"cand": [0, 0, 0], "sel": [0, 0, 0]} for x in ("A", "B")}
        for i in cand:
            g = G[i]
            if g in (1, 2):
                a, b = GI[i] == g, GW[i] == g
                o = overlap["A" if g == 1 else "B"]
                k = 0 if a and b else 1 if a else 2
                o["cand"][k] += 1
                o["sel"][k] += on[i]
    if view in HOLD:
        cand = rows_of[view]
    nt, ns, nd, nr = len(t["types"]), len(t["sources"]), len(t["dates"]), len(t["rules"])
    gc = [0] * 16                                       # (등급·label) 별 후보·선택
    tc, sc, dc = [[0] * 4 for _ in range(nt)], [[0] * 4 for _ in range(ns)], [[0] * 4 for _ in range(nd)]
    nid = t["types"].index("normal") if "normal" in t["types"] else -1
    rc, rtype = [[0] * 6 for _ in range(nr)], [nid] * nr      # 룰의 공격 유형 (정상 샘플만 있으면 normal)
    cols = (t["g"], t["y"], t["type"], t["src"], t["date"], t["rule"])
    rows = zip(*cols, on) if len(cand) == t["n"] else zip(*([c[i] for i in cand] for c in cols), [on[i] for i in cand])
    for (g, y, ty, s, d, r, o), n in Counter(rows).items():     # 같은 값 조합끼리 먼저 셈 (C 로 빠름)
        k = (g * 2 + y) * 2
        gc[k] += n
        gc[k + 1] += o * n
        for acc in (tc[ty], sc[s], dc[d]):
            acc[0] += n
            acc[1] += o * n
            acc[3 - y] += o * n                          # [후보, 선택, 공격 선택, 정상 선택]
        x = rc[r]
        x[0] += n
        x[1 + g] += n
        x[5] += o * n
        if ty != nid:
            rtype[r] = ty
    by_grade = [[C.GRADES[g], "공격" if y else "정상", gc[(g * 2 + y) * 2], gc[(g * 2 + y) * 2 + 1]]
                for g in range(4) for y in (1, 0)]

    def by(acc, names):
        return sorted(([names[k], *v] for k, v in enumerate(acc) if v[0]), key=lambda r: -r[1])

    rule_rows = sorted(([t["rules"][k], t["types"][rtype[k]], *v, ov["rules"].get(t["rules"][k], "")]
                        for k, v in enumerate(rc) if v[0]), key=lambda r: (-r[4], -r[2]))
    n_att = sum(t["y"][i] for i in sel)
    return {"n": t["n"], "cand": len(cand), "sel": len(sel), "sel_label": {"공격": n_att, "정상": len(sel) - n_att},
            "by_grade": by_grade, "why": why, "by_type": by(tc, t["types"]), "by_source": by(sc, t["sources"]),
            "by_date": by(dc, t["dates"]), "rules": rule_rows, "snapshot": t.get("snapshot"),
            "all_dates": t["dates"], "all_sources": t["sources"], "hold": hold, "unverified_att": unverified,
            "view": view if view in HOLD else "", "combined": bool(t.get("combined")), "overlap": overlap,
            "one_side": t.get("one_side"), "sel_grade": {C.GRADES[g]: sg[g] for g in (1, 2, 3)}, "fill": fill or None,
            "llm_ok_bad": llm_ok, "has_lk": bool(LK),
            "options": norm_options(options), "conditions": norm_conditions(conditions), "overrides": ov}


def lk_sig(t):
    """비교 표의 LLM 재확인 'label 맞음' 표시 지문 - 'LLM 맞음은 남기기' 초안을 담은 뒤 바뀌었는지 (확정 때 확인)."""
    return hashlib.sha1((t.get("lk") or "").encode()).hexdigest()[:16]


def read_table(base_name, dev):
    p = C.compared_files(base_name, dev)["table"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def combined_table(base_name):
    """두 장비 비교 결과를 짝지은 공통 선별 표 (둘 중 하나라도 없으면 None)."""
    ti, tw = read_table(base_name, "ips"), read_table(base_name, "waf")
    return combine(ti, tw) if ti and tw else None


# ── 버전 목록 (초안 저장·삭제, 기준 등록) ─────────────────────────────────

def save_draft(name, base_name, conditions, options, overrides, tables):
    """초안 저장 (tables = {장비: 표}). IPS·WAF 공통 선별 하나로 고르고 (mode = combined), 저장할 때의 장비별 비교 스냅숏과
    선택 수(두 장비 같음)를 함께 남긴다."""
    C.check_name(name)
    v = C.load_versions()
    base = C.get_version(v, base_name)
    if base.get("status") != "final":
        raise ValueError("기준 버전은 확정된 버전이어야 합니다")
    cur = next((x for x in v["versions"] if x["name"] == name), None)
    if cur and cur.get("status") == "final":
        raise ValueError(f"'{name}' 은 이미 확정된 버전입니다 - 다른 이름을 쓰세요")
    if cur and cur.get("status") == "candidate" and cur.get("base") != base_name:
        raise ValueError(f"'{name}' 은 기준 '{cur.get('base')}' 의 후보입니다 - 같은 기준으로만 선별을 다시 담습니다")
    if not tables or any(d not in tables for d in C.DEVICES):
        raise ValueError("IPS·WAF 비교 결과가 둘 다 있어야 합니다 - 먼저 두 장비 모두 비교하세요")
    for t in tables.values():
        if t["base"] != base_name:
            raise ValueError("비교 결과의 기준 버전이 다릅니다")
    tc = combine(tables["ips"], tables["waf"])
    pv = preview(tc, conditions, options, overrides)
    snap = {dev: tables[dev]["snapshot"] for dev in C.DEVICES}
    one = {"cand": pv["cand"], "sel": pv["sel"], "sel_label": pv["sel_label"], "by_grade": pv["by_grade"],
           "overlap": pv["overlap"], "one_side": pv["one_side"], "fill": pv["fill"]}
    counts = {dev: dict(one) for dev in C.DEVICES}
    item = {"name": name, "base": base_name, "status": "draft", "mode": "combined", "conditions": norm_conditions(conditions),
            "options": norm_options(options), "overrides": norm_overrides(overrides), "snapshot": snap,
            "draft_counts": counts, "created_at": (cur or {}).get("created_at") or C.now_iso(), "updated_at": C.now_iso()}
    if item["options"]["keep_llm_ok"]:                  # 비교 표의 LLM 재확인 표시가 담은 뒤 바뀌면 확정하지 않는다
        item["lk_sig"] = lk_sig(tc)
    if cur and cur.get("status") == "candidate":        # 후보: 선별만 바꾸고 rev·데이터 추가·목표는 그대로 (다음 내보내기 = rev+1)
        keep = {k: cur[k] for k in ("status", "rev", "revs", "additions", "targets", "counts", "files", "exported_at")
                if k in cur}
        item.update(keep, spec_changed_at=C.now_iso())
    if cur:
        cur.clear()
        cur.update(item)
    else:
        v["versions"].append(item)
    C.save_versions(v)
    return item


def delete_draft(name):
    v = C.load_versions()
    ver = C.get_version(v, name)
    if ver.get("status") != "draft":
        raise ValueError("확정된 버전은 화면에서 지우지 않습니다")
    v["versions"] = [x for x in v["versions"] if x["name"] != name]
    C.save_versions(v)


def register_base(name, tag, files, feature_code):
    """첫 기준 버전(기존 학습데이터) 등록: 장비별 학습 파일 CSV·원본 로그, 표준모델 피처 파일 경로."""
    C.check_name(name)
    v = C.load_versions()
    if any(x["name"] == name for x in v["versions"]):
        raise ValueError(f"'{name}' 은 이미 있습니다")
    out = {}
    for dev in C.DEVICES:
        f = (files or {}).get(dev) or {}
        csv_p, log_p, code = Path(f.get("train_csv", "")), Path(f.get("raw_log", "")), Path((feature_code or {}).get(dev, ""))
        for p, what in ((csv_p, "학습 파일"), (log_p, "원본 로그"), (code, "표준모델 피처 파일")):
            if not str(p) or not p.is_file():
                raise ValueError(f"{dev.upper()} {what}이 없습니다: {p}")
        head = csv_p.open(encoding="utf-8").readline()
        if not all(c in head for c in ("method", "payload", "label", f"{dev}_payload_")):
            raise ValueError(f"{dev.upper()} 학습 파일의 열이 표준모델 학습 형식(method, payload, label, {dev}_payload_*)이 아닙니다")
        out[dev] = {"train_csv": str(csv_p), "raw_log": str(log_p)}
        v["feature_code"][dev] = str(code)
    v["versions"].append({"name": name, "tag": tag or "", "kind": "external", "status": "final", "files": out,
                          "created_at": C.now_iso(), "finalized_at": C.now_iso()})
    C.save_versions(v)


def version_rows():
    """화면 버전 표: 이름·기준·상태·장비별 행 수(공격/정상)·추가 수."""
    v = C.load_versions()
    rows = []
    for x in v["versions"]:
        r = {"name": x["name"], "tag": x.get("tag", ""), "base": x.get("base", ""), "status": x.get("status"),
             "mode": x.get("mode", ""),                 # combined = IPS·WAF 공통 선별 ('' = 예전 장비별 초안·확정)
             "kind": x.get("kind", ""), "created_at": x.get("created_at", ""), "finalized_at": x.get("finalized_at", ""),
             "updated_at": x.get("updated_at", ""), "conditions": x.get("conditions"), "options": x.get("options"),
             "overrides": x.get("overrides"), "snapshot": x.get("snapshot"), "note": x.get("note", ""), "dev": {},
             # 1차 확정(후보)·최종 승인 (2026-10-30~): 최신 rev, rev 별 내보낸 수, 데이터 추가, 목표, 승인 기록
             "rev": x.get("rev"), "exported_at": x.get("exported_at", ""), "spec_changed_at": x.get("spec_changed_at", ""),
             "revs": [{k: r.get(k) for k in ("rev", "exported_at", "counts", "forced", "manual_rows")} for r in x.get("revs") or []],
             "additions": x.get("additions") or {}, "targets": x.get("targets") or {}, "approval": x.get("approval")}
        for dev in C.DEVICES:
            cnt = (x.get("counts") or {}).get(dev) or {}
            d = {"rows": cnt.get("rows"), "label": cnt.get("label"), "added": cnt.get("added"),
                 "added_label": cnt.get("added_label"), "by_grade": cnt.get("by_grade")}
            dc = (x.get("draft_counts") or {}).get(dev)
            if dc:
                d["draft"] = dc
            if x.get("status") in ("final", "candidate") and x.get("kind") != "external":
                d["files"] = {k: str(Path(p).name) for k, p in ((x.get("files") or {}).get(dev) or {}).items()}
            r["dev"][dev] = d
        rows.append(r)
    return rows


# ── 확정 실행 계획 (화면 6 의 전체 진행률: 단계마다 예상 시간) ─────────────────

CHECK_TEXT = "확정 전 확인 (비교 스냅숏·학습데이터 변경·디스크)"
MOVE_TEXT = "마무리 (버전 폴더로 옮기기·버전 목록 저장)"
FINAL_STEPS = (("select", "선별"), ("lines", "로그 줄"), ("log", "로그"), ("csv", "학습 파일"), ("list", "목록"), ("hash", "지문"))


def final_step_text(dev, step, n=0, csv_name="", log_name=""):
    """확정 단계 이름 (확정할 때와 실행 계획이 같은 글을 씀). n = 추가 샘플 수."""
    D = dev.upper()
    return {"select": f"{D} 선별 (초안 조건 그대로)", "lines": f"{D} 추가 샘플 {n:,}개의 로그 줄 읽기",
            "log": f"{D} 로그 내보내기 (기준 {log_name} + 추가 {n:,}줄)",
            "csv": f"{D} 학습 파일 내보내기 (기준 {csv_name} + 추가 {n:,}행)",
            "list": f"{D} 추가 샘플 목록·비교한 샘플 지문", "hash": f"{D} 파일 지문(sha256)"}[step]


def _std_csv(csv_p, dev):
    """기준 학습 파일이 표준 학습 형식(method, payload, label, 피처 열 이름순)이라 그대로 복사하는지 (첫 줄만 봄)."""
    import csv
    with open(csv_p, encoding="utf-8", newline="") as f:
        head = next(csv.reader([f.readline()]), [])
    feats = head[3:]
    return head[:3] == ["method", "payload", "label"] and feats == sorted(feats) and all(
        c.startswith(f"{dev}_payload_") for c in feats)


def finalize_plan(name, timing=None, est=None):
    """확정 실행 계획 [{key, dev, short, text, expected_s, tkey}]: 지난 확정 시간(cache/timing.json, 최근 3번 중앙값),
    없으면 파일 크기·수로 어림 (복사 약 150MB/초, 표준 열만 다시 쓰기 약 15MB/초, sha256 약 400MB/초)."""
    timing = C.read_json(C.CACHE / "timing.json", {}) if timing is None else timing
    v = C.load_versions()
    ver = C.get_version(v, name)
    base = C.get_version(v, ver["base"])
    mb = lambda p: p.stat().st_size / 2 ** 20 if p.is_file() else 0.0
    past = lambda tk, guess: C.past_median(timing, tk) or timing.get(tk) or guess
    plan = [{"key": "prep", "dev": "", "short": "준비", "text": "준비 (계산 모듈 읽기)",
             "expected_s": past("finalize:prep", 3.0), "tkey": "finalize:prep"},
            {"key": "check", "dev": "", "short": "확인", "text": CHECK_TEXT, "expected_s": past("finalize:check", 3.0),
             "tkey": "finalize:check"}]
    for dev in C.DEVICES:
        if dev not in (ver.get("snapshot") or {}):
            continue
        csv_p, log_p = C.version_files(v, base, dev)
        n = int((((ver.get("draft_counts") or {}).get(dev)) or {}).get("sel") or 0)
        rows = (C.read_json(C.compared_files(base["name"], dev)["summary"], {}) or {}).get("rows") or 0
        logs = sum(st[1] for st in (C.log_stat(d, dev) for d in C.dataset_dates()) if st) / 2 ** 20
        copy = mb(csv_p) / (150 if csv_p.is_file() and _std_csv(csv_p, dev) else 15)
        guess = {"select": 3 + rows * 2e-5, "lines": 2 + logs / 60, "log": 1 + mb(log_p) / 150,
                 "csv": 10 + copy + n * 3e-4, "list": 1 + rows * 1e-5 + n * 1e-4, "hash": 1 + (mb(csv_p) + mb(log_p)) / 400}
        for step, short in FINAL_STEPS:
            tk = f"finalize:{dev}:{step}"
            plan.append({"key": f"{dev}:{step}", "dev": dev, "short": short,
                         "text": final_step_text(dev, step, n, csv_p.name, log_p.name),
                         "expected_s": round(past(tk, guess[step]), 1), "tkey": tk})
    plan.append({"key": "move", "dev": "", "short": "마무리", "text": MOVE_TEXT, "expected_s": past("finalize:move", 2.0),
                 "tkey": "finalize:move"})
    return plan


# ── 확정 (실행기에서, pandas 사용) ───────────────────────────────────────

def _sha256(p, ctx=None):
    h = hashlib.sha256()
    total, done = Path(p).stat().st_size, 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
            done += len(b)
            if ctx is not None:
                ctx.progress(done >> 20, max(1, total >> 20), unit="MB")
    return h.hexdigest()


def _copy(src, dst, ctx, what):
    """파일 복사 (shutil.copyfile 과 같은 내용, 8MB 씩 읽어 진행 보고)."""
    total, done = Path(src).stat().st_size, 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for b in iter(lambda: fi.read(8 << 20), b""):
            fo.write(b)
            done += len(b)
            ctx.progress(done >> 20, max(1, total >> 20), unit="MB", what=what)


def _rename_dir(src, dst, ctx, wait_s=60.0):
    """임시 폴더 → 버전 폴더. 막 쓴 큰 파일을 백신 검사 등이 잠깐 잡고 있으면 Windows 가 이름 바꾸기를
    '액세스가 거부되었습니다'(PermissionError) 로 막는다 (이 PC 에서 약 470MB 를 쓴 직후 4초쯤)
    → 잠깐씩(0.25초부터 2초까지 늘려) 기다리며 다시, 모두 wait_s 초까지."""
    t0, k = time.monotonic(), 0
    while True:
        try:
            src.rename(dst)
            break
        except PermissionError as e:
            waited = time.monotonic() - t0
            if waited >= wait_s:
                raise PermissionError(f"{dst.name} 폴더로 옮기지 못했습니다 - {wait_s:.0f}초 동안 폴더가 잠겨 있었습니다 "
                                      f"(그 폴더를 연 탐색기 창·백신 검사를 확인하고 다시 확정하세요): {e}") from e
            k += 1
            ctx.progress(0, 0, text=f"폴더가 잠겨 있어 기다렸다 다시 ({k}번째 · {waited:.0f}초)")
            time.sleep(min(2.0, 0.25 * k))
    if k:
        ctx.log(f"  버전 폴더로 옮기기: 폴더가 잠겨 {k}번 다시 해서 됨 ({time.monotonic() - t0:.1f}초)")


def _check_ready(v, ver):
    """사람이 확인한 비교 결과 그대로인지: 저장할 때의 스냅숏 = 지금 비교 결과, 그 뒤 학습데이터 변경 없음."""
    base_name = ver["base"]
    for dev, snap in (ver.get("snapshot") or {}).items():
        summ = C.read_json(C.compared_files(base_name, dev)["summary"], None)
        if not summ or summ.get("snapshot") != snap:
            raise ValueError(f"{dev.upper()} 비교 결과가 초안을 저장할 때와 다릅니다 - 지금 결과로 확인하고 '버전 후보에 담기'를 다시 하세요")
        for date, st in (summ.get("log_stats") or {}).items():
            if C.log_stat(date, dev) != st:
                raise ValueError(f"{dev.upper()} {date} 학습데이터가 비교한 뒤 바뀌었습니다 (새 생성·수정 반영) - 다시 비교하세요")


def _forced(tc, cand, ver):
    """'데이터 추가'로 강제로 넣는 샘플 (공통 표의 줄 번호). 룰·공격 유형 단위는 조건(날짜·출처) 안에서, 샘플 단위(sk)는
    조건과 상관없이. 둘 다 제거(기준·신규 내부 중복)는 넣지 않는다 → (줄 번호 목록, 줄 번호 → 넣은 까닭, 중복이라 못 넣은 수)."""
    add = ver.get("additions") or {}
    rules, types, sks = set(add.get("rules") or []), set(add.get("types") or []), set(add.get("samples") or [])
    if not (rules or types or sks):
        return [], {}, 0
    R, T, G, SK, rname, tname = tc["rule"], tc["type"], tc["g"], tc["sk"], tc["rules"], tc["types"]
    in_cond, out, why, dup = set(cand), [], {}, 0
    for k in range(tc["n"]):
        hit = ("samples" if SK[k] in sks else "rules" if k in in_cond and rname[R[k]] in rules
               else "types" if k in in_cond and tname[T[k]] in types else None)
        if hit is None:
            continue
        if G[k] == 0:
            dup += 1
            continue
        out.append(k)
        why[k] = hit
    return out, why, dup


# ── 데이터 추가 (최종 승인 전 후보만): 강제로 넣기·외부 파일 ─────────────────────

def _candidate(v, name):
    ver = C.get_version(v, name)
    if ver.get("status") != "candidate":
        raise ValueError(f"'{name}' 은 1차 확정(후보) 버전이 아닙니다 - 데이터 추가·목표·승인·버리기는 후보에서만 합니다")
    return ver


def set_additions(name, kind, values, action="add"):
    """강제로 넣기 목록 바꾸기: kind = rules(룰 id) | types(공격 유형) | samples(샘플 지문 sk), action = add | remove | set.
    다음 내보내기(rev+1)부터 들어간다."""
    if kind not in ADD_KINDS or action not in ("add", "remove", "set"):
        raise ValueError("kind/action")
    vals = list(dict.fromkeys(str(x).strip() for x in (values or []) if str(x).strip()))
    if any(len(x) > 200 for x in vals):
        raise ValueError("값이 너무 깁니다")
    if kind == "samples" and any(not SK_RE.match(x) for x in vals):
        raise ValueError("샘플 지문(sk)은 16자리 16진수입니다 (4. 화면 줄의 sk)")
    v = C.load_versions()
    ver = _candidate(v, name)
    add = ver.setdefault("additions", {})
    cur = list(add.get(kind) or [])
    if action == "add":
        cur += [x for x in vals if x not in cur]
    elif action == "remove":
        drop = set(vals)
        cur = [x for x in cur if x not in drop]
    else:
        cur = vals
    add[kind] = cur
    ver["updated_at"] = C.now_iso()
    C.save_versions(v)
    return add


def _train_form(raw):
    """학습 로그 payload → 학습 파일 payload (5.0b raw_to_payload 와 같은 규칙: 요청 줄 끝 뒤의 '\\n' 만 줄바꿈으로)."""
    m = re.search(r" HTTP/\d(\.\d)?(?=\\n|$)", raw)
    if not m:
        return raw
    return raw[:m.end()] + raw[m.end():].replace("\\r\\n", "\n").replace("\\n", "\n")


def _h(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


_M14 = {}


def _one_line(http):
    """HTTP 원문 → 학습 로그 한 줄 payload (1.4 one_line 과 같은 규칙, 매핑 md 의 'payload 줄바꿈')."""
    if "M" not in _M14:
        M = _load("dataset_v", "1_4_build_dataset.py")
        _M14.update(M=M, st=M.load_mapping()[0])
    return _M14["M"].one_line(_M14["M"].normalize_http(http.replace("\r\n", "\n")), _M14["st"])


def parse_manual(text):
    """외부 파일 → ([{method, raw(학습 로그 payload), label, attack_type, line(로그 줄이면 그 줄)}], [오류]).
    CSV: 머리줄에 method·payload·label (attack_type 은 있으면). payload 는 HTTP 원문(줄바꿈 그대로) 또는 학습 로그처럼 '\\n' 표기.
    로그: '|||' 로 나눈 8필드 줄 (시각|||출발IP|||포트|||목적IP|||포트|||method|||payload|||label)."""
    rows, errors = [], []
    lines = [x for x in text.replace("\r\n", "\n").split("\n") if x.strip()]
    if not lines:
        return rows, ["빈 파일"]
    if "|||" in lines[0]:
        for i, line in enumerate(lines, 1):
            f = C.split_log_line(line)
            if len(f) != 8:
                errors.append(f"{i}번째 줄: '|||' 로 나눈 칸이 {len(f)}개 (8개여야 함)")
            elif f[7] not in ("anomalies", "normal"):
                errors.append(f"{i}번째 줄: label '{f[7][:20]}' (anomalies·normal 만)")
            elif not f[6].strip():
                errors.append(f"{i}번째 줄: payload 가 비었음")
            else:
                rows.append({"method": f[5], "raw": f[6], "label": f[7], "attack_type": "", "line": line})
    else:
        rd = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        cols = [c.strip().lower() for c in (rd.fieldnames or [])]
        miss = [c for c in ("method", "payload", "label") if c not in cols]
        if miss:
            return [], [f"CSV 머리줄에 {', '.join(miss)} 열이 없습니다 (method, payload, label[, attack_type])"]
        for i, r in enumerate(rd, 2):
            r = {(k or "").strip().lower(): (val or "") for k, val in r.items()}
            label, payload = r.get("label", "").strip(), r.get("payload", "")
            if label not in ("anomalies", "normal"):
                errors.append(f"{i}번째 줄: label '{label[:20]}' (anomalies·normal 만)")
            elif not payload.strip():
                errors.append(f"{i}번째 줄: payload 가 비었음")
            elif "|||" in payload or "|||" in r.get("method", ""):
                errors.append(f"{i}번째 줄: 로그 구분자 '|||' 가 있음")
            else:
                raw = _one_line(payload) if "\n" in payload.replace("\r\n", "\n") else payload
                rows.append({"method": r.get("method", "").strip(), "raw": raw, "label": label,
                             "attack_type": r.get("attack_type", "").strip(), "line": None})
    if len(rows) > MANUAL_MAX:
        errors.append(f"줄이 {len(rows):,}개 - 파일 하나에 {MANUAL_MAX:,}줄까지")
    return rows, errors


_BASE_PAY = {}                                          # 기준 학습 파일 → (수정 시각, payload 지문 집합)


def _base_payloads(path):
    """기준 학습 파일의 payload 지문 (중복 검사, 파일이 그대로면 지난 결과)."""
    import pandas as pd
    p = Path(path)
    mt = p.stat().st_mtime
    hit = _BASE_PAY.get(str(p))
    if hit and hit[0] == mt:
        return hit[1]
    s = pd.read_csv(p, usecols=["payload"], dtype=str, keep_default_na=False)["payload"]
    keys = {_h(x) for x in s}
    _BASE_PAY[str(p)] = (mt, keys)
    return keys


def manual_dups(v, ver, rows):
    """외부 파일 줄의 중복: 기준(두 장비 학습 파일 중 하나라도), 신규(이 후보 최신 rev 의 추가 샘플), 파일 안 → 줄마다 까닭(없으면 '')."""
    base = C.get_version(v, ver["base"])
    bkeys = set()
    for dev in C.DEVICES:
        csv_p, _ = C.version_files(v, base, dev)
        bkeys |= _base_payloads(csv_p)
    new = set()
    added = C.VERDIR / ver["name"] / f"{C.DEVICES[0]}_added.csv"
    if added.is_file():
        with open(added, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("origin") != "manual":
                    new.add(_h(r.get("payload", "")))
    seen, out = set(), []
    for r in rows:
        k = _h(r["raw"])
        out.append("기준" if _h(_train_form(r["raw"])) in bkeys else "신규" if k in new else "파일 안" if k in seen else "")
        seen.add(k)
    return out


def add_manual(name, filename, data):
    """외부 파일 넣기: 형식·label 을 검사하고(오류가 있으면 넣지 않음), 중복 수를 세어 manual/ 에 원본을 둔다.
    다음 내보내기(rev+1)부터 들어간다 (중복 줄은 빼고, added.csv 에 origin = manual)."""
    v = C.load_versions()
    ver = _candidate(v, name)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("UTF-8 텍스트 파일이 아닙니다 (CSV 또는 '|||' 로그)")
    rows, errors = parse_manual(text)
    if errors:
        raise ValueError(f"형식 오류 {len(errors)}건 - 고친 뒤 다시 넣으세요: " + "; ".join(errors[:5]))
    dup = manual_dups(v, ver, rows)
    d = C.VERDIR / name / "manual"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", Path(filename or "manual.csv").name)[-80:] or "manual.csv"
    fn = f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
    (d / fn).write_bytes(data)
    lab = Counter(r["label"] for r, x in zip(rows, dup) if not x)
    entry = {"file": fn, "orig": Path(filename or "").name, "format": "log" if rows[0]["line"] else "csv", "rows": len(rows),
             "use": sum(1 for x in dup if not x), "label": {k: lab.get(k, 0) for k in ("anomalies", "normal")},
             "dup": {k: dup.count(k) for k in ("기준", "신규", "파일 안")}, "sha256": hashlib.sha256(data).hexdigest(),
             "added_at": C.now_iso()}
    v = C.load_versions()
    ver = _candidate(v, name)
    ver.setdefault("additions", {}).setdefault("manual", []).append(entry)
    ver["updated_at"] = C.now_iso()
    C.save_versions(v)
    return entry


def remove_manual(name, fn):
    """넣은 외부 파일 빼기 (원본은 manual/_빠짐/ 으로 옮김)."""
    v = C.load_versions()
    ver = _candidate(v, name)
    add = ver.setdefault("additions", {})
    before = add.get("manual") or []
    add["manual"] = [m for m in before if m["file"] != fn]
    if len(add["manual"]) == len(before):
        raise ValueError("그 파일이 없습니다")
    src = C.VERDIR / name / "manual" / fn
    if src.is_file():
        (src.parent / "_빠짐").mkdir(exist_ok=True)
        os.replace(src, src.parent / "_빠짐" / fn)
    ver["updated_at"] = C.now_iso()
    C.save_versions(v)
    return add


def set_targets(name, targets):
    """최종 승인 목표: {장비: {precision, recall, f1 (0~1, 비우면 안 봄), not_below_base}, exp: 판단에 쓰는 실험}."""
    v = C.load_versions()
    ver = _candidate(v, name)
    out = {"exp": str((targets or {}).get("exp") or DEFAULT_EXP)}
    for dev in C.DEVICES:
        t = (targets or {}).get(dev) or {}
        d = {"not_below_base": bool(t.get("not_below_base"))}
        for m in METRICS:
            x = t.get(m)
            if x in (None, ""):
                d[m] = None
                continue
            try:
                x = float(x)
            except (TypeError, ValueError):
                raise ValueError(f"{dev.upper()} {m} 목표는 0~1 사이 수")
            if not 0 <= x <= 1:
                raise ValueError(f"{dev.upper()} {m} 목표는 0~1 사이 수")
            d[m] = round(x, 4)
        out[dev] = d
    ver["targets"] = out
    ver["updated_at"] = C.now_iso()
    C.save_versions(v)
    return out


def eval_results(name):
    """피처 테스트 결과 {rev: {장비: {실험: 결과}}} - 5.4 eval-version 이 versions/<이름>/eval/rev<n>/<장비>_<실험>.json 에 쓴다."""
    out = {}
    d = C.VERDIR / name / "eval"
    for rd in sorted(d.glob("rev*")) if d.is_dir() else []:
        m = re.match(r"rev(\d+)$", rd.name)
        if not m:
            continue
        for p in sorted(rd.glob("*.json")):
            mm = re.match(r"(ips|waf)_(.+)\.json$", p.name)
            r = C.read_json(p, None) if mm else None
            if isinstance(r, dict):
                out.setdefault(int(m.group(1)), {}).setdefault(mm.group(1), {})[mm.group(2)] = r
    return out


def check_targets(targets, res, exp=None):
    """승인 판단: 장비마다 실험 exp 결과가 목표(precision·recall·F1 최소값)와 '기준 버전보다 낮지 않음'을 넘는지.
    res = {장비: {실험: 결과}} (한 rev). → {complete, met, exp, dev: {장비: {has, met, fails, metrics, base}}}"""
    exp = exp or (targets or {}).get("exp") or DEFAULT_EXP
    out, complete, met = {}, True, True
    for dev in C.DEVICES:
        r = ((res or {}).get(dev) or {}).get(exp)
        t = (targets or {}).get(dev) or {}
        if not r:
            complete = False
            out[dev] = {"has": False, "met": False, "fails": ["테스트 결과 없음"]}
            continue
        b, fails = r.get("base") or {}, []
        if r.get("quick"):                              # 표본(빠른 확인) 결과만으로는 승인하지 않음 - 그래도 승인하려면 사유
            fails.append("빠른 확인(표본) 결과 - 전체 테스트 필요")
        for m in METRICS:
            x, want = r.get(m), t.get(m)
            if want is not None and (x is None or x < want):
                fails.append(f"{m} {x if x is None else round(x, 4)} < 목표 {want}")
            if t.get("not_below_base") and x is not None and b.get(m) is not None and round(x, 4) < round(b[m], 4):
                fails.append(f"{m} {round(x, 4)} < 기준 버전 {round(b[m], 4)}")
        out[dev] = {"has": True, "met": not fails, "fails": fails, "metrics": {m: r.get(m) for m in METRICS + ("miss", "fp")},
                    "base": {m: b.get(m) for m in METRICS + ("miss", "fp")}}
        met = met and not fails
    return {"complete": complete, "met": complete and met, "exp": exp, "dev": out}


def _server_gate(name, rev):
    """서버 반영 조건 (5_8_server.gate - 로컬 전용 서버 레시피 파일 5_8a 가 있을 때만 켜짐): 장비마다 서버 혼동행렬 4지표 95% 이상 +
    디코딩 입력·원본 입력 예측 일치 98% 이상. 꺼져 있으면 None (예전과 같은 승인)."""
    if not (BASE / "5_8_server.py").exists():
        return None
    g = _load("sup_server_v", "5_8_server.py").gate(name, int(rev))
    return g if g.get("active") else None


def approve(name, rev, reason="", exp=None):
    """최종 승인 → 확정 (잠금, 다음 버전의 기준). 최신 rev 의 두 장비 테스트가 목표를 넘어야 하고, 못 넘으면 사유가 있어야 한다.
    서버 반영 (2026-11-04~, 5_8): 서버 결과가 기준(4지표 95%·예측 일치 98%)에 못 미치거나 없어도 사유가 있어야 한다.
    승인 기록(시각·rev·실험·지표·기준 지표·목표·사유·서버 반영)을 manifest.json 과 versions.json 에 남긴다."""
    v = C.load_versions()
    ver = _candidate(v, name)
    if int(rev) != ver.get("rev"):
        raise ValueError(f"최신 rev {ver.get('rev')} 만 승인합니다 (그 rev 로 테스트한 결과로)")
    targets = ver.get("targets") or {}
    chk = check_targets(targets, eval_results(name).get(int(rev)), exp)
    if not chk["complete"]:
        raise ValueError(f"rev {rev} 의 IPS·WAF 테스트 결과(실험 {chk['exp']})가 둘 다 있어야 승인합니다")
    reason = str(reason or "").strip()
    if not chk["met"] and not reason:
        raise ValueError("목표에 못 미칩니다 - 그래도 승인하려면 사유를 적으세요: " +
                         "; ".join(f"{d.upper()} {', '.join(x['fails'])}" for d, x in chk["dev"].items() if x["fails"]))
    srv = _server_gate(name, rev)                       # 서버 반영 (2026-11-04~): 못 넘거나 결과가 없으면 사유 필요
    if srv and not srv["met"] and not reason:
        raise ValueError("서버 반영 기준(혼동행렬 4지표 95%·예측 일치 98%)에 못 미치거나 서버 결과가 없습니다 - 그래도 승인하려면 "
                         "사유를 적으세요: " + "; ".join(f"{d.upper()} {', '.join(x['fails'])}" for d, x in srv["dev"].items() if x["fails"]))
    if ver.get("spec_changed_at") and ver["spec_changed_at"] > (ver.get("exported_at") or ""):
        raise ValueError("선별을 다시 담은 뒤 내보내지 않았습니다 - 다시 내보내기(rev+1)·테스트 뒤 승인하세요")
    now = C.now_iso()
    rec = {"at": now, "rev": int(rev), "exp": chk["exp"], "targets": targets, "met": chk["met"] and (not srv or srv["met"]),
           "reason": reason[:1000],
           "metrics": {d: x.get("metrics") for d, x in chk["dev"].items()}, "base": {d: x.get("base") for d, x in chk["dev"].items()},
           "fails": {d: x["fails"] for d, x in chk["dev"].items()}}
    if srv:                                             # met = 피처 테스트 목표 + 서버 반영, targets_met = 피처 테스트 목표만
        rec.update(targets_met=chk["met"], server=srv)
    mp = C.VERDIR / name / "manifest.json"
    m = C.read_json(mp, None)
    if not m or m.get("rev") != int(rev):
        raise ValueError(f"{mp} 이 rev {rev} 의 것이 아닙니다 - 다시 내보내세요")
    m.update(status="final", finalized_at=now, approval=rec)
    C.write_json(mp, m)
    ver.update(status="final", finalized_at=now, approval=rec)
    C.save_versions(v)
    return rec


class _NoCtx:
    def progress(self, *a, **k):
        pass

    def log(self, msg):
        pass


def discard(name):
    """승인 전 후보 버리기: 폴더를 _보관/버린후보_<시각>_<이름>/ 으로 옮기고 목록에서 빼서 이름을 다시 쓸 수 있게 한다."""
    v = C.load_versions()
    _candidate(v, name)
    src, dst = C.VERDIR / name, None
    if src.exists():
        ARCHIVE.mkdir(exist_ok=True)
        dst = ARCHIVE / f"버린후보_{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        _rename_dir(src, dst, _NoCtx(), wait_s=20.0)
    v = C.load_versions()
    v["versions"] = [x for x in v["versions"] if x["name"] != name]
    C.save_versions(v)
    return {"moved_to": str(dst) if dst else ""}


def _commit_rev(tmp, out, prev_rev, ctx):
    """rev 2 부터: 지난 rev 의 작은 파일(manifest·added·considered)은 revs/rev<n>/ 에 남기고, 새 파일로 바꾼다
    (큰 학습 파일·로그는 같은 이름으로 바로 바꿔 쓴다 - 지난 rev 것은 남기지 않음, manifest 의 sha256 으로 확인)."""
    keep = out / "revs" / f"rev{prev_rev}"
    keep.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*"):
        if p.is_file() and not p.name.endswith(("_train.csv", "_raw.log")):
            shutil.copy2(p, keep / p.name)
    for p in sorted(tmp.iterdir()):
        for k in range(40):
            try:
                os.replace(p, out / p.name)
                break
            except PermissionError:
                if k == 39:
                    raise PermissionError(f"{out / p.name} 을 바꾸지 못했습니다 - 그 파일을 연 프로그램을 닫고 다시 내보내세요")
                ctx.progress(0, 0, text=f"{p.name} 가 잠겨 있어 기다렸다 다시 ({k + 1}번째)")
                time.sleep(0.5)
    shutil.rmtree(tmp, ignore_errors=True)


def export(name, ctx):
    """1차 확정 (후보): 초안 → rev 1, 후보 → rev+1 (데이터 추가 반영). 기준 버전 + 선별한 샘플(+ 강제로 넣은 샘플·외부 파일)의
    학습 파일·로그를 내보낸다. 후보는 피처 테스트·최종 승인 전이라 다음 버전의 기준이 될 수 없다."""
    import pandas as pd
    F = _load("sup_features_v", "5_0b_features.py")
    v = C.load_versions()
    ver = C.get_version(v, name)
    if ver.get("status") not in ("draft", "candidate"):
        raise ValueError(f"'{name}' 은 이미 최종 승인(확정)된 버전입니다 - 고칠 수 없습니다")
    rev = int(ver.get("rev") or 0) + 1 if ver.get("status") == "candidate" else 1
    base = C.get_version(v, ver["base"])
    devs = [d for d in C.DEVICES if d in (ver.get("snapshot") or {})]
    if not devs:
        raise ValueError("초안에 비교 결과가 없습니다")
    if ver.get("mode") != "combined" or devs != list(C.DEVICES):
        raise ValueError("예전 방식(장비별 선별)으로 담은 초안입니다 - 화면 6 버전 표의 '새 방식으로 다시 담기'를 누르세요 "
                         "(옵션·조정은 그대로 이어받고 IPS·WAF 에 같은 샘플을 넣음)")
    ctx.step("check", CHECK_TEXT)
    _check_ready(v, ver)
    need = 0
    for dev in devs:
        csv_p, log_p = C.version_files(v, base, dev)
        need += csv_p.stat().st_size + log_p.stat().st_size + 2 * 2 ** 20 * max(1, ver["draft_counts"][dev]["sel"] // 1000)
    free = C.disk_free_gb()
    if free < need / 2 ** 30 + 1.0:
        raise ValueError(f"디스크 여유 {free:.1f}GB 가 모자랍니다 (내보내기 약 {need / 2 ** 30:.1f}GB + 여유 1GB 필요)")
    out = C.VERDIR / name
    tmp = C.VERDIR / f"_{name}.tmp"
    if rev == 1 and out.exists():
        raise ValueError(f"{out} 가 이미 있습니다")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    adds = ver.get("additions") or {}
    manifest = {"name": name, "base": base["name"], "mode": "combined", "conditions": ver["conditions"],
                "options": ver["options"], "overrides": ver["overrides"], "snapshot": ver["snapshot"],
                "created_at": ver.get("created_at"), "status": "candidate", "rev": rev, "exported_at": None,
                "finalized_at": None, "approval": None,
                "additions": {k: adds.get(k) or [] for k in ADD_KINDS} | {"manual": [m["file"] for m in adds.get("manual") or []]},
                "dev": {}}
    counts, files, tc, picked, manual, mrows = {}, {}, None, {}, [], []
    try:
        for dev in devs:
            ctx.check()
            ctx.step(f"{dev}:select", final_step_text(dev, "select"))
            if tc is None:                                  # IPS·WAF 공통 선별 (한 번만, 두 장비가 같은 샘플)
                ctx.sub(0, 3, "선별용 표 읽기 (IPS·WAF 짝짓기)")
                tc = combined_table(base["name"])
                if (ver.get("options") or {}).get("keep_llm_ok") and ver.get("lk_sig") != lk_sig(tc):
                    raise ValueError("담은 뒤 비교 결과의 LLM 재확인 표시가 바뀌었습니다 - 지금 결과로 확인하고 '버전 후보에 담기'를 다시 하세요")
                ctx.sub(1, 3, f"공통 선별 ({tc['n']:,}개)")
                sel, cand, why = select(tc, ver["conditions"], ver["options"], ver["overrides"])
                forced, fwhy, fdup = _forced(tc, cand, ver)  # 데이터 추가: 강제로 넣기 (선별에 이미 든 것은 그대로)
                sset = set(sel)
                extra = [k for k in forced if k not in sset]
                added_by = ["선별"] * len(sel) + ["넣기:" + ADD_KINDS[fwhy[k]] for k in extra]
                sel = list(sel) + extra
                manifest["forced"] = {"rules": sum(1 for k in extra if fwhy[k] == "rules"),
                                      "types": sum(1 for k in extra if fwhy[k] == "types"),
                                      "samples": sum(1 for k in extra if fwhy[k] == "samples"),
                                      "already_selected": len(forced) - len(extra), "dup": fdup}
            ctx.sub(2, 3, "샘플별 등급 읽기 (rows.parquet)")
            R = pd.read_parquet(C.compared_files(base["name"], dev)["rows"])
            t_n = (C.read_json(C.compared_files(base["name"], dev)["summary"], {}) or {}).get("rows")
            if t_n is not None and len(R) != t_n:
                raise ValueError(f"{dev.upper()} 비교 결과 파일이 서로 맞지 않습니다 - 다시 비교하세요")
            ix = tc["iidx" if dev == "ips" else "widx"]     # 공통 표의 줄 → 이 장비 표(= rows.parquet)의 줄
            A = R.iloc[[ix[k] for k in sel]].reset_index(drop=True)
            picked[dev] = A["sk"].tolist()
            if dev != devs[0] and picked[dev] != picked[devs[0]]:
                raise ValueError("IPS·WAF 추가 샘플이 서로 다릅니다 - 비교 결과를 다시 만드세요")
            A["grade_dev"] = A["grade"]                     # 이 장비의 등급 (목록에 남김)
            A["grade"] = [C.GRADES[tc["g"][k]] for k in sel]   # 공통 등급 (선별 기준)
            A["added_by"] = added_by
            # 고른 줄의 로그 (날짜별로 한 번 읽음) - 내용 지문이 비교 때와 같은지 확인
            ctx.step(f"{dev}:lines", final_step_text(dev, "lines", len(A)))
            lines = {}
            groups = list(A.groupby("date"))
            w = [(C.log_stat(date, dev) or [None, 0])[1] for date, _ in groups]       # 날짜마다 로그 전체를 읽음
            for k, (date, grp) in enumerate(groups):
                ctx.sub(k, len(groups), f"날짜 {k + 1}/{len(groups)} ({date}) · 찾은 줄 {len(lines):,}/{len(A):,}", w)
                want = set(grp["line"].tolist())
                with open(C.log_path(date, dev), "rb") as f:
                    for i, line in enumerate(x for x in f if x.strip()):
                        if i in want:
                            lines[(date, i)] = line.rstrip(b"\r\n").decode("utf-8")
            raw = [lines[(d, i)] for d, i in zip(A["date"], A["line"])]
            payload = [C.split_log_line(x)[6] for x in raw]
            bad = sum(C.sample_key(lb, r, p) != ck for lb, r, p, ck in zip(A["label"], A["rule_id"], payload, A["ck"]))
            if bad:
                raise ValueError(f"{dev.upper()} 로그 줄 {bad}개가 비교할 때와 다릅니다 - 다시 비교하세요")
            csv_p, log_p = C.version_files(v, base, dev)
            if dev == devs[0] and adds.get("manual"):       # 외부 파일: 두 장비 모두 같은 줄 (기준 중 하나·신규·파일 안 중복은 뺌)
                ctx.sub(0, 1, "외부 파일 읽기·중복 검사")
                bkeys = set()
                for d2 in devs:
                    bkeys |= _base_payloads(C.version_files(v, base, d2)[0])
                seen = {_h(p) for p in payload}
                for m in adds["manual"]:
                    p = out / "manual" / m["file"]
                    rows, errors = parse_manual(p.read_text(encoding="utf-8-sig"))
                    if errors:
                        raise ValueError(f"외부 파일 {m['orig'] or m['file']}: {errors[0]}")
                    use = 0
                    for r in rows:
                        k = _h(r["raw"])
                        if k in seen or _h(_train_form(r["raw"])) in bkeys:
                            continue
                        seen.add(k)
                        manual.append(dict(r, file=m["file"]))
                        use += 1
                    mrows.append({"file": m["file"], "rows": len(rows), "use": use})
                manifest["manual"] = mrows
            # 원본 로그 = 기준 원본 로그 + 추가 줄 (+ 외부 파일 줄)
            ctx.step(f"{dev}:log", final_step_text(dev, "log", len(raw) + len(manual), log_name=log_p.name))
            mlines = [_manual_line(r, dev) for r in manual]
            lp = tmp / f"{dev}_raw.log"
            _copy(log_p, lp, ctx, "기준 로그 복사")
            with open(lp, "rb+") as f:
                f.seek(0, 2)
                if f.tell():
                    f.seek(-1, 2)
                    end = f.read(1)
                    f.seek(0, 2)
                    if end != b"\n":
                        f.write(b"\n")
                f.write(("\n".join(raw + mlines) + "\n").encode("utf-8") if raw or mlines else b"")
            # 학습 파일 = 기준 학습 파일(표준 열만) + 추가 샘플 (payload 는 학습 파일 형식, 표준모델 피처는 원본 코드와 같은 값)
            ctx.step(f"{dev}:csv", final_step_text(dev, "csv", len(A) + len(manual), csv_name=csv_p.name))
            ns = F.load_legacy(C.feature_code(v, dev))
            names = F.legacy_names(ns)
            head = list(pd.read_csv(csv_p, nrows=0).columns)
            std = ["method", "payload", "label"] + sorted(c for c in head if c.startswith(f"{dev}_payload_"))
            if sorted(std[3:]) != names:
                raise ValueError(f"{dev.upper()} 기준 학습 파일의 피처 열이 표준모델 피처 파일과 다릅니다")
            cp = tmp / f"{dev}_train.csv"
            what = "기준 학습 파일 복사" if head == std else "기준 학습 파일 (표준 열만) 쓰기"
            n_add = len(A) + len(manual)
            w = [csv_p.stat().st_size / 2 ** 20 / (150 if head == std else 15), 8, 1 + n_add * 3e-4]   # 하위 단계 예상 비율
            if n_add:
                ctx.sub(0, 3, what, w)
            if head == std:                                 # 표준 학습 형식이면 그대로 복사
                _copy(csv_p, cp, ctx, "" if n_add else what)
            else:                                           # 다른 열(proba 등)이 있으면 표준 열만
                first, done = True, 0
                total = ((base.get("counts") or {}).get(dev) or {}).get("rows") or 0
                for ch in pd.read_csv(csv_p, usecols=std, chunksize=50000, dtype=str, keep_default_na=False):
                    ch[std].to_csv(cp, mode="w" if first else "a", header=first, index=False, lineterminator="\n")
                    first = False
                    done += len(ch)
                    ctx.progress(done, max(done, total), what="" if n_add else what)
            ints = set(ns["regex_dict_comb"]) | set(ns["regex_dict_count"])
            parts = []
            if len(A):
                ctx.sub(1, 3, "피처 캐시 읽기", w)
                fd = C.CACHE / "features" / dev / F.code_hash(C.feature_code(v, dev))
                fc = pd.concat([pd.read_parquet(p, columns=["pk"] + names) for p in sorted(fd.glob("part-*.parquet"))],
                               ignore_index=True).drop_duplicates("pk", keep="last").set_index("pk")
                miss = [pk for pk in A["pk"] if pk not in fc.index]
                if miss:                                    # 캐시가 지워졌으면 다시 계산
                    todo = {pk: p for pk, p in zip(A["pk"], payload) if pk in set(miss)}
                    extra_f = F.compute_chunk(dev, str(C.feature_code(v, dev)), list(todo.values()))[names]
                    extra_f.index = list(todo.keys())
                    fc = pd.concat([fc, extra_f])
                X = fc.loc[A["pk"].to_numpy(), names].reset_index(drop=True)
                parts.append(pd.concat([pd.DataFrame({"method": A["method"], "payload": [F.raw_to_payload(p) for p in payload],
                                                      "label": A["label"]}), X], axis=1))
            if manual:                                      # 외부 파일 줄: 표준모델 피처를 지금 계산
                ctx.sub(1, 3, f"외부 파일 {len(manual):,}줄 피처 계산", w)
                Xm = F.compute_chunk(dev, str(C.feature_code(v, dev)), [r["raw"] for r in manual])[names].reset_index(drop=True)
                parts.append(pd.concat([pd.DataFrame({"method": [r["method"] for r in manual],
                                                      "payload": [F.raw_to_payload(r["raw"]) for r in manual],
                                                      "label": [r["label"] for r in manual]}), Xm], axis=1))
            if parts:
                ctx.sub(2, 3, f"추가 샘플 {n_add:,}행 쓰기", w)
                new = pd.concat(parts, ignore_index=True)[std]
                for c in names:
                    if c in ints:
                        new[c] = new[c].round().astype("int64")
                with open(cp, "rb+") as f:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
                new.to_csv(cp, mode="a", header=False, index=False, lineterminator="\n")
            # 추가한 샘플 목록 (학습 파일의 추가 행과 같은 순서)·이번에 비교한 샘플 지문
            ctx.step(f"{dev}:list", final_step_text(dev, "list"))
            ctx.sub(0, 2, f"추가 샘플 목록 ({dev}_added.csv)")
            cols = ["date", "line", "sk", "ck", "rule_id", "source", "method", "label", "attack_type", "origin", "cons",
                    "grade", "grade_dev", "holdout", "new_pat", "wrong_rf", "wrong_lg", "added_by"]
            add = A[cols].copy()
            add["gen_origin"] = add["origin"]               # 생성 출처 llm·augmented (origin 은 5.4 eval-version 규격: new·manual)
            add["origin"] = "new"
            add["payload"] = payload
            if manual:
                add = pd.concat([add, pd.DataFrame([{
                    "date": "", "line": "", "sk": "m" + _h(r["raw"])[:15], "ck": C.sample_key(r["label"], "manual", r["raw"]),
                    "rule_id": "", "source": "manual", "method": r["method"], "label": r["label"],
                    "attack_type": r["attack_type"], "origin": "manual", "cons": "", "grade": "manual", "grade_dev": "",
                    "holdout": "", "new_pat": "", "wrong_rf": "", "wrong_lg": "", "added_by": "manual:" + r["file"],
                    "gen_origin": "", "payload": r["raw"]} for r in manual])], ignore_index=True)
            add.to_csv(tmp / f"{dev}_added.csv", index=False, encoding="utf-8-sig")
            ctx.sub(1, 2, f"비교한 샘플 지문 ({dev}_considered.txt)")
            ck_all = R["ck"].to_numpy()
            (tmp / f"{dev}_considered.txt").write_text("\n".join(ck_all[ix[k]] for k in cand) + "\n", encoding="utf-8")
            base_cnt = (base.get("counts") or {}).get(dev) or {}
            lab = Counter(A["label"].tolist() + [r["label"] for r in manual])
            gl = A.groupby(["grade", "label"]).size()
            counts[dev] = {
                "rows": int((base_cnt.get("rows") or 0) + n_add),
                "label": {k: int((base_cnt.get("label") or {}).get(k, 0)) + int(lab.get(k, 0)) for k in ("anomalies", "normal")},
                "added": int(n_add), "added_label": {k: int(lab.get(k, 0)) for k in ("anomalies", "normal")},
                "by_origin": {"new": int(len(A)), "manual": len(manual)},
                "by_grade": [[g, lb, int(gl.get((g, lb), 0))] for g in C.GRADES[1:] for lb in ("anomalies", "normal")],
                "by_type": {str(k): int(n) for k, n in A["attack_type"].value_counts().items()},
                "by_source": {str(k): int(n) for k, n in A["source"].value_counts().items()},
                "by_date": {str(k): int(n) for k, n in A["date"].value_counts().sort_index().items()},
                "considered": len(cand), "why": why, "methods": base_cnt.get("methods")}
            files[dev] = {k: str(out / f"{dev}_{k}{ext}") for k, ext in (("train", ".csv"), ("raw", ".log"),
                                                                           ("added", ".csv"))}
            ctx.step(f"{dev}:hash", final_step_text(dev, "hash"))
            fl = sorted(tmp.glob(f"{dev}_*"))
            w, hashed = [p.stat().st_size for p in fl], {}
            for k, p in enumerate(fl):
                ctx.sub(k, len(fl), f"{p.name} ({k + 1}/{len(fl)})", w)
                hashed[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256(p, ctx)}
            manifest["dev"][dev] = {"counts": counts[dev], "base_files": {"train_csv": str(csv_p), "raw_log": str(log_p)},
                                    "base_rows": int(base_cnt.get("rows") or 0), "added_rows": int(n_add), "files": hashed}
        ctx.step("move", MOVE_TEXT)
        manifest["exported_at"] = C.now_iso()
        sks = sorted(picked[devs[0]])                       # 두 장비에 같은 샘플 (지문 목록의 sha1 로 남김)
        manifest["samples"] = {"added": len(sks) + len(manual), "new": len(sks), "manual": len(manual), "considered": len(cand),
                               "one_side": tc["one_side"], "sk_sha1": hashlib.sha1("\n".join(sks).encode()).hexdigest()}
        C.write_json(tmp / "manifest.json", manifest)
        if rev == 1:
            _rename_dir(tmp, out, ctx)
        else:
            _commit_rev(tmp, out, rev - 1, ctx)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    v = C.load_versions()
    ver = C.get_version(v, name)
    info = {"rev": rev, "exported_at": manifest["exported_at"],
            "counts": {d: {k: counts[d][k] for k in ("rows", "added", "added_label", "by_origin")} for d in devs},
            "forced": manifest.get("forced"), "manual_rows": len(manual), "additions": manifest["additions"]}
    ver.update(status="candidate", rev=rev, exported_at=manifest["exported_at"], counts=counts,
               files={dev: {"train_csv": str(out / f"{dev}_train.csv"), "raw_log": str(out / f"{dev}_raw.log")} for dev in devs},
               revs=(ver.get("revs") or []) + [info])
    C.save_versions(v)
    return {"rev": rev, **{dev: {"added": counts[dev]["added"], "rows": counts[dev]["rows"]} for dev in devs}}


finalize = export                                       # 예전 이름 (5_0 finalize = export)


def _manual_line(r, dev):
    """외부 파일 줄 → 학습 로그 한 줄. 로그 줄로 넣었으면 그대로, CSV 면 1.4 와 같은 규칙으로 시각·주소·포트를 채운다
    (시각 = 내보낸 시각, 출발 IP·포트 = payload 지문, 목적지 = Host 헤더 또는 매핑 md 기본값)."""
    if r.get("line"):
        return r["line"]
    if "M" not in _M14:
        _one_line("x")
    M, st = _M14["M"], _M14["st"]
    seed = int(_h(r["raw"]), 16)
    http = _train_form(r["raw"])
    d_ip, hport = M.dst_of(http, st["기본 목적지 IP"])
    port = int(hport or st["IPS 포트" if dev == "ips" else "WAF 포트"])
    fields = [time.strftime("%Y-%m-%d %H:%M:%S"), M.src_ip(seed), str(1024 + seed % 64511), d_ip, str(port),
              r["method"] or "manual", r["raw"], r["label"]]
    return "|||".join(fields)
