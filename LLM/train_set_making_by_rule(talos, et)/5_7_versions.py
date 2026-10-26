"""[5.7 선별·버전] 비교 결과(5.5)에서 학습에 넣을 신규 샘플을 고르고(선별), 버전(26 하반기 등)으로 담아 확정·내보낸다.  (화면 6)

화면(0_0_run_ui.py)도 선별 미리 보기에 이 모듈을 쓰므로 맨 위에서는 표준 라이브러리만 불러온다 (확정은 실행기가 pandas 로).
→ 화면의 미리 보기와 확정이 같은 select() 를 써서, 사람이 본 수와 내보내는 샘플이 같다.

선별 기본값 (화면에서 바꿀 수 있고, 확정 전에 사람이 룰·공격 유형 단위로 최종 판단)
  제거  항상 뺌 (중복)
  A     전부 (기준 모델이 틀림 = 기준 데이터에 부족한 데이터)
  B     패턴당 20개까지 (새 패턴) - 같은 패턴의 반복(예: SQLi 변형)으로 쏠리지 않게
  C     패턴당 20개까지 (이미 학습된 패턴, 분포 유지용)
  - 상한은 등급·label·패턴마다 적용하고, 넘치면 룰을 돌아가며 고른다 (같은 패턴 안에서도 여러 룰, 룰 안에서는 LLM 샘플 먼저).
  - '불일치'(4. 화면 검증) 빼기: 기본 켬.  보류 옵션(담당관 회신 전, 기본 끔): '정합 (룰 기준)' 공격 빼기, python-requests UA 정상 빼기.
  - 룰·공격 유형 조정: 빼기(그 샘플 모두 뺌) / 넣기(상한 없이 제거 외 모두 넣음, 룰만).
버전
  초안   이름, 기준 버전, 조건(날짜·출처), 선별 옵션, 룰·유형 조정 + 저장할 때의 비교 스냅숏(장비별)
  확정   비교 스냅숏이 지금 비교 결과와 같고 그 뒤 학습데이터가 바뀌지 않았을 때만 내보낸다 (사람이 확인한 그대로)
         11_supervised/versions/<이름>/
           <장비>_train.csv       기준 학습 파일 + 추가 샘플 (method, payload, label, 표준모델 피처 = 표준모델 학습 형식)
           <장비>_raw.log         기준 원본 로그 + 추가 샘플 로그 줄 (|||, 8필드)   ← 로그 형식 파일은 여기서만 만든다
           <장비>_added.csv       추가한 샘플 (날짜·줄·룰·등급·근거·payload)
           <장비>_considered.txt  이번 버전에서 비교한 샘플 지문 (다음 비교에서 뺌)
           manifest.json          기준·조건·옵션·수·파일 크기·sha256
         확정한 버전은 다음 버전(예: 27 상반기)의 기준이 된다.
"""

import hashlib
import importlib.util
import json
import shutil
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("sup_common_v", "5_0a_common.py")
DEFAULT_OPTIONS = {"A": "all", "B": 20, "C": 20,          # "all" = 전부, 0 = 빼기, n = 패턴당 n개까지
                   "exclude_bad": True, "exclude_rule_based": False, "exclude_python_ua_normal": False}
OPTION_TEXT = {"exclude_bad": "'불일치' 빼기", "exclude_rule_based": "'정합 (룰 기준)' 공격 빼기 (보류)",
               "exclude_python_ua_normal": "python-requests UA 정상 빼기 (보류)"}
HOLD = {"rule": "'정합 (룰 기준)' 공격", "pyua": "python-requests UA 정상"}   # 보류 옵션의 대상 (화면에서 따로 봄)
PY_UA = 1                                                  # 5.5 FLAG_BITS


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
    for k in ("exclude_bad", "exclude_rule_based", "exclude_python_ua_normal"):
        out[k] = bool((o or {}).get(k, out[k]))
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


def select(t, conditions=None, options=None, overrides=None):
    """화면 선별용 표(5.5 의 <장비>_table.json) → (고른 줄 번호 목록, 조건 안의 줄 번호 목록, 뺀 이유별 수)."""
    cond, opts, ov = norm_conditions(conditions), norm_options(options), norm_overrides(overrides)
    every = not cond["dates"] and not cond["sources"]
    date_ok = [not cond["dates"] or d in cond["dates"] for d in t["dates"]]
    src_ok = [not cond["sources"] or s in cond["sources"] for s in t["sources"]]
    rule_ov = [{"exclude": 1, "include": 2}.get(ov["rules"].get(r), 0) for r in t["rules"]]
    type_ex = [ov["types"].get(x) == "exclude" for x in t["types"]]
    caps = [None, opts["A"], opts["B"], opts["C"]]
    ex_bad, ex_rule, ex_py = opts["exclude_bad"], opts["exclude_rule_based"], opts["exclude_python_ua_normal"]
    n_dup = n_ex = n_bad = n_rule = n_py = n_grade = n_cap = 0
    sel, cand, groups = [], [], {}
    R, L = t["rule"], t["llm"]
    # 표의 줄은 날짜·줄 순서 (5.5) → 줄 번호 순서가 곧 날짜·줄 순서
    for i, d, s, g, y, p, r, ty, k, fl in zip(range(t["n"]), t["date"], t["src"], t["g"], t["y"], t["pat"], R, t["type"],
                                              t["cons"], t["fl"]):
        if not every and not (date_ok[d] and src_ok[s]):
            continue
        cand.append(i)
        if g == 0:
            n_dup += 1
            continue
        rc = rule_ov[r]
        if rc == 1 or type_ex[ty]:
            n_ex += 1
        elif ex_bad and k == "4":
            n_bad += 1
        elif ex_rule and k == "2" and y:
            n_rule += 1
        elif ex_py and not y and fl & PY_UA:
            n_py += 1
        elif rc == 2 or caps[g] == "all":
            sel.append(i)
        elif caps[g] == 0:
            n_grade += 1
        else:
            key = (p << 3) | (g << 1) | y                # 등급·label·패턴
            lst = groups.get(key)
            if lst is None:
                groups[key] = [i]
            else:
                lst.append(i)
    for key, idx in groups.items():
        cap = caps[(key >> 1) & 3]
        if len(idx) <= cap:
            sel.extend(idx)
            continue
        # 룰을 돌아가며: 룰마다 (LLM 먼저, 날짜·줄 순) 줄을 세우고 0번째, 1번째 … 순서로 cap 개
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
        sel.extend(picked)
        n_cap += len(idx) - cap
    sel.sort()
    why = {"제거 (중복)": n_dup, "'불일치'": n_bad, "'정합 (룰 기준)' 공격 (보류)": n_rule,
           "python-requests UA 정상 (보류)": n_py, "룰·유형 빼기": n_ex, "등급 빼기": n_grade, "패턴당 상한 초과": n_cap}
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
    sel, cand, why = select(t, conditions, options, overrides)
    ov = norm_overrides(overrides)
    on = bytearray(t["n"])
    for i in sel:
        on[i] = 1
    hold, rows_of, G, Y, K = {}, {}, t["g"], t["y"], t["cons"]
    for kind in HOLD:                                   # [후보, 지금 선택(옵션을 켜면 빠질 수), 제거, A, B, C]
        idx = rows_of[kind] = hold_rows(t, cand, kind)
        h = hold[kind] = [len(idx), sum(on[i] for i in idx), 0, 0, 0, 0]
        for i in idx:
            h[2 + G[i]] += 1
    unverified = sum(1 for i in cand if Y[i] and K[i] == "0")    # 검증 전이라 정합성을 모르는 공격
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
            "view": view if view in HOLD else "",
            "options": norm_options(options), "conditions": norm_conditions(conditions), "overrides": ov}


def read_table(base_name, dev):
    p = C.compared_files(base_name, dev)["table"]
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── 버전 목록 (초안 저장·삭제, 기준 등록) ─────────────────────────────────

def save_draft(name, base_name, conditions, options, overrides, tables):
    """초안 저장 (tables = {장비: 표}, 저장할 때의 비교 스냅숏·선택 수를 함께 남김)."""
    C.check_name(name)
    v = C.load_versions()
    base = C.get_version(v, base_name)
    if base.get("status") != "final":
        raise ValueError("기준 버전은 확정된 버전이어야 합니다")
    cur = next((x for x in v["versions"] if x["name"] == name), None)
    if cur and cur.get("status") == "final":
        raise ValueError(f"'{name}' 은 이미 확정된 버전입니다 - 다른 이름을 쓰세요")
    if not tables:
        raise ValueError("비교 결과가 없습니다 - 먼저 비교를 실행하세요")
    snap, counts = {}, {}
    for dev, t in tables.items():
        if t["base"] != base_name:
            raise ValueError("비교 결과의 기준 버전이 다릅니다")
        pv = preview(t, conditions, options, overrides)
        snap[dev] = t["snapshot"]
        counts[dev] = {"cand": pv["cand"], "sel": pv["sel"], "sel_label": pv["sel_label"], "by_grade": pv["by_grade"]}
    item = {"name": name, "base": base_name, "status": "draft", "conditions": norm_conditions(conditions),
            "options": norm_options(options), "overrides": norm_overrides(overrides), "snapshot": snap,
            "draft_counts": counts, "created_at": (cur or {}).get("created_at") or C.now_iso(), "updated_at": C.now_iso()}
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
             "kind": x.get("kind", ""), "created_at": x.get("created_at", ""), "finalized_at": x.get("finalized_at", ""),
             "updated_at": x.get("updated_at", ""), "conditions": x.get("conditions"), "options": x.get("options"),
             "overrides": x.get("overrides"), "snapshot": x.get("snapshot"), "note": x.get("note", ""), "dev": {}}
        for dev in C.DEVICES:
            cnt = (x.get("counts") or {}).get(dev) or {}
            d = {"rows": cnt.get("rows"), "label": cnt.get("label"), "added": cnt.get("added"),
                 "added_label": cnt.get("added_label"), "by_grade": cnt.get("by_grade")}
            dc = (x.get("draft_counts") or {}).get(dev)
            if dc:
                d["draft"] = dc
            if x.get("status") == "final" and x.get("kind") != "external":
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


def finalize(name, ctx):
    """초안 → 확정: 기준 버전 + 선별한 샘플의 학습 파일·로그를 내보내고 다음 버전의 기준으로 쓸 수 있게 한다."""
    import pandas as pd
    F = _load("sup_features_v", "5_0b_features.py")
    v = C.load_versions()
    ver = C.get_version(v, name)
    if ver.get("status") != "draft":
        raise ValueError(f"'{name}' 은 초안이 아닙니다")
    base = C.get_version(v, ver["base"])
    devs = [d for d in C.DEVICES if d in (ver.get("snapshot") or {})]
    if not devs:
        raise ValueError("초안에 비교 결과가 없습니다")
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
    if out.exists():
        raise ValueError(f"{out} 가 이미 있습니다")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    manifest = {"name": name, "base": base["name"], "conditions": ver["conditions"], "options": ver["options"],
                "overrides": ver["overrides"], "snapshot": ver["snapshot"], "created_at": ver.get("created_at"),
                "finalized_at": None, "dev": {}}
    counts, files = {}, {}
    try:
        for dev in devs:
            ctx.check()
            ctx.step(f"{dev}:select", final_step_text(dev, "select"))
            ctx.sub(0, 3, "선별용 표 읽기")
            t = read_table(base["name"], dev)
            ctx.sub(1, 3, f"선별 ({t['n']:,}행)")
            sel, cand, why = select(t, ver["conditions"], ver["options"], ver["overrides"])
            ctx.sub(2, 3, "샘플별 등급 읽기 (rows.parquet)")
            R = pd.read_parquet(C.compared_files(base["name"], dev)["rows"])
            if len(R) != t["n"]:
                raise ValueError(f"{dev.upper()} 비교 결과 파일이 서로 맞지 않습니다 - 다시 비교하세요")
            A = R.iloc[sel].reset_index(drop=True)
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
            # 원본 로그 = 기준 원본 로그 + 추가 줄
            csv_p, log_p = C.version_files(v, base, dev)
            ctx.step(f"{dev}:log", final_step_text(dev, "log", len(raw), log_name=log_p.name))
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
                f.write(("\n".join(raw) + "\n").encode("utf-8") if raw else b"")
            # 학습 파일 = 기준 학습 파일(표준 열만) + 추가 샘플 (payload 는 학습 파일 형식, 표준모델 피처는 원본 코드와 같은 값)
            ctx.step(f"{dev}:csv", final_step_text(dev, "csv", len(A), csv_name=csv_p.name))
            ns = F.load_legacy(C.feature_code(v, dev))
            names = F.legacy_names(ns)
            head = list(pd.read_csv(csv_p, nrows=0).columns)
            std = ["method", "payload", "label"] + sorted(c for c in head if c.startswith(f"{dev}_payload_"))
            if sorted(std[3:]) != names:
                raise ValueError(f"{dev.upper()} 기준 학습 파일의 피처 열이 표준모델 피처 파일과 다릅니다")
            cp = tmp / f"{dev}_train.csv"
            what = "기준 학습 파일 복사" if head == std else "기준 학습 파일 (표준 열만) 쓰기"
            w = [csv_p.stat().st_size / 2 ** 20 / (150 if head == std else 15), 8, 1 + len(A) * 3e-4]   # 하위 단계 예상 비율
            if len(A):
                ctx.sub(0, 3, what, w)
            if head == std:                                 # 표준 학습 형식이면 그대로 복사
                _copy(csv_p, cp, ctx, "" if len(A) else what)
            else:                                           # 다른 열(proba 등)이 있으면 표준 열만
                first, done = True, 0
                total = ((base.get("counts") or {}).get(dev) or {}).get("rows") or 0
                for ch in pd.read_csv(csv_p, usecols=std, chunksize=50000, dtype=str, keep_default_na=False):
                    ch[std].to_csv(cp, mode="w" if first else "a", header=first, index=False, lineterminator="\n")
                    first = False
                    done += len(ch)
                    ctx.progress(done, max(done, total), what="" if len(A) else what)
            if len(A):
                ctx.sub(1, 3, "피처 캐시 읽기", w)
                fd = C.CACHE / "features" / dev / F.code_hash(C.feature_code(v, dev))
                fc = pd.concat([pd.read_parquet(p, columns=["pk"] + names) for p in sorted(fd.glob("part-*.parquet"))],
                               ignore_index=True).drop_duplicates("pk", keep="last").set_index("pk")
                miss = [pk for pk in A["pk"] if pk not in fc.index]
                if miss:                                    # 캐시가 지워졌으면 다시 계산
                    todo = {pk: p for pk, p in zip(A["pk"], payload) if pk in set(miss)}
                    extra = F.compute_chunk(dev, str(C.feature_code(v, dev)), list(todo.values()))[names]
                    extra.index = list(todo.keys())
                    fc = pd.concat([fc, extra])
                ctx.sub(2, 3, f"추가 샘플 {len(A):,}행 쓰기", w)
                X = fc.loc[A["pk"].to_numpy(), names].reset_index(drop=True)
                ints = set(ns["regex_dict_comb"]) | set(ns["regex_dict_count"])
                for c in names:
                    if c in ints:
                        X[c] = X[c].round().astype("int64")
                new = pd.DataFrame({"method": A["method"], "payload": [F.raw_to_payload(p) for p in payload],
                                    "label": A["label"]})
                new = pd.concat([new, X], axis=1)[std]
                with open(cp, "rb+") as f:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.write(b"\n")
                new.to_csv(cp, mode="a", header=False, index=False, lineterminator="\n")
            # 추가한 샘플 목록·이번에 비교한 샘플 지문
            ctx.step(f"{dev}:list", final_step_text(dev, "list"))
            ctx.sub(0, 2, f"추가 샘플 목록 ({dev}_added.csv)")
            add = A[["date", "line", "sk", "ck", "rule_id", "source", "method", "label", "attack_type", "origin", "cons",
                     "grade", "holdout", "new_pat", "wrong_rf", "wrong_lg"]].copy()
            add["payload"] = payload
            add.to_csv(tmp / f"{dev}_added.csv", index=False, encoding="utf-8-sig")
            ctx.sub(1, 2, f"비교한 샘플 지문 ({dev}_considered.txt)")
            ck_all = R["ck"].to_numpy()
            (tmp / f"{dev}_considered.txt").write_text("\n".join(ck_all[i] for i in cand) + "\n", encoding="utf-8")
            base_cnt = (base.get("counts") or {}).get(dev) or {}
            lab = A["label"].value_counts()
            gl = A.groupby(["grade", "label"]).size()
            counts[dev] = {
                "rows": int((base_cnt.get("rows") or 0) + len(A)),
                "label": {k: int((base_cnt.get("label") or {}).get(k, 0)) + int(lab.get(k, 0)) for k in ("anomalies", "normal")},
                "added": int(len(A)), "added_label": {k: int(lab.get(k, 0)) for k in ("anomalies", "normal")},
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
                                    "files": hashed}
        ctx.step("move", MOVE_TEXT)
        manifest["finalized_at"] = C.now_iso()
        C.write_json(tmp / "manifest.json", manifest)
        _rename_dir(tmp, out, ctx)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    v = C.load_versions()
    ver = C.get_version(v, name)
    ver.update(status="final", finalized_at=manifest["finalized_at"], counts=counts,
               files={dev: {"train_csv": str(out / f"{dev}_train.csv"), "raw_log": str(out / f"{dev}_raw.log")}
                      for dev in devs})
    C.save_versions(v)
    return {dev: {"added": counts[dev]["added"], "rows": counts[dev]["rows"]} for dev in devs}
