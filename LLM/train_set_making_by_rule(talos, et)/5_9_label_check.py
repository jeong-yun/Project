"""5.9 라벨 최종 점검 (피처 테스트 전, 2026-11-12~): 후보 버전의 학습 파일(기존 + 신규)에서 label 이 틀렸을 법한 행을
cleanlab(Confident Learning, 오픈소스)으로 찾는다. 유료 LLM 을 쓰지 않는다.

- 왜 피처 테스트 전인가: 피처 테스트의 4지표는 label 이 맞다고 보고 잰다. label 이 틀린 행이 섞이면 피처 탓인지 label 탓인지 가를 수 없다.
- 모델은 피처와 따로 둔다: payload 의 낱말·기호 1~2개 묶음(해싱) + 로지스틱(SGD). 점검하려는 피처 정의에 기대지 않는다.
- 예측 확률은 그 행을 학습에 쓰지 않은 모델에서만 받는다 (5묶음 교차). 같은 룰의 신규 샘플은 한 묶음에 넣는다
  (증강 샘플끼리 외우지 않게 - 그 룰을 한 번도 못 본 모델이 본다).
- 두 단계로 남긴다. label 품질 점수 = 주어진 label 의 예측 확률 (0~1, 낮을수록 의심).
  ① cleanlab 의심: label 마다 기준(그 label 행들의 평균 확신)을 넘게 '반대라고 확신하는 행' - 짧은 목록, 먼저 볼 것.
  ② 예측이 label 과 다름: 점수 0.5 아래 전부 - 넓은 목록, 점수 낮은 순으로 본다.
  (2026-11-12 시험, 표본 10만 행에서 신규 행 label 을 일부러 뒤집음: 공격 → 정상 300행 중 ① 255 · ② 293, 정상 → 공격 300행 중 ① 28 · ② 274.
   ① 만으로는 'label 은 공격인데 내용은 정상' 쪽을 많이 놓쳐 ② 를 함께 남긴다.)
- 결과는 '의심' 목록이다. 고치거나 빼는 것은 사람이 정한다 (화면 4 에서 sk 로 확인 → 5.7 룰 빼기 · 3.6 LLM 재확인).

  pip install cleanlab          (pandas · numpy · scikit-learn 은 5.x 와 같음)
  python 5_9_label_check.py --version "26 하반기" [--dev ips waf] [--sample 100000]

결과: 11_supervised/versions/<이름>/label_check/rev<n>/<장비>_issues.csv (의심 행) · <장비>_summary.json (수 · 유형 · 룰별)
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FOLDS, MAX_CHARS, SNIPPET = 5, 3000, 300
META = ["sk", "rule_id", "source", "attack_type", "grade", "cons", "origin", "date", "line", "label"]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("sup_common_lc", "5_0a_common.py")


def read_rows(name, dev):
    """학습 파일(기존 + 신규)과 신규 행 정보 → (payload · label 표, 신규 행 표, 기존 행 수). 신규 행은 학습 파일 끝에 added.csv 순서로 있다."""
    d = C.VERDIR / name
    T = pd.read_csv(d / f"{dev}_train.csv", usecols=["payload", "label"], dtype=str, keep_default_na=False)
    A = pd.read_csv(d / f"{dev}_added.csv", usecols=META, dtype=str, keep_default_na=False)
    nb = len(T) - len(A)
    if nb < 0 or not (T["label"].to_numpy()[nb:] == A["label"].to_numpy()).all():
        raise ValueError(f"{dev.upper()} 학습 파일 끝과 added.csv 의 label 이 다릅니다 - 다시 내보내세요")
    return T, A, nb


def fold_of(keys, k):
    """묶음 열쇠 → 0..k-1 (같은 열쇠는 늘 같은 묶음, 실행마다 같음)."""
    return (pd.util.hash_pandas_object(pd.Series(keys, dtype=object), index=False).to_numpy() % k).astype(np.int8)


def vectors(payload):
    """payload → 낱말·기호 1~2개 묶음의 해싱 벡터 (소문자, 앞 MAX_CHARS 글자)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer="word", token_pattern=r"[a-z0-9_]+|[^\sa-z0-9_]", ngram_range=(1, 2), n_features=2 ** 20,
                           alternate_sign=False, lowercase=False, binary=True, dtype=np.float32)
    return hv.transform(payload.str.lower().str.slice(0, MAX_CHARS))


def held_out_probs(X, y, fold):
    """행마다 '그 행을 빼고 학습한 모델'이 본 공격 확률."""
    from sklearn.linear_model import SGDClassifier
    p = np.zeros(len(y))
    for f in range(FOLDS):
        te = fold == f
        m = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, tol=1e-4, random_state=C.SEED).fit(X[~te], y[~te])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def m3(y, p):
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"accuracy": round(float((y == p).mean()), 4), "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(2 * pr * rc / max(1e-12, pr + rc), 4)}


def side(issue, differ, y, mask):
    """mask 안의 수: ① cleanlab 의심 · ② 예측이 label 과 다름, 각각 'label 은 공격인데 정상처럼 보임' · 'label 은 정상인데 공격처럼 보임'."""
    return {"rows": int(mask.sum()),
            "issues": int((issue & mask).sum()), "issues_attack_label": int((issue & mask & y).sum()), "issues_normal_label": int((issue & mask & ~y).sum()),
            "differ": int((differ & mask).sum()), "differ_attack_label": int((differ & mask & y).sum()), "differ_normal_label": int((differ & mask & ~y).sum())}


def table(col, issue_a, differ_a):
    """신규 행의 값별 (① 의심 수, ② 다른 수, 행 수) - ② 많은 순."""
    g = pd.DataFrame({"k": col.to_numpy(), "i": issue_a, "d": differ_a}).groupby("k").agg(issues=("i", "sum"), differ=("d", "sum"), rows=("d", "count"))
    g = g[g["differ"] + g["issues"] > 0].sort_values(["differ", "issues", "rows"], ascending=[False, False, True])
    return [{"key": k, "issues": int(r["issues"]), "differ": int(r["differ"]), "rows": int(r["rows"]), "differ_share": round(float(r["differ"] / r["rows"]), 4)}
            for k, r in g.iterrows()]


def check(name, dev, rev, sample=0):
    import cleanlab
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    t0 = time.time()
    T, A, nb = read_rows(name, dev)
    part = np.arange(len(T)) >= nb
    if sample and sample < len(T):                              # 시험용: 무작위 일부 행만
        keep = np.sort(np.random.default_rng(C.SEED).choice(len(T), sample, replace=False))
        T, part, A = T.iloc[keep].reset_index(drop=True), part[keep], A.iloc[keep[keep >= nb] - nb].reset_index(drop=True)
        nb = int((~part).sum())
    else:
        keep = np.arange(len(T))
    y = T["label"].str.strip().str.lower().ne("normal").to_numpy()
    rule = A["rule_id"].to_numpy()
    keys = np.concatenate([T["payload"].to_numpy()[:nb],                           # 기존 행: 같은 payload 끼리
                           np.where(rule != "", "rule:" + rule, T["payload"].to_numpy()[nb:])])   # 신규 행: 같은 룰끼리
    fold = fold_of(keys, FOLDS)
    X = vectors(T["payload"])
    print(f"{dev.upper()} {len(T):,}행 (기존 {nb:,} · 신규 {len(A):,}) · 벡터 행당 {X.nnz / X.shape[0]:.0f}개 [{time.time() - t0:.0f}초]", flush=True)
    p = held_out_probs(X, y, fold)
    del X
    pp = np.c_[1 - p, p]
    lab = y.astype(int)
    issue = np.zeros(len(y), bool)
    issue[find_label_issues(lab, pp, return_indices_ranked_by="self_confidence", n_jobs=1)] = True
    score = get_label_quality_scores(lab, pp)
    differ = (p >= 0.5) != y
    ia, da = issue[nb:], differ[nb:]
    rules = table(A["rule_id"], ia, da)
    out = {"version": name, "rev": rev, "dev": dev, "at": C.now_iso(), "sample": int(sample or 0),
           "tool": {"cleanlab": cleanlab.__version__, "method": "confident learning (prune_by_noise_rate)",
                    "model": "낱말·기호 1~2개 묶음 해싱 2^20 + SGD 로지스틱", "folds": FOLDS, "group": "신규 = 룰, 기존 = payload"},
           "rows": int(len(y)), "model_held_out": {"all": m3(y, p >= 0.5), "base": m3(y[~part], p[~part] >= 0.5), "added": m3(y[part], p[part] >= 0.5)},
           "counts": {"all": side(issue, differ, y, np.ones(len(y), bool)), "base": side(issue, differ, y, ~part), "added": side(issue, differ, y, part)},
           "added_by_grade": table(A["grade"], ia, da), "added_by_type": table(A["attack_type"], ia, da), "added_by_source": table(A["source"], ia, da),
           "added_by_cons": table(A["cons"].map(lambda c: C.CONS_TEXT.get(c, c)), ia, da),
           "rules_listed": len(rules), "rules_all_rows_differ": sum(1 for r in rules if r["differ"] == r["rows"]),
           "rules_top": rules[:50], "seconds": round(time.time() - t0)}
    d = C.VERDIR / name / "label_check" / f"rev{rev}"
    d.mkdir(parents=True, exist_ok=True)
    ix = np.flatnonzero(issue | differ)
    ix = ix[np.argsort(score[ix])]
    R = pd.DataFrame({"구분": np.where(part[ix], "신규", "기존"), "학습 파일 줄": keep[ix], "label": T["label"].to_numpy()[ix],
                      "예측": np.where(p[ix] >= 0.5, "anomalies", "normal"), "label 품질 점수": np.round(score[ix], 4),
                      "cleanlab 의심": np.where(issue[ix], "예", "")})
    meta = A.drop(columns=["label"]).reindex(ix - nb).reset_index(drop=True)          # 기존 행은 빈칸
    R = pd.concat([R, meta.fillna("")], axis=1)
    R["payload 앞부분"] = T["payload"].to_numpy()[ix]
    R["payload 앞부분"] = R["payload 앞부분"].str.slice(0, SNIPPET)
    R.to_csv(d / f"{dev}_issues.csv", index=False, encoding="utf-8-sig")
    C.write_json(d / f"{dev}_summary.json", out)
    s = out["counts"]
    print(f"  모델 (안 본 행 기준): 정확도 {out['model_held_out']['all']['accuracy']:.4f} · F1 {out['model_held_out']['all']['f1']:.4f}"
          f" (기존 {out['model_held_out']['base']['f1']:.4f} · 신규 {out['model_held_out']['added']['f1']:.4f})")
    for k, t in (("base", "기존"), ("added", "신규")):
        print(f"  {t} {s[k]['rows']:,}행: ① cleanlab 의심 {s[k]['issues']:,} (label 공격 {s[k]['issues_attack_label']:,} · label 정상 {s[k]['issues_normal_label']:,})"
              f" · ② 예측이 label 과 다름 {s[k]['differ']:,} ({100 * s[k]['differ'] / max(1, s[k]['rows']):.2f}%, label 공격 {s[k]['differ_attack_label']:,}"
              f" · label 정상 {s[k]['differ_normal_label']:,})")
    print("  신규 등급별 (① · ② / 행): " + " · ".join(f"{r['key']} {r['issues']:,} · {r['differ']:,} / {r['rows']:,}" for r in out["added_by_grade"]))
    print(f"  ② 가 있는 룰 {out['rules_listed']:,} (모든 행이 ② 인 룰 {out['rules_all_rows_differ']:,}) → {d} [{out['seconds']}초]", flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="라벨 최종 점검 (cleanlab) - 피처 테스트 전")
    ap.add_argument("--version", required=True, help="1차 확정한 버전 이름 (예: 26 하반기)")
    ap.add_argument("--dev", nargs="+", default=list(C.DEVICES), choices=C.DEVICES)
    ap.add_argument("--sample", type=int, default=0, help="시험용: 무작위 N행만")
    a = ap.parse_args()
    man = C.read_json(C.VERDIR / a.version / "manifest.json", None)
    if not man or not man.get("rev"):
        raise SystemExit(f"'{a.version}' 은 1차 확정(후보 rev)을 한 버전이 아닙니다")
    for dev in a.dev:
        check(a.version, dev, int(man["rev"]), a.sample)


if __name__ == "__main__":
    main()
