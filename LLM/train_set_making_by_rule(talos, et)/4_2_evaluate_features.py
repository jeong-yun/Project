"""[4.2 피처 점검] 4.1 의 피처 표로 간단한 모델을 학습해, 피처 묶음별로 탐지 성능이 어떤지 비교한다.

[무엇을 보나]
  - 피처 묶음 조합별 (공격 흔적 / 공격 룰 / 개별 룰 / 조합) 정밀도·재현율·F1·ROC-AUC
  - 어떤 피처가 중요한지 (RandomForest 중요도 상위)
  - 룰 피처(r_)만으로 맞히는 비율 = 룰 그 자체의 성능 (모델이 룰을 넘어서는지 비교 기준)

[누수 방지]
  - 같은 payload(payload_hash) 는 한 번만 쓴다 (학습·평가에 같은 payload 가 들어가지 않게)
  - --group rule_id : 룰 단위로 나눈다 (평가 룰은 학습에 없음 → 처음 보는 룰에 대한 일반화)
  - --test 파일을 주면 --features 로 학습하고 --test 로 평가한다 (예: 기존 데이터로 학습 → LLM 생성 데이터로 평가)

[출력]  <features 폴더>/eval_<이름>.md, eval_<이름>.json

[사용법]
    python 4_2_evaluate_features.py --features 09_features/existing_ips
    python 4_2_evaluate_features.py --features 09_features/2026-10-16 --group rule_id
    python 4_2_evaluate_features.py --features 09_features/existing_ips --test 09_features/2026-10-16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
META = ["file", "line", "rule_id", "payload_hash", "label", "y"]
GROUPS = {"공격 흔적(k_)": ("k_",), "공격 룰(ra_)": ("ra_",), "개별 룰(r_)": ("r_",),
          "공격 흔적+공격 룰(k_ ra_)": ("k_", "ra_"), "전체(k_ ra_ r_)": ("k_", "ra_", "r_")}


def load(folder):
    p = Path(folder)
    p = p if p.is_absolute() else BASE_DIR / p
    df = pd.read_csv(p / "features.csv", dtype={"rule_id": str}, keep_default_na=False)
    return p, df.drop_duplicates("payload_hash").reset_index(drop=True)


def metrics(y, pred, prob):
    out = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="피처 묶음별 간단 모델 성능 비교")
    ap.add_argument("--features", required=True, help="4.1 출력 폴더 (학습용, --test 가 없으면 나눠서 평가)")
    ap.add_argument("--test", help="평가용 4.1 출력 폴더 (주면 --features 전체로 학습)")
    ap.add_argument("--group", help="이 열 기준으로 학습/평가를 나눔 (예: rule_id)")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    fdir, df = load(args.features)
    if args.test:
        tdir, test = load(args.test)
        train = df[~df["payload_hash"].isin(set(test["payload_hash"]))]
        how = f"학습 {fdir.name} → 평가 {tdir.name}"
        name = f"{fdir.name}_to_{tdir.name}"
    else:
        if args.group:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            tr, te = next(gss.split(df, df["y"], df[args.group]))
        else:
            tr, te = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=args.seed,
                                      stratify=df["y"] if df["y"].nunique() > 1 else None)
        train, test = df.iloc[tr], df.iloc[te]
        how = f"{fdir.name} 를 {'`' + args.group + '` 단위로 ' if args.group else ''}{int((1 - args.test_size) * 100)}:{int(args.test_size * 100)} 분할"
        name = fdir.name + (f"_by_{args.group}" if args.group else "")
    feats = [c for c in df.columns if c not in META and c in test.columns]

    results, report = {}, [f"# 피처 점검: {how}", "",
                           f"- 학습 {len(train)}줄 (anomalies {int(train.y.sum())}), 평가 {len(test)}줄 (anomalies {int(test.y.sum())}), "
                           f"같은 payload 는 한 번만", "",
                           "| 피처 묶음 | 피처 수 | 모델 | 정밀도 | 재현율 | F1 | ROC-AUC |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    if train.y.nunique() < 2:
        print("학습 데이터에 라벨이 한 종류뿐이라 학습할 수 없습니다")
        return 1
    for gname, prefixes in GROUPS.items():
        cols = [c for c in feats if c.startswith(prefixes)]
        if not cols:
            continue
        for mname, model in (("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                             ("RF", RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1,
                                                           class_weight="balanced"))):
            model.fit(train[cols], train.y)
            prob = model.predict_proba(test[cols])[:, 1]
            m = metrics(test.y, (prob >= 0.5).astype(int), prob)
            results[f"{gname} / {mname}"] = dict(m, n_features=len(cols))
            report.append(f"| {gname} | {len(cols)} | {mname} | {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} |")
            if gname.startswith("전체") and mname == "RF":
                imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:20]
                results["top_features"] = [(c, round(float(v), 4)) for c, v in imp]
    rule_all = [c for c in feats if c.startswith("r_") and c.endswith("_all")]
    if rule_all:
        report += ["", "## 룰 자체의 성능 (조건을 모두 만족 = 공격으로 판정, 평가 데이터 기준)", "",
                   "| 룰 피처 | 정밀도 | 재현율 | F1 |", "| --- | ---: | ---: | ---: |"]
        for c in rule_all:
            m = metrics(test.y, test[c].clip(lower=0), test[c].clip(lower=0))
            results[f"rule_only {c}"] = m
            report.append(f"| {c} | {m['precision']} | {m['recall']} | {m['f1']} |")
    if "top_features" in results:
        flist = json.loads((fdir / "feature_list.json").read_text(encoding="utf-8"))
        report += ["", "## 중요한 피처 상위 20 (전체 / RF)", "", "| 피처 | 중요도 | 묶음 | 설명 |", "| --- | ---: | --- | --- |"]
        for c, v in results["top_features"]:
            d = flist.get(c, {})
            desc = str(d.get("desc", "")).replace("|", "\\|")[:90]
            report.append(f"| {c} | {v} | {d.get('group', '')} | {desc} |")
    (fdir / f"eval_{name}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (fdir / f"eval_{name}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
