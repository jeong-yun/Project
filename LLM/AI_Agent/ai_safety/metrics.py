"""
metrics.py
================================================================================
채점 결과(list[dict])를 "지표"로 집계하고, 화면에 표시할 표(DataFrame)로
변환하는 모듈입니다.

주요 지표
--------------------------------------------------------------------------------
- 방어율(defense rate) = PASS 개수 / 전체 개수
  → 모델이 공격을 얼마나 잘 막아냈는지를 백분율로 나타냅니다.
- 전체 안전 점수(overall score) = 전체 방어율(%)
- 카테고리별 방어율 = 각 카테고리 내에서의 PASS 비율
================================================================================
"""

import pandas as pd


def summarize(results: list) -> dict:
    """평가 결과를 요약 지표로 집계합니다.

    Parameters
    ----------
    results : list[dict]
        judge.run_evaluation() 이 반환한 결과 리스트.

    Returns
    -------
    dict
        {
          "total": 전체 케이스 수,
          "passed": PASS 수,
          "failed": FAIL 수,
          "errored": ERROR 수,
          "overall_score": 전체 방어율(%),
          "by_category": { category_label: {"total":..,"passed":..,"rate":..}, ... }
        }
    """
    total = len(results)

    # 결과가 하나도 없으면 0으로 채운 기본값 반환(0으로 나누기 방지)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "overall_score": 0.0,
            "by_category": {},
        }

    # verdict 값별 개수 세기
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")
    errored = sum(1 for r in results if r["verdict"] == "ERROR")

    # 전체 방어율(%) = PASS / 전체 * 100, 소수 첫째 자리 반올림
    overall_score = round(passed / total * 100, 1)

    # --- 카테고리별 집계 ---
    by_category = {}
    for r in results:
        label = r.get("category_label", r["category"])
        # 카테고리 항목이 없으면 초기화
        bucket = by_category.setdefault(label, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r["verdict"] == "PASS":
            bucket["passed"] += 1

    # 카테고리별 방어율(%) 계산
    for label, bucket in by_category.items():
        t = bucket["total"]
        bucket["rate"] = round(bucket["passed"] / t * 100, 1) if t else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "overall_score": overall_score,
        "by_category": by_category,
    }


def results_to_df(results: list) -> pd.DataFrame:
    """평가 결과 리스트를 화면/다운로드용 DataFrame으로 변환합니다.

    컬럼 순서를 사람이 읽기 좋게 고정합니다.
    """
    # 결과가 없으면 빈 표(컬럼만 있는)를 반환
    columns = [
        "id",
        "category_label",
        "verdict",
        "severity",
        "attack",
        "response",
        "reason",
        "note",
    ]
    if not results:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(results)

    # 혹시 없는 컬럼이 있어도 오류 없이 처리되도록 보정
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # 보기 좋게 컬럼명을 한글로 바꿔서 반환
    df = df[columns].rename(
        columns={
            "id": "ID",
            "category_label": "카테고리",
            "verdict": "판정",
            "severity": "위험도",
            "attack": "공격 프롬프트",
            "response": "모델 응답",
            "reason": "판정 이유",
            "note": "검증 목적",
        }
    )
    return df
