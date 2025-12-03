print("=========================================")
print("제작: 데이터개발팀")
print("=========================================", end="\n\n\n")

import pandas as pd
import numpy as np
import sys
import glob
import warnings
import os
import time

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)  # 컬럼 자동 줄바꿈 제거
pd.set_option("display.max_colwidth", None)  # 컬럼 내용 자르지 않음
pd.set_option("display.width", 10000)  # 한 줄 최대 길이 설정


# =========================
# 1. 데이터 로드 & 컬럼 체크
# =========================
def load_data():
    folder_path = "sql_data"

    if not os.path.exists(folder_path):
        print('Error: "sql_data" 폴더가 없습니다.')
        print("폴더 생성을 하고 CSV 파일을 넣으세요.")
        sys.exit(1)

    # 폴더 내부 CSV 파일 검색
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if len(csv_files) == 0:
        print("폴더에 CSV 파일이 없습니다. 파일 1개가 필요합니다.")
        sys.exit(1)

    if len(csv_files) > 1:
        print("CSV 파일이 2개 이상 존재합니다. 파일은 반드시 1개여야 합니다.")
        print("폴더 내 파일:", csv_files)
        sys.exit(1)

    # 파일 한 개만 존재할 경우
    csv_file = csv_files[0]
    print(f"CSV 파일 로딩: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        sys.exit(1)

    print("\n[데이터 확인] 상위 5행")
    print(df.head(5))
    return df


def check_prime_columns(df):
    prime_columns = [
        "수집 시간(mgr_time)",
        "탐지 시간(event_time)",
        "출발지 IP(s_ip)",
        "http_status(http_status)",
        "목적지 IP(d_ip)",
        "http_query(http_query)",
    ]

    print("\n[주요 컬럼 목록]")
    print(prime_columns)

    missing = [col for col in prime_columns if col not in df.columns]
    if missing:
        print("\n다음 주요 컬럼이 누락되어 있습니다. 프로그램을 종료합니다.")
        print("누락 컬럼:", missing)
        return False
    else:
        print("\n주요 컬럼이 모두 존재합니다.")
        return True


# =========================
# 2. 전처리 & 피처 생성
# =========================
def preprocess_and_feature(df):
    # 시간 datetime으로 포맷 통일
    df["수집 시간(mgr_time)"] = pd.to_datetime(
        df["수집 시간(mgr_time)"], errors="coerce"
    )
    df["탐지 시간(event_time)"] = pd.to_datetime(
        df["탐지 시간(event_time)"], errors="coerce"
    )

    # 피처1: 특수문자 존재 여부
    df["has_spchar"] = df["http_query(http_query)"].str.contains(
        r'<|"|/|\'|\@|\+|\*', case=False, na=False
    )
    # 피처2: 주요구문 존재 여부
    df["has_sql_kw"] = df["http_query(http_query)"].str.contains(
        r"select|from|union", case=False, na=False
    )
    # 피처3: 주요 함수 존재 여부
    df["has_sql_func"] = df["http_query(http_query)"].str.contains(
        r"eval|cast|declare", case=False, na=False
    )
    # 피처4: 연결 조건 존재 여부
    df["has_logic_op"] = df["http_query(http_query)"].str.contains(
        r"\bor\b|\band\b", case=False, na=False
    )

    return df


def is_httpstatus_err(s):
    s = s.astype(str)
    return s.str.startswith("5") | s.str.startswith("4")


# =========================
# 3. 집계
# =========================
def aggregate_by_sip(df):
    # s_ip 단위로 집계
    agg = (
        df.groupby("출발지 IP(s_ip)")
        .agg(
            req_cnt=("출발지 IP(s_ip)", "size"),  # 전체 요청 수
            distinct_url_cnt=("http_url(http_url)", "nunique"),  # 서로 다른 URL 수
            spchar_req_cnt=("has_spchar", "sum"),  # 특수문자 포함 요청 수
            sql_keyword_req_cnt=("has_sql_kw", "sum"),  # select/from/union 포함
            sql_func_req_cnt=("has_sql_func", "sum"),  # eval/cast/declare 포함
            logic_op_req_cnt=("has_logic_op", "sum"),  # or/and 포함
            err_httpstatus_cnt=(
                "http_status(http_status)",
                lambda s: is_httpstatus_err(s).sum(),
            ),  # 5xx 응답 수
        )
        .reset_index()
    )

    # s_ip별 비율 확인 (0으로 나누는 것 방지)
    agg["req_cnt"] = agg["req_cnt"].replace(0, np.nan)

    agg["spchar_rate"] = agg["spchar_req_cnt"] / agg["req_cnt"]  # 특수문자 비율
    agg["sql_keyword_rate"] = (
        agg["sql_keyword_req_cnt"] / agg["req_cnt"]
    )  # sql 주요 구문 비율
    agg["sql_func_rate"] = agg["sql_func_req_cnt"] / agg["req_cnt"]  # sql 함수 비율
    agg["logic_op_rate"] = agg["logic_op_req_cnt"] / agg["req_cnt"]  # 연결어 비율
    agg["err_httpstatus_cnt"] = agg["err_httpstatus_cnt"] / agg["req_cnt"]  # 5xx 비율

    print("s_ip 집계 예시(5)")
    print(agg.head())
    return agg


# =========================
# 4. 검색 기능
# =========================
def risk_setting(agg):
    # 위험 점수 구하기
    risk_score_str = input(
        "\n위험 가중치를 입력하세요."
        "(sql 주요 키워드, sql 함수, 특수문자, 연결자 순, 예 1, 1.5, 0.7, 0.5): "
    )

    try:
        parts = risk_score_str.split(",")
        sql_keyword_w, sql_func_w, spchar_w, logic_op_w = map(float, parts)
    except Exception:
        print("가중치 입력 형식이 잘못되었습니다. 기본값 (1,1,1,1)을 사용합니다.")
        sql_keyword_w, sql_func_w, spchar_w, logic_op_w = 1.0, 1.0, 1.0, 1.0

    agg["risk_score"] = (
        sql_keyword_w * agg["sql_keyword_rate"].fillna(0)
        + sql_func_w * agg["sql_func_rate"].fillna(0)
        + spchar_w * agg["spchar_rate"].fillna(0)
        + logic_op_w * agg["logic_op_rate"].fillna(0)
    ) * np.log10(agg["req_cnt"].fillna(0) + 1)

    # 장비에 접속한 모든 s_ip 확인
    print("\n[장비에 접속한 모든 출발지 IP(s_ip)]")
    print(agg["출발지 IP(s_ip)"].unique())

    # 상위 n개 출력
    agg_sorted = agg.sort_values("risk_score", ascending=False)

    top_n_str = input("\n상위 공격 위험 IP 수 입력(예: 5): ")
    try:
        top_n = int(top_n_str)
    except ValueError:
        print("숫자가 아닙니다. 기본값 5를 사용합니다.")
        top_n = 5

    top_sip = agg_sorted.head(top_n)

    print(f"\n[공격 위험 상위 {top_n} IP]")
    print(top_sip)


def search_by_sip(df):
    print("\n=== [1] 출발지 IP(s_ip)별 데이터 검색 ===")
    print("모든 출발지 IP(s_ip): ", df["출발지 IP(s_ip)"].unique())
    sip = input("검색할 출발지 IP(s_ip)를 입력하세요: ").strip()

    if not sip:
        print("IP가 입력되지 않았습니다.")
        return

    result = df[df["출발지 IP(s_ip)"] == sip]

    if result.empty:
        print(f"'{sip}' 에 해당하는 데이터가 없습니다.")
    else:
        print(f"\n'{sip}' 에 해당하는 데이터 {len(result)}건")
        print(
            result.loc[
                :,
                [
                    "출발지 IP(s_ip)",
                    "http_query(http_query)",
                    "http_status(http_status)",
                    "has_spchar",
                    "has_sql_kw",
                    "has_sql_func",
                    "has_logic_op",
                ],
            ]
        )


def search_by_column(df):
    print("\n=== [2] 특정 컬럼 검색 ===")
    print("컬럼 목록:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    col_input = input("\n검색할 컬럼명을 그대로 입력하세요: ").strip()

    if col_input not in df.columns:
        print("해당 컬럼이 존재하지 않습니다.")
        return

    keyword = input(f"컬럼 '{col_input}'에서 찾을 값(문자열)을 입력하세요: ").strip()
    if keyword == "":
        print("검색값이 비어 있습니다.")
        return

    # 문자열 포함 검색 (대소문자 무시)
    mask = df[col_input].astype(str).str.contains(keyword, case=False, na=False)
    result = df[mask]

    if result.empty:
        print(f"\n컬럼 '{col_input}'에서 '{keyword}' 를 포함하는 데이터가 없습니다.")
    else:
        print(
            f"\n컬럼 '{col_input}'에서 '{keyword}' 를 포함하는 데이터 {len(result)}건"
        )
        print(result)


def search_by_feature_exist(df):
    print("\n=== [3] 피처에 해당하는 데이터 검색 ===")
    print("피처 존재여부 컬럼 목록: ")
    print(
        "1(특수문자 존재 여부), 2(sql 주요 키워드 존재 여부), 3(sql 함수 존재 여부), 4(연결자 존재 여부)"
    )

    col_input = input("\n검색할 피처의 번호를 입력하세요: ").strip()

    if col_input == "1":
        fea_col = "has_spchar"
    elif col_input == "2":
        fea_col = "has_sql_kw"
    elif col_input == "3":
        fea_col = "has_sql_func"
    elif col_input == "4":
        fea_col = "has_logic_op"
    else:
        print("해당 번호가 존재하지 않습니다.")
        return

    if fea_col not in df.columns:
        print("해당 컬럼이 존재하지 않습니다.")
        return

    result = df[df[fea_col] == True]

    print(f"{col_input}({fea_col})에 대한 결과입니다.")
    print(
        result.loc[
            :,
            [
                "출발지 IP(s_ip)",
                "http_query(http_query)",
                "http_status(http_status)",
                fea_col,
            ],
        ]
    )


def search_menu(df, agg):
    while True:
        print("\n==============================")
        print(" 검색 메뉴")
        print("==============================")
        print("0: 가중치 입력을 통한 위험도 상위 출발지 IP 조회")
        print("1: 특정 출발지 IP의 원본 데이터 검색")
        print("2: 특정 컬럼에서 값 검색")
        print("3: 피처 조건에 해당하는 원본 데이터 조회")
        print("99: 종료")
        choice = input("번호를 선택하세요: ").strip()

        if choice == "0":
            risk_setting(agg)
        elif choice == "1":
            search_by_sip(df)
        elif choice == "2":
            search_by_column(df)
        elif choice == "3":
            search_by_feature_exist(df)
        elif choice == "99":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 입력해주세요.")


# =========================
# 5. main 함수
# =========================
def main():
    # 0) 시작 시간
    start = time.time()

    # 1) 데이터 로드
    df = load_data()

    # 2) 주요 컬럼 체크 (없으면 종료)
    if not check_prime_columns(df):
        sys.exit(1)

    # 3) 전처리 & 피처 생성
    df = preprocess_and_feature(df)

    # 4) 집계 및 위험 점수 계산
    agg = aggregate_by_sip(df)

    # 5) 검색 메뉴 실행
    search_menu(df, agg)

    # 6) 진행 시간 확인
    end = time.time()
    print(f"실행시간 {end - start:.5f} sec")
    print("*" * 30)

    os.system("pause")


if __name__ == "__main__":
    main()
