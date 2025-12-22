print("=========================================")
print("제작: 이글루코퍼레이션 데이터개발팀")
print("=========================================", end="\n\n\n")

import pandas as pd
import numpy as np
import sys
import glob
import warnings
import os
import time, datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)  # 컬럼 자동 줄바꿈 제거
pd.set_option("display.max_colwidth", None)  # 컬럼 내용 자르지 않음
pd.set_option("display.width", 10000)  # 한 줄 최대 길이 설정
os.system("mode con: cols=10000")  # 문자열 최대한 한 줄로 출력


# =========================
# 1. 데이터 로드 & 컬럼 체크
# =========================
def load_data():
    folder_path = "xss_data"

    if not os.path.exists(folder_path):
        print('Error: "xss_data" 폴더가 없습니다.')
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

    if "(detail)" in df.columns:
        df = df.drop(columns={"(detail)"})

    print("\n[데이터 확인] 상위 5행")
    print(df.head(5))
    return df


def check_prime_columns(df):
    prime_columns = [
        "수집 시간(mgr_time)",
        "탐지 시간(event_time)",
        "출발지 IP(s_ip)",
        "http_status(http_status)",
        "http_url(http_url)",
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
    df["has_xss_core_token"] = df["http_query(http_query)"].str.contains(
        r"alert|script|document", case=False, na=False
    )
    # 피처3: 주요 이벤트 존재 여부
    df["has_xss_event_attr"] = df["http_query(http_query)"].str.contains(
        r"onload>|onload=|onload%3d|onload%3e|onmouseover=|onmouseover%3d",
        case=False,
        na=False,
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
            distinct_url_cnt=(
                "http_url(http_url)",
                "nunique",
            ),  # 서로 다른 URL 수
            xss_spchar_cnt=("has_spchar", "sum"),  # 특수문자 포함 요청 수
            xss_core_token_cnt=(
                "has_xss_core_token",
                "sum",
            ),  # alert/script/document 포함
            xss_event_attr_cnt=("has_xss_event_attr", "sum"),  # onload/onmouseover 포함
            err_httpstatus_cnt=(
                "http_status(http_status)",
                lambda s: is_httpstatus_err(s).sum(),
            ),  # 5xx, 4xx 응답 수
        )
        .reset_index()
    )

    # s_ip별 비율 확인 (0으로 나누는 것 방지)
    agg["req_cnt"] = agg["req_cnt"].replace(0, np.nan)

    agg["xss_spchar_rate"] = agg["xss_spchar_cnt"] / agg["req_cnt"]  # 특수문자 비율
    agg["xss_core_token_rate"] = (
        agg["xss_core_token_cnt"] / agg["req_cnt"]
    )  # xss 주요 구문 비율
    agg["xss_event_attr_rate"] = (
        agg["xss_event_attr_cnt"] / agg["req_cnt"]
    )  # xss 주요 이벤트 비율
    agg["err_httpstatus_rate"] = (
        agg["err_httpstatus_cnt"] / agg["req_cnt"]
    )  # 4xx,5xx 비율

    print(
        "\n이상행위 탐지 피처: xss_spchar_cnt(특수문자 횟수 합), xss_core_token_cnt(xss 주요 구문 횟수 합), xss_event_attr_cnt(xss 주요 이벤트 횟수 합)"
    )
    print(
        "추가 조건: req_count(전체 요청), distinct_url_cnt(서로 다른 URL 수), err_httpstatus_cnt(비정상(4xx, 5xx) 응답 수 합)"
    )
    print(
        "추가 조건: xss_spchar_cnt(특수문자 비율), xss_core_token_cnt(xss 주요 구문 비율), xss_event_attr_rate(xss 주요 이벤트 비율), err_httpstatus_rate(비정상 응답 수 비율)"
    )
    print("\ns_ip 집계 예시(5)")
    print(agg.head(5))

    # 위험도 선정(모두 1, 표시 x)
    xss_core_token_w, xss_event_attr_w, xss_spchar_w = 1.0, 1.0, 1.0

    agg["risk_score"] = (
        xss_core_token_w * agg["xss_core_token_rate"].fillna(0)
        + xss_event_attr_w * agg["xss_event_attr_rate"].fillna(0)
        + xss_spchar_w * agg["xss_spchar_rate"].fillna(0)
    ) * np.log10(agg["req_cnt"].fillna(0) + 1)

    agg_sorted = agg.sort_values("risk_score", ascending=False)
    agg_sorted.reset_index(drop=True, inplace=True)
    top_25 = agg_sorted.head(25)
    return top_25


# =========================
# 4. 검색 기능
# =========================
def risk_setting(agg, flag):
    print("\n=== [1] 가중치 입력을 통한 위험도 상위 출발지 IP 조회 ===")
    print(
        "위험 가중치는 위험 점수(위험 가중치*피처의 비율의 합)를 구하는데 사용됩니다."
        "\n - 위험 가중치를 잘못 입력하면 모두 같은 가중치(1)을 같게 됩니다."
    )
    # 위험 점수 구하기
    risk_score_str = input(
        "\n위험 가중치를 입력하세요."
        "xss 주요 키워드(alert, script 등), xss 주요 이벤트(onload, onmouseover 등), 특수문자 순서(예 1, 1.5, 0.5): "
    )

    try:
        parts = risk_score_str.split(",")
        xss_core_token_w, xss_event_attr_w, xss_spchar_w = map(float, parts)
    except Exception:
        print("\n가중치 입력 형식이 잘못되었습니다. 기본값 (1,1,1,1)을 사용합니다.")
        xss_core_token_w, xss_event_attr_w, xss_spchar_w = 1.0, 1.0, 1.0

    agg["risk_score"] = (
        xss_core_token_w * agg["xss_core_token_rate"].fillna(0)
        + xss_event_attr_w * agg["xss_event_attr_rate"].fillna(0)
        + xss_spchar_w * agg["xss_spchar_rate"].fillna(0)
    ) * np.log10(agg["req_cnt"].fillna(0) + 1)

    # 장비에 접속한 모든 s_ip 확인
    sip_num = len(agg["출발지 IP(s_ip)"].unique())
    print(
        "\n장비에 접속한 모든 출발지 IP(s_ip)의 수는 ",
        sip_num,
        "개 입니다.",
    )

    # 상위 n개 출력
    agg_sorted = agg.sort_values("risk_score", ascending=False)

    top_n_str = input(
        "\n위험 점수를 사용해서 위험도가 높은 출발지 IP를 조회합니다."
        "\n - 숫자 외 값을 입력하면 기본값(5)로 적용됩니다. 최대 값은 25입니다."
        "\n상위 공격 위험 출발지 IP 수 입력(예: 5): "
    )
    try:
        top_n = int(top_n_str)
    except ValueError:
        print("숫자가 아닙니다. 기본값 5를 사용합니다.")
        top_n = 5

    if top_n > 25:
        top_n = 25

    if sip_num < top_n:
        print(
            f"출발지 IP가 {sip_num}개 이므로 상위 출발지 IP 수를 {top_n}에서 {sip_num}으로 변경합니다."
        )
        top_n = sip_num

    top_sip = agg_sorted.head(top_n)

    print(f"\n[공격 위험 상위 {top_n} IP]")
    print(top_sip)
    flag = True
    return top_sip, top_n, flag


def search_by_sip(df, agg, n, flag):
    print("\n=== [2] 출발지 IP(s_ip)별 데이터 검색 ===")
    print("특정 출발지 IP를 입력 → 해당 IP에서 발생한 모든 데이터를 조회")

    # 위험 점수 진행 또는 기본
    if flag == True:  # 진행
        print(
            "\n메뉴 1(가중치 입력을 통한 위험도 상위 출발지 IP 조회)을 진행했습니다."
            "\n - 해당 분석 결과의 s_ip 순위 제공합니다."
        )

        print(f"\n[가중치 기반 위험도 분석 결과] 상위 {n}개 출발지 IP(s_ip): ")
        print(agg["출발지 IP(s_ip)"].to_list())

    elif flag == False:
        print(
            "\n메뉴 1(가중치 입력을 통한 위험도 상위 출발지 IP 조회)을 진행하지 않았습니다."
            "\n - 기본 가중치를 적용한 위험점수를 바탕으로 상위 위험 출발지 IP(s_ip)를 출력합니다."
        )

        n = len(agg["출발지 IP(s_ip)"].unique())
        print(f"\n[기본 가중치 기반] 상위 {n}개 출발지 IP(s_ip): ")
        print(agg["출발지 IP(s_ip)"].to_list())

    sip = input("\n검색할 출발지 IP(s_ip)를 입력하세요: ").strip()

    if not sip:
        print("IP가 입력되지 않았습니다. 메뉴로 이동합니다.")
        return

    result = df[df["출발지 IP(s_ip)"] == sip]

    print(
        "\n이상행위 탐지 피처: has_spchar(특수문자 존재 여부), xss 주요 키워드(alert, script 등), xss 주요 이벤트(onload, onmouseover 등)"
    )

    if result.empty:
        print(f"\n'{sip}' 에 해당하는 데이터가 없습니다.")
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
                    "has_xss_event_attr",
                    "has_xss_core_token",
                ],
            ]
        )


def search_by_column(df):
    print("\n=== [3] 특정 컬럼 검색 ===")
    print(
        "특정 컬럼에서 피처 정보외 검색하고 싶은 문자열이 포함된 데이터를 검색할 수 있습니다."
        "\n - 일반 문자열, 정규식 검색 가능"
        "\n - 단 sql like 검색(예시 *) 불가"
    )
    print("\n컬럼 목록:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    col_input = input(
        "\n검색할 컬럼명을 그대로 입력하세요(원본 데이터에 같은 컬럼이 존재하지 않으면 메뉴로 이동합니다.): "
    ).strip()

    if col_input not in df.columns:
        print("해당 컬럼이 존재하지 않습니다. 메뉴로 이동합니다.")
        return df

    keyword = input(
        f"컬럼 '{col_input}'에서 찾을 값(문자열, 정규식)을 입력하세요(검색어를 입력하지 않으면 메뉴로 이동합니다.): "
    ).strip()
    if keyword == "":
        print("검색값이 비어 있습니다. 메뉴로 이동합니다.")
        return df

    # 새로운 컬럼명
    new_col = f"{col_input}_{keyword}"

    # 문자열 포함 검색 (대소문자 무시)
    try:
        mask = df[col_input].astype(str).str.contains(keyword, case=False, na=False)
    except Exception as e:
        print(f"[오류] 검색 도중 문제가 발생했습니다: {e}")
        return df

    # 원본 데이터에 검색 결과 컬럼 추가
    df[new_col] = mask

    result = df[mask]

    prime_columns = [
        "수집 시간(mgr_time)",
        "탐지 시간(event_time)",
        "출발지 IP(s_ip)",
        "http_status(http_status)",
        "http_query(http_query)",
        "http_url(http_url)",
    ]

    if col_input not in prime_columns:
        prime_columns.append(col_input)

    display_cols = prime_columns.copy()
    if new_col not in display_cols:
        display_cols.append(new_col)

    if result.empty:
        print(f"\n컬럼 '{col_input}'에서 '{keyword}' 를 포함하는 데이터가 없습니다.")
    else:
        print(
            f"\n컬럼 '{col_input}'에서 '{keyword}' 를 포함하는 데이터 {len(result)}건"
        )
        print(result.loc[:, display_cols])

    return df


def search_by_feature_exist(df):
    print("\n=== [4] 피처에 해당하는 데이터 조회 ===")
    print(
        "피처 번호 중 하나를 입력하면 → 해당 피처 조건을 만족하는 원본 데이터만 추출하여 보여줍니다."
    )
    print("피처 존재여부 컬럼 목록: ")
    print(
        "1(특수문자 존재 여부), 2(xss 주요 키워드(alert, script 등) 존재 여부), 3(xss 주요 이벤트(onload, onmouseover 등) 존재 여부)"
    )

    col_input = input(
        "\n검색할 피처의 번호를 입력하세요(번호를 잘못 입력하면 메뉴로 이동합니다.): "
    ).strip()

    if col_input == "1":
        fea_col = "has_spchar"
    elif col_input == "2":
        fea_col = "has_xss_core_token"
    elif col_input == "3":
        fea_col = "has_xss_event_attr"
    else:
        print("해당 번호가 존재하지 않습니다.")
        return

    if fea_col not in df.columns:
        print("해당 컬럼이 존재하지 않습니다.")
        return

    result = df[df[fea_col] == True]

    if result.empty:
        print("\n검색 결과가 없습니다.")
        return

    print(
        f"\n[{col_input}] 피처({fea_col})에 대한 그룹화 결과입니다.(전체 {len(result)}건)\n"
    )

    grouped = (
        result.groupby(
            [
                "출발지 IP(s_ip)",
                "http_query(http_query)",
                "http_status(http_status)",
                fea_col,
            ]
        )
        .size()
        .reset_index(name="count")
    )

    print(grouped)


def data_download(df, agg):
    print("\n=== [5] 지금까지의 결과 데이터 다운로드 ===")
    print(
        "프로그램 실행 중 수정된 원본 데이터와, 출발지 IP 단위 분석 결과를 하나의 Excel 파일로 저장"
    )
    print(
        "두 데이터를 각각 다른 시트에 저장하여, 사용자가 Excel에서 동시에 확인할 수 있도록 구성"
    )

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    file_name = f"xss_result_{now}.xlsx"

    try:
        with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="원본 데이터", index=False)
            agg.to_excel(writer, sheet_name="s_ip 그룹화 데이터", index=False)

        print("\n 파일 저장 완료")
        print(f" - 파일 저장 위치: {os.path.abspath(file_name)}")
    except Exception as e:
        print(f"[오류] 저장 중 문제가 발생했습니다: {e}")


def search_menu(df, agg):
    # 위험도 수행 확인 용
    flag = False
    n = 25
    while True:
        print("\n==============================")
        print(" 검색 메뉴")
        print("==============================")
        print("1: 가중치 입력을 통한 위험도 상위 출발지 IP 조회")
        print("2: 특정 출발지 IP의 원본 데이터 검색")
        print("3: 특정 컬럼에서 값 검색")
        print("4: 피처 조건에 해당하는 원본 데이터 조회")
        print("5: 지금까지의 결과 데이터 다운로드")
        print("99: 종료")
        choice = input("번호를 선택하세요: ").strip()

        if choice == "1":
            agg, n, flag = risk_setting(agg, flag)
        elif choice == "2":
            search_by_sip(df, agg, n, flag)
        elif choice == "3":
            df = search_by_column(df)
        elif choice == "4":
            search_by_feature_exist(df)
        elif choice == "5":
            data_download(df, agg)
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
