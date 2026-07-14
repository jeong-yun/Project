"""
report.py
================================================================================
평가 결과 DataFrame을 다운로드 가능한 파일 바이트로 변환하는 모듈입니다.
english_word_maker/export_doc.py 의 Excel 생성 방식과 동일한 패턴을 사용합니다.
================================================================================
"""

from io import BytesIO

import pandas as pd


def get_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame을 CSV 바이트로 변환합니다.

    - 엑셀에서 한글이 깨지지 않도록 'utf-8-sig'(BOM 포함) 인코딩을 사용합니다.
    """
    # to_csv(index=False)로 인덱스 컬럼 없이 저장
    return df.to_csv(index=False).encode("utf-8-sig")


def get_excel_bytes(df: pd.DataFrame) -> BytesIO:
    """DataFrame을 Excel(xlsx) 바이트로 변환합니다.

    - engine="xlsxwriter" 는 requirements.txt 에 추가된 패키지입니다.
    - 반환값은 BytesIO 객체이며, Streamlit download_button 에 그대로 넘길 수 있습니다.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="SafetyEval")
    output.seek(0)
    return output
