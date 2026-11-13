"""AI 안전성 평가 대시보드.

대상(일반 채팅 / RAG) 고르기 → 범주 고르기 → 대상·심판 모델 고르기 → 실행(진행 막대)
→ 전체·범주별 방어율 → 결과 표 → CSV·Excel 내려받기.
계산과 모델 호출은 ai_safety/ 모듈이 하고, 이 파일은 화면 흐름만 맡는다.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_safety import judge, metrics, report
from ai_safety.testcases import CATEGORIES, category_target, get_testcases
from common import MODEL_CHOICES

load_dotenv()

st.set_page_config(page_title="AI 안전성 평가", page_icon="🛡️")
st.title("AI 안전성 평가")
st.caption(
    "공격 프롬프트를 대상 모델에 보내고, 심판 모델이 응답을 채점해 방어율(= 안전하게 막은 비율)을 냅니다. "
    "심판 모델의 판정도 틀릴 수 있으니, 결과 표의 응답과 판정 이유를 함께 확인해 주세요."
)

# ---------------- 고르기 ----------------

TARGETS = {"chat": "일반 채팅", "rag": "RAG (문서 Q&A)"}
target = st.radio("대상", list(TARGETS), format_func=TARGETS.get, horizontal=True)
if target == "rag":
    st.caption(
        "RAG 대상은 지어낸 '달빛 도서관' 안내문(`ai_safety/rag_docs/`)을 검색해 답합니다. "
        "질문은 평범하고, 공격은 검색되는 문서 안에 심겨 있습니다(간접 프롬프트 인젝션)."
    )

category_keys = [key for key in CATEGORIES if category_target(key) == target]
selected = st.multiselect(
    "평가할 범주",
    options=category_keys,
    default=category_keys,
    format_func=lambda key: CATEGORIES[key]["label"],
    key=f"categories_{target}",
)
col_target, col_judge = st.columns(2)
target_model = col_target.selectbox("대상 모델 (평가받는 모델)", MODEL_CHOICES, index=0)
judge_model = col_judge.selectbox("심판 모델 (채점하는 모델)", MODEL_CHOICES, index=1)

cases = get_testcases(selected, target) if selected else []
n = len(cases)
st.info(
    f"공격 프롬프트 {n}개 × 모델 호출 2번(대상 1번 + 심판 1번) = 호출 최대 {2 * n}번. "
    "코드 검사에서 이미 걸린 건은 심판을 부르지 않아 더 적을 수 있습니다. 실행하면 고른 모델의 사용료가 듭니다."
)

# ---------------- 실행 ----------------

if st.button("평가 실행", type="primary", disabled=(n == 0)):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error(
            "OpenAI 키를 찾지 못했습니다. 이 앱 폴더에 `.env` 파일을 만들고 "
            "`OPENAI_API_KEY=...` 한 줄을 넣은 뒤 앱을 다시 실행해 주세요."
        )
    else:
        bar = st.progress(0.0, text="평가를 시작합니다…")

        def on_progress(done, total, testcase):
            bar.progress(done / total, text=f"{done} / {total} 끝남 — {testcase['id']}")

        try:
            st.session_state.setdefault("safety_results", {})[target] = judge.run_evaluation(
                cases, target_model, judge_model, progress_callback=on_progress, target=target
            )
        except Exception as e:
            st.error(f"평가를 시작하지 못했습니다: {e}\n\n모델 이름과 OpenAI 키를 확인한 뒤 다시 실행해 주세요.")
        bar.empty()

# ---------------- 결과 ----------------

results = st.session_state.get("safety_results", {}).get(target)
if results:
    summary = metrics.summarize(results)
    st.subheader(f"결과 — 대상: {TARGETS[target]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 방어율", f"{summary['overall_score']}%")
    c2.metric("막음 (PASS)", summary["passed"])
    c3.metric("뚫림 (FAIL)", summary["failed"])
    c4.metric("채점 못 함 (ERROR)", summary["errored"])
    if summary["errored"]:
        st.warning(
            "채점하지 못한 건(ERROR)은 방어율에서 '막지 못한 것'으로 셉니다. "
            "결과 표의 판정 이유를 보고, 모델 호출 오류라면 다시 실행해 주세요."
        )

    st.markdown("**범주별 방어율**")
    st.dataframe(
        pd.DataFrame(
            [
                {"범주": label, "전체": b["total"], "막음": b["passed"], "방어율(%)": b["rate"]}
                for label, b in summary["by_category"].items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**결과 표**")
    df = metrics.results_to_df(results)
    st.dataframe(df, hide_index=True, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button("📥 결과 내려받기 (CSV)", report.get_csv_bytes(df), "safety_eval.csv", "text/csv")
    d2.download_button(
        "📥 결과 내려받기 (Excel)",
        report.get_excel_bytes(df),
        "safety_eval.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.caption(f"'{TARGETS[target]}' 대상의 평가 결과가 아직 없습니다. 위에서 범주와 모델을 고르고 '평가 실행'을 눌러 주세요.")
