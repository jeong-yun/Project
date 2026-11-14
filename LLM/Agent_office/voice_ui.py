"""🎙 음성 합성 팀 화면 — app.py 가 부르는 부분만 (features: [voice_kit]). 만드는 일은 core/voice.py 가 한다.

  check_form : '맡기기' 전에 양식 확인 (목소리 사용 확인 · 화자 이름 · zip 안전) — 문제가 있으면 TaskInputError
  preview    : 최종 결과 아래 — 🔍 데이터 점검 표(코드 계산) · 🎁 맞춤 실행 파일(zip) 내려받기 · AI 합성 음성 표시 안내
"""

import streamlit as st

from core import voice

FEATURE = "voice_kit"
check_form = voice.check_form


def tool_options(run: dict) -> dict:
    """이번 업무에서 voice_svc 툴이 계산한 값 (학습 설정 · 점검 숫자)."""
    return {k: v for t in run.get("tools") or [] if t.tool == "voice_svc" for k, v in t.harness_options.items()}


def preview(run: dict) -> None:
    opts = tool_options(run)
    stats, kit = opts.get("voice_dataset"), opts.get("voice_kit")
    if stats:
        with st.expander("🔍 데이터 점검 표 — 코드가 계산한 값 (LLM 이 아님)", expanded=True):
            st.markdown(voice.dataset_table(stats))
            st.caption("음원 내용은 모델로 보내지 않았습니다 — 이 숫자 요약만 팀의 근거 자료가 됐습니다.")
    if kit:
        with st.container(border=True):
            st.markdown("##### 🎁 맞춤 실행 파일")
            st.caption(f"화자 `{kit['singer']}` · epochs {kit['epochs']} · batch_size {kit['batch_size']} · 전처리 {kit['mode']}안 — "
                       f"Colab 노트북·preprocess.py·run.bat·안내(README) 묶음입니다. 보컬 분리·학습·변환은 내 PC 와 Colab 에서 직접 돌립니다.")
            if "voice_zip" not in run:  # 다시 그릴 때마다 만들지 않는다
                run["voice_zip"] = voice.kit_zip(kit)
            st.download_button("⬇ 맞춤 실행 파일 받기 (.zip)", data=run["voice_zip"], file_name=f"svc_{kit['singer']}_{run['run_id'][:8]}.zip",
                               mime="application/zip", key=f"dl_voice_{run['run_id']}")
    if stats or kit:
        st.caption(f"🏷 변환한 결과물을 남에게 보이거나 올릴 때는 '{voice.AI_LABEL}'이라고 표시하세요.")
