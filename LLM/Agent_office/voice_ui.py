"""🎤 노래 음색 변환 팀 · 🗣 일상 말 음성 변환 팀 화면 — app.py 가 부르는 부분만 (features: [voice_kit]). 만드는 일은 core/voice.py 가 한다.
두 팀이 같은 코드를 쓰고, 종류(kit·점검 숫자의 kind: song | speech)에 따라 글과 파일 이름만 바뀐다.

  check_form : '맡기기' 전에 양식 확인 (목소리 사용 확인 · 화자 이름 · zip 안전) — 문제가 있으면 TaskInputError
  preview    : 최종 결과 아래 — 🔍 데이터 점검 표(코드 계산) · 🎁 맞춤 실행 파일(zip) 내려받기 · AI 합성 음성 표시 안내
  tools_panel: 업무 양식 위 '🧰 모델 없이 바로 쓰는 도구' — 📈 학습 로그 진단 (코드 계산만)
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
            st.caption(("녹음" if stats.get("kind") == "speech" else "음원") + " 내용은 모델로 보내지 않았습니다 — 이 숫자 요약만 팀의 근거 자료가 됐습니다.")
    if kit:
        with st.container(border=True):
            st.markdown("##### 🎁 맞춤 실행 파일")
            speech = kit.get("kind") == "speech"
            work = ("말 녹음 자르기·학습·변환(음높이 자동 예측 -a)은 Colab 에서" if speech
                    else "보컬 분리·학습·변환은 내 PC 와 Colab 에서")
            st.caption(f"화자 `{kit['singer']}` · epochs {kit['epochs']} · batch_size {kit['batch_size']} · 전처리 {kit['mode']}안 — "
                       f"Colab 노트북·preprocess.py·run.bat·안내(README) 묶음입니다. {work} 직접 돌립니다.")
            if "voice_zip" not in run:  # 다시 그릴 때마다 만들지 않는다
                run["voice_zip"] = voice.kit_zip(kit)
            prefix = "speech" if speech else "svc"
            st.download_button("⬇ 맞춤 실행 파일 받기 (.zip)", data=run["voice_zip"], file_name=f"{prefix}_{kit['singer']}_{run['run_id'][:8]}.zip",
                               mime="application/zip", key=f"dl_voice_{run['run_id']}")
    if stats or kit:
        st.caption(f"🏷 변환한 결과물을 남에게 보이거나 올릴 때는 '{voice.AI_LABEL}'이라고 표시하세요.")


def tools_panel(team_id: str) -> None:
    """업무 양식 위 — 모델을 부르지 않고 코드만 쓰는 도구. 접혀 있고, 버튼을 누를 때만 계산한다 (다시 그릴 때는 결과만 보여 줌)."""
    with st.expander("🧰 모델 없이 바로 쓰는 도구 — 📈 학습 로그 진단"):
        _log_tool(team_id)


def _log_tool(team_id: str) -> None:
    key = f"voice_log_report_{team_id}"
    st.caption("Colab 학습(9번 셀) 화면의 진행 막대 줄이나, 10번 셀 TensorBoard 그래프에서 내려받은 CSV 를 붙여 넣으면 "
               "손실 값(loss) 추이를 코드로 계산해 다음에 할 일을 알려 드립니다. 모델을 부르지 않습니다.")
    with st.form(f"voice_log_form_{team_id}"):
        text = st.text_area("학습 로그 붙여 넣기", height=140, key=f"voice_log_text_{team_id}",
                            placeholder="Epoch 12/399 ━━━ 45/45 … loss/g/total: 45.123 … loss/d/total: 2.345\n"
                                        "또는 TensorBoard CSV — Wall time,Step,Value")
        file = st.file_uploader("또는 로그·CSV 파일 올리기", type=["txt", "log", "csv"], key=f"voice_log_file_{team_id}")
        go = st.form_submit_button("📈 진단하기")
    if go:
        raw = (text or "") + ("\n" + file.getvalue()[: voice.LOG_MAX_CHARS].decode("utf-8", "replace") if file else "")
        if raw.strip():
            st.session_state[key] = voice.log_report(voice.diagnose_log(raw))
        else:
            st.session_state.pop(key, None)
            st.warning("학습 로그를 붙여 넣거나 파일을 올려 주세요.")
    if st.session_state.get(key):
        st.markdown(st.session_state[key])
