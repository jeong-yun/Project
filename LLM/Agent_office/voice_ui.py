"""🎤 노래 음색 변환 팀 · 🗣 일상 말 음성 변환 팀 화면 — app.py 가 부르는 부분만 (features: [voice_kit]). 만드는 일은 core/voice.py 가 한다.
두 팀이 같은 코드를 쓰고, 종류(kit·점검 숫자의 kind: song | speech)에 따라 글과 파일 이름만 바뀐다.

  check_form : '맡기기' 전에 양식 확인 (목소리 사용 확인 · 화자 이름 · zip 안전) — 문제가 있으면 TaskInputError
  preview    : 최종 결과 아래 — 🔍 데이터 점검 표(코드 계산) · 🎁 맞춤 실행 파일(zip) 내려받기 · AI 합성 음성 표시 안내
  tools_panel: 업무 양식 위 '🧰 모델 없이 바로 쓰는 도구' — 📈 학습 로그 진단 · ✂️ 자르기만 (코드 계산만)
"""

import tempfile
from pathlib import Path

import streamlit as st

from core import voice
from core.tasks import TaskInputError

FEATURE = "voice_kit"
check_form = voice.check_form


def tool_options(run: dict) -> dict:
    """이번 업무에서 voice_svc 툴이 계산한 값 (학습 설정 · 점검 숫자)."""
    return {k: v for t in run.get("tools") or [] if t.tool == "voice_svc" for k, v in t.harness_options.items()}


def kit_name(run: dict, kit: dict) -> str:
    """맞춤 묶음 파일 이름 — 업무 번호 앞 8자리를 넣어 지난 업무 창이 저장 파일을 찾는다 (saved_kit)."""
    return f"{'speech' if kit.get('kind') == 'speech' else 'svc'}_{kit['singer']}_{run['run_id'][:8]}.zip"


def kit_bytes(run: dict, kit: dict) -> bytes:
    if "voice_zip" not in run:  # 다시 그릴 때마다 만들지 않는다
        run["voice_zip"] = voice.kit_zip(kit)
    return run["voice_zip"]


def adopted_files(run: dict) -> dict[str, bytes] | None:
    """채택 시 results/<팀>/ 에 함께 저장할 맞춤 묶음 (노트북·안내·원본 스크립트만 — 녹음·데이터 zip 은 저장하지 않는다)."""
    kit = tool_options(run).get("voice_kit")
    return {kit_name(run, kit): kit_bytes(run, kit)} if kit else None


def saved_kit(results_dir: Path, team_id: str, run_id: str) -> Path | None:
    """채택 때 저장한 맞춤 묶음 — 파일 이름 끝의 업무 번호로 찾는다 (없으면 None: 채택 전 기능·학습 계획이 아닌 업무)."""
    folder = results_dir / team_id
    hits = sorted(folder.glob(f"*_{run_id[:8]}.zip")) if folder.is_dir() else []
    return hits[-1] if hits else None


def history_download(results_dir: Path, team_id: str, run_id: str) -> None:
    """🕘 지난 업무 창 — 저장한 맞춤 묶음 다시 받기."""
    path = saved_kit(results_dir, team_id, run_id)
    if path:
        st.download_button("⬇ 맞춤 실행 파일 다시 받기 (.zip)", data=path.read_bytes(), file_name=path.name,
                           mime="application/zip", key=f"hist_voice_{run_id}")


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
            st.download_button("⬇ 맞춤 실행 파일 받기 (.zip)", data=kit_bytes(run, kit), file_name=kit_name(run, kit),
                               mime="application/zip", key=f"dl_voice_{run['run_id']}")
            st.caption("💾 채택하면 이 묶음을 결과 폴더에 저장해, 앱을 다시 켜도 '🕘 지난 업무'에서 다시 받을 수 있습니다.")
    if stats or kit:
        st.caption(f"🏷 변환한 결과물을 남에게 보이거나 올릴 때는 '{voice.AI_LABEL}'이라고 표시하세요.")


def tools_panel(team) -> None:
    """업무 양식 위 — 모델을 부르지 않고 코드만 쓰는 도구. 접혀 있고, 버튼을 누를 때만 계산한다 (다시 그릴 때는 결과만 보여 줌)."""
    kind = voice.kind_of(team.tasks[0].fixed if team.tasks else {})
    with st.expander("🧰 모델 없이 바로 쓰는 도구 — 📈 학습 로그 진단 · ✂️ 자르기만"):
        log_tab, cut_tab = st.tabs(["📈 학습 로그 진단", "✂️ 자르기만 (보컬이 분리된 wav)"])
        with log_tab:
            _log_tool(team.id)
        with cut_tab:
            _slice_tool(team.id, kind)


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


def _slice_tool(team_id: str, kind: str) -> None:
    key = f"voice_slice_{team_id}"
    cfg = voice.slice_defaults()
    st.caption(f"보컬만 남은 wav(또는 그런 wav 를 묶은 zip)를 올리면 원본 preprocess.py 와 같은 기준(소리 크기 {cfg['threshold_db']:g}dB · "
               f"{cfg['min_keep']:g}초 미만 버림 · {cfg['max_len']:g}초씩 나눔)으로 잘라 `dataset_raw/<화자>/` 구조의 zip 으로 드립니다. "
               "보컬 분리는 하지 않습니다 — " + ("배경 음악이 섞인 녹음은" if kind == "speech" else "반주가 섞인 곡은")
               + " 먼저 preprocess.py(A안)나 Colab(B안)으로 분리하세요. 올린 파일은 자르는 동안만 임시 폴더에 두고 지웁니다.")
    with st.form(f"voice_slice_form_{team_id}"):
        c1, c2 = st.columns(2)
        singer = c1.text_input("화자 이름 (영문·숫자·-_)", placeholder="예: myvoice", key=f"voice_slice_singer_{team_id}")
        source = c2.selectbox("목소리 출처", voice.SOURCES, key=f"voice_slice_source_{team_id}")
        consent = st.checkbox("이 목소리를 학습·변환에 써도 된다는 것을 확인했습니다", key=f"voice_slice_consent_{team_id}")
        upload = st.file_uploader("보컬만 남은 wav 또는 wav 묶음 zip", type=["wav", "zip"], key=f"voice_slice_file_{team_id}")
        go = st.form_submit_button("✂️ 자르기")
    if go:
        singer = (singer or "").strip()
        try:
            if not singer:
                raise TaskInputError("화자 이름을 적어 주세요 — `dataset_raw/<화자 이름>/` 폴더 이름이 됩니다. 예: myvoice")
            voice.check_form("plan", {"source": source, "consent": consent, "singer": singer})  # 출처·확인·이름 — 학습 계획과 같은 확인
            if upload is None:
                raise TaskInputError("자를 wav 또는 zip 파일을 올려 주세요.")
            with st.spinner("자르는 중…"), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / ("upload" + Path(upload.name).suffix.lower())
                src.write_bytes(upload.getvalue())
                data, summary = voice.slice_upload(src, upload.name, singer)
                out = Path(tmp) / "dataset_raw.zip"
                out.write_bytes(data)
                table = voice.dataset_table(voice.inspect_zip(out, "dataset_raw.zip", kind))  # 잘라 낸 조각을 데이터 점검과 같은 계산으로
            st.session_state[key] = {"zip": data, "summary": summary, "table": table}
        except (TaskInputError, voice.VoiceError) as e:
            st.session_state.pop(key, None)
            st.warning(str(e))
    res = st.session_state.get(key)
    if not res:
        return
    s = res["summary"]
    st.success(f"조각 {s['clips']:,}개 · {s['minutes']:.1f}분을 만들었습니다 (wav {len(s['files'])}개를 읽음"
               + (f" · wav 가 아니어서 건너뜀 {s['ignored']}개" if s["ignored"] else "")
               + (f" · 읽지 못했거나 너무 큰 wav {s['skipped']}개" if s["skipped"] else "") + ").")
    st.markdown("\n".join(["| 올린 파일 | 길이 | 조각 수 | 남긴 길이 |", "|---|---|---|---|",
                           *(f"| {f['name']} | {f['sec']:.1f}초 | {f['clips']}개 | {f['kept_sec']:.1f}초 |" for f in s["files"])]))
    with st.expander("🔍 잘라 낸 조각 점검 — '🔍 데이터 점검'과 같은 계산 (코드)"):
        st.markdown(res["table"])
    b1, b2 = st.columns([3, 1])
    b1.download_button("⬇ dataset_raw.zip 받기", data=res["zip"], file_name="dataset_raw.zip", mime="application/zip",
                       key=f"voice_slice_dl_{team_id}", use_container_width=True)
    if b2.button("✖ 지우기", key=f"voice_slice_clear_{team_id}", use_container_width=True):
        st.session_state.pop(key, None)
        st.rerun()
    st.caption("받은 `dataset_raw.zip` 을 Google Drive `MyDrive/svc_project/` 에 올리고, 노트북 1~3번 셀 → 4번 셀(A안 — zip 풀기)을 실행한 뒤 "
               "5~6번 셀(B안)" + (" · '4-C' 셀(C안)" if kind == "speech" else "") + "은 건너뛰고 7번 셀(특징 추출)부터 이어 갑니다. "
               f"변환한 결과물을 남에게 보이거나 올릴 때는 '{voice.AI_LABEL}'이라고 표시하세요.")
