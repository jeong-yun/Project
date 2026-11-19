"""🎧 듣기용 MP3 (core/listen.py) — 대본(절 목록) · 코드 검사 · 가짜 TTS 로 MP3 만들기 · 저장·재사용 · 화면.

실제 모델·실제 TTS 는 부르지 않는다 (use_script · fake_tts — tests/conftest.py)."""

import pytest

from core import call_monitor, listen, usage
from core.music import audio_duration
from tests.fakes import Script, silent_mp3

ENTRY = """[Perf]플래시어텐션(FlashAttention) (작성일: 26.10.20)
1. 정의 및 개념
 - GPU 메모리 계층(HBM·SRAM)을 고려해 어텐션을 타일 단위로 계산하는 정확한 어텐션 알고리즘임.
2. 작동 원리
 - 타일링과 재계산으로 HBM 읽기·쓰기를 줄임. 시퀀스 길이 2048 에서 GPT-2 학습 3배 빠름.
3. 좋아진 점
 - 메모리 사용량이 시퀀스 길이에 선형.
참고문헌
 - [논문·핵심기법] Dao et al. FlashAttention (arXiv:2205.14135, 2022)"""


def section(heading: str, text: str) -> listen.ScriptSection:
    return listen.ScriptSection(heading=heading, text=text)


def script(*texts: str) -> listen.ListenScript:
    return listen.ListenScript(sections=[section(f"절{i}", t) for i, t in enumerate(texts, 1)])


GOOD = script(
    "플래시어텐션은 GPU 메모리 계층을 고려해 어텐션을 조각조각 나눠 계산하는 방법입니다. 결과는 원래 어텐션과 똑같습니다.",
    "책상 위 작은 메모장과 멀리 있는 창고를 떠올려 보세요. 창고까지 덜 오가게 하는 것이 핵심입니다.",
    "시퀀스 길이 2048 에서 GPT-2 학습이 3배 빨라졌습니다.",
    "정리하면, 같은 답을 내면서 메모리를 덜 오가게 만든 어텐션입니다.",
)


# ─── 코드 검사 ─────────────────────────────────────────────────────


def test_good_script_passes_checks():
    assert listen.check_script(GOOD, ENTRY) == []


def test_numbers_and_english_terms_not_in_entry_are_flagged():
    bad = script("플래시어텐션은 2019년에 Google 이 만든 방법으로 속도가 7.5배 빠릅니다.")
    problems = " / ".join(listen.check_script(bad, ENTRY))
    assert "2019" in problems and "7.5" in problems
    assert "Google" in problems
    assert "GPU" not in problems  # 본문에 있는 영문 용어는 통과


def test_script_over_length_is_flagged():
    long = script("가" * (listen.MAX_SCRIPT_CHARS + 1))
    assert any("글자" in p and str(listen.MAX_SCRIPT_CHARS) in p for p in listen.check_script(long, ENTRY))


def test_empty_script_is_flagged():
    assert listen.check_script(listen.ListenScript(sections=[]), ENTRY)


def test_estimate_uses_reading_speed_and_price():
    minutes, usd = listen.estimate(listen.CHARS_PER_MIN * 4)
    assert minutes == pytest.approx(4)
    assert usd == pytest.approx(4 * listen.COST_PER_MIN_USD)


# ─── 문장 경계로 나누기 ───────────────────────────────────────────────


def test_split_text_keeps_sentences_under_limit():
    text = " ".join(f"{i}번째 문장은 이만큼 깁니다." for i in range(40))
    parts = listen.split_text(text, limit=100)
    assert len(parts) > 1 and all(len(p) <= 100 for p in parts)
    assert all(p.endswith(".") for p in parts)  # 문장 중간에서 끊지 않는다
    assert " ".join(parts) == text


def test_split_text_cuts_a_sentence_longer_than_limit():
    parts = listen.split_text("아" * 250, limit=100)
    assert [len(p) for p in parts] == [100, 100, 50]


# ─── 대본 만들기 (보조 모델 1번, 걸리면 1번 다시) ───────────────────────


def test_make_script_calls_utility_model_once_when_clean(use_script, models):
    s = use_script(Script(ListenScript=GOOD))
    result = listen.make_script(ENTRY, models)
    assert result.script == GOOD and result.warnings == [] and result.calls == 1
    prompt = s.prompts("ListenScript")[0]
    assert "타일 단위" in prompt  # 본문을 보냈다
    assert "인사" in prompt and "약어" in prompt  # 읽기용 규칙


def test_make_script_rewrites_once_with_problems_then_warns(use_script, models):
    bad = script("2019년에 Google 이 만들었습니다.")
    s = use_script(Script(ListenScript=[bad, bad]))
    result = listen.make_script(ENTRY, models)
    assert result.calls == 2  # 다시 쓰기는 한 번만
    assert "2019" in s.prompts("ListenScript")[1]  # 걸린 점을 알려 주고 다시 쓰게 했다
    assert any("2019" in w for w in result.warnings)  # 그래도 남으면 경고로


def test_make_script_rewrite_fixes_problems(use_script, models):
    use_script(Script(ListenScript=[script("Google 이 만들었습니다."), GOOD]))
    result = listen.make_script(ENTRY, models)
    assert result.calls == 2 and result.warnings == [] and result.script == GOOD


def test_make_script_is_recorded_in_call_monitor(use_script, models):
    use_script(Script(ListenScript=GOOD))
    with usage.scope(run_id="listen-run-1", team_id="concept"):
        listen.make_script(ENTRY, models)
    recs = call_monitor.calls("listen-run-1")
    assert [r.kind for r in recs] == ["listen"]


# ─── 음성 (가짜 TTS) ─────────────────────────────────────────────────


def test_make_audio_joins_chunks_with_pauses(fake_tts, models, tmp_path, monkeypatch):
    monkeypatch.setattr(listen, "TTS_CHUNK_CHARS", 40)
    out = tmp_path / "a.mp3"
    result = listen.make_audio(GOOD, out, models)
    assert out.exists() and result.path == out
    assert all(len(t) <= 40 for t in fake_tts)
    assert len(fake_tts) == result.chunks > len(GOOD.sections)  # 긴 절은 문장 경계로 나눠 보냈다
    pauses = (len(GOOD.sections) - 1) * listen.SECTION_PAUSE_SEC + (result.chunks - len(GOOD.sections)) * listen.CHUNK_PAUSE_SEC
    audio, sr = listen._decode(silent_mp3())  # MP3 는 앞뒤에 인코더 여백이 붙어 0.3초보다 조금 길다
    assert audio_duration(out) == pytest.approx(result.chunks * audio.shape[1] / sr + pauses, abs=0.1)
    assert result.seconds == pytest.approx(audio_duration(out), abs=0.01)


def test_make_audio_records_monitor_and_usage(fake_tts, models, tmp_path):
    events = []
    usage.set_sink(events.append)
    with usage.scope(run_id="listen-run-2", team_id="concept"):
        listen.make_audio(GOOD, tmp_path / "a.mp3", models)
    recs = call_monitor.calls("listen-run-2")
    assert len(recs) == len(fake_tts) and all(r.kind == "listen" and r.status == "done" for r in recs)
    assert [r.chars for r in recs] == [len(t) for t in fake_tts]
    assert recs[0].model == listen.TTS_MODEL
    assert len(events) == len(fake_tts) and all(e.model == listen.TTS_MODEL and e.run_id == "listen-run-2" for e in events)


def test_make_audio_without_key_explains_next_step(models, tmp_path, no_real_tts):
    with pytest.raises(listen.ListenError, match="OPENAI_API_KEY"):
        listen.make_audio(GOOD, tmp_path / "a.mp3", models)
    assert no_real_tts == []  # 키가 없으면 부르지도 않는다


@pytest.mark.parametrize("error, words", [
    ("Error code: 401 - invalid_api_key", "키"),
    ("Error code: 429 - insufficient_quota", "크레딧"),
    ("Error code: 404 - model_not_found", "2027-01-06"),
    ("Connection error.", "연결"),
])
def test_tts_failure_becomes_plain_message(fake_tts, models, tmp_path, monkeypatch, error, words):
    def fail(text, api_key):
        raise RuntimeError(error)

    monkeypatch.setattr(listen, "tts_request", fail)
    with usage.scope(run_id="listen-run-3"):
        with pytest.raises(listen.ListenError) as e:
            listen.make_audio(GOOD, tmp_path / "a.mp3", models)
    assert words in str(e.value)
    assert not (tmp_path / "a.mp3").exists()
    assert call_monitor.calls("listen-run-3")[-1].status == "failed"


def test_real_tts_is_blocked_in_tests(no_real_tts):
    with pytest.raises(RuntimeError, match="실제 TTS 금지"):
        listen.tts_request("안녕하세요", "test-key")
    assert no_real_tts == ["안녕하세요"]
    no_real_tts.clear()  # 막히는 것을 본 테스트이므로 끝날 때의 실패 처리는 끈다


# ─── 저장·재사용 ────────────────────────────────────────────────────


def test_script_is_saved_and_reused_without_model_call(use_script, models, tmp_path):
    json_path, mp3_path = listen.paths(tmp_path, "run12345678", ENTRY)
    assert json_path.parent == tmp_path and mp3_path.suffix == ".mp3" and "run12345" in mp3_path.name
    listen.save_script(json_path, listen.ScriptResult(GOOD, ["경고"], 1))
    use_script(Script())  # 모델을 부르면 'no fake output' 오류
    loaded = listen.load_script(json_path)
    assert loaded.script == GOOD and loaded.warnings == ["경고"]


def test_paths_change_when_entry_changes(tmp_path):
    assert listen.paths(tmp_path, "r1", ENTRY) != listen.paths(tmp_path, "r1", ENTRY + " 수정")


def test_saved_audio_finds_latest_mp3_of_the_run(tmp_path):
    assert listen.saved_audio(tmp_path / "없음", "run12345678") is None
    _, mp3 = listen.paths(tmp_path, "run12345678", ENTRY)
    mp3.write_bytes(b"x")
    listen.paths(tmp_path, "other999", ENTRY)[1].write_bytes(b"y")
    assert listen.saved_audio(tmp_path, "run12345678") == mp3


# ─── 화면 (concept_ui.listen_panel) ──────────────────────────────────


def _panel(entry, folder, saved):
    import concept_ui
    from core.config import load_models

    concept_ui.listen_panel(entry, {"run_id": "run12345678"}, folder, load_models(), lambda run: saved.append(run["run_id"]))


def test_panel_before_script_shows_estimate_and_button(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_panel, kwargs={"entry": ENTRY, "folder": tmp_path, "saved": []}, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("듣기용 MP3 만들기" in b.label for b in at.button)
    assert any("추정" in c.value for c in at.caption)


def test_panel_makes_audio_on_click_only(fake_tts, use_script, tmp_path):
    from streamlit.testing.v1 import AppTest

    s = use_script(Script(ListenScript=GOOD))
    saved: list[str] = []
    at = AppTest.from_function(_panel, kwargs={"entry": ENTRY, "folder": tmp_path, "saved": saved}, default_timeout=30).run()
    assert fake_tts == [] and s.calls == []  # 그리기만으로는 부르지 않는다
    next(b for b in at.button if "듣기용 MP3 만들기" in b.label).click().run()
    assert not at.exception, at.exception
    assert len(s.calls) == 1 and fake_tts == []  # 1단계: 대본만 — 음성 전에 예상 요금을 보여 준다
    assert any("$" in c.value and "추정" in c.value for c in at.caption)
    next(b for b in at.button if "음성 만들기" in b.label).click().run()
    assert not at.exception, at.exception
    assert fake_tts and len(at.get("audio")) == 1
    assert any("MP3 받기" in b.label for b in at.get("download_button"))
    n_tts = len(fake_tts)
    at.run()  # 다시 그려도 TTS·모델을 또 부르지 않는다 (저장한 파일 재사용)
    assert len(fake_tts) == n_tts and len(s.calls) == 1
    assert saved  # ⏱ 호출 기록 저장을 불렀다
