"""음원 편집(core/audio_edit.py) 테스트 — 사인파 MP3 로 연산마다 결과를 측정해 확인한다 (네트워크·API 없이)."""

import numpy as np
import pytest
from pydantic import ValidationError

from core import audio_edit as ae
from core.music import audio_duration
from tests.fakes import Script

SR = 44100


def step(op, **kw):
    values = dict(op=op, start_sec=0, end_sec=0, seconds=0, amount=0, bass_db=0, treble_db=0, position="none")
    return ae.EditStep(**{**values, **kw})


def plan(*steps, regen=False, reason=""):
    return ae.EditPlan(analysis="", needs_regeneration=regen, regeneration_reason=reason, steps=list(steps), summary="요약")


def sine(seconds=6.0, freq=220.0, amp=0.2):
    t = np.arange(int(SR * seconds)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x])


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "take1.mp3"
    ae.save_audio(path, sine(), SR)
    return ae.start_history(path)


def dominant_freq(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio[0]))
    return float(np.fft.rfftfreq(audio.shape[1], 1 / SR)[spectrum.argmax()])


def apply(history, tmp_path, *steps):
    result = ae.apply_edit(history, plan(*steps), "테스트", tmp_path)
    assert result.version and not result.problems, result.problems
    audio, _ = ae.load_audio(result.version.path)
    return result, audio


# ─── 요청 거르기 · 계획 검사 ───────────────────────────────────────────


@pytest.mark.parametrize("request_text, expect", [
    ("가사를 좀 더 밝게 바꿔줘", "ACE-Step"),
    ("후렴 멜로디 수정해줘", "다시 생성"),
    ("보컬 크게 해줘", "Demucs"),
    ("반주만 남겨줘", "Demucs"),
    ("30초로 줄여", ""),
    ("키 한 음 낮춰", ""),
    ("끝에 페이드아웃", ""),
])
def test_requests_needing_regeneration_or_stems(request_text, expect):
    message = ae.request_needs_regeneration(request_text)
    assert (expect in message) if expect else message == ""


def test_unknown_op_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        step("remove_vocals")


@pytest.mark.parametrize("bad, problem", [
    (step("trim", start_sec=2, end_sec=30), "벗어남"),
    (step("trim", start_sec=3, end_sec=3.5), "1초보다 짧음"),
    (step("loop", seconds=9999), "목표 길이"),
    (step("fade", seconds=5, position="end"), "페이드"),
    (step("fade", seconds=1), "position"),
    (step("tempo", amount=0), "0 제외"),
    (step("tempo", amount=300), "-50~100"),
    (step("pitch_shift", amount=-24), "-12~12"),
    (step("gain", amount=40), "-30~12"),
    (step("reverb", amount=2), "0.01~1"),
    (step("normalize", amount=0), "-30~-5"),
    (step("eq"), "저음·고음"),
    (step("concat", start_sec=0, end_sec=2, amount=9), "횟수"),
])
def test_validate_plan_rejects_out_of_range(bad, problem):
    problems = ae.validate_plan(plan(bad), duration=6.0)
    assert problems and problem in problems[0], problems


def test_validate_plan_tracks_duration_through_steps():
    # trim 으로 3초가 된 뒤의 구간 2~5초는 범위를 벗어난다
    problems = ae.validate_plan(plan(step("trim", end_sec=3), step("concat", start_sec=2, end_sec=5)), 6.0)
    assert problems and "2번째 concat" in problems[0]
    assert ae.validate_plan(plan(*[step("gain", amount=-1)] * 9), 6.0)[0].startswith("편집 연산이 9개")
    assert ae.validate_plan(plan(), 6.0) == ["편집 연산이 없습니다"]
    assert ae.validate_plan(plan(regen=True), 6.0) == []
    ok = plan(step("trim", end_sec=4), step("fade", seconds=1, position="end"), step("tempo", amount=25))
    assert ae.validate_plan(ok, 6.0) == [] and ae.expected_duration(ok, 6.0) == pytest.approx(3.2)


# ─── 연산별 결과 (사인파 MP3 → 처리 → MP3 저장 → 다시 재기) ─────────────


def test_trim_and_fade(history, tmp_path):
    result, audio = apply(history, tmp_path, step("trim", start_sec=1, end_sec=4), step("fade", seconds=1, position="end"))
    assert result.version.duration == pytest.approx(3.0, abs=0.05)
    assert np.abs(audio[:, -2000:]).max() < 0.02 < np.abs(audio[:, SR: 2 * SR]).max()  # 끝은 조용, 가운데는 소리
    assert any("계획" in c and "일치" in c for c in result.checks)


def test_loop_to_target_length(history, tmp_path):
    result, _ = apply(history, tmp_path, step("loop", seconds=15))
    assert result.version.duration == pytest.approx(15.0, abs=0.05)


def test_concat_appends_segment(history, tmp_path):
    start = history.now.duration
    result, _ = apply(history, tmp_path, step("concat", start_sec=1, end_sec=3, amount=2))
    assert result.version.duration == pytest.approx(start + 4.0, abs=0.1)


def test_tempo_changes_length_keeps_pitch(history, tmp_path):
    start = history.now.duration
    result, audio = apply(history, tmp_path, step("tempo", amount=25))
    assert result.version.duration == pytest.approx(start / 1.25, abs=0.1)
    assert dominant_freq(audio) == pytest.approx(220, abs=5)


def test_pitch_shift_changes_pitch_keeps_length(history, tmp_path):
    start = history.now.duration
    result, audio = apply(history, tmp_path, step("pitch_shift", amount=12))
    assert result.version.duration == pytest.approx(start, abs=0.05)
    assert dominant_freq(audio) == pytest.approx(440, abs=8)  # 한 옥타브 위


def test_gain_lowers_loudness(history, tmp_path):
    before = history.now.lufs
    result, _ = apply(history, tmp_path, step("gain", amount=-6))
    assert result.version.lufs == pytest.approx(before - 6, abs=0.5)


def test_eq_bass_boost_raises_low_tone(tmp_path):
    path = tmp_path / "low.mp3"
    ae.save_audio(path, sine(freq=80), SR)
    history = ae.start_history(path)
    result, _ = apply(history, tmp_path, step("eq", bass_db=6))
    assert result.version.lufs > history.versions[0].lufs + 3


def test_reverb_adds_tail_same_length(history, tmp_path):
    original, _ = ae.load_audio(history.now.path)
    result, audio = apply(history, tmp_path, step("reverb", amount=0.3))
    assert result.version.duration == pytest.approx(history.versions[0].duration, abs=0.05)
    assert not np.allclose(audio[:, : original.shape[1]], original[:, : audio.shape[1]], atol=1e-3)


def test_compress_reduces_loud_part(tmp_path):
    loud = np.concatenate([sine(3, amp=0.05), sine(3, amp=0.9)], axis=1)
    path = tmp_path / "dyn.mp3"
    ae.save_audio(path, loud, SR)
    history = ae.start_history(path)
    _, audio = apply(history, tmp_path, step("compress", amount=-24))
    assert np.abs(audio[:, 4 * SR:]).max() < 0.8


def test_normalize_hits_target_lufs_without_clipping(history, tmp_path):
    result, audio = apply(history, tmp_path, step("normalize", amount=-12))
    assert result.version.lufs == pytest.approx(-12, abs=ae.LUFS_TOLERANCE)
    assert any("라우드니스" in c and "일치" in c for c in result.checks)
    assert np.abs(audio).max() <= 1.0


def test_peak_limit_keeps_under_ceiling():
    loud = sine(1, amp=1.5)
    limited = ae.peak_limit(loud, SR, ceiling_db=-1)
    assert np.abs(limited).max() <= 10 ** (-1 / 20) + 1e-4
    quiet = sine(1, amp=0.1)
    assert ae.peak_limit(quiet, SR) is quiet  # 넘치지 않으면 그대로


def test_clipping_is_prevented_after_gain(history, tmp_path):
    result, audio = apply(history, tmp_path, step("gain", amount=12))  # 0.2 → 0.8 (넘치지 않음)
    assert np.abs(audio).max() <= 1.0 and result.version.lufs > history.versions[0].lufs + 10


# ─── 버전 기록 · 되돌리기 · 화면 흐름 ────────────────────────────────────


def test_history_undo_redo_and_branch(history, tmp_path):
    apply(history, tmp_path, step("gain", amount=-3))
    apply(history, tmp_path, step("trim", end_sec=3))
    assert [v.label for v in history.versions] == ["v1", "v2", "v3"] and history.now.label == "v3"
    assert history.previous.label == "v2"
    assert history.undo() and history.undo() and not history.undo()
    assert history.now.label == "v1" and history.redo() and history.now.label == "v2"
    apply(history, tmp_path, step("fade", seconds=1, position="start"))  # v2 에서 새로 편집 → v3 는 버림
    assert [v.label for v in history.versions] == ["v1", "v2", "v4"] and history.current == 2


def test_invalid_or_regeneration_plan_makes_no_version(history, tmp_path):
    bad = ae.apply_edit(history, plan(step("trim", start_sec=0, end_sec=99)), "99초까지", tmp_path)
    assert bad.version is None and "벗어남" in bad.problems[0] and len(history.versions) == 1
    regen = ae.apply_edit(history, plan(regen=True, reason="가사를 바꾸는 요청입니다."), "가사 바꿔", tmp_path)
    assert regen.version is None and "가사를 바꾸는 요청" in regen.problems[0] and "ACE-Step" in regen.problems[0]


def test_plan_and_apply_uses_utility_model(history, tmp_path, use_script, models):
    script = use_script(Script(EditPlan=plan(step("trim", end_sec=2), step("fade", seconds=0.5, position="end"))))
    edit_plan, result = ae.plan_and_apply(history, "2초로 줄이고 끝에 페이드아웃", models, tmp_path)
    assert edit_plan.steps[0].op == "trim" and result.version.label == "v2"
    assert result.version.duration == pytest.approx(2.0, abs=0.05) and result.version.note == "2초로 줄이고 끝에 페이드아웃"
    prompt = script.prompts("EditPlan")[0]
    assert "trim, loop, fade" in prompt and "지금 곡 길이: 6.0초" in prompt


def test_plan_and_apply_short_circuits_regeneration(history, tmp_path, use_script, models):
    script = use_script(Script())
    edit_plan, result = ae.plan_and_apply(history, "가사 바꿔줘", models, tmp_path)
    assert edit_plan is None and result.version is None and "다시 생성" in result.problems[0]
    assert script.calls == []  # 규칙으로 걸러 모델을 부르지 않는다


def test_saved_mp3_duration_matches_mutagen(tmp_path):
    path = tmp_path / "x.mp3"
    ae.save_audio(path, sine(2.5), SR)
    assert audio_duration(path) == pytest.approx(2.5, abs=0.05)
