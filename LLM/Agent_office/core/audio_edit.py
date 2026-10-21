"""음악 팀 편집 단계 — 사람의 편집 요청 → 편집 계획(JSON) → CPU 오디오 처리 → 측정으로 검증 → 버전 기록.

Lyria 는 한 번에 만들기만 하고 고칠 수 없으므로(single-turn), 만든 곡의 편집은 모두 후처리로 한다.
  1) 요청 해석   : 보조(utility) 모델이 구조화 출력으로 편집 계획(EditPlan)을 만든다 — 연산은 정해진 목록(OPS)에서만
                  가사·멜로디·장르처럼 후처리로 안 되는 요청은 '다시 생성 필요'로 돌려보낸다 (규칙 + 모델 판단)
  2) 계획 검사   : validate_plan — 연산별 값 범위·구간이 곡 길이 안인지 코드로 확인 (잘못되면 실행하지 않음)
  3) 실행       : numpy(자르기·반복·페이드·이어 붙이기·음량), pedalboard(템포·키 = 내장 Rubber Band, EQ·리버브·컴프레서·리미터,
                  MP3 읽기/쓰기 = 내장 LAME), pyloudnorm(라우드니스 LUFS) — 모두 CPU, ffmpeg 불필요
  4) 검증       : 결과 파일의 길이(mutagen)와 라우드니스를 다시 재서 계획에서 계산한 기대값과 비교
  5) 버전 기록   : v1(원본) → v2 → … 되돌리기(↩)·다시 하기가 가능하다. 되돌린 뒤 새로 편집하면 그 뒤 버전은 버린다

pydub·pyrubberband 는 ffmpeg·rubberband 실행 파일이 필요해 쓰지 않는다 (이 PC 에 없음 — pedalboard 가 같은 일을 내장으로 한다).
보컬/반주 분리(Demucs)는 torch·torchaudio(+ffmpeg)가 필요해 아직 넣지 않았다 → README 7-x 의 '나중 단계' 참고.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory
from core.music import audio_duration, fmt_time
from core.schemas import ModelsConfig

OPS = ("trim", "loop", "fade", "concat", "tempo", "pitch_shift", "gain", "eq", "reverb", "compress", "normalize")
MAX_STEPS = 8
MAX_SECONDS = 600  # 편집 결과 최대 길이 (10분)
DECLICK_SEC = 0.005  # 이어 붙이는 곳의 '틱' 소리를 없애는 아주 짧은 페이드
TRUE_PEAK_LIMIT_DB = -1.0
LUFS_TOLERANCE = 1.5
DURATION_TOLERANCE = 0.15  # 초 — MP3 프레임 단위 오차

# 후처리로는 안 되는 요청 (규칙) — 모델 판단과 함께 쓴다
REGEN_PATTERN = re.compile(r"(가사|멜로디|선율|코드\s*진행|장르|편곡|화음|보컬\s*톤|목소리|악기\s*구성|분위기).{0,10}(바꿔|바꾸|수정|고쳐|변경|다르게|새로|추가|넣어)")
STEMS_PATTERN = re.compile(r"(보컬|목소리|반주|드럼|베이스).{0,6}(크게|작게|키워|줄여|빼|없애|만\s*남|분리)|반주만|보컬만|\bMR\b|inst만")


class EditStep(BaseModel):
    """편집 연산 하나. 쓰지 않는 값은 0 (OpenAI strict 스키마 호환을 위해 모든 필드 필수)."""

    op: Literal["trim", "loop", "fade", "concat", "tempo", "pitch_shift", "gain", "eq", "reverb", "compress", "normalize"] = Field(
        description="trim=구간만 남기기, loop=반복해 늘리기, fade=페이드, concat=구간을 끝에 이어 붙이기, tempo=빠르기(음높이 유지), "
        "pitch_shift=키(반음), gain=음량(dB), eq=저음·고음, reverb=울림, compress=소리 크기 고르게, normalize=라우드니스 맞추기"
    )
    start_sec: float = Field(description="trim·loop·concat: 구간 시작(초). 그 외 0")
    end_sec: float = Field(description="trim·loop·concat: 구간 끝(초). 0 이면 곡 끝까지. 그 외 0")
    seconds: float = Field(description="fade: 페이드 길이(초), loop: 반복해서 만들 전체 길이(초). 그 외 0")
    amount: float = Field(
        description="tempo: 빠르기 변화 %(+10 = 10% 빠르게), pitch_shift: 반음 수(한 음 = 2, 반음 = 1, 낮추면 음수), "
        "gain: dB, reverb: 울림 양 0~1(조금 = 0.15), compress: 문턱 dB(-30~-10), normalize: 목표 LUFS(보통 -14), "
        "concat: 이어 붙일 횟수(1~4). 그 외 0"
    )
    bass_db: float = Field(description="eq: 저음(200Hz 이하) 증감 dB. 그 외 0")
    treble_db: float = Field(description="eq: 고음(4kHz 이상) 증감 dB. 그 외 0")
    position: Literal["start", "end", "both", "none"] = Field(description="fade: start(페이드인)/end(페이드아웃)/both. 그 외 none")


class EditPlan(BaseModel):
    analysis: str = Field(description="요청을 어떤 연산으로 바꿀지 먼저 생각한 내용 (한국어)")
    needs_regeneration: bool = Field(description="가사·멜로디·장르·악기 구성·보컬 교체처럼 후처리로 할 수 없으면 true")
    regeneration_reason: str = Field(description="needs_regeneration 이 true 면 이유, 아니면 빈 문자열")
    steps: list[EditStep] = Field(description="순서대로 실행할 편집 연산 (최대 8개). 다시 생성이 필요하면 빈 목록")
    summary: str = Field(description="사람에게 보여 줄 한 줄 요약 (한국어)")


# ─── 1) 요청 해석 ─────────────────────────────────────────────────────


def request_needs_regeneration(request: str) -> str:
    """후처리로 안 되는 요청이면 사람에게 보여 줄 설명, 아니면 빈 문자열 (규칙 기반 — 모델을 부르기 전에 거른다)."""
    if STEMS_PATTERN.search(request):
        return ("보컬·반주를 따로 키우거나 빼려면 음원을 악기별로 나누는 스템 분리(Demucs)가 필요합니다. "
                "이 PC 에는 아직 설치되어 있지 않습니다 (torch·ffmpeg 필요 — README 7-x '나중 단계'). "
                "지금은 'eq 로 저음/고음 조절'이나 Lyria 로 반주만 다시 만들기를 쓸 수 있습니다.")
    if REGEN_PATTERN.search(request):
        return regeneration_message("가사·멜로디·장르·악기 구성은 음원 안에 섞여 있어 후처리로 바꿀 수 없습니다.")
    return ""


def regeneration_message(reason: str) -> str:
    return (f"{reason} 이 요청은 곡을 다시 생성해야 합니다 — 위의 '🔁 Lyria 로 다시 만들기'(유료, 결과가 매번 다름)를 쓰거나, "
            "가사·멜로디 일부만 바꾸는 편집은 GPU 생성 모델(예: ACE-Step 1.5)이 필요한데 이 PC(내장 그래픽)에서는 쓸 수 없습니다.")


def plan_edit(request: str, duration: float, models: ModelsConfig) -> tuple[EditPlan, list[str]]:
    """보조 모델이 편집 요청을 편집 계획으로 바꾼다 (구조화 출력)."""
    messages = [
        SystemMessage("\n".join([
            "너는 음원 편집 엔지니어다. 사람의 편집 요청을 아래 연산만 써서 순서대로 실행할 편집 계획으로 바꾼다.",
            f"- 쓸 수 있는 연산: {', '.join(OPS)} (그 밖의 일은 할 수 없다)",
            f"- 지금 곡 길이: {duration:.1f}초 ({fmt_time(duration)}). 구간은 이 길이 안에서 정한다.",
            "- '30초로 줄여' → trim(0~30) 뒤 fade end 2초. '2분이 되게' → loop(seconds=120) 뒤 fade end.",
            "- '키 한 음 낮춰' → pitch_shift amount=-2 (한 음 = 2반음, 반음 = 1). '템포 10% 빠르게' → tempo amount=10.",
            "- '리버브 조금' → reverb amount 0.15. '음량 맞춰'·'소리 크기 맞춰' → normalize amount=-14.",
            "- 가사·멜로디·장르·악기 구성·보컬 교체·보컬만 키우기는 할 수 없다 → needs_regeneration=true, steps 는 빈 목록.",
            "- 쓰지 않는 값은 0, position 은 fade 가 아니면 none.",
        ])),
        HumanMessage(request.strip()[:1000]),
    ]
    call = llm_factory.invoke_structured(models.utility, models, EditPlan, messages, "음원 편집 계획")
    return call.parsed, call.errors


# ─── 2) 계획 검사 ─────────────────────────────────────────────────────


def _segment(step: EditStep, duration: float) -> tuple[float, float]:
    end = step.end_sec if step.end_sec > 0 else duration
    return step.start_sec, min(end, duration)


def next_duration(step: EditStep, duration: float) -> float:
    """이 연산을 하고 난 뒤의 길이 (검사·검증의 기대값)."""
    if step.op == "trim":
        start, end = _segment(step, duration)
        return end - start
    if step.op == "loop":
        return step.seconds
    if step.op == "concat":
        start, end = _segment(step, duration)
        return duration + (end - start) * max(1, int(step.amount or 1))
    if step.op == "tempo":
        return duration / (1 + step.amount / 100)
    return duration


def expected_duration(plan: EditPlan, duration: float) -> float:
    for step in plan.steps:
        duration = next_duration(step, duration)
    return duration


def validate_plan(plan: EditPlan, duration: float, max_steps: int = MAX_STEPS) -> list[str]:
    """실행 전에 연산별 값 범위·구간을 확인한다. 문제 목록이 비어 있어야 실행한다."""
    if plan.needs_regeneration:
        return []
    if not plan.steps:
        return ["편집 연산이 없습니다"]
    if len(plan.steps) > max_steps:
        return [f"편집 연산이 {len(plan.steps)}개로 너무 많습니다 (최대 {max_steps}개)"]
    problems: list[str] = []
    for i, s in enumerate(plan.steps, 1):
        where = f"{i}번째 {s.op}"
        if s.op not in OPS:
            problems.append(f"{where}: 허용되지 않은 연산")
            continue
        if s.op in ("trim", "loop", "concat"):
            start, end = _segment(s, duration)
            if not (0 <= start < end) or s.end_sec > duration + 0.5:
                problems.append(f"{where}: 구간 {start:.1f}~{s.end_sec or duration:.1f}초가 곡 길이 {duration:.1f}초를 벗어남")
            elif end - start < 1:
                problems.append(f"{where}: 구간이 1초보다 짧음")
        if s.op == "loop" and not (1 <= s.seconds <= MAX_SECONDS):
            problems.append(f"{where}: 목표 길이 {s.seconds}초는 1~{MAX_SECONDS}초여야 함")
        if s.op == "concat" and not (0 <= s.amount <= 4):
            problems.append(f"{where}: 이어 붙이는 횟수 {s.amount} 는 1~4")
        if s.op == "fade":
            if s.position == "none":
                problems.append(f"{where}: position(start/end/both)이 없음")
            if not (0.1 <= s.seconds <= duration / 2):
                problems.append(f"{where}: 페이드 {s.seconds}초는 0.1초~곡 길이의 절반({duration / 2:.1f}초)")
        ranges = {"tempo": (-50, 100), "pitch_shift": (-12, 12), "gain": (-30, 12), "reverb": (0.01, 1),
                  "compress": (-40, -1), "normalize": (-30, -5)}
        if s.op in ranges:
            low, high = ranges[s.op]
            if not (low <= s.amount <= high) or (s.op in ("tempo", "pitch_shift", "gain") and s.amount == 0):
                problems.append(f"{where}: 값 {s.amount} 는 {low}~{high} (0 제외)")
        if s.op == "eq" and (not (-12 <= s.bass_db <= 12 and -12 <= s.treble_db <= 12) or s.bass_db == s.treble_db == 0):
            problems.append(f"{where}: 저음·고음 dB 는 -12~12 이고 하나는 0 이 아니어야 함")
        duration = next_duration(s, duration)
        if duration > MAX_SECONDS:
            problems.append(f"{where}: 결과 길이 {duration:.0f}초가 최대 {MAX_SECONDS}초를 넘음")
            break
    return problems


# ─── 3) 실행 (CPU) ────────────────────────────────────────────────────


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """(채널, 샘플) float32 배열과 샘플레이트. MP3·WAV 모두 (pedalboard 내장 디코더)."""
    from pedalboard.io import AudioFile

    with AudioFile(str(path)) as f:
        return f.read(f.frames).astype(np.float32), int(f.samplerate)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    """확장자에 맞춰 저장 (.mp3 는 내장 LAME, VBR V2 ≈ 190kbps)."""
    from pedalboard.io import AudioFile

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"quality": "V2"} if path.suffix.lower() == ".mp3" else {}
    with AudioFile(str(path), "w", sr, audio.shape[0], **kwargs) as f:
        f.write(np.ascontiguousarray(audio, dtype=np.float32))


def loudness(audio: np.ndarray, sr: int) -> float | None:
    """통합 라우드니스(LUFS). 너무 짧거나 무음이면 None."""
    import pyloudnorm

    if audio.shape[1] < sr * 0.5:
        return None
    value = pyloudnorm.Meter(sr).integrated_loudness(audio.T.astype(np.float64))
    return float(value) if np.isfinite(value) else None


def _declick(audio: np.ndarray, sr: int) -> np.ndarray:
    n = min(int(sr * DECLICK_SEC), audio.shape[1] // 2)
    if n > 0:
        audio = audio.copy()
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio[:, :n] *= ramp
        audio[:, -n:] *= ramp[::-1]
    return audio


def _slice(audio: np.ndarray, sr: int, step: EditStep) -> np.ndarray:
    start, end = _segment(step, audio.shape[1] / sr)
    return audio[:, int(round(start * sr)): int(round(end * sr))]


def op_trim(audio, sr, s: EditStep):
    return _declick(_slice(audio, sr, s), sr)


def op_loop(audio, sr, s: EditStep):
    piece = _declick(_slice(audio, sr, s), sr)
    target = int(round(s.seconds * sr))
    reps = int(np.ceil(target / piece.shape[1]))
    return np.concatenate([piece] * reps, axis=1)[:, :target]


def op_fade(audio, sr, s: EditStep):
    audio = audio.copy()
    n = min(int(s.seconds * sr), audio.shape[1])
    curve = np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32)) ** 2  # 부드러운 곡선
    if s.position in ("start", "both"):
        audio[:, :n] *= curve
    if s.position in ("end", "both"):
        audio[:, -n:] *= curve[::-1]
    return audio


def op_concat(audio, sr, s: EditStep):
    piece = _declick(_slice(audio, sr, s), sr)
    return np.concatenate([_declick(audio, sr), *[piece] * max(1, int(s.amount or 1))], axis=1)


def op_tempo(audio, sr, s: EditStep, high_quality: bool = False):
    from pedalboard import time_stretch

    # 음높이는 유지. high_quality 는 3분 곡 기준 약 30초 → 90초로 느려진다 (settings.yaml music.edit_high_quality)
    return time_stretch(audio, sr, stretch_factor=1 + s.amount / 100, high_quality=high_quality).astype(np.float32)


def op_pitch_shift(audio, sr, s: EditStep, high_quality: bool = False):
    from pedalboard import time_stretch

    out = time_stretch(audio, sr, stretch_factor=1.0, pitch_shift_in_semitones=s.amount, high_quality=high_quality)
    return out.astype(np.float32)[:, : audio.shape[1]]  # 길이는 그대로


def op_gain(audio, sr, s: EditStep):
    return audio * np.float32(10 ** (s.amount / 20))


def op_eq(audio, sr, s: EditStep):
    from pedalboard import HighShelfFilter, LowShelfFilter, Pedalboard

    board = Pedalboard([LowShelfFilter(cutoff_frequency_hz=200, gain_db=s.bass_db),
                        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=s.treble_db)])
    return board(audio, sr)


def op_reverb(audio, sr, s: EditStep):
    from pedalboard import Reverb

    return Reverb(room_size=0.5, wet_level=s.amount, dry_level=1.0 - s.amount * 0.3)(audio, sr)


def op_compress(audio, sr, s: EditStep):
    from pedalboard import Compressor

    return Compressor(threshold_db=s.amount, ratio=3.0, attack_ms=10, release_ms=200)(audio, sr)


def peak_limit(audio: np.ndarray, sr: int, ceiling_db: float = TRUE_PEAK_LIMIT_DB, window_ms: float = 10) -> np.ndarray:
    """간단한 미리보기(lookahead) 피크 리미터 — 넘치는 곳 주변만 부드럽게 줄인다.

    pedalboard.Limiter 는 출력 쪽에 보정 이득(threshold 만큼)을 더해 라우드니스 목표를 넘기므로 직접 만든다.
    필요한 이득(≤1)에 최소값 필터 → 같은 폭의 평균 필터를 걸면, 평균이 항상 필요한 이득 이하라 넘치지 않는다.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    ceiling = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(audio), axis=0)
    if peak.size == 0 or peak.max() <= ceiling:
        return audio
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    size = max(3, int(sr * window_ms / 1000) | 1)
    gain = uniform_filter1d(minimum_filter1d(need, size=size), size=size)
    return (audio * gain.astype(np.float32)).astype(np.float32)


def op_normalize(audio, sr, s: EditStep):
    # 목표 LUFS 로 맞춘 뒤 피크를 제한하면 라우드니스가 조금 내려가므로 두 번 보정한다
    for _ in range(2):
        current = loudness(audio, sr)
        if current is None:
            return audio
        audio = peak_limit(audio * np.float32(10 ** ((s.amount - current) / 20)), sr)
    return audio


OP_FUNCS = {
    "trim": op_trim, "loop": op_loop, "fade": op_fade, "concat": op_concat, "tempo": op_tempo,
    "pitch_shift": op_pitch_shift, "gain": op_gain, "eq": op_eq, "reverb": op_reverb, "compress": op_compress,
    "normalize": op_normalize,
}


def run_plan(audio: np.ndarray, sr: int, plan: EditPlan, high_quality: bool = False) -> tuple[np.ndarray, list[str]]:
    """계획대로 연산을 차례로 실행한다. 마지막에 0dBFS 를 넘으면 낮춰 클리핑을 막는다."""
    notes = []
    for step in plan.steps:
        extra = {"high_quality": high_quality} if step.op in ("tempo", "pitch_shift") else {}
        audio = OP_FUNCS[step.op](audio, sr, step, **extra).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio * np.float32(0.98 / peak)
        notes.append(f"최대 음량이 넘쳐(peak {peak:.2f}) 전체를 {20 * np.log10(0.98 / peak):.1f}dB 낮춤 (클리핑 방지)")
    return audio, notes


# ─── 4) 검증 · 5) 버전 기록 ─────────────────────────────────────────────


@dataclass
class Version:
    label: str  # v1, v2, …
    path: Path
    duration: float
    lufs: float | None
    note: str  # 무엇을 했는지 (원본 / 편집 요청)
    plan: EditPlan | None = None


@dataclass
class History:
    versions: list[Version] = field(default_factory=list)
    current: int = -1

    @property
    def now(self) -> Version:
        return self.versions[self.current]

    @property
    def previous(self) -> Version | None:
        return self.versions[self.current - 1] if self.current > 0 else None

    def add(self, version: Version) -> None:
        """되돌린 상태에서 새로 편집하면 그 뒤 버전은 버린다 (일반적인 실행 취소 규칙)."""
        del self.versions[self.current + 1:]
        self.versions.append(version)
        self.current = len(self.versions) - 1

    def undo(self) -> bool:
        if self.current <= 0:
            return False
        self.current -= 1
        return True

    def redo(self) -> bool:
        if self.current >= len(self.versions) - 1:
            return False
        self.current += 1
        return True

    def next_label(self) -> str:
        return f"v{max((int(v.label[1:]) for v in self.versions), default=0) + 1}"


def start_history(path: Path, note: str = "원본 (Lyria 생성)") -> History:
    audio, sr = load_audio(path)
    history = History()
    history.add(Version(label="v1", path=path, duration=audio_duration(path), lufs=loudness(audio, sr), note=note))
    return history


@dataclass
class EditResult:
    version: Version | None
    checks: list[str]  # 통과한 검증 내용
    problems: list[str]  # 계획 오류·검증 실패 (version 은 None 이거나 경고와 함께 만들어짐)


def verify(plan: EditPlan, out_path: Path, expected: float, audio: np.ndarray, sr: int) -> tuple[list[str], list[str], float, float | None]:
    """결과 파일을 다시 재서 계획과 비교한다 → (통과, 실패, 실제 길이, 실제 LUFS)."""
    checks, problems = [], []
    measured = audio_duration(out_path)
    if abs(measured - expected) <= max(DURATION_TOLERANCE, expected * 0.01):
        checks.append(f"길이 {measured:.2f}초 — 계획 {expected:.2f}초와 일치")
    else:
        problems.append(f"길이가 계획과 다름: 실제 {measured:.2f}초 / 계획 {expected:.2f}초")
    lufs = loudness(audio, sr)
    targets = [s.amount for s in plan.steps if s.op == "normalize"]
    after_last_norm = plan.steps[max(i for i, s in enumerate(plan.steps) if s.op == "normalize") + 1:] if targets else []
    if targets and lufs is not None and not any(s.op in ("gain", "eq", "reverb", "compress") for s in after_last_norm):
        if abs(lufs - targets[-1]) <= LUFS_TOLERANCE:
            checks.append(f"라우드니스 {lufs:.1f} LUFS — 목표 {targets[-1]:.0f} LUFS 와 일치")
        else:
            problems.append(f"라우드니스가 목표와 다름: 실제 {lufs:.1f} / 목표 {targets[-1]:.0f} LUFS")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    checks.append(f"최대 음량 {20 * np.log10(max(peak, 1e-9)):.1f} dBFS" + (" (클리핑 없음)" if peak <= 1.0 else ""))
    return checks, problems, measured, lufs


def apply_edit(history: History, plan: EditPlan, request: str, out_dir: Path, high_quality: bool = False) -> EditResult:
    """현재 버전에 편집 계획을 적용해 새 버전을 만든다. 계획이 잘못되면 실행하지 않는다."""
    src = history.now
    audio, sr = load_audio(src.path)
    duration = audio.shape[1] / sr  # 디코딩된 실제 샘플 기준 (MP3 앞뒤 여백 포함)
    problems = validate_plan(plan, duration)
    if plan.needs_regeneration:
        return EditResult(None, [], [regeneration_message(plan.regeneration_reason or "후처리로 할 수 없는 요청입니다.")])
    if problems:
        return EditResult(None, [], problems)
    out, notes = run_plan(audio, sr, plan, high_quality)
    label = history.next_label()
    out_path = out_dir / f"{label}.mp3"
    save_audio(out_path, out, sr)
    # 계획에서 계산한 길이와, 저장한 파일을 다시 잰 길이를 비교한다
    checks, failed, measured, lufs = verify(plan, out_path, expected_duration(plan, duration), out, sr)
    version = Version(label=label, path=out_path, duration=measured, lufs=lufs, note=request.strip() or plan.summary, plan=plan)
    history.add(version)
    return EditResult(version, checks + notes, failed)


def plan_and_apply(
    history: History, request: str, models: ModelsConfig, out_dir: Path, high_quality: bool = False
) -> tuple[EditPlan | None, EditResult]:
    """화면에서 부르는 한 번에 하기: 규칙 거르기 → 모델 계획 → 검사 → 실행 → 검증."""
    message = request_needs_regeneration(request)
    if message:
        return None, EditResult(None, [], [message])
    plan, _errors = plan_edit(request, history.now.duration, models)
    return plan, apply_edit(history, plan, request, out_dir, high_quality)
