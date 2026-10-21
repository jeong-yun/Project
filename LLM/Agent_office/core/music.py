"""음악 팀 — 최종 답변의 가사·Lyria 프롬프트 블록 → Google Gemini Lyria 로 MP3 생성 → 길이 확인 → (선택) 듣고 채점.

1) 블록 뽑기   : 최종 답변의 ```lyrics(가사, 섹션 태그) / ```lyria(영어 스타일 프롬프트, [m:ss - m:ss] 시간표) 블록
2) 형식 검사   : check_answer — 하네스 검증기 music_prompt 가 쓴다 (블록 존재, 반주의 'Instrumental only, no vocals',
                가사 섹션 태그·줄 수·음절 수, 아티스트 이름, 시간표 구간이 길이에 맞는지). 모두 코드 계산이라 결정론적
3) 생성       : google-genai 의 interactions API (client.interactions.create) — 30초는 lyria-3-clip-preview, 그 이상은 lyria-3.5
                오디오는 interaction.output_audio.data (base64, MP3 44.1kHz). 유료이고 한 번에 만들기만 한다(수정 불가)
4) 길이 확인   : mutagen 으로 MP3 길이를 재서 요청 길이와 비교 (Lyria 3.5 는 길이를 프롬프트로만 정하므로 어긋날 수 있다)
5) 듣고 채점   : (선택) 오디오를 이해하는 Gemini 모델에 MP3 와 브리프를 주고 1~5점 채점 → 점수가 낮으면 최대 N번 다시 생성

외부 호출(call_lyria, call_gemini_audio)은 함수 인자로 바꿔 끼울 수 있다 → 테스트는 가짜 함수로 네트워크 없이 돈다.
설정은 config/settings.yaml 의 music: (모델·가격·채점 기준·자동 재생성 횟수).
"""

import base64
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from core.harness.base import extract_code_blocks

# settings.yaml 의 music: 가 없을 때 쓰는 기본값
DEFAULTS = {
    "auto_generate": True,
    "clip_model": "lyria-3-clip-preview",
    "song_model": "lyria-3.5",
    "clip_seconds": 30,
    "cost_usd": {"lyria-3-clip-preview": 0.04, "lyria-3.5": 0.08},
    "grade": True,
    "grade_model": "gemini-3.5-flash",
    "grade_threshold": 3,
    "max_regenerations": 1,
    "duration_tolerance": 0.25,
}

SONG, INSTRUMENTAL = "song", "instrumental"
INSTRUMENTAL_PHRASE = "Instrumental only, no vocals."

SECTION_TAG = re.compile(
    r"^\s*\[\s*(intro|verse|pre-?chorus|chorus|post-?chorus|hook|refrain|bridge|outro|interlude|break|rap|drop|"
    r"도입|벌스|프리코러스|코러스|후렴|브릿지|간주|아웃트로|마무리)[^\]]*\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CHORUS_TAG = re.compile(r"\[\s*(chorus|hook|refrain|후렴|코러스)", re.IGNORECASE)
TIMESTAMP = re.compile(r"\[\s*(\d{1,2}):(\d{2})\s*[-–~]\s*(\d{1,2}):(\d{2})\s*\]")
HANGUL_OR_KANA = re.compile(r"[가-힣぀-ヿ一-鿿]")
LATIN_WORD = re.compile(r"[A-Za-z']+")
VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)

# 아티스트 이름 휴리스틱 — Lyria 는 실존 가수 목소리·스타일 요청을 거부한다. 흔한 이름 + 'in the style of' 류 표현만 본다
KNOWN_ARTISTS = [
    "아이유", "방탄소년단", "블랙핑크", "뉴진스", "임영웅", "조용필", "이선희", "아이브", "에스파", "세븐틴", "트와이스",
    "빅뱅", "싸이", "성시경", "백예린", "악뮤", "볼빨간사춘기", "태연", "아이콘", "BTS", "BLACKPINK", "NewJeans", "IU",
    "Taylor Swift", "Beatles", "Ed Sheeran", "Billie Eilish", "Ariana Grande", "Bruno Mars", "Michael Jackson",
    "Beyonce", "Beyoncé", "Coldplay", "Adele", "Justin Bieber", "Lady Gaga", "Eminem", "The Weeknd", "Dua Lipa",
    "Olivia Rodrigo", "Daft Punk", "Hans Zimmer", "Joe Hisaishi", "Ryuichi Sakamoto", "YOASOBI", "Kenshi Yonezu",
    "Mariya Takeuchi", "Tatsuro Yamashita",
]
ARTIST_PHRASES = [
    r"\bin the style of\b", r"\bsounds? like\s+[A-Z]", r"\b(?:voice|vocals?) of\s+[A-Z]", r"\bsung by\b",
    r"\bcover of\b", r"\bimpersonat", r"(?:의|같은)\s*(?:목소리로|창법으로|스타일로)",
]


class MusicError(RuntimeError):
    """음원을 만들 수 없는 상태 (블록 없음·키 없음·Lyria 거부 등) — 화면에 그대로 보여 준다."""


# ─── 1) 블록 뽑기 · 길이 ───────────────────────────────────────────────


@dataclass
class MusicBlocks:
    lyrics: str  # ```lyrics 블록 (반주면 빈 문자열)
    prompt: str  # ```lyria 블록 (영어 스타일 프롬프트)


def extract_blocks(answer: str) -> MusicBlocks:
    """최종 답변에서 가사·Lyria 프롬프트 블록을 뽑는다. 같은 블록이 여러 개면 첫 번째를 쓴다."""
    lyrics = extract_code_blocks(answer, "lyrics")
    prompt = extract_code_blocks(answer, "lyria")
    return MusicBlocks(lyrics=(lyrics[0].strip() if lyrics else ""), prompt=(prompt[0].strip() if prompt else ""))


def parse_length(text: str | int | float | None) -> int | None:
    """'30초 (클립)', '1분 30초', '2분', '2:30', '90s', '90' → 초. 알 수 없으면 None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text) if text > 0 else None
    s = str(text).strip().lower()
    m = re.search(r"(\d+):(\d{2})", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:분|min)", s)
    seconds = re.search(r"(\d+)\s*(?:초|sec|s\b)", s)
    if minutes or seconds:
        total = float(minutes.group(1)) * 60 if minutes else 0
        return int(total + (int(seconds.group(1)) if seconds else 0)) or None
    m = re.fullmatch(r"(\d+)", s)
    return int(m.group(1)) if m and int(m.group(1)) > 0 else None


def fmt_time(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def lyric_lines(lyrics: str) -> list[str]:
    """섹션 태그·빈 줄을 뺀 가사 줄."""
    return [ln.strip() for ln in lyrics.splitlines() if ln.strip() and not SECTION_TAG.match(ln)]


def count_syllables(line: str) -> int:
    """음절 수 근사 — 한글·가나·한자는 글자당 1, 영어는 단어마다 모음 묶음 수(최소 1)."""
    count = len(HANGUL_OR_KANA.findall(line))
    for word in LATIN_WORD.findall(line):
        groups = len(VOWEL_GROUP.findall(word))
        if word.lower().endswith("e") and groups > 1 and not word.lower().endswith(("le", "ee")):
            groups -= 1  # 묵음 e (make, love)
        count += max(1, groups)
    return count


def lyric_limits(length_sec: int) -> dict:
    """곡 길이에 맞는 가사 분량 범위 — 대략 노래 4초에 한 줄, 초당 0.6~4.5 음절."""
    return {
        "min_lines": max(4, round(length_sec / 15)),
        "max_lines": round(length_sec / 2.5) + 4,
        "max_line_syllables": 28,
        "min_syllables": round(length_sec * 0.6),
        "max_syllables": round(length_sec * 4.5),
    }


def timestamp_sections(prompt: str) -> list[tuple[int, int]]:
    return [
        (int(a) * 60 + int(b), int(c) * 60 + int(d)) for a, b, c, d in TIMESTAMP.findall(prompt)
    ]


def find_artists(text: str, extra: list[str] | None = None) -> list[str]:
    """실존 아티스트 이름·모방 표현 (간단한 휴리스틱)."""
    found = []
    for name in [*KNOWN_ARTISTS, *(extra or [])]:
        if re.fullmatch(r"[A-Za-z .'é]+", name):
            hit = re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", text, re.IGNORECASE if len(name) > 3 else 0)
        else:
            hit = name in text
        if hit:
            found.append(name)
    found += [m.group(0) for p in ARTIST_PHRASES for m in [re.search(p, text, re.IGNORECASE if p.startswith(r"\bin") else 0)] if m]
    return list(dict.fromkeys(found))


def infer_kind(blocks: MusicBlocks, kind: str | None = None) -> str:
    if kind in (SONG, INSTRUMENTAL):
        return kind
    return SONG if blocks.lyrics else INSTRUMENTAL


# ─── 2) 형식 검사 (하네스 music_prompt) ─────────────────────────────────


def check_answer(answer: str, options: dict) -> list[str]:
    """음악 결과 형식 문제 목록 (빈 목록이면 통과). options: music_kind(song/instrumental), music_length('2분' 등)."""
    blocks = extract_blocks(answer)
    kind = options.get("music_kind")
    if not blocks.prompt:
        if kind:
            return ["```lyria 블록(Lyria 스타일 프롬프트)이 없다. 최종 답변에 ```lyria 블록을 넣어라"]
        return []  # 자유 질문(작사 팁 등)은 음원을 만들지 않으므로 검사하지 않는다
    kind = infer_kind(blocks, kind)
    length = parse_length(options.get("music_length"))
    problems: list[str] = []

    has_instrumental = INSTRUMENTAL_PHRASE.lower().rstrip(".") in " ".join(blocks.prompt.lower().split())
    if kind == INSTRUMENTAL:
        if not has_instrumental:
            problems.append(f"반주 요청인데 ```lyria 블록에 '{INSTRUMENTAL_PHRASE}' 문장이 없다")
        if lyric_lines(blocks.lyrics):
            problems.append("반주 요청인데 ```lyrics 가사 블록이 있다. 가사를 빼라")
    else:
        if has_instrumental:
            problems.append("노래(보컬) 요청인데 ```lyria 블록에 'Instrumental only' 가 있다")
        problems += _check_lyrics(blocks.lyrics, length)

    artists = find_artists(blocks.prompt + "\n" + blocks.lyrics, options.get("banned_artists"))
    if artists:
        problems.append(f"실존 아티스트 이름·모방 표현이 있다 (Lyria 가 거부함): {artists}. 장르·시대·악기·음색으로 바꿔 써라")
    problems += _check_timestamps(blocks.prompt, length)
    return problems


def _check_lyrics(lyrics: str, length: int | None) -> list[str]:
    if not lyrics.strip():
        return ["노래 요청인데 ```lyrics 가사 블록이 없다"]
    problems = []
    tags = SECTION_TAG.findall(lyrics)
    if not tags:
        problems.append("가사에 [Verse]·[Chorus] 같은 섹션 태그가 없다 (태그는 한 줄에 단독으로)")
    elif length and length >= 60 and not CHORUS_TAG.search(lyrics):
        problems.append("1분 이상 노래인데 [Chorus](후렴) 섹션이 없다")
    lines = lyric_lines(lyrics)
    if not length:
        return problems
    lim = lyric_limits(length)
    if len(lines) < lim["min_lines"]:
        problems.append(f"가사가 {len(lines)}줄로 너무 짧다 ({fmt_time(length)} 곡은 {lim['min_lines']}~{lim['max_lines']}줄)")
    elif len(lines) > lim["max_lines"]:
        problems.append(f"가사가 {len(lines)}줄로 너무 길다 ({fmt_time(length)} 곡은 {lim['min_lines']}~{lim['max_lines']}줄)")
    long_lines = [ln for ln in lines if count_syllables(ln) > lim["max_line_syllables"]]
    if long_lines:
        problems.append(f"한 줄이 너무 길어 부르기 어렵다 ({lim['max_line_syllables']}음절 초과): {long_lines[:3]}")
    total = sum(count_syllables(ln) for ln in lines)
    if total > lim["max_syllables"]:
        problems.append(f"가사 음절 수 {total} 가 {fmt_time(length)} 에 너무 많다 (최대 약 {lim['max_syllables']})")
    elif total < lim["min_syllables"]:
        problems.append(f"가사 음절 수 {total} 가 {fmt_time(length)} 에 너무 적다 (최소 약 {lim['min_syllables']})")
    return problems


def _check_timestamps(prompt: str, length: int | None) -> list[str]:
    sections = timestamp_sections(prompt)
    if not sections:
        return ["```lyria 블록에 [0:00 - 0:15] 형식의 시간표 구간이 없다"]
    problems = []
    bad = [f"[{fmt_time(s)} - {fmt_time(e)}]" for s, e in sections if e <= s]
    if bad:
        problems.append(f"끝이 시작보다 빠른 구간: {bad}")
    if sections[0][0] > 2:
        problems.append(f"첫 구간이 0:00 이 아니라 {fmt_time(sections[0][0])} 에서 시작한다")
    for (s1, e1), (s2, e2) in zip(sections, sections[1:]):
        if s2 < e1 - 1:
            problems.append(f"구간이 겹친다: [{fmt_time(s1)} - {fmt_time(e1)}] 과 [{fmt_time(s2)} - {fmt_time(e2)}]")
        elif s2 > e1 + 2:
            problems.append(f"구간 사이가 비어 있다: {fmt_time(e1)} ~ {fmt_time(s2)}")
    if length:
        last = max(e for _, e in sections)
        if last > length + max(3, length * 0.1):
            problems.append(f"시간표가 {fmt_time(last)} 까지라 요청 길이 {fmt_time(length)} 를 넘는다")
        elif last < length * 0.85:
            problems.append(f"시간표가 {fmt_time(last)} 에서 끝나 요청 길이 {fmt_time(length)} 를 채우지 못한다")
    return problems


# ─── 3) 생성 (Lyria) ──────────────────────────────────────────────────


def music_settings(settings: dict) -> dict:
    cfg = {**DEFAULTS, **(settings.get("music") or {})}
    cfg["cost_usd"] = {**DEFAULTS["cost_usd"], **(cfg.get("cost_usd") or {})}
    return cfg


def pick_model(length_sec: int | None, cfg: dict) -> str:
    """30초 이하는 클립 모델(항상 30초), 그 이상은 전체 곡 모델."""
    return cfg["clip_model"] if length_sec and length_sec <= cfg["clip_seconds"] else cfg["song_model"]


def estimate_cost(model: str, cfg: dict, takes: int = 1) -> float:
    return round(float(cfg["cost_usd"].get(model, 0.0)) * takes, 2)


def build_lyria_input(blocks: MusicBlocks, kind: str, length_sec: int | None, model: str, cfg: dict) -> str:
    """Lyria 에 보낼 입력 — 스타일 프롬프트 (+ 길이) (+ 가사). 가사는 프롬프트 뒤에 섹션 태그 그대로 붙인다."""
    prompt = blocks.prompt.strip()
    if kind == INSTRUMENTAL and INSTRUMENTAL_PHRASE.lower().rstrip(".") not in prompt.lower():
        prompt = f"{INSTRUMENTAL_PHRASE} {prompt}"
    parts = []
    if length_sec and model != cfg["clip_model"]:  # 클립은 항상 30초, 전체 곡은 길이를 프롬프트로 정한다
        parts.append(f"Create a {fmt_time(length_sec)} long {'instrumental track' if kind == INSTRUMENTAL else 'song'}.")
    parts.append(prompt)
    if kind == SONG and blocks.lyrics.strip():
        parts.append("Lyrics:\n" + blocks.lyrics.strip())
    return "\n\n".join(parts)


@dataclass
class LyriaOutput:
    audio: bytes  # MP3
    text: str = ""  # 모델이 함께 돌려준 글 (가사·구성 설명)


LYRIA_TIMEOUT_SEC = 300  # 곡 1개 생성 최대 대기 — 넘으면 오류로 끝낸다 (화면이 '진행 중'에 멈추지 않게)
GRADE_TIMEOUT_SEC = 120  # 채점(Gemini 가 듣기) 최대 대기


def _genai_client(timeout_sec: float = LYRIA_TIMEOUT_SEC):
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise MusicError("GOOGLE_API_KEY 가 없어 Lyria 를 부를 수 없습니다 (.env 확인)")
    from google import genai
    from google.genai import types

    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)))


def call_lyria(model: str, prompt: str) -> LyriaOutput:
    """Lyria 1회 호출 (유료). 테스트에서는 가짜 함수로 바꾼다."""
    # 클라이언트를 변수에 잡아 둔다 — `_genai_client().interactions.create(...)` 처럼 임시 객체로 쓰면
    # google-genai 가 Client 가 사라질 때(__del__) 연결을 닫아 'client has been closed' 로 요청이 나가지도 못한다
    client = _genai_client(LYRIA_TIMEOUT_SEC)
    interaction = client.interactions.create(model=model, input=prompt)
    audio = interaction.output_audio
    text = interaction.output_text or ""
    if audio is None or not audio.data:
        raise MusicError("Lyria 가 음원을 돌려주지 않았습니다 (안전 필터·아티스트 이름 등으로 거부됐을 수 있음)" + (f": {text[:200]}" if text else ""))
    data = audio.data
    return LyriaOutput(audio=base64.b64decode(data) if isinstance(data, str) else bytes(data), text=text)


def audio_duration(path: Path) -> float:
    """MP3 길이(초) — mutagen (ffmpeg 불필요)."""
    from mutagen.mp3 import MP3

    return float(MP3(str(path)).info.length)


def duration_note(measured: float, length_sec: int | None, tolerance: float) -> str:
    """요청 길이와 많이 다르면 안내 문장 (편집 단계에서 자르기·반복으로 맞출 수 있다)."""
    if not length_sec or abs(measured - length_sec) <= length_sec * tolerance:
        return ""
    return (f"요청 길이 {fmt_time(length_sec)} 와 실제 길이 {fmt_time(measured)} 가 다릅니다 — "
            "아래 편집에서 '30초로 줄여'·'2분이 되게 반복해' 처럼 맞출 수 있습니다.")


# ─── 5) 듣고 채점 (Gemini 오디오 이해) ───────────────────────────────────


class TrackGrade(BaseModel):
    """채점 결과 — 들은 내용을 먼저 쓰고 점수를 매긴다 (필드 순서 = 생각 순서)."""

    heard: str = Field(description="들은 내용 요약: 장르·빠르기·악기·보컬 유무·가사 언어·구성")
    vocals_present: bool = Field(description="사람 목소리(노래)가 들리는가")
    matches_brief: bool = Field(description="브리프(장르·분위기·보컬 여부·언어)와 대체로 맞는가")
    issues: list[str] = Field(description="브리프와 다른 점 (없으면 빈 목록)")
    score: int = Field(description="브리프 일치도 1~5 (5 = 완전히 맞음)", ge=1, le=5)


GradeCall = Callable[[str, bytes, str], TrackGrade]


def call_gemini_audio(model: str, audio: bytes, prompt: str) -> TrackGrade:
    """오디오 + 지시문 → TrackGrade (구조화 출력). 테스트에서는 가짜 함수로 바꾼다."""
    from google.genai import types

    client = _genai_client(GRADE_TIMEOUT_SEC)  # 변수에 잡아 둔다 (call_lyria 주석 참고)
    response = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=audio, mime_type="audio/mp3"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TrackGrade, temperature=0.2),
    )
    return response.parsed if isinstance(response.parsed, TrackGrade) else TrackGrade.model_validate_json(response.text)


def listen_and_grade(audio: bytes, brief: str, kind: str, length_sec: int | None, cfg: dict, call: GradeCall = call_gemini_audio) -> TrackGrade:
    """Gemini 가 곡을 듣고 브리프와 비교해 채점한다."""
    expect = "보컬 없는 반주" if kind == INSTRUMENTAL else "보컬이 있는 노래"
    prompt = "\n".join([
        "너는 음반 검수자다. 첨부한 음원을 듣고 아래 브리프와 비교해 채점한다. 들은 것만 근거로 쓴다.",
        f"- 기대: {expect}" + (f", 길이 약 {fmt_time(length_sec)}" if length_sec else ""),
        "- 보컬 여부가 기대와 다르면 2점 이하. 장르·분위기·언어가 다르면 감점.",
        "[브리프]",
        brief.strip()[:4000],
    ])
    grade = call(cfg["grade_model"], audio, prompt)
    if kind == INSTRUMENTAL and grade.vocals_present:  # 반주에 목소리가 들리면 코드로도 낮춘다
        grade.score = min(grade.score, 2)
        grade.issues = [*grade.issues, "반주인데 보컬이 들림"]
    return grade


# ─── 생성 흐름: 생성 → 길이 확인 → 채점 → (점수 낮으면) 다시 생성 ─────────────


@dataclass
class Take:
    path: Path
    duration: float
    grade: TrackGrade | None = None
    text: str = ""


@dataclass
class MusicTrack:
    kind: str
    model: str
    length_sec: int | None
    lyria_input: str  # Lyria 에 실제로 보낸 입력
    blocks: MusicBlocks
    takes: list[Take] = field(default_factory=list)
    chosen: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Take:
        return self.takes[self.chosen]


def make_track(
    answer: str,
    out_dir: Path,
    cfg: dict,
    kind: str | None = None,
    length: str | int | None = None,
    brief: str = "",
    lyria: Callable[[str, str], LyriaOutput] = call_lyria,
    grader: GradeCall | None = call_gemini_audio,
) -> MusicTrack:
    """최종 답변 → MP3. 채점 점수가 grade_threshold 보다 낮으면 max_regenerations 번까지 다시 만들고, 가장 좋은 것을 고른다."""
    blocks = extract_blocks(answer)
    if not blocks.prompt:
        raise MusicError("최종 답변에 ```lyria 블록이 없어 음원을 만들 수 없습니다")
    kind = infer_kind(blocks, kind)
    length_sec = parse_length(length)
    model = pick_model(length_sec, cfg)
    track = MusicTrack(kind=kind, model=model, length_sec=length_sec, blocks=blocks,
                       lyria_input=build_lyria_input(blocks, kind, length_sec, model, cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_takes = 1 + max(0, int(cfg.get("max_regenerations", 1)))
    for n in range(1, max_takes + 1):
        out = lyria(model, track.lyria_input)
        track.cost_usd = estimate_cost(model, cfg, n)
        path = out_dir / f"take{n}.mp3"
        path.write_bytes(out.audio)
        take = Take(path=path, duration=audio_duration(path), text=out.text)
        track.takes.append(take)
        if cfg.get("grade") and grader is not None:
            try:
                take.grade = listen_and_grade(out.audio, brief or answer, kind, length_sec, cfg, grader)
            except Exception as e:  # noqa: BLE001 — 채점 실패가 음원을 막지 않게
                track.notes.append(f"채점 실패 (음원은 그대로 사용): {type(e).__name__}: {str(e)[:200]}")
        if take.grade is None or take.grade.score >= int(cfg["grade_threshold"]):
            break
        if n < max_takes:
            track.notes.append(f"{n}번째 곡 채점 {take.grade.score}/5 — 기준 {cfg['grade_threshold']}점 미만이라 다시 생성합니다: "
                               + "; ".join(take.grade.issues[:3]))
    graded = [(t.grade.score if t.grade else 0, -i) for i, t in enumerate(track.takes)]
    track.chosen = -max(graded)[1] if any(t.grade for t in track.takes) else len(track.takes) - 1
    note = duration_note(track.best.duration, length_sec if model != cfg["clip_model"] else cfg["clip_seconds"], cfg["duration_tolerance"])
    if note:
        track.notes.append(note)
    return track


def recover_track(folder: Path, answer: str, cfg: dict, kind: str | None = None, length: str | int | None = None) -> MusicTrack | None:
    """서버를 다시 켜서 화면 상태가 사라졌을 때, 이미 만든 곡(folder/take*.mp3)을 다시 불러온다 (다시 생성 = 비용이 들므로)."""
    takes = sorted(folder.glob("take*.mp3"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not takes:
        return None
    blocks = extract_blocks(answer)
    kind = infer_kind(blocks, kind)
    length_sec = parse_length(length)
    model = pick_model(length_sec, cfg)
    return MusicTrack(
        kind=kind, model=model, length_sec=length_sec, blocks=blocks,
        lyria_input=build_lyria_input(blocks, kind, length_sec, model, cfg) if blocks.prompt else "",
        takes=[Take(path=p, duration=audio_duration(p)) for p in takes], chosen=len(takes) - 1,
        cost_usd=estimate_cost(model, cfg, len(takes)),
        notes=["서버를 다시 켜서 이전에 만든 곡을 다시 불러왔습니다 (채점·편집 기록은 원본부터 다시 시작)"],
    )
