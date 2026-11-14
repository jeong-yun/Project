"""🎙 음성 합성 팀 — 노래 음색 변환(SVC, so-vits-svc-fork) 준비를 돕는 코드. 모델 호출·외부 실행·패키지 설치 없음.

보컬 분리·학습·변환은 앱 밖(사용자 PC 의 preprocess.py · Colab 노트북)에서 돈다. 이 모듈이 하는 일:
  1) 책임 있는 사용 : check_form(양식 — 목소리 출처·확인 체크·화자 이름) · misuse(사칭·무단 복제 요청 낱말)
  2) 맞춤 파일      : plan_settings(epochs·batch 를 규칙으로) → build_notebook(JSON 으로 읽어 셀 값만 바꿈) → kit_zip
  3) 데이터 점검    : inspect_zip(dataset_raw.zip 을 풀지 않고 메모리에서 wav 만 읽어 숫자 계산) → dataset_table
  4) 하네스 검사    : check_guard · check_plan · check_data (core/harness/validators.py 의 voice_* 검증기가 쓴다)

원본 스크립트·노트북은 teams/voice/assets/ 에 있다 (사용자가 만든 파일 그대로). 슬라이싱 기준(−40dB·2초·15초)은
그 스크립트의 slice_wav 기본값을 읽어 쓴다 → 기준이 한 곳에만 있다.
"""

import ast
import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from statistics import median

from core.config import TEAMS_DIR
from core.music import KNOWN_ARTISTS
from core.tasks import TaskInputError

ASSET_DIR = TEAMS_DIR / "voice" / "assets"
NOTEBOOK = "svc_colab_training.ipynb"
SCRIPT_FILES = ("preprocess.py", "run.bat")  # 묶음에 원본 그대로 넣는 파일
MB = 1024 * 1024

# ─── 1) 책임 있는 사용 ────────────────────────────────────────────────

SOURCES = ("내 목소리", "본인 동의를 받은 사람의 목소리", "사용이 허락된 공개 데이터셋")
SINGER = re.compile(r"[A-Za-z0-9_-]{1,40}")  # 화자 이름 = 폴더 이름 (dataset_raw/<화자>/)
AI_LABEL = "AI 로 합성한 음성"
CONSENT_HELP = ("목소리 사용 확인이 필요합니다 — '이 목소리를 학습·변환에 써도 된다는 것을 확인했습니다'에 표시해 주세요. "
                "다른 사람의 목소리라면 먼저 그 사람에게 학습·변환에 써도 된다는 동의를 받아야 합니다. "
                "확인이 없으면 이 팀은 학습 계획과 노트북을 만들지 않습니다.")
MISUSE = [  # (찾을 표현, 화면·프롬프트에 쓸 이유)
    (r"보이스\s*피싱|사칭|(?:인|한|는)\s?척(?=하|을|\s|$|[.,!?])|속이(?:려|기|고|는|다)|속여|속일|속였", "남을 속이거나 다른 사람인 척하는 용도"),
    (r"(?:본인|음성|목소리|화자)\s*인증|인증\s*(?:을|를)?\s*(?:우회|통과|뚫)", "본인 인증을 통과하거나 피해 가는 용도"),
    (r"(?:동의|허락)\s*(?:없이|없는|안\s*받|받지\s*않)|몰래|무단", "본인 동의를 받지 않은 목소리"),
]
RUN_STEPS = re.compile(r"svc\s+(?:train|infer|pre-\w+)|preprocess\.py|run\.bat|dataset_raw", re.IGNORECASE)


class VoiceError(ValueError):
    """올린 zip·자산 파일 문제 — 화면에 그대로 보여 줄 쉬운 말로 쓴다."""


def voice_targets(text: str) -> list[str]:
    """'<실존 가수> 목소리·음색'처럼 가수의 목소리를 대상으로 삼은 표현. (그 가수의 곡을 부른 커버는 해당 없음)

    이름 목록은 음악 팀의 것을 재사용한다 (core/music.KNOWN_ARTISTS) — 목록에 없는 사람은 코드로 알 수 없다."""
    found = []
    for name in KNOWN_ARTISTS:
        head = r"(?<![A-Za-z])" if re.fullmatch(r"[A-Za-z .'é]+", name) else ""
        if re.search(head + re.escape(name) + r"\s*(?:님|씨)?\s*(?:의\s*)?(?:목소리|음색|보이스|창법|voice)", text, re.IGNORECASE):
            found.append(name)
    return found


def is_artist_name(singer: str) -> bool:
    key = re.sub(r"[^0-9a-z가-힣]", "", singer.lower())
    return bool(key) and key in {re.sub(r"[^0-9a-z가-힣]", "", n.lower()) for n in KNOWN_ARTISTS}


def misuse(text: str) -> str:
    """범위 밖(사칭·무단 복제) 요청이면 그 이유, 아니면 빈 글. 낱말 확인이라 완벽하지 않다 — 팀 지침·양식 확인과 함께 쓴다."""
    for pattern, reason in MISUSE:
        if re.search(pattern, text):
            return reason
    return "실존 가수·유명인의 목소리를 복제하는 용도" if voice_targets(text) else ""


def refusal_message(reason: str) -> str:
    return (f"이 팀은 {reason}에는 도움을 드리지 않습니다. "
            "내 목소리이거나 본인에게 동의를 받은 목소리로, 노래 음색 변환을 준비하는 일만 돕습니다. "
            "그런 경우라면 목소리 출처를 적어 다시 맡겨 주세요.")


def check_form(task_id: str, values: dict, extra: str = "") -> None:
    """업무 양식 확인 — 문제가 있으면 TaskInputError (화면에 그대로 보인다). 맡기기 전에 부른다."""
    reason = misuse(" ".join(str(values.get(k) or "") for k in ("goal", "note")) + " " + extra)
    if reason:
        raise TaskInputError(refusal_message(reason))
    if task_id == "plan":
        if values.get("source") not in SOURCES:
            raise TaskInputError("목소리 출처를 골라 주세요: " + " / ".join(SOURCES))
        if values.get("consent") is not True:
            raise TaskInputError(CONSENT_HELP)
        singer = str(values.get("singer") or "").strip()
        if singer and not SINGER.fullmatch(singer):
            raise TaskInputError("화자 이름은 영문·숫자·`-`·`_` 만 쓸 수 있습니다 (폴더 이름이 됩니다, 40자까지). 예: myvoice, jiyun_01")
        if is_artist_name(singer):
            raise TaskInputError(refusal_message("실존 가수·유명인의 목소리를 복제하는 용도"))
    if task_id == "check" and values.get("dataset"):
        try:
            scan_zip(values["dataset"])
        except VoiceError as e:
            raise TaskInputError(str(e)) from e


# ─── 2) 맞춤 파일 (학습 설정은 코드가 규칙으로 정한다) ─────────────────────

# 근거는 원본 파일에 적힌 값뿐이다: 노트북 8번 셀 'epochs 400 — 데이터 30분~1시간 기준', 'batch_size 16 — T4 기준, 메모리 부족(OOM) 시 8',
# preprocess.py '10분 미만 주의, 30분~1시간 이상 권장'. 그 밖의 구간은 재 보지 않았으므로 기본값을 그대로 둔다.
DEFAULT_EPOCHS, DEFAULT_BATCH, LOW_MEMORY_BATCH = 400, 16, 8
MIN_MINUTES, GOOD_MINUTES, BASE_MAX_MINUTES = 10, 30, 60
VERDICTS = ("부족", "주의", "충분")
TIME_NOTES = [
    "보컬 분리 — Colab GPU: 곡당 약 1분 (노트북 B안 설명에 적힌 어림값)",
    "보컬 분리 — GPU 없는 PC: 곡당 수 분~수십 분 (preprocess.py 안내 문구). 빠른 모델(--fast)을 쓰면 줄어든다",
    "so-vits-svc-fork 설치: 2~3분 (노트북 3번 셀)",
    "학습에 걸리는 시간: 재 보지 않음 — 데이터 분량과 epochs 에 따라 다르다. 체크포인트가 Drive 에 저장되므로 세션이 끊기면 이어서 학습한다",
]


def verdict(minutes: float) -> str:
    """보컬 분량 판정 — 10분 미만 부족 · 30분 미만 주의 · 그 이상 충분 (preprocess.py 의 안내 기준)."""
    return VERDICTS[0] if minutes < MIN_MINUTES else VERDICTS[1] if minutes < GOOD_MINUTES else VERDICTS[2]


def plan_settings(minutes: float | None, oom: bool = False) -> dict:
    """{epochs, batch_size, notes} — 모델이 아니라 규칙이 정한다. 근거 없는 구간은 기본값 + '늘려서 이어 학습' 안내."""
    notes = []
    if minutes is None:
        notes.append("보컬 분량을 모름 — epochs 는 기본값. 전처리 뒤 '🔍 데이터 점검'으로 분량을 확인한다")
    elif minutes < MIN_MINUTES:
        notes.append(f"보컬 약 {minutes:g}분 — 10분 미만이라 부족하다. 학습 전에 음원을 더 모으는 것이 먼저다 (30분~1시간 이상 권장)")
    elif minutes < GOOD_MINUTES:
        notes.append(f"보컬 약 {minutes:g}분 — 권장(30분~1시간)보다 적다. epochs 는 기본값 그대로 (이 구간의 알맞은 값은 재 보지 않음)")
    elif minutes <= BASE_MAX_MINUTES:
        notes.append(f"보컬 약 {minutes:g}분 — 기본값의 기준 구간(30분~1시간)이다")
    else:
        notes.append(f"보컬 약 {minutes:g}분 — 기준 구간(30분~1시간)보다 많다. 기본값으로 시작하고, 부족하면 epochs 를 늘려 다시 실행한다 (이어서 학습됨)")
    if oom:
        notes.append("메모리 부족(OOM)을 겪었다고 해서 batch_size 를 8 로 낮췄다 (노트북 8번 셀의 안내)")
    else:
        notes.append("학습 중 메모리 부족(OOM) 오류가 나면 batch_size 를 8 로 낮춰 다시 실행한다")
    return {"epochs": DEFAULT_EPOCHS, "batch_size": LOW_MEMORY_BATCH if oom else DEFAULT_BATCH, "notes": notes}


def kit_from_inputs(inputs: dict) -> dict | None:
    """양식 값 → 맞춤 파일 설정. 목소리 사용 확인이 없거나 화자 이름이 맞지 않으면 None (계획·노트북을 만들지 않는다)."""
    singer = str(inputs.get("singer") or "").strip()
    if inputs.get("consent") is not True or inputs.get("source") not in SOURCES:
        return None
    if not SINGER.fullmatch(singer) or is_artist_name(singer):
        return None
    minutes = float(inputs["minutes"]) if inputs.get("minutes") not in (None, "") else None
    mode = "B" if str(inputs.get("where") or "").startswith("Colab") else "A"
    has_gpu = str(inputs.get("gpu") or "").startswith("있")
    settings = plan_settings(minutes, bool(inputs.get("oom")))
    return {"singer": singer, "source": inputs["source"], "mode": mode, "fast": mode == "A" and not has_gpu,
            "minutes": minutes, "songs": int(inputs["songs"]) if inputs.get("songs") else None, **settings}


def _set_line(nb: dict, pattern: str, value: str) -> None:
    """코드 셀에서 pattern 에 맞는 줄 한 곳의 값만 바꾼다. 못 찾거나 여러 곳이면 원본이 바뀐 것이므로 멈춘다."""
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            new, n = re.subn(pattern, lambda m: m.group(1) + value, line)
            if n:
                cell["source"][i] = new
                hits += n
    if hits != 1:
        raise VoiceError(f"노트북 원본에서 값을 넣을 줄을 {hits}곳 찾았습니다 (1곳이어야 함): {pattern}")


def build_notebook(kit: dict) -> str:
    """원본 노트북을 JSON 으로 읽어 화자 이름·epochs·batch_size 줄의 값만 바꾼다 (셀 수·순서는 그대로)."""
    nb = json.loads((ASSET_DIR / NOTEBOOK).read_text(encoding="utf-8"))
    _set_line(nb, r"^(SINGER = )'[^']*'", f"'{kit['singer']}'")
    _set_line(nb, r"^(cfg\['train'\]\['epochs'\] = )\d+", str(kit["epochs"]))
    _set_line(nb, r"^(cfg\['train'\]\['batch_size'\] = )\d+", str(kit["batch_size"]))
    skip = "4번 셀(A안)을 실행하고 5~6번 셀(B안)은 건너뜁니다" if kit["mode"] == "A" else "4번 셀(A안)은 건너뛰고 5~6번 셀(B안)을 실행합니다"
    nb["cells"][0]["source"] += [
        "\n", "\n",
        f"**이 노트북의 맞춤 값** (Agent Office 🎙 음성 합성 팀이 양식 값으로 넣음): 화자 `{kit['singer']}` · "
        f"epochs {kit['epochs']} · batch_size {kit['batch_size']} · 전처리 {kit['mode']}안 — {skip}.\n",
        "\n",
        f"목소리 출처: {kit['source']} (양식에서 사용해도 된다고 확인함). 변환한 결과물을 남에게 보이거나 올릴 때는 '{AI_LABEL}'이라고 표시하세요.\n",
    ]
    return json.dumps(nb, ensure_ascii=False, indent=1)


def kit_readme(kit: dict) -> str:
    """묶음에 넣는 안내 글 — 양식 값이 들어간 실행 순서."""
    singer, fast = kit["singer"], " --fast" if kit["fast"] else ""
    if kit["mode"] == "A":
        prep = [
            f"1. **내 PC**: `run.bat` 을 더블클릭하고 음원 폴더와 화자 이름 `{singer}` 를 입력합니다"
            + (" (빠른 모델을 물으면 `y` — GPU 가 없는 PC)." if kit["fast"] else " (빠른 모델을 물으면 `N` — 품질 우선 모델).")
            + f"\n   명령으로 하려면: `python preprocess.py --input <음원 폴더> --singer {singer}{fast}`"
            + f"\n   → `dataset_raw/{singer}/` 에 5~15초 클립이 생깁니다. 처음 실행하면 필요한 패키지를 스스로 설치합니다 (수백 MB, 인터넷 필요).",
            "2. `dataset_raw` 폴더를 `dataset_raw.zip` 으로 묶습니다. (Agent Office 의 '🔍 데이터 점검'에 올리면 분량·클립 길이를 확인할 수 있습니다)",
            "3. `dataset_raw.zip` 을 Google Drive `MyDrive/svc_project/` 에 올립니다.",
            f"4. **Colab**: `{NOTEBOOK}` 을 열고 `런타임 → 런타임 유형 변경 → T4 GPU` 를 고른 뒤 1~4번 셀을 실행합니다. 5~6번 셀(B안)은 건너뜁니다.",
        ]
    else:
        prep = [
            "1. 원본 곡(mp3·wav)을 `songs.zip` 으로 묶어 Google Drive `MyDrive/svc_project/` 에 올립니다.",
            f"2. **Colab**: `{NOTEBOOK}` 을 열고 `런타임 → 런타임 유형 변경 → T4 GPU` 를 고른 뒤 1~3번 셀을 실행합니다. 4번 셀(A안)은 건너뜁니다.",
            f"3. 5~6번 셀(B안)을 실행합니다 — Colab GPU 로 보컬을 분리하고 `dataset_raw/{singer}/` 로 자릅니다 (화자 이름은 이미 넣어 두었습니다).",
        ]
    n = len(prep)
    return "\n".join([
        f"# 내 맞춤 SVC 실행 안내 — 화자 `{singer}`",
        "",
        "Agent Office 🎙 음성 합성 팀이 양식 값으로 만든 묶음입니다. 보컬 분리·학습·변환은 이 앱이 아니라 내 PC 와 Colab 에서 돕니다.",
        "",
        "## 먼저 확인한 것",
        f"- 목소리 출처: **{kit['source']}** — 학습·변환에 써도 된다고 양식에서 확인했습니다.",
        f"- 변환한 결과물을 남에게 보이거나 올릴 때는 **'{AI_LABEL}'** 이라고 표시합니다.",
        "- 다른 사람인 척 남을 속이는 데(전화·음성 메시지 등) 쓰지 않습니다.",
        "",
        "## 파일",
        "| 파일 | 용도 |",
        "|---|---|",
        f"| `{NOTEBOOK}` | Colab 학습·변환 노트북 — 화자 `{singer}` · epochs {kit['epochs']} · batch_size {kit['batch_size']} 를 넣어 둠 |",
        "| `preprocess.py` | 내 PC 전처리: wav 변환 → 보컬 분리 → 5~15초로 자르기 |",
        "| `run.bat` | `preprocess.py` 를 더블클릭으로 실행 (Python 3.10 이상이 설치돼 있어야 함) |",
        "",
        f"## 순서 ({kit['mode']}안)",
        *prep,
        f"{n + 1}. 7~9번 셀: 특징 추출 → 학습 설정(epochs {kit['epochs']} · batch_size {kit['batch_size']}) → 학습. "
        "체크포인트는 Drive 에 저장됩니다. 세션이 끊기면 1~3번 셀을 실행한 뒤 9번 셀만 다시 실행하면 이어집니다.",
        f"{n + 2}. 변환: 바꿀 노래의 **보컬만 분리한 wav** 를 `svc_project/cover.wav` 로 올리고 11번 셀을 실행합니다. "
        "키는 `-t` 값(반음)으로 맞춥니다 — 원래 부른 사람과 성별이 다르면 -12 또는 +12 를 먼저 해 봅니다.",
        f"{n + 3}. 12번 셀로 듣고, 반주는 오디오 편집기에서 변환된 보컬(`cover_converted.wav`)과 합칩니다.",
        "",
        "## 설정 메모 (코드가 규칙으로 정함)",
        *[f"- {note}" for note in kit["notes"]],
        "",
        "## 걸리는 시간 (원본 파일에 적힌 어림값만)",
        *[f"- {note}" for note in TIME_NOTES],
        "",
    ])


def kit_files(kit: dict) -> dict[str, bytes]:
    files = {NOTEBOOK: build_notebook(kit).encode("utf-8"), "README.md": kit_readme(kit).encode("utf-8")}
    files.update({name: (ASSET_DIR / name).read_bytes() for name in SCRIPT_FILES})
    return files


def kit_zip(kit: dict) -> bytes:
    """맞춤 노트북 + 안내 + 원본 스크립트 묶음 (.zip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in kit_files(kit).items():
            zf.writestr(name, data)
    return buf.getvalue()


def plan_evidence(kit: dict) -> str:
    """팀원에게 줄 근거 자료 — 코드가 정한 값. 팀은 이 값을 그대로 쓴다."""
    prep = "내 PC (A안)" + (" · 빠른 모델 --fast" if kit["fast"] else " · 품질 우선 모델") if kit["mode"] == "A" else "Colab (B안)"
    lines = [
        "### 🎛 코드가 정한 학습 설정 — 이 값을 그대로 쓴다 (바꿔 쓰지 않는다)",
        "| 항목 | 값 |", "|---|---|",
        f"| 목소리 출처 | {kit['source']} — 양식에서 사용해도 된다고 확인함 |",
        f"| 화자 이름 (폴더) | {kit['singer']} → dataset_raw/{kit['singer']}/ |",
        f"| 전처리 위치 | {prep} |",
        f"| epochs | {kit['epochs']} |",
        f"| batch_size | {kit['batch_size']} |",
        "",
        *[f"- {note}" for note in kit["notes"]],
        "",
        "### ⏱ 걸리는 시간 — 원본 파일에 적힌 어림값만 쓴다 (그 밖의 숫자는 '재 보지 않음')",
        *[f"- {note}" for note in TIME_NOTES],
    ]
    if kit["songs"] and kit["mode"] == "B":
        lines.append(f"- 이번 요청: 곡 {kit['songs']}개 × 곡당 약 1분 = 보컬 분리 약 {kit['songs']}분 (어림, 재 보지 않음)")
    lines += [
        "",
        "### 🎁 사용자가 받는 파일 (코드가 만듦 — 최종 결과 아래 '맞춤 실행 파일'에서 내려받는다)",
        f"- `{NOTEBOOK}` (위 화자 이름·epochs·batch_size 가 들어감) · `preprocess.py` · `run.bat` · `README.md`",
        f"- 실행 명령: `python preprocess.py --input <음원 폴더> --singer {kit['singer']}" + (" --fast`" if kit["fast"] else "`"),
    ]
    return "\n".join(lines)


# ─── 3) 데이터 점검 (dataset_raw.zip) ───────────────────────────────────

MAX_ZIP_BYTES = 200 * MB  # 올린 zip 크기 (Streamlit 올리기 한도와 같음)
MAX_TOTAL_BYTES = 2048 * MB  # 풀린 크기 합 — 넘으면 점검하지 않는다
MAX_CLIP_BYTES = 100 * MB  # wav 하나 — 넘으면 그 파일만 건너뛴다
MAX_ENTRIES = 20000
FRAME_SEC = 0.010  # slice_wav 와 같은 10ms 프레임
SILENT_RATIO = 0.10  # 소리 나는 프레임이 이보다 적으면 '거의 무음'
CLIP_LEVEL = 0.999  # 이 이상인 샘플이 여럿이면 '클리핑 의심'
LONG_SLACK = 0.05  # 15.0초로 딱 잘린 클립을 '초과'로 세지 않게


@lru_cache(maxsize=1)
def slice_defaults() -> dict:
    """preprocess.py 의 slice_wav 기본값 {threshold_db, min_keep, max_len} — 스크립트를 실행하지 않고 읽기만 한다."""
    tree = ast.parse((ASSET_DIR / "preprocess.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "slice_wav")
    names = [a.arg for a in fn.args.args][-len(fn.args.defaults):]
    values = dict(zip(names, (ast.literal_eval(d) for d in fn.args.defaults)))
    return {k: float(values[k]) for k in ("threshold_db", "min_keep", "max_len")}


def _entry_name(info: zipfile.ZipInfo) -> str:
    """zip 안 경로 ('/' 로 통일). Windows 에서 묶은 한글 이름(UTF-8 표시 없음)은 cp949 로 다시 읽는다."""
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name.replace("\\", "/")


def scan_zip(path: str | Path) -> list[tuple[zipfile.ZipInfo, str]]:
    """zip 안전 확인 → [(항목, 경로)]. 크기 초과·폴더 밖 경로·깨진 파일이면 VoiceError (아무것도 풀지 않는다)."""
    path = Path(path)
    if not path.exists():
        raise VoiceError("올린 zip 파일을 찾지 못했습니다 — 다시 올려 주세요.")
    size = path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise VoiceError(f"zip 파일이 너무 큽니다 ({size / MB:.0f}MB) — {MAX_ZIP_BYTES // MB}MB 까지 점검합니다. "
                         "화자 폴더 하나만 묶거나 클립 일부만 묶어 다시 올려 주세요.")
    try:
        with zipfile.ZipFile(path) as zf:
            entries = [(i, _entry_name(i)) for i in zf.infolist() if not i.is_dir()]
    except (zipfile.BadZipFile, OSError) as e:
        raise VoiceError("zip 파일을 열 수 없습니다 — `dataset_raw` 폴더를 zip 으로 다시 묶어 올려 주세요.") from e
    if len(entries) > MAX_ENTRIES:
        raise VoiceError(f"zip 안에 파일이 너무 많습니다 ({len(entries):,}개) — {MAX_ENTRIES:,}개까지 점검합니다.")
    for _, name in entries:
        if name.startswith("/") or re.match(r"[A-Za-z]:", name) or ".." in name.split("/"):
            raise VoiceError(f"zip 안에 폴더 밖을 가리키는 경로가 있어 점검하지 않았습니다: {name[:80]} — `dataset_raw` 폴더만 다시 묶어 올려 주세요.")
    total = sum(i.file_size for i, _ in entries)
    if total > MAX_TOTAL_BYTES:
        raise VoiceError(f"압축을 풀면 너무 큽니다 ({total / MB:,.0f}MB) — {MAX_TOTAL_BYTES // MB:,}MB 까지 점검합니다. 클립 일부만 묶어 다시 올려 주세요.")
    return entries


def _singer_of(parts: tuple[str, ...], issues: set[str]) -> str:
    """경로에서 화자 폴더를 찾고, dataset_raw/<화자>/*.wav 와 다르면 issues 에 적는다."""
    if "dataset_raw" not in parts[:-1]:
        issues.add("맨 위 폴더가 `dataset_raw` 가 아니다 — 노트북 4번 셀은 `/content/dataset_raw/<화자>/` 를 읽는다. `dataset_raw` 폴더째 묶어야 한다")
        return parts[-2] if len(parts) >= 2 else "(폴더 없음)"
    at = parts.index("dataset_raw")
    if at:
        issues.add("`dataset_raw` 가 다른 폴더 안에 들어 있다 — `dataset_raw` 가 zip 의 맨 위에 오게 묶어야 한다")
    rest = parts[at + 1:]
    if len(rest) == 1:
        issues.add("화자 폴더 없이 `dataset_raw` 바로 아래에 wav 가 있다 — `dataset_raw/<화자>/` 안에 넣어야 한다")
        return "(폴더 없음)"
    if len(rest) > 2:
        issues.add("화자 폴더 안에 하위 폴더가 있다 — 클립은 `dataset_raw/<화자>/` 바로 아래에 두는 구조다")
    return rest[0]


def read_wav(data: bytes):
    """wav 바이트 → (샘플 배열 [길이, 채널] −1~1, 샘플레이트). 표준 라이브러리 wave 로 정수 PCM(preprocess.py 가 쓰는 16비트 포함)을 읽고,
    소수점 형식 wav 는 soundfile 이 설치돼 있을 때만 읽는다 (requirements 에 없는 패키지라 없어도 된다). 못 읽으면 ValueError."""
    import wave

    import numpy as np

    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, sr, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        try:
            import soundfile as sf
        except ImportError:
            raise ValueError(f"읽을 수 없는 wav 형식: {e}") from e
        try:
            return sf.read(io.BytesIO(data), always_2d=True)
        except Exception as e2:  # noqa: BLE001
            raise ValueError(str(e2)) from e2
    if width == 1:
        x = (np.frombuffer(frames, np.uint8).astype(np.float64) - 128) / 128
    elif width == 3:
        raw = np.frombuffer(frames, np.uint8).reshape(-1, 3)
        x = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int8).astype(np.int32) << 16)) / 2 ** 23
    elif width in (2, 4):
        x = np.frombuffer(frames, f"<i{width}").astype(np.float64) / 2 ** (8 * width - 1)
    else:
        raise ValueError(f"지원하지 않는 샘플 크기: {width}바이트")
    if not sr or not channels:
        raise ValueError("샘플레이트·채널 정보가 없다")
    return x[: len(x) // channels * channels].reshape(-1, channels), sr


def _clip_stats(audio, sr: int, threshold_db: float) -> dict:
    """클립 하나 — 길이·소리 나는 프레임 비율(slice_wav 와 같은 10ms 프레임 RMS)·클리핑 의심."""
    import numpy as np

    mono = audio.mean(axis=1)
    hop = max(1, int(sr * FRAME_SEC))
    n = len(mono) // hop
    voiced = 0.0
    if n:
        rms = np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        voiced = float((20 * np.log10(rms + 1e-12) > threshold_db).mean())
    hot = int((np.abs(audio) >= CLIP_LEVEL).sum())
    return {"sec": len(mono) / sr, "voiced": voiced, "clipped": hot >= max(5, int(audio.size * 0.0001))}


def inspect_zip(path: str | Path, name: str = "") -> dict:
    """dataset_raw.zip 점검 — 화자 폴더별 클립 수·길이·샘플레이트·무음·클리핑. 디스크에 풀지 않고 wav 만 메모리에서 읽는다."""
    cfg = slice_defaults()
    entries = scan_zip(path)
    issues: set[str] = set()
    singers: dict[str, dict] = {}
    ignored = skipped = read = 0
    with zipfile.ZipFile(path) as zf:
        for info, entry in entries:
            parts = PurePosixPath(entry).parts
            if not entry.lower().endswith(".wav"):
                ignored += 1
                continue
            if info.file_size > MAX_CLIP_BYTES:
                skipped += 1
                continue
            with zf.open(info) as f:
                data = f.read(MAX_CLIP_BYTES + 1)  # 적힌 크기를 믿지 않고 읽는 양에도 한도를 둔다
            read += len(data)
            if read > MAX_TOTAL_BYTES:
                raise VoiceError(f"압축을 풀면 너무 큽니다 — {MAX_TOTAL_BYTES // MB:,}MB 까지 점검합니다. 클립 일부만 묶어 다시 올려 주세요.")
            try:
                audio, sr = read_wav(data)
            except ValueError:  # 깨졌거나 읽지 못하는 wav 는 세기만 하고 넘어간다
                skipped += 1
                continue
            if len(data) > MAX_CLIP_BYTES or not len(audio):
                skipped += 1
                continue
            clip = _clip_stats(audio, sr, cfg["threshold_db"])
            row = singers.setdefault(_singer_of(parts, issues), {"secs": [], "silent": 0, "clipped": 0, "rates": set(), "channels": set()})
            row["secs"].append(clip["sec"])
            row["silent"] += clip["voiced"] < SILENT_RATIO
            row["clipped"] += clip["clipped"]
            row["rates"].add(int(sr))
            row["channels"].add(int(audio.shape[1]))
    rows = []
    for singer, row in sorted(singers.items()):
        secs = row["secs"]
        minutes = round(sum(secs) / 60, 1)
        rows.append({
            "name": singer, "clips": len(secs), "minutes": minutes,
            "min_sec": round(min(secs), 1), "median_sec": round(median(secs), 1), "max_sec": round(max(secs), 1),
            "short": sum(s < cfg["min_keep"] for s in secs), "long": sum(s > cfg["max_len"] + LONG_SLACK for s in secs),
            "silent": int(row["silent"]), "clipped": int(row["clipped"]),
            "sample_rates": sorted(row["rates"]), "channels": sorted(row["channels"]), "verdict": verdict(minutes),
        })
    if not rows:
        issues.add("읽을 수 있는 wav 클립이 하나도 없다 — `preprocess.py` 가 만든 `dataset_raw` 폴더를 묶었는지 확인한다")
    if len({r for row in rows for r in row["sample_rates"]}) > 1:
        issues.add("샘플레이트가 섞여 있다 — 학습 전에 `svc pre-resample` 이 맞춰 주지만, 같은 방법으로 만든 클립인지 확인한다")
    total = round(sum(sum(s["secs"]) for s in singers.values()) / 60, 1)
    worst = min((r["verdict"] for r in rows), key=VERDICTS.index, default=VERDICTS[0])
    return {"file": name or Path(path).name, "singers": rows, "clips": sum(r["clips"] for r in rows), "minutes": total,
            "verdict": worst, "issues": sorted(issues), "ignored": ignored, "skipped": skipped,
            "min_keep": cfg["min_keep"], "max_len": cfg["max_len"], "threshold_db": cfg["threshold_db"]}


def dataset_table(stats: dict) -> str:
    """점검 표 (마크다운) — 숫자는 모두 코드 계산값."""
    lo, hi = f"{stats['min_keep']:g}초 미만", f"{stats['max_len']:g}초 초과"
    lines = [
        f"| 화자 폴더 | 클립 수 | 총 길이 | 클립 길이 (최소 · 중앙 · 최대) | {lo} | {hi} | 거의 무음 | 클리핑 의심 | 샘플레이트 | 채널 | 분량 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in stats["singers"]:
        lines.append(
            f"| {r['name']} | {r['clips']:,}개 | {r['minutes']:.1f}분 | {r['min_sec']:.1f} · {r['median_sec']:.1f} · {r['max_sec']:.1f}초 "
            f"| {r['short']}개 | {r['long']}개 | {r['silent']}개 | {r['clipped']}개 | {' · '.join(f'{x}Hz' for x in r['sample_rates'])} "
            f"| {' · '.join(str(x) for x in r['channels'])} | {r['verdict']} |")
    lines += [
        "",
        f"- 전체: 클립 {stats['clips']:,}개 · {stats['minutes']:.1f}분 · 분량 판정 **{stats['verdict']}** "
        f"(화자별 보컬 {MIN_MINUTES}분 미만 부족 · {GOOD_MINUTES}분 미만 주의 · 그 이상 충분 — preprocess.py 의 안내 기준)",
        "- 폴더 구조: " + ("맞음 (`dataset_raw/<화자>/*.wav`)" if not stats["issues"] else "확인 필요"),
        *[f"  - ⚠️ {issue}" for issue in stats["issues"]],
    ]
    if stats["ignored"] or stats["skipped"]:
        lines.append(f"- wav 가 아니어서 세지 않은 파일 {stats['ignored']}개 · 읽지 못했거나 너무 큰 wav {stats['skipped']}개")
    lines.append(f"- 기준: 소리 크기 {stats['threshold_db']:g}dB 보다 큰 10ms 구간이 {SILENT_RATIO:.0%} 미만이면 '거의 무음', "
                 f"최댓값({CLIP_LEVEL})에 닿은 샘플이 여럿이면 '클리핑 의심'. 자르는 기준(−40dB · {stats['min_keep']:g}초 · {stats['max_len']:g}초)은 preprocess.py 와 같다.")
    return "\n".join(lines)


# ─── 4) 하네스 검사 ────────────────────────────────────────────────────


def check_guard(answer: str, options: dict) -> list[str]:
    """모든 답에: 실존 가수 목소리를 대상으로 적지 않았는지 + 범위 밖 요청에 실행 방법을 주지 않았는지."""
    problems = []
    names = voice_targets(answer)
    if names:
        problems.append(f"실존 가수·유명인의 목소리를 대상으로 적었다: {names}. 본인 동의 없는 목소리 복제는 돕지 않는다 — "
                        "이름을 되풀이하지 말고 '요청하신 가수'처럼 쓰고, 그 목소리의 학습·변환 방법은 안내하지 마라")
    blocked = options.get("voice_blocked")
    if blocked and RUN_STEPS.search(answer):
        problems.append(f"이 요청은 범위 밖이다 ({blocked}). 그런데 답에 실행 방법(명령·스크립트·데이터 폴더)이 들어 있다 — "
                        "실행 방법을 모두 빼고, 왜 돕지 않는지와 무엇을 하면 되는지(내 목소리 또는 본인 동의를 받은 목소리로 다시 맡기기)만 안내하라")
    return problems


def _numbers_after(pattern: str, text: str) -> list[int]:
    return [int(m.group(1).replace(",", "")) for m in re.finditer(pattern, text, re.IGNORECASE)]


def check_plan(answer: str, options: dict) -> list[str]:
    """학습 계획 — 출처·동의 문장, AI 합성 표시, 네 단계, 코드가 정한 값(화자·epochs·batch_size)과 같은지."""
    kit = options.get("voice_kit")
    if options.get("voice_blocked") or not kit:
        return []  # 범위 밖 요청은 check_guard 가 본다
    problems = []
    if not any(kit["source"] in line and re.search("확인|동의|허락", line) for line in answer.splitlines()):
        problems.append(f"목소리 출처와 사용 확인을 한 줄로 밝혀라 (예: '목소리 출처: {kit['source']} — 학습·변환에 써도 된다고 확인함')")
    if not (re.search(r"AI\s*(?:로|가|를 써서)?\s*(?:합성|생성|변환)", answer) and "표시" in answer):
        problems.append(f"결과물을 남에게 보이거나 올릴 때 '{AI_LABEL}'이라고 표시하라는 안내가 없다")
    stages = {"전처리": "전처리", "Drive 에 올리기": "업로드|올리|올린|올립", "학습": "학습", "변환": r"svc infer|음색 변환|변환 실행|변환합|변환한|변환할|변환\b"}
    missing = [name for name, pattern in stages.items() if not re.search(pattern, answer)]
    if missing:
        problems.append(f"실행 단계가 빠졌다: {missing} — 전처리 → Drive 에 올리기 → 학습 → 변환 순서로 모두 써라")
    if kit["singer"] not in answer:
        problems.append(f"화자 이름 `{kit['singer']}` 이 답에 없다 — 명령·폴더 경로에 이 이름을 그대로 써라")
    wrong = sorted({s for s in re.findall(r"(?:--singer\s+|SINGER\s*=\s*['\"]|dataset_raw/)([A-Za-z0-9_-]+)", answer) if s != kit["singer"]})
    if wrong:
        problems.append(f"화자 이름이 코드가 만든 파일과 다르다: {wrong} → `{kit['singer']}` 로 써라")
    for label, key, pattern in (("epochs", "epochs", r"epochs?\b['\"\]\s]*[=:]\s*([\d,]+)"),
                                ("batch_size", "batch_size", r"batch(?:_| )?size\b['\"\]\s]*[=:]\s*([\d,]+)")):
        if not re.search(rf"{label.replace('_', '(?:_| )?')}[^0-9\n]{{0,20}}{kit[key]}(?!\d)", answer, re.IGNORECASE):
            problems.append(f"{label} 값 {kit[key]} 이 답에 없다 — 코드가 정한 값을 그대로 써라")
        bad = sorted({n for n in _numbers_after(pattern, answer) if n != kit[key]})
        if bad:
            problems.append(f"{label} 를 {bad} 로 적었다 — 코드가 만든 노트북의 값은 {kit[key]} 이다. 같은 값으로 써라")
    return problems


def check_data(answer: str, options: dict) -> list[str]:
    """데이터 점검 — 답의 숫자(총 클립 수·총 길이)와 분량 판정이 코드가 계산한 표와 같은지."""
    stats = options.get("voice_dataset")
    if options.get("voice_blocked"):
        return []
    if not stats:  # 점검하지 못함 — 숫자를 지어내면 안 된다
        made_up = re.search(r"\d[\d,]*\s*개(?:의)?\s*클립|클립\s*(?:수)?\s*\d[\d,]*\s*개|총\s*\d+(?:\.\d+)?\s*분", answer)
        return ["데이터를 점검하지 못했는데 클립 수·총 길이 숫자를 적었다 — 숫자를 빼고, 왜 점검하지 못했는지와 다시 올리는 방법만 안내하라"] if made_up else []
    problems = []
    plain = answer.replace(",", "")
    if not re.search(rf"(?<![\d.]){stats['clips']}\s*개", plain):
        problems.append(f"전체 클립 수 {stats['clips']:,}개가 답에 없다 — 점검 표의 값을 그대로 써라")
    if not re.search(rf"(?<![\d.]){stats['minutes']:.1f}\s*분", plain):
        problems.append(f"전체 길이 {stats['minutes']:.1f}분이 답에 없다 — 점검 표의 값을 그대로 써라")
    ok_minutes = [stats["minutes"], *(r["minutes"] for r in stats["singers"])]
    for m in re.finditer(r"(?:총|전체|합계)[^\n\d]{0,12}(\d+(?:\.\d+)?)\s*분", plain):
        if not any(abs(float(m.group(1)) - x) < 0.05 for x in ok_minutes):
            problems.append(f"'{m.group(0)}' — 점검 표에 없는 길이다 (표: 전체 {stats['minutes']:.1f}분). 표의 값으로 고쳐라")
    ok_counts = {stats["clips"]} | {r[k] for r in stats["singers"] for k in ("clips", "short", "long", "silent", "clipped")}
    for m in re.finditer(r"(?:총|전체|합계)[^\n\d]{0,12}(\d+)\s*개(?:의)?\s*클립|클립\s*(?:수)?[^\n\d]{0,6}(\d+)\s*개", plain):
        if int(m.group(1) or m.group(2)) not in ok_counts:
            problems.append(f"'{m.group(0)}' — 점검 표에 없는 클립 수다 (표: 전체 {stats['clips']:,}개). 표의 값으로 고쳐라")
    if stats["verdict"] not in answer:
        problems.append(f"분량 판정 '{stats['verdict']}' 이 답에 없다 — 코드의 판정을 그대로 밝히고 다음 조치를 써라")
    return list(dict.fromkeys(problems))
