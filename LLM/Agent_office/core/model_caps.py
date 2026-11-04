"""모델 능력 판단 — '이 모델이 이 일을 할 수 있는가'와 '제공처에서 사라진 모델'을 한곳에서 판단한다.

왜 필요한가
  - 텍스트 전용 모델(gpt-oss·Solar 등)에 사진을 보내면 실패한다. 화면에서 골라도, 대안 모델로 넘어가다 걸려도 마찬가지.
  - 제공처가 지운 모델이 선택지에 남아 있으면 고른 순간부터 호출이 404 로 실패한다.

능력 (Caps): vision(사진 입력) · structured_output(JSON 스키마 답) · streaming · context(대략의 입력 토큰) · text_only(= vision False)
판단 순서 (뒤가 앞을 덮는다 — 사람이 지정한 값이 가장 우선):
  1) 이름 규칙 NAME_RULES          2) config/model_caps.yaml (근거 적음)
  3) 제공처 목록 API 메타데이터     4) 🧪 사진 시험·연결 시험 결과      5) 사람이 지정 (화면 '👁 모델 능력')
  3)~5) 는 데이터 폴더(AGENT_OFFICE_DATA_DIR)의 model_catalog.json · model_caps.json 에서 읽는다. yaml 은 안 고친다.

일(JobNeeds)이 필요한 능력을 말하면, 그 능력이 없는 모델은
  - 화면 선택지에서 숨긴다 → selectable_models(needs, keys) = (보일 모델, 숨긴 모델+이유)
  - 실행 후보에서 건너뛴다 → runtime_specs(specs, needs) (llm_factory.available_candidates 가 부른다)
  사진을 읽는 일은 '모름'도 숨긴다 (보내 보기 전엔 알 수 없으니 🧪 사진 시험으로 확인). 그 밖의 능력은 '모름'이면 허용한다.

사라진 모델 (removal)
  - 제공처 목록을 '최근(TTL 안)에 성공적으로' 받았는데 그 목록에 없으면 사라진 것으로 본다 → 숨김·건너뜀.
  - 목록 받기가 실패했거나(오프라인) 오래됐으면 아무것도 숨기지 않는다 (잘못 숨기지 않게).
  - 목록 API 가 없는 provider(claude_cli)와 사람이 직접 입력한 모델은 숨기지 않는다 (⚠️ 표시만 — 사람이 넣은 것이 우선).
  - '추천 대체 모델로 바꾸기' = model_caps.json 의 replace 에 기록 → 실행 때 사라진 모델 자리에 대체 모델을 쓴다 (yaml 은 그대로).

다른 화면·기능에서 쓰는 법 (한 함수로 거른다)
    from core import model_caps
    shown, hidden = model_caps.selectable_models(model_caps.IMAGE_READING, keys)      # 선택지
    specs, notes = model_caps.runtime_specs(rank_models.candidates, model_caps.IMAGE_READING)  # 실행 후보
  메시지에 사진이 들어 있으면 llm_factory 가 알아서 vision 을 요구한다 (needs_for_messages) — 따로 안 넘겨도 된다.
"""

import base64
import fnmatch
import io
import json
import os
import random
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from core.memo import SigCache, file_sig
from core.schemas import ModelsConfig, ModelSpec

CAPS_FILE = "model_caps.json"  # 데이터 폴더 — {"probe": {...}, "user": {...}, "replace": {...}}
FIELDS = ("vision", "structured_output", "streaming", "context")
SOURCES = ("name", "yaml", "api", "probe", "user")  # 뒤로 갈수록 우선
SOURCE_LABELS = {
    "name": "이름 규칙", "yaml": "능력표(config/model_caps.yaml)", "api": "제공처 목록 API",
    "probe": "🧪 시험 결과", "user": "사람이 지정",
}
FIXED_LIST = {"claude_cli"}  # 목록 API 가 없는 provider — 사라짐 판단 안 함 (model_catalog.FIXED_LIST 와 같음)


# ─── 일이 필요로 하는 능력 ────────────────────────────────────────────


@dataclass(frozen=True)
class JobNeeds:
    """어떤 일이 모델에게 요구하는 능력. label 은 숨긴 이유에 쓰는 일 이름."""

    vision: bool = False
    structured_output: bool = False
    streaming: bool = False
    min_context: int = 0
    label: str = ""

    def __or__(self, other: "JobNeeds") -> "JobNeeds":
        return JobNeeds(self.vision or other.vision, self.structured_output or other.structured_output,
                        self.streaming or other.streaming, max(self.min_context, other.min_context),
                        self.label or other.label)

    @property
    def empty(self) -> bool:
        return not (self.vision or self.structured_output or self.streaming or self.min_context)


ANY = JobNeeds()  # 능력 조건 없음 — 사라진 모델만 거른다
TEAM_WORK = JobNeeds(structured_output=True, label="팀 업무")  # 팀원 답변·검토·통합 (구조화 출력)
IMAGE_READING = JobNeeds(vision=True, structured_output=True, label="사진 읽기")  # 도면 그림·JD 이미지 등
LONG_CONTEXT = 200_000  # '아주 긴 통합'에 쓸 만한 문맥 기준 (필요한 곳에서 JobNeeds(min_context=...) 로)


def needs_for_messages(messages, base: JobNeeds = ANY) -> JobNeeds:
    """메시지에 사진 블록(image_url·image)이 있으면 vision 을 더한다."""
    return base | JobNeeds(vision=True, label=base.label or "사진 읽기") if has_images(messages) else base


def has_images(messages) -> bool:
    for m in messages or []:
        content = getattr(m, "content", m)
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in ("image_url", "image", "input_image") for b in content
        ):
            return True
    return False


# ─── 능력 (한 모델) ──────────────────────────────────────────────────


@dataclass
class Caps:
    key: str
    vision: bool | None = None
    structured_output: bool | None = None
    streaming: bool | None = None
    context: int | None = None
    source: dict[str, str] = field(default_factory=dict)  # 능력 → 출처 (SOURCES 중 하나)
    note: dict[str, str] = field(default_factory=dict)  # 능력 → 근거 한 줄

    @property
    def text_only(self) -> bool:
        return self.vision is False

    def why(self, name: str) -> str:
        src = SOURCE_LABELS.get(self.source.get(name, ""), "")
        note = self.note.get(name, "")
        return " — ".join(x for x in (src, note) if x)

    def apply(self, values: dict, source: str, note: str = "") -> None:
        for name in FIELDS:
            value = values.get(name)
            if value is None:
                continue
            setattr(self, name, int(value) if name == "context" else bool(value))
            self.source[name] = source
            if note:
                self.note[name] = note
            else:
                self.note.pop(name, None)


# 이름 규칙 — (provider 와일드카드, 모델 이름 정규식, 능력, 근거). 능력표(yaml)에 없는 새 모델에 쓴다.
NAME_RULES: list[tuple[str, str, dict, str]] = [
    ("*", r"gpt-oss", {"vision": False}, "gpt-oss 계열은 텍스트 전용"),
    ("*", r"^solar", {"vision": False}, "Solar 대화 모델은 텍스트 전용"),
    ("*", r"(^|[-_/])(qwq|deepseek-r1|llama-3\.[13]|mistral-small-3\.0)", {"vision": False}, "텍스트 전용 계열"),
    ("google", r"^(gemini|gemma-3|gemma-4)", {"vision": True}, "Gemini·Gemma 3+ 는 사진 입력 지원"),
    ("openai", r"^(gpt-(4o|4\.1|4-turbo|[5-9])|o[134])", {"vision": True}, "GPT-4o·4.1·4 Turbo·5 이후·o 계열은 사진 입력 지원"),
    ("openai", r"^(gpt-3\.5|gpt-4$|gpt-4-\d{4})", {"vision": False}, "GPT-3.5·초기 GPT-4 는 텍스트 전용"),
    ("anthropic", r"^claude", {"vision": True}, "Claude 계열은 사진 입력 지원"),
    ("claude_cli", r".*", {"vision": True}, "Claude CLI 는 stream-json 입력으로 사진 전달"),
]
PROVIDER_DEFAULTS = {p: {"streaming": True} for p in ("openai", "google", "anthropic", "upstage", "freellm", "claude_cli")}


def name_caps(key: str) -> list[tuple[dict, str]]:
    provider, _, model = key.partition(":")
    low = model.lower().rsplit("/", 1)[-1]
    out = [(PROVIDER_DEFAULTS.get(provider, {}), "")]
    for prov, pattern, values, note in NAME_RULES:
        if fnmatch.fnmatch(provider, prov) and re.search(pattern, low):
            out.append((values, note))
            break  # 첫 규칙만 (위가 더 구체적)
    return out


# ─── 능력표(yaml) ────────────────────────────────────────────────────


_yaml_cache = SigCache()


def yaml_path() -> Path:
    from core.config import CONFIG_DIR

    return CONFIG_DIR / "model_caps.yaml"


def yaml_rules(path: Path | None = None) -> list[dict]:
    path = path or yaml_path()
    if not path.exists():
        return []

    def read() -> list[dict]:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [r for r in data.get("models") or [] if isinstance(r, dict) and r.get("match")]

    return _yaml_cache.get(str(path), file_sig(path), read)


def _api_values(meta: dict | None) -> dict:
    """model_catalog.json 의 caps 메타 → 능력 (input_modalities·context)."""
    if not isinstance(meta, dict):
        return {}
    out: dict = {}
    mods = meta.get("input_modalities")
    if isinstance(mods, list) and mods:
        out["vision"] = any(str(m).lower() in ("image", "images", "vision") for m in mods)
    if isinstance(meta.get("vision"), bool):
        out["vision"] = meta["vision"]
    if meta.get("context"):
        try:
            out["context"] = int(meta["context"])
        except (TypeError, ValueError):
            pass
    return out


# ─── 데이터 폴더의 기록 (model_caps.json) ─────────────────────────────


_lock = threading.Lock()
_store_cache = SigCache()


class CapsStore:
    """{"probe": {key: {"vision"|"structured_output": bool, "at": ts, "detail": str}},
        "user": {key: {"vision": bool, ...}}, "replace": {사라진 key: 대체 key}}"""

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / CAPS_FILE

    def read(self) -> dict:
        def load() -> dict:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            return data if isinstance(data, dict) else {}

        data = _store_cache.get(str(self.path), file_sig(self.path), load) if self.path.exists() else {}
        return {s: dict(data.get(s) or {}) for s in ("probe", "user", "replace")}

    def _write(self, data: dict) -> None:
        with _lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def _update(self, section: str, key: str, value) -> None:
        data = self.read()
        if value is None:
            data[section].pop(key, None)
        else:
            data[section][key] = value
        self._write(data)

    def record_probe(self, key: str, values: dict, detail: str = "", now: float | None = None) -> None:
        old = self.read()["probe"].get(key) or {}
        self._update("probe", key, {**old, **values, "at": time.time() if now is None else now, "detail": detail[:200]})

    def set_user(self, key: str, name: str, value: bool | None) -> None:
        """사람이 지정 (value None = 자동으로 되돌림)."""
        current = dict(self.read()["user"].get(key) or {})
        if value is None:
            current.pop(name, None)
        else:
            current[name] = value
        self._update("user", key, current or None)

    def set_replacement(self, removed: str, new: str | None) -> None:
        self._update("replace", removed, new or None)

    def replacements(self) -> dict[str, str]:
        return {k: v for k, v in self.read()["replace"].items() if isinstance(v, str) and v}


# ─── 상태: 기본 데이터 폴더 ──────────────────────────────────────────


_data_dir: Path | None = None
_UNSET = object()


def configure(data_dir: Path | str | None) -> None:
    """앱이 시작할 때 데이터 폴더를 알려 준다 → 실행 후보 거르기(runtime_specs)가 목록·시험 결과를 쓴다.
    None(테스트 기본) = 이름 규칙·능력표만, 사라진 모델 판단 안 함."""
    global _data_dir
    _data_dir = Path(data_dir) if data_dir else None


def _dir(data_dir) -> Path | None:
    return _data_dir if data_dir is _UNSET else (Path(data_dir) if data_dir else None)


def _catalog(data_dir: Path | None) -> dict:
    if not data_dir:
        return {}
    from core import model_catalog

    return model_catalog.load_catalog(data_dir)


def _store(data_dir: Path | None) -> dict:
    return CapsStore(data_dir).read() if data_dir else {"probe": {}, "user": {}, "replace": {}}


def caps_for(key: str, data_dir=_UNSET) -> Caps:
    """한 모델의 능력 — 이름 규칙 < 능력표 < 목록 API < 시험 결과 < 사람이 지정."""
    data_dir = _dir(data_dir)
    caps = Caps(key)
    for values, note in name_caps(key):
        caps.apply(values, "name", note)
    for rule in yaml_rules():
        if fnmatch.fnmatch(key, str(rule["match"])):
            caps.apply(rule, "yaml", str(rule.get("source") or ""))
    provider, _, model = key.partition(":")
    meta = ((_catalog(data_dir).get("providers") or {}).get(provider) or {}).get("models", {}).get(model) or {}
    caps.apply(_api_values(meta.get("caps")), "api", "목록 API 메타데이터")
    store = _store(data_dir)
    probe = store["probe"].get(key) or {}
    if probe:
        when = time.strftime("%Y-%m-%d", time.localtime(float(probe.get("at") or 0)))
        caps.apply({k: probe.get(k) for k in ("vision", "structured_output")}, "probe",
                   f"{when} {probe.get('detail') or ''}".strip())
    caps.apply(store["user"].get(key) or {}, "user")
    return caps


# ─── 사라진 모델 ─────────────────────────────────────────────────────


@dataclass
class Removal:
    key: str
    checked_at: float  # 목록을 받은 시각
    suggest: str = ""  # 이름으로 고른 비슷한 최신 모델
    replacement: str = ""  # 사람이 고른 대체 모델 (model_caps.json replace)


def _custom_keys(data_dir: Path | None) -> set[str]:
    if not data_dir:
        return set()
    from core import model_catalog

    return set(model_catalog.CustomStore(data_dir).keys())


def removal(key: str, data_dir=_UNSET, now: float | None = None) -> Removal | None:
    """제공처 목록에서 사라진 모델이면 Removal, 아니면 None.
    목록을 최근(TTL 안)에, 성공적으로 받은 경우에만 판단한다 — 실패·오래됨·목록 API 없음·직접 입력 모델은 None."""
    data_dir = _dir(data_dir)
    if not data_dir:
        return None
    from core import model_catalog

    provider, _, model = key.partition(":")
    if provider in FIXED_LIST:
        return None
    entry = (_catalog(data_dir).get("providers") or {}).get(provider) or {}
    fetched = float(entry.get("fetched_at") or 0)
    now = time.time() if now is None else now
    if not fetched or entry.get("error") or now - fetched > model_catalog.TTL_SEC:
        return None  # 이번 확인이 실패했거나 오래됨 → 잘못 숨기지 않는다
    found = entry.get("models") or {}
    all_ids = set(entry.get("all_ids") or found)
    if model in all_ids or key in _custom_keys(data_dir):
        return None
    replacement = _store(data_dir)["replace"].get(key) or ""
    return Removal(key, fetched, model_catalog._suggest(key, found), replacement)


# ─── 판단: 이 모델이 이 일을 할 수 있는가 ─────────────────────────────


def reasons(key: str, needs: JobNeeds, data_dir=_UNSET, now: float | None = None) -> list[str]:
    """이 모델을 이 일에 쓰면 안 되는 이유들 (빈 목록 = 써도 됨)."""
    data_dir = _dir(data_dir)
    out = []
    gone = removal(key, data_dir, now)
    if gone:
        day = time.strftime("%m-%d %H:%M", time.localtime(gone.checked_at))
        hint = (f" → 대체: {gone.replacement.partition(':')[2]}" if gone.replacement
                else f" (추천 대체: {gone.suggest.partition(':')[2]})" if gone.suggest else "")
        out.append(f"제공처 목록에서 사라짐 ({day} 확인){hint}")
    if needs.empty:
        return out
    caps = caps_for(key, data_dir)
    job = needs.label or "이 일"
    if needs.vision and caps.vision is not True:
        out.append(f"사진을 못 읽음 ({caps.why('vision')})" if caps.vision is False
                   else "사진 읽기 확인 안 됨 — '👁 모델 능력'의 🧪 사진 시험으로 확인")
    if needs.structured_output and caps.structured_output is False:
        out.append(f"형식(구조화) 출력 안 됨 — {job}에 필요 ({caps.why('structured_output')})")
    if needs.streaming and caps.streaming is False:
        out.append("스트리밍 안 됨")
    if needs.min_context and caps.context and caps.context < needs.min_context:
        out.append(f"문맥이 짧음 (약 {caps.context:,} < 필요 {needs.min_context:,} 토큰)")
    return out


def can(key: str, needs: JobNeeds, data_dir=_UNSET, now: float | None = None) -> bool:
    return not reasons(key, needs, data_dir, now)


@dataclass
class Hidden:
    key: str
    reasons: list[str]


def selectable_models(needs: JobNeeds, keys: Iterable[str], data_dir=_UNSET,
                      now: float | None = None) -> tuple[list[str], list[Hidden]]:
    """선택지 거르기 — (보일 모델 키, 숨긴 모델+이유). 순서는 그대로. 모든 모델 선택 화면이 이 함수를 쓴다.
    keys = "provider:model" 목록 (상하 관계 규칙·무료만·disabled_providers 는 부르는 쪽이 먼저 적용)."""
    shown, hidden = [], []
    for key in keys:
        why = reasons(key, needs, data_dir, now)
        if why:
            hidden.append(Hidden(key, why))
        else:
            shown.append(key)
    return shown, hidden


def runtime_specs(specs: Iterable[ModelSpec], needs: JobNeeds = ANY, data_dir=_UNSET,
                  now: float | None = None) -> tuple[list[ModelSpec], list[str]]:
    """실행 후보 거르기 — (쓸 후보, 건너뛴 기록). 사라진 모델은 대체 모델이 있으면 그 자리에 대체 모델을,
    없으면 건너뛴다. 능력이 없는 모델도 건너뛴다. 순서는 그대로, 같은 모델은 한 번만."""
    data_dir = _dir(data_dir)
    kept, notes, seen = [], [], set()
    for spec in specs:
        key = f"{spec.provider}:{spec.model}"
        gone = removal(key, data_dir, now)
        if gone and gone.replacement:
            provider, _, model = gone.replacement.partition(":")
            new = spec.model_copy(update={"provider": provider, "model": model})
            _notice(key, f"{spec.label}: 제공처 목록에서 사라짐 → 대체 모델 {model} 사용")
            notes.append(f"{spec.label}: 제공처 목록에서 사라짐 → 대체 모델 {model}")
            spec, key = new, gone.replacement
        why = reasons(key, needs, data_dir, now)
        if why:
            notes.append(f"{spec.label}: 건너뜀 — {why[0]}")
            if gone and not gone.replacement:
                _notice(key, f"{spec.label}: 제공처 목록에서 사라져 건너뜀"
                        + (f" — 추천 대체 {gone.suggest.partition(':')[2]}" if gone.suggest else ""))
            continue
        if key not in seen:
            seen.add(key)
            kept.append(spec)
    return kept, notes


_notices: dict[str, str] = {}


def _notice(key: str, text: str) -> None:
    _notices.setdefault(key, text)


def runtime_notices() -> dict[str, str]:
    """실행 중 사라진 모델을 건너뛰거나 대체한 기록 (한 번씩만) — 화면이 ⚠️ 알림으로 한 번 보여 준다."""
    return dict(_notices)


# ─── 저장된 팀원 선택 점검 ───────────────────────────────────────────


@dataclass
class ChoiceIssue:
    node_id: str
    key: str  # 저장된 선택
    reasons: list[str]
    use: str  # 대신 쓸 선택 ("" = 기본 = 직급의 다음 후보)


def check_choices(choices: dict[str, str], needs: JobNeeds = TEAM_WORK, data_dir=_UNSET,
                  now: float | None = None) -> tuple[dict[str, str], list[ChoiceIssue]]:
    """저장된 팀원 선택 중 사라졌거나 이 일을 못 하는 모델 → (실행에 쓸 선택, 경고 목록).
    저장값은 바꾸지 않는다 — 대체 모델이 있으면 그것으로, 없으면 기본(직급의 다음 후보)으로 '이번에만' 바꾼다."""
    data_dir = _dir(data_dir)
    use, issues = dict(choices), []
    for node_id, key in choices.items():
        why = reasons(key, needs, data_dir, now)
        if not why:
            continue
        gone = removal(key, data_dir, now)
        alt = gone.replacement if gone and gone.replacement and can(gone.replacement, needs, data_dir, now) else ""
        if alt:
            use[node_id] = alt
        else:
            use.pop(node_id, None)
        issues.append(ChoiceIssue(node_id, key, why, alt))
    return use, issues


def removed_defaults(models: ModelsConfig, keys: Iterable[str] | None = None, data_dir=_UNSET,
                     now: float | None = None) -> list[Removal]:
    """models.yaml(직급·보조·강도)에 쓰였는데 제공처에서 사라진 모델 — '추천 대체 모델로 바꾸기' 대상."""
    data_dir = _dir(data_dir)
    if keys is None:
        keys = {f"{s.provider}:{s.model}" for r in [*models.ranks.values(), models.utility] for s in r.candidates}
        keys |= {k for row in models.model_tiers for k in row}
    out = [r for k in sorted(keys) if (r := removal(k, data_dir, now))]
    return out


# ─── 🧪 사진 시험 ───────────────────────────────────────────────────


PROBE_LETTERS = "ABDEFGHJKMNPRTUWXYZ"  # 헷갈리는 글자(I·L·O·Q·S·C·V) 제외
PROBE_COLORS = [((255, 214, 0), (20, 40, 200)), ((0, 140, 70), (255, 255, 255)), ((200, 30, 30), (255, 255, 255))]


def probe_image(letter: str, colors=None) -> bytes:
    """두 색 PNG 에 큰 글자 하나 (128×128, 약 1KB)."""
    from PIL import Image, ImageDraw, ImageFont

    bg, fg = colors or PROBE_COLORS[0]
    img = Image.new("RGB", (128, 128), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 96)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
        except OSError:
            font = ImageFont.load_default(size=96)
    box = draw.textbbox((0, 0), letter, font=font)
    x = (128 - (box[2] - box[0])) / 2 - box[0]
    y = (128 - (box[3] - box[1])) / 2 - box[1]
    draw.text((x, y), letter, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PROBE_PROMPT = "What single capital letter is shown in this image? Reply with only that letter."
NOT_VISION_ERRORS = ("image", "vision", "multimodal", "multi-modal", "content type", "image_url", "unsupported content")


@dataclass
class ProbeResult:
    ok: bool | None  # True = 읽음, False = 못 읽음, None = 판단 못 함 (연결·한도 오류 — 기록 안 함)
    expected: str = ""
    reply: str = ""
    error: str = ""
    elapsed_sec: float = 0.0
    recorded: bool = False


def vision_probe(provider: str, model: str, models: ModelsConfig, data_dir=_UNSET, letter: str | None = None,
                 llm=None) -> ProbeResult:
    """작은 그림(글자 하나)을 보내 답을 확인한다 → vision 을 시험 결과로 기록 (데이터 폴더 model_caps.json).
    llm 을 주면 그 모델로 (테스트). 꺼진 provider 는 부르지 않는다. 토큰은 사용량에 '사진 시험'으로 기록된다."""
    from langchain_core.messages import HumanMessage

    from core import llm_factory, usage

    data_dir = _dir(data_dir)
    letter = letter or random.choice(PROBE_LETTERS)
    result = ProbeResult(ok=None, expected=letter)
    if llm is None:
        reason = llm_factory.unavailable_reason(provider, models)
        if reason:
            result.error = f"지금 쓸 수 없는 provider — {reason}"
            return result
    url = "data:image/png;base64," + base64.b64encode(probe_image(letter, random.choice(PROBE_COLORS))).decode()
    message = HumanMessage(content=[{"type": "text", "text": PROBE_PROMPT}, {"type": "image_url", "image_url": {"url": url}}])
    started = time.perf_counter()
    try:
        with usage.scope(purpose="사진 시험"):
            llm = llm or llm_factory.create_chat_model(ModelSpec(provider=provider, model=model), models)
            with llm_factory.provider_slot(provider):
                reply = llm.invoke([message])
        content = getattr(reply, "content", reply)
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])
        result.reply = text.strip()[:100]
        found = re.findall(r"\b[A-Z]\b", result.reply.upper())
        result.ok = bool(found) and found[0] == letter
    except Exception as e:  # noqa: BLE001 — provider 별 예외
        result.error = f"{type(e).__name__}: {str(e)[:300]}"
        low = result.error.lower()
        if not llm_factory.is_transient(e) and any(w in low for w in NOT_VISION_ERRORS):
            result.ok = False  # '사진은 못 받는다'는 오류 — 텍스트 전용 모델
    result.elapsed_sec = round(time.perf_counter() - started, 2)
    if result.ok is not None and data_dir:
        detail = (f"글자 {letter} → 답 “{result.reply[:20]}”" if not result.error else f"오류: {result.error[:120]}")
        CapsStore(data_dir).record_probe(f"{provider}:{model}", {"vision": result.ok}, detail)
        result.recorded = True
    return result


def record_connection_test(key: str, structured_ok: bool | None, data_dir=_UNSET) -> None:
    """🧪 연결 시험의 구조화 출력 결과를 능력으로 기록 (None = 시험 안 함 → 기록 안 함)."""
    data_dir = _dir(data_dir)
    if structured_ok is None or not data_dir:
        return
    CapsStore(data_dir).record_probe(key, {"structured_output": bool(structured_ok)},
                                     "연결 시험 구조화 출력 " + ("성공" if structured_ok else "실패"))


def summary(key: str, data_dir=_UNSET) -> str:
    """'👁 사진 O · 📝 형식 O · 약 100만 토큰' 같은 한 줄."""
    caps = caps_for(key, data_dir)

    def mark(v):
        return "O" if v is True else "X" if v is False else "?"

    ctx = f" · 약 {caps.context // 1000:,}k 토큰" if caps.context else ""
    return f"👁 사진 {mark(caps.vision)} · 📝 형식 출력 {mark(caps.structured_output)}{ctx}"


__all__ = [
    "ANY", "TEAM_WORK", "IMAGE_READING", "JobNeeds", "Caps", "Hidden", "Removal", "ChoiceIssue", "CapsStore",
    "configure", "caps_for", "removal", "reasons", "can", "selectable_models", "runtime_specs", "check_choices",
    "removed_defaults", "needs_for_messages", "has_images", "vision_probe", "record_connection_test", "summary",
    "runtime_notices",
]
