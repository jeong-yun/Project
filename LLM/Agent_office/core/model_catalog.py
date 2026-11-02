"""모델 목록 — 각 사 모델 목록 API 로 새 모델을 찾고(자동), 사람이 직접 입력한 모델을 더한다.

config/models.yaml 에 없는 모델도 '🎛 팀원 모델 선택'에서 고를 수 있게 한다. yaml 은 바꾸지 않는다.

자동 찾기
  - provider 마다 공식 목록 API 를 부른다 — OpenAI /v1/models, Google Gemini models.list, Anthropic /v1/models(켜져 있을 때),
    Upstage /v1/models(OpenAI 호환), FreeLLMAPI /v1/models. claude_cli(구독)는 목록 API 가 없어 CLI 가 받는 이름(별칭)을 쓴다.
  - 키가 있고 disabled_providers 에 없는 provider 만 부른다. 대화(텍스트 생성) 모델만 남긴다 (임베딩·음성·이미지·검열·파인튜닝 제외).
  - 결과는 데이터 폴더의 model_catalog.json 에 저장하고 하루(TTL_SEC) 동안 다시 부르지 않는다.
    앱을 켤 때 오래됐으면 백그라운드 스레드로 새로 받는다 (짧은 시간 제한, 실패해도 앱은 그대로 — 저장된 목록 → yaml 순으로 쓴다).
  - 처음 받은 뒤 새로 나타난 모델은 🆕 (NEW_DAYS 일 동안), yaml 에 있는데 목록에서 사라진 모델은 ⚠️ + 대신 쓸 모델 제안 (자동으로 바꾸지 않음).
  - 찾은 모델은 '고를 수 있게' 될 뿐이다 — 직급의 기본(primary)·대안 모델은 절대 자동으로 바뀌지 않는다.

직접 입력
  - provider + 모델 이름(+ 표시 이름). 데이터 폴더의 custom_models.json 에 저장한다 (config/*.yaml 은 안 고침).
  - 🧪 연결 시험(connection_test): 아주 짧은 호출 한 번 + 구조화 출력 한 번으로 실제로 되는지 확인한다.

강도(상하 관계 규칙용)
  yaml 의 model_tiers > 사람이 고른 급(custom_models.json 의 tiers) > 이름으로 추정(guess_tier) 순서로 정한다.
  새 모델은 새 줄을 만들지 않고 기존 줄 하나에 '같은 급'으로 들어간다 → core/member_models.py 의 규칙이 그대로 동작한다.
"""

import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from core.memo import SigCache, file_sig
from core.schemas import ModelsConfig, ModelSpec, Team

CATALOG_FILE = "model_catalog.json"
CUSTOM_FILE = "custom_models.json"
TTL_SEC = 24 * 3600  # 이 시간이 지나면 앱을 켤 때 목록을 다시 받는다
NEW_DAYS = 7  # 새로 나타난 모델에 🆕 를 붙이는 기간
HTTP_TIMEOUT = 6.0  # 초 — provider 하나의 목록 요청
FREE_PROVIDER = "freellm"

OPENAI_URL = "https://api.openai.com/v1/models"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_URL = "https://api.anthropic.com/v1/models"
UPSTAGE_URL = "https://api.upstage.ai/v1/models"
# claude -p --model 이 받는 이름 — 별칭(opus·sonnet·haiku = 그 계열의 최신)과 정식 이름 (Claude Code 문서의 model 설정)
CLAUDE_CLI_MODELS = [
    ("opus", "Opus (최신 별칭)"), ("sonnet", "Sonnet (최신 별칭)"), ("haiku", "Haiku (최신 별칭)"),
    ("claude-opus-5-5", "Claude Opus 5.5"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("claude-haiku-5-5", "Claude Haiku 5.5"),
]


# ─── HTTP (테스트에서 가짜로 바꾼다) ─────────────────────────────────────


def http_get(url: str, headers: dict, params: dict | None = None, timeout: float = HTTP_TIMEOUT) -> dict:
    """GET → JSON. 200 이 아니면 오류 (응답 본문은 앞부분만 — 키가 섞여 나올 일은 없지만 길 수 있음)."""
    import httpx

    response = httpx.get(url, headers=headers, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


# ─── 목록 읽기·거르기 (provider 별) ───────────────────────────────────────


@dataclass
class Found:
    """목록 API 에서 찾은 모델 하나."""

    id: str
    display: str = ""
    created: float = 0.0  # 각 사가 알려 준 출시(등록) 시각 (모르면 0)
    price: dict | None = None  # 목록 API 가 가격을 알려 주면 {input, output} (USD/100만 토큰)


NOT_CHAT = (  # 이름에 이것이 들어 있으면 대화 모델이 아니다 (메타데이터가 없는 OpenAI 호환 목록용)
    "embed", "whisper", "tts", "dall-e", "image", "imagen", "davinci", "babbage", "moderation", "audio", "realtime",
    "transcribe", "sora", "veo", "lyria", "search", "computer-use", "aqa", "rerank", "guard", "ocr", "document-parse",
    "layout", "native-audio", "live", "banana", "robotics",  # banana = Gemini 이미지 생성(Nano Banana)
)
# OpenAI: 앱은 Chat Completions 로 부르므로 Responses API 전용(*-pro·codex)·옛 Completions 전용(instruct) 모델은 뺀다
# (2026-11-02 실측 목록: gpt-5.5-pro·gpt-5.3-codex·gpt-3.5-turbo-instruct 등). 꼭 써 보고 싶으면 직접 입력 → 🧪 연결 시험.
OPENAI_NOT_CHAT = ("codex", "-pro", "instruct")
# FreeLLMAPI 의 자동 선택·라우터 이름 — 응답 형식이 들쭉날쭉해 쓰지 않는다 (models.yaml 의 auto 참고)
FREELLM_ROUTERS = ("auto", "router", "default")
SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{6}|\d{4})$")  # gpt-5.4-2026-03-05 · claude-x-20260101 · solar-pro4-260806 · gpt-4-0613


def _chat_name(name: str) -> bool:
    low = name.lower()
    return not any(word in low for word in NOT_CHAT)


def _drop_snapshots(found: list[Found]) -> list[Found]:
    """날짜가 붙은 고정판(snapshot)은 날짜 없는 이름이 함께 있으면 뺀다 — 목록이 너무 길어지지 않게."""
    ids = {f.id for f in found}
    return [f for f in found if not (SNAPSHOT.search(f.id) and SNAPSHOT.sub("", f.id) in ids)]


def _price_meta(item: dict) -> dict | None:
    """목록 API 가 가격을 주는 경우(OpenRouter 형식 pricing.prompt/completion = 토큰당 USD) → 100만 토큰당."""
    pricing = item.get("pricing") if isinstance(item, dict) else None
    if not isinstance(pricing, dict):
        return None
    try:
        price = {"input": float(pricing["prompt"]) * 1e6, "output": float(pricing["completion"]) * 1e6}
    except (KeyError, TypeError, ValueError):
        return None
    return {k: round(v, 6) for k, v in price.items()}


def parse_openai_like(data: dict, drop_words: tuple[str, ...] = ()) -> list[Found]:
    """OpenAI 형식 {"data": [{"id", "created", "owned_by"}]} — OpenAI·Upstage·FreeLLMAPI. drop_words = 이 provider 만의 제외 단어."""
    found = []
    for item in data.get("data") or []:
        model_id = str(item.get("id") or "") if isinstance(item, dict) else ""
        if (not model_id or model_id.startswith("ft:") or not _chat_name(model_id)
                or any(word in model_id.lower() for word in drop_words)):
            continue  # ft: = 파인튜닝한 내 모델
        found.append(Found(model_id, str(item.get("name") or ""), float(item.get("created") or 0), _price_meta(item)))
    return _drop_snapshots(found)


def parse_google(data: dict) -> list[Found]:
    """Gemini models.list — supportedGenerationMethods 에 generateContent 가 있는 모델만 (임베딩·AQA 등 제외)."""
    found = []
    for item in data.get("models") or []:
        name = str(item.get("name") or "").removeprefix("models/")
        if not name or "generateContent" not in (item.get("supportedGenerationMethods") or []) or not _chat_name(name):
            continue
        found.append(Found(name, str(item.get("displayName") or "")))
    return found


def parse_anthropic(data: dict) -> list[Found]:
    """Anthropic /v1/models — 모두 대화 모델."""
    found = []
    for item in data.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            created = item.get("created_at") or ""
            try:
                ts = time.mktime(time.strptime(str(created)[:19], "%Y-%m-%dT%H:%M:%S")) if created else 0.0
            except ValueError:
                ts = 0.0
            found.append(Found(str(item["id"]), str(item.get("display_name") or ""), ts))
    return _drop_snapshots(found)


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def fetch_openai(key: str) -> tuple[list[Found], list[str]]:
    data = http_get(OPENAI_URL, _bearer(key))
    return parse_openai_like(data, OPENAI_NOT_CHAT), [str(i.get("id")) for i in data.get("data") or []]


def fetch_google(key: str) -> tuple[list[Found], list[str]]:
    found, ids, token = [], [], None
    for _ in range(10):  # 페이지 나눔 (보통 한 페이지)
        params = {"pageSize": 1000, **({"pageToken": token} if token else {})}
        data = http_get(GOOGLE_URL, {"x-goog-api-key": key}, params)  # 키는 주소가 아니라 헤더로
        found += parse_google(data)
        ids += [str(m.get("name") or "").removeprefix("models/") for m in data.get("models") or []]
        token = data.get("nextPageToken")
        if not token:
            break
    return found, ids


def fetch_anthropic(key: str) -> tuple[list[Found], list[str]]:
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    found, after = [], None
    for _ in range(10):
        data = http_get(ANTHROPIC_URL, headers, {"limit": 1000, **({"after_id": after} if after else {})})
        found += parse_anthropic(data)
        if not data.get("has_more"):
            break
        after = data.get("last_id")
    return found, [f.id for f in found]


def fetch_upstage(key: str) -> tuple[list[Found], list[str]]:
    data = http_get(UPSTAGE_URL, _bearer(key))
    return parse_openai_like(data), [str(i.get("id")) for i in data.get("data") or []]


def fetch_freellm(key: str) -> tuple[list[Found], list[str]]:
    from core.llm_factory import freellm_base_url

    data = http_get(freellm_base_url().rstrip("/") + "/models", _bearer(key))
    # auto(FreeLLMAPI 자동 선택)는 쓰지 않는다 — 응답에 role 이 빠져 LangChain 이 거부함 (models.yaml 참고)
    return parse_openai_like(data, FREELLM_ROUTERS), [str(i.get("id")) for i in data.get("data") or []]


def fetch_claude_cli(key: str) -> tuple[list[Found], list[str]]:
    found = [Found(model_id, display) for model_id, display in CLAUDE_CLI_MODELS]
    return found, [f.id for f in found]


FETCHERS: dict[str, Callable[[str], tuple[list[Found], list[str]]]] = {
    "openai": fetch_openai, "google": fetch_google, "anthropic": fetch_anthropic,
    "upstage": fetch_upstage, "freellm": fetch_freellm, "claude_cli": fetch_claude_cli,
}
FIXED_LIST = {"claude_cli"}  # 목록 API 가 없어 고정 목록 — '목록에서 사라짐' 판단에 쓰지 않는다
# 찾은 모델을 '🎛 팀원 모델 선택' 선택지에 바로 넣는 provider. FreeLLMAPI 는 등록한 무료 키에 따라 수백 개라(2026-11-02 실측 266개)
# 선택지에 다 넣지 않는다 — '직접 입력'의 '찾은 모델에서 고르기'로 하나씩 넣는다. settings.yaml model_catalog.selector_providers 로 바꿀 수 있다.
SELECTOR_PROVIDERS = ("openai", "google", "anthropic", "upstage", "claude_cli")


# ─── 저장된 목록 (model_catalog.json) ─────────────────────────────────────


_lock = threading.Lock()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # 쓰는 도중 꺼져도 예전 파일이 깨지지 않게


_cache = SigCache()


def load_catalog(data_dir: Path | str) -> dict:
    """{"providers": {provider: {"fetched_at", "error", "models": {id: {...}}, "all_ids": [...]}}} — 파일이 그대로면 다시 안 읽음."""
    path = Path(data_dir) / CATALOG_FILE
    return _cache.get(str(path), file_sig(path), lambda: _read_json(path)) if path.exists() else {}


def catalog_prices(data_dir: Path | str) -> dict[str, dict]:
    """{모델: {input, output}} — 목록 API 가 가격을 알려 준 모델만 (usage.set_catalog_prices 에 연결)."""
    return {
        model_id: meta["price"]
        for entry in (load_catalog(data_dir).get("providers") or {}).values()
        for model_id, meta in (entry.get("models") or {}).items()
        if isinstance(meta, dict) and meta.get("price")
    }


def provider_key(provider: str, models: ModelsConfig) -> str | None:
    """목록을 받을 수 있으면 키(claude_cli 는 빈 문자열), 아니면 None — 꺼짐·키 없음·FreeLLMAPI 서버 꺼짐."""
    from core import llm_factory

    if provider not in FETCHERS or not llm_factory.is_available(provider, models):
        return None
    return os.getenv(models.providers.get(provider, "")) or ""


def refresh(models: ModelsConfig, data_dir: Path | str, now: float | None = None) -> dict:
    """켜져 있고 키가 있는 provider 의 목록을 동시에 받아 저장한다. 실패한 provider 는 예전 목록을 그대로 두고 오류만 적는다."""
    now = time.time() if now is None else now
    targets = {p: k for p in models.providers if (k := provider_key(p, models)) is not None}

    def one(provider: str):
        try:
            return provider, FETCHERS[provider](targets[provider]), ""
        except Exception as e:  # noqa: BLE001 — 네트워크·키·형식 오류 모두 '이번엔 못 받음'
            return provider, None, f"{type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        results = list(pool.map(one, targets))
    path = Path(data_dir) / CATALOG_FILE
    with _lock:
        data = _read_json(path)
        providers = data.setdefault("providers", {})
        for provider, got, error in results:
            entry = providers.setdefault(provider, {"models": {}})
            entry["checked_at"] = now
            entry["error"] = error
            if got is None:
                continue
            found, all_ids = got
            old = entry.get("models") or {}
            baseline = not old  # 처음 받은 목록은 모두 '원래 있던 모델' (🆕 아님)
            entry["models"] = {
                f.id: {
                    "display": f.display, "created": f.created, "price": f.price,
                    "first_seen": (old.get(f.id) or {}).get("first_seen", 0.0 if baseline else now),
                }
                for f in found
            }
            entry["all_ids"] = sorted(set(all_ids))
            entry["fetched_at"] = now
        data["refreshed_at"] = now
        _write_json(path, data)
    return data


_refreshing: threading.Thread | None = None


def is_stale(data_dir: Path | str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now - float(load_catalog(data_dir).get("refreshed_at") or 0) > TTL_SEC


def start_refresh(models: ModelsConfig, data_dir: Path | str, force: bool = False) -> threading.Thread | None:
    """백그라운드로 목록을 받는다 (오래됐거나 force). 이미 받는 중이면 그 스레드를 돌려준다. 화면은 기다리지 않는다."""
    global _refreshing
    with _lock:
        if _refreshing and _refreshing.is_alive():
            return _refreshing
        if not force and not is_stale(data_dir):
            return None

        def work() -> None:
            try:
                refresh(models, data_dir)
            except Exception:  # noqa: BLE001 — 목록을 못 받아도 앱은 저장된 목록·yaml 로 그대로 동작
                pass

        _refreshing = threading.Thread(target=work, name="model-catalog-refresh", daemon=True)
        _refreshing.start()
        return _refreshing


REAL_START = start_refresh  # 테스트가 start_refresh 를 막아 둬도 원래 함수를 부를 수 있게


def refreshing() -> bool:
    return bool(_refreshing and _refreshing.is_alive())


# ─── 직접 입력한 모델·사람이 고른 급 (custom_models.json) ──────────────────


MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,119}$")


class CustomModelError(ValueError):
    """잘못 입력한 모델 (provider 없음·이름 형식)."""


class CustomStore:
    """{"models": [{"provider", "model", "display", "added_at"}], "tiers": {"provider:model": "같은 급 기준 모델 키"}}"""

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / CUSTOM_FILE

    def read(self) -> dict:
        data = load_catalog_file(self.path)
        models = [m for m in data.get("models") or [] if isinstance(m, dict) and m.get("provider") and m.get("model")]
        tiers = {k: v for k, v in (data.get("tiers") or {}).items() if isinstance(k, str) and isinstance(v, str)}
        return {"models": models, "tiers": tiers}

    def models(self) -> list[dict]:
        return self.read()["models"]

    def keys(self) -> list[str]:
        return [f"{m['provider']}:{m['model']}" for m in self.models()]

    def tiers(self) -> dict[str, str]:
        return self.read()["tiers"]

    def _save(self, data: dict) -> None:
        with _lock:
            _write_json(self.path, data)

    def add(self, provider: str, model: str, display: str, models: ModelsConfig) -> str:
        provider, model, display = provider.strip(), model.strip(), display.strip()
        if provider not in models.providers:
            raise CustomModelError(f"모르는 provider: {provider} (가능: {', '.join(models.providers)})")
        if not MODEL_NAME.match(model):
            raise CustomModelError("모델 이름은 영문·숫자·. _ - : / @ + 만 쓸 수 있습니다 (빈칸 없이, 120자 이내).")
        data = self.read()
        key = f"{provider}:{model}"
        data["models"] = [m for m in data["models"] if f"{m['provider']}:{m['model']}" != key]
        data["models"].append({"provider": provider, "model": model, "display": display[:60], "added_at": time.time()})
        self._save(data)
        return key

    def remove(self, key: str) -> None:
        data = self.read()
        data["models"] = [m for m in data["models"] if f"{m['provider']}:{m['model']}" != key]
        data["tiers"].pop(key, None)
        self._save(data)

    def set_tier(self, key: str, same_as: str | None) -> None:
        """key 의 강도를 'same_as 와 같은 급'으로 (None = 자동 추정으로 되돌림)."""
        data = self.read()
        if same_as:
            data["tiers"][key] = same_as
        else:
            data["tiers"].pop(key, None)
        self._save(data)


_custom_cache = SigCache()


def load_catalog_file(path: Path) -> dict:
    return _custom_cache.get(str(path), file_sig(path), lambda: _read_json(path)) if path.exists() else {}


# ─── 강도 추정 (이름 규칙) ───────────────────────────────────────────────


CLASS_WORDS = (  # (단어, 크기 등급) — 4 = 가장 큼(pro·opus), 3 = 보통, 2 = mini·flash, 1 = nano·lite·haiku
    ("opus", 4), ("ultra", 4), ("max", 4), ("large", 4),
    ("nano", 1), ("lite", 1), ("tiny", 1), ("haiku", 1),
    ("mini", 2), ("flash", 2), ("small", 2), ("medium", 2),
    ("sonnet", 3), ("fable", 3),
)
CLASS_STEP = 2  # 크기 등급 하나 = 강도 줄 두 칸 (yaml: gpt-5.4 → mini → nano 가 두 줄씩 차이)
VERSION_STEP = 0.5  # 버전이 이만큼 높으면 한 줄 위 (높으면 적어도 한 줄)
MAX_VERSION_SHIFT = 2  # 버전 차이로는 최대 두 줄까지만 (실측: solar-pro2 → pro4 를 네 줄 올리면 gpt-5.5 급이 됨)


@dataclass
class NameInfo:
    family: str
    version: float | None
    size: int


def name_info(model: str) -> NameInfo:
    """모델 이름 → (계열, 버전, 크기 등급). 예) gpt-5.4-mini → (gpt, 5.4, 2), claude-opus-5-5 → (claude, 5.5, 4)."""
    name = model.lower().rsplit("/", 1)[-1]
    name = SNAPSHOT.sub("", re.sub(r"-(preview|exp|experimental|latest|chat-latest)(-.*)?$", "", name))
    tokens = [t for t in re.split(r"[-_:]", name) if t]
    head = tokens[0] if tokens else name
    family = re.match(r"[a-z]+", head).group(0) if re.match(r"[a-z]+", head) else head
    if "oss" in tokens:
        family = f"{family}-oss"
    version = None
    params = None
    rest = [head[len(family):]] + tokens[1:] if head != family else tokens[1:]
    for i, tok in enumerate(rest):
        if m := re.fullmatch(r"(\d+(?:\.\d+)?)b", tok):
            params = float(m.group(1))
            continue
        m = re.fullmatch(r"[a-z]*?(\d+(?:\.\d+)?)[a-z]?", tok)
        if m and version is None:
            version = float(m.group(1))
            nxt = rest[i + 1] if i + 1 < len(rest) else ""
            if "." not in m.group(1) and re.fullmatch(r"\d", nxt):  # claude-opus-5-5 → 5.5
                version = float(f"{m.group(1)}.{nxt}")
    size = 3
    words = set(tokens) | {re.sub(r"\d+$", "", t) for t in tokens}
    for word, level in CLASS_WORDS:
        if word in words:
            size = level
            break
    else:
        if "pro" in words:
            size = 4
        elif params is not None:
            size = 1 if params <= 30 else 2 if params <= 150 else 3
    return NameInfo(family, version, size)


def _free_rows(tiers: list[list[str]]) -> set[int]:
    return {i for i, row in enumerate(tiers) if row and all(k.startswith(FREE_PROVIDER + ":") for k in row)}


def guess_tier(key: str, tiers: list[list[str]]) -> int:
    """yaml 에 없는 모델의 강도 줄 추정 — 같은 계열(가능하면 같은 provider)의 비슷한 크기 모델을 기준으로
    크기 등급·버전 차이만큼 위아래로 옮긴다. 무료(freellm) 모델은 무료 줄 안에서만, 유료는 무료 줄 위에서만."""
    provider, _, model = key.partition(":")
    info = name_info(model)
    free = _free_rows(tiers)
    is_free = provider == FREE_PROVIDER
    allowed = sorted(free) if is_free else [i for i in range(len(tiers)) if i not in free]
    if not allowed:
        allowed = list(range(len(tiers)))
    anchors = [(i, k, name_info(k.partition(":")[2])) for i, row in enumerate(tiers) for k in row if i in allowed]
    same_family = [a for a in anchors if a[2].family == info.family]
    pool = ([a for a in same_family if a[1].startswith(provider + ":")] or same_family
            or [a for a in anchors if a[1].startswith(provider + ":")])
    if not pool:
        return allowed[len(allowed) // 2]  # 기준이 없으면 가운데

    def closeness(a):
        size_gap = abs(a[2].size - info.size)
        version_gap = abs((a[2].version or 0) - (info.version or 0)) if a[2].family == info.family else 0
        return (size_gap, version_gap, -a[0])  # 같으면 약한 쪽(아래 줄) — 상하 규칙에 더 안전

    row, _, anchor = min(pool, key=closeness)
    row -= (info.size - anchor.size) * CLASS_STEP
    if anchor.family == info.family and info.version is not None and anchor.version is not None:
        diff = info.version - anchor.version
        shift = round(diff / VERSION_STEP)
        if diff > 0:
            shift = max(shift, 1)
        row -= max(-MAX_VERSION_SHIFT, min(MAX_VERSION_SHIFT, shift))
    return min(max(row, allowed[0]), allowed[-1])


# ─── 합치기: yaml + 찾은 모델 + 직접 입력 ─────────────────────────────────


@dataclass
class ModelInfo:
    """고를 수 있는 모델 하나의 출처·표시 정보 (화면 라벨용)."""

    key: str
    source: str  # yaml | discovered | custom
    tier_source: str  # yaml | user | guess
    display: str = ""
    new: bool = False  # 최근 NEW_DAYS 일 안에 목록에 새로 나타남
    selectable: bool = True  # 선택지(강도 순서)에 들어감 — 선택지에 바로 안 넣는 provider 의 찾은 모델은 False
    missing: bool = False  # yaml 에 있는데 provider 목록에서 사라짐
    suggest: str = ""  # 사라진 모델 대신 쓸 만한 모델 키
    price: dict | None = None  # 목록 API 가 알려 준 가격


@dataclass
class Merged:
    models: ModelsConfig  # model_tiers 를 넓힌 설정 (나머지는 yaml 그대로)
    info: dict[str, ModelInfo] = field(default_factory=dict)
    yaml_missing: list[str] = field(default_factory=list)  # 직급·팀 설정에 쓰였는데 model_tiers 에 없던 모델 (추정으로 넣음)


def _used_keys(models: ModelsConfig, teams: list[Team]) -> list[str]:
    from core.config import used_model_keys

    return sorted(used_model_keys(models, teams))


def _suggest(key: str, found: dict[str, dict]) -> str:
    """사라진 모델 대신 — 같은 provider·같은 계열·같은 크기에서 버전이 가장 높은 모델 (없으면 같은 계열)."""
    provider, _, model = key.partition(":")
    info = name_info(model)
    same = [(m, name_info(m)) for m in found if name_info(m).family == info.family]
    best = [x for x in same if x[1].size == info.size] or same
    if not best:
        return ""
    return f"{provider}:{max(best, key=lambda x: (x[1].version or 0, -len(x[0])))[0]}"


def merge(models: ModelsConfig, data_dir: Path | str | None, teams: list[Team] = (), now: float | None = None,
          selector_providers: tuple[str, ...] | list[str] = SELECTOR_PROVIDERS) -> Merged:
    """yaml 의 강도 순서에 찾은 모델·직접 입력한 모델을 끼워 넣은 설정. 데이터 폴더가 없으면(테스트) yaml 만.
    selector_providers 밖의 provider 에서 찾은 모델은 정보(🆕 등)만 남기고 선택지에는 넣지 않는다 (직접 입력하면 들어감)."""
    now = time.time() if now is None else now
    tiers = [list(row) for row in models.model_tiers] or [[]]
    listed = {k for row in tiers for k in row}
    info = {k: ModelInfo(k, "yaml", "yaml") for k in listed}
    catalog = load_catalog(data_dir) if data_dir else {}
    store = CustomStore(data_dir) if data_dir else None
    user_tiers = store.tiers() if store else {}
    row_of = {k: i for i, row in enumerate(tiers) for k in row}

    extra: dict[str, ModelInfo] = {}
    yaml_missing = [k for k in _used_keys(models, teams) if k not in listed]
    for key in yaml_missing:
        extra[key] = ModelInfo(key, "yaml", "guess")
    for provider, entry in (catalog.get("providers") or {}).items():
        if provider not in models.providers:
            continue
        found = entry.get("models") or {}
        for model_id, meta in found.items():
            key = f"{provider}:{model_id}"
            first = float(meta.get("first_seen") or 0)
            is_new = bool(first) and now - first < NEW_DAYS * 86400
            if key in info:
                info[key].new, info[key].price = is_new, meta.get("price")
            elif key not in extra:
                extra[key] = ModelInfo(key, "discovered", "guess", meta.get("display") or "", is_new, price=meta.get("price"),
                                       selectable=provider in selector_providers)
        if provider in FIXED_LIST or not entry.get("fetched_at"):
            continue
        all_ids = set(entry.get("all_ids") or found)
        for key in [k for k in listed if k.startswith(provider + ":")]:
            if key.partition(":")[2] not in all_ids:
                info[key].missing, info[key].suggest = True, _suggest(key, found)
    for m in store.models() if store else []:
        key = f"{m['provider']}:{m['model']}"
        if m["provider"] not in models.providers:
            continue
        target = info.get(key) or extra.get(key)
        if target is None:
            extra[key] = ModelInfo(key, "custom", "guess", m.get("display") or "")
        else:  # 목록에서도 찾은 모델을 직접 입력함 → 직접 입력으로 표시 (yaml 모델은 yaml 그대로)
            if target.source == "discovered":
                target.source, target.selectable = "custom", True
            target.display = m.get("display") or target.display
    for key, item in extra.items():
        info[key] = item
        if not item.selectable:
            continue
        anchor = user_tiers.get(key)
        if anchor in row_of:
            row, item.tier_source = row_of[anchor], "user"
        else:
            row = guess_tier(key, models.model_tiers or tiers)
        tiers[row].append(key)
    merged = models.model_copy(update={"model_tiers": [row for row in tiers if row]})
    return Merged(merged, info, yaml_missing)


def tier_rows(models: ModelsConfig) -> list[str]:
    """yaml 강도 줄마다 대표 모델 키 (화면의 '이 모델은 [○○ 와 같은 급]' 선택지) — 위가 가장 강함."""
    return [row[0] for row in models.model_tiers if row]


# ─── 🧪 연결 시험 ───────────────────────────────────────────────────────


TEST_PROMPT = "Reply with exactly: OK"
TEST_TOKENS = (60, 40)  # 미리 보여 줄 비용 추정용 (입력, 출력) — 텍스트·구조화 출력 두 번 합계의 대략값


class Ping(BaseModel):
    """구조화 출력 시험용 아주 작은 형식."""

    ok: bool
    word: str


@dataclass
class TestResult:
    ok: bool
    text_ok: bool = False
    structured_ok: bool | None = None  # None = 시험 안 함
    reply: str = ""
    error: str = ""
    elapsed_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None = 가격 미등록


def estimate_test_cost(provider: str, model: str) -> float | None:
    from core import usage

    return usage.estimate_cost(provider, model, TEST_TOKENS[0], TEST_TOKENS[1], calls=2)


def connection_test(provider: str, model: str, models: ModelsConfig, structured: bool = True) -> TestResult:
    """아주 짧은 호출(텍스트) 한 번 + 구조화 출력 한 번. 꺼진 provider(disabled·키 없음·무료만)는 부르지 않는다.
    토큰·비용은 평소처럼 사용량에 기록된다 (용도 '연결 시험')."""
    from core import llm_factory, usage

    reason = llm_factory.unavailable_reason(provider, models)
    if reason:
        return TestResult(ok=False, error=f"지금 쓸 수 없는 provider — {reason}")
    spec = ModelSpec(provider=provider, model=model)
    result = TestResult(ok=False)
    started = time.perf_counter()
    try:
        with usage.scope(purpose="연결 시험"):
            llm = llm_factory.create_chat_model(spec, models)
            with llm_factory.provider_slot(provider):
                reply = llm.invoke(TEST_PROMPT)
            result.reply = str(getattr(reply, "content", reply))[:200]
            result.text_ok = bool(result.reply.strip())
            meta = getattr(reply, "usage_metadata", None) or {}
            result.input_tokens += int(meta.get("input_tokens") or 0)
            result.output_tokens += int(meta.get("output_tokens") or 0)
            if structured:
                result.structured_ok = False
                bound = llm.with_structured_output(Ping, include_raw=True, **llm_factory.structured_output_kwargs(spec))
                with llm_factory.provider_slot(provider):
                    out = bound.invoke("Return ok=true and word='pong'.")
                result.structured_ok = out.get("parsed") is not None
                meta = getattr(out.get("raw"), "usage_metadata", None) or {}
                result.input_tokens += int(meta.get("input_tokens") or 0)
                result.output_tokens += int(meta.get("output_tokens") or 0)
                if not result.structured_ok:
                    result.error = f"구조화 출력 실패: {str(out.get('parsing_error'))[:200]}"
    except Exception as e:  # noqa: BLE001 — provider 별 예외가 제각각
        result.error = f"{type(e).__name__}: {str(e)[:300]}"
    result.elapsed_sec = round(time.perf_counter() - started, 2)
    result.ok = result.text_ok and result.structured_ok is not False
    result.cost_usd = usage.estimate_cost(provider, model, result.input_tokens, result.output_tokens, calls=2)
    return result
