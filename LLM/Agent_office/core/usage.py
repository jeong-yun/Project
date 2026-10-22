"""모델 사용량·비용 기록 — 메인 화면(🏠)의 모델별 토큰·추정 비용, API 키별 사용 내역.

어떻게 모으나
  - LangChain 모델 호출: llm_factory.create_chat_model 이 만드는 모든 모델에 UsageCallback 을 붙인다
    → 에이전트 초안·검토·통합, 비평가, 브리프, 교훈, 검색 계획, 지도, 편집 계획 … 모든 호출이 한 곳에서 잡힌다.
  - 그 밖의 SDK 직접 호출(음악 팀 Lyria·채점)은 record() 를 직접 부른다.
  - 어느 팀·어느 실행의 호출인지는 scope() (ContextVar) 로 붙인다. 백그라운드 스레드에도 이어진다
    (core/live.py 가 실행 스레드에서 scope 를 열고, 동시 실행 스레드풀은 context 를 복사한다).

API 키
  키 원문은 저장하지 않는다. sha256 앞 12자리를 키 id 로, 화면에는 '앞 3자 … 뒤 4자'만 보인다.
  키를 바꾸면 새 id 가 생기고, 예전 id 의 사용 내역은 '이전 키'로 남는다. 사람이 키마다 이름(별칭)을 붙일 수 있다.

저장은 sink 함수(app.py 가 Storage.record_usage 를 연결)가 맡는다. sink 가 없으면(테스트 등) 아무것도 하지 않는다.
기록 실패가 모델 호출을 막지 않도록 모든 예외를 삼킨다.
"""

import hashlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from core.config import CONFIG_DIR, load_yaml

PRICING_FILE = CONFIG_DIR / "pricing.yaml"


@dataclass
class UsageEvent:
    provider: str
    model: str
    key_id: str  # API 키 지문 (키 없음 = "")
    key_masked: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 1
    cost_usd: float | None = None  # 가격 미등록이면 None
    team_id: str = ""
    run_id: str = ""
    purpose: str = ""
    ts: float = field(default_factory=time.time)


# ─── 가격 ───────────────────────────────────────────────────────────────


_pricing_cache: tuple[float, dict] | None = None


def load_pricing(path: Path = PRICING_FILE) -> dict:
    """가격표 (파일이 바뀌면 다시 읽는다 — 앱을 다시 켜지 않아도 반영)."""
    global _pricing_cache
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"models": {}, "per_call": {}, "free_providers": []}
    if _pricing_cache and _pricing_cache[0] == mtime:
        return _pricing_cache[1]
    data = load_yaml(path) or {}
    data.setdefault("models", {})
    data.setdefault("per_call", {})
    data.setdefault("free_providers", [])
    _pricing_cache = (mtime, data)
    return data


_overrides: Callable[[], dict] | None = None


def set_price_overrides(getter: Callable[[], dict] | None) -> None:
    """화면에서 고친 가격(app.py: Storage.price_overrides)을 가격표 위에 덮어쓴다. {모델: {input, output, ...}}"""
    global _overrides
    _overrides = getter


def price_for(model: str, pricing: dict | None = None, day: str | None = None) -> dict | None:
    """모델의 100만 토큰당 가격 {input, output, cached_input?} — 덮어쓰기 > 날짜별 변경(changes) > 기본."""
    pricing = pricing or load_pricing()
    if _overrides is not None:
        try:
            over = _overrides().get(model)
        except Exception:  # noqa: BLE001
            over = None
        if over:
            return over
    price = pricing["models"].get(model)
    if not price:
        return None
    day = day or time.strftime("%Y-%m-%d")
    current = dict(price)
    for change in sorted(price.get("changes") or [], key=lambda c: str(c["from"])):
        if str(change["from"]) <= day:
            current.update({k: v for k, v in change.items() if k != "from"})
    return current


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
                  calls: int = 1, pricing: dict | None = None, day: str | None = None) -> float | None:
    """추정 비용($). 무료 provider = 0, 가격 미등록 = None."""
    pricing = pricing or load_pricing()
    if provider in pricing["free_providers"]:
        return 0.0
    if model in pricing["per_call"]:
        return round(float(pricing["per_call"][model]) * calls, 6)
    price = price_for(model, pricing, day)
    if not price:
        return None
    cached = min(cached_tokens, input_tokens)
    cached_rate = price.get("cached_input", price["input"])
    cost = (input_tokens - cached) * price["input"] + cached * cached_rate + output_tokens * price["output"]
    return round(cost / 1_000_000, 6)


# ─── API 키 지문 ─────────────────────────────────────────────────────────


def key_fingerprint(key: str | None) -> tuple[str, str]:
    """(키 id, 화면 표시용 가린 키). 키 원문은 어디에도 저장하지 않는다."""
    if not key:
        return "", "(키 없음)"
    key = key.strip()
    key_id = hashlib.sha256(key.encode()).hexdigest()[:12]
    masked = f"{key[:3]}…{key[-4:]}" if len(key) > 10 else "…" + key[-2:]
    return key_id, masked


# ─── 어느 팀·실행의 호출인가 ────────────────────────────────────────────────


_scope: ContextVar[dict] = ContextVar("usage_scope", default={})


@contextmanager
def scope(**tags: str) -> Iterator[None]:
    """with usage.scope(team_id="stock", run_id="…", purpose="결정 전 문의"): … — 안쪽 호출 기록에 붙는다."""
    token = _scope.set({**_scope.get(), **{k: v for k, v in tags.items() if v}})
    try:
        yield
    finally:
        _scope.reset(token)


# ─── 기록 ───────────────────────────────────────────────────────────────


_sink: Callable[[UsageEvent], None] | None = None


def set_sink(sink: Callable[[UsageEvent], None] | None) -> None:
    """기록을 저장할 함수 (app.py: Storage.record_usage). None 이면 기록하지 않는다."""
    global _sink
    _sink = sink


def record(provider: str, model: str, api_key: str | None, input_tokens: int = 0, output_tokens: int = 0,
           cached_tokens: int = 0, calls: int = 1, purpose: str = "") -> UsageEvent | None:
    if _sink is None:
        return None
    try:
        key_id, masked = key_fingerprint(api_key)
        tags = _scope.get()
        event = UsageEvent(
            provider=provider, model=model, key_id=key_id, key_masked=masked,
            input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0), cached_tokens=int(cached_tokens or 0),
            calls=calls, cost_usd=estimate_cost(provider, model, input_tokens or 0, output_tokens or 0, cached_tokens or 0, calls),
            team_id=tags.get("team_id", ""), run_id=tags.get("run_id", ""), purpose=purpose or tags.get("purpose", ""),
        )
        _sink(event)
        return event
    except Exception:  # noqa: BLE001 — 기록 실패가 모델 호출·업무를 막지 않게
        return None


def _usage_from_result(response) -> tuple[int, int, int]:
    """LLMResult → (입력, 출력, 캐시된 입력) 토큰. 스트리밍이면 조각을 합친 마지막 메시지에 들어 있다."""
    for gens in response.generations or []:
        for gen in gens:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                details = meta.get("input_token_details") or {}
                return meta.get("input_tokens", 0), meta.get("output_tokens", 0), details.get("cache_read", 0) or 0
    usage = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage") or {}
    return (usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
            usage.get("completion_tokens") or usage.get("output_tokens") or 0, 0)


class UsageCallback(BaseCallbackHandler):
    """모델 하나(provider·model·키)에 붙여 호출이 끝날 때마다 토큰을 기록한다."""

    raise_error = False

    def __init__(self, provider: str, model: str, api_key: str | None):
        self.provider, self.model, self.api_key = provider, model, api_key

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            tokens_in, tokens_out, cached = _usage_from_result(response)
        except Exception:  # noqa: BLE001
            tokens_in = tokens_out = cached = 0
        record(self.provider, self.model, self.api_key, tokens_in, tokens_out, cached)
