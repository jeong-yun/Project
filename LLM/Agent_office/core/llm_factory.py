"""provider/model 설정 → LangChain ChatModel 생성.

새 provider 를 추가하려면:
  1) core/schemas.py 의 Provider 에 이름 추가
  2) config/models.yaml 의 providers 에 .env 키 이름 추가
  3) 아래 create_chat_model 에 분기 추가

claude_cli(Claude 구독)는 API 키 대신 'CLI 설치·로그인'으로 사용 가능 여부를 정한다 — core/claude_cli.py
"""

import os
import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from core import usage
from core.schemas import ModelsConfig, ModelSpec, RankModels


class NoAvailableModelError(RuntimeError):
    """사용 가능한 모델이 없거나 모든 후보 모델 호출이 실패한 경우."""


# ─── 일시적 오류 재시도 ──────────────────────────────────────────────
# 503(과부하)·429(요청 과다)·시간 초과·연결 끊김은 잠시 뒤 다시 하면 대부분 성공한다.
# → 같은 모델로 RETRY_DELAYS 만큼 기다렸다 다시 시도하고, 그래도 안 되면 다음 후보(대안 모델)로 넘어간다.
# 키 오류·모델 없음(404)·잘못된 요청(400) 같은 영구 오류는 기다려도 소용없으므로 바로 다음 후보로 간다.
RETRY_DELAYS = (2.0, 5.0)  # 초
TRANSIENT_MARKERS = (
    "503", "502", "500 ", "429", "unavailable", "overloaded", "high demand", "try again later",
    "ratelimit", "rate limit", "resourceexhausted", "timeout", "timed out", "connection", "temporarily",
    "deadlineexceeded", "internalservererror", "apiconnectionerror",
)


# 할당량 소진(무료 일일 한도·크레딧 부족)은 몇 초 기다려도 풀리지 않는다.
# → 그 모델을 QUOTA_COOLDOWN_SEC 동안 후보에서 빼서, 다음 호출부터는 바로 대안 모델을 쓴다 (매번 실패를 기다리지 않음).
QUOTA_COOLDOWN_SEC = 600
_cooldown: dict[str, float] = {}  # "provider:model" → 다시 써 볼 시각 (time.monotonic)


# FreeLLMAPI 가 '이 모델의 무료 한도가 찼다'고 알려 주는 문구
FREE_TIER_EXHAUSTED = ("all models exhausted", "rate-limited after", "on cooldown")
# 요청 자체가 무료 한도보다 큼 (예: Groq gpt-oss 분당 8천 토큰) — 같은 모델로는 다시 해도 안 됨
TOO_LARGE = ("413", "request too large", "tpm_limit")


SHORT_RESET_MAX = 150.0  # 초 — 무료 한도가 이 안에 풀린다고 하면 기다렸다 같은 모델로 (분당 토큰·요청 한도)
RESET_HINT = re.compile(r"soonest reset ~\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.IGNORECASE)


def reset_seconds(error: Exception) -> float | None:
    """FreeLLMAPI 의 'Soonest reset ~2m' 처럼 한도가 풀리는 시간이 적혀 있으면 초 단위로."""
    m = RESET_HINT.search(str(error))
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    return value * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]


UNKNOWN_RESET_WAIT = 30.0  # 초 — 한도에 걸렸다는데 풀리는 시간이 안 적힌 경우 (실측: Groq 분당 토큰 한도)


def _unknown_reset(error: Exception) -> bool:
    text = str(error).lower()
    return "rate-limited or on cooldown" in text and "soonest reset" not in text


def retry_delay(error: Exception, attempt: int) -> float | None:
    """같은 모델로 다시 시도하기 전 기다릴 시간 (None = 다시 하지 말고 다음 후보로).

    - 곧 풀리는 무료 한도(분당 한도, 150초 이내): 그 시간만큼 한 번만 기다린다 — 실측: Groq 분당 8천 토큰 한도
    - 그 밖의 일시적 오류(503·429·연결 끊김): RETRY_DELAYS(2초·5초)
    """
    reset = reset_seconds(error)
    if reset is not None and reset <= SHORT_RESET_MAX:
        return reset + 2 if attempt == 0 else None
    if reset is None and _unknown_reset(error):
        return UNKNOWN_RESET_WAIT if attempt == 0 else None
    if attempt < len(RETRY_DELAYS) and is_transient(error):
        return RETRY_DELAYS[attempt]
    return None


def is_quota_exhausted(error: Exception) -> bool:
    if getattr(error, "quota_exhausted", False):
        return True  # 예외가 직접 알려 줌 (claude_cli 구독 한도)
    reset = reset_seconds(error)
    if (reset is not None and reset <= SHORT_RESET_MAX) or (reset is None and _unknown_reset(error)):
        return False  # 곧 풀리는(또는 시간이 안 적힌) 분당 한도 — 쿨다운하지 않고 잠깐 기다렸다 다시
    text = f"{type(error).__name__} {error}".lower()
    return (
        ("quota" in text and ("free_tier" in text or "limit: 0" in text or "perday" in text))
        or "insufficient_quota" in text
        or any(marker in text for marker in FREE_TIER_EXHAUSTED)
    )


def is_transient(error: Exception) -> bool:
    """잠시 뒤 다시 하면 될 가능성이 높은 오류인가 (provider 마다 예외 종류가 달라 이름·메시지로 판단)."""
    if getattr(error, "transient", None) is not None:
        return bool(error.transient)  # 예외가 직접 알려 줌 (claude_cli: 로그인 안 됨·시간 초과는 False, 과부하는 True)
    if is_quota_exhausted(error):
        return False  # 무료 할당량 소진·크레딧 부족 — 기다려도 안 됨
    text = f"{type(error).__name__} {error}".lower()
    if any(marker in text for marker in TOO_LARGE):
        return False  # 프롬프트가 무료 한도보다 큼 → 바로 다음 후보(문맥이 큰 모델)로
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _model_key(spec: ModelSpec) -> str:
    return f"{spec.provider}:{spec.model}"


def note_failure(spec: ModelSpec, error: Exception) -> None:
    """호출 실패 기록 — 할당량 소진이면 잠시 후보에서 뺀다."""
    if is_quota_exhausted(error):
        seconds = getattr(error, "cooldown_sec", None) or QUOTA_COOLDOWN_SEC  # 구독 한도는 더 오래 쉰다
        _cooldown[_model_key(spec)] = time.monotonic() + seconds


def cooling_down(spec: ModelSpec) -> bool:
    return _cooldown.get(_model_key(spec), 0.0) > time.monotonic()


def sleep(seconds: float) -> None:
    """재시도 대기 (테스트에서 0 초로 바꿀 수 있게 함수로 분리)."""
    time.sleep(seconds)


# ─── provider 별 동시 호출 제한 ──────────────────────────────────────
# 형제 에이전트·자료 조사가 동시에 진행되므로, 무료 티어(분당 토큰·요청 한도, 429)가 한꺼번에 터지지 않게
# provider 마다 '동시에 진행 중인 LLM 호출 수'를 세마포어로 제한한다. 값은 settings.yaml 의 llm_concurrency.
# 자리는 호출 1회 동안만 잡는다 — 재시도 대기(sleep) 중에는 자리를 비워 다른 에이전트가 쓰게 한다.
DEFAULT_CONCURRENCY = 4
_concurrency: dict[str, int] | None = None  # None = 아직 settings.yaml 을 읽지 않음
_slots: dict[str, threading.BoundedSemaphore] = {}
_slots_lock = threading.Lock()


def set_concurrency(limits: dict | None) -> None:
    """동시 호출 제한을 바꾼다 ({"default": 4, "freellm": 2, …}, 0 이하 = 제한 없음). 값이 같으면 그대로 둔다."""
    global _concurrency
    limits = {k: int(v) for k, v in (limits or {}).items()}
    with _slots_lock:
        if limits != _concurrency:
            _concurrency = limits
            _slots.clear()  # 진행 중인 호출은 예전 세마포어를 그대로 놓고 나간다


def concurrency_limit(provider: str) -> int:
    global _concurrency
    if _concurrency is None:
        from core.config import load_settings  # 순환 import 를 피하려고 필요할 때 읽는다

        try:
            set_concurrency(load_settings().get("llm_concurrency"))
        except (OSError, ValueError):
            set_concurrency({})
    limits = _concurrency or {}
    return limits.get(provider, limits.get("default", DEFAULT_CONCURRENCY))


@contextmanager
def provider_slot(provider: str) -> Iterator[None]:
    """provider 의 동시 호출 자리 하나를 잡는다 (자리가 없으면 빌 때까지 기다림)."""
    limit = concurrency_limit(provider)
    if limit <= 0:
        yield
        return
    with _slots_lock:
        slot = _slots.get(provider)
        if slot is None:
            slot = _slots[provider] = threading.BoundedSemaphore(limit)
    with slot:
        yield

# Upstage Solar 는 OpenAI 호환 API 를 제공하므로 별도 패키지 없이 ChatOpenAI 로 호출한다.
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
# langchain-anthropic 의 기본 max_tokens(1024)는 긴 답변에 부족하다.
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


def has_key(provider: str, models: ModelsConfig) -> bool:
    if provider == "claude_cli":
        return claude_cli_status() == "ok"  # API 키 대신 'CLI 가 있고 구독 로그인이 되어 있음'
    env_key = models.providers.get(provider)
    return bool(env_key and os.getenv(env_key))


def claude_cli_status() -> str:
    """Claude 구독 CLI 상태: ok | not_installed | logged_out (5분 캐시 — core/claude_cli.py)."""
    from core import claude_cli

    return claude_cli.cli_status()


# ─── FreeLLMAPI (무료 모델 프록시) ─────────────────────────────────────
# 내 PC 의 FreeLLMAPI 서버(Docker)가 Google·Groq 등의 무료 티어를 OpenAI 호환 주소 하나로 묶어 준다.
# 서버가 꺼져 있으면 호출마다 연결 오류·재시도로 시간을 버리므로, 접속 가능 여부를 짧게 확인해 캐시한다.
FREELLM_DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"  # Windows 는 localhost 보다 127.0.0.1 이 빠름
FREELLM_CHECK_TTL = 30.0  # 초
_reachable: dict[str, tuple[float, bool]] = {}


def freellm_base_url() -> str:
    return os.getenv("FREELLMAPI_BASE_URL") or FREELLM_DEFAULT_BASE_URL


class _RoleFixTransport:
    """FreeLLMAPI 응답 보정 — 가끔 choices[].message 에 role 이 빠져 와서 LangChain 이 거부한다
    (2026-10-20 실측: 무료 한도 근처에서 'ValidationError: ChatMessage role None'). JSON 응답이면 role 을 assistant 로 채운다.
    스트리밍(SSE)은 건드리지 않는다."""

    def __init__(self, inner=None):
        import httpx

        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request):
        import json

        import httpx

        response = self._inner.handle_request(request)
        if "event-stream" in response.headers.get("content-type", "") or not request.url.path.endswith("/chat/completions"):
            return response  # 스트리밍(SSE)·다른 경로는 그대로 (content-type 이 json 이 아니게 오는 경우도 있어 경로로 판단)
        response.read()
        try:
            data = json.loads(response.content)
        except ValueError:
            return response
        changed = False
        for choice in data.get("choices") or [] if isinstance(data, dict) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and not message.get("role"):
                message["role"] = "assistant"
                changed = True
        if not changed:
            return response
        headers = [(k, v) for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=json.dumps(data).encode(), request=request)

    def close(self) -> None:
        self._inner.close()


_freellm_client = None
_freellm_client_lock = threading.Lock()


def freellm_http_client():
    """FreeLLMAPI 전용 HTTP 클라이언트 (한 번 만들어 재사용 — 매번 만들면 SSL 인증서를 다시 읽어 느려짐).

    여러 에이전트가 동시에 처음 부를 수 있으므로 잠금 안에서 한 번만 만든다 (httpx.Client 는 스레드 간 공유 가능).
    """
    global _freellm_client
    with _freellm_client_lock:
        if _freellm_client is None:
            import httpx

            _freellm_client = httpx.Client(transport=_RoleFixTransport(), timeout=180)
    return _freellm_client


def freellm_reachable() -> bool:
    """FreeLLMAPI 서버 포트가 열려 있는가 (30초 캐시, 0.5초 안에 판단)."""
    url = freellm_base_url()
    cached = _reachable.get(url)
    if cached and time.monotonic() - cached[0] < FREELLM_CHECK_TTL:
        return cached[1]
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.5):
            ok = True
    except OSError:
        ok = False
    _reachable[url] = (time.monotonic(), ok)
    return ok


def is_available(provider: str, models: ModelsConfig) -> bool:
    """키가 있고 disabled_providers 에 없으면 사용 가능 (freellm 은 서버가 켜져 있어야 함, claude_cli 는 CLI 로그인)."""
    if provider in models.disabled_providers or not has_key(provider, models):
        return False
    return freellm_reachable() if provider == "freellm" else True


def provider_status(models: ModelsConfig) -> dict[str, str]:
    """provider 별 상태: ok | disabled | no_key | offline | not_installed | logged_out (키 값은 노출하지 않는다)."""
    def status(provider: str) -> str:
        if provider in models.disabled_providers:
            return "disabled"
        if provider == "claude_cli":
            return claude_cli_status()
        if not has_key(provider, models):
            return "no_key"
        if provider == "freellm" and not freellm_reachable():
            return "offline"
        return "ok"

    return {provider: status(provider) for provider in models.providers}


def claude_cli_settings() -> dict:
    """settings.yaml 의 claude_cli (timeout_sec, effort). 못 읽으면 기본값."""
    from core.config import load_settings  # 순환 import 를 피하려고 필요할 때 읽는다

    try:
        return load_settings().get("claude_cli") or {}
    except (OSError, ValueError):
        return {}


def free_only(models: ModelsConfig) -> ModelsConfig:
    """무료 모델만 쓰는 설정 (유료 provider 를 모두 끈 사본) — 화면의 '🆓 무료 모델만 사용'.

    claude_cli(구독)도 끈다 — 청구는 없지만 사람이 쓰는 Claude 와 한도를 나눠 쓰므로 'FreeLLMAPI 만'이라는 뜻을 지킨다.
    """
    paid = [p for p in models.providers if p != "freellm"]
    return models.model_copy(update={"disabled_providers": sorted(set(models.disabled_providers) | set(paid))})


def available_candidates(rank_models: RankModels, models: ModelsConfig) -> list[tuple[ModelSpec, bool]]:
    """사용 가능한 후보 모델을 우선순위대로 반환. (spec, is_fallback) 목록.

    할당량 소진으로 쉬는 중인 모델은 뒤로 미룬다 (모두 쉬는 중이면 그래도 시도한다).
    """
    usable = [(spec, i > 0) for i, spec in enumerate(rank_models.candidates) if is_available(spec.provider, models)]
    return [c for c in usable if not cooling_down(c[0])] + [c for c in usable if cooling_down(c[0])]


def create_chat_model(spec: ModelSpec, models: ModelsConfig) -> BaseChatModel:
    """ModelSpec 으로 ChatModel 을 만든다. provider 패키지는 필요할 때만 import 한다."""
    if spec.provider == "claude_cli":
        # 구독 호출 — API 키 없음, 토큰은 기록하되 비용은 0 (pricing.yaml free_providers → 메인 화면 '구독')
        from core.claude_cli import ClaudeCLIChatModel

        options = claude_cli_settings()
        return ClaudeCLIChatModel(
            model=spec.model, timeout=options.get("timeout_sec") or ClaudeCLIChatModel.model_fields["timeout"].default,
            effort=options.get("effort") or None, callbacks=[usage.UsageCallback(spec.provider, spec.model, None)],
        )
    api_key = os.getenv(models.providers[spec.provider])
    # 호출이 끝날 때마다 토큰·추정 비용을 기록한다 (메인 화면 🏠 — core/usage.py). 기록 실패는 호출을 막지 않는다
    kwargs = {"model": spec.model, "callbacks": [usage.UsageCallback(spec.provider, spec.model, api_key)]}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature

    if spec.provider in ("openai", "upstage", "freellm"):
        from langchain_openai import ChatOpenAI

        if spec.max_tokens:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.provider == "upstage":
            kwargs["base_url"] = UPSTAGE_BASE_URL
        if spec.provider == "freellm":
            # 무료 티어는 한도에 자주 걸리므로 라이브러리 재시도는 끄고 공통 재시도·대안 모델 로직에 맡긴다
            kwargs.update(base_url=freellm_base_url(), max_retries=0, timeout=180, http_client=freellm_http_client())
        return ChatOpenAI(api_key=api_key, stream_usage=True, **kwargs)

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs["max_tokens"] = spec.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS
        return ChatAnthropic(api_key=api_key, **kwargs)

    if spec.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if spec.max_tokens:
            kwargs["max_output_tokens"] = spec.max_tokens
        # 라이브러리 자체 재시도(기본 6회, 지수 대기)는 할당량 소진 때 수 분씩 멈추게 한다 → 재시도는 위의 공통 로직이 맡는다
        kwargs.setdefault("max_retries", 1)
        return ChatGoogleGenerativeAI(google_api_key=api_key, **kwargs)

    raise ValueError(f"지원하지 않는 provider: {spec.provider}")


def structured_output_kwargs(spec: ModelSpec) -> dict:
    """provider 별 with_structured_output 옵션.

    - openai   : 기본값(json_schema strict) 사용
    - upstage  : OpenAI 호환 API 의 json_schema 지원이 불확실 → function_calling
    - freellm  : json_schema (2026-10-20 실측: gpt-oss-20b·Gemini 계열은 function_calling 이 실패하고 json_schema 는 성공)
    - anthropic/google : 기본값(tool calling). google 은 추가 인자를 받지 않는다.
    - claude_cli : 추가 인자 없음 — ClaudeCLIChatModel.with_structured_output 이 claude -p --json-schema 로 처리
    """
    if spec.provider == "upstage":
        return {"method": "function_calling"}
    if spec.provider == "freellm":
        return {"method": "json_schema"}
    return {}


FREE_SERVER_OFF = "무료 서버 꺼짐 — Docker Desktop 실행 후 FreeLLMAPI 켜기"


STATUS_REASONS = {  # provider_status 값 → 쓸 수 없는 이유 (화면 '고른 모델 꺼짐'·실패 메시지)
    "ok": "",
    "disabled": "비활성 (models.yaml disabled_providers 또는 '무료 모델만 사용')",
    "no_key": "API 키 없음 (.env)",
    "offline": FREE_SERVER_OFF,
    "not_installed": "Claude CLI 없음",
    "logged_out": "Claude 로그인 필요 (claude auth login)",
}


def unavailable_reason(provider: str, models: ModelsConfig) -> str:
    """provider 를 지금 쓸 수 없는 이유 (쓸 수 있으면 빈 문자열)."""
    status = provider_status(models).get(provider, "no_key")
    return STATUS_REASONS.get(status, status)


def require_candidates(rank_models: RankModels, models: ModelsConfig, who: str) -> list[tuple[ModelSpec, bool]]:
    candidates = available_candidates(rank_models, models)
    if not candidates:
        providers = sorted({s.provider for s in rank_models.candidates})
        if providers == ["freellm"]:
            # 팀원 모델 선택으로 무료 모델만 쓰게 된 팀원 — 유료 모델로 넘어가지 않고 이유를 알려 준다
            raise NoAvailableModelError(
                f"{who}: 고른 무료 모델을 쓸 수 없습니다 — {unavailable_reason('freellm', models)} (유료 모델로 넘어가지 않습니다)"
            )
        raise NoAvailableModelError(f"{who}: 사용 가능한 API 키가 없습니다. .env 에 {providers} 중 하나를 설정하세요.")
    return candidates


@dataclass
class StructuredCall:
    parsed: BaseModel
    spec: ModelSpec
    is_fallback: bool
    elapsed_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    errors: list[str] = field(default_factory=list)


def invoke_structured(
    rank_models: RankModels,
    models: ModelsConfig,
    schema: type[BaseModel],
    messages: list[BaseMessage],
    who: str,
) -> StructuredCall:
    """구조화 출력 호출. 일시적 오류는 같은 모델로 재시도하고, 그래도 실패하면 다음 후보로 넘어간다."""
    errors: list[str] = []
    for spec, is_fallback in require_candidates(rank_models, models, who):
        out = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            started = time.perf_counter()
            try:
                llm = create_chat_model(spec, models)
                structured = llm.with_structured_output(schema, include_raw=True, **structured_output_kwargs(spec))
                with provider_slot(spec.provider):  # 동시 호출 제한 — 자리가 빌 때까지 기다린 시간은 빼고 잰다
                    started = time.perf_counter()
                    out = structured.invoke(messages)
                if out.get("parsed") is None:
                    raise ValueError(f"구조화 출력 파싱 실패: {out.get('parsing_error')}")
                break
            except Exception as e:  # noqa: BLE001 — provider 별 예외 타입이 제각각
                # provider 에러 메시지는 매우 길 수 있어 앞부분만 남긴다
                errors.append(f"{spec.label}: {type(e).__name__}: {str(e)[:300]}")
                note_failure(spec, e)
                out = None
                delay = retry_delay(e, attempt)
                if delay is not None:
                    sleep(delay)  # 잠깐 기다렸다 같은 모델로 다시 (곧 풀리는 무료 한도면 그 시간만큼)
                    continue
                break  # 영구 오류이거나 재시도를 다 썼으면 다음 후보로
        if out is None:
            continue
        usage = getattr(out.get("raw"), "usage_metadata", None) or {}
        return StructuredCall(
            parsed=out["parsed"],
            spec=spec,
            is_fallback=is_fallback,  # 후보 순서로 판단 (같은 모델 재시도 성공은 대안이 아님)
            elapsed_sec=round(time.perf_counter() - started, 2),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            errors=errors,
        )
    raise NoAvailableModelError(f"{who}: 모든 후보 모델 호출에 실패했습니다.\n" + "\n".join(errors))
