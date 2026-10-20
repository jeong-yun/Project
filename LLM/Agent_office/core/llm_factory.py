"""provider/model 설정 → LangChain ChatModel 생성.

새 provider 를 추가하려면:
  1) core/schemas.py 의 Provider 에 이름 추가
  2) config/models.yaml 의 providers 에 .env 키 이름 추가
  3) 아래 create_chat_model 에 분기 추가
"""

import os
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

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


def is_transient(error: Exception) -> bool:
    """잠시 뒤 다시 하면 될 가능성이 높은 오류인가 (provider 마다 예외 종류가 달라 이름·메시지로 판단)."""
    text = f"{type(error).__name__} {error}".lower()
    if "quota" in text and ("free_tier" in text or "limit: 0" in text):
        return False  # 무료 할당량 0 (예: Gemini Pro) — 기다려도 안 됨
    return any(marker in text for marker in TRANSIENT_MARKERS)


def sleep(seconds: float) -> None:
    """재시도 대기 (테스트에서 0 초로 바꿀 수 있게 함수로 분리)."""
    time.sleep(seconds)

# Upstage Solar 는 OpenAI 호환 API 를 제공하므로 별도 패키지 없이 ChatOpenAI 로 호출한다.
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
# langchain-anthropic 의 기본 max_tokens(1024)는 긴 답변에 부족하다.
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


def has_key(provider: str, models: ModelsConfig) -> bool:
    env_key = models.providers.get(provider)
    return bool(env_key and os.getenv(env_key))


def is_available(provider: str, models: ModelsConfig) -> bool:
    """키가 있고 disabled_providers 에 없으면 사용 가능."""
    return has_key(provider, models) and provider not in models.disabled_providers


def provider_status(models: ModelsConfig) -> dict[str, str]:
    """provider 별 상태: ok | disabled | no_key (키 값은 노출하지 않는다)."""
    return {
        provider: "disabled" if provider in models.disabled_providers
        else "ok" if has_key(provider, models)
        else "no_key"
        for provider in models.providers
    }


def available_candidates(rank_models: RankModels, models: ModelsConfig) -> list[tuple[ModelSpec, bool]]:
    """사용 가능한 후보 모델을 우선순위대로 반환. (spec, is_fallback) 목록."""
    return [
        (spec, i > 0)
        for i, spec in enumerate(rank_models.candidates)
        if is_available(spec.provider, models)
    ]


def create_chat_model(spec: ModelSpec, models: ModelsConfig) -> BaseChatModel:
    """ModelSpec 으로 ChatModel 을 만든다. provider 패키지는 필요할 때만 import 한다."""
    api_key = os.getenv(models.providers[spec.provider])
    kwargs = {"model": spec.model}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature

    if spec.provider in ("openai", "upstage"):
        from langchain_openai import ChatOpenAI

        if spec.max_tokens:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.provider == "upstage":
            kwargs["base_url"] = UPSTAGE_BASE_URL
        return ChatOpenAI(api_key=api_key, stream_usage=True, **kwargs)

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs["max_tokens"] = spec.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS
        return ChatAnthropic(api_key=api_key, **kwargs)

    if spec.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if spec.max_tokens:
            kwargs["max_output_tokens"] = spec.max_tokens
        return ChatGoogleGenerativeAI(google_api_key=api_key, **kwargs)

    raise ValueError(f"지원하지 않는 provider: {spec.provider}")


def structured_output_kwargs(spec: ModelSpec) -> dict:
    """provider 별 with_structured_output 옵션.

    - openai   : 기본값(json_schema strict) 사용
    - upstage  : OpenAI 호환 API 의 json_schema 지원이 불확실 → function_calling
    - anthropic/google : 기본값(tool calling). google 은 추가 인자를 받지 않는다.
    """
    if spec.provider == "upstage":
        return {"method": "function_calling"}
    return {}


def require_candidates(rank_models: RankModels, models: ModelsConfig, who: str) -> list[tuple[ModelSpec, bool]]:
    candidates = available_candidates(rank_models, models)
    if not candidates:
        providers = sorted({s.provider for s in rank_models.candidates})
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
                out = llm.with_structured_output(schema, include_raw=True, **structured_output_kwargs(spec)).invoke(
                    messages
                )
                if out.get("parsed") is None:
                    raise ValueError(f"구조화 출력 파싱 실패: {out.get('parsing_error')}")
                break
            except Exception as e:  # noqa: BLE001 — provider 별 예외 타입이 제각각
                # provider 에러 메시지는 매우 길 수 있어 앞부분만 남긴다
                errors.append(f"{spec.label}: {type(e).__name__}: {str(e)[:300]}")
                out = None
                if attempt < len(RETRY_DELAYS) and is_transient(e):
                    sleep(RETRY_DELAYS[attempt])  # 잠깐 기다렸다 같은 모델로 다시
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
