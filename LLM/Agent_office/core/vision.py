"""그림 읽기 공용 부분 — 비전 모델(그림을 볼 수 있는 LLM)로 그림을 읽는다.

쓰는 곳
  - 설계 팀 도면 그림 → 방 배치 (core/floorplan_import.py from_image)
  - 포트폴리오·자소서 팀 JD·요약 문서 그림·스캔 PDF → 글자 옮겨 적기 (transcribe — career_ui.py 의 확인 단계)

그림을 읽을 수 있는지는 모델 단위로 core/model_caps 가 판단한다 (👁 사진 읽기 능력).
  텍스트 전용(gpt-oss·Solar 등), 그림 전달이 확인 안 된 freellm(FreeLLMAPI 무료 프록시), 제공처에서 사라진 모델은 빠진다.
  claude_cli(구독)는 사진도 넘긴다(core/claude_cli.py — stream-json 입력).
  → '🆓 무료 모델만 사용'을 켜거나 팀원이 무료 모델만 쓰도록 골라져 있으면 그림을 읽을 모델이 없다 (유료로 넘어가지 않음).

큰 그림: 가로가 MAX_WIDTH 를 넘으면 줄이고, 줄인 뒤에도 세로가 TILE_HEIGHT 를 넘는 긴 화면 캡처는
위에서 아래로 조각(조금씩 겹침)으로 잘라 한 번의 호출에 순서대로 넣는다.
"""

import base64
import io
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory, model_caps, usage
from core.schemas import ModelsConfig, ModelSpec, RankModels

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
VISION_PROVIDERS = ("openai", "google", "anthropic", "claude_cli")  # 참고용 — 실제 판단은 core/model_caps (모델 단위)
NO_VISION_MODEL = "그림을 읽을 수 있는 모델이 없습니다 — 텍스트로 붙여 넣거나 PDF/Word 로 올려 주세요."
MAX_WIDTH = 1600  # 이보다 넓은 그림은 줄인다 (글자는 이 정도면 충분히 읽힌다)
TILE_HEIGHT = 2000  # 이보다 긴 그림은 조각으로 자른다
TILE_OVERLAP = 80  # 조각 사이 겹치는 높이 (줄이 잘려 사라지지 않게)
MAX_TILES = 8  # 그림 하나의 최대 조각 수 (넘으면 더 줄여서 맞춘다)
PDF_SCALE = 2.0  # PDF 쪽을 그림으로 바꿀 때 배율 (72dpi × 2)


READ_IMAGES = model_caps.JobNeeds(vision=True, label="사진 읽기")  # 옮겨 적기는 형식 출력이 필요 없다


def is_vision(spec: ModelSpec) -> bool:
    """이 모델이 그림을 읽을 수 있는가 — 모델 능력표·🧪 시험·사라진 모델까지 core/model_caps 로 판단."""
    return model_caps.can(f"{spec.provider}:{spec.model}", READ_IMAGES)


def vision_models(models: ModelsConfig, pool: list[ModelSpec] | None = None) -> RankModels | None:
    """후보 중 그림을 읽을 수 있는 것만 순서대로 (같은 모델은 한 번).

    pool 을 안 주면 보조 모델 → 직급 모델 순으로 찾는다 (설계 팀).
    포트폴리오·자소서 팀은 팀원 모델 후보(🎛 팀원 모델 선택 반영)를 pool 로 넘긴다.
    """
    if pool is None:
        pool = [*models.utility.candidates, *[s for r in models.ranks.values() for s in r.candidates]]
    seen, picked = set(), []
    for spec in pool:
        if is_vision(spec) and spec.label not in seen:
            seen.add(spec.label)
            picked.append(spec)
    if not picked:
        return None
    return RankModels(primary=picked[0], fallback=picked[1:4])


def usable_vision_models(models: ModelsConfig, pool: list[ModelSpec] | None = None) -> list[ModelSpec]:
    """지금 실제로 쓸 수 있는(키 있음·비활성 아님·'무료만' 아님) 그림 읽기 모델 — 화면의 모델 고르기 목록."""
    if pool is None:
        pool = [*models.utility.candidates, *[s for r in models.ranks.values() for s in r.candidates]]
    seen, out = set(), []
    for spec in pool:
        if is_vision(spec) and spec.label not in seen and llm_factory.is_available(spec.provider, models):
            seen.add(spec.label)
            out.append(spec)
    return out


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def image_part(data: bytes, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(data, mime)}}


# ─── 그림 준비 (줄이기·자르기) ───────────────────────────────────────────


def prepare_image(data: bytes) -> list[bytes]:
    """그림 → 보낼 PNG 조각 목록 (위에서 아래 순서). 작은 그림은 한 장 그대로."""
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")  # 투명 배경(PNG·WEBP)은 흰색 위에
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
        image = background
    width, height = image.size
    scale = min(1.0, MAX_WIDTH / width)
    # 조각이 너무 많아질 만큼 긴 그림은 더 줄인다
    max_height = MAX_TILES * (TILE_HEIGHT - TILE_OVERLAP)
    if height * scale > max_height:
        scale = max_height / height
    if scale < 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    width, height = image.size
    tiles, top = [], 0
    while True:
        bottom = min(height, top + TILE_HEIGHT)
        tiles.append(image.crop((0, top, width, bottom)))
        if bottom >= height:
            break
        top = bottom - TILE_OVERLAP
    out = []
    for tile in tiles:
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def pdf_page_png(data: bytes, index: int = 0) -> bytes:
    """PDF 한 쪽 → PNG (pypdfium2)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        page = pdf[index]
        image = page.render(scale=PDF_SCALE).to_pil()
        page.close()
    finally:
        pdf.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def pdf_first_page_png(data: bytes) -> bytes:
    return pdf_page_png(data, 0)


# ─── 글자 옮겨 적기 ────────────────────────────────────────────────────


class Transcript(BaseModel):
    """비전 모델 구조화 출력 — OpenAI strict 호환을 위해 기본값 없이 모두 필수."""

    text: str = Field(description="그림 속 글자를 위에서 아래 순서로 글자 그대로 옮긴 것 (제목·글머리표·줄바꿈 유지)")
    unclear: list[str] = Field(description="흐리거나 잘려서 확실하지 않은 부분 (그 줄을 그대로). 없으면 빈 목록")


TRANSCRIBE_PROMPT = "\n".join([
    "너는 그림 속 글자를 옮겨 적는 담당이다. 채용 공고·경력 정리 문서 화면 캡처가 들어온다.",
    "- 그림 속 글자를 글자 그대로(verbatim) 옮긴다. 요약하지 않고, 고쳐 쓰지 않고, 없는 내용을 더하지 않는다.",
    "- 제목·소제목·글머리표(-, •, 1.)·줄바꿈을 그대로 살린다. 표는 줄마다 칸을 ' | ' 로 이어 쓴다.",
    "- 숫자·날짜·단위·영문 약어는 보이는 그대로 쓴다 (예: 3년 이상, ISMS-P, 2,000만 원).",
    "- 메뉴·버튼·광고처럼 공고 본문이 아닌 화면 요소는 옮기지 않는다.",
    "- 여러 조각이 함께 오면 한 그림을 위에서 아래로 자른 것이다. 조각 사이 겹친 줄은 한 번만 쓴다.",
    "- 읽기 어려운 글자는 추측하지 말고 [?] 로 두고, 그 줄을 unclear 에 적는다.",
])


@dataclass
class TranscribeResult:
    text: str
    model: str  # 읽은 모델 이름 (화면 표시)
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = 0.0  # 추정 비용($) — 가격 미등록이면 None
    unclear: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def transcribe(images: list[bytes], rank_models: RankModels, models: ModelsConfig, who: str = "그림 글자 읽기",
               hint: str = "") -> TranscribeResult:
    """그림 조각들(한 그림을 자른 것) → 글자. 일시 오류는 재시도, 실패하면 대안 모델 (llm_factory.invoke_structured).

    쓸 수 있는 모델이 없으면 llm_factory.NoAvailableModelError.
    """
    content: list[dict] = [{"type": "text", "text": "이 그림의 글자를 그대로 옮겨 적어 줘." + (f" ({hint})" if hint else "")
                            + (f" 조각 {len(images)}개 = 한 그림을 위에서 아래로 자른 것." if len(images) > 1 else "")}]
    content += [image_part(png, "image/png") for png in images]
    messages = [SystemMessage(TRANSCRIBE_PROMPT), HumanMessage(content=content)]
    call = llm_factory.invoke_structured(rank_models, models, Transcript, messages, who)
    parsed: Transcript = call.parsed
    tokens_in, tokens_out = call.input_tokens or 0, call.output_tokens or 0
    cost = usage.estimate_cost(call.spec.provider, call.spec.model, tokens_in, tokens_out)
    return TranscribeResult(text=parsed.text.strip(), model=call.spec.label, provider=call.spec.provider,
                            input_tokens=tokens_in, output_tokens=tokens_out, cost=cost,
                            unclear=list(parsed.unclear), notes=call.errors[-3:])
