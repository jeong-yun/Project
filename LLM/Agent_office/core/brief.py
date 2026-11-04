"""요청 정리(Desired State Brief) — '과정 지시' 대신 '최종 상태 정의'로 일을 맡긴다.

좋은 결과를 얻는 질문은 "1단계로 A 하고, 2단계로 B 해" 처럼 과정을 쪼개는 것이 아니라
"내가 도달하고 싶은 최종 상태는 이것이고, 이 기준을 만족하면 완료다" 라고 목표를 정의하는 것이다.
과정(누가 무엇을 할지)은 supervisor 가 스스로 구상해 하위 에이전트에게 나눈다.

브리프 구성
  goal              최종 목표 — 결과물이 무엇이고 어디에 쓰이는지
  context           배경 — 왜 필요한지, 이미 아는 것
  success_criteria  완료 기준 — 이것들을 모두 만족하면 '끝' (검토자·비평가의 채점 기준으로도 쓰인다)
  constraints       제약 — 분량, 대상 독자, 피해야 할 것
  output_format     결과 형식 — 표, 단계별 목록, 보고서 등

사용자가 화면에서 직접 채우면 그대로 쓰고, 비워 두면 보조 모델이 질문에서 추론해 채운다.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm_factory
from core.config import date_label, today
from core.llm_factory import NoAvailableModelError
from core.schemas import ModelsConfig


class DesiredState(BaseModel):
    """최종 상태 정의 (LLM 구조화 출력 겸 화면 입력 형식)."""

    goal: str = Field(description="최종 목표 — 사용자가 받고 싶은 결과물과 그 용도를 한두 문장으로")
    context: str = Field(description="배경·맥락. 질문에 없으면 빈 문자열")
    success_criteria: list[str] = Field(description="완료 기준 2~4개. 검증 가능한 문장으로 (예: '정식 명칭과 출처를 제시한다')")
    constraints: list[str] = Field(description="제약 조건 (분량·대상·금지 사항). 없으면 빈 목록")
    output_format: str = Field(description="결과 형식 (예: '결론 → 근거 → 출처 순의 마크다운')")
    assumptions: list[str] = Field(description="질문이 모호해 스스로 정한 가정. 없으면 빈 목록")

    def is_empty(self) -> bool:
        return not (self.goal.strip() or self.success_criteria)

    def as_prompt(self, question: str) -> str:
        """에이전트들에게 전달할 '업무 브리프' 텍스트."""
        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {x}" for x in items) or "- (없음)"

        parts = [
            f"[사용자 원 질문]\n{question}",
            f"[최종 목표 — 이 상태에 도달하면 완료]\n{self.goal}",
        ]
        if self.context.strip():
            parts.append(f"[배경]\n{self.context}")
        parts.append(f"[완료 기준 — 모두 만족해야 함]\n{bullets(self.success_criteria)}")
        if self.constraints:
            parts.append(f"[제약]\n{bullets(self.constraints)}")
        if self.output_format.strip():
            parts.append(f"[결과 형식]\n{self.output_format}")
        if self.assumptions:
            parts.append(f"[모호한 부분에 대한 가정]\n{bullets(self.assumptions)}")
        return "\n\n".join(parts)


def empty_state() -> DesiredState:
    return DesiredState(goal="", context="", success_criteria=[], constraints=[], output_format="", assumptions=[])


def draft_brief(question: str, models: ModelsConfig, partial: DesiredState | None = None) -> tuple[DesiredState, list[str]]:
    """질문(+사용자가 일부 채운 내용)으로 브리프를 완성한다. 사용자가 채운 항목은 그대로 둔다.

    모델 호출이 실패하면 최소 브리프(목표 = 질문)로 진행한다 — 답변을 막지 않는다.
    """
    partial = partial or empty_state()
    filled = partial.model_dump_json(indent=1) if not partial.is_empty() else "(없음)"
    messages = [
        SystemMessage(
            "\n".join(
                [
                    "너는 요청 정리 담당이다. 사용자의 질문을 '최종 상태 정의(Desired State)' 브리프로 바꾼다.",
                    f"- 오늘 날짜: {date_label(today())} ('최근·현재'는 이 날짜 기준).",
                    "- 과정(1단계, 2단계…)을 지시하지 말고, 도달해야 할 결과물과 완료 기준을 정의한다.",
                    "- 완료 기준은 나중에 검토자가 '충족/미충족'을 판정할 수 있도록 구체적이고 검증 가능하게 쓴다.",
                    "- 사실 확인이 필요한 질문이면 '출처를 제시한다' 같은 기준을 포함한다.",
                    "- 질문이 모호하면 가장 그럴듯한 해석을 assumptions 에 적는다 (약어는 여러 의미 가능성을 고려).",
                    "- 약어·용어가 여러 뜻일 수 있으면 완료 기준에 '가능한 의미를 구분하되, 최신 동향(뉴스)에서",
                    "  가장 화제인 의미를 중심으로 설명한다' 를 넣는다.",
                    "- 사용자가 이미 채운 항목은 그 내용을 존중해 그대로 쓰고, 빈 항목만 채운다.",
                ]
            )
        ),
        HumanMessage(f"[질문]\n{question}\n\n[사용자가 이미 채운 브리프]\n{filled}"),
    ]
    try:
        call = llm_factory.invoke_structured(models.utility, models, DesiredState, messages, "요청 정리")
    except NoAvailableModelError as e:
        fallback = partial.model_copy(update={"goal": partial.goal or question})
        return fallback, [f"요청 정리 실패 → 질문을 그대로 목표로 사용: {e}"]
    brief: DesiredState = call.parsed
    # 사용자가 직접 쓴 값이 있으면 모델 값보다 우선
    for field in ("goal", "context", "output_format"):
        if getattr(partial, field).strip():
            setattr(brief, field, getattr(partial, field))
    for field in ("success_criteria", "constraints"):
        if getattr(partial, field):
            setattr(brief, field, getattr(partial, field))
    return brief, call.errors


# 자유 질문 예시는 팀마다 teams/<팀>/team.yaml 의 free_examples 에 둔다 (그 팀 분야 질문만 — core/scope.py).
