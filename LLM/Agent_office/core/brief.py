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


# ─── 화면에 보여 줄 질문 예시 ('최종 상태'를 정의하는 질문 쓰는 법) ───────────

EXAMPLES: list[dict] = [
    {
        "title": "개념 조사 — 새 기술 용어",
        "question": "HBF(High Bandwidth Flash)가 무엇인지 알려줘",
        "goal": "팀 회의에서 5분 안에 공유할 수 있는 HBF 설명 자료",
        "success_criteria": [
            "정식 명칭과 한 줄 정의를 제시한다",
            "HBM 과의 차이를 표로 비교한다",
            "주요 개발 기업·시점 등 사실 정보마다 출처 번호를 단다",
        ],
        "constraints": ["비전공자도 이해할 수 있게", "A4 1장 분량"],
        "output_format": "한 줄 요약 → 비교 표 → 핵심 포인트 3개 → 출처",
    },
    {
        "title": "의사결정 — 선택지 비교",
        "question": "사내 문서 검색 챗봇에 쓸 벡터 DB 를 골라줘",
        "goal": "다음 주 아키텍처 회의에서 바로 결정할 수 있는 추천안",
        "success_criteria": [
            "후보 3개를 비용·운영 난이도·성능 기준으로 비교한다",
            "추천 1개와 그 이유, 선택하지 않은 이유를 쓴다",
            "도입 리스크와 대응 방안을 1개 이상 쓴다",
        ],
        "constraints": ["사내 서버(온프레미스) 운영 가능해야 함", "문서 100만 건 규모"],
        "output_format": "결론(추천) → 비교 표 → 리스크 → 다음 행동",
    },
    {
        "title": "계획 — 일정·실행안",
        "question": "3박 4일 오사카 가족 여행 계획 세워줘",
        "goal": "그대로 따라 하면 되는 일자별 여행 계획표",
        "success_criteria": [
            "일자별 오전·오후·저녁 일정이 있다",
            "이동 수단과 대략의 이동 시간을 적는다",
            "아이(초등학생) 동반에 맞는 장소인지 근거를 쓴다",
        ],
        "constraints": ["1인 하루 예산 15만 원 이내", "하루 이동은 2시간 이내"],
        "output_format": "일자별 표 → 예산 합계 → 준비물 체크리스트",
    },
    {
        "title": "코드 — 동작하는 결과물",
        "question": "CSV 로그에서 IP 별 요청 수 상위 10개를 뽑는 파이썬 함수 만들어줘",
        "goal": "바로 붙여 넣어 실행할 수 있는 파이썬 함수와 사용 예",
        "success_criteria": [
            "표준 라이브러리만 사용한다",
            "```python 코드 블록이 문법 오류 없이 실행된다 (하네스 검증)",
            "헤더가 없거나 빈 줄이 있어도 동작한다",
        ],
        "constraints": ["함수 하나 + 사용 예 1개"],
        "output_format": "코드 블록 → 사용 예 → 주의사항",
    },
]
