"""데이터 툴 등록부.

툴 = 질문·업무 입력을 받아 '사실 자료'를 만들어 주는 함수. 결과(ToolResult)는 웹 검색 결과와 함께
모든 에이전트 프롬프트에 근거 자료로 들어가고, 화면 타임라인에도 표시된다.

  stock_data      : 종목 주가·수익률·이동평균·변동성·밸류에이션, 보유 종목 평가손익 (yfinance)
  travel_calendar : 여행 기간 N박 M일, 날짜별 요일, 계절, D-day (계산만, 외부 호출 없음)

새 툴 추가 방법
  1) 이 폴더에 run(question, inputs, models) -> ToolResult | None 함수를 만든다 (쓸 자료가 없으면 None)
  2) 아래 TOOLS 에 이름을 등록한다
  3) team.yaml 의 tools 에 이름을 적는다
"""

from collections.abc import Callable

from core.schemas import ModelsConfig, ToolResult
from core.tools import stock_data, travel_calendar

ToolFn = Callable[[str, dict, ModelsConfig], ToolResult | None]

TOOLS: dict[str, ToolFn] = {
    "stock_data": stock_data.run,
    "travel_calendar": travel_calendar.run,
}


def run_tools(names: list[str], question: str, inputs: dict, models: ModelsConfig) -> list[ToolResult]:
    """팀에 등록된 툴을 차례로 실행한다. 한 툴이 실패해도 나머지는 계속 (실패는 결과 notes 에 기록)."""
    results = []
    for name in names:
        if name not in TOOLS:
            continue
        try:
            result = TOOLS[name](question, inputs, models)
        except Exception as e:  # noqa: BLE001 — 외부 데이터 오류가 답변을 막지 않게
            result = ToolResult(tool=name, title=name, content="", notes=[f"{name} 실행 실패: {type(e).__name__}: {str(e)[:200]}"])
        if result is not None:
            results.append(result)
    return results
