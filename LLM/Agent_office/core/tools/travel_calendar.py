"""여행 달력 툴 — 여행 기간을 날짜별로 펼쳐 준다 (계산만, 외부 호출 없음).

LLM 은 날짜·요일 계산을 자주 틀린다 (예: 11월 20일이 무슨 요일인지, 3박 4일인지 4박 5일인지).
그래서 날짜 관련 사실은 코드로 계산해 모든 에이전트에게 근거 자료로 준다.
계산한 여행 일수는 하네스(itinerary_days 검증기)에도 넘겨, 일정이 모든 날을 덮는지 검사한다.
"""

from datetime import date, timedelta

from core.config import WEEKDAYS, date_label, today
from core.schemas import ModelsConfig, ToolResult

SEASONS = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    if not (inputs.get("start_date") and inputs.get("end_date")):
        return None
    start, end = date.fromisoformat(str(inputs["start_date"])), date.fromisoformat(str(inputs["end_date"]))
    days = (end - start).days + 1
    d_day = (start - today()).days

    lines = [
        f"- 여행 기간: {date_label(start)} ~ {date_label(end)} · **{days - 1}박 {days}일**",
        f"- 오늘({date_label(today())}) 기준 출발까지 **D-{d_day}**" if d_day >= 0 else f"- 출발일이 오늘보다 {-d_day}일 지났음 (날짜 확인 필요)",
        f"- 계절(북반구 기준): {SEASONS[start.month]} ({start.month}월)" + (" — 남반구 목적지는 계절이 반대" if inputs.get("scope") == "해외" else ""),
        "",
        "| 일차 | 날짜 | 요일 | 비고 |",
        "|---|---|---|---|",
    ]
    for i in range(days):
        d = start + timedelta(days=i)
        note = "출발일" if i == 0 else ("귀가일" if i == days - 1 else "")
        if d.weekday() >= 5:
            note = (note + " · 주말").strip(" ·")
        lines.append(f"| {i + 1}일차 | {d.isoformat()} | {WEEKDAYS[d.weekday()]} | {note} |")
    weekend = sum((start + timedelta(days=i)).weekday() >= 5 for i in range(days))
    lines += ["", f"- 주말 포함 {weekend}일. 공휴일·현지 축제·휴무일은 웹 검색 자료로 확인한다."]

    return ToolResult(
        tool="travel_calendar",
        title=f"🗓 여행 달력 {days - 1}박 {days}일",
        content="\n".join(lines),
        harness_options={"trip_days": days, "trip_start": start.isoformat()},
    )
