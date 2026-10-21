"""설계 팀 도면 자료 툴 — 첨부 도면을 레이아웃 JSON 으로 바꿔 모든 에이전트에게 근거 자료로 준다.

워크플로우가 시작되기 전에(자료 조사 단계) 돈다. 에이전트는 파일을 직접 보지 않고 이 툴이 만든 JSON 텍스트를 고친다.
  - 첨부 파일(업무 양식의 file 칸) → core/floorplan_import.py (JSON 정확 / DXF 정확·근사 / 그림·PDF 근사 / DWG 안내)
  - 첨부가 없고 '최근 저장한 설계안 사용'을 고르면 → results/design/ 의 가장 최근 *_layout.json
  - 읽은 도면은 코드로 완성·검증해 면적표·검증 결과도 함께 준다
  - 항상: 방 종류별 최소 치수 기준표 (하네스 floorplan 검증기가 쓰는 기준과 같다)
"""

import os
from pathlib import Path

from core.config import BASE_DIR, load_settings
from core.floorplan import area_table, check, report_markdown, standards_table
from core.floorplan_import import import_file, latest_saved_layout, to_json_text
from core.schemas import ModelsConfig, ToolResult


def results_dir() -> Path:
    """앱과 같은 규칙: AGENT_OFFICE_RESULTS_DIR → settings.yaml storage.results_dir → results/"""
    env = os.getenv("AGENT_OFFICE_RESULTS_DIR")
    if env:
        return Path(env)
    return BASE_DIR / load_settings().get("storage", {}).get("results_dir", "results")


def run(question: str, inputs: dict, models: ModelsConfig) -> ToolResult | None:
    standards = "### 📏 방 종류별 최소 치수 (코드 검증 기준)\n" + standards_table()
    path = Path(str(inputs["attachment"])) if inputs.get("attachment") else None
    source_name = inputs.get("attachment_name") or (path.name if path else "")
    if path is None and inputs.get("use_previous"):
        path = latest_saved_layout(results_dir())
        source_name = path.name if path else ""
        if path is None:
            return ToolResult(tool="floorplan_input", title="📐 도면 자료", content=standards,
                              notes=["저장된 설계안(results/design/*_layout.json)이 없습니다 — 먼저 평면도를 만들어 채택하세요."])
    if path is None:
        return ToolResult(tool="floorplan_input", title="📏 설계 기준표", content=standards)
    if not path.exists():
        return ToolResult(tool="floorplan_input", title="📐 도면 자료", content=standards, notes=[f"첨부 파일을 찾지 못했습니다: {path}"])

    result = import_file(path, models)
    lines = [f"- 첨부: `{source_name}` → 읽은 방법: **{result.method}**"]
    lines += [f"- 참고: {n}" for n in result.notes]
    if result.layout is None:
        lines += ["", "**도면을 읽지 못했다.** 사용자의 요청 문장만으로 새로 설계하고, 그 사실을 가정(assumptions)에 적어라.", "", standards]
        return ToolResult(tool="floorplan_input", title="📐 도면 자료 (읽기 실패)", content="\n".join(lines), notes=result.notes)

    layout = result.layout
    if layout.needs_check:
        lines.append("- ⚠️ 확인 필요: " + " / ".join(layout.needs_check))
    resolved, rep = check(layout)
    lines += [
        "",
        "### 첨부 도면의 현재 상태 (코드가 완성·검증)",
        area_table(resolved),
        "",
        report_markdown(rep),
        "",
        "### 첨부 도면의 레이아웃 JSON — 이 JSON 을 출발점으로 요청한 부분만 고쳐 답한다",
        "```json",
        to_json_text(resolved),
        "```",
        "",
        standards,
    ]
    return ToolResult(tool="floorplan_input", title=f"📐 첨부 도면 ({'정확' if result.exact else '근사'})", content="\n".join(lines),
                      notes=result.notes)
