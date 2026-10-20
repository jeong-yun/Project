"""채택한 결과를 파일로 저장 — results/<팀>/<날짜-시각>_<질문>.md (+ 여행 지도 .html).

사람이 '채택'한 내용만 저장한다 (반려·버리기는 저장하지 않음).
파일에는 나중에 봐도 맥락을 알 수 있도록 질문·업무 브리프·최종 답변·사람의 결정·근거 자료 목록을 함께 적는다.
저장 위치는 settings.yaml 의 storage.results_dir (기본 results/) — .gitignore 로 git 에는 올라가지 않는다
(보유 종목·여행 일정 같은 개인 정보가 들어갈 수 있으므로). 올리고 싶으면 .gitignore 에서 results/ 줄을 지운다.
"""

import re
from datetime import datetime
from pathlib import Path

from core.schemas import HumanDecision


def slugify(text: str, limit: int = 40) -> str:
    """파일 이름에 쓸 수 있게: 한글·영문·숫자만 남기고 나머지는 _ 로."""
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text.strip()).strip("_")
    return (slug[:limit].rstrip("_")) or "result"


def result_markdown(
    run: dict, decision: HumanDecision, team_name: str, names: dict[str, str], disclaimer: str = "", appendix: str = ""
) -> str:
    """저장할 마크다운 본문."""
    brief = run.get("brief")
    lines = [
        f"# {run['question'].splitlines()[0][:80]}",
        "",
        f"- 팀: {team_name}",
        f"- 채택: {names.get(decision.adopted_node_id or '', decision.adopted_node_id)} 의 안"
        + (" (사람이 직접 수정)" if decision.edited else ""),
        f"- 만족도: {decision.score}/5",
        f"- 저장 시각: {datetime.now():%Y-%m-%d %H:%M}",
    ]
    if decision.comment:
        lines.append(f"- 코멘트: {decision.comment}")
    lines += ["", "## 질문", "", run["question"]]
    if brief:
        lines += ["", "## 업무 브리프 (최종 상태 정의)", "", f"**목표**: {brief.goal}", ""]
        lines += [f"- [완료 기준] {c}" for c in brief.success_criteria]
        lines += [f"- [제약] {c}" for c in brief.constraints]
    lines += ["", "## 최종 결과", "", decision.final_answer.strip()]
    if appendix:
        lines += ["", "## 코드 검산 (점수·평균가·손익비)", "", appendix]
    if disclaimer:
        lines += ["", f"> {disclaimer}"]

    sources = []
    for tool in run.get("tools") or []:
        sources.append(f"- {tool.title} (데이터 툴)")
    web = run.get("web")
    if web:
        sources += [f"- [{i}] [{r.title}]({r.url})" for i, r in enumerate(web.results, 1)]
    if sources:
        lines += ["", "## 근거 자료", "", *sources]
    return "\n".join(lines) + "\n"


def export_adopted(
    run: dict,
    decision: HumanDecision,
    team_id: str,
    team_name: str,
    names: dict[str, str],
    results_root: Path,
    disclaimer: str = "",
    map_html: str | None = None,
    appendix: str = "",
) -> list[Path]:
    """채택 결과를 파일로 저장하고, 저장한 파일 경로 목록을 돌려준다."""
    folder = results_root / team_id
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(run['question'])}"
    md_path = folder / f"{stem}.md"
    md_path.write_text(result_markdown(run, decision, team_name, names, disclaimer, appendix), encoding="utf-8")
    paths = [md_path]
    if map_html:
        map_path = folder / f"{stem}_지도.html"
        map_path.write_text(map_html, encoding="utf-8")
        paths.append(map_path)
    return paths


def list_results(results_root: Path, team_id: str, limit: int = 20) -> list[Path]:
    """저장된 결과 파일 (최근 것부터)."""
    folder = results_root / team_id
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.suffix in (".md", ".html")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
