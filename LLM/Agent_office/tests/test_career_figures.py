"""👔 포트폴리오·자소서 팀 — 원본 그림(흐름도) 살리기 · mermaid 유지 (docs/TODO_커리어팀_그림_mermaid.md).

그림은 코드로 만든 작은 PNG, 문서는 가짜 글(가명)뿐이다 — 실제 경력 문서·그림 아님. API 키·네트워크·비용 없음.
"""

import io
import json
import zipfile
from pathlib import Path

from core import career, doc_export
from tests.test_career_format import DATED_SOURCE, check, summary


def tiny_png(color: tuple[int, int, int] = (200, 30, 30), size: tuple[int, int] = (4, 3)) -> bytes:
    """코드로 만든 작은 PNG (외부 라이브러리 없이) — 실제 그림 아님."""
    import struct
    import zlib

    w, h = size
    raw = b"".join(b"\x00" + bytes(color) * w for _ in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def figure_docx(tmp_path, name: str = "기술경력서.docx", colors=((200, 30, 30), (30, 30, 200))) -> str:
    """가짜 기술경력서 — 제목 1개, 표 칸 안 그림(흐름도) 2개 (두 번째는 병합 칸), 가명만."""
    import docx

    d = docx.Document()
    d.add_heading("1. 회사 업무", level=1)
    t = d.add_table(rows=3, cols=3)
    for c, v in enumerate(["업무 명", "기간", "흐름도"]):
        t.cell(0, c).text = v
    t.cell(1, 0).text, t.cell(1, 1).text = "탐지 모델 고도화", "2026-06-01 ~ 2026-10-31"
    t.cell(2, 0).text, t.cell(2, 1).text = "로그 수집 개선", "2025.03 ~ 2025.12"
    for row, color in zip((1, 2), colors):
        img = tmp_path / f"src_{name}_{row}.png"
        img.write_bytes(tiny_png(color))
        cell = t.cell(row, 2)
        if row == 2:  # 병합 칸 — row.cells 에 같은 칸이 두 번 나온다
            cell = t.cell(row, 1).merge(t.cell(row, 2))
        cell.paragraphs[-1].add_run().add_picture(str(img))
    path = tmp_path / name
    buf = io.BytesIO()
    d.save(buf)
    path.write_bytes(buf.getvalue())
    return str(path)


MERMAID = '```mermaid\nflowchart LR\n  subgraph S1[수집]\n    A1["1. 수집<br/>• 하루 1,200건"] --> A2["2. 분류"]\n  end\n```'


# ─── 1단계: 문서 읽기 — 그림 자리 표시 + 원본 보관 ──────────────────────────


def test_docx_figures_marked_in_place_and_merged_cells_counted_once(tmp_path):
    from core.doc_text import read_file

    doc = read_file(figure_docx(tmp_path))
    assert len(doc.images) == 2 and doc.images[0].ext == "png"
    assert "탐지 모델 고도화 | 2026-06-01 ~ 2026-10-31 | [그림 1]" in doc.text
    assert doc.text.count("[그림 2]") == 1  # 병합 칸이어도 한 번
    assert doc.table_heads == [["업무 명", "기간", "흐름도"]]
    assert doc.units == "문단 1개 · 표 1개"


# ─── 2단계: 전체 번호 + 그림 파일 보관 ───────────────────────────────────────


def test_figures_numbered_across_documents_and_saved_by_hash(tmp_path):
    from core.tools import career_docs

    old = figure_docx(tmp_path)
    new = figure_docx(tmp_path, "최신.docx", colors=((30, 160, 30), (30, 30, 200)))  # 두 번째 그림은 old 의 두 번째와 같은 내용
    res = career_docs.run("q", {"career_kind": "portfolio_update", "portfolio": old, "portfolio_name": "기술경력서.docx",
                                "docs": [new], "docs_names": ["최신.docx"]}, None)
    figs = res.harness_options["career_figures"]
    assert len(figs) == 4 and all(Path(p).parent == tmp_path and Path(p).name.startswith("figure_") for p in figs)
    assert figs[3] == figs[1] and len(set(figs)) == 3  # 같은 그림은 한 파일 (내용 해시 이름)
    old_text = res.harness_options["career_old_portfolio"]
    assert "[그림 1]" in old_text and "[그림 2]" in old_text and "[그림 3]" not in old_text
    assert "[그림 3]" in res.content and "[그림 4]" in res.content
    assert "| 최신.docx | docx | 문단 1개 · 표 1개 |" in res.content and "2개 ([그림 3]~[그림 4]) |" in res.content
    assert "그림 2개 ([그림 1]~[그림 2])" in res.content and "그림 자리" in res.content
    letter = career_docs.run("q", {"career_kind": "cover_letter", "docs": [new], "docs_names": ["최신.docx"],
                                   "questions": "1. 동기 (500자)"}, None)
    assert "career_figures" not in letter.harness_options  # 자소서에는 그림을 넣지 않는다


# ─── 3단계: 검사 ──────────────────────────────────────────────────────────


def test_portfolio_format_requires_every_figure_mark_and_numbers_pass_fact_check():
    fmt = {"mode": "original", "headings": ["1. 회사 업무"], "table_heads": [["업무 명", "기간", "흐름도"]], "sources": ["a.docx"]}
    ev = summary([("모델", "탐지 모델 고도화")])
    table = ("## 1. 회사 업무\n| 업무 명 | 기간 | 흐름도 |\n|---|---|---|\n| 탐지 모델 고도화 | 26.06 ~ 26.10 | [그림 1] |\n"
             "| 로그 수집 개선 | 25.03 ~ 25.12 | [그림 2] |")
    two = check("portfolio_format", table + ev, career_format=fmt, career_figures=["a.png", "b.png"])
    assert two.passed and "그림 표시 2개 모두 남음" in two.log
    lost = check("portfolio_format", table.replace("| [그림 2] |", "| - |") + ev, career_format=fmt, career_figures=["a.png", "b.png"])
    assert not lost.passed and "[그림 2] 이 빠졌다" in lost.log
    standard = check("portfolio_format", "## 요약\n없음" + ev, career_figures=["a.png"])  # 표준 형식 검사에도 붙는다
    assert not standard.passed and "[그림 1] 이 빠졌다" in standard.log
    source = "탐지 모델 고도화 | 2026-06-01 ~ 2026-10-31 | [그림 11]\n로그 수집 개선 | 2025.03 ~ 2025.12 | [그림 12]"
    big = table.replace("[그림 1]", "[그림 11]").replace("[그림 2]", "[그림 12]") + ev
    assert check("career_facts", big, career_source=source, career_reference="").passed  # [그림 12] 의 12 는 원문에 같은 표시


# ─── 4단계: 내보내기 — mermaid 유지 · Word 에 원본 그림 ─────────────────────


def test_body_keeps_mermaid_but_drops_json_and_mermaid_numbers_are_fact_checked():
    answer = "## 흐름\n" + MERMAID + summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    body = career.body_text(answer)
    assert "```mermaid" in body and "flowchart LR" in body and "evidence" not in body
    assert career.body_text("본문\n```python\nprint(1)\n```\n끝") == "본문\n\n끝"  # 다른 코드는 지금처럼 뺀다
    opts = {"career_source": DATED_SOURCE, "career_reference": ""}
    assert check("career_facts", answer, **opts).passed
    res = check("career_facts", answer.replace("하루 1,200건\"", "하루 3,400건\""), **opts)
    assert not res.passed and "3,400" in res.log  # 흐름도 노드의 숫자도 본문 사실 규칙
    doc = career.render_document(answer, "portfolio", title="포트폴리오")
    assert "```mermaid" in doc and 'A1["1. 수집<br/>• 하루 1,200건"]' in doc and '"evidence"' not in doc


def test_docx_export_puts_original_figures_in_place_and_mermaid_as_code_box(tmp_path):
    import docx

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(tiny_png())
    b.write_bytes(tiny_png((10, 10, 10), (3000, 10)))  # 아주 넓은 그림 — 본문 폭으로 줄인다
    md = ("# 경력서\n| 업무 명 | 흐름도 |\n|---|---|\n| 탐지 모델 | [그림 1] |\n\n- 로그 수집 [그림 2]\n\n[그림 3]\n\n" + MERMAID
          + "\n\n```json\n{\"evidence\": []}\n```")
    data = doc_export.to_docx(md, "경력서", figures=[str(a), str(b), str(tmp_path / "없는.png")])
    d = docx.Document(io.BytesIO(data))
    assert len(d.inline_shapes) == 2
    assert max(s.width for s in d.inline_shapes) <= docx.shared.Cm(doc_export.BODY_WIDTH_CM)
    text = "\n".join(p.text for p in d.paragraphs)
    assert "[그림 3]" in text and "[그림 1]" not in text and "[그림 2]" not in text  # 파일이 없는 그림만 글자로
    assert "flowchart LR" in text and "mermaid.live" in text and "evidence" not in text
    assert d.tables[0].cell(1, 1).text.strip() == ""  # 표 칸의 표시는 그림으로 바뀜
    plain = docx.Document(io.BytesIO(doc_export.to_docx(md, "경력서")))  # 그림 정보가 없으면 지금처럼 글자
    assert len(plain.inline_shapes) == 0 and "[그림 1]" in plain.tables[0].cell(1, 1).text


def test_hwpx_export_keeps_figure_marks_as_text_and_mermaid_lines():
    data = doc_export.to_hwpx("# 경력서\n| 업무 | 흐름도 |\n|---|---|\n| a | [그림 1] |\n\n" + MERMAID, "경력서")
    section = zipfile.ZipFile(io.BytesIO(data)).read("Contents/section0.xml").decode("utf-8")
    assert "[그림 1 — 원본 그림은 Word 파일 참고]" in section
    assert "flowchart LR" in section and "&lt;br/&gt;• 하루 1,200건" in section


# ─── 형식 피드백 5단계: 형식별 출력 두 개 · 원본 docx 를 틀로 ──────────────────

SPLIT = ("# 경력서 (가명)\n<!-- format:docx -->\n## 1. 회사 업무\n| 업무 명 | 기간 |\n|---|---|\n| 탐지 모델 고도화 | 26.06 ~ 26.10 |\n"
         "<!-- format:md -->\n## 회사\n### 탐지 모델 고도화 (26.06 ~ 26.10)\n" + MERMAID + "\n")


def test_format_sections_pick_body_per_output():
    answer = SPLIT + summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    secs = career.format_sections(career.body_text(answer))
    assert set(secs) == {"docx", "md"} and secs["docx"].startswith("# 경력서 (가명)") and secs["md"].startswith("# 경력서 (가명)")
    assert "| 업무 명 | 기간 |" in career.body_for(answer, "docx") and "### 탐지 모델" not in career.body_for(answer, "hwpx")
    assert "```mermaid" in career.body_for(answer, "md") and "| 업무 명" not in career.body_for(answer, "md")
    assert career.body_for("# 하나\n본문", "docx") == "# 하나\n본문"  # 구간이 없으면 지금처럼 본문 하나
    md, word = (career.render_document(answer, "portfolio", title="포트폴리오", target=t) for t in ("md", "docx"))
    assert "format:" not in md + word and "## 회사" in md and "## 1. 회사 업무" in word
    assert md.count("# 부록") == word.count("# 부록") == 1  # 부록(근거 인용)은 두 형식 모두
    assert career.section_problems(answer) == []
    half = answer.replace("<!-- format:md -->\n## 회사\n### 탐지 모델 고도화 (26.06 ~ 26.10)\n" + MERMAID, "<!-- format:md -->\n")
    assert career.section_problems(half) and "format:md" in career.section_problems(half)[0]
    res = check("portfolio_format", half, career_format={"mode": "original", "headings": [], "table_heads": []})
    assert not res.passed and "format:md" in res.log


def test_docx_template_keeps_original_styles_margins_header_but_not_old_body(tmp_path):
    import docx
    from docx.shared import Cm

    t = docx.Document()
    t.sections[0].left_margin = Cm(1.5)
    t.sections[0].header.paragraphs[0].text = "머리글 (가명)"
    t.styles["Normal"].font.name = "Batang"
    t.add_paragraph("옛 본문 (지워져야 함)")
    old_table = t.add_table(rows=1, cols=2)
    old_table.style = "Light Grid Accent 1"
    path = tmp_path / "틀.docx"
    t.save(str(path))
    data = doc_export.to_docx("# 경력서\n## 1. 회사 업무\n| 업무 명 | 기간 |\n|---|---|\n| 탐지 모델 | 26.06 |\n\n- 글머리", "경력서",
                              template=str(path))
    d = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in d.paragraphs)
    assert "옛 본문" not in text and "1. 회사 업무" in text
    assert abs(d.sections[0].left_margin - Cm(1.5)) < Cm(0.01)  # Word 는 여백을 트윕 단위로 저장 (반올림)
    assert d.sections[0].header.paragraphs[0].text == "머리글 (가명)"
    assert d.styles["Normal"].font.name == "Batang" and len(d.tables) == 1 and d.tables[0].style.name == "Light Grid Accent 1"
    assert all(r.font.name is None for p in d.paragraphs for r in p.runs)  # 틀의 글꼴을 덮어쓰지 않는다
    assert d.core_properties.author == ""
    broken = tmp_path / "깨짐.docx"
    broken.write_bytes(b"not a docx")
    assert docx.Document(io.BytesIO(doc_export.to_docx("# 제목", "t", template=str(broken)))).paragraphs  # 깨진 틀 → 기본 양식


# ─── 5단계: 가짜 docx + 가짜 모델로 포트폴리오 최신화 끝까지 ────────────────────

NEW_MD = "최신 요약 (가명)\n탐지 모델 고도화 (2026-06-01 ~ 2026-10-31)\n하루 1,200건을 자동 분류했다.\n"
Q = "하루 1,200건을 자동 분류했다."


def update_answer(with_second_figure: bool = True) -> str:
    rows = ("| 탐지 모델 고도화 | 26.06 ~ 26.10 | [그림 1] |\n| 로그 수집 개선 | 25.03 ~ 25.12 | "
            + ("[그림 2]" if with_second_figure else "-") + " |")
    data = {"changes": [{"part": "1. 회사 업무 > 탐지 모델 고도화", "before": "탐지 모델 고도화", "after": "탐지 모델 고도화 (26.06 ~ 26.10)",
                         "why": "날짜 표기 통일", "source": Q, "kind": "문서"}],
            "evidence": [{"claim": "자동 분류", "quote": Q}], "jd_coverage": [], "needs_check": []}
    return ("# 기술 경력서 (최신화)\n## 1. 회사 업무\n| 업무 명 | 기간 | 흐름도 |\n|---|---|---|\n" + rows + "\n\n"
            + MERMAID + "\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```")


def test_fake_model_portfolio_update_keeps_original_figures_in_word(tmp_path, ranks, models, use_script):
    import docx

    import career_ui
    from core.config import load_team, team_org
    from core.tasks import build_request
    from core.workflow import build_office_graph, stream_office
    from tests.fakes import Script, make_review

    new = tmp_path / "최신.md"
    new.write_text(NEW_MD, encoding="utf-8")
    fact = ("## 사실 카드\n| # | 경험 | 원문 인용 |\n|---|---|---|\n| 1 | 자동 분류 | " + Q + " |\n\n```json\n"
            + json.dumps({"evidence": [{"claim": "자동 분류", "quote": Q}], "jd_coverage": [], "needs_check": []}, ensure_ascii=False)
            + "\n```")
    reviews = [make_review("revise", 8, update_answer(False)), make_review("revise", 8, update_answer()),
               make_review("approve", 9), make_review("approve", 9)]
    script = use_script(Script(text=fact, ReviewResult=reviews))
    team = load_team("career")
    task = next(t for t in team.tasks if t.id == "portfolio_update")
    values = {"portfolio": figure_docx(tmp_path), "portfolio_name": "기술경력서.docx", "docs": [str(new)], "docs_names": ["최신.md"],
              "docs_name": "최신.md", "fmt_docx": True, "fmt_md": True}
    req = build_request(task, values, "Word 는 기존 기술경력서 형식과 유사하게")
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    events = list(stream_office(graph, org, req.question, "lead", brief=req.brief, auto_brief=False, tool_inputs=req.tool_inputs,
                                extra_validators=req.validators, harness_options=req.harness_options))
    tools = next(p for k, p in events if k == "tools")
    assert len(tools[0].harness_options["career_figures"]) == 2
    assert tools[0].harness_options["career_format"]["table_heads"] == [["업무 명", "기간", "흐름도"]]
    feedback = next(p for p in script.prompts("ReviewResult") if "자동 검증(하네스)" in p)
    assert "[그림 2] 이 빠졌다" in feedback  # 그림 표시를 지운 초안은 되돌려 보낸다
    final = next(s for k, s in events if k == "step" and s.node_id == "lead")
    assert {"career_facts", "portfolio_format", "portfolio_changes"} <= {h.validator for h in final.harness}
    assert all(h.passed for h in final.harness), [h.log for h in final.harness if not h.passed]

    run = {"run_id": "r9", "tools": tools, "career_inputs": req.tool_inputs}
    out = career_ui.build(run, final.content)
    assert out["figures"] == 2 and out["missing_figures"] == [] and out["facts"].ok
    word = docx.Document(io.BytesIO(out["files"]["포트폴리오(최신화).docx"]))
    assert len(word.inline_shapes) == 2  # 원본 그림 2개가 Word 의 같은 자리(표 칸)에
    md = out["files"]["포트폴리오(최신화).md"].decode("utf-8")
    assert "[그림 1]" in md and "[그림 2]" in md and "```mermaid" in md and "flowchart LR" in md
    assert out["template"] and tools[0].harness_options["career_format"]["template"] == values["portfolio"]  # 원본 Word 를 틀로
    assert "탐지 모델 고도화 | 2026-06-01" not in "\n".join(c.text for t in word.tables for row in t.rows for c in row.cells)  # 옛 본문은 지움
    media = [n for n in zipfile.ZipFile(io.BytesIO(out["files"]["포트폴리오(최신화).docx"])).namelist() if n.startswith("word/media/")]
    assert len(media) == 2  # 넣은 그림 2개만 — 틀(원본)의 옛 본문 그림이 숨어 남지 않는다
