"""마크다운 → Word(.docx) · 한글(.hwpx) 파일 (포트폴리오·자소서 팀 내려받기).

마크다운은 이 팀 결과에 나오는 모양만 다룬다: 제목(# ~ ####), 문단, 글머리표(- · 1.), 표(| a | b |), 인용(>), 구분선(---),
문단 안의 **굵게**·`코드`·[글](주소), ```mermaid 흐름도(코드 상자 + 이미지로 바꾸라는 안내 — JSON 등 다른 코드 블록은 뺀다).
'[그림 N]' 표시: .docx 는 그 자리에 원본 그림(to_docx figures), .hwpx 는 '[그림 N — 원본 그림은 Word 파일 참고]' 글자.
그 밖의 문법은 글자 그대로 들어간다.

  .docx : python-docx — 제목 스타일·표(Table Grid, 머리 줄 음영)·글머리표, 글꼴 맑은 고딕
  .hwpx : 한글 2014 이상이 여는 공개 XML 형식(OWPML, KS X 6101) — zip 안에 mimetype·version.xml·META-INF·Contents/header.xml·
          Contents/section0.xml 을 직접 쓴다 (외부 라이브러리 없음). 바이너리 .hwp 는 형식이 비공개에 가까워 만들지 않는다
          (.hwpx 를 한글에서 열어 '다른 이름으로 저장 → .hwp' 하면 된다).
"""

import io
import re
import zipfile
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

FONT = "맑은 고딕"

# ─── 마크다운 → 블록 ───────────────────────────────────────────────────


@dataclass
class Block:
    kind: str  # h · p · li · table · quote · hr · code
    text: str = ""
    lang: str = ""  # code 블록의 언어 (mermaid)
    level: int = 0  # 제목 수준(1~4) / 글머리 들여쓰기
    ordered: str = ""  # 번호 목록이면 '1.' 같은 번호
    rows: list[list[str]] = field(default_factory=list)


CODE_LANGS = ("mermaid",)  # 문서에 남기는 코드 블록 언어
MERMAID_NOTE = "(mermaid 흐름도 — Word 는 mermaid 를 그리지 못합니다. mermaid.live 등에서 그림으로 바꿔 이 자리에 넣으세요)"
FIGURE_RX = re.compile(r"\[그림\s*(\d{1,3})\]")
HWPX_FIGURE = "[그림 {n} — 원본 그림은 Word 파일 참고]"
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def xml_safe(text: str) -> str:
    """Word·한글 XML 에 쓸 수 없는 제어 문자를 뺀다 (쪽 나눔 \\x0c · 세로 탭 \\x0b 은 줄바꿈으로).

    PDF 글자 층·옛 문서에서 온 글에 섞이면 python-docx 는 예외로 멈추고(.docx 를 못 만듦), .hwpx 는 열리지 않는 파일이 된다.
    """
    return XML_BAD.sub("", str(text or "").replace("\x0c", "\n").replace("\x0b", "\n"))


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = xml_safe(text).replace("\r\n", "\n").split("\n")
    i, para = 0, []

    def flush() -> None:
        if para:
            blocks.append(Block("p", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):  # 코드 블록 — mermaid 흐름도만 문서에 넣고 나머지(JSON 요약 등)는 뺀다
            flush()
            lang = stripped[3:].strip().split(" ")[0].lower()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            if lang in CODE_LANGS:
                blocks.append(Block("code", "\n".join(code).strip("\n"), lang=lang))
            continue
        if not stripped:
            flush()
        elif m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            flush()
            blocks.append(Block("h", m.group(2).strip().rstrip("#").strip(), level=min(len(m.group(1)), 4)))
        elif re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush()
            blocks.append(Block("hr"))
        elif stripped.startswith("|") and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_cells(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            width = max(len(r) for r in rows)
            blocks.append(Block("table", rows=[r + [""] * (width - len(r)) for r in rows]))
            continue
        elif m := re.match(r"^(\s*)([-*+]|\d{1,2}[.)])\s+(.*)$", line):
            flush()
            marker = m.group(2)
            blocks.append(Block("li", m.group(3).strip(), level=len(m.group(1).replace("\t", "  ")) // 2,
                                ordered=marker if marker[0].isdigit() else ""))
        elif stripped.startswith(">"):
            flush()
            blocks.append(Block("quote", stripped.lstrip(">").strip()))
        else:
            para.append(line)
        i += 1
    flush()
    _renumber(blocks)
    return blocks


def _renumber(blocks: list[Block]) -> None:
    """번호 목록을 목록마다 첫 번호부터 다시 센다 — 프로젝트마다 '1. 2. 3.' 이 이어지지 않게, '1. 1. 1.' 로 적어도 1·2·3 으로."""
    counters: dict[int, tuple[int, str]] = {}  # 들여쓰기 수준 → (지금 번호, 뒤 기호 '.' 또는 ')')
    for block in blocks:
        if block.kind != "li":
            counters.clear()
            continue
        for deeper in [lv for lv in counters if lv > block.level]:
            del counters[deeper]
        if not block.ordered:
            counters.pop(block.level, None)
            continue
        if block.level in counters:
            number, mark = counters[block.level][0] + 1, counters[block.level][1]
        else:
            number, mark = int(block.ordered[:-1]), block.ordered[-1]
        counters[block.level] = (number, mark)
        block.ordered = f"{number}{mark}"


INLINE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)")


def inline_runs(text: str) -> list[tuple[str, bool]]:
    """문단 글 → [(글, 굵게?)] — **굵게**만 살리고 `코드`·[글](주소)는 글자로."""
    runs, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], False))
        if m.group(1) or m.group(2):
            runs.append((m.group(1) or m.group(2), True))
        elif m.group(3):
            runs.append((m.group(3), False))
        else:
            runs.append((f"{m.group(4)} ({m.group(5)})", False))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], False))
    return [(t.replace("\\|", "|").replace("<br>", "\n"), b) for t, b in runs if t]


def plain(text: str) -> str:
    return "".join(t for t, _ in inline_runs(text))


# ─── Word (.docx) ─────────────────────────────────────────────────────


def _font(run_or_style, size: float | None = None, bold: bool | None = None) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    font = run_or_style.font
    font.name = FONT
    if size:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    element = run_or_style.element if hasattr(run_or_style, "element") else run_or_style._element
    rpr = element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)


def _add_runs(paragraph, text: str, bold_all: bool = False, keep_font: bool = False) -> None:
    for chunk, bold in inline_runs(text):
        run = paragraph.add_run(chunk)
        run.bold = bold or bold_all
        if not keep_font:  # 틀(원본 docx)을 쓰면 틀의 글꼴을 그대로
            _font(run)


def _shade(cell, color: str) -> None:
    from docx.oxml.ns import qn

    tcpr = cell._element.get_or_add_tcPr()
    shd = tcpr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): color})
    tcpr.append(shd)


BODY_WIDTH_CM = 16.0  # A4 본문 폭 (기본 여백) — 그림이 이보다 크면 줄인다


def split_figures(text: str) -> tuple[str, list[int]]:
    """글 → ('[그림 N]' 을 뺀 글, 그림 번호들)."""
    numbers = [int(n) for n in FIGURE_RX.findall(text or "")]
    rest = FIGURE_RX.sub("", text or "") if numbers else (text or "")
    return re.sub(r"[ \t]{2,}", " ", rest).strip(), numbers


def _figure_path(n: int, figures: list[str] | None) -> str | None:
    from pathlib import Path

    if figures and 1 <= n <= len(figures) and Path(figures[n - 1]).is_file():
        return figures[n - 1]
    return None


def _add_picture(paragraph, path: str, max_cm: float) -> bool:
    """원본 그림을 원래 크기로 넣되 max_cm 를 넘으면 줄인다. Word 가 못 읽는 형식(emf 등)이면 False."""
    from docx.image.image import Image
    from docx.shared import Cm, Emu

    try:
        width = min(int(Image.from_file(path).width), int(Cm(max_cm)))
        paragraph.add_run().add_picture(path, width=Emu(width))
        return True
    except Exception:  # noqa: BLE001 — 깨진 그림·지원하지 않는 형식 → 글자 표시로
        return False


def _cell_paragraphs(cell, reuse_first: bool):
    """표 칸에 그림 문단을 차례로 — 칸 글이 비었으면 첫 문단을 쓰고, 그다음부터 새 문단."""
    used = [not reuse_first]

    def next_paragraph():
        if not used[0]:
            used[0] = True
            return cell.paragraphs[0]
        return cell.add_paragraph()
    return next_paragraph


def _shade_paragraph(paragraph, color: str) -> None:
    from docx.oxml.ns import qn

    ppr = paragraph._p.get_or_add_pPr()
    ppr.append(ppr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): color}))


def _clear_body(document) -> str | None:
    """틀(원본 docx)의 본문을 비운다 — 용지·여백·머리글/바닥글(구역 설정)과 스타일은 남긴다. 원본 표의 스타일 이름을 돌려준다."""
    from docx.oxml.ns import qn

    table_style = None
    for table in document.tables:
        if table.style is not None:
            table_style = table.style.name
            break
    body = document.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    part = document.part  # 지운 본문이 쓰던 그림은 파일 안에 숨어 남지 않게 연결을 끊는다 (머리글·바닥글 그림은 그대로)
    for rid, rel in list(part.rels.items()):
        if rel.reltype.endswith("/image"):
            part.drop_rel(rid)
    return table_style


def to_docx(markdown: str, title: str = "", figures: list[str] | None = None, template: str | None = None) -> bytes:
    """마크다운 → Word. figures = 원본 그림 경로 ([그림 1] = 0번) — 글의 '[그림 N]' 자리(문단·글머리·표 칸 아래)에 원본 그림을 넣는다.

    그림 파일이 없거나 Word 가 못 읽는 형식이면 '[그림 N]' 글자를 그대로 둔다. ```mermaid 는 회색 코드 상자 + 이미지로 바꾸라는 안내.
    template = 원본 docx (형식 지정이 있을 때) — 그 파일의 스타일·용지·여백·머리글/바닥글·표 스타일을 틀로 쓰고 본문만 새로 쓴다
    (원본 본문은 지운다. 원본 표 칸을 제자리에서 채우지는 않는다 — 표 대응은 다음 단계). 틀을 열지 못하면 기본 양식.
    """
    import docx
    from docx.shared import Pt, RGBColor

    document, table_style = None, None
    if template:
        try:
            document = docx.Document(template)
            table_style = _clear_body(document)
        except Exception:  # noqa: BLE001 — 깨진 틀은 기본 양식으로
            document, table_style = None, None
    styled = document is not None  # 틀의 글꼴·색을 그대로 둔다
    if document is None:
        document = docx.Document()
        _font(document.styles["Normal"], 10.5)
        for level, size in ((1, 18), (2, 14), (3, 12), (4, 11)):
            style = document.styles[f"Heading {level}"]
            _font(style, size, True)
            style.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    if title:
        document.core_properties.title = xml_safe(title)
    document.core_properties.author = ""  # 개인 정보(작성자 이름)를 파일 속성에 남기지 않는다
    document.core_properties.last_modified_by = ""

    def add_runs(paragraph, text: str, bold_all: bool = False) -> None:
        _add_runs(paragraph, text, bold_all, keep_font=styled)

    def add_heading(level: int):
        try:
            return document.add_heading(level=level)
        except KeyError:  # 틀에 제목 스타일이 없음 → 굵은 문단
            p = document.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            return p

    def add_bullet(level: int):
        try:
            return document.add_paragraph(style="List Bullet 2" if level else "List Bullet")
        except KeyError:  # 틀에 글머리 스타일이 없음 → 들여쓴 '•' 문단
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(12 + 12 * level)
            p.add_run("◦ " if level else "• ")
            return p

    def pictures(numbers: list[int], new_paragraph, max_cm: float, indent: float = 0) -> None:
        """그림 번호마다 새 문단에 원본 그림 (없으면 '[그림 N]' 글자)."""
        for n in numbers:
            p = new_paragraph()
            if indent:
                p.paragraph_format.left_indent = Pt(indent)
            path = _figure_path(n, figures)
            if not (path and _add_picture(p, path, max_cm)):
                add_runs(p, f"[그림 {n}]")

    for block in parse_markdown(markdown):
        text, numbers = split_figures(block.text) if figures and block.kind in ("p", "li") else (block.text, [])
        if block.kind == "h":
            heading = add_heading(block.level)
            add_runs(heading, block.text, bold_all=heading.style is None or not heading.style.name.lower().startswith("heading"))
        elif block.kind == "p":
            if text or not numbers:
                add_runs(document.add_paragraph(), text)
            pictures(numbers, document.add_paragraph, BODY_WIDTH_CM)
        elif block.kind == "li" and block.ordered:
            # Word 의 'List Number' 는 문서 전체가 번호 하나를 이어 쓴다 → 목록마다 다시 센 번호를 글자로 쓰고 내어쓰기
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(18 + 18 * block.level)
            p.paragraph_format.first_line_indent = Pt(-18)
            add_runs(p, f"{block.ordered}\t{text}")
            p.paragraph_format.tab_stops.add_tab_stop(Pt(18 + 18 * block.level))
            pictures(numbers, document.add_paragraph, BODY_WIDTH_CM - 1, 18 + 18 * block.level)
        elif block.kind == "li":
            add_runs(add_bullet(block.level), text)
            pictures(numbers, document.add_paragraph, BODY_WIDTH_CM - 1, 18 + 18 * block.level)
        elif block.kind == "code":
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(6)
            _shade_paragraph(p, "F2F2F2")
            run = p.add_run(block.text)  # 줄바꿈은 python-docx 가 줄 나눔(w:br)으로 바꾼다
            _font(run, 9)
            run.font.name = "Consolas"
            note = document.add_paragraph()
            note_run = note.add_run(MERMAID_NOTE if block.lang == "mermaid" else f"({block.lang} 코드)")
            _font(note_run, 9)
            note_run.italic = True
            note_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        elif block.kind == "quote":
            p = document.add_paragraph()
            p.paragraph_format.left_indent = Pt(12)
            for chunk, bold in inline_runs(block.text):
                run = p.add_run(chunk)
                run.bold, run.italic = bold, True
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
                _font(run)
        elif block.kind == "hr":
            document.add_paragraph("")
        elif block.kind == "table":
            table = document.add_table(rows=len(block.rows), cols=len(block.rows[0]))
            for name in (table_style, "Table Grid"):  # 틀의 원본 표 스타일 → 없으면 격자
                try:
                    if name:
                        table.style = name
                        break
                except (KeyError, ValueError):
                    continue
            cell_cm = max(3.0, BODY_WIDTH_CM / len(block.rows[0]) - 0.4)
            for r, row in enumerate(block.rows):
                for c, value in enumerate(row):
                    cell = table.cell(r, c)
                    cell.text = ""
                    value, numbers = split_figures(value) if figures else (value, [])
                    add_runs(cell.paragraphs[0], value, bold_all=r == 0)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9.5)
                    # 표 칸의 그림 — 칸 폭에 맞춰 줄인다 (원본 기술경력서는 흐름도가 모두 표 칸 안에 있었다)
                    pictures(numbers, _cell_paragraphs(cell, reuse_first=not value), cell_cm)
                    if r == 0 and not table_style:  # 틀의 표 스타일이 있으면 그 머리 줄 모양을 따른다
                        _shade(cell, "E8EEF7")
            document.add_paragraph("")
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


# ─── 한글 (.hwpx) ─────────────────────────────────────────────────────

NS = {
    "ha": "http://www.hancom.co.kr/hwpml/2011/app",
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hp10": "http://www.hancom.co.kr/hwpml/2016/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hhs": "http://www.hancom.co.kr/hwpml/2011/history",
    "hm": "http://www.hancom.co.kr/hwpml/2011/master-page",
    "hpf": "http://www.hancom.co.kr/schema/2011/hpf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "opf": "http://www.idpf.org/2007/opf/",
    "ooxmlchart": "http://www.hancom.co.kr/hwpml/2016/ooxmlchart",
    "hwpunitchar": "http://www.hancom.co.kr/hwpml/2016/HwpUnitChar",
    "epub": "http://www.idpf.org/2007/ops",
    "config": "urn:oasis:names:tc:opendocument:xmlns:config:1.0",
}
XMLNS = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
LANGS = ("HANGUL", "LATIN", "HANJA", "JAPANESE", "OTHER", "SYMBOL", "USER")
PAGE_W, PAGE_H, MARGIN_LR = 59528, 84186, 7087  # A4, 좌우 여백 25mm (1mm = 283.46 HWPUNIT)
TEXT_W = PAGE_W - 2 * MARGIN_LR

# 글자 모양 id: 0 본문 · 1 본문 굵게 · 2 제목1 · 3 제목2 · 4 제목3 · 5 표 · 6 표 굵게 · 7 인용(회색)
CHAR_PR = [(1050, False, "#000000"), (1050, True, "#000000"), (1800, True, "#1F3A5F"), (1400, True, "#1F3A5F"),
           (1200, True, "#1F3A5F"), (950, False, "#000000"), (950, True, "#000000"), (1000, False, "#555555")]
# 문단 모양 id: 0 본문(양쪽) · 1 제목(왼쪽, 위 간격) · 2 글머리 · 3 표 칸 · 4 인용(들여쓰기)
PARA_PR = [("JUSTIFY", 0, 0, 0, 160), ("LEFT", 0, 1400, 400, 150), ("LEFT", 1400, 0, 0, 150),
           ("LEFT", 0, 0, 0, 130), ("LEFT", 1400, 0, 0, 150)]


def _char_pr(i: int, height: int, bold: bool, color: str) -> str:
    seven = lambda v: " ".join(f'{lang.lower()}="{v}"' for lang in LANGS)  # noqa: E731
    return (f'<hh:charPr id="{i}" height="{height}" textColor="{color}" shadeColor="none" useFontSpace="0" useKerning="0" '
            f'symMark="NONE" borderFillIDRef="1"><hh:fontRef {seven(0)}/><hh:ratio {seven(100)}/><hh:spacing {seven(0)}/>'
            f'<hh:relSz {seven(100)}/><hh:offset {seven(0)}/>{"<hh:bold/>" if bold else ""}'
            '<hh:underline type="NONE" shape="SOLID" color="#000000"/><hh:strikeout shape="NONE" color="#000000"/>'
            '<hh:outline type="NONE"/><hh:shadow type="NONE" color="#B2B2B2" offsetX="10" offsetY="10"/></hh:charPr>')


def _para_pr(i: int, align: str, left: int, prev: int, nxt: int, line: int) -> str:
    unit = lambda tag, v: f'<hc:{tag} value="{v}" unit="HWPUNIT"/>'  # noqa: E731
    return (f'<hh:paraPr id="{i}" tabPrIDRef="0" condense="0" fontLineHeight="0" snapToGrid="1" suppressLineNumbers="0" '
            f'checked="0"><hh:align horizontal="{align}" vertical="BASELINE"/><hh:heading type="NONE" idRef="0" level="0"/>'
            '<hh:breakSetting breakLatinWord="KEEP_WORD" breakNonLatinWord="KEEP_WORD" widowOrphan="0" keepWithNext="0" '
            'keepLines="0" pageBreakBefore="0" lineWrap="BREAK"/><hh:autoSpacing eAsianEng="0" eAsianNum="0"/>'
            f'<hh:margin>{unit("intent", 0)}{unit("left", left)}{unit("right", 0)}{unit("prev", prev)}{unit("next", nxt)}</hh:margin>'
            f'<hh:lineSpacing type="PERCENT" value="{line}" unit="HWPUNIT"/>'
            '<hh:border borderFillIDRef="1" offsetLeft="0" offsetRight="0" offsetTop="0" offsetBottom="0" connect="0" '
            'ignoreMargin="0"/></hh:paraPr>')


def _border_fill(i: int, line: str, fill: str = "") -> str:
    side = lambda tag: f'<hh:{tag} type="{line}" width="0.12 mm" color="#000000"/>'  # noqa: E731
    brush = (f'<hc:fillBrush><hc:winBrush faceColor="{fill}" hatchColor="#999999" alpha="0"/></hc:fillBrush>' if fill else "")
    return (f'<hh:borderFill id="{i}" threeD="0" shadow="0" centerLine="NONE" breakCellSeparateLine="0">'
            '<hh:slash type="NONE" Crooked="0" isCounter="0"/><hh:backSlash type="NONE" Crooked="0" isCounter="0"/>'
            f'{side("leftBorder")}{side("rightBorder")}{side("topBorder")}{side("bottomBorder")}'
            f'<hh:diagonal type="SOLID" width="0.1 mm" color="#000000"/>{brush}</hh:borderFill>')


def _header_xml() -> str:
    font = (f'<hh:font id="0" face="{FONT}" type="TTF" isEmbedded="0"><hh:typeInfo familyType="FCAT_GOTHIC" weight="6" '
            'proportion="4" contrast="0" strokeVariation="1" armStyle="1" letterform="1" midline="1" xHeight="1"/></hh:font>')
    faces = "".join(f'<hh:fontface lang="{lang}" fontCnt="1">{font}</hh:fontface>' for lang in LANGS)
    fills = _border_fill(1, "NONE") + _border_fill(2, "SOLID") + _border_fill(3, "SOLID", "#E8EEF7")
    chars = "".join(_char_pr(i, *c) for i, c in enumerate(CHAR_PR))
    paras = "".join(_para_pr(i, *p) for i, p in enumerate(PARA_PR))
    return (f'{DECL}<hh:head {XMLNS} version="1.4" secCnt="1">'
            '<hh:beginNum page="1" footnote="1" endnote="1" pic="1" tbl="1" equation="1"/><hh:refList>'
            f'<hh:fontfaces itemCnt="{len(LANGS)}">{faces}</hh:fontfaces>'
            f'<hh:borderFills itemCnt="3">{fills}</hh:borderFills>'
            f'<hh:charProperties itemCnt="{len(CHAR_PR)}">{chars}</hh:charProperties>'
            '<hh:tabProperties itemCnt="1"><hh:tabPr id="0" autoTabLeft="0" autoTabRight="0"/></hh:tabProperties>'
            f'<hh:paraProperties itemCnt="{len(PARA_PR)}">{paras}</hh:paraProperties>'
            '<hh:styles itemCnt="1"><hh:style id="0" type="PARA" name="바탕글" engName="Normal" paraPrIDRef="0" charPrIDRef="0" '
            'nextStyleIDRef="0" langID="1042" lockForm="0"/></hh:styles>'
            '</hh:refList><hh:compatibleDocument targetProgram="HWP201X"><hh:layoutCompatibility/></hh:compatibleDocument>'
            '<hh:docOption><hh:linkinfo path="" pageInherit="0" footnoteInherit="0"/></hh:docOption>'
            '<hh:trackchageConfig flags="56"/></hh:head>')


SEC_PR = (
    '<hp:secPr id="" textDirection="HORIZONTAL" spaceColumns="1134" tabStop="8000" tabStopVal="4000" tabStopUnit="HWPUNIT" '
    'outlineShapeIDRef="0" memoShapeIDRef="0" textVerticalWidthHead="0" masterPageCnt="0">'
    '<hp:grid lineGrid="0" charGrid="0" wonggojiFormat="0"/><hp:startNum pageStartsOn="BOTH" page="0" pic="0" tbl="0" equation="0"/>'
    '<hp:visibility hideFirstHeader="0" hideFirstFooter="0" hideFirstMasterPage="0" border="SHOW_ALL" fill="SHOW_ALL" '
    'hideFirstPageNum="0" hideFirstEmptyLine="0" showLineNumber="0"/>'
    '<hp:lineNumberShape restartType="0" countBy="0" distance="0" startNumber="0"/>'
    f'<hp:pagePr landscape="WIDELY" width="{PAGE_W}" height="{PAGE_H}" gutterType="LEFT_ONLY">'
    f'<hp:margin header="4252" footer="4252" gutter="0" left="{MARGIN_LR}" right="{MARGIN_LR}" top="5668" bottom="4252"/></hp:pagePr>'
    '<hp:footNotePr><hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
    '<hp:noteLine length="-1" type="SOLID" width="0.12 mm" color="#000000"/>'
    '<hp:noteSpacing betweenNotes="283" belowLine="567" aboveLine="850"/><hp:numbering type="CONTINUOUS" newNum="1"/>'
    '<hp:placement place="EACH_COLUMN" beneathText="0"/></hp:footNotePr>'
    '<hp:endNotePr><hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
    '<hp:noteLine length="14692344" type="SOLID" width="0.12 mm" color="#000000"/>'
    '<hp:noteSpacing betweenNotes="0" belowLine="567" aboveLine="850"/><hp:numbering type="CONTINUOUS" newNum="1"/>'
    '<hp:placement place="END_OF_DOCUMENT" beneathText="0"/></hp:endNotePr>'
    + "".join(f'<hp:pageBorderFill type="{t}" borderFillIDRef="1" textBorder="PAPER" headerInside="0" footerInside="0" '
              'fillArea="PAPER"><hp:offset left="1417" right="1417" top="1417" bottom="1417"/></hp:pageBorderFill>'
              for t in ("BOTH", "EVEN", "ODD"))
    + '</hp:secPr>'
)
COL_PR = '<hp:ctrl><hp:colPr id="" type="NEWSPAPER" layout="LEFT" colCount="1" sameSz="1" sameGap="0"/></hp:ctrl>'


class _Hwpx:
    def __init__(self) -> None:
        self.paras: list[str] = []
        self.next_id = 0

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    @staticmethod
    def _runs(text: str, char_pr: int, bold_pr: int) -> str:
        out = []
        for chunk, bold in inline_runs(text):
            parts = chunk.split("\n")
            body = "<hp:lineBreak/>".join(escape(p) for p in parts)
            out.append(f'<hp:run charPrIDRef="{bold_pr if bold else char_pr}"><hp:t>{body}</hp:t></hp:run>')
        return "".join(out) or f'<hp:run charPrIDRef="{char_pr}"/>'

    def para(self, text: str, para_pr: int = 0, char_pr: int = 0, bold_pr: int = 1, extra: str = "") -> str:
        return (f'<hp:p id="{self._id()}" paraPrIDRef="{para_pr}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                f'{extra}{self._runs(text, char_pr, bold_pr)}</hp:p>')

    def raw_para(self, text: str, para_pr: int = 0, char_pr: int = 0) -> str:
        """마크다운으로 읽지 않고 글자 그대로 한 문단 (코드 줄)."""
        return (f'<hp:p id="{self._id()}" paraPrIDRef="{para_pr}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                f'<hp:run charPrIDRef="{char_pr}"><hp:t>{escape(text)}</hp:t></hp:run></hp:p>')

    def table(self, rows: list[list[str]]) -> str:
        n_rows, n_cols = len(rows), len(rows[0])
        widths = _col_widths(rows, TEXT_W)
        row_h = 1800
        trs = []
        for r, row in enumerate(rows):
            tcs = []
            for c, value in enumerate(row):
                head = r == 0
                cell_p = self.para(value, 3, 6 if head else 5, 6)
                tcs.append(
                    f'<hp:tc name="" header="{int(head)}" hasMargin="0" protect="0" editable="0" dirty="0" '
                    f'borderFillIDRef="{3 if head else 2}"><hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" '
                    'vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" '
                    f'hasNumRef="0">{cell_p}</hp:subList><hp:cellAddr colAddr="{c}" rowAddr="{r}"/>'
                    f'<hp:cellSpan colSpan="1" rowSpan="1"/><hp:cellSz width="{widths[c]}" height="{row_h}"/>'
                    '<hp:cellMargin left="510" right="510" top="141" bottom="141"/></hp:tc>')
            trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
        tbl = (f'<hp:tbl id="{self._id()}" zOrder="0" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" '
               f'lock="0" dropcapstyle="None" pageBreak="CELL" repeatHeader="1" rowCnt="{n_rows}" colCnt="{n_cols}" '
               'cellSpacing="0" borderFillIDRef="2" noAdjust="0">'
               f'<hp:sz width="{sum(widths)}" widthRelTo="ABSOLUTE" height="{row_h * n_rows}" heightRelTo="ABSOLUTE" protect="0"/>'
               '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" '
               'horzRelTo="COLUMN" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               '<hp:outMargin left="283" right="283" top="283" bottom="283"/>'
               '<hp:inMargin left="510" right="510" top="141" bottom="141"/>' + "".join(trs) + "</hp:tbl>")
        return (f'<hp:p id="{self._id()}" paraPrIDRef="3" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
                f'<hp:run charPrIDRef="5">{tbl}<hp:t/></hp:run></hp:p>')


def _col_widths(rows: list[list[str]], total: int) -> list[int]:
    """칸 너비 — 글 길이에 비례하되 한 칸이 너무 좁거나 넓지 않게."""
    n = len(rows[0])
    lengths = [max(4, min(40, max(len(plain(r[c])) for r in rows))) for c in range(n)]
    s = sum(lengths)
    widths = [int(total * v / s) for v in lengths]
    widths[-1] += total - sum(widths)
    return widths


def to_hwpx(markdown: str, title: str = "") -> bytes:
    doc = _Hwpx()
    blocks = parse_markdown(markdown)
    body: list[str] = []
    first_extra = f'<hp:run charPrIDRef="0">{SEC_PR}{COL_PR}</hp:run>'  # 첫 문단에 구역(용지·여백) 정의
    def figure_text(text: str) -> str:  # hwpx 에는 그림을 넣지 않는다 (BinData·manifest 가 필요 — 다음 단계)
        return FIGURE_RX.sub(lambda m: HWPX_FIGURE.format(n=m.group(1)), text or "")

    for block in blocks:
        extra, first_extra = first_extra, ""
        block.text = figure_text(block.text)
        block.rows = [[figure_text(c) for c in row] for row in block.rows]
        if block.kind == "h":
            char_pr = {1: 2, 2: 3}.get(block.level, 4)
            body.append(doc.para(block.text, 1, char_pr, char_pr, extra))
        elif block.kind == "p":
            body.append(doc.para(block.text, 0, 0, 1, extra))
        elif block.kind == "li":
            bullet = f"{block.ordered} " if block.ordered else ("◦ " if block.level else "• ")
            body.append(doc.para(bullet + block.text, 2, 0, 1, extra))
        elif block.kind == "quote":
            body.append(doc.para(block.text, 4, 7, 1, extra))
        elif block.kind == "hr":
            body.append(doc.para("", 0, 0, 1, extra))
        elif block.kind == "code":  # mermaid — 코드 글자 그대로 + 안내 (`·** 같은 글자를 마크다운으로 읽지 않게 한 줄씩 그대로)
            body.append(doc.para("", 4, 7, 7, extra))
            body += [doc.raw_para(line, 4, 7) for line in block.text.split("\n")]
            body.append(doc.para(MERMAID_NOTE.replace("Word", "한글"), 4, 7, 7))
        elif block.kind == "table":
            if extra:
                body.append(doc.para("", 0, 0, 1, extra))
            body.append(doc.table(block.rows))
    if not body:
        body.append(doc.para("", 0, 0, 1, first_extra))
    section = f'{DECL}<hs:sec {XMLNS}>' + "".join(body) + "</hs:sec>"

    files = {
        "version.xml": (f'{DECL}<hv:HCFVersion xmlns:hv="http://www.hancom.co.kr/hwpml/2011/version" '
                        'tagetApplication="WORDPROCESSOR" major="5" minor="1" micro="0" buildNumber="1" os="1" xmlVersion="1.4" '
                        'application="Hancom Office Hangul" appVersion="12.0.0.0"/>'),
        "META-INF/container.xml": (f'{DECL}<ocf:container xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container" '
                                   f'xmlns:hpf="{NS["hpf"]}"><ocf:rootfiles><ocf:rootfile full-path="Contents/content.hpf" '
                                   'media-type="application/hwpml-package+xml"/></ocf:rootfiles></ocf:container>'),
        "META-INF/manifest.xml": f'{DECL}<odf:manifest xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>',
        "Contents/content.hpf": (f'{DECL}<opf:package {XMLNS} version="" unique-identifier="" id=""><opf:metadata>'
                                 f'<opf:title>{escape(xml_safe(title))}</opf:title><opf:language>ko</opf:language></opf:metadata>'
                                 '<opf:manifest><opf:item id="header" href="Contents/header.xml" media-type="application/xml"/>'
                                 '<opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>'
                                 '<opf:item id="settings" href="settings.xml" media-type="application/xml"/></opf:manifest>'
                                 '<opf:spine><opf:itemref idref="header" linear="yes"/><opf:itemref idref="section0" linear="yes"/>'
                                 '</opf:spine></opf:package>'),
        "Contents/header.xml": _header_xml(),
        "Contents/section0.xml": section,
        "settings.xml": (f'{DECL}<ha:HWPApplicationSetting xmlns:ha="{NS["ha"]}" xmlns:config="{NS["config"]}">'
                         '<ha:CaretPosition listIDRef="0" paraIDRef="0" pos="0"/></ha:HWPApplicationSetting>'),
        "Preview/PrvText.txt": "\n".join(plain(b.text) for b in blocks if b.text)[:1000],
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        # mimetype 은 맨 앞에, 압축하지 않고 (ODF·EPUB 와 같은 규칙)
        z.writestr(zipfile.ZipInfo("mimetype"), "application/hwp+zip", compress_type=zipfile.ZIP_STORED)
        for name, text in files.items():
            z.writestr(name, text.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
    return out.getvalue()
