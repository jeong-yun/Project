"""문서 → 글자 (외부 호출 없음) — 포트폴리오·자소서 팀이 올린 요약 문서를 읽는다.

  .docx      : python-docx (문단과 표 칸을 문서 순서대로 + 절 제목·표 머리글 + 그림 — 그림 자리에 '[그림 N]', 원본 그림은 images)
  .pdf       : pypdfium2 (쪽마다 글자 층 — 스캔 그림만 있는 PDF 는 글자가 없다)
  .hwpx      : zip 안 Contents/section*.xml 의 <hp:t> (순수 파이썬)
  .hwp       : 한글 5.x 바이너리 — olefile 로 BodyText/Section* 를 열고 zlib 로 풀어 문단 글자(PARA_TEXT)만 꺼낸다
               암호 문서·배포용 문서는 읽을 수 없다 → .hwpx/.docx/.pdf 로 다시 저장해 달라고 안내
  회사 문서보안(DRM)으로 암호화된 파일(예: 머리가 'SCDSA' — 한글·Word 로 저장하면 자동으로 씌워지는 PC 가 있다)은
  내용을 읽을 수 없으므로 DRM 해제 후 올려 달라고 안내한다.
  .md · .txt : 그대로 (utf-8 → cp949 순서로 시도)
  .csv · .json · .yaml · .log · .py · .sql · .tf : 글 그대로 (PLAIN_SUFFIXES — 📎 첨부 파일, core/tools/attachments.py)

큰 파일은 MAX_CHARS 까지만, PDF 는 MAX_PAGES 쪽까지만 읽고 그 사실을 notes 에 남긴다.
"""

import io
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

SUFFIXES = ("docx", "pdf", "hwp", "hwpx", "md", "txt")
PLAIN_SUFFIXES = ("csv", "json", "yaml", "yml", "log", "py", "sql", "tf")  # 글 그대로 읽는 자료·설정·코드 파일
MAX_CHARS = 30_000  # 파일 하나에서 읽는 최대 글자 수
MAX_PAGES = 40  # PDF 최대 쪽 수
HWP_ADVICE = "한글에서 '다른 이름으로 저장'으로 .hwpx(또는 .docx·.pdf)로 저장해 올려 주세요."
DRM_MARKS = (b"SCDSA",)  # 문서보안(DRM) 암호화 파일의 머리
DRM_ADVICE = ("회사 문서보안(DRM)으로 암호화된 파일이라 내용을 읽을 수 없습니다. DRM 을 해제(반출 신청 등)한 파일이나, "
              "보안 프로그램이 없는 PC 에서 저장한 파일을 올려 주세요.")


class DocReadError(ValueError):
    """읽을 수 없는 문서 — 화면에 그대로 보여 줄 안내 문장."""


@dataclass
class DocText:
    name: str
    kind: str  # docx / pdf / hwpx / hwp / md / txt
    text: str
    units: str = ""  # 읽은 범위 (예: '3쪽', '구역 2개', '문단 41개')
    notes: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)  # 절 제목 (docx 제목 스타일 · md '#') — 원본 형식 검사용
    table_heads: list[list[str]] = field(default_factory=list)  # 표마다 첫 줄 칸 글 (docx 표 · md 표 머리글)
    images: list["DocImage"] = field(default_factory=list)  # docx 그림 (글의 [그림 N] 순서 — N 은 이 문서 안 번호)
    path: str = ""  # 읽은 파일 경로 (read_file) — 그림을 같은 폴더에 보관할 때 쓴다
    figure_range: tuple[int, int] = (0, 0)  # 업무 전체 그림 번호 (처음, 끝) — career_docs.number_figures 가 매긴다

    @property
    def chars(self) -> int:
        return len(self.text)


# XML(Word·한글 파일)에 쓸 수 없는 제어 문자 — PDF 글자 층(하이픈 표시 \x02 · 쪽 나눔 \x0c)·옛 문서(\x0b)에 섞여 들어온다.
# 그대로 두면 python-docx 가 'All strings must be XML compatible' 로 멈춰 Word 를 못 만들고 채택 저장까지 실패했다 (2026-11-05).
XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").replace("\x0b", "\n")
    text = XML_BAD.sub("", text)
    text = re.sub(r"[ \t ]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _limit(doc: DocText) -> DocText:
    if doc.chars > MAX_CHARS:
        doc.notes.append(f"{doc.chars:,}자 중 앞 {MAX_CHARS:,}자만 읽었습니다 (나머지는 요약해 다시 올려 주세요).")
        doc.text = doc.text[:MAX_CHARS]
    return doc


# ─── 형식별 ──────────────────────────────────────────────────────────


@dataclass
class DocImage:
    """문서 안 그림 하나 (원본 그대로 — 다시 그리지 않고 결과 Word 에 옮긴다)."""
    data: bytes
    ext: str  # png / jpeg / gif / bmp / emf …


FIGURE_MARK = "[그림 {n}]"
FIGURE_RX = re.compile(r"\[그림\s*(\d{1,3})\]")


@dataclass
class DocxRead:
    text: str
    units: str
    headings: list[str] = field(default_factory=list)
    table_heads: list[list[str]] = field(default_factory=list)
    images: list[DocImage] = field(default_factory=list)


HEADING_STYLES = ("heading", "title", "제목")  # Word 기본 제목 스타일 이름 (영문·한글 Word)
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_V = "{urn:schemas-microsoft-com:vml}"


def _row_cells(row) -> list:
    """표 한 줄의 칸 — 병합 칸은 row.cells 에 같은 칸(_tc)이 되풀이되므로 한 번만."""
    seen, out = set(), []
    for cell in row.cells:
        if id(cell._tc) in seen:
            continue
        seen.add(id(cell._tc))
        out.append(cell)
    return out


def _pictures(element, part) -> list[DocImage]:
    """요소 안의 그림 (DrawingML a:blip · 옛 VML v:imagedata) — 문서 순서대로, 원본 바이트 그대로."""
    out = []
    for node in element.iter(f"{_A}blip", f"{_V}imagedata"):
        rid = node.get(f"{_R}embed") or node.get(f"{_R}id")
        target = part.related_parts.get(rid) if rid else None
        blob = getattr(target, "blob", None)
        if blob:
            ext = str(getattr(target, "partname", "")).rsplit(".", 1)[-1].lower() or "png"
            out.append(DocImage(blob, "jpeg" if ext == "jpg" else ext))
    return out


def read_docx_full(data: bytes) -> DocxRead:
    """Word → 글 (문단과 표를 문서 순서대로) + 절 제목 + 표 머리글 + 그림.

    표는 줄마다 칸을 ' | ' 로 이어 붙인다. 병합 칸은 한 번만 (같은 글·그림이 되풀이되지 않게).
    그림이 있던 자리(문단 끝·표 칸 끝)에 '[그림 N]' (이 문서 안 번호)을 남기고, 그림은 images 에 순서대로 모은다.
    """
    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 — 깨진 파일·다른 형식
        raise DocReadError(f"Word(.docx) 파일을 열지 못했습니다 ({type(e).__name__}). 파일이 손상되지 않았는지 확인하세요.") from e
    out = DocxRead("", "")
    part = document.part

    def with_figures(text: str, element) -> str:
        marks = []
        for image in _pictures(element, part):
            out.images.append(image)
            marks.append(FIGURE_MARK.format(n=len(out.images)))
        return " ".join([text, *marks]).strip() if marks else text

    lines: list[str] = []
    n_para = n_table = 0
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            n_para += 1
            lines.append(with_figures(para.text, child))
            style = (para.style.name if para.style is not None else "") or ""
            if para.text.strip() and style.lower().startswith(HEADING_STYLES):
                out.headings.append(para.text.strip())
        elif tag == "tbl":
            table = Table(child, document)
            n_table += 1
            for r, row in enumerate(table.rows):
                cells = [with_figures(c.text.strip(), c._tc) for c in _row_cells(row)]
                if r == 0 and any(cells):
                    out.table_heads.append([FIGURE_RX.sub("", c).strip() for c in cells if FIGURE_RX.sub("", c).strip()])
                lines.append(" | ".join(c for c in cells if c))
    out.text = "\n".join(lines)
    out.units = f"문단 {n_para}개 · 표 {n_table}개"
    return out


def read_docx(data: bytes) -> tuple[str, str]:
    doc = read_docx_full(data)
    return doc.text, doc.units


MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
MD_TABLE_HEAD = re.compile(r"^\s*\|(.+)\|\s*\n\s*\|?\s*:?-{3,}", re.M)


def md_structure(text: str) -> tuple[list[str], list[list[str]]]:
    """마크다운 → (절 제목, 표 머리글). 코드 블록 안은 보지 않는다."""
    text = re.sub(r"```.*?```", "", text or "", flags=re.S)
    heads = [m.group(1).strip() for m in MD_HEADING.finditer(text)]
    tables = [[c.strip() for c in m.group(1).split("|") if c.strip()] for m in MD_TABLE_HEAD.finditer(text)]
    return heads, [t for t in tables if t]


def pdf_page_texts(data: bytes) -> tuple[list[str], int]:
    """PDF 쪽마다 글자 층 (앞 MAX_PAGES 쪽) + 전체 쪽 수. 글자 층이 없는 쪽(스캔 그림)은 빈 글자."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as e:  # noqa: BLE001
        raise DocReadError(f"PDF 를 열지 못했습니다 ({type(e).__name__}). 암호가 걸려 있으면 풀어서 올려 주세요.") from e
    try:
        total = len(pdf)
        pages = []
        for i in range(min(total, MAX_PAGES)):
            page = pdf[i]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
    finally:
        pdf.close()
    return pages, total


def read_pdf(data: bytes) -> tuple[str, str, list[str]]:
    pages, total = pdf_page_texts(data)
    notes: list[str] = []
    if total > MAX_PAGES:
        notes.append(f"PDF {total}쪽 중 앞 {MAX_PAGES}쪽만 읽었습니다.")
    text = "\n\n".join(pages)
    if total and not text.strip():
        notes.append("PDF 에 글자 층이 없습니다 (스캔한 그림 PDF). 포트폴리오·자소서 팀 양식에 올리면 그림 읽기(확인 단계)로 읽습니다.")
    return text, f"{min(total, MAX_PAGES)}쪽", notes


def read_hwpx(data: bytes) -> tuple[str, str]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise DocReadError("한글(.hwpx) 파일을 열지 못했습니다 (zip 형식이 아님). .hwp 파일이면 확장자를 확인하세요.") from e
    sections = sorted((n for n in archive.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)),
                      key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[-1]).group()))
    if not sections:
        raise DocReadError("한글(.hwpx) 안에 본문(Contents/section*.xml)이 없습니다.")
    lines: list[str] = []
    for name in sections:
        root = ElementTree.fromstring(archive.read(name))
        for para in root.iter():
            if para.tag.rsplit("}", 1)[-1] != "p":
                continue
            # 문단 바로 아래 run 의 글자만 (표 안 문단은 따로 iter 에서 다시 나온다 — 중복 방지)
            parts = []
            for run in para:
                if run.tag.rsplit("}", 1)[-1] != "run":
                    continue
                for t in run:
                    if t.tag.rsplit("}", 1)[-1] == "t":
                        parts.append("".join(t.itertext()))
                    elif t.tag.rsplit("}", 1)[-1] == "tab":
                        parts.append("\t")
            lines.append("".join(parts))
    return "\n".join(lines), f"구역 {len(sections)}개"


# 한글 5.x 바이너리 — 레코드 머리(4바이트): 태그 10비트 · 수준 10비트 · 크기 12비트 (0xFFF 면 다음 4바이트가 크기)
HWPTAG_PARA_TEXT = 0x10 + 51
_CHAR_CONTROLS = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}  # 글자 1개 크기 — 나머지 0~31 은 8글자(16바이트) 크기 컨트롤


def _hwp_para_text(payload: bytes) -> str:
    out, i, n = [], 0, len(payload) // 2
    while i < n:
        code = struct.unpack_from("<H", payload, i * 2)[0]
        if code >= 32:
            out.append(chr(code))
            i += 1
        elif code in _CHAR_CONTROLS:
            if code in (10, 13):
                out.append("\n")
            elif code in (30, 31):
                out.append(" ")
            i += 1
        else:  # 인라인·확장 컨트롤 (탭 9 는 인라인) — 8글자를 건너뛴다
            if code == 9:
                out.append("\t")
            i += 8
    return "".join(out)


def _hwp_records(stream: bytes):
    pos = 0
    while pos + 4 <= len(stream):
        head = struct.unpack_from("<I", stream, pos)[0]
        pos += 4
        tag, size = head & 0x3FF, (head >> 20) & 0xFFF
        if size == 0xFFF:
            if pos + 4 > len(stream):
                return
            size = struct.unpack_from("<I", stream, pos)[0]
            pos += 4
        yield tag, stream[pos:pos + size]
        pos += size


def read_hwp(data: bytes) -> tuple[str, str]:
    import olefile

    if not data.startswith(olefile.MAGIC):
        raise DocReadError("한글(.hwp) 파일이 아니거나 손상됐습니다. " + HWP_ADVICE)
    ole = olefile.OleFileIO(data)
    try:
        if not ole.exists("FileHeader"):
            raise DocReadError("한글 5.x(.hwp) 문서가 아닙니다 (한글 97 이전 문서일 수 있음). " + HWP_ADVICE)
        header = ole.openstream("FileHeader").read()
        if not header.startswith(b"HWP Document File"):
            raise DocReadError("한글 5.x(.hwp) 문서가 아닙니다. " + HWP_ADVICE)
        flags = struct.unpack_from("<I", header, 36)[0]
        if flags & 0x2:
            raise DocReadError("암호가 걸린 한글 문서는 읽을 수 없습니다. 암호를 풀고 " + HWP_ADVICE)
        if flags & 0x4:
            raise DocReadError("배포용(복사 방지) 한글 문서는 읽을 수 없습니다. " + HWP_ADVICE)
        compressed = bool(flags & 0x1)
        sections = sorted((e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")),
                          key=lambda e: int(e[1][7:] or 0))
        if not sections:
            raise DocReadError("한글 문서에 본문(BodyText)이 없습니다. " + HWP_ADVICE)
        paras: list[str] = []
        for entry in sections:
            raw = ole.openstream(entry).read()
            if compressed:
                try:
                    raw = zlib.decompressobj(-15).decompress(raw)
                except zlib.error as e:
                    raise DocReadError("한글 문서 본문을 풀지 못했습니다. " + HWP_ADVICE) from e
            paras += [_hwp_para_text(payload).rstrip("\n") for tag, payload in _hwp_records(raw) if tag == HWPTAG_PARA_TEXT]
    finally:
        ole.close()
    return "\n".join(paras), f"구역 {len(sections)}개 · 문단 {len(paras)}개"


def read_plain(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_bytes(name: str, data: bytes) -> DocText:
    """파일 이름(확장자)과 내용 → DocText. 읽을 수 없으면 DocReadError."""
    kind = Path(name).suffix.lower().lstrip(".")
    notes: list[str] = []
    if data.startswith(DRM_MARKS):
        raise DocReadError(DRM_ADVICE)
    headings: list[str] = []
    table_heads: list[list[str]] = []
    images: list[DocImage] = []
    if kind == "docx":
        full = read_docx_full(data)
        text, units, headings, table_heads, images = full.text, full.units, full.headings, full.table_heads, full.images
    elif kind == "pdf":
        text, units, notes = read_pdf(data)
    elif kind == "hwpx":
        text, units = read_hwpx(data)
    elif kind == "hwp":
        text, units = read_hwp(data)
    elif kind in ("md", "txt", *PLAIN_SUFFIXES):
        text = read_plain(data)
        units = f"{text.count(chr(10)) + 1}줄"
        if kind == "md":
            headings, table_heads = md_structure(text)
    elif kind == "doc":
        raise DocReadError("옛 Word(.doc) 는 읽지 못합니다. Word 에서 .docx 로 저장해 올려 주세요.")
    else:
        raise DocReadError(f"지원하지 않는 형식입니다: .{kind} (지원: {', '.join('.' + s for s in SUFFIXES)})")
    doc = DocText(name=name, kind=kind, text=_clean(text), units=units, notes=notes, headings=headings, table_heads=table_heads,
                  images=images)
    if not doc.text and not notes:
        doc.notes.append("읽은 글자가 없습니다.")
    return _limit(doc)


def read_file(path: str | Path, name: str | None = None) -> DocText:
    path = Path(path)
    doc = read_bytes(name or path.name, path.read_bytes())
    doc.path = str(path)
    return doc
