"""문서 → 글자 (외부 호출 없음) — 포트폴리오·자소서 팀이 올린 요약 문서를 읽는다.

  .docx      : python-docx (문단 + 표 칸)
  .pdf       : pypdfium2 (쪽마다 글자 층 — 스캔 그림만 있는 PDF 는 글자가 없다)
  .hwpx      : zip 안 Contents/section*.xml 의 <hp:t> (순수 파이썬)
  .hwp       : 한글 5.x 바이너리 — olefile 로 BodyText/Section* 를 열고 zlib 로 풀어 문단 글자(PARA_TEXT)만 꺼낸다
               암호 문서·배포용 문서는 읽을 수 없다 → .hwpx/.docx/.pdf 로 다시 저장해 달라고 안내
  회사 문서보안(DRM)으로 암호화된 파일(예: 머리가 'SCDSA' — 한글·Word 로 저장하면 자동으로 씌워지는 PC 가 있다)은
  내용을 읽을 수 없으므로 DRM 해제 후 올려 달라고 안내한다.
  .md · .txt : 그대로 (utf-8 → cp949 순서로 시도)

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

    @property
    def chars(self) -> int:
        return len(self.text)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t ]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _limit(doc: DocText) -> DocText:
    if doc.chars > MAX_CHARS:
        doc.notes.append(f"{doc.chars:,}자 중 앞 {MAX_CHARS:,}자만 읽었습니다 (나머지는 요약해 다시 올려 주세요).")
        doc.text = doc.text[:MAX_CHARS]
    return doc


# ─── 형식별 ──────────────────────────────────────────────────────────


def read_docx(data: bytes) -> tuple[str, str]:
    import docx  # python-docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 — 깨진 파일·다른 형식
        raise DocReadError(f"Word(.docx) 파일을 열지 못했습니다 ({type(e).__name__}). 파일이 손상되지 않았는지 확인하세요.") from e
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:  # 표는 줄마다 칸을 ' | ' 로 이어 붙인다
        for row in table.rows:
            cells = list(dict.fromkeys(c.text.strip() for c in row.cells))  # 병합 칸은 같은 글이 되풀이된다
            lines.append(" | ".join(c for c in cells if c))
    return "\n".join(lines), f"문단 {len(document.paragraphs)}개 · 표 {len(document.tables)}개"


def read_pdf(data: bytes) -> tuple[str, str, list[str]]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as e:  # noqa: BLE001
        raise DocReadError(f"PDF 를 열지 못했습니다 ({type(e).__name__}). 암호가 걸려 있으면 풀어서 올려 주세요.") from e
    notes: list[str] = []
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
    if total > MAX_PAGES:
        notes.append(f"PDF {total}쪽 중 앞 {MAX_PAGES}쪽만 읽었습니다.")
    text = "\n\n".join(pages)
    if total and not text.strip():
        notes.append("PDF 에 글자 층이 없습니다 (스캔한 그림 PDF). 글자가 들어 있는 PDF 나 .docx 로 올려 주세요.")
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
    if kind == "docx":
        text, units = read_docx(data)
    elif kind == "pdf":
        text, units, notes = read_pdf(data)
    elif kind == "hwpx":
        text, units = read_hwpx(data)
    elif kind == "hwp":
        text, units = read_hwp(data)
    elif kind in ("md", "txt"):
        text = read_plain(data)
        units = f"{text.count(chr(10)) + 1}줄"
    elif kind == "doc":
        raise DocReadError("옛 Word(.doc) 는 읽지 못합니다. Word 에서 .docx 로 저장해 올려 주세요.")
    else:
        raise DocReadError(f"지원하지 않는 형식입니다: .{kind} (지원: {', '.join('.' + s for s in SUFFIXES)})")
    doc = DocText(name=name, kind=kind, text=_clean(text), units=units, notes=notes)
    if not doc.text and not notes:
        doc.notes.append("읽은 글자가 없습니다.")
    return _limit(doc)


def read_file(path: str | Path, name: str | None = None) -> DocText:
    path = Path(path)
    return read_bytes(name or path.name, path.read_bytes())
