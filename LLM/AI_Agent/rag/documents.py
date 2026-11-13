"""문서 읽기와 조각 나누기.

조각(chunk) 하나는 dict 다: {"source": 파일 이름, "title": 제목 경로, "order": 그 파일 안 순번, "text": 본문}
"""

import re
from io import BytesIO
from pathlib import Path

from common import BASE_DIR

# 기본 문서 묶음: 이 저장소에 공개된 AI 거버넌스 문서 (읽기만 한다 — 복사하지 않고 경로로 읽는다)
DEFAULT_DOCS_DIR = BASE_DIR.parent / "train_set_making_by_rule(talos, et)" / "docs" / "ai_governance"
DEFAULT_DOCS_GLOB = "[0-9][0-9]_*.md"  # 01_… ~ 07_… (README 등은 뺀다)

# 조각 하나의 글자 수 상한. 값은 rag/eval_retrieval.py 의 측정으로 골랐다 (README '검색 품질').
MAX_CHARS = 1000

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class UnsupportedDocument(ValueError):
    """읽을 수 없는 문서 형식. 메시지에 무엇을 하면 되는지 적는다."""


def read_document(name: str, data: bytes) -> str:
    """파일 이름과 내용(bytes)을 받아 글로 바꾼다. .md·.txt 는 그대로, .pdf 는 pypdf 가 설치돼 있을 때만."""
    suffix = Path(name).suffix.lower()
    if suffix in (".md", ".txt"):
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp949", errors="replace")
    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise UnsupportedDocument(
                f"'{name}': PDF 를 읽는 패키지(pypdf)가 설치돼 있지 않습니다. "
                "`pip install pypdf` 로 설치하거나, 내용을 .md·.txt 파일로 저장해 올려 주세요."
            ) from None
        pages = pypdf.PdfReader(BytesIO(data)).pages
        return "\n\n".join((page.extract_text() or "").strip() for page in pages)
    raise UnsupportedDocument(f"'{name}': 읽을 수 없는 형식입니다. .md · .txt · .pdf 파일을 올려 주세요.")


def load_default_documents() -> list[tuple[str, str]]:
    """기본 문서 묶음을 [(파일 이름, 글)] 로 읽는다. 폴더가 없으면 빈 목록."""
    if not DEFAULT_DOCS_DIR.is_dir():
        return []
    return [(p.name, read_document(p.name, p.read_bytes())) for p in sorted(DEFAULT_DOCS_DIR.glob(DEFAULT_DOCS_GLOB))]


def _cut(body: str, max_chars: int) -> list[str]:
    """글을 줄 단위로 이어 붙이되 max_chars 를 넘지 않게 자른다. 한 줄이 상한보다 길면 그 줄을 잘라 넣는다."""
    pieces, cur = [], ""
    for line in body.split("\n"):
        while len(line) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + 1 + len(line) > max_chars:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    pieces.append(cur)
    return [p for p in pieces if p.strip()]


def split_text(text: str, source: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """글을 제목(# ~ ######) 단위로 나누고, 긴 절은 max_chars 이하로 더 자른다.

    조각의 title 은 '큰 제목 > 작은 제목' 처럼 위 제목까지 이은 경로다. 코드 블록(```) 안의 # 은 제목으로 보지 않는다.
    """
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []  # 지금 위치의 제목 경로 [(단계, 제목)]
    lines: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(lines).strip("\n")
        if body.strip():
            sections.append((" > ".join(title for _, title in path), body))
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, match.group(2)))
        else:
            lines.append(line)
    flush()

    chunks: list[dict] = []
    for title, body in sections:
        for piece in _cut(body, max_chars):
            chunks.append({"source": source, "title": title, "order": len(chunks) + 1, "text": piece})
    return chunks


def split_documents(docs: list[tuple[str, str]], max_chars: int = MAX_CHARS) -> list[dict]:
    """[(파일 이름, 글)] → 모든 파일의 조각을 한 목록으로."""
    return [chunk for name, text in docs for chunk in split_text(text, name, max_chars)]
