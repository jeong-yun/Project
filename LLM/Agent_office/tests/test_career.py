"""👔 포트폴리오·자소서 팀 — 문서 읽기(docx·pdf·hwpx·hwp·md), 문항·글자 수, 지어낸 사실 검사, JD 대응, 문서 만들기(.docx·.hwpx·.md),
팀 설정, 가짜 모델로 두 업무 끝까지, 공개 교훈 공유(knowledge/)에서 빠지는지, 화면 양식.

모든 문서·인물은 가상이다 (실제 사람·회사 아님). API 키·네트워크·비용 없음 (conftest 가 모델·검색을 막는다).
"""

import io
import json
import struct
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree

import pytest

from core import career, doc_export, feedback_sync
from core.config import list_teams, load_team, validate_config
from core.doc_text import DRM_ADVICE, DocReadError, read_bytes, read_file
from core.harness import REGISTRY
from core.schemas import AgentAnswer, HumanDecision, Lesson, ModelSpec
from core.storage import Storage
from core.tasks import build_request, store_uploads
from core.tools import career_docs

SOURCE = """가상 인물 김샘플 경력 요약
보안 로그 분석 자동화 (2022.04 ~ 2022.12)
일일 분류 시간을 180분에서 45분으로 줄였다.
오탐으로 닫던 이벤트를 하루 1,200건 자동 분류했다.
ISMS-P 인증 심사 증적 준비 지원 — 담당 항목에서 결함 지적 0건."""
JD = """[주요 업무]
- 보안 로그 분석 및 이상 징후 탐지
[자격 요건]
- Python 을 활용한 로그 분석 경험 3년 이상
[우대 사항]
- 침해사고 대응 경험"""


def check(name: str, answer: str, **options):
    return REGISTRY[name].check(answer, options)


def summary(evidence: list[tuple[str, str]], coverage: list[dict] | None = None, needs: list[str] | None = None) -> str:
    data = {"evidence": [{"claim": c, "quote": q} for c, q in evidence], "jd_coverage": coverage or [], "needs_check": needs or []}
    return "\n\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


# ─── 가상 문서 만들기 (테스트 안에서만) ────────────────────────────────────


def make_docx(paragraphs: list[str], table: list[list[str]]) -> bytes:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    t = d.add_table(rows=len(table), cols=len(table[0]))
    for r, row in enumerate(table):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


def make_pdf(lines: list[str]) -> bytes:
    """글자 층이 있는 최소 PDF (Helvetica, 영문) — 외부 라이브러리 없이."""
    content = "BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def make_hwp(paragraphs: list[str], flags: int = 1) -> bytes:
    """한글 5.x(.hwp) 최소 파일 — OLE 복합 문서에 FileHeader 와 BodyText/Section0 (문단 글자 레코드)만.

    스트림을 4096바이트 이상으로 채워 작은 스트림 영역(mini stream) 없이 일반 섹터만 쓴다 (실제 한글 파일은 olefile 이 둘 다 읽는다).
    """
    def record(tag: int, payload: bytes) -> bytes:
        return struct.pack("<I", tag | (len(payload) << 20)) + payload

    body = b""
    for text in paragraphs:
        chars = text.encode("utf-16-le")
        if "\t" in text:  # 탭(인라인 컨트롤, 8글자 크기)과 표(확장 컨트롤 11)가 섞인 문단도 흉내
            chars = text.replace("\t", "").encode("utf-16-le")[:4] + struct.pack("<H", 9) + b"\x00" * 14 + \
                text.replace("\t", "").encode("utf-16-le")[4:]
        body += record(66, b"\x00" * 22) + record(67, chars + struct.pack("<H", 13))
    body += record(67, struct.pack("<H", 11) + b"\x00" * 14 + "표 안 글".encode("utf-16-le") + struct.pack("<H", 13))
    section = zlib.compress(body)[2:-4] if flags & 1 else body  # 한글은 raw deflate
    header = b"HWP Document File".ljust(32, b"\x00") + struct.pack("<II", 0x05000300, flags)
    streams = [header.ljust(4096, b"\x00"), section.ljust(max(4096, len(section)), b"\x00")]

    data, chains, sector = b"", [], 2  # 0 = FAT, 1 = 디렉터리
    for s in streams:
        s = s.ljust(-(-len(s) // 512) * 512, b"\x00")
        n = len(s) // 512
        chains.append((sector, n))
        data += s
        sector += n
    fat = [0xFFFFFFFD, 0xFFFFFFFE]
    for start, n in chains:
        fat += [start + i + 1 for i in range(n - 1)] + [0xFFFFFFFE]
    fat += [0xFFFFFFFF] * (128 - len(fat))

    def entry(name: str, kind: int, left=0xFFFFFFFF, right=0xFFFFFFFF, child=0xFFFFFFFF, start=0xFFFFFFFE, size=0) -> bytes:
        raw = (name + "\x00").encode("utf-16-le")
        return (raw.ljust(64, b"\x00") + struct.pack("<HBB", len(raw), kind, 1) + struct.pack("<III", left, right, child)
                + b"\x00" * 16 + struct.pack("<I", 0) + b"\x00" * 16 + struct.pack("<IQ", start, size))

    sizes = [len(x) for x in streams]
    directory = (entry("Root Entry", 5, child=1) + entry("FileHeader", 2, left=2, start=chains[0][0], size=sizes[0])
                 + entry("BodyText", 1, child=3) + entry("Section0", 2, start=chains[1][0], size=sizes[1]))
    head = (bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 16 + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\x00" * 6
            + struct.pack("<IIIIIIIII", 0, 1, 1, 0, 4096, 0xFFFFFFFE, 0, 0xFFFFFFFE, 0)
            + struct.pack("<I", 0) + struct.pack("<I", 0xFFFFFFFF) * 108)
    return head + struct.pack("<128I", *fat) + directory + data


# ─── 문서 읽기 ────────────────────────────────────────────────────────────


def test_read_docx_with_table():
    doc = read_bytes("요약.docx", make_docx(["보안 로그 분석 자동화", "분류 시간을 180분에서 45분으로"], [["기술", "Python"], ["기간", "2022.04"]]))
    assert doc.kind == "docx" and "180분에서 45분" in doc.text and "기술 | Python" in doc.text
    assert doc.units == "문단 2개 · 표 1개"


def test_read_pdf_text_layer():
    doc = read_bytes("resume.pdf", make_pdf(["Kim Sample (fictional)", "Triage time 180 to 45 minutes"]))
    assert doc.units == "1쪽" and "Triage time 180 to 45 minutes" in doc.text


def test_read_hwpx_made_by_exporter():
    data = doc_export.to_hwpx("# 경력\n\n분류 시간을 **180분**에서 45분으로\n\n| 기술 | 기간 |\n|---|---|\n| Python | 2022.04 |")
    doc = read_bytes("요약.hwpx", data)
    assert "분류 시간을 180분에서 45분으로" in doc.text and "Python" in doc.text and doc.units == "구역 1개"


@pytest.mark.parametrize("flags", [1, 0])  # 압축 / 압축 안 함
def test_read_hwp_binary(flags):
    doc = read_bytes("요약.hwp", make_hwp(["보안 로그 분석 자동화", "하루 1,200건\t자동 분류"], flags))
    assert "보안 로그 분석 자동화" in doc.text and "하루\t 1,200건자동 분류" in doc.text  # 탭 컨트롤(8글자 크기)은 탭 한 글자로
    assert "표 안 글" in doc.text and doc.units.startswith("구역 1개")


def test_hwp_encrypted_distribution_drm_and_unknown_formats_explain_what_to_do():
    with pytest.raises(DocReadError, match="암호"):
        read_bytes("a.hwp", make_hwp(["비밀"], flags=0x3))
    with pytest.raises(DocReadError, match="배포용"):
        read_bytes("a.hwp", make_hwp(["배포"], flags=0x5))
    with pytest.raises(DocReadError, match="hwpx"):
        read_bytes("a.hwp", b"not an ole file at all" * 30)
    with pytest.raises(DocReadError) as e:
        read_bytes("요약.docx", b"SCDSA004" + b"\x00" * 200)  # 회사 문서보안(DRM)으로 암호화된 파일
    assert str(e.value) == DRM_ADVICE
    with pytest.raises(DocReadError, match=".docx 로 저장"):
        read_bytes("old.doc", b"x")
    with pytest.raises(DocReadError, match="지원하지 않는 형식"):
        read_bytes("a.xlsx", b"x")


def test_read_plain_text_cp949_and_long_file_is_cut():
    assert "가상 인물" in read_bytes("a.txt", "가상 인물".encode("cp949")).text
    doc = read_bytes("big.md", ("가" * 40_000).encode("utf-8"))
    assert doc.chars == 30_000 and "앞 30,000자만" in doc.notes[0]


# ─── 문항 · 글자 수 ───────────────────────────────────────────────────────


def test_parse_questions_reads_limits_and_space_rule():
    qs = career.parse_questions("1. 지원 동기 (700자)\n2) 강점과 경험 (1,000자 이내, 공백 제외)\nQ3. 입사 후 포부 [500자]\n\n성장 과정\n"
                                "⑤ 협업 경험(공백 제외 800자)")
    assert [(q.no, q.text, q.limit, q.no_spaces) for q in qs] == [
        (1, "지원 동기", 700, False), (2, "강점과 경험", 1000, True), (3, "입사 후 포부", 500, False),
        (4, "성장 과정", None, False), (5, "협업 경험", 800, True)]
    assert career.count_chars("가 나\n다") == (5, 3)


def cover(answers: dict[int, str], extra: str = "") -> str:
    return "\n\n".join(f"## 문항 {n}. 질문 {n}\n{text}" for n, text in answers.items()) + extra


def test_char_limits_validator():
    qs = [{"no": 1, "text": "지원 동기", "limit": 20, "no_spaces": False}, {"no": 2, "text": "경험", "limit": 10, "no_spaces": True}]
    ok = cover({1: "가" * 15, 2: "나 " * 9})  # 2번: 공백 포함 18자, 공백 제외 9자 → 통과
    res = check("char_limits", ok, career_questions=qs)
    assert res.passed, res.log
    assert "1번 15/20자" in res.log and "2번 9/10자" in res.log
    over = check("char_limits", cover({1: "가" * 25, 2: "나" * 9}), career_questions=qs)
    assert not over.passed and "문항 1 이 공백 포함 25자로 제한 20자를 5자 넘는다" in over.log
    short = check("char_limits", cover({1: "가" * 3, 2: "나" * 9}), career_questions=qs)
    assert not short.passed and "50%에 못 미친다" in short.log
    missing = check("char_limits", cover({1: "가" * 15}), career_questions=qs)
    assert not missing.passed and "문항 2 답변이 없다" in missing.log
    assert check("char_limits", "아무 답", career_questions=[]).passed


def test_split_answers_drops_model_counts_quotes_and_json():
    text = cover({1: "본문 첫 줄\n> 메모\n(글자 수: 999자)\n본문 둘째 줄"}, summary([("a", "b")]))
    assert career.split_answers(text) == {1: "본문 첫 줄\n본문 둘째 줄"}


# ─── 지어낸 사실 검사 ─────────────────────────────────────────────────────


def test_invented_number_fails_sourced_number_passes():
    opts = {"career_source": SOURCE, "career_reference": JD}
    good = cover({1: "로그 분류 시간을 180분에서 45분으로 줄였습니다. 하루 1,200건을 자동 분류했습니다."},
                 summary([("분류 시간 단축", "일일 분류 시간을 180분에서 45분으로 줄였다.")]))
    assert check("career_facts", good, **opts).passed
    invented = good.replace("하루 1,200건을", "하루 3,500건을")
    res = check("career_facts", invented, **opts)
    assert not res.passed and "3,500건" in res.log
    unit = good.replace("하루 1,200건을", "하루 1,200명을")  # 숫자는 같아도 단위가 다르면 다른 사실
    assert not check("career_facts", unit, **opts).passed
    marked = invented.replace("자동 분류했습니다.", "자동 분류했습니다 [확인 필요].")
    assert check("career_facts", marked, **opts).passed  # 확인 필요 표시한 문장은 통과 (사람이 확인)


def test_percentages_computed_from_source_numbers_and_standard_names_pass():
    opts = {"career_source": SOURCE, "career_reference": JD}
    text = cover({1: "분류 시간을 180분에서 45분으로 75% 줄였습니다. ISO 27001 과 Python 3.11 을 공부했고 3가지 원칙을 지킵니다. "
                     "JD 의 3년 이상 요건을 압니다."}, summary([("분류", "180분에서 45분으로")]))
    assert check("career_facts", text, **opts).passed
    assert not check("career_facts", text.replace("75%", "80%"), **opts).passed


def test_quotes_must_be_verbatim_and_evidence_required():
    opts = {"career_source": SOURCE, "career_reference": ""}
    bad = cover({1: "자동화를 했습니다."}, summary([("자동화", "로그 분석을 자동화해 팀 생산성을 높였다")]))
    res = check("career_facts", bad, **opts)
    assert not res.passed and "올린 문서에 없는 인용 1건" in res.log
    gap = summary([("x", "결함 지적 0건")], [{"id": "R3", "status": "공백", "quote": "지어낸 인용이지만 공백이라 보지 않음"}])
    assert check("career_facts", cover({1: "결함 지적이 없었습니다."}, gap), **opts).passed
    none = check("career_facts", cover({1: "자동화를 했습니다."}), **opts)
    assert not none.passed and "evidence" in none.log
    assert check("career_facts", "아무 답", **{}).passed  # 툴 정보가 없으면 검사하지 않는다


# ─── JD 대응 ──────────────────────────────────────────────────────────────


def test_jd_requirements_and_first_pass_coverage():
    reqs = career.extract_requirements("회사 소개 문장입니다.\n" + JD)
    assert [(r.id, r.section) for r in reqs] == [("R1", "업무"), ("R2", "자격"), ("R3", "우대")]
    assert "Python" in reqs[1].keywords and "경험" not in reqs[1].keywords
    rows = career.coverage(reqs, SOURCE)
    assert [c.status for c in rows] == ["부분", "근거 있음", "공백"]  # R2 키워드(Python·로그·분석) 중 둘이 한 줄에
    assert rows[0].snippet.startswith("보안 로그 분석 자동화")
    plain = career.extract_requirements("보안 관제 경험이 있는 분을 찾습니다. AWS 를 다뤄 보신 분이면 좋습니다.")
    assert len(plain) == 2  # 글머리표·제목이 없으면 문장 단위


def test_jd_coverage_validator():
    ids = ["R1", "R2", "R3"]
    full = summary([("a", "b")], [{"id": "R1", "status": "근거 있음", "quote": "보안 로그 분석 자동화"},
                                  {"id": "R2", "status": "부분", "quote": "180분에서 45분"}, {"id": "R3", "status": "공백", "quote": ""}])
    assert check("jd_coverage", full, career_requirements=ids).passed
    res = check("jd_coverage", full.replace('"R3"', '"R9"').replace('"부분"', '"있음"'), career_requirements=ids)
    assert not res.passed and "빠진 요구사항: R3" in res.log and "R2(있음)" in res.log
    unquoted = check("jd_coverage", full.replace('"quote": "180분에서 45분"', '"quote": ""'), career_requirements=ids)
    assert not unquoted.passed and "quote(원문 인용)가 없다: R2" in unquoted.log
    assert check("jd_coverage", "답", career_requirements=[]).passed


# ─── 포트폴리오 형식 ──────────────────────────────────────────────────────

CARD = "- **문제**: 수작업\n- **역할**: 개발\n- **기술**: Python\n- **결과**: 180분 → 45분\n- **배운 점**: 자동화"
PORTFOLIO = ("# 포트폴리오 — 보안 관리자\n## 요약\n가상 인물.\n## 핵심 역량\n| 영역 | 기술·도구 | 근거 |\n|---|---|---|\n| 관제 | Python | 1번 |\n"
             f"## 프로젝트\n### 1. 로그 자동화 (2022.04 ~ 2022.12)\n{CARD}\n### 2. ISMS-P 증적\n{CARD}\n## 직무 강점\n- 증적 정리")


def test_portfolio_format_validator():
    assert check("portfolio_format", PORTFOLIO).passed
    res = check("portfolio_format", PORTFOLIO.replace("- **배운 점**: 자동화", "", 1).replace("## 요약", "## 소개"))
    assert not res.passed and "'## 요약' 절이 없다" in res.log and "빠진 항목: 배운 점" in res.log


# ─── 문서 만들기 ─────────────────────────────────────────────────────────


def test_render_document_adds_code_counts_and_appendix():
    qs = career.parse_questions("지원 동기 (100자)\n경험 (50자, 공백 제외)")
    text = cover({1: "가나다 라마", 2: "바사"}, summary([("자동화", "180분에서 45분으로")],
                                                       [{"id": "R1", "status": "공백", "quote": ""}], ["팀 규모"]))
    md = career.render_document(text, "cover_letter", qs, "자기소개서", {"R1": "침해사고 대응 경험"})
    assert md.startswith("# 자기소개서")
    assert "> 글자 수: 공백 포함 6자 · 공백 제외 5자 · 제한 100자(공백 포함) ✅" in md
    assert "> 글자 수: 공백 포함 2자 · 공백 제외 2자 · 제한 50자(공백 제외) ✅" in md
    assert "| R1 | 침해사고 대응 경험 | 공백 | - |" in md and "- 팀 규모" in md and "| 자동화 | 180분에서 45분으로 |" in md
    assert "```" not in md


def test_docx_export_has_headings_tables_and_no_author():
    import docx

    data = doc_export.to_docx(PORTFOLIO + "\n> 참고 메모", "포트폴리오")
    d = docx.Document(io.BytesIO(data))
    headings = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
    assert headings[:3] == ["포트폴리오 — 보안 관리자", "요약", "핵심 역량"]
    assert d.tables[0].rows[1].cells[1].text == "Python"
    bold = [r.text for p in d.paragraphs for r in p.runs if r.bold]
    assert "문제" in bold and d.core_properties.author == ""
    bullets = [p for p in d.paragraphs if p.style.name == "List Bullet"]
    assert bullets and bullets[0].text == "문제: 수작업"


def test_hwpx_export_is_valid_package():
    data = doc_export.to_hwpx(PORTFOLIO, "포트폴리오")
    z = zipfile.ZipFile(io.BytesIO(data))
    first = z.infolist()[0]
    assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
    assert z.read("mimetype") == b"application/hwp+zip"
    required = {"version.xml", "META-INF/container.xml", "META-INF/manifest.xml", "Contents/content.hpf",
                "Contents/header.xml", "Contents/section0.xml", "settings.xml"}
    assert required <= set(z.namelist())
    for name in required:
        ElementTree.fromstring(z.read(name))  # 모든 XML 이 올바른 형식
    section = z.read("Contents/section0.xml").decode("utf-8")
    assert section.count("<hp:secPr") == 1 and section.count("<hp:tbl ") == 1 and 'rowCnt="2" colCnt="3"' in section
    header = ElementTree.fromstring(z.read("Contents/header.xml"))
    ns = {"hh": doc_export.NS["hh"]}
    n_char = len(header.findall(".//hh:charPr", ns))
    assert n_char == int(header.find(".//hh:charProperties", ns).get("itemCnt"))
    refs = {int(x) for x in __import__("re").findall(r'charPrIDRef="(\d+)"', section)}
    assert max(refs) < n_char  # 본문이 가리키는 글자 모양이 모두 머리(header)에 있다
    text = read_bytes("p.hwpx", data).text
    assert "포트폴리오 — 보안 관리자" in text and "• 문제: 수작업" in text
    assert "<script" not in doc_export.to_hwpx("<script>alert(1)</script> & 기호").decode("latin-1")  # 글자는 XML 이스케이프


# ─── 팀 설정 · 툴 · 업무 양식 ──────────────────────────────────────────────


def test_team_config_is_valid_and_private(ranks, models):
    teams = list_teams()
    validate_config(ranks, models, teams)
    team = load_team("career")
    assert team.private and "web_search" not in team.tools and team.tools == ["career_docs"]
    assert [t.id for t in team.tasks] == ["cover_letter", "portfolio", "jd_recommend", "portfolio_update"]
    assert {"career_facts", "char_limits", "jd_coverage"} <= set(team.harness.validators)
    nodes = {n.id: n for n in team.org.walk()}
    assert [nodes[i].rank for i in ("lead", "gm_role", "am_writer", "staff_facts")] == ["team_lead", "gm", "assistant_manager", "staff"]
    assert "30년 차 인사 팀장" in team.personas["team_lead"]
    gm = team.personas["gm"]
    assert all(word in gm for word in ("정보보호 관리자", "인프라", "데이터 엔지니어링"))
    docs_field = next(f for f in team.tasks[0].fields if f.key == "docs")
    assert docs_field.type == "file" and docs_field.multiple and set(docs_field.options) == {"docx", "pdf", "hwp", "hwpx", "md", "txt", "png", "jpg", "jpeg", "webp"}
    import yaml

    from core.config import BASE_DIR

    depts = yaml.safe_load((BASE_DIR / "config" / "departments.yaml").read_text(encoding="utf-8"))["departments"]
    assert any(d["name"] == "커리어부" and d["teams"] == ["career"] for d in depts)
    for p in (BASE_DIR / "teams" / "career" / "skills").glob("*.md"):
        assert "<thought>" not in p.read_text(encoding="utf-8")  # 모델 거절을 부르는 형식 강제 표현 없음


def test_store_uploads_multiple_files_and_build_request(tmp_path):
    class Up:
        def __init__(self, name, data):
            self.name, self._data = name, data

        def getvalue(self):
            return self._data

    values = store_uploads({"docs": [Up("요약 1.md", "가".encode()), Up("b.txt", b"x")], "questions": "지원 동기 (700자)"}, tmp_path)
    assert len(values["docs"]) == 2 and all(Path(p).exists() for p in values["docs"])
    assert values["docs_name"] == "요약 1.md, b.txt" and values["docs_names"] == ["요약 1.md", "b.txt"]
    task = load_team("career").tasks[0]
    req = build_request(task, {**values, "docs": values["docs"], "fmt_docx": True})
    assert "올린 문서: 요약 1.md, b.txt" in req.question and "[JD 붙여 넣기]" not in req.question  # 빈 JD 줄은 빠진다
    assert req.tool_inputs["career_kind"] == "cover_letter" and req.tool_inputs["docs"] == values["docs"]
    empty = build_request(task, {"docs": [], "questions": "x"})
    assert "docs" not in empty.tool_inputs  # 빈 목록은 비어 있는 값


def test_tool_reads_uploads_jd_and_questions(tmp_path):
    good = tmp_path / "ab12cd34_요약.docx"
    good.write_bytes(make_docx(["보안 로그 분석 자동화", "분류 시간을 180분에서 45분으로 줄였다"], [["기술", "Python"]]))
    drm = tmp_path / "ab12cd34_잠긴.hwp"
    drm.write_bytes(b"SCDSA004" + b"\x00" * 64)
    jd_file = tmp_path / "jd.hwpx"
    jd_file.write_bytes(doc_export.to_hwpx(JD))
    inputs = {"docs": [str(good), str(drm)], "docs_names": ["요약.docx", "잠긴.hwp"], "jd_file": str(jd_file), "jd_file_name": "jd.hwpx",
              "questions": "1. 지원 동기 (700자)\n2. 경험 (1,000자, 공백 제외)", "career_kind": "cover_letter"}
    res = career_docs.run("질문", inputs, None)
    assert "문서 1개" in res.title and "JD 요구사항 3개" in res.title
    assert any("잠긴.hwp" in n and "DRM" in n for n in res.notes)
    assert "| 요약.docx | docx | 문단 2개 · 표 1개 |" in res.content and "<<<문서: 요약.docx" in res.content
    assert "| 2 | 경험 | 1,000자, 공백 제외 |" in res.content
    opts = res.harness_options
    assert "180분에서 45분" in opts["career_source"] and opts["career_requirements"] == ["R1", "R2", "R3"]
    assert opts["career_questions"][1] == {"no": 2, "text": "경험", "limit": 1000, "no_spaces": True}
    assert "3년 이상" in opts["career_reference"]


def test_tool_sample_and_portfolio_role(tmp_path):
    res = career_docs.run("질문", {"use_sample": True, "career_kind": "portfolio", "role": "기타 (아래에 직접 입력)",
                                   "role_custom": "보안 데이터 분석가"}, None)
    assert "김샘플" in res.harness_options["career_source"] and "가상" in res.content
    assert "목표 직무: 보안 데이터 분석가" in res.content and res.harness_options["career_questions"] == []
    none = career_docs.run("질문", {"career_kind": "portfolio"}, None)
    assert any("읽은 요약 문서가 없습니다" in n for n in none.notes)


# ─── 가짜 모델로 두 업무 끝까지 ───────────────────────────────────────────

FACT_SHEET = ("## 사실 카드\n| # | 경험 | 숫자 | 원문 인용 |\n|---|---|---|---|\n| 1 | 로그 자동화 | 180분 → 45분 | 일일 분류 시간을 180분에서 45분으로 줄였다. |"
              + summary([("분류 시간 단축", "일일 분류 시간을 180분에서 45분으로 줄였다.")]))


def run_task(task_id: str, values: dict, reviews: list, ranks, models, use_script):
    from core.config import team_org
    from core.workflow import build_office_graph, stream_office
    from tests.fakes import Script

    team = load_team("career")
    script = use_script(Script(text=FACT_SHEET, ReviewResult=reviews))
    task = next(t for t in team.tasks if t.id == task_id)
    req = build_request(task, values)
    org = team_org(team, ranks)
    graph = build_office_graph(org, ranks, team, models, {}, {"critic": {"enabled": False}})
    events = list(stream_office(graph, org, req.question, "lead", brief=req.brief, auto_brief=False, tool_inputs=req.tool_inputs,
                                extra_validators=req.validators, harness_options=req.harness_options))
    return events, script


def test_fake_model_cover_letter_retries_invented_number(tmp_path, ranks, models, use_script):
    from tests.fakes import make_review

    src = tmp_path / "ab12cd34_요약.md"
    src.write_text(SOURCE, encoding="utf-8")
    coverage = [{"id": "R1", "requirement": "로그 분석", "status": "근거 있음", "quote": "보안 로그 분석 자동화"},
                {"id": "R2", "status": "부분", "quote": "일일 분류 시간을 180분에서 45분으로 줄였다."},
                {"id": "R3", "status": "공백", "quote": ""}]
    body = "보안 로그 분류를 자동화해 일일 분류 시간을 180분에서 45분으로 줄였습니다. " * 3
    good = cover({1: body}, summary([("분류 시간 단축", "일일 분류 시간을 180분에서 45분으로 줄였다.")], coverage))
    bad = good.replace("45분으로 줄였습니다. 보안", "30분으로 줄였습니다. 보안", 1)  # 지어낸 숫자
    reviews = [make_review("revise", 8, bad), make_review("revise", 8, good), make_review("approve", 9), make_review("approve", 9)]
    events, script = run_task("cover_letter", {"docs": [str(src)], "docs_names": ["요약.md"], "jd": JD, "questions": "지원 동기 (200자)",
                                               "fmt_docx": True, "fmt_hwpx": True}, reviews, ranks, models, use_script)
    tools = next(p for k, p in events if k == "tools")
    assert tools[0].tool == "career_docs"
    first = script.prompts("text")[0]
    assert "올린 문서 원문" in first and "[사실 추출" in first and "[자소서 — 쓰는 법]" not in first  # 사원은 맡은 절만
    assert "30분" in next(p for p in script.prompts("ReviewResult") if "자동 검증(하네스)" in p)  # 지어낸 숫자가 피드백으로
    steps = [p for k, p in events if k == "step"]
    facts = next(s for s in steps if s.node_id == "staff_facts")
    assert {h.validator for h in facts.harness} == {"career_facts"} and all(h.passed for h in facts.harness)
    writer = next(s for s in steps if s.node_id == "am_writer")
    assert writer.attempts == 2 and {h.validator for h in writer.harness} == {"career_facts", "char_limits", "jd_coverage"}
    final = next(s for s in steps if s.node_id == "lead")
    assert all(h.passed for h in final.harness) and final.content == good

    import career_ui

    run = {"run_id": "r1", "tools": tools, "career_inputs": {"career_kind": "cover_letter", "fmt_md": "예", "fmt_docx": True,
                                                             "fmt_hwpx": True}}
    out = career_ui.build(run, final.content)
    assert set(out["files"]) == {"자기소개서.md", "자기소개서.docx", "자기소개서.hwpx"} and out["facts"].ok
    assert out["rows"][0].limit == 200 and out["rows"][0].ok
    assert career_ui.build(run, final.content) is out  # 같은 글이면 다시 만들지 않는다
    edited = HumanDecision(action="adopt", adopted_node_id="lead", final_answer=good.replace("180분", "180 분"), edited=True, score=4)
    assert career_ui.adopted_files(run, edited)["자기소개서.md"].decode("utf-8").count("180 분에서 45분으로 줄였습니다") == 3  # 고친 내용으로 다시


def test_restored_run_without_tool_results_still_counts_chars():
    """서버를 다시 켜 되살린 결정 대기 업무 — 툴 결과·입력값이 없어도 질문 글에서 문항을 다시 읽는다."""
    import career_ui

    task = load_team("career").tasks[0]
    req = build_request(task, {"questions": "1. 지원 동기 (50자)\n2. 경험 (30자, 공백 제외)", "tone": "담백하게"})
    assert [(q.limit, q.no_spaces) for q in career.questions_from_request(req.question)] == [(50, False), (30, True)]
    out = career_ui.build({"run_id": "r", "question": req.question}, cover({1: "가" * 40, 2: "나 " * 20}))
    assert [(r.limit, r.counted, r.ok) for r in out["rows"]] == [(50, 40, True), (30, 20, True)]
    assert set(out["files"]) == {"자기소개서.md", "자기소개서.docx", "자기소개서.hwpx"} and out["facts"] is None
    assert career_ui.build({"run_id": "p", "question": "x"}, PORTFOLIO)["title"] == "포트폴리오"


def test_fake_model_portfolio(tmp_path, ranks, models, use_script):
    from tests.fakes import make_review

    gaps = [{"id": f"R{i}", "status": "공백", "quote": ""} for i in range(1, 8)]  # 샘플을 켜면 샘플 JD(요구사항 7개)도 쓴다
    final_text = PORTFOLIO.replace("180분 → 45분", "180분에서 45분으로") + summary([("자동화", "180분에서 45분으로")], gaps)
    reviews = [make_review("revise", 8, final_text), make_review("approve", 9), make_review("approve", 9)]
    events, _ = run_task("portfolio", {"use_sample": True, "role": "보안 관리자", "fmt_docx": True, "fmt_hwpx": True, "fmt_md": True},
                         reviews, ranks, models, use_script)
    steps = [p for k, p in events if k == "step"]
    final = next(s for s in steps if s.node_id == "lead")
    assert {h.validator for h in final.harness} == {"career_facts", "char_limits", "jd_coverage", "portfolio_format"}
    assert all(h.passed for h in final.harness), [h.log for h in final.harness if not h.passed]
    facts = next(s for s in steps if s.node_id == "staff_facts")
    assert "portfolio_format" not in {h.validator for h in facts.harness}  # 사원은 사실 카드만


# ─── 공개 교훈 공유에서 빠진다 · 결과 저장 ──────────────────────────────────


def test_private_team_is_excluded_from_knowledge_sync(tmp_path):
    storage = Storage(tmp_path / "runs.db")
    for rid, team in (("c1", "career"), ("g1", "general")):
        storage.create_run(rid, team, "lead", "가상 인물 김샘플 자소서")
        storage.save_steps(rid, [AgentAnswer(node_id="lead", rank_id="team_lead", kind="integrate", content="답",
                                             model=ModelSpec(provider="openai", model="x"))], "awaiting_human")
        storage.save_decision(rid, HumanDecision(action="adopt", adopted_node_id="lead", final_answer="답", score=4))
    storage.add_lessons("career", "c1", [Lesson(agent_id="lead", kind="improve", lesson="김샘플의 가나다시큐리티 경력을 앞에 둔다")])
    storage.add_lessons("general", "g1", [Lesson(agent_id="lead", kind="praise", lesson="출처를 단다")])
    assert feedback_sync.private_teams() == {"career"}
    folder = tmp_path / "knowledge"
    out = feedback_sync.export(storage, folder)
    everything = "".join(p.read_text(encoding="utf-8") for p in folder.iterdir())
    assert out["lessons"] == 1 and out["decisions"] == 1 and "career" not in everything and "김샘플" not in everything
    assert "출처를 단다" in everything
    # 다른 PC 의 knowledge/ 에 이 팀 교훈이 섞여 있어도 가져오지 않는다
    (folder / "lessons.jsonl").write_text(json.dumps({"team_id": "career", "agent_id": "lead", "kind": "improve", "lesson": "x",
                                                      "active": True, "created_at": "2026-01-01", "run_id": ""}) + "\n", encoding="utf-8")
    other = Storage(tmp_path / "other.db")
    assert feedback_sync.import_lessons(other, folder) == 0


def test_results_and_uploads_are_git_ignored():
    import subprocess

    from core.config import BASE_DIR

    for path in ("results/career/x_자기소개서.docx", "results/career/x.hwpx", "data/uploads/ab_요약.hwp"):
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=BASE_DIR, capture_output=True)
        assert out.returncode == 0, f"{path} 가 git 에 올라갈 수 있습니다"


def test_exporter_lists_docx_and_hwpx(tmp_path):
    from core.exporter import MIME, export_adopted, list_results

    decision = HumanDecision(action="adopt", adopted_node_id="lead", final_answer="답", score=5)
    paths = export_adopted({"question": "자소서", "brief": None}, decision, "career", "팀", {"lead": "팀장"}, tmp_path,
                           extra_files={"자기소개서.docx": b"d", "자기소개서.hwpx": b"h"})
    assert {p.suffix for p in list_results(tmp_path, "career")} == {".md", ".docx", ".hwpx"} and len(paths) == 3
    assert MIME[".hwpx"] == "application/hwp+zip"


def test_sample_file_from_tool_is_fictional():
    text = read_file(career_docs.SAMPLES_DIR / "sample_profile.md").text
    assert "가상 인물" in text and "실제 사람·회사가 아닙니다" in text


# ─── 화면 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch, set_keys):
    from streamlit.testing.v1 import AppTest

    set_keys()
    monkeypatch.setenv("AGENT_OFFICE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_OFFICE_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.query_params["team"] = "career"
    at.query_params["view"] = "team"
    return at


def test_career_team_page_renders_both_forms(app):
    at = app.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="dept_task_career").value == "head"  # 부서 화면 기본 = 👔 부서장에게 맡기기
    at = at.selectbox(key="dept_task_career").set_value("career:cover_letter").run()  # 부서 업무 목록에서 직접 고르기
    picker = at.selectbox(key="dept_task_career")
    assert [o.split(" › ")[1] for o in picker.options[1:3]] == ["✍️ 자소서 작성", "📁 포트폴리오 작성"]
    assert len(at.get("file_uploader")) == 2  # 요약 문서(여러 개) + JD 파일
    assert any("LLM 회사로 전송" in c.value for c in at.caption)
    assert next(c for c in at.checkbox if c.label == "Word(.docx) 로도 받기").value is True
    at = at.button[[b.label for b in at.button].index("맡기기")].click().run()
    assert any("필수 입력이 비어 있습니다: 자소서 문항" in w.value for w in at.warning)

    at = at.selectbox(key="dept_task_career").set_value("career:portfolio").run()
    assert not at.exception, at.exception
    assert at.selectbox(key="tf_career_portfolio_role").options[0] == "보안 관리자"
    assert {c.label for c in at.checkbox} >= {"Word(.docx)", "한글(.hwpx)", "Markdown(.md)"}
