"""🔄 포트폴리오 최신화 양식 — JD(선택) · 목표 직무(선택) · 지금 포트폴리오 여러 판(Word판·md판).

가짜 문서(가명)·코드로 만든 docx 만. 모델 호출 없음.
"""

import io
from pathlib import Path

import docx

from core import career
from core.config import load_team
from core.tasks import build_request
from core.tools import career_docs
from tests.test_career_format import check, summary

TASK = next(t for t in load_team("career").tasks if t.id == "portfolio_update")
NEW = "최신 요약 (가명)\n탐지 모델 고도화 (2026-06-01 ~ 2026-10-31)\n하루 1,200건을 자동 분류했다.\n"
OLD_MD = "# 경력서 (가명)\n## 회사\n### 탐지 모델 고도화 (26.06 ~ 진행 중)\n- 처리 1,200건\n## 개인\n- 로그 요약 도구\n"
JD = "[가 회사 보안 분석가 (가명)]\n[자격 요건]\n- 로그 분석 자동화 경험\n[우대 사항]\n- 탐지 모델 운영 경험"


class Upload:
    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


def old_docx(count: str = "1,200건") -> bytes:
    d = docx.Document()
    d.add_heading("1. 회사 업무", level=1)
    t = d.add_table(rows=2, cols=3)
    for c, v in enumerate(["업무 명", "기간", "결과"]):
        t.cell(0, c).text = v
    for c, v in enumerate(["탐지 모델 고도화", "2026-06-01 ~ 현재", f"처리 {count}"]):
        t.cell(1, c).text = v
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def request(tmp_path, extra: str = "", **values):
    from core.tasks import store_uploads

    base = {"docs": [Upload("new.md", NEW.encode())], "portfolio": [Upload("기술경력서.docx", old_docx())]}
    return build_request(TASK, store_uploads({**base, **values}, tmp_path), extra)


# ─── A2: 목표 직무는 선택 ──────────────────────────────────────────────────


def test_role_is_optional_and_default_keeps_current_direction(tmp_path):
    role = next(f for f in TASK.fields if f.key == "role")
    assert role.options[0].startswith("지정 안 함") and not role.required
    req = request(tmp_path, role=role.options[0])
    assert career_docs.target_role(req.tool_inputs) == ""
    res = career_docs.run(req.question, req.tool_inputs, None)
    assert "### 🧭 목표 직무: (지정 안 함 — 지금 포트폴리오의 직무 방향을 그대로" in res.content
    assert career_docs.target_role({"role": "기타 (아래에 직접 입력)", "role_custom": "보안 데이터 분석가"}) == "보안 데이터 분석가"
    assert career_docs.target_role({"role": "보안 관리자"}) == "보안 관리자"
    lenses = Path(__file__).resolve().parents[1] / "teams" / "career" / "skills" / "role_lenses.md"
    assert "지정 안 함" in lenses.read_text(encoding="utf-8")  # 부장 직무 검토도 원래 방향으로


# ─── A1: JD 는 선택 — 있을 때만 jd_coverage ─────────────────────────────────


def test_jd_is_optional_and_coverage_required_only_with_jd(tmp_path):
    fields = {f.key: f for f in TASK.fields}
    assert fields["jd"].type == "textarea" and fields["jd_file"].multiple and "png" in fields["jd_file"].options
    no_jd = career_docs.run("q", request(tmp_path).tool_inputs, None)
    assert no_jd.harness_options["career_requirements"] == [] and "JD 없음 — 지금 포트폴리오의 방향·순서를 그대로" in no_jd.content
    sample = career_docs.run("q", {**request(tmp_path).tool_inputs, "use_sample": True}, None)
    assert sample.harness_options["career_requirements"] == []  # 샘플을 켜도 JD 를 몰래 넣지 않는다
    req = request(tmp_path, jd=JD)
    assert "가 회사 보안 분석가" not in req.question and "'📋 JD 원문'" in req.question  # JD 원문은 근거 자료에만 (질문에는 가리키는 말만)
    with_jd = career_docs.run(req.question, req.tool_inputs, None)
    ids = with_jd.harness_options["career_requirements"]
    assert ids == ["R1", "R2"] and "같은 구분 안에서만" in with_jd.content and "가 회사 보안 분석가" in with_jd.harness_options["career_reference"]
    answer = "## 1. 회사 업무\n본문" + summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    assert check("jd_coverage", answer, career_requirements=[]).passed
    assert not check("jd_coverage", answer, career_requirements=ids).passed
    pic = career_docs.vision_items({"career_kind": "portfolio_update", "jd_file": [str(tmp_path / "a.png")]})
    assert list(pic) == ["jd"]  # 그림 JD 는 자소서·포트폴리오와 같은 확인 단계 (여러 장 → 한 JD)


# ─── A3: 지금 포트폴리오 여러 판 ────────────────────────────────────────────


def test_several_portfolio_versions_map_to_outputs_and_differences_are_listed(tmp_path):
    req = request(tmp_path, portfolio=[Upload("기술경력서.docx", old_docx("1,300건")), Upload("경력서.md", OLD_MD.encode())])
    assert req.tool_inputs["portfolio_names"] == ["기술경력서.docx", "경력서.md"]
    res = career_docs.run(req.question, req.tool_inputs, None)
    opts = res.harness_options
    assert "처리 1,300건" in opts["career_source"] and "처리 1,200건" in opts["career_source"]  # 모든 판이 사실의 출처
    fmt = opts["career_format"]  # 판이 둘(docx·md)이면 추가 요청이 없어도 판마다 형식 유지
    assert set(fmt["targets"]) == {"docx", "md"} and fmt["template"].endswith("기술경력서.docx")
    assert fmt["targets"]["docx"]["table_heads"] == [["업무 명", "기간", "결과"]] and "회사" in fmt["targets"]["md"]["headings"]
    assert "Word(.docx)·한글(.hwpx) 출력의 원본" in res.content and "Markdown(.md) 출력의 원본" in res.content
    assert "판끼리 다른 곳" in res.content and "기술경력서.docx 에만" in res.content and "1,300" in res.content

    ev = summary([("자동 분류", "하루 1,200건을 자동 분류했다.")])
    word = "<!-- format:docx -->\n## 1. 회사 업무\n| 업무 명 | 기간 | 결과 |\n|---|---|---|\n| 탐지 모델 고도화 | 26.06 ~ 진행 중 | 처리 1,200건 |\n"
    md = "<!-- format:md -->\n# 경력서 (가명)\n## 회사\n### 탐지 모델 고도화 (26.06 ~ 진행 중)\n## 개인\n- 로그 요약 도구\n"
    assert check("portfolio_format", word + md + ev, career_format=fmt).passed
    swapped = word.replace("format:docx", "format:md") + md.replace("format:md", "format:docx")
    res = check("portfolio_format", swapped + ev, career_format=fmt)
    assert not res.passed and "[docx 구간" in res.log  # Word 구간에 md 형식을 쓰면 Word판 원본과 다르다
    one = career_docs.run("q", request(tmp_path).tool_inputs, None)
    assert "career_format" not in one.harness_options  # 한 판·형식 지정 없음 → 지금처럼 표준 형식 검사


def test_version_differences_ignore_same_date_in_other_notation():
    a = career_docs.DocText("a.docx", "docx", "탐지 모델 (2026-06-01 ~ 현재) 처리 1,200건")
    b = career_docs.DocText("b.md", "md", "탐지 모델 (26.06 ~ 현재) 처리 1,200건")
    assert career_docs.version_differences([a, b]) == []
    c = career_docs.DocText("c.md", "md", "탐지 모델 (26.06 ~ 현재) 처리 1,500건")
    assert career_docs.version_differences([a, c]) == ["a.docx 에만: 탐지 모델 (2026-06-01 ~ 현재) 처리 1,200건",
                                                      "c.md 에만: 탐지 모델 (26.06 ~ 현재) 처리 1,500건"]
    assert career.numbers_in("26.06")  # (날짜 표기 허용과 같은 규칙)
