"""👔 포트폴리오·자소서 팀 화면 — app.py 가 부르는 이 팀 전용 부분만 모았다 (features: [career_docs]).

  preview       : 최종 결과 아래 '📄 문서' 상자 — 문항별 글자 수(코드 계산) · 사실 검사 요약 · 내려받기(.md · .docx · .hwpx)
  adopted_files : 채택 시 results/career/ 에 함께 저장할 파일 (채택한·고친 내용으로 다시 만든다)

문서 만드는 일은 core/career.py(본문·부록) · core/doc_export.py(.docx · .hwpx)가 한다 (LLM 호출 없음, 1초 안).
"""

import hashlib
import re

import streamlit as st

from core import career
from core.doc_export import to_docx, to_hwpx
from core.schemas import HumanDecision

FEATURE = "career_docs"
MIME = {
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".hwpx": "application/hwp+zip",
}
NOTICE = ("🔒 올린 문서·JD 는 이 PC 의 `data/uploads/`, 채택한 결과는 `results/career/` 에만 저장되고 GitHub 에 올라가지 않습니다 "
          "(교훈 공유에서도 이 팀은 빠집니다). 다만 내용은 고른 LLM 회사로 전송됩니다.")


def _options(run: dict, text: str) -> dict:
    """툴 결과 + 업무 입력에서 문서 만들 때 쓸 값.

    서버를 다시 켜서 되살린 결정 대기 업무(core/live.py restore_run)는 툴 결과·입력값이 없다 →
    답변 모양으로 업무 종류를, 저장된 질문 글의 '[문항]' 줄로 문항을 다시 읽고, 세 형식을 모두 준다 (사실 검사는 원문이 없어 생략).
    """
    opts: dict = {}
    for tool in run.get("tools") or []:
        opts.update(tool.harness_options)
    inputs = run.get("career_inputs") or {}
    if "career_questions" not in opts and not inputs:
        opts["career_questions"] = [q.__dict__ for q in career.questions_from_request(str(run.get("question") or ""))]
    looks_cover = bool(re.search(r"^#{2,4}\s*문항\s*\d", text, re.M))  # 포트폴리오의 '### 1. 프로젝트' 와 구분
    kind = str(inputs.get("career_kind") or ("cover_letter" if looks_cover else "portfolio"))
    formats = [f for f in ("md", "docx", "hwpx") if inputs.get(f"fmt_{f}")] or (["md", "docx"] if inputs else ["md", "docx", "hwpx"])
    return {**opts, "kind": kind, "formats": formats}


def build(run: dict, text: str) -> dict:
    """답변 → {markdown, files, rows, facts}. 같은 글이면 다시 만들지 않는다."""
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cached = run.get("career")
    if cached and cached.get("key") == key:
        return cached
    opts = _options(run, text)
    questions = [career.Question(**q) for q in opts.get("career_questions") or []]
    kind = opts["kind"]
    title = "자기소개서" if kind == "cover_letter" else "포트폴리오"
    markdown = career.render_document(text, kind, questions, title, opts.get("career_requirement_text"))
    files: dict[str, bytes] = {}
    for fmt in opts["formats"]:
        if fmt == "md":
            files[f"{title}.md"] = markdown.encode("utf-8")
        elif fmt == "docx":
            files[f"{title}.docx"] = to_docx(markdown, title)
        elif fmt == "hwpx":
            files[f"{title}.hwpx"] = to_hwpx(markdown, title)
    facts = career.check_facts(text, str(opts.get("career_source") or ""), str(opts.get("career_reference") or "")) \
        if "career_source" in opts else None
    out = {"key": key, "markdown": markdown, "files": files, "rows": career.limit_rows(text, questions), "facts": facts,
           "title": title}
    run["career"] = out
    return out


def preview(run: dict, final_text: str) -> None:
    """최종 결과 아래 📄 문서 상자."""
    rid = run["run_id"]
    try:
        out = build(run, final_text)
    except Exception as e:  # noqa: BLE001 — 문서 만들기 실패가 결과·결정을 막지 않게
        st.warning(f"문서를 만들지 못했습니다: {type(e).__name__}: {e}")
        return
    with st.container(border=True):
        st.markdown(f"##### 📄 {out['title']} 문서")
        if out["rows"]:
            lines = ["| 문항 | 공백 포함 | 공백 제외 | 제한 | 판정 |", "|---|---|---|---|---|"]
            for r in out["rows"]:
                limit = f"{r.limit:,}자 ({'공백 제외' if r.no_spaces else '공백 포함'})" if r.limit else "-"
                mark = "⚠️ 답변 없음" if not r.with_spaces else ("✅" if r.ok else f"⚠️ {r.counted - r.limit:,}자 초과")
                lines.append(f"| {r.no}. {r.question[:24]} | {r.with_spaces:,} | {r.without_spaces:,} | {limit} | {mark} |")
            st.markdown("**문항별 글자 수 (코드가 셈)**\n\n" + "\n".join(lines))
        facts = out["facts"]
        if facts is not None:
            if facts.ok:
                st.success(f"🔎 사실 검사 통과 — 인용 {facts.quotes_checked}건이 올린 문서에 그대로 있고, 본문 숫자가 모두 문서·JD·문항에 있습니다.")
            else:
                problems = [*(f"문서에 없는 인용: {q}" for q in facts.missing_quotes),
                            *(f"문서에 없는 숫자: {n}" for n in facts.unsourced_numbers)]
                if facts.no_evidence:
                    problems.append("근거 인용(evidence)이 없습니다")
                st.error("🔎 사실 검사 — 제출 전에 확인하세요\n\n" + "\n".join(f"- {p}" for p in problems[:12]))
        with st.expander("문서 미리보기 (내려받을 내용)"):
            st.markdown(out["markdown"])
        cols = st.columns(max(len(out["files"]), 1))
        for col, (name, data) in zip(cols, out["files"].items()):
            suffix = "." + name.rsplit(".", 1)[-1]
            col.download_button(f"⬇ {name}", data=data, file_name=f"career_{rid[:8]}_{name}", mime=MIME.get(suffix),
                                key=f"dlcareer_{rid}_{name}", use_container_width=True)
        st.caption("한글(.hwpx)은 한글 2014 이상에서 열립니다. 바이너리 .hwp 가 필요하면 한글에서 열어 '다른 이름으로 저장 → .hwp' 하세요. "
                   "결과를 직접 고쳐 채택하면 고친 내용으로 다시 만들어 저장합니다.")
        st.caption(NOTICE)


def adopted_files(run: dict, decision: HumanDecision) -> dict[str, bytes]:
    """채택한(사람이 고쳤으면 고친) 내용으로 문서를 다시 만든다 → results/career/<stem>_자기소개서.docx 등."""
    return dict(build(run, decision.final_answer)["files"])
