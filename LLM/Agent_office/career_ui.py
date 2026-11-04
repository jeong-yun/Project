"""👔 포트폴리오·자소서 팀 화면 — app.py 가 부르는 이 팀 전용 부분만 모았다 (features: [career_docs]).

  confirm_step  : 맡기기 전 확인 단계 — JD·요약 문서에 그림(png·jpg·jpeg·webp)·스캔 PDF 가 있으면 비전 모델로 글자를 옮기고,
                  사람이 고친 뒤 '이 내용으로 진행'을 눌러야 업무가 시작된다 (확인한 글 = JD·사실의 출처)
  preview       : 최종 결과 아래 '📄 문서' 상자 — 문항별 글자 수(코드 계산) · 사실 검사 요약 · 내려받기(.md · .docx · .hwpx)
  adopted_files : 채택 시 results/career/ 에 함께 저장할 파일 (채택한·고친 내용으로 다시 만든다)

문서 만드는 일은 core/career.py(본문·부록) · core/doc_export.py(.docx · .hwpx)가 한다 (LLM 호출 없음, 1초 안).
그림 읽기는 core/vision.py (설계 팀 도면 그림 읽기와 같은 부분) — 이 팀에서 외부 LLM 을 부르는 유일한 맡기기 전 단계.
"""

import hashlib
import re

import streamlit as st

from core import career, llm_factory, usage, vision
from core.doc_export import to_docx, to_hwpx
from core.schemas import HumanDecision, ModelsConfig, ModelSpec, RankModels
from core.tools import career_docs

FEATURE = "career_docs"
MIME = {
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".hwpx": "application/hwp+zip",
}
NOTICE = ("🔒 올린 문서·JD 는 이 PC 의 `data/uploads/`, 채택한 결과는 `results/career/` 에만 저장되고 GitHub 에 올라가지 않습니다 "
          "(교훈 공유에서도 이 팀은 빠집니다). 다만 내용은 고른 LLM 회사로 전송됩니다.")


VISION_NOTICE = ("🔒 그림은 고른 모델의 회사(OpenAI · Google · Anthropic)로 전송됩니다 — 이 팀은 비공개 팀이라 결과는 이 PC 에만 "
                 "저장되지만, 그림 읽기는 외부 호출입니다. 보내기 싫으면 '취소'하고 글로 붙여 넣거나 PDF(글자 층)·Word 로 올려 주세요.")
FIELD_TITLES = {"jd": "JD (채용 공고)", "docs": "요약 문서"}


# ─── 맡기기 전 확인 단계 (그림·스캔 PDF → 글자) ───────────────────────────


def _cost_text(cost: float | None) -> str:
    if cost is None:
        return "추정 비용: 가격 미등록 (🏠 가격표에서 넣으면 계산)"
    return f"추정 비용 ${cost:.4f}"


def reader_pool(chains: list[RankModels]) -> list[ModelSpec]:
    """그림 읽기 후보 — 팀원 모델 후보(🎛 팀원 모델 선택 반영)를 아래 직급부터 (가벼운 모델 먼저), 같은 모델은 한 번."""
    seen, pool = set(), []
    for chain in chains:
        for spec in chain.candidates:
            if spec.label not in seen:
                seen.add(spec.label)
                pool.append(spec)
    return pool


def no_model_reason(pool: list[ModelSpec], models: ModelsConfig) -> str:
    """그림을 읽을 모델이 없는 이유 (화면 안내)."""
    seen = [s for s in pool if vision.is_vision(s)]
    if not seen:
        return ("이 팀원들이 쓰는 모델 중 그림을 볼 수 있는 모델(OpenAI · Google · Anthropic)이 없습니다 "
                "(무료 모델·Claude 구독은 그림을 넘기지 않습니다 — '🎛 팀원 모델 선택'에서 무료 모델만 쓰게 골랐을 수 있습니다).")
    reasons = sorted({f"{s.provider}: {llm_factory.unavailable_reason(s.provider, models)}" for s in seen})
    return "그림을 볼 수 있는 모델을 지금 쓸 수 없습니다 — " + " / ".join(reasons)


def read_items(items: dict[str, list], spec: ModelSpec, pool: list[ModelSpec], models: ModelsConfig,
               team_id: str = "") -> dict[str, list[career_docs.FileTranscript]]:
    """확인이 필요한 칸의 파일을 모두 글자로 옮긴다 — 고른 모델이 primary, 안 되면 나머지 그림 모델로 (llm_factory 재시도·대안)."""
    others = [s for s in vision.usable_vision_models(models, pool) if s.label != spec.label]
    rank_models = RankModels(primary=spec, fallback=others[:3])
    out: dict[str, list[career_docs.FileTranscript]] = {}
    with usage.scope(team_id=team_id, purpose="그림 글자 읽기"):
        for fld, files in items.items():
            out[fld] = [career_docs.transcribe_file(f.path, f.name, rank_models, models) for f in files]
    return out


def apply_confirmed(request: dict, items: dict[str, list], texts: dict[str, object], summary: str) -> dict:
    """확인한 글을 업무 입력에 넣는다 → career_docs 툴이 파일 대신 이 글을 JD·사실의 출처로 쓴다."""
    inputs = dict(request["tool_inputs"])
    if "jd" in items:
        inputs["jd_confirmed"] = str(texts.get("jd") or "")
    if "docs" in items:
        confirmed = dict(inputs.get("docs_confirmed") or {})
        confirmed.update({f.path: str(t) for f, t in zip(items["docs"], texts.get("docs") or [])})
        inputs["docs_confirmed"] = confirmed
    inputs["vision_summary"] = summary
    return {**request, "tool_inputs": inputs}


def confirm_step(request: dict, team_id: str, task_id: str, models: ModelsConfig, chains: list[RankModels]) -> dict:
    """맡기기 전 확인 단계. 그림·스캔 PDF 가 없으면 request 를 그대로 돌려준다.

    있으면 request 를 붙잡아 두고(세션) 아래 상자를 그린다 → '이 내용으로 진행'을 누른 실행에서 확인한 글을 넣은 request 를 돌려준다.
    chains: 이번 업무에 참여할 팀원들의 모델 후보 (아래 직급부터) — 여기서 그림을 볼 수 있고 지금 쓸 수 있는 모델만 고른다.
    """
    key = f"career_vision_{team_id}_{task_id}"
    if request:
        items = career_docs.vision_items(request.get("tool_inputs") or {})
        if not items:
            st.session_state.pop(key, None)
            return request
        nonce = (st.session_state.get(key) or {}).get("nonce", 0) + 1  # 다시 맡기면 입력칸을 새로 (지난 글이 남지 않게)
        st.session_state[key] = {"request": request, "items": items, "reads": None, "error": "", "nonce": nonce}
    pending = st.session_state.get(key)
    if not pending:
        return {}
    items, wkey = pending["items"], f"{key}_{pending['nonce']}"

    with st.container(border=True):
        st.markdown("##### 🖼 그림에서 읽은 글 확인 — 업무 시작 전")
        names = " · ".join(f"{FIELD_TITLES[f]}: {', '.join(i.name for i in files)}" for f, files in items.items())
        st.caption(f"그림·스캔 PDF 가 있어 비전 모델로 글자를 옮깁니다 ({names}). 잘못 읽은 글자는 그대로 사실 검사의 근거가 되므로 "
                   "**숫자·회사 이름·자격 요건을 원본과 대조해 고친 뒤** 진행하세요.")
        st.caption(VISION_NOTICE)
        pool = reader_pool(chains)
        usable = vision.usable_vision_models(models, pool)
        if not usable:
            st.error(f"{vision.NO_VISION_MODEL}\n\n{no_model_reason(pool, models)}")
            if st.button("✖ 취소", key=f"{wkey}_cancel0"):
                st.session_state.pop(key, None)
                st.rerun()
            return {}
        labels = [s.label for s in usable]
        picked = st.selectbox("그림 읽을 모델 (팀원 모델 후보 중 그림을 볼 수 있는 것 — 가벼운 모델 먼저)", labels,
                              key=f"{wkey}_model", help="🆓 무료 모델만 사용 · 끈 provider · 🎛 팀원 모델 선택 규칙을 따릅니다.")
        spec = usable[labels.index(picked)]
        c1, c2 = st.columns(2)
        reread = c1.button("🔁 고른 모델로 다시 읽기", key=f"{wkey}_reread")
        if c2.button("✖ 취소 (업무 맡기지 않음)", key=f"{wkey}_cancel"):
            st.session_state.pop(key, None)
            st.rerun()
        if pending["reads"] is None and not pending["error"] or reread:
            with st.spinner(f"{spec.label} 로 그림 글자를 읽는 중…"):
                try:
                    pending["reads"], pending["error"] = read_items(items, spec, pool, models, team_id), ""
                except llm_factory.NoAvailableModelError as e:
                    pending["reads"], pending["error"] = None, f"{vision.NO_VISION_MODEL}\n\n{e}"
                except Exception as e:  # noqa: BLE001 — 그림 깨짐 등
                    pending["reads"], pending["error"] = None, f"그림을 읽지 못했습니다: {type(e).__name__}: {str(e)[:300]}"
            reads = pending["reads"] or {}
            for fld, files in reads.items():  # 새로 읽은 글로 입력칸을 채운다
                if fld == "jd":
                    st.session_state[f"{wkey}_text_jd"] = career_docs.join_transcripts(files)
                else:
                    for i, f in enumerate(files):
                        st.session_state[f"{wkey}_text_docs_{i}"] = f.text
        if pending["error"]:
            st.error(pending["error"])
            return {}

        reads = pending["reads"]
        all_reads = [r for files in reads.values() for f in files for r in f.reads]
        used = sorted({r.model for r in all_reads})
        costs = [f.cost for files in reads.values() for f in files]
        total = None if any(c is None for c in costs) else round(sum(costs), 6)
        tokens_in, tokens_out = sum(r.input_tokens for r in all_reads), sum(r.output_tokens for r in all_reads)
        st.info(f"🤖 읽은 모델: {', '.join(used) or '-'} · 호출 {len(all_reads)}회 · 토큰 입력 {tokens_in:,} / 출력 {tokens_out:,} · "
                f"{_cost_text(total)}")
        texts: dict[str, object] = {}
        if "jd" in reads:
            st.markdown("**JD (채용 공고)** — 이 글이 JD 로 쓰입니다 (JD 의 숫자는 답변에 써도 되는 출처가 됩니다)")
            for f in reads["jd"]:
                st.caption(f"· {f.name}: {f.how}" + "".join(f" — ⚠️ 흐림: {u}" for r in f.reads for u in r.unclear[:3]))
            texts["jd"] = st.text_area("JD 글 (고칠 수 있음)", key=f"{wkey}_text_jd", height=260)
        if "docs" in reads:
            st.markdown("**요약 문서** — 확인한 글이 **사실의 출처**가 됩니다 (인용·숫자는 이 글과 대조합니다)")
            texts["docs"] = []
            for i, f in enumerate(reads["docs"]):
                st.caption(f"· {f.name}: {f.how}" + "".join(f" — ⚠️ 흐림: {u}" for r in f.reads for u in r.unclear[:3]))
                texts["docs"].append(st.text_area(f"{f.name} (고칠 수 있음)", key=f"{wkey}_text_docs_{i}", height=220))
        if st.button("✅ 이 내용으로 진행", type="primary", key=f"{wkey}_go"):
            if "jd" in texts and not str(texts["jd"]).strip():
                st.warning("JD 글이 비어 있습니다. 내용을 넣거나 취소하세요.")
                return {}
            summary = (f"그림·스캔 PDF 는 비전 모델({', '.join(used)})이 옮기고 사람이 확인·수정한 글입니다 "
                       f"({' · '.join(FIELD_TITLES[f] for f in reads)})")
            st.session_state.pop(key, None)
            return apply_confirmed(pending["request"], items, texts, summary)
    return {}


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
