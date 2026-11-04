"""👔 포트폴리오·자소서 팀 화면 — app.py 가 부르는 이 팀 전용 부분만 모았다 (features: [career_docs]).

  confirm_step  : 맡기기 전 확인 단계 — JD·요약 문서에 그림(png·jpg·jpeg·webp)·스캔 PDF 가 있으면 비전 모델로 글자를 옮기고,
                  사람이 고친 뒤 '이 내용으로 진행'을 눌러야 업무가 시작된다 (확인한 글 = JD·사실의 출처)
                  🎯 JD 추천은 JD 파일마다 따로 확인한다 (파일 하나 = JD 하나)
  web_step      : 🔄 포트폴리오 최신화의 외부 자료 승인 — 검색어 초안(직무·일반 기술 용어만)을 보여 주고, 사람이 고치거나 지운 뒤
                  '승인'을 눌러야 검색·GitHub 글 가져오기를 한다 (기본 꺼짐, core/career_search.py 가 개인 정보가 든 검색어를 막는다)
  preview       : 최종 결과 아래 '📄 문서' 상자 — 문항별 글자 수(코드 계산) · 사실 검사 요약 · 내려받기(.md · .docx · .hwpx)
  adopted_files : 채택 시 results/career/ 에 함께 저장할 파일 (채택한·고친 내용으로 다시 만든다)

문서 만드는 일은 core/career.py(본문·부록) · core/doc_export.py(.docx · .hwpx)가 한다 (LLM 호출 없음, 1초 안).
그림 읽기는 core/vision.py (설계 팀 도면 그림 읽기와 같은 부분) — 이 팀에서 외부 LLM 을 부르는 유일한 맡기기 전 단계.
"""

import hashlib
import re

import streamlit as st

from core import career, career_search, llm_factory, search, usage, vision
from core.config import load_settings
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
FIELD_TITLES = {"jd": "JD (채용 공고)", "jds": "JD (파일마다 하나)", "docs": "요약 문서"}


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
    for fld in ("docs", "jds"):  # 파일마다 따로 확인한 글 → {경로: 글}
        if fld in items:
            confirmed = dict(inputs.get(f"{fld}_confirmed") or {})
            confirmed.update({f.path: str(t) for f, t in zip(items[fld], texts.get(fld) or [])})
            inputs[f"{fld}_confirmed"] = confirmed
    inputs["vision_summary"] = summary
    return {**request, "tool_inputs": inputs}


def confirm_step(request: dict, team_id: str, task_id: str, models: ModelsConfig, chains: list[RankModels]) -> dict:
    """맡기기 전 확인 단계 — ① 그림·스캔 PDF 글자 확인 → ② (포트폴리오 최신화) 외부 검색어·GitHub 주소 승인.

    둘 다 필요 없으면 request 를 그대로 돌려준다. 확인을 기다리는 동안은 빈 dict (업무를 시작하지 않는다).
    """
    return web_step(vision_step(request, team_id, task_id, models, chains), team_id, task_id, models)


def vision_step(request: dict, team_id: str, task_id: str, models: ModelsConfig, chains: list[RankModels]) -> dict:
    """그림·스캔 PDF 확인 단계. 그림·스캔 PDF 가 없으면 request 를 그대로 돌려준다.

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
                        st.session_state[f"{wkey}_text_{fld}_{i}"] = f.text
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
        for fld, intro in (("jds", "**JD (파일마다 하나)** — 확인한 글이 그 JD 가 됩니다 (요구사항·점수는 이 글로 계산)"),
                           ("docs", "**요약 문서** — 확인한 글이 **사실의 출처**가 됩니다 (인용·숫자는 이 글과 대조합니다)")):
            if fld not in reads:
                continue
            st.markdown(intro)
            texts[fld] = []
            for i, f in enumerate(reads[fld]):
                st.caption(f"· {f.name}: {f.how}" + "".join(f" — ⚠️ 흐림: {u}" for r in f.reads for u in r.unclear[:3]))
                texts[fld].append(st.text_area(f"{f.name} (고칠 수 있음)", key=f"{wkey}_text_{fld}_{i}", height=220))
        if st.button("✅ 이 내용으로 진행", type="primary", key=f"{wkey}_go"):
            if "jd" in texts and not str(texts["jd"]).strip() or any(not str(t).strip() for t in texts.get("jds") or []):
                st.warning("JD 글이 비어 있습니다. 내용을 넣거나 취소하세요.")
                return {}
            summary = (f"그림·스캔 PDF 는 비전 모델({', '.join(used)})이 옮기고 사람이 확인·수정한 글입니다 "
                       f"({' · '.join(FIELD_TITLES[f] for f in reads)})")
            st.session_state.pop(key, None)
            return apply_confirmed(pending["request"], items, texts, summary)
    return {}


# ─── 맡기기 전 외부 자료 승인 (🔄 포트폴리오 최신화) ────────────────────────

WEB_NOTICE = ("🔒 이 팀은 평소 웹 검색을 하지 않습니다. 이 단계에서 **승인 버튼을 눌러야만** 아래 검색어를 검색 엔진(DuckDuckGo)으로 "
              "보냅니다. 검색어에는 직무 이름·일반 기술 용어·연도만 넣습니다 — 이름·회사·프로젝트 이름·숫자·문서 문장이 있으면 "
              "그 검색어는 보내지 않습니다. 검색 결과는 용어·트렌드에만 쓰고, 내 경력·숫자는 내 문서에서만 씁니다 (출처는 URL 로 표시).")


def needs_web(inputs: dict) -> bool:
    """포트폴리오 최신화에서 외부 검색을 켰거나 GitHub 주소를 넣었고, 아직 승인 단계를 거치지 않았는가."""
    return (inputs.get("career_kind") == "portfolio_update" and not inputs.get("web_approved")
            and bool(inputs.get("web_on") or str(inputs.get("github_url") or "").strip()))


def private_text(inputs: dict) -> str:
    """검색어 검사에 쓰는 '내 글' — 요약 문서·지금 포트폴리오 원문 + 올린 파일 이름 (이 글의 낱말이 검색어로 나가지 않게)."""
    docs, _ = career_docs.read_docs(inputs)
    old, _ = career_docs.read_portfolio(inputs)
    names = [str(inputs.get(k) or "") for k in ("docs_name", "portfolio_name")]
    return "\n".join([d.text for d in docs] + ([old.text] if old else []) + names)


def approved_request(request: dict, queries: list[str], results: list[dict], github: tuple[str, str] | None,
                     notes: list[str]) -> dict:
    """승인한 검색어·결과·GitHub 글을 업무 입력에 넣는다 → career_docs 툴이 '외부 자료' 절로 보여 준다."""
    inputs = dict(request["tool_inputs"])
    inputs.update(web_approved=True, web_queries=list(queries), web_results=list(results), web_notes=list(notes))
    if github:
        inputs["github_text"], inputs["github_raw"] = github
    return {**request, "tool_inputs": inputs}


def web_step(request: dict, team_id: str, task_id: str, models: ModelsConfig) -> dict:
    """외부 자료 승인 단계 — 검색어 초안을 보여 주고, 사람이 고치거나 지운 뒤 '승인'을 눌러야 검색·GitHub 가져오기를 한다.

    필요 없으면 request 를 그대로 돌려준다. 기다리는 동안은 빈 dict.
    """
    key = f"career_web_{team_id}_{task_id}"
    if request:
        if not needs_web(request.get("tool_inputs") or {}):
            st.session_state.pop(key, None)
            return request
        nonce = (st.session_state.get(key) or {}).get("nonce", 0) + 1
        st.session_state[key] = {"request": request, "notes": None, "nonce": nonce}
    pending = st.session_state.get(key)
    if not pending:
        return {}
    inputs, wkey = pending["request"]["tool_inputs"], f"{key}_{pending['nonce']}"
    role = career_docs.target_role(inputs)
    private = private_text(inputs)
    settings = load_settings()

    with st.container(border=True):
        st.markdown("##### 🌐 외부 자료 확인 — 업무 시작 전 (보내기 전에 검색어를 확인하세요)")
        st.caption(WEB_NOTICE)
        checks: list[tuple[str, list[str]]] = []
        if inputs.get("web_on"):
            if pending["notes"] is None:
                with st.spinner("검색어 초안을 만드는 중… (직무 이름·일반 기술 용어만 씁니다)"):
                    queries, pending["notes"] = career_search.draft_queries(role, private, models, settings)
                st.session_state[f"{wkey}_queries"] = "\n".join(queries)
            for note in pending["notes"]:
                st.caption(f"· {note}")
            text = st.text_area("보낼 검색어 (한 줄에 하나 — 고치거나 지울 수 있습니다. 승인 전에는 아무것도 보내지 않습니다)",
                                key=f"{wkey}_queries", height=110)
            checks = [(q, career_search.query_problems(q, private, role)) for q in (ln.strip() for ln in text.splitlines()) if q]
            if checks:
                st.markdown("\n".join(f"- ✅ `{q}`" if not p else f"- ⛔ `{q}` — {' / '.join(p)} → **보내지 않음**" for q, p in checks))
            else:
                st.caption("검색어가 없습니다 — 검색하지 않습니다.")
        github_url = str(inputs.get("github_url") or "").strip()
        raw = career_search.github_raw_url(github_url) if github_url else None
        if github_url and raw:
            use = "내 문서로 — 사실의 출처" if inputs.get("github_as_source") else "외부 자료 — 용어·프로젝트 이름 참고, 출처 URL 표시"
            st.markdown(f"📥 가져올 GitHub 글: `{raw}` ({use})")
        elif github_url:
            st.error("이 GitHub 주소는 가져올 수 없습니다 — `https://github.com/계정/저장소` 또는 그 안의 .md·.txt 파일 주소만 됩니다. "
                     "이 주소는 빼고 진행합니다.")
        send = [q for q, p in checks if not p]
        label = "🔍 승인 — 검색하고 진행" if send else ("📥 승인 — GitHub 글 가져오고 진행" if raw else "▶ 진행")
        c1, c2, c3 = st.columns(3)
        go = c1.button(label, type="primary", key=f"{wkey}_go")
        skip = c2.button("🚫 외부 자료 없이 진행", key=f"{wkey}_skip")
        if c3.button("✖ 취소 (업무 맡기지 않음)", key=f"{wkey}_cancel"):
            st.session_state.pop(key, None)
            st.rerun()
        if skip:
            st.session_state.pop(key, None)
            return approved_request(pending["request"], [], [], None, ["외부 자료 없이 진행 (사용자 선택)"])
        if go:
            notes, results, github = [], [], None
            if send:
                with st.spinner(f"승인한 검색어 {len(send)}개로 검색하는 중…"):
                    context = search.search_queries(send, settings)
                results, notes = [r.model_dump() for r in context.results], list(context.notes)
            if len(send) < len(checks):
                notes.append(f"개인 정보 검사에 걸린 검색어 {len(checks) - len(send)}개는 보내지 않았습니다.")
            if raw:
                try:
                    github = career_search.fetch_github(github_url)
                except Exception as e:  # noqa: BLE001 — 가져오지 못해도 업무는 내 문서로 진행
                    notes.append(f"GitHub 글을 가져오지 못했습니다: {type(e).__name__}: {str(e)[:150]}")
            st.session_state.pop(key, None)
            return approved_request(pending["request"], send, results, github, notes)
    return {}


# ─── 결과 문서 ────────────────────────────────────────────────────────

TITLES = {"cover_letter": "자기소개서", "portfolio": "포트폴리오", "jd_recommend": "JD 추천", "portfolio_update": "포트폴리오(최신화)"}


def _kind(inputs: dict, text: str) -> str:
    """업무 종류 — 입력값이 없으면(되살린 업무) 답변 모양으로."""
    if inputs.get("career_kind"):
        return str(inputs["career_kind"])
    data = career.answer_json(text) or {}
    if "jd_fit" in data:
        return "jd_recommend"
    if "changes" in data or "no_changes" in data:
        return "portfolio_update"
    return "cover_letter" if re.search(r"^#{2,4}\s*문항\s*\d", text, re.M) else "portfolio"  # 포트폴리오의 '### 1. 프로젝트' 와 구분


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
    formats = [f for f in ("md", "docx", "hwpx") if inputs.get(f"fmt_{f}")] or (["md", "docx"] if inputs else ["md", "docx", "hwpx"])
    return {**opts, "kind": _kind(inputs, text), "formats": formats, "web": list(inputs.get("web_results") or [])}


def build(run: dict, text: str) -> dict:
    """답변 → {markdown, files, rows, facts, ranking, changes}. 같은 글이면 다시 만들지 않는다."""
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cached = run.get("career")
    if cached and cached.get("key") == key:
        return cached
    opts = _options(run, text)
    questions = [career.Question(**q) for q in opts.get("career_questions") or []]
    kind = opts["kind"]
    title = TITLES.get(kind, "포트폴리오")
    source = str(opts["career_source"]) if "career_source" in opts else None
    jds = opts.get("career_jds") or []
    markdown = career.render_document(text, kind, questions, title, opts.get("career_requirement_text"), jds=jds, source=source,
                                      web=opts["web"])
    files: dict[str, bytes] = {}
    for fmt in opts["formats"]:
        if fmt == "md":
            files[f"{title}.md"] = markdown.encode("utf-8")
        elif fmt == "docx":
            files[f"{title}.docx"] = to_docx(markdown, title)
        elif fmt == "hwpx":
            files[f"{title}.hwpx"] = to_hwpx(markdown, title)
    facts = career.check_facts(text, source or "", str(opts.get("career_reference") or "")) if source is not None else None
    out = {"key": key, "markdown": markdown, "files": files, "rows": career.limit_rows(text, questions), "facts": facts,
           "title": title, "kind": kind,
           "ranking": career.ranked(career.score_jds(text, jds, source)) if kind == "jd_recommend" and jds else [],
           "changes": career._items(career.answer_json(text), "changes") if kind == "portfolio_update" else [],
           "change_problems": career.check_changes(text, str(opts["career_old_portfolio"]), source or "",
                                                   list(opts.get("career_web_urls") or []), int(opts.get("career_web_count") or 0),
                                                   str(opts.get("career_reference") or ""))
           if kind == "portfolio_update" and "career_old_portfolio" in opts else []}
    run["career"] = out
    return out


def _ranking_box(out: dict) -> None:
    """🎯 JD 추천 — 코드가 계산한 순위·점수 표."""
    if not out.get("ranking"):
        return
    lines = ["| 순위 | JD | 적합도 | 점수 근거 |", "|---|---|---|---|"]
    lines += [f"| {i} | {s.id}. {career._cell(s.title)} | {f'**{s.score:g}점**' if s.score is not None else '-'} | {career._cell(s.basis)} |"
              for i, s in enumerate(out["ranking"], 1)]
    top = out["ranking"][0]
    st.markdown("**추천 순위 (점수는 코드가 판정으로 계산 — 모델이 정한 숫자가 아님)**\n\n" + "\n".join(lines))
    if top.score is not None:
        st.success(f"🎯 가장 맞는 JD: **{top.id}. {top.title}** — {top.score:g}점")
    st.caption(career.SCORE_RULE)


def _changes_box(out: dict) -> None:
    """🔄 포트폴리오 최신화 — 바뀐 점 요약 · 변경 목록 검사."""
    if out.get("kind") != "portfolio_update":
        return
    changes = out.get("changes") or []
    outside = sum(str(c.get("kind")).strip() == "외부" for c in changes)
    st.markdown(f"**바뀐 점 {len(changes)}건** (내 문서 {len(changes) - outside} · 외부 자료 {outside}) — 전체 표는 문서 부록에 있습니다.")
    for c in changes[:8]:
        mark = "🌐" if str(c.get("kind")).strip() == "외부" else "📄"
        before = career._cell(c.get("before"))[:60] or "(새로 추가)"
        st.markdown(f"- {mark} **{career._cell(c.get('part'))}**: {before} → {career._cell(c.get('after'))[:80]}")
    if out.get("change_problems"):
        st.error("🔎 변경 목록 검사 — 확인하세요\n\n" + "\n".join(f"- {p}" for p in out["change_problems"][:8]))


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
        _ranking_box(out)
        _changes_box(out)
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
