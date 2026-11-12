"""👔 포트폴리오·자소서 팀 화면 — app.py 가 부르는 이 팀 전용 부분만 모았다 (features: [career_docs]).

  confirm_step  : 맡기기 전 확인 단계 — JD·요약 문서에 그림(png·jpg·jpeg·webp)·스캔 PDF 가 있으면 비전 모델로 글자를 옮기고,
                  사람이 고친 뒤 '이 내용으로 진행'을 눌러야 업무가 시작된다 (확인한 글 = JD·사실의 출처)
                  🎯 JD 추천은 JD 파일마다 따로 확인한다 (파일 하나 = JD 하나)
  web_step      : 외부 자료 승인 (네 양식 + 💬 자유 질문) — '읽기'와 '보내기'를 나눠 보여 준다.
                  읽기 = 사람이 넣은 GitHub·웹 주소의 글 (주소만 나간다 — 가져올 주소 목록을 확인하고 승인),
                  보내기 = 검색어 (기본 꺼짐 — 초안을 고치거나 지운 뒤 승인, core/career_search.py 가 개인 정보가 든 검색어를 막는다).
                  승인 전에는 아무것도 나가지 않는다. 가져오지 못한 주소는 왜 안 됐고 무엇을 하면 되는지 알린 뒤 진행·취소를 고르게 한다
  followup_step : 💬 수정 요청·추가 질문 글에 적은 주소 — 같은 확인 상자를 거쳐 가져오고, 그 글을 그 업무의 근거 자료에 더한다
                  (수정본을 쓰는 팀장 · 부장 검토 · 사실 검사 · 문서 미리보기가 모두 같은 근거를 본다)
  preview       : 최종 결과 아래 '📄 문서' 상자 — 문항별 글자 수(코드 계산) · 사실 검사 요약 · 내려받기(.md · .docx · .hwpx)
  adopted_files : 채택 시 results/career/ 에 함께 저장할 파일 (채택한·고친 내용으로 다시 만든다)

문서 만드는 일은 core/career.py(본문·부록) · core/doc_export.py(.docx · .hwpx)가 한다 (LLM 호출 없음, 1초 안).
그림 읽기는 core/vision.py (설계 팀 도면 그림 읽기와 같은 부분) — 이 팀에서 외부 LLM 을 부르는 유일한 맡기기 전 단계.
"""

import hashlib
import re
from urllib.parse import unquote

import streamlit as st

from core import career, career_search, llm_factory, search, usage, vision
from core.config import load_settings
from core.doc_export import to_docx, to_hwpx
from core.schemas import HumanDecision, ModelsConfig, ModelSpec, RankModels, ToolResult
from core.tools import career_docs

FEATURE = "career_docs"
MIME = {
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".hwpx": "application/hwp+zip",
}
NOTICE = ("🔒 올린 문서·JD 는 이 PC 의 `data/uploads/`, 채택한 결과는 `results/career/` 에만 저장되고 GitHub 에 올라가지 않습니다 "
          "(교훈 공유에서도 이 팀은 빠집니다). 다만 내용은 고른 AI 모델 회사로 전송됩니다.")


VISION_NOTICE = ("🔒 그림은 고른 모델의 회사(OpenAI · Google · Anthropic)로 전송됩니다 — 이 팀은 비공개 팀이라 결과는 이 PC 에만 "
                 "저장되지만, 그림 읽기는 외부 호출입니다. 보내기 싫으면 '취소'하고 글로 붙여 넣거나 PDF(글자 층)·Word 로 올려 주세요.")
FIELD_TITLES = {"jd": "JD (채용 공고)", "jds": "JD (파일마다 하나)", "docs": "요약 문서", "attachments": "첨부 파일"}


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
               team_id: str = "", cache: dict | None = None, progress=None) -> dict[str, list[career_docs.FileTranscript]]:
    """확인이 필요한 칸의 파일을 모두 글자로 옮긴다 — 고른 모델이 primary, 안 되면 나머지 그림 모델로 (llm_factory 재시도·대안).

    cache: 같은 그림(내용 해시)·같은 모델은 다시 읽지 않는다 (화면이 다시 그려져 읽기가 끊겨도 읽은 그림은 남는다).
    progress(i, n, name): 파일마다 '그림 i/n 읽는 중' 표시 (여러 장을 차례로 읽는 것이 반복처럼 보이지 않게).
    """
    others = [s for s in vision.usable_vision_models(models, pool) if s.label != spec.label]
    rank_models = RankModels(primary=spec, fallback=others[:3])
    out: dict[str, list[career_docs.FileTranscript]] = {}
    total = sum(len(files) for files in items.values())
    done = 0
    with usage.scope(team_id=team_id, purpose="그림 글자 읽기"):
        for fld, files in items.items():
            out[fld] = []
            for f in files:
                done += 1
                if progress:
                    progress(done, total, f.name)
                out[fld].append(career_docs.transcribe_file(f.path, f.name, rank_models, models, cache))
    return out


CACHE_KEY = "career_vision_cache"  # 세션 안 그림 읽기 결과 — (그림 해시|모델) → 읽은 결과


def apply_confirmed(request: dict, items: dict[str, list], texts: dict[str, object], summary: str,
                    groups: dict[str, int] | None = None) -> dict:
    """확인한 글을 업무 입력에 넣는다 → career_docs 툴이 파일 대신 이 글을 JD·사실의 출처로 쓴다.

    groups = {파일 경로: JD 묶음 번호} (JD 추천) — 같은 번호의 그림은 이어 붙여 JD 하나 (career_docs.read_jds).
    """
    inputs = dict(request["tool_inputs"])
    if "jd" in items:
        inputs["jd_confirmed"] = str(texts.get("jd") or "")
    for fld in ("docs", "jds", "attachments"):  # 파일마다 따로 확인한 글 → {경로: 글}
        if fld in items:
            confirmed = dict(inputs.get(f"{fld}_confirmed") or {})
            confirmed.update({f.path: str(t) for f, t in zip(items[fld], texts.get(fld) or [])})
            inputs[f"{fld}_confirmed"] = confirmed
    if groups:
        inputs["jds_group"] = {**(inputs.get("jds_group") or {}), **{path: f"G{g}" for path, g in groups.items()}}
    inputs["vision_summary"] = summary
    return {**request, "tool_inputs": inputs}


def confirm_step(request: dict, team_id: str, task_id: str, models: ModelsConfig, chains: list[RankModels]) -> dict:
    """맡기기 전 확인 단계 — ① 그림·스캔 PDF 글자 확인 → ② 외부 자료(검색어 · GitHub·웹 주소) 승인.

    둘 다 필요 없으면 request 를 그대로 돌려준다. 확인을 기다리는 동안은 빈 dict (업무를 시작하지 않는다).
    """
    return web_step(vision_step(request, team_id, task_id, models, chains), team_id, task_id, models)


def vision_step(request: dict, team_id: str, task_id: str, models: ModelsConfig, chains: list[RankModels],
                find=career_docs.vision_items, notice: str = VISION_NOTICE) -> dict:
    """그림·스캔 PDF 확인 단계. 그림·스캔 PDF 가 없으면 request 를 그대로 돌려준다.

    find / notice: 확인할 파일을 찾는 함수와 전송 안내 — 📎 자유 질문 첨부(attach_ui.vision_step)가 자기 것을 넘겨 같은 상자를 쓴다.

    있으면 request 를 붙잡아 두고(세션) 아래 상자를 그린다 → '이 내용으로 진행'을 누른 실행에서 확인한 글을 넣은 request 를 돌려준다.
    chains: 이번 업무에 참여할 팀원들의 모델 후보 (아래 직급부터) — 여기서 그림을 볼 수 있고 지금 쓸 수 있는 모델만 고른다.
    """
    key = f"career_vision_{team_id}_{task_id}"
    if request:
        items = find(request.get("tool_inputs") or {})
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
        what = "숫자·이름·날짜를" if "attachments" in items else "숫자·회사 이름·자격 요건을"
        st.caption(f"그림·스캔 PDF 가 있어 비전 모델로 글자를 옮깁니다 ({names}). 잘못 읽은 글자는 그대로 사실 검사의 근거가 되므로 "
                   f"**{what} 원본과 대조해 고친 뒤** 진행하세요.")
        st.caption(notice)
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
            cache = st.session_state.setdefault(CACHE_KEY, {})
            if reread:  # '다시 읽기'는 고른 모델로 새로 — 이 파일들의 지난 결과를 쓰지 않는다
                cache = {}
            bar = st.progress(0.0, text=f"{spec.label} 로 그림 글자를 읽는 중…")

            def progress(i: int, n: int, name: str) -> None:
                bar.progress((i - 1) / n, text=f"🖼 그림 {i}/{n} 읽는 중 — {name} ({spec.label}) · 읽은 그림은 다시 읽지 않습니다")

            with st.spinner(f"{spec.label} 로 그림 글자를 읽는 중…"):
                try:
                    pending["reads"], pending["error"] = read_items(items, spec, pool, models, team_id, cache, progress), ""
                    if reread:
                        st.session_state[CACHE_KEY].update(cache)
                except llm_factory.NoAvailableModelError as e:
                    pending["reads"], pending["error"] = None, f"{vision.NO_VISION_MODEL}\n\n{e}"
                except Exception as e:  # noqa: BLE001 — 그림 깨짐 등
                    pending["reads"], pending["error"] = None, f"그림을 읽지 못했습니다: {type(e).__name__}: {str(e)[:300]}"
            bar.empty()
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
        groups: dict[str, int] = {}
        if "jd" in reads:
            st.markdown("**JD (채용 공고)** — 이 글이 JD 로 쓰입니다 (JD 의 숫자는 답변에 써도 되는 출처가 됩니다)")
            for f in reads["jd"]:
                st.caption(f"· {f.name}: {f.how}" + "".join(f" — ⚠️ 흐림: {u}" for r in f.reads for u in r.unclear[:3]))
            texts["jd"] = st.text_area("JD 글 (고칠 수 있음)", key=f"{wkey}_text_jd", height=260)
        for fld, intro in (("jds", "**JD (파일마다 하나)** — 확인한 글이 그 JD 가 됩니다 (요구사항·점수는 이 글로 계산)"),
                           ("docs", "**요약 문서** — 확인한 글이 **사실의 출처**가 됩니다 (인용·숫자는 이 글과 대조합니다)"),
                           ("attachments", "**첨부 파일** — 확인한 글이 팀의 **근거 자료**로 들어갑니다")):
            if fld not in reads:
                continue
            st.markdown(intro)
            if fld == "jds":
                st.caption("📎 긴 JD 가 캡처 여러 장이면 같은 'JD 묶음' 번호를 고르세요 (올린 순서대로 이어 붙여 JD 하나). "
                           "캡처 한 장에 JD 가 여러 개면 글 안에서 JD 사이에 '---' 만 있는 줄을 넣어 나누세요.")
            texts[fld] = []
            n_files = len(reads[fld])
            for i, f in enumerate(reads[fld]):
                st.caption(f"· {f.name}: {f.how}" + "".join(f" — ⚠️ 흐림: {u}" for r in f.reads for u in r.unclear[:3]))
                if fld == "jds" and n_files > 1:
                    groups[items[fld][i].path] = st.selectbox(f"{f.name} — JD 묶음", list(range(1, n_files + 1)), index=i,
                                                              key=f"{wkey}_group_{i}", format_func=lambda g: f"JD 묶음 {g}")
                texts[fld].append(st.text_area(f"{f.name} (고칠 수 있음)", key=f"{wkey}_text_{fld}_{i}", height=220))
            if fld == "jds":  # 지금 묶음·'---' 나누기로 만들어질 최종 JD 목록 (붙여 넣은 JD 가 있으면 그것이 먼저)
                draft = apply_confirmed(pending["request"], items, texts, "", groups)["tool_inputs"]
                st.markdown("**최종 JD 목록 (이대로 진행하면 J 번호가 이렇게 매겨집니다)**\n\n" + career_docs.jd_list_preview(draft))
        if st.button("✅ 이 내용으로 진행", type="primary", key=f"{wkey}_go"):
            if "jd" in texts and not str(texts["jd"]).strip() or any(not str(t).strip() for t in texts.get("jds") or []):
                st.warning("JD 글이 비어 있습니다. 내용을 넣거나 취소하세요.")
                return {}
            summary = (f"그림·스캔 PDF 는 비전 모델({', '.join(used)})이 옮기고 사람이 확인·수정한 글입니다 "
                       f"({' · '.join(FIELD_TITLES[f] for f in reads)})")
            st.session_state.pop(key, None)
            return apply_confirmed(pending["request"], items, texts, summary, groups)
    return {}


# ─── 맡기기 전 외부 자료 승인 — 읽기(주소의 글) · 보내기(검색어) ─────────────

READ_NOTICE = ("📥 **읽기** — 주소의 글을 가져올 때는 **그 주소만 나갑니다** — 내 문서 내용은 나가지 않습니다. 그 주소 하나만 읽고 "
               "링크를 따라가지 않습니다. 가져온 글은 용어·프로젝트 이름을 참고하는 외부 자료이고(출처 URL 표시) 내 경력·숫자의 근거가 "
               "아닙니다 — '내 저장소'로 표시한 GitHub 글만 사실의 출처가 됩니다. 승인 전에는 아무것도 가져오지 않습니다.")
SEND_NOTICE = ("🔍 **보내기** — 검색어는 검색 엔진(DuckDuckGo)으로 나갑니다 — 아래에서 확인하고 승인하세요. 검색어에는 직무 이름·"
               "일반 기술 용어·연도만 넣습니다 — 이름·회사·프로젝트 이름·숫자·문서 문장이 있으면 그 검색어는 보내지 않습니다. "
               "검색 결과는 용어·트렌드에만 쓰고, 내 경력·숫자는 내 문서에서만 씁니다 (출처는 URL 로 표시).")
URL_LIMITS = {"ref": career_search.MAX_URLS, "jd": career_docs.MAX_JDS}
USE_LABEL = {"ref": "참고 주소", "jd": "JD 주소"}


def wanted_urls(inputs: dict, question: str = "") -> list[tuple[str, str]]:
    """가져오려는 주소 [(주소, 쓰임)] — 'jd' = JD 칸에 줄째로 넣은 주소, 'ref' = 참고 주소 칸 (💬 자유 질문은 질문 글 속의 주소)."""
    refs = career_docs.ref_urls(inputs)
    if not inputs.get("career_kind"):  # 자유 질문에는 주소 칸이 없다 — 질문에 적은 주소를 가져올지 묻는다
        refs += [u for u in career_search.find_urls(question) if u not in refs]
    jds = career_docs.jd_urls(inputs)
    return [(u, "jd") for u in jds] + [(u, "ref") for u in refs if u not in jds]


def needs_web(inputs: dict, question: str = "") -> bool:
    """외부 검색을 켰거나 가져올 주소가 있고, 아직 승인 단계를 거치지 않았는가 (네 양식 + 💬 자유 질문)."""
    return not inputs.get("web_approved") and bool(inputs.get("web_on") or wanted_urls(inputs, question))


def private_text(inputs: dict) -> str:
    """검색어·주소 검사에 쓰는 '내 글' — 요약 문서·지금 포트폴리오 원문 + 올린 파일 이름 (이 글의 낱말이 밖으로 나가지 않게)."""
    docs, _ = career_docs.read_docs(inputs)
    olds, _ = career_docs.read_portfolios(inputs)
    refs, _ = career_docs.read_format_refs(inputs)
    names = [str(inputs.get(k) or "") for k in ("docs_name", "portfolio_name", "format_refs_name")]
    return "\n".join([d.text for d in docs] + [d.text for d in olds] + [d.text for d in refs] + names)


def url_plan(inputs: dict, question: str, private: str, role: str) -> list[dict]:
    """가져올 주소마다 {entered, use, target, problem} — 네트워크 없이 주소 모양만 본다 (승인 전 화면).

    target 이 None 이면 받지 않는 주소이고 problem 에 '이유 — 다음에 할 일'. 참고 주소는 5개, JD 주소는 8개까지.
    """
    plan, seen, counts = [], set(), {"ref": 0, "jd": 0}
    for url, use in wanted_urls(inputs, question):
        item = {"entered": url, "use": use, "target": None, "problem": ""}
        try:
            item["target"] = career_search.target(url, private, role)
        except career_search.FetchProblem as e:
            item["problem"] = str(e)
        if item["target"] and item["target"].fetch in seen:
            continue  # 같은 글을 가리키는 주소는 한 번만
        if item["target"] and counts[use] >= URL_LIMITS[use]:
            item["target"], item["problem"] = None, f"{USE_LABEL[use]}는 {URL_LIMITS[use]}개까지 읽습니다 — 꼭 필요한 주소만 남겨 주세요."
        if item["target"]:
            seen.add(item["target"].fetch)
            counts[use] += 1
        plan.append(item)
    return plan


def fetch_urls(plan: list[dict]) -> tuple[list[dict], list[str]]:
    """승인한 주소를 차례로 가져온다 → (가져온 글 [{url, entered, raw, kind, use, text}], 가져오지 못한 주소와 이유).

    하나가 실패해도 나머지는 가져온다.
    """
    docs, failed = [], []
    for item in plan:
        target = item["target"]
        if target is None:
            continue
        try:
            docs.append({"url": target.url, "entered": item["entered"], "raw": target.fetch, "kind": target.kind, "use": item["use"],
                         "folder": target.folder, "text": career_search.fetch(target)})
        except career_search.FetchProblem as e:
            failed.append(f"{career_docs.FAILED_URL} {item['entered']}: {e}")
    return docs, failed


def read_line(doc: dict) -> str:
    """가져온 글 한 줄 안내 — 무엇을 읽었는지 (파일 이름 · 글자 수 · 폴더 주소면 README.md 만 읽었다는 것)."""
    name = unquote(str(doc["raw"]).rsplit("/", 1)[-1]) if doc.get("kind") == "github" else "웹 페이지"
    only = " — 폴더 전체가 아니라 그 폴더의 README.md 만 읽었습니다 (다른 파일은 파일 주소로 넣어 주세요)" if doc.get("folder") else ""
    return f"읽은 글 {doc['url']}: {name} {len(str(doc['text'])):,}자{only}"


def approved_request(request: dict, queries: list[str], results: list[dict], docs: list[dict], notes: list[str],
                     as_source: bool | None = None) -> dict:
    """승인한 검색어·결과·가져온 글을 업무 입력에 넣는다 → career_docs 툴이 '🌐' 절(외부 자료) · JD · 내 문서로 보여 준다."""
    inputs = dict(request["tool_inputs"])
    inputs.update(web_approved=True, web_queries=list(queries), web_results=list(results), web_notes=list(notes), fetched=list(docs))
    if as_source is not None:  # 💬 자유 질문 — 승인 상자에서 고른 '내 저장소'
        inputs["github_as_source"] = bool(as_source)
    return {**request, "tool_inputs": inputs}


def _plan_line(item: dict, as_source: bool) -> str:
    target = item["target"]
    if target is None:
        return f"- ⛔ `{item['entered']}` — {item['problem']} → **가져오지 않음**"
    what = ("GitHub" + (" 폴더·저장소 — 폴더 전체가 아니라 README.md 만 읽습니다" if target.folder else " 글")
            if target.kind == "github" else "웹 페이지")
    if item["use"] == "jd":
        use = "JD 로 씁니다"
    elif target.kind == "github" and as_source:
        use = "내 문서로 — 사실의 출처"
    else:
        use = "외부 자료 — 용어·프로젝트 이름 참고, 출처 URL 표시"
    return f"- ✅ `{target.fetch}` — {what} · {use}"


def _failed_box(pending: dict, key: str, wkey: str, cancel: str = "✖ 취소 (업무 맡기지 않음)") -> bool:
    """승인 뒤 — 가져오지 못한 주소가 있다. 어느 주소가 왜 안 됐고 무엇을 하면 되는지 보여 주고, 가져온 글로 진행할지 고르게 한다.

    '▶ 가져온 글로 진행'을 누른 실행에서 True (pending['result'] 를 쓰면 된다). 기다리는 동안은 False.
    """
    st.warning("가져오지 못한 주소가 있습니다 — 아래 안내대로 고쳐 다시 보내거나(취소), 가져온 글만으로 진행하세요.\n\n"
               + "\n".join(f"- ⛔ {line.removeprefix(career_docs.FAILED_URL).strip()}" for line in pending["failed"]))
    st.caption(pending["summary"])
    c1, c2 = st.columns(2)
    go = c1.button("▶ 가져온 글로 진행", type="primary", key=f"{wkey}_use")
    if c2.button(cancel, key=f"{wkey}_drop"):
        st.session_state.pop(key, None)
        st.rerun()
    if go:
        st.session_state.pop(key, None)
    return go


def _read_section(plan: list[dict], inputs: dict, wkey: str, ask_source: bool) -> bool | None:
    """확인 상자의 '읽기' 부분 — 안내 · ('내 저장소' 체크) · 가져올 주소 목록. 체크 칸을 그렸으면 고른 값, 아니면 None."""
    st.caption(READ_NOTICE)
    picked = None
    if ask_source and any(item["target"] and item["target"].kind == "github" for item in plan):
        picked = st.checkbox("내 저장소입니다 — GitHub 글을 내 문서(사실의 출처)로도 쓰기", key=f"{wkey}_mine",
                             value=bool(inputs.get("github_as_source")),
                             help="끄면 용어·프로젝트 이름을 참고하는 외부 자료로만 씁니다 (그 글의 사실·숫자를 내 경력으로 쓰지 못합니다). "
                                  "프로젝트 내용을 포트폴리오에 넣으려면 켜세요 — 내가 쓴 저장소일 때만.")
    as_source = bool(inputs.get("github_as_source") if picked is None else picked)
    st.markdown("**가져올 주소**\n\n" + "\n".join(_plan_line(item, as_source) for item in plan))
    return picked


def web_step(request: dict, team_id: str, task_id: str, models: ModelsConfig) -> dict:
    """외부 자료 승인 단계 — 가져올 주소(읽기)와 검색어 초안(보내기)을 보여 주고, 사람이 '승인'을 눌러야 가져오고 검색한다.

    필요 없으면(검색 꺼짐 + 주소 없음) request 를 그대로 돌려준다. 기다리는 동안은 빈 dict.
    주소만 넣고 검색을 켜지 않았으면 검색어 칸 없이 가져올 주소 목록만 확인한다.
    """
    key = f"career_web_{team_id}_{task_id}"
    if request:
        if not needs_web(request.get("tool_inputs") or {}, str(request.get("question") or "")):
            st.session_state.pop(key, None)
            return request
        nonce = (st.session_state.get(key) or {}).get("nonce", 0) + 1
        st.session_state[key] = {"request": request, "notes": None, "nonce": nonce, "result": None, "failed": []}
    pending = st.session_state.get(key)
    if not pending:
        return {}
    inputs, wkey = pending["request"]["tool_inputs"], f"{key}_{pending['nonce']}"

    with st.container(border=True):
        st.markdown("##### 🌐 외부 자료 확인 — 업무 시작 전")
        if pending.get("result"):
            return pending["result"] if _failed_box(pending, key, wkey) else {}
        role = career_docs.target_role(inputs)
        private = private_text(inputs)
        settings = load_settings()
        plan = url_plan(inputs, str(pending["request"].get("question") or ""), private, role)
        ready = [item for item in plan if item["target"]]
        # 💬 자유 질문에는 '내 저장소' 칸이 없다 → 이 상자에서 고른다 (양식은 양식의 체크 칸을 따른다)
        picked_source = _read_section(plan, inputs, wkey, ask_source=not inputs.get("career_kind")) if plan else None
        checks: list[tuple[str, list[str]]] = []
        if inputs.get("web_on"):
            st.caption(SEND_NOTICE)
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
        send = [q for q, p in checks if not p]
        label = ("✅ 승인 — 검색하고 글 가져와 진행" if send and ready else "🔍 승인 — 검색하고 진행" if send
                 else "📥 승인 — 글 가져오고 진행" if ready else "▶ 진행")
        c1, c2, c3 = st.columns(3)
        go = c1.button(label, type="primary", key=f"{wkey}_go")
        skip = c2.button("🚫 외부 자료 없이 진행", key=f"{wkey}_skip")
        if c3.button("✖ 취소 (업무 맡기지 않음)", key=f"{wkey}_cancel"):
            st.session_state.pop(key, None)
            st.rerun()
        if skip:
            st.session_state.pop(key, None)
            return approved_request(pending["request"], [], [], [], ["외부 자료 없이 진행 (사용자 선택)"])
        if go:
            notes, results = [], []
            if send:
                with st.spinner(f"승인한 검색어 {len(send)}개로 검색하는 중…"):
                    context = search.search_queries(send, settings)
                results, notes = [r.model_dump() for r in context.results], list(context.notes)
            if len(send) < len(checks):
                notes.append(f"개인 정보 검사에 걸린 검색어 {len(checks) - len(send)}개는 보내지 않았습니다.")
            notes += [f"{career_docs.FAILED_URL} {item['entered']}: {item['problem']}" for item in plan if not item["target"]]
            with st.spinner(f"승인한 주소 {len(ready)}개의 글을 가져오는 중…"):
                docs, failed = fetch_urls(ready)
            result = approved_request(pending["request"], send, results, docs, [*notes, *failed, *map(read_line, docs)], picked_source)
            if failed:  # 왜 안 됐는지 먼저 보여 주고, 가져온 글만으로 진행할지 고르게 한다
                pending.update(result=result, failed=failed, summary=f"가져온 글 {len(docs)}개 · 검색 결과 {len(results)}건")
                st.rerun()
            st.session_state.pop(key, None)
            return result
    return {}


# ─── 💬 수정 요청·추가 질문 글에 적은 주소 (결과가 나온 뒤) ───────────────────


def add_followup_docs(run: dict, docs: list[dict], as_source: bool, problems: list[str]) -> None:
    """확인 후 가져온 글을 이 업무의 근거 자료에 더한다 — 수정본을 쓰는 팀장 · 부장 검토 · 사실 검사 · 문서 미리보기가 같은 근거를 본다.

    글은 따로 묶은 툴 결과로 더하고(근거 자료가 길어 줄일 때 짧은 묶음은 통째로 남는다), 검사 값(career_source · career_web_urls)은
    원래 career_docs 결과에 더한다 (검사 값은 뒤 툴 결과가 앞의 것을 덮어쓰기 때문).
    서버 메모리에만 있다 — 앱을 다시 켜면 이 글은 근거에서 빠지므로, 그 뒤의 수정 요청에는 주소를 다시 적어야 한다.
    """
    text, source, urls = career_docs.followup_addendum(docs, as_source, problems)
    tools = run["tools"] = run.get("tools") or []
    main = next((t for t in tools if t.tool == "career_docs"), None)
    extra = ToolResult(tool="career_docs", title=f"🌐 수정 요청에서 가져온 글 ({len(docs)}개)", content=text)
    if text:
        tools.append(extra)
    options = (main or extra).harness_options
    if source:
        options["career_source"] = "\n\n".join(x for x in (str(options.get("career_source") or ""), source) if x)
    if urls and "career_web_urls" in options:
        options["career_web_urls"] = [*options["career_web_urls"], *urls]
    run.pop("career", None)  # 문서 미리보기(사실 검사)를 새 근거로 다시 만든다
    run.setdefault("followup_fetched", []).extend(read_line(d) + (" · 내 문서로" if as_source and d["kind"] == "github" else " · 참고용 외부 자료")
                                                  for d in docs)


def followup_step(run: dict, request: str, prepare=None) -> str:
    """💬 수정 요청·추가 질문 글에 적은 GitHub·웹 주소 — 가져올 주소를 보여 주고 확인받은 뒤 가져와, 이 업무의 근거 자료에 더한다.

    돌려주는 값 = 지금 보낼 요청 글 (주소가 없으면 받은 글 그대로 — 네트워크 호출 없음). 확인을 기다리는 동안은 빈 글.
    바로 가져오지 않고 확인을 거치는 이유: 다른 곳과 같은 원칙(확인 전에는 아무것도 나가지 않는다)이고, 이 상자가 '내 저장소입니다' 체크와
    실패 안내가 놓일 자리이기 때문이다 — 수정 요청은 보낸 뒤에는 백그라운드로 돌아 중간에 물을 수 없다.
    prepare: 근거 자료에 더하기 직전에 부르는 함수 (앱을 다시 켜 되살린 업무는 근거 자료를 먼저 되찾아야 한다 — app.py followup_evidence).
    """
    key = f"career_followup_web_{run['run_id']}"
    if request:
        if not career_search.find_urls(request):
            st.session_state.pop(key, None)
            return request
        nonce = (st.session_state.get(key) or {}).get("nonce", 0) + 1
        st.session_state[key] = {"request": request, "nonce": nonce, "result": None, "failed": []}
    if run.get("followup_fetched"):
        st.caption("📥 수정 요청에서 가져와 근거 자료에 더한 글\n\n" + "\n".join(f"- {line}" for line in run["followup_fetched"]))
    pending = st.session_state.get(key)
    if not pending:
        return ""
    wkey = f"{key}_{pending['nonce']}"
    inputs = run.get("career_inputs") or {}
    private = "\n".join(str(t.harness_options.get("career_source") or "") for t in run.get("tools") or [])

    def done(docs: list[dict], as_source: bool, problems: list[str]) -> str:
        if prepare:
            prepare()
        add_followup_docs(run, docs, as_source, problems)
        st.session_state.pop(key, None)
        return pending["request"]

    with st.container(border=True):
        st.markdown("##### 🌐 수정 요청에 적은 주소 — 보내기 전 확인")
        if pending.get("result"):
            return done(*pending["result"]) if _failed_box(pending, key, wkey, "✖ 취소 (보내지 않음)") else ""
        plan = url_plan({}, pending["request"], private, career_docs.target_role(inputs))
        ready = [item for item in plan if item["target"]]
        as_source = bool(_read_section(plan, inputs, wkey, ask_source=True))
        c1, c2, c3 = st.columns(3)
        go = c1.button("📥 승인 — 글 가져오고 보내기" if ready else "▶ 그대로 보내기", type="primary", key=f"{wkey}_go")
        skip = bool(ready) and c2.button("🚫 가져오지 않고 보내기", key=f"{wkey}_skip")
        if c3.button("✖ 취소 (보내지 않음)", key=f"{wkey}_cancel"):
            st.session_state.pop(key, None)
            st.rerun()
        if skip:
            st.session_state.pop(key, None)
            return pending["request"]
        if go:
            problems = [f"{career_docs.FAILED_URL} {item['entered']}: {item['problem']}" for item in plan if not item["target"]]
            with st.spinner(f"주소 {len(ready)}개의 글을 가져오는 중…"):
                docs, failed = fetch_urls(ready)
            if failed:  # 팀이 '읽을 수 없습니다'라고만 답하기 전에 — 어느 주소가 왜 안 됐고 무엇을 하면 되는지 먼저 알린다
                pending.update(result=(docs, as_source, problems + failed), failed=failed, summary=f"가져온 글 {len(docs)}개")
                st.rerun()
            return done(docs, as_source, problems)
    return ""


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
    def render(target: str) -> str:  # 형식 구간(<!-- format:docx --> · <!-- format:md -->)이 있으면 형식마다 그 구간
        return career.render_document(text, kind, questions, title, opts.get("career_requirement_text"), jds=jds, source=source,
                                      web=opts["web"], target=target)

    markdown = render("md")
    word = render("docx")
    fmt_opts = opts.get("career_format") if isinstance(opts.get("career_format"), dict) else {}
    files: dict[str, bytes] = {}
    for fmt in opts["formats"]:
        if fmt == "md":
            files[f"{title}.md"] = markdown.encode("utf-8")
        elif fmt == "docx":  # '[그림 N]' 자리에 원본 그림 (툴이 보관한 경로 — 되살린 업무는 없어서 글자 표시만), 원본 docx 가 있으면 틀로
            files[f"{title}.docx"] = to_docx(word, title, figures=opts.get("career_figures"), template=fmt_opts.get("template"))
        elif fmt == "hwpx":
            files[f"{title}.hwpx"] = to_hwpx(word, title)
    facts = career.check_facts(text, source or "", str(opts.get("career_reference") or "")) if source is not None else None
    marks = sorted(career.figure_marks(career.body_text(text)))
    figures = list(opts.get("career_figures") or [])
    out = {"key": key, "markdown": markdown, "files": files, "rows": career.limit_rows(text, questions), "facts": facts,
           "title": title, "kind": kind, "figure_marks": marks, "figures": len(figures),
           "split": word != markdown, "template": bool(fmt_opts.get("template")),
           "missing_figures": career.missing_figures(text, len(figures)) if figures else [],
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


def _figures_box(out: dict) -> None:
    """🖼 원본 그림 — Word 에 넣은 그림 수 · 빠진 표시 · 그림 파일이 없는(되살린) 업무 안내."""
    marks, figures = out.get("figure_marks") or [], out.get("figures") or 0
    if figures:
        placed = len([n for n in marks if n <= figures])
        st.caption(f"🖼 원본 그림 {figures}개 중 {placed}개를 Word(.docx) 의 '[그림 N]' 자리에 넣었습니다 "
                   "(한글 .hwpx 와 .md 에는 '[그림 N]' 글자로 남습니다).")
        if out.get("missing_figures"):
            st.warning("결과에서 빠진 그림 표시: " + ", ".join(f"[그림 {n}]" for n in out["missing_figures"])
                       + " — 원본 Word 에서 직접 복사해 붙이거나, 다시 맡기세요.")
    elif marks:
        st.caption("🖼 결과에 '[그림 N]' 표시가 있지만 원본 그림 파일 정보가 없습니다 (서버를 다시 켜 되살린 업무 등) — "
                   "Word 에는 글자로만 남습니다. 원본 Word 의 그림을 직접 붙이거나 업무를 다시 맡기세요.")


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
        _figures_box(out)
        if out.get("split"):
            st.caption("📐 형식별 본문: 미리보기·.md 는 md 형식 구간, Word(.docx)·한글(.hwpx) 은 Word 형식 구간으로 만들었습니다.")
        if out.get("template"):
            st.caption("📐 Word(.docx) 는 원본 Word 파일을 틀로 썼습니다 (스타일·여백·머리글/바닥글·표 스타일 유지, 본문은 새로 씀).")
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
