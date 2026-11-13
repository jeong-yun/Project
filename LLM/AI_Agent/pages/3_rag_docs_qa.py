"""문서 Q&A (RAG) 화면.

문서 묶음 고르기 → 질문 → 낱말 검색(BM25)으로 조각 찾기 → 찾은 조각을 넣어 모델이 답하기 → 답 + 출처 검사 + 찾은 조각.
계산은 rag/ 모듈이 하고, 이 파일은 화면 흐름만 맡는다.
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import common
from rag import answer, documents, search

load_dotenv()

st.set_page_config(page_title="문서 Q&A", page_icon="📄")
st.title("문서 Q&A (RAG)")
st.caption(
    "문서에서 질문과 관련된 조각을 먼저 찾고, 그 조각만 근거로 답합니다. "
    "답에 붙은 [1]·[2] 는 아래 '찾은 조각'의 번호입니다."
)

# ---------------- 문서 묶음 ----------------

DEFAULT_LABEL = "기본 문서 (AI 거버넌스 문서)"
UPLOAD_LABEL = "직접 올리기"
source = st.radio("문서 묶음", (DEFAULT_LABEL, UPLOAD_LABEL), horizontal=True)

docs: list[tuple[str, str]] = []
if source == DEFAULT_LABEL:
    docs = documents.load_default_documents()
    if not docs:
        st.warning(
            "기본 문서 폴더를 찾지 못했습니다. 저장소 전체를 받았는지 확인하거나 '직접 올리기'를 골라 주세요.\n\n"
            f"찾아본 곳: `{documents.DEFAULT_DOCS_DIR}`"
        )
else:
    uploaded = st.file_uploader(
        "문서 올리기 (.md · .txt · .pdf, 여러 개 가능)", type=["md", "txt", "pdf"], accept_multiple_files=True
    )
    for file in uploaded or []:
        try:
            docs.append((file.name, documents.read_document(file.name, file.getvalue())))
        except Exception as e:  # 읽지 못한 파일은 건너뛰고 이유를 알린다
            st.warning(str(e) if isinstance(e, documents.UnsupportedDocument) else f"'{file.name}' 을 읽지 못했습니다: {e}")

st.caption(
    "올린 문서는 이 PC 에서만 읽고 따로 저장하지 않습니다. 다만 **질문과, 찾은 조각 "
    f"{answer.TOP_K}개의 내용은 고른 모델(OpenAI)로 전송됩니다.** "
    "회사 문서나 개인정보가 든 문서는 외부 모델로 보내도 되는지 먼저 확인해 주세요."
)

model = st.selectbox("답변 모델", common.MODEL_CHOICES, index=0)

chunks = documents.split_documents(docs)
index = search.BM25Index(chunks)
if docs:
    st.caption(f"문서 {len(docs)}개를 조각 {len(chunks)}개로 나눴습니다 (조각마다 최대 {documents.MAX_CHARS}자).")

# ---------------- 질문 ----------------

question = st.chat_input("문서에 대해 질문해 보세요", disabled=not chunks)
if question:
    hits = index.search(question, answer.TOP_K)
    result = {"question": question, "hits": hits, "answer": None, "error": None}
    if not hits:
        result["answer"] = answer.ask(question, index, chain=None)["answer"]  # 조각이 없으면 모델을 부르지 않는다
    elif not os.environ.get("OPENAI_API_KEY"):
        result["error"] = (
            "OpenAI 키를 찾지 못해 답은 만들지 못했습니다 (찾은 조각은 아래에 있습니다). "
            "이 앱 폴더에 `.env` 파일을 만들고 `OPENAI_API_KEY=...` 한 줄을 넣은 뒤 앱을 다시 실행해 주세요."
        )
    else:
        try:
            with st.spinner("답을 만드는 중…"):
                result["answer"] = answer.ask(question, index, answer.build_chain(common.make_llm(model)))["answer"]
        except Exception as e:
            result["error"] = f"모델 호출에 실패했습니다: {e}\n\n모델 이름과 OpenAI 키를 확인한 뒤 다시 질문해 주세요."
    st.session_state["rag_result"] = result

# ---------------- 결과 ----------------

result = st.session_state.get("rag_result")
if result:
    st.chat_message("user").write(result["question"])
    if result["error"]:
        st.error(result["error"])
    if result["answer"]:
        st.chat_message("assistant").write(result["answer"])
        check = answer.check_citations(result["answer"], result["hits"], result["question"])
        if result["hits"] and not check["ok"]:
            st.warning("출처 검사에서 확인이 필요한 점이 있습니다 — 아래 조각과 직접 비교해 주세요.\n\n- " + "\n- ".join(check["problems"]))
        elif result["hits"] and not check["not_in_docs"]:
            st.caption("출처 검사: 출처 번호가 찾은 조각 번호와 맞고, 답의 숫자·영문 낱말이 인용한 조각에 있습니다. (문장의 뜻까지 검사한 것은 아닙니다.)")

    if result["hits"]:
        st.subheader("찾은 조각")
        st.dataframe(
            pd.DataFrame(
                [
                    {"번호": n, "파일": hit["source"], "제목": hit["title"], "점수": hit["score"]}
                    for n, hit in enumerate(result["hits"], 1)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        for n, hit in enumerate(result["hits"], 1):
            with st.expander(f"[{n}] {hit['source']} · {hit['title'] or '제목 없음'}"):
                st.text(hit["text"])
