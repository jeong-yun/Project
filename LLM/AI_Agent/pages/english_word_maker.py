import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote import logging

# --- 단어장 기능 모듈 ---
from english_word_maker.parsing import parse_yaml_style_answer
from english_word_maker.export_doc import (
    get_excel_bytes,
    get_word_bytes_from_vocab,
    get_pdf_bytes_from_vocab,
)
from english_word_maker.tts_utils import (
    tts_bytes,
    get_audio_zip_from_vocab,
)

load_dotenv()
logging.langsmith("English_word_maker_project")

st.set_page_config(page_title="영어 단어장", page_icon="📚")
st.title("영어 단어장 생성 & 발음 앱")

# ---------------- 기본 설정 ----------------

SAVE_DIR = Path("English_word")
SAVE_PATH = SAVE_DIR / "vocab.json"

if "messages_vocab" not in st.session_state:
    st.session_state["messages_vocab"] = []

if "vocab" not in st.session_state:
    st.session_state["vocab"] = {}
    if SAVE_PATH.exists():
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                st.session_state["vocab"] = json.load(f)
        except Exception as e:
            st.warning(f"저장된 단어장 로드 실패: {e}")

if "last_word_key" not in st.session_state:
    st.session_state["last_word_key"] = None

# ---------------- 유틸 함수 ----------------


def add_message(role, content):
    st.session_state["messages_vocab"].append(ChatMessage(role=role, content=content))


def print_messages():
    for chat_message in st.session_state["messages_vocab"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def save_vocab_to_disk():
    try:
        SAVE_DIR.mkdir(exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                st.session_state.get("vocab", {}),
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        st.error(f"단어장 저장 중 오류 발생: {e}")


def vocab_to_df(vocab: dict) -> pd.DataFrame:
    rows = []
    for word_key, info in vocab.items():
        senses_ko = info.get("senses_ko", {})
        senses_en = info.get("senses_en", {})
        examples = info.get("examples", {})
        synonyms = info.get("synonyms", [])
        antonyms = info.get("antonyms", [])
        collocs = info.get("collocations", [])
        derivatives = info.get("derivatives", [])
        tags = info.get("tags", [])
        similar_words = info.get("similar_words", [])
        confusable_raw = info.get("confusable", "")

        rows.append(
            {
                "Word": info.get("word", word_key),
                "Domain": info.get("domain", ""),
                "POS": info.get("pos", ""),
                "IPA": info.get("ipa", ""),
                "Level": info.get("level", ""),
                "Frequency": info.get("frequency", ""),
                "Etymology": info.get("etymology", ""),
                "LookupCount": info.get("lookup_count", 1),
                "Senses_Ko": json.dumps(senses_ko, ensure_ascii=False),
                "Senses_En": json.dumps(senses_en, ensure_ascii=False),
                "Examples": json.dumps(examples, ensure_ascii=False),
                "Synonyms": json.dumps(synonyms, ensure_ascii=False),
                "Antonyms": json.dumps(antonyms, ensure_ascii=False),
                "Collocations": json.dumps(collocs, ensure_ascii=False),
                "Derivatives": json.dumps(derivatives, ensure_ascii=False),
                "Tags": json.dumps(tags, ensure_ascii=False),
                "SimilarWords": json.dumps(similar_words, ensure_ascii=False),
                "ConfusableRaw": confusable_raw,
            }
        )

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(
            columns=[
                "Word",
                "Domain",
                "POS",
                "IPA",
                "Level",
                "Frequency",
                "Etymology",
                "LookupCount",
                "Senses_Ko",
                "Senses_En",
                "Examples",
                "Synonyms",
                "Antonyms",
                "Collocations",
                "Derivatives",
                "Tags",
                "SimilarWords",
                "ConfusableRaw",
            ]
        )


def update_vocab_from_answer(vocab: dict, answer_text: str):
    """LLM 응답 전체 텍스트를 파싱해서 vocab에 반영"""
    parsed = parse_yaml_style_answer(answer_text)
    word = (parsed.get("word") or "").strip()
    if not word:
        return None

    word_key = word.lower()
    if word_key in vocab:
        vocab[word_key]["lookup_count"] = vocab[word_key].get("lookup_count", 0) + 1
        # 기존 항목에 새 정보가 있으면 덮어쓰기(원하면 merge 로직 추가 가능)
        vocab[word_key].update(parsed)
    else:
        entry = parsed.copy()
        entry["lookup_count"] = 1
        vocab[word_key] = entry

    return word_key


# ---------------- LLM 체인 ----------------


def create_chain():
    # 항상 English.yaml 프롬프트 로드
    prompt = load_prompt("prompts/English.yaml", encoding="utf-8")  # 수정

    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


# ---------------- 사이드바 ----------------

with st.sidebar:
    st.markdown("단어장 옵션")
    st.write("프롬프트: English (고정)")

print_messages()

# ---------------- 메인 채팅 입력 ----------------

user_input = st.chat_input("단어 또는 질문을 입력하세요")

if user_input:
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    chain = create_chain()

    answer_text = chain.invoke({"word": user_input})
    st.chat_message("assistant").markdown(answer_text)
    add_message("assistant", answer_text)

    vocab = st.session_state["vocab"]
    word_key = update_vocab_from_answer(vocab, answer_text)
    if word_key:
        st.session_state["last_word_key"] = word_key

# ---------------- 최근 조회한 단어 발음 ----------------

st.subheader("최근 조회한 단어 발음")

last_key = st.session_state.get("last_word_key")
vocab = st.session_state.get("vocab", {})

if last_key and last_key in vocab:
    entry = vocab[last_key]
    word = entry.get("word", last_key)
    examples = entry.get("examples", {})

    st.write(f"**{word}** 발음")

    if st.button("🔊 단어 발음 듣기"):
        audio = tts_bytes(word)
        if audio:
            st.audio(audio, format="audio/mp3")

    if isinstance(examples, dict) and examples:
        st.write("예문 발음 (최대 3개)")
        for idx, (label, ex) in enumerate(examples.items()):
            if idx >= 3:
                break
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"({label}) {ex}")
            with col2:
                if st.button(f"🔊 예문 {idx+1}", key=f"ex_{last_key}_{idx}"):
                    audio = tts_bytes(ex)
                    if audio:
                        st.audio(audio, format="audio/mp3")
else:
    st.info("아직 조회한 단어가 없습니다.")

# ---------------- 단어장 테이블 & 다운로드 ----------------

st.subheader("현재 단어장")

df = vocab_to_df(vocab)
st.dataframe(df, use_container_width=True)

if not df.empty:
    save_vocab_to_disk()

    excel_bytes = get_excel_bytes(df)
    st.download_button(
        label="📥 단어장 다운로드 (Excel)",
        data=excel_bytes,
        file_name="vocab.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    word_bytes = get_word_bytes_from_vocab(vocab)
    st.download_button(
        label="📥 단어장 다운로드 (Word)",
        data=word_bytes,
        file_name="vocab.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    pdf_bytes = get_pdf_bytes_from_vocab(vocab)
    st.download_button(
        label="📥 단어장 다운로드 (PDF)",
        data=pdf_bytes,
        file_name="vocab.pdf",
        mime="application/pdf",
    )

    audio_zip = get_audio_zip_from_vocab(vocab)
    st.download_button(
        label="🔊 단어/예문 발음 다운로드 (ZIP)",
        data=audio_zip,
        file_name="vocab_audio.zip",
        mime="application/zip",
    )
else:
    st.info("아직 저장된 단어가 없습니다.")
