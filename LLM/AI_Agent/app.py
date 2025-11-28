import streamlit as st
from dotenv import load_dotenv  # 설정 값

from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# logging
from langchain_teddynote import logging

logging.langsmith("My Project(app)")
load_dotenv()

st.set_page_config(page_title="My ChatGPT", page_icon="🤖")
st.title("홈 - 일반 질문")

# --- 사이드바 ---
with st.sidebar:
    st.markdown("### 페이지")
    st.markdown("- 이 페이지: 일반 Q&A")
    st.markdown(
        "- 사이드바 상단 메뉴에서 **1_english_word_maker** 페이지로 이동할 수 있어요."
    )
    option = st.selectbox(
        "Please Select Prompt",
        ("Basic", "Summary"),
        index=0,
    )

# --- 대화 상태 저장 ---
if "messages_home" not in st.session_state:
    st.session_state["messages_home"] = []


def add_message_home(role, content):
    st.session_state["messages_home"].append(ChatMessage(role=role, content=content))


def print_messages_home():
    for chat_message in st.session_state["messages_home"]:
        st.chat_message(chat_message.role).write(chat_message.content)


print_messages_home()


# --- 체인 생성 (단어장 기능 없이) ---
def create_chain_home(prompt_type: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다. 자세히 답변해 주세요."),
            ("user", "#Question:\n{question}"),
        ]
    )

    if prompt_type == "Summary":
        prompt = hub.pull("baektest/chain-of-density-map-korean", include_model=True)

    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


# --- 사용자 입력 ---
user_input = st.chat_input("내용을 입력하세요")

if user_input:
    st.chat_message("user").write(user_input)
    add_message_home("user", user_input)

    chain = create_chain_home(option)

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in chain.stream({"question": user_input}):
            ai_answer += token
            container.markdown(ai_answer)

    add_message_home("assistant", ai_answer)

# --- 단어장 페이지로 이동 링크 (선택사항) ---
# st.markdown("---")
# st.markdown("📚 단어장을 사용하려면 아래 링크를 눌러 이동하세요.")

try:
    st.page_link(
        "pages/1_english_word_maker.py",
        label="➡ 영어 단어장 페이지로 이동",
        page_title="📖 영어 단어장",
        icon="📖",
    )
except Exception:
    st.info("사이드바 상단의 페이지 메뉴에서 '단어장'을 선택해 주세요.")
