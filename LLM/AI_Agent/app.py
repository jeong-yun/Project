import streamlit as st
from dotenv import load_dotenv  # 설정 값

from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from common import PROMPTS_DIR, as_data, enable_langsmith, make_llm

enable_langsmith("My Project(app)")
load_dotenv()

st.set_page_config(page_title="My ChatGPT", page_icon="🤖")
st.title("홈 - 일반 질문")

# --- 사이드바 ---
with st.sidebar:
    st.markdown("### 페이지")
    st.markdown("- 이 페이지: 일반 Q&A")
    st.markdown("- 사이드바 상단 메뉴에서 **영어 단어장** · **AI 안전성 평가** · **문서 Q&A** 페이지로 이동할 수 있어요.")
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
        # 저장소 안의 프롬프트 파일을 쓴다 (예전에는 남이 올린 공개 프롬프트를 실행할 때마다 내려받았다).
        # 요약할 글은 '지시가 아닌 데이터' 표시선 안에 넣는다.
        summary = load_prompt(str(PROMPTS_DIR / "summary.yaml"), encoding="utf-8")
        prompt = RunnableLambda(lambda x: {"question": as_data(x["question"])}) | summary

    llm = make_llm("gpt-5.1", temperature=0)
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

# --- 다른 페이지로 이동 링크 ---
PAGE_LINKS = [
    ("pages/1_english_word_maker.py", "➡ 영어 단어장", "📖"),
    ("pages/2_ai_safety_eval.py", "➡ AI 안전성 평가", "🛡️"),
    ("pages/3_rag_docs_qa.py", "➡ 문서 Q&A (RAG)", "📄"),
]
try:
    for path, label, icon in PAGE_LINKS:
        st.page_link(path, label=label, icon=icon)
except Exception:
    # `streamlit run app.py` 가 아닌 방식으로 열면 링크를 만들 수 없다
    st.info("사이드바 상단의 페이지 메뉴에서 페이지를 선택해 주세요.")
