# web
import streamlit as st
from langchain_core.messages.chat import ChatMessage

# LLM
from langchain_core.prompts import ChatPromptTemplate  # prompt
from langchain_core.prompts import load_prompt  # prompt load

from langchain import hub  # prompt hub
from langchain_openai import ChatOpenAI  # AI
from langchain_core.output_parsers import StrOutputParser  # 출력
from dotenv import load_dotenv  # 설정 값

# 단어장 다운로드
import json, re
import pandas as pd
from io import BytesIO
from docx import Document
from fpdf import FPDF
from pathlib import Path  # 단어장 위치

load_dotenv()

st.title("my chatgpt")

# 단어장 저장 위치
SAVE_DIR = Path("English_word")
SAVE_PATH = SAVE_DIR / "vocab.json"

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위해 생성
    st.session_state["messages"] = []

# 단어장 생성을 위한 초기 코드
if "vocab" not in st.session_state:
    st.session_state["vocab"] = {}
    if SAVE_PATH.exists():
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # loaded 는 {word_key: entry_dict} 형태라고 가정
            st.session_state["vocab"] = loaded
        except Exception as e:
            st.warning(f"저장된 단어장 로드 실패: {e}")

# 사이드바 생성
with st.sidebar:
    # 대화 초기화 버튼
    clear_bar = st.button("대화 초기화")

    option = st.selectbox(
        "Please Select Prompt", ("Basic", "English", "Summary"), index=0
    )


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 대화 초기화
if clear_bar:
    st.session_state["messages"] = []
# 이전 대화 기록 출력
print_messages()


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# chain 생성
def create_chain(prompt_type):
    # 프롬프트(기본 모드)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다. 자세히 답변해 주세요."),
            ("user", "#Question:\n{question}"),
        ]
    )
    if prompt_type == "English":
        prompt = load_prompt("prompts/English.yaml", encoding="utf-8")
        prompt
    elif prompt_type == "Summary":
        prompt = hub.pull("baektest/chain-of-density-map-korean", include_model=True)
        # prompt = hub.pull("teddynote/chain-of-density-prompt", include_model=True)

    # AI
    llm = ChatOpenAI(model_name="gpt-5.1", temperature=0)  # gpt-5 오류 발생
    # 출력
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 프롬프트 설정에 따른 단어장 생성
def parse_yaml_style_answer(answer_text: str) -> dict:
    """
    YAML 프롬프트에서 나온 번호 기반 텍스트를 파싱해서
    { "word": ..., "domain": ..., ... } 형태의 dict로 변환
    """
    # 번호 -> 내부 key 매핑
    field_map = {
        1: "word",
        2: "domain",
        3: "pos",
        4: "ipa",
        5: "meaning",
        6: "senses",
        7: "level",
        8: "frequency",
        9: "etymology",
        10: "examples",
        11: "synonyms",
        12: "antonyms",
        13: "collocations",
        14: "derivatives",
        15: "tags",
        16: "confusable",
        17: "lookup_count_raw",  # LLM이 적어준 값(우리는 참고만)
    }

    # "1) ", "2) " ... 로 시작하는 부분 찾기
    pattern = re.compile(r"^\s*(\d+)\)\s", re.MULTILINE)
    matches = list(pattern.finditer(answer_text))

    result = {}

    if not matches:
        return result  # 파싱 실패 시 빈 dict

    for idx, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()

        if idx + 1 < len(matches):
            end = matches[idx + 1].start()
        else:
            end = len(answer_text)

        block = answer_text[start:end].strip()

        # "N) 제목(Title): 내용..." 형태에서 첫 번째 ":" 뒤를 값으로 사용
        header_split = block.split(":", 1)
        if len(header_split) == 2:
            value = header_split[1].strip()
        else:
            value = ""

        key = field_map.get(num)
        if key:
            result[key] = value

    # word가 여러 줄일 수 있으니 첫 줄만 사용
    if "word" in result and result["word"]:
        first_line = result["word"].splitlines()[0].strip()
        result["word"] = first_line

    # 태그 / 유의어 / 반의어 / 콜로케이션 등은 쉼표 기준으로 리스트로 쪼개도 됨
    for list_key in ["synonyms", "antonyms", "collocations", "derivatives", "tags"]:
        if list_key in result and result[list_key]:
            # "a, b, c" -> ["a", "b", "c"]
            items = [x.strip() for x in result[list_key].split(",") if x.strip()]
            result[list_key] = items

    return result


# 단어장 생성용 함수
def update_vocab_from_answer(answer_text: str):
    # LLM이 넘겨준 JSON 텍스트(answer_text)를 파싱해서 단어장에 반영
    parsed = parse_yaml_style_answer(answer_text)
    word = parsed.get("word", "")

    if not word:
        return  # 단어 못 찾으면 무시

    word_key = word.lower().strip()
    vocab = st.session_state["vocab"]

    if word_key in vocab:
        # 이미 있는 단어 → 조회수 +1
        vocab[word_key]["lookup_count"] += 1

    else:
        # 새 단어 → parsed 정보를 그대로 넣고 lookup_count는 1부터 시작
        entry = parsed.copy()
        entry["lookup_count"] = 1  # LLM이 써준 15번 필드는 무시하고 우리가 관리
        vocab[word_key] = entry


# 단어장을 저장하기 위한 작업(DF화)
def vocab_to_df() -> pd.DataFrame:
    """session_state['vocab'] 를 보기 좋은 DataFrame으로 변환"""
    vocab = st.session_state.get("vocab", {})
    rows = []

    for word_key, info in vocab.items():
        rows.append(
            {
                "Word": info.get("word", word_key),
                "Domain": info.get("domain", ""),
                "POS": info.get("pos", ""),
                "IPA": info.get("ipa", ""),
                "Meaning": info.get("meaning", ""),
                "Senses": info.get("senses", ""),
                "Level": info.get("level", ""),
                "Frequency": info.get("frequency", ""),
                "Etymology": info.get("etymology", ""),
                "Examples": info.get("examples", ""),
                "Synonyms": ", ".join(info.get("synonyms", [])),
                "Antonyms": ", ".join(info.get("antonyms", [])),
                "Collocations": ", ".join(info.get("collocations", [])),
                "Derivatives": ", ".join(info.get("derivatives", [])),
                "Tags": ", ".join(info.get("tags", [])),
                "Confusable": info.get("confusable_word", ""),
                "LookupCount": info.get("lookup_count", 1),
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
                "Meaning",
                "Senses",
                "Level",
                "Frequency",
                "Etymology",
                "Examples",
                "Synonyms",
                "Antonyms",
                "Collocations",
                "Derivatives",
                "Tags",
                "Confusable",
                "LookupCount",
            ]
        )


# Excel
def get_excel_bytes(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Vocab")
    output.seek(0)
    return output


# Word(docx)
def get_word_bytes(df: pd.DataFrame):
    doc = Document()
    doc.add_heading("단어장", level=1)

    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = col

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            row_cells[i].text = str(row[col])

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# PDF 용 text 정리 함수
def clean_text(value) -> str:
    """fpdf2에 넣기 전에 텍스트를 안전한 형태로 정리"""
    if value is None:
        return ""
    text = str(value)

    # 줄바꿈/캐리지리턴/탭 같은 제어 문자 정리
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # fpdf가 싫어할 수 있는 control char 제거 (ASCII 32 미만)
    text = "".join(ch if ord(ch) >= 32 else " " for ch in text)

    return text


KOREAN_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"


# PDF
def get_pdf_bytes(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1️⃣ 유니코드 TTF 폰트 등록 + 사용
    pdf.add_font("malgun", "", KOREAN_FONT_PATH, uni=True)
    pdf.set_font("malgun", size=12)

    # 2️⃣ 여기서부터는 아무 텍스트나(한글/영문/특수기호) 써도 됨
    for _, row in df.iterrows():
        # 모든 텍스트를 한 번 정리
        word = clean_text(row.get("Word", ""))
        pos = clean_text(row.get("POS", ""))
        lookup = clean_text(row.get("LookupCount", ""))

        meaning_src = row.get("Meaning", "")
        meaning = clean_text(meaning_src)

        # 1) 첫 줄: multi_cell 대신 cell 사용 (줄바꿈 알고리즘 안 탄다)
        line1 = f"{word} ({pos})  [lookup {lookup}]"
        safe_line1 = line1[:200]  # 혹시 모를 경우를 위해 너무 길면 잘라줌
        pdf.cell(0, 8, safe_line1, ln=1)  # ln=1 → 이 줄 출력 후 줄바꿈

        # 2) 의미: 길 수 있으니 우리가 직접 잘라서 여러 줄로 찍기
        prefix = "- mean: "
        text = prefix + meaning
        max_chars = 80  # 한 줄에 찍을 최대 문자 수 (대략)

        while text:
            chunk = text[:max_chars]
            pdf.cell(0, 8, chunk, ln=1)
            text = text[max_chars:]

        # ===== 3) 예시 문장 (최대 2개) =====
        examples_raw = str(row.get("Examples", "") or "")
        examples_raw = clean_text(examples_raw)
        if examples_raw:
            pdf.cell(0, 8, "- examples:", ln=1)
            # 줄 단위로 쪼개서 앞 2개만 사용
            example_lines = [e.strip() for e in examples_raw.split("\n") if e.strip()]
            for ex in example_lines[:2]:
                ex_line = f"  · {ex}"
                # 너무 길면 자르기
                while ex_line:
                    chunk = ex_line[:max_chars]
                    pdf.cell(0, 8, chunk, ln=1)
                    ex_line = ex_line[max_chars:]

        # ===== 4) 유의어 (최대 3개) =====
        syn_raw = str(row.get("Synonyms", "") or "")
        syn_list = [s.strip() for s in syn_raw.split(",") if s.strip()]
        syn_list = syn_list[:3]  # 최대 3개
        if syn_list:
            syn_line = "- synonyms: " + ", ".join(syn_list)
            syn_line = clean_text(syn_line)
            while syn_line:
                chunk = syn_line[:max_chars]
                pdf.cell(0, 8, chunk, ln=1)
                syn_line = syn_line[max_chars:]

        # ===== 5) 반의어 (최대 3개) =====
        ant_raw = str(row.get("Antonyms", "") or "")
        ant_list = [a.strip() for a in ant_raw.split(",") if a.strip()]
        ant_list = ant_list[:3]
        if ant_list:
            ant_line = "- antonyms: " + ", ".join(ant_list)
            ant_line = clean_text(ant_line)
            while ant_line:
                chunk = ant_line[:max_chars]
                pdf.cell(0, 8, chunk, ln=1)
                ant_line = ant_line[max_chars:]

        # 단어 사이에 빈 줄 하나
        pdf.ln(4)

    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# 단어장 저장 함수
def save_vocab_to_disk():
    """현재 session_state['vocab']를 JSON으로 저장"""
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


# 사용자 입력
user_input = st.chat_input("내용 입력")

# 입력 시
if user_input:
    # web 사용자 대화 출력(사용자 입력)
    st.chat_message("user").write(user_input)

    # 사용자 대화 기록 저장
    add_message("user", user_input)

    # chain 생성
    chain = create_chain(option)
    if option == "English":
        # 단어장 agent: 전체 응답을 한번에 받아서 처리
        answer_text = chain.invoke({"word": user_input})
        # 화면 출력
        st.chat_message("assistant").markdown(answer_text)
        add_message("assistant", answer_text)
        # 단어장 업데이트
        update_vocab_from_answer(answer_text)
    else:
        responce = chain.stream({user_input})
        # web assistant 대화 출력 방법2
        with st.chat_message("assistant"):
            # 빈 공간 생성
            container = st.empty()
            ai_answer = ""
            for token in responce:
                ai_answer += token
                container.markdown(ai_answer)

        # assistant 대화 저장
        add_message("assistant", ai_answer)

st.subheader("현재 단어장")
df = vocab_to_df()
st.dataframe(df, use_container_width=True)

# Excel
if not df.empty:
    # 공통: 버튼 누르면 서버에 단어장 Json 저장
    save_vocab_to_disk()

    # 1) Excel download
    excel_bytes = get_excel_bytes(df)
    st.download_button(
        label="📥 단어장 다운로드 (Excel)",
        data=excel_bytes,
        file_name="vocab.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # 2) Word(docs) download
    word_bytes = get_word_bytes(df)
    st.download_button(
        label="📥 단어장 다운로드 (Word)",
        data=word_bytes,
        file_name="vocab.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    # 3) PDF download
    pdf_bytes = get_pdf_bytes(df)
    st.download_button(
        label="📥 단어장 다운로드 (PDF)",
        data=pdf_bytes,
        file_name="vocab.pdf",
        mime="application/pdf",
    )
else:
    st.info("아직 저장된 단어가 없습니다.")
