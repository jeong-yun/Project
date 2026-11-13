"""여러 화면·모듈이 함께 쓰는 것: 폴더 경로, 모델 만들기, '지시가 아닌 데이터' 표시선.

경로는 이 파일 위치를 기준으로 잡는다 — 어느 폴더에서 실행해도 같은 파일을 읽는다.
"""

from pathlib import Path

from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
VOCAB_DIR = BASE_DIR / "English_word"  # 단어장 저장 (vocab.json, audio/)

# 화면에서 고를 수 있는 모델 이름
MODEL_CHOICES = ("gpt-5.1", "gpt-4.1", "gpt-4.1-mini")


def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """모델 이름에 맞게 ChatOpenAI 객체를 만든다.

    gpt-5 계열(gpt-5, gpt-5.1 등)은 temperature 를 바꿀 수 없어 0 을 주면 400 오류가 난다고 알려져 있다
    (이 저장소에서 실제 호출로 확인하지는 않았다). 그래서 gpt-5 계열이면 temperature 를 넘기지 않는다.
    """
    if model.startswith("gpt-5"):
        return ChatOpenAI(model=model)
    return ChatOpenAI(model=model, temperature=temperature)


# --- '지시가 아닌 데이터' 표시선 -------------------------------------------------
# 모델 응답·문서 조각처럼 밖에서 온 글을 프롬프트에 넣을 때 이 표시선으로 감싼다.
DATA_OPEN = "<<<데이터 시작>>>"
DATA_CLOSE = "<<<데이터 끝>>>"


def as_data(text) -> str:
    """표시선 안에 넣을 글에서 표시선 흉내를 무력화한다.

    글 안에 '<<<데이터 끝>>>' 을 적어 표시선을 일찍 닫은 척하는 것을 막는다 (꺾쇠를 다른 글자로 바꾼다).
    """
    return str(text).replace("<<<", "‹‹‹").replace(">>>", "›››")
