"""테스트용 가짜 — 실제 모델·발음(TTS) 대신 정해 둔 답을 낸다."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

# '키가 있을 때'의 화면 흐름을 볼 때 환경 변수에 넣는 가짜 값. 실제 키가 아니다.
FAKE_KEY = "test-not-a-real-key"


def fake_chat(*answers: str):
    """정해 둔 답을 차례로 내는 가짜 채팅 모델 (LangChain 제공). 답이 떨어지면 처음부터 다시 낸다."""
    return FakeListChatModel(responses=list(answers))


class FakeTTSClient:
    """OpenAI 클라이언트의 발음 만들기(audio.speech.create) 자리에 끼우는 가짜. 부른 횟수를 센다."""

    def __init__(self):
        self.calls = 0
        self.audio = self
        self.speech = self

    def create(self, **kwargs):
        self.calls += 1
        return self

    def read(self) -> bytes:
        return b"mp3"
