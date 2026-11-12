"""테스트용 가짜 모델 — 실제 모델 대신 정해 둔 답을 낸다."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel


def fake_chat(*answers: str):
    """정해 둔 답을 차례로 내는 가짜 채팅 모델 (LangChain 제공). 답이 떨어지면 처음부터 다시 낸다."""
    return FakeListChatModel(responses=list(answers))
