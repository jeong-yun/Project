"""낱말 검색 (BM25). 외부 패키지·임베딩 호출 없이 순수 파이썬으로 돈다.

BM25 는 '질문의 낱말이 조각에 얼마나 나오는가'로 점수를 매긴다. 드문 낱말일수록, 짧은 조각일수록 점수가 높다.
"""

import math
import re
from collections import Counter

_WORD = re.compile(r"[0-9a-z]+(?:[._-][0-9a-z]+)*|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """글을 검색용 낱말로 나눈다.

    - 한글: 글자 2개씩 묶는다 ("대출기간은" → 대출·출기·기간·간은). 조사를 떼는 형태소 분석 없이도 낱말이 겹친다.
    - 영문·숫자: 소문자로 통째 한 낱말. 점·밑줄·붙임표로 이어진 것(meta.json, gpt-5)은 통째 + 나눈 것 둘 다.
    """
    tokens: list[str] = []
    for match in _WORD.finditer(text.lower()):
        word = match.group()
        if "가" <= word[0] <= "힣":
            tokens += [word] if len(word) == 1 else [word[i : i + 2] for i in range(len(word) - 1)]
        else:
            tokens.append(word)
            parts = re.split(r"[._-]", word)
            if len(parts) > 1:
                tokens += parts
    return tokens


class BM25Index:
    """조각 목록으로 만든 검색 색인. 조각의 제목과 본문을 함께 색인한다."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c["title"] + "\n" + c["text"])) for c in self.chunks]
        self.length = [sum(tf.values()) for tf in self.tf]
        self.avg_length = (sum(self.length) / len(self.length)) if self.length else 1.0
        n = len(self.chunks)
        df = Counter(token for tf in self.tf for token in tf)
        self.idf = {token: math.log(1 + (n - d + 0.5) / (d + 0.5)) for token, d in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        """질문과 가장 잘 맞는 조각 k 개를 점수 높은 순으로. 각 항목 = 조각 + "score". 겹치는 낱말이 없으면 빈 목록."""
        words = [w for w in set(tokenize(query)) if w in self.idf]
        scored = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.length[i] / self.avg_length)
            score = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in words if w in tf)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{**self.chunks[i], "score": round(score, 3)} for score, i in scored[:k]]
