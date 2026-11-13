"""검색 품질 측정 — 모델을 부르지 않는다.

기본 문서(AI 거버넌스 문서 7개)에 대한 질문 15개와 '정답이 들어 있는 조각'을 사람이 문서를 읽고 정했다.
정답 조각 = 그 파일의 조각 가운데 `must_contain` 글귀가 들어 있는 조각.
잰 것: 상위 k 개 조각 안에 정답 조각이 드는 질문의 비율.

실행:  python -m rag.eval_retrieval      (조각 크기별 비교 표를 찍는다)

주의: 질문을 검색기를 만든 쪽이 썼다. 문서의 낱말을 그대로 쓴 질문이 많아 실제 사용자의 질문보다 쉬울 수 있다.
"""

from rag import documents, search

# (질문, 정답 파일 이름의 앞부분, 정답 조각에 들어 있는 글귀, 정답이 있는 절 — 사람이 읽는 메모)
QUESTIONS = [
    ("검토 역할에 쓰는 모델과 max_tokens 값은 무엇인가?", "01_", "max_tokens 32000", "1.1 목록"),
    ("외부 LLM 제공자에게 보내지 않는 데이터에는 어떤 것이 있나?", "01_", "LLM 제공자에게 가지 않는 데이터", "1.3 데이터 흐름"),
    ("생성 모델이 정규식을 잘못 읽어 룰과 안 맞는 샘플을 만드는 것은 NIST AI 600-1 의 어떤 위험 범주인가?", "02_", "Confabulation (그럴듯한 오류)", "2.5 NIST AI 600-1 …"),
    ("위험 점수는 어떻게 계산하고 몇 점부터 높음 등급인가?", "03_", "위험 점수 = L × I", "3.1 평가 척도"),
    ("API 키 유출 위험의 원인과 지금 통제는 무엇인가?", "03_", "`.env` 를 실수로 커밋", "R06 API 키·OINKCODE 유출"),
    ("정규식 검사가 끝나지 않고 멈추는 문제는 어떻게 통제하나?", "03_", "3_0a_common.REGEX_TIMEOUT", "R07 룰 정규식 검사의 ReDoS"),
    ("외부 룰 문구에 악의적인 지시가 들어 있을 때 어떤 통제가 있나?", "03_", "출력은 JSON 으로만 해석", "R09 룰 텍스트를 통한 프롬프트 인젝션"),
    ("하루치 학습 로그를 전체 검증했을 때 라벨 불일치는 몇 줄이었나?", "04_", "불일치 110줄", "라벨 정확성"),
    ("검토 모델이 안전 사유로 지운 비율은 검토한 공격 샘플 전체 대비 얼마였나?", "04_", "14.0%", "검토 모델의 안전 사유 삭제율"),
    ("요청 100개를 처리하는 시간은 동시 요청을 늘린 뒤 얼마나 줄었나?", "04_", "약 28분", "비용·속도"),
    ("다시 만들기에서 사람은 무엇을 결정하나?", "05_", "대체 모델로 다시 검토할지 결정", "5.1 감독 지점"),
    ("인공지능 기본법은 언제 제정되고 언제 시행되나?", "06_", "2026-01-22 시행", "6.1 근거 자료"),
    ("이 파이프라인은 고영향 AI 에 해당하나?", "06_", "고영향 AI 에 해당하지 않을 가능성이 높다", "6.2 이 파이프라인은 고영향 AI 인가?"),
    ("참고 예시를 마스킹할 때 Cookie 헤더는 어떻게 바꾸나?", "07_", "이름은 두고 값만", "7.3 바꾸는 값"),
    ("마스킹한 뒤 룰 조건을 더는 만족하지 않는 예시는 어떻게 하나?", "07_", "그 예시는 **보내지 않음**", "7.4 룰 보존 확인"),
]


def measure(chunks: list[dict], k: int = 3) -> dict:
    """질문마다 상위 k 개 안에 정답 조각이 드는지 센다.

    반환: {"total", "hits"(상위 k 안), "top1"(1위), "missed": [놓친 질문], "no_answer_chunk": [정답 조각이 문서에 없는 질문]}
    """
    index = search.BM25Index(chunks)

    def is_answer(chunk, prefix, phrase):
        return chunk["source"].startswith(prefix) and phrase in chunk["text"]

    result = {"total": len(QUESTIONS), "hits": 0, "top1": 0, "missed": [], "no_answer_chunk": []}
    for question, prefix, phrase, _section in QUESTIONS:
        if not any(is_answer(c, prefix, phrase) for c in chunks):
            result["no_answer_chunk"].append(question)
        found = [is_answer(hit, prefix, phrase) for hit in index.search(question, k)]
        if any(found):
            result["hits"] += 1
            result["top1"] += int(found[0])
        else:
            result["missed"].append(question)
    return result


if __name__ == "__main__":
    docs = documents.load_default_documents()
    if not docs:
        raise SystemExit(f"기본 문서 폴더가 없습니다: {documents.DEFAULT_DOCS_DIR}")
    print(f"문서 {len(docs)}개, 질문 {len(QUESTIONS)}개 (지금 쓰는 조각 크기 = {documents.MAX_CHARS}자)")
    print("조각 크기(자) | 조각 수 | 1위 | 상위 3개 안 | 상위 5개 안 | 정답 조각이 잘려 없는 질문")
    for max_chars in (300, 500, 800, 1000, 1500, 2500):
        chunks = documents.split_documents(docs, max_chars)
        top3, top5 = measure(chunks, 3), measure(chunks, 5)
        print(
            f"{max_chars:>12} | {len(chunks):>7} | {top3['top1']:>2}/{top3['total']} | "
            f"{top3['hits']:>2}/{top3['total']}       | {top5['hits']:>2}/{top5['total']}       | {len(top3['no_answer_chunk'])}"
        )
    for question in measure(documents.split_documents(docs), 3)["missed"]:
        print("놓친 질문(지금 설정, 상위 3개):", question)
