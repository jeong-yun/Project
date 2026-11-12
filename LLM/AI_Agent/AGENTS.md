# AGENTS.md — AI_Agent 전용 규칙

저장소 루트 [AGENTS.md](../../AGENTS.md) 의 공통 원칙을 먼저 따르고, 이 폴더에서는 아래를 더한다.

## 실행·테스트

- 실행: 이 폴더에서 `streamlit run app.py` (`.env` 에 `OPENAI_API_KEY` 필요 — `.env` 는 커밋하지 않는다).
- 테스트: 이 폴더에서 `python -m pytest -q`. 바꾼 부분의 테스트 파일부터 돌리고, 끝내기 전에 전체를 돌린다.
- **테스트는 실제 모델·임베딩·발음(TTS)을 부르지 않는다.** `tests/conftest.py` 가 키를 지우고 바깥 요청을 막는다. 모델 자리에는 `tests/fakes.py` 의 가짜 모델이나 `RunnableLambda` 를 끼운다. 실제 호출이 꼭 필요하면 사용자에게 먼저 묻는다.
- 화면 테스트는 `streamlit.testing.v1.AppTest` 로 한다. 모든 화면은 키 없이 열려야 한다.

## 이 앱에서 이미 쓰는 방식 (새로 만들기 전에 재사용)

- 기능은 모듈 패키지(`english_word_maker/`, `ai_safety/`)에, 화면은 `pages/` 에 둔다.
- 폴더 경로·모델 만들기는 `common.py` 를 쓴다 (`BASE_DIR`, `PROMPTS_DIR`, `make_llm`). 화면에서 `ChatOpenAI` 를 직접 만들지 않는다.
- 밖에서 온 글(모델 응답, 문서 조각)을 프롬프트에 넣을 때는 `common.as_data` 를 거쳐 표시선(`DATA_OPEN`~`DATA_CLOSE`) 안에 넣는다.
- 모델이 낸 판정·숫자는 코드가 다시 확인한다 (예: `ai_safety/judge.py` 의 `_checked_verdict`). 모델에게 다시 물어 검증하지 않는다.
