# AGENTS.md — AI 코딩 도구 공통 원칙

이 저장소에서 코드를 쓰는 모든 AI 도구(Claude Code, Codex, Gemini CLI, Copilot 등)가 따르는 **원천 파일**이다.
도구별 파일(`CLAUDE.md`, `.gemini/settings.json`, `.github/copilot-instructions.md`)은 이 파일을 가리키기만 한다. 규칙을 바꿀 때는 여기만 고친다.
원칙은 Addy Osmani 의 글에서 영감을 받아 이 저장소에 맞게 다듬었다.

## 1. 손대기 전 5가지 질문

1. 진짜 문제는 무엇이고, 더 단순한 해결은 없는가?
2. 지금 뺄 수 있는 20% 는 무엇인가?
3. 최악의 경우 속도·메모리·규모는? (예: Streamlit 다시 그리기마다 도는 코드, 큰 YAML·DB, 모델 호출 수)
4. 한 달 뒤 다른 사람이 1분 안에 이해하는가?
5. 지금 아무도 안 쓰는 '미래용' 옵션·추상화를 넣지 않았는가?

## 2. 규칙

**단순하게 — 더 뺄 게 없을 때 완성이다**
- 새 파일·추상화·설정 옵션은 지금 쓰는 곳이 2곳 이상일 때만 만든다.
- 먼저 기존 함수·모듈을 찾아 재사용한다 (예: Agent Office 의 `core/`, 가짜 모델 `tests/fakes.py`).
- 요청 범위 밖의 정리·리팩터링은 하지 않고, 따로 제안만 한다.

**측정한다 — 추측하지 않는다**
- 성능 주장은 측정값과 함께 쓴다. 같은 PC·같은 데이터 복사본에서 전→후 중앙값과 측정 조건(CPU 부하 등)을 적는다. 본보기: `LLM/Agent_office/today_worked.md` 의 '화면 전환 속도' 기록, `scripts/bench_ui.py`.
- 재 보지 않은 숫자는 '추정' 또는 '아직 실측 안 함'으로 쓴다.
- LLM 이 낸 숫자·코드·diff 는 코드가 다시 검산·실행해 확인한다 (Agent Office 하네스 `core/harness/`, 수정안 재검사 `core/iac_patch.py`). LLM 에게 다시 묻는 것으로 검증을 대신하지 않는다.

**작게, 빨리 확인한다**
- 한 번에 한 가지를 바꾸고 관련 테스트를 바로 돌린다. 버그는 실패하는 테스트부터 쓴다.
- 끝내기 전에 바꾼 폴더의 테스트 전체를 돌린다.

**사용자가 느끼는 가치가 먼저다**
- 화면 문구·오류 메시지는 쉬운 한국어로 쓰고, 사용자가 다음에 무엇을 하면 되는지 적는다. 내부 용어·영어 약어는 풀어 쓴다.
- 내부 구조의 깔끔함보다 사용자가 느끼는 차이(빨라짐, 실패가 줄어듦, 덜 헷갈림)가 먼저다.

## 3. 이 저장소에서 꼭 지킬 것

- **공개 저장소다.** 비밀 값·회사 자료·개인 문서·`.env`·`env.zip`·`data/`·`results/`·`LLM/Agent_office/teams/stock/avatars/` 를 커밋하지 않는다. 올리기 전 검사 안내는 [SECURITY.md](SECURITY.md).
- **테스트는 가짜 모델만 쓴다.** `tests/conftest.py` 가 실제 호출을 막는다. `.env` 의 실제 키로 진짜 모델을 부르지 않는다. 실측이 꼭 필요하면 사용자에게 먼저 묻는다.
- **CI 는 리눅스(Python 3.11)에서 의존성을 새로 설치해 돈다.** 패키지 버전 차이, 부동소수점 차이(정확히 같음 대신 근사 비교), Streamlit 버전별 표시 차이(예: 알림 앞 이모지가 아이콘으로 떨어짐), Windows 경로·한글 파일 이름을 조심한다.
- 사용자가 앱을 실행하는 폴더 `Project_run` 은 건드리지 않는다. 작업은 따로 만든 worktree·브랜치에서 한다.
- commit·push 는 사용자가 요청할 때만 한다.
- 답변·보고는 쉬운 한국어로 하고, 확인한 것과 확인하지 못한 것을 나눠 적는다.

## 4. 프로젝트 지도

| 폴더 | 내용 |
|---|---|
| `LLM/Agent_office` | 주력. Streamlit + LangGraph 멀티 에이전트 앱. 테스트: 이 폴더에서 `python -m pytest -q` |
| `LLM/train_set_making_by_rule(talos, et)` | Snort 룰 기반 IPS/WAF 학습 데이터 파이프라인 (코드만 공개, `tests/` 있음) |
| `LLM/AI_Agent`, `LLM/LLM 에이전트 기반 게임 IP …` | 작은 LLM 실험 |
| `Arcus`, `분석`, `PredictStock`, `Titanic`, `wine_qualityN_data`, `GAME`, `Contect` | 학습용 실험 (유지보수 안 함) |

문서: [README.md](README.md)(목록) · [PROJECTS.md](PROJECTS.md)(프로젝트 요약·진행) · [SECURITY.md](SECURITY.md)(비밀·자동 검사) · `LLM/Agent_office/today_worked.md`(날짜별 기록: 작업·기술·성과·다음 할 일) · `LLM/Agent_office/README.md`·`PLAN.md`.

## 5. 검토

끝내기 전에 1번 질문으로 내 변경을 다시 읽고 줄인다. 사람이 아무 모델에게 검토를 맡길 때 쓰는 프롬프트: [docs/AI_검토_프롬프트.md](docs/AI_검토_프롬프트.md).
