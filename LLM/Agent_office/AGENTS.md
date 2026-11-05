# AGENTS.md — Agent Office 전용 규칙

저장소 루트 [AGENTS.md](../../AGENTS.md) 의 공통 원칙을 먼저 따르고, 이 폴더에서는 아래를 더한다.

## 실행·테스트

- 테스트: 이 폴더에서 `python -m pytest -q`. 바꾼 부분의 테스트 파일부터 돌리고, 끝내기 전에 전체를 돌린다.
- 가짜 모델: `tests/conftest.py` 가 실제 웹 검색·LLM 호출을 막고, `tests/fakes.py` 의 가짜 모델을 쓴다. 새 테스트도 이것을 재사용한다.
- 사용자가 앱을 실행하는 폴더 `Project_run` 은 건드리지 않는다.

## CI 와 다른 점 조심

CI(`agent-office-tests`)는 리눅스·Python 3.11 에서 의존성을 새로 설치해 돈다.
- 패키지 버전 차이: 내 PC 에서만 되는 동작에 기대지 않는다.
- 부동소수점: 정확히 같음 대신 근사 비교(`pytest.approx`).
- Streamlit 버전별 표시 차이: 예를 들어 새 버전은 알림 앞 이모지를 아이콘으로 떼어 낸다. 화면 문자열 전체를 비교하지 않는다.
- Windows 경로·한글 파일 이름.

## 이 앱에서 이미 쓰는 방식 (새로 만들기 전에 재사용)

- 결과 검증은 코드가 한다: 하네스 `core/harness/`(검증기 등록 방법은 `base.py` 맨 위), 수정 diff 재검사 `core/iac_patch.py`, 채택 뒤 `calc_report` 로 다시 검산.
- 속도: Streamlit 은 클릭마다 `app.py` 전체를 다시 실행한다. 다시 그리기마다 도는 코드를 조심하고, 전후 측정은 `scripts/bench_ui.py` 로 한다. 파일 지문 캐시는 `core/memo.py`.
- 기록: 끝낸 작업은 `today_worked.md` 에 날짜별로 **작업 · 기술 · 성과 · 다음 할 일** 형식으로 적는다.
