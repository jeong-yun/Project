# 2. NIST AI RMF 대응표

> 기준 문서
> - NIST AI 100-1, *AI Risk Management Framework (AI RMF 1.0)*, 2023-01: https://doi.org/10.6028/NIST.AI.100-1
> - NIST AI 600-1, *AI RMF: Generative AI Profile*, 2024-07-26: https://doi.org/10.6028/NIST.AI.600-1 (소개 페이지: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
>
> AI RMF 의 네 기능(Govern·Map·Measure·Manage) 과 그 아래 **카테고리 번호**(예: GOVERN 6)까지만 적었다. 카테고리 이름은 작성자가 줄여 옮긴 것이며, 세부 번호(GOVERN 6.1 등)는 원문 대조가 필요하다 (확인 필요).
> '빈틈' 열은 지금 없는 것이다. 제안은 [README 의 다음에 할 일](README.md#다음에-할-일).

## 2.1 GOVERN — 정책·책임·제3자

| 카테고리 | 이 파이프라인의 장치 (파일·함수·기록) | 빈틈 |
| --- | --- | --- |
| GOVERN 1 정책·절차가 정해져 있음 | 라벨 정의(룰 기준)를 문서로 고정: [`진행_설계_요약.md`](../보고용/진행_설계_요약.md) 3장, `LLM_설정.md` `로컬 검사 실패 처리`·`판정 불가 처리`; 생성 안전 규칙: [`LLM_프롬프트.md`](../코드참조/LLM_프롬프트.md) 생성 규칙 5 (문서용 IP·`example.com` 만, 실제 악성코드·C&C·파괴 명령 금지); 선별 기준: [`선별조건.md`](../코드참조/선별조건.md) | 외부 LLM 에 보내도 되는 데이터의 기준 문서가 없음 (R05) |
| GOVERN 2 책임과 역할 | 실행·모델 설정·다시 만들기·버전 확정이 모두 사람이 누르는 버튼 (화면 2·4·6) | 역할(운영자·데이터 책임자) 구분과 승인자 기록이 없음 (R12) |
| GOVERN 4 위험을 알리는 문화·소통 | 화면 확인 창에 비용·구독 사용량 안내 (`0_0_run_ui.py` 3361·3429행), 진행 문서의 '배운 점' | — |
| GOVERN 6 제3자(공급자) 위험 | provider 표로 공급자를 한곳에서 관리 (`LLM_설정.md` `## provider`); 크레딧 없음·로그인·구독 한도 감지 후 정지 (`3_0a_common.FATAL_MARKS`); Claude CLI 를 도구·MCP·훅 없이 격리 실행 (`_call_claude_cli`) | 공급자별 데이터 보관·학습 사용 조건 기록 없음 |

## 2.2 MAP — 맥락·분류·위험 파악

| 카테고리 | 장치 | 빈틈 |
| --- | --- | --- |
| MAP 1 맥락(목적·사용자·쓰임) | 목적: 룰을 정답지로 IPS/WAF 탐지 모델 학습데이터 생성 ([README](../../README.md), 요약 1장); 쓰임: 내부 학습데이터, 확정 버전만 학습에 사용 | — |
| MAP 2 시스템 분류 | [1. AI 시스템 목록](01_AI_시스템_목록.md) (역할별 모델·입출력) | — |
| MAP 3 능력·이득·비용 | 모델 비교 기록 (`LLM_설정.md` 17~22행: 비용·남은 샘플·고쳐 살린 비율) | 비교가 일회성, 고정 세트 없음 |
| MAP 4 구성요소·제3자 위험 | 외부 입력(룰 피드)·외부 LLM 경계를 데이터 흐름도로 표시 ([1.3](01_AI_시스템_목록.md#13-데이터-흐름)); 룰 받을 때 버전 비교·sha256 기록 (`1_1_fetch_rules.py`) | 참고 예시의 전송 전 마스킹 단계 없음 (R05, [7장 제안](07_마스킹_제안.md)) |
| MAP 5 영향 | [6. AI 기본법 체크리스트](06_AI기본법_고영향_체크리스트.md) 의 고영향 판단, 위험 등록부 영향 열 | 다운스트림 탐지 모델 배포처별 영향은 이 문서 범위 밖 |

## 2.3 MEASURE — 측정·평가

| 카테고리 | 장치 | 빈틈 |
| --- | --- | --- |
| MEASURE 1 방법·지표를 정함 | 라벨 정합성 4단계(정합·정합(룰 기준)·확인 필요·불일치) — README '데이터 프레임의 열'; 등급 A·B·C 기준 — `5_7_versions.py`, README | 공격 유형별 삭제율 같은 편향 지표가 정의만 있고 자동 기록 없음 |
| MEASURE 2 신뢰성 특성 평가 | 유효성: 코드 검사 `local_check`, 화면 4 전체 검증 (190,226줄 → 불일치 0.06% → 0); 편향: 공격 유형별 '검토에서 뺀 샘플' 표; 보안: 정규식 시간 제한; 평가 오염 방지: rule_id 지문 30% holdout | 실제 Snort 로 탐지 확인 없음 (근사 검사) |
| MEASURE 3 시간에 따른 위험 추적 | `meta.json` (요청마다 모델·토큰·샘플·수정·제외 수), `rule_status.csv` 상태 플래그, 진행 현황 (`3_5_progress.py`), 검증 결과 `<장비>_verify.json` | 실행 사이 추세(유형별 삭제율·거절률)를 모으는 표 없음 |
| MEASURE 4 효과에 대한 피드백 | 사람 검토 의견을 지시문·코드에 반영한 기록 (요약 5장 '검토 의견' 행: 헤더 현실화) | 피드백 접수 기록 형식 없음 |

## 2.4 MANAGE — 처리·대응·복구

| 카테고리 | 장치 | 빈틈 |
| --- | --- | --- |
| MANAGE 1 위험 우선순위·대응 결정 | [3. 위험 등록부](03_위험_등록부.md) | — (이 문서로 처음 생김) |
| MANAGE 2 이득 극대화·피해 최소화 | 거절 룰 대체 검토 (`3_3_review.py --redo-safety`), 증강은 코드만(LLM 비용 없음), 기존 데이터에 충분한 룰은 생성 안 함 (`기존 데이터 충분 기준` 1000) | — |
| MANAGE 3 제3자 위험 관리 | 모델 교체 가능 구조, `mock` 시험, 크레딧 소진 시 정지 후 이어서 | 생성 모델 대체 경로 시험 기록 없음 |
| MANAGE 4 대응·복구·소통 | 멈춰도 이어서 (끝난 요청 건너뜀), 옛 결과 `_replaced/` 보관, 사람 수정은 `edits.json` 덧씌우기라 되돌림, 버전 확정은 비교 스냅숏이 같을 때만 | 확정 버전을 되돌리는 절차 없음 (README: 화면에서 되돌리지 않음) |

## 2.5 NIST AI 600-1 생성형 AI 위험 범주와의 연결

AI 600-1 은 생성형 AI 에 고유하거나 더 커지는 위험 12가지를 든다. 이 파이프라인에 해당하는 것만 골랐다.

| 600-1 위험 범주 | 이 파이프라인에서의 모습 | 위험 ID |
| --- | --- | --- |
| Confabulation (그럴듯한 오류) | 생성 모델이 정규식을 잘못 읽고 룰과 안 맞는 샘플을 만듦 | R02 |
| Harmful Bias or Homogenization | 안전 거부로 위험 유형 공격 샘플이 빠짐, 문서용 주소·인코딩 쏠림 | R01, R10 |
| Data Privacy | 참고 예시에 섞일 수 있는 민감값(Host·Cookie 등)이 LLM 제공자에게 전달될 가능성 | R05 |
| Information Security | 키 유출, 프롬프트 인젝션, ReDoS | R06, R07, R09 |
| Information Integrity | 라벨·로그 표기 어긋남, 설정·기록 불일치 | R02, R03, R11 |
| Human-AI Configuration | 사람이 샘플 단위가 아니라 실행·결과 단위로만 개입, 검토 모델 판단에 기대는 정도 | R01, R12 |
| Value Chain and Component Integration | 외부 룰 피드·외부 LLM·CLI 의존, 모델 변경 | R05, R08 |
| Intellectual Property | 룰 라이선스(Talos 재배포 제한) | R13 |
| CBRN Information or Capabilities / Dangerous, Violent, or Hateful Content | 해당 거의 없음. 공격 페이로드는 생성하지만 실제 악성코드·C&C·파괴 명령은 지시문으로 금지 | (R01 의 반대편: 안전 규칙) |
| Obscene, Degrading, and/or Abusive Content / Environmental Impacts | 해당 없음으로 판단 (환경 영향은 토큰 사용량으로 간접 관찰 가능) | — |

> 600-1 의 범주 이름은 원문 영어 그대로 두었다. 각 범주 정의는 원문 2장을 본다.
