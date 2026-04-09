# 프로젝트 진행 기록: LLM 에이전트 기반 게임 IP 리뷰 감성 분석 파이프라인

## 🏗 아키텍처 및 데이터 파이프라인(Architecture)
1. **데이터 전처리(Data Preprocessing):** - 20자 이하의 무의미한 리뷰(노이즈) 필터링 (Pandas)
2. **데이터 적재(Data Warehouse):** - Docker 환경에 ClickHouse OLAP 데이터베이스 구축 및 대규모 텍스트 데이터 적재
3. **LLM 감성 추출(LLM Extraction):** - LangChain & Pydantic을 활용하여 리뷰 내 4가지 핵심 요소(게임플레이, 그래픽, 최적화, 과금)에 대한 감성 스코어 및 핵심 키워드를 JSON 형태로 파싱
4. **피처 엔지니어링(Feature Engineering):** - 게임별 요소 평균 스코어 산출
   - **이상 탐지(Anomaly Detection) 기법 적용:** 점수의 분산(Variance)을 계산하여 단기간의 리뷰 폭격(Review Bombing) 및 최적화/과금 논란 수치화
   - 핵심 키워드 One-Hot Encoding

## 💡 핵심 구현 및 어필 포인트
* **복합 감성 추출:** 하나의 리뷰에 칭찬("이 게임은 훌륭하다")과 비판("하지만 커뮤니티 수준은 최악이다")이 혼재되어 있을 경우, 이를 분리하여 요소별로 정확히 수치화(-1.0 ~ 1.0)하는 프롬프트 엔지니어링 구현.
* **대용량 처리 최적화:** API Rate Limit 및 예외 상황에 대비하여 `try-except` 및 배치(Batch) 처리 적용. ClickHouse의 `MergeTree` 엔진을 활용하여 파티셔닝 기반의 쿼리 성능 향상.
* **보안/이상 탐지 관점의 피처 결합:** 보안 로그 탐지 원리를 응용하여, 감성 스코어의 '분산(Variance)'을 파생 변수로 생성해 유저들의 평가 엇갈림(논란)을 머신러닝 피처로 포착.


## 1. 데이터 준비(Data Preparation)
* **데이터 수집:** Kaggle의 오픈 데이터셋 활용 (`Steam Reviews Dataset`)
* **데이터 전처리 (Python):**
  * 리뷰 텍스트 길이 20자 이하 데이터 필터링 수행.
  * "yes.", "no" 등 문맥이 부족하여 LLM 감성 분석이 불가능한 무의미한 노이즈 문자열을 사전 제거하여 분석 품질 및 API 비용 효율성 확보.

## 2. 데이터 베이스 구축 및 적재(Infrastructure & DB Setup)
대용량 비정형 데이터의 빠른 집계와 처리를 위해 Docker 환경에 ClickHouse OLAP 데이터베이스 구축.
* **Docker 설치:** 운영체제 아키텍처(x64 AMD64 / ARM64)에 맞는 Docker Desktop 설치 완료.
* **Docker 목록 조회:** docker ps -a
* **Docker 서버 실행:** docker start clickhouse-server
* **ClickHouse 서버 컨테이너 실행:**
```bash
docker run -d --name clickhouse-server -p 8123:8123 -p 9000:9000 -e CLICKHOUSE_PASSWORD=1234 --ulimit nofile=262144:262144 clickhouse/clickhouse-server
```
* **트러블슈팅(Docker Desktop is unable to start 에러 해결):**
  * 원인: Windows WSL2 시스템 충돌.
  * 해결: 터미널에서 wsl --update 및 wsl --shutdown 명령어로 WSL 시스템 초기화 후 Docker 서비스 재시작하여 정상 구동 확인.

## 3. 초기 데이터 적재(Data Loading)
 - 테이블 생성 (DDL): 시계열 및 추천 여부 기반의 빠른 검색을 위해 MergeTree 엔진과 파티셔닝 적용.
 ```SQL
CREATE TABLE steam_reviews (
    date_posted Date,
    funny Int32,
    helpful Int32,
    hour_played Int32,
    is_early_access_review UInt8,
    recommendation String,
    review String,
    title String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date_posted)
ORDER BY (title, date_posted, recommendation);
```
 -  데이터 적재 및 확인: Python clickhouse-connect 및 CLI 도구를 활용하여 전처리된 CSV 데이터를 테이블에 성공적으로 적재 및 건수 검증 완료.

## 4. LLM 추출 스키마 설계(Schema Design) 및 LLM 활용 감성 분석 파이프라인(LLM Analysis Pipeline)
LLM의 자유로운 출력을 정형화된 데이터베이스에 적재하기 위한 규격화 작업.
 - Python (Pydantic): gameplay, graphics, optimization, monetization 등 게임 핵심 요소별 점수(Score)와 핵심 키워드(Keywords)를 추출하기 위한 클래스 구조체 정의.
 - SQL (ClickHouse): 감성 분석 결과값 전용으로 저장할 타겟 테이블(steam_reviews_sentiment) 스키마 설계 완료.
Python 스크립트와 ClickHouse 서버를 연동하여 텍스트 리뷰를 구조화된 감성 스코어로 변환하는 자동화 파이프라인 구축.

## 5. 피처 엔지니어릴(Feature Engineering)
1. 게임별/기간별 데이터 집계 및 파생 변수 생성(Feature Engineering): '해당 게임의 전체적인 평가 지표'를 보고 흥행 예측, 리뷰 단위의 데이터를 게임 단위(또는 월별 단위)로 묶어서(Group By) 특징(Feature) 생성
  - 요소별 평균 점수 산출: 각 게임의 gameplay_score, optimization_score 등의 평균값을 계산합니다. Null 값은 계산에서 제외(Drop 또는 Ignore)하여 실제 언급된 리뷰들의 평균만 취합.
  - 이상 징후(Anomaly) 피처 추가: 보안 통제 시스템에서 네트워크 트래픽의 비정상적인 스파이크나 이상 징후를 탐지할 때 변동성을 중요한 피처로 잡듯, 특정 기간 내 최적화나 과금 스코어의 분산(Variance)이 급격히 커지는 현상을 '리뷰 폭격(Review Bombing)'이라는 파생 변수로 생성해 모델에 주입
 - 키워드 빈도 추출: core_keywords 배열에서 가장 빈도수가 높은 단어들을 추출해 One-Hot Encoding 형태로 변환하거나 카테고리화.
2. 
3. 
