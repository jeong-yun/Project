# 프로젝트 진행 기록: LLM 에이전트 기반 게임 IP 리뷰 감성 분석 파이프라인

## 1. 데이터 준비 (Data Preparation)

* **데이터 수집:** Kaggle의 오픈 데이터셋 활용 (`Steam Reviews Dataset`)
* **데이터 전처리 (Python):**
  * 리뷰 텍스트 길이 20자 이하 데이터 필터링 수행.
  * "yes.", "no" 등 문맥이 부족하여 LLM 감성 분석이 불가능한 무의미한 노이즈 문자열을 사전 제거하여 분석 품질 및 API 비용 효율성 확보.

## 2. 데이터 베이스 구축 및 적재 (Infrastructure & DB Setup)

대용량 비정형 데이터의 빠른 집계와 처리를 위해 Docker 환경에 ClickHouse OLAP 데이터베이스 구축.

* **Docker 설치:** 운영체제 아키텍처(x64 AMD64 / ARM64)에 맞는 Docker Desktop 설치 완료.
* **ClickHouse 서버 컨테이너 실행:**

```bash
docker run -d --name clickhouse-server -p 8123:8123 -p 9000:9000 -e CLICKHOUSE_PASSWORD=1234 --ulimit nofile=262144:262144 clickhouse/clickhouse-server
```

* **트러블슈팅(Docker Desktop is unable to start 에러 해결):**
  * 원인: Windows WSL2 시스템 충돌.
  * 해결: 터미널에서 wsl --update 및 wsl --shutdown 명령어로 WSL 시스템 초기화 후 Docker 서비스 재시작하여 정상 구동 확인.
