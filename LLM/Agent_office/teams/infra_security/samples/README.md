# 샘플상사 (가상 회사) IaC

🏗️ 인프라 보안 팀 시험용 **가상** 인프라 코드다. 실제 회사·계정·사람·IP 와 관계없다.

- 이름 '샘플상사', 버킷·DB 이름(samplecorp-*), 레지스트리·내려받기 주소(*.example.com)는 지어낸 것이다.
- IP 는 문서 전용 대역(RFC 5737: 192.0.2.0/24 · 198.51.100.0/24 · 203.0.113.0/24)만 쓴다. `0.0.0.0/0` 은 '인터넷 전체'라는 뜻의 표기다.
- 비밀 값 자리는 `PLACEHOLDER` 로만 채웠다 (진짜 키·비밀번호 없음).

| 폴더 | 내용 |
| --- | --- |
| `sample_corp/terraform` | AWS 모양 Terraform: S3 버킷, 보안 그룹, IAM 정책, RDS |
| `sample_corp/k8s` | 쿠버네티스 매니페스트: 웹 Deployment, 야간 보고서 CronJob |
| `sample_corp/docker` | 웹 이미지 Dockerfile |
| `ground_truth.yaml` | 일부러 넣은 잘못된 설정 **17개** 정답지 (스캔 폴더 밖 — 모델에게 주지 않는다) |

내장 규칙(`config/iac_rules.yaml`, `core/iac.py`)이 정답지 17개를 빠짐없이 찾고 그 밖에는 찾지 않는지
`tests/test_infra_security.py` 가 센다.
