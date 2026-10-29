# 보안 정책 (Security Policy)

> English: Please report vulnerabilities privately via GitHub's **"Report a vulnerability"** button on the Security tab. Do not open a public issue.

## 취약점 알리기

- 이 레포의 **Security 탭 → "Report a vulnerability"** 버튼으로 비공개로 알려 주세요 (GitHub 비공개 취약점 신고).
- 공개 이슈·PR·댓글에는 취약점 내용이나 키 값을 적지 마세요.
- 알려 주실 때 있으면 좋은 것: 어느 파일·기능인지, 재현 방법, 예상되는 영향.
- 개인 프로젝트라 정해진 응답 기한은 없지만, 확인되면 수정 내용과 함께 신고 건에 답을 남깁니다.

## 지원 범위

| 대상 | 범위 |
|---|---|
| `LLM/Agent_office` (Agent Office 앱) | ✅ `main` 브랜치 최신 코드 |
| `LLM/train_set_making_by_rule(talos, et)` (학습 데이터 파이프라인) | ✅ `main` 브랜치 최신 코드 |
| `.github/` (CI·의존성 설정) | ✅ |
| 그 밖의 폴더 (Titanic, wine_qualityN_data, PredictStock, GAME, 분석 등 학습용 실험) | ❌ 유지보수하지 않음 — 단, **키·개인정보가 보이면 어느 폴더든 알려 주세요** |

## 비밀은 레포에 넣지 않는다

- API 키·토큰·비밀번호는 **`.env` 파일에만** 둡니다. 각 프로젝트의 `.env.example` 을 `.env` 로 복사해서 값을 채웁니다.
- 코드·노트북에 키를 직접 적지 않습니다. 노트북에서도 `os.environ["..."]` 처럼 환경 변수에서 읽습니다.
- 루트 `.gitignore` 가 어디에 있든 막는 것:
  - 비밀 설정: `.env`, `.env.*` (`.env.example` 만 예외), `env.zip`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`, `credentials*.json`, `service-account*.json`
  - 데이터·실행 결과: 모든 `data/`, `results/` 폴더 (Kaggle 공개 예제인 `Titanic/data/` 만 예외)
  - 개인 이미지: `LLM/Agent_office/teams/stock/avatars/`
  - 부산물: `__pycache__/`, `.venv/`, `.pytest_cache/`, `.ipynb_checkpoints/`
- 폴더별 `.gitignore`(Agent Office, 학습 데이터 파이프라인)는 그 폴더의 원본·중간 데이터와 로그를 추가로 막습니다.

## 자동으로 도는 검사

| 검사 | 언제 | 하는 일 |
|---|---|---|
| `secret-scan` (gitleaks) | push·PR 마다 | 새로 들어온 커밋에 키·토큰이 있으면 실패. 값은 로그에 가려서 보여 줌 |
| `secret-scan` 수동 실행 | Actions 탭에서 직접 | 전체 히스토리와 현재 파일 전체를 점검 |
| `agent-office-tests` | `main` 으로 가는 push·PR | Agent Office 테스트 전체 (API 키·네트워크 없이 실행) |
| Dependabot | 매주 월요일 | 파이썬 패키지·GitHub Actions 보안 업데이트 PR (자동 병합 안 함) |

## 키가 새어 나갔을 때

1. **먼저 그 키를 발급한 곳에서 폐기(revoke)하고 새 키를 받습니다.** 공개 레포에 한 번 올라간 키는 지워도 이미 복사됐다고 봅니다.
2. 현재 파일에서 키를 지우고 환경 변수로 바꿉니다.
3. 히스토리 다시 쓰기(force push)는 기본으로 하지 않습니다. 키를 폐기하면 남은 기록은 쓸모가 없어지기 때문입니다.
4. 폐기를 마친 키만 `.gitleaksignore` 에 gitleaks 지문(fingerprint)과 폐기 날짜를 적어 정기 점검에서 빼 줍니다.

## 커밋 전 자동 검사 (선택)

내 PC 에서 커밋하기 전에 키·개인 키·큰 파일을 미리 막고 싶다면 [pre-commit](https://pre-commit.com/) 을 씁니다. 한 번만 `pip install pre-commit` 후 레포 루트에서 `pre-commit install` 을 실행하면, 이후 `git commit` 때마다 `.pre-commit-config.yaml` 의 검사(gitleaks, 개인 키 탐지, 1MB 넘는 새 파일 차단, 병합 충돌 표시 확인)가 돌고, 걸리면 커밋이 멈춥니다. 전체 파일을 한 번 점검하려면 `pre-commit run --all-files`, 잠시 끄려면 `pre-commit uninstall` 을 실행합니다. CI 의 `secret-scan` 과 같은 gitleaks 버전·설정을 쓰므로, 로컬에서 통과하면 CI 에서도 대체로 통과합니다.
