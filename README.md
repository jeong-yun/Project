# Project

개인 프로젝트 모음. 요약·실측 수치·진행 기록은 [PROJECTS.md](PROJECTS.md), 보안 정책은 [SECURITY.md](SECURITY.md).

## LLM
- [Agent_office](LLM/Agent_office/README.md) — 직급형 멀티 에이전트 협업 앱 (LangGraph · Streamlit). 부서 7개 · 팀 12개, 보안 거버넌스 · 인프라 보안 · 자율 실행 · 교육
- [train_set_making_by_rule(talos, et)](<LLM/train_set_making_by_rule(talos, et)/README.md>) — Snort 룰 기반 IPS/WAF 학습데이터 생성 파이프라인 (공개본, 코드만) + AI 거버넌스 문서
- [AI_Agent](LLM/AI_Agent/README.md) — LangChain · Streamlit 작은 LLM 앱: 채팅 · 영어 단어장 · AI 안전성 평가 · 문서 Q&A(RAG)
- LLM 에이전트 기반 게임 IP 리뷰 감성 분석 파이프라인

## 그 밖
- [GAME/Soccer](GAME/Soccer/READ_ME.md) — 축구 게임 (선수 선택·정보·경기 중 교체, 포메이션, AI 전술)
- Arcus — 임베딩 실험
- 분석 — 보안 로그 분석 스크립트 (XSS · SQL injection · 정규식)
- PredictStock · Titanic · wine_qualityN_data — 데이터 분석·예측 연습
- Contect

## 지금 진행 중 (2026-11-05)

| 프로젝트 | 상태 | 내용 |
|---|---|---|
| Agent Office | ✅ main 반영 | 🎓 교육 부서(선생님 생성 · 수준 퀴즈 코드 채점 · 레벨업, 직무 학습 · 취미·교양 학습 팀), 부서·역할별 ⭐ 추천 모델, 진행 % · 남은 시간, 결정 전 추가 질문 백그라운드 실행, 문서 작성 팀 원본 형식 · 그림 유지 — 자동 테스트 926건 |
| 저장소 공통 | 🔧 작업 중 | **올리기 전 검사** — push 직전 비밀 값 검사(gitleaks) · 바뀐 부분 테스트(두 PC 공통, Docker 없이) + `check/main` 검증 브랜치(GitHub 에서 전체 검사 통과 시에만 main 반영) |
| 저장소 공통 | ✅ main 반영 | **AI 코딩 원칙** — 루트 [`AGENTS.md`](AGENTS.md) 하나를 저장소 안 모든 하위 프로젝트의 공통 지침으로(단순성 우선 · 측정 기반 · 작게 검증 · 사용자 중심), 하위 프로젝트는 자기 폴더에 짧은 `AGENTS.md` 만 추가. Claude Code · Codex · Gemini CLI · Cursor 가 같은 파일을 읽음, 사람용 [검토 프롬프트](docs/AI_검토_프롬프트.md) |
| train_set_making_by_rule | 진행 | 생성 마무리 · 코드 검사 보완 (원본에서 만들어 공개본으로 내보냄) |

- **측정 예정:** 교육 팀 · 추천 모델의 실제 모델 실행 시간 · 비용 (지금 수치는 가짜 모델 테스트와 가정치 기준)
- 실측 숫자와 날짜별 기록은 [PROJECTS.md](PROJECTS.md) · [Agent Office 작업 기록](LLM/Agent_office/today_worked.md)
