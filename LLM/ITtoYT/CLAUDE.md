# ITtoYT

IT 개념을 웹 검증한 뒤 NotebookLM(Gemini Notebook)으로 short(60초) / 일반 버전 영상 소재를 만드는 프로젝트. **한국어 우선.**

## 작업 방식
- 개념 하나를 받으면 `ittoyt` 스킬로 시작한다. 단계별로는 `it-verify` → `nlm-pack`.
- 산출물은 `IT 개념/<slug>/` 에 `00-` / `10-` / `20-` / `30-` 접두사 규칙으로 저장한다. 이후 자동화가 이 규칙에 의존하므로 임의로 바꾸지 않는다.
- 파이프라인 전체 설계와 현재 자동화 경계는 [docs/pipeline.md](docs/pipeline.md).

## 반드시 알아야 할 제약
- **NotebookLM에 Video Overview 생성 API가 없다.** 영상 생성은 수동 클릭 단계다(1단계 = 반자동).
- **Short·Cinematic 포맷은 영어 내러이션만 지원한다.** 한국어 short은 별도 경로가 필요하다.
- 세부 근거: [.claude/skills/nlm-pack/references/notebooklm-settings.md](.claude/skills/nlm-pack/references/notebooklm-settings.md)

## 원칙
검증되지 않은 내용은 소스팩에 넣지 않는다. 기억에서 나온 그럴듯한 설명이 이 프로젝트의 가장 큰 리스크다.
