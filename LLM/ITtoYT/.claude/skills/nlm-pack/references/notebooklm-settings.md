# NotebookLM (Gemini Notebook) 설정·제약 레퍼런스

조사 기준일: 2026-08-20. NotebookLM은 기능 변경이 잦다 — 6개월 이상 지났으면 아래 항목을 재확인한다.
제품명이 **Gemini Notebook** 으로 변경되었다(구 NotebookLM). UI 문구가 다를 수 있다.

## Video Overview 포맷 3종

| 포맷 | 화면비 | 길이 | 내러이션 언어 | 비주얼 스타일 선택 |
|---|---|---|---|---|
| **Short** | 9:16 세로 | 약 60초 | **영어만** (18세 이상) | 없음 |
| **Explainer** | 16:9 | 구조적 개요 (수분) | 지원 언어 다수 (**한국어 가능**) | 있음 |
| **Cinematic** | 16:9 | 몰입형 | **영어만** (18세 이상) | 없음 |

Explainer 비주얼 스타일 프리셋: Classic, Whiteboard, Watercolor, Retro Print, Heritage, Paper-craft, Kawaii, Anime + 자동선택/커스텀.
→ **IT 개념 설명에는 Whiteboard 또는 Classic 을 권장한다.** 다이어그램·화살표 표현이 개념 설명과 맞물리고, Anime/Kawaii 는 코드·구조도를 뭉갠다.

공통 옵션: **steering prompt**(집중할 주제를 직접 입력 가능), 언어 선택.
Short 포맷은 2026-06-30 롤아웃, AI Ultra/Pro 대상(무료 플랜 추후).

## 파이프라인에 직결되는 제약

### 1. 한국어 short 은 NotebookLM으로 만들 수 없다
Short·Cinematic 은 영어 내러이션만 지원한다. 한국어 채널이라면 short 은 아래 중 하나를 골라야 한다.

| 경로 | 방법 | 장점 | 단점 |
|---|---|---|---|
| **A. 자체 제작** (권장) | `20-source-short.md` 나레이션 원고 → TTS + 편집 | 60초·9:16·한국어를 정확히 통제 | 편집 도구가 별도로 필요 |
| **B. Explainer 리프레임** | 한국어 Explainer 생성 → ffmpeg로 9:16 크롭 + 구간 잘라내기 | NotebookLM 한 번으로 두 버전 | 세로 구도가 어색하고 60초 압축이 어렵다 |
| **C. 영어 Short + 자막** | 영어 Short 생성 → 한국어 자막 | 제작이 가장 빠름 | 한국어 채널에는 부적합 |

`meta.json` 의 `short.route` 에 `A`/`B`/`C` 를 기록한다.

### 2. 영상 생성 API는 없다
- 소비자용 공개 API 없음.
- Enterprise(Gemini Notebook Enterprise)에는 `notebooks.create`, `notebooks.sources.batchCreate`, `notebooks.sources.uploadFile` 가 있으나 **Video Overview 생성 메서드는 없다.**
- Podcast(오디오) API는 존재하지만 **deprecated 이며 신규 고객 allowlist를 하지 않는다.**
→ 결론: 영상 생성 단계는 사람이 UI에서 클릭하거나 브라우저 자동화뿐이다. 이 스킬은 클릭 직전까지를 완성하는 것을 목표로 한다.

### 3. 소스는 여러 개로 나누는 게 낫다
팩트시트와 소스팩을 **각각 별도 소스로 업로드**한다. NotebookLM은 소스 단위로 인용하므로, 대본형 문서(소스팩)와 근거 문서(팩트시트)를 분리하면 대본 구조는 소스팩에서 가져오고 사실 확인은 팩트시트에서 하게 된다.
단, short 노트북에는 `20-source-short.md` **하나만** 넣는다. 소스가 많으면 60초 안에 곁가지가 끼어든다.

## UI 조작 순서 (일반 버전)

1. 새 노트북 생성 → 제목: `<개념명> (long)`
2. 소스 추가 → `10-source-long.md`, `00-research.md` 업로드
3. Studio 패널 → Video Overview 선택
4. 포맷 **Explainer**
5. 언어 **한국어**
6. 비주얼 스타일 **Whiteboard**(또는 Classic)
7. steering prompt 붙여넣기
8. 생성 → 완료 후 다운로드

## UI 조작 순서 (short 버전, 경로 C를 택한 경우에만)

1. 새 노트북 → 제목: `<개념명> (short)`
2. 소스: `20-source-short.md` 만
3. Video Overview → 포맷 **Short** → 언어 English
4. steering prompt 붙여넣기 → 생성

경로 A를 택했으면 NotebookLM을 쓰지 않고 나레이션 원고를 TTS로 넘긴다.
경로 B를 택했으면 일반 버전 영상을 만든 뒤 ffmpeg 후처리 단계로 넘긴다.

## 출처

- [Generate Video Overviews in Gemini Notebook — Google 지원](https://support.google.com/gemininotebook/answer/16454555?hl=en)
- [Create and manage notebooks (API) — Google Cloud](https://docs.cloud.google.com/gemini/enterprise/notebooklm-enterprise/docs/api-notebooks)
- [Add and manage data sources in a notebook (API) — Google Cloud](https://docs.cloud.google.com/gemini/enterprise/notebooklm-enterprise/docs/api-notebooks-sources)
- [Generate podcasts (API method) — Google Cloud](https://docs.cloud.google.com/gemini/enterprise/notebooklm-enterprise/docs/podcast-api)
- [NotebookLM Short Video Overviews (2026)](https://notebooklm-guide.com/notebooklm-short-video-overviews/)
- [Google Adds Short Video Overviews to NotebookLM — TechTimes, 2026-07-02](https://www.techtimes.com/articles/319540/20260702/google-adds-short-video-overviews-notebooklm-powered-nano-banana-2-lite.htm)
