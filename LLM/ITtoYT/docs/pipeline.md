# ITtoYT 파이프라인 설계

## 목표
IT 개념 하나를 웹으로 교차검증한 뒤, NotebookLM으로 **short(60초) 버전**과 **일반 버전** 영상을 만든다. 한국어 우선.

## 전체 흐름

```
개념 입력
  │
  ├─[1] it-verify      웹 리서치 + 교차검증        → 00-research.md      (자동)
  ├─[2] nlm-pack       소스팩 2종 + steering prompt → 10/20/30-*.md       (자동)
  ├─[3] NotebookLM     Video Overview 생성          → mp4                 (수동 · 2단계에서 자동화)
  ├─[4] 후처리         리프레임/자막/인트로         → 최종 mp4            (미구현)
  └─[5] 업로드         제목·설명·태그 + YouTube API                       (미구현)
```

## 왜 [3]이 수동인가
NotebookLM(Gemini Notebook)에 **Video Overview 생성 API가 없다.** Enterprise API에도 노트북·소스 관리 메서드만 있고 영상 생성은 없으며, 오디오(Podcast) API는 deprecated에 신규 allowlist 중단 상태다.
남은 선택지는 브라우저 자동화뿐이고, 이는 UI 변경에 취약하며 Google 계정 리스크가 있다. 그래서 **1단계는 반자동, 2단계에서 자동화 판단**으로 결정했다.

근거와 세부 제약: `.claude/skills/nlm-pack/references/notebooklm-settings.md`

## 한국어 short 제약
NotebookLM의 Short 포맷(9:16, 60초)은 **영어 내러이션만** 지원한다. 따라서 한국어 short은 세 경로 중 선택:
- **A. 자체 제작(권장)**: 나레이션 원고 → TTS + 편집
- **B. Explainer 리프레임**: 한국어 16:9 생성 → ffmpeg 크롭
- **C. 영어 Short + 한국어 자막**

`nlm-pack` 이 원고까지 만들어 두므로 어느 경로로 가도 입력이 준비된다.

## 디렉터리 규칙

```
ITtoYT/
├── .claude/skills/
│   ├── it-verify/SKILL.md
│   ├── nlm-pack/SKILL.md + references/notebooklm-settings.md
│   └── ittoyt/SKILL.md
├── IT 개념/<slug>/
│   ├── 00-research.md      검증 팩트시트 (출처·등급)
│   ├── 10-source-long.md   일반(Explainer) 소스팩
│   ├── 20-source-short.md  short 소스팩 + 나레이션 원고
│   ├── 30-checklist.md     NotebookLM 조작 순서 + steering prompt
│   └── meta.json           상태 추적
├── docs/pipeline.md
└── scripts/                (2단계: ffmpeg 후처리, YouTube 업로드)
```

`slug` 는 영문 kebab-case. 이후 스크립트가 경로를 다루기 쉽게 하려는 것이다.

## 상태 머신
`meta.json` 의 `stage`: `verified` → `packed` → `generated` → `published`
2단계 자동화는 이 필드를 스캔해 다음 작업을 고른다. 배치 실행의 진입점이 된다.

## 2단계 계획 (요청 시 착수)
| 항목 | 형태 | 비고 |
|---|---|---|
| 영상 후처리 | Python + ffmpeg | 9:16 리프레임, 자막 번인, 인트로/아웃트로 |
| YouTube 메타데이터 | 스킬 | 제목 A/B 후보, 설명문, 태그, 챕터 |
| YouTube 업로드 | Python + Data API | OAuth 설정 필요 |
| NotebookLM 자동화 | 브라우저 자동화 | 계정 리스크 검토 후 판단 |

후처리·업로드는 결정론적이라 LLM이 아니라 스크립트가 맞다. 스킬은 판단이 필요한 곳(리서치·대본·메타데이터)에만 둔다.
