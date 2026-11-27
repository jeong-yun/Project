# Antigravity CLI(agy) 설치·사용법 — 내려받기부터 이 저장소에서 쓰기까지

Google 의 터미널용 AI 코딩 도구 **Antigravity CLI**(실행 파일 `agy`)를 이 저장소에서 쓰는 방법이다.

1. **Claude Code 처럼 그냥 쓰면 된다.** 저장소 폴더에서 `agy` 를 치면 된다. 저장소에 따로 넣을 설정 파일은 없다 — `agy` 가 [AGENTS.md](../AGENTS.md) 를 직접 읽는다.
2. **PC 마다 한 번** 할 일은 세 가지다: 설치 → 저장소 폴더에서 `agy` 실행 → Google 계정 로그인.
3. **Claude Code 와 다른 점 세 가지**만 기억한다.
   - 어느 폴더에서 켜느냐에 따라 읽는 규칙이 달라진다 ([2절](#2-이-저장소에-적용된-것)).
   - 되는 PC 에서는 처음부터 auto 모드로 켜져서, 파일 고치기와 안전하다고 본 명령을 묻지 않고 실행한다 ([3절](#3-평소-쓰는-법)).
   - 다른 AI 도구(Claude Code 등) 안에서 `agy` 를 부르지 않는다 ([4절](#4-이-저장소에서-지킬-것)).

> 기준: agy 1.5.18 · Windows 11 · 2026-11-27. 근거는 공식 문서(맨 아래 링크)와 agy 가 PC 에 풀어 두는 내장 안내·변경 기록이다.
> 이 PC 에서 직접 해 본 것과 문서로만 확인한 것은 [7절](#7-확인한-것--확인하지-못한-것)에 나눠 적었다.

## 한눈에 보기 (Claude Code 와 비교)

| | Claude Code | Antigravity CLI |
|---|---|---|
| 실행 | 저장소 폴더에서 `claude` | 저장소 폴더에서 `agy` |
| 로그인 | Anthropic 계정 | Google 계정 (첫 실행 때 브라우저가 열림) |
| 읽는 규칙 파일 | `AGENTS.md` (이 저장소에서 직접 확인한 동작) | `AGENTS.md` · `GEMINI.md` |
| 하위 폴더의 `AGENTS.md` | 그 폴더 파일을 열 때 저절로 읽음 (2.1.317 에서 확인. 루트에 `CLAUDE.md` 가 없을 때) | **저절로 읽지 않음** — 그 폴더에서 켜야 읽음 |
| 처음 켰을 때 모드 | 묻는 모드 | 되는 PC 에서는 **auto** (묻지 않고 고치고 실행). 아니면 묻는 모드 |
| 저장소에 두는 설정 | `.claude/` (이 저장소는 올리지 않는다) | 이 저장소에는 없음. 설정·권한 규칙은 사용자 홈 폴더에만 있다 |
| 내 설정 위치 | `~/.claude/` | `~/.gemini/antigravity-cli/settings.json` |
| 권한 모드 바꾸기 | Shift+Tab | Shift+Tab (같다). 실행 옵션은 `--mode` |
| 입력 | `/명령` · `@파일` · `!셸 명령` | 같다 |
| 끝내기 | `/exit` | `/exit` |

## 1. 처음 한 번 (PC 마다)

이 PC 와 다른 PC 모두 같다. 설치 폴더가 내 사용자 폴더 안이라 관리자 권한은 필요 없다.

### 1-1. 내려받기·설치

PowerShell 을 열고 공식 설치 명령을 실행한다.

```
irm https://antigravity.google/cli/install.ps1 | iex
```

PowerShell 에서 막히면 명령 프롬프트(CMD)에서 아래 공식 명령을 쓴다.

```
curl -fsSL https://antigravity.google/cli/install.cmd -o install.cmd && install.cmd && del install.cmd
```

- 설치 위치: `C:\Users\<사용자>\AppData\Local\agy\bin\agy.exe`. 어느 폴더에서든 `agy` 로 실행되게 설치 명령이 이 폴더를 등록(사용자 PATH)한다.
- **설치가 끝나면 터미널 창을 새로 연다.** 설치 전에 열려 있던 창은 `agy` 를 찾지 못한다.
- 확인: 새 창에서 `agy --version` 이 버전 번호(예: `1.5.18`)를 보여 주면 된다.
- 이미 설치된 PC 에서 설치 명령을 다시 실행하면 아무것도 바꾸지 않고 끝난다. 업데이트는 `agy` 가 스스로 한다. 손으로 하려면 `agy update`.
- winget·npm 설치는 공식 문서에 없다. 위 두 명령만 쓴다.

### 1-2. 저장소 준비

`agy` 를 처음 켜기 전에 저장소부터 받는다. 평소 하던 준비 그대로다.

1. 저장소를 받는다(`git clone` 또는 `git pull`).
2. `Project\tools\setup_prepush.bat` 를 한 번 실행한다 — [올리기 전 검사](올리기_전_검사.md).

### 1-3. 계정 로그인

1. 새 터미널에서 **저장소 폴더(`Project`)로 간 뒤** `agy` 를 실행한다 (뒤에 아무것도 붙이지 않는다).
   - 새 터미널이 처음 열리는 사용자 홈 폴더(`C:\Users\<사용자>`)에서는 켜지 않는다. `agy` 는 켠 폴더를 작업 폴더로 삼아, 그 안의 파일을 묻지 않고 읽는다.
2. 처음에는 화면 색, 그리는 방식, **이 폴더를 믿는지**를 묻는다.
   - 그리는 방식은 Alt-Screen(터미널 전체를 덮어 그림)과 Inline(터미널 기록에 이어서 출력) 중 아무거나 고른다. 나중에 `/config` 에서 바꿀 수 있다.
   - 내 저장소 폴더이므로 '믿는다'를 고른다.
3. 브라우저가 열린다. 개인 Google 계정이면 **Google OAuth** 를 고르고 로그인한다. 끝나면 터미널로 돌아와 이어진다.
4. 입력 칸이 뜨면 로그인된 것이다. `/exit` 로 나간다.

화면이 나오는 순서는 위와 다를 수 있다(확인하지 못함). 아래 두 화면도 나올 수 있다(문서로만 확인).
- 개인 계정은 약관과 함께 제품 개선용 수집 동의 칸이 나온다. 원하지 않으면 여기서 끈다.
- Gemini CLI 를 쓰던 PC 에서는 옛 설정 중 무엇을 가져올지 고르는 목록이 나온다.

**로그인 뒤**
- 로그인 정보는 Windows 자격 증명 관리자에 저장된다. 다음부터는 묻지 않는다. PC 마다 따로 로그인한다.
- 로그인됐는지 터미널에서 보려면 `agy models` — 모델 목록이 나오면 로그인된 것이다. 안 됐으면 `Please sign in` 이 나온다.
- 로그아웃은 `agy` 안에서 `/logout`.
- 원격 접속(SSH)한 PC 에서는 브라우저가 열리지 않는다. 화면에 나온 주소를 내 PC 브라우저에 붙여 넣고, 받은 코드를 터미널에 붙여 넣는다.

**요금·한도**
- 개인 Google 계정이면 무료 요금제에서도 쓸 수 있다.
- 무료·Plus 는 한도가 주 단위로 다시 찬다. Pro·Ultra 는 5시간마다 다시 차지만 주 단위 한도가 따로 있어서, 주 한도에 닿으면 5시간이 지나도 다시 차지 않는다.
- 회사·학교(Google Workspace) 계정은 작은 시작 한도만 받는다.
- 남은 한도는 `agy` 안에서 `/usage` 로 본다.

**보내는 것**
- 내가 쓴 글과 `agy` 가 읽은 파일 내용은 답을 만들기 위해 Google 로 간다.
- 제품 개선용 수집(Enable Telemetry)은 `/config` 에서 끌 수 있다. 이 선택은 파일이 아니라 계정에 저장된다.

## 2. 이 저장소에 적용된 것

**저장소 쪽 설정은 필요 없다.** `agy` 는 켠 폴더에서 저장소 루트(`.git` 이 있는 폴더)까지 **위로 올라가며** `AGENTS.md` 와 `GEMINI.md` 를 읽고, 모두 합쳐 규칙으로 쓴다.

| 저장소의 파일 | agy 가 쓰는가 | 메모 |
|---|---|---|
| 루트 `AGENTS.md` | ✅ 저장소 안 어디서 켜든 읽는다 | 공통 규칙. 따로 연결할 필요 없다 |
| 하위 폴더 `AGENTS.md` (`LLM/Agent_office` 등) | ⚠️ 그 폴더(또는 그 아래)에서 켰을 때만 | 아래 표 참고 |
| `.gemini/settings.json` | ❌ 필요 없다 | Gemini CLI 가 `AGENTS.md` 를 읽게 하는 연결 파일. agy 와는 상관없다. 그대로 둔다 |
| `CLAUDE.md` · `.claude/` | ❌ 읽지 않는다 | Claude Code 전용 |
| `.agents/` 폴더 | (없음) | agy 용 스킬·훅을 팀과 나눌 때 저장소에 두는 폴더. 지금은 쓸 곳이 없어 만들지 않았다 |

### 어느 폴더에서 켜느냐

`agy` 는 **켠 폴더보다 아래에 있는 `AGENTS.md` 는 읽지 않는다.** 작업 중에 그 폴더의 파일을 열어도 마찬가지다. Claude Code 와 가장 다른 점이다.

| 하려는 일 | 켜는 폴더 | 실리는 규칙 |
|---|---|---|
| Agent Office 작업 | `Project\LLM\Agent_office` | 루트 + Agent Office |
| AI_Agent 작업 | `Project\LLM\AI_Agent` | 루트 + AI_Agent |
| 저장소 공통 일 (문서, 루트 설정) | `Project` | 루트만 |
| (주의) `Project` 보다 위 폴더, 사용자 홈 폴더 | 켜지 않는다 | 이 저장소의 `AGENTS.md` 가 하나도 실리지 않는다 |

PowerShell 에서:

```
cd C:\...\Project\LLM\Agent_office
agy
```

- 저장소 루트에서 켜고 하위 프로젝트를 건드리는 경우를 위해, 루트 `AGENTS.md` 에 "하위 프로젝트의 파일을 읽거나 고치기 전에 그 프로젝트 폴더에 `AGENTS.md` 가 있으면 먼저 읽는다"를 넣어 두었다. 그래도 하위 프로젝트 일은 그 폴더에서 켜는 쪽이 확실하다.
- 무엇이 실렸는지는 `agy` 안에서 `/context` 로 본다. 화면 아래쪽 파일 목록(System files)에 `AGENTS.md` 경로가 나오면 실린 것이다 (문서로만 확인 — 7절).
- `AGENTS.md` 를 고치면 다시 켜지 않아도 다음 메시지부터 반영된다.
- 규칙 파일 하나는 24,000바이트까지 읽는다. 넘으면 뒷부분이 잘린다. 지금 가장 큰 루트 `AGENTS.md` 가 약 6,200바이트다.

### 저장소를 복제해도 따라오지 않는 것

아래는 저장소가 아니라 PC(내 사용자 계정)에 있다. 새 PC 에서는 다시 정한다.

| 무엇 | 어디 | 다시 정하는 법 |
|---|---|---|
| 로그인 | Windows 자격 증명 관리자 | `agy` 실행 후 로그인 |
| 설정 (모드·모델·믿는 폴더·권한 규칙) | `C:\Users\<사용자>\.gemini\antigravity-cli\settings.json` | `/config` · `/permissions` · `/model` |
| 대화 기록 | `C:\Users\<사용자>\.gemini\antigravity-cli\conversations\` | (PC 마다 따로) |

## 3. 평소 쓰는 법

### 켜고 끄기

| 하려는 일 | 명령 |
|---|---|
| 켜기 | 작업할 폴더에서 `agy` |
| 방금 하던 대화 이어서 | 그 대화를 하던 **같은 폴더**에서 `agy -c`. 대화는 켠 폴더별로 따로 남는다 |
| 예전 대화 골라서 | `agy` 안에서 `/resume` |
| 첫 질문과 함께 켜기 | `agy -i "질문"` |
| 계획만 세우는 모드로 켜기 | `agy --mode plan` |
| 끄기 | `/exit` (또는 입력 칸이 비었을 때 Ctrl+D, 또는 Ctrl+C 두 번) |

### 권한 모드 (Shift+Tab 으로 바꾼다)

| 모드 | 파일 고치기 | 명령 실행 |
|---|---|---|
| default | 고치기 전에 바뀔 내용을 보여 주고 묻는다 | 묻는다 |
| accept-edits | 묻지 않는다 | 묻는다 |
| auto | 묻지 않는다 | 격리된 공간(샌드박스)에서 묻지 않고 돌린다. 샌드박스 밖에서 돌려야 하는 명령은 `agy` 가 스스로 따져, 안전하고 되돌릴 수 있다고 본 것은 묻지 않고 돌리고 위험한 것만 묻는다 |
| plan | 고치지 않고 계획부터 쓴다 | — |

- 지금 모드는 화면 오른쪽 아래에 나온다. 아무 표시가 없으면 default 다.
- **auto 는 되는 PC 에서만 나오고, 되는 PC 에서는 처음부터 auto 로 켜진다.** 개인 계정이고 Windows 가 11 24H2 에 2026년 8월 업데이트 이상이어야 한다. 조건이 안 되면 default 로 켜지고, Shift+Tab 이 accept-edits 다음에 바로 plan 으로 넘어간다.
- 처음부터 묻게 하고 싶으면 `/config` 에서 Agent Mode 를 default 로 바꾼다.
- 물을 때: `y` 승인, `n` 거부. 파일 고치기 확인에서는 `f` 로 바뀔 내용을 전체 화면으로 본다.
- '항상 허용'은 두 가지다. '이 대화에서만'은 대화가 끝나면 사라지고, 설정에 저장하는 쪽(Persist to settings.json)만 내 설정 파일에 남는다. 쌓인 규칙은 `/permissions` 로 본다.

### 자주 쓰는 명령

| 명령 | 하는 일 |
|---|---|
| `/help` | 명령·단축키 도움말 |
| `/model` | 모델 고르기 |
| `/usage` | 남은 한도 |
| `/context` | 지금 실린 규칙 파일과 대화 길이 |
| `/plan 할 일` | 고치기 전에 계획부터 |
| `/diff` | 이번에 바뀐 내용 보기 |
| `/rewind` | 대화를 앞 시점으로 되돌리기 |
| `/compact` | 긴 대화 줄이기 |
| `/clear` | 새 대화 시작 |
| `/btw 질문` | 하던 일을 끊지 않고 곁질문 |
| `/config` | 설정 화면 |
| `/permissions` | 권한 규칙 관리 |
| `/antigravity-guide 질문` | agy 사용법을 agy 에게 묻기 |
| `/logout` | 로그아웃 |

### 입력

- `@` — 파일·폴더 경로를 골라 넣는다.
- `!명령` — 터미널 명령을 바로 실행한다. Windows 에서는 PowerShell 로 돈다.
- 줄바꿈 — Shift+Enter 또는 Ctrl+J. 터미널이 Shift+Enter 를 넘겨주지 않으면 줄 끝에 `\` 를 치고 Enter.
- 그림 붙여넣기 — Ctrl+V. 터미널이 가로채면 Alt+V.
- Esc — 답하는 중에 멈춘다.
- Ctrl+R — 계획서 같은 결과물을 열어 검토한다. 검토하라는 알림이 뜨면 누른다.

## 4. 이 저장소에서 지킬 것

`agy` 도 [AGENTS.md](../AGENTS.md) 를 따른다. `agy` 에서 특히 챙길 것만 적는다.

- **다른 AI 도구 안에서 `agy` 를 부르지 않는다.** 사람이 터미널에서 직접 쓴다. 공식 안내는 Claude Code 같은 다른 도구로 Antigravity 로그인을 쓰는 것을 약관 위반으로 보고, 계정이 정지될 수 있다고 적는다. `agy -p` 를 다른 코딩 도구 안에 넣어 쓰는 것도 개인 요금제에서는 지원하지 않는다.
- **`--dangerously-skip-permissions` 를 쓰지 않는다.** 묻지 않고 전부 실행한다. commit·push 는 내가 시킬 때만 한다는 규칙과 맞지 않는다.
- **앱을 돌리는 폴더에서 켜지 않는다.** `agy` 는 켠 폴더의 파일을 바로 고친다. `agy` 에는 작업 폴더를 따로 만들어 주는 옵션이 없으므로, Agent Office 를 실행 중이면 직접 만든 다른 폴더에서 켠다.
  - 예: `Project` 폴더에서 `git worktree add ..\Project-agy -b agy-work` 를 실행하고, `..\Project-agy\LLM\Agent_office` 로 가서 `agy` 를 켠다.
  - auto 모드에서는 켠 폴더 밖 파일도 git 저장소 안에 있으면 묻지 않고 고칠 수 있다(1.5.16 변경 기록). 앱을 돌리는 PC 에서 auto 가 켜져 있으면 Shift+Tab 으로 default 나 accept-edits 로 바꿔 쓴다.
- **커밋을 시킬 때는 메시지를 함께 준다.** `agy` 가 돌리는 git 은 편집기를 열 수 없어서, 메시지 없는 `git commit` 은 바로 실패한다.
- **올리기 전 검사는 `agy` 가 push 해도 돈다.** git 훅이라 누가 push 하든 같다 (auto 모드의 샌드박스 안에서도 같은지는 확인하지 못함).
- **`.env` · `data/` · `results/` 를 읽히거나 올리지 않는다.** `agy` 에 `.env` 보호가 들어 있지만 기준은 `AGENTS.md` 다. 더 막고 싶으면 `/permissions` 에서 거부(deny) 규칙을 넣는다. 예: `read_file(.env)` — 읽기를 막으면 쓰기도 함께 막힌다. 하위 폴더의 `.env` 까지 막히는지는 넣은 뒤 직접 확인한다.
- **`/learn` 을 쓰면 `AGENTS.md` 가 바뀔 수 있다.** 제안서를 먼저 보여 주고 승인해야 쓴다. 어느 파일을 고치려는지 보고 승인한다. `LLM/train_set_making_by_rule(talos, et)` 안의 파일은 여기서 고치지 않는다.
- **처음 써 본 뒤 `git status` 를 한 번 본다.** `agy` 는 자기 파일을 사용자 홈 폴더에 두므로 저장소에 새 파일이 생기지 않아야 한다.

## 5. 안 될 때

| 증상 | 할 일 |
|---|---|
| `agy` 를 찾을 수 없다고 나온다 | 터미널 창을 새로 연다. 그래도 안 되면 PowerShell 에서 `& "$env:LOCALAPPDATA\agy\bin\agy.exe" install` 을 실행하고 창을 새로 연다 |
| `Please sign in` | `agy` 를 뒤에 아무것도 붙이지 않고 실행해 로그인한다 |
| 브라우저 로그인이 안 된다 | 개인 Google 계정인지(회사·학교 계정은 따로 취급), 만 18세 이상으로 확인된 계정인지, 계정의 국가가 지원 지역인지 본다. 그래도 안 되면 `agy update` 뒤 다시 한다 |
| 하위 프로젝트 규칙을 안 지킨다 | 그 폴더에서 켰는지 본다. `/context` 로 실린 파일을 확인한다 |
| `agy -c` 를 했는데 새 대화가 뜬다 | 그 대화를 하던 폴더에서 켰는지 본다 |
| 매번 물어서 번거롭다 | Shift+Tab 으로 accept-edits. 명령 허용은 `/permissions` (Windows 는 명령이 글자 그대로 같아야 허용된다) |
| auto 모드가 안 보인다 | Windows 가 11 24H2 + 2026년 8월 업데이트에 못 미쳤거나, 회사 계정으로 로그인한 경우다. default 나 accept-edits 로 쓴다 |
| 한도를 다 썼다 | `/usage` 로 다시 차는 때를 본다 |
| 설정이 꼬였다 | `/config` 로 고친다. 설정 파일은 `C:\Users\<사용자>\.gemini\antigravity-cli\settings.json` |
| 더 알고 싶다 | `agy` 안에서 `/antigravity-guide 질문`, 또는 아래 공식 문서 |

## 6. 다른 PC 에서 따라 하기

1절을 줄인 순서다. 자세한 것은 괄호 안 절을 본다.

1. PowerShell 에서 `irm https://antigravity.google/cli/install.ps1 | iex` (1-1)
2. 터미널 창을 **새로** 열고 `agy --version`
3. 저장소 받기 → `Project\tools\setup_prepush.bat` 실행 (1-2)
4. 작업할 폴더로 가서 `agy`. Agent Office 면 `Project\LLM\Agent_office`. 그 PC 가 `Project` 폴더에서 앱을 돌리고 있으면 4절대로 따로 만든 폴더에서 켠다
5. 처음 한 번: 화면 색·그리는 방식 고르기 → 폴더를 믿는지 물으면 '믿는다' → 브라우저에서 Google 로그인 (1-3)
6. 일을 시키기 전에 화면 오른쪽 아래 모드를 본다. `auto` 면 묻지 않고 고친다. 묻게 하려면 Shift+Tab 으로 바꾼다 (3절)
7. `/context` 로 `AGENTS.md` 가 실렸는지 한 번 본다

## 7. 확인한 것 / 확인하지 못한 것

**이 PC 에서 직접 해 본 것 (2026-11-27)**
- `agy.exe` 1.5.18 이 `AppData\Local\agy\bin` 에 설치돼 있고 Google 서명이 유효하다. 사용자 PATH 에 들어 있다.
- `agy --version`, `agy --help`, `agy help <하위 명령>` 출력.
- 로그인 전 `agy models` 는 `Please sign in`, 브라우저 로그인 뒤에는 모델 목록이 나온다.
- 이 PC 는 **auto 모드가 안 된다.** agy 기록 파일에 "샌드박스를 쓸 수 없어 default 로 돌아간다"고 남아 있다 (Windows 11 24H2 빌드 26100, 2026년 8월 업데이트 전).
- 설치만 한 상태에서는 저장소 폴더에 새 파일이 생기지 않았다.

**공식 문서·내장 안내·변경 기록으로만 확인한 것**
- 규칙 파일을 읽는 방식(위로만 올라감, 하위 폴더는 읽지 않음), 크기 한도, 모드, 권한 규칙, 설정 위치, 요금·한도, 약관 안내.
- 설치 명령의 동작(설치 위치, PATH 등록, 이미 설치된 PC 에서 다시 실행했을 때), 첫 실행 화면, 3절의 명령·단축키, 5절의 대처.

**확인하지 못한 것 — 처음 쓸 때 한 번 본다**
- [ ] `/context` 에 루트 `AGENTS.md` 가 실제로 나오는가. `LLM\Agent_office` 에서 켜면 두 개가 나오는가.
- [ ] `git worktree` 로 만든 폴더에서 켜도 저장소 루트를 제대로 찾는가 (변경 기록 1.1.12 에는 worktree 를 챙겼다고 나온다).
- [ ] `agy --add-dir LLM\Agent_office` 로 루트에서 켜도 Agent Office 규칙이 실리는가 (문서로는 그렇게 읽힌다).
- [ ] 한글 입력, 줄바꿈(Shift+Enter)이 이 PC 터미널에서 문제없는가.
- [ ] 다른 PC 에서 처음 설치할 때 1절이 적힌 순서대로 되는가.
- [ ] auto 모드가 되는 PC 에서 `python -m pytest` 와 올리기 전 검사가 샌드박스 안에서 도는가.

대화형 실행을 확인하지 못한 까닭: 이 문서는 Claude Code 로 작성했다. 4절의 약관 안내 때문에 Claude Code 가 `agy` 대화를 대신 실행하지 않았다. 위 항목은 사람이 터미널에서 직접 본 뒤 이 문서를 고친다.

## 공식 문서

- 설치: <https://antigravity.google/docs/getting-started?tab=cli>
- 로그인: <https://antigravity.google/docs/install?tab=cli>
- 명령·단축키·설정 키: <https://antigravity.google/docs/cli/reference>
- 규칙 파일(`AGENTS.md`): <https://antigravity.google/docs/rules>
- 모드: <https://antigravity.google/docs/cli/modes>
- 권한: <https://antigravity.google/docs/permissions?tab=cli>
- 문제 해결: <https://antigravity.google/docs/cli/troubleshooting>
- 요금제: <https://antigravity.google/docs/plans> · 자주 묻는 질문: <https://antigravity.google/docs/faq>
- 저장소: <https://github.com/google-antigravity/antigravity-cli>
