# FreeLLMAPI(무료 모델 서버) 설치 안내 — 새 Windows 11 PC 에서 처음부터

Agent Office 의 `freellm` provider 는 **FreeLLMAPI** 라는 프로그램을 내 PC 의 Docker 안에서 돌려서 씁니다.
이 문서는 아무것도 없는 Windows 11 PC 에서 그것을 켜고 Agent Office 와 연결하기까지를 순서대로 적은 것입니다.

- FreeLLMAPI 저장소: <https://github.com/tashfeenahmed/freellmapi> (MIT 라이선스)
- FreeLLMAPI 공식 설치 문서(영어): <https://github.com/tashfeenahmed/freellmapi/blob/main/docs/en/install/01-install.md>
- 이 문서는 2026-11-04 기준, 저장소의 `docker-compose.yml`·`.env.example`·설치 문서를 읽고 정리했습니다.

## 필요 사양 (PC 최소 사항)

FreeLLMAPI 는 Docker 안에서 돌기 때문에 **Docker Desktop for Windows 의 공식 최소 사양**이 곧 이 PC 의 최소 사양입니다 (WSL 2 방식 기준).

| 항목 | 최소 사항 | 확인 방법 |
|---|---|---|
| Windows | **Windows 11 64비트 23H2(빌드 22631) 이상** 또는 **Windows 10 64비트 22H2(빌드 19045)** — Pro·Enterprise·Education 판 | `Win+R` → `winver` |
| WSL | **WSL 2, 버전 2.1.5 이상** (Windows 기능 'WSL 2' 켜짐) | `wsl --version` (없으면 2-1 의 `wsl --install`) |
| CPU | **64비트 + SLAT**(2단계 주소 변환) 지원 — 최근 10년 안의 Intel·AMD CPU 는 대부분 지원 | 작업 관리자 → 성능 → CPU |
| 가상화 | **BIOS/UEFI 에서 하드웨어 가상화 켜짐** (Intel VT-x / AMD SVM) | 작업 관리자 → 성능 → CPU → **가상화: 사용** |
| RAM | **8GB 이상** | 작업 관리자 → 성능 → 메모리 |
| 디스크 | 공식 최소값은 없음 → **C 드라이브 10GB 이상 여유 권장** (Docker Desktop·WSL2 몇 GB + FreeLLMAPI 이미지 1GB 안쪽) | 파일 탐색기 → 내 PC |
| 그 밖 | Windows 서비스 **Server(LanmanServer)** 가 '자동' 시작 | `services.msc` → Server |
| 인터넷 | 이미지 받기·공급자(Google·Groq) 호출에 필요. 회사망이면 프록시 확인 (8-8) | |

- **Windows Home 판**은 공식 지원 목록에 없습니다. **Windows Server** 도 지원하지 않습니다. Microsoft 지원이 끝난 Windows 버전도 지원하지 않습니다.
- **ARM64 Windows**(Snapdragon 노트북 등)는 'Docker Desktop for Windows - Arm' **Early Access(정식 출시 전)** 로만 있습니다. 같은 Windows·WSL·RAM 조건이고,
  FreeLLMAPI 이미지는 `linux/arm64` 도 제공하므로 돌아갈 수는 있지만 정식 지원이 아니라 권하지 않습니다.
- **라이선스**: Docker Desktop 은 개인·교육·비상업 오픈소스·소규모 회사(직원 250명 미만 **그리고** 연 매출 1천만 달러 미만)는 무료입니다.
  직원 250명 이상 **또는** 연 매출 1천만 달러 이상인 회사의 업무용, 그리고 정부 기관은 **유료 구독**이 필요합니다. 회사 PC 라면 먼저 회사 정책을 확인하세요.
- 출처: Docker 공식 문서 [Install Docker Desktop on Windows — System requirements](https://docs.docker.com/desktop/setup/install/windows-install/),
  라이선스 [Docker Desktop license agreement](https://docs.docker.com/subscription/desktop-license/) (2026-11-04 확인).

**빠른 순서 (자세한 설명은 아래 각 절)**

1. 위 필요 사양 확인 → Docker Desktop 설치·실행 (WSL2 필요) → [2절](#2-준비물)
2. FreeLLMAPI 저장소를 `C:\Users\<사용자>\freellmapi` 에 받기 → [3-1](#3-1-저장소-받기)
3. `.env` 만들기 (`ENCRYPTION_KEY`, `PORT`) → [3-2](#3-2-env-만들기-암호화-키)
4. `docker compose pull` → `docker compose up -d` → [3-3](#3-3-켜기)
5. <http://127.0.0.1:3001> 에서 계정 만들기 → Keys 페이지에 Google AI Studio·Groq 무료 키 등록 → [4절](#4-공급자-무료-키-등록-대시보드)
6. Keys 페이지 위쪽의 **통합 키**(`freellmapi-…`)를 `LLM/Agent_office/.env` 에 넣기 → [5절](#5-통합-키를-agent-office-에-넣기)
7. `python scripts/check_freellm.py` 로 점검 → 앱 사이드바에 `✅ freellm` 확인 → [6절](#6-agent-office-에서-확인)

---

## 1. 이것이 무엇이고 왜 필요한가

**FreeLLMAPI** 는 Google AI Studio·Groq·Cerebras·OpenRouter 등 여러 회사의 **무료 등급(free tier)** API 키를 한곳에 모아,
OpenAI 와 똑같은 주소 형식(`http://127.0.0.1:3001/v1`) 하나와 **통합 키 하나**로 쓰게 해 주는 내 PC 의 중계 서버(프록시)입니다.
공급자 키는 서버 안의 SQLite 데이터베이스에 암호화해 저장하고, Agent Office 는 통합 키만 압니다.

Agent Office 에서 쓰는 곳:

- **유료 모델의 마지막 대안**: 직급마다 `config/models.yaml` 의 유료 후보(OpenAI·Google·Claude 구독 등)가 모두 실패하면(키 없음·크레딧 부족·장애)
  마지막으로 무료 모델(`freellm:…`)을 씁니다. 직급별 무료 모델 배정은 README 의 '2-1. 무료 모델' 절과 `config/models.yaml` 위쪽 설명에 있습니다.
- **🆓 무료 모델만 사용** (사이드바 토글): 켜면 유료 provider 를 모두 끄고 처음부터 무료 모델만 씁니다. 비용 0원이지만 하루 호출 수가 적습니다.
- **🎛 팀원 모델 선택** 에서 무료 모델을 고른 팀원.

**FreeLLMAPI 가 꺼져 있으면**

| 누가 | 어떻게 되나 |
|---|---|
| 기본 팀원 (모델을 따로 고르지 않음) | 유료 후보로 계속 일합니다. 유료도 모두 안 되면 마지막 대안이 없어 실패합니다. |
| 무료 모델을 고른 팀원 | **유료로 넘어가지 않고** '무료 서버 꺼짐 — Docker Desktop 실행 후 FreeLLMAPI 켜기' 로 실패합니다 (실무자는 '작성 실패'로 남고 나머지는 계속, 관리자면 업무가 멈춤). |
| 🆓 무료 모델만 사용 을 켠 경우 | 쓸 모델이 없어 업무가 시작되지 않습니다. 사이드바에 'FreeLLMAPI 가 꺼져 있거나 키가 없습니다' 경고가 나옵니다. |

> 참고: FreeLLMAPI 는 Windows 용 **데스크톱 앱(.exe)** 도 있습니다 (저장소 Releases). 이 문서는 지금 쓰는 PC 와 같은 **Docker 방식**만 다룹니다.
> 데스크톱 앱은 기본 포트가 31415 라서 쓰려면 `FREELLMAPI_BASE_URL` 도 그 포트로 바꿔야 합니다.

---

## 2. 준비물

### 2-1. Docker Desktop for Windows

1. 공식 안내·다운로드: <https://docs.docker.com/desktop/setup/install/windows-install/>
   → **Docker Desktop for Windows - x86_64** 설치 파일을 받아 실행합니다.
2. 설치 화면에서 **Use WSL 2 instead of Hyper-V (recommended)** 를 켠 채로 진행합니다 (WSL2 = Windows 안에서 리눅스를 돌리는 기능, Docker 가 이걸 씁니다).
3. 설치가 끝나면 **PC 를 다시 시작**합니다.
4. Docker Desktop 을 실행합니다. 처음에 나오는 **로그인(Sign in)은 건너뛰어도 됩니다** (Skip / Continue without signing in). 설문도 건너뛰어도 됩니다.
5. 창 왼쪽 아래가 초록색 **Engine running** 이면 준비 완료입니다.

**WSL2 가 없다고 나오면** — 관리자 PowerShell(시작 → PowerShell 오른쪽 클릭 → 관리자로 실행)에서:

```powershell
wsl --install
```

다시 시작한 뒤에도 안 되면 `wsl --update` 를 한 번 더 실행합니다.

**가상화(Virtualization)가 꺼져 있다고 나오면**

- 작업 관리자(Ctrl+Shift+Esc) → **성능** → **CPU** → 오른쪽 아래 **가상화: 사용** 인지 봅니다.
- '사용 안 함'이면 PC 를 켤 때 BIOS/UEFI 설정(보통 F2·Del 키)에 들어가 **Intel VT-x**(Intel Virtualization Technology) 또는 **AMD SVM** 을 켜고 저장합니다.
- 회사 PC 라 BIOS 가 잠겨 있으면 IT 담당자에게 요청합니다.

**확인** (일반 PowerShell):

```powershell
docker --version
docker compose version
docker run --rm hello-world
```

마지막 줄이 `Hello from Docker!` 를 출력하면 Docker 가 정상입니다.

> **라이선스 주의**: 큰 회사(직원 250명 이상 또는 연 매출 1천만 달러 이상)의 업무용은 유료 구독이 필요합니다 — 맨 위 [필요 사양](#필요-사양-pc-최소-사항) 참고.

### 2-2. Git (선택)

- 있으면 `git clone` 한 줄로 받고 나중에 `git pull` 로 compose 파일을 새로 받을 수 있습니다.
- 설치: <https://git-scm.com/download/win> 또는 PowerShell 에서 `winget install --id Git.Git -e`
- **없어도 됩니다** — 3-1 의 zip 내려받기를 쓰세요.

### 2-3. 디스크 공간

- FreeLLMAPI 이미지: 1GB 안쪽 (+ 데이터베이스는 수십 MB 수준).
- Docker Desktop·WSL2 자체가 몇 GB 를 쓰므로 **C 드라이브에 10GB 이상** 여유를 두는 것이 안전합니다.

---

## 3. 설치

아래 명령은 모두 **일반 PowerShell** 에서 실행합니다. `<사용자>` 는 Windows 사용자 폴더 이름입니다 (`$env:USERNAME` 으로 확인).

### 3-1. 저장소 받기

**Git 이 있으면**

```powershell
cd $env:USERPROFILE
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi
```

**Git 이 없으면 (zip)**

1. <https://github.com/tashfeenahmed/freellmapi/archive/refs/heads/main.zip> 을 내려받습니다.
2. 압축을 풀면 `freellmapi-main` 폴더가 생깁니다. 이것을 **`C:\Users\<사용자>\freellmapi` 로 이름을 바꿔** 옮깁니다.
   (폴더 이름이 Docker 의 프로젝트 이름이 되어 컨테이너 `freellmapi-freellmapi-1`, 데이터 볼륨 `freellmapi_freellmapi-data` 가 됩니다.
   다른 이름이면 이 문서의 이름들도 그 이름으로 바뀝니다.)
3. `cd $env:USERPROFILE\freellmapi`

> 실제로 필요한 파일은 `docker-compose.yml` 과 3-2 에서 만들 `.env` 둘뿐입니다 (프로그램은 이미 만들어진 이미지를 받아 씀).

### 3-2. `.env` 만들기 (암호화 키)

`.env.example` 에는 설정이 아주 많지만, Docker 로 혼자 쓰는 데 **필요한 것은 두 개**입니다.

| 변수 | 뜻 | 값 |
|---|---|---|
| `ENCRYPTION_KEY` | 등록한 공급자 키를 데이터베이스에 암호화해 넣을 때 쓰는 키. **Docker(운영 모드)에서는 꼭 있어야 합니다.** | 64자 16진수 (아래 명령이 만들어 줌) |
| `PORT` | 내 PC 에서 열 포트 | `3001` (Agent Office 기본값과 같음) |

그 밖에 알아 두면 좋은 것 (보통은 넣지 않음):

| 변수 | 언제 |
|---|---|
| `HOST_BIND` | 기본 `127.0.0.1`(이 PC 에서만 접속). **바꾸지 마세요** — 9절 참고 |
| `PROXY_URL` | 회사 프록시를 거쳐야 Google·Groq 에 닿을 때 — 8-8 참고 |
| `WATCHTOWER_TOKEN` | 대시보드의 'Update now' 자동 업데이트를 쓸 때 — 7-3 참고 |

`freellmapi` 폴더에서 아래를 그대로 실행하면 무작위 키로 `.env` 를 만듭니다 (키를 화면에 찍지 않음):

```powershell
$Bytes = New-Object Byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($Bytes)
$Key = -join ($Bytes | ForEach-Object { "{0:x2}" -f $_ })
Set-Content -Path .env -Encoding ascii -Value "ENCRYPTION_KEY=$Key", "PORT=3001"
Remove-Variable Key, Bytes
```

- 공식 문서의 `Out-File -Encoding utf8` 대신 `ascii` 로 저장합니다. Windows PowerShell 5.1 의 utf8 은 파일 맨 앞에 보이지 않는 BOM 글자를 붙여
  첫 줄 변수 이름이 깨질 수 있기 때문입니다.
- **이 `.env` 는 비밀번호와 같습니다.** 깃에 올리거나 메신저로 보내지 마세요. 잃어버리면 등록한 공급자 키를 다시 등록해야 합니다.
- 이미 `.env` 가 있으면 덮어쓰지 마세요 (기존 키를 잃으면 저장된 공급자 키를 못 읽습니다).

### 3-3. 켜기

```powershell
docker compose pull
docker compose up -d
```

- `pull` 이 `ghcr.io/tashfeenahmed/freellmapi:latest` 이미지를 받습니다 (처음 한 번, 몇 분).
- `up -d` 가 컨테이너를 뒤에서(detached) 켭니다. compose 파일에 `restart: unless-stopped` 가 있어서 Docker Desktop 이 켜질 때마다 자동으로 다시 켜집니다.

### 3-4. 켜졌는지 확인

```powershell
docker ps
```

`freellmapi-freellmapi-1` 줄의 STATUS 가 15~30초 뒤 `Up … (healthy)` 이면 정상입니다 (`health: starting` 이면 조금 더 기다림).

```powershell
curl.exe http://127.0.0.1:3001/api/ping
```

응답이 오면 서버가 살아 있습니다 (compose 의 healthcheck 도 이 주소를 봅니다). PowerShell 에서는 `curl` 이 다른 명령의 별칭이라 **`curl.exe`** 로 씁니다.

로그 보기: `docker compose logs -f freellmapi` (끝내려면 Ctrl+C)

### 3-5. 대시보드 열고 계정 만들기

브라우저에서 <http://127.0.0.1:3001> 을 엽니다.

- 처음에는 **계정 만들기** 화면이 나옵니다. 이메일과 비밀번호를 정합니다.
  이메일로 메일이 가지는 않습니다 (로그인 아이디 용도). 같은 PC 의 브라우저면 추가 코드 없이 만들 수 있습니다.
- 비밀번호를 잊으면: 로그인 화면 **Forgot password?** → **Send reset code** → `docker compose logs freellmapi` 에 찍힌
  `Password-reset code: …` 를 15분 안에 입력합니다.

---

## 4. 공급자 무료 키 등록 (대시보드)

대시보드 왼쪽 메뉴 **Keys** 페이지에서 공급자(platform)를 고르고 키를 붙여 넣어 추가합니다. 추가하면 FreeLLMAPI 가 키를 바로 시험하고,
상태 점(초록 = 정상)과 마지막 확인 시각을 보여 줍니다. 공급자를 고르면 키 발급 페이지 링크도 같이 나옵니다.

**Agent Office 의 무료 모델 배정(`config/models.yaml`)은 Google AI Studio 와 Groq 두 곳을 기준으로 했습니다. 이 둘은 꼭 등록하세요.**

| 공급자 | 키 발급 주소 | 메모 |
|---|---|---|
| **Google AI Studio** (Gemini) | <https://aistudio.google.com/apikey> | Google 계정으로 로그인 → **Create API key**. 카드 등록 없이 무료 등급. Gemini Flash·Flash-Lite 계열을 씁니다 (Pro 계열은 무료 한도 0). |
| **Groq** (gpt-oss) | <https://console.groq.com/keys> | 가입 → **Create API Key**. `gsk_…` 로 시작. gpt-oss-20b·120b 를 씁니다. |
| Cerebras (선택) | <https://cloud.cerebras.ai> | 빠른 오픈 모델. |
| OpenRouter (선택) | <https://openrouter.ai/keys> | 이름 끝이 `:free` 인 모델만 무료. |
| Mistral (선택) | <https://console.mistral.ai/api-keys/> | |

- 그 밖에 지원하는 공급자 목록: 저장소 README 의 'Supported providers' 와 <https://freellmapi.co/models.html>.
- 발급한 키는 **이 대시보드에만** 붙여 넣습니다. Agent Office 의 `.env` 에 Google·Groq 키를 넣는 것이 아닙니다
  (Agent Office 의 `GOOGLE_API_KEY` 는 유료 `google` provider 용으로 따로 있습니다).
- **두 PC 에 같은 공급자 키를 등록하면 하루 한도를 두 PC 가 나눠 씁니다.** 한도를 따로 쓰려면 PC 마다 새 키(가능하면 다른 프로젝트·계정)를 발급하세요.

**모델별 한도 보기**

- 대시보드 **Models** 페이지: 모델마다 하루 요청 수(RPD)·분당 요청(RPM)·분당 토큰(TPM)·신뢰도·속도가 나오고, 위에서부터 라우팅 순서입니다.
- **Analytics** 페이지: 최근 호출 수·성공률·지연 시간.
- 전체 목록과 한도: <https://freellmapi.co/models.html>
- 2026-10-20 실측 기준 Gemini 무료 모델은 **모델마다 하루 20회·분당 10회**, Groq gpt-oss 는 **분당 8천 토큰** 정도라 하루에 돌릴 수 있는 업무 수가 적습니다.

---

## 5. 통합 키를 Agent Office 에 넣기

1. 대시보드 **Keys** 페이지 **위쪽(헤더)** 에 있는 **unified API key**(`freellmapi-…` 로 시작)를 복사합니다.
   이 키는 그 PC 의 FreeLLMAPI 서버에서만 통합니다 — 다른 PC 의 통합 키를 그대로 쓰면 401 이 납니다.
2. `LLM/Agent_office/.env` 를 엽니다 (없으면 같은 폴더의 `.env.example` 을 복사해 `.env` 로 이름을 바꿈). 아래 두 줄을 넣습니다:

   ```dotenv
   FREELLMAPI_BASE_URL=http://127.0.0.1:3001/v1
   FREELLMAPI_API_KEY=freellmapi-여기에-복사한-통합-키
   ```

   - `FREELLMAPI_BASE_URL` 은 기본값과 같으면 생략해도 됩니다. 포트를 바꿨다면 그 포트로 적습니다. `localhost` 보다 `127.0.0.1` 이 Windows 에서 빠릅니다.
   - Agent Office 는 `LLM/Agent_office/.env` 를 먼저 읽고, 없는 값은 `LLM/.env` 에서 채웁니다.
3. **`.env` 는 깃에 절대 올라가지 않습니다** (`LLM/Agent_office/.gitignore` 에 있음). 커밋 전에 `git status` 에 `.env` 가 보이지 않는지 한 번 더 봅니다.
4. 이미 Streamlit 앱이 켜져 있었다면 **앱을 껐다 다시 켭니다** (`.env` 는 시작할 때 읽음).

---

## 6. Agent Office 에서 확인

### 6-1. 점검 스크립트 (권장)

`LLM/Agent_office` 폴더에서:

```powershell
python scripts/check_freellm.py
```

서버 켜짐 → 통합 키 → 쓸 수 있는 모델 수 → `config/models.yaml` 의 무료 모델이 서버 목록에 있는지를 차례로 보고 한국어로 알려 줍니다.
키 값은 출력하지 않습니다. 무료 한도를 쓰지 않습니다. 실제 답변까지 시험하려면 (한도 1회 사용):

```powershell
python scripts/check_freellm.py --chat gpt-oss-20b
```

예시 출력:

```text
FreeLLMAPI 점검 — 주소 http://127.0.0.1:3001/v1
✅ 서버 켜짐 (http://127.0.0.1:3001)
✅ 통합 키 맞음 (키 길이 NN자)
✅ 쓸 수 있는 모델 2NN개
✅ config/models.yaml 의 무료 모델 N개 모두 있음
```

### 6-2. 앱 화면

- 사이드바 **⚙️ 모델·검증 상태** 를 펼쳐 `✅ freellm — 사용 가능` 인지 봅니다.
  - `🔌 freellm — 서버 꺼짐` → Docker Desktop·컨테이너가 꺼짐 (3-3, 8-2)
  - `⚪ freellm — 키 없음` → `.env` 의 `FREELLMAPI_API_KEY` 가 비었거나 앱을 다시 켜지 않음 (5절)
  - 이 표시는 **포트가 열렸는지만** 봅니다. 키가 틀려도 ✅ 로 보일 수 있으니 6-1 스크립트로 키까지 확인하세요.
- 사이드바 **🧩 모델 목록 · 직접 입력** → **🔄 모델 목록 새로 고침** 을 누르면 FreeLLMAPI 의 `/v1/models` 목록을 받아 옵니다 (등록한 키에 따라 수백 개).
- **🆓 무료 모델만 사용** 을 켜고 짧은 질문을 하나 맡겨 봅니다.

### 6-3. 직접 한 줄로 시험 (curl)

`<통합키>` 자리에 키를 넣어 실행합니다. **명령 기록에 키가 남으므로** 시험 후 PowerShell 창을 닫거나, 6-1 스크립트를 쓰는 편이 낫습니다.

```powershell
curl.exe -s http://127.0.0.1:3001/v1/models -H "Authorization: Bearer <통합키>"
```

```powershell
curl.exe -s http://127.0.0.1:3001/v1/chat/completions -H "Authorization: Bearer <통합키>" -H "Content-Type: application/json" -d '{\"model\":\"gpt-oss-20b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'
```

---

## 7. 평소 사용

### 7-1. 자동으로 켜기

- Docker Desktop → 오른쪽 위 톱니바퀴(**Settings**) → **General** → **Start Docker Desktop when you sign in to your computer** 를 켭니다.
- FreeLLMAPI 컨테이너는 `restart: unless-stopped` 라서 Docker Desktop 이 켜지면 따라 켜집니다.
  단, **직접 `docker compose stop`/`down` 으로 끈 경우는 다시 켜지지 않으니** 그때는 `docker compose up -d` 를 실행합니다.

### 7-2. 끄기·켜기 (`freellmapi` 폴더에서)

| 하고 싶은 것 | 명령 |
|---|---|
| 잠시 끄기 / 다시 켜기 | `docker compose stop` / `docker compose start` |
| 켜기 (설정·이미지가 바뀐 뒤에도) | `docker compose up -d` |
| 컨테이너 지우기 (데이터는 남음) | `docker compose down` |
| 상태·로그 | `docker ps` / `docker compose logs -f freellmapi` |

다른 폴더에서는 `docker compose -f C:/Users/<사용자>/freellmapi/docker-compose.yml up -d` 처럼 `-f` 로 파일을 가리킵니다.

> ⚠️ `docker compose down -v` 는 **데이터 볼륨까지 지웁니다** (등록한 공급자 키·계정·통합 키가 모두 사라짐). 지울 생각이 아니면 `-v` 를 붙이지 마세요.

### 7-3. 업데이트

**직접 (권장)**

```powershell
docker compose pull
docker compose up -d
```

데이터 볼륨과 `.env` 는 그대로 이어집니다. 모델 목록(카탈로그)은 업데이트 없이도 하루 두 번 스스로 받아 옵니다.
compose 파일 자체가 바뀌었을 수 있으니 가끔 `git pull` (zip 이면 새로 받아 `docker-compose.yml` 만 교체)도 합니다.

**대시보드에서 'Update now' (선택)**

`.env` 에 `WATCHTOWER_TOKEN=<긴 무작위 값>` 을 넣고 `docker compose --profile autoupdate up -d` 로 켜면,
Settings → Check for updates 에 **Update now** 가 생기고 Watchtower 보조 컨테이너가 이미지를 바꿔 줍니다.
다만 이 보조 컨테이너는 **Docker 소켓(`/var/run/docker.sock`)을 마운트**하는데, compose 파일 설명대로 이것은 호스트의 관리자 권한과 같습니다.
그래서 **기본은 끈 채로 두고 직접 업데이트하는 것을 권합니다.**

### 7-4. 백업 · 다른 PC 로 옮기기

등록한 공급자 키·계정·통합 키·설정은 Docker 볼륨 **`freellmapi_freellmapi-data`** (컨테이너 안 `/app/server/data`)에 있습니다.
이름은 `docker volume ls` 로 확인합니다.

**가장 쉬운 방법은 옮기지 않는 것**입니다 — 새 PC 에 3~5절을 새로 하고, 공급자 키를 다시 등록(또는 새로 발급)하면 됩니다.
통합 키·계정·순서 설정까지 그대로 가져가고 싶을 때만 아래처럼 옮깁니다.

**반드시 함께 옮길 것: 같은 `.env` (`ENCRYPTION_KEY`).** 볼륨 속 공급자 키는 이 키로 암호화돼 있어서 키가 다르면 읽을 수 없습니다.

1. **원래 PC — 백업** (`freellmapi` 폴더에서, 일관된 복사를 위해 잠시 끔):

   ```powershell
   docker compose stop
   docker run --rm -v freellmapi_freellmapi-data:/data -v "${PWD}:/backup" alpine tar czf /backup/freellmapi-data.tgz -C /data .
   docker compose start
   ```

   `freellmapi-data.tgz` 가 생깁니다. (`alpine` 은 Docker Hub 의 공식 작은 리눅스 이미지로, 압축만 하고 바로 지워집니다.)

2. `freellmapi-data.tgz` 와 `.env` 를 **USB 등 개인 매체로** 새 PC 의 `C:\Users\<사용자>\freellmapi` 에 옮깁니다.

3. **새 PC — 복원** (2절·3-1 까지 하고, 3-2 의 새 `.env` 는 만들지 않고 옮긴 `.env` 사용):

   ```powershell
   docker compose pull
   docker compose up -d
   docker compose stop
   docker run --rm -v freellmapi_freellmapi-data:/data -v "${PWD}:/backup" alpine sh -c "rm -rf /data/* && tar xzf /backup/freellmapi-data.tgz -C /data"
   docker compose start
   ```

4. 대시보드에 원래 계정으로 로그인되고 Keys 페이지에 공급자 키가 초록색이면 성공입니다. 통합 키도 원래와 같으므로 Agent Office 의 `.env` 도 그대로 씁니다.

> 🔒 `freellmapi-data.tgz` 와 `.env` 에는 **공급자 키(암호화)와 그것을 푸는 키가 함께** 들어 있습니다.
> 둘을 같이 가진 사람은 내 키를 쓸 수 있으니 깃·클라우드 공유 폴더·메신저에 올리지 말고, 옮긴 뒤에는 지우세요.

> 같은 공급자 키를 두 PC 가 함께 쓰게 되면 하루 한도도 나눠 씁니다 (4절).

---

## 8. 문제 해결

먼저 `python scripts/check_freellm.py` 를 돌리면 어느 단계에서 막혔는지 바로 나옵니다.

### 8-1. 포트 3001 을 이미 쓰고 있음

`docker compose up -d` 가 `port is already allocated` / `bind: address already in use` 로 실패할 때:

```powershell
netstat -ano | findstr :3001
Get-Process -Id <맨 끝 숫자(PID)>
```

- 그 프로그램을 끄거나,
- `freellmapi\.env` 의 `PORT=3002` 처럼 바꾸고 `docker compose up -d` → Agent Office `.env` 의 `FREELLMAPI_BASE_URL=http://127.0.0.1:3002/v1` 로 맞춥니다.
  (컨테이너 안쪽은 계속 3001, 바깥 포트만 바뀝니다.)

### 8-2. Docker 가 꺼져 있음

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Docker Desktop 앱이 꺼져 있다는 뜻입니다. 시작 메뉴에서 Docker Desktop 을 켜고 **Engine running** 이 될 때까지(30초~1분) 기다린 뒤 다시 실행합니다.
앱 사이드바의 '🔌 서버 꺼짐', '무료 서버 꺼짐 — Docker Desktop 실행 후 FreeLLMAPI 켜기' 도 대부분 이 경우입니다. 자동 시작은 7-1.

### 8-3. WSL2 가 없거나 오래됨

Docker Desktop 이 `WSL 2 installation is incomplete` / `WSL needs updating` 을 띄우면 관리자 PowerShell 에서
`wsl --install` (처음) 또는 `wsl --update` 후 다시 시작합니다. 가상화가 꺼져 있으면 2-1 참고.
'Virtual Machine Platform' 기능이 꺼져 있다는 메시지면 `wsl --install` 이 켜 줍니다.

### 8-4. 401 Unauthorized

- Agent Office `.env` 의 `FREELLMAPI_API_KEY` 가 이 PC 대시보드 Keys 페이지의 통합 키와 다름 (앞뒤 공백·따옴표 포함 여부 확인).
- 다른 PC 의 통합 키를 그대로 가져옴 → 이 PC 의 키로 바꿈 (볼륨을 옮긴 경우는 같음).
- 대시보드에서 통합 키를 새로 만들었으면 `.env` 도 바꾸고 앱을 다시 켭니다.

### 8-5. 429 · 하루 한도

- 무료 등급은 모델마다 분당·하루 한도가 작습니다 (4절). 대시보드 **Models** 페이지에서 남은 한도를 봅니다.
- Agent Office 는 곧 풀리는 한도(150초 이내)는 기다렸다 다시 하고, 하루 한도가 찬 모델은 10분 동안 건너뜁니다.
- 자주 걸리면 `config/settings.yaml` 의 `llm_concurrency` 에서 `freellm` 동시 호출을 1 로 낮추거나, 공급자 키를 더 등록합니다 (Cerebras·OpenRouter 등).
- 다음 날 한도가 다시 차는 시각은 공급자마다 다릅니다 (Google 은 미국 태평양 시간 자정 기준).

### 8-6. 모델이 404 (없어진 모델)

무료 모델은 공급자가 자주 바꾸고 없앱니다 (예: gemini-2.5 계열이 404). `scripts/check_freellm.py` 가
'config/models.yaml 의 무료 모델 중 서버 목록에 없음' 으로 알려 줍니다. 앱의 **🔄 모델 목록 새로 고침** 에서 ⚠️(사라진 모델)를 확인하고,
`config/models.yaml` 의 그 이름을 목록에 있는 비슷한 모델로 바꿉니다. 해당 공급자 키를 아직 등록하지 않아 안 보이는 경우도 있습니다.

### 8-7. `auto` 모델을 쓰면 오류

FreeLLMAPI 의 `auto`(자동 선택)는 가끔 응답에 `role` 이 빠져 와서 LangChain 이 거부합니다. 그래서 Agent Office 는 `auto`·`router`·`default` 를
쓰지 않고 모델 목록에서도 뺍니다. `config/models.yaml` 이나 '직접 입력'에 `auto` 를 넣지 말고 정확한 모델 이름을 쓰세요.

### 8-8. 회사 네트워크 · 프록시

컨테이너는 켜졌는데 Keys 페이지의 공급자 키가 빨간색이거나 모든 호출이 시간 초과일 때:

- **어디서 막히는지 보기** (`freellmapi` 폴더에서):

  ```powershell
  docker compose exec freellmapi node -e "fetch('https://generativelanguage.googleapis.com/').then(r=>console.log('ok',r.status)).catch(e=>console.log('fail',e.cause?.code||e.message))"
  ```

  `ok 404` 처럼 숫자가 나오면 인터넷은 됨, `fail …` 이면 컨테이너에서 밖으로 못 나감.
- **회사 프록시를 거쳐야 하면** `freellmapi\.env` 에 `PROXY_URL=http://프록시주소:포트` 를 넣고 `docker compose up -d`.
  (대시보드 Keys → Outbound proxy 에서도 설정 가능.) 내 PC 에서 도는 프록시 프로그램이면 `127.0.0.1` 이 아니라
  `PROXY_URL=http://host.docker.internal:포트` 로 적습니다 — 컨테이너 안에서 127.0.0.1 은 컨테이너 자신입니다.
- 이미지 내려받기(`docker compose pull`)부터 막히면 Docker Desktop → Settings → **Resources → Proxies** 에 프록시를 넣습니다.
- 회사 보안 장비가 HTTPS 를 중간에서 풀어 보는 환경(SSL 검사)이면 컨테이너가 회사 인증서를 몰라 `certificate` 오류가 날 수 있습니다. IT 담당자에게 문의하세요.

### 8-9. 방화벽 · 보안 프로그램

- 포트를 `127.0.0.1` 에만 열기 때문에 Windows 방화벽 규칙은 필요 없습니다. 방화벽 허용 창이 뜨면 '공용 네트워크'는 허용하지 않아도 됩니다.
- 회사 백신·DRM·보안 에이전트가 WSL2·Docker 를 막는 경우가 있습니다 (Docker Desktop 이 시작하다 멈춤, `wsl` 오류).
  이 경우 개인 설정으로 풀기보다 IT 담당자에게 예외를 요청하세요.

### 8-10. 그 밖

- 대시보드 비밀번호 분실: 3-5.
- 컨테이너가 계속 재시작: `docker compose logs freellmapi` 에서 `ENCRYPTION_KEY` 관련 오류면 `.env` 를 확인 (3-2, BOM 주의).
- FreeLLMAPI 자체 문제 해결 문서: <https://github.com/tashfeenahmed/freellmapi/blob/main/docs/en/troubleshooting/01-common-issues.md>

---

## 9. 보안

- **`HOST_BIND` 는 기본값 `127.0.0.1` 그대로 두세요.** 이 PC 에서만 접속됩니다. `0.0.0.0` 으로 바꾸면 같은 네트워크(회사망·카페 와이파이)의 다른 기기가
  대시보드와 API 에 접근할 수 있고, 지키는 것은 통합 키 하나뿐입니다. 인터넷에 직접 열거나 포트 포워딩하지 마세요.
- **개인용입니다.** FreeLLMAPI 는 한 사람이 쓰도록 만든 프로그램입니다. 통합 키를 다른 사람과 나눠 쓰지 마세요.
  각 공급자의 무료 등급 이용 조건(계정당 사용, 재판매 금지 등)도 지켜야 합니다.
- **민감한 데이터는 무료 모델에 보내지 마세요.** 무료 등급은 공급자에 따라 입력을 저장하거나 서비스 개선·학습에 씁니다.
  - Google Gemini API **무료 등급(Unpaid Services)**: 보낸 내용과 답변을 Google 제품·기술 개선에 쓰고, 사람 검토자가 읽을 수 있다고 약관에 적혀 있습니다
    → <https://aistudio.google.com/docs/terms> (Unpaid Services 절, 예전 주소 ai.google.dev/gemini-api/terms 에서 넘어옴)
  - Groq: 기본적으로 추론 요청 데이터를 보관하지 않는다고 하지만, 문제 해결·남용 조사용 임시 로그(최대 30일) 등 예외가 있습니다
    → <https://console.groq.com/docs/your-data>
  - 그 밖의 공급자는 각자의 약관을 확인하세요. 일부는 학습에 쓴다고 명시합니다 (예: Kilo Gateway 의 익명 무료 경로).
  - 회사 자료·개인정보·고객 정보가 들어가는 업무는 🆓 무료 모델만 사용 을 끄고 유료 키(또는 Claude 구독)로 처리하세요.
- **비밀 파일 세 가지** — `freellmapi\.env`(암호화 키), Docker 볼륨/백업 `freellmapi-data.tgz`(공급자 키), `LLM/Agent_office/.env`(통합 키).
  셋 다 깃에 올리지 않고, 화면 공유·캡처에 나오지 않게 합니다.
- 키가 새어 나갔다고 의심되면: 각 공급자 콘솔(4절 주소)에서 그 키를 지우고 새로 발급 → 대시보드에 다시 등록 → 통합 키도 새로 만들어 Agent Office `.env` 교체.

---

## 10. 지우기

```powershell
cd $env:USERPROFILE\freellmapi
docker compose down -v
```

`-v` 로 데이터 볼륨까지 지웁니다 (공급자 키·계정 모두 삭제). 그다음 `freellmapi` 폴더를 지우고, 필요하면 Docker Desktop 을
설정 → 앱 에서 제거합니다. 등록했던 공급자 키도 각 공급자 콘솔에서 지워 두면 안전합니다.
