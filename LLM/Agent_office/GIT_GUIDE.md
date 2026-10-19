# Git / GitHub 에 올리기 — 단계별 가이드

처음 GitHub 에 올리는 절차입니다. (현재 코드는 이미 `jeong-yun/Project` 저장소의 `LLM/Agent_office` 에 올라가 있으므로, 다른 PC 에서 시작할 때는 10장부터 보면 됩니다.)
Windows **PowerShell** 기준이며, 명령은 한 줄씩 순서대로 실행합니다.

> 현재 PC 상태 (2026-10-19 확인)
> - Git 2.51 설치됨, Git Credential Manager(`manager`) 설정됨 → 첫 push 때 브라우저로 GitHub 로그인 창이 뜸
> - git 전역 사용자 이름·메일 미설정 → 아래 4단계에서 **이 저장소에만** 설정
> - GitHub CLI(`gh`) 미설치 → 저장소는 GitHub 웹사이트에서 만듦

---

## 0. 전체 흐름 한눈에 보기

```
[1] GitHub 에 빈 저장소 만들기 (웹)
[2] 폴더 위치 정하기 (langchain-kr 밖으로 복사 권장)
[3] git init (저장소 시작)
[4] 이 저장소에만 이름·메일 설정
[5] 올리면 안 되는 파일 확인 (.env, data/)
[6] git add → 목록 확인 → git commit
[7] GitHub 주소 연결 → git push
[8] GitHub 에서 확인
```

---

## 1. GitHub 에 빈 저장소 만들기 (웹)

1. https://github.com 로그인 → 오른쪽 위 **+** → **New repository**
2. 입력
   - **Repository name**: 예) `agent-office`
   - **Public / Private**: 개인 프로젝트·포트폴리오면 Public, 비공개면 Private
   - ⚠️ **Add a README / .gitignore / license 는 모두 체크하지 않는다** (이미 폴더에 있음. 체크하면 첫 push 가 충돌함)
3. **Create repository** → 화면에 나오는 주소를 복사해 둔다
   예) `https://github.com/<내계정>/agent-office.git`

---

## 2. 폴더 위치 정하기

### 방법 A (권장) — langchain-kr 밖으로 복사해서 쓰기

저장소 안에 저장소가 있으면(중첩 저장소) 헷갈리기 쉬우므로 밖으로 꺼내는 것을 권장합니다.

```powershell
# 프로젝트 폴더 상위로 이동
cd "C:\Users\Jyun\Desktop\프로젝트"

# 폴더 통째로 복사 (data/ 의 교훈·기록도 함께 복사됨 — git 에는 안 올라감)
Copy-Item -Recurse "langchain-kr\langchain-kr\26-Agent-Office" ".\agent-office"

cd ".\agent-office"
```

복사한 폴더에는 `.env` 가 없습니다. 지금까지는 langchain-kr 루트의 `.env` 를 함께 읽었으므로,
**새 폴더에서도 실행하려면 `.env` 를 만들어야 합니다.**

```powershell
Copy-Item .env.example .env
notepad .env        # OPENAI_API_KEY, GOOGLE_API_KEY 등 값을 채우고 저장
```

> 복사가 잘 됐고 새 폴더에서 앱이 실행되는 것을 확인했다면, 원래 `langchain-kr\langchain-kr\26-Agent-Office` 는 지워도 됩니다 (선택).

### 방법 B — langchain-kr 안에 그대로 두고 쓰기

```powershell
cd "C:\Users\Jyun\Desktop\프로젝트\langchain-kr\langchain-kr\26-Agent-Office"

# 바깥 저장소(langchain-kr)가 이 폴더를 추적하지 않도록 '로컬 제외' 등록 (커밋되지 않는 설정)
Add-Content -Path "..\.git\info\exclude" -Value "26-Agent-Office/"
```

이후 단계는 A·B 모두 같습니다. **반드시 26-Agent-Office(또는 agent-office) 폴더 안에서** 실행하세요.

---

## 3. git init — 저장소 시작

```powershell
# 현재 위치가 프로젝트 폴더인지 확인 (app.py, README.md 가 보여야 함)
ls

# 저장소 만들기 (기본 브랜치 이름 main)
git init -b main
```

`Initialized empty Git repository in ...\.git` 이 나오면 성공입니다.

---

## 4. 이 저장소에만 이름·메일 설정

커밋에 기록될 작성자 정보입니다. `--global` 을 붙이지 않으면 **이 저장소에만** 적용됩니다.

```powershell
git config user.name "내 이름 또는 GitHub 아이디"
git config user.email "사용할메일@example.com"

# 확인
git config user.name
git config user.email
```

> GitHub 프로필에 커밋이 내 것으로 표시되려면, 이 메일이 GitHub 계정(Settings → Emails)에 등록된 메일이어야 합니다.
> 메일을 공개하고 싶지 않으면 GitHub 가 주는 `숫자+아이디@users.noreply.github.com` 주소를 써도 됩니다.

---

## 5. 올리면 안 되는 파일 확인 (가장 중요)

| 파일·폴더 | 이유 | 처리 |
|---|---|---|
| `.env` | **API 키** — 공개되면 남이 내 비용으로 사용 | `.gitignore` 로 제외 |
| `data/` | 실행 기록·교훈 DB, 평가 결과 (개인 데이터) | `.gitignore` 로 제외 |
| `__pycache__/`, `.pytest_cache/`, `.venv/` | 자동 생성 파일 | `.gitignore` 로 제외 |
| `.env.example` | 값이 빈 키 목록 (다른 PC 설정용) | **올린다** |

확인 명령:

```powershell
# 둘 다 경로가 출력되면 제외되고 있는 것 (아무것도 안 나오면 위험!)
git check-ignore .env
git check-ignore data/runs.db
```

---

## 6. 파일 추가 → 목록 확인 → 커밋

```powershell
# 1) 모든 파일을 '올릴 목록(staging)'에 추가 (.gitignore 대상은 자동 제외)
git add .

# 2) 올릴 목록 확인 — .env, data/ 가 없어야 한다
git status
```

`git status` 의 초록색 `new file:` 목록에 아래가 **있으면 정상**입니다.
`app.py`, `core/...`, `config/...`, `teams/...`, `tests/...`, `evals/...`, `assets/avatars/...`, `README.md`, `.env.example`, `.gitignore` 등

아래가 **보이면 멈추고** 6-1 로 갑니다: `.env`, `data/runs.db`, `data/checkpoints.db`

```powershell
# 3) 커밋 (스냅샷 저장)
git commit -m "Agent Office 초기 버전: 조직도형 멀티 에이전트 + 검증 하네스"

# 4) 커밋 확인
git log --oneline
```

### 6-1. 실수로 .env 가 목록에 들어갔다면

```powershell
git rm --cached .env          # 목록에서만 빼기 (파일은 그대로)
git status                    # 다시 확인
```

> 이미 **push 까지 했다면** 키가 공개된 것입니다. 저장소에서 지워도 기록에 남으므로
> 각 사 콘솔(OpenAI·Google·Anthropic)에서 **키를 즉시 폐기하고 새로 발급**하세요.

---

## 7. GitHub 주소 연결 → push

```powershell
# 1단계에서 복사한 주소
git remote add origin https://github.com/<내계정>/agent-office.git

# 연결 확인
git remote -v

# 올리기 (-u: 앞으로 git push 만 쳐도 이 주소·브랜치로 올라감)
git push -u origin main
```

- 처음 push 하면 **브라우저 로그인 창**(Git Credential Manager)이 뜹니다 → GitHub 로그인 → 승인
- 한 번 로그인하면 Windows 자격 증명 관리자에 저장되어 다음부터는 묻지 않습니다.

---

## 8. GitHub 에서 확인

1. 저장소 페이지 새로고침 → 파일 목록과 README 가 보이는지 확인
2. **`.env` 와 `data/` 폴더가 목록에 없는지 다시 확인**
3. 커밋 작성자가 내 계정으로 표시되는지 확인 (아니면 4단계 메일 확인)

---

## 8-1. (다른 방법) 이미 있는 저장소의 폴더에 올리기 — 예: `jeong-yun/Project` 의 `LLM/Agent_offic`

새 저장소를 만들지 않고, 기존 저장소 안의 폴더에 올리는 경우입니다 (1~8단계 대신 이 절차).

⚠️ git 이 연결하는 주소는 **저장소 전체 주소**뿐입니다.
- ❌ `https://github.com/jeong-yun/Project/tree/main/LLM/Agent_offic` — 브라우저의 폴더 페이지 주소 → `not found`
- ✅ `https://github.com/jeong-yun/Project.git` — 저장소 주소 (초록색 **Code** 버튼에서 복사)

```powershell
# 1) 저장소 전체를 PC 로 내려받기
cd "C:\Users\Jyun\Desktop\프로젝트"
git clone https://github.com/jeong-yun/Project.git
cd Project

# 2) 이 저장소에만 작성자 설정
git config user.name "jeong-yun"
git config user.email "사용할메일@example.com"

# 3) 프로젝트 파일을 LLM\Agent_offic 폴더 '안으로' 복사 (폴더째가 아니라 내용물을)
Copy-Item -Recurse -Force "..\langchain-kr\langchain-kr\26-Agent-Office\*" ".\LLM\Agent_offic\"

# 4) .gitignore 가 함께 복사됐는지 확인 (목록에 .gitignore, .env.example 이 보여야 함)
Get-ChildItem -Force .\LLM\Agent_offic | Select-Object Name

# 5) 추가 → 목록 확인 → 커밋 → 올리기
git add LLM/Agent_offic
git status          # LLM/Agent_offic/data/ , .env , __pycache__ 가 없어야 한다
git commit -m "Agent Office 프로젝트 추가"
git push            # 처음이면 브라우저 로그인 창이 뜬다
```

- 폴더 안의 `.gitignore` 는 상위 저장소 안에서도 그 폴더에 적용되므로 `data/`(교훈·기록)는 자동으로 빠집니다.
- 폴더 이름을 `Agent_office` 로 고치고 싶으면 5) 전에: `git mv LLM/Agent_offic LLM/Agent_office` (이후 명령의 경로도 바꿀 것)
- 이후 수정은 `Project` 폴더에서 작업하고 `git add` → `git commit` → `git push` (9장과 같음).
  langchain-kr 쪽 `26-Agent-Office` 와 두 군데에서 따로 고치면 내용이 어긋나므로, 올린 뒤에는 `Project\LLM\Agent_offic` 에서만 작업하는 것을 권장합니다.
  그 폴더에서 앱을 실행하려면 `.env` 를 새로 만들어야 합니다 (`Copy-Item .env.example .env`).

---

## 9. 이후 작업할 때 (매번 반복)

```powershell
git status                          # 무엇이 바뀌었는지
git add .                           # 바뀐 파일 추가
git status                          # .env / data/ 없는지 확인 습관
git commit -m "무엇을 왜 바꿨는지 한 줄"
git push                            # GitHub 에 올리기
```

커밋 메시지 예시
- `조직도에 보안 로그 분석 팀 추가`
- `HBF 검색 실패 수정: 뉴스 검색 추가`
- `today_worked: 10-20 진행 기록`

> 하루 작업이 끝나면 `today_worked.md` 를 갱신하고 함께 커밋하면 JD 용 기록이 자동으로 쌓입니다.

---

## 10. 다른 PC 에서 내려받아 실행하기

현재 코드는 `jeong-yun/Project` 저장소의 `LLM/Agent_office` 폴더에 있습니다.

```powershell
cd 원하는\폴더
git clone https://github.com/jeong-yun/Project.git
cd Project\LLM\Agent_office

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env                        # API 키 입력

streamlit run app.py
```

- git 으로 옮겨지지 않는 것: `.env`(API 키), `data/`(교훈·기록), `results/`(채택 결과 파일), `.venv/`(가상환경).
  교훈을 이어 쓰려면 `data\runs.db`, 저장한 결과를 옮기려면 `results\` 폴더를 같은 위치에 직접 복사합니다.
- 두 PC 를 오가며 작업할 때: **시작 전 `git pull` → (작업) → `git add` · `git commit` · `git push`**.
  `git pull` 뒤에는 새 패키지가 있을 수 있으니 `pip install -r requirements.txt` 를 한 번 더 실행합니다.
- 새 PC 에서 처음 커밋할 때는 작성자 설정이 필요합니다: `git config user.name "이름"`, `git config user.email "메일"`

---

## 11. 자주 나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `fatal: not a git repository` | 프로젝트 폴더 밖에서 실행 | `cd` 로 프로젝트 폴더에 들어가서 실행 |
| `Author identity unknown` | 4단계 이름·메일 미설정 | `git config user.name` / `user.email` 설정 |
| push 시 `rejected (fetch first)` | GitHub 저장소를 만들 때 README 등을 체크함 | `git pull --rebase origin main` 후 `git push` |
| push 시 `remote origin already exists` | 주소를 이미 연결함 | `git remote set-url origin <새주소>` |
| 커밋 메일을 잘못 썼음 (push 전) | 4단계 설정 오류 | 메일 다시 설정 → `git commit --amend --reset-author --no-edit` |
| `.env` 를 push 해 버림 | 5단계 확인 누락 | 6-1 참고 — **키 폐기·재발급이 우선** |
| PowerShell 에서 `Activate.ps1` 실행 차단 | 실행 정책 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| (방법 B) langchain-kr 에서 `26-Agent-Office/` 가 계속 보임 | 로컬 제외 미등록 | 2단계 방법 B 의 `exclude` 등록 |
