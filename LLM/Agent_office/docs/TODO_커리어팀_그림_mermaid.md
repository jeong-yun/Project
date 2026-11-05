# 할 일 — 👔 포트폴리오·자소서 팀: 원본 그림(흐름도) 살리기 · mermaid 유지 (2026-11-04)

> **다른 PC 에서 이어서 하는 법**
> 1. `cd Project` → `git pull`
> 2. `LLM/Agent_office` 에서 Claude Code 를 열고: **"docs/TODO_커리어팀_그림_mermaid.md 를 읽고 1단계부터 진행해줘"**
> 3. 단계마다 `python -m pytest tests/test_career*.py -q` 통과 → 커밋. 끝나면 맨 아래 '진행 기록'에 적기.
>
> ⚠️ 이 저장소는 **공개**입니다. 테스트용 docx·그림은 코드로 만든 가짜(작은 PNG + 가짜 글)만 쓰고, 실제 경력 문서·그림을 넣지 마세요.
> 관련 할 일: [TODO_커리어팀_형식_피드백.md](TODO_커리어팀_형식_피드백.md) (수정안 4 '그림·흐름도', 6 '받지 못한 참고 파일').

---

## 무슨 일이 있었나

🔄 포트폴리오 최신화 결과(Word·md)에 **흐름도가 하나도 없었다**. `- **흐름도**:` 칸만 8곳 비어 있음.

## 원인 (코드로 확인, 커밋 7a4693d 기준)

| # | 원인 | 위치 |
|---|---|---|
| 1 | **원본 Word 의 흐름도는 그림** — 실제 문서: 그림(`w:drawing`) 6개, **모두 표 칸 안**(본문 문단에는 0개). 문서 읽기는 문단 글자·표 칸 글자만 꺼내고 그림은 건너뜀 → 모델은 그림이 있다는 것조차 모름 | `core/doc_text.py` `read_docx()` (66행~) |
| 2 | **mermaid 원본이 들어오지 않음** — 추가 요청에 'md 는 경력서.md 와 유사하게'라고 했지만 그 파일은 이름만 전달. 올라간 md 에도 ```` ```mermaid ```` 블록 0개. 양식에 '형식 참고 파일' 칸이 없음 | `teams/career/team.yaml` `portfolio_update` fields |
| 3 | **지침에 그림·mermaid 규칙이 없음** — 사실 규칙('문서에 없는 것은 쓰지 않는다')만 있어 모델이 흐름도 칸을 비움 | `teams/career/skills/portfolio.md` · `career_writing.md` · `recommend_update.md` |
| 4 | **md·Word 내보내기가 코드 블록을 모두 지움** — 끝의 JSON 요약을 빼려는 것인데 ```` ```mermaid ```` 도 함께 사라짐. 모델이 mermaid 를 써 와도 결과 파일에 남지 않는다 | `core/career.py` `body_text()` (296행: `re.sub(r"```.*?```", "")`) → `render_document()` 851행 / `core/doc_export.py` `parse_markdown()` 57행 (코드 블록 건너뜀) |

## 설계 (이렇게 고치기로 함)

- **원본 그림은 다시 그리지 않고 그대로 옮긴다.** 문서를 읽을 때 그림 자리에 `[그림 N]` 표시를 남기고, 원본 그림 파일을 보관 → 모델은 표시를 같은 자리에 그대로 둠 → Word 내보내기가 표시 자리에 원본 그림을 넣음.
- **mermaid 는 글(코드)로 유지한다.** md 에는 그대로, Word 에는 그릴 수 없으므로 '원본 그림이 있으면 그림, 없으면 코드 상자 + 이미지로 바꾸라는 안내'.
- 번호는 업무 하나 안에서 **전체 번호** (지금 포트폴리오 먼저, 그다음 요약 문서 순서) — 문서마다 1부터 매기면 겹침.

## 작업 순서

### 1단계 — 문서 읽기: 그림 자리 표시 + 원본 보관 (`core/doc_text.py`)
- `DocImage(data: bytes, ext: str)` 와 `DocText.images: list[DocImage]` 추가.
- `read_docx`: 문단·표 칸마다 `.//a:blip` 의 `r:embed` → `document.part.related_parts[rId].blob` 로 그림을 차례로 모으고, 그 자리 글 끝에 `[그림 N]` (문서 안 번호).
  - **표 칸이 중요** (실제 문서는 그림이 모두 표 칸 안). 병합 칸은 `row.cells` 에 같은 `_tc` 가 되풀이되므로 **`_tc` 객체로 한 번만** 셀 것 (지금 글자는 `dict.fromkeys` 로 중복 제거 중).
  - 기존 `read_docx(data) -> (text, units)` 를 쓰는 테스트가 있으니(`tests/test_career.py` `test_read_docx_with_table`) 시그니처는 유지하고, 그림까지 돌려주는 함수를 따로(`read_docx_full`) 두고 `read_bytes` 가 그것을 쓰게.
  - `_clean()`·`_limit()` 이 `[그림 N]` 을 깨지 않는지 확인. (가능하면 hwpx 의 `hp:pic`·PDF 는 다음 단계 — 이번엔 docx 만)
- 테스트: python-docx 로 표 칸 안에 작은 PNG 를 넣은 가짜 docx → 글에 `[그림 1]`, `images` 1개, 병합 칸이어도 1개.

### 2단계 — 전체 번호 + 그림 파일 보관 (`core/tools/career_docs.py`)
- 읽은 문서들(`_run_update` 는 지금 포트폴리오 `old` 먼저, 그다음 `docs`; `run` 은 `docs`)의 `[그림 n]` 을 전체 번호로 다시 매기고(`re.sub`), 원본 그림을
  올린 파일과 같은 폴더(`data/uploads/`)에 **내용 해시 이름**(`figure_<sha1 12자>.<ext>`)으로 저장.
- `harness_options["career_figures"] = [그림 경로 …]` (번호 순서, 1번 = 인덱스 0). 체크포인트에 bytes 를 넣지 않기 위해 경로만.
- 근거 자료 표(📄 읽은 문서)에 '그림 N개' 칸 추가 — 사람이 그림이 읽혔는지 볼 수 있게.
- `data/uploads/` 는 gitignore 인지 확인 (그림도 개인 자료).

### 3단계 — 지침 + 검사
- `skills/career_writing.md`(공통)·`portfolio.md`·`recommend_update.md` 에:
  - "근거 자료의 `[그림 N]` 표시는 그 그림이 있던 항목(예: 그 프로젝트의 흐름도 칸) 아래에 **한 줄로 글자 그대로** 남긴다. 지우거나 번호를 바꾸거나 설명으로 바꾸지 않는다."
  - "사람이 mermaid 코드를 주면 그대로 재사용. 새 흐름도는 mermaid(flowchart LR · subgraph · 번호 노드 · `<br/>•` 목록)로, 노드 문구·숫자도 본문 사실 규칙을 따른다. 원본 그림이 있는 자리에는 새로 그리지 않는다."
  - 자소서(`cover_letter`)에는 그림을 넣지 않는다.
- 검사: `portfolio_format` 하네스(`core/harness/validators.py` 383행 — 대리·부장·최종 단계에서 돎)에
  "`career_figures` 가 있는데 원문의 `[그림 N]` 중 결과에 없는 번호" → 실패 메시지 '그림 N 이 빠졌다 — 그 항목 아래에 [그림 N] 한 줄을 남겨라'.
  (`career_facts` 에 넣지 말 것 — 사원 '사실 카드' 단계에서도 돌아 표시를 억지로 요구하게 됨)
- `[그림 12]` 의 12 는 원문에 같은 표시가 있으므로 숫자 검사에 걸리지 않음 — 테스트로 확인.

### 4단계 — 내보내기
- `core/career.py` `body_text()`: ```` ```json ```` 블록(과 언어 표시 없는 끝 JSON)만 지우고 ```` ```mermaid ```` 는 남김.
  → `render_document()`(md) 에 mermaid 가 그대로 들어감. `check_facts`·`check_portfolio` 도 `body_text` 를 쓰니 mermaid 노드 글의 숫자도 사실 검사를 받게 됨 (의도한 동작 — 테스트 추가).
- `core/doc_export.py`:
  - `to_docx(markdown, title, figures: list[str] | None = None)` — 문단·글머리·표 칸 글에서 `[그림 N]` 을 찾아, 글은 남은 부분만 쓰고 그 아래에 `run.add_picture(경로)`.
    너비는 원래 크기를 쓰되 본문 폭(약 16cm, 표 칸이면 칸 폭)을 넘으면 줄임. 파일이 없으면 `[그림 N]` 글자 그대로.
  - `parse_markdown`: ```` ```mermaid ```` 블록을 새 블록 종류 `code`(lang=mermaid)로 → Word 에서는 회색 음영·고정폭 글꼴 상자 + 한 줄 안내 '(mermaid 흐름도 — 이미지로 바꿔 넣으세요: mermaid.live 등)'. JSON 블록은 지금처럼 건너뜀.
  - hwpx: 그림 넣기는 BinData·manifest 가 필요해 범위가 큼 → 이번엔 `[그림 N — 원본 그림은 Word 파일 참고]` 글자로. mermaid 는 코드 글자 그대로.
- `career_ui.py` `build()`(358행): `to_docx(markdown, title, figures=opts.get("career_figures"))`.
- 서버를 다시 켜 되살린 결정 대기 업무는 툴 결과(`career_figures`)가 없음 → 그림 없이 표시만 남음 (화면에 한 줄 안내).

### 5단계 — 확인
- 가짜 docx(표 칸 그림 2개) + 가짜 모델로 `portfolio_update` 끝까지 → 결과 .docx 에 그림 2개(`docx.Document(...).inline_shapes`), .md 에 `[그림 1]`·mermaid 블록 유지.
- 실제 개인 문서로 한 번 실행 (저장소 밖 자료) → Word 에서 흐름도 6개 자리 확인.

## 이미 저장된 결과 (이 PC 의 results/career)

이번 결과는 `[그림 N]` 표시 없이 만들어져 자동으로 그림 자리를 맞출 수 없다 (그림 6개 ↔ 빈 흐름도 칸 8곳).
위 수정 뒤 **업무를 다시 돌리거나**, 그 전까지는 원본 Word 의 그림을 결과 Word 에 직접 복사해 붙인다.

## 진행 기록

- 2026-11-04: 원인 확인·문서 작성 (코드 변경 없음). 같은 날 Word 번호 목록이 문서 전체로 이어지던 문제는 고침(7a4693d).
