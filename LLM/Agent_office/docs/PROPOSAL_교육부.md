# 🎓 교육부 제안서

배우고 싶은 분야(예: 로봇, 보안 관리자, 영어, 이탈리아어)와 지금 지식 수준(예: 신규)을 고르면, 그 내용에 맞춰 가르치는
**선생님**을 만든다. 지식 수준이 오르면 선생님도 그 수준에 맞춰 가르친다.

목적은 두 가지다.

- **직무 전환**: 합격을 목표로 개인 프로젝트, JD 추천, 질문 답변, 커리큘럼, 프로젝트 관리
- **취미·교양**: 일상 활용, 여행 활용, 추가 지식 제안(예: 한국사, 유럽 역사), 커리큘럼, 필수 단어·용어

## 1. 부서와 팀

🎓 교육부(`config/departments.yaml`) 아래에 두 팀을 둔다.

| 팀 | 폴더 | 맡는 일 |
|---|---|---|
| 🧭 직무 전환 교육팀 | `teams/edu_career` | 새 직무 합격이 목표인 학습 |
| 🌍 취미·교양 교육팀 | `teams/edu_hobby` | 일상·여행·교양을 위한 학습 |

부서장(교육부 부장)은 요청을 보고 직무 전환인지 취미인지 판단해 팀에 배정한다. 이미 있는 부서장 배정 흐름
(`building_ui.assign` = 부서 안 팀 고르기 + 범위 확인)을 그대로 쓴다.

## 2. 선생님 = 과목마다 만드는 담당 교사

처음에 **🎓 선생님 만들기**를 한다. 입력은 분야, 목적(직무 전환/취미 — 팀이 정함), 지금 수준, 목표
(예: 6개월 안에 로봇 SW 엔지니어 합격 / 이탈리아 여행 회화), 주당 학습 시간이다.

- 모델이 선생님 프로필을 만든다: 이름, 전문 분야, 경력 설정, 가르치는 방식, 수준별 말투·난이도 규칙, 첫 커리큘럼 뼈대.
- 사람이 확인·수정한 뒤 채택하면 저장한다. 이것이 **과목(Course)** 하나다.
- 선생님은 그 팀 팀장의 페르소나로 들어간다. 팀원(커리큘럼 설계·평가(퀴즈)·자료 조사)이 받친다.
- 한 사람이 여러 과목을 가질 수 있다(로봇 + 이탈리아어).

## 3. 수준

- 공통 5단계: 입문(신규) · 초급 · 중급 · 고급 · 전문. 언어 과목은 CEFR(A1~C1)을 함께 표시한다.
- 시작할 때 **수준 확인 퀴즈**를 볼 수 있다. 모델이 문제를 내고 **채점은 코드**가 한다(객관식 정답 대조). 아니면 사람이 직접 고른다.
- 수준을 올리는 방법은 두 가지다. ① 레벨업 퀴즈 80% 이상, ② 사람이 직접 올림.
- 수준이 바뀌면 선생님 말투·난이도와 **남은** 커리큘럼을 그 수준에 맞게 다시 만든다. 끝난 부분은 기록으로 남긴다.
- 수준 변화 이력을 남긴다.

## 4. 학습 기록

과목마다 저장한다(개인 자료).

- 커리큘럼(주차·단원·완료 여부), 진도 %, 퀴즈 점수, 질문 기록 요약
- 직무 전환: 프로젝트·마일스톤. 취미: 단어장 (2·3단계가 `Course.extras` 에 넣는다)
- 저장 위치는 `data/learning/`(`.gitignore` 의 `**/data/`). 두 팀은 `private: true` 라서 교훈·결정 요약이 공개 `knowledge/` 로 나가지 않는다.

## 5. 2·3단계 (다른 세션)

- 직무 전환 팀 업무: 커리큘럼, 개인 프로젝트 설계·관리(마일스톤·진행 체크·다음 할 일), JD 추천(커리어부 '🎯 업무(JD) 추천'과 연결), 질문 답변, 합격 로드맵
- 취미 팀 업무: 커리큘럼, 필수 단어·용어장(수준별·예문·내려받기), 일상·여행 상황별 표현, 추가 지식(역사·문화) 제안, 질문 답변, 복습 퀴즈

---

## 2·3단계 세션용 안내

1단계(기반)가 만든 공용 틀과 약속이다. 2단계(🧭 직무 전환)와 3단계(🌍 취미·교양)는 **서로 다른 파일만** 고치도록 나눴다 — 두 세션이 동시에 일해도 충돌하지 않는다.

### 1단계가 만든 것

| 파일 | 내용 |
|---|---|
| `core/learning.py` | 수준 표 `LEVELS`(입문(신규)·초급·중급·고급·전문 + CEFR A1~C1) · `TutorProfile` · `Unit` · `Course`(`extras` 포함) · `LearningStore`(`data/learning/<id>.json`, 잠금 + 원자적 교체) · `parse_quiz`/`grade_quiz`/`can_level_up`/`placement_level` · `mark_unit_done`/`next_unit`/`progress_pct` · `change_level`/`plan_remaining` · `course_context` · `register_context` |
| `core/tools/learning_course.py` | 툴 `learning_course` — 업무 입력 `course_id` → 근거 자료 [선생님 프로필 · 수준과 그 규칙 · 커리큘럼·진도 · 최근 기록 · 팀 전용 extras 글], `harness_options` = `level` · `kind` · `course_id` · `language` |
| `core/harness/learning_validators.py` | 공용 검증기 `quiz_format`(퀴즈 JSON·정답이 보기 안에) · `level_fit`(입문·초급에 풀이 없는 용어, 조금 넘으면 경고만) |
| `teams/edu_career/team.yaml` · `teams/edu_hobby/team.yaml` | 조직도(선생님 = 팀장 + 커리큘럼 설계 · 평가(퀴즈) · 자료 조사), `private: true`, `features: [edu_course]`, `skills_from: [_shared/education]`, 공용 업무 3개(`create_tutor` · `quiz` · `ask`) |
| `teams/_shared/education/teaching.md` | 공용 스킬 — 절 `[공통` · `[선생님 만들기` · `[퀴즈` · `[질문 답변` |
| `education_ui.py` | 📚 내 과목 카드 · 과목 고르면 업무 양식 `course_id` 자동 채움 · 단원 완료 체크 · 수준 바꾸기 · 선생님 미리 보기 · 퀴즈 풀기·코드 채점·레벨업 · 채택 시 과목 저장 · **확장 지점** |
| `app.py` | 교육부일 때만 `education_ui` 를 부르는 몇 줄 (import · `is_edu`/`edu_ctx` · 내 과목 패널 · 결과 상자 · 미리 보기 · 채택 · `run["edu_inputs"]`) |
| `tests/test_education.py` | 25건 — 데이터 모델 · 저장소(동시 쓰기) · 채점 · 레벨업 · 진도 · 툴 문맥 · 검증기 · 부서장 배정 · AppTest(과목 만들기 → 퀴즈 → 레벨업 → 커리큘럼 재생성) |

### 어떤 파일에 무엇을 추가하나

| | 2단계 — 🧭 직무 전환 | 3단계 — 🌍 취미·교양 |
|---|---|---|
| 팀 업무 | `teams/edu_career/team.yaml` 의 `tasks` **맨 아래에 추가** | `teams/edu_hobby/team.yaml` 의 `tasks` **맨 아래에 추가** |
| 팀 스킬(선택) | `teams/edu_career/skills/*.md` (새 폴더) | `teams/edu_hobby/skills/*.md` (새 폴더) |
| 코드 계산·검증기·문맥 글 | **새 파일** `core/learning_career.py` | **새 파일** `core/learning_hobby.py` |
| 화면 | **새 파일** `edu_career_ui.py` | **새 파일** `edu_hobby_ui.py` |
| 테스트 | **새 파일** `tests/test_edu_career.py` | **새 파일** `tests/test_edu_hobby.py` |
| 쓰는 extras 키 | `projects` · `roadmap` · `jd_links` | `vocab` · `phrases` · `topics` |

- 새 파일 네 개는 **있으면 자동으로 읽힌다** — `core/learning.py` 의 `load_extensions()` 가 `core.learning_career` · `core.learning_hobby` 를,
  `education_ui.py` 맨 아래가 `edu_career_ui` · `edu_hobby_ui` 를 import 한다. 그 안에서 등록만 하면 되고 공용 파일은 고치지 않는다.
- 새 검증기는 그 모듈 안에 `@register` 로 둔다 (`core/harness/validators.py` 를 두 세션이 함께 고치지 않게). `core/harness/learning_validators.py` 가 import 될 때 `load_extensions()` 를 불러 함께 등록된다.
- 업무에서 과목을 쓰려면 양식에 `course_id` 칸(`{key: course_id, label: "과목 (📚 내 과목에서 고르면 자동)", required: true}`)과
  `fixed: {edu_kind: career|hobby, edu_task: <업무 id>}` 를 둔다. 그러면 ① 카드에서 과목을 고르면 칸이 채워지고 ② 툴이 과목 문맥을 근거 자료로 주고 ③ 화면이 업무 종류를 안다.

### 건드리면 안 되는 공용 파일 (허용되는 변경만)

| 파일 | 허용 | 금지 |
|---|---|---|
| `core/learning.py` | `EXTRAS_KEYS` 에 **한 줄 추가**(새 extras 키) | `Course`·`Unit`·`TutorProfile` 칸 추가·이름 바꾸기, `LEVELS`·`LEVEL_UP_PCT` 변경, 함수 동작 변경 |
| `education_ui.py` | 없음 — 확장 지점(`register_*`)을 **다른 파일에서** 부른다 | 공용 업무(`create`·`quiz`·`ask`) 처리 변경, 카드·패널 구조 변경 |
| `core/tools/learning_course.py` · `core/harness/learning_validators.py` | 없음 (팀 전용 문맥은 `register_context`, 검증기는 자기 모듈) | 문맥 글 형식·`harness_options` 키 변경 |
| 두 `team.yaml` 의 공용 업무 3개 · 조직도 · `personas.team_lead` · `harness` | 2·3단계 업무에 필요한 검증기는 그 **업무의** `validators` 에 적기 | 공용 업무 3개·조직도 id 변경 (테스트가 확인), `private: true` 끄기 |
| `config/departments.yaml` 교육부 · `app.py` 교육부 줄 | 없음 | — |

공용 파일을 꼭 고쳐야 하면 먼저 그 이유를 부모 세션에 묻는다 (두 세션이 같은 줄을 고치면 합칠 때 충돌한다).

### 확장 지점 (등록 함수)

```python
# core/learning_career.py (2단계) — core/learning_hobby.py 도 같은 모양
from core import learning
from core.harness.base import Validator, register

# 'projects' · 'roadmap' · 'jd_links' 는 learning.EXTRAS_KEYS 에 이미 있다 — 새 키가 필요할 때만 learning.py 에 한 줄 추가


def projects_text(course: learning.Course) -> str:
    """extras['projects'] → 근거 자료 글 (툴이 과목 문맥 아래 '### 🗂 개인 프로젝트·마일스톤'으로 붙인다)."""
    rows = ["| 프로젝트 | 상태 | 다음 마일스톤 |", "|---|---|---|"]
    for p in course.extras.get("projects", []):
        nxt = next((m["title"] for m in p.get("milestones", []) if not m.get("done")), "-")
        rows.append(f"| {p['title']} | {p.get('status', '')} | {nxt} |")
    return "\n".join(rows)


learning.register_context("projects", projects_text)


@register
class ProjectPlanValidator(Validator):
    name = "project_plan"
    description = "개인 프로젝트 계획 JSON — 마일스톤마다 제목·기한 (교육부 직무 전환)"

    def check(self, answer, options):
        data = learning.find_json(answer, "projects")
        return self.ok("프로젝트 계획 형식 통과") if data else self.fail('```json {"projects": [...]} 블록이 없다')
```

```python
# edu_career_ui.py (2단계) — edu_hobby_ui.py 도 같은 모양
import streamlit as st

import education_ui
from core import learning


def projects_section(ctx: education_ui.Ctx, course: learning.Course) -> None:
    """과목 '📖 자세히' 안 — 프로젝트·마일스톤 체크."""
    st.markdown("**🛠 개인 프로젝트**")
    ...  # 고칠 때는 ctx.store.update(course.id, lambda c: ...)


def on_project_adopt(ctx, run, decision) -> str | None:
    """'🛠 개인 프로젝트 설계' 채택 → extras['projects'] 에 저장. 돌려준 글은 결과 아래 초록 상자로 보인다."""
    data = learning.find_json(decision.final_answer, "projects")
    if not data:
        return None
    cid = run["edu_inputs"]["course_id"]
    ctx.store.update(cid, lambda c: c.extras.__setitem__("projects", data["projects"]))
    return f"🛠 프로젝트 {len(data['projects'])}개를 과목에 저장했습니다."


education_ui.register_section("career", projects_section)          # 📖 자세히 안에 칸 추가
education_ui.register_quick("career", "🛠 프로젝트", "project_plan")   # 과목 카드에 버튼 추가 (누르면 그 업무 + course_id 자동)
education_ui.register_adopt("project_plan", on_project_adopt)       # 그 업무를 채택할 때 과목에 기록
# education_ui.register_preview("project_plan", fn)                 # 그 업무 결과 아래 상자 (선택) — fn(ctx, run, final_text)
```

- `register_quick` 의 업무 id 는 team.yaml 업무 `id`, `register_preview` / `register_adopt` 의 키는 업무의 `fixed.edu_task` 다 (둘을 같게 두면 헷갈리지 않는다).
  공용 `create` · `quiz` · `ask` 는 다시 등록할 수 없다 (ValueError).
- 수준을 바꾸는 일은 화면이 한다 (`education_ui.change_level_now`). 팀 업무가 수준을 바꿔야 하면 `learning.change_level(...)` 만 쓴다 — 이력이 남는다.
- 단원 완료는 `learning.mark_unit_done(course, unit_id)`, 다음 단원은 `learning.next_unit(course)`, 진도는 `course.progress`.
- 과목 기록은 개인 자료다 — `data/learning/` 밖(저장소 안)에 쓰지 않는다. 샘플·테스트 과목은 가상 인물만.

### 업무 하나를 추가하는 예시 (2단계 — 개인 프로젝트 설계)

```yaml
# teams/edu_career/team.yaml 의 tasks 맨 아래에
  - id: project_plan
    title: 🛠 개인 프로젝트 설계
    description: >-
      과목을 고르면 지금 수준과 목표 직무에 맞는 포트폴리오용 개인 프로젝트를 설계하고 마일스톤을 나눕니다. 채택하면 과목에 저장됩니다.
    fields:
      - {key: course_id, label: 과목 (📚 내 과목에서 고르면 자동), required: true}
      - {key: weeks, label: 기간 (주), type: select, options: ["4", "8", "12"]}
      - {key: interest, label: 관심 주제 (선택), placeholder: "예: 로그 분석 자동화"}
    fixed: {edu_kind: career, edu_task: project_plan}
    question: |
      과목 {course_id} 학습자에게 {weeks}주짜리 개인 프로젝트를 설계해줘.
      관심 주제: {interest}
    goal: "지금 수준에서 끝낼 수 있고 목표 직무 합격에 도움이 되는 개인 프로젝트 1~2개 + 주차별 마일스톤"
    success_criteria:
      - "근거 자료의 지금 수준·다음 단원과 이어진다"
      - "마일스톤마다 끝났는지 확인할 수 있는 결과물을 쓴다"
      - '답변 맨 끝에 ```json {"projects": [...]} 블록 1개'
    output_format: "## 프로젝트 → ## 마일스톤 표 → ```json 1개"
    instructions: |
      [JSON] {"projects": [{"id": "P1", "title": "…", "goal": "…", "status": "계획",
               "milestones": [{"id": "M1", "title": "…", "due": "1주차", "done": false}]}]}
    validators: [project_plan]
```

그다음 `core/learning_career.py`(검증기 `project_plan` · `register_context("projects", …)`)와 `edu_career_ui.py`(`register_quick` · `register_adopt` · `register_section`)를 위 예시처럼 만들고,
`tests/test_edu_career.py` 에 검증기·문맥 글·채택 저장 테스트를 둔다 (가짜 모델 `use_script` 만, 샘플 과목은 가상 인물).
`tests/test_education.py` 는 공용 업무 3개가 **맨 앞**에 있는지와 모든 업무에 `fixed.edu_kind` 가 있는지만 확인하므로, 뒤에 업무를 더해도 고칠 필요가 없다 (이 파일은 고치지 않는다).

### 확인할 것

- 전체 `pytest` (특히 `tests/test_education.py` · `tests/test_scope.py` — 새 업무 예시·scope 낱말이 다른 팀 범위를 침범하지 않는지).
- 팀 `scope.keywords` 를 더하면 `tests/test_scope.py::test_every_team_has_scope_and_own_examples` 가 모든 팀 예시를 다시 확인한다.
  여행부(`이탈리아` 같은 지명)·커리어부(`자소서`·`JD`) 낱말과 겹치지 않게 '배우기' 쪽 낱말을 쓴다.
- JD 추천(2단계)은 커리어부 '🎯 업무(JD) 추천'(`teams/career`, `core/career.py score_jds`)을 **불러 쓰고 고치지 않는다** — 그 팀은 다른 세션 담당이다.
- 미리 보기는 8501 이 아닌 포트로. `streamlit.web.bootstrap` 으로 띄우면 `load_config_options(flag_options={"server_port": …})` 를 먼저 불러야 포트가 먹는다.

---

## 진행 기록

### 3단계 — 🌍 취미·교양 교육팀 전문 업무 (2026-11-05, 브랜치 `claude/edu-hobby-814269`)

| 업무 (`team.yaml` id = `fixed.edu_task`) | 채택하면 | 검증기 (`core/learning_hobby.py`) |
|---|---|---|
| 📅 커리큘럼 짜기 (`hobby_curriculum`) | 과목의 남은(안 끝낸) 단원을 교체 — 모델은 1주부터 쓰고 코드가 끝낸 주 다음으로 밂 | `hobby_curriculum` — 주차·단원 형식, 주마다 시간 합 ≤ 주당 시간(0.1시간 정수), 언어는 네 영역 |
| 📖 필수 단어·용어장 (`vocab`) | `extras["vocab"]` 에 쌓음, 같은 단어는 합침 | `vocab_list` — 칸·중복, 수준 범위는 경고만 |
| 🗣 일상·여행 상황별 표현 (`phrases`) | `extras["phrases"]` | `phrase_list` |
| 🧭 더 배울 것 (`topics`) | `extras["topics"]` + ➕ 새 과목으로(공용 선생님 만들기 양식 미리 채움) | `topic_list` |
| 🔁 복습 퀴즈 (`review`) | 채점할 때 바로 저장(퀴즈 기록 + 항목별 다음 복습일) | `review_quiz` — 공용 퀴즈 형식 + 문제마다 `ref` 가 과목에 있음 |

- 파일: `teams/edu_hobby/team.yaml`(tasks 맨 아래 5개 + `harness.final_only` 에 검증기 5개) · `teams/edu_hobby/skills/hobby_teaching.md` · `core/learning_hobby.py` · `edu_hobby_ui.py` · `tests/test_edu_hobby.py`(17건).
- extras 항목 모양은 learning.py 머리말의 키에 이 팀 칸을 더함 — vocab: `reading·pos·example_ko·related` + 복습 칸 `known·box·due·last_review·reviews·wrong`, phrases: `reading·dialogue·tip` + 복습 칸, topics: `first_unit·field·added_at`. `EXTRAS_KEYS` 는 고치지 않음(키 3개는 1단계에 이미 있음).
- 📖 자세히 칸(`register_section`): 단어장 표('외움' 체크 = `st.data_editor`, 저장할 때마다 표 키를 바꿔 줄 순서가 어긋나지 않게) · 검색 · CSV(Anki)·.md · 표현 모음 · ⚡ 바로 복습(모델 없이) · 추가 주제.
- **공용 파일 변경 1곳 (이유)**: `education_ui.result_text` 는 '📝 퀴즈' 결과만 정답 JSON 을 감춤 → 3단계 복습 퀴즈의 정답이 결과 상자에 그대로 보임.
  `ANSWER_TASKS = {"quiz"}` + `register_answer_hidden(edu_task)` 로 바꾸고 `edu_hobby_ui` 가 `review` 를 등록 (공용 퀴즈 동작은 그대로, 머리말 확장 지점 목록에 한 줄).
  2단계도 퀴즈형 업무가 있으면 `education_ui.register_answer_hidden("<업무>")` 한 줄이면 됨.
- **2단계도 확인할 것 — `harness.final_only`**: 업무 `validators` 는 팀원 초안(자료 조사·커리큘럼 설계)마다 돌기 때문에 JSON 형식 검증기는 그대로 두면 팀원이 JSON 없다고 다시 쓰게 됨.
  공용 `quiz_format`·`json_keys` 처럼 팀 `harness.final_only` 에도 이름을 적어 최종 답변에만 돌게 함 (`tests/test_education.py` 의 final_only 확인은 '포함'이라 통과).
- **✈️ 여행부 연결 (제안만 — 여행팀 파일은 안 고침)**:
  ① 🗣 표현 미리 보기에 '✈️ 이 나라 여행 계획 맡기기' 버튼 → `travel_abroad` 업무 `plan` 의 `destination` 칸을 과목 나라로 채워 엶 (교육부 `open_task` 와 같은 방식, 부서가 달라 `dept`·`team` 쿼리도 바꿔야 함).
  ② 반대로 여행팀 결과 아래 '🗣 현지 표현 배우기' → 그 나라 말 취미 과목이 있으면 `phrases` 업무를 `situation` = 공항/호텔로 채워 엶, 없으면 선생님 만들기 양식을 채움.
  ③ 여행 날짜(`travel_calendar` 의 출국일)를 커리큘럼 `weeks` 기본값으로 — '출국까지 N주' 에 맞춰 짜기. ①·② 는 여행팀 세션과 상의, ③ 은 여행팀 run 기록을 읽기만 하면 됨.
- AppTest 주의: 채택 뒤 다시 그리기에서 결정 양식의 키 없는 칸(만족도 슬라이더)이 트리에 남아, 그 화면에서 더 누르면 KeyError — 새 AppTest 로 이어 확인했음 (실제 브라우저는 문제없음).

