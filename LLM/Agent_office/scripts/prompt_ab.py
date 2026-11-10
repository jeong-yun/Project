"""팀장 프롬프트 '2단계 방식' 비교 시험 — 현재 프롬프트와 시험 프롬프트 중 어느 쪽 결과가 더 좋은지 실제 모델로 잰다.

비교하는 방식 (팀장의 마지막 검토·통합 호출만 바꾼다. 팀원·모델·설정은 같다)
  C  : 지금 프롬프트 그대로
  T1 : 같은 대화에서 두 번 묻기 — ① "이 일을 그 분야 최고인 사람이 한다면 어떤 도구를 사용할지 알려줘"
       → 그 답 뒤에 ② "방금 말한 도구 그대로 사용해서 써서 줘" (팀장 호출 1번 추가)
  T2 : 팀장 시스템 프롬프트 끝에 한 문단을 접어 넣기 (호출 수 그대로)

어떻게 공정하게 비교하나
  팀원 초안과 중간 관리자 검토는 앱의 워크플로우(core/workflow.py)를 그대로 한 번만 돌려 저장하고(draft),
  팀장의 마지막 호출만 방식별로 다시 돌린다(lead). 세 방식이 똑같은 하위 결과를 받는다.
  교훈(저장소)·웹 검색·부장 비평(자가 교정)·하네스 재시도는 세 방식 모두 뺐다 — 프롬프트 차이만 보려고.

실행 (Agent_office 폴더에서, 실제 모델을 부르므로 비용이 든다 — 먼저 --fake 로 끝까지 도는지 확인)
    python scripts/prompt_ab.py plan                      # 호출 없음: 누가 어느 모델을 쓰는지, 예상 호출 수
    python scripts/prompt_ab.py draft --fake              # 가짜 모델로 팀원 초안 단계
    python scripts/prompt_ab.py lead --fake               # 가짜 모델로 팀장 단계 (C·T1·T2)
    python scripts/prompt_ab.py grade --fake              # 가짜 모델로 눈가림 채점
    python scripts/prompt_ab.py draft --env <.env 경로>   # 실제 모델 (키는 그 파일에서 환경 변수로만 읽는다)
    python scripts/prompt_ab.py lead  --env <.env 경로>
    python scripts/prompt_ab.py measure                   # 호출 없음: 하네스·점검표·글자 수·코드 실행 검사
    python scripts/prompt_ab.py grade --env <.env 경로>   # 다른 회사 모델이 방식을 가린 채 채점 (순서를 뒤집어 한 번 더)
    python scripts/prompt_ab.py table                     # 호출 없음: 비교표
    옵션: --only <질문 id> · --rest <provider:model> (오늘 한도가 찬 모델을 후보 맨 뒤로) · --graders · --lead-model

결과는 results/prompt_ab/ 에 쌓인다 (.gitignore 대상 — 커밋하지 않는다). 가짜 모델 결과는 results/prompt_ab/_fake/.
호출 상한(--max-calls, --max-usd)을 넘으면 새 호출을 시작하지 않고 멈춘다. 이미 만든 결과는 다시 만들지 않는다(--force 로 다시).
"""

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core import claude_cli, governance, harness, llm_factory, usage  # noqa: E402
from core.agent import OrgAgent  # noqa: E402
from core.brief import DesiredState  # noqa: E402
from core.config import load_models, load_ranks, load_settings, load_team, load_verification, team_org  # noqa: E402
from core.harness.base import extract_code_blocks  # noqa: E402
from core.schemas import AgentAnswer, ModelSpec, RankModels  # noqa: E402
from core.tasks import build_request  # noqa: E402
from core.workflow import build_office_graph, stream_office  # noqa: E402

VARIANTS = ("C", "T1", "T2")
# ── 시험 프롬프트에 넣는 문장 ─────────────────────────────────────────
T1_ASK_TOOLS = "이 일을 그 분야 최고인 사람이 한다면 어떤 도구를 사용할지 알려줘"  # 사용자가 준 문장 그대로
T1_ASK_TOOLS_NOTE = "(지금은 도구만 답하고, 결과물은 아직 쓰지 않는다.)"  # 첫 답에서 결과물까지 써 버리지 않게 붙인 한 줄
T1_ASK_WRITE = "방금 말한 도구 그대로 사용해서 써서 줘"  # 사용자가 준 문장 그대로
T2_RULE = ("[작성 방법]\n먼저 이 일의 최고 전문가가 쓸 도구(방법·틀)를 정하고, 그 도구를 그대로 써서 작성하라. "
           "정한 도구는 analysis 맨 앞에 '사용 도구: …' 한 줄로 적는다.")

# ── 시험 질문 (회사·개인 정보 없는 일반 질문) — check 는 질문이 요구한 항목 점검표 (이름, 정규식) ──
QUESTIONS = [
    {
        "id": "general_onboarding", "team": "general",
        "question": "신입 개발자 3명이 한꺼번에 들어오는 10명 규모 백엔드 팀의 4주 온보딩 계획을 만들어줘",
        "brief": {
            "goal": "다음 달 입사하는 신입 3명에게 그대로 적용할 수 있는 4주 온보딩 계획",
            "success_criteria": [
                "1주차부터 4주차까지 주차별 목표·주요 활동·점검 방법을 표로 쓴다",
                "멘토(사수) 배정 방식과 멘토가 매주 할 일을 쓴다",
                "온보딩이 실패하는 흔한 위험을 2개 이상 쓰고 각각의 대응 방안을 쓴다",
                "4주 뒤 온보딩이 성공했는지 판단할 측정 가능한 기준을 3개 이상 쓴다",
            ],
            "constraints": ["기존 팀원의 추가 부담은 1인당 주 3시간 이내", "유료 교육·외부 강의 없이"],
            "output_format": "한 줄 요약 → 주차별 표 → 멘토 운영 → 위험과 대응 → 성공 기준 → 다음 행동",
        },
        "check": [
            ("1~4주차 모두", r"(?s)(?=.*1\s*주차)(?=.*2\s*주차)(?=.*3\s*주차)(?=.*4\s*주차)"),
            ("주차별 표", r"(?m)^\|.*주차.*\|"), ("멘토 배정·할 일", r"멘토"),
            ("위험과 대응", r"(?s)(위험|리스크).*대응"), ("측정 가능한 성공 기준", r"성공\s*(판단\s*)?기준"),
            ("부담 주 3시간 제약", r"3\s*시간"), ("다음 행동", r"다음\s*행동"),
        ],
    },
    {
        "id": "general_code", "team": "general",
        "question": "CSV 접속 로그에서 IP 별 요청 수 상위 N개를 뽑는 파이썬 함수를 만들어줘",
        "brief": {
            "goal": "바로 붙여 넣어 실행할 수 있는 파이썬 함수와 사용 예",
            "success_criteria": [
                "함수 이름은 top_ips, 인자는 (path, n=10), 반환값은 (ip, 요청 수) 튜플의 리스트이며 요청 수 내림차순, "
                "요청 수가 같으면 IP 문자열 오름차순이다",
                "표준 라이브러리만 사용한다",
                "IP 는 CSV 의 'ip' 열에서 읽고, 헤더가 없으면 첫 번째 열을 IP 로 본다. 빈 줄과 IP 가 빈 행은 건너뛴다",
                "```python 코드 블록이 문법 오류 없이 실행된다 (하네스 검증)",
            ],
            "constraints": ["함수 하나 + 사용 예 1개", "파일은 UTF-8"],
            "output_format": "코드 블록 → 사용 예 → 주의사항",
        },
        "check": [("python 코드 블록", r"```python"), ("함수 top_ips", r"def\s+top_ips\s*\("), ("사용 예", r"사용\s*예"),
                  ("주의사항", r"주의")],
    },
    {
        "id": "travel_busan", "team": "travel_domestic", "task": "plan",
        "inputs": {"destination": "부산", "transport": "KTX·기차", "start_date": "2026-12-04", "end_date": "2026-12-06",
                   "departure": "서울", "people": "성인 2, 초등학생 1", "budget": "총 80만원", "style": "맛집·바다 위주, 걷기 적게"},
        "check": [
            ("1~3일차 모두", r"(?s)(?=.*1\s*일차)(?=.*2\s*일차)(?=.*3\s*일차)"), ("날짜·요일", r"12[월/.\-]\s*0?4.*금|금.*12[월/.\-]\s*0?4"),
            ("KTX 소요 시간·요금", r"(?s)KTX.*(시간|분).*원"), ("숙소 지역·가격대", r"(?s)숙소.*원"),
            ("예산 표와 합계", r"합계"), ("날씨·주의사항", r"날씨"), ("아이 동반 고려", r"초등학생|아이|어린이"),
        ],
    },
    {
        "id": "travel_daytrip", "team": "travel_domestic",
        "question": "서울에서 기차로 다녀올 수 있는 겨울 당일치기 여행지 3곳을 추천해줘",
        "brief": {
            "goal": "이번 주말 바로 고를 수 있는 겨울 당일치기 후보 비교와 추천 1곳의 하루 일정",
            "success_criteria": [
                "후보 3곳마다 이동 수단·편도 소요 시간·대표 코스를 쓴다",
                "후보마다 겨울에 가기 좋은 이유를 쓴다",
                "3곳을 비교 표로 정리하고, 1곳을 추천해 그 하루 일정을 시간대별로 쓴다",
            ],
            "constraints": ["편도 2시간 이내", "대중교통만", "성인 2명"],
            "output_format": "비교 표 → 추천 1곳 일정 → 주의사항",
        },
        "check": [("비교 표", r"(?m)^\|.*\|.*\|"), ("편도 소요 시간", r"편도"), ("겨울 이유", r"겨울"),
                  ("추천 1곳", r"추천"), ("시간대별 일정", r"\d{1,2}\s*[:시]\s*\d{0,2}"), ("주의사항", r"주의")],
    },
    {
        "id": "gov_control", "team": "security_governance", "task": "control_check",
        "inputs": {"framework": "ISMS-P", "scope": "접근통제·로그 관리", "use_sample": True},
        "check": [("판정 표", r"(?m)^\|.*판정.*\|"), ("증적 인용", r"증적"), ("개선안", r"개선"), ("json 블록 1개", r"```json")],
    },
    {
        "id": "gov_drift", "team": "security_governance", "task": "drift_check",
        "inputs": {"framework": "ISMS-P", "use_sample_drift": True},
        "check": [("영향 표", r"(?m)^\|.*영향.*\|"), ("조치", r"조치|되돌"), ("json 블록 1개", r"```json")],
    },
]

# ── 눈가림 채점 ──────────────────────────────────────────────────────
GRADERS = ("openai:gpt-5.4", "google:gemini-3.8-flash")  # 팀장(Claude)과 다른 회사 모델
LABELS = ("가", "나", "다")
CRITERIA = {"fulfil": "요구 충족", "accuracy": "정확성·근거", "structure": "구조·읽기 쉬움", "usefulness": "쓸모",
            "no_fabrication": "지어낸 내용 없음"}


class AnswerGrade(BaseModel):
    label: str = Field(description="답변 표시 (가/나/다)")
    fulfil: int = Field(description="요구 충족 1~5 — 완료 기준·제약·결과 형식을 빠짐없이 지켰는가")
    accuracy: int = Field(description="정확성과 근거 1~5 — 사실·숫자가 맞고, 근거·출처 또는 '확인 필요' 표시가 있는가")
    structure: int = Field(description="구조와 읽기 쉬움 1~5 — 결론이 먼저 보이고 군더더기 없이 읽히는가")
    usefulness: int = Field(description="쓸모 1~5 — 읽는 사람이 그대로 가져다 쓸 수 있는가")
    no_fabrication: int = Field(description="지어낸 내용 없음 1~5 — 5 = 근거 없는 숫자·사실·출처가 없음, 1 = 많음")
    note: str = Field(description="이 답변의 가장 큰 장점과 단점을 한두 문장으로")


class GradeSheet(BaseModel):
    analysis: str = Field(description="점수를 매기기 전에 세 답변의 차이를 3문장 이내로 분석")
    grades: list[AnswerGrade] = Field(description="답변마다 하나씩 (가·나·다 모두)")
    ranking: list[str] = Field(description="좋은 답변부터 순서대로 답변 표시")


# ── 호출 상한·기록 ───────────────────────────────────────────────────


class BudgetExceeded(BaseException):
    """상한을 넘어 새 호출을 막음 — '모델 실패'로 삼켜져 대안 모델로 넘어가지 않게 BaseException."""


class Budget:
    """모델 호출 시도·끝난 호출을 calls.jsonl 에 한 줄씩 남기고, 상한을 넘으면 새 호출을 막는다. 키 정보는 남기지 않는다."""

    def __init__(self, path: Path, max_calls: int, max_usd: float):
        self.path, self.max_calls, self.max_usd = path, max_calls, max_usd
        self.calls, self.usd, self.lock = 0, 0.0, threading.Lock()
        for row in read_jsonl(path):
            self.calls += row["type"] == "attempt"
            self.usd += row.get("cost_usd") or 0.0 if row["type"] == "done" else 0.0

    def _write(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 1), **row}, ensure_ascii=False) + "\n")

    def attempt(self, spec: ModelSpec) -> None:
        with self.lock:
            if self.calls >= self.max_calls or self.usd >= self.max_usd:
                raise BudgetExceeded(f"상한 도달 — 호출 {self.calls}/{self.max_calls}회, 추정 ${self.usd:.2f}/${self.max_usd}")
            self.calls += 1
            self._write({"type": "attempt", "provider": spec.provider, "model": spec.model,
                         "run": usage.current("run_id"), "purpose": usage.current("purpose"), "agent": usage.current("agent_id")})

    def done(self, event: usage.UsageEvent) -> None:
        with self.lock:
            self.usd += event.cost_usd or 0.0
            self._write({"type": "done", "provider": event.provider, "model": event.model, "run": event.run_id,
                         "purpose": event.purpose, "agent": event.agent_id, "in": event.input_tokens,
                         "out": event.output_tokens, "cost_usd": event.cost_usd,
                         "api_equiv_usd": api_equiv(event.provider, event.model, event.input_tokens, event.output_tokens)})


def api_equiv(provider: str, model: str, tokens_in, tokens_out) -> float | None:
    """구독(claude_cli)은 청구가 0 이라, 같은 호출을 API 로 했다면 얼마인지 가격표로 환산한 값 (방식끼리 비교용)."""
    return usage.estimate_cost("anthropic" if provider == "claude_cli" else provider, model, tokens_in or 0, tokens_out or 0)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── 준비 ─────────────────────────────────────────────────────────────


class Ctx:
    def __init__(self, args):
        self.fake = bool(getattr(args, "fake", False))
        self.out = ROOT / "results" / "prompt_ab" / ("_fake" if self.fake else "")
        self.ranks, self.models, self.settings = load_ranks(), load_models(), load_settings()
        self.verification = load_verification()
        self.force = bool(getattr(args, "force", False))
        only = getattr(args, "only", None)
        self.questions = [q for q in QUESTIONS if not only or q["id"] in only.split(",")]
        self.lead_model = getattr(args, "lead_model", None)
        if self.fake:
            for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "UPSTAGE_API_KEY"):
                os.environ[key] = "fake"
            claude_cli.find_executable = lambda: None  # 구독 CLI 를 부르지 않는다
            claude_cli._status_cache.clear()
        elif getattr(args, "env", None):
            load_dotenv(args.env)  # 키는 이 프로세스의 환경 변수로만 — 화면·파일에 남기지 않는다
        for key in filter(None, (getattr(args, "rest", None) or "").split(",")):
            # 오늘 한도가 이미 찬 모델 — 앱이 한도 오류를 본 뒤 하는 것처럼 후보 맨 뒤로 미룬다 (헛도는 시도·시간 초과를 줄임)
            llm_factory._cooldown[key] = time.monotonic() + 24 * 3600
            llm_factory._cooldown_why[key] = "사용 한도 소진"
        self.budget = Budget(self.out / "calls.jsonl", getattr(args, "max_calls", 150), getattr(args, "max_usd", 5.0))

    def install(self) -> None:
        """모든 모델 호출이 상한 검사·기록을 거치게 한다 (가짜 모드면 가짜 모델로 바꾼다)."""
        real_create, script = llm_factory.create_chat_model, fake_script() if self.fake else None

        def create(spec, models):
            self.budget.attempt(spec)
            if script is not None:
                from tests.fakes import FakeOfficeModel

                return FakeOfficeModel(script=script)
            return real_create(spec, models)

        llm_factory.create_chat_model = create
        usage.set_sink(self.budget.done)


def fake_script():
    from core.schemas import Assignment, Delegation
    from tests.fakes import Script, make_review

    def delegation(messages):
        ids = re.findall(r"- id: (\S+) /", str(messages[-1].content))
        return Delegation(analysis="가짜 분석", assignments=[Assignment(child_id=i, subtask=f"{i} 업무") for i in ids])

    def grade(messages):
        labels = re.findall(r"(?m)^### 답변 (\S+)$", str(messages[-1].content))
        return GradeSheet(analysis="가짜 채점", ranking=labels, grades=[
            AnswerGrade(label=x, fulfil=3, accuracy=3, structure=3, usefulness=3, no_fabrication=3, note="가짜") for x in labels])

    return Script(text="가짜 글 — 1일차 2일차 3일차 예산 숙소 교통", Delegation=delegation, GradeSheet=grade,
                  ReviewResult=make_review(answer="가짜 통합 답변 — 1일차 2일차 3일차 예산 숙소 교통 판정 증적 개선 영향 조치"))


def request_for(q: dict, team) -> tuple[str, dict]:
    """질문 → (질문 글, stream_office 인자). 브리프를 직접 채워 '요청 정리' 호출을 없앤다 (점검표가 흔들리지 않게)."""
    if "task" in q:
        req = build_request(next(t for t in team.tasks if t.id == q["task"]), q["inputs"])
        return req.question, {"brief": req.brief, "auto_brief": False, "tool_inputs": req.tool_inputs,
                              "extra_validators": req.validators, "harness_options": req.harness_options}
    brief = DesiredState(context="", assumptions=[], **q["brief"])
    return q["question"], {"brief": brief, "auto_brief": False}


def build(team, ctx: Ctx):
    """앱과 같은 함수로 그래프를 만들고, 그때 만들어진 에이전트들을 함께 돌려준다 (저장소 없음 → 교훈 없음)."""
    org, found, init = team_org(team, ctx.ranks), {}, OrgAgent.__init__

    def spy(self, node, *a, **k):
        init(self, node, *a, **k)
        found[node.id] = self

    OrgAgent.__init__ = spy
    try:
        graph = build_office_graph(org, ctx.ranks, team, ctx.models, ctx.settings, ctx.verification)
    finally:
        OrgAgent.__init__ = init
    return org, graph, found


# ── draft: 팀원 초안·중간 검토를 한 번만 (팀장의 마지막 호출 직전에 멈춘다) ─────────────


class StopAtLead(Exception):
    """팀장의 검토·통합 프롬프트를 만들려는 순간 — 여기서 멈추고 팀장이 받을 입력을 저장한다."""


def step_row(s: AgentAnswer) -> dict:
    return {"node": s.node_id, "kind": s.kind, "model": s.model.label, "fallback": s.is_fallback, "sec": s.elapsed_sec,
            "in": s.input_tokens, "out": s.output_tokens, "attempts": s.attempts, "chars": len(s.content),
            "errors": s.errors, "harness_ok": all(r.passed for r in s.harness), "log": s.verification_log}


def run_draft(q: dict, ctx: Ctx) -> None:
    path = ctx.out / q["id"] / "phase_a.json"
    if path.exists() and not ctx.force:
        print(f"- {q['id']}: 이미 있음 — 건너뜀")
        return
    team = load_team(q["team"])
    org, graph, _ = build(team, ctx)
    question, kwargs = request_for(q, team)
    captured: dict = {}
    steps: list[AgentAnswer] = []
    tools: list = []
    original = OrgAgent.review_messages

    def hook(self, question, task, child_results, web_context="", lessons=None, rewrite_answer=True):
        if self.node.id == org.id:
            captured.update(question=question, task=task, children=child_results, web=web_context, rewrite=rewrite_answer)
            raise StopAtLead()
        return original(self, question, task, child_results, web_context, lessons, rewrite_answer)

    OrgAgent.review_messages = hook
    started = time.perf_counter()
    try:
        with usage.scope(team_id=team.id, run_id=q["id"], purpose="A:팀원"):
            for kind, payload in stream_office(graph, org, question, org.id, use_web=False, **kwargs):
                if kind == "tools":
                    tools = payload
                elif kind == "step":
                    steps.append(payload)
                    print(f"   · {payload.node_id:<14} {payload.kind:<9} {payload.model.label:<34} {payload.elapsed_sec:>6}s"
                          f" {len(payload.content):>6}자{' ⚠️ ' + str(len(payload.errors)) + '건 실패' if payload.errors else ''}")
    except Exception:  # noqa: BLE001 — StopAtLead (그래프가 감싸서 올려도 저장된 입력이 있으면 정상)
        if not captured:
            raise
    finally:
        OrgAgent.review_messages = original
    options = {**team.harness.options, **kwargs.get("harness_options", {})}
    for t in tools:  # core/workflow.py harness_plan 과 같은 순서
        options = {**t.harness_options, **options}
    validators = list(dict.fromkeys([*team.harness.validators, *kwargs.get("extra_validators", [])]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "id": q["id"], "team": team.id, "user_question": question, "question": captured["question"], "task": captured["task"],
        "web": captured["web"], "rewrite": captured["rewrite"], "validators": validators, "options": options,
        "children": [c.model_dump(mode="json") for c in captured["children"]],
        "steps": [step_row(s) for s in steps], "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"- {q['id']}: 팀원 단계 {len(steps)}개 · {time.perf_counter() - started:.0f}초 · 팀장이 받을 하위 결과 {len(captured['children'])}개 저장")


# ── lead: 팀장의 마지막 호출을 방식별로 ──────────────────────────────────


def lead_agent(team, ctx: Ctx) -> OrgAgent:
    org, _, agents = build(team, ctx)
    agent = agents[org.id]
    if ctx.lead_model:
        provider, _, model = ctx.lead_model.partition(":")
        spec = ModelSpec(provider=provider, model=model)
    else:
        spec = llm_factory.available_candidates(agent.rank_models, ctx.models)[0][0]
    agent.rank_models = RankModels(primary=spec)  # 대안 모델로 넘어가지 않는다 — 방식끼리 같은 모델로만 비교
    return agent


def run_variant(agent: OrgAgent, cap: dict, variant: str) -> tuple[AgentAnswer, AgentAnswer | None]:
    """방식 하나 = 팀장의 검토·통합 1회 (T1 은 앞에 '도구 묻기' 1회). (최종 답변, T1 의 첫 답)"""
    children = [AgentAnswer.model_validate(c) for c in cap["children"]]
    system, human = agent.review_messages(cap["question"], cap["task"], children, cap["web"], None, cap["rewrite"])
    if variant == "C":
        return agent.review([system, human], children, cap["rewrite"]), None
    if variant == "T2":
        return agent.review([SystemMessage(f"{system.content}\n\n{T2_RULE}"), human], children, cap["rewrite"]), None
    ask = [system, HumanMessage(f"{human.content}\n\n{T1_ASK_TOOLS}\n{T1_ASK_TOOLS_NOTE}")]
    tools = agent.draft(ask)  # 형식 없는 글로 도구를 답하게 한다
    return agent.review([*ask, AIMessage(tools.content), HumanMessage(T1_ASK_WRITE)], children, cap["rewrite"]), tools


def run_lead(q: dict, ctx: Ctx, variants) -> None:
    cap_path = ctx.out / q["id"] / "phase_a.json"
    if not cap_path.exists():
        print(f"- {q['id']}: 팀원 단계 결과가 없음 — 먼저 draft 를 돌리세요")
        return
    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    team = load_team(q["team"])
    agent = lead_agent(team, ctx)
    for variant in variants:
        meta_path = ctx.out / q["id"] / f"{variant}.json"
        if meta_path.exists() and not ctx.force:
            print(f"- {q['id']} {variant}: 이미 있음 — 건너뜀")
            continue
        started = time.perf_counter()
        try:
            with usage.scope(team_id=team.id, run_id=q["id"], purpose=f"B:{variant}", agent_id=agent.node.id):
                answer, tools = run_variant(agent, cap, variant)
        except llm_factory.NoAvailableModelError as e:  # 실패는 다시 돌리지 않고 기록만
            meta_path.write_text(json.dumps({"variant": variant, "failed": str(e)[:1500]}, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            print(f"- {q['id']} {variant}: ❌ 실패 — {str(e).splitlines()[-1][:200]}")
            continue
        parts = [a for a in (tools, answer) if a is not None]
        results = harness.run_validators(cap["validators"], answer.content, cap["options"])
        meta = {
            "variant": variant, "model": answer.model.label, "kind": answer.kind, "calls": len(parts),
            "failed_attempts": [e for a in parts for e in a.errors],
            "sec": round(sum(a.elapsed_sec for a in parts), 1), "wall_sec": round(time.perf_counter() - started, 1),
            "in": sum(a.input_tokens or 0 for a in parts), "out": sum(a.output_tokens or 0 for a in parts),
            "chars": len(answer.content), "verdict": answer.review.verdict, "score": answer.review.score,
            "analysis": answer.review.analysis, "issues": answer.review.issues, "comment": answer.review.comment,
            "log": answer.verification_log,
            "harness": [{"validator": r.validator, "passed": r.passed, "log": r.log} for r in results],
        }
        meta["cost_usd"] = sum(usage.estimate_cost(a.model.provider, a.model.model, a.input_tokens or 0, a.output_tokens or 0) or 0
                               for a in parts)
        meta["api_equiv_usd"] = round(sum(api_equiv(a.model.provider, a.model.model, a.input_tokens, a.output_tokens) or 0
                                          for a in parts), 4)
        (ctx.out / q["id"] / f"{variant}.md").write_text(answer.content, encoding="utf-8")
        if tools is not None:
            (ctx.out / q["id"] / "T1_도구_답.md").write_text(tools.content, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = "통과" if all(r.passed for r in results) else "실패: " + ", ".join(r.validator for r in results if not r.passed)
        print(f"- {q['id']} {variant}: {meta['model']} · 호출 {meta['calls']} · {meta['sec']}초 · {meta['chars']:,}자 · "
              f"{answer.review.verdict} · 하네스 {ok}")


# ── measure: 코드로 재는 것 (호출 없음) ───────────────────────────────────

CODE_TESTER = r'''
import json, os, sys, tempfile
ns = {}
exec(compile(open(sys.argv[1], encoding="utf-8").read(), "answer", "exec"), ns)
f = ns["top_ips"]
def run(text, *a):
    path = os.path.join(tempfile.mkdtemp(), "log.csv")
    open(path, "w", encoding="utf-8", newline="").write(text)
    return [(str(ip), int(n)) for ip, n in f(path, *a)]
A, B, C = "10.0.0.1", "10.0.0.2", "10.0.0.3"
cases = {
 "헤더 ip 첫 열": (lambda: run(f"ip,path\n{A},/a\n{A},/b\n{A},/c\n{B},/a\n{B},/b\n{C},/a\n"), [(A, 3), (B, 2), (C, 1)]),
 "헤더 ip 둘째 열": (lambda: run(f"time,ip,path\n1,{A},/a\n2,{A},/b\n3,{B},/a\n"), [(A, 2), (B, 1)]),
 "헤더 없음": (lambda: run(f"{A},/a\n{A},/b\n{B},/a\n"), [(A, 2), (B, 1)]),
 "빈 줄·빈 IP": (lambda: run(f"ip,path\n{A},/a\n\n,/x\n{A},/b\n   \n{B},/a\n"), [(A, 2), (B, 1)]),
 "동률은 IP 문자열 오름차순": (lambda: run(f"ip\n{B}\n10.0.0.10\n{C}\n{C}\n"), [(C, 2), ("10.0.0.10", 1), (B, 1)]),
 "n 지정": (lambda: run(f"ip\n{A}\n{A}\n{B}\n{C}\n{C}\n{C}\n", 2), [(C, 3), (A, 2)]),
 "기본 n=10": (lambda: len(run("ip\n" + "".join(f"10.0.1.{i}\n" for i in range(12)))), 10),
}
out = {}
for name, (call, want) in cases.items():
    try:
        got = call()
        out[name] = got == want
    except Exception as e:
        out[name] = f"{type(e).__name__}: {str(e)[:80]}"
print(json.dumps(out))
'''


def code_tests(answer: str) -> dict:
    """답변의 파이썬 코드에서 import·함수·상수 정의만 남겨(사용 예 실행 제외) top_ips 를 실제 CSV 로 돌려 본다."""
    kept = []
    for block in extract_code_blocks(answer, "python", "py"):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
        kept += [ast.unparse(node) for node in tree.body if isinstance(node, keep)]
    if not any(re.match(r"def top_ips\b", k) for k in kept):
        return {"top_ips 함수 있음": False}
    with tempfile.TemporaryDirectory() as tmp:
        code, tester = Path(tmp) / "answer_code.py", Path(tmp) / "tester.py"
        code.write_text("\n\n".join(kept), encoding="utf-8")
        tester.write_text(CODE_TESTER, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(tester), str(code)], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"30초 안에 끝남": False}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"코드 실행": f"실패 — {proc.stderr.strip()[-200:]}"}


def won(text: str) -> list[int]:
    """글 속 금액(원) — '780,000원', '78만 원', '78만 5천원'."""
    out = [int(m.group(1).replace(",", "")) for m in re.finditer(r"(\d[\d,]{3,})\s*원", text)]
    out += [int(float(m.group(1).replace(",", "")) * 10000) + int(m.group(2) or 0) * 1000
            for m in re.finditer(r"(\d[\d,.]*)\s*만\s*(?:(\d)\s*천)?\s*원", text)]
    return out


def measure_one(q: dict, variant: str, ctx: Ctx) -> dict | None:
    folder = ctx.out / q["id"]
    if not (folder / f"{variant}.md").exists():
        return None
    text = (folder / f"{variant}.md").read_text(encoding="utf-8")
    meta = json.loads((folder / f"{variant}.json").read_text(encoding="utf-8"))
    cap = json.loads((folder / "phase_a.json").read_text(encoding="utf-8"))
    checks = {name: bool(re.search(pattern, text, re.IGNORECASE)) for name, pattern in q["check"]}
    out = {"variant": variant, "chars": len(text), "harness_ok": all(h["passed"] for h in meta["harness"]),
           "harness_failed": [h["validator"] for h in meta["harness"] if not h["passed"]],
           "check_hit": sum(checks.values()), "check_total": len(checks), "check_missing": [k for k, v in checks.items() if not v],
           "same_as_child": any(text.strip() == c["content"].strip() for c in cap["children"])}
    if q["id"] == "general_code":
        tests = code_tests(text)
        out.update(code_pass=sum(v is True for v in tests.values()), code_total=len(tests),
                   code_fail={k: v for k, v in tests.items() if v is not True})
    if q["team"] == "security_governance":
        out["governance"] = governance.answer_metrics(text, cap["options"])
        out["json_blocks"] = len(extract_code_blocks(text, "json"))
    if q["id"] == "travel_busan":
        # 합계 줄에 적힌 금액들 (범위로 쓴 답은 하단·상단이 함께 잡힌다 — 80만 원 안인지는 사람이 읽고 판정)
        out["budget_amounts"] = sorted({n for line in text.splitlines() if "합계" in line for n in won(line)})
    return out


def run_measure(ctx: Ctx) -> None:
    rows = {q["id"]: [m for v in VARIANTS if (m := measure_one(q, v, ctx))] for q in ctx.questions}
    (ctx.out / "measure.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for qid, items in rows.items():
        for m in items:
            extra = {k: v for k, v in m.items() if k in ("code_pass", "code_fail", "governance", "json_blocks", "budget_amounts",
                                                         "same_as_child", "check_missing", "harness_failed") and v not in ([], {}, False)}
            print(f"{qid:<20} {m['variant']:<3} {m['chars']:>6,}자 하네스 {'통과' if m['harness_ok'] else '실패'} "
                  f"점검표 {m['check_hit']}/{m['check_total']} {json.dumps(extra, ensure_ascii=False)}")


# ── grade: 방식을 가린 채 다른 회사 모델이 채점 ─────────────────────────────

GRADER_SYSTEM = "\n".join([
    "너는 결과물 품질 심사자다. 같은 업무 요청에 대한 답변 3개(가·나·다)를 아래 기준표로 각각 채점한다.",
    "- 답변을 누가 어떤 방식으로 썼는지는 주어지지 않는다. 추측하지 말고 답변 글만 보고 판단한다.",
    "- 분량이 아니라 [업무 브리프]의 완료 기준·제약·결과 형식을 얼마나 지켰는지로 채점한다. 길다고 점수를 더 주지 않는다.",
    "- 근거 자료가 주어졌으면 답변의 사실·숫자·인용을 그 자료와 대조한다. 자료에 없는 구체적 숫자·출처·고유명사를",
    "  '확인 필요' 표시 없이 사실처럼 쓴 것은 지어낸 내용으로 본다.",
    "- 기준마다 1~5 정수 (5 = 매우 좋음). 차이가 없으면 같은 점수를 줘도 된다.",
    "- 먼저 analysis 에 세 답변의 차이를 짧게 분석하고, 그다음 답변마다 점수를 매긴다.",
    "기준: " + " / ".join(f"{k} = {v}" for k, v in CRITERIA.items()),
])


def grade_prompt(cap: dict, answers: dict[str, str]) -> str:
    parts = [f"[업무 브리프]\n{cap['question']}"]
    if cap["web"]:
        parts.append(f"[근거 자료 — 답변을 쓴 쪽도 같은 자료를 받았다]\n{cap['web'][:12000]}")
    parts += [f"### 답변 {label}\n{text[:20000]}" for label, text in answers.items()]
    return "\n\n".join(parts)


def run_grade(ctx: Ctx, graders, passes: int) -> None:
    log = ctx.out / "grades.jsonl"
    done = {(r["question"], r["grader"], r["pass"]) for r in read_jsonl(log)}
    for q in ctx.questions:
        folder = ctx.out / q["id"]
        texts = {v: (folder / f"{v}.md").read_text(encoding="utf-8") for v in VARIANTS if (folder / f"{v}.md").exists()}
        if len(texts) < 2:
            print(f"- {q['id']}: 비교할 결과가 2개 미만 — 건너뜀")
            continue
        cap = json.loads((folder / "phase_a.json").read_text(encoding="utf-8"))
        order = sorted(texts)
        random.Random(q["id"]).shuffle(order)  # 질문마다 정해진 무작위 순서 (다시 돌려도 같음)
        for grader in graders:
            provider, _, model = grader.partition(":")
            spec = ModelSpec(provider=provider, model=model, temperature=0.0 if provider == "google" else None)
            for p in range(1, passes + 1):
                if (q["id"], grader, p) in done and not ctx.force:
                    continue
                # 1회: 정한 순서 · 2회: 뒤집기 · 3회: 가운데 답변을 맨 뒤로 (뒤집어도 가운데 자리는 그대로라서)
                shown = {1: order, 2: order[::-1]}.get(p, [order[0], *order[2:], order[1]])
                mapping = dict(zip(LABELS, shown))  # 표시 → 방식 (모델에게는 보내지 않는다)
                messages = [SystemMessage(GRADER_SYSTEM), HumanMessage(grade_prompt(cap, {k: texts[v] for k, v in mapping.items()}))]
                try:
                    with usage.scope(team_id=q["team"], run_id=q["id"], purpose=f"C:채점{p}"):
                        call = llm_factory.invoke_structured(RankModels(primary=spec), ctx.models, GradeSheet, messages, "눈가림 채점")
                except llm_factory.NoAvailableModelError as e:
                    print(f"- {q['id']} {grader} {p}회차: ❌ 실패 — {str(e).splitlines()[-1][:200]}")
                    continue
                sheet: GradeSheet = call.parsed
                scores = {mapping[g.label.strip()]: {**{k: getattr(g, k) for k in CRITERIA}, "note": g.note}
                          for g in sheet.grades if g.label.strip() in mapping}
                row = {"question": q["id"], "grader": grader, "pass": p, "order": shown, "scores": scores,
                       "ranking": [mapping.get(x.strip(), x) for x in sheet.ranking], "analysis": sheet.analysis,
                       "sec": call.elapsed_sec, "in": call.input_tokens, "out": call.output_tokens}
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                totals = " ".join(f"{v}={sum(s[k] for k in CRITERIA)}" for v, s in sorted(scores.items()))
                print(f"- {q['id']} {grader} {p}회차 (보인 순서 {'·'.join(shown)}): {totals} / 순위 {' > '.join(row['ranking'])}")


# ── plan · table (호출 없음) ──────────────────────────────────────────


def show_plan(ctx: Ctx) -> None:
    total = 0
    for team_id in dict.fromkeys(q["team"] for q in ctx.questions):
        team = load_team(team_id)
        org, _, agents = build(team, ctx)
        n_q = sum(q["team"] == team_id for q in ctx.questions)
        managers = [n for n in org.walk() if n.children]
        # 팀원 단계 호출 수 = 실무자 초안 + 하위 2명 이상인 관리자의 분배 + 팀장을 뺀 관리자의 검토·통합
        per_q = sum(not n.children for n in org.walk()) + sum(len(n.children) > 1 for n in managers) + len(managers) - 1
        total += n_q * (per_q + 4)
        print(f"\n{team.name} — 질문 {n_q}개 · 질문당 팀원 단계 {per_q}회 + 팀장 단계 4회(C 1 · T1 2 · T2 1)")
        for node in org.walk():
            first = llm_factory.available_candidates(agents[node.id].rank_models, ctx.models)
            print(f"   {node.id:<14} {agents[node.id].name:<12} {first[0][0].label if first else '쓸 수 있는 모델 없음'}")
    grading = len(ctx.questions) * len(GRADERS) * 2
    print(f"\n예상 호출: 팀원·팀장 {total}회 + 눈가림 채점 {grading}회 = {total + grading}회 (재시도·실패 제외)")


def run_table(ctx: Ctx) -> None:
    grades = [r for r in read_jsonl(ctx.out / "grades.jsonl") if len(r["scores"]) >= 2]  # 표시가 깨진 채점은 뺀다
    cols = sorted({(r["grader"], r["pass"]) for r in grades})
    measure = json.loads((ctx.out / "measure.json").read_text(encoding="utf-8")) if (ctx.out / "measure.json").exists() else {}

    def total(row: dict, variant: str) -> int | None:
        s = row["scores"].get(variant)
        return sum(s[k] for k in CRITERIA) if s else None

    print("| 질문 | 방식 | 하네스 | 점검표 | 글자 수 | 걸린 시간(초) | 호출 | 토큰 입력/출력 | API 환산($) | " +
          " | ".join(f"{g.split(':')[1]} {p}회" for g, p in cols) + " | 채점 평균 |")
    print("|---|---|---|---|---|---|---|---|---|" + "---|" * (len(cols) + 1))
    for q in ctx.questions:
        for v in VARIANTS:
            meta_path = ctx.out / q["id"] / f"{v}.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "failed" in meta:
                print(f"| {q['id']} | {v} | 호출 실패 |" + " |" * (len(cols) + 7))
                continue
            m = next((x for x in measure.get(q["id"], []) if x["variant"] == v), {})
            got = [next((total(r, v) for r in grades if (r["question"], r["grader"], r["pass"]) == (q["id"], g, p)), None) for g, p in cols]
            valid = [x for x in got if x is not None]
            print(f"| {q['id']} | {v} | {'통과' if m.get('harness_ok') else '실패' if m else '-'} | "
                  f"{m.get('check_hit', '-')}/{m.get('check_total', '-')} | {meta['chars']:,} | {meta['sec']} | {meta['calls']} | "
                  f"{meta['in']:,}/{meta['out']:,} | {meta['api_equiv_usd']} | " + " | ".join("-" if x is None else str(x) for x in got) +
                  f" | {sum(valid) / len(valid):.1f} |" if valid else " | - |")

    print()
    print("채점 요약 (25점 만점, 표시가 깨진 채점 제외)")
    for grader in [None, *sorted({g for g, _ in cols})]:
        rows = [r for r in grades if grader is None or r["grader"] == grader]
        if not rows:
            continue
        means = {v: [t for r in rows if (t := total(r, v)) is not None] for v in VARIANTS}
        firsts = {v: 0.0 for v in VARIANTS}
        for r in rows:
            best = max(t for v in VARIANTS if (t := total(r, v)) is not None)
            winners = [v for v in VARIANTS if total(r, v) == best]
            for v in winners:
                firsts[v] += 1 / len(winners)  # 공동 1위는 나눠 센다
        print(f"- {grader or '전체'} ({len(rows)}회): 평균 " + " · ".join(f"{v} {sum(x) / len(x):.2f}" for v, x in means.items() if x) +
              " / 1위 횟수 " + " · ".join(f"{v} {n:.1f}" for v, n in firsts.items()))
    by_crit = {v: {k: [r["scores"][v][k] for r in grades if v in r["scores"]] for k in CRITERIA} for v in VARIANTS}
    for k, name in CRITERIA.items():
        print(f"- 기준별 평균 {name}: " + " · ".join(f"{v} {sum(by_crit[v][k]) / len(by_crit[v][k]):.2f}" for v in VARIANTS if by_crit[v][k]))
    by_pos: dict[int, list[int]] = {}
    for r in grades:
        for i, v in enumerate(r["order"]):
            if (t := total(r, v)) is not None:
                by_pos.setdefault(i + 1, []).append(t)
    print("- 보인 자리별 평균 (순서 영향): " + " · ".join(f"{i}번째 {sum(x) / len(x):.2f}" for i, x in sorted(by_pos.items())))
    for i in sorted(by_pos):  # 같은 자리에 놓였을 때끼리 비교 — 자리 운을 뺀 방식 차이
        cell = {v: [t for r in grades if len(r["order"]) >= i and r["order"][i - 1] == v and (t := total(r, v)) is not None] for v in VARIANTS}
        print(f"  · {i}번째 자리에 놓였을 때: " + " · ".join(f"{v} {sum(x) / len(x):.2f} ({len(x)}회)" for v, x in cell.items() if x))
    flips = same = 0
    for qid, grader in sorted({(r["question"], r["grader"]) for r in grades}):
        pair = [r for r in grades if (r["question"], r["grader"]) == (qid, grader)]
        if len(pair) == 2:
            tops = [{v for v in VARIANTS if total(r, v) == max(t for x in VARIANTS if (t := total(r, x)) is not None)} for r in pair]
            same += bool(tops[0] & tops[1])
            flips += not (tops[0] & tops[1])
    print(f"- 순서를 뒤집었을 때: 1위가 그대로(공동 포함) {same}쌍 · 1위가 바뀜 {flips}쌍")
    calls = read_jsonl(ctx.out / "calls.jsonl")
    done = [c for c in calls if c["type"] == "done"]
    print()
    print(f"모델 호출 시도 {sum(c['type'] == 'attempt' for c in calls)}회 (끝난 호출 {len(done)}회) · "
          f"청구 추정 ${sum(c.get('cost_usd') or 0 for c in done):.3f} · "
          f"구독 호출 {sum(c['provider'] == 'claude_cli' for c in done)}회를 API 로 환산하면 "
          f"${sum((c.get('api_equiv_usd') or 0) for c in done if c['provider'] == 'claude_cli'):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="팀장 프롬프트 2단계 방식 비교 시험")
    parser.add_argument("command", choices=["plan", "draft", "lead", "measure", "grade", "table"])
    parser.add_argument("--fake", action="store_true", help="가짜 모델로 (호출·비용 없음)")
    parser.add_argument("--env", help="API 키가 든 .env 경로 (환경 변수로만 읽는다)")
    parser.add_argument("--only", help="이 질문 id 만 (쉼표로 여러 개)")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--graders", default=",".join(GRADERS))
    parser.add_argument("--passes", type=int, default=2, help="채점 횟수 (2회째는 순서를 뒤집고, 3회째는 가운데 답변을 맨 뒤로)")
    parser.add_argument("--lead-model", help="팀장 모델을 직접 정한다 (provider:model) — 기본은 앱이 지금 쓰는 첫 모델")
    parser.add_argument("--max-calls", type=int, default=150, help="모델 호출 시도 상한 (지금까지 합계)")
    parser.add_argument("--max-usd", type=float, default=5.0, help="청구 추정 비용 상한")
    parser.add_argument("--rest", help="오늘 한도가 찬 모델을 후보 맨 뒤로 (provider:model, 쉼표로 여러 개)")
    parser.add_argument("--force", action="store_true", help="이미 있는 결과도 다시 만든다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ctx = Ctx(args)
    if args.command == "plan":
        return show_plan(ctx)
    if args.command == "measure":
        return run_measure(ctx)
    if args.command == "table":
        return run_table(ctx)
    ctx.install()
    try:
        if args.command == "draft":
            for q in ctx.questions:
                print(f"\n▶ {q['id']}")
                run_draft(q, ctx)
        elif args.command == "lead":
            for q in ctx.questions:
                run_lead(q, ctx, [v for v in args.variants.split(",") if v in VARIANTS])
        else:
            run_grade(ctx, args.graders.split(","), args.passes)
    except BudgetExceeded as e:
        print(f"\n⛔ {e} — 여기서 멈춥니다. 이미 만든 결과는 남아 있습니다.")
    print(f"\n지금까지 모델 호출 시도 {ctx.budget.calls}회 · 청구 추정 ${ctx.budget.usd:.3f}")


if __name__ == "__main__":
    main()
