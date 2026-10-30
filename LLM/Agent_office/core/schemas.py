"""데이터 모양(스키마) 정의 — 이 프로젝트에서 주고받는 모든 데이터의 '형식'이 여기 모여 있다.

크게 세 종류로 나뉜다.
  1) 설정 스키마   : config/*.yaml, teams/*/team.yaml 을 읽어 검증하는 모델 (오타·누락을 실행 전에 잡는다)
  2) LLM 출력 스키마: LLM 에게 "이 JSON 형식으로만 답하라"고 강제하는 구조화 출력 모델
                     (with_structured_output 에 넘긴다. 필드 순서 = LLM 이 생각하는 순서)
  3) 실행 결과 스키마: 워크플로우가 만들어 내는 단계별 결과, 사람의 결정, 교훈 등

pydantic BaseModel 을 쓰면 (a) 타입 검증, (b) dict/JSON 변환, (c) LLM 용 JSON 스키마 생성이 자동으로 된다.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# freellm = FreeLLMAPI 프록시 (무료 티어 모델 모음, OpenAI 호환) — 유료 키를 쓸 수 없을 때의 무료 대안
# claude_cli = 이 PC 에 로그인된 Claude 구독을 Claude Code CLI(claude -p)로 호출 (core/claude_cli.py) — API 크레딧 대신 구독 한도
Provider = Literal["openai", "anthropic", "google", "upstage", "freellm", "claude_cli"]


# ════════════════════════════════════════════════════════════════════
# 1) 설정 스키마
# ════════════════════════════════════════════════════════════════════


class ModelSpec(BaseModel):
    """LLM 하나를 가리키는 설정 (config/models.yaml 의 한 줄)."""

    provider: Provider
    model: str
    # None 이면 provider 기본값 사용 (예: gpt-5 계열은 temperature 를 지원하지 않음)
    temperature: float | None = None
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"


class RankModels(BaseModel):
    """직급 하나에 배정된 모델 목록 — primary 가 안 되면 fallback 을 순서대로 시도한다."""

    primary: ModelSpec
    fallback: list[ModelSpec] = Field(default_factory=list)

    @property
    def candidates(self) -> list[ModelSpec]:
        return [self.primary, *self.fallback]


class ModelsConfig(BaseModel):
    """config/models.yaml 전체."""

    providers: dict[Provider, str]  # provider → .env 키 이름
    # 키가 있어도 쓰지 않을 provider (크레딧 부족 등)
    disabled_providers: list[Provider] = Field(default_factory=list)
    ranks: dict[str, RankModels]
    utility: RankModels  # 검색어 생성·교훈 생성·비평(critic) 등 보조 작업용
    # 모델 강도 순서 — 위(앞)가 가장 강함, 한 줄 안은 같은 강도. 항목은 "provider:model" (팀원 모델 선택 규칙에 쓴다)
    model_tiers: list[list[str]] = Field(default_factory=list)


class Rank(BaseModel):
    """직급 (config/ranks.yaml). 직급은 '어떤 모델을 쓰는지'와 기본 아바타를 정한다."""

    id: str
    name: str
    emoji: str = "👤"
    role: str
    avatar: str | None = None


class OrgNode(BaseModel):
    """조직도의 한 칸 = 에이전트 한 명.

    같은 직급이라도 전문 분야가 다른 에이전트를 여러 명 둘 수 있다 (예: 부장·리서치, 부장·실행).
      - rank      : 직급 id → 사용할 모델과 기본 아바타가 결정된다
      - specialty : 전문 분야 → 프롬프트에 들어가 '그 분야 전문가'로 행동하게 한다
      - children  : 하위 에이전트들. 비어 있으면 실무자(leaf) — 직접 초안을 쓴다
    """

    id: str
    rank: str
    name: str | None = None  # 화면 표시 이름. 없으면 직급 이름
    specialty: str = ""
    persona: str = ""  # 비우면 team.personas[rank] → ranks.yaml 의 role 순서로 대체
    avatar: str | None = None  # 이 에이전트 전용 이미지 경로 (선택)
    # 팀 스킬 중 이 에이전트에게 넣을 절 — 절 제목 줄('[1단계·…')의 앞부분 목록. 비우면 스킬 전문 (OrgAgent)
    #   실무자에게 맡은 일에 필요한 절만 주면 프롬프트가 짧아지고, 최종본 형식 같은 남의 일까지 쓰지 않는다
    skill_sections: list[str] = Field(default_factory=list)
    # 실무자 초안의 산출물 범위 — 있으면 브리프의 완료 기준(팀 전체 최종 답변 기준) 대신 이 범위만 쓴다
    output_scope: str = ""
    children: list["OrgNode"] = Field(default_factory=list)

    def walk(self) -> list["OrgNode"]:
        """자신과 모든 하위 노드를 위→아래(전위 순회) 순서로 반환."""
        nodes = [self]
        for child in self.children:
            nodes += child.walk()
        return nodes


class HarnessConfig(BaseModel):
    """결정론적 검증 하네스 설정 (team.yaml 의 harness)."""

    validators: list[str] = Field(default_factory=list)  # core/harness/validators.py 의 REGISTRY 이름
    max_retries: int = 2  # 검증 실패 시 에러 로그를 돌려주고 다시 쓰게 하는 최대 횟수 (Backtracking)
    options: dict = Field(default_factory=dict)  # 검증기별 옵션 (예: sql_schema 파일 경로)
    # 최종 답변(문의 대상 에이전트의 결과)에만 돌릴 검증기 — 하위 실무자는 업무의 일부만 맡으므로 전체 형식을 요구하지 않는다
    final_only: list[str] = Field(default_factory=list)
    # final_only 검증기를 최종 답변 외에 이 에이전트들의 결과에도 적용 — {검증기: [에이전트 id]}
    # 예: 설계 팀 {json_keys: [dgm_layout]} → 방 프로그램을 받아 JSON 을 쓰는 차장 단계부터 JSON 필수
    also_at: dict[str, list[str]] = Field(default_factory=dict)


class TaskField(BaseModel):
    """업무 입력 양식의 칸 하나 — 화면에 입력란으로 그려진다."""

    key: str  # 템플릿에서 {key} 로 쓰는 이름
    label: str
    # file: 파일 첨부 (options = 받을 확장자, 예: [json, dxf, png]) — 저장한 파일 경로가 값이 되고 {key}_name 에 원래 파일 이름
    # checkbox: 켜면 True ("예"), 끄면 빈 값 → 그 {key} 가 들어간 줄은 빠진다
    type: Literal["text", "textarea", "number", "date", "select", "file", "checkbox"] = "text"
    required: bool = False
    options: list[str] = Field(default_factory=list)  # select 일 때 선택지 / file 일 때 받을 확장자
    default: bool | str | float | int | None = None  # checkbox 는 true/false
    placeholder: str = ""
    help: str = ""


class TaskTemplate(BaseModel):
    """팀의 정형 업무 (예: 주가 전망). 입력값으로 질문과 '최종 상태' 브리프를 자동으로 만든다.

    question·goal·success_criteria 등에는 {입력칸key} 를 쓸 수 있다.
    값이 비어 있는 {key} 가 들어간 줄은 통째로 빠진다 → 선택 입력을 자연스럽게 처리.
    """

    id: str
    title: str
    description: str = ""
    fields: list[TaskField] = Field(default_factory=list)
    question: str
    goal: str
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_format: str = ""
    # 업무 수행 지침 (사고 절차·채점 루브릭·퓨샷 예시·출력 JSON 형식 등) — 브리프 [배경]에 들어가 모든 에이전트가 본다
    instructions: str = ""
    validators: list[str] = Field(default_factory=list)  # 이 업무에서 추가로 돌릴 하네스 검증기
    harness_options: dict = Field(default_factory=dict)  # 검증기 옵션 (예: required_terms). 문자열 값의 {key} 도 입력값으로 바뀐다
    fixed: dict[str, str] = Field(default_factory=dict)  # 화면에 칸 없이 늘 같은 입력값 (예: 해외 여행팀 {scope: 해외})


class LegacyLink(BaseModel):
    """팀을 나눈 뒤 옛 팀 id 의 기록을 이어 받기 — DB·파일은 고치지 않고 읽을 때만 붙인다 (core/storage.py)."""

    team: str  # 옛 팀 id (예: travel)
    match: str = ""  # 옛 업무 중 질문에 이 말이 있는 것만 이 팀에 보인다. 어느 팀의 말도 없으면 모든 팀에 보인다


class Team(BaseModel):
    """팀 (teams/<id>/team.yaml). 팀마다 조직도·검토 기준·툴·하네스를 따로 가진다."""

    id: str
    name: str
    description: str = ""
    personas: dict[str, str] = Field(default_factory=dict)  # 직급 id → 기본 페르소나
    review_criteria: list[str] = Field(default_factory=list)
    output_format: str = "markdown"
    tools: list[str] = Field(default_factory=list)  # 예: [web_search, stock_data]
    model_overrides: dict[str, RankModels] = Field(default_factory=dict)
    disclaimer: str = ""
    org: OrgNode | None = None  # 없으면 ranks.yaml 순서대로 일렬 조직도를 자동 생성
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    rubric: dict | None = None  # 없으면 config/verification.yaml 의 review_rubric 사용
    tasks: list[TaskTemplate] = Field(default_factory=list)  # 정형 업무 양식 (없으면 자유 질문만)
    search: dict = Field(default_factory=dict)  # 웹 검색 설정 덮어쓰기 (hint, news_timelimit 등)
    features: list[str] = Field(default_factory=list)  # 추가 기능 (예: route_map = 여행 경로 지도)
    # 검증 설정 덮어쓰기 (config/verification.yaml 의 critic 위에 덮어씀)
    #   critic: {scope: all, max_self_corrections: 2, rules: [이 팀의 비평 규칙 …]}
    verification: dict = Field(default_factory=dict)
    # 하위 1명 검토(review)에서 고쳐 쓰는 방식
    #   rewrite: 지적을 반영한 답변 전문을 다시 씀 (기본)
    #   edit   : 고칠 곳만 '찾을 문장 → 바꿀 문장'(edits)으로 — 긴 답변(개념 카탈로그 항목 1만 자)을 통째로 다시 쓰는 시간을 줄인다
    review_mode: Literal["rewrite", "edit"] = "rewrite"
    skills_from: list[str] = Field(default_factory=list)  # 함께 쓰는 스킬 폴더 (teams/ 기준, 예: _shared/travel)
    legacy: LegacyLink | None = None  # 나뉘기 전 옛 팀의 지난 업무·교훈·팀원 모델 선택·저장 파일을 이어 받기


# ════════════════════════════════════════════════════════════════════
# 2) LLM 구조화 출력 스키마
#    ※ 필드 순서가 곧 LLM 의 '생각 순서'다. 분석(analysis)을 맨 앞에 두어
#      점수·결론을 내기 전에 근거를 먼저 쓰게 강제한다 (CoT 강제).
#    ※ OpenAI strict json_schema 와 호환되도록 기본값 없이 모든 필드를 필수로 둔다.
# ════════════════════════════════════════════════════════════════════


class Assignment(BaseModel):
    child_id: str = Field(description="업무를 맡길 하위 에이전트 id (목록에 있는 id 그대로)")
    subtask: str = Field(description="그 하위 에이전트에게 줄 구체적인 업무 지시 (1~3문장)")


class Delegation(BaseModel):
    """관리자(하위가 있는 에이전트)의 업무 분배 결과 — 위 → 아래."""

    analysis: str = Field(description="먼저 질문 의도와 필요한 전문 분야를 2~3문장으로 분석")
    assignments: list[Assignment] = Field(description="업무를 맡길 하위 에이전트와 업무 지시. 필요한 하위만 고른다")


class ReviewResult(BaseModel):
    """관리자의 검토·통합 결과 — 아래 → 위."""

    analysis: str = Field(
        description="점수를 매기기 전에, 검토 대상 답변에서 오류·누락·근거 부족 같은 취약점이 있는지 3문장 이내로 먼저 분석"
    )
    issues: list[str] = Field(description="분석에서 찾은 문제점 목록. 없으면 빈 목록")
    score: int = Field(description="채점 기준(루브릭)에 따른 검토 대상 답변의 점수 (1~10). 분석과 문제점에 근거해 마지막에 매긴다")
    verdict: Literal["approve", "revise"] = Field(
        description="approve: 그대로 승인 / revise: 보완(또는 여러 답변을 통합)해 새로 작성"
    )
    revised_answer: str = Field(
        description="revise 이면 지적을 모두 반영한(통합한) 완성 답변 전문. approve 이면 빈 문자열"
    )
    comment: str = Field(description="하위 에이전트에게 남기는 한두 문장의 조언")


class TextEdit(BaseModel):
    find: str = Field(description="검토 대상 답변에 있는 바꿀 부분을 글자 그대로 (그 답변 안에서 한 번만 나오도록 앞뒤를 조금 포함)")
    replace: str = Field(description="그 부분을 바꿀 새 글 (지우려면 빈 문자열)")


class ReviewEdits(ReviewResult):
    """검토 결과 + 고칠 곳 목록 — 팀의 review_mode: edit (답변 전문을 다시 쓰지 않고 고칠 곳만 바꾼다)."""

    edits: list[TextEdit] = Field(
        description="revise 이면 고칠 곳마다 찾을 부분 → 바꿀 글. revised_answer 를 비웠을 때 이것을 검토 대상 답변에 차례로 적용한다"
    )


class CriticResult(BaseModel):
    """비평가(Critic, LLM-as-a-Judge) — 검토의 '중간 과정' 논리만 따로 채점한다."""

    analysis: str = Field(description="점수 전에, 검토 지적 → 수정 답변으로 이어지는 논리를 3문장 이내로 먼저 분석")
    logical_leaps: list[str] = Field(description="근거 없이 결론으로 비약한 부분. 없으면 빈 목록")
    rule_violations: list[str] = Field(
        description="어긴 규칙과 위치를 'A 부분이 B 규칙에 어긋남' 형식으로. 없으면 빈 목록"
    )
    severity: Literal["none", "minor", "severe"] = Field(
        description=(
            "위 지적 중 가장 심각한 것의 등급. none: 지적 없음 / minor: 표현·형식·사소한 누락 (결론은 그대로 유효) / "
            "severe: 사실 오류, 근거 자료와 어긋나는 숫자, 하위 결과의 올바른 사실을 버림, 결론을 뒤집는 비약"
        )
    )
    score: int = Field(description="중간 과정 논리의 완결성 (1~5, 5=비약 없음). 위 분석에 근거해 마지막에 매긴다")

    @model_validator(mode="before")
    @classmethod
    def _fill_severity(cls, data):
        # severity 가 생기기 전에 저장된 실행 기록(runs.db)·모델이 빠뜨린 경우 — 지적이 있으면 minor, 없으면 none.
        # (기본값 대신 여기서 채운다: 기본값이 있으면 OpenAI strict 구조화 출력이 필드를 required 로 못 받는다)
        if isinstance(data, dict) and not data.get("severity"):
            data = {**data, "severity": "minor" if data.get("logical_leaps") or data.get("rule_violations") else "none"}
        return data


class Lesson(BaseModel):
    agent_id: str = Field(description="교훈을 받을 에이전트 id (괄호 안의 id 그대로)")
    kind: Literal["praise", "improve"] = Field(description="praise: 잘한 점 유지 / improve: 개선할 점")
    lesson: str = Field(description="다음 업무에 바로 적용할 수 있는 한 문장의 일반화된 교훈")


class LessonSet(BaseModel):
    """피드백(교훈) 생성용 구조화 출력."""

    lessons: list[Lesson] = Field(description="참여한 에이전트마다 1개씩의 교훈")


# ════════════════════════════════════════════════════════════════════
# 3) 실행 결과 스키마
# ════════════════════════════════════════════════════════════════════


class ToolResult(BaseModel):
    """데이터 툴(주가 조회, 여행 달력 등)의 결과 — 웹 검색 결과와 함께 모든 에이전트의 근거 자료가 된다."""

    tool: str  # 툴 이름 (예: stock_data)
    title: str  # 화면 표시 제목 (예: 📈 시장 데이터)
    content: str  # 프롬프트에 들어갈 본문 (마크다운)
    notes: list[str] = Field(default_factory=list)  # 경고·실패 기록
    harness_options: dict = Field(default_factory=dict)  # 툴이 계산해 하네스에 넘기는 값 (예: 여행 일수)


class HarnessResult(BaseModel):
    """결정론적 검증기 한 개의 결과 (LLM 이 아니라 코드가 판정)."""

    validator: str
    passed: bool
    score: float  # 1.0 통과 / 0.0 실패 (RLVR 의 '검증 가능한 보상')
    log: str = ""  # 실패 시 에러 로그 — 그대로 LLM 에게 돌려줘 다시 쓰게 한다


StepKind = Literal["plan", "draft", "review", "integrate"]


class AgentAnswer(BaseModel):
    """에이전트 한 단계의 결과. 타임라인 한 줄 = AgentAnswer 하나.

      plan      : 관리자가 하위에게 업무를 나눠 줌 (content = 분배 요약)
      draft     : 실무자(leaf)가 초안 작성
      review    : 관리자가 하위 1명의 결과를 검토·보완
      integrate : 관리자가 하위 여러 명의 결과를 통합
    """

    node_id: str  # 어느 에이전트(조직도 칸)의 결과인지
    rank_id: str  # 그 에이전트의 직급 (모델·아바타 결정용)
    kind: StepKind = "draft"
    content: str  # 이 단계를 거친 뒤의 답변 (plan 이면 분배 요약)
    review: ReviewResult | None = None  # review / integrate 일 때
    delegation: Delegation | None = None  # plan 일 때
    critic: CriticResult | None = None  # 비평가가 채점했으면
    harness: list[HarnessResult] = Field(default_factory=list)  # 하네스 검증 결과 (마지막 시도)
    verification_log: list[str] = Field(default_factory=list)  # 자가 교정·재시도 기록 (사람이 읽는 용)
    attempts: int = 1  # 자가 교정·하네스 재시도를 포함한 총 작성 횟수
    model: ModelSpec
    is_fallback: bool = False
    elapsed_sec: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    # 실패하고 건너뛴 모델들 ("provider · model: 에러 메시지")
    errors: list[str] = Field(default_factory=list)


class HumanDecision(BaseModel):
    """사람의 최종 결정 (LangGraph interrupt 재개 값)."""

    action: Literal["adopt", "reject"]
    adopted_node_id: str | None = None  # 어느 에이전트의 안을 채택했는지 (reject 면 None)
    final_answer: str = ""  # 채택 후 사람이 수정했을 수 있는 최종 답변
    edited: bool = False  # 채택안을 사람이 직접 수정했는지
    score: int = 3  # 최종 결과 만족도 1~5
    comment: str = ""
