"""⏭ 진행 중인 호출을 다른 모델로 — 고를 수 있는 모델과 '왜 안 되는지'.

🎛 팀원 모델 선택과 같은 규칙을 쓴다 (core/member_models.py):
  - 팀원(에이전트) 자신의 호출(분배·초안·검토·자가 교정·하네스 재작성): 상사보다 강한 모델 금지, 하위가 있는 관리자는 맨 아래 줄 금지,
    무료 모델을 쓰게 된 팀원은 무료 모델만
  - 공용 호출(비평가·요청 정리·검색어·교훈): 모델 목록 전체
  - 모두: 지금 쓸 수 없는 provider(키 없음·비활성·🆓 무료 모델만·무료 서버 꺼짐·Claude 로그인 필요)는 안 됨
"""

from dataclasses import dataclass

from core import llm_factory, member_models, model_caps
from core.call_monitor import CallRecord
from core.schemas import ModelsConfig, ModelSpec, OrgNode, Team

AGENT_KINDS = {"plan", "draft", "review", "self_correct", "harness", "directions", "followup"}


@dataclass
class Option:
    key: str  # "provider:model"
    allowed: bool
    reason: str = ""  # 안 되는 이유 (되면 빈 문자열)


def options(rec: CallRecord, org: OrgNode, team: Team | None, base_models: ModelsConfig, run_models: ModelsConfig,
            member_choices: dict[str, str] | None = None) -> list[Option]:
    """이 호출을 바꿔 쓸 수 있는 모델 (강한 순). base_models = 원래 설정, run_models = 이번 실행 설정(무료만 반영)."""
    nodes = {n.id: n for n in org.walk()}
    node = nodes.get(rec.agent_id) if rec.kind in AGENT_KINDS else None
    tiers = member_models.tier_map(base_models)
    last = len(base_models.model_tiers) - 1
    plan = member_models.plan(org, base_models, team, member_choices or {})[node.id] if node else None
    locked = bool(plan and member_models.free_locked(plan))
    status = llm_factory.provider_status(run_models)
    out = []
    for key in member_models.catalog(base_models):
        provider, model = member_models.split_key(key)
        reason = ""
        if provider == rec.provider and model == rec.model:
            reason = "지금 쓰는 모델"
        elif plan is not None and tiers[key] < plan.floor:
            reason = "상사보다 강한 모델 (팀원 규칙)"
        elif node is not None and node.children and tiers[key] >= last:
            reason = "하위가 있는 관리자는 맨 아래 줄 모델을 쓸 수 없음"
        elif locked and provider != member_models.FREE_PROVIDER:
            reason = "무료 모델을 쓰게 된 팀원 — 무료 모델만"
        elif provider in run_models.disabled_providers and provider not in base_models.disabled_providers:
            reason = "🆓 무료 모델만 사용 중"
        elif caps_why := model_caps.reasons(key, model_caps.TEAM_WORK):  # 제공처에서 사라짐·형식 출력 못 함 (core/model_caps)
            reason = caps_why[0]
        else:
            reason = llm_factory.STATUS_REASONS.get(status.get(provider, "no_key"), status.get(provider, ""))
        out.append(Option(key, not reason, reason))
    return out


def spec_for(key: str, base_models: ModelsConfig, team: Team | None, rank_id: str = "") -> ModelSpec:
    return member_models.spec_for(key, base_models, team, rank_id or next(iter(base_models.ranks), ""))
