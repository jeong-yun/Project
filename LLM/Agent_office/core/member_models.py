"""팀원별 모델 선택 — 화면에서 에이전트마다 모델을 고르고, 상하 관계 규칙을 지킨다.

강도 순서는 config/models.yaml 의 model_tiers (위 = 가장 강함, 한 줄 = 같은 강도).
상사 = 조직도에서 그 팀원 위로 팀장까지 이어지는 모든 사람.

규칙
  - 팀원은 상사보다 강한 모델을 쓸 수 없다.
  - 상사의 모델을 화면에서 직접 고른 경우, 그 아래 팀원은 '더 낮은' 줄의 모델만 쓴다 (같은 강도도 안 됨).
    예) 부장 = Claude Opus 5.5 → 그 부장 아래 차장·대리·사원은 Opus 보다 낮은 모델만
  - '기본(직급 설정)' 팀원의 모델은 직급의 primary 로 본다. 기본 상사와는 같은 강도까지 허용한다.
    → 팀장 기본값이 이미 가장 강한 Opus 라서, 이렇게 해야 '부장 = Opus' 를 고를 수 있다.
      (models.yaml 의 직급 설정에도 대리의 대안 gpt-5.4-mini = 차장 primary 처럼 같은 강도가 있다)
  - 상사를 바꿔 규칙을 어기게 된 팀원은 '허용되는 가장 강한 모델'로 자동으로 낮춘다.
    직접 고른 팀원은 저장값을 바꾸고, 기본 팀원은 저장하지 않고 그때그때 낮춘다 (상사를 되돌리면 기본도 돌아온다).
  - 실행 때: 고른 모델이 primary, 대안(fallback)은 직급 목록 그대로 두되 상사 규칙에 걸리는 모델은 뺀다
    → 대안 모델로 넘어가도 규칙을 넘지 않는다. 사용 불가(키 없음·비활성·무료만)·쿨다운·동시 호출 제한은 llm_factory 가 그대로 맡는다.
  - 무료(freellm) 모델을 쓰게 된 팀원(직접 고름·자동으로 낮춰짐)은 대안도 무료 모델만 — 무료 서버가 꺼져 있어도
    유료 API·구독으로 넘어가지 않고 '무료 서버 꺼짐'으로 실패한다 (특정 API 를 많이 쓰지 않으려고 고른 것이므로).
  - 본인과 상사 중 아무도 고르지 않았으면 지금과 완전히 같다 (직급 설정 그대로).

저장: 데이터 폴더(AGENT_OFFICE_DATA_DIR)의 member_models.json — {팀 id: {에이전트 id: "provider:model"}}.
config/*.yaml 은 바꾸지 않는다. 바꾼 선택은 다음에 맡기는 업무부터 쓰인다 (진행 중인 업무는 시작할 때의 모델 그대로).
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from core.config import rank_models
from core.schemas import ModelsConfig, ModelSpec, OrgNode, RankModels, Team

FILE_NAME = "member_models.json"
COST_LABELS = {"claude_cli": "구독", "freellm": "무료"}  # 그 밖의 provider = 유료 API
FREE_PROVIDER = "freellm"


def model_key(spec: ModelSpec) -> str:
    return f"{spec.provider}:{spec.model}"


def split_key(key: str) -> tuple[str, str]:
    provider, _, model = key.partition(":")
    return provider, model


def cost_label(provider: str) -> str:
    return COST_LABELS.get(provider, "유료 API")


def tier_map(models: ModelsConfig) -> dict[str, int]:
    """{"provider:model": 줄 번호} — 0 이 가장 강함."""
    return {key: i for i, tier in enumerate(models.model_tiers) for key in tier}


def catalog(models: ModelsConfig) -> list[str]:
    """고를 수 있는 모델 전체 (강한 순)."""
    return [key for tier in models.model_tiers for key in tier]


def spec_for(key: str, models: ModelsConfig, team: Team | None, rank_id: str) -> ModelSpec:
    """모델 키 → ModelSpec. 그 직급 목록에 있으면 그 설정(temperature 등)을, 없으면 다른 직급·보조 작업의 설정을 쓴다."""
    chains = [rank_models(rank_id, models, team), *[rank_models(r, models, team) for r in models.ranks], models.utility]
    for chain in chains:
        for spec in chain.candidates:
            if model_key(spec) == key:
                return spec
    provider, model = split_key(key)
    return ModelSpec(provider=provider, model=model)


def strongest_allowed(models: ModelsConfig, chain: RankModels, floor: int) -> str:
    """floor 줄 이하(같거나 약한)에서 가장 강한 모델 — 같은 줄이면 직급 목록에 있는 것, 비활성 provider 가 아닌 것 우선."""
    own = {model_key(s) for s in chain.candidates}
    ordered = [
        key
        for tier in models.model_tiers[floor:]
        for key in sorted(tier, key=lambda k: (k not in own, split_key(k)[0] in models.disabled_providers))
    ]
    usable = [k for k in ordered if split_key(k)[0] not in models.disabled_providers]
    return (usable or ordered or [models.model_tiers[-1][0]])[0]


@dataclass
class MemberPlan:
    """에이전트 한 명의 모델 결정 결과."""

    node_id: str
    rank_id: str
    choice: str  # 사람이 고른 모델 ("" = 기본(직급 설정))
    effective: str  # 실제로 쓸 primary 모델
    floor: int  # 쓸 수 있는 가장 강한 줄 번호 (이 값 이상 = 같거나 약한 모델만)
    limited: bool  # 고른 모델·기본 모델이 상사 규칙에 걸려 자동으로 낮춰짐
    chosen_on_chain: bool  # 본인이나 상사 중 누군가 직접 골랐는가 (아니면 직급 설정 그대로)


def plan(org: OrgNode, models: ModelsConfig, team: Team | None, choices: dict[str, str]) -> dict[str, MemberPlan]:
    """조직도 전체의 모델 결정 (위 → 아래 순서의 dict)."""
    tiers = tier_map(models)
    out: dict[str, MemberPlan] = {}

    def visit(node: OrgNode, floor: int, chosen_above: bool) -> None:
        choice = choices.get(node.id, "")
        if choice not in tiers:
            choice = ""  # 설정에서 빠진 모델이 저장돼 있으면 기본으로
        chain = rank_models(node.rank, models, team)
        wanted = choice or model_key(chain.primary)
        on_chain = chosen_above or bool(choice)
        if not on_chain or tiers.get(wanted, len(tiers)) >= floor:
            effective, limited = wanted, False
        else:
            effective, limited = strongest_allowed(models, chain, floor), True
        out[node.id] = MemberPlan(node.id, node.rank, choice, effective, floor, limited, on_chain)
        # 아래로 내려가는 한도: 직접 고른 상사보다는 한 줄 아래부터, 기본 상사와는 같은 줄까지
        child_floor = max(floor, tiers.get(effective, 0) + (1 if choice else 0))
        for child in node.children:
            visit(child, child_floor, on_chain)

    visit(org, 0, False)
    return out


def allowed_keys(models: ModelsConfig, node: OrgNode, member: MemberPlan) -> list[str]:
    """이 팀원이 고를 수 있는 모델 (강한 순). 하위가 있으면 맨 아래 줄은 뺀다 (하위가 쓸 더 낮은 모델이 남도록)."""
    tiers = tier_map(models)
    last = len(models.model_tiers) - 1
    return [k for k in catalog(models) if tiers[k] >= member.floor and not (node.children and tiers[k] >= last)]


def free_locked(member: MemberPlan) -> bool:
    """무료 모델을 쓰게 된 팀원인가 (직접 골랐거나 상사 규칙으로 낮춰짐) → 대안도 무료 모델만."""
    return member.chosen_on_chain and split_key(member.effective)[0] == FREE_PROVIDER


def member_rank_models(member: MemberPlan, models: ModelsConfig, team: Team | None) -> RankModels:
    """실행 때 쓸 모델 후보 — 고른(또는 낮춰진) 모델이 primary, 대안은 직급 목록에서 상사 규칙에 걸리는 것을 뺀 것.
    무료 모델을 쓰게 된 팀원은 대안도 무료 모델만 남긴다 (유료 API·구독으로 넘어가지 않음)."""
    base = rank_models(member.rank_id, models, team)
    if not member.chosen_on_chain:
        return base  # 본인·상사 모두 기본 → 지금과 같다
    tiers = tier_map(models)
    fallback = [
        s for s in base.candidates
        if model_key(s) != member.effective and tiers.get(model_key(s), len(tiers)) >= member.floor
        and (not free_locked(member) or s.provider == FREE_PROVIDER)
    ]
    return RankModels(primary=spec_for(member.effective, models, team, member.rank_id), fallback=fallback)


def team_rank_models(org: OrgNode, models: ModelsConfig, team: Team | None, choices: dict[str, str] | None) -> dict[str, RankModels]:
    """{에이전트 id: 모델 후보} — 워크플로우를 만들 때 한 번 계산한다."""
    plans = plan(org, models, team, choices or {})
    return {node_id: member_rank_models(p, models, team) for node_id, p in plans.items()}


@dataclass
class Change:
    node_id: str
    before: str
    after: str


def apply_choice(
    org: OrgNode, models: ModelsConfig, team: Team | None, choices: dict[str, str], node_id: str, key: str
) -> tuple[dict[str, str], list[Change]]:
    """node_id 의 모델을 key 로 바꾸고 ("" = 기본), 규칙을 어기게 된 하위를 낮춘다.

    돌려주는 값 = (새 선택, 다른 팀원 중 실제 모델이 바뀐 목록). 직접 고른 하위는 저장값도 '허용되는 가장 강한 모델'로 바꾼다.
    """
    before = plan(org, models, team, choices)
    new = {k: v for k, v in choices.items() if k in before}
    if key:
        new[node_id] = key
    else:
        new.pop(node_id, None)
    while True:  # 위에서부터 하나씩 고친다 (위를 고치면 아래 한도가 바뀌므로 다시 계산)
        current = plan(org, models, team, new)
        broken = next((p for p in current.values() if p.choice and p.limited), None)
        if broken is None:
            break
        new[broken.node_id] = broken.effective
    changes = [
        Change(nid, before[nid].effective, p.effective)
        for nid, p in current.items()
        if nid != node_id and p.effective != before[nid].effective
    ]
    return new, changes


# ─── 저장 (데이터 폴더의 JSON 파일) ─────────────────────────────────────


class MemberModelStore:
    """팀원별 모델 선택 저장소 — {팀 id: {에이전트 id: "provider:model"}}."""

    _lock = threading.Lock()

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / FILE_NAME

    def _read_all(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def load(self, team_id: str) -> dict[str, str]:
        team = self._read_all().get(team_id) or {}
        return {k: v for k, v in team.items() if isinstance(k, str) and isinstance(v, str) and v} if isinstance(team, dict) else {}

    def save(self, team_id: str, choices: dict[str, str]) -> None:
        with self._lock:
            data = self._read_all()
            cleaned = {k: v for k, v in choices.items() if v}
            if cleaned:
                data[team_id] = cleaned
            else:
                data.pop(team_id, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # 쓰는 도중 꺼져도 예전 파일이 깨지지 않게
