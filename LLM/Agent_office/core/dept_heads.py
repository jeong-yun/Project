"""👔 부서 부장 — 부서마다 한 명. ① 그 부서 모든 팀의 ⚖️ 비평가 ② 🧭 부서장(요청을 알맞은 팀으로 보냄).

부장은 팀 조직도 안의 팀원이 아니다 (팀 조직도는 그대로). 부서 전체를 보는 사람이다.
  이름·관점(persona) : config/departments.yaml 의 부서마다 head: {name, persona} (없으면 '<부서 이름> 부장' + 부서 소개)
  모델               : 화면 '🎛 부서별 모델'에서 고른다 → 데이터 폴더 department_models.json {부서 id: "provider:model"}
                       (config/*.yaml 은 바꾸지 않는다. 바꾼 모델은 다음에 맡기는 업무부터)

비평가 모델 정하는 순서 (critic_plan)
  1. 부장 모델을 화면에서 골랐으면 그 모델
  2. ⭐ 추천 (config/model_recommendations.yaml heads — core/model_recs.py). 비공개 팀이면 추천에서 무료 모델을 뺀다.
     추천이 모두 꺼져 있으면(비공개 팀 + '🆓 무료 모델만' 등) 다음 순서로
  3. 비공개 팀(team.yaml private: true)이면 팀장 모델 후보 그대로 — 문서가 팀원이 쓰는 제공처 밖으로 나가지 않게
  4. config/models.yaml 의 critic: = 부장 기본 (팀장과 같거나 더 강한 모델 — 지금은 Claude 구독 Opus 5.5 → gpt-5.5 → … → 무료)
  5. utility (critic 을 지웠을 때만 — 예전 동작)
  고른 모델의 대안(fallback) = 3·4 의 후보 중 고른 모델보다 강하지 않은 것 (대안으로 넘어가도 더 비싼 모델로 올라가지 않음).
  무료(freellm) 모델을 골랐으면 대안도 무료만 (팀원 모델 선택과 같은 규칙). '🆓 무료 모델만 사용'이면 유료가 꺼지므로
  후보 중 무료 모델이 쓰인다 — 후보에 무료 모델이 하나도 없으면 utility 의 무료 모델을 끝에 붙인다.

강도 규칙 (결정)
  부장은 조직의 맨 위다 — 기본 모델은 팀장과 같거나 더 강하게 둔다 (2026-11-05: 예전 기본 utility(gemini-3.5-flash)가
  팀장(Opus 5.5)보다 약했던 문제를 고침). 화면에서 고를 때는 위아래 제한을 두지 않는다 — 비용·한도 때문에 가벼운 모델도 고를 수 있다.
  대신 비평가가 채점 대상(팀장)보다 약한 모델이면 화면과 검증 기록에 경고한다 (member_models.head_warning).

비용·한도 (기본 = claude_cli Opus 5.5)
  비평: 업무 하나에 1번 + 불합격이면 (자가 교정 + 다시 비평) × 최대 max_self_corrections 번 — 모두 구독 한도를 쓴다.
  💬 수정 요청(core/followup.py): 수정본마다 검토 1번 + 추가로 고칠 점이 있으면 (부장 직접 수정 1 + 다시 검토 1),
  불합격이면 (팀장 다시 고침 + 다시 검토 1) — 둘을 합쳐 같은 상한. 부장의 직접 수정도 부장 모델(구독 한도)을 쓴다.
  🧭 팀 배정(route_request): 팀이 둘 이상인 부서에서 맡길 때마다 모델 1번 (낱말 비교는 모델이 실패할 때만). claude -p 는 한 번에
  약 8초의 시작 시간이 더해진다. 🛎 인포메이션 센터(core/reception.py)는 이 기본을 쓰지 않고 가벼운 assist_chain 을 쓴다.

비공개 팀 (결정)
  부장 모델을 고르지 않았으면 팀장 모델로 비평한다 (위 2). 골랐는데 그 제공처가 팀원들이 쓰는 제공처에 없으면 경고한다.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from core import member_models, model_recs
from core.schemas import ModelsConfig, ModelSpec, RankModels, Team

FILE_NAME = "department_models.json"
FREE = member_models.FREE_PROVIDER

SOURCE_LABELS = {
    "chosen": "부장 모델 (화면에서 고름)",
    "recommended": "⭐ 추천 (config/model_recommendations.yaml)",
    "private": "팀장 모델 (비공개 팀 — 팀 밖 제공처로 보내지 않음)",
    "critic": "부장 기본 — models.yaml critic",
    "utility": "models.yaml utility (critic 없음)",
}


@dataclass
class DeptHead:
    """부서 부장 한 명."""

    dept_id: str
    dept_name: str
    emoji: str
    name: str  # 예: '보안부 부장'
    persona: str  # 비평·배정할 때의 관점

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.name}"

    def critic_persona(self) -> str:
        return f"너는 {self.name}이다. {self.persona}".strip()


def head_of(dept) -> DeptHead:
    """building.Department → 부장. departments.yaml 에 head 가 없으면 부서 이름·소개로 만든다."""
    raw = getattr(dept, "head", None) or {}
    name = str(raw.get("name") or f"{dept.name} 부장")
    persona = str(raw.get("persona") or f"부서 소개: {dept.description} 이 부서 결과물을 마지막으로 확인하는 책임자로서 본다.")
    return DeptHead(dept_id=dept.id, dept_name=dept.name, emoji=dept.emoji, name=name, persona=persona)


def dept_private(dept, teams_by_id: dict) -> bool:
    """이 부서에 비공개 팀이 하나라도 있는가 — 🧭 배정 요청에 개인 문서가 들어갈 수 있으므로 추천에서 무료 모델을 뺀다."""
    return any(getattr(teams_by_id.get(t), "private", False) for t in getattr(dept, "teams", []))


# ─── 모델 후보 ─────────────────────────────────────────────────────────


def base_chain(models: ModelsConfig) -> tuple[RankModels, str]:
    """고르지 않았을 때의 비평가·부서장 모델 — models.yaml critic → utility."""
    return (models.critic, "critic") if models.critic else (models.utility, "utility")


def assist_chain(models: ModelsConfig, choice: str = "") -> RankModels:
    """🛎 인포메이션 센터처럼 부서장이 아닌 가벼운 안내 호출 — 고른 모델이 있으면 그것, 없으면 utility (부장 기본을 쓰지 않음)."""
    choice = valid_choice(choice, models)
    return _with_free_tail(chosen_chain(choice, models), models) if choice else models.utility


def valid_choice(key: str, models: ModelsConfig) -> str:
    """저장된 선택이 지금 고를 수 있는 모델이면 그대로, 아니면 '' (지운 직접 입력 모델 등)."""
    return key if key and key in member_models.tier_map(models) else ""


def spec_of(key: str, models: ModelsConfig) -> ModelSpec:
    """모델 키 → ModelSpec. 설정(critic·utility·직급) 어딘가에 있으면 그 temperature 등을 그대로 쓴다."""
    chains = [*([models.critic] if models.critic else []), models.utility, *models.ranks.values()]
    for chain in chains:
        for spec in chain.candidates:
            if member_models.model_key(spec) == key:
                return spec
    provider, model = member_models.split_key(key)
    return ModelSpec(provider=provider, model=model)


def _with_free_tail(chain: RankModels, models: ModelsConfig) -> RankModels:
    """후보에 무료 모델이 없으면 utility 의 무료 모델을 끝에 붙인다 ('🆓 무료 모델만 사용'에서도 비평이 돌게)."""
    if any(s.provider == FREE for s in chain.candidates):
        return chain
    free = [s for s in models.utility.candidates if s.provider == FREE]
    return RankModels(primary=chain.primary, fallback=[*chain.fallback, *free])


def chosen_chain(key: str, models: ModelsConfig) -> RankModels:
    """고른 모델이 primary, 대안 = 기본 후보 중 고른 모델보다 강하지 않은 것 (무료를 골랐으면 무료만)."""
    tiers = member_models.tier_map(models)
    floor = tiers.get(key, len(tiers))
    base, _ = base_chain(models)
    pool = [*base.candidates, *[s for s in models.utility.candidates if s not in base.candidates]]
    free_locked = member_models.split_key(key)[0] == FREE
    fallback, seen = [], {key}
    for spec in pool:
        k = member_models.model_key(spec)
        if k in seen or tiers.get(k, len(tiers)) < floor or (free_locked and spec.provider != FREE):
            continue
        seen.add(k)
        fallback.append(spec)
    return RankModels(primary=spec_of(key, models), fallback=fallback)


def head_chain(models: ModelsConfig, choice: str = "", dept_id: str | None = None,
               private: bool = False) -> tuple[RankModels, str]:
    """부장 모델 후보와 출처 — 🧭 팀 배정에 쓴다 (비공개 팀 규칙은 팀이 정해진 뒤 critic_plan 에서).
    dept_id 를 주면 ⭐ 추천(heads)을 기본으로 쓴다. private = 비공개 팀만 있는 부서 (추천에서 무료 모델을 뺀다)."""
    choice = valid_choice(choice, models)
    if choice:
        return _with_free_tail(chosen_chain(choice, models), models), "chosen"
    rec = model_recs.head_rec(dept_id, models, private)
    if rec is not None:
        return (rec.chain if private else _with_free_tail(rec.chain, models)), "recommended"
    chain, source = base_chain(models)
    return _with_free_tail(chain, models), source


def head_why(dept_id: str | None, models: ModelsConfig, private: bool = False) -> str:
    """부서장 ⭐ 추천 이유 (없으면 '')."""
    rec = model_recs.head_rec(dept_id, models, private)
    return rec.why if rec else ""


# ─── 비평가 계획 (팀 하나) ────────────────────────────────────────────


@dataclass
class CriticPlan:
    chain: RankModels
    source: str  # chosen | private | critic | utility
    head: DeptHead | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def who(self) -> str:
        return self.head.label if self.head else "⚖️ 비평가"

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)


def _name(key: str) -> str:
    return member_models.split_key(key)[1]


def critic_warnings(models: ModelsConfig, team: Team | None, critic_key: str, lead_key: str,
                    member_providers: set[str]) -> list[str]:
    """비평가가 채점 대상(팀장)보다 약한가 · 비공개 팀인데 비평가 제공처가 팀원 제공처 밖인가."""
    out = []
    weak = member_models.head_warning(models, critic_key, lead_key, "채점 대상 팀장")
    if weak:
        out.append(weak)
    provider = member_models.split_key(critic_key)[0]
    if team is not None and team.private and member_providers and provider not in member_providers:
        out.append(f"🔒 비공개 팀: 비평가 제공처({provider})가 팀원 제공처({' · '.join(sorted(member_providers))})에 없어 "
                   f"문서 내용이 {provider} 로도 보내집니다.")
    return out


def critic_plan(models: ModelsConfig, team: Team | None, chains: dict[str, RankModels], lead_id: str,
                head: DeptHead | None = None, choice: str = "") -> CriticPlan:
    """이 팀의 비평가 모델 후보. chains = 팀원별 모델 후보 (core/member_models.team_rank_models), lead_id = 팀장 id."""
    choice = valid_choice(choice, models)
    lead = chains.get(lead_id)
    private = bool(team is not None and team.private)
    rec = None if choice else model_recs.head_rec(head.dept_id if head else None, models, private)
    if choice:
        chain, source = _with_free_tail(chosen_chain(choice, models), models), "chosen"
    elif rec is not None:  # ⭐ 추천 — 비공개 팀이면 무료 모델을 뺀 목록 그대로 (무료 꼬리도 붙이지 않음)
        chain, source = (rec.chain if private else _with_free_tail(rec.chain, models)), "recommended"
    elif team is not None and team.private and lead is not None:
        chain, source = lead, "private"
    else:
        chain, source = head_chain(models)
    lead_key = member_models.model_key(lead.primary) if lead else ""
    providers = {c.primary.provider for c in chains.values()}
    warnings = critic_warnings(models, team, member_models.model_key(chain.primary), lead_key, providers)
    return CriticPlan(chain=chain, source=source, head=head, warnings=warnings)


# ─── 저장 (데이터 폴더의 JSON 파일) ─────────────────────────────────────


class DeptModelStore:
    """부서 부장 모델 저장소 — {부서 id: "provider:model"}. 없으면 기본(critic → utility)."""

    _lock = threading.Lock()

    def __init__(self, data_dir: Path | str):
        self.path = Path(data_dir) / FILE_NAME

    def load_all(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v}

    def get(self, dept_id: str) -> str:
        return self.load_all().get(dept_id, "")

    def set(self, dept_id: str, key: str) -> None:
        with self._lock:
            data = self.load_all()
            if key:
                data[dept_id] = key
            else:
                data.pop(dept_id, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # 쓰는 도중 꺼져도 예전 파일이 깨지지 않게
