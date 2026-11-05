"""⭐ 역할별 추천 모델 — config/model_recommendations.yaml 을 읽어 '고르지 않은 사람'의 기본 모델 목록을 정한다.

적용 순서: 사람이 화면에서 고른 모델 > 이 추천 > models.yaml 의 직급 기본(ranks)·부장 기본(critic).
  - 팀원·팀장: core/member_models.plan 이 default_chain() 으로 추천을 기본으로 쓴다 (상하 규칙은 plan 이 그대로 지킨다).
  - 부서장:    core/dept_heads.head_chain / critic_plan 이 head_rec() 를 쓴다.
저장된 사람 선택(member_models.json · department_models.json)은 건드리지 않는다.

추천 목록을 만들 때 하는 일 (rec_chain)
  1) 강도 순서(model_tiers)에 없는 모델은 뺀다 (점검 경고)
  2) 비공개 팀이면 private_exclude(기본 freellm — 제3자 무료 티어 경유)를 뺀다
  3) 팀 업무(형식 출력)를 못 하거나 제공처에서 사라진 모델은 뺀다 (core/model_caps)
  4) 남은 목록이 모두 비활성 provider(models.yaml disabled_providers · '🆓 무료 모델만')면 None → 직급 기본으로 돌아간다
     (비공개 팀 + 무료만 모드에서 생긴다 — 화면에 ⚠️)

점검 (check — 앱 시작 때·테스트): 없는 팀·사람·역할·모델, 그림 역할의 맨 앞 모델이 그림을 못 읽음,
비공개 팀에 무료 모델, 상하 규칙(부서장 ≥ 팀장 ≥ … ) 위반을 문장 목록으로 돌려준다 (막지 않고 ⚠️ 로 알림).

테스트는 conftest 가 configure(None) 으로 끈다 (예전 직급 기본 그대로) — 추천 테스트만 켠다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.memo import SigCache, file_sig
from core.schemas import ModelsConfig, OrgNode, RankModels, Team

FILE_NAME = "model_recommendations.yaml"
ROLE_KINDS = ("lead", "manager", "drafter", "head")
_UNSET = object()
_path: Path | None | object = _UNSET  # _UNSET = config/model_recommendations.yaml, None = 끔
_cache = SigCache()


def configure(path: Path | str | None | object = _UNSET) -> None:
    """추천 파일 위치를 바꾼다. None = 끔 (테스트), 인자 없이 = 기본 파일."""
    global _path
    _path = path if path is _UNSET or path is None else Path(path)
    _load_memo[:] = [None, 0.0, None]
    _chain_memo.clear()


def path() -> Path | None:
    if _path is None:
        return None
    if _path is _UNSET:
        from core.config import CONFIG_DIR

        return CONFIG_DIR / FILE_NAME
    return _path  # type: ignore[return-value]


@dataclass
class PersonRec:
    """yaml 의 한 사람 (아직 모델 설정과 맞춰 보기 전)."""

    role: str
    why: str
    keys: list[str]
    needs_image: bool = False
    explicit: bool = False  # chain 을 사람 칸에 직접 적음 (역할의 chain 이 아님)


@dataclass
class Recs:
    """추천 파일 전체."""

    heads: dict[str, PersonRec] = field(default_factory=dict)
    teams: dict[str, dict[str, PersonRec]] = field(default_factory=dict)
    roles: dict[str, dict] = field(default_factory=dict)
    private_exclude: list[str] = field(default_factory=lambda: ["freellm"])
    estimate: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)  # 읽을 때 찾은 형식 문제 (없는 역할 등)

    @property
    def empty(self) -> bool:
        return not (self.heads or self.teams)


def _person(raw, roles: dict, where: str, problems: list[str]) -> PersonRec | None:
    if not isinstance(raw, dict):
        problems.append(f"{where}: {{role, why}} 형식이 아닙니다")
        return None
    role = str(raw.get("role") or "")
    spec = roles.get(role) or {}
    if role and role not in roles:
        problems.append(f"{where}: 없는 역할 '{role}' (roles 에 있는 것: {', '.join(roles)})")
    keys = [str(k) for k in (raw.get("chain") or spec.get("chain") or [])]
    if not keys:
        problems.append(f"{where}: 모델 목록(chain)이 비었습니다")
        return None
    needs = {str(n) for n in (raw.get("needs") or spec.get("needs") or [])}
    return PersonRec(role=role, why=str(raw.get("why") or spec.get("about") or ""), keys=keys, needs_image="image" in needs,
                     explicit=bool(raw.get("chain")))


def parse(data: dict) -> Recs:
    if not isinstance(data, dict) or data.get("enabled") is False:
        return Recs()
    problems: list[str] = []
    roles = {str(k): v for k, v in (data.get("roles") or {}).items() if isinstance(v, dict)}
    heads = {}
    for dept_id, raw in (data.get("heads") or {}).items():
        rec = _person(raw, roles, f"heads.{dept_id}", problems)
        if rec:
            heads[str(dept_id)] = rec
    teams: dict[str, dict[str, PersonRec]] = {}
    for team_id, people in (data.get("teams") or {}).items():
        for node_id, raw in (people or {}).items():
            rec = _person(raw, roles, f"teams.{team_id}.{node_id}", problems)
            if rec:
                teams.setdefault(str(team_id), {})[str(node_id)] = rec
    exclude = data.get("private_exclude")
    return Recs(heads=heads, teams=teams, roles=roles, problems=problems,
                private_exclude=[str(p) for p in exclude] if isinstance(exclude, list) else ["freellm"],
                estimate=data.get("estimate") or {})


LOAD_TTL = 2.0  # 초 — 파일이 바뀌었는지 이 간격으로만 확인 (한 화면에서 수백 번 부른다)
_load_memo: list = [None, 0.0, None]  # [경로, 확인 시각, Recs]


def load() -> Recs:
    """추천 파일 (바뀌면 다시 읽는다). 없거나 꺼져 있으면 빈 Recs."""
    p = path()
    now = time.monotonic()
    if _load_memo[0] == p and now - _load_memo[1] < LOAD_TTL:
        return _load_memo[2]
    recs = _load(p)
    if recs is not _load_memo[2]:
        _chain_memo.clear()  # 파일이 바뀜 → 사람별 계산도 새로
    _load_memo[:] = [p, now, recs]
    return recs


def _load(p: Path | None) -> Recs:
    if p is None or not p.exists():
        return Recs()

    def read() -> Recs:
        import yaml

        try:
            return parse(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        except (OSError, ValueError, yaml.YAMLError) as e:
            return Recs(problems=[f"{FILE_NAME} 를 읽지 못했습니다: {e}"])

    return _cache.get(str(p), file_sig(p), read)


# ─── 한 사람의 추천 목록 ────────────────────────────────────────────


@dataclass
class Rec:
    """모델 설정과 맞춘 추천 — chain 은 실행 후보 (앞이 우선)."""

    chain: RankModels
    why: str
    role: str
    needs_image: bool = False
    dropped: list[str] = field(default_factory=list)  # 뺀 모델과 이유 (점검·화면)

    @property
    def keys(self) -> list[str]:
        return [f"{s.provider}:{s.model}" for s in self.chain.candidates]


def _private(team: Team | None) -> bool:
    return bool(team is not None and team.private)


CHAIN_TTL = 30.0  # 초 — 모델 능력(🧪 시험 결과 등)이 바뀌면 이 시간 안에 반영 (화면은 1초마다 다시 그리므로 캐시)
_chain_memo: dict[tuple, tuple[float, Rec | None]] = {}


def rec_chain(person: PersonRec, models: ModelsConfig, team: Team | None = None, rank_id: str = "",
              private: bool | None = None) -> Rec | None:
    """yaml 의 한 사람 → 실행 후보. 쓸 모델이 하나도 없으면(모두 비활성 등) None → 직급 기본.
    모델 능력 판단(파일 확인)이 느려서 같은 입력은 CHAIN_TTL 동안 다시 계산하지 않는다."""
    private = _private(team) if private is None else private
    key = (id(person), private, rank_id, team.id if team else "", tuple(sorted(models.disabled_providers)),
           tuple(tuple(t) for t in models.model_tiers), tuple(models.ranks))
    now = time.monotonic()
    hit = _chain_memo.get(key)
    if hit and now - hit[0] < CHAIN_TTL:
        return hit[1]
    if len(_chain_memo) > 2000:
        _chain_memo.clear()
    rec = _rec_chain(person, models, team, rank_id, private)
    _chain_memo[key] = (now, rec)
    return rec


def _rec_chain(person: PersonRec, models: ModelsConfig, team: Team | None, rank_id: str, private: bool) -> Rec | None:
    from core import member_models, model_caps

    tiers = member_models.tier_map(models)
    exclude = set(load().private_exclude) if private else set()
    keep, dropped = [], []
    for key in dict.fromkeys(person.keys):
        provider = key.partition(":")[0]
        if key not in tiers:
            dropped.append(f"{key}: 강도 순서(model_tiers)에 없음")
        elif provider in exclude:
            dropped.append(f"{key}: 비공개 팀이라 뺌 (무료 모델 = 제3자 경유)")
        elif not model_caps.can(key, model_caps.TEAM_WORK):
            dropped.append(f"{key}: " + (model_caps.reasons(key, model_caps.TEAM_WORK) or ["팀 업무를 못 함"])[0])
        else:
            keep.append(key)
    if not keep or all(k.partition(":")[0] in models.disabled_providers for k in keep):
        return None
    rank_id = rank_id or next(iter(models.ranks), "")
    specs = [member_models.spec_for(k, models, team, rank_id) for k in keep]
    return Rec(chain=RankModels(primary=specs[0], fallback=specs[1:]), why=person.why, role=person.role,
               needs_image=person.needs_image, dropped=dropped)


def member_rec(team: Team | None, node: OrgNode, models: ModelsConfig) -> Rec | None:
    """팀원·팀장 한 명의 추천 (없으면 None → 직급 기본)."""
    if team is None:
        return None
    person = load().teams.get(team.id, {}).get(node.id)
    return rec_chain(person, models, team, node.rank) if person else None


def head_rec(dept_id: str | None, models: ModelsConfig, private: bool = False) -> Rec | None:
    """부서장 한 명의 추천. private = 비평할 팀이 비공개 팀 (무료 모델을 뺀다)."""
    if not dept_id:
        return None
    person = load().heads.get(dept_id)
    return rec_chain(person, models, None, private=private) if person else None


def private_free_fallback(team: Team | None, node: OrgNode, models: ModelsConfig) -> bool:
    """비공개 팀 추천이 '무료만' 같은 이유로 모두 꺼져 직급 기본(무료 모델 포함)으로 돌아갔는가 — 화면 ⚠️."""
    if not _private(team):
        return False
    person = load().teams.get(team.id, {}).get(node.id)
    return bool(person) and member_rec(team, node, models) is None


def first_usable(chain: RankModels, models: ModelsConfig, status: dict[str, str] | None = None) -> str:
    """후보 중 지금 쓸 첫 모델 — 비활성 provider 는 건너뛰고, status(provider → ok/no_key/…)가 있으면 ok 인 것만.
    하나도 없으면 맨 앞 모델."""
    keys = [f"{s.provider}:{s.model}" for s in chain.candidates]
    for key in keys:
        provider = key.partition(":")[0]
        if provider in models.disabled_providers:
            continue
        if status is not None and status.get(provider, "ok") != "ok":
            continue
        return key
    return keys[0]


# ─── 점검 ─────────────────────────────────────────────────────────────


_check_memo: dict[tuple, tuple[float, list[str]]] = {}


def check_cached(models: ModelsConfig, teams: list[Team], ranks, departments=None) -> list[str]:
    """check 를 CHAIN_TTL 동안 다시 하지 않는다 (앱은 화면을 다시 그릴 때마다 부른다)."""
    key = (id(load()), tuple(tuple(t) for t in models.model_tiers), tuple(sorted(models.disabled_providers)),
           tuple((t.id, t.private) for t in teams), tuple((d.id, tuple(d.teams)) for d in departments or []))
    now = time.monotonic()
    hit = _check_memo.get(key)
    if hit and now - hit[0] < CHAIN_TTL:
        return hit[1]
    _check_memo.clear()
    out = check(models, teams, ranks, departments)
    _check_memo[key] = (now, out)
    return out


def check(models: ModelsConfig, teams: list[Team], ranks, departments=None) -> list[str]:
    """추천 파일이 설정·규칙과 맞는지 — 문장 목록 (비어 있으면 문제 없음). departments = building.departments(...)."""
    from core import member_models, model_caps
    from core.config import team_org

    recs = load()
    out = list(recs.problems)
    if recs.empty:
        return out
    tiers = member_models.tier_map(models)
    teams_by_id = {t.id: t for t in teams}

    def tier(key: str) -> int:
        return tiers.get(key, len(tiers))

    def name(key: str) -> str:
        return key.partition(":")[2]

    def check_person(where: str, person: PersonRec, private: bool) -> None:
        for key in person.keys:
            if key not in tiers:
                out.append(f"{where}: {key} 가 models.yaml model_tiers 에 없습니다 (추천에서 빠짐)")
            elif private and person.explicit and key.partition(":")[0] in recs.private_exclude:
                out.append(f"{where}: 🔒 비공개 팀인데 {key} (무료 모델 = 제3자 경유)가 들어 있습니다 — 자동으로 뺍니다")
        usable = [k for k in person.keys if k in tiers and not (private and k.partition(":")[0] in recs.private_exclude)]
        if usable and person.needs_image and not model_caps.can(usable[0], model_caps.IMAGE_READING):
            out.append(f"{where}: 그림을 읽는 역할인데 맨 앞 모델 {name(usable[0])} 가 그림을 읽지 못합니다")
        if usable and not any(model_caps.can(k, model_caps.TEAM_WORK) for k in usable):
            out.append(f"{where}: 팀 업무(형식 출력)를 할 수 있는 모델이 없습니다")

    for team_id, people in recs.teams.items():
        team = teams_by_id.get(team_id)
        if team is None:
            out.append(f"teams.{team_id}: 없는 팀입니다")
            continue
        org = team_org(team, ranks)
        nodes = {n.id: n for n in org.walk()}
        for node_id, person in people.items():
            if node_id not in nodes:
                out.append(f"teams.{team_id}.{node_id}: 조직도에 없는 사람입니다 ({', '.join(nodes)})")
                continue
            check_person(f"{team.name} · {nodes[node_id].name or node_id}", person, team.private)
        # 상하 규칙 — 아무도 고르지 않았을 때(추천 기본)의 맨 앞 모델로
        plans = member_models.plan(org, models, team, {})

        def visit(node: OrgNode, boss: OrgNode | None) -> None:
            if boss is not None:
                mine, theirs = plans[node.id].wanted, plans[boss.id].wanted
                if tier(mine) < tier(theirs):
                    out.append(f"{team.name}: 상하 규칙 위반 — {node.name or node.id}({name(mine)}) 가 상사 "
                               f"{boss.name or boss.id}({name(theirs)}) 보다 강합니다")
            for child in node.children:
                visit(child, node)

        visit(org, None)

    depts = list(departments or [])
    dept_ids = {d.id for d in depts}
    for dept_id, person in recs.heads.items():
        if depts and dept_id not in dept_ids:
            out.append(f"heads.{dept_id}: 없는 부서입니다")
            continue
        dept = next((d for d in depts if d.id == dept_id), None)
        private = bool(dept) and any(teams_by_id[t].private for t in dept.teams if t in teams_by_id)
        check_person(f"{dept.name if dept else dept_id} 부서장", person, private)
        if dept is None:
            continue
        for tid in dept.teams:
            team = teams_by_id.get(tid)
            if team is None:
                continue
            rec = head_rec(dept_id, models, team.private)
            org = team_org(team, ranks)
            lead = member_models.plan(org, models, team, {})[org.id].wanted
            if rec and tier(member_models.model_key(rec.chain.primary)) > tier(lead):
                out.append(f"{dept.name} 부서장({name(member_models.model_key(rec.chain.primary))})이 "
                           f"{team.name} 팀장({name(lead)})보다 약합니다 — 부서장 ≥ 팀장")
    return out


# ─── 1회 실행 예상 (비용·구독 호출) ─────────────────────────────────


@dataclass
class Estimate:
    usd: float = 0.0
    subscription_calls: int = 0  # claude_cli
    free_calls: int = 0  # freellm
    paid_calls: int = 0
    unknown: list[str] = field(default_factory=list)  # 가격 미등록 모델
    gemini_calls: dict[str, int] = field(default_factory=dict)  # 모델별 Gemini 호출 (하루 한도)

    def add(self, key: str, calls: int, input_tokens: int, output_tokens: int) -> None:
        from core import usage

        provider, _, model = key.partition(":")
        if provider == "claude_cli":
            self.subscription_calls += calls
        elif provider == "freellm":
            self.free_calls += calls
        else:
            self.paid_calls += calls
        if provider == "google" or (provider == "freellm" and model.startswith("gemini")):
            self.gemini_calls[model] = self.gemini_calls.get(model, 0) + calls
        cost = usage.estimate_cost(provider, model, input_tokens * calls, output_tokens * calls)
        if cost is None:
            self.unknown.append(model)
        else:
            self.usd += cost

    def __iadd__(self, other: "Estimate") -> "Estimate":
        self.usd += other.usd
        self.subscription_calls += other.subscription_calls
        self.free_calls += other.free_calls
        self.paid_calls += other.paid_calls
        self.unknown += [u for u in other.unknown if u not in self.unknown]
        for k, v in other.gemini_calls.items():
            self.gemini_calls[k] = self.gemini_calls.get(k, 0) + v
        return self


def role_kind(node: OrgNode, root: OrgNode) -> str:
    if node.id == root.id:
        return "lead"
    return "manager" if node.children else "drafter"


def _tokens(kind: str, history: list[dict] | None, est: dict) -> tuple[int, int]:
    """한 번 부를 때의 [입력, 출력] 토큰 — 지난 기록(usage_by_agent)이 있으면 평균, 없으면 yaml 가정."""
    default = (est.get("tokens") or {}).get(kind) or {"lead": [9000, 2500], "manager": [7000, 1800],
                                                       "drafter": [4000, 1500], "head": [8000, 900]}[kind]
    calls = sum(r.get("calls") or 0 for r in history or [])
    tokens = sum(r.get("tokens") or 0 for r in history or [])
    if calls and tokens:
        share = float(est.get("output_share", 0.25))
        per = tokens / calls
        return int(per * (1 - share)), int(per * share)
    return int(default[0]), int(default[1])


def estimate_team(org: OrgNode, effective: dict[str, str], head_key: str = "",
                  history: dict[str, list[dict]] | None = None) -> Estimate:
    """팀 업무 1번 (+ 부서장 비평 1번) 예상. effective = {에이전트 id: 실제로 쓸 모델}."""
    est = load().estimate or {}
    calls = {"lead": 2, "manager": 2, "drafter": 1, "head": 1, **(est.get("calls") or {})}
    out = Estimate()
    for node in org.walk():
        key = effective.get(node.id)
        if not key:
            continue
        kind = role_kind(node, org)
        out.add(key, int(calls[kind]), *_tokens(kind, (history or {}).get(node.id), est))
    if head_key:
        out.add(head_key, int(calls["head"]), *_tokens("head", None, est))
    return out


def rank_effective(org: OrgNode, models: ModelsConfig, team: Team | None,
                   status: dict[str, str] | None = None) -> dict[str, str]:
    """추천이 없었다면(직급 기본) 쓸 모델 — 화면의 '예상 비용 변화' 비교용."""
    from core.config import rank_models

    return {n.id: first_usable(rank_models(n.rank, models, team), models, status) for n in org.walk()}


def estimate_text(e: Estimate) -> str:
    """'1회 예상 $0.031 (≈43원) · 구독 호출 3회 · Gemini 2회' 한 줄."""
    krw = float((load().estimate or {}).get("krw_per_usd") or 1400)
    parts = [f"1회 예상 ${e.usd:,.3f} (≈{e.usd * krw:,.0f}원)" + (" + 가격 미등록 " + ", ".join(sorted(set(e.unknown))) if e.unknown else "")]
    parts.append(f"구독 호출 {e.subscription_calls}회")
    gemini = sum(e.gemini_calls.values())
    if gemini:
        parts.append(f"Gemini {gemini}회")
    if e.free_calls:
        parts.append(f"무료 {e.free_calls}회")
    return " · ".join(parts)
