"""설정(YAML)·환경변수 로더.

읽는 파일
  config/ranks.yaml        직급 (모델·기본 아바타 기준)
  config/models.yaml       직급 → 모델 매핑
  config/settings.yaml     앱 전역 설정 (웹 검색, 저장 위치 등)
  config/verification.yaml 검증 설정 (채점 루브릭·비평가·자가 교정)
  teams/<id>/team.yaml     팀 설정 (조직도·검토 기준·툴·하네스)

모든 경로는 이 파일 위치(26-Agent-Office 폴더) 기준이므로 어느 폴더에서 실행해도 같게 동작한다.
읽은 YAML 은 core/schemas.py 의 pydantic 모델로 검증되어, 오타가 있으면 앱 시작 시 바로 알려 준다.
"""

import os
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core.schemas import ModelsConfig, OrgNode, Rank, RankModels, Team

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TEAMS_DIR = BASE_DIR / "teams"
ASSETS_DIR = BASE_DIR / "assets"
AVATAR_EXTS = ("png", "jpg", "jpeg", "webp")


class ConfigError(ValueError):
    """설정 파일 간 정합성 오류 (예: 조직도에 없는 직급을 씀)."""


def load_env() -> None:
    """API 키 읽기: 폴더 내 .env 를 우선 읽고, 없는 값은 상위(레포 루트) .env 에서 보충한다."""
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env", override=False)


WEEKDAYS = "월화수목금토일"


def today() -> date:
    """오늘 날짜 — '현 시점' 기준. 테스트에서는 환경변수 AGENT_OFFICE_TODAY=YYYY-MM-DD 로 고정한다."""
    fixed = os.getenv("AGENT_OFFICE_TODAY")
    return date.fromisoformat(fixed) if fixed else date.today()


def date_label(d: date) -> str:
    return f"{d.isoformat()} ({WEEKDAYS[d.weekday()]})"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_ranks() -> list[Rank]:
    """낮은 직급 → 높은 직급 순서의 직급 목록."""
    ranks = [Rank(**r) for r in load_yaml(CONFIG_DIR / "ranks.yaml")["ranks"]]
    ids = [r.id for r in ranks]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"ranks.yaml 에 중복된 직급 id 가 있습니다: {ids}")
    return ranks


def load_models() -> ModelsConfig:
    return ModelsConfig(**load_yaml(CONFIG_DIR / "models.yaml"))


def load_settings() -> dict:
    return load_yaml(CONFIG_DIR / "settings.yaml")


def load_verification() -> dict:
    """검증 설정. 파일이 없으면 빈 dict (각 기능이 자체 기본값 사용)."""
    path = CONFIG_DIR / "verification.yaml"
    return load_yaml(path) if path.exists() else {}


def list_teams() -> list[Team]:
    """teams/ 아래 team.yaml 이 있는 폴더 목록. '_' 로 시작하는 폴더(템플릿)는 제외."""
    return [
        load_team(team_dir.name)
        for team_dir in sorted(TEAMS_DIR.iterdir())
        if team_dir.is_dir()
        and not team_dir.name.startswith("_")
        and (team_dir / "team.yaml").exists()
    ]


def load_team(team_id: str) -> Team:
    data = load_yaml(TEAMS_DIR / team_id / "team.yaml")
    data["id"] = team_id  # 폴더 이름을 팀 id 로 사용
    return Team(**data)


def load_skills(team_id: str) -> str:
    """팀 스킬(반복 업무 지침) — teams/<팀>/skills/*.md 를 모두 읽어 하나의 텍스트로 합친다.

    스킬은 사람이 직접 쓰는 '고정 지침'이고, 교훈(lessons)은 사람의 결정에서 자동으로 쌓이는 '학습'이다.
    둘 다 모든 에이전트의 시스템 프롬프트에 들어간다. 파일을 추가·수정하면 다음 질문부터 반영된다.
    """
    skills_dir = TEAMS_DIR / team_id / "skills"
    if not skills_dir.exists():
        return ""
    parts = [f"## 스킬: {p.stem}\n{p.read_text(encoding='utf-8').strip()}" for p in sorted(skills_dir.glob("*.md"))]
    return "\n\n".join(parts)


# ─── 조직도 ──────────────────────────────────────────────────────────


def default_org(ranks: list[Rank]) -> OrgNode:
    """team.yaml 에 org 가 없을 때: 최상위 직급부터 한 줄로 내려가는 조직도를 만든다.

    ranks 가 [사원, 대리, 차장, 부장, 팀장] 이면  팀장 → 부장 → 차장 → 대리 → 사원
    """
    node = None
    for rank in ranks:  # 낮은 직급부터 하나씩 위로 감싼다
        node = OrgNode(id=rank.id, rank=rank.id, children=[node] if node else [])
    return node


def team_org(team: Team, ranks: list[Rank]) -> OrgNode:
    return team.org or default_org(ranks)


def subtree(root: OrgNode, node_id: str) -> OrgNode:
    """조직도에서 node_id 를 꼭대기로 하는 하위 조직도. (클릭한 에이전트가 그 팀의 supervisor 가 된다)"""
    for node in root.walk():
        if node.id == node_id:
            return node
    raise KeyError(f"조직도에 없는 에이전트 id: {node_id}")


def parent_map(root: OrgNode) -> dict[str, str]:
    """{자식 id: 부모 id} — 피드백에서 '누가 내 결과를 검토했는지' 찾을 때 쓴다."""
    return {child.id: node.id for node in root.walk() for child in node.children}


def node_name(node: OrgNode, ranks_by_id: dict[str, Rank]) -> str:
    return node.name or ranks_by_id[node.rank].name


def validate_config(ranks: list[Rank], models: ModelsConfig, teams: list[Team]) -> None:
    """직급·모델·팀(조직도) 설정이 서로 맞는지 확인한다. 문제가 있으면 ConfigError."""
    rank_ids = {r.id for r in ranks}
    missing = rank_ids - models.ranks.keys()
    if missing:
        raise ConfigError(f"models.yaml 에 모델이 지정되지 않은 직급: {sorted(missing)}")
    for team in teams:
        unknown = (team.personas.keys() | team.model_overrides.keys()) - rank_ids
        if unknown:
            raise ConfigError(f"teams/{team.id}/team.yaml 에 없는 직급 id: {sorted(unknown)}")
        nodes = team_org(team, ranks).walk()
        node_ids = [n.id for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ConfigError(f"teams/{team.id}/team.yaml 조직도에 중복된 에이전트 id 가 있습니다: {node_ids}")
        bad_rank = [n.id for n in nodes if n.rank not in rank_ids]
        if bad_rank:
            raise ConfigError(f"teams/{team.id}/team.yaml 조직도의 rank 가 ranks.yaml 에 없음: {bad_rank}")
        from core.harness import REGISTRY  # 순환 import 방지를 위해 여기서 import
        from core.tools import TOOLS

        used_validators = set(team.harness.validators) | {v for t in team.tasks for v in t.validators}
        unknown_validators = used_validators - REGISTRY.keys()
        if unknown_validators:
            raise ConfigError(
                f"teams/{team.id}/team.yaml 에 없는 검증기: {sorted(unknown_validators)} (사용 가능: {sorted(REGISTRY)})"
            )
        known_tools = {"web_search", *TOOLS}
        unknown_tools = set(team.tools) - known_tools
        if unknown_tools:
            raise ConfigError(f"teams/{team.id}/team.yaml tools 에 없는 툴: {sorted(unknown_tools)} (사용 가능: {sorted(known_tools)})")
        task_ids = [t.id for t in team.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ConfigError(f"teams/{team.id}/team.yaml 에 중복된 업무 id 가 있습니다: {task_ids}")


def rank_models(rank_id: str, models: ModelsConfig, team: Team | None = None) -> RankModels:
    """팀 override 가 있으면 우선 적용한 직급 모델 설정."""
    if team and rank_id in team.model_overrides:
        return team.model_overrides[rank_id]
    return models.ranks[rank_id]


# ─── 아바타 이미지 ────────────────────────────────────────────────────


def _first_existing(base: Path, stem: str) -> Path | None:
    for ext in AVATAR_EXTS:
        path = base / f"{stem}.{ext}"
        if path.exists():
            return path
    return None


def resolve_avatar(team_id: str, rank: Rank, working: bool = False, node: OrgNode | None = None) -> Path | None:
    """아바타 이미지 경로를 찾는다. 없으면 None (화면에서 emoji 로 대체).

    찾는 순서 (먼저 찾은 것을 사용):
      1) node.avatar 로 지정한 파일
      2) teams/<팀>/avatars/<에이전트id>.png   ← 에이전트 한 명만 바꾸고 싶을 때
      3) teams/<팀>/avatars/<직급id>.png       ← 팀 전체에서 직급 이미지를 바꿀 때
      4) assets/avatars/<직급id>.png           ← 공용 기본 이미지
    working=True 면 각 단계에서 '<이름>_working.png'(답변·검토 중 이미지)를 먼저 찾는다.
    """
    team_dir = TEAMS_DIR / team_id / "avatars"
    common = BASE_DIR / rank.avatar if rank.avatar else None

    candidates: list[tuple[Path, str]] = []  # (폴더, 파일 이름에서 확장자 뺀 부분)
    if node and node.avatar:
        custom = BASE_DIR / node.avatar
        candidates.append((custom.parent, custom.stem))
    if node:
        candidates.append((team_dir, node.id))
    candidates.append((team_dir, rank.id))
    if common:
        candidates.append((common.parent, common.stem))

    if working:
        for folder, stem in candidates:
            found = _first_existing(folder, f"{stem}_working")
            if found:
                return found
    for folder, stem in candidates:
        found = _first_existing(folder, stem)
        if found:
            return found
    return None
