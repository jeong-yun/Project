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

import copy
import json
import os
import re
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core.feedback_sync import knowledge_dir
from core.memo import SigCache, file_sig
from core.schemas import ModelsConfig, OrgNode, Rank, RankModels, Team

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TEAMS_DIR = BASE_DIR / "teams"
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


# libyaml(C)이 있으면 그것으로 — 순수 파이썬 파서보다 열 배쯤 빠르다 (결과는 safe_load 와 같음)
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_yaml_cache = SigCache()


def _parse_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_YAML_LOADER) or {}


def load_yaml(path: Path) -> dict:
    """YAML 읽기. 화면을 다시 그릴 때마다 읽으므로 파일이 그대로면 이전에 읽은 것을 쓴다 (복사본이라 고쳐 써도 안전)."""
    return copy.deepcopy(_yaml_cache.get(str(path), file_sig(path), lambda: _parse_yaml(path)))


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
    if extra := shared_rules(team_id, "critic"):  # knowledge/ 에 올린 부서장 검토 기준을 team.yaml 의 것 뒤에 붙인다
        verification = data["verification"] = data.get("verification") or {}
        critic = verification["critic"] = verification.get("critic") or {}
        critic["rules"] = list(dict.fromkeys([*(critic.get("rules") or []), *extra]))
    return Team(**data)


def load_skills(team_id: str) -> str:
    """팀 스킬(반복 업무 지침) — teams/<팀>/skills/*.md 를 모두 읽어 하나의 텍스트로 합친다.

    스킬은 사람이 직접 쓰는 '고정 지침'이고, 교훈(lessons)은 사람의 결정에서 자동으로 쌓이는 '학습'이다.
    둘 다 모든 에이전트의 시스템 프롬프트에 들어간다. 파일을 추가·수정하면 다음 질문부터 반영된다.
    team.yaml 의 skills_from 에 적은 공용 스킬 폴더(여러 팀이 함께 쓰는 지침)도 먼저 읽는다.
    knowledge/ 에 올린 고정 규칙은 그 팀의 promoted_rules 절에 합친다 (teams/ 쪽 파일이 없으면 절을 새로 붙인다).
    """
    files = [p for folder in skill_dirs(team_id) for p in sorted(folder.glob("*.md"))]
    sections = [(p.stem, p.read_text(encoding="utf-8").strip()) for p in files]
    if extra := shared_rules(team_id, "skill"):
        own = promoted_skill_path(team_id)
        at = next((i for i, p in enumerate(files) if p == own), None)
        base = sections[at][1] if at is not None else PROMOTED_HEAD
        merged = (PROMOTED_SKILL, "\n".join([base, *(f"- {r}" for r in extra if f"- {r}" not in base.splitlines())]))
        if at is None:
            sections.append(merged)
        else:
            sections[at] = merged
    return "\n\n".join(f"## 스킬: {name}\n{text}" for name, text in sections)


def _team_yaml(team_id: str) -> dict:
    """읽기 전용 — 복사하지 않고 캐시를 그대로 준다 (아바타 찾기처럼 자주 불리는 곳용, 고쳐 쓰지 말 것)."""
    path = TEAMS_DIR / team_id / "team.yaml"
    return _yaml_cache.get(str(path), file_sig(path), lambda: _parse_yaml(path)) if path.exists() else {}


def skill_dirs(team_id: str) -> list[Path]:
    """스킬 폴더 — skills_from 의 공용 폴더가 먼저, 그 팀의 skills/ 가 나중 (있는 폴더만)."""
    shared = _team_yaml(team_id).get("skills_from") or []
    return [d for d in (*(TEAMS_DIR / s for s in shared), TEAMS_DIR / team_id / "skills") if d.is_dir()]


# ─── 되풀이된 교훈 → 고정 규칙 (core/feedback.py promote_lesson 이 사람이 확인한 뒤에만 부른다) ─────────────
# 어디에 쓰나 — 교훈 동기화(core/feedback_sync.py)와 같은 규칙:
#   AGENT_OFFICE_KNOWLEDGE_DIR 가 있으면(실행용 사본) 그 폴더의 promoted_rules.yaml 한 파일에 쓴다. 실행용 사본의 추적 파일
#   (teams/)을 바꾸면 사본 갱신('git merge --ff-only main')이 깨지고 GitHub 에도 안 올라가기 때문. 읽을 때 팀 설정에 합친다.
#   변수가 없거나(개발·테스트) 비공개 팀(private: true)이면 예전처럼 teams/<팀>/ 에 쓴다. 예전에 teams/ 에 쓴 규칙도 계속 읽는다.
# 공용 스킬 폴더(skills_from)는 다른 팀도 읽으므로 그 팀의 skills/ 에만 쓴다.

PROMOTED_SKILL = "promoted_rules"  # teams/<팀>/skills/promoted_rules.md
PROMOTED_FILE = "promoted_rules.yaml"  # knowledge/ 안 — {팀: {skill: [규칙…], critic: [규칙…]}}
# 첫 줄은 '[' 로 시작하지 않는다 — skill_sections(맡은 절만 받는 실무자)가 절 제목으로 보고 걸러 내지 않게 (규칙 줄도 '- ')
PROMOTED_HEAD = "사람 피드백에서 올린 고정 규칙 — 같은 지적이 여러 번 되풀이되어 교훈에서 올렸다 (📚 교훈 관리의 '⬆ 고정 규칙으로 올리기')"
CRITIC_RULES = ("verification", "critic", "rules")  # team.yaml 안의 부서장 검토 기준 위치


def promoted_skill_path(team_id: str) -> Path:
    return TEAMS_DIR / team_id / "skills" / f"{PROMOTED_SKILL}.md"


def shared_rules_path(team_id: str) -> Path | None:
    """그 팀의 고정 규칙을 둘 knowledge/ 파일 — 변수가 없거나 비공개 팀이면 None (teams/ 에 쓴다)."""
    folder = knowledge_dir()
    return folder / PROMOTED_FILE if folder and not _team_yaml(team_id).get("private") else None


def shared_rules(team_id: str, kind: str) -> list[str]:
    """knowledge/ 에 올린 그 팀의 규칙 (kind: skill·critic). 없으면 []."""
    path = shared_rules_path(team_id)
    if path is None or not path.exists():
        return []
    return list(((load_yaml(path).get(team_id) or {}).get(kind)) or [])


def _add_shared_rule(path: Path, team_id: str, kind: str, rule: str) -> Path:
    data = load_yaml(path) if path.exists() else {}
    rules = data.setdefault(team_id, {}).setdefault(kind, [])
    if rule not in rules:
        rules.append(rule)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 교훈에서 올린 고정 규칙 (core/config.py 가 씀) — 팀: skill(팀 업무 지침) · critic(부서장 검토 기준)\n"
                        + yaml.safe_dump(data, allow_unicode=True, sort_keys=True, width=1000), encoding="utf-8")
    return path


def add_skill_rule(team_id: str, rule: str) -> Path:
    """팀 업무 지침(스킬)에 규칙 한 줄을 더한다 — 이미 같은 줄이 있으면 그대로. 바꾼(또는 확인한) 파일 경로."""
    if shared := shared_rules_path(team_id):
        return _add_shared_rule(shared, team_id, "skill", rule)
    path = promoted_skill_path(team_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else [PROMOTED_HEAD]
    if f"- {rule}" not in lines:
        path.write_text("\n".join([*lines, f"- {rule}"]).strip() + "\n", encoding="utf-8")
    return path


def add_critic_rule(team_id: str, rule: str) -> Path:
    """team.yaml 의 verification.critic.rules 에 한 줄을 더한다 — 주석·순서를 그대로 두려고 글자로 끼워 넣고,
    다시 읽어 '그 한 줄만 늘었는지' 코드로 확인한 뒤에만 저장한다 (아니면 ConfigError, 파일은 그대로). 이미 있으면 그대로.
    knowledge/ 를 쓰는 경우(shared_rules_path)는 team.yaml 대신 그 파일에 더한다."""
    if shared := shared_rules_path(team_id):
        return _add_shared_rule(shared, team_id, "critic", rule)
    path = TEAMS_DIR / team_id / "team.yaml"
    text = path.read_text(encoding="utf-8")
    before = yaml.safe_load(text) or {}
    node = before
    for key in CRITIC_RULES[:-1]:
        node = node.get(key) or {} if isinstance(node, dict) else {}
    if rule in ((node.get(CRITIC_RULES[-1]) if isinstance(node, dict) else None) or []):
        return path
    new = _insert_yaml_item(text, CRITIC_RULES, rule)
    expected = copy.deepcopy(before)
    node = expected
    for key in CRITIC_RULES[:-1]:
        node[key] = node.get(key) or {}
        node = node[key]
    node[CRITIC_RULES[-1]] = [*(node.get(CRITIC_RULES[-1]) or []), rule]
    if yaml.safe_load(new) != expected:
        raise ConfigError(f"teams/{team_id}/team.yaml 에 부서장 검토 기준을 자동으로 넣지 못했습니다 (파일 모양이 예상과 달라 "
                          "바꾸지 않았습니다). 파일의 verification → critic → rules 목록에 직접 한 줄 더해 주세요.")
    path.write_text(new, encoding="utf-8")
    return path


def _content(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _insert_yaml_item(text: str, keys: tuple[str, ...], item: str) -> str:
    """YAML 글에서 keys 경로의 목록 끝에 item 을 한 줄로 끼워 넣는다 (없는 키는 만든다). 맞는지는 부른 쪽이 다시 읽어 확인한다."""
    lines = text.splitlines()
    start, end, parent = 0, len(lines), -1  # 지금 보는 블록 [start, end) 과 그 블록을 연 키의 들여쓰기
    for depth, key in enumerate(keys):
        body = [i for i in range(start, end) if _content(lines[i])]
        child = _indent(lines[body[0]]) if body else parent + 2
        found = next((i for i in body if _indent(lines[i]) == child
                      and re.match(rf"^ *{re.escape(key)}\s*:\s*(#.*)?$", lines[i])), None)
        last = body[-1] if body else start - 1  # 블록의 마지막 내용 줄 — 뒤따르는 주석은 다음 절 것이라 그 앞에 넣는다
        if found is None:  # 남은 키를 블록 끝에 만든다
            new = [" " * (child + 2 * j) + f"{k}:" for j, k in enumerate(keys[depth:])]
            new.append(" " * (child + 2 * len(keys[depth:])) + "- " + json.dumps(item, ensure_ascii=False))
            return "\n".join(lines[:last + 1] + new + lines[last + 1:]) + "\n"
        is_list = depth == len(keys) - 1
        start, parent, end = found + 1, child, len(lines)
        for i in range(found + 1, len(lines)):  # 블록 끝 = 들여쓰기가 키보다 얕거나 같은 다음 내용 줄 (목록은 같은 깊이의 '- ' 도 포함)
            if _content(lines[i]) and _indent(lines[i]) <= child and not (is_list and _indent(lines[i]) == child
                                                                            and lines[i].lstrip().startswith("- ")):
                end = i
                break
    items = [i for i in range(start, end) if _content(lines[i])]
    first = next((i for i in items if lines[i].lstrip().startswith("- ")), None)
    pad = _indent(lines[first]) if first is not None else parent + 2
    at = (items[-1] if items else start - 1) + 1
    return "\n".join(lines[:at] + [" " * pad + "- " + json.dumps(item, ensure_ascii=False)] + lines[at:]) + "\n"


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


def validate_config(ranks: list[Rank], models: ModelsConfig, teams: list[Team]) -> None:
    """직급·모델·팀(조직도) 설정이 서로 맞는지 확인한다. 문제가 있으면 ConfigError."""
    rank_ids = {r.id for r in ranks}
    missing = rank_ids - models.ranks.keys()
    if missing:
        raise ConfigError(f"models.yaml 에 모델이 지정되지 않은 직급: {sorted(missing)}")
    validate_model_tiers(models, teams)
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

        bad_also = {v: ids for v, ids in team.harness.also_at.items() if v not in team.harness.final_only or set(ids) - set(node_ids)}
        if bad_also:
            raise ConfigError(
                f"teams/{team.id}/team.yaml harness.also_at 오류: {bad_also} — 검증기는 final_only 에 있어야 하고 에이전트 id 는 조직도에 있어야 함"
            )
        used_validators = set(team.harness.validators) | set(team.harness.final_only) | {v for t in team.tasks for v in t.validators}
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
        missing_skills = [d for d in team.skills_from if not (TEAMS_DIR / d).is_dir()]
        if missing_skills:
            raise ConfigError(f"teams/{team.id}/team.yaml skills_from 에 없는 폴더: {missing_skills} (teams/ 기준 경로)")
    team_ids = {t.id for t in teams}
    taken = [t.id for t in teams if t.legacy and t.legacy.team in team_ids]
    if taken:
        raise ConfigError(f"legacy.team 은 지금은 없는 옛 팀 id 여야 합니다 (지금 있는 팀과 겹침): {taken}")


def used_model_keys(models: ModelsConfig, teams: list[Team] = ()) -> set[str]:
    """설정에 쓰인 모든 모델 ("provider:model") — 직급·보조 작업·비평가(critic)·팀 override."""
    chains = [*models.ranks.values(), models.utility, *([models.critic] if models.critic else []),
              *[rm for t in teams for rm in t.model_overrides.values()]]
    return {f"{s.provider}:{s.model}" for chain in chains for s in chain.candidates}


def validate_model_tiers(models: ModelsConfig, teams: list[Team] = ()) -> list[str]:
    """models.yaml 의 model_tiers(모델 강도 순서)가 올바른가 — 형식, 중복. 형식·중복은 ConfigError.

    쓰였는데 model_tiers 에 없는 모델은 오류로 멈추지 않고 목록으로 돌려준다 — 앱이 이름으로 강도를 추정해 넣는다
    (core/model_catalog.py merge). 화면 사이드바에 '강도 추정으로 넣은 모델'로 알린다.
    """
    listed = [key for tier in models.model_tiers for key in tier]
    bad = [k for k in listed if k.partition(":")[0] not in models.providers or not k.partition(":")[2]]
    if bad:
        raise ConfigError(f"models.yaml model_tiers 형식 오류 (provider:model, provider 는 {sorted(models.providers)}): {bad}")
    dup = sorted({k for k in listed if listed.count(k) > 1})
    if dup:
        raise ConfigError(f"models.yaml model_tiers 에 두 번 나온 모델: {dup}")
    if any(not tier for tier in models.model_tiers):
        raise ConfigError("models.yaml model_tiers 에 빈 줄([])이 있습니다.")
    return sorted(used_model_keys(models, teams) - set(listed))


def rank_models(rank_id: str, models: ModelsConfig, team: Team | None = None) -> RankModels:
    """팀 override 가 있으면 우선 적용한 직급 모델 설정."""
    if team and rank_id in team.model_overrides:
        return team.model_overrides[rank_id]
    return models.ranks[rank_id]


# ─── 아바타 이미지 ────────────────────────────────────────────────────


_listing_cache = SigCache()


def _listing(folder: Path) -> frozenset[str]:
    """폴더 안 파일 이름들 (소문자 — Windows 처럼 대소문자 무시). 파일을 넣거나 지우면 폴더 수정 시각이 바뀌어 다시 읽는다."""
    def read() -> frozenset[str]:
        try:
            return frozenset(name.lower() for name in os.listdir(folder))
        except OSError:
            return frozenset()

    return _listing_cache.get(str(folder), file_sig(folder), read)


def _first_existing(base: Path, stem: str) -> Path | None:
    names = _listing(base)  # 파일마다 있는지 묻지 않고 폴더 목록 한 번으로 (조직도를 그릴 때마다 수십 번 불림)
    for ext in AVATAR_EXTS:
        path = base / f"{stem}.{ext}"
        if path.name.lower() in names:
            return path
    return None


def resolve_avatar(team_id: str, rank: Rank, working: bool = False, node: OrgNode | None = None) -> Path | None:
    """아바타 이미지 경로를 찾는다. 없으면 None (화면에서 emoji 로 대체).

    찾는 순서 (먼저 찾은 것을 사용):
      1) teams/<팀>/avatars/<에이전트id>.png   ← 에이전트 한 명만 바꾸고 싶을 때
      2) teams/<팀>/avatars/<직급id>.png       ← 팀 전체에서 직급 이미지를 바꿀 때
      3) assets/avatars/<직급id>.png           ← 공용 기본 이미지
    working=True 면 각 단계에서 '<이름>_working.png'(답변·검토 중 이미지)를 먼저 찾는다.
    """
    team_dirs = [TEAMS_DIR / team_id / "avatars"]
    legacy = (_team_yaml(team_id).get("legacy") or {}).get("team")
    if legacy:  # 팀을 나눈 경우 옛 팀 폴더(teams/<옛 id>/avatars)에 남은 이미지도 찾는다
        team_dirs.append(TEAMS_DIR / legacy / "avatars")
    common = BASE_DIR / rank.avatar if rank.avatar else None

    candidates: list[tuple[Path, str]] = []  # (폴더, 파일 이름에서 확장자 뺀 부분)
    for team_dir in team_dirs:
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
