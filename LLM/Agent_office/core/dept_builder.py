"""🆕 새 부서 만들기 — 🛎 인포메이션 센터의 제안(core/reception.NewDeptProposal) → 파일 초안 → 검증 → 저장 (화면 그리기 없음).

  draft()    : 제안 → (부서 항목 YAML, teams/<새 팀>/team.yaml 내용). team.yaml 은 teams/_template/team.yaml 을 바탕으로
               조직도·페르소나·scope·검토 기준·자유 질문 예시를 채운다. **도구(tools)는 비운다** — 웹 검색처럼 밖으로 나가는
               도구는 사람이 확인하고 켠다 (필요한 도구는 주석으로만 적는다).
  validate() : 사람이 미리 보기에서 고친 글을 다시 읽어 확인 — YAML 문법, id 모양, 기존 부서·팀과 겹침,
               Team 스키마(core/schemas.py), 직급·툴·검증기(core/config.validate_config).
  save()     : 사람이 [확인하고 저장]을 눌렀을 때만. teams/<id>/team.yaml 을 새로 쓰고, config/departments.yaml 의
               'other:' 앞에 부서 항목을 끼워 넣는다 (주석은 그대로). 다시 읽어 확인이 안 되면 둘 다 되돌린다.
  audit()    : 자율 실행과 같은 해시 체인 감사 로그(data/autonomy/audit.jsonl)에 dept_created / dept_deferred 한 줄.
"""

import re
from datetime import datetime
from pathlib import Path

import yaml

from core.config import CONFIG_DIR, TEAMS_DIR, ConfigError, load_yaml, validate_config
from core.schemas import Team

ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
OTHER_RE = re.compile(r"^(?:#[^\n]*\n)*other:", re.MULTILINE)  # 'other:' 와 바로 위 주석 줄들
DEFAULT_COLOR = "#14B8A6"
COLORS = ["#14B8A6", "#F97316", "#6366F1", "#84CC16", "#EC4899", "#06B6D4", "#EAB308"]


class DeptBuildError(ValueError):
    """초안을 저장할 수 없는 이유 (화면에 그대로 보여 준다)."""


def _dump(data) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)


def pick_color(used: set[str]) -> str:
    return next((c for c in COLORS if c.upper() not in {u.upper() for u in used}), DEFAULT_COLOR)


def draft(p: dict, question: str, rank_ids: list[str], used_colors: set[str] = frozenset(),
          template: Path | None = None) -> tuple[str, str]:
    """제안(dict) → (부서 항목 YAML, team.yaml 글)."""
    dept = {
        "id": p["dept_id"], "name": p["dept_name"], "emoji": p.get("emoji") or "🆕", "color": pick_color(set(used_colors)),
        "description": p["dept_description"], "teams": [p["team_id"]],
        "head": {"name": f"{p['dept_name']} 부장", "persona": p.get("head_persona") or "요청한 내용에 답했는지, 근거 없는 단정이 없는지를 본다."},
    }
    base = load_yaml(template or TEAMS_DIR / "_template" / "team.yaml")
    members = p["members"]
    lead, rest = members[0], members[1:]
    team = {
        **{k: v for k, v in base.items() if k not in ("id",)},
        "name": p["team_name"],
        "description": p["team_description"],
        "scope": {"summary": p["scope_summary"], "accepts": list(p.get("accepts") or []),
                  "rejects": list(p.get("rejects") or []), "keywords": list(p.get("keywords") or [])},
        "free_examples": [{"title": "처음 맡긴 질문", "question": question, "goal": "", "success_criteria": [],
                           "constraints": [], "output_format": ""}],
        "personas": {r: "" for r in rank_ids},
        "org": {"id": lead["id"], "rank": lead["rank"], "name": lead["name"], "specialty": lead["specialty"],
                "children": [{"id": m["id"], "rank": m["rank"], "name": m["name"], "specialty": m["specialty"]} for m in rest]},
        "review_criteria": list(p.get("review_criteria") or []) or ["요청한 내용에 빠짐없이 답했는가"],
        "tools": [],  # 위험한 도구(웹 검색 등)는 기본으로 끈다 — 필요하면 사람이 켠다
        "tasks": [],
    }
    tools_note = "\n".join(f"#   - {t}" for t in p.get("tools_needed") or []) or "#   - (없음)"
    header = (f"# 🛎 인포메이션 센터가 만든 초안 ({datetime.now():%Y-%m-%d}) — teams/_template/team.yaml 바탕\n"
              f"# 원래 질문: {' '.join(question.split())[:120]}\n"
              "# 필요한 도구 (제안 — 아직 켜지 않음. 쓸 수 있는 툴만 tools 에 적을 수 있다: web_search 등 core/tools):\n"
              f"{tools_note}\n")
    return _dump(dept), header + _dump(team)


def validate(dept_text: str, team_text: str, *, team_ids: set[str], dept_ids: set[str], ranks, models,
             teams_dir: Path | None = None) -> tuple[dict, dict]:
    """(부서 dict, team.yaml dict) — 문제가 있으면 DeptBuildError (사람이 읽을 이유)."""
    try:
        dept = yaml.safe_load(dept_text)
        team = yaml.safe_load(team_text)
    except yaml.YAMLError as e:
        raise DeptBuildError(f"YAML 문법 오류 — {e}") from e
    if not isinstance(dept, dict) or not isinstance(team, dict):
        raise DeptBuildError("부서 항목과 team.yaml 은 '키: 값' 모양이어야 합니다.")
    did, teams = str(dept.get("id") or ""), dept.get("teams") or []
    if not ID_RE.match(did):
        raise DeptBuildError(f"부서 id '{did}' — 영문 소문자로 시작하는 소문자·숫자·밑줄 2~31자여야 합니다.")
    if did in dept_ids or did == "etc":
        raise DeptBuildError(f"부서 id '{did}' 는 이미 있습니다.")
    if not dept.get("name"):
        raise DeptBuildError("부서 이름(name)이 비었습니다.")
    if not (isinstance(teams, list) and len(teams) == 1):
        raise DeptBuildError("부서의 teams 에는 새 팀 id 하나만 적어 주세요.")
    tid = str(teams[0])
    if not ID_RE.match(tid) or tid.startswith("_"):
        raise DeptBuildError(f"팀 id '{tid}' — 영문 소문자로 시작하는 소문자·숫자·밑줄 2~31자여야 합니다.")
    if tid in team_ids or (Path(teams_dir or TEAMS_DIR) / tid).exists():
        raise DeptBuildError(f"팀 id '{tid}' 는 이미 있습니다.")
    head = dept.get("head")
    if head is not None and not isinstance(head, dict):
        raise DeptBuildError("head 는 {name, persona} 모양이어야 합니다.")
    try:
        model = Team(**{**team, "id": tid})
    except Exception as e:  # noqa: BLE001 — pydantic 오류 글을 그대로 보여 준다
        raise DeptBuildError(f"team.yaml 이 팀 형식과 맞지 않습니다 — {e}") from e
    if model.org is None or not model.org.children:
        raise DeptBuildError("조직도(org)에 팀장과 팀원(children)이 있어야 합니다.")
    try:
        validate_config(ranks, models, [model])
    except ConfigError as e:
        raise DeptBuildError(str(e).replace(f"teams/{tid}/team.yaml", "team.yaml")) from e
    return dept, team


def insert_department(text: str, dept: dict) -> str:
    """departments.yaml 글에 부서 항목을 끼워 넣는다 — 'other:'(와 바로 위 주석) 앞, 없으면 맨 끝."""
    block = "\n".join("  " + line if line else line for line in _dump([dept]).rstrip().splitlines()) + "\n"
    m = OTHER_RE.search(text)
    if m:
        head = text[:m.start()].rstrip("\n") + "\n\n"
        return head + block + "\n" + text[m.start():]
    if not re.search(r"^departments:", text, re.MULTILINE):
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + "departments:\n"
    return text.rstrip("\n") + "\n\n" + block


def save(dept: dict, team_text: str, *, config_path: Path | None = None, teams_dir: Path | None = None) -> Path:
    """사람이 확인한 초안을 저장 → 새 team.yaml 경로. 실패하면 되돌리고 DeptBuildError."""
    config_path = Path(config_path or CONFIG_DIR / "departments.yaml")
    teams_dir = Path(teams_dir or TEAMS_DIR)
    tid = dept["teams"][0]
    folder = teams_dir / tid
    if folder.exists():
        raise DeptBuildError(f"teams/{tid} 폴더가 이미 있습니다.")
    before = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    folder.mkdir(parents=True)
    path = folder / "team.yaml"
    try:
        path.write_text(team_text, encoding="utf-8")
        config_path.write_text(insert_department(before, dept), encoding="utf-8")
        after = load_yaml(config_path)  # 다시 읽어 확인
        if not any(d.get("id") == dept["id"] and tid in (d.get("teams") or []) for d in after.get("departments") or []):
            raise DeptBuildError("departments.yaml 에 부서가 들어가지 않았습니다.")
    except Exception as e:
        path.unlink(missing_ok=True)
        folder.rmdir()
        config_path.write_text(before, encoding="utf-8")
        raise e if isinstance(e, DeptBuildError) else DeptBuildError(f"저장하지 못했습니다 — {e}") from e
    return path


def audit(data_dir: Path, settings: dict, event: str, **fields) -> None:
    """자율 실행과 같은 해시 체인 감사 로그에 한 줄 (실패해도 화면을 막지 않는다)."""
    from core import autonomy

    try:
        cfg = autonomy.load_config(settings)
        autonomy.AuditLog(Path(data_dir) / cfg.audit_file).append(event, decider="user", via="info_center", **fields)
    except Exception:  # noqa: BLE001
        pass
