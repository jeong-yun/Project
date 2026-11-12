"""🧭 자유 질문 범위 확인 — 팀마다 자기 분야 질문만 받는다 (화면 그리기 없음).

  team.yaml 의 scope   : summary(한 줄) · accepts(다루는 것) · rejects(다루지 않는 것) · keywords · accept_all
  확인 순서 (check)    : '💬 자유 질문'을 맡길 때만. 정형 업무 양식은 이미 그 팀 업무라 확인하지 않는다.
    0) accept_all (기본 팀 = 범용 창구) 또는 scope 없음 → 통과
    1) 낱말 확인 (모델 없음) — 이 팀 keywords 가 다른 어느 팀보다 많이 맞으면 통과.
       직전 결과에 대한 짧은 후속 질문('이거 더 자세히' 등)도 통과.
    2) 애매하면 🧭 부서장 배정(core/building.route_request)을 모든 팀 대상으로 1회 → 이 팀을 고르면 통과,
       다른 팀을 고르면 막고 그 팀을 알려 준다. 모델을 못 쓰면(낱말 비교) 다른 팀 keywords 가 맞을 때만 막는다.
  기록 (log_decision)  : data/scope_log.jsonl 한 줄씩 — 범위 기준을 다듬을 때 본다.
  💬 결정 전 문의(core/followup.py)는 이 확인을 거치지 않는다.
"""

import json
import re
import time
from pathlib import Path

from core.schemas import Team

FOLLOWUP_MAX_CHARS = 60  # 이보다 짧은 질문만 '직전 결과에 대한 후속 질문'으로 봐 준다
FOLLOWUP_MARKERS = ("이거", "이것", "위의", "위 내용", "방금", "앞의", "앞에서", "더 자세히", "다시", "그럼", "그러면",
                    "추가로", "여기서", "그중", "그 중", "이 중", "이중", "위에서")
ROUTER_DEPT = {"id": "all", "name": "부서장 회의", "emoji": "🧭", "color": "#64748B",
               "description": "모든 부서의 팀 중 질문에 가장 알맞은 팀을 고른다. 팀 소개의 '다루는 질문'·'다루지 않는 질문'을 기준으로."}


# ─── 1) 낱말 확인 (모델 없음) ──────────────────────────────────────────


def _keyword_pattern(word: str) -> re.Pattern:
    """영문 낱말은 앞뒤가 영문이 아닐 때만 (AI 가 'MAIN' 에 맞지 않게), 한글은 그대로 (조사가 붙어도 맞게)."""
    body = re.escape(word.strip())
    if re.fullmatch(r"[A-Za-z0-9&.\- ]+", word.strip()):
        return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)
    return re.compile(body, re.IGNORECASE)


def keyword_hits(text: str, team: Team) -> list[str]:
    """질문에 맞은 이 팀의 범위 낱말."""
    if not team.scope:
        return []
    return [w for w in team.scope.keywords if w.strip() and _keyword_pattern(w).search(text)]


def _bigrams(text: str) -> set[str]:
    words = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return {w[i:i + 2] for w in words if len(w) > 1 for i in range(len(w) - 1)}


def is_followup(question: str, last_text: str, others_hit: bool) -> bool:
    """직전 결과에 대한 짧은 후속 질문인가 — 짧고, (직전 결과와 낱말이 겹치거나 / 후속 표현이 있고 다른 팀 낱말이 없음)."""
    if not last_text.strip() or len(question.strip()) > FOLLOWUP_MAX_CHARS:
        return False
    overlap = len(_bigrams(question) & _bigrams(last_text))
    if overlap >= 3:
        return True
    return not others_hit and any(m in question for m in FOLLOWUP_MARKERS)


def heuristic(question: str, team: Team, teams: list[Team], last_text: str = "") -> dict | None:
    """모델 없이 '분명히 이 팀 질문'이면 통과 결정(dict), 애매하면 None."""
    if team.scope is None or team.scope.accept_all:
        return _decision(True, team.id, "accept_all", "이 팀은 무엇이든 받는 범용 창구입니다." if team.scope else "범위가 정해지지 않은 팀입니다.")
    mine = keyword_hits(question, team)
    others = {t.id: keyword_hits(question, t) for t in teams if t.id != team.id and t.scope and not t.scope.accept_all}
    best_other = max((len(v) for v in others.values()), default=0)
    if mine and len(mine) > best_other:
        return _decision(True, team.id, "keyword", f"이 팀 분야 낱말이 맞습니다: {', '.join(mine[:4])}")
    if is_followup(question, last_text, best_other > 0):
        return _decision(True, team.id, "followup", "직전 결과에 대한 짧은 후속 질문입니다.")
    return None


# ─── 2) 🧭 부서장 배정 재사용 (보조 모델 1회) ─────────────────────────


def with_scope(team: Team) -> Team:
    """부서장 배정 프롬프트(building.route_messages)는 팀 소개(description)를 읽는다 → 소개 뒤에 '다루는/다루지 않는 질문'을 붙인 사본."""
    if not team.scope:
        return team
    s = team.scope
    extra = f" [다루는 질문: {s.summary}" + (f" / 다루지 않음: {'; '.join(s.rejects)}" if s.rejects else "") + "]"
    return team.model_copy(update={"description": team.description + extra})


def router_check(question: str, team: Team, teams: list[Team], models, route=None) -> dict:
    """부서장 배정(모든 팀 대상)으로 가장 알맞은 팀을 고른다 → 이 팀이면 통과, 아니면 막고 그 팀을 알려 준다."""
    from core import building  # 무거운 import 방지

    route = route or building.route_request
    dept = building.Department(**ROUTER_DEPT, teams=[t.id for t in teams])
    pick = route(question, dept, teams, models)  # 팀 소개에 '다루는/다루지 않는 질문'을 붙이는 일은 route_messages 가 한다 (with_scope)
    picked, how = pick["team_id"], pick.get("how", "")
    if how == "keyword":
        # 보조 모델을 못 씀 → 낱말 비교는 덜 확실하다. 이 팀 낱말은 없고 다른 팀 범위 낱말이 맞을 때만 막는다
        hits = {t.id: len(keyword_hits(question, t)) for t in teams if t.scope and not t.scope.accept_all}
        best = max(hits, key=lambda k: hits[k], default=team.id)
        if hits.get(team.id, 0) == 0 and best != team.id and hits.get(best, 0) > 0:
            return _decision(False, best, "router_keyword", f"보조 모델 없이 낱말로 판단 — {', '.join(keyword_hits(question, _by_id(teams, best))[:4])}",
                             note=pick.get("note", ""))
        return _decision(True, team.id, "router_keyword", "보조 모델을 쓰지 못해 그대로 받습니다 (낱말로는 분명한 다른 팀이 없음).",
                         note=pick.get("note", ""))
    if picked == team.id:
        return _decision(True, team.id, "router", pick.get("reason", ""), model=pick.get("model", ""))
    return _decision(False, picked, "router", pick.get("reason", ""), model=pick.get("model", ""))


def check(question: str, team: Team, teams: list[Team], models, last_text: str = "", route=None, extra: str = "") -> dict:
    """자유 질문 범위 확인 — {in_scope, team_id(맞는 팀), stage, reason, model, note}.

    extra = 질문 뒤에 붙일 참고 한 줄 (📎 첨부 파일 이름 — core/tools/attachments.hint). 낱말 확인은 질문만 보고,
    애매해서 부서장 배정으로 넘어갈 때만 붙인다 → 질문만으로 분명한 것을 파일 이름이 뒤집지 않는다."""
    return heuristic(question, team, teams, last_text) or router_check(question + extra, team, teams, models, route=route)


def _decision(in_scope: bool, team_id: str, stage: str, reason: str, model: str = "", note: str = "") -> dict:
    return {"in_scope": in_scope, "team_id": team_id, "stage": stage, "reason": reason, "model": model, "note": note}


def _by_id(teams: list[Team], team_id: str) -> Team:
    return next(t for t in teams if t.id == team_id)


# ─── 화면 글 · 기록 ─────────────────────────────────────────────────


def dept_name(team_id: str, teams: list[Team], cfg: dict | None = None) -> str:
    """팀이 속한 부서 이름 (config/departments.yaml)."""
    from core import building

    cfg = building.load_config() if cfg is None else cfg
    return next((d.name for d in building.departments([t.id for t in teams], cfg) if team_id in d.teams), "")


def blocked_message(decision: dict, team: Team, teams: list[Team], cfg: dict | None = None) -> str:
    """'이 질문은 ○○팀 범위가 아닙니다 — ◇◇팀(◇◇부)이 맞습니다'."""
    other = _by_id(teams, decision["team_id"])
    dept = dept_name(other.id, teams, cfg)
    return f"이 질문은 **{team.name}** 범위가 아닙니다 — **{other.name}**" + (f"({dept})" if dept else "") + "이 맞습니다."


def summary_line(team: Team) -> str:
    """'이 팀이 다루는 질문: …' 한 줄 (scope 없으면 빈 글)."""
    return team.scope.summary if team.scope else ""


def log_decision(path: Path, team: Team, question: str, decision: dict) -> None:
    """범위 판단 한 줄 기록 (data/scope_log.jsonl) — 실패해도 업무를 막지 않는다."""
    row = {"ts": round(time.time(), 1), "team": team.id, "question": question[:300], **decision}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_log(path: Path, limit: int = 200) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:]
