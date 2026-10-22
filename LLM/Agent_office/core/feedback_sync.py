"""피드백(교훈·결정 요약)을 git 에 올릴 수 있는 파일로 — knowledge/ 폴더 (공개 저장소에 올라가도 되는 것만).

data/runs.db 는 .gitignore 라 GitHub 에 올라가지 않는다. 그중 '다음 업무를 더 잘하게 만드는 학습'만 텍스트로 꺼내 둔다.

올리는 것
  lessons.jsonl   : 에이전트별 교훈 (팀·에이전트·칭찬/개선·문장·켜짐 여부·작성 시각)
  decisions.jsonl : 결정 요약 (팀·문의 대상·채택/반려·채택한 에이전트·직접 수정 여부·만족도·시각)
  교훈.md          : 사람이 GitHub 에서 읽기 좋게 팀·에이전트별로 묶은 교훈
올리지 않는 것 (공개 저장소)
  질문·최종 답변·코멘트 원문 (보유 종목·매수가·여행 일정 같은 개인 정보 — 교훈 끝에 붙은 '사람 코멘트'도 뗀다),
  사용량·API 키 정보

언제
  - 결정·교훈이 바뀔 때마다 자동으로 다시 쓴다 — 환경변수 AGENT_OFFICE_KNOWLEDGE_DIR 가 있을 때만
    (실행용 Agent_Office_실행.bat 이 지정한다. 테스트·개발용 실행은 지정하지 않으므로 파일을 건드리지 않는다)
  - 앱 시작 때 knowledge/ 의 교훈 중 DB 에 없는 것을 가져온다 → 다른 PC 에서 clone 하면 교훈이 이어진다
  - 수동: python scripts/sync_feedback.py [export|import] [--dir 경로]
파일 내용이 그대로면 다시 쓰지 않는다 (git 에 불필요한 변경이 생기지 않게). 실패해도 업무·결정은 막지 않는다.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

ENV = "AGENT_OFFICE_KNOWLEDGE_DIR"
LESSONS, DECISIONS, READABLE = "lessons.jsonl", "decisions.jsonl", "교훈.md"
DECIDED = ("adopted", "rejected")
# core/feedback.py 가 교훈 끝에 붙이는 사람 코멘트 원문 — '(사람 코멘트: …)' 또는 ' 사람 코멘트: …' (결정 전 문의 기록 포함)
COMMENT = re.compile(r"\s*\(?사람 코멘트:", re.S)


def public_lesson(text: str) -> str:
    """공개 저장소에 올릴 교훈 문장 — 끝에 붙은 사람 코멘트 원문을 뗀다 (코멘트에 개인 정보가 들어갈 수 있다)."""
    return COMMENT.split(text, maxsplit=1)[0].rstrip()


def knowledge_dir() -> Path | None:
    value = os.getenv(ENV)
    return Path(value) if value else None


def _dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def lesson_rows(storage) -> list[dict]:
    rows = storage._run_sql("SELECT * FROM lessons ORDER BY created_at, id", ())
    return [
        {"team_id": r["team_id"], "agent_id": r["agent_id"], "kind": r["kind"], "lesson": public_lesson(r["lesson"]),
         "active": bool(r["active"]), "created_at": r["created_at"], "run_id": r["run_id"] or ""}
        for r in rows
    ]


def decision_rows(storage) -> list[dict]:
    """결정 요약 — 질문·답변·코멘트 원문은 넣지 않는다."""
    placeholders = ",".join("?" * len(DECIDED))
    rows = storage._run_sql(
        f"SELECT id, team_id, root_id, status, decision, created_at, decided_at FROM runs "
        f"WHERE status IN ({placeholders}) ORDER BY created_at, id", DECIDED,
    )
    out = []
    for r in rows:
        decision = json.loads(r["decision"]) if r["decision"] else {}
        out.append({
            "run_id": r["id"], "team_id": r["team_id"], "root_id": r["root_id"], "status": r["status"],
            "adopted_agent": decision.get("adopted_node_id") or "", "edited": bool(decision.get("edited")),
            "score": decision.get("score"), "commented": bool((decision.get("comment") or "").strip()),
            "created_at": r["created_at"], "decided_at": r["decided_at"] or "",
        })
    return out


def readable(lessons: list[dict], decisions: list[dict]) -> str:
    """교훈.md — 팀 → 에이전트별 교훈 (최신 먼저), 팀마다 결정 수·평균 만족도."""
    by_team: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in lessons:
        by_team[item["team_id"]][item["agent_id"]].append(item)
    stats: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        stats[d["team_id"]].append(d)
    lines = [
        "# 에이전트 교훈 (자동 생성 — 직접 고치지 마세요)",
        "",
        "사람이 결과를 채택·반려할 때 에이전트마다 만든 교훈입니다. 다음 업무부터 해당 에이전트의 프롬프트에 들어갑니다.",
        "원본은 `data/runs.db`, 이 파일은 `core/feedback_sync.py` 가 만듭니다 (질문·답변 원문은 올리지 않음).",
        "👍 = 유지할 점 · 🔧 = 고칠 점 · ~~취소선~~ = 화면에서 끈 교훈",
        "",
    ]
    for team in sorted(set(by_team) | set(stats)):
        ds = stats.get(team, [])
        scores = [d["score"] for d in ds if isinstance(d.get("score"), int)]
        adopted = sum(d["status"] == "adopted" for d in ds)
        summary = f"결정 {len(ds)}건 (채택 {adopted} · 반려 {len(ds) - adopted})" + (
            f" · 평균 만족도 {sum(scores) / len(scores):.1f}/5" if scores else "")
        lines += [f"## {team}", "", summary, ""]
        for agent in sorted(by_team.get(team, {})):
            lines.append(f"### {agent}")
            for item in sorted(by_team[team][agent], key=lambda x: x["created_at"], reverse=True):
                text = f"{'👍' if item['kind'] == 'praise' else '🔧'} {item['lesson']} ({item['created_at'][:10]})"
                lines.append(f"- {text}" if item["active"] else f"- ~~{text}~~")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export(storage, dest: Path) -> dict:
    """DB → knowledge/ 파일. 바뀐 파일 이름 목록과 개수를 돌려준다."""
    dest.mkdir(parents=True, exist_ok=True)
    lessons, decisions = lesson_rows(storage), decision_rows(storage)
    changed = [name for name, text in (
        (LESSONS, _dump(lessons)), (DECISIONS, _dump(decisions)), (READABLE, readable(lessons, decisions)),
    ) if _write_if_changed(dest / name, text)]
    return {"lessons": len(lessons), "decisions": len(decisions), "changed": changed}


def import_lessons(storage, src: Path) -> int:
    """knowledge/lessons.jsonl 의 교훈 중 DB 에 없는 것을 넣는다 (팀·에이전트·문장·작성 시각이 같으면 같은 교훈). 넣은 개수."""
    path = src / LESSONS
    if not path.exists():
        return 0
    have = {(r["team_id"], r["agent_id"], r["lesson"], r["created_at"]) for r in lesson_rows(storage)}  # 코멘트 뗀 문장끼리 비교
    new = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (item["team_id"], item["agent_id"], item["lesson"], item["created_at"])
        if key not in have:
            have.add(key)
            new.append(item)
    if new:
        storage.insert_lessons_raw(new)
    return len(new)


def sync(storage, folder: Path | None = None, *, pull: bool = False) -> dict | None:
    """자동 동기화 (환경변수로 켠 경우만). pull=True 면 먼저 가져온 뒤 내보낸다. 실패는 삼킨다."""
    folder = folder or knowledge_dir()
    if folder is None:
        return None
    try:
        imported = import_lessons(storage, folder) if pull else 0
        return {**export(storage, folder), "imported": imported}
    except Exception as e:  # noqa: BLE001 — 동기화 실패가 업무·결정을 막지 않게
        return {"error": f"{type(e).__name__}: {e}"}
