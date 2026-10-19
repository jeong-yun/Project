"""실행 이력·교훈 저장소 (SQLite 파일: data/runs.db).

테이블
  runs    : 질문 1건의 기록 (단계별 결과, 사람의 결정)
  lessons : 사람의 결정에서 만든 에이전트별 교훈 → 다음 실행 때 시스템 프롬프트에 주입

저장 정책 (사람이 피드백을 남긴 것만 보관)
  - 사람이 채택/반려한 실행(adopted / rejected)  → 보관 (+ 교훈 저장)
  - 결정 없이 버려진 실행(reviewing / awaiting_human / error) → 삭제 대상
      · 같은 화면에서 새 질문을 하면 직전 미결정 실행을 바로 삭제
      · '결과 버리기' 버튼으로 즉시 삭제
      · 앱 시작 시 오래된(settings.yaml storage.purge_undecided_after_hours) 미결정 실행 일괄 삭제
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from core.schemas import AgentAnswer, HumanDecision, Lesson

DECIDED = ("adopted", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,           -- 실행 id (= LangGraph thread_id)
    team_id     TEXT NOT NULL,
    root_id     TEXT NOT NULL,              -- 질문을 받은 에이전트(supervisor) id
    question    TEXT NOT NULL,
    steps       TEXT NOT NULL DEFAULT '[]', -- JSON list[AgentAnswer] (타임라인)
    status      TEXT NOT NULL,              -- reviewing | awaiting_human | adopted | rejected | error
    decision    TEXT,                       -- JSON HumanDecision
    created_at  TEXT NOT NULL,
    decided_at  TEXT
);
CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,              -- 조직도의 에이전트 id
    run_id      TEXT,
    kind        TEXT NOT NULL,              -- praise | improve
    lesson      TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1, -- 0 이면 프롬프트에 주입하지 않음
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_team_agent ON lessons(team_id, agent_id, active);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            self._migrate(conn)
            conn.executescript(SCHEMA)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """이전 버전(직급 일렬 구조) DB 를 새 구조로 옮긴다.

        - lessons.rank_id → agent_id 로 이름 변경 (교훈은 보존)
        - 옛 runs 테이블(형식이 다름)은 runs_legacy 로 이름만 바꿔 보존 (화면에는 표시하지 않음)
        """
        def columns(table: str) -> set[str]:
            return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

        if "rank_id" in columns("lessons"):
            conn.execute("ALTER TABLE lessons RENAME COLUMN rank_id TO agent_id")
            conn.execute("DROP INDEX IF EXISTS idx_lessons_team_rank")
        if "start_rank" in columns("runs"):
            conn.execute("ALTER TABLE runs RENAME TO runs_legacy")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # row["컬럼명"] 으로 접근
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        # closing: 연결 닫기 / 두 번째 conn: 오류 없으면 commit, 오류면 rollback
        with closing(self._connect()) as conn, conn:
            return conn.execute(sql, params).fetchall()

    # ─── runs ────────────────────────────────────────────────────

    def create_run(self, run_id: str, team_id: str, root_id: str, question: str) -> None:
        self._execute(
            "INSERT INTO runs (id, team_id, root_id, question, status, created_at) VALUES (?,?,?,?,?,?)",
            (run_id, team_id, root_id, question, "reviewing", _now()),
        )

    def save_steps(self, run_id: str, steps: list[AgentAnswer], status: str) -> None:
        payload = json.dumps([s.model_dump() for s in steps], ensure_ascii=False)
        self._execute("UPDATE runs SET steps=?, status=? WHERE id=?", (payload, status, run_id))

    def save_decision(self, run_id: str, decision: HumanDecision) -> None:
        status = "adopted" if decision.action == "adopt" else "rejected"
        self._execute(
            "UPDATE runs SET decision=?, status=?, decided_at=? WHERE id=?",
            (decision.model_dump_json(), status, _now(), run_id),
        )

    def get_run(self, run_id: str) -> dict | None:
        rows = self._execute("SELECT * FROM runs WHERE id=?", (run_id,))
        return self._run_row(rows[0]) if rows else None

    def list_runs(self, team_id: str | None = None, limit: int = 50) -> list[dict]:
        if team_id:
            rows = self._execute(
                "SELECT * FROM runs WHERE team_id=? ORDER BY created_at DESC LIMIT ?", (team_id, limit)
            )
        else:
            rows = self._execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._run_row(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        """실행 기록 삭제 (사람 피드백이 없는 불필요한 결과 정리용)."""
        self._execute("DELETE FROM runs WHERE id=?", (run_id,))

    def purge_undecided(self, older_than_hours: float) -> list[str]:
        """결정 없이 남은 오래된 실행을 삭제하고, 삭제한 id 목록을 반환 (체크포인트도 지우도록)."""
        cutoff = (datetime.now() - timedelta(hours=older_than_hours)).isoformat(timespec="seconds")
        placeholders = ",".join("?" * len(DECIDED))
        rows = self._execute(
            f"SELECT id FROM runs WHERE status NOT IN ({placeholders}) AND created_at < ?", (*DECIDED, cutoff)
        )
        ids = [r["id"] for r in rows]
        for run_id in ids:
            self.delete_run(run_id)
        return ids

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        run = dict(row)
        run["steps"] = [AgentAnswer(**s) for s in json.loads(run["steps"])]
        run["decision"] = HumanDecision.model_validate_json(run["decision"]) if run["decision"] else None
        return run

    # ─── lessons ─────────────────────────────────────────────────

    def add_lessons(self, team_id: str, run_id: str | None, lessons: list[Lesson]) -> None:
        now = _now()
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO lessons (team_id, agent_id, run_id, kind, lesson, created_at) VALUES (?,?,?,?,?,?)",
                [(team_id, l.agent_id, run_id, l.kind, l.lesson, now) for l in lessons],
            )

    def recent_lessons(self, team_id: str, agent_id: str, limit: int) -> list[str]:
        """프롬프트에 주입할 최근 활성 교훈 (오래된 것 → 최신 순)."""
        rows = self._execute(
            "SELECT lesson FROM lessons WHERE team_id=? AND agent_id=? AND active=1 ORDER BY id DESC LIMIT ?",
            (team_id, agent_id, limit),
        )
        return [r["lesson"] for r in reversed(rows)]

    def list_lessons(self, team_id: str, agent_id: str | None = None) -> list[dict]:
        if agent_id:
            rows = self._execute(
                "SELECT * FROM lessons WHERE team_id=? AND agent_id=? ORDER BY id DESC", (team_id, agent_id)
            )
        else:
            rows = self._execute("SELECT * FROM lessons WHERE team_id=? ORDER BY id DESC", (team_id,))
        return [dict(r) for r in rows]

    def lesson_counts(self, team_id: str) -> dict[str, int]:
        rows = self._execute(
            "SELECT agent_id, COUNT(*) AS n FROM lessons WHERE team_id=? AND active=1 GROUP BY agent_id", (team_id,)
        )
        return {r["agent_id"]: r["n"] for r in rows}

    def update_lesson(self, lesson_id: int, lesson: str | None = None, active: bool | None = None) -> None:
        if lesson is not None:
            self._execute("UPDATE lessons SET lesson=? WHERE id=?", (lesson, lesson_id))
        if active is not None:
            self._execute("UPDATE lessons SET active=? WHERE id=?", (int(active), lesson_id))

    def delete_lesson(self, lesson_id: int) -> None:
        self._execute("DELETE FROM lessons WHERE id=?", (lesson_id,))
