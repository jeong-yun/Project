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
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,              -- 호출이 끝난 시각 (epoch 초)
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    key_id        TEXT NOT NULL,              -- API 키 지문 (sha256 앞 12자리, 키 없음 = '')
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    calls         INTEGER NOT NULL DEFAULT 1,
    cost_usd      REAL,                       -- 기록 당시 가격표로 계산한 추정 비용 (가격 미등록 = NULL)
    team_id       TEXT NOT NULL DEFAULT '',
    run_id        TEXT NOT NULL DEFAULT '',
    purpose       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,             -- 키 원문은 저장하지 않는다
    provider    TEXT NOT NULL,
    masked      TEXT NOT NULL,                -- 화면 표시용 '앞 3자…뒤 4자'
    label       TEXT NOT NULL DEFAULT '',     -- 사람이 붙인 이름 (예: 회사 OpenAI 키)
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS price_overrides (
    model         TEXT PRIMARY KEY,           -- 화면 '⚙️ 가격표'에서 고친 가격 (config/pricing.yaml 보다 우선)
    input         REAL NOT NULL,
    output        REAL NOT NULL,
    cached_input  REAL
);
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
        # 동시 실행(형제 에이전트)·사용량 기록이 한꺼번에 쓰므로 잠금을 넉넉히 기다린다
        conn = sqlite3.connect(self.db_path, timeout=30)
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

    # ─── 사용량·비용 (core/usage.py 가 호출마다 record_usage 로 넘긴다) ───────────

    def record_usage(self, event) -> None:
        """호출 1건 기록 + 처음 본 API 키면 키 목록에 추가 (마지막 사용 시각 갱신)."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO usage (ts, provider, model, key_id, input_tokens, output_tokens, cached_tokens, calls, "
                "cost_usd, team_id, run_id, purpose) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (event.ts, event.provider, event.model, event.key_id, event.input_tokens, event.output_tokens,
                 event.cached_tokens, event.calls, event.cost_usd, event.team_id, event.run_id, event.purpose),
            )
            conn.execute(
                "INSERT INTO api_keys (key_id, provider, masked, first_seen, last_seen) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key_id) DO UPDATE SET last_seen=excluded.last_seen",
                (event.key_id, event.provider, event.key_masked, event.ts, event.ts),
            )

    def usage_summary(self, since: float | None = None, key_id: str | None = None) -> list[dict]:
        """(키 · provider · 모델)별 합계 — 호출 수·토큰·추정 비용. 비용이 NULL(가격 미등록)인 행은 priced_calls 로 구분."""
        where, params = [], []
        if since is not None:
            where.append("ts >= ?")
            params.append(since)
        if key_id is not None:
            where.append("key_id = ?")
            params.append(key_id)
        sql = (
            "SELECT key_id, provider, model, SUM(calls) AS calls, SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens, SUM(cached_tokens) AS cached_tokens, SUM(cost_usd) AS cost_usd, "
            "SUM(CASE WHEN cost_usd IS NULL THEN calls ELSE 0 END) AS unpriced_calls, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM usage" + (" WHERE " + " AND ".join(where) if where else "")
            + " GROUP BY key_id, provider, model ORDER BY cost_usd DESC, calls DESC"
        )
        return [dict(r) for r in self._execute(sql, tuple(params))]

    def usage_daily(self, since: float | None = None) -> list[dict]:
        """날짜별 추정 비용·토큰 (현지 시각 기준)."""
        sql = ("SELECT date(ts, 'unixepoch', 'localtime') AS day, SUM(cost_usd) AS cost_usd, "
               "SUM(input_tokens + output_tokens) AS tokens, SUM(calls) AS calls FROM usage"
               + (" WHERE ts >= ?" if since is not None else "") + " GROUP BY day ORDER BY day")
        return [dict(r) for r in self._execute(sql, (since,) if since is not None else ())]

    def api_keys(self) -> list[dict]:
        return [dict(r) for r in self._execute("SELECT * FROM api_keys ORDER BY last_seen DESC")]

    def set_key_label(self, key_id: str, label: str) -> None:
        self._execute("UPDATE api_keys SET label=? WHERE key_id=?", (label.strip(), key_id))

    def remember_key(self, key_id: str, provider: str, masked: str) -> None:
        """지금 .env 에 있는 키를 (아직 호출 전이어도) 목록에 올려 이름을 미리 붙일 수 있게 한다."""
        now = __import__("time").time()
        self._execute(
            "INSERT INTO api_keys (key_id, provider, masked, first_seen, last_seen) VALUES (?,?,?,?,?) "
            "ON CONFLICT(key_id) DO NOTHING",
            (key_id, provider, masked, now, now),
        )

    def price_overrides(self) -> dict[str, dict]:
        rows = self._execute("SELECT * FROM price_overrides")
        return {r["model"]: {k: r[k] for k in ("input", "output", "cached_input") if r[k] is not None} for r in rows}

    def set_price_override(self, model: str, input_price: float, output_price: float, cached_input: float | None = None) -> None:
        self._execute(
            "INSERT INTO price_overrides (model, input, output, cached_input) VALUES (?,?,?,?) "
            "ON CONFLICT(model) DO UPDATE SET input=excluded.input, output=excluded.output, cached_input=excluded.cached_input",
            (model, input_price, output_price, cached_input),
        )

    def clear_price_override(self, model: str) -> None:
        self._execute("DELETE FROM price_overrides WHERE model=?", (model,))

    def reprice_unpriced(self, estimate) -> int:
        """가격이 없어 비용이 비어 있던 기록을 지금 가격표로 다시 계산 (가격을 새로 넣은 뒤). 고친 행 수."""
        rows = self._execute("SELECT id, ts, provider, model, input_tokens, output_tokens, cached_tokens, calls "
                             "FROM usage WHERE cost_usd IS NULL")
        fixed = 0
        with closing(self._connect()) as conn, conn:
            for r in rows:
                day = __import__("time").strftime("%Y-%m-%d", __import__("time").localtime(r["ts"]))
                cost = estimate(r["provider"], r["model"], r["input_tokens"], r["output_tokens"], r["cached_tokens"],
                                r["calls"], day=day)
                if cost is not None:
                    conn.execute("UPDATE usage SET cost_usd=? WHERE id=?", (cost, r["id"]))
                    fixed += 1
        return fixed
