"""SQLite metadata store for DistilAgent runs.

Schema:
    runs:       one row per dataset-build invocation (status, config, counts)
    events:     structured event log (for SSE replay on reconnect)
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from threading import Lock

_LOCK = Lock()


def get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id              TEXT PRIMARY KEY,
                created_at      REAL NOT NULL,
                started_at      REAL,
                finished_at     REAL,
                status          TEXT NOT NULL,      -- queued | running | paused | done | failed | cancelled
                config_json     TEXT NOT NULL,
                output_path     TEXT,
                checkpoint_path TEXT,
                log_path        TEXT,
                pid             INTEGER,
                completed       INTEGER DEFAULT 0,
                failed          INTEGER DEFAULT 0,
                total_planned   INTEGER DEFAULT 0,
                error           TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id    TEXT NOT NULL,
                ts        REAL NOT NULL,
                kind      TEXT NOT NULL,       -- sample | heartbeat | error | plan | done | info
                payload   TEXT NOT NULL,        -- JSON blob
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id)"
        )


def new_run(conn: sqlite3.Connection, config: Dict[str, Any]) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _LOCK, conn:
        conn.execute(
            "INSERT INTO runs(id, created_at, status, config_json) VALUES (?,?,?,?)",
            (run_id, time.time(), "queued", json.dumps(config)),
        )
    return run_id


def update_run(conn: sqlite3.Connection, run_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [run_id]
    with _LOCK, conn:
        conn.execute(f"UPDATE runs SET {cols} WHERE id=?", vals)


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _row_to_dict(row)


def list_runs(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def append_event(conn: sqlite3.Connection, run_id: str, kind: str, payload: Dict[str, Any]) -> int:
    with _LOCK, conn:
        cur = conn.execute(
            "INSERT INTO events(run_id, ts, kind, payload) VALUES (?,?,?,?)",
            (run_id, time.time(), kind, json.dumps(payload)),
        )
        return cur.lastrowid


def events_since(
    conn: sqlite3.Connection, run_id: str, last_id: int = 0, limit: int = 500
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, ts, kind, payload FROM events "
        "WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
        (run_id, last_id, limit),
    ).fetchall()
    return [
        {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload"])}
        for r in rows
    ]


def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    if d.get("config_json"):
        d["config"] = json.loads(d.pop("config_json"))
    return d
