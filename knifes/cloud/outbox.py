from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class EventOutbox:
    """Persist small event records; snapshots stay on disk until upload."""

    def __init__(self, path: Path | None, scope: str) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.execute("PRAGMA busy_timeout = 5000")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS pending_events (
                scope TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, due_at REAL NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                PRIMARY KEY (scope, event_id)
            )"""
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS pending_due ON pending_events (scope, failed, due_at)"
        )
        self._db.commit()

    def put(self, payload: dict[str, Any]) -> None:
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Evento sem event_id.")
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO pending_events (scope, event_id, payload) VALUES (?, ?, ?)",
                (self.scope, event_id, json.dumps(payload, ensure_ascii=False)),
            )

    def next_due(self) -> tuple[dict[str, Any], int] | None:
        with self._lock:
            row = self._db.execute(
                """SELECT payload, attempts FROM pending_events
                   WHERE scope = ? AND failed = 0 AND due_at <= ?
                   ORDER BY due_at, rowid LIMIT 1""",
                (self.scope, time.time()),
            ).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def acknowledge(self, event_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM pending_events WHERE scope = ? AND event_id = ?",
                (self.scope, event_id),
            )

    def reject(self, event_id: str, attempts: int, error: str, retryable: bool) -> None:
        delay = min(60.0, 2.0 ** min(attempts + 1, 6))
        with self._lock, self._db:
            self._db.execute(
                """UPDATE pending_events SET attempts = ?, due_at = ?, failed = ?, last_error = ?
                   WHERE scope = ? AND event_id = ?""",
                (attempts + 1, time.time() + delay, int(not retryable), error[:500], self.scope, event_id),
            )

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT failed, COUNT(*) FROM pending_events WHERE scope = ? GROUP BY failed",
                (self.scope,),
            ).fetchall()
        counts = dict(rows)
        return {"pending_events": counts.get(0, 0), "failed_events": counts.get(1, 0)}

    def close(self) -> None:
        with self._lock:
            self._db.close()
