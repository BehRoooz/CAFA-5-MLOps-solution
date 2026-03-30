from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
                """
            )
            self.conn.commit()

    def create_job(self, job_id: str, request_json: dict[str, Any]) -> None:
        progress = {"embedded_sequences": 0, "total_sequences": 0, "percent": 0.0}
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO jobs (job_id, status, request_json, progress_json, error_json, created_at, started_at, finished_at)
                VALUES (?, 'queued', ?, ?, NULL, ?, NULL, NULL)
                """,
                (job_id, json.dumps(request_json), json.dumps(progress), utc_now()),
            )
            self.conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def get_next_queued_job_id(self) -> str | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        return row["job_id"] if row else None

    def mark_running(self, job_id: str) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (utc_now(), job_id),
            )
            self.conn.commit()

    def update_progress(self, job_id: str, embedded: int, total: int) -> None:
        percent = 0.0 if total == 0 else (100.0 * embedded / total)
        progress = {
            "embedded_sequences": int(embedded),
            "total_sequences": int(total),
            "percent": round(percent, 2),
        }
        with self.lock:
            self.conn.execute(
                "UPDATE jobs SET progress_json = ? WHERE job_id = ?",
                (json.dumps(progress), job_id),
            )
            self.conn.commit()

    def mark_succeeded(self, job_id: str) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE jobs SET status = 'succeeded', finished_at = ?, error_json = NULL WHERE job_id = ?",
                (utc_now(), job_id),
            )
            self.conn.commit()

    def mark_failed(self, job_id: str, error_json: dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE jobs SET status = 'failed', error_json = ?, finished_at = ? WHERE job_id = ?",
                (json.dumps(error_json), utc_now(), job_id),
            )
            self.conn.commit()

    def insert_artifact(
        self,
        job_id: str,
        name: str,
        path: str,
        dtype: str,
        shape: list[int],
        size_bytes: int,
    ) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO artifacts (job_id, name, path, dtype, shape_json, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, name, path, dtype, json.dumps(shape), size_bytes),
            )
            self.conn.commit()

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT job_id, name, path, dtype, shape_json, size_bytes FROM artifacts WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        return [
            {
                "name": row["name"],
                "path": row["path"],
                "dtype": row["dtype"],
                "shape": json.loads(row["shape_json"]),
                "size_bytes": row["size_bytes"],
            }
            for row in rows
        ]

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "request": json.loads(row["request_json"]),
            "progress": json.loads(row["progress_json"]),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }