"""Saved jobs and preferences in SQLite, so they survive between sessions."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS saved_jobs (
    job_id   TEXT PRIMARY KEY,   -- PRIMARY KEY: the same job can't be saved twice
    title    TEXT NOT NULL,
    company  TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS preferences (
    key        TEXT PRIMARY KEY, -- one value per key; saving again replaces it
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def save_job(self, job_id: str, title: str, company: str) -> bool:
        """Save a job; False if it was already saved."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO saved_jobs (job_id, title, company, saved_at) "
            "VALUES (?, ?, ?, ?)", (job_id, title, company, _now()))
        self.conn.commit()
        return cur.rowcount == 1

    def list_saved(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT job_id, title, company FROM saved_jobs ORDER BY saved_at, job_id")
        return [{"job_id": r[0], "title": r[1], "company": r[2]} for r in rows]

    def set_preference(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at", (key, value, _now()))
        self.conn.commit()

    def get_preferences(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM preferences ORDER BY key")
        return dict(rows.fetchall())

    def clear(self) -> None:
        """Delete all saved jobs and preferences."""
        self.conn.execute("DELETE FROM saved_jobs")
        self.conn.execute("DELETE FROM preferences")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def MemoryStore() -> Store:
    """A throw-away in-memory store."""
    return Store(":memory:")
