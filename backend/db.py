"""SQLite log of batch prediction runs (optional catalog feature).

Only aggregate statistics are stored - never the uploaded beats - so the
history table contains no patient-level data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_runs (
    batch_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    model_used      TEXT NOT NULL,
    source_name     TEXT,
    n_beats         INTEGER NOT NULL,
    mean_risk       REAL NOT NULL,
    n_high_risk     INTEGER NOT NULL,
    n_ventricular   INTEGER NOT NULL,
    abnormal_pct    REAL NOT NULL,
    class_counts    TEXT NOT NULL,
    latency_ms      REAL NOT NULL
);
"""


class PredictionLog:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def log_batch(self, batch_id: str, model_used: str, summary: dict,
                  latency_ms: float, source_name: str | None = None) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO batch_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (batch_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 model_used, source_name, summary["n_beats"], summary["mean_risk"],
                 summary["n_high_risk"], summary["n_ventricular_alerts"],
                 summary["abnormal_burden_pct"], json.dumps(summary["class_counts"]),
                 latency_ms),
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM batch_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["class_counts"] = json.loads(d["class_counts"])
            out.append(d)
        return out
