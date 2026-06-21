"""
Database Layer
================
SQLite schema for the Project Cost Performance Analyzer. This replaces
the v1 approach of baking computed results into a static JSON blob
embedded in HTML — here, the pipeline writes results into real tables,
and the API reads live from the database on every request. Changing the
source Excel files and re-running the pipeline genuinely changes what
the API returns next time, with no manual re-embedding step.

Tables:
  projects          - one row per project, including status (in_progress / completed)
  weekly_metrics     - one row per project/task/week with all computed EVM + anomaly fields
  forecasts          - one row per project: ARIMA-based forecast incl. completion week/cost
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "cost_analyzer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id          TEXT PRIMARY KEY,
    project_name         TEXT NOT NULL,
    status               TEXT NOT NULL CHECK(status IN ('in_progress','completed')),
    total_budget_bac     REAL NOT NULL,
    planned_total_weeks  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS weekly_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id              TEXT NOT NULL REFERENCES projects(project_id),
    wbs_task                TEXT NOT NULL,
    week                    INTEGER NOT NULL,
    planned_weekly_cost     REAL NOT NULL,
    acwp_actual_cost        REAL NOT NULL,
    etc_remaining           REAL NOT NULL,
    acwp_cumulative         REAL NOT NULL,
    bcwp_earned_value       REAL NOT NULL,
    cost_variance           REAL NOT NULL,
    cpi                     REAL,
    cpi_weekly               REAL,
    eac                     REAL NOT NULL,
    vac                     REAL NOT NULL,
    anomaly_zscore           INTEGER NOT NULL DEFAULT 0,
    anomaly_isolation_forest INTEGER NOT NULL DEFAULT 0,
    if_anomaly_score          REAL,
    UNIQUE(project_id, wbs_task, week)
);

CREATE TABLE IF NOT EXISTS forecasts (
    project_id              TEXT PRIMARY KEY REFERENCES projects(project_id),
    method                   TEXT NOT NULL,
    avg_forecast_cpi          REAL NOT NULL,
    risk_flag                TEXT NOT NULL,
    forecast_finish_week      INTEGER,
    forecast_final_cost       REAL,
    forecast_weeks_json       TEXT,
    forecast_cpi_json          TEXT,
    forecast_lower_json        TEXT,
    forecast_upper_json        TEXT,
    historical_weeks_json      TEXT,
    historical_cpi_json        TEXT
);

CREATE TABLE IF NOT EXISTS validation_metrics (
    method            TEXT PRIMARY KEY,
    true_positives    INTEGER,
    false_positives   INTEGER,
    false_negatives   INTEGER,
    precision_score   REAL,
    recall_score      REAL,
    f1_score          REAL,
    total_planted     INTEGER
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(fresh: bool = False):
    """Creates the schema. If fresh=True, wipes existing data first
    (used when re-seeding from scratch)."""
    if fresh and DB_PATH.exists():
        DB_PATH.unlink()
    with get_connection() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db(fresh=True)
    print(f"Database initialized at {DB_PATH}")
