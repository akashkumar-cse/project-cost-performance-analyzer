"""
FastAPI Backend
==================
Single process that does two things:

  1. Serves a REST API over the SQLite database populated by the
     pipeline (projects, weekly detail, anomalies, forecasts,
     validation metrics) - this is what makes the dashboard genuinely
     live: re-run the pipeline after changing source data, and the API
     returns fresh numbers on the next request, no manual re-embedding
     step required.

  2. Serves the dashboard's static frontend files and proxies natural-
     language questions to the Gemini API using a key read from a
     server-side .env file - the browser never sees this key, unlike
     the v1 approach where the user had to paste their own key into
     the page. This is the actually-secure pattern for "store it in
     the backend."

Run with:  python app.py        (from inside backend/)
Then open: http://127.0.0.1:8000
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from database import get_connection, DB_PATH

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Project Cost Performance Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _row_to_dict(row):
    return {k: row[k] for k in row.keys()}


def _project_status_label(cpi: Optional[float]) -> str:
    if cpi is None:
        return "watch"
    if cpi >= 1.0:
        return "good"
    if cpi >= 0.95:
        return "watch"
    return "risk"


# ─────────────────────────────────────────────────────────────────────
# API: Projects
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    """All projects with their latest computed CPI, cost variance, and
    (for in-progress projects) forecast summary - this is the main
    payload the dashboard's project cards are built from."""
    with get_connection() as conn:
        projects = conn.execute("SELECT * FROM projects ORDER BY status, project_id").fetchall()
        results = []
        for p in projects:
            p = _row_to_dict(p)
            latest = conn.execute(
                """SELECT week, SUM(acwp_cumulative) as acwp, SUM(bcwp_earned_value) as bcwp,
                          SUM(eac) as eac, SUM(vac) as vac
                   FROM weekly_metrics WHERE project_id=? AND week=(SELECT MAX(week) FROM weekly_metrics WHERE project_id=?)
                   GROUP BY week""",
                (p["project_id"], p["project_id"]),
            ).fetchone()

            anomaly_count = conn.execute(
                "SELECT COUNT(*) as c FROM weekly_metrics WHERE project_id=? AND (anomaly_zscore=1 OR anomaly_isolation_forest=1)",
                (p["project_id"],),
            ).fetchone()["c"]

            current_cpi = round(latest["bcwp"] / latest["acwp"], 3) if latest and latest["acwp"] else None

            forecast = None
            if p["status"] == "in_progress":
                f = conn.execute("SELECT * FROM forecasts WHERE project_id=?", (p["project_id"],)).fetchone()
                if f:
                    f = _row_to_dict(f)
                    forecast = {
                        "method": f["method"],
                        "risk_flag": f["risk_flag"],
                        "avg_forecast_cpi": f["avg_forecast_cpi"],
                        "forecast_finish_week": f["forecast_finish_week"],
                        "planned_finish_week": p["planned_total_weeks"],
                        "forecast_final_cost": f["forecast_final_cost"],
                        "planned_total_budget": p["total_budget_bac"],
                        "forecast_overrun": round(f["forecast_final_cost"] - p["total_budget_bac"], 2),
                        "weeks_elapsed": latest["week"] if latest else 0,
                    }

            results.append({
                "project_id": p["project_id"],
                "project_name": p["project_name"],
                "status": p["status"],
                "current_cpi": current_cpi,
                "status_label": _project_status_label(current_cpi),
                "cost_variance": round(latest["bcwp"] - latest["acwp"], 2) if latest else None,
                "vac": round(latest["vac"], 2) if latest else None,
                "weeks_elapsed": latest["week"] if latest else 0,
                "planned_total_weeks": p["planned_total_weeks"],
                "total_budget_bac": p["total_budget_bac"],
                "anomaly_count": anomaly_count,
                "forecast": forecast,
            })
        return results


@app.get("/api/projects/{project_id}/trend")
def get_project_trend(project_id: str):
    """Historical CPI series + forecast (if in-progress) for the trend chart."""
    with get_connection() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if not proj:
            raise HTTPException(404, "Project not found")
        proj = _row_to_dict(proj)

        weekly = conn.execute(
            """SELECT week, SUM(acwp_cumulative) as acwp, SUM(bcwp_earned_value) as bcwp
               FROM weekly_metrics WHERE project_id=? GROUP BY week ORDER BY week""",
            (project_id,),
        ).fetchall()
        historical_weeks = [w["week"] for w in weekly]
        historical_cpi = [round(w["bcwp"] / w["acwp"], 3) if w["acwp"] else None for w in weekly]

        forecast_data = {}
        if proj["status"] == "in_progress":
            f = conn.execute("SELECT * FROM forecasts WHERE project_id=?", (project_id,)).fetchone()
            if f:
                f = _row_to_dict(f)
                forecast_data = {
                    "method": f["method"],
                    "forecast_weeks": json.loads(f["forecast_weeks_json"]),
                    "forecast_cpi": json.loads(f["forecast_cpi_json"]),
                    "forecast_lower": json.loads(f["forecast_lower_json"]),
                    "forecast_upper": json.loads(f["forecast_upper_json"]),
                    "forecast_finish_week": f["forecast_finish_week"],
                    "forecast_final_cost": f["forecast_final_cost"],
                }

        anomalous_weeks = conn.execute(
            """SELECT DISTINCT week FROM weekly_metrics
               WHERE project_id=? AND (anomaly_zscore=1 OR anomaly_isolation_forest=1) ORDER BY week""",
            (project_id,),
        ).fetchall()

        return {
            "project_id": project_id,
            "project_name": proj["project_name"],
            "status": proj["status"],
            "planned_total_weeks": proj["planned_total_weeks"],
            "historical_weeks": historical_weeks,
            "historical_cpi": historical_cpi,
            "anomalous_weeks": [w["week"] for w in anomalous_weeks],
            **forecast_data,
        }


# ─────────────────────────────────────────────────────────────────────
# API: Weekly detail table
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/weekly-detail")
def get_weekly_detail(project_id: Optional[str] = None, anomalies_only: bool = False):
    with get_connection() as conn:
        query = """SELECT wm.*, p.project_name FROM weekly_metrics wm
                   JOIN projects p ON wm.project_id = p.project_id WHERE 1=1"""
        params = []
        if project_id:
            query += " AND wm.project_id=?"
            params.append(project_id)
        if anomalies_only:
            query += " AND (wm.anomaly_zscore=1 OR wm.anomaly_isolation_forest=1)"
        query += " ORDER BY wm.project_id, wm.wbs_task, wm.week"
        rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# API: Model validation metrics
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/validation-metrics")
def get_validation_metrics():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM validation_metrics").fetchall()
        return {r["method"]: _row_to_dict(r) for r in rows}


# ─────────────────────────────────────────────────────────────────────
# API: Gemini chat (server-side key, never exposed to the browser)
# ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list = []


def _build_context() -> str:
    """Builds the grounding context sent to Gemini - pipeline OUTPUT
    only (project summaries, anomaly counts, forecasts), never the raw
    source Excel files. Same architectural boundary as v1: ML does the
    analysis, the LLM only narrates already-computed results."""
    projects = get_projects()
    metrics = get_validation_metrics()

    lines = ["You are a project cost performance assistant. Answer using ONLY the data below, "
             "which comes from a validated ML pipeline (Isolation Forest for anomaly detection, "
             "ARIMA for forecasting). Be concise and specific with numbers. "
             "Do not invent figures not present here.\n"]

    for p in projects:
        lines.append(f"\n{p['project_id']} ({p['project_name']}) - status: {p['status']}")
        lines.append(f"  Week {p['weeks_elapsed']} of {p['planned_total_weeks']} planned")
        lines.append(f"  Current CPI: {p['current_cpi']} | Cost Variance: {p['cost_variance']} | VAC: {p['vac']}")
        lines.append(f"  Budget (BAC): {p['total_budget_bac']:.0f} | Anomalies flagged: {p['anomaly_count']}")
        if p["forecast"]:
            f = p["forecast"]
            lines.append(f"  Forecast: {f['risk_flag']}, projected finish week {f['forecast_finish_week']} "
                         f"(planned {f['planned_finish_week']}), projected final cost {f['forecast_final_cost']:.0f} "
                         f"(overrun {f['forecast_overrun']:.0f})")

    lines.append(f"\nModel validation (Isolation Forest vs Z-score baseline, against known test anomalies):")
    for method, m in metrics.items():
        lines.append(f"  {method}: precision={m['precision_score']}, recall={m['recall_score']}, f1={m['f1_score']}")

    return "\n".join(lines)


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            500,
            "GEMINI_API_KEY is not configured on the server. Add it to backend/.env "
            "(see .env.example) and restart the server."
        )

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=GEMINI_API_KEY)
    system_prompt = _build_context()

    contents = []
    for turn in req.history[-8:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=turn.get("content", ""))]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=req.message)]))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(system_instruction=system_prompt, max_output_tokens=1000),
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(502, f"Gemini API error: {str(e)}")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database": DB_PATH.exists(),
        "gemini_configured": bool(GEMINI_API_KEY),
    }


# ─────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"Gemini configured: {bool(GEMINI_API_KEY)}")
    print(f"Database: {DB_PATH}")
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
