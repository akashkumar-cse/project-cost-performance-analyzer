"""
Synthetic Project Cost Data Generator (v2)
=============================================
Same EVM simulation approach as v1, with one important addition: each
project now has a STATUS — 'in_progress' or 'completed'. This matters
because the whole point of cost performance forecasting is to act
*before* a project finishes, not after. In-progress projects get a
forward-looking forecast (completion week, final cost). Completed
projects are kept for retrospective analysis only — comparing what
actually happened against what was predicted along the way.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# project_id -> (name, status, total_planned_weeks, weeks_elapsed_so_far)
# in_progress projects: weeks_elapsed < total_planned_weeks (we only have
#   data up to "today" - the rest is genuinely unknown, which is what the
#   forecast is for)
# completed projects: weeks_elapsed == total_planned_weeks (full history)
PROJECTS = {
    "PRJ-101": {"name": "Avionics Display Upgrade",       "status": "in_progress", "total_weeks": 32, "weeks_elapsed": 22},
    "PRJ-102": {"name": "Cabin Pressure Sensor Suite",    "status": "in_progress", "total_weeks": 28, "weeks_elapsed": 18},
    "PRJ-103": {"name": "Flight Control Software v4",     "status": "in_progress", "total_weeks": 36, "weeks_elapsed": 14},
    "PRJ-104": {"name": "Ground Ops Telemetry Link",      "status": "in_progress", "total_weeks": 24, "weeks_elapsed": 9},
    "PRJ-201": {"name": "Cockpit HMI Redesign (closed)",  "status": "completed",   "total_weeks": 26, "weeks_elapsed": 26},
    "PRJ-202": {"name": "Hydraulic Test Bench Automation","status": "completed",   "total_weeks": 20, "weeks_elapsed": 20},
}

WBS_TASKS = ["Design", "Integration", "Testing", "Certification", "Documentation"]
WEEKLY_BUDGET_RANGE = (8000, 35000)
NOISE_STD_PCT = 0.08

ANOMALY_LOG = []


def maybe_plant_anomaly(proj, task, week, base_actual, base_etc):
    r = np.random.random()
    if r < 0.02:
        factor = np.random.uniform(2.2, 3.5)
        ANOMALY_LOG.append((proj, task, week, "cost_spike"))
        return base_actual * factor, base_etc
    elif r < 0.04:
        factor = np.random.uniform(2.0, 4.0)
        ANOMALY_LOG.append((proj, task, week, "etc_blowup"))
        return base_actual, base_etc * factor
    elif r < 0.06:
        ANOMALY_LOG.append((proj, task, week, "underreport"))
        return base_actual * np.random.uniform(0.02, 0.1), base_etc
    return base_actual, base_etc


# ---------------------------------------------------------------------
# 1. Budget baseline file
# ---------------------------------------------------------------------
budget_rows = []
for proj, meta in PROJECTS.items():
    for task in WBS_TASKS:
        weekly_budget = np.random.uniform(*WEEKLY_BUDGET_RANGE)
        total_budget = weekly_budget * meta["total_weeks"]
        budget_rows.append({
            "Project_ID": proj,
            "Project_Name": meta["name"],
            "Status": meta["status"],
            "Planned_Total_Weeks": meta["total_weeks"],
            "WBS_Task": task,
            "Planned_Weekly_Cost": round(weekly_budget, 2),
            "Total_Budget_BAC": round(total_budget, 2),
        })
budget_df = pd.DataFrame(budget_rows)
budget_df.to_excel(DATA_DIR / "project_budget_baseline.xlsx", index=False)

# ---------------------------------------------------------------------
# 2. Weekly actuals - only up to weeks_elapsed for each project (we
#    cannot have actuals for weeks that haven't happened yet - that's
#    exactly what the forecast is supposed to predict)
# ---------------------------------------------------------------------
all_weekly_rows = []
for proj, meta in PROJECTS.items():
    proj_budget = budget_df[budget_df.Project_ID == proj]
    project_drift = np.random.uniform(0.85, 1.15)

    for _, brow in proj_budget.iterrows():
        task = brow.WBS_Task
        weekly_planned = brow.Planned_Weekly_Cost
        task_drift = project_drift * np.random.uniform(0.97, 1.03)
        weeks_elapsed = meta["weeks_elapsed"]
        total_weeks = meta["total_weeks"]

        for week in range(1, weeks_elapsed + 1):
            noise = np.random.normal(1.0, NOISE_STD_PCT)
            base_actual = weekly_planned * task_drift * noise
            weeks_left = total_weeks - week
            base_etc = max(weekly_planned * weeks_left * np.random.uniform(0.95, 1.1), 0)

            actual, etc = maybe_plant_anomaly(proj, task, week, base_actual, base_etc)

            all_weekly_rows.append({
                "Project_ID": proj,
                "WBS_Task": task,
                "Week": week,
                "Planned_Weekly_Cost": round(weekly_planned, 2),
                "ACWP_Actual_Cost": round(actual, 2),
                "ETC_Remaining": round(etc, 2),
            })

weekly_df = pd.DataFrame(all_weekly_rows)

# Split into 4-week-block files (periodic export simulation)
weekly_df["Block"] = ((weekly_df["Week"] - 1) // 4) + 1
for block_num, block_df in weekly_df.groupby("Block"):
    fname = DATA_DIR / f"weekly_actuals_block_{int(block_num):02d}.xlsx"
    block_df.drop(columns=["Block"]).to_excel(fname, index=False)

anomaly_log_df = pd.DataFrame(ANOMALY_LOG, columns=["Project_ID", "WBS_Task", "Week", "Anomaly_Type"])
anomaly_log_df.to_csv(DATA_DIR / "_ground_truth_anomalies.csv", index=False)

print(f"Projects: {len(PROJECTS)} ({sum(1 for m in PROJECTS.values() if m['status']=='in_progress')} in progress, "
      f"{sum(1 for m in PROJECTS.values() if m['status']=='completed')} completed)")
print(f"Budget baseline file: 1 file, {len(budget_df)} rows")
print(f"Weekly actuals files: {weekly_df['Block'].nunique()} files, {len(weekly_df)} total rows")
print(f"Planted anomalies: {len(anomaly_log_df)} ({len(anomaly_log_df)/len(weekly_df)*100:.1f}% of rows)")
