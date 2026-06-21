"""
Data Consolidation Module (v2)
=================================
Same EVM calculation logic as v1 (BCWS, BCWP proxy, ACWP, CV, CPI, EAC,
VAC), but now writes results into SQLite instead of an in-memory
DataFrame destined for a JSON export. This is what makes the system
genuinely "live" - the API reads these tables on every request, so
re-running this script after changing the source files actually changes
what the dashboard shows.
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_connection, init_db

DATA_DIR = Path(__file__).parent.parent / "data"


def load_budget_baseline() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "project_budget_baseline.xlsx")


def load_all_weekly_actuals() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / "weekly_actuals_block_*.xlsx")))
    if not files:
        raise FileNotFoundError("No weekly actuals files found. Run seed_data.py first.")
    frames = [pd.read_excel(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["Project_ID", "WBS_Task", "Week"], inplace=True)
    return combined.reset_index(drop=True)


def compute_evm_metrics(weekly_df: pd.DataFrame, budget_df: pd.DataFrame) -> pd.DataFrame:
    df = weekly_df.merge(
        budget_df[["Project_ID", "Project_Name", "Status", "Planned_Total_Weeks", "WBS_Task", "Total_Budget_BAC"]],
        on=["Project_ID", "WBS_Task"], how="left",
    )

    df["BCWS_Planned_Cumulative"] = df["Planned_Weekly_Cost"] * df["Week"]
    df["ACWP_Cumulative"] = df.groupby(["Project_ID", "WBS_Task"])["ACWP_Actual_Cost"].cumsum()
    df["BCWP_Earned_Value"] = df["BCWS_Planned_Cumulative"]  # planned-schedule proxy, see README

    df["Cost_Variance_CV"] = df["BCWP_Earned_Value"] - df["ACWP_Cumulative"]
    df["CPI"] = (df["BCWP_Earned_Value"] / df["ACWP_Cumulative"].replace(0, np.nan)).round(3)
    df["CPI_Weekly"] = (df["Planned_Weekly_Cost"] / df["ACWP_Actual_Cost"].replace(0, np.nan)).round(3)

    df["EAC_Estimate_At_Completion"] = df["ACWP_Cumulative"] + df["ETC_Remaining"]
    df["VAC_Variance_At_Completion"] = df["Total_Budget_BAC"] - df["EAC_Estimate_At_Completion"]

    return df


def get_consolidated_dataset() -> pd.DataFrame:
    budget_df = load_budget_baseline()
    weekly_df = load_all_weekly_actuals()
    return compute_evm_metrics(weekly_df, budget_df)


def write_to_database(df: pd.DataFrame, budget_df: pd.DataFrame):
    init_db(fresh=True)
    with get_connection() as conn:
        # projects table
        for _, row in budget_df.drop_duplicates("Project_ID").iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO projects
                   (project_id, project_name, status, total_budget_bac, planned_total_weeks)
                   VALUES (?,?,?,?,?)""",
                (row.Project_ID, row.Project_Name, row.Status,
                 float(budget_df[budget_df.Project_ID == row.Project_ID].Total_Budget_BAC.sum()),
                 int(row.Planned_Total_Weeks)),
            )

        # weekly_metrics table
        for _, r in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO weekly_metrics
                   (project_id, wbs_task, week, planned_weekly_cost, acwp_actual_cost,
                    etc_remaining, acwp_cumulative, bcwp_earned_value, cost_variance,
                    cpi, cpi_weekly, eac, vac)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r.Project_ID, r.WBS_Task, int(r.Week), float(r.Planned_Weekly_Cost),
                 float(r.ACWP_Actual_Cost), float(r.ETC_Remaining), float(r.ACWP_Cumulative),
                 float(r.BCWP_Earned_Value), float(r.Cost_Variance_CV),
                 None if pd.isna(r.CPI) else float(r.CPI),
                 None if pd.isna(r.CPI_Weekly) else float(r.CPI_Weekly),
                 float(r.EAC_Estimate_At_Completion), float(r.VAC_Variance_At_Completion)),
            )


if __name__ == "__main__":
    budget_df = load_budget_baseline()
    df = get_consolidated_dataset()
    write_to_database(df, budget_df)
    print(f"Wrote {len(budget_df.Project_ID.unique())} projects, {len(df)} weekly_metrics rows to database.")
