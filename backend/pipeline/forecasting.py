"""
Cost Trend Forecasting Module (v2)
=====================================
Extends v1's ARIMA-based CPI forecasting with two new, more directly
actionable outputs: a forecasted COMPLETION WEEK and a forecasted FINAL
COST. The whole point of forecasting cost performance is to answer two
questions a project engineer actually needs before a project finishes -
"when will this be done" and "how much will it actually cost" - not
just "is the trend going up or down."

Only IN-PROGRESS projects get a forward forecast. Completed projects
have no future to forecast - they get a retrospective accuracy check
instead (see backend/app.py for how the two are presented differently).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_connection

warnings.filterwarnings("ignore")


def _fit_cpi_forecast(series: pd.Series, forecast_weeks: int):
    """Fits ARIMA(1,1,1) on a weekly CPI series, falls back to linear
    trend on short/flat series where ARIMA can't converge meaningfully."""
    last_week = series.index.max()
    forecast_index = list(range(last_week + 1, last_week + 1 + forecast_weeks))

    if len(series) >= 5:
        try:
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            forecast_result = fitted.get_forecast(steps=forecast_weeks)
            forecast_values = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int(alpha=0.2)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
            method = "ARIMA(1,1,1)"
            forecast_values = np.clip(forecast_values, 0.1, 3.0)
            return forecast_index, forecast_values, lower, upper, method
        except Exception:
            pass

    # Fallback: linear trend extrapolation
    x = series.index.values
    y = series.values
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0, y[-1] if len(y) else 1.0
    forecast_values = slope * np.array(forecast_index) + intercept
    forecast_values = np.clip(forecast_values, 0.1, 3.0)
    spread = series.std() if len(series) > 1 and series.std() > 0 else 0.05
    lower = forecast_values - spread
    upper = forecast_values + spread
    return forecast_index, forecast_values, lower, upper, "Linear trend (fallback)"


def forecast_project(project_row: dict, weekly_df: pd.DataFrame, forecast_weeks: int = 8) -> dict:
    """Builds the full forecast payload for one project: CPI trend,
    forecasted completion week, and forecasted final cost.

    The completion-week and final-cost estimates are derived from the
    project's CURRENT cost performance trajectory, not just its most
    recent single data point - they answer "if this project continues
    performing the way it has been trending, when does it actually
    finish and what does it actually cost," which is the operationally
    useful question, vs. just reporting a CPI number in isolation.
    """
    proj_id = project_row["project_id"]
    proj_weekly = weekly_df[weekly_df.project_id == proj_id]

    weekly_agg = (
        proj_weekly.groupby("week")
        .apply(lambda g: g.acwp_cumulative.sum() and (g.bcwp_earned_value.sum() / g.acwp_cumulative.sum()))
        .rename("cpi")
        .reset_index()
        .sort_values("week")
    )
    series = weekly_agg.set_index("week")["cpi"].astype(float)

    if series.empty:
        return None

    forecast_weeks_idx, forecast_cpi, lower, upper, method = _fit_cpi_forecast(series, forecast_weeks)

    avg_forecast_cpi = float(np.mean(forecast_cpi))
    risk_flag = (
        "AT RISK" if avg_forecast_cpi < 0.95 else
        "WATCH" if avg_forecast_cpi < 1.0 else
        "ON TRACK"
    )

    # --- Forecasted completion week & final cost ---
    # Current ACWP (actual spend to date) and remaining work, scaled by
    # the forecasted CPI trend rather than assuming the original plan
    # still holds - if a project is running at CPI 0.85, it will burn
    # through its remaining budget faster than planned, which should
    # push the completion week out and the final cost up.
    latest_week_num = proj_weekly.week.max()
    latest_week_data = proj_weekly[proj_weekly.week == latest_week_num]
    current_acwp = float(latest_week_data.acwp_cumulative.sum())

    # IMPORTANT: don't blindly trust the single latest week's ETC. A
    # week flagged as anomalous (e.g. a planted/real ETC spike from
    # scope re-estimation) can sit exactly on the most recent week,
    # which would make the forecast inherit that spike as if it were
    # the genuine current estimate. Instead, take a per-task ETC reading
    # from the most recent NON-anomalous week available (falling back to
    # the literal latest week only if every recent week is flagged,
    # which would itself be worth surfacing rather than silently
    # excluding everything).
    def _robust_etc_for_task(task_df: pd.DataFrame) -> float:
        clean = task_df[(task_df.anomaly_zscore == 0) & (task_df.anomaly_isolation_forest == 0)]
        source = clean if not clean.empty else task_df
        return float(source.sort_values("week").iloc[-1].etc_remaining)

    current_etc = float(
        proj_weekly.groupby("wbs_task").apply(_robust_etc_for_task).sum()
    )

    planned_total_weeks = int(project_row["planned_total_weeks"])
    weeks_elapsed = int(proj_weekly.week.max())
    total_budget = float(project_row["total_budget_bac"])

    # If CPI is below 1.0, the team is burning more actual cost per unit
    # of planned work than budgeted - remaining work will take longer
    # AND cost more, scaled inversely by the forecasted CPI.
    cpi_efficiency = max(avg_forecast_cpi, 0.3)  # guard against extreme/unstable forecasts
    weeks_remaining_planned = planned_total_weeks - weeks_elapsed
    forecast_weeks_remaining = weeks_remaining_planned / cpi_efficiency
    forecast_finish_week = int(round(weeks_elapsed + forecast_weeks_remaining))

    forecast_final_cost = current_acwp + (current_etc / cpi_efficiency)

    return {
        "project_id": proj_id,
        "method": method,
        "historical_weeks": series.index.tolist(),
        "historical_cpi": [round(v, 3) for v in series.values],
        "forecast_weeks": forecast_weeks_idx,
        "forecast_cpi": [round(v, 3) for v in forecast_cpi],
        "forecast_lower": [round(v, 3) for v in lower],
        "forecast_upper": [round(v, 3) for v in upper],
        "avg_forecast_cpi": round(avg_forecast_cpi, 3),
        "risk_flag": risk_flag,
        "forecast_finish_week": forecast_finish_week,
        "planned_finish_week": planned_total_weeks,
        "weeks_elapsed": weeks_elapsed,
        "forecast_final_cost": round(forecast_final_cost, 2),
        "planned_total_budget": round(total_budget, 2),
        "forecast_overrun": round(forecast_final_cost - total_budget, 2),
    }


def run_and_persist(forecast_weeks: int = 8):
    import json

    with get_connection() as conn:
        projects = pd.read_sql_query("SELECT * FROM projects WHERE status='in_progress'", conn)
        weekly_df = pd.read_sql_query("SELECT * FROM weekly_metrics", conn)

    results = []
    for _, prow in projects.iterrows():
        f = forecast_project(prow.to_dict(), weekly_df, forecast_weeks)
        if f:
            results.append(f)

    with get_connection() as conn:
        for f in results:
            conn.execute(
                """INSERT OR REPLACE INTO forecasts
                   (project_id, method, avg_forecast_cpi, risk_flag,
                    forecast_finish_week, forecast_final_cost,
                    forecast_weeks_json, forecast_cpi_json, forecast_lower_json,
                    forecast_upper_json, historical_weeks_json, historical_cpi_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f["project_id"], f["method"], f["avg_forecast_cpi"], f["risk_flag"],
                 f["forecast_finish_week"], f["forecast_final_cost"],
                 json.dumps(f["forecast_weeks"]), json.dumps(f["forecast_cpi"]),
                 json.dumps(f["forecast_lower"]), json.dumps(f["forecast_upper"]),
                 json.dumps(f["historical_weeks"]), json.dumps(f["historical_cpi"])),
            )

    for f in results:
        print(f"{f['project_id']}: {f['risk_flag']} | forecast finish wk {f['forecast_finish_week']} "
              f"(planned {f['planned_finish_week']}) | forecast cost ${f['forecast_final_cost']:,.0f} "
              f"(budget ${f['planned_total_budget']:,.0f})")
    return results


if __name__ == "__main__":
    run_and_persist()
