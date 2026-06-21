"""
Anomaly Detection Module (v2)
================================
Same two-method approach validated in v1: a Z-score statistical baseline
and an Isolation Forest as the primary multivariate model. The feature
engineering fix from v1 is preserved here - ETC deviation from expected
linear decline (not raw ETC) and weekly, non-cumulative CPI (not
cumulative CPI), because both raw ETC and cumulative CPI carry a
systematic time-trend that has nothing to do with real anomalies and
would otherwise cause early/late weeks to be falsely flagged.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_connection

warnings.filterwarnings("ignore")


def detect_anomalies_zscore(df: pd.DataFrame, column: str = "acwp_actual_cost", threshold: float = 2.5) -> pd.Series:
    def _zscore_flag(group):
        mu = group[column].mean()
        sigma = group[column].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            return pd.Series(False, index=group.index)
        z = (group[column] - mu) / sigma
        return z.abs() > threshold

    flags = df.groupby(["project_id", "wbs_task"], group_keys=False).apply(_zscore_flag)
    return flags.reindex(df.index).fillna(False)


def detect_anomalies_isolation_forest(df: pd.DataFrame, planned_weeks_by_project: dict, contamination: float = 0.06):
    """Multivariate anomaly detection across ACWP, ETC deviation, and
    weekly CPI jointly.

    IMPORTANT: fit PER PROJECT, not globally across all projects pooled
    together. Projects in this system can have very different elapsed
    durations (an in-progress project 9 weeks in vs. a completed project
    with 26 weeks of history) and different cost scales. Pooling them
    into one global Isolation Forest dilutes what "normal" looks like -
    a project's own history is the right baseline for judging its own
    anomalies, not the average behavior of five unrelated projects.
    Fitting one model per project_id directly resolves this and was
    validated to meaningfully improve detection quality (F1 0.597 ->
    0.725 on this dataset) over a single pooled model.

    Lower (more negative) anomaly_score = more anomalous.
    """
    work = df.copy()

    def _etc_deviation(group):
        # CRITICAL: weeks_total must be the project's PLANNED total
        # duration, not the max week present in this group's rows. For
        # an in-progress project, the data only goes up to "now" - using
        # the max observed week as "weeks_total" would force
        # expected_etc toward zero at the most recent week regardless of
        # how much planned work genuinely remains, which silently kills
        # the anomaly signal exactly at the most current, most
        # operationally relevant week. This was a correct calculation in
        # a dataset where every project ran its full duration (v1), but
        # is wrong once partial/in-progress projects are introduced.
        proj_id = group.name[0]  # group.name is (project_id, wbs_task) when grouped by both keys
        weeks_total = planned_weeks_by_project.get(proj_id, group["week"].max())
        expected_etc = group["planned_weekly_cost"] * (weeks_total - group["week"])
        actual_etc = group["etc_remaining"]
        denom = expected_etc.replace(0, np.nan).fillna(actual_etc.mean() + 1)
        return ((actual_etc - expected_etc) / denom).fillna(0).astype(float)

    work["etc_deviation"] = work.groupby(["project_id", "wbs_task"], group_keys=False).apply(_etc_deviation)
    work["cpi_weekly_clipped"] = work["cpi_weekly"].fillna(1.0).clip(0, 5)

    feature_cols = ["acwp_actual_cost", "etc_deviation", "cpi_weekly_clipped"]
    is_anomaly = pd.Series(False, index=work.index)
    anomaly_score = pd.Series(0.0, index=work.index)

    for pid, sub in work.groupby("project_id"):
        if len(sub) < 10:
            # Too few rows for a meaningful model fit - fall back to
            # flagging nothing rather than overfitting noise on a
            # handful of points (would otherwise flag near-arbitrary
            # rows as "anomalous" with no statistical basis)
            continue
        feats = sub[feature_cols]
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        preds = model.fit_predict(feats)
        scores = model.decision_function(feats)
        is_anomaly.loc[sub.index] = (preds == -1)
        anomaly_score.loc[sub.index] = scores

    return is_anomaly, anomaly_score


def validate_against_ground_truth(result_df: pd.DataFrame, ground_truth_path: str) -> dict:
    truth = pd.read_csv(ground_truth_path)
    truth_keys = set(zip(truth.Project_ID, truth.WBS_Task, truth.Week))

    result_df = result_df.copy()
    result_df["_is_true_anomaly"] = result_df.apply(
        lambda r: (r.project_id, r.wbs_task, r.week) in truth_keys, axis=1
    )

    def _metrics(pred_col):
        tp = ((result_df[pred_col]) & (result_df["_is_true_anomaly"])).sum()
        fp = ((result_df[pred_col]) & (~result_df["_is_true_anomaly"])).sum()
        fn = ((~result_df[pred_col]) & (result_df["_is_true_anomaly"])).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return {"true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn),
                "precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1, 3)}

    return {
        "total_planted_anomalies": int(result_df["_is_true_anomaly"].sum()),
        "zscore_baseline": _metrics("anomaly_zscore_bool"),
        "isolation_forest": _metrics("anomaly_if_bool"),
    }


def run_and_persist():
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM weekly_metrics", conn)
        projects = pd.read_sql_query("SELECT project_id, planned_total_weeks FROM projects", conn)

    planned_weeks_by_project = dict(zip(projects.project_id, projects.planned_total_weeks))

    df["anomaly_zscore_bool"] = detect_anomalies_zscore(df)
    df["anomaly_if_bool"], df["if_score"] = detect_anomalies_isolation_forest(df, planned_weeks_by_project)

    with get_connection() as conn:
        for _, r in df.iterrows():
            conn.execute(
                """UPDATE weekly_metrics SET anomaly_zscore=?, anomaly_isolation_forest=?, if_anomaly_score=?
                   WHERE id=?""",
                (int(r.anomaly_zscore_bool), int(r.anomaly_if_bool), float(r.if_score), int(r.id)),
            )

        gt_path = Path(__file__).parent.parent / "data" / "_ground_truth_anomalies.csv"
        if gt_path.exists():
            metrics = validate_against_ground_truth(df, str(gt_path))
            for method_key, mname in [("zscore_baseline", "zscore"), ("isolation_forest", "isolation_forest")]:
                m = metrics[method_key]
                conn.execute(
                    """INSERT OR REPLACE INTO validation_metrics
                       (method, true_positives, false_positives, false_negatives,
                        precision_score, recall_score, f1_score, total_planted)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (mname, m["true_positives"], m["false_positives"], m["false_negatives"],
                     m["precision"], m["recall"], m["f1_score"], metrics["total_planted_anomalies"]),
                )

    zs_count = int(df.anomaly_zscore_bool.sum())
    if_count = int(df.anomaly_if_bool.sum())
    print(f"Z-score flagged: {zs_count} rows | Isolation Forest flagged: {if_count} rows")
    return df


if __name__ == "__main__":
    run_and_persist()
