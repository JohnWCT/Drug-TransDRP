"""Cross-fold metric aggregation and reporting utilities."""

import pandas as pd
import numpy as np

def _metric_value_columns(df: pd.DataFrame) -> list[str]:
    skip = {"drug_id", "n", "fold", "samples_used", "n_cancer_types", "k_eff"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

def aggregate_per_drug_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std of per-drug metrics across folds."""
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    if "drug_id" not in combined.columns:
        return pd.DataFrame()
    metric_cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for drug_id, grp in combined.groupby("drug_id"):
        row: dict[str, object] = {"drug_id": drug_id, "n_folds": int(grp["fold"].nunique()) if "fold" in grp else len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_scalar_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std for single-row metric tables (e.g. kmeans, latent)."""
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for col in cols:
        vals = combined[col].dropna()
        rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=0)) if len(vals) > 1 else float("nan"),
                "n_folds": len(vals),
            }
        )
    return pd.DataFrame(rows)

def aggregate_summary_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std of summary metrics (macro / weighted / overall) across folds."""
    if not fold_frames:
        return pd.DataFrame()

    pivoted_frames = []
    for f_idx, df in enumerate(fold_frames):
        if df.empty:
            continue
        if "metric" not in df.columns or "aggregation" not in df.columns or "metric_value" not in df.columns:
            continue
        pivoted = df.pivot(index="metric", columns="aggregation", values="metric_value").reset_index()
        pivoted["fold"] = f_idx
        pivoted_frames.append(pivoted)

    if not pivoted_frames:
        return pd.DataFrame()

    combined = pd.concat(pivoted_frames, ignore_index=True)

    rows: list[dict[str, object]] = []
    for metric_name, grp in combined.groupby("metric"):
        row: dict[str, object] = {"metric": metric_name, "n_folds": int(grp["fold"].nunique())}
        for col in ("macro", "weighted", "overall"):
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

def build_combined_eval_summary(
    src_fold_frames: list[pd.DataFrame],
    tgt_fold_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Integrate source_test and target_eval fold-mean/std summaries."""
    frames: list[pd.DataFrame] = []
    if src_fold_frames:
        frames.append(
            aggregate_summary_metrics(src_fold_frames).assign(domain="source_test")
        )
    if tgt_fold_frames:
        frames.append(
            aggregate_summary_metrics(tgt_fold_frames).assign(domain="target_eval")
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    front = ["domain", "metric", "n_folds"]
    rest = [c for c in combined.columns if c not in front]
    return combined[front + rest]


def aggregate_target_eval_metrics_by_dataset(
    fold_frames: list[pd.DataFrame],
    dataset_col: str = "eval_dataset",
) -> pd.DataFrame:
    """Aggregate summary metrics grouped by eval_dataset across folds."""
    if not fold_frames:
        return pd.DataFrame()

    pivoted_frames = []
    for f_idx, df in enumerate(fold_frames):
        if df.empty:
            continue
        if "metric" not in df.columns or "aggregation" not in df.columns or "metric_value" not in df.columns:
            continue
        pivoted = df.pivot(index="metric", columns="aggregation", values="metric_value").reset_index()
        pivoted["fold"] = f_idx
        if dataset_col in df.columns:
            pivoted[dataset_col] = df[dataset_col].iloc[0]
        pivoted_frames.append(pivoted)

    if not pivoted_frames:
        return pd.DataFrame()

    combined = pd.concat(pivoted_frames, ignore_index=True)
    group_cols = [dataset_col, "metric"] if dataset_col in combined.columns else ["metric"]

    rows: list[dict[str, object]] = []
    for keys, grp in combined.groupby(group_cols, dropna=False):
        if isinstance(keys, tuple):
            eval_ds, metric_name = keys
        else:
            eval_ds, metric_name = "", keys
        row: dict[str, object] = {
            "eval_dataset": eval_ds,
            "metric": metric_name,
            "n_folds": int(grp["fold"].nunique()) if "fold" in grp.columns else len(grp),
        }
        for col in ("macro", "weighted", "overall"):
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_drug_metrics_by_dataset(
    fold_frames: list[pd.DataFrame],
    dataset_col: str = "eval_dataset",
) -> pd.DataFrame:
    """Mean / std of per-drug metrics grouped by eval_dataset across folds."""
    if not fold_frames:
        return pd.DataFrame()

    combined = pd.concat(fold_frames, ignore_index=True)
    if "drug_id" not in combined.columns or dataset_col not in combined.columns:
        return pd.DataFrame()

    metric_cols = _metric_value_columns(combined)
    extra_cols = [c for c in ("has_supervised_source_label", "is_target_eval_only") if c in combined.columns]
    rows: list[dict[str, object]] = []

    for (eval_ds, drug_id), grp in combined.groupby([dataset_col, "drug_id"], dropna=False):
        row: dict[str, object] = {
            "eval_dataset": eval_ds,
            "drug_id": drug_id,
            "n_folds": int(grp["fold"].nunique()) if "fold" in grp.columns else len(grp),
        }
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        for col in extra_cols:
            row[col] = grp[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)
