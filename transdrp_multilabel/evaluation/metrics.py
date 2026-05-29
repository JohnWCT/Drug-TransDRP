import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = (
    "auroc",
    "aupr",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
)
REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson", "spearman")

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")

def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")

def _classification_scores(g: pd.DataFrame) -> np.ndarray:
    if "probability" in g.columns and g["probability"].notna().any():
        return g["probability"].fillna(g["pred_score"]).astype(float).to_numpy()
    return g["pred_score"].astype(float).to_numpy()

def compute_classification_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug_id, g in pred_df.groupby("drug_id"):
        y = g["ground_truth"].astype(int).to_numpy()
        y_score = _classification_scores(g)
        pred = g["pred_label"].astype(int).to_numpy()
        rows.append(
            {
                "drug_id": drug_id,
                "n_observed": len(g),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "auroc": _safe_auc(y, y_score),
                "aupr": _safe_aupr(y, y_score),
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            }
        )
    return pd.DataFrame(rows)

def compute_classification_metrics_overall(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {m: float("nan") for m in CLASSIFICATION_METRICS}
    y = pred_df["ground_truth"].astype(int).to_numpy()
    y_score = _classification_scores(pred_df)
    pred = pred_df["pred_label"].astype(int).to_numpy()
    return {
        "auroc": _safe_auc(y, y_score),
        "aupr": _safe_aupr(y, y_score),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }

def compute_classification_metrics_summary(per_drug: pd.DataFrame, pred_df: pd.DataFrame = None) -> pd.DataFrame:
    overall = compute_classification_metrics_overall(pred_df) if pred_df is not None else {}
    rows = []
    for m in CLASSIFICATION_METRICS:
        vals = per_drug[m].dropna() if m in per_drug.columns else pd.Series(dtype=float)
        macro_val = float(vals.mean()) if len(vals) else float("nan")

        w = per_drug["n_observed"].values if "n_observed" in per_drug.columns else np.array([])
        v = per_drug[m].values if m in per_drug.columns else np.array([])
        mask = ~np.isnan(v.astype(float))
        weighted = float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")

        for agg, val in (
            ("macro", macro_val),
            ("micro", overall.get(m, float("nan"))),
            ("weighted", weighted),
            ("overall", overall.get(m, float("nan"))),
        ):
            rows.append(
                {
                    "metric_name": f"{agg}_{m}",
                    "metric": m,
                    "aggregation": agg,
                    "metric_value": val,
                    "n_valid_drugs": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)

def compute_regression_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug_id, g in pred_df.groupby("drug_id"):
        y = g["ground_truth"].astype(float).to_numpy()
        p = g["pred_score"].astype(float).to_numpy()
        mae = float(mean_absolute_error(y, p))
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_std = float(np.std(y.astype(np.float64)))
            pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
            sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")

        rows.append(
            {
                "drug_id": drug_id,
                "n_observed": len(g),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson": float(pr) if pr == pr else float("nan"),
                "spearman": float(sp) if sp == sp else float("nan"),
            }
        )
    return pd.DataFrame(rows)

def compute_regression_metrics_overall(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {m: float("nan") for m in REGRESSION_METRICS}
    y = pred_df["ground_truth"].astype(float).to_numpy()
    p = pred_df["pred_score"].astype(float).to_numpy()
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_std = float(np.std(y.astype(np.float64)))
        pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
        sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": float(pr) if pr == pr else float("nan"),
        "spearman": float(sp) if sp == sp else float("nan"),
    }

def compute_regression_metrics_summary(per_drug: pd.DataFrame, pred_df: pd.DataFrame = None) -> pd.DataFrame:
    overall = compute_regression_metrics_overall(pred_df) if pred_df is not None else {}
    rows = []
    for m in REGRESSION_METRICS:
        vals = per_drug[m].dropna() if m in per_drug.columns else pd.Series(dtype=float)
        macro_val = float(vals.mean()) if len(vals) else float("nan")

        w = per_drug["n_observed"].values if "n_observed" in per_drug.columns else np.array([])
        v = per_drug[m].values if m in per_drug.columns else np.array([])
        mask = ~np.isnan(v.astype(float))
        weighted = float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")

        for agg, val in (
            ("macro", macro_val),
            ("micro", overall.get(m, float("nan"))),
            ("weighted", weighted),
            ("overall", overall.get(m, float("nan"))),
        ):
            rows.append(
                {
                    "metric_name": f"{agg}_{m}",
                    "metric": m,
                    "aggregation": agg,
                    "metric_value": val,
                    "n_valid_drugs": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)

def compute_metrics_from_predictions(
    pred_df: pd.DataFrame,
    task_type: str,
    domain: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred_df.empty or "drug_id" not in pred_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    sub = pred_df[pred_df["domain"] == domain] if "domain" in pred_df.columns else pred_df
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    if task_type == "regression":
        per = compute_regression_metrics_per_drug(sub)
        summ = compute_regression_metrics_summary(per, sub)
        return per, summ

    per = compute_classification_metrics_per_drug(sub)
    summ = compute_classification_metrics_summary(per, sub)
    return per, summ

def compute_prediction_bundle(
    val_pred: pd.DataFrame, val_true: pd.DataFrame, val_mask: pd.DataFrame,
    tgt_pred: pd.DataFrame, tgt_true: pd.DataFrame, tgt_mask: pd.DataFrame,
    task_type: str
):
    """Auxiliary bundle metric computation for validation monitoring."""
    from transdrp_multilabel.contracts import PredictionBundle, DrugIndex

    # Mock drug index for indexing columns
    n_drugs = val_pred.shape[1]
    drug_ids = [f"drug_{j}" for j in range(n_drugs)]
    mock_idx = DrugIndex(drug_ids=drug_ids, drug_to_index={d: j for j, d in enumerate(drug_ids)}, index_to_drug={j: d for j, d in enumerate(drug_ids)})

    # Create prediction tables
    from transdrp_multilabel.evaluation.prediction import build_prediction_long_table
    val_long = build_prediction_long_table(
        val_pred.values, val_true.values, val_mask.values,
        [f"val_{i}" for i in range(len(val_pred))], mock_idx,
        "source", "val", task_type, 0.5, 0.0
    )
    tgt_long = build_prediction_long_table(
        tgt_pred.values, tgt_true.values, tgt_mask.values,
        [f"tgt_{i}" for i in range(len(tgt_pred))], mock_idx,
        "target", "test", task_type, 0.5, 0.0
    )

    val_per, val_sum = compute_metrics_from_predictions(val_long, task_type, "source")
    tgt_per, tgt_sum = compute_metrics_from_predictions(tgt_long, task_type, "target")

    return PredictionBundle(
        source_predictions=val_long,
        target_predictions=tgt_long,
        source_metrics_per_drug=val_per,
        target_metrics_per_drug=tgt_per,
        source_metrics_summary=val_sum,
        target_metrics_summary=tgt_sum
    )
