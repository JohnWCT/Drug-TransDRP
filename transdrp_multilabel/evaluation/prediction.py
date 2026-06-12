from __future__ import annotations

from typing import Literal, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transdrp_multilabel.contracts import DrugIndex


def predict_matrix(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            _, yp, _ = model(batch_x, 0, node_x.to(device), edge_index.to(device))
            preds.append(yp.cpu().numpy())
    return np.vstack(preds) if preds else np.zeros((0, 0), dtype=np.float32)


def build_prediction_long_table(
    scores: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    sample_ids: list[str],
    drug_index: DrugIndex,
    domain: Literal["source", "target"],
    split: str,
    task_type: str,
    prediction_threshold: float,
    regression_binary_threshold: float,
    fold: int = 0,
    seed: int = 0,
    cancer_type_table: pd.DataFrame = None,
    eval_drug_indices: Optional[List[int]] = None,
    eval_dataset: str | None = None,
    source_drug_ids: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    cancer_map = {}
    if cancer_type_table is not None:
        sub = cancer_type_table[cancer_type_table["domain"] == domain]
        cancer_map = dict(zip(sub["sample_id"].astype(str), sub["cancer_type"].astype(str)))

    drug_indices = eval_drug_indices if eval_drug_indices is not None else list(range(len(drug_index.drug_ids)))
    src_set = source_drug_ids or set()

    for i, sid in enumerate(sample_ids):
        for j in drug_indices:
            if mask[i, j] < 0.5:
                continue
            gt = float(y[i, j])
            score = float(scores[i, j])
            drug_id = drug_index.index_to_drug[j]
            has_sup = drug_id in src_set if src_set else (domain == "source")
            is_tgt_only = (not has_sup) and domain == "target"

            row = {
                "sample_id": sid,
                "drug_id": drug_id,
                "drug_index": j,
                "domain": domain,
                "split": split,
                "eval_dataset": eval_dataset or "",
                "fold": fold,
                "seed": seed,
                "task_type": task_type,
                "ground_truth": gt,
                "mask": 1,
                "pred_score": score,
                "cancer_type": cancer_map.get(sid, ""),
                "has_supervised_source_label": has_sup,
                "is_target_eval_only": is_tgt_only,
            }

            if task_type == "classification":
                prob = 1.0 / (1.0 + np.exp(-score))
                row["probability"] = float(prob)
                row["pred_label"] = int(prob >= prediction_threshold)
                row["confidence"] = float(max(prob, 1.0 - prob))
            elif task_type == "regression" and domain == "target":
                row["pred_label"] = int(score > regression_binary_threshold)
            elif task_type == "regression" and domain == "source":
                row["ground_truth_binary"] = int(gt > regression_binary_threshold)
                row["pred_label"] = int(score > regression_binary_threshold)
            rows.append(row)

    return pd.DataFrame(rows)
