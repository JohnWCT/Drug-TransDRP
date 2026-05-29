from typing import Literal
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
    edge_index: torch.Tensor
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            # original model returns (domain_output, class_output, feature)
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
) -> pd.DataFrame:
    rows = []
    cancer_map = {}
    if cancer_type_table is not None:
        sub = cancer_type_table[cancer_type_table["domain"] == domain]
        cancer_map = dict(zip(sub["sample_id"].astype(str), sub["cancer_type"].astype(str)))

    for i, sid in enumerate(sample_ids):
        for j, _drug_id in enumerate(drug_index.drug_ids):
            if mask[i, j] < 0.5:
                continue
            gt = float(y[i, j])
            score = float(scores[i, j])

            row = {
                "sample_id": sid,
                "drug_id": drug_index.index_to_drug[j],
                "drug_index": j,
                "domain": domain,
                "split": split,
                "fold": fold,
                "seed": seed,
                "task_type": task_type,
                "ground_truth": gt,
                "mask": 1,
                "pred_score": score,
                "cancer_type": cancer_map.get(sid, "Unknown"),
            }

            # Binary probability mapping for classification or target evaluation
            if task_type == "classification" or (task_type == "regression" and domain == "target"):
                prob = 1.0 / (1.0 + np.exp(-score))
                row["probability"] = float(prob)
                row["pred_label"] = int(prob >= prediction_threshold)
            elif task_type == "regression" and domain == "source":
                # For regression source responder, lower score is sensitive (< threshold)
                row["ground_truth_binary"] = int(gt < regression_binary_threshold)
                row["pred_label"] = int(score < regression_binary_threshold)
            rows.append(row)

    return pd.DataFrame(rows)
