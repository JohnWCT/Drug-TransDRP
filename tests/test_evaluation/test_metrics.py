import pytest
import pandas as pd
from transdrp_multilabel.evaluation.metrics import compute_metrics_from_predictions

def test_compute_metrics_classification():
    pred_df = pd.DataFrame({
        "drug_id": ["druga", "druga", "drugb", "drugb"],
        "ground_truth": [1, 0, 1, 0],
        "pred_score": [1.0, 0.0, 0.0, 1.0], # logits
        "pred_label": [1, 0, 0, 1],
        "domain": ["source"] * 4
    })

    per, summ = compute_metrics_from_predictions(pred_df, "classification", "source")
    assert len(per) == 2
    assert "auroc" in per.columns
    assert len(summ) > 0

def test_compute_metrics_regression():
    pred_df = pd.DataFrame({
        "drug_id": ["druga", "druga", "drugb", "drugb"],
        "ground_truth": [1.0, 2.0, 3.0, 4.0],
        "pred_score": [1.0, 2.1, 2.9, 4.0],
        "domain": ["source"] * 4
    })

    per, summ = compute_metrics_from_predictions(pred_df, "regression", "source")
    assert len(per) == 2
    assert "mae" in per.columns
    assert per.loc[per["drug_id"] == "druga", "mae"].values[0] == pytest.approx(0.05)
