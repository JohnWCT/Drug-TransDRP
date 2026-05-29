import pytest
import numpy as np
import pandas as pd
from transdrp_multilabel.data.drug_index import DrugIndex
from transdrp_multilabel.evaluation.prediction import build_prediction_long_table

def test_build_prediction_long_table():
    scores = np.array([[10.0, -10.0], [0.0, 5.0]])
    y = np.array([[1.0, 0.0], [1.0, 1.0]])
    mask = np.array([[1.0, 1.0], [1.0, 1.0]])

    idx = DrugIndex(
        drug_ids=["druga", "drugb"],
        drug_to_index={"druga": 0, "drugb": 1},
        index_to_drug={0: "druga", 1: "drugb"}
    )

    df = build_prediction_long_table(
        scores,
        y,
        mask,
        ["S1", "S2"],
        idx,
        "source",
        "test",
        "classification",
        prediction_threshold=0.5,
        regression_binary_threshold=1.0
    )

    assert len(df) == 4
    assert list(df["pred_label"]) == [1, 0, 1, 1]
    assert df.loc[0, "sample_id"] == "S1"
