import pytest
import pandas as pd
import numpy as np
from transdrp_multilabel.data.drug_index import build_drug_index_from_union
from transdrp_multilabel.data.response_matrix import long_to_response_matrix

def test_long_to_response_matrix():
    src_df = pd.DataFrame({
        "Sample_ID": ["S1", "S1", "S2"],
        "drug_name": ["DrugA", "DrugB", "DrugB"],
        "Label": [1.0, 0.0, 1.0]
    })

    idx = build_drug_index_from_union(src_df, src_df, "drug_name")
    sample_ids = ["S1", "S2"]

    matrix = long_to_response_matrix(
        src_df,
        sample_ids,
        idx,
        "Sample_ID",
        "drug_name",
        "Label",
        "source",
        "binary",
        "mean"
    )

    assert matrix.y.shape == (2, 2)
    assert matrix.mask.shape == (2, 2)
    # S1 DrugA observed, S1 DrugB observed
    assert np.allclose(matrix.mask[0], [1.0, 1.0])
    # S2 DrugA missing, S2 DrugB observed
    assert np.allclose(matrix.mask[1], [0.0, 1.0])
    assert matrix.y[0, 0] == 1.0
    assert matrix.y[0, 1] == 0.0
    assert matrix.y[1, 1] == 1.0
