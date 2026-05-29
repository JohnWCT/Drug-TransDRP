import pytest
import pandas as pd
from transdrp_multilabel.data.drug_index import build_drug_index_from_union

def test_build_drug_index_from_union():
    src_df = pd.DataFrame({"drug_name": [" DrugA ", "DrugB"]})
    tgt_df = pd.DataFrame({"drug_name": ["DrugB", "drugc"]})

    idx = build_drug_index_from_union(src_df, tgt_df, "drug_name")

    # normalized to lowercase
    assert idx.drug_ids == ["druga", "drugb", "drugc"]
    assert idx.drug_to_index["druga"] == 0
    assert idx.index_to_drug[2] == "drugc"
    assert idx.n_drugs == 3
