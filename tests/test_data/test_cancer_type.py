import pytest
import pandas as pd
from transdrp_multilabel.data.cancer_type import load_and_align_cancer_types

def _write_maps(tmp_path):
    src_df = pd.DataFrame({
        "Sample_ID": ["S1", "S2"],
        "Cancer_type": ["COAD", "READ"]
    })
    tgt_df = pd.DataFrame({
        "Patient_id": ["T1", "T2"],
        "Cancer_type": ["COAD", "BRCA"]
    })
    src_path = tmp_path / "src_ct.csv"
    tgt_path = tmp_path / "tgt_ct.csv"
    src_df.to_csv(src_path, index=False)
    tgt_df.to_csv(tgt_path, index=False)
    return str(src_path), str(tgt_path)

def test_load_and_align_cancer_types_complete(tmp_path):
    src_path, tgt_path = _write_maps(tmp_path)
    table = load_and_align_cancer_types(
        ["S1", "S2"],
        ["T1", "T2"],
        src_path,
        tgt_path,
        "Cancer_type",
    )
    assert len(table) == 4
    assert set(table["sample_id"]) == {"S1", "S2", "T1", "T2"}
    assert table[table["sample_id"] == "S1"].iloc[0]["cancer_type"] == "COAD"

def test_missing_sample_raises(tmp_path):
    # req.10: every sample must have a cancer type; S3 is unmapped -> hard error.
    src_path, tgt_path = _write_maps(tmp_path)
    with pytest.raises(ValueError):
        load_and_align_cancer_types(["S1", "S2", "S3"], ["T1", "T2"], src_path, tgt_path, "Cancer_type")

def test_missing_path_raises(tmp_path):
    src_path, tgt_path = _write_maps(tmp_path)
    with pytest.raises(ValueError):
        load_and_align_cancer_types(["S1"], ["T1"], None, tgt_path, "Cancer_type")
