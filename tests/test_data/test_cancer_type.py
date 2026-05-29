import pytest
import pandas as pd
from transdrp_multilabel.data.cancer_type import load_and_align_cancer_types

def test_load_and_align_cancer_types(tmp_path):
    # Setup files
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

    table = load_and_align_cancer_types(
        ["S1", "S2", "S3"],
        ["T1", "T2"],
        str(src_path),
        str(tgt_path),
        "Cancer_type"
    )

    assert len(table) == 5
    assert set(table["sample_id"]) == {"S1", "S2", "S3", "T1", "T2"}

    # S3 is missing in src_df, should default to Unknown
    s3_row = table[table["sample_id"] == "S3"]
    assert s3_row.iloc[0]["cancer_type"] == "Unknown"

    s1_row = table[table["sample_id"] == "S1"]
    assert s1_row.iloc[0]["cancer_type"] == "COAD"
