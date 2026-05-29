import pytest
import pandas as pd
from transdrp_multilabel.data.omics import align_omics_features, read_omics_table
from transdrp_multilabel.contracts import OmicsTable

def test_align_omics_features():
    src_df = pd.DataFrame({"gene_a": [1.0], "gene_b": [2.0]}, index=["S1"])
    tgt_df = pd.DataFrame({"gene_b": [3.0], "gene_c": [4.0]}, index=["T1"])

    src = OmicsTable(x=src_df, sample_ids=["S1"], feature_names=["gene_a", "gene_b"], domain="source")
    tgt = OmicsTable(x=tgt_df, sample_ids=["T1"], feature_names=["gene_b", "gene_c"], domain="target")

    src_al, tgt_al, report = align_omics_features(src, tgt)

    assert src_al.feature_names == ["gene_b"]
    assert tgt_al.feature_names == ["gene_b"]
    assert list(src_al.x.columns) == ["gene_b"]
    assert list(tgt_al.x.columns) == ["gene_b"]
    assert len(report) == 3

def test_align_omics_features_no_overlap():
    src_df = pd.DataFrame({"gene_a": [1.0]}, index=["S1"])
    tgt_df = pd.DataFrame({"gene_c": [4.0]}, index=["T1"])

    src = OmicsTable(x=src_df, sample_ids=["S1"], feature_names=["gene_a"], domain="source")
    tgt = OmicsTable(x=tgt_df, sample_ids=["T1"], feature_names=["gene_c"], domain="target")

    with pytest.raises(ValueError, match="No overlapping features"):
        align_omics_features(src, tgt)
