import pytest
from transdrp_multilabel.data.sample_id import (
    sample_match_key,
    normalize_source_sample_id,
    normalize_target_omics_sample_id,
    normalize_target_response_sample_id,
    tcga_patient_key,
    tcga_segment_count,
)

def test_tcga_segment_count():
    assert tcga_segment_count("TCGA-AA-0001") == 3
    assert tcga_segment_count("TCGA-AA-0001-01A") == 4
    assert tcga_segment_count("CCLE_sample") == 1

def test_tcga_patient_key():
    assert tcga_patient_key("TCGA-AA-0001-01A") == "TCGA-AA-0001"
    assert tcga_patient_key("non-TCGA-sample") == "non-TCGA-sample"

def test_normalize_sample_ids():
    assert normalize_source_sample_id(" CCLE_1 ") == "CCLE_1"
    assert normalize_target_omics_sample_id(" TCGA-AA-0001-01A ") == "TCGA-AA-0001-01A"
    assert normalize_target_response_sample_id(" TCGA-AA-0001-01A ") == "TCGA-AA-0001"

def test_sample_match_key():
    assert sample_match_key("TCGA-AA-0001-01A") == "TCGA-AA-0001"
    assert sample_match_key("ACH-0001") == "ACH-0001"
