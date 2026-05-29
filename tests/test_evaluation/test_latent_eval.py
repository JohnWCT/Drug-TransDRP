import pytest
import numpy as np
from transdrp_multilabel.evaluation.latent_eval import (
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    calculate_fid,
    calculate_mmd,
    calculate_wasserstein,
)

def test_latent_eval_distribution():
    # 4 samples, 2 latent dimensions
    src = {"S1": [0.0, 1.0], "S2": [0.5, 0.5], "S3": [0.0, 0.0], "S4": [1.0, 1.0]}
    tgt = {"T1": [0.1, 0.9], "T2": [0.4, 0.6], "T3": [0.1, 0.1], "T4": [0.9, 0.9]}

    metrics = compute_distribution_metrics(src, tgt)
    assert metrics["source_n"] == 4.0
    assert metrics["target_n"] == 4.0
    assert metrics["fid_source_target"] >= 0.0
    assert metrics["mmd_source_target"] >= 0.0
    assert metrics["wasserstein_source_target"] >= 0.0

def test_latent_eval_kmeans():
    combined = {
        "S1": [1.0, 0.0], "S2": [0.9, 0.1],
        "T1": [0.0, 1.0], "T2": [0.1, 0.9]
    }

    # Cancer map separating them perfectly into COAD and READ
    cancer_map = {
        "S1": "COAD", "S2": "COAD",
        "T1": "READ", "T2": "READ"
    }

    metrics = compute_kmeans_cancer_type_metrics(combined, cancer_map, random_state=42)
    assert metrics["samples_used"] == 4.0
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["silhouette"] > 0.0
