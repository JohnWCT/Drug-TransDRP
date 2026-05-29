import pytest
import os
import pandas as pd
from transdrp_multilabel.export.visualization import run_tsne, plot_tsne_by_domain, plot_tsne_by_cancer_type

def test_visualization_flow(tmp_path):
    # Setup dataframe with 10 samples and 4 latent features
    data = {
        "sample_id": [f"S{i}" for i in range(10)],
        "domain": ["source"] * 5 + ["target"] * 5,
        "latent_0": [0.1 * i for i in range(10)],
        "latent_1": [0.2 * i for i in range(10)],
        "latent_2": [0.3 * i for i in range(10)],
        "latent_3": [0.4 * i for i in range(10)],
    }
    df = pd.DataFrame(data)

    tsne_df = run_tsne(df, seed=42)
    assert tsne_df is not None
    assert "tsne_1" in tsne_df.columns
    assert "tsne_2" in tsne_df.columns

    domain_path = tmp_path / "tsne_domain.png"
    assert plot_tsne_by_domain(tsne_df, str(domain_path))
    assert domain_path.is_file()

    ct_df = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(10)],
        "cancer_type": ["COAD"] * 5 + ["READ"] * 5
    })

    ct_path = tmp_path / "tsne_ct.png"
    assert plot_tsne_by_cancer_type(tsne_df, ct_df, str(ct_path))
    assert ct_path.is_file()
