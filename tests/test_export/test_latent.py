import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transdrp_multilabel.contracts import OmicsTable
from transdrp_multilabel.export.latent import extract_latent_table

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 8)
        self.encoder.norm_flag = True

def test_extract_latent_table():
    model = MockModel()
    src_df = pd.DataFrame(np.random.randn(5, 10))
    src_df.index = [f"S{i}" for i in range(5)]
    src = OmicsTable(
        x=src_df,
        sample_ids=[f"S{i}" for i in range(5)],
        feature_names=[f"g{i}" for i in range(10)],
        domain="source"
    )

    df = extract_latent_table(model, src, batch_size=2, device="cpu")
    assert len(df) == 5
    assert df.loc[0, "domain"] == "source"
    assert "latent_0" in df.columns
    assert "latent_7" in df.columns
