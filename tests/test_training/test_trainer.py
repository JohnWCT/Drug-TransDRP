import pytest
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from transdrp_multilabel.training.trainer import get_tissue_prototypes

class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 8)
    def forward(self, x):
        return self.fc(x)

def test_get_tissue_prototypes():
    encoder = MockEncoder()
    # 6 samples, 10 features
    x = torch.randn(6, 10)
    cancer_table = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(6)],
        "cancer_type": ["COAD", "COAD", "READ", "READ", "Unknown", "Unknown"],
        "domain": ["source"] * 6
    })

    loader = get_tissue_prototypes(
        encoder,
        x,
        cancer_table,
        [f"S{i}" for i in range(6)],
        domain="source",
        device="cpu"
    )

    # Check that it returns a DataLoader containing the prototypes
    assert loader is not None
    batch = next(iter(loader))
    types, protos = batch
    assert len(types) == 3 # COAD, READ, Unknown
    assert protos.shape == (3, 8)
