import pytest
import os
import torch
import torch.nn as nn
from transdrp_multilabel.model.checkpoint import save_checkpoint, load_checkpoint

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

def test_checkpoint_save_load(tmp_path):
    model = DummyModel()
    original_weight = model.fc.weight.clone()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, str(ckpt_path))
    assert ckpt_path.is_file()

    model2 = DummyModel()
    # Randomize weight first to verify load works
    with torch.no_grad():
        model2.fc.weight.copy_(torch.randn_like(model2.fc.weight))

    load_checkpoint(model2, str(ckpt_path), device="cpu")
    assert torch.allclose(model2.fc.weight, original_weight)
