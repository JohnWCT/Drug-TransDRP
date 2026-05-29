import pytest
import torch
from transdrp_multilabel.training.losses import masked_bce_with_logits, masked_mae

def test_masked_bce_with_logits():
    # 2 samples, 3 drugs
    y_pred = torch.tensor([[0.0, 10.0, -10.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    y_true = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)

    # Fully observed
    mask = torch.ones((2, 3), dtype=torch.float32)
    loss = masked_bce_with_logits(y_pred, y_true, mask)
    assert loss.item() > 0

    # MAsked (ignore second sample completely)
    mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    loss_masked = masked_bce_with_logits(y_pred, y_true, mask)
    # The first sample predictions are very close to targets, so loss should be low
    assert loss_masked.item() < loss.item()

    # Zero mask case
    mask = torch.zeros((2, 3), dtype=torch.float32)
    assert masked_bce_with_logits(y_pred, y_true, mask).item() == 0.0

def test_masked_mae():
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 5.0], [3.0, 8.0]], dtype=torch.float32)

    # mask out index (0, 1) and (1, 1), leaving only exact matches
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    loss = masked_mae(y_pred, y_true, mask)
    assert loss.item() == 0.0
