import torch
import torch.nn.functional as F

def safe_masked_mean(raw_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (raw_loss * mask).sum() / denom

def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, y.double(), reduction="none")
    return safe_masked_mean(raw, mask)

def masked_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = torch.abs(pred - y)
    return safe_masked_mean(raw, mask)
