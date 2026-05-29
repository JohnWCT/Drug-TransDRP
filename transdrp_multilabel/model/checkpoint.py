import os
import torch
import torch.nn as nn

def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file '{path}' not found.")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
