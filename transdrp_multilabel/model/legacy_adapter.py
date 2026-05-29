import sys
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import TransDRP root modules
_PKG_ROOT = Path(__file__).resolve().parents[1]
_TRANSDRP_ROOT = _PKG_ROOT.parent
if str(_TRANSDRP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRANSDRP_ROOT))

import config as legacy_config
import pretraining
from models import FeatMLP
from transdrp_multilabel.contracts import TransDRPMultilabelConfig, PreparedPretrainData

def build_legacy_transdrp_components(
    config: TransDRPMultilabelConfig,
    n_features: int,
) -> dict[str, Any]:
    """Build shared encoder using original FeatMLP."""
    encoder = FeatMLP(
        input_dim=n_features,
        output_dim=config.latent_dim,
        hidden_dims=list(config.encoder_hidden_dims),
        drop=config.drop
    )
    return {"shared_encoder": encoder, "latent_dim": config.latent_dim}

def build_pretrain_dataloaders(
    prepared: PreparedPretrainData,
    batch_size: int,
    seed: int,
) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
    """Build pretraining dataloaders yielding (x, tissue_index) aligned with TransDRP pretraining."""
    from sklearn.model_selection import train_test_split

    src_x = prepared.source_omics.x.values.astype("float32")
    tgt_x = prepared.target_omics.x.values.astype("float32")

    # Tissue mapping setup: default to 0 if not provided
    src_tissue = np.zeros(len(src_x), dtype=np.int64)
    tgt_tissue = np.zeros(len(tgt_x), dtype=np.int64)

    # Update legacy config's tissue_map to be a dictionary mapping { 'Unknown': 0 }
    legacy_config.tissue_map = {"Unknown": 0}

    # Train-test split
    src_tr_x, src_te_x, src_tr_t, src_te_t = train_test_split(src_x, src_tissue, test_size=0.25, random_state=seed)
    tgt_tr_x, tgt_te_x, tgt_tr_t, tgt_te_t = train_test_split(tgt_x, tgt_tissue, test_size=0.25, random_state=seed)

    gen = torch.Generator().manual_seed(seed)

    def _dl(x_arr: np.ndarray, t_arr: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(x_arr), torch.from_numpy(t_arr))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=gen if shuffle else None,
            drop_last=shuffle
        )

    s_train = _dl(src_tr_x, src_tr_t, True)
    s_test = _dl(src_te_x, src_te_t, False)
    t_train = _dl(tgt_tr_x, tgt_tr_t, True)
    t_test = _dl(tgt_te_x, tgt_te_t, False)

    return (s_train, s_test), (t_train, t_test)

def run_pretrain(
    config: TransDRPMultilabelConfig,
    prepared: PreparedPretrainData,
    model_save_folder: str,
) -> nn.Module:
    """Run unsupervised pretraining using TransDRP pretraining.training."""
    n_features = len(prepared.source_omics.feature_names)
    s_dl, t_dl = build_pretrain_dataloaders(prepared, config.batch_size, config.seed)

    kwargs: dict[str, Any] = {
        "input_dim": n_features,
        "latent_dim": config.latent_dim,
        "encoder_hidden_dims": list(config.encoder_hidden_dims),
        "decoder_hidden_dims": list(config.decoder_hidden_dims),
        "lr": config.lr,
        "batch_size": config.batch_size,
        "pretrain_num_epochs": config.epochs,  # mapped from config.epochs
        "norm_flag": config.norm_flag,
        "retrain_flag": config.retrain_flag,
        "drop": config.drop,
        "device": config.device,
        "model_save_folder": model_save_folder,
        "alph": config.alph,
        "beta": config.beta
    }

    shared_encoder = pretraining.training(s_dataloaders=s_dl, t_dataloaders=t_dl, **kwargs)
    return shared_encoder
