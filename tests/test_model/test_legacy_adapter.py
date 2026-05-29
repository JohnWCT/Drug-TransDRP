import pytest
import os
import torch
import pandas as pd
import numpy as np
from transdrp_multilabel.config import TransDRPMultilabelConfig
from transdrp_multilabel.contracts import PreparedPretrainData, OmicsTable
from transdrp_multilabel.model.legacy_adapter import (
    build_legacy_transdrp_components,
    build_pretrain_dataloaders,
)

def test_legacy_components():
    config = TransDRPMultilabelConfig(
        task_type="classification",
        source_omics_path="dummy",
        target_omics_path="dummy",
        source_response_path=None,
        target_response_path=None,
        source_sample_col="Sample_ID",
        target_sample_col="tissue_id",
        target_response_sample_col="Patient_id",
        drug_col="drug_name",
        source_response_col="Label",
        target_response_col="Label",
        method="transdrp_ae",
        pretrain_checkpoint=None,
        output_dir="dummy",
        overwrite=True,
        batch_size=4,
        epochs=1,
        lr=0.01,
        seed=42,
        n_splits=2,
        source_test_size=0.25,
        metric=None,
        reg_loss="mae",
        prediction_threshold=0.5,
        regression_binary_threshold=1.0,
        source_cancer_type_path=None,
        target_cancer_type_path=None,
        cancer_type_col="Cancer_type",
        drug_smiles_path=None,
        alph=0.2,
        beta=0.3,
        latent_dim=8,
        encoder_hidden_dims=(16, 8),
        decoder_hidden_dims=(8, 16),
        classifier_hidden_dims=(16, 8),
        drop=0.1,
        norm_flag=True,
        retrain_flag=True,
        device="cpu"
    )

    comp = build_legacy_transdrp_components(config, n_features=50)
    assert comp["shared_encoder"] is not None
    assert comp["latent_dim"] == 8

def test_build_pretrain_dataloaders():
    src_df = pd.DataFrame(np.random.randn(8, 10))
    tgt_df = pd.DataFrame(np.random.randn(8, 10))
    src = OmicsTable(x=src_df, sample_ids=[f"S{i}" for i in range(8)], feature_names=[f"g{i}" for i in range(10)], domain="source")
    tgt = OmicsTable(x=tgt_df, sample_ids=[f"T{i}" for i in range(8)], feature_names=[f"g{i}" for i in range(10)], domain="target")

    prep = PreparedPretrainData(source_omics=src, target_omics=tgt, feature_alignment=pd.DataFrame())

    s_dl, t_dl = build_pretrain_dataloaders(prep, batch_size=2, seed=42)
    assert len(s_dl[0]) == 3 # 8 * 0.75 = 6 training samples / 2 = 3 batches
    assert len(t_dl[0]) == 3
