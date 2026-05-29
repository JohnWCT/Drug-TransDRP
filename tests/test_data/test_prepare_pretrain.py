import pytest
import os
from transdrp_multilabel.config import TransDRPMultilabelConfig
from transdrp_multilabel.data.prepare_pretrain import prepare_pretrain_data

def test_prepare_pretrain_data(tmp_data_dir):
    config = TransDRPMultilabelConfig(
        task_type="classification",
        source_omics_path=os.path.join(tmp_data_dir, "source_omics.csv"),
        target_omics_path=os.path.join(tmp_data_dir, "target_omics.csv"),
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
        encoder_hidden_dims=(32, 16),
        decoder_hidden_dims=(16, 32),
        classifier_hidden_dims=(16, 8),
        drop=0.1,
        norm_flag=True,
        retrain_flag=True,
        device="cpu"
    )

    prepared = prepare_pretrain_data(config)
    assert len(prepared.source_omics.sample_ids) == 10
    assert len(prepared.target_omics.sample_ids) == 10
    assert len(prepared.source_omics.feature_names) == 20
    assert len(prepared.target_omics.feature_names) == 20
