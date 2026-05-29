import os
import sys
import shutil
import tempfile
import pandas as pd
import numpy as np
import torch

from transdrp_multilabel.config import TransDRPMultilabelConfig
from transdrp_multilabel.training.runners import PretrainRunner, FineTuneRunner
from transdrp_multilabel.io import read_csv, write_csv

def generate_synthetic_data(temp_dir: str):
    # 1. Omics Data
    genes = [f"gene_{i}" for i in range(50)]

    src_samples = [f"CCLE_sample_{i}" for i in range(20)]
    src_omics = pd.DataFrame(
        np.random.randn(20, 50),
        columns=genes
    )
    src_omics.insert(0, "Sample_ID", src_samples)
    write_csv(src_omics, os.path.join(temp_dir, "source_omics.csv"))

    tgt_samples = [f"TCGA-AA-00{i}-01A" for i in range(20)]
    tgt_omics = pd.DataFrame(
        np.random.randn(20, 50),
        columns=genes
    )
    tgt_omics.insert(0, "tissue_id", tgt_samples)
    write_csv(tgt_omics, os.path.join(temp_dir, "target_omics.csv"))

    # 2. Response Data
    drugs = ["DrugA", "DrugB", "DrugC"]

    src_resp_rows = []
    for sid in src_samples:
        for d in drugs:
            src_resp_rows.append({
                "Sample_ID": sid,
                "drug_name": d,
                "Label": float(np.random.choice([0.0, 1.0]))  # classification
            })
    src_resp = pd.DataFrame(src_resp_rows)
    write_csv(src_resp, os.path.join(temp_dir, "source_response_clf.csv"))

    # Continuous labels for regression
    src_resp_reg = src_resp.copy()
    src_resp_reg["Label"] = np.random.randn(len(src_resp)) * 2.0
    write_csv(src_resp_reg, os.path.join(temp_dir, "source_response_reg.csv"))

    # Target patient labels: note TCGA response uses Patient_id (normalized)
    tgt_patient_ids = [f"TCGA-AA-00{i}" for i in range(20)]
    tgt_resp_rows = []
    for pid in tgt_patient_ids:
        for d in drugs:
            tgt_resp_rows.append({
                "Patient_id": pid,
                "drug_name": d,
                "Label": float(np.random.choice([0.0, 1.0]))
            })
    tgt_resp = pd.DataFrame(tgt_resp_rows)
    write_csv(tgt_resp, os.path.join(temp_dir, "target_response.csv"))

    # 3. Cancer Types
    src_ct = pd.DataFrame({
        "Sample_ID": src_samples,
        "Cancer_type": ["COAD"] * 10 + ["READ"] * 10
    })
    write_csv(src_ct, os.path.join(temp_dir, "source_cancer_types.csv"))

    tgt_ct = pd.DataFrame({
        "Patient_id": tgt_patient_ids,
        "Cancer_type": ["COAD"] * 10 + ["READ"] * 10
    })
    write_csv(tgt_ct, os.path.join(temp_dir, "target_cancer_types.csv"))

    # 4. SMILES Mapping
    # SMILES for 3 arbitrary drugs: DrugA, DrugB, DrugC
    smiles_df = pd.DataFrame({
        "drug_id": ["druga", "drugb", "drugc"],
        "Isosmiles": [
            "CC1=C(C(C(=C(C1=O)C)O)O)C", # dummy SMILES
            "CN(C)C(=N)N=C(N)N",
            "CC1(C(C2C(C(O1)O)OC3C2(C(=C)C(C3=O)O)C)O)C"
        ]
    }).set_index("drug_id")
    write_csv(smiles_df.reset_index(), os.path.join(temp_dir, "drug_smiles.csv"))

def print_directory_tree(path: str, indent: str = "") -> None:
    if not os.path.exists(path):
        return
    items = sorted(os.listdir(path))
    for idx, item in enumerate(items):
        is_last = (idx == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(f"{indent}{prefix}{item}/")
            next_indent = indent + ("    " if is_last else "│   ")
            print_directory_tree(full_path, next_indent)
        else:
            size = os.path.getsize(full_path)
            if size < 1024:
                size_str = f"{size} B"
            else:
                size_str = f"{size / 1024:.1f} KB"
            print(f"{indent}{prefix}{item} ({size_str})")

def run_smoke_test():
    is_pytest = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ

    if is_pytest:
        smoke_dir = tempfile.mkdtemp(prefix="transdrp_smoke_")
        print(f"Running under Pytest. Using temporary directory {smoke_dir}...")
    else:
        smoke_dir = "/workspace/TransDRP/smoke_outputs"
        print(f"Running directly. Using persistent directory {smoke_dir}...")
        if os.path.exists(smoke_dir):
            shutil.rmtree(smoke_dir, ignore_errors=True)
        os.makedirs(smoke_dir, exist_ok=True)

    output_dir = os.path.join(smoke_dir, "outputs")

    try:
        print(f"Generating synthetic datasets in {smoke_dir}...")
        generate_synthetic_data(smoke_dir)

        # 1. Run Pre-training
        print("\n--- Running Stage 1 Pre-training Smoke Test ---")
        pre_config = TransDRPMultilabelConfig(
            task_type="classification",
            source_omics_path=os.path.join(smoke_dir, "source_omics.csv"),
            target_omics_path=os.path.join(smoke_dir, "target_omics.csv"),
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
            output_dir=output_dir,
            overwrite=True,
            batch_size=8,
            epochs=2,
            lr=0.001,
            seed=2024,
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
            encoder_hidden_dims=(32, 16, 8),
            decoder_hidden_dims=(16, 32),
            classifier_hidden_dims=(16, 8),
            drop=0.1,
            norm_flag=True,
            retrain_flag=True,
            device="cpu"
        )

        PretrainRunner(pre_config).run()

        # 2. Run Classification Fine-tuning
        print("\n--- Running Stage 2 Classification Fine-tuning Smoke Test ---")
        clf_config = TransDRPMultilabelConfig(
            task_type="classification",
            source_omics_path=os.path.join(smoke_dir, "source_omics.csv"),
            target_omics_path=os.path.join(smoke_dir, "target_omics.csv"),
            source_response_path=os.path.join(smoke_dir, "source_response_clf.csv"),
            target_response_path=os.path.join(smoke_dir, "target_response.csv"),
            source_sample_col="Sample_ID",
            target_sample_col="tissue_id",
            target_response_sample_col="Patient_id",
            drug_col="drug_name",
            source_response_col="Label",
            target_response_col="Label",
            method="transdrp_ft",
            pretrain_checkpoint=os.path.join(output_dir, "pretrain", "checkpoint.pt"),
            output_dir=output_dir,
            overwrite=True,
            batch_size=8,
            epochs=2,
            lr=0.001,
            seed=2024,
            n_splits=2,
            source_test_size=0.25,
            metric="macro_auroc",
            reg_loss="mae",
            prediction_threshold=0.5,
            regression_binary_threshold=1.0,
            source_cancer_type_path=os.path.join(smoke_dir, "source_cancer_types.csv"),
            target_cancer_type_path=os.path.join(smoke_dir, "target_cancer_types.csv"),
            cancer_type_col="Cancer_type",
            drug_smiles_path=os.path.join(smoke_dir, "drug_smiles.csv"),
            alph=0.2,
            beta=0.3,
            latent_dim=8,
            encoder_hidden_dims=(32, 16, 8),
            decoder_hidden_dims=(16, 32),
            classifier_hidden_dims=(16, 8),
            drop=0.1,
            norm_flag=True,
            retrain_flag=True,
            device="cpu"
        )
        FineTuneRunner(clf_config).run()

        # 3. Run Regression Fine-tuning
        print("\n--- Running Stage 2 Regression Fine-tuning Smoke Test ---")
        reg_config = TransDRPMultilabelConfig(
            task_type="regression",
            source_omics_path=os.path.join(smoke_dir, "source_omics.csv"),
            target_omics_path=os.path.join(smoke_dir, "target_omics.csv"),
            source_response_path=os.path.join(smoke_dir, "source_response_reg.csv"),
            target_response_path=os.path.join(smoke_dir, "target_response.csv"),
            source_sample_col="Sample_ID",
            target_sample_col="tissue_id",
            target_response_sample_col="Patient_id",
            drug_col="drug_name",
            source_response_col="Label",
            target_response_col="Label",
            method="transdrp_ft",
            pretrain_checkpoint=os.path.join(output_dir, "pretrain", "checkpoint.pt"),
            output_dir=output_dir,
            overwrite=True,
            batch_size=8,
            epochs=2,
            lr=0.001,
            seed=2024,
            n_splits=2,
            source_test_size=0.25,
            metric="macro_mae",
            reg_loss="mae",
            prediction_threshold=0.5,
            regression_binary_threshold=0.0,  # binarization threshold for regression
            source_cancer_type_path=os.path.join(smoke_dir, "source_cancer_types.csv"),
            target_cancer_type_path=os.path.join(smoke_dir, "target_cancer_types.csv"),
            cancer_type_col="Cancer_type",
            drug_smiles_path=os.path.join(smoke_dir, "drug_smiles.csv"),
            alph=0.2,
            beta=0.3,
            latent_dim=8,
            encoder_hidden_dims=(32, 16, 8),
            decoder_hidden_dims=(16, 32),
            classifier_hidden_dims=(16, 8),
            drop=0.1,
            norm_flag=True,
            retrain_flag=True,
            device="cpu"
        )
        FineTuneRunner(reg_config).run()

        print("\nSmoke tests completed successfully!")

        if not is_pytest:
            print("\n================ Generated Outputs Tree ================")
            print_directory_tree(smoke_dir)
            print("========================================================")
            print(f"\nAll outputs are preserved in: {smoke_dir}")

    finally:
        if is_pytest:
            print(f"Cleaning up temporary directory {smoke_dir}...")
            shutil.rmtree(smoke_dir, ignore_errors=True)

if __name__ == "__main__":
    run_smoke_test()
