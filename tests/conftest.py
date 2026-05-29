import pytest
import os
import numpy as np
import pandas as pd

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory with synthetic CSV files for testing."""
    genes = [f"gene_{i}" for i in range(20)]

    # Source omics
    src_samples = [f"CCLE_sample_{i}" for i in range(10)]
    src_omics = pd.DataFrame(np.random.randn(10, 20), columns=genes)
    src_omics.insert(0, "Sample_ID", src_samples)
    src_omics.to_csv(tmp_path / "source_omics.csv", index=False)

    # Target omics
    tgt_samples = [f"TCGA-AA-00{i}-01A" for i in range(10)]
    tgt_omics = pd.DataFrame(np.random.randn(10, 20), columns=genes)
    tgt_omics.insert(0, "tissue_id", tgt_samples)
    tgt_omics.to_csv(tmp_path / "target_omics.csv", index=False)

    # Drug SMILES
    drugs = ["druga", "drugb", "drugc"]
    smiles_df = pd.DataFrame({
        "drug_id": drugs,
        "Isosmiles": [
            "CC1=C(C(C(=C(C1=O)C)O)O)C",
            "CN(C)C(=N)N=C(N)N",
            "CC1(C(C2C(C(O1)O)OC3C2(C(=C)C(C3=O)O)C)O)C",
        ],
    })
    smiles_df.to_csv(tmp_path / "drug_smiles.csv", index=False)

    # Source response (classification)
    src_resp_rows = []
    for sid in src_samples:
        for d in ["DrugA", "DrugB", "DrugC"]:
            src_resp_rows.append({
                "Sample_ID": sid,
                "drug_name": d,
                "Label": float(np.random.choice([0.0, 1.0])),
            })
    pd.DataFrame(src_resp_rows).to_csv(tmp_path / "source_response.csv", index=False)

    # Target response
    tgt_patient_ids = [f"TCGA-AA-00{i}" for i in range(10)]
    tgt_resp_rows = []
    for pid in tgt_patient_ids:
        for d in ["DrugA", "DrugB", "DrugC"]:
            tgt_resp_rows.append({
                "Patient_id": pid,
                "drug_name": d,
                "Label": float(np.random.choice([0.0, 1.0])),
            })
    pd.DataFrame(tgt_resp_rows).to_csv(tmp_path / "target_response.csv", index=False)

    # Cancer types
    pd.DataFrame({
        "Sample_ID": src_samples,
        "Cancer_type": ["COAD"] * 5 + ["READ"] * 5,
    }).to_csv(tmp_path / "source_cancer_types.csv", index=False)

    pd.DataFrame({
        "Patient_id": tgt_patient_ids,
        "Cancer_type": ["COAD"] * 5 + ["READ"] * 5,
    }).to_csv(tmp_path / "target_cancer_types.csv", index=False)

    return tmp_path
