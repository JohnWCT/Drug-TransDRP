from typing import Literal
import pandas as pd
from transdrp_multilabel.contracts import TransDRPMultilabelConfig, OmicsTable, PreparedFineTuneData
from transdrp_multilabel.data.sample_id import sample_match_key
from transdrp_multilabel.data.cancer_type import load_and_align_cancer_types
from transdrp_multilabel.data.drug_index import (
    build_drug_availability,
    build_drug_index_from_source,
)
from transdrp_multilabel.data.omics import align_omics_features, read_omics_table
from transdrp_multilabel.data.response_matrix import long_to_response_matrix
from transdrp_multilabel.data.split import split_source_samples
from transdrp_multilabel.io import read_csv
from transdrp_multilabel.validators import (
    validate_omics_table,
    validate_response_long_table,
    validate_response_matrix,
)

def prepare_finetune_data(config: TransDRPMultilabelConfig) -> PreparedFineTuneData:
    if not config.source_response_path or not config.target_response_path:
        raise ValueError("Fine-tune requires both source_response_path and target_response_path.")
    if not config.drug_smiles_path:
        raise ValueError("Fine-tune requires drug_smiles_path for GNN drug node features.")

    src_omics_df = read_csv(config.source_omics_path)
    tgt_omics_df = read_csv(config.target_omics_path)
    validate_omics_table(src_omics_df, config.source_sample_col)
    validate_omics_table(tgt_omics_df, config.target_sample_col)

    source = read_omics_table(config.source_omics_path, config.source_sample_col, "source")
    target = read_omics_table(config.target_omics_path, config.target_sample_col, "target")
    source, target, alignment = align_omics_features(source, target)

    src_resp_df = read_csv(config.source_response_path)
    tgt_resp_df = read_csv(config.target_response_path)

    validate_response_long_table(
        src_resp_df,
        config.source_sample_col,
        config.drug_col,
        config.source_response_col,
        config.task_type,
        "source",
    )
    validate_response_long_table(
        tgt_resp_df,
        config.target_response_sample_col,
        config.drug_col,
        config.target_response_col,
        config.task_type,
        "target",
    )

    # Check that patient response matches target omics samples
    resp_keys = {
        sample_match_key(s, column_hint=config.target_response_sample_col)
        for s in tgt_resp_df[config.target_response_sample_col].astype(str)
    }
    keep = [
        sid
        for sid in target.sample_ids
        if sample_match_key(sid, column_hint=config.target_sample_col) in resp_keys
    ]
    if keep:
        if len(keep) < len(target.sample_ids):
            target = OmicsTable(
                x=target.x.loc[keep].copy(),
                sample_ids=keep,
                feature_names=target.feature_names,
                domain="target",
            )
    else:
        raise ValueError("No target omics samples matched to target response after ID normalization.")

    # Build drug availability report over the union (for reporting only) and
    # build the FINAL drug index from source drugs only. target-only drugs are
    # dropped here: they are not evaluated and are not required to have SMILES.
    drug_availability = build_drug_availability(src_resp_df, tgt_resp_df, config.drug_col)
    drug_index = build_drug_index_from_source(src_resp_df, config.drug_col)

    # Load SMILES and perform strict presence validation (Raise error and stop)
    # NOTE: validation runs ONLY on the final source-drug index, so a target-only
    # drug missing from the SMILES table will NOT halt execution.
    smiles_df = pd.read_csv(config.drug_smiles_path, header=0)
    cols_lower = [str(c).lower().strip() for c in smiles_df.columns]
    smiles_df.columns = cols_lower

    first_col = cols_lower[0]
    name_col = None
    for possible in ["name", "drug_name", "drug_id"]:
        if possible in cols_lower and possible != first_col:
            name_col = possible
            break

    smiles_keys = set()
    for _, row in smiles_df.iterrows():
        key1 = str(row[first_col]).strip().lower()
        smiles_keys.add(key1)
        if name_col:
            key2 = str(row[name_col]).strip().lower()
            smiles_keys.add(key2)

    missing_drugs = []
    for d in drug_index.drug_ids:
        if str(d).strip().lower() not in smiles_keys:
            missing_drugs.append(d)

    if missing_drugs:
        missing_lines = "\n".join(f"  - {d}" for d in missing_drugs)
        raise ValueError(
            f"Missing SMILES for {len(missing_drugs)} source drugs:\n"
            f"{missing_lines}\n\n"
            f"These drugs are in the final source drug index and cannot be excluded "
            f"automatically. Please fix drug_smiles_path ('{config.drug_smiles_path}') "
            f"or provide a drug alias mapping. Halted execution."
        )

    # Convert response tables to wide matrices
    src_sem: Literal["binary", "continuous"] = (
        "binary" if config.task_type == "classification" else "continuous"
    )
    # Target is always evaluation-only binary labels
    tgt_sem: Literal["binary"] = "binary"

    source_response = long_to_response_matrix(
        src_resp_df,
        list(source.sample_ids),
        drug_index,
        config.source_sample_col,
        config.drug_col,
        config.source_response_col,
        "source",
        src_sem,
        "mean",
        omics_sample_id_col=config.source_sample_col,
        response_sample_id_col=config.source_sample_col,
    )
    target_response = long_to_response_matrix(
        tgt_resp_df,
        list(target.sample_ids),
        drug_index,
        config.target_response_sample_col,
        config.drug_col,
        config.target_response_col,
        "target",
        tgt_sem,
        "mean",
        omics_sample_id_col=config.target_sample_col,
        response_sample_id_col=config.target_response_sample_col,
    )

    validate_response_matrix(source_response)
    validate_response_matrix(target_response)

    folds = split_source_samples(
        list(source.sample_ids),
        source_response.y,
        source_response.mask,
        config.source_test_size,
        config.n_splits,
        config.seed,
    )

    cancer_type_table = load_and_align_cancer_types(
        list(source.sample_ids),
        list(target.sample_ids),
        config.source_cancer_type_path,
        config.target_cancer_type_path,
        config.cancer_type_col,
    )

    return PreparedFineTuneData(
        source_omics=source,
        target_omics=target,
        source_response=source_response,
        target_response=target_response,
        drug_index=drug_index,
        folds=folds,
        cancer_type_table=cancer_type_table,
        drug_availability=drug_availability,
    )
