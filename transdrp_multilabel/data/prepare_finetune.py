from __future__ import annotations

from typing import Dict, Literal
import pandas as pd
from transdrp_multilabel.contracts import (
    TransDRPMultilabelConfig,
    OmicsTable,
    PreparedFineTuneData,
    TargetEvalDataset,
)
from transdrp_multilabel.data.sample_id import sample_match_key
from transdrp_multilabel.data.cancer_type import load_and_align_cancer_types
from transdrp_multilabel.data.drug_index import (
    build_drug_availability_from_eval_union,
    build_drug_index_from_source,
    build_drug_index_from_eval_union,
    _drug_set,
)
from transdrp_multilabel.data.omics import align_omics_features, read_omics_table
from transdrp_multilabel.data.response_matrix import long_to_response_matrix
from transdrp_multilabel.data.split import split_source_samples
from transdrp_multilabel.data.target_eval import prepare_target_eval_dataset
from transdrp_multilabel.config import optional_data_path
from transdrp_multilabel.io import read_csv
from transdrp_multilabel.validators import (
    validate_omics_table,
    validate_response_long_table,
    validate_response_matrix,
)


def _collect_target_eval_sample_keys(
    dfs: list[pd.DataFrame | None],
    sample_col: str,
) -> set[str]:
    keys: set[str] = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        for s in df[sample_col].astype(str):
            keys.add(sample_match_key(s, column_hint=sample_col))
    return keys


def prepare_finetune_data(config: TransDRPMultilabelConfig) -> PreparedFineTuneData:
    primary_path = optional_data_path(config.target_eval_primary_response_path) or optional_data_path(
        config.target_response_path
    )
    aux_path = optional_data_path(config.target_eval_aux_response_path)
    target_only_path = optional_data_path(config.target_eval_target_only_response_path)
    if not config.source_response_path or not primary_path:
        raise ValueError(
            "Fine-tune requires source_response_path and "
            "(target_eval_primary_response_path or target_response_path)."
        )
    if not config.drug_smiles_path:
        raise ValueError("Fine-tune requires drug_smiles_path for GNN drug node features.")

    src_omics_df = read_csv(config.source_omics_path)
    tgt_omics_df = read_csv(config.target_omics_path)
    validate_omics_table(src_omics_df, config.source_sample_col)
    validate_omics_table(tgt_omics_df, config.target_sample_col)

    source = read_omics_table(config.source_omics_path, config.source_sample_col, "source")
    target = read_omics_table(config.target_omics_path, config.target_sample_col, "target")
    source, target, _alignment = align_omics_features(source, target)

    src_resp_df = read_csv(config.source_response_path)
    primary_tgt_resp_df = read_csv(primary_path)
    aux_tgt_resp_df = read_csv(aux_path) if aux_path else None
    target_only_resp_df = read_csv(target_only_path) if target_only_path else None

    validate_response_long_table(
        src_resp_df,
        config.source_sample_col,
        config.drug_col,
        config.source_response_col,
        config.task_type,
        "source",
    )
    for df, name in (
        (primary_tgt_resp_df, config.target_eval_primary_name),
        (aux_tgt_resp_df, config.target_eval_aux_name),
        (target_only_resp_df, config.target_eval_target_only_name),
    ):
        if df is not None:
            validate_response_long_table(
                df,
                config.target_response_sample_col,
                config.drug_col,
                config.target_response_col,
                config.task_type,
                "target",
            )

    # Target omics: keep samples appearing in ANY target eval response
    resp_keys = _collect_target_eval_sample_keys(
        [primary_tgt_resp_df, aux_tgt_resp_df, target_only_resp_df],
        config.target_response_sample_col,
    )
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
        raise ValueError(
            "No target omics samples matched to any target eval response after ID normalization."
        )

    if config.drug_graph_edge_strategy == "source_cooccurrence":
        # Source-only mode: final drug index is limited to source drugs.
        # Target eval rows containing out-of-source drugs are skipped downstream.
        drug_index = build_drug_index_from_source(
            source_response=src_resp_df,
            drug_col=config.drug_col,
        )
    else:
        drug_index = build_drug_index_from_eval_union(
            source_response=src_resp_df,
            primary_target_response=primary_tgt_resp_df,
            auxiliary_target_response=aux_tgt_resp_df,
            target_only_response=target_only_resp_df,
            drug_col=config.drug_col,
        )
    source_drug_ids = _drug_set(src_resp_df, config.drug_col)

    smiles_df = pd.read_csv(config.drug_smiles_path, header=0)
    drug_availability = build_drug_availability_from_eval_union(
        source_response=src_resp_df,
        primary_target_response=primary_tgt_resp_df,
        auxiliary_target_response=aux_tgt_resp_df,
        target_only_response=target_only_resp_df,
        drug_col=config.drug_col,
        drug_index=drug_index,
        smiles_df=smiles_df,
    )

    # Strict SMILES check on final drug index
    missing_smiles = drug_availability.loc[~drug_availability["has_smiles"], "drug_id"].tolist()
    if missing_smiles:
        missing_lines = "\n".join(f"  - {d}" for d in missing_smiles)
        raise ValueError(
            f"Missing SMILES for final drug index drugs ({len(missing_smiles)}):\n"
            f"{missing_lines}\n\n"
            f"Please fix drug_smiles_path ('{config.drug_smiles_path}') "
            f"or provide a drug alias mapping."
        )

    src_sem: Literal["binary", "continuous"] = (
        "binary" if config.task_type == "classification" else "continuous"
    )
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

    # Target-only drugs: ensure source mask stays zero (no supervised loss)
    for j, did in enumerate(drug_index.drug_ids):
        if did not in source_drug_ids:
            source_response.mask[:, j] = 0.0
            source_response.y[:, j] = 0.0

    target_eval_datasets: Dict[str, TargetEvalDataset] = {}
    eval_specs = [
        (config.target_eval_primary_name, primary_tgt_resp_df),
        (config.target_eval_aux_name, aux_tgt_resp_df),
        (config.target_eval_target_only_name, target_only_resp_df),
    ]
    for eval_name, eval_df in eval_specs:
        if eval_df is None:
            continue
        target_eval_datasets[eval_name] = prepare_target_eval_dataset(
            eval_response_df=eval_df,
            eval_dataset_name=eval_name,
            target_sample_ids=list(target.sample_ids),
            drug_index=drug_index,
            sample_col=config.target_response_sample_col,
            drug_col=config.drug_col,
            label_col=config.target_response_col,
            task_type=config.task_type,
            duplicate_strategy="mean",
            omics_sample_id_col=config.target_sample_col,
        )

    if config.target_eval_primary_name not in target_eval_datasets:
        raise ValueError("Primary target eval dataset could not be built.")

    target_response = target_eval_datasets[config.target_eval_primary_name].response

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
        target_eval_datasets=target_eval_datasets,
        source_drug_ids=source_drug_ids,
        drug_availability=drug_availability,
    )
