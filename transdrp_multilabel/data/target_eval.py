"""Target eval dataset preparation (inference/evaluation only, no training)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from transdrp_multilabel.contracts import DrugIndex, ResponseMatrix, TargetEvalDataset
from transdrp_multilabel.data.drug_index import _normalize_drug
from transdrp_multilabel.data.response_matrix import long_to_response_matrix
from transdrp_multilabel.data.sample_id import sample_match_key


def _validate_required_columns(
    df: pd.DataFrame,
    sample_col: str,
    drug_col: str,
    label_col: str,
) -> None:
    missing = [c for c in (sample_col, drug_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Target eval response missing required columns: {missing}")


def prepare_target_eval_dataset(
    eval_response_df: pd.DataFrame,
    eval_dataset_name: str,
    target_sample_ids: list[str],
    drug_index: DrugIndex,
    sample_col: str = "Patient_id",
    drug_col: str = "drug_name",
    label_col: str = "Label",
    task_type: str = "classification",
    duplicate_strategy: str = "mean",
    omics_sample_id_col: str = "tissue_id",
) -> TargetEvalDataset:
    """Build a single target eval dataset with y/mask aligned to target omics."""
    _validate_required_columns(eval_response_df, sample_col, drug_col, label_col)

    raw = eval_response_df.copy()
    input_rows = len(raw)

    df = raw[[sample_col, drug_col, label_col]].copy()
    df[sample_col] = df[sample_col].astype(str).str.strip()
    df[drug_col] = df[drug_col].astype(str).map(_normalize_drug)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    omics_keys = {
        sample_match_key(sid, column_hint=omics_sample_id_col) for sid in target_sample_ids
    }
    final_drug_set = set(drug_index.drug_ids)

    skipped_sample = 0
    skipped_drug = 0
    usable_mask = []
    for _, row in df.iterrows():
        sid_key = sample_match_key(row[sample_col], column_hint=sample_col)
        did = str(row[drug_col])
        if sid_key not in omics_keys:
            skipped_sample += 1
            usable_mask.append(False)
            continue
        if did not in final_drug_set:
            skipped_drug += 1
            usable_mask.append(False)
            continue
        if pd.isna(row[label_col]):
            usable_mask.append(False)
            continue
        usable_mask.append(True)

    usable_rows = int(sum(usable_mask))

    label_semantics: Literal["binary", "continuous"] = (
        "binary" if task_type == "classification" else "continuous"
    )
    response = long_to_response_matrix(
        df,
        target_sample_ids,
        drug_index,
        sample_col,
        drug_col,
        label_col,
        "target",
        label_semantics,
        duplicate_strategy,
        omics_sample_id_col=omics_sample_id_col,
        response_sample_id_col=sample_col,
    )

    observed = response.mask > 0
    label_0 = int(((response.y == 0) & observed).sum())
    label_1 = int(((response.y == 1) & observed).sum())
    n_obs_drugs = len({drug_index.index_to_drug[j] for j in range(drug_index.n_drugs) if observed[:, j].any()})
    n_obs_patients = len({target_sample_ids[i] for i in range(len(target_sample_ids)) if observed[i].any()})

    report = pd.DataFrame([{
        "eval_dataset": eval_dataset_name,
        "input_rows": input_rows,
        "usable_rows": usable_rows,
        "skipped_rows_sample_not_in_target_omics": skipped_sample,
        "skipped_rows_drug_not_in_final_index": skipped_drug,
        "n_observed_patients": n_obs_patients,
        "n_observed_drugs": n_obs_drugs,
        "label_0_count": label_0,
        "label_1_count": label_1,
    }])

    return TargetEvalDataset(
        name=eval_dataset_name,
        response=response,
        raw_response=raw,
        report=report,
    )
