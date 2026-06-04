from typing import Literal
import numpy as np
import pandas as pd
from transdrp_multilabel.contracts import DrugIndex, ResponseMatrix
from transdrp_multilabel.data.drug_index import _normalize_drug
from transdrp_multilabel.data.sample_id import sample_match_key

def long_to_response_matrix(
    response_df: pd.DataFrame,
    sample_ids: list[str],
    drug_index: DrugIndex,
    sample_id_col: str,
    drug_col: str,
    response_col: str,
    domain: Literal["source", "target"],
    label_semantics: Literal["binary", "continuous"],
    duplicate_strategy: Literal["mean", "median", "first", "error"] = "mean",
    *,
    omics_sample_id_col: str = None,
    response_sample_id_col: str = None,
) -> ResponseMatrix:
    df = response_df[[sample_id_col, drug_col, response_col]].copy()
    df[sample_id_col] = df[sample_id_col].astype(str).str.strip()
    df[drug_col] = df[drug_col].astype(str).map(_normalize_drug)
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")

    dup = df.duplicated(subset=[sample_id_col, drug_col], keep=False)
    if dup.any():
        if duplicate_strategy == "error":
            raise ValueError(f"Duplicate sample-drug rows found in {domain} response table.")
        elif duplicate_strategy == "first":
            df = df.drop_duplicates(subset=[sample_id_col, drug_col], keep="first")
        elif duplicate_strategy == "mean":
            df = df.groupby([sample_id_col, drug_col], as_index=False)[response_col].mean()
        elif duplicate_strategy == "median":
            df = df.groupby([sample_id_col, drug_col], as_index=False)[response_col].median()
        else:
            raise ValueError(f"Unknown duplicate strategy: {duplicate_strategy}")

    n_samples = len(sample_ids)
    n_drugs = len(drug_index.drug_ids)
    y = np.zeros((n_samples, n_drugs), dtype=np.float32)
    mask = np.zeros((n_samples, n_drugs), dtype=np.float32)

    omics_hint = omics_sample_id_col or sample_id_col
    resp_hint = response_sample_id_col or sample_id_col

    sample_to_row = {
        sample_match_key(sid, column_hint=omics_hint): i for i, sid in enumerate(sample_ids)
    }

    for _, row in df.iterrows():
        sid_key = sample_match_key(row[sample_id_col], column_hint=resp_hint)
        did = str(row[drug_col])
        if sid_key not in sample_to_row or did not in drug_index.drug_to_index:
            continue
        if pd.isna(row[response_col]):
            continue
        i = sample_to_row[sid_key]
        j = drug_index.drug_to_index[did]
        y[i, j] = float(row[response_col])
        mask[i, j] = 1.0

    return ResponseMatrix(
        y=y,
        mask=mask,
        sample_ids=sample_ids,
        drug_index=drug_index,
        domain=domain,
        label_semantics=label_semantics,
    )
