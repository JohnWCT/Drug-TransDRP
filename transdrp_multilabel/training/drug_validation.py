"""Runner-layer drug-set validation after data preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from transdrp_multilabel.contracts import DrugIndex, PreparedFineTuneData, TransDRPMultilabelConfig
from transdrp_multilabel.io import read_csv


def _normalize_drug(d: object) -> str:
    return str(d).strip().lower()


def _drugs_from_response(df: pd.DataFrame, drug_col: str) -> set[str]:
    return {_normalize_drug(d) for d in df[drug_col].astype(str) if str(d).strip()}


def validate_final_drug_index(
    prepared: PreparedFineTuneData,
    config: TransDRPMultilabelConfig,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Ensure final drug index equals all source drugs and excludes target-only drugs."""
    src_df = read_csv(config.source_response_path)
    tgt_df = read_csv(config.target_response_path)

    source_drugs = _drugs_from_response(src_df, config.drug_col)
    target_drugs = _drugs_from_response(tgt_df, config.drug_col)
    target_only = target_drugs - source_drugs
    source_only = source_drugs - target_drugs
    shared_drugs = source_drugs & target_drugs

    final_drugs = set(prepared.drug_index.drug_ids)

    if final_drugs != source_drugs:
        extra = final_drugs - source_drugs
        missing = source_drugs - final_drugs
        parts = ["final drug_index must equal all source drugs."]
        if extra:
            parts.append(f"Unexpected drugs in index: {sorted(extra)[:10]}")
        if missing:
            parts.append(f"Missing source drugs in index: {sorted(missing)[:10]}")
        raise ValueError(" ".join(parts))

    leaked = target_only & final_drugs
    if leaked:
        raise ValueError(
            f"target-only drugs must not appear in final drug_index: {sorted(leaked)[:10]}"
        )

    if prepared.drug_availability is not None:
        bad = prepared.drug_availability[
            (prepared.drug_availability["category"] == "target_only")
            & (prepared.drug_availability["in_final_index"] == True)  # noqa: E712
        ]
        if not bad.empty:
            raise ValueError(
                "drug_availability_report marks target-only drugs as in_final_index."
            )

    return source_drugs, target_drugs, shared_drugs, target_only


def target_eval_drug_ids(
    drug_index: DrugIndex,
    shared_drugs: set[str],
    target_mask: Optional[np.ndarray] = None,
) -> list[str]:
    """Drugs used for target prediction / metrics = source ∩ target with observations."""
    if target_mask is None:
        return sorted(shared_drugs)

    eval_drugs = sorted(
        d for d in shared_drugs
        if d in drug_index.drug_to_index and target_mask[:, drug_index.drug_to_index[d]].sum() > 0
    )
    unexpected = {
        drug_index.index_to_drug[j]
        for j in range(drug_index.n_drugs)
        if target_mask[:, j].sum() > 0
    } - shared_drugs
    if unexpected:
        raise ValueError(
            f"target-only drugs observed in target mask: {sorted(unexpected)[:10]}"
        )
    return eval_drugs


def eval_drug_indices(drug_index: DrugIndex, eval_drug_ids_list: list[str]) -> list[int]:
    return [drug_index.drug_to_index[d] for d in eval_drug_ids_list]
