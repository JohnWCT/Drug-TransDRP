"""Runner-layer drug-set validation after data preparation."""

from __future__ import annotations

import pandas as pd

from transdrp_multilabel.contracts import DrugIndex


def _normalize_drug(d: object) -> str:
    return str(d).strip().lower()


def _drugs_from_response(df: pd.DataFrame | None, drug_col: str) -> set[str]:
    if df is None or df.empty:
        return set()
    return {_normalize_drug(d) for d in df[drug_col].astype(str) if str(d).strip()}


def validate_final_drug_index(
    drug_index: DrugIndex,
    source_response: pd.DataFrame,
    primary_target_response: pd.DataFrame | None,
    auxiliary_target_response: pd.DataFrame | None,
    target_only_response: pd.DataFrame | None,
    drug_col: str,
    smiles_path: str | None = None,
) -> pd.DataFrame:
    """Ensure final drug index equals source ∪ all target eval drugs."""
    source_drugs = _drugs_from_response(source_response, drug_col)
    primary_drugs = _drugs_from_response(primary_target_response, drug_col)
    aux_drugs = _drugs_from_response(auxiliary_target_response, drug_col)
    target_only_drugs = _drugs_from_response(target_only_response, drug_col)
    all_target_eval = primary_drugs | aux_drugs | target_only_drugs
    expected = source_drugs | all_target_eval
    final_drugs = set(drug_index.drug_ids)

    if final_drugs != expected:
        extra = final_drugs - expected
        missing = expected - final_drugs
        parts = ["final drug_index must equal source ∪ target eval drugs."]
        if extra:
            parts.append(f"Unexpected drugs in index: {sorted(extra)[:10]}")
        if missing:
            parts.append(f"Missing drugs in index: {sorted(missing)[:10]}")
        raise ValueError(" ".join(parts))

    if smiles_path:
        from transdrp_multilabel.training.runners import get_drug_features
        get_drug_features(drug_index, smiles_path)

    rows = []
    for d in drug_index.drug_ids:
        in_src = d in source_drugs
        in_any_tgt = d in all_target_eval
        rows.append({
            "drug_id": d,
            "drug_index": drug_index.drug_to_index[d],
            "in_source": in_src,
            "in_any_target_eval": in_any_tgt,
            "has_supervised_source_label": in_src,
            "is_target_eval_only": (not in_src) and in_any_tgt,
        })
    return pd.DataFrame(rows)


def eval_drug_indices(drug_index: DrugIndex, eval_drug_ids_list: list[str]) -> list[int]:
    return [drug_index.drug_to_index[d] for d in eval_drug_ids_list if d in drug_index.drug_to_index]
