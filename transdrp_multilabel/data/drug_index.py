from __future__ import annotations

import pandas as pd
from transdrp_multilabel.contracts import DrugIndex
from transdrp_multilabel.io import write_csv


def _normalize_drug(s: str) -> str:
    return str(s).strip().lower()


def _drug_set(response: pd.DataFrame | None, drug_col: str) -> set[str]:
    if response is None or response.empty:
        return set()
    return {
        _normalize_drug(d)
        for d in response[drug_col].astype(str)
        if str(d).strip() and str(d).lower() != "nan"
    }


def build_drug_index_from_source(
    source_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    """Build drug index from source response drugs only."""
    src = _drug_set(source_response, drug_col)
    drug_ids = sorted(src)
    if not drug_ids:
        raise ValueError("Source drug list is empty; cannot build final drug index.")
    if any(not d for d in drug_ids):
        raise ValueError("Empty drug_id found in source response table.")

    drug_to_index = {d: i for i, d in enumerate(drug_ids)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=drug_ids, drug_to_index=drug_to_index, index_to_drug=index_to_drug)


def build_drug_index_from_eval_union(
    source_response: pd.DataFrame,
    primary_target_response: pd.DataFrame | None,
    auxiliary_target_response: pd.DataFrame | None,
    target_only_response: pd.DataFrame | None,
    drug_col: str,
) -> DrugIndex:
    """Final drug index = source ∪ primary ∪ auxiliary ∪ target-only eval drugs."""
    src = _drug_set(source_response, drug_col)
    primary = _drug_set(primary_target_response, drug_col)
    auxiliary = _drug_set(auxiliary_target_response, drug_col)
    target_only = _drug_set(target_only_response, drug_col)

    union = sorted(src | primary | auxiliary | target_only)
    if not union:
        raise ValueError("Drug list is empty after eval union; cannot build final drug index.")
    if any(not d for d in union):
        raise ValueError("Empty drug_id found in response tables.")

    drug_to_index = {d: i for i, d in enumerate(union)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=union, drug_to_index=drug_to_index, index_to_drug=index_to_drug)


def build_drug_availability(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> pd.DataFrame:
    """Legacy availability report for source vs single target response."""
    src = _drug_set(source_response, drug_col)
    tgt = _drug_set(target_response, drug_col)
    union = sorted(src | tgt)

    rows = []
    for d in union:
        in_src = d in src
        in_tgt = d in tgt
        if in_src and in_tgt:
            category = "source_and_target"
        elif in_src:
            category = "source_only"
        else:
            category = "target_only"
        rows.append(
            {
                "drug_id": d,
                "in_source": in_src,
                "in_target": in_tgt,
                "category": category,
                "in_final_index": in_src,
            }
        )
    return pd.DataFrame(rows)


def build_drug_availability_from_eval_union(
    source_response: pd.DataFrame,
    primary_target_response: pd.DataFrame | None,
    auxiliary_target_response: pd.DataFrame | None,
    target_only_response: pd.DataFrame | None,
    drug_col: str,
    drug_index: DrugIndex,
    smiles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Report drug availability across source and all target eval response tables."""
    src = _drug_set(source_response, drug_col)
    in_primary = _drug_set(primary_target_response, drug_col)
    in_auxiliary = _drug_set(auxiliary_target_response, drug_col)
    in_target_only_eval = _drug_set(target_only_response, drug_col)
    in_any_target_eval = in_primary | in_auxiliary | in_target_only_eval

    smiles_keys: set[str] = set()
    if smiles_df is not None and not smiles_df.empty:
        cols_lower = [str(c).lower().strip() for c in smiles_df.columns]
        smiles_df = smiles_df.copy()
        smiles_df.columns = cols_lower
        first_col = cols_lower[0]
        name_col = None
        for possible in ("name", "drug_name", "drug_id"):
            if possible in cols_lower and possible != first_col:
                name_col = possible
                break
        for _, row in smiles_df.iterrows():
            smiles_keys.add(str(row[first_col]).strip().lower())
            if name_col:
                smiles_keys.add(str(row[name_col]).strip().lower())

    rows = []
    for d in drug_index.drug_ids:
        in_src = d in src
        in_p = d in in_primary
        in_a = d in in_auxiliary
        in_to = d in in_target_only_eval
        in_any = d in in_any_target_eval

        if in_src and in_any:
            category = "source_and_target_eval"
        elif in_src:
            category = "source_only"
        elif in_to and not in_p and not in_a:
            category = "target_only_eval_only"
        else:
            category = "target_eval_only"

        rows.append(
            {
                "drug_id": d,
                "drug_index": drug_index.drug_to_index[d],
                "in_source": in_src,
                "in_target_primary": in_p,
                "in_target_auxiliary": in_a,
                "in_target_only_eval": in_to,
                "in_any_target_eval": in_any,
                "category": category,
                "in_final_index": True,
                "has_smiles": d in smiles_keys if smiles_keys else False,
                "has_supervised_source_label": in_src,
                "is_target_eval_only": (not in_src) and in_any,
            }
        )
    return pd.DataFrame(rows)


def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    """Deprecated: kept for backward compatibility."""
    src = _drug_set(source_response, drug_col)
    tgt = _drug_set(target_response, drug_col)
    union = sorted(list(src | tgt))
    if not union:
        raise ValueError("Drug list is empty after taking union.")
    if any(not d for d in union):
        raise ValueError("Empty drug_id found in response tables.")

    drug_to_index = {d: i for i, d in enumerate(union)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=union, drug_to_index=drug_to_index, index_to_drug=index_to_drug)


def save_drug_list(drug_index: DrugIndex, path: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {"drug_id": list(drug_index.drug_ids), "drug_index": list(range(len(drug_index.drug_ids)))}
    )
    write_csv(df, path)
    return df


def load_drug_list(path: str) -> DrugIndex:
    df = pd.read_csv(path)
    drug_ids = [str(d).strip().lower() for d in df["drug_id"].tolist()]
    drug_to_index = {d: i for i, d in enumerate(drug_ids)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=drug_ids, drug_to_index=drug_to_index, index_to_drug=index_to_drug)
