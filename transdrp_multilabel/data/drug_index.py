import pandas as pd
from transdrp_multilabel.contracts import DrugIndex
from transdrp_multilabel.io import write_csv

def _normalize_drug(s: str) -> str:
    return str(s).strip().lower()

def _drug_set(response: pd.DataFrame, drug_col: str) -> set[str]:
    return {_normalize_drug(d) for d in response[drug_col].astype(str) if str(d).strip()}

def build_drug_index_from_source(
    source_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    """Final drug index = source drugs only.

    Per the final modification decision (sections 2/13/15), the model's drug
    index is built strictly from source drugs. target-only drugs are dropped
    (no SMILES requirement, no evaluation). source-only drugs are kept so the
    source training/validation task stays complete; they will simply carry an
    all-zero target mask later.
    """
    src = _drug_set(source_response, drug_col)
    drug_ids = sorted(src)
    if not drug_ids:
        raise ValueError("Source drug list is empty; cannot build final drug index.")
    if any(not d for d in drug_ids):
        raise ValueError("Empty drug_id found in source response table.")

    drug_to_index = {d: i for i, d in enumerate(drug_ids)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=drug_ids, drug_to_index=drug_to_index, index_to_drug=index_to_drug)

def build_drug_availability(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> pd.DataFrame:
    """Categorize every drug seen in either domain into the availability report.

    category:
        source_and_target : drug present in both source and target responses
        source_only        : drug present only in source (kept in final index)
        target_only        : drug present only in target (dropped from final index)
    in_final_index == in_source, since the final drug index is source-only.
    """
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

def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    """Deprecated: kept for backward compatibility. The final pipeline uses
    build_drug_index_from_source (source-only) instead of the source∪target union."""
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
