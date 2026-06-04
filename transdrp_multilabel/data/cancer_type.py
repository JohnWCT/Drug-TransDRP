from __future__ import annotations
import pandas as pd
from transdrp_multilabel.data.sample_id import tcga_patient_key
from transdrp_multilabel.io import read_csv

UNKNOWN = "Unknown"

def load_cancer_type_mapping(path: str, sample_col: str = None, cancer_type_col: str = None) -> dict[str, str]:
    df = read_csv(path)
    lookup = {str(c).lower(): c for c in df.columns}

    # Auto-detect sample/patient column
    sid_c = None
    if sample_col and sample_col in df.columns:
        sid_c = sample_col
    else:
        for possible in [sample_col, "sample_id", "patient_id", "tissue_id", "unnamed: 0"]:
            if possible and possible.lower() in lookup:
                sid_c = lookup[possible.lower()]
                break
        if sid_c is None:
            sid_c = df.columns[0]  # default to first column

    # Auto-detect cancer/tissue type column
    ct_c = None
    if cancer_type_col and cancer_type_col in df.columns:
        ct_c = cancer_type_col
    else:
        for possible in [cancer_type_col, "cancer_type", "primary_disease", "tissue", "cancer", "_primary_disease"]:
            if possible and possible.lower() in lookup:
                ct_c = lookup[possible.lower()]
                break
        if ct_c is None:
            # Fallback to look for primary_disease or tissue
            for col in df.columns:
                if "disease" in col.lower() or "tissue" in col.lower() or "cancer" in col.lower():
                    ct_c = col
                    break
        if ct_c is None:
            ct_c = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sid = str(row[sid_c]).strip()
        ct = str(row[ct_c]).strip()
        if not sid or pd.isna(row[ct_c]):
            continue
        # Normalize target patient key if it looks like TCGA
        key = tcga_patient_key(sid) if sid.startswith("TCGA-") else sid
        out[key] = ct
    return out

def load_and_align_cancer_types(
    source_ids: list[str],
    target_ids: list[str],
    source_path: str | None,
    target_path: str | None,
    cancer_type_col: str | None = None,
) -> pd.DataFrame:
    # Cancer type maps are mandatory (req.10): both paths must be provided and
    # every source/target sample must resolve to a cancer type. Missing paths or
    # unmapped samples raise immediately instead of silently using "Unknown".
    if not source_path:
        raise ValueError("source_cancer_type_path is required but was not provided.")
    if not target_path:
        raise ValueError("target_cancer_type_path is required but was not provided.")

    source_map = load_cancer_type_mapping(source_path, cancer_type_col=cancer_type_col)
    target_map = load_cancer_type_mapping(target_path, cancer_type_col=cancer_type_col)

    rows = []
    missing_source: list[str] = []
    for sid in source_ids:
        ct = source_map.get(sid)
        if ct is None or str(ct).strip() == "" or ct == UNKNOWN:
            missing_source.append(sid)
            ct = UNKNOWN
        rows.append({"sample_id": sid, "domain": "source", "cancer_type": ct})

    missing_target: list[str] = []
    for sid in target_ids:
        key = tcga_patient_key(sid) if sid.startswith("TCGA-") else sid
        ct = target_map.get(key, target_map.get(sid))
        if ct is None or str(ct).strip() == "" or ct == UNKNOWN:
            missing_target.append(sid)
            ct = UNKNOWN
        rows.append({"sample_id": sid, "domain": "target", "cancer_type": ct})

    if missing_source or missing_target:
        def _fmt(ids: list[str]) -> str:
            shown = ids[:10]
            extra = f" ... (+{len(ids) - 10} more)" if len(ids) > 10 else ""
            return ", ".join(shown) + extra

        msg = ["Cancer type mapping is incomplete (every sample must have a cancer type)."]
        if missing_source:
            msg.append(f"- {len(missing_source)} source samples missing in '{source_path}': {_fmt(missing_source)}")
        if missing_target:
            msg.append(f"- {len(missing_target)} target samples missing in '{target_path}': {_fmt(missing_target)}")
        raise ValueError("\n".join(msg))

    return pd.DataFrame(rows)
