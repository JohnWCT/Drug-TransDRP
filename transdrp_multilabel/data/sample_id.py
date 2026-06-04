import re

_TCGA_PREFIX = re.compile(r"^TCGA-", re.IGNORECASE)

def tcga_segment_count(sample_id: str) -> int:
    return len(str(sample_id).strip().split("-"))

def tcga_patient_key(sample_id: str) -> str:
    sid = str(sample_id).strip()
    parts = sid.split("-")
    if _TCGA_PREFIX.match(sid) and len(parts) >= 3:
        return "-".join(parts[:3])
    return sid

def normalize_source_sample_id(sample_id: str) -> str:
    return str(sample_id).strip()

def normalize_target_omics_sample_id(sample_id: str) -> str:
    return str(sample_id).strip()

def normalize_target_response_sample_id(sample_id: str) -> str:
    return tcga_patient_key(sample_id)

def build_target_omics_response_join_key(sample_id: str) -> str:
    return tcga_patient_key(sample_id)

def sample_match_key(sample_id: str, column_hint: str = None) -> str:
    sid = str(sample_id).strip()
    hint = (column_hint or "").lower()
    if _TCGA_PREFIX.match(sid):
        return tcga_patient_key(sid)
    if hint in {"tissue_id", "barcode_id", "patient_id", "sample_id"} and tcga_segment_count(sid) >= 3:
        return tcga_patient_key(sid)
    return sid
