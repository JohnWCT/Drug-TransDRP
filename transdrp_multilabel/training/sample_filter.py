"""Runner-layer sample filtering (cancer type) before data preparation."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import pandas as pd

from transdrp_multilabel.contracts import TransDRPMultilabelConfig
from transdrp_multilabel.data.cancer_type import load_cancer_type_mapping
from transdrp_multilabel.data.sample_id import tcga_patient_key
from transdrp_multilabel.io import read_csv, write_csv

_INVALID_CANCER_VALUES = frozenset({"", "unknown", "nan", "none"})


def _is_invalid_cancer_type(ct: object) -> bool:
    if ct is None or (isinstance(ct, float) and pd.isna(ct)):
        return True
    s = str(ct).strip()
    if not s:
        return True
    return s.lower() in _INVALID_CANCER_VALUES


def _lookup_cancer_type(sample_id: str, cancer_map: dict[str, str], domain: str) -> Optional[str]:
    sid = str(sample_id).strip()
    if domain == "target" and sid.startswith("TCGA-"):
        key = tcga_patient_key(sid)
        ct = cancer_map.get(key, cancer_map.get(sid))
    else:
        ct = cancer_map.get(sid)
    if _is_invalid_cancer_type(ct):
        return None
    return str(ct).strip()


def filter_config_by_cancer_type(
    config: TransDRPMultilabelConfig,
) -> tuple[TransDRPMultilabelConfig, pd.DataFrame]:
    """Remove samples with missing / empty / Unknown cancer types from omics inputs.

    Writes filtered omics CSVs under ``{output_dir}/.filtered_inputs/`` and returns
    a config pointing at those files plus a sample-filtering report.
    """
    if not config.source_cancer_type_path or not config.target_cancer_type_path:
        raise ValueError("Both source_cancer_type_path and target_cancer_type_path are required.")

    source_map = load_cancer_type_mapping(
        config.source_cancer_type_path, cancer_type_col=config.cancer_type_col
    )
    target_map = load_cancer_type_mapping(
        config.target_cancer_type_path, cancer_type_col=config.cancer_type_col
    )

    filter_dir = os.path.join(config.output_dir, ".filtered_inputs")
    os.makedirs(filter_dir, exist_ok=True)

    report_rows: list[dict] = []

    def _filter_omics(
        omics_path: str,
        sample_col: str,
        domain: str,
        cancer_map: dict[str, str],
        out_name: str,
    ) -> str:
        df = read_csv(omics_path)
        kept_rows = []
        for _, row in df.iterrows():
            sid = str(row[sample_col]).strip()
            ct = _lookup_cancer_type(sid, cancer_map, domain)
            if ct is None:
                raw = cancer_map.get(sid)
                if domain == "target" and sid.startswith("TCGA-"):
                    raw = cancer_map.get(tcga_patient_key(sid), cancer_map.get(sid))
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    reason = "missing cancer type"
                elif str(raw).strip().lower() == "unknown":
                    reason = "Unknown"
                elif not str(raw).strip():
                    reason = "empty cancer type"
                else:
                    reason = "invalid cancer type"
                report_rows.append(
                    {
                        "domain": domain,
                        "sample_id": sid,
                        "reason": reason,
                        "kept": False,
                        "stage": "fine-tune-filter",
                    }
                )
            else:
                kept_rows.append(row)
                report_rows.append(
                    {
                        "domain": domain,
                        "sample_id": sid,
                        "reason": "",
                        "kept": True,
                        "stage": "fine-tune-filter",
                    }
                )

        if not kept_rows:
            raise ValueError(
                f"All {domain} samples were removed by cancer-type filtering; "
                f"check {config.source_cancer_type_path if domain == 'source' else config.target_cancer_type_path}."
            )

        filtered = pd.DataFrame(kept_rows)
        out_path = os.path.join(filter_dir, out_name)
        write_csv(filtered, out_path)
        return out_path

    src_out = _filter_omics(
        config.source_omics_path,
        config.source_sample_col,
        "source",
        source_map,
        "source_omics_filtered.csv",
    )
    tgt_out = _filter_omics(
        config.target_omics_path,
        config.target_sample_col,
        "target",
        target_map,
        "target_omics_filtered.csv",
    )

    filtered_config = replace(
        config,
        source_omics_path=src_out,
        target_omics_path=tgt_out,
    )
    return filtered_config, pd.DataFrame(report_rows)
