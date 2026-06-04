from typing import Literal
import pandas as pd
from transdrp_multilabel.contracts import OmicsTable
from transdrp_multilabel.data.sample_id import (
    normalize_source_sample_id,
    normalize_target_omics_sample_id,
)
from transdrp_multilabel.io import read_csv
from transdrp_multilabel.validators import validate_omics_table

def read_omics_table(
    path: str,
    sample_id_col: str,
    domain: Literal["source", "target"],
) -> OmicsTable:
    df = read_csv(path)
    validate_omics_table(df, sample_id_col)
    norm_fn = normalize_source_sample_id if domain == "source" else normalize_target_omics_sample_id
    sids = [norm_fn(s) for s in df[sample_id_col].astype(str)]
    feat_df = df.drop(columns=[sample_id_col])
    numeric = feat_df.select_dtypes(include="number")
    x = numeric.copy()
    x.index = sids
    return OmicsTable(
        x=x,
        sample_ids=sids,
        feature_names=list(x.columns.astype(str)),
        domain=domain,
    )

def align_omics_features(
    source: OmicsTable,
    target: OmicsTable,
) -> tuple[OmicsTable, OmicsTable, pd.DataFrame]:
    common = sorted(list(set(source.feature_names) & set(target.feature_names)))
    if not common:
        raise ValueError("No overlapping features (genes) between source and target omics.")
    src_x = source.x.loc[list(source.sample_ids), common]
    tgt_x = target.x.loc[list(target.sample_ids), common]

    report_rows = []
    all_feats = sorted(list(set(source.feature_names) | set(target.feature_names)))
    src_set, tgt_set = set(source.feature_names), set(target.feature_names)
    for f in all_feats:
        report_rows.append(
            {
                "feature_name": f,
                "in_source": f in src_set,
                "in_target": f in tgt_set,
                "used": f in common,
            }
        )
    report = pd.DataFrame(report_rows)

    src_aligned = OmicsTable(
        x=src_x.copy(),
        sample_ids=source.sample_ids,
        feature_names=common,
        domain="source",
    )
    tgt_aligned = OmicsTable(
        x=tgt_x.copy(),
        sample_ids=target.sample_ids,
        feature_names=common,
        domain="target",
    )
    src_aligned.x.index = list(source.sample_ids)
    tgt_aligned.x.index = list(target.sample_ids)

    return src_aligned, tgt_aligned, report
