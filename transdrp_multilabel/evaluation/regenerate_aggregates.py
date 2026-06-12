"""Regenerate cross-fold aggregate CSVs from existing fold_* outputs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from transdrp_multilabel.evaluation.reports import (
    aggregate_per_drug_metrics,
    aggregate_per_drug_metrics_by_dataset,
    aggregate_summary_metrics,
    aggregate_target_eval_metrics_by_dataset,
    build_combined_eval_summary,
)
from transdrp_multilabel.io import write_csv

_EVAL_DATASETS = ("primary", "auxiliary", "target_only")


def _read_fold_csvs(output_dir: Path, pattern: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for fold_dir in sorted(output_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        m = re.match(r"fold_(\d+)$", fold_dir.name)
        if not m:
            continue
        fold_id = int(m.group(1))
        path = fold_dir / pattern
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["fold"] = fold_id
        frames.append(df)
    return frames


def _attach_drug_flags(per_drug_std: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    if per_drug_std.empty:
        return per_drug_std
    out = per_drug_std.copy()
    for col in ("has_supervised_source_label", "is_target_eval_only"):
        if col in combined.columns:
            flags = combined.groupby("drug_id")[col].first()
            out[col] = out["drug_id"].map(flags)
    return out


def regenerate_cross_fold_reports(output_dir: str | Path) -> None:
    out = Path(output_dir)

    src_per = _read_fold_csvs(out, "source_test_metrics_per_drug.csv")
    src_sum = _read_fold_csvs(out, "source_test_metrics_summary.csv")
    if src_per:
        write_csv(
            pd.concat(src_per, ignore_index=True),
            out / "source_test_metrics_per_drug_across_folds.csv",
        )
        write_csv(aggregate_per_drug_metrics(src_per), out / "source_test_metrics_per_drug_fold_mean_std.csv")
    if src_sum:
        write_csv(
            pd.concat(src_sum, ignore_index=True),
            out / "source_test_metrics_summary_across_folds.csv",
        )
        write_csv(aggregate_summary_metrics(src_sum), out / "source_test_metrics_summary_fold_mean_std.csv")

    all_tgt_per_across: list[pd.DataFrame] = []
    all_tgt_sum_across: list[pd.DataFrame] = []
    all_tgt_per_fold_std: list[pd.DataFrame] = []
    all_tgt_sum_fold_std: list[pd.DataFrame] = []
    primary_per: list[pd.DataFrame] = []
    primary_sum: list[pd.DataFrame] = []

    for eval_name in _EVAL_DATASETS:
        per_frames = _read_fold_csvs(out, f"target_{eval_name}_metrics_per_drug.csv")
        sum_frames = _read_fold_csvs(out, f"target_{eval_name}_metrics_summary.csv")
        for df in per_frames:
            df["eval_dataset"] = eval_name
        for df in sum_frames:
            df["eval_dataset"] = eval_name

        if per_frames:
            combined = pd.concat(per_frames, ignore_index=True)
            write_csv(combined, out / f"target_{eval_name}_metrics_per_drug_across_folds.csv")
            per_std = _attach_drug_flags(aggregate_per_drug_metrics(per_frames), combined)
            write_csv(per_std, out / f"target_{eval_name}_metrics_per_drug_fold_mean_std.csv")
            all_tgt_per_across.append(combined)
            ds_per_std = aggregate_per_drug_metrics_by_dataset(per_frames)
            if not ds_per_std.empty:
                all_tgt_per_fold_std.append(ds_per_std)
            if eval_name == "primary":
                primary_per = per_frames

        if sum_frames:
            combined = pd.concat(sum_frames, ignore_index=True)
            write_csv(combined, out / f"target_{eval_name}_metrics_summary_across_folds.csv")
            write_csv(
                aggregate_summary_metrics(sum_frames),
                out / f"target_{eval_name}_metrics_summary_fold_mean_std.csv",
            )
            all_tgt_sum_across.append(combined)
            ds_sum_std = aggregate_target_eval_metrics_by_dataset(sum_frames)
            if not ds_sum_std.empty:
                all_tgt_sum_fold_std.append(ds_sum_std)
            if eval_name == "primary":
                primary_sum = sum_frames

    if primary_per:
        write_csv(
            pd.concat(primary_per, ignore_index=True),
            out / "target_eval_metrics_per_drug_across_folds.csv",
        )
        write_csv(
            aggregate_per_drug_metrics(primary_per),
            out / "target_eval_metrics_per_drug_fold_mean_std.csv",
        )
    if primary_sum:
        write_csv(
            pd.concat(primary_sum, ignore_index=True),
            out / "target_eval_metrics_summary_across_folds.csv",
        )
        write_csv(
            aggregate_summary_metrics(primary_sum),
            out / "target_eval_metrics_summary_fold_mean_std.csv",
        )

    if all_tgt_per_across:
        write_csv(
            pd.concat(all_tgt_per_across, ignore_index=True),
            out / "target_eval_metrics_per_drug_by_dataset_across_folds.csv",
        )
    if all_tgt_per_fold_std:
        write_csv(
            pd.concat(all_tgt_per_fold_std, ignore_index=True),
            out / "target_eval_metrics_per_drug_by_dataset_fold_mean_std.csv",
        )
    if all_tgt_sum_across:
        write_csv(
            pd.concat(all_tgt_sum_across, ignore_index=True),
            out / "target_eval_metrics_summary_by_dataset_across_folds.csv",
        )
    if all_tgt_sum_fold_std:
        write_csv(
            pd.concat(all_tgt_sum_fold_std, ignore_index=True),
            out / "target_eval_metrics_summary_by_dataset_fold_mean_std.csv",
        )

    if src_sum or primary_sum:
        write_csv(
            build_combined_eval_summary(src_sum, primary_sum),
            out / "eval_metrics_summary_fold_mean_std.csv",
        )
