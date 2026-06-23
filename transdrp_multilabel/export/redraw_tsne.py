"""Re-render DAPL-style dual-panel t-SNE from saved fold latent PKLs."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import pandas as pd

from transdrp_multilabel.data.cancer_type import load_and_align_cancer_types
from transdrp_multilabel.export.visualization import (
    TSNE_MAX_POINTS,
    plot_tsne_dual_from_latent_frames,
)


def _latent_dict_to_df(latent_dict: dict[str, list[float]], domain: str) -> pd.DataFrame:
    """Convert fold-level latent PKL (sample_id -> vector) to visualization table."""
    rows: list[dict] = []
    for sample_id, vector in latent_dict.items():
        row: dict = {"sample_id": sample_id, "domain": domain}
        for j, value in enumerate(vector):
            row[f"latent_{j}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_latent_pkl(path: Path) -> dict[str, list[float]]:
    with open(path, "rb") as handle:
        latent = pickle.load(handle)
    if not isinstance(latent, dict):
        raise TypeError(f"expected dict in {path}, got {type(latent).__name__}")
    return latent


def _discover_fold_dirs(output_dir: Path) -> list[tuple[int, Path]]:
    folds: list[tuple[int, Path]] = []
    for fold_dir in sorted(output_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        match = re.match(r"fold_(\d+)$", fold_dir.name)
        if not match:
            continue
        folds.append((int(match.group(1)), fold_dir))
    return folds


def _load_run_config(output_dir: Path) -> dict:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {output_dir}")
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def redraw_fold_tsne(
    fold_dir: Path,
    fold_id: int,
    cancer_type_table: pd.DataFrame,
    output_name: str = "tsne_latent_dual.png",
    max_points: int = TSNE_MAX_POINTS,
) -> Path | None:
    """Render one fold's dual-panel t-SNE PNG from saved latent PKLs."""
    src_path = fold_dir / "source_latent_representation.pkl"
    tgt_path = fold_dir / "target_latent_representation.pkl"
    if not src_path.is_file() or not tgt_path.is_file():
        return None

    source_latent = _latent_dict_to_df(_load_latent_pkl(src_path), domain="source")
    target_latent = _latent_dict_to_df(_load_latent_pkl(tgt_path), domain="target")
    save_path = fold_dir / output_name

    ok = plot_tsne_dual_from_latent_frames(
        source_latent,
        target_latent,
        cancer_type_table,
        save_path,
        suptitle=f"TransDRP Latent t-SNE (Fold {fold_id})",
        max_points=max_points,
    )
    return save_path if ok else None


def redraw_tsne_for_output_dir(
    output_dir: str | Path,
    max_points: int = TSNE_MAX_POINTS,
    output_name: str = "tsne_latent_dual.png",
) -> list[Path]:
    """Re-render t-SNE for every fold_* directory under an experiment output root."""
    out = Path(output_dir)
    config = _load_run_config(out)

    source_path = config.get("source_cancer_type_path")
    target_path = config.get("target_cancer_type_path")
    cancer_type_col = config.get("cancer_type_col")
    if not source_path or not target_path:
        raise ValueError("config.json must include source_cancer_type_path and target_cancer_type_path")

    written: list[Path] = []
    for fold_id, fold_dir in _discover_fold_dirs(out):
        src_path = fold_dir / "source_latent_representation.pkl"
        tgt_path = fold_dir / "target_latent_representation.pkl"
        if not src_path.is_file() or not tgt_path.is_file():
            continue

        source_ids = list(_load_latent_pkl(src_path).keys())
        target_ids = list(_load_latent_pkl(tgt_path).keys())
        cancer_type_table = load_and_align_cancer_types(
            source_ids,
            target_ids,
            source_path,
            target_path,
            cancer_type_col=cancer_type_col,
        )

        save_path = redraw_fold_tsne(
            fold_dir,
            fold_id,
            cancer_type_table,
            output_name=output_name,
            max_points=max_points,
        )
        if save_path is not None:
            written.append(save_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render DAPL-style dual-panel t-SNE from saved fold latent PKLs.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="outputs_transdrp_eval3",
        help="Experiment output directory containing fold_* and config.json",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=TSNE_MAX_POINTS,
        help=f"Subsample cap before t-SNE (default: {TSNE_MAX_POINTS})",
    )
    parser.add_argument(
        "--output-name",
        default="tsne_latent_dual.png",
        help="PNG filename written inside each fold directory",
    )
    args = parser.parse_args()

    written = redraw_tsne_for_output_dir(
        args.output_dir,
        max_points=args.max_points,
        output_name=args.output_name,
    )
    if not written:
        raise SystemExit(f"No t-SNE plots were written under {args.output_dir}")
    for path in written:
        print(f"[redraw_tsne] wrote {path}")


if __name__ == "__main__":
    main()
