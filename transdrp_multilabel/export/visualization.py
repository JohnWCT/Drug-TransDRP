"""DAPL-style dual-panel latent t-SNE visualization."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

TSNE_RANDOM_STATE = 42
TSNE_MAX_POINTS = 3000
SOURCE_COLOR = "#1f77b4"
TARGET_COLOR = "#ff7f0e"
DEFAULT_SUPTITLE = "Latent t-SNE (Source / Target)"


def _sanitize_latent(z: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _fit_tsne(all_feats: np.ndarray) -> np.ndarray:
    n = len(all_feats)
    if n < 2:
        raise ValueError("t-SNE requires at least 2 samples")
    perplexity = float(min(30, max(2, n - 1)))
    kwargs = dict(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        perplexity=perplexity,
        init="random",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tsne = TSNE(**kwargs, learning_rate="auto")
        except TypeError:
            tsne = TSNE(**kwargs, learning_rate=200.0)
        return tsne.fit_transform(all_feats)


def plot_latent_tsne_dual(
    source_z: np.ndarray,
    target_z: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    mapping_int2str: dict[int, str],
    save_path: str | Path,
    suptitle: str = DEFAULT_SUPTITLE,
    max_points: int = TSNE_MAX_POINTS,
) -> bool:
    """Render 1x2 t-SNE figure (domain + cancer type) per DAPL pretrain_tsne spec."""
    source_z = _sanitize_latent(source_z)
    target_z = _sanitize_latent(target_z)
    if source_z.size == 0 and target_z.size == 0:
        return False
    if source_z.ndim == 1:
        source_z = source_z.reshape(1, -1)
    if target_z.ndim == 1:
        target_z = target_z.reshape(1, -1)
    if len(source_z) == 0:
        source_z = np.empty((0, target_z.shape[1]), dtype=np.float64)
    if len(target_z) == 0:
        target_z = np.empty((0, source_z.shape[1]), dtype=np.float64)

    source_labels = np.asarray(source_labels, dtype=int).reshape(-1)
    target_labels = np.asarray(target_labels, dtype=int).reshape(-1)

    all_feats = np.vstack([source_z, target_z])
    all_labels = np.concatenate([source_labels, target_labels])
    n_source = len(source_z)
    domain_flags = np.array(["source"] * n_source + ["target"] * len(target_z))

    if len(all_feats) > max_points:
        rng = np.random.default_rng(TSNE_RANDOM_STATE)
        idx = rng.choice(len(all_feats), max_points, replace=False)
        all_feats = all_feats[idx]
        all_labels = all_labels[idx]
        domain_flags = domain_flags[idx]
        n_source = int((domain_flags == "source").sum())

    try:
        emb = _fit_tsne(all_feats)
    except ValueError as exc:
        logger.warning("skipping t-SNE: %s", exc)
        return False

    emb_s = emb[domain_flags == "source"]
    emb_t = emb[domain_flags == "target"]
    lab_s = all_labels[domain_flags == "source"]
    lab_t = all_labels[domain_flags == "target"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))

    ax_a.scatter(
        emb_s[:, 0],
        emb_s[:, 1],
        c=SOURCE_COLOR,
        s=14,
        alpha=0.85,
        marker="o",
        edgecolors="k",
        linewidths=0.3,
        label="Source (CCLE)",
    )
    ax_a.scatter(
        emb_t[:, 0],
        emb_t[:, 1],
        c=TARGET_COLOR,
        s=12,
        alpha=0.55,
        marker="^",
        edgecolors="k",
        linewidths=0.3,
        label="Target (TCGA)",
    )
    ax_a.set_title("A. t-SNE by Domain (Source / Target)")
    ax_a.set_xlabel("Dimension 1")
    ax_a.set_ylabel("Dimension 2")
    ax_a.legend(loc="best", fontsize=8)
    ax_a.grid(alpha=0.2)

    unique = np.unique(all_labels)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(20, len(unique)))
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", max(20, len(unique)))
    colors = {int(lab): cmap(i % cmap.N) for i, lab in enumerate(unique)}

    for lab in np.unique(lab_s):
        idx = lab_s == lab
        ax_b.scatter(
            emb_s[idx, 0],
            emb_s[idx, 1],
            c=[colors[int(lab)]],
            s=14,
            alpha=0.85,
            marker="o",
            edgecolors="k",
            linewidths=0.3,
        )
    for lab in np.unique(lab_t):
        idx = lab_t == lab
        ax_b.scatter(
            emb_t[idx, 0],
            emb_t[idx, 1],
            c=[colors[int(lab)]],
            s=12,
            alpha=0.85,
            marker="^",
            edgecolors="k",
            linewidths=0.3,
        )
    ax_b.set_title("B. t-SNE by Cancer Type")
    ax_b.set_xlabel("Dimension 1")
    ax_b.set_ylabel("Dimension 2")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[int(lab)],
            markersize=6,
            label=mapping_int2str.get(int(lab), str(lab)),
        )
        for lab in unique
    ]
    ax_b.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    ax_b.grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return True


def _latent_matrix(df: pd.DataFrame) -> np.ndarray:
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("missing latent columns")
    return _sanitize_latent(df[latent_cols].values)


def _encode_cancer_labels(labels: list[str]) -> tuple[np.ndarray, dict[int, str]]:
    cleaned = [
        str(x).strip() if str(x).strip() and str(x).lower() != "nan" else "Unknown"
        for x in labels
    ]
    unique = sorted(set(cleaned))
    str_to_int = {name: i for i, name in enumerate(unique)}
    mapping = {i: name for name, i in str_to_int.items()}
    encoded = np.array([str_to_int[x] for x in cleaned], dtype=int)
    return encoded, mapping


def plot_tsne_dual_from_latent_frames(
    source_latent_df: pd.DataFrame,
    target_latent_df: pd.DataFrame,
    cancer_type_df: pd.DataFrame,
    save_path: str | Path,
    suptitle: str = DEFAULT_SUPTITLE,
    max_points: int = TSNE_MAX_POINTS,
) -> bool:
    """TransDRP adapter: latent tables + cancer_type_table."""
    if source_latent_df.empty and target_latent_df.empty:
        return False
    if len(source_latent_df) + len(target_latent_df) < 2:
        logger.warning("skipping t-SNE: insufficient samples")
        return False

    source_z = _latent_matrix(source_latent_df) if len(source_latent_df) else np.empty((0, 0))
    target_z = _latent_matrix(target_latent_df) if len(target_latent_df) else np.empty((0, 0))
    if source_z.size == 0:
        source_z = np.empty((0, target_z.shape[1]), dtype=np.float64)
    if target_z.size == 0:
        target_z = np.empty((0, source_z.shape[1]), dtype=np.float64)

    ct_map = dict(
        zip(
            cancer_type_df["sample_id"].astype(str),
            cancer_type_df["cancer_type"].astype(str),
        )
    )
    source_ids = source_latent_df["sample_id"].astype(str).tolist()
    target_ids = target_latent_df["sample_id"].astype(str).tolist()
    source_ct = [ct_map.get(sid, "Unknown") for sid in source_ids]
    target_ct = [ct_map.get(sid, "Unknown") for sid in target_ids]

    all_ct = source_ct + target_ct
    _, mapping = _encode_cancer_labels(all_ct)
    str_to_int = {v: k for k, v in mapping.items()}
    source_labels = np.array([str_to_int[x] for x in source_ct], dtype=int)
    target_labels = np.array([str_to_int[x] for x in target_ct], dtype=int)

    return plot_latent_tsne_dual(
        source_z,
        target_z,
        source_labels,
        target_labels,
        mapping,
        save_path,
        suptitle=suptitle,
        max_points=max_points,
    )


# Backward-compatible aliases (deprecated separate plots → dual panel)
def run_tsne(latent_df: pd.DataFrame, seed: int) -> pd.DataFrame | None:
    del seed
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    if len(latent_df) < 2 or not latent_cols:
        return None
    try:
        emb = _fit_tsne(_latent_matrix(latent_df))
    except ValueError:
        return None
    out = latent_df[["sample_id", "domain"]].copy()
    out["tsne_1"] = emb[:, 0]
    out["tsne_2"] = emb[:, 1]
    return out


def plot_tsne_by_domain(tsne_df: pd.DataFrame, output_path: str) -> bool:
    del tsne_df, output_path
    logger.warning("plot_tsne_by_domain is deprecated; use plot_tsne_dual_from_latent_frames")
    return False


def plot_tsne_by_cancer_type(
    tsne_df: pd.DataFrame, cancer_type_df: pd.DataFrame, output_path: str
) -> bool:
    del tsne_df, cancer_type_df, output_path
    logger.warning("plot_tsne_by_cancer_type is deprecated; use plot_tsne_dual_from_latent_frames")
    return False
