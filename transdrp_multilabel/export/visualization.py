from __future__ import annotations
import warnings
import logging
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def run_tsne(latent_df: pd.DataFrame, seed: int) -> pd.DataFrame | None:
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    if len(latent_df) < 5 or not latent_cols:
        logger.warning("skipping t-SNE: insufficient samples or latent columns")
        return None
    x = latent_df[latent_cols].values
    if not pd.DataFrame(x).apply(pd.to_numeric).notna().all().all():
        logger.warning("skipping t-SNE: non-finite latent values")
        return None

    perplexity = min(30, max(2, len(latent_df) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
        emb = tsne.fit_transform(x)

    out = latent_df[["sample_id", "domain"]].copy()
    out["tsne_1"] = emb[:, 0]
    out["tsne_2"] = emb[:, 1]
    return out

def plot_tsne_by_domain(tsne_df: pd.DataFrame, output_path: str) -> bool:
    if tsne_df is None or tsne_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, g in tsne_df.groupby("domain"):
        ax.scatter(g["tsne_1"], g["tsne_2"], label=domain, alpha=0.7, s=20)
    ax.legend()
    ax.set_title("t-SNE by domain")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_tsne_by_cancer_type(
    tsne_df: pd.DataFrame, cancer_type_df: pd.DataFrame, output_path: str
) -> bool:
    if tsne_df is None or tsne_df.empty:
        return False
    # cancer_type_df should have 'sample_id' and 'cancer_type' columns
    merged = tsne_df.merge(cancer_type_df, on="sample_id", how="left")
    if "cancer_type" not in merged.columns:
        logger.warning("skipping cancer-type t-SNE: missing cancer_type column")
        return False
    if merged["cancer_type"].astype(str).eq("Unknown").all():
        logger.warning("skipping cancer-type t-SNE: all Unknown")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = merged["cancer_type"].astype(str)
    for ct in labels.unique():
        mask = labels == ct
        ax.scatter(
            merged.loc[mask, "tsne_1"], merged.loc[mask, "tsne_2"], label=ct, alpha=0.6, s=15
        )
    ax.legend(fontsize=6, loc="best")
    ax.set_title("t-SNE by cancer type")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True
