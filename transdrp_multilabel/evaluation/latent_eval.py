"""Latent distribution metrics and K-Means clustering metrics."""

import contextlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance

UNKNOWN_LABEL = "Unknown"

def _to_matrix(latent_dict: Dict[str, List[float]], sample_ids: List[str]) -> np.ndarray:
    return np.asarray([latent_dict[sid] for sid in sample_ids], dtype=np.float64)

def calculate_fid(source: np.ndarray, target: np.ndarray) -> float:
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    diff = mu_s - mu_t
    cov_s = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    if cov_s.ndim == 0:
        cov_s = np.array([[cov_s]])
    cov_t = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6
    if cov_t.ndim == 0:
        cov_t = np.array([[cov_t]])

    covmean = sqrtm(cov_s @ cov_t)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(cov_s + cov_t - 2 * covmean))
    return fid

def calculate_mmd(source: np.ndarray, target: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        gamma = 1.0 / source.shape[1]
    n = min(500, source.shape[0], target.shape[0])
    rng = np.random.default_rng(0)
    xs = source[rng.choice(source.shape[0], n, replace=False)]
    xt = target[rng.choice(target.shape[0], n, replace=False)]

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xx = np.sum(x * x, axis=1, keepdims=True)
        yy = np.sum(y * y, axis=1, keepdims=True)
        xy = x @ y.T
        return np.exp(-gamma * (xx - 2 * xy + yy.T))

    k_ss = kernel(xs, xs).mean()
    k_tt = kernel(xt, xt).mean()
    k_st = kernel(xs, xt).mean()
    return float(k_ss + k_tt - 2 * k_st)

def calculate_wasserstein(source: np.ndarray, target: np.ndarray) -> float:
    dists = [wasserstein_distance(source[:, j], target[:, j]) for j in range(source.shape[1])]
    return float(np.mean(dists))

def compute_distribution_metrics(
    source_latent: Dict[str, List[float]],
    target_latent: Dict[str, List[float]],
) -> Dict[str, float]:
    src_ids = sorted(source_latent.keys())
    tgt_ids = sorted(target_latent.keys())

    if not src_ids or not tgt_ids:
        return {
            "source_n": float(len(src_ids)),
            "target_n": float(len(tgt_ids)),
            "fid_source_target": float("nan"),
            "mmd_source_target": float("nan"),
            "wasserstein_source_target": float("nan"),
        }

    src = _to_matrix(source_latent, src_ids)
    tgt = _to_matrix(target_latent, tgt_ids)
    return {
        "source_n": float(len(src_ids)),
        "target_n": float(len(tgt_ids)),
        "fid_source_target": calculate_fid(src, tgt),
        "mmd_source_target": calculate_mmd(src, tgt),
        "wasserstein_source_target": calculate_wasserstein(src, tgt),
    }

def compute_kmeans_cancer_type_metrics(
    combined_latent: Dict[str, List[float]],
    cancer_map: Dict[str, str],
    random_state: int,
) -> Dict[str, float]:
    ids = []
    labels = []
    for sid, ct in cancer_map.items():
        if sid not in combined_latent:
            continue
        if ct == UNKNOWN_LABEL:
            continue
        ids.append(sid)
        labels.append(ct)

    if len(ids) < 2:
        return {
            "k_eff": float("nan"),
            "samples_used": float(len(ids)),
            "ari": float("nan"),
            "nmi": float("nan"),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_cancer_types": 0.0,
        }

    x = _to_matrix(combined_latent, ids)
    unique_labels = sorted(set(labels))
    k = len(unique_labels)
    k_eff = int(max(2, min(k, len(ids) - 1)))

    pred = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10).fit_predict(x)
    y_true = np.asarray(labels)

    metrics: Dict[str, float] = {
        "k_eff": float(k_eff),
        "samples_used": float(len(ids)),
        "ari": float(adjusted_rand_score(y_true, pred)),
        "nmi": float(normalized_mutual_info_score(y_true, pred)),
        "n_cancer_types": float(k),
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
    }

    with contextlib.suppress(Exception):
        metrics["silhouette"] = float(silhouette_score(x, pred))
    with contextlib.suppress(Exception):
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, pred))
    with contextlib.suppress(Exception):
        metrics["davies_bouldin"] = float(davies_bouldin_score(x, pred))

    return metrics
