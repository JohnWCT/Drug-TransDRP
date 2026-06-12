"""Hybrid drug graph: source co-occurrence + molecular similarity edges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from transdrp_multilabel.contracts import DrugIndex


def build_source_cooccurrence_adjacency(
    y_source: np.ndarray,
    mask_source: np.ndarray,
    task_type: str,
    reg_binary_threshold: float,
    threshold_label: float = 0.1,
) -> np.ndarray:
    """Build binary co-occurrence adjacency from SOURCE labels only."""
    n_drugs = y_source.shape[1]

    if task_type == "classification":
        labels = y_source.copy()
        labels[mask_source == 0] = 0
    else:
        labels = (y_source > reg_binary_threshold).astype(float)
        labels[mask_source == 0] = 0.0

    label_graph = np.eye(n_drugs, dtype=float)
    for idx in range(n_drugs):
        for j in range(n_drugs):
            if idx == j:
                continue
            overlap = np.sum((labels[:, idx] == 1.0) & (labels[:, j] == 1.0))
            label_graph[idx, j] = overlap

    row, col = np.diag_indices_from(label_graph)
    label_graph[row, col] = np.sum(labels, axis=0)

    for col_idx in range(n_drugs):
        normalizer = np.sum(label_graph[:, col_idx])
        if normalizer > 0:
            label_graph[:, col_idx] = label_graph[:, col_idx] / normalizer

    adj = label_graph - np.diag(np.diag(label_graph))
    adj = (adj >= threshold_label).astype(int)
    return adj


def _load_smiles_lookup(smiles_path: str) -> dict[str, str]:
    smiles_df = pd.read_csv(smiles_path, header=0)
    cols_lower = [str(c).lower().strip() for c in smiles_df.columns]
    smiles_df.columns = cols_lower

    smiles_col = None
    for possible in ("isosmiles", "smiles"):
        if possible in cols_lower:
            smiles_col = possible
            break
    if smiles_col is None:
        raise KeyError(f"Could not find smiles or isosmiles column in {smiles_path}")

    first_col = cols_lower[0]
    name_col = None
    for possible in ("name", "drug_name", "drug_id"):
        if possible in cols_lower and possible != first_col:
            name_col = possible
            break

    lookup: dict[str, str] = {}
    for _, row in smiles_df.iterrows():
        smiles_val = str(row[smiles_col]).strip()
        if not smiles_val or pd.isna(row[smiles_col]):
            continue
        key1 = str(row[first_col]).strip().lower()
        lookup[key1] = smiles_val
        if name_col:
            key2 = str(row[name_col]).strip().lower()
            lookup[key2] = smiles_val
    return lookup


def _fingerprint_for_drug(drug_id: str, lookup: dict[str, str]):
    smiles = lookup.get(str(drug_id).strip().lower())
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.RDKFingerprint(mol, fpSize=2048)


def build_molecular_similarity_edges(
    drug_index: DrugIndex,
    smiles_path: str,
    source_drug_ids: set[str],
    target_eval_only_drug_ids: set[str],
    k: int = 3,
    threshold: float = 0.3,
    force_top1_if_isolated: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Connect target-eval-only drugs to source drugs via Tanimoto similarity."""
    lookup = _load_smiles_lookup(smiles_path)

    source_ids_sorted = sorted(
        d for d in drug_index.drug_ids if d in source_drug_ids
    )
    source_fps = []
    for d in source_ids_sorted:
        fp = _fingerprint_for_drug(d, lookup)
        if fp is None:
            raise ValueError(f"Missing or invalid SMILES for source drug '{d}' in hybrid graph.")
        source_fps.append(fp)

    edge_rows: list[tuple[int, int]] = []
    report_rows: list[dict] = []

    for tgt_d in sorted(target_eval_only_drug_ids):
        if tgt_d not in drug_index.drug_to_index:
            continue
        tgt_idx = drug_index.drug_to_index[tgt_d]
        tgt_fp = _fingerprint_for_drug(tgt_d, lookup)
        if tgt_fp is None:
            raise ValueError(f"Missing or invalid SMILES for target-eval-only drug '{tgt_d}'.")

        sims = BulkTanimotoSimilarity(tgt_fp, source_fps)
        candidates = [
            (source_ids_sorted[i], float(sims[i]))
            for i in range(len(source_ids_sorted))
            if sims[i] >= threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:k]
        forced = False

        if not selected and force_top1_if_isolated and source_ids_sorted:
            best_i = int(np.argmax(sims))
            selected = [(source_ids_sorted[best_i], float(sims[best_i]))]
            forced = True

        for src_d, sim in selected:
            src_idx = drug_index.drug_to_index[src_d]
            edge_rows.append((tgt_idx, src_idx))
            edge_rows.append((src_idx, tgt_idx))
            report_rows.append({
                "drug_id": tgt_d,
                "drug_index": tgt_idx,
                "edge_type": "molecular_similarity",
                "neighbor_drug_id": src_d,
                "neighbor_drug_index": src_idx,
                "similarity": float(sim),
                "source": "tanimoto_rdkit",
                "forced_top1_edge": forced,
            })

    if edge_rows:
        edge_arr = np.array(edge_rows, dtype=int).T
    else:
        edge_arr = np.zeros((2, 0), dtype=int)

    report = pd.DataFrame(report_rows)
    return edge_arr, report


def build_hybrid_drug_graph(
    y_source: np.ndarray,
    mask_source: np.ndarray,
    drug_index: DrugIndex,
    source_drug_ids: set[str],
    smiles_path: str,
    task_type: str,
    reg_binary_threshold: float,
    threshold_label: float,
    similarity_k: int = 3,
    similarity_threshold: float = 0.3,
    force_top1_if_isolated: bool = True,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Merge source co-occurrence edges, molecular similarity edges, and self-loops."""
    n_drugs = len(drug_index.drug_ids)
    target_eval_only = set(drug_index.drug_ids) - source_drug_ids

    cooc_adj = build_source_cooccurrence_adjacency(
        y_source, mask_source, task_type, reg_binary_threshold, threshold_label
    )

    report_parts: list[pd.DataFrame] = []

    # Source co-occurrence edge report
    cooc_rows = []
    x_idx, y_idx = np.where(cooc_adj == 1)
    for i, j in zip(x_idx, y_idx):
        cooc_rows.append({
            "drug_id": drug_index.index_to_drug[i],
            "drug_index": i,
            "edge_type": "source_cooccurrence",
            "neighbor_drug_id": drug_index.index_to_drug[j],
            "neighbor_drug_index": j,
            "similarity": float("nan"),
            "source": "source_response_labels",
            "forced_top1_edge": False,
        })
    if cooc_rows:
        report_parts.append(pd.DataFrame(cooc_rows))

    sim_edges, sim_report = build_molecular_similarity_edges(
        drug_index=drug_index,
        smiles_path=smiles_path,
        source_drug_ids=source_drug_ids,
        target_eval_only_drug_ids=target_eval_only,
        k=similarity_k,
        threshold=similarity_threshold,
        force_top1_if_isolated=force_top1_if_isolated,
    )
    if not sim_report.empty:
        report_parts.append(sim_report)

    # Self-loops
    self_rows = []
    for i in range(n_drugs):
        self_rows.append({
            "drug_id": drug_index.index_to_drug[i],
            "drug_index": i,
            "edge_type": "self_loop",
            "neighbor_drug_id": drug_index.index_to_drug[i],
            "neighbor_drug_index": i,
            "similarity": float("nan"),
            "source": "self",
            "forced_top1_edge": False,
        })
    report_parts.append(pd.DataFrame(self_rows))

    # Combine edge indices
    edge_list: list[tuple[int, int]] = []
    x_idx, y_idx = np.where(cooc_adj == 1)
    for i, j in zip(x_idx, y_idx):
        edge_list.append((int(i), int(j)))

    if sim_edges.shape[1] > 0:
        for col in range(sim_edges.shape[1]):
            edge_list.append((int(sim_edges[0, col]), int(sim_edges[1, col])))

    for i in range(n_drugs):
        edge_list.append((i, i))

    if edge_list:
        unique_edges = list({(a, b) for a, b in edge_list})
        edge_index = np.array(unique_edges, dtype=int).T
    else:
        edge_index = np.zeros((2, 0), dtype=int)

    drug_graph_edge_report = pd.concat(report_parts, ignore_index=True) if report_parts else pd.DataFrame()
    return torch.from_numpy(edge_index).long(), drug_graph_edge_report


def build_drug_graph(
    y_source: np.ndarray,
    mask_source: np.ndarray,
    task_type: str,
    reg_binary_threshold: float,
    threshold_label: float = 0.1,
) -> torch.Tensor:
    """Legacy source-only co-occurrence graph (no hybrid edges)."""
    adj = build_source_cooccurrence_adjacency(
        y_source, mask_source, task_type, reg_binary_threshold, threshold_label
    )
    x_indices, y_indices = np.where(adj == 1)
    edge_index = np.vstack((x_indices, y_indices))
    # Add self-loops
    self_loops = np.arange(adj.shape[0])
    edge_index = np.hstack([edge_index, np.vstack([self_loops, self_loops])])
    return torch.from_numpy(edge_index).long()
