"""Target eval zero-shot drug reporting."""

from __future__ import annotations

import pandas as pd

from transdrp_multilabel.contracts import DrugIndex, PreparedFineTuneData


def build_target_eval_zero_shot_drug_report(
    prepared: PreparedFineTuneData,
    drug_graph_edge_report: pd.DataFrame,
) -> pd.DataFrame:
    """Report zero-shot / extrapolation drugs per eval dataset."""
    rows: list[dict] = []

    sim_edges = drug_graph_edge_report[
        drug_graph_edge_report["edge_type"] == "molecular_similarity"
    ] if not drug_graph_edge_report.empty else pd.DataFrame()

    for eval_name, eval_ds in prepared.target_eval_datasets.items():
        resp = eval_ds.response
        for j, drug_id in enumerate(prepared.drug_index.drug_ids):
            mask_col = resp.mask[:, j]
            if mask_col.sum() == 0:
                continue

            in_source = drug_id in prepared.source_drug_ids
            is_tgt_only = not in_source

            drug_sim = sim_edges[sim_edges["drug_id"] == drug_id] if not sim_edges.empty else pd.DataFrame()
            n_sim = len(drug_sim)
            max_sim = float(drug_sim["similarity"].max()) if n_sim else float("nan")
            mean_topk = float(drug_sim["similarity"].mean()) if n_sim else float("nan")
            forced = bool(drug_sim["forced_top1_edge"].any()) if n_sim else False

            y_col = resp.y[:, j]
            obs = mask_col > 0
            n_pos = int(((y_col == 1) & obs).sum())
            n_neg = int(((y_col == 0) & obs).sum())

            rows.append({
                "drug_id": drug_id,
                "drug_index": j,
                "eval_dataset": eval_name,
                "in_source": in_source,
                "has_supervised_source_label": in_source,
                "is_target_eval_only": is_tgt_only,
                "n_eval_observed_rows": int(obs.sum()),
                "n_eval_positive": n_pos,
                "n_eval_negative": n_neg,
                "n_similarity_edges": n_sim,
                "max_similarity_to_source": max_sim,
                "mean_similarity_to_source_topk": mean_topk,
                "forced_top1_edge": forced,
            })

    return pd.DataFrame(rows)
