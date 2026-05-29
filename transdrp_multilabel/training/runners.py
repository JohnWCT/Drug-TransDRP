import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem

# Import TransDRP root modules
_PKG_ROOT = Path(__file__).resolve().parents[1]
_TRANSDRP_ROOT = _PKG_ROOT.parent
if str(_TRANSDRP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRANSDRP_ROOT))

import config as legacy_config
from transdrp_multilabel.contracts import TransDRPMultilabelConfig, DrugIndex
from transdrp_multilabel.data.prepare_pretrain import prepare_pretrain_data
from transdrp_multilabel.data.prepare_finetune import prepare_finetune_data
from transdrp_multilabel.data.sample_id import sample_match_key
from transdrp_multilabel.model.legacy_adapter import build_legacy_transdrp_components, run_pretrain
from transdrp_multilabel.model.checkpoint import save_checkpoint, load_checkpoint
from transdrp_multilabel.model.heads import MultiOutputDrugHead
from transdrp_multilabel.training.trainer import train_finetune, get_tissue_prototypes
from transdrp_multilabel.evaluation.prediction import predict_matrix, build_prediction_long_table
from transdrp_multilabel.evaluation.metrics import compute_metrics_from_predictions
from transdrp_multilabel.evaluation.latent_eval import compute_distribution_metrics, compute_kmeans_cancer_type_metrics
from transdrp_multilabel.evaluation.reports import (
    aggregate_per_drug_metrics,
    aggregate_scalar_metrics,
    aggregate_summary_metrics,
    build_combined_eval_summary
)
from transdrp_multilabel.export.latent import extract_latent_table
from transdrp_multilabel.export.visualization import run_tsne, plot_tsne_by_domain, plot_tsne_by_cancer_type
from transdrp_multilabel.io import write_json, ensure_clean_dir, write_csv
from transdrp_multilabel.config import config_to_dict
import pickle

def get_drug_features(drug_index: DrugIndex, smiles_path: str) -> torch.Tensor:
    """Compute 64-bit RDKit fingerprints for all unique drugs."""
    smiles_df = pd.read_csv(smiles_path, header=0)
    cols_lower = [str(c).lower().strip() for c in smiles_df.columns]
    smiles_df.columns = cols_lower

    smiles_col = None
    for possible in ["isosmiles", "smiles"]:
        if possible in cols_lower:
            smiles_col = possible
            break
    if smiles_col is None:
        raise KeyError(f"Could not find smiles or isosmiles column in {smiles_path}")

    first_col = cols_lower[0]
    name_col = None
    for possible in ["name", "drug_name", "drug_id"]:
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

    fp_list = []
    missing_drugs = []
    for d in drug_index.drug_ids:
        d_lower = str(d).strip().lower()
        if d_lower in lookup:
            smiles = lookup[d_lower]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmiles("CC")
            fp = Chem.RDKFingerprint(mol, fpSize=64)
            fp_list.append(np.array(fp))
        else:
            missing_drugs.append(d)

    if missing_drugs:
        raise ValueError(
            f"Smiles file '{smiles_path}' does not contain structural records for the "
            f"following {len(missing_drugs)} drugs: {missing_drugs}. Halted execution."
        )

    return torch.from_numpy(np.array(fp_list)).float()

def build_drug_graph(y_source: np.ndarray, mask_source: np.ndarray, task_type: str, reg_binary_threshold: float) -> torch.Tensor:
    """Construct drug co-occurrence adjacency matrix."""
    n_drugs = y_source.shape[1]

    # 1. Binarize labels
    if task_type == "classification":
        labels = y_source.copy()
        labels[mask_source == 0] = 0
    else:
        # User Decision 1 (Option B): Binarize regression values with threshold
        labels = (y_source < reg_binary_threshold).astype(float)
        labels[mask_source == 0] = 0.0

    # 2. Build overlap co-occurrence matrix
    label_graph = np.eye(n_drugs, dtype=float)
    for idx in range(n_drugs):
        for j in range(n_drugs):
            if idx == j:
                continue
            # Count overlaps
            overlap = np.sum((labels[:, idx] == 1.0) & (labels[:, j] == 1.0))
            label_graph[idx, j] = overlap

    # Set diagonals to sum of active instances
    row, col = np.diag_indices_from(label_graph)
    label_graph[row, col] = np.sum(labels, axis=0)

    # Column-wise normalization
    for col_idx in range(n_drugs):
        normalizer = np.sum(label_graph[:, col_idx])
        if normalizer > 0:
            label_graph[:, col_idx] = label_graph[:, col_idx] / normalizer

    # Apply default co-occurrence threshold of 0.1
    adj = label_graph - np.diag(np.diag(label_graph))
    adj = (adj >= 0.1).astype(int)

    # Extract edge index
    x_indices, y_indices = np.where(adj == 1)
    edge_index = np.vstack((x_indices, y_indices))

    return torch.from_numpy(edge_index).long()

class PretrainRunner:
    def __init__(self, config: TransDRPMultilabelConfig) -> None:
        self.config = config

    def run(self) -> None:
        ensure_clean_dir(self.config.output_dir, self.config.overwrite)

        # Save config
        config_dict = config_to_dict(self.config)
        write_json(config_dict, os.path.join(self.config.output_dir, "config.json"))

        # Prepare data
        prepared = prepare_pretrain_data(self.config)
        write_csv(prepared.feature_alignment, os.path.join(self.config.output_dir, "feature_alignment_report.csv"))

        # Run pretrain
        pre_dir = os.path.join(self.config.output_dir, "pretrain")
        os.makedirs(pre_dir, exist_ok=True)
        shared_encoder = run_pretrain(self.config, prepared, pre_dir)

        # Save checkpoint
        save_checkpoint(shared_encoder, os.path.join(pre_dir, "checkpoint.pt"))
        print("\nPretraining completed successfully.")

class FineTuneRunner:
    def __init__(self, config: TransDRPMultilabelConfig) -> None:
        self.config = config

    def _clean_output_dir(self) -> None:
        import shutil
        output_dir = self.config.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            return

        chk_path = Path(self.config.pretrain_checkpoint).resolve() if self.config.pretrain_checkpoint else None
        out_path = Path(output_dir).resolve()

        def clean_dir_except(dir_path: Path, preserve_path: Path) -> None:
            for item in dir_path.iterdir():
                if preserve_path and (item == preserve_path or item in preserve_path.parents):
                    if item.is_dir():
                        clean_dir_except(item, preserve_path)
                else:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

        for item in out_path.iterdir():
            if chk_path and (item == chk_path or item in chk_path.parents):
                if item.is_dir():
                    clean_dir_except(item, chk_path)
            else:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def run(self) -> None:
        # Check overwrite
        if os.path.exists(self.config.output_dir) and not self.config.overwrite:
            raise FileExistsError(f"Directory {self.config.output_dir} already exists and overwrite is set to False.")

        if self.config.overwrite:
            self._clean_output_dir()
        else:
            os.makedirs(self.config.output_dir, exist_ok=True)

        # Save config
        config_dict = config_to_dict(self.config)
        write_json(config_dict, os.path.join(self.config.output_dir, "config.json"))

        # Prepare data
        prepared = prepare_finetune_data(self.config)

        # Save drug list
        drug_df = pd.DataFrame({
            "drug_id": prepared.drug_index.drug_ids,
            "drug_index": list(range(len(prepared.drug_index.drug_ids)))
        })
        write_csv(drug_df, os.path.join(self.config.output_dir, "drug_list.csv"))

        # Set up dynamic GNN features
        node_x = get_drug_features(prepared.drug_index, self.config.drug_smiles_path)

        # Build global cancer type map
        cancer_types = sorted(list(prepared.cancer_type_table["cancer_type"].unique()))
        if "Unknown" not in cancer_types:
            cancer_types.append("Unknown")
        cancer_type_map = {name: idx for idx, name in enumerate(cancer_types)}



        # Align variables for training dataloaders
        src_x = prepared.source_omics.x.values.astype("float32")
        src_y = prepared.source_response.y.astype("float32")
        src_m = prepared.source_response.mask.astype("float32")

        tgt_x = prepared.target_omics.x.values.astype("float32")
        tgt_y = prepared.target_response.y.astype("float32")
        tgt_m = prepared.target_response.mask.astype("float32")

        # Map samples to tissue index
        ct_map = dict(zip(prepared.cancer_type_table["sample_id"], prepared.cancer_type_table["cancer_type"]))
        src_tissue = np.array([cancer_type_map.get(ct_map.get(sid, "Unknown"), cancer_type_map["Unknown"]) for sid in prepared.source_omics.sample_ids])
        tgt_tissue = np.array([cancer_type_map.get(ct_map.get(sid, "Unknown"), cancer_type_map["Unknown"]) for sid in prepared.target_omics.sample_ids])

        all_folds_src_per_drug = []
        all_folds_tgt_per_drug = []
        all_folds_src_summary = []
        all_folds_tgt_summary = []
        all_folds_latent_metrics = []
        all_folds_kmeans = []
        all_folds_summary_rows = []

        # Write pre-training/alignment reports
        # 1. feature_alignment_report.csv
        common = sorted(list(set(prepared.source_omics.feature_names) & set(prepared.target_omics.feature_names)))
        report_rows = []
        all_feats = sorted(list(set(prepared.source_omics.feature_names) | set(prepared.target_omics.feature_names)))
        src_set, tgt_set = set(prepared.source_omics.feature_names), set(prepared.target_omics.feature_names)
        for f in all_feats:
            report_rows.append({
                "feature_name": f,
                "in_source": f in src_set,
                "in_target": f in tgt_set,
                "used": f in common
            })
        feature_align_df = pd.DataFrame(report_rows)
        write_csv(feature_align_df, os.path.join(self.config.output_dir, "feature_alignment_report.csv"))

        # 2. data_alignment_report.csv
        data_align_df = pd.DataFrame([
            {"metric": "source_samples", "value": len(prepared.source_omics.sample_ids)},
            {"metric": "target_samples", "value": len(prepared.target_omics.sample_ids)},
            {"metric": "source_observed_positions", "value": int(prepared.source_response.mask.sum())},
            {"metric": "target_observed_positions", "value": int(prepared.target_response.mask.sum())},
            {"metric": "n_drugs", "value": prepared.drug_index.n_drugs}
        ])
        write_csv(data_align_df, os.path.join(self.config.output_dir, "data_alignment_report.csv"))

        # 3. drug_availability_report.csv
        drug_avail_rows = []
        for j, d in enumerate(prepared.drug_index.drug_ids):
            src_obs = int(src_m[:, j].sum())
            tgt_obs = int(tgt_m[:, j].sum())
            drug_avail_rows.append({
                "drug_id": d,
                "drug_index": j,
                "source_observed": src_obs,
                "target_observed": tgt_obs
            })
        drug_avail_df = pd.DataFrame(drug_avail_rows)
        write_csv(drug_avail_df, os.path.join(self.config.output_dir, "drug_availability_report.csv"))

        # Run across folds
        for fold in prepared.folds:
            print(f"\n================ Running Fold {fold.fold_id} ================")

            # Map sample IDs to indices
            src_sample_to_idx = {sid: idx for idx, sid in enumerate(prepared.source_omics.sample_ids)}
            train_idx = [src_sample_to_idx[sid] for sid in fold.train_sample_ids]
            val_idx = [src_sample_to_idx[sid] for sid in fold.val_sample_ids]
            test_idx = [src_sample_to_idx[sid] for sid in fold.test_sample_ids]

            # Build training subset drug graph dynamically (no leak)
            y_train_subset = src_y[train_idx]
            mask_train_subset = src_m[train_idx]
            edge_index = build_drug_graph(y_train_subset, mask_train_subset, self.config.task_type, self.config.regression_binary_threshold)

            # Create dataloaders
            train_ds = TensorDataset(
                torch.from_numpy(src_x[train_idx]),
                torch.from_numpy(src_y[train_idx]),
                torch.from_numpy(src_m[train_idx]),
                torch.from_numpy(src_tissue[train_idx])
            )
            val_ds = TensorDataset(
                torch.from_numpy(src_x[val_idx]),
                torch.from_numpy(src_y[val_idx]),
                torch.from_numpy(src_m[val_idx]),
                torch.from_numpy(src_tissue[val_idx])
            )
            target_ds = TensorDataset(
                torch.from_numpy(tgt_x),
                torch.from_numpy(tgt_y),
                torch.from_numpy(tgt_m),
                torch.from_numpy(tgt_tissue)
            )

            train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
            target_loader = DataLoader(target_ds, batch_size=self.config.batch_size, shuffle=True)


            # Initialize model components
            n_features = src_x.shape[1]
            n_drugs = len(prepared.drug_index.drug_ids)
            components = build_legacy_transdrp_components(self.config, n_features)
            shared_encoder = components["shared_encoder"]

            # Load Stage 1 checkpoint
            load_checkpoint(shared_encoder, self.config.pretrain_checkpoint, device=self.config.device)
            shared_encoder.to(self.config.device)


            # Build heads and tissue prototypes
            classifier = MultiOutputDrugHead(
                input_dim=self.config.latent_dim + 64,  # concatenated latent + drug SMILES fingerprint
                hidden_dims=list(self.config.classifier_hidden_dims),
                drug_num=n_drugs,
                drop=self.config.drop
            )

            # Compute tissue prototypes loader using Stage 1 aligned encoders
            src_proto_loader = get_tissue_prototypes(shared_encoder, torch.from_numpy(src_x[train_idx]), prepared.cancer_type_table, fold.train_sample_ids, "source", self.config.device)
            tgt_proto_loader = get_tissue_prototypes(shared_encoder, torch.from_numpy(tgt_x), prepared.cancer_type_table, prepared.target_omics.sample_ids, "target", self.config.device)
            prototypes = [src_proto_loader, tgt_proto_loader]

            # Train GRL & Predictor
            da_network = train_finetune(
                config=self.config,
                encoder=shared_encoder,
                classifier=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                target_loader=target_loader,
                node_x=node_x,
                edge_index=edge_index,
                prototypes=prototypes,
                output_dir=self.config.output_dir,
                fold_id=fold.fold_id
            )

            # Predict wide matrices
            src_test_x = src_x[test_idx]
            src_test_y = src_y[test_idx]
            src_test_m = src_m[test_idx]

            src_test_scores = predict_matrix(da_network, src_test_x, self.config.batch_size, self.config.device, node_x, edge_index)
            tgt_scores = predict_matrix(da_network, tgt_x, self.config.batch_size, self.config.device, node_x, edge_index)

            # Save predictions results
            src_pred_long = build_prediction_long_table(
                src_test_scores, src_test_y, src_test_m, fold.test_sample_ids,
                prepared.drug_index, "source", "test", self.config.task_type,
                self.config.prediction_threshold, self.config.regression_binary_threshold,
                fold.fold_id, self.config.seed, prepared.cancer_type_table
            )
            tgt_pred_long = build_prediction_long_table(
                tgt_scores, tgt_y, tgt_m, prepared.target_omics.sample_ids,
                prepared.drug_index, "target", "test", self.config.task_type,
                self.config.prediction_threshold, self.config.regression_binary_threshold,
                fold.fold_id, self.config.seed, prepared.cancer_type_table
            )

            fold_dir = os.path.join(self.config.output_dir, f"fold_{fold.fold_id}")
            write_csv(src_pred_long, os.path.join(fold_dir, "source_prediction_results.csv"))
            write_csv(tgt_pred_long, os.path.join(fold_dir, "target_prediction_results.csv"))

            # Compute fold metrics
            src_per, src_sum = compute_metrics_from_predictions(src_pred_long, self.config.task_type, "source")
            tgt_per, tgt_sum = compute_metrics_from_predictions(tgt_pred_long, self.config.task_type, "target")

            write_csv(src_per, os.path.join(fold_dir, "source_metrics_per_drug.csv"))
            write_csv(src_sum, os.path.join(fold_dir, "source_metrics_summary.csv"))
            write_csv(tgt_per, os.path.join(fold_dir, "target_metrics_per_drug.csv"))
            write_csv(tgt_sum, os.path.join(fold_dir, "target_metrics_summary.csv"))

            all_folds_src_per_drug.append(src_per.assign(fold=fold.fold_id))
            all_folds_tgt_per_drug.append(tgt_per.assign(fold=fold.fold_id))
            all_folds_src_summary.append(src_sum.assign(fold=fold.fold_id))
            all_folds_tgt_summary.append(tgt_sum.assign(fold=fold.fold_id))

            # Export latents
            src_latent = extract_latent_table(da_network, prepared.source_omics, self.config.batch_size, self.config.device)
            tgt_latent = extract_latent_table(da_network, prepared.target_omics, self.config.batch_size, self.config.device)
            write_csv(src_latent, os.path.join(fold_dir, "source_latent_representation.csv"))
            write_csv(tgt_latent, os.path.join(fold_dir, "target_latent_representation.csv"))

            # Save pickled latent dicts
            latent_cols = [c for c in src_latent.columns if c.startswith("latent_")]
            src_latent_dict = {row["sample_id"]: row[latent_cols].tolist() for _, row in src_latent.iterrows()}
            tgt_latent_dict = {row["sample_id"]: row[latent_cols].tolist() for _, row in tgt_latent.iterrows()}
            with open(os.path.join(fold_dir, "source_latent_representation.pkl"), "wb") as f:
                pickle.dump(src_latent_dict, f)
            with open(os.path.join(fold_dir, "target_latent_representation.pkl"), "wb") as f:
                pickle.dump(tgt_latent_dict, f)

            combined_latent_dict = {**src_latent_dict, **tgt_latent_dict}
            with open(os.path.join(fold_dir, "latent_representation.pkl"), "wb") as f:
                pickle.dump(combined_latent_dict, f)

            # Distribution metrics (FID, MMD, Wasserstein)
            dist_metrics = compute_distribution_metrics(src_latent_dict, tgt_latent_dict)
            dist_df = pd.DataFrame([dist_metrics]).assign(fold=fold.fold_id)
            write_csv(dist_df, os.path.join(fold_dir, "latent_distribution_metrics.csv"))
            all_folds_latent_metrics.append(dist_df)

            # K-Means clustering metrics
            cancer_map_str = dict(zip(prepared.cancer_type_table["sample_id"].astype(str), prepared.cancer_type_table["cancer_type"].astype(str)))
            kmeans_metrics = compute_kmeans_cancer_type_metrics(combined_latent_dict, cancer_map_str, self.config.seed)
            kmeans_df = pd.DataFrame([kmeans_metrics]).assign(fold=fold.fold_id)
            write_csv(kmeans_df, os.path.join(fold_dir, "kmeans_cancer_type_metrics.csv"))
            all_folds_kmeans.append(kmeans_df)

            # Save checkpoint load report
            checkpoint_load = {
                "checkpoint_path": self.config.pretrain_checkpoint,
                "status": "success",
                "device": self.config.device
            }
            write_json(checkpoint_load, os.path.join(fold_dir, "checkpoint_load_report.json"))

            # Save selection report and collect for fold_summary
            log_path = os.path.join(fold_dir, "training_log.csv")
            train_log_path = os.path.join(fold_dir, "train_log.csv")
            if os.path.exists(log_path):
                os.rename(log_path, train_log_path)

            if os.path.exists(train_log_path):
                log_df = pd.read_csv(train_log_path)
                if self.config.task_type == "classification":
                    best_row = log_df.loc[log_df["val_metric_val"].idxmax()]
                else:
                    best_row = log_df.loc[log_df["val_metric_val"].idxmin()]
                best_epoch = int(best_row["epoch"])
                best_val = float(best_row["val_metric_val"])
                metric_name = str(best_row["val_metric_name"])
            else:
                best_epoch = 0
                best_val = 0.0
                metric_name = self.config.metric or ("macro_auroc" if self.config.task_type == "classification" else "macro_mae")

            selection_rep = {
                "best_epoch": best_epoch,
                "best_metric_val": best_val,
                "metric_name": metric_name
            }
            write_json(selection_rep, os.path.join(fold_dir, "selection_report.json"))

            all_folds_summary_rows.append({
                "fold_id": fold.fold_id,
                "best_epoch": best_epoch,
                "best_metric_name": metric_name,
                "best_metric_value": best_val,
                "seed": self.config.seed,
                "task_type": self.config.task_type,
                "n_drugs": n_drugs
            })

            # Generate t-SNE visualizations
            combined_latent_df = pd.concat([src_latent, tgt_latent], ignore_index=True)
            tsne_df = run_tsne(combined_latent_df, self.config.seed)
            if tsne_df is not None:
                plot_tsne_by_domain(tsne_df, os.path.join(fold_dir, "tsne_domain_mixing.png"))
                plot_tsne_by_cancer_type(tsne_df, prepared.cancer_type_table, os.path.join(fold_dir, "tsne_cancer_type.png"))

        # Write global summaries
        # 1. source_split.csv
        split_rows = []
        for fold in prepared.folds:
            for sid in fold.train_sample_ids:
                split_rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "train"})
            for sid in fold.val_sample_ids:
                split_rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "val"})
            for sid in fold.test_sample_ids:
                split_rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "test"})
        write_csv(pd.DataFrame(split_rows), os.path.join(self.config.output_dir, "source_split.csv"))

        # 2. cancer_type_summary.csv
        source_ct_sub = prepared.cancer_type_table[prepared.cancer_type_table["domain"] == "source"]
        target_ct_sub = prepared.cancer_type_table[prepared.cancer_type_table["domain"] == "target"]
        cancer_type_summary_df = pd.DataFrame([
            {
                "domain": "source",
                "n_samples": len(prepared.source_omics.sample_ids),
                "n_unknown": int((source_ct_sub["cancer_type"] == "Unknown").sum()),
                "cancer_type_path": self.config.source_cancer_type_path
            },
            {
                "domain": "target",
                "n_samples": len(prepared.target_omics.sample_ids),
                "n_unknown": int((target_ct_sub["cancer_type"] == "Unknown").sum()),
                "cancer_type_path": self.config.target_cancer_type_path
            }
        ])
        write_csv(cancer_type_summary_df, os.path.join(self.config.output_dir, "cancer_type_summary.csv"))

        # 3. fold_summary.csv
        fold_summary_df = pd.DataFrame(all_folds_summary_rows)
        write_csv(fold_summary_df, os.path.join(self.config.output_dir, "fold_summary.csv"))

        # 4. metric aggregates
        if all_folds_src_summary:
            write_csv(pd.concat(all_folds_src_summary, ignore_index=True), os.path.join(self.config.output_dir, "source_test_metrics_summary_across_folds.csv"))
            write_csv(aggregate_summary_metrics(all_folds_src_summary), os.path.join(self.config.output_dir, "source_test_metrics_summary_fold_mean_std.csv"))

        if all_folds_tgt_summary:
            write_csv(pd.concat(all_folds_tgt_summary, ignore_index=True), os.path.join(self.config.output_dir, "target_eval_metrics_summary_across_folds.csv"))
            write_csv(aggregate_summary_metrics(all_folds_tgt_summary), os.path.join(self.config.output_dir, "target_eval_metrics_summary_fold_mean_std.csv"))

        if all_folds_src_summary or all_folds_tgt_summary:
            write_csv(build_combined_eval_summary(all_folds_src_summary, all_folds_tgt_summary), os.path.join(self.config.output_dir, "eval_metrics_summary_fold_mean_std.csv"))

        if all_folds_src_per_drug:
            write_csv(aggregate_per_drug_metrics(all_folds_src_per_drug), os.path.join(self.config.output_dir, "source_test_metrics_per_drug_fold_mean_std.csv"))

        if all_folds_tgt_per_drug:
            write_csv(aggregate_per_drug_metrics(all_folds_tgt_per_drug), os.path.join(self.config.output_dir, "target_eval_metrics_per_drug_fold_mean_std.csv"))

        if all_folds_latent_metrics:
            write_csv(pd.concat(all_folds_latent_metrics, ignore_index=True), os.path.join(self.config.output_dir, "latent_metrics_summary.csv"))

        if all_folds_kmeans:
            write_csv(pd.concat(all_folds_kmeans, ignore_index=True), os.path.join(self.config.output_dir, "kmeans_cancer_type_summary.csv"))
            write_csv(aggregate_scalar_metrics(all_folds_kmeans), os.path.join(self.config.output_dir, "kmeans_cancer_type_fold_mean_std.csv"))

        # Write run_manifest.json
        manifest_files = [
            "config.json",
            "drug_list.csv",
            "feature_alignment_report.csv",
            "cancer_type_summary.csv",
            "data_alignment_report.csv",
            "drug_availability_report.csv",
            "source_split.csv",
            "fold_summary.csv",
            "source_test_metrics_summary_across_folds.csv",
            "source_test_metrics_summary_fold_mean_std.csv",
            "target_eval_metrics_summary_across_folds.csv",
            "target_eval_metrics_summary_fold_mean_std.csv",
            "eval_metrics_summary_fold_mean_std.csv",
            "source_test_metrics_per_drug_fold_mean_std.csv",
            "target_eval_metrics_per_drug_fold_mean_std.csv",
            "latent_metrics_summary.csv",
            "kmeans_cancer_type_summary.csv",
            "kmeans_cancer_type_fold_mean_std.csv"
        ]
        for fold in prepared.folds:
            f_prefix = f"fold_{fold.fold_id}"
            manifest_files.extend([
                f"{f_prefix}/checkpoint_load_report.json",
                f"{f_prefix}/train_log.csv",
                f"{f_prefix}/selection_report.json",
                f"{f_prefix}/source_prediction_results.csv",
                f"{f_prefix}/target_prediction_results.csv",
                f"{f_prefix}/source_metrics_per_drug.csv",
                f"{f_prefix}/target_metrics_per_drug.csv",
                f"{f_prefix}/source_metrics_summary.csv",
                f"{f_prefix}/target_metrics_summary.csv",
                f"{f_prefix}/source_latent_representation.csv",
                f"{f_prefix}/source_latent_representation.pkl",
                f"{f_prefix}/target_latent_representation.csv",
                f"{f_prefix}/target_latent_representation.pkl",
                f"{f_prefix}/latent_representation.pkl",
                f"{f_prefix}/latent_distribution_metrics.csv",
                f"{f_prefix}/kmeans_cancer_type_metrics.csv",
                f"{f_prefix}/tsne_domain_mixing.png",
                f"{f_prefix}/tsne_cancer_type.png"
            ])
        write_json({"artifacts": manifest_files}, os.path.join(self.config.output_dir, "run_manifest.json"))

        print("\nMulti-label TransDRP execution completed successfully.")
