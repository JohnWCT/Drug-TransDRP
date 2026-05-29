import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import cycle

from models import AdversarialNetwork
from myloss import Adversarial_loss, InfoMax_loss
from transdrp_multilabel.contracts import TransDRPMultilabelConfig
from transdrp_multilabel.training.losses import masked_bce_with_logits, masked_mae
from transdrp_multilabel.training.selection import MetricSelector
from transdrp_multilabel.evaluation.metrics import compute_prediction_bundle
from transdrp_multilabel.model.checkpoint import save_checkpoint

# Simple Dataset wrapper to recreate PrototypeData aligned with myloss
class SimplePrototypeDataset(Dataset):
    def __init__(self, types, expressions):
        self.types = types
        self.expressions = expressions
    def __len__(self):
        return len(self.types)
    def __getitem__(self, idx):
        return self.types[idx], self.expressions[idx]

def get_tissue_prototypes(
    encoder: nn.Module,
    omics_x: torch.Tensor,
    cancer_type_table: pd.DataFrame,
    sample_ids: list[str],
    domain: str,
    device: str
) -> DataLoader:
    """Calculate mean latent representations for each tissue/cancer type."""
    encoder.eval()
    with torch.no_grad():
        z = encoder(omics_x.to(device))
        if getattr(encoder, "norm_flag", False):
            z = torch.nn.functional.normalize(z, p=2, dim=1)

    # Map sample IDs to cancer types
    cancer_type_map = dict(zip(cancer_type_table["sample_id"], cancer_type_table["cancer_type"]))

    unique_types = sorted(list(set(cancer_type_table["cancer_type"].unique())))
    type_to_idx = {name: idx for idx, name in enumerate(unique_types)}

    # Group latent features by tissue type
    type_feats = {idx: [] for idx in type_to_idx.values()}
    for sid, latent in zip(sample_ids, z):
        ct = cancer_type_map.get(sid, "Unknown")
        if ct in type_to_idx:
            type_feats[type_to_idx[ct]].append(latent.unsqueeze(0))

    type_ids_list = []
    type_protos_list = []
    for idx, feats in type_feats.items():
        if feats:
            mean_feat = torch.cat(feats, dim=0).mean(dim=0)
            type_ids_list.append(idx)
            type_protos_list.append(mean_feat.unsqueeze(0))

    if not type_ids_list:
        # Fallback if no types matched
        type_ids_list = [0]
        type_protos_list = [torch.zeros((1, z.size(1)))]

    type_ids_tensor = torch.tensor(type_ids_list, dtype=torch.long)
    type_protos_tensor = torch.cat(type_protos_list, dim=0).float()

    dataset = SimplePrototypeDataset(type_ids_tensor, type_protos_tensor)
    return DataLoader(dataset, batch_size=len(type_ids_tensor), shuffle=False)

def train_finetune(
    config: TransDRPMultilabelConfig,
    encoder: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_loader: DataLoader,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    prototypes: list,
    output_dir: str,
    fold_id: int
) -> nn.Module:
    """Run Stage 2 training: Fine-tuning and domain adaptation."""
    # Wrap encoder and classifier into the AdversarialNetwork wrapper
    # Note: We pass len(drug_ids) as the third argument (fix_source) to match legacy behavior
    n_drugs = node_x.size(0)
    da_network = AdversarialNetwork(encoder, classifier, fix_source=n_drugs).to(config.device)
    node_x = node_x.to(config.device)
    edge_index = edge_index.to(config.device)

    optimizer = torch.optim.AdamW(da_network.parameters(), lr=config.lr)
    loss_fn_domain = nn.BCEWithLogitsLoss()

    best_metric_value = None
    best_model_state = None
    best_epoch = 0

    selector = MetricSelector(config.task_type, config.metric)
    log_rows = []

    print(f"\n====== Starting Fine-tuning & Domain Adaptation (Fold {fold_id}) ======")

    for epoch in range(config.epochs):
        da_network.train()
        train_loss_sum = 0.0
        len_loader = min(len(train_loader), len(target_loader))

        # Iteration-based training
        for i, batch in enumerate(zip(target_loader, cycle(train_loader))):
            p = float(i + epoch * len_loader) / int(config.epochs) / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Source batch
            s_x = batch[1][0].to(config.device)
            s_y = batch[1][1].to(config.device)
            s_mask = batch[1][2].to(config.device)
            s_type = batch[1][3].to(config.device)

            # Target batch
            t_x = batch[0][0].to(config.device)
            t_type = batch[0][3].to(config.device)

            optimizer.zero_grad()

            # Forward passes
            domain_s, s_yp, s_feat = da_network(s_x, alpha, node_x, edge_index)
            domain_t, t_yp, t_feat = da_network(t_x, alpha, node_x, edge_index)

            # 1. Domain Adaptation Loss
            transfer_loss = Adversarial_loss(domain_s, domain_t, loss_fn_domain)

            # 2. Supervised Prediction Loss on Source
            if config.task_type == "classification":
                pred_loss = masked_bce_with_logits(s_yp, s_y, s_mask)
            else:
                pred_loss = masked_mae(s_yp, s_y, s_mask)

            # 3. Tissue Contrastive Loss (InfoMax)
            contrastive_loss = InfoMax_loss(da_network.encoder.output_layer[0].in_features)(
                s_feat, t_feat, s_type, t_type, prototypes
            )

            # Combined Loss
            total_loss = (
                config.alph * transfer_loss +
                config.beta * contrastive_loss +
                (1 - 2 * config.alph) * pred_loss
            )

            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()

        avg_train_loss = train_loss_sum / len_loader

        # Evaluation and check early stopping on Validation Set
        da_network.eval()
        with torch.no_grad():
            # Get predictions on Validation domain (Source) and Patient domain (Target)
            # Create a mock bundle to compute metrics
            val_preds = []
            val_trues = []
            val_masks = []
            for x_batch, y_batch, mask_batch, _ in val_loader:
                _, yp, _ = da_network(x_batch.to(config.device), 0, node_x, edge_index)
                val_preds.append(yp.cpu())
                val_trues.append(y_batch.cpu())
                val_masks.append(mask_batch.cpu())

            val_preds_cat = torch.cat(val_preds, dim=0).numpy()
            val_trues_cat = torch.cat(val_trues, dim=0).numpy()
            val_masks_cat = torch.cat(val_masks, dim=0).numpy()

            # Target (Patient) predictions
            tgt_preds = []
            tgt_trues = []
            tgt_masks = []
            for x_batch, y_batch, mask_batch, _ in target_loader:
                _, yp, _ = da_network(x_batch.to(config.device), 0, node_x, edge_index)
                tgt_preds.append(yp.cpu())
                tgt_trues.append(y_batch.cpu())
                tgt_masks.append(mask_batch.cpu())

            tgt_preds_cat = torch.cat(tgt_preds, dim=0).numpy()
            tgt_trues_cat = torch.cat(tgt_trues, dim=0).numpy()
            tgt_masks_cat = torch.cat(tgt_masks, dim=0).numpy()

            # Construct DataFrames
            val_pred_df = pd.DataFrame(val_preds_cat)
            val_true_df = pd.DataFrame(val_trues_cat)
            val_mask_df = pd.DataFrame(val_masks_cat)

            tgt_pred_df = pd.DataFrame(tgt_preds_cat)
            tgt_true_df = pd.DataFrame(tgt_trues_cat)
            tgt_mask_df = pd.DataFrame(tgt_masks_cat)

            # Calculate validation and target metrics
            bundle = compute_prediction_bundle(
                val_pred_df, val_true_df, val_mask_df,
                tgt_pred_df, tgt_true_df, tgt_mask_df,
                config.task_type
            )

            # Extract validation selector metric
            val_summary = bundle.source_metrics_summary
            metric_name, metric_val, direction = selector.select_metric(val_summary)

            # Record logging info
            log_rows.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_metric_name": metric_name,
                "val_metric_val": metric_val,
                "target_macro_auroc": bundle.target_metrics_summary.loc[
                    bundle.target_metrics_summary["metric_name"] == "macro_auroc", "metric_value"
                ].values[0] if "macro_auroc" in bundle.target_metrics_summary["metric_name"].values else np.nan
            })

            if selector.is_better(metric_val, best_metric_value, metric_name):
                best_metric_value = metric_val
                best_model_state = copy.deepcopy(da_network.state_dict())
                best_epoch = epoch + 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val {metric_name}: {metric_val:.4f}")

    # Save best checkpoint
    fold_dir = os.path.join(output_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    best_model_path = os.path.join(fold_dir, "best_model.pt")
    torch.save(best_model_state, best_model_path)

    # Save training logs
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(fold_dir, "training_log.csv"), index=False)

    print(f"Fold {fold_id} completed. Best Epoch: {best_epoch} with Val {metric_name} = {best_metric_value:.4f}")

    # Load back the best state to return the trained model
    da_network.load_state_dict(best_model_state)
    return da_network
