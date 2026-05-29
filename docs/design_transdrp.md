# Design: Multi-label / Multi-drug TransDRP System Architecture

> **Document Status:** Final Architecture Design based on `proposal_transdrp.md` and user-approved design decisions.
>
> **Scope:** This document defines the modular architecture for converting TransDRP from a single-drug (9 drugs hardcoded) model into a multi-label / multi-drug single-model workflow. It specifies module boundaries, data contracts, model wrapping, dynamic GNN batching, loss computations, and output file layouts.
>
> **Primary Design Rule:** Preserve TransDRP's two-stage encoder pre-training and fine-tuning/domain adaptation philosophy, while replacing the single-drug predictor interface with a multi-output dynamic predictor and masked multi-drug data handling.

---

## 1. Modular System Layout

The refactored system is located in the independent package `TransDRP/transdrp_multilabel/`. This isolates the multi-label logic from the legacy single-drug codebase.

```text
TransDRP/
  pretrain_multilabel_hyper_main.py      # Entry point for Stage 1 pre-training
  drug_ft_multilabel_hyper_main.py       # Entry point for Stage 2 fine-tuning

  transdrp_multilabel/
    __init__.py
    config.py                            # CLI parsing & parameter resolution
    contracts.py                         # Dataclass contracts
    seed.py                              # Random seed control
    io.py                                # File read/write utilities
    validators.py                        # Table shape & format validators

    data/
      __init__.py
      sample_id.py                       # Sample ID normalization & TCGA matching
      omics.py                           # Gene feature intersection & alignment
      drug_index.py                      # Drug list union extraction
      response_matrix.py                 # Long to wide response + mask matrix
      split.py                           # Stratified KFold splits
      cancer_type.py                     # Cancer type mapping
      prepare_pretrain.py                # Pipeline for Stage 1 data preparation
      prepare_finetune.py                # Pipeline for Stage 2 data preparation

    model/
      __init__.py
      heads.py                           # Dynamic GNN multi-output head
      legacy_adapter.py                  # Adapt original TransDRP models
      checkpoint.py                      # Encoder weight save/load

    training/
      __init__.py
      losses.py                          # Masked BCE, MAE, GRL and InfoMax losses
      selection.py                       # Early stopping and validation checkpointing
      trainer.py                         # Stage 1 and Stage 2 training loops
      runners.py                         # Run orchestrators

    evaluation/
      __init__.py
      prediction.py                      # Prediction tables writer
      metrics.py                         # Per-drug & summary metrics calculations

    export/
      __init__.py
      latent.py                          # Latent vector exporter
      visualization.py                   # t-SNE plotter
```

---

## 2. Core Architectural Decisions

### AD-01: Drug Graph Construction under Regression Run (User Decision 1 - Option B)
In regression runs, drug labels are continuous (e.g., Z-score, LN_IC50). To construct the GNN's drug co-occurrence adjacency matrix `config.label_graph`:
* **Design**: The continuous response values of the source domain are temporarily binarized using a threshold (e.g., `Z_SCORE < 0` or a user-provided threshold) **only during the graph building phase**.
* **Rationale**: This allows us to calculate the co-occurrence overlap matrix exactly as done in the classification task, keeping drug graph construction consistent and stable.

### AD-02: Tissue Contrastive Alignment under Regression Run (User Decision 2 - Option A)
* **Design**: During regression fine-tuning, the tissue-specific contrastive learning loss ([InfoMax_loss](file:///home/wasijk/Drug/TransDRP/myloss.py#L62)) remains active.
* **Rationale**: Tissue contrastive alignment is independent of the task label type. It aligns the latent feature representations of identical cancer/tissue types between cell lines and patients.

### AD-03: GNN Batching (User Decision 3 - Option A)
* **Design**: We will represent each sample in a batch as a disjoint drug graph of size $N$ drugs.
* **Implementation**: We pack a list of PyTorch Geometric `Data(x, edge_index)` objects and batch them using `Batch.from_data_list(data_list)`.
* **Rationale**: This enables PyG GNN layers (`GATConv`) to perform sparse graph propagation efficiently on both CPU and GPU without rewriting standard layers.

### AD-04: Target Labels usage
* **Design**: Target domain (TCGA) response labels are used **exclusively for evaluation** and do not contribute to the supervised loss during domain adaptation fine-tuning, preserving the unsupervised domain adaptation setting.

---

## 3. Module Boundaries & Data Contracts

### 3.1 Data contracts: `contracts.py`

#### `TransDRPMultilabelConfig`
Holds all runtime parameters including directories, column mappings (`source_response_col` to dynamically select the label column like `neg_log2_auc` or `Label`), task type, and TransDRP-specific coefficients (`alph`, `beta`).

#### `ResponseMatrix`
Represents the wide response and observed mask matrix.
* `y.shape == mask.shape == [N_samples, N_drugs]`.
* `mask[i, j] = 1.0` if `y[i, j]` is observed, and `0.0` if missing.

---

## 4. Detailed Component Workflows

### 4.1 Stage 1: Autoencoder Pre-training
1. **Inputs**: `source_omics_path`, `target_omics_path`.
2. **Data Pipeline**:
   * Align source and target feature columns (intersection of genes).
   * Initialize `s_AE` (source autoencoder) and `t_AE` (target autoencoder) sharing a `shared_encoder` and `shared_decoder`.
3. **Training**:
   * Optimize the reconstruction loss (MSE) on both domains and the orthogonality constraint between private and shared latent spaces.
4. **Output**: Save `pretrain/checkpoint.pt` containing the `shared_encoder` weights.

### 4.2 Stage 2: Fine-tuning & Domain Adaptation
1. **Inputs**: Checkpoint from Stage 1, source/target omics tables, source/target response tables, source/target cancer type mapping paths, and drug SMILES CSV.
2. **Data Pipeline**:
   * Align omics features.
   * Extract unique drugs union and validate that all drugs exist in the SMILES CSV (raise error if any are missing).
   * Read response values from the column specified in `source_response_col` (e.g. `neg_log2_auc` or `Label`).
   * Construct GNN node features `config.drug_feat` (RDKit 64-bit fingerprints) and GNN drug graph adjacency matrix `config.label_graph` (binarized co-occurrence overlap based on response binarization threshold).
   * Generate K-fold splits on source samples.
3. **Model Construction**:
   * Load the `shared_encoder` from the checkpoint.
   * Initialize `MultiOutputDrugHead` dynamically for $N$ drugs.
   * Combine them into the `AdversarialNetwork` wrapper including a domain discriminator.
4. **Training & Adaptation**:
   * Optimize the combined loss function:
     $$\text{Total Loss} = \alpha \cdot \text{TransferLoss} + \beta \cdot \text{ContrastiveLoss} + (1 - 2\alpha) \cdot \text{ClassifierLoss}$$
     * `TransferLoss`: Domain adversarial loss via GRL.
     * `ClassifierLoss`: Masked BCE (classification) or masked MAE (regression) on source.
     * `ContrastiveLoss`: InfoMax cross-domain alignment between tissue prototypes.
5. **Early Stopping**:
   * Select checkpointing model based on validation macro prediction metric (macro AUROC or macro MAE).
6. **Outputs**:
   * Save best model to `fold_{i}/best_model.pt`.
   * Export fold predictions (`source_prediction_results.csv`, `target_prediction_results.csv`).
   * Export fold metrics (`source_metrics_per_drug.csv`, `target_metrics_per_drug.csv`, `source_metrics_summary.csv`, `target_metrics_summary.csv`).
   * Export fold latent representations (`source_latent_representation.csv`, `target_latent_representation.csv`, `.pkl` formats, and `latent_representation.pkl` for combined).
   * Compute and write distribution metrics (`latent_distribution_metrics.csv` containing FID, MMD, Wasserstein).
   * Compute and write K-Means clustering cancer type metrics (`kmeans_cancer_type_metrics.csv` containing ARI, NMI, silhouette, calinski_harabasz, davies_bouldin).
   * Save t-SNE visualizations (`tsne_domain_mixing.png`, `tsne_cancer_type.png`).
   * Generate training loss history (`train_log.csv`).
   * Aggregate final fold summaries at the root directory level (`source_test_metrics_summary_across_folds.csv`, `source_test_metrics_summary_fold_mean_std.csv`, `target_eval_metrics_summary_across_folds.csv`, `target_eval_metrics_summary_fold_mean_std.csv`, `eval_metrics_summary_fold_mean_std.csv`, `source_test_metrics_per_drug_fold_mean_std.csv`, `target_eval_metrics_per_drug_fold_mean_std.csv`, `latent_metrics_summary.csv`, `kmeans_cancer_type_summary.csv`, `kmeans_cancer_type_fold_mean_std.csv`, `cancer_type_summary.csv`, `fold_summary.csv`).
