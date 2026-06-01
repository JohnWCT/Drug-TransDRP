# Proposal: Convert TransDRP to Multi-label / Multi-drug Single-model Framework

> **Target Document Level:** This file defines the high-level requirements, design boundaries, data contracts, and training workflows for converting TransDRP. Function-level API designs will be detailed in the subsequent design phase.

---

## 1. Project Goal

The original TransDRP implementation is structured around a predefined list of 9 drugs:

```python
# original main.py
main(args=args, params_dict=param,
     drug=["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", 
           "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"])
```

The objective is to convert this workflow into a **multi-label / multi-drug single-model framework** capable of predicting responses for all drugs present in your local datasets.

The new workflow will:
1. Dynamically build the final drug index from **all source drugs** in your local input tables. target-only drugs (present only in the target response) are dropped: they are not evaluated and are not required to have SMILES. source-only drugs are kept for source training/validation.
2. Train a single, shared model to predict sensitivity scores across all source drugs simultaneously.
3. Replace hardcoded data-loading paths with general CLI paths for local tables.
4. Support both **classification** and **regression** tasks.
5. Match the entry script structure and output directory schema of the CODE-AE benchmark.

The core TransDRP components—Autoencoder (AE) pre-training, Graph MLP classifier with GNN drug propagation, Gradient Reversal Layer (GRL) domain adaptation, and tissue prototype contrastive alignment—will be preserved.

---

## 2. Key Design Decisions

| Category | Decision | Details / Rationale |
|---|---|---|
| **Workflow** | Two-stage pipeline | Preserve TransDRP's flow: Stage 1 (Unsupervised AE Pre-training) $\rightarrow$ Stage 2 (Fine-tuning & Domain Adaptation). |
| **Task Type** | Support both Classification & Regression | `--task_type classification` and `--task_type regression` are supported. |
| **Model Head** | Shared Encoder + GNN Predictor | Replace the fixed 9-output GNN classifier with a dynamically sized GNN head outputting a `[Batch_Size, N_drugs]` tensor. |
| **Target Labels** | Evaluation-Only | Consistent with TransDRP and CODE-AE designs, target labels (TCGA) are only used for validation/testing, not in the supervised training loss. |
| **Drug List** | Source drugs only | The final drug index = all source drugs. target-only drugs are dropped (not evaluated, no SMILES required); source-only drugs are kept. A `drug_availability_report.csv` records the `source_and_target` / `source_only` / `target_only` categories. |
| **SMILES Missing** | Raise Error & Stop | If a **source** drug (i.e. one in the final drug index) lacks a molecular structure in the SMILES mapping file, the program raises an error and halts execution. target-only drugs are exempt. |
| **Cancer Type Data** | External CLI Paths | Tissue/cancer types for contrastive alignment are provided via `--source_cancer_type_path` and `--target_cancer_type_path`. |
| **Data Format** | Wide Matrix + Mask | Long tables are converted to wide matrices `[N_samples, N_drugs]` plus a binary mask `[N_samples, N_drugs]` indicating observed records. |
| **Early Stopping** | Macro Average Metric | Validation macro metrics (e.g., `macro_auroc` for classification, `macro_mae` for regression) drive model checkpoint selection. |
| **Visualizations** | Export Latents & t-SNE | Save final sample latents and generate t-SNE plots for domain mixing and cancer type clustering. |

---

## 3. Proposed Entry Points

To align with the CODE-AE multi-label benchmark, we will add two new entry points under the `TransDRP/` workspace:

### 3.1 Pre-training Entry: `pretrain_multilabel_hyper_main.py`
Trains the shared autoencoder to align source (CCLE) and target (TCGA) features without utilizing drug labels.

```bash
python pretrain_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --epochs 100 \
  --output_dir outputs_transdrp_multilabel
```

### 3.2 Fine-tuning Entry: `drug_ft_multilabel_hyper_main.py`
Loads the pre-trained autoencoder checkpoint, dynamically initializes the GNN prediction head for $N$ drugs, and performs domain adaptation training.

#### Regression CLI Run Example:
```bash
python drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --source_omics_path /workspace/DAPL-master/data_Winnie/CCLE_impact_hotspot.csv \
  --target_omics_path /workspace/DAPL-master/data_Winnie/TCGA_impact_hotspot.csv \
  --source_response_path /workspace/DAPL-master/data_Winnie/PRISM_drug_sensitivity.csv \
  --target_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_response_col neg_log2_auc \
  --source_cancer_type_path /workspace/DAPL-master/data_Winnie/CCLE_cancer_type.csv \
  --target_cancer_type_path /workspace/DAPL-master/data_Winnie/TCGA_cancer_type.csv \
  --drug_smiles_path data/local/drug_smiles.csv \
  --pretrain_checkpoint outputs_transdrp_multilabel/pretrain/checkpoint.pt \
  --epochs 300 \
  --n_splits 5 \
  --output_dir outputs_transdrp_multilabel
```

#### Classification CLI Run Example:
```bash
python drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_response_col Label \
  --source_cancer_type_path /workspace/DAPL-master/data/ccle_sample_info_df.csv \
  --target_cancer_type_path /workspace/DAPL-master/data/TCGA/xena_sample_info_df.csv \
  --drug_smiles_path data/local/drug_smiles.csv \
  --pretrain_checkpoint outputs_transdrp_multilabel/pretrain/checkpoint.pt \
  --epochs 300 \
  --n_splits 5 \
  --output_dir outputs_transdrp_multilabel
```

---

## 4. Input & Data Design

### 4.1 Omics Input
Tables of shape `[N_samples, N_features]` containing continuous gene expression values.
* Features will be aligned by taking the intersection of genes present in both source and target.

### 4.2 Response Long Tables
* **Source response**: Contains `Sample_ID`, `drug_name`, and the response label. The response label column name is specified by `--source_response_col` (e.g. `neg_log2_auc` for PRISM regression; `Label` for GDSC2 classification).
* **Target response**: Contains `Patient_id`, `drug_name`, and `Label` (always binary labels for evaluation, e.g., in `PMID27354694_DR_OMICS_ad_intersect_pretrain.csv`).

### 4.3 Drug SMILES Mapping File
CSV containing mapping of drug names to their SMILES string (e.g. columns `drug_id` and `Isosmiles`).
* The script reads this file to compute the 64-bit molecular RDKit fingerprints.
* **If any source drug (in the final drug index) is missing from this file, execution terminates immediately with a ValueError.** target-only drugs are dropped before this check and are therefore exempt.

### 4.4 Cancer Type Mapping File
CSV containing mappings between sample/patient IDs and their respective tissue/cancer types. Used to extract tissue categories for CCLE and TCGA samples to calculate prototypes for the contrastive alignment loss ([InfoMax_loss](file:///home/wasijk/Drug/TransDRP/myloss.py#L62)).
* Paths are specified via `--source_cancer_type_path` and `--target_cancer_type_path`.

---

## 5. Model & Loss Adjustments

### 5.1 Dynamic Graph Construction
The drug correlation graph ([config.label_graph](file:///home/wasijk/Drug/TransDRP/config.py#L37)) is constructed dynamically:
* **Classification**: Built based on binary overlap co-occurrence similarity.
* **Regression**: Built by temporarily binarizing the continuous response values only during the graph building phase to compute the co-occurrence overlap matrix, keeping graph construction identical. Direction follows `-logAUC`: lower AUC → higher `-logAUC` → more sensitive, so `sensitive = 1 if value > threshold` (threshold = `-log(0.5)`; `--regression_binary_threshold`).

### 5.2 Multi-output Graph MLP
The [GraphMLP](file:///home/wasijk/Drug/TransDRP/models.py#L50)'s GNN layers will process a graph containing $N$ drug nodes.
* Dynamic node features: Shape `[N_drugs, 64]`.
* Output shape: `[Batch_Size, N_drugs]`.

### 5.3 Masked Losses
All prediction losses apply binary masks to ignore missing values:
* **Classification**: Masked BCE with logits loss.
  $$\text{Loss}_{\text{pred}} = \frac{\sum (L_{\text{BCE}}(yp, y) \cdot M)}{\sum M}$$
* **Regression**: Masked MAE loss for continuous source labels.
  $$\text{Loss}_{\text{pred\_source}} = \frac{\sum (|yp_{\text{source}} - y_{\text{source}}| \cdot M_{\text{source}})}{\sum M_{\text{source}}}$$

---

## 6. Output & Metrics Structure

The outputs generated in the `--output_dir` follow the CODE-AE benchmark directory structure and incorporate complete fold summaries, K-Means clustering, and latent distribution metrics (FID, MMD, Wasserstein):

```text
{output_dir}/
  config.json
  drug_list.csv
  data_alignment_report.csv
  drug_availability_report.csv
  feature_alignment_report.csv
  run_manifest.json
  source_split.csv
  pretrain/
    checkpoint.pt                        # stage 1 shared encoder weights
  fold_0/
    best_model.pt                        # stage 2 domain-adapted network
    source_prediction_results.csv
    target_prediction_results.csv
    source_metrics_per_drug.csv
    target_metrics_per_drug.csv
    source_metrics_summary.csv           # macro/micro/weighted summary
    target_metrics_summary.csv
    source_latent_representation.csv     # exported sample latents
    target_latent_representation.csv
    source_latent_representation.pkl
    target_latent_representation.pkl
    latent_representation.pkl
    latent_distribution_metrics.csv      # FID, MMD, Wasserstein distance
    kmeans_cancer_type_metrics.csv       # K-Means clustering metrics (ARI, NMI, etc.)
    tsne_domain_mixing.png               # t-SNE visualization
    tsne_cancer_type.png
    train_log.csv
  ...
  source_test_metrics_summary_across_folds.csv
  source_test_metrics_summary_fold_mean_std.csv
  target_eval_metrics_summary_across_folds.csv
  target_eval_metrics_summary_fold_mean_std.csv
  eval_metrics_summary_fold_mean_std.csv # combined source/target summary
  source_test_metrics_per_drug_fold_mean_std.csv
  target_eval_metrics_per_drug_fold_mean_std.csv
  latent_metrics_summary.csv             # distribution metrics across folds
  kmeans_cancer_type_summary.csv         # k-means metrics across folds
  kmeans_cancer_type_fold_mean_std.csv   # mean/std of k-means metrics
  cancer_type_summary.csv
  fold_summary.csv
```
