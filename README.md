# Multi-label TransDRP (Drug Response Prediction) System

本專案是將原始的 **TransDRP**（基於 Transformer 與 GNN 的藥物敏感性預測模型）重構成 **多標籤 / 多藥物單一模型框架 (Multi-label / Multi-drug Single-model Framework)**。

原始的 TransDRP 只支援寫死（hardcoded）的 9 種藥物獨立訓練，而重構後的版本能夠**動態從您的本地數據集中提取所有藥物的聯集**，並利用單個共享的 GNN 預測頭同時預測所有藥物的敏感度分數。此外，本系統同時支援**分類（Classification）**與**回歸（Regression）**任務。

---

## 核心設計特點
1. **動態藥物圖結構建構 (Dynamic Graph Construction)**：依據訓練集中各藥物在樣本間的共同敏感性（Co-occurrence）頻次，並套用大於等於 `0.1` 的正規化閾值過濾規則，動態建構用於 GNN（GATConv）傳遞的藥物交互作用圖。
2. **多輸出 GNN 預測頭 (Multi-output GNN Head)**：將多體學特徵與 RDKit 計算的 64 位元藥物分子指紋特徵進行融合，透過 disjoint batching 方式送入 PyTorch Geometric 的 GNN 進行端到端預測。
3. **遮罩損失函數 (Masked Loss Functions)**：自動對缺失的藥物-樣本對套用二值遮罩，計算 Masked BCE (分類) 或 Masked MAE (回歸)，避免未觀測數據干擾梯度更新。
4. **跨領域組織對齊 (Tissue Contrastive Alignment)**：不論分類或回歸任務，皆保留跨領域（細胞株與患者組織）的對比學習損失（InfoMax），對齊潛在空間特徵。

---

## 專案結構
```text
TransDRP/
  ├── pretrain_multilabel_hyper_main.py  # 階段 1：無監督 Autoencoder 預訓練入口
  ├── drug_ft_multilabel_hyper_main.py   # 階段 2：微調與域適應入口
  ├── train_params.json                  # 超參數預設設定檔
  ├── transdrp_multilabel/               # 重構後的的核心包
  │   ├── data/                          # 數據加載、對齊、K-Fold 劃分
  │   ├── model/                         # GNN 頭、模型封裝器、Checkpoint
  │   ├── training/                      # 遮罩 Loss、早停、訓練循環
  │   ├── evaluation/                    # 藥物預測表建構、指標計算
  │   ├── export/                        # 潛在特徵導出、t-SNE 視覺化
  │   └── smoke/                         # 仿真合成數據 Smoke Test 模組
  └── tests/                             # 單元測試與集成測試目錄
```

---

## 環境配置

本專案在 Docker 容器 `transDRP` 中執行。容器內已配置好以下環境：
- **Python** (3.9+)
- **PyTorch** (2.1.0+ / 支持 CUDA 12.1)
- **PyTorch Geometric (PyG)** (2.4.0+)
- **RDKit**

---

## 使用指南

本系統提供兩個主要的命令列入口點，分別用於**階段 1：無監督預訓練**以及**階段 2：微調與域適應**。

---

### 1. 階段 1：無監督預訓練 (`pretrain_multilabel_hyper_main.py`)

預訓練階段主要訓練一個共享的 Autoencoder 以對齊細胞株與患者的體學特徵，此階段不使用任何藥物反應標籤。

#### 執行範例
```bash
docker exec -it transDRP python pretrain_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data_Winnie/CCLE_impact_hotspot.csv \
  --target_omics_path /workspace/DAPL-master/data_Winnie/TCGA_impact_hotspot.csv \
  --batch_size 128 \
  --epochs 300 \
  --output_dir outputs_transdrp_multilabel \
  --device cuda
```

#### 參數說明

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `--source_omics_path` | **[必填]** 來源域（細胞株，如 CCLE）的體學特徵資料 CSV 路徑。 | - |
| `--target_omics_path` | **[必填]** 目標域（患者組織，如 TCGA）的體學特徵資料 CSV 路徑。 | - |
| `--output_dir` | **[必填]** 輸出結果與 Checkpoint 的儲存目錄路徑。 | - |
| `--source_sample_col` | 來源域體學資料中樣本 ID 的欄位名稱。 | `Sample_ID` |
| `--target_sample_col` | 目標域體學資料中樣本 ID 的欄位名稱。 | `tissue_id` |
| `--method` | 預訓練模型方法。 | `transdrp_ae` |
| `--overwrite` / `--no-overwrite` | 是否覆寫已存在之輸出目錄。 | `--overwrite` |
| `--epochs` | 預訓練的 Epoch 數。若未指定，則載入 `train_params.json` 中的 `pretrain_num_epochs`。 | 載入設定檔 |
| `--batch_size` | 訓練 Batch Size。若未指定，則載入 `train_params.json` 中的 `unlabeled.batch_size`。 | 載入設定檔 |
| `--lr` | 學習率。若未指定，則載入 `train_params.json` 中的 `unlabeled.lr`。 | 載入設定檔 |
| `--seed` | 隨機數種子，以確保實驗可重複性。 | `2024` |
| `--norm_flag` / `--no-norm_flag` | 是否對資料進行歸一化。 | 載入設定檔 |
| `--retrain_flag` / `--no-retrain_flag` | 是否強制重新訓練模型。 | `--retrain_flag` |
| `--device` | 指定運行設備（如 `cuda`, `cpu`, `cuda:1` 等）。 | 自動偵測 |

---

### 2. 階段 2：微調與域適應 (`drug_ft_multilabel_hyper_main.py`)

微調階段載入第一階段預訓練好的編碼器權重，並初始化 GNN 預測頭，以多標籤/多藥物框架進行端到端的對齊與預測微調訓練。

#### 執行範例 (以回歸任務為例)
```bash
docker exec -it transDRP python drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --source_omics_path /workspace/DAPL-master/data_Winnie/CCLE_impact_hotspot.csv \
  --target_omics_path /workspace/DAPL-master/data_Winnie/TCGA_impact_hotspot.csv \
  --source_response_path /workspace/DAPL-master/data_Winnie/PRISM_drug_sensitivity.csv \
  --target_response_path /workspace/DAPL-master/data_Winnie/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_response_col neg_log2_auc \
  --source_cancer_type_path /workspace/DAPL-master/data_Winnie/CCLE_cancer_type.csv \
  --target_cancer_type_path /workspace/DAPL-master/data_Winnie/TCGA_cancer_type.csv \
  --drug_smiles_path /workspace/DAPL-master/data_Winnie/drug_smiles.csv \
  --pretrain_checkpoint outputs_transdrp_multilabel/pretrain/checkpoint.pt \
  --batch_size 512 \
  --epochs 300 \
  --n_splits 5 \
  --output_dir outputs_transdrp_multilabel \
  --device cuda
```

#### 參數說明

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `--task_type` | **[必填]** 預測任務類型，可選 `classification` (分類) 或 `regression` (回歸)。 | - |
| `--source_omics_path` | **[必填]** 來源域（細胞株）的體學特徵資料 CSV 路徑。 | - |
| `--target_omics_path` | **[必填]** 目標域（患者組織）的體學特徵資料 CSV 路徑。 | - |
| `--source_response_path` | **[必填]** 來源域藥物敏感性反應資料 CSV 路徑。 | - |
| `--target_response_path` | **[必填]** 目標域藥物敏感性反應資料 CSV 路徑。 | - |
| `--pretrain_checkpoint` | **[必填]** 第一階段預訓練生成的權重 Checkpoint 檔案路徑。 | - |
| `--output_dir` | **[必填]** 輸出結果、模型與交叉驗證折疊評估的儲存目錄路徑。 | - |
| `--source_sample_col` | 來源體學資料中的樣本 ID 欄位。 | `Sample_ID` |
| `--target_sample_col` | 目標體學資料中的樣本 ID 欄位。 | `tissue_id` |
| `--target_response_sample_col`| 目標藥物反應資料中的患者/組織 ID 欄位。 | `Patient_id` |
| `--drug_col` | 藥物名稱在敏感性反應資料中的欄位名稱。 | `drug_name` |
| `--source_response_col` | 來源藥物敏感性指標的欄位名稱 (例如 `Label` 或 `neg_log2_auc`)。 | `Label` |
| `--target_response_col` | 目標藥物敏感性指標的欄位名稱 (例如 `Label`)。 | `Label` |
| `--drug_smiles_path` | 藥物 SMILES 對照 CSV 路徑。若提供，則會利用 RDKit 計算 64 位元指紋並構建 GNN 的動態藥物相關圖。| `None` |
| `--source_cancer_type_path` | 來源域樣本的癌症類型對照 CSV 路徑。 | `None` |
| `--target_cancer_type_path` | 目標域樣本的癌症類型對照 CSV 路徑。 | `None` |
| `--cancer_type_col` | 癌症類型對照資料中的癌症類別欄位名稱。 | `Cancer_type` |
| `--metric` | 早停與模型評估的主要指標。若未指定，分類預設為 `macro_auroc`，回歸預設為 `macro_mae`。 | 載入任務對應預設值 |
| `--reg_loss` | 回歸任務之損失函數，目前僅支援 `mae`。 | `mae` |
| `--prediction_threshold` | 分類任務在計算特定指標時的預測閾值。 | `0.5` |
| `--regression_binary_threshold`| 回歸任務評估時用於特定二值化轉換的閾值。 | `1.0` |
| `--n_splits` | 交叉驗證的折數 (K-Fold)。 | `5` |
| `--source_test_size` | 來源域在評估時作為測試集的比例。 | `0.25` |
| `--epochs` | 訓練 Epoch 數。若未指定，則載入 `train_params.json` 中的 `uda_num_epochs`。 | 載入設定檔 |
| `--batch_size` | 訓練 Batch Size。若未指定，則載入 `train_params.json` 中的 `labeled.batch_size`。 | 載入設定檔 |
| `--lr` | 學習率。若未指定，則載入 `train_params.json` 中的 `labeled.lr`。 | 載入設定檔 |
| `--seed` | 隨機數種子。 | `2024` |
| `--alph` | 遷移學習對齊損失權重。若未指定，則載入 `train_params.json` 中的 `alph`。 | 載入設定檔 |
| `--beta` | 跨領域對比對齊損失權重。若未指定，則載入 `train_params.json` 中的 `beta`。 | 載入設定檔 |
| `--norm_flag` / `--no-norm_flag` | 是否對資料進行歸一化。 | 載入設定檔 |
| `--retrain_flag` / `--no-retrain_flag` | 是否強制重新訓練模型。 | `--retrain_flag` |
| `--device` | 指定運行設備。 | 自動偵測 |

## 授權條款 (License)
本專案基於 **MIT License** 授權開源。
