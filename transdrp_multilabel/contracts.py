from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, List, Dict, Any, Set
import pandas as pd
import numpy as np

@dataclass
class TransDRPMultilabelConfig:
    task_type: Literal["classification", "regression"]

    source_omics_path: str
    target_omics_path: str
    source_response_path: Optional[str]
    target_response_path: Optional[str]

    source_sample_col: str
    target_sample_col: str
    target_response_sample_col: str
    drug_col: str
    source_response_col: str
    target_response_col: str

    method: str
    pretrain_checkpoint: Optional[str]
    output_dir: str
    overwrite: bool

    batch_size: int
    epochs: int
    lr: float
    seed: int
    n_splits: int
    source_test_size: float

    metric: Optional[str]
    reg_loss: Literal["mae"]
    prediction_threshold: float
    regression_binary_threshold: float

    source_cancer_type_path: Optional[str]
    target_cancer_type_path: Optional[str]
    cancer_type_col: Optional[str]

    drug_smiles_path: Optional[str]

    # TransDRP specific hyperparameters
    alph: float
    beta: float
    latent_dim: int
    encoder_hidden_dims: Tuple[int, ...]
    decoder_hidden_dims: Tuple[int, ...]
    classifier_hidden_dims: Tuple[int, ...]
    drop: float
    norm_flag: bool
    retrain_flag: bool
    device: str
    threshold_label: float = 0.1

    # Target eval response paths (optional; primary falls back to target_response_path)
    target_eval_primary_response_path: Optional[str] = None  # see config.DEFAULT_TARGET_EVAL_*
    target_eval_aux_response_path: Optional[str] = None
    target_eval_target_only_response_path: Optional[str] = None

    target_eval_primary_name: str = "primary"
    target_eval_aux_name: str = "auxiliary"
    target_eval_target_only_name: str = "target_only"

    drug_graph_edge_strategy: str = "hybrid"
    drug_graph_similarity_k: int = 3
    drug_graph_similarity_threshold: float = 0.3
    drug_graph_force_top1_if_isolated: bool = True

@dataclass
class OmicsTable:
    x: pd.DataFrame
    sample_ids: List[str]
    feature_names: List[str]
    domain: Literal["source", "target"]

@dataclass
class DrugIndex:
    drug_ids: List[str]
    drug_to_index: Dict[str, int]
    index_to_drug: Dict[int, str]

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)

@dataclass
class ResponseMatrix:
    y: np.ndarray
    mask: np.ndarray
    sample_ids: List[str]
    drug_index: DrugIndex
    domain: Literal["source", "target"]
    label_semantics: Literal["binary", "continuous"]

@dataclass
class PreparedPretrainData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    feature_alignment: pd.DataFrame

@dataclass
class TargetEvalDataset:
    name: str
    response: ResponseMatrix
    raw_response: pd.DataFrame
    report: pd.DataFrame

@dataclass
class SourceFold:
    fold_id: int
    train_sample_ids: List[str]
    val_sample_ids: List[str]
    test_sample_ids: List[str]

@dataclass
class PreparedFineTuneData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    source_response: ResponseMatrix
    target_response: ResponseMatrix
    drug_index: DrugIndex
    folds: List[SourceFold]
    cancer_type_table: Optional[pd.DataFrame]
    target_eval_datasets: Dict[str, TargetEvalDataset]
    source_drug_ids: Set[str]
    drug_availability: Optional[pd.DataFrame] = None

@dataclass
class TrainingResult:
    fold_id: int
    best_model_path: str
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    train_log: pd.DataFrame

@dataclass
class PredictionBundle:
    source_predictions: pd.DataFrame
    target_predictions: pd.DataFrame
    source_metrics_per_drug: pd.DataFrame
    target_metrics_per_drug: pd.DataFrame
    source_metrics_summary: pd.DataFrame
    target_metrics_summary: pd.DataFrame
