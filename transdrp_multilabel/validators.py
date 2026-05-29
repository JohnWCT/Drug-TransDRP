import pandas as pd
import numpy as np
from typing import List
from transdrp_multilabel.contracts import DrugIndex, ResponseMatrix, SourceFold

def validate_omics_table(df: pd.DataFrame, sample_id_col: str) -> None:
    if sample_id_col not in df.columns:
        raise ValueError(f"Sample ID column '{sample_id_col}' not found in omics table.")
    if df[sample_id_col].duplicated().any():
        raise ValueError(f"Duplicate sample IDs found in omics table under '{sample_id_col}'.")

    # Check that all other columns are numeric
    non_numeric = []
    for col in df.columns:
        if col == sample_id_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(f"Non-numeric columns found in omics: {non_numeric[:5]}")

def validate_response_long_table(
    df: pd.DataFrame,
    sample_id_col: str,
    drug_col: str,
    response_col: str,
    task_type: str,
    domain: str,
) -> None:
    for col in [sample_id_col, drug_col, response_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in response table.")

    # Check data types
    if df[drug_col].isnull().any():
        raise ValueError(f"Null values found in drug column '{drug_col}'.")
    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"Response column '{response_col}' must be numeric.")

    # Check binary label constraint for classification, or target domain regardless of task_type
    if task_type == "classification" or domain == "target":
        unique_vals = df[response_col].dropna().unique()
        invalid_vals = [v for v in unique_vals if v not in (0, 1, 0.0, 1.0)]
        if invalid_vals:
            raise ValueError(
                f"Binary response expected for classification or target domain, but found invalid values: {invalid_vals[:5]}"
            )

def validate_drug_index(drug_index: DrugIndex) -> None:
    if not drug_index.drug_ids:
        raise ValueError("Drug index cannot be empty.")
    if len(drug_index.drug_ids) != len(drug_index.drug_to_index):
        raise ValueError("Mismatch in drug index counts.")
    for idx, drug in enumerate(drug_index.drug_ids):
        if drug_index.drug_to_index.get(drug) != idx:
            raise ValueError(f"Invalid drug mapping for drug '{drug}'. Expected index {idx}.")

def validate_response_matrix(matrix: ResponseMatrix) -> None:
    if matrix.y.shape != matrix.mask.shape:
        raise ValueError(f"Shape mismatch: y shape {matrix.y.shape} vs mask shape {matrix.mask.shape}.")
    if not np.all(np.isin(matrix.mask, [0, 1])):
        raise ValueError("Mask must contain only binary 0 or 1 values.")
    if matrix.y.shape[1] != len(matrix.drug_index.drug_ids):
        raise ValueError(
            f"Matrix column count {matrix.y.shape[1]} does not match drug index count {len(matrix.drug_index.drug_ids)}."
        )
    if matrix.y.shape[0] != len(matrix.sample_ids):
        raise ValueError(
            f"Matrix row count {matrix.y.shape[0]} does not match sample count {len(matrix.sample_ids)}."
        )

def validate_folds(folds: List[SourceFold]) -> None:
    if not folds:
        raise ValueError("Source folds list cannot be empty.")
    for fold in folds:
        train_set = set(fold.train_sample_ids)
        val_set = set(fold.val_sample_ids)
        test_set = set(fold.test_sample_ids)

        # Check intersections
        train_val = train_set.intersection(val_set)
        train_test = train_set.intersection(test_set)
        val_test = val_set.intersection(test_set)

        if train_val:
            raise ValueError(f"Fold {fold.fold_id}: Overlap between train and validation samples: {list(train_val)[:5]}")
        if train_test:
            raise ValueError(f"Fold {fold.fold_id}: Overlap between train and test samples: {list(train_test)[:5]}")
        if val_test:
            raise ValueError(f"Fold {fold.fold_id}: Overlap between validation and test samples: {list(val_test)[:5]}")
