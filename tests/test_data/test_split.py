import pytest
import numpy as np
from transdrp_multilabel.data.split import split_source_samples

def test_split_source_samples():
    sample_ids = [f"S{i}" for i in range(20)]
    # Random response labels and mask
    y = np.random.choice([0.0, 1.0], size=(20, 5))
    mask = np.ones((20, 5))

    folds = split_source_samples(
        sample_ids,
        y,
        mask,
        test_size=0.2,
        n_splits=3,
        seed=42
    )

    assert len(folds) == 3
    for f in folds:
        assert len(f.train_sample_ids) > 0
        assert len(f.val_sample_ids) > 0
        assert len(f.test_sample_ids) == 4  # 20 * 0.2 = 4
        # Assert no intersection
        train_s = set(f.train_sample_ids)
        val_s = set(f.val_sample_ids)
        test_s = set(f.test_sample_ids)
        assert len(train_s & val_s) == 0
        assert len(train_s & test_s) == 0
        assert len(val_s & test_s) == 0
