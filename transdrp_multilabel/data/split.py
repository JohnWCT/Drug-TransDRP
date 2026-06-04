import numpy as np
from sklearn.model_selection import KFold, train_test_split
from transdrp_multilabel.contracts import SourceFold

def split_source_samples(
    sample_ids: list[str],
    y_source: np.ndarray,
    mask_source: np.ndarray,
    test_size: float,
    n_splits: int,
    seed: int,
) -> list[SourceFold]:
    n = len(sample_ids)
    indices = np.arange(n)
    pseudo = np.zeros(n, dtype=int)
    for i in range(n):
        obs = mask_source[i] > 0
        if obs.any():
            pseudo[i] = int(np.round(y_source[i, obs].mean()) >= 0.5)

    eff_test_size = test_size
    if n > 0:
        min_test = max(1, int(round(n * test_size)))
        if min_test >= n:
            min_test = max(1, n // 10)
        eff_test_size = min_test / n

    try:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed, stratify=pseudo
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed
        )

    test_ids = [sample_ids[int(i)] for i in sorted(test_idx)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[SourceFold] = []

    for fold_id, (tr, va) in enumerate(kf.split(train_val_idx)):
        train_ids = [sample_ids[int(train_val_idx[i])] for i in tr]
        val_ids = [sample_ids[int(train_val_idx[i])] for i in va]
        folds.append(
            SourceFold(
                fold_id=fold_id,
                train_sample_ids=train_ids,
                val_sample_ids=val_ids,
                test_sample_ids=test_ids,
            )
        )
    return folds
