import pytest
import pandas as pd
from transdrp_multilabel.training.selection import MetricSelector

def test_metric_selector_classification():
    sel = MetricSelector(task_type="classification", requested_metric=None)

    summary = pd.DataFrame([
        {"metric_name": "macro_auroc", "metric_value": 0.7, "aggregation": "macro"}
    ])

    name, val, direction = sel.select_metric(summary)
    assert name == "macro_auroc"
    assert val == 0.7
    assert direction == "higher"

    assert sel.is_better(0.8, 0.7, name)
    assert not sel.is_better(0.6, 0.7, name)
    assert sel.is_better(0.7, None, name)

def test_metric_selector_regression():
    sel = MetricSelector(task_type="regression", requested_metric="macro_mae")

    summary = pd.DataFrame([
        {"metric_name": "macro_mae", "metric_value": 1.5, "aggregation": "macro"}
    ])

    name, val, direction = sel.select_metric(summary)
    assert name == "macro_mae"
    assert val == 1.5
    assert direction == "lower"

    assert sel.is_better(1.2, 1.5, name)
    assert not sel.is_better(1.7, 1.5, name)
