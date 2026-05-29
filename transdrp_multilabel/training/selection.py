import math
from typing import Optional, Tuple
import pandas as pd

CLASSIFICATION_PRIMARY = "macro_auroc"
CLASSIFICATION_FALLBACKS = ("macro_aupr", "macro_balanced_accuracy", "macro_f1")
REGRESSION_PRIMARY = "macro_mae"

class MetricSelector:
    def __init__(self, task_type: str, requested_metric: Optional[str] = None) -> None:
        self.task_type = task_type
        self.requested_metric = requested_metric
        self._fallback_reason: Optional[str] = None

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    def select_metric(self, metrics_summary: pd.DataFrame) -> Tuple[str, float, str]:
        if metrics_summary.empty:
            raise ValueError("metrics summary is empty")

        if "metric_name" in metrics_summary.columns:
            names = metrics_summary["metric_name"].astype(str)
            values = metrics_summary["metric_value"].astype(float)
        else:
            row = metrics_summary.iloc[0]
            names = pd.Series([str(row.get("metric", row.name))])
            values = pd.Series([float(row.get("macro", row.get("metric_value", float("nan"))))])

        candidates = []
        if self.requested_metric:
            candidates.append(self.requested_metric)
        if self.task_type == "classification":
            candidates.extend([CLASSIFICATION_PRIMARY, *CLASSIFICATION_FALLBACKS])
        else:
            candidates.extend([REGRESSION_PRIMARY, "macro_rmse"])

        for name in candidates:
            if name in names.values:
                idx = names[names == name].index[0]
                val = float(values.loc[idx])
                if not math.isnan(val):
                    direction = self._direction(name)
                    return name, val, direction
                self._fallback_reason = f"{name} is NaN"

        for name, val in zip(names, values):
            if not math.isnan(float(val)):
                direction = self._direction(str(name))
                return str(name), float(val), direction
        raise ValueError("No valid metric found in metrics summary.")

    def is_better(self, new_value: float, best_value: Optional[float], metric_name: str) -> bool:
        if best_value is None or math.isnan(best_value):
            return True
        if math.isnan(new_value):
            return False
        if self._direction(metric_name) == "higher":
            return new_value > best_value
        return new_value < best_value

    def _direction(self, metric_name: str) -> str:
        lower_better = ("mae", "rmse", "loss")
        if any(k in metric_name.lower() for k in lower_better):
            return "lower"
        return "higher"
