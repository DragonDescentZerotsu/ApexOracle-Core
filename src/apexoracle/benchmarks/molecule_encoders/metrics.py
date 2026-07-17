"""Shared masked metrics for the Fig. 2b multi-task MIC benchmark."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def masked_r2_per_task(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_mask: np.ndarray,
) -> list[Optional[float]]:
    """Compute R2 independently per task, returning None when undefined."""

    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    if labels.shape != predictions.shape or labels.shape != mask.shape:
        raise ValueError(
            "labels, predictions and label_mask must have identical shapes; "
            f"received {labels.shape}, {predictions.shape}, {mask.shape}"
        )
    if labels.ndim != 2:
        raise ValueError("metric inputs must be 2D arrays")

    values: list[Optional[float]] = []
    for task_index in range(labels.shape[1]):
        observed = mask[:, task_index]
        y_true = labels[observed, task_index]
        y_pred = predictions[observed, task_index]
        if len(y_true) < 2:
            values.append(None)
            continue
        ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_total == 0:
            values.append(None)
            continue
        ss_residual = float(np.sum((y_true - y_pred) ** 2))
        values.append(1.0 - ss_residual / ss_total)
    return values


def finite_macro_mean(values: Sequence[Optional[float]]) -> float:
    """Average defined finite task metrics with an explicit empty check."""

    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        raise ValueError("no finite task metrics are available")
    return float(np.mean(finite))
