"""Shared paper-compatible loss and metric helpers for Fig. 2b runners."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class LegacyMaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked_loss = (prediction - target).square() * mask
        if self.reduction == "mean":
            return masked_loss.sum() / (mask.sum() + 1e-8)
        if self.reduction == "sum":
            return masked_loss.sum()
        return masked_loss


def legacy_r2_per_task(
    labels: np.ndarray, predictions: np.ndarray, masks: np.ndarray
) -> list[float | None]:
    """Preserve the original per-task R2 behavior, including degenerate NaN."""

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    masks = np.asarray(masks)
    values: list[float | None] = []
    for task_index in range(labels.shape[1]):
        observed = masks[:, task_index].astype(bool)
        y_true = labels[observed, task_index]
        y_pred = predictions[observed, task_index]
        if len(y_true) == 0:
            values.append(None)
            continue
        total = np.sum((y_true - np.mean(y_true)) ** 2)
        residual = np.sum((y_true - y_pred) ** 2)
        values.append(float(1 - residual / total))
    return values


def finite_mean_or_nan(values: Iterable[float | None]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")
