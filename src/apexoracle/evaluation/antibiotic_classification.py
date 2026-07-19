"""Metrics and best-checkpoint tracking for three-strain classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(labels, logits) -> dict[str, float]:
    labels_array = np.asarray(labels)
    logits_array = np.asarray(logits)
    return {
        "auroc": float(roc_auc_score(labels_array, logits_array)),
        "auprc": float(average_precision_score(labels_array, logits_array)),
    }


def ensemble_classification_predictions(predictions_by_member) -> np.ndarray:
    return np.mean(np.asarray(predictions_by_member), axis=0)


@dataclass
class LegacyClassificationBestTracker:
    """Preserve strict AUROC selection and the historical AUPRC-save ordering.

    The legacy drivers write a checkpoint immediately after an AUROC
    improvement and only then update ``best_auprc``. Consequently, the AUPRC
    stored inside a checkpoint may lag the metric reported for that same epoch.
    Callers use ``checkpoint_auprc`` before ``finish_epoch`` to keep that exact
    contract visible and testable.
    """

    best_auroc: float = -10.0
    best_auprc: float = -10.0
    best_predictions: list[float] | None = None

    def update_auroc(self, *, auroc: float, predictions: list[float]) -> bool:
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.best_predictions = predictions
            return True
        return False

    @property
    def checkpoint_auprc(self) -> float:
        return self.best_auprc

    def finish_epoch(self, *, auprc: float) -> None:
        if auprc > self.best_auprc:
            self.best_auprc = auprc
