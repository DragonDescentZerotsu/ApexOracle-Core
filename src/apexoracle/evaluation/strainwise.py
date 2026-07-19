"""Metrics and ensemble helpers for strain-wise regression."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def calculate_r2(all_labels, all_preds) -> float:
    labels = np.array(all_labels)
    predictions = np.array(all_preds)
    ss_total = np.sum((labels - np.mean(labels)) ** 2)
    ss_residual = np.sum((labels - predictions) ** 2)
    return 1 - (ss_residual / ss_total)


def summarize_predictions(labels, predictions) -> dict[str, float]:
    return {
        "r2": calculate_r2(labels, predictions),
        "spearman": spearmanr(labels, predictions)[0],
        "pearson": pearsonr(labels, predictions)[0],
    }


def ensemble_predictions(predictions_by_member) -> np.ndarray:
    return np.mean(np.array(predictions_by_member), axis=0)


def specieswise_metrics(labels_by_species, predictions_by_species) -> dict:
    """Return the legacy ``[R2, MSE, Spearman, Pearson]`` list per species."""

    metrics = {}
    for species_name in predictions_by_species.keys():
        labels = labels_by_species[species_name]
        predictions = predictions_by_species[species_name]
        r2 = calculate_r2(labels, predictions)
        mse = np.mean((np.array(labels) - np.array(predictions)) ** 2)
        if len(labels) > 1:
            spearman = spearmanr(labels, predictions)[0]
            pearson = pearsonr(labels, predictions)[0]
        else:
            spearman = pearson = None
        metrics[species_name] = [r2, mse, spearman, pearson]
    return metrics
