"""Metrics and ensemble helpers shared by hierarchical MIC holdouts."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class HierarchicalMicPredictionAccumulator:
    """Preserve legacy loss/prediction partitions and species insertion order."""

    losses: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    genome_text_losses: list[float] = field(default_factory=list)
    genome_text_labels: list[float] = field(default_factory=list)
    genome_text_predictions: list[float] = field(default_factory=list)
    text_only_losses: list[float] = field(default_factory=list)
    text_only_labels: list[float] = field(default_factory=list)
    text_only_predictions: list[float] = field(default_factory=list)
    species_labels: dict[str, list[float]] = field(default_factory=dict)
    species_predictions: dict[str, list[float]] = field(default_factory=dict)
    baseline_predictions: list[float] = field(default_factory=list)

    def add_batch(
        self,
        result,
        *,
        has_genome: bool,
        atcc_id_to_species: dict,
        original_strain_to_species: dict,
        baseline_mean: float | None = None,
    ) -> None:
        loss = result.loss.item()
        labels = result.labels.detach().cpu().flatten().tolist()
        predictions = result.logits.detach().cpu().flatten().tolist()
        self.losses.append(loss)
        self.labels.extend(labels)
        self.predictions.extend(predictions)
        if has_genome:
            self.genome_text_losses.append(loss)
            self.genome_text_labels.extend(labels)
            self.genome_text_predictions.extend(predictions)
        else:
            self.text_only_losses.append(loss)
            self.text_only_labels.extend(labels)
            self.text_only_predictions.extend(predictions)
        if baseline_mean is not None:
            self.baseline_predictions.extend(
                np.full(
                    result.logits.detach().cpu().flatten().shape, baseline_mean
                ).tolist()
            )

        for strain_name, label, prediction in zip(
            result.strain_names, labels, predictions
        ):
            species_name = atcc_id_to_species.get(strain_name, None)
            if species_name is None:
                species_name = original_strain_to_species[strain_name]
            if species_name not in self.species_predictions:
                self.species_predictions[species_name] = [prediction]
                self.species_labels[species_name] = [label]
            else:
                self.species_predictions[species_name].append(prediction)
                self.species_labels[species_name].append(label)


def summarize_partition_or_sentinel(
    labels, predictions, *, sentinel: float = -1000
) -> dict[str, float]:
    if len(labels) <= 1:
        return {"r2": sentinel, "spearman": sentinel, "pearson": sentinel}
    return summarize_predictions(labels, predictions)


@dataclass
class LegacyBestMetricTracker:
    """Track paper-era strict improvements without copying prediction lists."""

    best_r2: float = -10
    best_spearman: float = -10
    best_pearson: float = -10
    best_predictions: list[float] | None = None

    def update(
        self,
        *,
        r2: float,
        spearman: float,
        pearson: float,
        predictions: list[float],
    ) -> bool:
        r2_improved = r2 > self.best_r2
        if r2_improved:
            self.best_r2 = r2
            self.best_predictions = predictions
        if spearman > self.best_spearman:
            self.best_spearman = spearman
        if pearson > self.best_pearson:
            self.best_pearson = pearson
        return r2_improved


StrainwisePredictionAccumulator = HierarchicalMicPredictionAccumulator
