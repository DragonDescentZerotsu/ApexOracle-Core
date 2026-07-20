from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from apexoracle.evaluation.paired_classification import (
    holm_adjust,
    paired_bootstrap,
    paired_randomization_test,
    percentile_interval,
)
from apexoracle.evaluation.fig1b_results import combine_fold_sources


def test_paired_resampling_detects_a_clear_ordering():
    labels = np.array([0, 1] * 20)
    candidate = np.where(labels == 1, 0.9, 0.1)
    baseline = np.linspace(0.0, 1.0, len(labels))
    samples = paired_bootstrap(
        labels,
        candidate,
        baseline,
        metric=average_precision_score,
        iterations=100,
        seed=42,
    )
    assert percentile_interval(samples["difference"])[0] > 0
    assert paired_randomization_test(
        labels,
        candidate,
        baseline,
        metric=average_precision_score,
        iterations=199,
        seed=42,
    ) <= 0.05


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust([0.03, 0.001, 0.02])
    assert adjusted == pytest.approx([0.04, 0.003, 0.04])


def test_fold_assembler_aligns_ids_and_averages_unique_members(tmp_path):
    first = pd.DataFrame(
        {
            "molecule_id": ["b", "a", "c", "d"],
            "label": [1, 0, 1, 0],
            "prediction": [0.8, 0.2, 0.7, 0.1],
            "prediction_ensemble_0": [0.8, 0.2, 0.7, 0.1],
        }
    )
    second = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "label": [0, 1, 1, 0],
            "prediction": [0.4, 0.6, 0.9, 0.3],
            "prediction_ensemble_1": [0.4, 0.6, 0.9, 0.3],
        }
    )
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    first.to_csv(paths[0], index=False)
    second.to_csv(paths[1], index=False)
    combined, report = combine_fold_sources(paths, group=0, fold=0)
    assert combined.molecule_id.tolist() == ["a", "b", "c", "d"]
    assert combined.prediction.tolist() == pytest.approx([0.3, 0.7, 0.8, 0.2])
    assert report["num_members"] == 2
