from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from apexoracle.evaluation.paired_classification import (
    align_predictions,
    holm_adjust,
    paired_bootstrap,
    paired_randomization_test,
    percentile_interval,
)
from apexoracle.evaluation.fig1b_results import assemble_config, combine_fold_sources
from apexoracle.evaluation.fig1b_plot import build_plot_rows


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


def test_fig1b_plot_rows_require_all_groups_and_keep_one_baseline_per_group():
    results = []
    for group in range(3):
        for family, candidate in (
            ("fine_tune_vs_baseline", 0.7),
            ("strict_zero_shot_vs_baseline", 0.6),
        ):
            results.append(
                {
                    "family": family,
                    "group": group,
                    "metric": "auprc",
                    "candidate": candidate,
                    "candidate_95ci": [candidate - 0.1, candidate + 0.1],
                    "baseline": 0.5,
                    "baseline_95ci": [0.4, 0.6],
                }
            )
    frame = build_plot_rows({"results": results})
    assert len(frame) == 9
    assert frame.groupby("group").size().tolist() == [3, 3, 3]
    assert (frame["method"] == "Chemprop baseline (common folds)").sum() == 3


def test_fold_assembler_can_restrict_oof_to_a_reference_prediction_set(tmp_path):
    predictions = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d", "e", "f"],
            "label": [0, 1, 0, 0, 1, 0],
            "prediction": [0.1, 0.9, 0.2, 0.3, 0.8, 0.4],
        }
    )
    reference = predictions.iloc[:5]
    predictions.iloc[:3].to_csv(tmp_path / "fold_0.csv", index=False)
    predictions.iloc[3:].to_csv(tmp_path / "fold_1.csv", index=False)
    reference.to_csv(tmp_path / "reference.csv", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        "description: test\n"
        "groups:\n"
        "  - group: 0\n"
        "    reference_predictions: reference.csv\n"
        "    folds:\n"
        "      - fold: 0\n"
        "        sources: [fold_0.csv]\n"
        "      - fold: 1\n"
        "        sources: [fold_1.csv]\n",
        encoding="utf-8",
    )
    report = assemble_config(config, repo_root=tmp_path, output_root=tmp_path / "out")
    group = report["groups"][0]
    assert group["num_examples"] == 5
    assert group["num_examples_all_eligible"] == 6
    assert group["excluded_molecule_ids"] == ["f"]


def test_prediction_alignment_requires_explicit_candidate_superset(tmp_path):
    candidate = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c"],
            "label": [0, 1, 0],
            "prediction": [0.1, 0.9, 0.2],
        }
    )
    baseline = candidate.iloc[:2]
    candidate_path = tmp_path / "candidate.csv"
    baseline_path = tmp_path / "baseline.csv"
    candidate.to_csv(candidate_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    with pytest.raises(ValueError, match="molecule-ID sets differ"):
        align_predictions(candidate_path, baseline_path)
    aligned = align_predictions(
        candidate_path, baseline_path, candidate_may_be_superset=True
    )
    assert aligned.molecule_id.tolist() == ["a", "b"]
    assert aligned.attrs["excluded_candidate_molecule_ids"] == ["c"]
