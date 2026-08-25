"""Tests for DRAVP potency parsing and the two-head masked objective."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from apexoracle.data.antiviral_peptide import (
    aggregate_metric,
    cross_check_dual_unit,
    parse_activity,
    to_micromolar,
)
from apexoracle.training.antiviral_peptide import (
    build_label_matrix,
    label_to_potency,
    masked_mse_loss,
    per_head_metrics,
    potency_to_label,
)


def test_mass_unit_is_converted_with_the_molecular_weight() -> None:
    # DRAVP states both readings for this peptide: 7.62 µg/ml is 5.0 µM.
    assert to_micromolar(7.62, "µg/ml", 1524.0) == pytest.approx(5.0, rel=1e-3)


def test_dravp_dual_unit_statement_validates_the_conversion() -> None:
    text = "inhibition of viral infection in Huh7.5.1 cells(IC50 = 7.62 μg/ml (5.0 μM))."
    assert cross_check_dual_unit(text, 1524.0) == pytest.approx(1.0, abs=0.05)


def test_molar_units_do_not_need_a_weight() -> None:
    assert to_micromolar(2.5, "nM", None) == pytest.approx(0.0025)
    assert to_micromolar(2.5, "mM", None) == pytest.approx(2500.0)


def test_mass_unit_without_a_weight_is_dropped_rather_than_guessed() -> None:
    assert to_micromolar(10.0, "µg/ml", None) is None
    assert to_micromolar(10.0, "µg/ml", "not a number") is None


def test_range_collapses_to_its_mean() -> None:
    records = parse_activity("(IC50=0.06-0.15 µM)", None)
    assert [r.value_micromolar for r in records] == [pytest.approx(0.105)]
    assert records[0].censoring == "none"


def test_right_censored_value_keeps_the_paper_multiplier() -> None:
    records = parse_activity("(IC50 > 100 µM)", None)
    # parse_concentration doubles a ">" reading, matching the MIC pipeline.
    assert records[0].value_micromolar == pytest.approx(200.0)
    assert records[0].censoring == "right"


def test_half_maximal_inhibition_statement_is_recovered_as_left_censored() -> None:
    records = parse_activity(">50% inhibition at 100 µM", None)
    assert len(records) == 1
    assert records[0].metric == "IC50"
    assert records[0].censoring == "left"
    assert records[0].from_inhibition is True
    assert records[0].value_micromolar == pytest.approx(100.0)


def test_weak_inhibition_statement_is_not_recovered() -> None:
    assert parse_activity("20% inhibition at 100 µM", None) == []


def test_explicit_potency_suppresses_the_inhibition_fallback() -> None:
    records = parse_activity("(IC50 = 4 µM), >90% inhibition at 100 µM", None)
    assert [r.from_inhibition for r in records] == [False]


def test_aggregation_is_geometric_so_it_averages_the_labels() -> None:
    combined = aggregate_metric([1.0, 100.0])
    assert combined == pytest.approx(10.0)
    assert potency_to_label(torch.tensor([combined])).item() == pytest.approx(
        potency_to_label(torch.tensor([1.0, 100.0])).mean().item()
    )


def test_label_transform_round_trips() -> None:
    values = torch.tensor([0.01, 1.0, 10.0, 1000.0])
    assert torch.allclose(label_to_potency(potency_to_label(values)), values, rtol=1e-5)


def test_ten_micromolar_maps_to_zero_so_zero_is_not_a_safe_fill_value() -> None:
    assert potency_to_label(torch.tensor([10.0])).item() == pytest.approx(0.0)


def test_label_matrix_carries_gaps_as_nan() -> None:
    frame = pd.DataFrame({"IC50_uM": [1.0, None], "EC50_uM": [None, 100.0]})
    labels = build_label_matrix(frame)
    assert labels.shape == (2, 2)
    assert math.isnan(labels[0, 1].item())
    assert math.isnan(labels[1, 0].item())


def test_masked_loss_ignores_unobserved_entries_entirely() -> None:
    labels = torch.tensor([[1.0, float("nan")], [float("nan"), 2.0]])
    logits = torch.zeros_like(labels)
    baseline = masked_mse_loss(logits, labels)

    polluted = logits.clone()
    polluted[0, 1] = 1e6
    polluted[1, 0] = -1e6
    assert torch.allclose(masked_mse_loss(polluted, labels).loss, baseline.loss)


def test_masked_loss_divides_by_observed_not_numel() -> None:
    labels = torch.tensor([[2.0, float("nan")]])
    logits = torch.zeros_like(labels)
    result = masked_mse_loss(logits, labels)
    assert int(result.observed) == 1
    # Dividing by numel would give 2.0; dividing by observed gives 4.0.
    assert result.loss.item() == pytest.approx(4.0)


def test_all_missing_batch_stays_differentiable() -> None:
    labels = torch.full((2, 2), float("nan"))
    logits = torch.zeros_like(labels, requires_grad=True)
    result = masked_mse_loss(logits, labels)
    assert int(result.observed) == 0
    result.loss.backward()
    assert logits.grad is not None


def test_per_head_metrics_report_counts_and_r2() -> None:
    labels = torch.tensor([[1.0, float("nan")], [2.0, float("nan")], [3.0, 5.0]])
    logits = labels.clone().nan_to_num()
    metrics = per_head_metrics(logits, labels)
    assert metrics["IC50_n"] == 3.0
    assert metrics["IC50_r2"] == pytest.approx(1.0)
    assert metrics["EC50_n"] == 1.0
    assert math.isnan(metrics["EC50_r2"])
