from __future__ import annotations

import numpy as np

from apexoracle.evaluation.generated_candidate_diversity import (
    REPLACEMENT_RULES,
    best_topology_aware_sequence_alignment,
    histogram_rows,
    match_replacement_rule,
    normalize_legacy_arg,
    sample_distinct_ordered_pairs,
    summarize_similarities,
    topology_aware_equal,
)


def test_legacy_arg_normalization_and_replacement_mapping() -> None:
    assert normalize_legacy_arg("MLNIQBKLBL") == "MLNIQRKLRL"
    rule = match_replacement_rule("MIKLLIKLAIGYLRLQBGQPLLNPGKGAB")
    assert rule is not None
    assert rule.apexoracle_id == "ApexOracle-5"
    assert len(REPLACEMENT_RULES) == 8


def test_cyclic_matching_accepts_rotation_but_not_linearization() -> None:
    assert topology_aware_equal("cyclo-ABC", "cyclo-BCA")
    assert not topology_aware_equal("ABC", "cyclo-BCA")
    assert not topology_aware_equal("cyclo-ABC", "cyclo-ACB")


def test_selected_peptide_sequence_similarity_preserves_topology() -> None:
    assert (
        best_topology_aware_sequence_alignment(
            "AAAA",
            "AAAA",
            left_is_cyclic=False,
            right_is_cyclic=True,
        )
        is None
    )
    linear = best_topology_aware_sequence_alignment(
        "ACDE",
        "ACDE",
        left_is_cyclic=False,
        right_is_cyclic=False,
    )
    assert linear is not None
    assert linear.metrics.pid == 1.0
    cyclic = best_topology_aware_sequence_alignment(
        "ACDE",
        "DEAC",
        left_is_cyclic=True,
        right_is_cyclic=True,
    )
    assert cyclic is not None
    assert cyclic.metrics.pid == 1.0
    assert cyclic.metrics.matches == 4
    asymmetric_tie_left = "KMLNQNNKEGRLNLRLSILSTLRRGKLLLV"
    asymmetric_tie_right = "MNLAAFFIFKNPPSKWKYKR"
    forward = best_topology_aware_sequence_alignment(
        asymmetric_tie_left,
        asymmetric_tie_right,
        left_is_cyclic=False,
        right_is_cyclic=False,
    )
    reverse = best_topology_aware_sequence_alignment(
        asymmetric_tie_right,
        asymmetric_tie_left,
        left_is_cyclic=False,
        right_is_cyclic=False,
    )
    assert forward is not None and reverse is not None
    assert forward.metrics.pid == reverse.metrics.pid


def test_pair_sampling_is_reproducible_and_excludes_self_pairs() -> None:
    left_a, right_a = sample_distinct_ordered_pairs(73, 1_000, 20260802)
    left_b, right_b = sample_distinct_ordered_pairs(73, 1_000, 20260802)
    assert np.array_equal(left_a, left_b)
    assert np.array_equal(right_a, right_b)
    assert np.all(left_a != right_a)
    assert np.all((right_a >= 0) & (right_a < 73))


def test_similarity_summary_and_histogram_preserve_denominator() -> None:
    values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    summary = summarize_similarities(values)
    assert summary["n_pairs"] == 5
    assert summary["median"] == 0.5
    assert summary["fraction_ge_0_5"] == 0.6
    rows = histogram_rows(values)
    assert sum(int(row["count"]) for row in rows) == 5
    assert np.isclose(sum(float(row["fraction"]) for row in rows), 1.0)
