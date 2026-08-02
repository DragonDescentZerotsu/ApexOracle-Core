from __future__ import annotations

import pandas as pd
import pytest

from apexoracle.evaluation.hierarchical_mic_molecule_overlap import (
    IDENTITY_DBAASP_ID,
    IDENTITY_MODEL_INPUT,
    aggregate_group_summaries,
    apply_legacy_token_length_filter,
    concatenate_routes,
    partition_test_by_train_molecules,
    summarize_overlap,
)


def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["DBAASP_id", "strain_name", "SMILES", "MIC"])


def test_overlap_uses_union_of_routes_and_exact_model_input_identity():
    genome_train = frame([(1, "train-a", "[1, 2]", 8.0)])
    text_train = frame([(2, "train-b", "[3, 4]", 32.0)])
    test = frame(
        [
            (1, "test-a", "[1,2]", 4.0),
            # Different database ID but the exact same model input as ID 2.
            (999, "test-a", "[3, 4]", 16.0),
            (3, "test-b", "[5, 6]", 64.0),
        ]
    )
    train = concatenate_routes([genome_train, text_train])

    by_id = summarize_overlap(
        train, test, identity_definition=IDENTITY_DBAASP_ID
    )
    by_input = summarize_overlap(
        train, test, identity_definition=IDENTITY_MODEL_INPUT
    )
    assert by_id["test_train_seen_measurement_instances"] == 1
    assert by_input["test_train_seen_measurement_instances"] == 2
    _, _, unseen = partition_test_by_train_molecules(
        train, test, identity_definition=IDENTITY_MODEL_INPUT
    )
    assert unseen["DBAASP_id"].tolist() == [3]


def test_legacy_length_filter_matches_512_token_boundary():
    eligible = list(range(512))
    too_long = list(range(513))
    data = frame(
        [
            (1, "a", str(eligible), 8.0),
            (2, "b", str(too_long), 8.0),
        ]
    )
    filtered = apply_legacy_token_length_filter(data)
    assert filtered["DBAASP_id"].tolist() == [1]


def test_summary_counts_measurements_and_unique_molecules_separately():
    train = frame([(1, "train", "[1]", 8.0)])
    test = frame(
        [
            (2, "test-a", "[2]", 8.0),
            (2, "test-b", "[2]", 32.0),
            (1, "test-c", "[1]", 64.0),
        ]
    )
    summary = summarize_overlap(
        train,
        test,
        identity_definition=IDENTITY_DBAASP_ID,
        low_mic_threshold_um=16.0,
    )
    assert summary["test_measurement_instances"] == 3
    assert summary["test_train_unseen_measurement_instances"] == 2
    assert summary["test_train_unseen_unique_molecules"] == 1
    assert summary["test_train_unseen_low_mic_measurement_instances"] == 1
    assert summary["test_train_unseen_low_mic_fraction"] == pytest.approx(0.5)


def test_group_aggregation_uses_total_numerator_and_denominator():
    first = {
        "identity_definition": IDENTITY_DBAASP_ID,
        "train_measurement_instances": 10,
        "test_measurement_instances": 10,
        "test_train_seen_measurement_instances": 9,
        "test_train_unseen_measurement_instances": 1,
        "train_unique_molecules": 4,
        "test_unique_molecules": 4,
        "test_train_seen_unique_molecules": 3,
        "test_train_unseen_unique_molecules": 1,
        "test_pathogens": 2,
        "test_train_unseen_pathogens": 1,
        "test_train_unseen_low_mic_threshold_um": 16.0,
        "test_train_unseen_low_mic_measurement_instances": 1,
    }
    second = {
        **first,
        "test_measurement_instances": 90,
        "test_train_seen_measurement_instances": 45,
        "test_train_unseen_measurement_instances": 45,
        "test_train_unseen_low_mic_measurement_instances": 9,
    }
    total = aggregate_group_summaries([first, second])
    assert total["test_train_seen_fraction"] == pytest.approx(54 / 100)
    assert total["test_train_unseen_low_mic_fraction"] == pytest.approx(10 / 46)
