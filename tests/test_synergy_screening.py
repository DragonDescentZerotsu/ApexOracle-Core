from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from apexoracle.evaluation.synergy_screening import (
    ScreeningConfig,
    inverse_fici_target,
    legacy_ddp_rank_block_order,
    select_legacy_positional_rows,
    split_legacy_rank_frames,
    validate_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/synergy/legacy_screening.yaml"


def test_screening_profiles_freeze_safe_outputs_and_historical_row_gap():
    fixed = ScreeningConfig.load(
        CONFIG,
        REPO_ROOT,
        profile="DBAASP_train_best",
    )
    best = ScreeningConfig.load(
        CONFIG,
        REPO_ROOT,
        profile="inhouse_best",
    )
    assert len(fixed.checkpoint_paths) == len(best.checkpoint_paths) == 7
    assert fixed.smiles_rows == 5_918_520
    assert fixed.sequence_rows == 5_949_525
    assert fixed.sequence_rows - fixed.smiles_rows == 31_005
    assert fixed.historical_world_size == 4
    assert fixed.output_path.parent == REPO_ROOT / "results/synergy_screening"
    assert fixed.output_path != fixed.observed_output
    assert fixed.observed_output_rows == 2_476_932
    assert best.observed_output_rows == 934_201
    assert validate_inputs(fixed, verify_hashes=False) == {}


def test_screening_refuses_outputs_inside_original_data_tree():
    config = ScreeningConfig.load(
        CONFIG,
        REPO_ROOT,
        profile="DBAASP_train_best",
    )
    unsafe = replace(
        config,
        output_path=REPO_ROOT / "DataPrepare/Data/unsafe.csv",
    )
    with pytest.raises(ValueError, match="original data"):
        unsafe.validate(REPO_ROOT)


def test_inverse_fici_target_matches_legacy_formula():
    logits = torch.tensor([0.0, 1.0, 2.0])
    torch.testing.assert_close(
        inverse_fici_target(logits),
        torch.tensor([10.0, 1.0, 0.1]),
    )


def test_legacy_positional_selection_preserves_historical_misalignment():
    sequences = pd.DataFrame(
        [
            [1, 2, "NA", "A", "B", -1.0],
            [1, 3, "NA", "A", "filtered-out", -1.0],
            [1, 4, "NA", "A", "C", -1.0],
            [1, 5, "NA", "A", "D", -1.0],
        ],
        columns=(
            "DBAASP_id",
            "antibio_id_or_name",
            "strain_name",
            "AMP_smiles",
            "antibiotic_smiles",
            "FICI",
        ),
    )
    # These predictions conceptually correspond to pairs (1,2), (1,4), (1,5),
    # but the old code selected sequence rows with the shorter table's positions.
    predictions = np.asarray([0.8, 0.4, 0.3])
    selected = select_legacy_positional_rows(
        sequences,
        predictions,
        threshold=0.5,
    )
    assert selected["antibio_id_or_name"].tolist() == [3, 4]
    assert selected["FICI"].tolist() == [0.4, 0.3]


def test_legacy_ddp_rank_block_order_preserves_four_rank_gather_bug():
    predictions = np.arange(12)
    reordered = legacy_ddp_rank_block_order(predictions, world_size=4)
    assert reordered.tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    frame = pd.DataFrame({"value": predictions})
    rank_frames = split_legacy_rank_frames(frame, world_size=4)
    assert [part["value"].tolist() for part in rank_frames] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]
