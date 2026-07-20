from __future__ import annotations

from pathlib import Path

import pandas as pd

from apexoracle.benchmarks.fig1b_baselines import (
    _feature_arguments,
    build_target_table,
    load_prepared_folds,
    prepare_common_folds,
)


def _config(tmp_path: Path) -> dict:
    tokenized = pd.DataFrame(
        {
            "DBAASP_id": [f"m{i}" for i in range(10)] + ["too-long"],
            "strain_name": ["target"] * 11,
            "SMILES": ["[1, 2]"] * 10 + ["[1, 2, 3, 4]"],
            "MIC": [0, 1] * 5 + [1],
        }
    )
    canonical = pd.DataFrame(
        {
            "DBAASP_id": tokenized["DBAASP_id"],
            "SMILES": [f"C{'C' * i}" for i in range(11)],
            "MIC": tokenized["MIC"],
        }
    )
    tokenized.to_csv(tmp_path / "tokenized.csv", index=False)
    canonical.to_csv(tmp_path / "canonical.csv", index=False)
    return {
        "protocol": "fig1b_common_outer_folds_chemprop_sensitivity",
        "tokenized_records": "tokenized.csv",
        "max_token_length": 3,
        "outer_folds": 5,
        "outer_seed": 42,
        "validation_fraction_of_outer_train": 0.25,
        "validation_seed": 4200,
        "targets": [
            {
                "group": 0,
                "strain": "target",
                "display_name": "Target",
                "records": "canonical.csv",
                "profile": "test",
            }
        ],
    }


def test_target_table_restores_smiles_and_applies_original_token_filter(tmp_path):
    table, metadata = build_target_table(tmp_path, _config(tmp_path), 0)
    assert table["molecule_id"].tolist() == [f"m{i}" for i in range(10)]
    assert table["smiles"].str.startswith("C").all()
    assert metadata["num_source_rows"] == 11
    assert metadata["num_eligible_rows"] == 10


def test_common_folds_cover_every_id_once_and_keep_test_out_of_validation(tmp_path):
    output = tmp_path / "output"
    report = prepare_common_folds(
        tmp_path, _config(tmp_path), group=0, output_dir=output
    )
    folds = pd.read_csv(output / "folds.csv")
    assert sorted(folds["fold"].value_counts().tolist()) == [2, 2, 2, 2, 2]
    assert sum(item["test"] for item in report["folds"]) == 10
    for fold in range(5):
        test_ids = set(pd.read_csv(output / f"fold_{fold}/test_manifest.csv").molecule_id)
        train_ids = set(pd.read_csv(output / f"fold_{fold}/train_manifest.csv").molecule_id)
        val_ids = set(
            pd.read_csv(output / f"fold_{fold}/validation_manifest.csv").molecule_id
        )
        assert not test_ids & train_ids
        assert not test_ids & val_ids
        assert not train_ids & val_ids


def test_feature_arguments_preserve_paper_specific_baseline_contract():
    assert _feature_arguments({"features_generator": None}) == []
    assert _feature_arguments(
        {
            "features_generator": "rdkit_2d_normalized",
            "no_features_scaling": True,
        }
    ) == ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]


def test_prepared_folds_can_be_reused_without_rewriting(tmp_path):
    output = tmp_path / "output"
    config = _config(tmp_path)
    expected = prepare_common_folds(tmp_path, config, group=0, output_dir=output)
    before = (output / "folds.csv").stat().st_mtime_ns
    observed = load_prepared_folds(config, group=0, output_dir=output)
    assert observed == expected
    assert (output / "folds.csv").stat().st_mtime_ns == before
