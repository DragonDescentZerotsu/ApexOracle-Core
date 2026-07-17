import csv

import numpy as np
import pytest

from apexoracle.benchmarks.molecule_encoders.data import SharedBenchmarkData
from apexoracle.benchmarks.molecule_encoders.encoders import _tokenize_without_dropping
from apexoracle.benchmarks.molecule_encoders.feature_cache import (
    FeatureCache,
    load_feature_cache,
    save_feature_cache,
)
from apexoracle.benchmarks.molecule_encoders.protocol import DEFAULT_TARGET_COLUMNS
from apexoracle.benchmarks.molecule_encoders.training import (
    HeadTrainingConfig,
    train_shared_heads,
)


def _benchmark(number_of_rows=25):
    ids = tuple(str(index) for index in range(number_of_rows))
    features = np.stack(
        [
            np.linspace(-1.0, 1.0, number_of_rows),
            np.sin(np.arange(number_of_rows)),
            np.cos(np.arange(number_of_rows)),
            np.ones(number_of_rows),
        ],
        axis=1,
    ).astype(np.float32)
    labels = np.stack(
        [features[:, 0] * (task + 1) + features[:, 1] for task in range(19)],
        axis=1,
    ).astype(np.float32)
    benchmark = SharedBenchmarkData(
        molecule_ids=ids,
        smiles=tuple(f"SMILES-{value}" for value in ids),
        apex_sequences=tuple("AXD" for _ in ids),
        labels=labels,
        label_mask=np.ones_like(labels, dtype=bool),
        folds=np.arange(number_of_rows) % 5,
        target_columns=tuple(DEFAULT_TARGET_COLUMNS),
    )
    return benchmark, features


class _FakeTokenizer:
    unk_token_id = 99

    def __call__(self, text, *, truncation, max_length=None, **_kwargs):
        content = [99 if symbol == "?" else 7 for symbol in text]
        tokens = [1, *content, 2]
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


def test_hf_tokenization_truncates_and_records_but_never_drops():
    token_ids, truncated, with_unknown = _tokenize_without_dropping(
        _FakeTokenizer(),
        ("AAAA", "A?"),
        max_length=4,
    )

    assert len(token_ids) == 2
    assert [len(values) for values in token_ids] == [4, 4]
    assert truncated == 1
    assert with_unknown == 1


def test_feature_cache_reorders_to_shared_ids_and_rejects_missing_ids(tmp_path):
    benchmark, features = _benchmark()
    reversed_ids = tuple(reversed(benchmark.molecule_ids))
    cache_path = tmp_path / "features.npz"
    save_feature_cache(
        cache_path,
        encoder_name="test_encoder",
        molecule_ids=reversed_ids,
        features=features[::-1],
        metadata={"checkpoint": "unit-test"},
    )
    cache = load_feature_cache(cache_path, benchmark, expected_encoder="test_encoder")
    np.testing.assert_allclose(cache.features, features)
    assert cache.molecule_ids == benchmark.molecule_ids

    bad_path = tmp_path / "missing.npz"
    save_feature_cache(
        bad_path,
        encoder_name="test_encoder",
        molecule_ids=reversed_ids[:-1],
        features=features[::-1][:-1],
    )
    with pytest.raises(ValueError, match="exactly the shared molecule IDs"):
        load_feature_cache(bad_path, benchmark)


def test_shared_head_runner_writes_one_outer_test_prediction_per_label(tmp_path):
    benchmark, features = _benchmark()
    cache = FeatureCache(
        encoder_name="test_encoder",
        features=features,
        molecule_ids=benchmark.molecule_ids,
        metadata={"cache_version": "unit-test"},
    )
    config = HeadTrainingConfig(
        hidden_dim_1=8,
        hidden_dim_2=4,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=2,
        patience=1,
        validation_fraction=0.25,
        seed=7,
    )
    metrics = train_shared_heads(
        benchmark,
        cache,
        tmp_path / "results",
        config=config,
        device="cpu",
    )

    assert len(metrics["folds"]) == 5
    assert sum(fold["test_size"] for fold in metrics["folds"]) == len(benchmark)
    with (tmp_path / "results" / "predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == len(benchmark) * len(DEFAULT_TARGET_COLUMNS)
    assert len({(row["dbaasp_id"], row["task"]) for row in predictions}) == len(predictions)
