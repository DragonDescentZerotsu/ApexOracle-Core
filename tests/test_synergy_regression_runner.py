from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from apexoracle.data.synergy import SYNERGY_COLUMNS, synergy_regression_target
from apexoracle.training import synergy_regression_runner as runner
from apexoracle.training.synergy import (
    legacy_synergy_regression_checkpoint_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "synergy" / "legacy_regression_producer.yaml"


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("a", "b", "strain", "A", "B", 0.5),
            ("c", "d", "strain", "C", "D", 0.25),
        ],
        columns=SYNERGY_COLUMNS,
    )


def test_regression_target_matches_legacy_log_transform():
    assert synergy_regression_target(0.5) == pytest.approx(-np.log10(0.05))
    assert synergy_regression_target(2.0) == pytest.approx(-np.log10(0.2))
    with pytest.raises(ValueError, match="positive"):
        synergy_regression_target(0.0)


def test_regression_loaders_use_continuous_targets_and_both_train_routes():
    tables = runner.RegressionTables(
        train_genome_text=_table(),
        train_combined_text=_table(),
        test_genome_text=_table(),
        unique_smiles_tokenized=4,
    )
    molecules = {
        key: torch.arange(4, dtype=torch.float32)
        for key in ("a", "b", "c", "d")
    }
    features = runner.RegressionRuntimeFeatures(
        genomes={"strain": torch.ones(2, 3)},
        atcc_text={"strain": torch.ones(2, 2)},
        text_only={},
        molecules=molecules,
    )
    loaders = runner.build_regression_loaders(tables, features, batch_size=2)

    genome_batch = next(iter(loaders.genome_test))
    text_batch = next(iter(loaders.text_train))
    expected = sorted([synergy_regression_target(0.5), synergy_regression_target(0.25)])
    assert sorted(genome_batch["label"].tolist()) == pytest.approx(expected)
    assert sorted(text_batch["label"].tolist()) == pytest.approx(expected)
    assert "padded_genome_embeddings" in genome_batch
    assert "padded_genome_embeddings" not in text_batch


def test_train_epoch_preserves_genome_then_text_zip_longest_order(monkeypatch):
    calls = []

    def fake_step(batch, *, has_genome, **kwargs):
        calls.append((batch, has_genome))

    monkeypatch.setattr(runner, "synergy_pair_step", fake_step)
    components = SimpleNamespace(
        genome_attention=nn.Linear(1, 1),
        text_attention=nn.Linear(1, 1),
        prediction_head=nn.Linear(1, 1),
        missing_genome_embedding=nn.Parameter(
            torch.zeros(1), requires_grad=False
        ),
    )
    loaders = runner.RegressionLoaders(
        genome_train=["g0", "g1"],
        text_train=["t0"],
        genome_test=[],
    )
    runner.train_regression_epoch(
        loaders,
        components=components,
        device=torch.device("cpu"),
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD(components.prediction_head.parameters(), lr=0.1),
        scaler=object(),
        autocast_enabled=False,
        epoch=0,
    )
    assert calls == [("g0", True), ("t0", False), ("g1", True)]


def test_legacy_r2_matches_direct_formula():
    labels = np.asarray([1.0, 2.0, 4.0])
    predictions = np.asarray([1.5, 1.5, 3.0])
    expected = 1 - np.sum((labels - predictions) ** 2) / np.sum(
        (labels - labels.mean()) ** 2
    )
    assert runner.legacy_r2(labels, predictions) == expected


def test_regression_checkpoint_payload_keeps_full_six_key_state():
    genome = nn.Linear(2, 2)
    text = nn.Linear(2, 2)
    head = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(
        list(genome.parameters()) + list(text.parameters()) + list(head.parameters())
    )
    missing = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
    payload = legacy_synergy_regression_checkpoint_payload(
        r2=0.25,
        optimizer=optimizer,
        prediction_head=head,
        genome_attention=genome,
        text_attention=text,
        missing_genome_embedding=missing,
    )
    assert set(payload) == {
        "R2",
        "optimizer_state_dict",
        "re_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }
    assert set(payload["co_cross_attn_genome"]) == {"weight", "bias"}


def test_fixed_epoch_uses_best_r2_so_far_but_current_parameters(
    monkeypatch, tmp_path
):
    components = SimpleNamespace(
        genome_attention=nn.Linear(1, 1),
        text_attention=nn.Linear(1, 1),
        prediction_head=nn.Linear(2, 1),
        missing_genome_embedding=nn.Parameter(
            torch.zeros(1), requires_grad=False
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_legacy_synergy_components",
        lambda *args, **kwargs: components,
    )
    monkeypatch.setattr(runner, "train_regression_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner.optim.lr_scheduler,
        "CosineAnnealingLR",
        lambda *args, **kwargs: SimpleNamespace(step=lambda: None),
    )
    scores = iter([-9.0, 0.1, 0.2, 0.15, 0.3, 0.25, 0.28])

    class FakeEvaluation:
        labels = (0.0, 1.0)
        predictions = (0.0, 1.0)
        pair_keys = (("a", "b", "strain"), ("c", "d", "strain"))

        def metrics(self):
            return {"r2": next(scores), "spearman": 1.0, "pearson": 1.0, "mse": 0.0}

    monkeypatch.setattr(
        runner,
        "evaluate_regression",
        lambda *args, **kwargs: FakeEvaluation(),
    )
    saved = []
    monkeypatch.setattr(
        torch,
        "save",
        lambda payload, path: saved.append((payload["R2"], Path(path).name)),
    )
    config = SimpleNamespace(
        seeds=(42,),
        epochs=6,
        fixed_epoch_index=5,
        learning_rate=1e-5,
        weight_decay=0.0,
        eta_min=1e-10,
        molecule_dim=1,
        genome_dim=1,
        text_dim=1,
        attention_heads=1,
        lora_rank=64,
        paths=SimpleNamespace(base_checkpoint=tmp_path / "base", output_dir=tmp_path),
    )
    result = runner.run_member(
        config,
        runner.RegressionLoaders([], [], []),
        member=0,
        device=torch.device("cpu"),
    )
    assert result["best_r2"] == 0.3
    assert saved[-1] == (0.3, "fold_0_ensemble_0_fixed_epoch.ckpt")


def test_config_rejects_original_data_or_checkpoint_outputs():
    for output in (
        REPO_ROOT / "DataPrepare" / "Data" / "unsafe",
        REPO_ROOT / "Checkpoints" / "unsafe",
    ):
        with pytest.raises(ValueError, match="must not overwrite"):
            runner.RegressionProducerConfig.load(
                CONFIG,
                REPO_ROOT,
                epochs=1,
                ensemble_members=1,
                output_dir=output,
            )
