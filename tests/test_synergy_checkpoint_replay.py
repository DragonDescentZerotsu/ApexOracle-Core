import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_replay_script():
    path = ROOT / "scripts/reproduce/replay_synergy_checkpoints.py"
    spec = importlib.util.spec_from_file_location("synergy_checkpoint_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load_replay_script()


def test_frozen_replay_audit_matches_archived_log_precision() -> None:
    audit = json.loads(
        (ROOT / "experiments/synergy/checkpoint_replay_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "passed"
    assert audit["protocol"]["evaluation_rows_after_token_filter"] == 2371
    assert len(audit["folds"]) == 3
    for fold in audit["folds"]:
        assert round(fold["auroc"], 4) == fold["archived_log_auroc"]
        assert round(fold["auprc"], 4) == fold["archived_log_auprc"]
    assert audit["repeatability"]["all_four_prediction_csv_files_byte_identical"]


def test_pair_identity_is_stable_and_order_sensitive() -> None:
    first = REPLAY.pair_identity(("molecule-a", "molecule-b", "strain"))
    assert len(first) == 64
    assert first == REPLAY.pair_identity(("molecule-a", "molecule-b", "strain"))
    assert first != REPLAY.pair_identity(("molecule-b", "molecule-a", "strain"))


def test_prediction_table_averages_members_and_removes_raw_ids() -> None:
    table, metrics = REPLAY.build_prediction_table(
        fold=1,
        routes=["genome_text", "text_only"],
        pair_keys=[("private-a", "private-b", "strain-1"), ("x", "y", "strain-2")],
        labels=[0.0, 1.0],
        member_predictions=[[0.1, 0.8], [0.3, 1.0]],
    )
    assert table["ensemble_probability"].tolist() == pytest.approx([0.2, 0.9])
    assert table["label"].tolist() == [0, 1]
    assert table["fold"].tolist() == [1, 1]
    assert "private-a" not in table.to_csv(index=False)
    assert metrics == pytest.approx({"auroc": 1.0, "auprc": 1.0})


def test_prediction_table_retains_repeated_measurements() -> None:
    table, _ = REPLAY.build_prediction_table(
        fold=0,
        routes=["text_only", "text_only"],
        pair_keys=[("a", "b", "strain"), ("a", "b", "strain")],
        labels=[0.0, 1.0],
        member_predictions=[[0.2, 0.8]],
    )
    assert table["pair_identity"].nunique() == 1
    assert table["measurement_index"].tolist() == [0, 1]
    assert table["label"].tolist() == [0, 1]
