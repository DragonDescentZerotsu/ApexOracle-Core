from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from apexoracle.benchmarks.molecule_encoders.apex_adapter import (
    build_apex_vocabulary,
    legacy_onehot_encoding,
)
from apexoracle.benchmarks.molecule_encoders.apex_model import (
    ApexEncoder,
    load_aaindex_embedding,
)
from apexoracle.benchmarks.molecule_encoders.assets import APEX_AAINDEX_RELATIVE_PATH
from apexoracle.benchmarks.molecule_encoders.legacy_training import (
    LegacyMaskedMSELoss,
    finite_mean_or_nan,
    legacy_r2_per_task,
)
from apexoracle.resources import resolve_weight


REPO_ROOT = Path(__file__).resolve().parents[1]
AAINDEX = REPO_ROOT / APEX_AAINDEX_RELATIVE_PATH
try:
    CHECKPOINT = resolve_weight("fig2b_apex_encoder", repo_root=REPO_ROOT)
except FileNotFoundError:
    CHECKPOINT = REPO_ROOT / "weights/molecule_encoders/apex/APEX_pretrained_encoder_state_dict_best.ckpt"


def _inline_legacy_forward(model: ApexEncoder, token_ids: torch.Tensor) -> torch.Tensor:
    embedded = model.peptideEmb.aa_embedding(token_ids)
    recurrent, _ = model.rnn(embedded)
    recurrent = model.layernorm(recurrent)
    attention_1 = functional.softmax(
        model.attn1(torch.cat((recurrent, embedded), 2)), dim=2
    )
    attended = torch.bmm(attention_1, recurrent)
    attention_2 = functional.softmax(model.attn2(attended), dim=1)
    return model.fc0(torch.sum(attention_2 * attended, dim=1))


def test_apex_encoder_forward_matches_inline_legacy_operations() -> None:
    torch.manual_seed(7)
    embedding = np.arange(23 * 8, dtype=np.float32).reshape(23, 8) / 100
    model = ApexEncoder(
        embedding, 8, num_rnn_layers=3, hidden_dim=4, max_length=6
    ).eval()
    token_ids = torch.tensor([[1, 3, 4, 2, 0, 0], [1, 3, 0, 5, 2, 0]])

    with torch.inference_mode():
        actual = model(token_ids)
        expected = _inline_legacy_forward(model, token_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_legacy_loss_and_metrics_preserve_direct_formulas() -> None:
    prediction = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    target = torch.tensor([[0.0, 2.0], [1.0, 9.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    direct = (prediction - target).square() * mask

    torch.testing.assert_close(
        LegacyMaskedMSELoss()(prediction, target, mask),
        direct.sum() / (mask.sum() + 1e-8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        LegacyMaskedMSELoss("sum")(prediction, target, mask), direct.sum()
    )
    torch.testing.assert_close(
        LegacyMaskedMSELoss("none")(prediction, target, mask), direct
    )

    labels = np.array([[0.0, 5.0], [2.0, 6.0], [4.0, 7.0]])
    predictions = np.array([[0.5, 0.0], [1.5, 5.5], [5.0, 6.5]])
    masks = np.array([[1, 0], [1, 1], [1, 1]])
    values = legacy_r2_per_task(labels, predictions, masks)
    expected = []
    for task in range(2):
        observed = masks[:, task].astype(bool)
        y_true = labels[observed, task]
        y_pred = predictions[observed, task]
        expected.append(
            1
            - np.sum((y_true - y_pred) ** 2)
            / np.sum((y_true - np.mean(y_true)) ** 2)
        )
    assert values == expected
    assert finite_mean_or_nan([values[0], None, float("nan")]) == values[0]


@pytest.mark.skipif(
    not AAINDEX.exists() or not CHECKPOINT.exists(),
    reason="ignored APEX paper assets are not installed",
)
def test_real_apex_checkpoint_strict_load_and_frozen_features() -> None:
    vocabulary, _ = build_apex_vocabulary()
    embedding, _ = load_aaindex_embedding(AAINDEX, vocabulary)
    assert embedding.shape == (23, 566)
    assert (
        hashlib.sha256(np.ascontiguousarray(embedding).tobytes()).hexdigest()
        == "9b81565083ca302f56d0131f8a38dbf19825a898fe80f881f98786132cac3454"
    )
    token_ids = legacy_onehot_encoding(
        ["ACDEFGHIK", "AXD", "A" * 60, "C"], 52, vocabulary
    )
    model = ApexEncoder(
        embedding, embedding.shape[1], num_rnn_layers=3, hidden_dim=128
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        features = model(torch.from_numpy(token_ids).long()).float().numpy()

    assert features.shape == (4, 128)
    assert (
        hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
        == "dd1ac1a9f359f60b55a83910cdd198b5aeca00788b2d5a8c6aa691d956643cb3"
    )
