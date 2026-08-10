from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.prediction.hierarchical_mic import EXAMPLE_FORMAT, predict_mic_bundle


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(5)
    genome = FirstTokenAttentionGenome(8, 12, 4, 0.1)
    text = FirstTokenAttentionGenome(8, 16, 4, 0.1)
    head = RegressionHead(28, 10, 6, 1, 0.2)
    torch.save(
        {
            "format": "apexoracle_hierarchical_mic_inference_v1",
            "source_checkpoint": "synthetic.pth",
            "source_checkpoint_size": 1,
            "source_checkpoint_sha256": "0" * 64,
            "archived_r2": 0.5,
            "re_head_state_dict": head.state_dict(),
            "co_cross_attn_genome": genome.state_dict(),
            "co_cross_attn_text": text.state_dict(),
            "learnable_embedding_weight": torch.zeros(1, 12),
        },
        path,
    )


def test_text_only_bundle_returns_inverse_transformed_mic(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    example = tmp_path / "example.pt"
    _write_checkpoint(checkpoint)
    torch.save(
        {
            "format": EXAMPLE_FORMAT,
            "molecule_embedding": torch.randn(1, 8),
            "text_embeddings": torch.randn(3, 16),
            "metadata": {"example": "synthetic"},
        },
        example,
    )

    result = predict_mic_bundle(checkpoint, example)

    assert result["route"] == "text_only"
    assert result["input_metadata"] == {"example": "synthetic"}
    assert len(result["prediction_z"]) == 1
    assert result["predicted_mic_um"][0] > 0


def test_bundle_rejects_wrong_feature_dimension(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    example = tmp_path / "example.pt"
    _write_checkpoint(checkpoint)
    torch.save(
        {
            "format": EXAMPLE_FORMAT,
            "molecule_embedding": torch.randn(1, 7),
            "text_embeddings": torch.randn(3, 16),
        },
        example,
    )

    with pytest.raises(ValueError, match="feature dimension 8"):
        predict_mic_bundle(checkpoint, example)
