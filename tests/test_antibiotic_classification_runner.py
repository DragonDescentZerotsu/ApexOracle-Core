from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apexoracle.data.antibiotic_classification import TARGET_STRAINS
from apexoracle.training.antibiotic_classification_runner import (
    AntibioticClassificationConfig,
    AntibioticClassificationPaths,
    ClassificationLoaders,
    _train_full_shared_epoch,
    build_full_fusion_model,
    evaluate_target,
    load_checkpoint_into_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_config_exposes_three_distinct_behavior_frozen_modes():
    path = REPO_ROOT / "configs/antibiotic_classification/legacy_three_strain.yaml"
    strict = AntibioticClassificationConfig.load(
        path, REPO_ROOT, mode="strict-zero-shot"
    )
    fine = AntibioticClassificationConfig.load(path, REPO_ROOT, mode="fine-tune")
    molecule = AntibioticClassificationConfig.load(
        path, REPO_ROOT, mode="molecule-only"
    )
    assert strict.target_names == TARGET_STRAINS
    assert (
        strict.full_fusion and not strict.target_training and strict.num_folds is None
    )
    assert fine.full_fusion and fine.target_training and fine.num_folds == 5
    assert not molecule.full_fusion and molecule.target_training
    assert molecule.num_folds == 5
    assert strict.batch_size == fine.batch_size == molecule.batch_size == 70
    assert strict.freeze_epochs == 3000


def _small_config(tmp_path: Path) -> AntibioticClassificationConfig:
    paths = AntibioticClassificationPaths(
        genome_embeddings=tmp_path,
        atcc_text_embeddings=tmp_path,
        text_only_embeddings=tmp_path,
        peptide_embeddings=tmp_path / "pep.pt",
        small_molecule_embeddings=tmp_path / "sm.pt",
        output_dir=tmp_path,
    )
    return AntibioticClassificationConfig(
        protocol_family="paper_legacy_three_strain_antibiotic_classification",
        mode="strict-zero-shot",
        target_names=TARGET_STRAINS,
        target_modalities=("genome_text", "genome_text", "text_only"),
        molecule_embedding_dim=6,
        genome_embedding_dim=8,
        text_embedding_dim=4,
        attention_heads=2,
        attention_dropout=0.1,
        fusion_head_hidden_dims=(3, 4),
        molecule_only_head_hidden_dims=(3, 4),
        head_dropout=0.2,
        ensembles=1,
        ensemble_seeds=(42,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-5,
        weight_decay=0.0,
        scheduler_eta_min=1e-10,
        freeze_epochs=3000,
        genome_embedding_scale=1e14,
        text_embedding_scale=1.0,
        target_training=False,
        num_folds=None,
        full_fusion=True,
        evidence="synthetic_test",
        paths=paths,
    )


def _batch(*, device: torch.device, has_genome: bool, classification: bool) -> dict:
    output = {
        "label": (
            torch.tensor([1.0, 0.0]) if classification else torch.tensor([0.5, -0.5])
        ),
        "padded_text_embeddings": torch.randn(
            2, 3, 4, device=device, dtype=torch.bfloat16
        ),
        "text_attn_masks": torch.tensor(
            [[1, 1, 1], [1, 1, 0]], device=device, dtype=torch.uint8
        ),
        "strain_names": ["a", "b"],
        "molecule_ids": ["m-a", "m-b"],
        "mol_emb": torch.randn(2, 6),
    }
    if has_genome:
        output["padded_genome_embeddings"] = torch.randn(
            2, 2, 8, device=device, dtype=torch.bfloat16
        )
        output["genome_attn_masks"] = torch.ones(2, 2, device=device, dtype=torch.uint8)
    return output


def test_loader_accepts_only_original_or_documented_stripped_capsule_schema(tmp_path):
    config = _small_config(tmp_path)
    source = build_full_fusion_model(
        config, genome_dim=8, text_dim=4, device=torch.device("cpu")
    )
    payload = {
        "auroc": 0.7,
        "auprc": 0.6,
        "re_head_state_dict": source.regression_head.state_dict(),
        "cls_head_state_dict": source.classification_head.state_dict(),
        "co_cross_attn_genome": source.genome_attention.state_dict(),
        "co_cross_attn_text": source.text_attention.state_dict(),
        "learnable_embedding_weight": source.missing_genome_embedding,
    }
    stripped = tmp_path / "stripped.pth"
    torch.save(payload, stripped)
    destination = build_full_fusion_model(
        config, genome_dim=8, text_dim=4, device=torch.device("cpu")
    )
    assert load_checkpoint_into_model(
        config, destination, stripped, device=torch.device("cpu")
    ) == {"auroc": 0.7, "auprc": 0.6}

    invalid = tmp_path / "invalid.pth"
    torch.save(
        {key: value for key, value in payload.items() if key != "auprc"}, invalid
    )
    with pytest.raises(ValueError, match="Unexpected full-fusion checkpoint keys"):
        load_checkpoint_into_model(
            config, destination, invalid, device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_one_epoch_full_fusion_cuda_smoke_preserves_selection_cls_train_mode(tmp_path):
    device = torch.device("cuda:0")
    config = _small_config(tmp_path)
    model = build_full_fusion_model(config, genome_dim=8, text_dim=4, device=device)
    target = _batch(device=device, has_genome=True, classification=True)
    loaders = ClassificationLoaders(
        mic_genome_text_train=[
            _batch(device=device, has_genome=True, classification=False)
        ],
        mic_text_route_train=[
            _batch(device=device, has_genome=False, classification=False)
        ],
        auxiliary_genome_text_train=[
            _batch(device=device, has_genome=True, classification=True)
        ],
        auxiliary_text_only_train=[
            _batch(device=device, has_genome=False, classification=True)
        ],
        target_train=None,
        target_test=[target],
        target_has_genome=True,
        target_dataset=None,
    )
    losses = _train_full_shared_epoch(loaders, model, config, device=device, epoch=0)
    assert len(losses["regression"]) == 2
    assert len(losses["auxiliary"]) == 2
    assert losses["target"] == []
    evaluation = evaluate_target(loaders, model, full_fusion=True, device=device)
    assert len(evaluation.labels) == len(evaluation.logits) == 2
    assert all(torch.isfinite(torch.tensor(evaluation.logits)))
    assert not model.genome_attention.training
    assert not model.text_attention.training
    assert not model.regression_head.training
    assert model.classification_head.training
    assert model.optimizer.state
