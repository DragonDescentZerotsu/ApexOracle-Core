from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from apexoracle.data.antibiotic_classification import (
    AntibioticGenomeTextDataset,
    AntibioticTextOnlyDataset,
    TARGET_STRAINS,
    collate_antibiotic_genome_text_classification,
    collate_antibiotic_text_classification,
    legacy_target_folds,
    prepare_antibiotic_classification_frames,
)
from apexoracle.data.hierarchical_mic_preparation import PreparedHierarchicalMicData
from apexoracle.evaluation.antibiotic_classification import (
    LegacyClassificationBestTracker,
    classification_metrics,
    ensemble_classification_predictions,
)
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.training.antibiotic_classification import (
    full_fusion_classification_forward,
    legacy_full_fusion_checkpoint_payload,
    legacy_molecule_only_checkpoint_payload,
    molecule_only_forward,
    molecule_only_step,
    set_legacy_full_fusion_selection_modes,
    set_legacy_full_fusion_training_modes,
)
from apexoracle.training.antibiotic_classification_runner import checkpoint_filename


COLUMNS = ["DBAASP_id", "strain_name", "SMILES", "MIC"]


def _group(records: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: records[records[:, 1] == name] for name in set(records[:, 1].tolist())
    }


def _prepared() -> PreparedHierarchicalMicData:
    genome = np.array([[1, "g-a", "[1]", 10.0], [2, "g-b", "[1]", 20.0]], dtype=object)
    text = np.array(
        [
            [1, "g-a", "[1]", 10.0],
            [2, "g-b", "[1]", 20.0],
            [3, "text-a", "[1]", 30.0],
        ],
        dtype=object,
    )
    auxiliary = np.array(
        [
            [10, "#004", "[1]", 1.0],
            [11, "#004", "[1]", 0.0],
            [12, "17978", "[1]", 1.0],
            [13, "Staphylococcus aureus RN4220", "[1]", 0.0],
        ],
        dtype=object,
    )
    return PreparedHierarchicalMicData(
        columns=COLUMNS,
        genome_text_records=genome,
        genome_or_text_records=text,
        small_molecule_records=auxiliary,
        genome_text_groups=_group(genome),
        genome_or_text_groups=_group(text),
        small_molecule_groups=_group(auxiliary),
        atcc_id_to_species={},
        original_strain_to_species={},
        species_to_strains={},
        taxonomy_aliases={},
    )


def test_three_mode_frame_adapter_preserves_duplicate_mic_route_and_target_roles():
    prepared = _prepared()
    group_zero = prepare_antibiotic_classification_frames(prepared, 0)
    assert group_zero.target_has_genome is True
    assert set(group_zero.target["strain_name"]) == {"#004"}
    assert set(group_zero.auxiliary_genome_text_train["strain_name"]) == {"17978"}
    assert set(group_zero.auxiliary_text_only_train["strain_name"]) == {
        "Staphylococcus aureus RN4220"
    }
    assert group_zero.mic_genome_text_train["DBAASP_id"].tolist() == [1, 2]
    assert group_zero.mic_text_route_train["DBAASP_id"].tolist() == [1, 2, 3]

    group_two = prepare_antibiotic_classification_frames(prepared, 2)
    assert group_two.target_has_genome is False
    assert group_two.auxiliary_text_only_train is None
    assert set(group_two.auxiliary_genome_text_train["strain_name"]) == {
        "#004",
        "17978",
    }
    assert set(group_two.target["strain_name"]) == {TARGET_STRAINS[2]}


def test_target_folds_are_the_exact_post_filter_legacy_kfold_indices():
    shared = legacy_target_folds(23, num_folds=5)
    reference = tuple(
        KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(23))
    )
    assert len(shared) == len(reference) == 5
    for (shared_train, shared_test), (legacy_train, legacy_test) in zip(
        shared, reference
    ):
        np.testing.assert_array_equal(shared_train, legacy_train)
        np.testing.assert_array_equal(shared_test, legacy_test)


def test_classification_datasets_add_ids_without_changing_frozen_tensor_contract():
    frame = pd.DataFrame(
        [[10, "#004", "[1, 2]", 1.0], [11, "#004", "[1, 2, 3]", 0.0]],
        columns=COLUMNS,
    )
    genomes = {"#004": torch.ones(2, 4, dtype=torch.bfloat16)}
    texts = {"#004": torch.ones(3, 4, dtype=torch.bfloat16)}
    molecules = {
        10: torch.arange(6, dtype=torch.float32).reshape(1, 6),
        11: torch.arange(6, dtype=torch.float32).reshape(1, 6) + 1,
    }
    genome_dataset = AntibioticGenomeTextDataset(
        frame.copy(), None, genomes, texts, "test", molecules, {}
    )
    genome_batch = collate_antibiotic_genome_text_classification(
        [genome_dataset[0], genome_dataset[1]]
    )
    assert genome_batch["molecule_ids"] == [10, 11]
    assert genome_batch["label"].tolist() == pytest.approx([1.0, 0.0])
    assert genome_batch["padded_genome_embeddings"].dtype == torch.bfloat16

    text_dataset = AntibioticTextOnlyDataset(
        frame.copy(), None, texts, "test", molecules, {}
    )
    text_batch = collate_antibiotic_text_classification(
        [text_dataset[0], text_dataset[1]]
    )
    assert text_batch["molecule_ids"] == [10, 11]
    assert "padded_genome_embeddings" not in text_batch


def test_full_fusion_selection_changes_three_modes_but_leaves_cls_dropout_active():
    genome = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1))
    text = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1))
    regression = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1))
    classification = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2))
    set_legacy_full_fusion_training_modes(genome, text, regression, classification)
    assert all(module.training for module in (genome, text, regression, classification))
    set_legacy_full_fusion_selection_modes(genome, text, regression, classification)
    assert not genome.training
    assert not text.training
    assert not regression.training
    assert classification.training


def test_full_fusion_forward_matches_direct_legacy_operations():
    torch.manual_seed(17)
    genome_attention = FirstTokenAttentionGenome(6, 8, 2, 0.1).eval()
    text_attention = FirstTokenAttentionGenome(6, 4, 2, 0.1).eval()
    cls_head = RegressionHead(12, 6, 3, 1, 0.2).eval()
    missing = nn.Parameter(torch.randn(1, 8))
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "padded_genome_embeddings": torch.randn(2, 3, 8),
        "genome_attn_masks": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8),
        "padded_text_embeddings": torch.randn(2, 2, 4),
        "text_attn_masks": torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8),
        "strain_names": ["a", "b"],
        "mol_emb": torch.randn(2, 6),
    }
    result = full_fusion_classification_forward(
        batch,
        device=torch.device("cpu"),
        genome_attention=genome_attention,
        text_attention=text_attention,
        classification_head=cls_head,
        criterion=nn.BCEWithLogitsLoss(),
        missing_genome_embedding=missing,
        has_genome=True,
        autocast_enabled=False,
    )
    genome_part = genome_attention(
        batch["mol_emb"],
        batch["padded_genome_embeddings"],
        1 - batch["genome_attn_masks"],
    )
    text_part = text_attention(
        batch["mol_emb"],
        batch["padded_text_embeddings"],
        1 - batch["text_attn_masks"],
    )
    expected_logits = cls_head(torch.cat((genome_part, text_part), dim=1)).squeeze()
    expected_loss = nn.BCEWithLogitsLoss()(expected_logits, batch["label"])
    torch.testing.assert_close(result.logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(result.loss, expected_loss, rtol=0, atol=0)


def test_molecule_only_forward_and_optimizer_step_match_direct_head_path():
    torch.manual_seed(23)
    head = RegressionHead(6, 3, 2, 1, 0.2).eval()
    batch = {
        "label": torch.tensor([1.0, 0.0]),
        "mol_emb": torch.randn(2, 6),
        "strain_names": ["a", "b"],
    }
    result = molecule_only_forward(
        batch,
        device=torch.device("cpu"),
        classification_head=head,
        criterion=nn.BCEWithLogitsLoss(),
        autocast_enabled=False,
    )
    expected = head(batch["mol_emb"]).squeeze()
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)

    train_head = RegressionHead(6, 3, 2, 1, 0.2).train()
    optimizer = torch.optim.Adam(train_head.parameters(), lr=1e-5)
    before = {
        name: value.detach().clone() for name, value in train_head.state_dict().items()
    }
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    molecule_only_step(
        batch,
        device=torch.device("cpu"),
        classification_head=train_head,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scaler=scaler,
        autocast_enabled=False,
    )
    assert any(
        not torch.equal(before[name], value)
        for name, value in train_head.state_dict().items()
    )


def test_metric_tracker_and_checkpoint_payload_preserve_legacy_auprc_lag():
    tracker = LegacyClassificationBestTracker()
    assert tracker.update_auroc(auroc=0.7, predictions=[0.1, 0.9])
    assert tracker.checkpoint_auprc == -10.0
    tracker.finish_epoch(auprc=0.6)
    assert tracker.best_auprc == 0.6
    assert tracker.update_auroc(auroc=0.8, predictions=[0.2, 0.8])
    assert tracker.checkpoint_auprc == 0.6

    genome = FirstTokenAttentionGenome(6, 8, 2, 0.1)
    text = FirstTokenAttentionGenome(6, 4, 2, 0.1)
    regression = RegressionHead(12, 6, 3, 1, 0.2)
    classification = RegressionHead(12, 6, 3, 1, 0.2)
    missing = nn.Parameter(torch.randn(1, 8))
    optimizer = torch.optim.Adam(genome.parameters(), lr=1e-5)
    full = legacy_full_fusion_checkpoint_payload(
        auroc=tracker.best_auroc,
        checkpoint_auprc=tracker.checkpoint_auprc,
        optimizer=optimizer,
        regression_head=regression,
        classification_head=classification,
        genome_attention=genome,
        text_attention=text,
        missing_genome_embedding=missing,
    )
    assert set(full) == {
        "auroc",
        "auprc",
        "optimizer_state_dict",
        "re_head_state_dict",
        "cls_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }
    assert full["auprc"] == 0.6
    molecule = legacy_molecule_only_checkpoint_payload(
        auroc=0.8,
        checkpoint_auprc=0.6,
        optimizer=torch.optim.Adam(classification.parameters(), lr=1e-5),
        classification_head=classification,
    )
    assert set(molecule) == {
        "auroc",
        "auprc",
        "optimizer_state_dict",
        "cls_head_state_dict",
    }


def test_metrics_ensemble_and_checkpoint_names_match_legacy_contracts():
    labels = [0.0, 0.0, 1.0, 1.0]
    first = [0.1, 0.3, 0.7, 0.9]
    second = [0.2, 0.4, 0.6, 0.8]
    ensembled = ensemble_classification_predictions([first, second])
    assert ensembled.tolist() == pytest.approx([0.15, 0.35, 0.65, 0.85])
    assert classification_metrics(labels, ensembled) == pytest.approx(
        {"auroc": 1.0, "auprc": 1.0}
    )
    assert checkpoint_filename("strict-zero-shot", 2, 9, None).endswith(
        "group_2_ensemble_9.pth"
    )
    assert checkpoint_filename("fine-tune", 1, 3, 4).endswith(
        "group_1_ensemble_3_fold_4.pth"
    )
    assert checkpoint_filename("molecule-only", 0, 0, 0).endswith(
        "group_0_ensemble_0_fold_0.pth"
    )


def test_canonical_entrypoint_does_not_execute_a_root_legacy_driver():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root / "scripts/reproduce/run_antibiotic_classification.py"
    ).read_text(encoding="utf-8")
    assert "runpy" not in source
    assert "antibiotic_3_strain_compare" not in source
    assert "antibiotic_classification_runner" in source
