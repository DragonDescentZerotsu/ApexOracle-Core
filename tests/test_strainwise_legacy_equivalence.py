from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import pearsonr, spearmanr

from apexoracle.data.strainwise import (
    StrainEmbeddingDataset,
    TextOnlyStrainEmbeddingDataset,
    collate_genome_text_classification,
    collate_genome_text_regression,
    collate_text_classification,
    collate_text_regression,
)
from apexoracle.data.strainwise_protocol import build_legacy_three_fold_groups
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.models.strainwise_checkpoint import (
    load_legacy_strainwise_checkpoint,
    predict_genome_text,
    predict_text_only,
)
from apexoracle.data.strain_mapping import (
    exclude_wrong_species_atcc_map,
    get_atcc_id_to_species_name_map,
    get_original_strain_id_to_species_name_map,
    get_original_strain_names_with_genome_embedding,
    merge_strain_maps,
)
from apexoracle.features.precomputed import (
    get_embedded_genome_ids,
    load_all_embeddings,
    load_text_only_embeddings,
)
from apexoracle.evaluation.strainwise import (
    calculate_r2,
    ensemble_predictions,
    specieswise_metrics,
    summarize_predictions,
)
from apexoracle.training.strainwise import strainwise_optimizer_step

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_legacy_module():
    path = REPO_ROOT / "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py"
    spec = importlib.util.spec_from_file_location("legacy_strainwise_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("legacy_strainwise_test")
    return module


def _synthetic_batch():
    return [
        {
            "label": torch.tensor(10.0),
            "genome_embedding": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            "text_embedding": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "strain_name": "strain-a",
            "mol_emb": torch.arange(6, dtype=torch.float32),
        },
        {
            "label": torch.tensor(100.0),
            "genome_embedding": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "text_embedding": torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            "strain_name": "strain-b",
            "mol_emb": torch.arange(6, dtype=torch.float32) + 1,
        },
    ]


def _assert_batches_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert left[key].dtype == right[key].dtype
            assert left[key].shape == right[key].shape
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            assert left[key] == right[key]


def test_all_collators_match_executable_legacy_reference():
    legacy = _load_legacy_module()
    batch = _synthetic_batch()
    pairs = [
        (legacy.legacy_collate_fn, collate_genome_text_regression, batch),
        (
            legacy.legacy_collate_fn_text_only,
            collate_text_regression,
            [{key: value for key, value in item.items() if key != "genome_embedding"} for item in batch],
        ),
        (legacy.legacy_collate_fn_cls, collate_genome_text_classification, batch),
        (
            legacy.legacy_collate_fn_text_only_cls,
            collate_text_classification,
            [{key: value for key, value in item.items() if key != "genome_embedding"} for item in batch],
        ),
    ]
    for legacy_fn, shared_fn, fn_batch in pairs:
        _assert_batches_equal(legacy_fn(fn_batch), shared_fn(fn_batch))


def test_dataset_lookup_and_length_filter_match_legacy_reference():
    legacy = _load_legacy_module()
    frame = pd.DataFrame(
        [
            [1, "strain-a", "[1, 2, 3]", 10.0],
            [2, "strain-b", "[1, 2, 3, 4, 5]", 100.0],
        ],
        columns=["DBAASP_id", "strain_name", "SMILES", "MIC"],
    )
    genomes = {
        "strain-a": torch.ones(2, 4, dtype=torch.bfloat16),
        "strain-b": torch.ones(3, 4, dtype=torch.bfloat16),
    }
    texts = {
        "strain-a": torch.ones(1, 4, dtype=torch.bfloat16),
        "strain-b": torch.ones(2, 4, dtype=torch.bfloat16),
    }
    peptide = {1: torch.arange(6).reshape(1, 6)}
    small_molecule = {2: torch.arange(6).reshape(1, 6) + 10}
    legacy_dataset = legacy.LegacySMILESDatasetWithGenomeAndText(
        frame.copy(), None, genomes, texts, "legacy", peptide, small_molecule, max_length=4
    )
    shared_dataset = StrainEmbeddingDataset(
        frame.copy(), None, genomes, texts, "shared", peptide, small_molecule, max_length=4
    )
    assert len(legacy_dataset) == len(shared_dataset) == 1
    legacy_item = legacy_dataset[0]
    shared_item = shared_dataset[0]
    assert legacy_item.keys() == shared_item.keys()
    for key in legacy_item:
        if torch.is_tensor(legacy_item[key]):
            torch.testing.assert_close(legacy_item[key], shared_item[key], rtol=0, atol=0)
        else:
            assert legacy_item[key] == shared_item[key]


def test_regression_head_matches_legacy_reference():
    legacy = _load_legacy_module()
    torch.manual_seed(7)
    reference = legacy.LegacyRegressionHead(12, 8, 4, 1, 0.2).eval()
    shared = RegressionHead(12, 8, 4, 1, 0.2).eval()
    shared.load_state_dict(reference.state_dict(), strict=True)
    features = torch.randn(3, 12)
    torch.testing.assert_close(reference(features), shared(features), rtol=0, atol=0)
    assert list(reference.state_dict()) == list(shared.state_dict())


def test_cross_attention_matches_legacy_reference():
    legacy = _load_legacy_module()
    torch.manual_seed(11)
    reference = legacy.LegacyFirstTokenAttentionGenome(6, 8, 2, 0.1).eval()
    shared = FirstTokenAttentionGenome(6, 8, 2, 0.1).eval()
    shared.load_state_dict(reference.state_dict(), strict=True)
    molecule = torch.randn(2, 6)
    sequence = torch.randn(2, 4, 8)
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    torch.testing.assert_close(
        reference(molecule, sequence, padding_mask),
        shared(molecule, sequence, padding_mask),
        rtol=0,
        atol=0,
    )
    assert list(reference.state_dict()) == list(shared.state_dict())


def test_split_builder_preserves_legacy_mutation_semantics():
    first = {
        "Species old": ["a", "b", "c"],
        "Species new": ["d", "e"],
        "Species stable": ["x", "y", "z"],
    }
    second = {key: list(value) for key, value in first.items()}
    legacy = _load_legacy_module()

    # Local executable reference of the loop, using the same operations as the
    # historical driver. This comparison explicitly includes input mutation.
    legacy_train = [[], [], []]
    legacy_test = [[], [], []]
    taxonomy = {"Species old": "Species new", "Species new": "Species old"}
    for fold in range(3):
        repeated = []
        for species, ids in first.items():
            if species in repeated:
                continue
            merged = ids
            if species in taxonomy:
                repeated.append(taxonomy[species])
                other = first.get(taxonomy[species])
                if other is not None:
                    merged.extend(other)
            merged.sort()
            if len(merged) >= 6:
                merged[1], merged[2] = merged[2], merged[1]
            if len(merged) == 1:
                legacy_train[fold].extend(merged)
            elif len(merged) == 2:
                legacy_train[fold].append(merged[fold % 2])
                legacy_test[fold].append(merged[(fold + 1) % 2])
            else:
                chunk = len(merged) // 3
                held_out = merged[fold * chunk : (fold + 1) * chunk]
                legacy_train[fold].extend(list(set(merged) - set(held_out)))
                legacy_test[fold].extend(held_out)

    shared_train, shared_test = build_legacy_three_fold_groups(second, taxonomy)
    assert shared_train == legacy_train
    assert shared_test == legacy_test
    assert second == first


def test_checkpoint_loader_is_strict_and_preserves_legacy_contract(tmp_path):
    torch.manual_seed(19)
    genome_attention = FirstTokenAttentionGenome(6, 8, 4, 0.1)
    text_attention = FirstTokenAttentionGenome(6, 4, 4, 0.1)
    regression_head = RegressionHead(12, 3, 2, 1, 0.2)
    classification_head = RegressionHead(12, 3, 2, 1, 0.2)
    checkpoint_path = tmp_path / "legacy.pth"
    torch.save(
        {
            "R2": np.float64(0.5),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "re_head_state_dict": regression_head.state_dict(),
            "cls_head_state_dict": classification_head.state_dict(),
            "co_cross_attn_genome": genome_attention.state_dict(),
            "co_cross_attn_text": text_attention.state_dict(),
            "learnable_embedding_weight": torch.randn(1, 8),
        },
        checkpoint_path,
    )
    components, contract = load_legacy_strainwise_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    assert contract["molecule_dim"] == 6
    assert contract["genome_dim"] == 8
    assert contract["text_dim"] == 4
    assert components.archived_r2 == 0.5
    components.eval()
    molecule = torch.randn(2, 6)
    genome = torch.randn(2, 3, 8)
    text = torch.randn(2, 2, 4)
    genome_mask = torch.tensor([[False, False, True], [False, False, False]])
    text_mask = torch.tensor([[False, True], [False, False]])
    assert predict_genome_text(
        components, molecule, genome, genome_mask, text, text_mask
    ).shape == (2, 1)
    assert predict_text_only(components, molecule, text, text_mask).shape == (2, 1)


def test_checkpoint_contract_records_optional_misnamed_mdlm_payload(tmp_path):
    torch.manual_seed(23)
    genome_attention = FirstTokenAttentionGenome(6, 8, 4, 0.1)
    text_attention = FirstTokenAttentionGenome(6, 4, 4, 0.1)
    head = RegressionHead(12, 3, 2, 1, 0.2)
    checkpoint = {
        "R2": np.float64(0.25),
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "re_head_state_dict": head.state_dict(),
        "cls_head_state_dict": head.state_dict(),
        "co_cross_attn_genome": genome_attention.state_dict(),
        "co_cross_attn_text": text_attention.state_dict(),
        "learnable_embedding_weight": torch.randn(1, 8),
        "ChemBERTa_state_dict": {
            "backbone.vocab_embed.embedding": torch.randn(10, 6)
        },
    }
    path = tmp_path / "optional.pth"
    torch.save(checkpoint, path)
    _, contract = load_legacy_strainwise_checkpoint(path)
    payload = contract["optional_payloads"]["ChemBERTa_state_dict"]
    assert payload["interpretation"] == "historically_misnamed_mdlm_backbone_state_dict"
    assert payload["vocab_embedding_shape"] == [10, 6]


def test_embedding_filename_parsing_and_loading_match_legacy_reference(tmp_path):
    legacy = _load_legacy_module()
    atcc_folder = tmp_path / "atcc"
    text_folder = tmp_path / "text"
    atcc_folder.mkdir()
    text_folder.mkdir()
    torch.save(torch.ones(2, 3), atcc_folder / "Escherichia_coli_ATCC_25922.pt")
    torch.save(torch.ones(1, 3) * 2, atcc_folder / "Enterobacter_ATCC_BAA_2468.pt")
    torch.save(torch.ones(2, 4), text_folder / "Species～strain^variant.pt")

    assert legacy.legacy_get_embedded_genome_IDs(atcc_folder) == get_embedded_genome_ids(
        atcc_folder
    )
    legacy_atcc = legacy.legacy_load_all_genome_embeddings(
        atcc_folder, 3.0, torch.device("cpu"), "legacy"
    )
    shared_atcc = load_all_embeddings(
        atcc_folder, 3.0, torch.device("cpu"), "shared"
    )
    assert legacy_atcc.keys() == shared_atcc.keys()
    for key in legacy_atcc:
        torch.testing.assert_close(legacy_atcc[key], shared_atcc[key], rtol=0, atol=0)

    legacy_text = legacy.legacy_load_text_wo_genome_embeddings(
        text_folder, 2.0, torch.device("cpu"), "legacy"
    )
    shared_text = load_text_only_embeddings(
        text_folder, 2.0, torch.device("cpu"), "shared"
    )
    assert legacy_text.keys() == shared_text.keys()
    for key in legacy_text:
        torch.testing.assert_close(legacy_text[key], shared_text[key], rtol=0, atol=0)


def test_strain_mapping_utilities_match_legacy_reference(tmp_path, capsys):
    legacy = _load_legacy_module()
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "Species one*ATCC 123": 1,
                "Species two ATCC BAA 456": 1,
                "Species three without id": 1,
            }
        )
    )
    legacy_mapping = legacy.legacy_get_original_strain_name_with_genome_embedding(
        mapping_path, ["123", "BAA-456"]
    )
    shared_mapping = get_original_strain_names_with_genome_embedding(
        mapping_path, ["123", "BAA-456"]
    )
    assert legacy_mapping == shared_mapping

    atcc_folder = tmp_path / "fasta"
    text_folder = tmp_path / "text"
    atcc_folder.mkdir()
    text_folder.mkdir()
    (atcc_folder / "Escherichia_coli_ATCC_25922.fasta").touch()
    (atcc_folder / "Salmonella_enterica_subsp_enterica_ATCC_14028.fasta").touch()
    (text_folder / "Escherichia～coli～K12.pt").touch()
    assert _normalized_maps(
        legacy.legacy_get_ATCC_ID_to_species_name_map(atcc_folder)
    ) == _normalized_maps(get_atcc_id_to_species_name_map(atcc_folder))
    assert _normalized_maps(
        legacy.legacy_get_original_strain_ID_to_species_name_map(text_folder)
    ) == _normalized_maps(get_original_strain_id_to_species_name_map(text_folder))
    first = {"a": np.array(["1", "2"])}
    second = {"a": np.array(["3"]), "b": np.array(["4"])}
    assert legacy.legacy_merge_dict(first, second) == merge_strain_maps(first, second)

    records = np.array(
        [
            [1, "Escherichia coli ATCC 25922", "[]", 1.0],
            [2, "Wrong species ATCC 25922", "[]", 1.0],
            [3, "Manual strain", "[]", 1.0],
        ],
        dtype=object,
    )
    legacy_filtered = legacy.legacy_exclude_wrong_species_ATCC_map(
        records.copy(), {"25922": "Escherichia"}
    )
    capsys.readouterr()
    shared_filtered = exclude_wrong_species_atcc_map(
        records.copy(), {"25922": "Escherichia"}
    )
    capsys.readouterr()
    np.testing.assert_array_equal(legacy_filtered, shared_filtered)


def test_shared_metrics_match_legacy_reference_operations():
    legacy = _load_legacy_module()
    labels = [0.0, 1.0, 2.0, 4.0]
    member_predictions = [
        [0.2, 0.8, 2.2, 3.7],
        [-0.1, 1.1, 1.9, 4.2],
    ]
    expected_ensemble = np.mean(np.array(member_predictions), axis=0)
    np.testing.assert_array_equal(
        ensemble_predictions(member_predictions), expected_ensemble
    )
    metrics = summarize_predictions(labels, expected_ensemble)
    assert metrics["r2"] == legacy.legacy_calculate_r2(labels, expected_ensemble)
    assert metrics["spearman"] == spearmanr(labels, expected_ensemble)[0]
    assert metrics["pearson"] == pearsonr(labels, expected_ensemble)[0]
    assert calculate_r2(labels, expected_ensemble) == metrics["r2"]


def test_specieswise_metrics_match_legacy_reference_operations():
    labels = {"Species A": [0.0, 1.0, 2.0], "Species B": [3.0]}
    predictions = {"Species A": [0.2, 0.8, 2.1], "Species B": [2.5]}
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = specieswise_metrics(labels, predictions)
        expected = {}
        for species_name in predictions.keys():
            species_labels = labels[species_name]
            species_predictions = predictions[species_name]
            r2 = calculate_r2(species_labels, species_predictions)
            mse = np.mean(
                (np.array(species_labels) - np.array(species_predictions)) ** 2
            )
            if len(species_labels) > 1:
                spearman = spearmanr(species_labels, species_predictions)[0]
                pearson = pearsonr(species_labels, species_predictions)[0]
            else:
                spearman = pearson = None
            expected[species_name] = [r2, mse, spearman, pearson]

    assert actual.keys() == expected.keys()
    for species_name in expected:
        for actual_value, expected_value in zip(
            actual[species_name], expected[species_name]
        ):
            if expected_value is None:
                assert actual_value is None
            else:
                assert actual_value == expected_value


@dataclass
class _TrainingComponents:
    genome_attention: FirstTokenAttentionGenome
    text_attention: FirstTokenAttentionGenome
    regression_head: RegressionHead
    classification_head: RegressionHead
    missing_genome_embedding: torch.nn.Parameter
    optimizer: torch.optim.Optimizer


def _make_training_components() -> _TrainingComponents:
    torch.manual_seed(101)
    genome_attention = FirstTokenAttentionGenome(6, 8, 2, 0.1)
    text_attention = FirstTokenAttentionGenome(6, 4, 2, 0.1)
    regression_head = RegressionHead(12, 7, 3, 1, 0.2)
    classification_head = RegressionHead(12, 7, 3, 1, 0.2)
    missing_genome_embedding = torch.nn.Parameter(torch.randn(1, 8))
    optimizer = torch.optim.Adam(genome_attention.parameters(), lr=1e-3)
    optimizer.add_param_group({"params": text_attention.parameters(), "lr": 1e-3})
    optimizer.add_param_group({"params": regression_head.parameters(), "lr": 1e-3})
    optimizer.add_param_group({"params": classification_head.parameters(), "lr": 1e-3})
    optimizer.add_param_group({"params": [missing_genome_embedding], "lr": 1e-3})
    return _TrainingComponents(
        genome_attention,
        text_attention,
        regression_head,
        classification_head,
        missing_genome_embedding,
        optimizer,
    )


def _training_batch(*, has_genome: bool, classification: bool) -> dict:
    generator = torch.Generator().manual_seed(303)
    batch = {
        "label": (
            torch.tensor([0.0, 1.0])
            if classification
            else torch.tensor([0.5, 1.5])
        ),
        "padded_text_embeddings": torch.randn(2, 3, 4, generator=generator),
        "text_attn_masks": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.uint8),
        "strain_names": ["strain-a", "strain-b"],
        "mol_emb": torch.randn(2, 6, generator=generator),
    }
    if has_genome:
        batch.update(
            {
                "padded_genome_embeddings": torch.randn(
                    2, 2, 8, generator=generator
                ),
                "genome_attn_masks": torch.tensor(
                    [[1, 1], [1, 0]], dtype=torch.uint8
                ),
            }
        )
    return batch


def _legacy_inline_optimizer_step(
    components: _TrainingComponents,
    batch: dict,
    *,
    has_genome: bool,
    classification: bool,
):
    head = (
        components.classification_head
        if classification
        else components.regression_head
    )
    criterion = (
        torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss()
    )
    optimizer = components.optimizer
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    optimizer.zero_grad()
    labels = batch["label"]
    molecule = batch["mol_emb"]
    text = batch["padded_text_embeddings"]
    text_mask = batch["text_attn_masks"]
    if has_genome:
        genome = batch["padded_genome_embeddings"]
        genome_mask = batch["genome_attn_masks"]
    else:
        genome = components.missing_genome_embedding[:, None, :].expand(
            molecule.shape[0], 1, -1
        )
        genome_mask = torch.from_numpy(np.array([1]))[None, :].expand(
            molecule.shape[0], -1
        )
    genome_output = components.genome_attention(molecule, genome, 1 - genome_mask)
    text_output = components.text_attention(molecule, text, 1 - text_mask)
    if classification:
        fused = torch.cat((genome_output, text_output), dim=1)
    else:
        fused = torch.cat(
            (genome_output.reshape(-1, 8), text_output.reshape(-1, 4)), dim=1
        )
    logits = head(fused).squeeze()
    loss = criterion(logits, labels.squeeze())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not has_genome:
        torch.nn.utils.clip_grad_norm_(
            [components.missing_genome_embedding], max_norm=1.0
        )
    torch.nn.utils.clip_grad_norm_(components.genome_attention.parameters(), max_norm=1.0)
    # Intentional historical behavior: classification also clips reg_head.
    torch.nn.utils.clip_grad_norm_(components.regression_head.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return logits, loss


def _named_training_parameters(components: _TrainingComponents) -> dict:
    named = {}
    for prefix, module in (
        ("genome", components.genome_attention),
        ("text", components.text_attention),
        ("regression", components.regression_head),
        ("classification", components.classification_head),
    ):
        named.update({f"{prefix}.{name}": value for name, value in module.named_parameters()})
    named["missing_genome_embedding"] = components.missing_genome_embedding
    return named


def test_all_four_optimizer_steps_match_legacy_operations():
    for has_genome in (True, False):
        for classification in (False, True):
            legacy_components = _make_training_components()
            shared_components = _make_training_components()
            batch = _training_batch(
                has_genome=has_genome, classification=classification
            )

            torch.manual_seed(707)
            legacy_logits, legacy_loss = _legacy_inline_optimizer_step(
                legacy_components,
                batch,
                has_genome=has_genome,
                classification=classification,
            )
            shared_head = (
                shared_components.classification_head
                if classification
                else shared_components.regression_head
            )
            shared_criterion = (
                torch.nn.BCEWithLogitsLoss()
                if classification
                else torch.nn.MSELoss()
            )
            torch.manual_seed(707)
            shared_result = strainwise_optimizer_step(
                batch,
                device=torch.device("cpu"),
                genome_attention=shared_components.genome_attention,
                text_attention=shared_components.text_attention,
                prediction_head=shared_head,
                legacy_regression_head_for_clipping=shared_components.regression_head,
                criterion=shared_criterion,
                missing_genome_embedding=shared_components.missing_genome_embedding,
                optimizer=shared_components.optimizer,
                scaler=torch.amp.GradScaler("cuda", enabled=False),
                has_genome=has_genome,
                reshape_outputs=not classification,
                autocast_enabled=False,
                epoch=0,
                freeze_epochs=0,
            )

            torch.testing.assert_close(
                shared_result.logits, legacy_logits, rtol=0, atol=0
            )
            torch.testing.assert_close(
                shared_result.loss, legacy_loss, rtol=0, atol=0
            )
            legacy_parameters = _named_training_parameters(legacy_components)
            shared_parameters = _named_training_parameters(shared_components)
            assert legacy_parameters.keys() == shared_parameters.keys()
            for name in legacy_parameters:
                legacy_parameter = legacy_parameters[name]
                shared_parameter = shared_parameters[name]
                torch.testing.assert_close(
                    shared_parameter, legacy_parameter, rtol=0, atol=0, msg=name
                )
                if legacy_parameter.grad is None:
                    assert shared_parameter.grad is None, name
                else:
                    torch.testing.assert_close(
                        shared_parameter.grad,
                        legacy_parameter.grad,
                        rtol=0,
                        atol=0,
                        msg=name,
                    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not visible")
def test_cuda_amp_optimizer_step_matches_legacy_operations():
    device = torch.device("cuda:0")
    legacy_components = _make_training_components()
    shared_components = _make_training_components()
    for components in (legacy_components, shared_components):
        components.genome_attention.to(device)
        components.text_attention.to(device)
        components.regression_head.to(device)
        components.classification_head.to(device)
        components.missing_genome_embedding.data = (
            components.missing_genome_embedding.data.to(device)
        )
    batch = _training_batch(has_genome=True, classification=False)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    torch.manual_seed(909)
    legacy_components.optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=True):
        genome_output = legacy_components.genome_attention(
            batch["mol_emb"],
            batch["padded_genome_embeddings"],
            1 - batch["genome_attn_masks"],
        )
        text_output = legacy_components.text_attention(
            batch["mol_emb"],
            batch["padded_text_embeddings"],
            1 - batch["text_attn_masks"],
        )
        fused = torch.cat(
            (genome_output.reshape(-1, 8), text_output.reshape(-1, 4)), dim=1
        )
        legacy_logits = legacy_components.regression_head(fused).squeeze()
        legacy_loss = torch.nn.MSELoss()(legacy_logits, batch["label"].squeeze())
    legacy_scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    legacy_scaler.scale(legacy_loss).backward()
    legacy_scaler.step(legacy_components.optimizer)
    legacy_scaler.update()

    torch.manual_seed(909)
    shared_result = strainwise_optimizer_step(
        batch,
        device=device,
        genome_attention=shared_components.genome_attention,
        text_attention=shared_components.text_attention,
        prediction_head=shared_components.regression_head,
        legacy_regression_head_for_clipping=shared_components.regression_head,
        criterion=torch.nn.MSELoss(),
        missing_genome_embedding=shared_components.missing_genome_embedding,
        optimizer=shared_components.optimizer,
        scaler=torch.amp.GradScaler("cuda", init_scale=128.0),
        has_genome=True,
        reshape_outputs=True,
        autocast_enabled=True,
        epoch=0,
        freeze_epochs=5000,
    )
    torch.testing.assert_close(shared_result.logits, legacy_logits, rtol=0, atol=0)
    torch.testing.assert_close(shared_result.loss, legacy_loss, rtol=0, atol=0)
    legacy_parameters = _named_training_parameters(legacy_components)
    shared_parameters = _named_training_parameters(shared_components)
    for name in legacy_parameters:
        torch.testing.assert_close(
            shared_parameters[name], legacy_parameters[name], rtol=0, atol=0, msg=name
        )
        if legacy_parameters[name].grad is None:
            assert shared_parameters[name].grad is None, name
        else:
            assert torch.isfinite(legacy_parameters[name].grad).all(), name
            assert torch.isfinite(shared_parameters[name].grad).all(), name
            torch.testing.assert_close(
                shared_parameters[name].grad,
                legacy_parameters[name].grad,
                rtol=0,
                atol=0,
            )


def _normalized_maps(result):
    first, second = result
    return first, {key: tuple(value.tolist()) for key, value in second.items()}
