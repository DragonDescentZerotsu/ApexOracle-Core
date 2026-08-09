from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from apexoracle.data.hierarchical_mic_preparation import (
    HoldoutSplit,
    PreparedHierarchicalMicData,
    load_fixed_strain_holdout_manifest,
    prepare_hierarchical_mic_data,
)
from apexoracle.training.hierarchical_mic_runner import (
    DEFAULT_CONFIG,
    HierarchicalMicConfig,
    HierarchicalMicPaths,
    build_model_bundle,
    checkpoint_filename,
    legacy_group_token,
    main,
    prepare_holdout_frames,
    run_holdout,
    RuntimeFeatures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _group(records: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: records[np.where(records[:, 1] == name)[0]] for name in set(records[:, 1])
    }


def _prepared() -> PreparedHierarchicalMicData:
    columns = ["DBAASP_id", "strain_name", "SMILES", "MIC"]
    genome = np.array(
        [
            [1, "g-held", "[1]", 10.0],
            [2, "g-train", "[1]", 20.0],
            [3, "#004", "[1]", 30.0],
            [4, "17978", "[1]", 40.0],
        ],
        dtype=object,
    )
    text_extra = np.array(
        [
            [5, "t-held", "[1]", 50.0],
            [6, "Staphylococcus aureus RN4220", "[1]", 60.0],
            [7, "t-train", "[1]", 70.0],
        ],
        dtype=object,
    )
    all_records = np.concatenate((genome, text_extra))
    auxiliary = np.array(
        [
            [101, "#004", "[1]", 1.0],
            [102, "17978", "[1]", 0.0],
            [103, "Staphylococcus aureus RN4220", "[1]", 1.0],
        ],
        dtype=object,
    )
    return PreparedHierarchicalMicData(
        columns=columns,
        genome_text_records=genome,
        genome_or_text_records=all_records,
        small_molecule_records=auxiliary,
        genome_text_groups=_group(genome),
        genome_or_text_groups=_group(all_records),
        small_molecule_groups=_group(auxiliary),
        atcc_id_to_species={
            "g-held": "species-held",
            "g-train": "species-train",
            "#004": "species-004",
            "17978": "species-17978",
        },
        original_strain_to_species={
            "t-held": "species-text-held",
            "t-train": "species-text-train",
            "Staphylococcus aureus RN4220": "Staphylococcus aureus",
        },
        species_to_strains={},
        taxonomy_aliases={},
    )


def _small_config(protocol: str = "strain") -> HierarchicalMicConfig:
    paths = HierarchicalMicPaths(
        genome_embeddings=Path("genome"),
        atcc_text_embeddings=Path("atcc-text"),
        text_only_embeddings=Path("text-only"),
        peptide_embeddings=Path("peptide.pt"),
        small_molecule_embeddings=Path("small.pt"),
        output_dir=Path("output"),
    )
    return HierarchicalMicConfig(
        protocol_family="paper_legacy_hierarchical_mic_holdouts",
        holdout_protocol=protocol,
        holdout_adapter=(
            "legacy_within_species_three_fold"
            if protocol == "strain"
            else "taxonomy_tree_agglomerative_clusters"
        ),
        holdout_group_names=("group",),
        holdout_clusters=None if protocol == "strain" else 1,
        holdout_tree=None if protocol == "strain" else Path("tree.phy"),
        molecule_embedding_dim=6,
        genome_embedding_dim=4,
        text_embedding_dim=4,
        attention_heads=2,
        attention_dropout=0.1,
        head_hidden_dims=(2, 128),
        head_dropout=0.2,
        regression_targets=1,
        ensembles_per_group=1,
        ensemble_seeds=(42,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-5,
        weight_decay=0.0,
        scheduler_eta_min=1e-10,
        freeze_epochs=5000,
        genome_embedding_scale=1e14,
        text_embedding_scale=1.0,
        paths=paths,
    )


@pytest.mark.parametrize(
    ("protocol", "expected_groups", "output_fragment"),
    [
        ("strain", 3, "strain_wise_w_SM_b_attn"),
        ("species", 11, "11_species_w_SM"),
        ("phylum", 3, "3_species_w_SM"),
    ],
)
def test_one_config_loads_all_holdout_protocols(
    protocol: str, expected_groups: int, output_fragment: str
):
    config = HierarchicalMicConfig.load(
        REPO_ROOT / DEFAULT_CONFIG,
        REPO_ROOT,
        holdout_protocol=protocol,
    )
    assert config.holdout_protocol == protocol
    assert len(config.holdout_group_names) == expected_groups
    assert output_fragment in str(config.paths.output_dir)
    assert config.ensembles_per_group == 7
    assert config.head_hidden_dims == (3072, 128)
    if protocol == "strain":
        assert config.holdout_tree is None
    else:
        assert config.holdout_tree is not None
        assert config.holdout_tree.is_absolute()


def test_config_has_one_shared_model_and_no_per_holdout_model_copy():
    raw = yaml.safe_load((REPO_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))
    assert set(raw["holdouts"]) == {"strain", "species", "phylum"}
    assert "model" in raw
    assert all("model" not in holdout for holdout in raw["holdouts"].values())


def test_holdout_adapter_is_the_only_input_to_common_frame_builder():
    prepared = _prepared()
    split = HoldoutSplit(
        protocol="strain",
        group_names=("group",),
        test_groups=(("g-held", "t-held", "#004"),),
    )
    frames = prepare_holdout_frames(prepared, split, 0)

    assert set(frames.genome_text_test["strain_name"]) == {"g-held", "#004"}
    assert set(frames.text_only_test["strain_name"]) == {"t-held"}
    assert set(frames.small_molecule_genome_text_train["strain_name"]) == {"17978"}
    assert set(frames.small_molecule_text_only_train["strain_name"]) == {
        "Staphylococcus aureus RN4220"
    }


def test_holdout_frames_preserve_appended_mic_audit_columns() -> None:
    prepared = _prepared()
    prepared.columns.append("censor_class")
    prepared.genome_text_records = np.column_stack(
        [prepared.genome_text_records, ["none"] * len(prepared.genome_text_records)]
    )
    prepared.genome_or_text_records = np.column_stack(
        [
            prepared.genome_or_text_records,
            ["none"] * len(prepared.genome_or_text_records),
        ]
    )
    prepared.genome_text_groups = _group(prepared.genome_text_records)
    prepared.genome_or_text_groups = _group(prepared.genome_or_text_records)
    split = HoldoutSplit(
        protocol="strain",
        group_names=("group",),
        test_groups=(("g-held", "t-held", "#004"),),
    )

    frames = prepare_holdout_frames(prepared, split, 0)

    assert "censor_class" in frames.genome_text_test.columns
    assert list(frames.small_molecule_genome_text_train.columns) == prepared.columns[:4]
    assert list(frames.small_molecule_text_only_train.columns) == prepared.columns[:4]


def test_prepare_data_rejects_path_and_frame_inputs_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_hierarchical_mic_data(
            tmp_path,
            mic_data_path=tmp_path / "mic.csv",
            mic_frame=pd.DataFrame(),
        )


def test_shared_model_bundle_preserves_five_legacy_optimizer_groups():
    config = _small_config()
    bundle = build_model_bundle(
        config, genome_dim=4, text_dim=4, device=torch.device("cpu")
    )
    assert [len(group["params"]) for group in bundle.optimizer.param_groups] == [
        18,
        18,
        6,
        6,
        1,
    ]
    assert all(
        module.training
        for module in (
            bundle.genome_attention,
            bundle.text_attention,
            bundle.regression_head,
            bundle.classification_head,
        )
    )


def test_checkpoint_and_log_tokens_preserve_each_legacy_driver_contract():
    strain = HoldoutSplit("strain", ("fold 1",), (("x",),))
    phylum = HoldoutSplit("phylum", ("Fungi",), (("x",),))
    strain_config = _small_config("strain")
    phylum_config = _small_config("phylum")

    assert legacy_group_token(strain_config, strain, 0) == "0"
    assert legacy_group_token(phylum_config, phylum, 0) == "Fungi"
    assert checkpoint_filename(strain_config, strain, 0, 2).endswith(
        "Strain_wise_best_R2_group_0_ensemble_2.pth"
    )
    assert checkpoint_filename(phylum_config, phylum, 0, 2).endswith(
        "Species_wise_best_R2_group_Fungi_ensemble_2.pth"
    )


def test_canonical_entrypoint_does_not_execute_a_root_legacy_driver():
    source = (REPO_ROOT / "scripts/reproduce/run_hierarchical_mic.py").read_text(
        encoding="utf-8"
    )
    assert "runpy" not in source
    assert "DP_inhouse" not in source
    assert "hierarchical_mic_runner" in source


def test_cli_requires_explicit_legacy_split_acknowledgement():
    with pytest.raises(SystemExit, match="Refusing an ambiguous rerun"):
        main(["--protocol", "strain", "--test-group", "0"])


def test_fixed_strain_manifest_loader(tmp_path: Path):
    manifest = tmp_path / "strain.json"
    manifest.write_text(
        """
{
  "protocol": "deterministic_legacy_codepath_candidate",
  "historical_membership_status": "not_fully_recovered",
  "folds": [
    {"fold": 1, "test_strain_ids": ["b", "a", "a"]},
    {"fold": 0, "test_strain_ids": ["c"]}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    split = load_fixed_strain_holdout_manifest(manifest)
    assert split.protocol == "strain"
    assert split.group_names == ("fold 1", "fold 2")
    assert split.test_groups == (("c",), ("a", "b"))


def test_run_holdout_rejects_invalid_ensemble_selection():
    config = _small_config("strain")
    with pytest.raises(ValueError, match="Duplicate ensemble"):
        run_holdout(
            config,
            _prepared(),
            HoldoutSplit(
                protocol="strain",
                group_names=("group",),
                test_groups=(("g-held",),),
            ),
            RuntimeFeatures({}, {}, {}, {}, {}),
            group=0,
            device=torch.device("cpu"),
            ensemble_indices=(0, 0),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_unified_runner_one_epoch_cuda_smoke(tmp_path: Path):
    prepared = _prepared()
    extra_auxiliary = np.array(
        [
            [104, "17978", "[1]", 1.0],
            [105, "Staphylococcus aureus RN4220", "[1]", 0.0],
        ],
        dtype=object,
    )
    prepared.small_molecule_records = np.concatenate(
        (prepared.small_molecule_records, extra_auxiliary)
    )
    prepared.small_molecule_groups = _group(prepared.small_molecule_records)
    split = HoldoutSplit(
        protocol="strain",
        group_names=("group",),
        test_groups=(("g-held", "t-held", "#004"),),
    )
    config = _small_config("strain")
    config = HierarchicalMicConfig(
        **{
            **config.__dict__,
            "paths": HierarchicalMicPaths(
                **{**config.paths.__dict__, "output_dir": tmp_path}
            ),
        }
    )
    device = torch.device("cuda:0")
    torch.manual_seed(123)
    all_ids = set(prepared.genome_or_text_records[:, 0]) | set(
        prepared.small_molecule_records[:, 0]
    )
    molecule_embeddings = {molecule_id: torch.randn(1, 6) for molecule_id in all_ids}
    genome_names = set(prepared.genome_text_records[:, 1])
    text_only_names = set(prepared.genome_or_text_records[:, 1]) - genome_names
    features = RuntimeFeatures(
        genome_embeddings={
            name: torch.randn(2, 4, dtype=torch.bfloat16, device=device)
            for name in genome_names
        },
        atcc_text_embeddings={
            name: torch.randn(2, 4, dtype=torch.bfloat16, device=device)
            for name in genome_names
        },
        text_only_embeddings={
            name: torch.randn(2, 4, dtype=torch.bfloat16, device=device)
            for name in text_only_names
        },
        peptide_embeddings=molecule_embeddings,
        small_molecule_embeddings={},
    )

    metrics = run_holdout(
        config,
        prepared,
        split,
        features,
        group=0,
        device=device,
    )

    assert set(metrics) == {"r2", "spearman", "pearson"}
    checkpoint = tmp_path / checkpoint_filename(config, split, 0, 0)
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "R2",
        "optimizer_state_dict",
        "re_head_state_dict",
        "cls_head_state_dict",
        "co_cross_attn_genome",
        "co_cross_attn_text",
        "learnable_embedding_weight",
    }
