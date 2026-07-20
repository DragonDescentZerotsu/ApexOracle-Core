"""Shared runner for the paper-era hierarchical MIC holdout protocols.

The orchestration in this module intentionally preserves several historical
behaviors that would normally be changed in a new training pipeline: evaluation
is performed while every module remains in ``train`` mode, loaders are combined
with ``zip_longest``, each non-empty modality batch performs its own optimizer
step, and the held-out group selects checkpoints directly by R2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from apexoracle.data.hierarchical_mic import (
    OnlineMoleculeStrainDataset,
    OnlineMoleculeTextOnlyDataset,
    StrainEmbeddingDataset,
    TextOnlyStrainEmbeddingDataset,
    collate_genome_text_classification,
    collate_genome_text_regression,
    collate_text_classification,
    collate_text_regression,
    collate_online_genome_text_classification,
    collate_online_genome_text_regression,
    collate_online_text_classification,
    collate_online_text_regression,
)
from apexoracle.data.hierarchical_mic_preparation import (
    HoldoutSplit,
    PreparedHierarchicalMicData,
    build_holdout_split,
    holdout_record_counts,
    prepare_hierarchical_mic_data,
)
from apexoracle.evaluation.hierarchical_mic import (
    HierarchicalMicPredictionAccumulator,
    LegacyBestMetricTracker,
    calculate_r2,
    ensemble_predictions,
    specieswise_metrics,
    summarize_partition_or_sentinel,
    summarize_predictions,
)
from apexoracle.features.precomputed import (
    load_all_embeddings,
    load_text_only_embeddings,
)
from apexoracle.features.kmer import load_legacy_projected_embeddings
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.models.hf_molecule_encoder import (
    HFMoleculeEncoderConfig,
    load_legacy_molecule_encoder,
    load_legacy_tokenizer,
    unfreeze_legacy_molecule_encoder,
)
from apexoracle.training.hierarchical_mic import (
    build_legacy_cosine_scheduler,
    hierarchical_mic_batch_forward,
    hierarchical_mic_optimizer_step,
    legacy_hierarchical_checkpoint_payload,
    legacy_zip_longest_loaders,
)


LOGGER = logging.getLogger("apexoracle.hierarchical_mic_runner")
DEFAULT_CONFIG = Path("configs/hierarchical_mic/legacy_mdlm.yaml")


@dataclass(frozen=True)
class HierarchicalMicPaths:
    genome_embeddings: Path
    atcc_text_embeddings: Path
    text_only_embeddings: Path
    peptide_embeddings: Path
    small_molecule_embeddings: Path
    output_dir: Path
    mic_records: Path = Path(
        "DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv"
    )
    small_molecule_records: Path = Path(
        "DataPrepare/Data/small_molecule/processed/"
        "small_molecule_Evo_binary_data_SELFIES.csv"
    )

    @classmethod
    def from_config(
        cls,
        repo_root: Path,
        values: Mapping[str, str],
        *,
        output_path: str,
        output_override: Path | None = None,
    ) -> "HierarchicalMicPaths":
        def resolve(value: str) -> Path:
            expanded = os.path.expandvars(value)
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in path: {value}")
            path = Path(expanded)
            return path if path.is_absolute() else repo_root / path

        output = output_override or resolve(output_path)
        if not output.is_absolute():
            output = repo_root / output
        return cls(
            genome_embeddings=resolve(values["genome_embeddings"]),
            atcc_text_embeddings=resolve(values["atcc_text_embeddings"]),
            text_only_embeddings=resolve(values["text_only_embeddings"]),
            peptide_embeddings=resolve(values["peptide_embeddings"]),
            small_molecule_embeddings=resolve(values["small_molecule_embeddings"]),
            mic_records=resolve(
                values.get(
                    "mic_records",
                    "DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
                )
            ),
            small_molecule_records=resolve(
                values.get(
                    "small_molecule_records",
                    "DataPrepare/Data/small_molecule/processed/"
                    "small_molecule_Evo_binary_data_SELFIES.csv",
                )
            ),
            output_dir=output,
        )

    def required_inputs(
        self, *, include_precomputed_molecule_embeddings: bool = True
    ) -> tuple[Path, ...]:
        shared = (
            self.genome_embeddings,
            self.atcc_text_embeddings,
            self.text_only_embeddings,
            self.mic_records,
            self.small_molecule_records,
        )
        if not include_precomputed_molecule_embeddings:
            return shared
        return shared + (self.peptide_embeddings, self.small_molecule_embeddings)


@dataclass(frozen=True)
class GenomeEmbeddingAdapterConfig:
    kind: str
    input_dim: int
    output_dim: int
    trainable: bool
    initialization: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GenomeEmbeddingAdapterConfig":
        return cls(
            kind=str(values["kind"]),
            input_dim=int(values["input_dim"]),
            output_dim=int(values["output_dim"]),
            trainable=bool(values["trainable"]),
            initialization=str(values["initialization"]),
        )

    def validate(self, *, configured_output_dim: int) -> None:
        if self.kind != "legacy_frozen_kmer_projection":
            raise ValueError(f"Unsupported genome embedding adapter: {self.kind}")
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("Genome adapter dimensions must be positive")
        if self.output_dim != configured_output_dim:
            raise ValueError(
                "Genome adapter output dimension must match model genome dimension"
            )
        if self.trainable:
            raise ValueError("The legacy k-mer projection must remain frozen")
        if self.initialization != "xavier_uniform_gain_0.01_unseeded_process_state":
            raise ValueError("Unexpected legacy k-mer projection initialization")


@dataclass(frozen=True)
class HierarchicalMicConfig:
    protocol_family: str
    holdout_protocol: str
    holdout_adapter: str
    holdout_group_names: tuple[str, ...]
    holdout_clusters: int | None
    holdout_tree: Path | None
    molecule_embedding_dim: int
    genome_embedding_dim: int
    text_embedding_dim: int
    attention_heads: int
    attention_dropout: float
    head_hidden_dims: tuple[int, int]
    head_dropout: float
    regression_targets: int
    ensembles_per_group: int
    ensemble_seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler_eta_min: float
    freeze_epochs: int
    genome_embedding_scale: float
    text_embedding_scale: float
    paths: HierarchicalMicPaths
    molecule_encoder: HFMoleculeEncoderConfig | None = None
    genome_embedding_adapter: GenomeEmbeddingAdapterConfig | None = None

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        holdout_protocol: str,
        molecule_encoder_name: str | None = None,
        epochs: int | None = None,
        weight_decay: float | None = None,
        output_dir: Path | None = None,
    ) -> "HierarchicalMicConfig":
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Config must contain a mapping: {config_path}")
        if holdout_protocol not in {"strain", "species", "phylum"}:
            raise ValueError(f"Unsupported holdout protocol: {holdout_protocol}")
        model = raw["model"]
        training = raw["training"]
        holdout = raw["holdouts"][holdout_protocol]
        outputs = raw["outputs"]
        if holdout_protocol not in outputs:
            raise ValueError(f"No output path configured for {holdout_protocol}")
        encoder_profile = None
        encoder_profiles = raw.get("molecule_encoders")
        if encoder_profiles is not None:
            if molecule_encoder_name is None:
                available = ", ".join(sorted(encoder_profiles))
                raise ValueError(
                    "This config requires --molecule-encoder; available: " + available
                )
            if molecule_encoder_name not in encoder_profiles:
                raise ValueError(f"Unknown molecule encoder: {molecule_encoder_name}")
            encoder_profile = encoder_profiles[molecule_encoder_name]
        elif molecule_encoder_name is not None:
            raise ValueError(
                "--molecule-encoder was provided for a config without encoder profiles"
            )

        output_path = (
            str(encoder_profile["output"])
            if encoder_profile is not None
            else outputs[holdout_protocol]
        )
        paths = HierarchicalMicPaths.from_config(
            repo_root,
            raw["paths"],
            output_path=output_path,
            output_override=output_dir,
        )
        adapter_values = raw.get("genome_embedding_adapter")
        genome_embedding_adapter = (
            None
            if adapter_values is None
            else GenomeEmbeddingAdapterConfig.from_mapping(adapter_values)
        )
        molecule_encoder = None
        if encoder_profile is not None:
            molecule_encoder = HFMoleculeEncoderConfig(
                name=molecule_encoder_name,
                model_name=str(encoder_profile["model_name"]),
                revision=str(encoder_profile["revision"]),
                hidden_size=int(encoder_profile["hidden_size"]),
                tokenizer_kind=str(encoder_profile.get("tokenizer_kind", "auto")),
                trust_remote_code=bool(
                    encoder_profile.get("trust_remote_code", False)
                ),
                initial_mode=str(encoder_profile["initial_mode"]),
                pooling=str(encoder_profile.get("pooling", "first_token")),
                max_length=int(encoder_profile.get("max_length", 512)),
                optimizer_learning_rate=float(
                    encoder_profile["optimizer_learning_rate"]
                ),
                optimizer_weight_decay_multiplier=float(
                    encoder_profile["optimizer_weight_decay_multiplier"]
                ),
                checkpoint_state_key=str(encoder_profile["checkpoint_state_key"]),
            )

        def training_value(name: str):
            if encoder_profile is not None and name in encoder_profile:
                return encoder_profile[name]
            return training[name]

        config = cls(
            protocol_family=str(raw["protocol_family"]),
            holdout_protocol=holdout_protocol,
            holdout_adapter=str(holdout["adapter"]),
            holdout_group_names=tuple(str(name) for name in holdout["group_names"]),
            holdout_clusters=(
                None if "clusters" not in holdout else int(holdout["clusters"])
            ),
            holdout_tree=(
                None
                if "tree" not in holdout
                else (
                    Path(holdout["tree"])
                    if Path(holdout["tree"]).is_absolute()
                    else repo_root / holdout["tree"]
                )
            ),
            molecule_embedding_dim=int(
                model["molecule_embedding_dim"]
                if molecule_encoder is None
                else molecule_encoder.hidden_size
            ),
            genome_embedding_dim=int(model["genome_embedding_dim"]),
            text_embedding_dim=int(model["text_embedding_dim"]),
            attention_heads=int(model["attention_heads"]),
            attention_dropout=float(model["attention_dropout"]),
            head_hidden_dims=tuple(int(value) for value in model["head_hidden_dims"]),
            head_dropout=float(model["head_dropout"]),
            regression_targets=int(model["regression_targets"]),
            ensembles_per_group=int(training_value("ensembles_per_group")),
            ensemble_seeds=tuple(int(seed) for seed in training["ensemble_seeds"]),
            epochs=int(training["epochs"] if epochs is None else epochs),
            batch_size=int(training_value("batch_size")),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(
                training["weight_decay"] if weight_decay is None else weight_decay
            ),
            scheduler_eta_min=float(training["scheduler_eta_min"]),
            freeze_epochs=int(training_value("freeze_epochs")),
            genome_embedding_scale=float(training["genome_embedding_scale"]),
            text_embedding_scale=float(training["text_embedding_scale"]),
            molecule_encoder=molecule_encoder,
            genome_embedding_adapter=genome_embedding_adapter,
            paths=paths,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol_family != "paper_legacy_hierarchical_mic_holdouts":
            raise ValueError(f"Unsupported protocol family: {self.protocol_family}")
        if self.holdout_protocol not in {"strain", "species", "phylum"}:
            raise ValueError(f"Unsupported holdout protocol: {self.holdout_protocol}")
        expected_adapter = {
            "strain": "legacy_within_species_three_fold",
            "species": "taxonomy_tree_agglomerative_clusters",
            "phylum": "taxonomy_tree_agglomerative_clusters",
        }[self.holdout_protocol]
        if self.holdout_adapter != expected_adapter:
            raise ValueError(
                f"Unsupported {self.holdout_protocol} adapter: {self.holdout_adapter}"
            )
        if self.holdout_protocol == "strain":
            if self.holdout_clusters is not None or self.holdout_tree is not None:
                raise ValueError("Strain holdout must not configure a tree or clusters")
        elif self.holdout_clusters is None or self.holdout_tree is None:
            raise ValueError(
                f"{self.holdout_protocol} holdout requires clusters and a taxonomy tree"
            )
        if self.ensembles_per_group < 1:
            raise ValueError("ensembles_per_group must be positive")
        if len(self.ensemble_seeds) < self.ensembles_per_group:
            raise ValueError("Not enough ensemble seeds for ensembles_per_group")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.genome_embedding_dim % self.attention_heads:
            raise ValueError(
                "genome_embedding_dim must be divisible by attention_heads"
            )
        if self.text_embedding_dim % self.attention_heads:
            raise ValueError("text_embedding_dim must be divisible by attention_heads")
        expected_first_hidden = (
            self.genome_embedding_dim + self.text_embedding_dim
        ) // 4
        if self.head_hidden_dims != (expected_first_hidden, 128):
            raise ValueError(
                "Legacy head dimensions must be "
                f"({expected_first_hidden}, 128), got {self.head_hidden_dims}"
            )
        if self.molecule_encoder is not None:
            self.molecule_encoder.validate()
        if self.genome_embedding_adapter is not None:
            self.genome_embedding_adapter.validate(
                configured_output_dim=self.genome_embedding_dim
            )


@dataclass
class HoldoutFrames:
    genome_text_train: pd.DataFrame
    genome_text_test: pd.DataFrame
    text_only_train: pd.DataFrame
    text_only_test: pd.DataFrame
    small_molecule_genome_text_train: pd.DataFrame | None
    small_molecule_text_only_train: pd.DataFrame | None
    genome_text_train_mean_mic: float
    text_only_train_mean_mic: float


@dataclass
class RuntimeFeatures:
    genome_embeddings: Mapping[str, torch.Tensor]
    atcc_text_embeddings: Mapping[str, torch.Tensor]
    text_only_embeddings: Mapping[str, torch.Tensor]
    peptide_embeddings: Mapping[Any, torch.Tensor] | None
    small_molecule_embeddings: Mapping[Any, torch.Tensor] | None
    tokenizer: Any | None = None
    genome_embedding_adapter: nn.Module | None = None

    @property
    def all_text_embeddings(self) -> dict[str, torch.Tensor]:
        return dict(self.atcc_text_embeddings) | dict(self.text_only_embeddings)


@dataclass
class HoldoutLoaders:
    genome_text_train: DataLoader
    genome_text_test: DataLoader
    text_only_train: DataLoader
    text_only_test: DataLoader
    small_molecule_genome_text_train: DataLoader | list[None]
    small_molecule_text_only_train: DataLoader | list[None]


@dataclass
class ModelBundle:
    molecule_encoder: nn.Module | None
    genome_attention: nn.Module
    text_attention: nn.Module
    regression_head: nn.Module
    classification_head: nn.Module
    missing_genome_embedding: nn.Parameter
    regression_criterion: nn.Module
    classification_criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    scaler: Any


def _concatenate_groups(
    groups: Mapping[str, np.ndarray], names: set[str], columns: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        np.concatenate([groups[name] for name in names]), columns=columns
    )


def prepare_holdout_frames(
    prepared: PreparedHierarchicalMicData, split: HoldoutSplit, group: int
) -> HoldoutFrames:
    """Construct the six common partitions from one protocol's held-out strains."""

    if group < 0 or group >= len(split.test_groups):
        raise ValueError(f"group must be in [0, {len(split.test_groups) - 1}]")
    all_genome_names = set(prepared.genome_text_records[:, 1])
    all_names = set(prepared.genome_or_text_records[:, 1])
    test_names = set(split.test_groups[group])

    genome_test_names = test_names & all_genome_names
    genome_train_names = all_genome_names - genome_test_names
    text_test_names = (test_names & all_names) - genome_test_names
    text_train_names = all_names - text_test_names - genome_test_names

    genome_text_train = _concatenate_groups(
        prepared.genome_text_groups, genome_train_names, prepared.columns
    )
    genome_text_test = _concatenate_groups(
        prepared.genome_text_groups, genome_test_names, prepared.columns
    )
    text_only_train = _concatenate_groups(
        prepared.genome_or_text_groups, text_train_names, prepared.columns
    )
    text_only_test = _concatenate_groups(
        prepared.genome_or_text_groups, text_test_names, prepared.columns
    )

    genome_auxiliary = []
    text_auxiliary = []
    if split.protocol == "strain":
        # The strain-wise final driver checks these candidates in fixed order.
        for strain_name in ("#004", "17978"):
            if strain_name in genome_train_names:
                genome_auxiliary.extend(
                    line
                    for line in prepared.small_molecule_records
                    if line[1] == strain_name
                )
        if "Staphylococcus aureus RN4220" in text_train_names:
            text_auxiliary.extend(
                line
                for line in prepared.small_molecule_records
                if line[1] == "Staphylococcus aureus RN4220"
            )
    else:
        # Species/phylum legacy drivers iterate these set differences and then
        # concatenate per-strain blocks. Preserve that process-hash-dependent
        # ordering rather than silently adopting the strain-wise fixed order.
        genome_auxiliary_train = {"#004", "17978"} - (test_names & {"#004", "17978"})
        for strain_name in genome_auxiliary_train:
            genome_auxiliary.extend(prepared.small_molecule_groups[strain_name])
        text_auxiliary_train = {"Staphylococcus aureus RN4220"} - (
            test_names & {"Staphylococcus aureus RN4220"}
        )
        for strain_name in text_auxiliary_train:
            text_auxiliary.extend(prepared.small_molecule_groups[strain_name])

    return HoldoutFrames(
        genome_text_train=genome_text_train,
        genome_text_test=genome_text_test,
        text_only_train=text_only_train,
        text_only_test=text_only_test,
        small_molecule_genome_text_train=(
            pd.DataFrame(genome_auxiliary, columns=prepared.columns)
            if genome_auxiliary
            else None
        ),
        small_molecule_text_only_train=(
            pd.DataFrame(text_auxiliary, columns=prepared.columns)
            if text_auxiliary
            else None
        ),
        genome_text_train_mean_mic=-np.log10(genome_text_train["MIC"].mean() / 10),
        text_only_train_mean_mic=-np.log10(text_only_train["MIC"].mean() / 10),
    )


def load_runtime_features(
    config: HierarchicalMicConfig, device: torch.device, repo_root: Path
) -> RuntimeFeatures:
    paths = config.paths
    online = config.molecule_encoder is not None
    genome_adapter = None
    if config.genome_embedding_adapter is None:
        genome_embeddings = load_all_embeddings(
            paths.genome_embeddings,
            config.genome_embedding_scale,
            device,
            "genome",
        )
    else:
        adapter = config.genome_embedding_adapter
        genome_embeddings, genome_adapter = load_legacy_projected_embeddings(
            paths.genome_embeddings,
            input_dim=adapter.input_dim,
            output_dim=adapter.output_dim,
            scale=config.genome_embedding_scale,
            device=device,
        )
    return RuntimeFeatures(
        genome_embeddings=genome_embeddings,
        atcc_text_embeddings=load_all_embeddings(
            paths.atcc_text_embeddings,
            config.text_embedding_scale,
            device,
            "text (with corresponding genome)",
        ),
        text_only_embeddings=load_text_only_embeddings(
            paths.text_only_embeddings,
            config.text_embedding_scale,
            device,
            "text (without corresponding genome)",
        ),
        peptide_embeddings=(
            None if online else torch.load(paths.peptide_embeddings)
        ),
        small_molecule_embeddings=(
            None if online else torch.load(paths.small_molecule_embeddings)
        ),
        tokenizer=(
            load_legacy_tokenizer(config.molecule_encoder, repo_root)
            if online
            else None
        ),
        genome_embedding_adapter=genome_adapter,
    )


def build_holdout_loaders(
    frames: HoldoutFrames,
    features: RuntimeFeatures,
    *,
    batch_size: int,
) -> tuple[HoldoutLoaders, int, int]:
    all_text_embeddings = features.all_text_embeddings
    online = features.tokenizer is not None
    if online:
        tokenizer = features.tokenizer
        genome_text_train = OnlineMoleculeStrainDataset(
            frames.genome_text_train,
            tokenizer,
            features.genome_embeddings,
            features.atcc_text_embeddings,
            "peptide genome-text training set",
        )
        genome_text_test = OnlineMoleculeStrainDataset(
            frames.genome_text_test,
            tokenizer,
            features.genome_embeddings,
            features.atcc_text_embeddings,
            "peptide genome-text test set",
        )
        text_only_train = OnlineMoleculeTextOnlyDataset(
            frames.text_only_train,
            tokenizer,
            all_text_embeddings,
            "peptide text-only training set",
        )
        text_only_test = OnlineMoleculeTextOnlyDataset(
            frames.text_only_test,
            tokenizer,
            all_text_embeddings,
            "peptide text-only test set",
        )
        pad_token_id = tokenizer.pad_token_id
        genome_regression_collate = partial(
            collate_online_genome_text_regression,
            pad_token_id=pad_token_id,
        )
        text_regression_collate = partial(
            collate_online_text_regression,
            pad_token_id=pad_token_id,
        )
        genome_classification_collate = partial(
            collate_online_genome_text_classification,
            pad_token_id=pad_token_id,
        )
        text_classification_collate = partial(
            collate_online_text_classification,
            pad_token_id=pad_token_id,
        )
    else:
        common = (
            None,
            features.peptide_embeddings,
            features.small_molecule_embeddings,
        )
        genome_text_train = StrainEmbeddingDataset(
            frames.genome_text_train,
            common[0],
            features.genome_embeddings,
            features.atcc_text_embeddings,
            "peptide genome-text training set",
            common[1],
            common[2],
        )
        genome_text_test = StrainEmbeddingDataset(
            frames.genome_text_test,
            common[0],
            features.genome_embeddings,
            features.atcc_text_embeddings,
            "peptide genome-text test set",
            common[1],
            common[2],
        )
        text_only_train = TextOnlyStrainEmbeddingDataset(
            frames.text_only_train,
            common[0],
            all_text_embeddings,
            "peptide text-only training set",
            common[1],
            common[2],
        )
        text_only_test = TextOnlyStrainEmbeddingDataset(
            frames.text_only_test,
            common[0],
            all_text_embeddings,
            "peptide text-only test set",
            common[1],
            common[2],
        )
        genome_regression_collate = collate_genome_text_regression
        text_regression_collate = collate_text_regression
        genome_classification_collate = collate_genome_text_classification
        text_classification_collate = collate_text_classification

    genome_auxiliary = None
    if frames.small_molecule_genome_text_train is not None:
        if online:
            genome_auxiliary = OnlineMoleculeStrainDataset(
                frames.small_molecule_genome_text_train,
                features.tokenizer,
                features.genome_embeddings,
                features.atcc_text_embeddings,
                "small-molecule genome-text training set",
            )
        else:
            genome_auxiliary = StrainEmbeddingDataset(
                frames.small_molecule_genome_text_train,
                common[0],
                features.genome_embeddings,
                features.atcc_text_embeddings,
                "small-molecule genome-text training set",
                common[1],
                common[2],
            )
    text_auxiliary = None
    if frames.small_molecule_text_only_train is not None:
        if online:
            text_auxiliary = OnlineMoleculeTextOnlyDataset(
                frames.small_molecule_text_only_train,
                features.tokenizer,
                all_text_embeddings,
                "small-molecule text-only training set",
            )
        else:
            text_auxiliary = TextOnlyStrainEmbeddingDataset(
                frames.small_molecule_text_only_train,
                common[0],
                all_text_embeddings,
                "small-molecule text-only training set",
                common[1],
                common[2],
            )

    loaders = HoldoutLoaders(
        genome_text_train=DataLoader(
            genome_text_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=genome_regression_collate,
        ),
        genome_text_test=DataLoader(
            genome_text_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=genome_regression_collate,
        ),
        text_only_train=DataLoader(
            text_only_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=text_regression_collate,
        ),
        text_only_test=DataLoader(
            text_only_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=text_regression_collate,
        ),
        small_molecule_genome_text_train=(
            DataLoader(
                genome_auxiliary,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=genome_classification_collate,
            )
            if genome_auxiliary is not None
            else [None]
        ),
        small_molecule_text_only_train=(
            DataLoader(
                text_auxiliary,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=text_classification_collate,
            )
            if text_auxiliary is not None
            else [None]
        ),
    )
    first = genome_text_train[0]
    return loaders, first["genome_embedding"].shape[1], first["text_embedding"].shape[1]


def build_model_bundle(
    config: HierarchicalMicConfig,
    *,
    genome_dim: int,
    text_dim: int,
    device: torch.device,
) -> ModelBundle:
    if (
        genome_dim != config.genome_embedding_dim
        or text_dim != config.text_embedding_dim
    ):
        raise ValueError(
            "Runtime embedding dimensions do not match config: "
            f"got genome={genome_dim}, text={text_dim}; expected "
            f"genome={config.genome_embedding_dim}, text={config.text_embedding_dim}"
        )
    molecule_encoder = (
        None
        if config.molecule_encoder is None
        else load_legacy_molecule_encoder(
            config.molecule_encoder,
            device=device,
        )
    )
    genome_attention = FirstTokenAttentionGenome(
        config.molecule_embedding_dim,
        genome_dim,
        config.attention_heads,
        config.attention_dropout,
    ).to(device)
    text_attention = FirstTokenAttentionGenome(
        config.molecule_embedding_dim,
        text_dim,
        config.attention_heads,
        config.attention_dropout,
    ).to(device)
    fused_dim = genome_dim + text_dim
    regression_head = RegressionHead(
        fused_dim,
        config.head_hidden_dims[0],
        config.head_hidden_dims[1],
        config.regression_targets,
        config.head_dropout,
    ).to(device)
    classification_head = RegressionHead(
        fused_dim,
        config.head_hidden_dims[0],
        config.head_hidden_dims[1],
        1,
        config.head_dropout,
    ).to(device)
    missing_genome_embedding = nn.Parameter(torch.randn(1, genome_dim, device=device))

    optimizer = optim.Adam(
        genome_attention.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.add_param_group(
        {
            "params": text_attention.parameters(),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    optimizer.add_param_group(
        {
            "params": regression_head.parameters(),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    optimizer.add_param_group(
        {
            "params": classification_head.parameters(),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    optimizer.add_param_group(
        {
            "params": [missing_genome_embedding],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    if molecule_encoder is not None:
        optimizer.add_param_group(
            {
                "params": molecule_encoder.parameters(),
                "lr": config.molecule_encoder.optimizer_learning_rate,
                "weight_decay": (
                    config.weight_decay
                    * config.molecule_encoder.optimizer_weight_decay_multiplier
                ),
            }
        )
    scheduler = build_legacy_cosine_scheduler(
        optimizer,
        num_epochs=config.epochs,
        min_lr=config.scheduler_eta_min,
    )
    return ModelBundle(
        molecule_encoder=molecule_encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        regression_head=regression_head,
        classification_head=classification_head,
        missing_genome_embedding=missing_genome_embedding,
        regression_criterion=nn.MSELoss(),
        classification_criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.cuda.amp.GradScaler(),
    )


def _partition_metrics(
    accumulator: HierarchicalMicPredictionAccumulator,
) -> dict[str, Any]:
    overall = summarize_predictions(accumulator.labels, accumulator.predictions)
    return {
        "overall": overall,
        "genome_text": summarize_partition_or_sentinel(
            accumulator.genome_text_labels, accumulator.genome_text_predictions
        ),
        "text_only": summarize_partition_or_sentinel(
            accumulator.text_only_labels, accumulator.text_only_predictions
        ),
        "species": specieswise_metrics(
            accumulator.species_labels, accumulator.species_predictions
        ),
    }


def evaluate_holdout(
    loaders: HoldoutLoaders,
    model: ModelBundle,
    prepared: PreparedHierarchicalMicData,
    *,
    device: torch.device,
    baseline_means: tuple[float, float] | None = None,
    progress_desc: str,
) -> tuple[HierarchicalMicPredictionAccumulator, dict[str, Any]]:
    accumulator = HierarchicalMicPredictionAccumulator()
    batches, total = legacy_zip_longest_loaders(
        loaders.genome_text_test, loaders.text_only_test
    )
    with torch.no_grad():
        for genome_batch, text_batch in tqdm(
            batches, desc=progress_desc, leave=False, total=total
        ):
            if genome_batch is not None:
                result = hierarchical_mic_batch_forward(
                    genome_batch,
                    device=device,
                    molecule_encoder=model.molecule_encoder,
                    genome_attention=model.genome_attention,
                    text_attention=model.text_attention,
                    prediction_head=model.regression_head,
                    criterion=model.regression_criterion,
                    missing_genome_embedding=model.missing_genome_embedding,
                    has_genome=True,
                    reshape_outputs=True,
                    autocast_enabled=True,
                )
                accumulator.add_batch(
                    result,
                    has_genome=True,
                    atcc_id_to_species=prepared.atcc_id_to_species,
                    original_strain_to_species=prepared.original_strain_to_species,
                    baseline_mean=None if baseline_means is None else baseline_means[0],
                )
            if text_batch is not None:
                result = hierarchical_mic_batch_forward(
                    text_batch,
                    device=device,
                    molecule_encoder=model.molecule_encoder,
                    genome_attention=model.genome_attention,
                    text_attention=model.text_attention,
                    prediction_head=model.regression_head,
                    criterion=model.regression_criterion,
                    missing_genome_embedding=model.missing_genome_embedding,
                    has_genome=False,
                    reshape_outputs=True,
                    autocast_enabled=True,
                )
                accumulator.add_batch(
                    result,
                    has_genome=False,
                    atcc_id_to_species=prepared.atcc_id_to_species,
                    original_strain_to_species=prepared.original_strain_to_species,
                    baseline_mean=None if baseline_means is None else baseline_means[1],
                )
    return accumulator, _partition_metrics(accumulator)


def _train_regression_batch(
    batch: dict,
    accumulator: HierarchicalMicPredictionAccumulator,
    model: ModelBundle,
    prepared: PreparedHierarchicalMicData,
    *,
    device: torch.device,
    has_genome: bool,
    epoch: int,
    freeze_epochs: int,
) -> None:
    result = hierarchical_mic_optimizer_step(
        batch,
        device=device,
        molecule_encoder=model.molecule_encoder,
        genome_attention=model.genome_attention,
        text_attention=model.text_attention,
        prediction_head=model.regression_head,
        legacy_regression_head_for_clipping=model.regression_head,
        criterion=model.regression_criterion,
        missing_genome_embedding=model.missing_genome_embedding,
        optimizer=model.optimizer,
        scaler=model.scaler,
        has_genome=has_genome,
        reshape_outputs=True,
        autocast_enabled=True,
        epoch=epoch,
        freeze_epochs=freeze_epochs,
    )
    accumulator.add_batch(
        result,
        has_genome=has_genome,
        atcc_id_to_species=prepared.atcc_id_to_species,
        original_strain_to_species=prepared.original_strain_to_species,
    )


def _train_classification_batch(
    batch: dict,
    model: ModelBundle,
    *,
    device: torch.device,
    has_genome: bool,
    epoch: int,
    freeze_epochs: int,
) -> float:
    result = hierarchical_mic_optimizer_step(
        batch,
        device=device,
        molecule_encoder=model.molecule_encoder,
        genome_attention=model.genome_attention,
        text_attention=model.text_attention,
        prediction_head=model.classification_head,
        legacy_regression_head_for_clipping=model.regression_head,
        criterion=model.classification_criterion,
        missing_genome_embedding=model.missing_genome_embedding,
        optimizer=model.optimizer,
        scaler=model.scaler,
        has_genome=has_genome,
        reshape_outputs=False,
        autocast_enabled=True,
        epoch=epoch,
        freeze_epochs=freeze_epochs,
    )
    return result.loss.item()


def train_epoch(
    loaders: HoldoutLoaders,
    model: ModelBundle,
    prepared: PreparedHierarchicalMicData,
    *,
    device: torch.device,
    epoch: int,
    freeze_epochs: int,
    progress_desc: str,
) -> tuple[HierarchicalMicPredictionAccumulator, dict[str, list[float]]]:
    regression = HierarchicalMicPredictionAccumulator()
    classification_losses = {"all": [], "genome_text": [], "text_only": []}
    batches, total = legacy_zip_longest_loaders(
        loaders.genome_text_train,
        loaders.text_only_train,
        loaders.small_molecule_genome_text_train,
        loaders.small_molecule_text_only_train,
    )
    for genome_batch, text_batch, genome_cls_batch, text_cls_batch in tqdm(
        batches, desc=progress_desc, leave=False, total=total
    ):
        if genome_batch is not None:
            _train_regression_batch(
                genome_batch,
                regression,
                model,
                prepared,
                device=device,
                has_genome=True,
                epoch=epoch,
                freeze_epochs=freeze_epochs,
            )
        if text_batch is not None:
            _train_regression_batch(
                text_batch,
                regression,
                model,
                prepared,
                device=device,
                has_genome=False,
                epoch=epoch,
                freeze_epochs=freeze_epochs,
            )
        if genome_cls_batch is not None:
            loss = _train_classification_batch(
                genome_cls_batch,
                model,
                device=device,
                has_genome=True,
                epoch=epoch,
                freeze_epochs=freeze_epochs,
            )
            classification_losses["all"].append(loss)
            classification_losses["genome_text"].append(loss)
        if text_cls_batch is not None:
            loss = _train_classification_batch(
                text_cls_batch,
                model,
                device=device,
                has_genome=False,
                epoch=epoch,
                freeze_epochs=freeze_epochs,
            )
            classification_losses["all"].append(loss)
            classification_losses["text_only"].append(loss)
    model.scheduler.step()
    return regression, classification_losses


def _mean(values: Sequence[float]) -> float:
    return float(np.array(values).mean())


def _log_species(metrics: Mapping[str, Sequence[float]], prefix: str) -> None:
    LOGGER.info("\n %s species wise R2, MSE, Spearman, Pearson:", prefix)
    for species_name, values in metrics.items():
        formatted = ", ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in values
        )
        LOGGER.info("    %s:  %s", species_name, formatted)


def _log_initial_evaluation(
    accumulator: HierarchicalMicPredictionAccumulator,
    metrics: Mapping[str, Any],
    *,
    ensemble: int,
    num_ensembles: int,
    epoch: int,
    num_epochs: int,
) -> None:
    _log_species(metrics["species"], "Test")
    baseline_r2 = calculate_r2(accumulator.labels, accumulator.baseline_predictions)
    LOGGER.info(
        "\n Ensemble %d/%d Epoch %d/%d\n"
        "    Test Loss: %.6f, genome text Test Loss: %.6f, text only Test Loss: %.6f\n"
        "    Test R2: %.6f, genome text Test R2: %.6f, text only Test R2: %.6f, "
        "Test train mean MIC R2: %.6f",
        ensemble + 1,
        num_ensembles,
        epoch,
        num_epochs,
        _mean(accumulator.losses),
        _mean(accumulator.genome_text_losses),
        _mean(accumulator.text_only_losses),
        metrics["overall"]["r2"],
        metrics["genome_text"]["r2"],
        metrics["text_only"]["r2"],
        baseline_r2,
    )


def _log_epoch(
    train: HierarchicalMicPredictionAccumulator,
    test: HierarchicalMicPredictionAccumulator,
    train_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    classification_losses: Mapping[str, Sequence[float]],
    tracker: LegacyBestMetricTracker,
    *,
    ensemble: int,
    num_ensembles: int,
    epoch: int,
    num_epochs: int,
) -> None:
    _log_species(train_metrics["species"], "Train")
    _log_species(test_metrics["species"], "Test")
    LOGGER.info(
        "\n Ensemble %d/%d Epoch %d/%d\n"
        "    Regression Training Loss: %.4f, Test Loss: %.4f\n"
        "      Genome text Training Loss: %.4f, Genome Text Test Loss: %.4f\n"
        "      Text only Training Loss: %.4f, Text only Test Loss: %.4f\n"
        "    Classification Training Loss: %.4f\n"
        "      Genome text Training Loss: %.4f\n"
        "      Text only Training Loss: %.4f\n"
        "    Train R2: %.4f, Test R2: %.4f, Best Test R2: %.4f\n"
        "      Genome Text Train R2: %.4f, Genome Text Test R2: %.4f\n"
        "      Text only Train R2: %.4f, Text only Test R2: %.4f\n"
        "    Train spearman:%.4f, Test spearman:%.4f, Best Test spearman:%.4f\n"
        "      Genome Text Train spearman:%.4f, Genome Text Test spearman:%.4f\n"
        "      Text only Train spearman:%.4f, Text only Test spearman:%.4f\n"
        "    Train pearson:%.4f, Test pearson:%.4f, Best Test pearson:%.4f\n"
        "      Genome Text Train pearson:%.4f, Genome Text Test pearson:%.4f\n"
        "      Text onlyTrain pearson:%.4f, Text only Test pearson:%.4f",
        ensemble + 1,
        num_ensembles,
        epoch + 1,
        num_epochs,
        _mean(train.losses),
        _mean(test.losses),
        _mean(train.genome_text_losses),
        _mean(test.genome_text_losses),
        _mean(train.text_only_losses),
        _mean(test.text_only_losses),
        _mean(classification_losses["all"]),
        _mean(classification_losses["genome_text"]),
        _mean(classification_losses["text_only"]),
        train_metrics["overall"]["r2"],
        test_metrics["overall"]["r2"],
        tracker.best_r2,
        train_metrics["genome_text"]["r2"],
        test_metrics["genome_text"]["r2"],
        train_metrics["text_only"]["r2"],
        test_metrics["text_only"]["r2"],
        train_metrics["overall"]["spearman"],
        test_metrics["overall"]["spearman"],
        tracker.best_spearman,
        train_metrics["genome_text"]["spearman"],
        test_metrics["genome_text"]["spearman"],
        train_metrics["text_only"]["spearman"],
        test_metrics["text_only"]["spearman"],
        train_metrics["overall"]["pearson"],
        test_metrics["overall"]["pearson"],
        tracker.best_pearson,
        train_metrics["genome_text"]["pearson"],
        test_metrics["genome_text"]["pearson"],
        train_metrics["text_only"]["pearson"],
        test_metrics["text_only"]["pearson"],
    )


def run_holdout(
    config: HierarchicalMicConfig,
    prepared: PreparedHierarchicalMicData,
    split: HoldoutSplit,
    features: RuntimeFeatures,
    *,
    group: int,
    device: torch.device,
) -> dict[str, float]:
    frames = prepare_holdout_frames(prepared, split, group)
    loaders, genome_dim, text_dim = build_holdout_loaders(
        frames, features, batch_size=config.batch_size
    )
    LOGGER.info(
        "\n training data 1 data type: %s\n",
        next(iter(features.genome_embeddings.values())).dtype,
    )

    ensemble_predictions_by_member = []
    final_labels: list[float] | None = None
    for ensemble in tqdm(range(config.ensembles_per_group), desc=" Doing ensembles "):
        torch.manual_seed(config.ensemble_seeds[ensemble])
        torch.cuda.manual_seed(config.ensemble_seeds[ensemble])
        LOGGER.info(" num of frozen epochs: %d\n", config.freeze_epochs)
        model = build_model_bundle(
            config, genome_dim=genome_dim, text_dim=text_dim, device=device
        )
        # Modules deliberately remain in their default train mode for all evaluation.
        if not all(
            module.training
            for module in (
                model.genome_attention,
                model.text_attention,
                model.regression_head,
                model.classification_head,
            )
        ):
            raise AssertionError("Legacy evaluation requires train-mode modules")
        if model.molecule_encoder is not None:
            expected_training = config.molecule_encoder.initial_mode == "train"
            if model.molecule_encoder.training != expected_training:
                raise AssertionError(
                    "Online molecule encoder mode differs from its legacy profile"
                )

        tracker = LegacyBestMetricTracker()
        for epoch in tqdm(
            range(config.epochs),
            desc=f" Training ensemble {ensemble + 1}/{config.ensembles_per_group} ",
            leave=False,
        ):
            if epoch == config.freeze_epochs and model.molecule_encoder is not None:
                unfreeze_legacy_molecule_encoder(model.molecule_encoder)
            if epoch == 0:
                initial, initial_metrics = evaluate_holdout(
                    loaders,
                    model,
                    prepared,
                    device=device,
                    baseline_means=(
                        frames.genome_text_train_mean_mic,
                        frames.text_only_train_mean_mic,
                    ),
                    progress_desc=f" Epoch {epoch}/{config.epochs} | evaluating",
                )
                _log_initial_evaluation(
                    initial,
                    initial_metrics,
                    ensemble=ensemble,
                    num_ensembles=config.ensembles_per_group,
                    epoch=epoch,
                    num_epochs=config.epochs,
                )

            train, classification_losses = train_epoch(
                loaders,
                model,
                prepared,
                device=device,
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
                progress_desc=(
                    f" Ensemble {ensemble + 1}/{config.ensembles_per_group} "
                    f"Epoch {epoch + 1}/{config.epochs} | training"
                ),
            )
            train_metrics = _partition_metrics(train)
            test, test_metrics = evaluate_holdout(
                loaders,
                model,
                prepared,
                device=device,
                progress_desc=(
                    f" Ensemble {ensemble + 1}/{config.ensembles_per_group} "
                    f"Epoch {epoch + 1}/{config.epochs} | evaluating"
                ),
            )
            final_labels = test.labels
            overall = test_metrics["overall"]
            improved = tracker.update(
                r2=overall["r2"],
                spearman=overall["spearman"],
                pearson=overall["pearson"],
                predictions=test.predictions,
            )
            if improved:
                torch.save(
                    legacy_hierarchical_checkpoint_payload(
                        r2=tracker.best_r2,
                        optimizer=model.optimizer,
                        regression_head=model.regression_head,
                        classification_head=model.classification_head,
                        genome_attention=model.genome_attention,
                        text_attention=model.text_attention,
                        missing_genome_embedding=model.missing_genome_embedding,
                        molecule_encoder=model.molecule_encoder,
                        molecule_encoder_state_key=(
                            None
                            if config.molecule_encoder is None
                            else config.molecule_encoder.checkpoint_state_key
                        ),
                        genome_embedding_adapter=features.genome_embedding_adapter,
                    ),
                    config.paths.output_dir
                    / checkpoint_filename(config, split, group, ensemble),
                )
            _log_epoch(
                train,
                test,
                train_metrics,
                test_metrics,
                classification_losses,
                tracker,
                ensemble=ensemble,
                num_ensembles=config.ensembles_per_group,
                epoch=epoch,
                num_epochs=config.epochs,
            )
        if tracker.best_predictions is not None and tracker.best_r2 > -10:
            ensemble_predictions_by_member.append(tracker.best_predictions)

    if final_labels is None:
        raise RuntimeError("No held-out evaluation labels were produced")
    LOGGER.info(
        "\n len of ensembled test predictions: %d", len(ensemble_predictions_by_member)
    )
    predictions = ensemble_predictions(ensemble_predictions_by_member)
    metrics = summarize_predictions(final_labels, predictions)
    LOGGER.info("\n Ensemble R2 of %s: %.4f", split.group_names[group], metrics["r2"])
    LOGGER.info(
        " Ensemble spearman of %s: %.4f", split.group_names[group], metrics["spearman"]
    )
    LOGGER.info(
        " Ensemble pearson of %s: %.4f", split.group_names[group], metrics["pearson"]
    )
    return metrics


def legacy_group_token(
    config: HierarchicalMicConfig, split: HoldoutSplit, group: int
) -> str:
    return (
        split.group_names[group] if config.holdout_protocol == "phylum" else str(group)
    )


def checkpoint_filename(
    config: HierarchicalMicConfig,
    split: HoldoutSplit,
    group: int,
    ensemble: int,
) -> str:
    scope = "Strain_wise" if config.holdout_protocol == "strain" else "Species_wise"
    return (
        f"genome_text_learnable_emb_{scope}_best_R2_"
        f"group_{legacy_group_token(config, split, group)}_ensemble_{ensemble}.pth"
    )


def configure_logging(output_dir: Path, group_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        output_dir / f"log_group_{group_label}.log", mode="w"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def validate_paths(config: HierarchicalMicConfig, repo_root: Path) -> None:
    missing = [
        path
        for path in config.paths.required_inputs(
            include_precomputed_molecule_embeddings=(
                config.molecule_encoder is None
            )
        )
        if not path.exists()
    ]
    if config.holdout_tree is not None and not config.holdout_tree.exists():
        missing.append(config.holdout_tree)
    preparation_inputs = (
        repo_root
        / "DataPrepare/Data/Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        repo_root / "DataPrepare/Data/Genome/old_to_new_NCBI_taxonomy.json",
    )
    missing.extend(path for path in preparation_inputs if not path.exists())
    if (
        config.molecule_encoder is not None
        and config.molecule_encoder.tokenizer_kind == "vendored_peptideclm"
    ):
        from apexoracle.vendor.peptideclm_tokenizer import RESOURCE_ROOT

        missing.extend(
            path
            for path in (
                RESOURCE_ROOT / "new_vocab.txt",
                RESOURCE_ROOT / "new_splits.txt",
            )
            if not path.exists()
        )
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Required hierarchical MIC resources are missing:\n{formatted}"
        )


def dry_run_report(
    config: HierarchicalMicConfig,
    prepared: PreparedHierarchicalMicData,
    split: HoldoutSplit,
    *,
    group: int,
    repo_root: Path,
) -> dict[str, Any]:
    frames = prepare_holdout_frames(prepared, split, group)
    if config.molecule_encoder is None:
        counts = holdout_record_counts(prepared, split, group)
    else:
        tokenizer = load_legacy_tokenizer(config.molecule_encoder, repo_root)

        def online_count(frame: pd.DataFrame) -> dict[str, int]:
            retained = sum(
                len(
                    tokenizer(
                        value,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                    )["input_ids"].squeeze(0)
                )
                <= config.molecule_encoder.max_length
                for value in frame["SMILES"]
            )
            return {
                "before_length_filter": len(frame),
                "after_length_filter": retained,
            }

        counts = {
            "genome_text_train": online_count(frames.genome_text_train),
            "genome_text_test": online_count(frames.genome_text_test),
            "text_only_train": online_count(frames.text_only_train),
            "text_only_test": online_count(frames.text_only_test),
        }
    return {
        "status": "dry_run_ok",
        "runner": "apexoracle.training.hierarchical_mic_runner",
        "legacy_driver_imported": False,
        "protocol_family": config.protocol_family,
        "holdout_protocol": config.holdout_protocol,
        "repo_root": str(repo_root),
        "group": group,
        "group_name": split.group_names[group],
        "available_groups": list(split.group_names),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "historical_membership_claim": "not_claimed",
        "ensembles": config.ensembles_per_group,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "molecule_encoder": (
            "precomputed_mdlm"
            if config.molecule_encoder is None
            else config.molecule_encoder.name
        ),
        "genome_embedding_adapter": (
            None
            if config.genome_embedding_adapter is None
            else {
                "kind": config.genome_embedding_adapter.kind,
                "input_dim": config.genome_embedding_adapter.input_dim,
                "output_dim": config.genome_embedding_adapter.output_dim,
                "trainable": config.genome_embedding_adapter.trainable,
                "initialization": config.genome_embedding_adapter.initialization,
            }
        ),
        "record_counts": counts,
        "auxiliary_before_length_filter": {
            "genome_text": (
                0
                if frames.small_molecule_genome_text_train is None
                else len(frames.small_molecule_genome_text_train)
            ),
            "text_only": (
                0
                if frames.small_molecule_text_only_train is None
                else len(frames.small_molecule_text_only_train)
            ),
        },
        "paths": {
            "genome_embeddings": str(config.paths.genome_embeddings),
            "atcc_text_embeddings": str(config.paths.atcc_text_embeddings),
            "text_only_embeddings": str(config.paths.text_only_embeddings),
            "peptide_embeddings": str(config.paths.peptide_embeddings),
            "small_molecule_embeddings": str(config.paths.small_molecule_embeddings),
            "mic_records": str(config.paths.mic_records),
            "small_molecule_records": str(config.paths.small_molecule_records),
            "output_dir": str(config.paths.output_dir),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one shared MDLM hierarchical MIC holdout benchmark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--protocol", choices=["strain", "species", "phylum"], required=True
    )
    parser.add_argument("--test-group", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--molecule-encoder",
        choices=["chemberta_mtr", "chemberta_mlm", "molformer", "peptideclm"],
        help="Select a Fig. 2c online encoder profile when using the comparator config.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-dynamic-legacy-split",
        action="store_true",
        help=(
            "Required because archived process hash seeds were not recorded; a fresh "
            "run follows the selected legacy split code but does not overclaim missing historical provenance."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if not args.acknowledge_dynamic_legacy_split:
        raise SystemExit(
            "Refusing an ambiguous rerun. Pass --acknowledge-dynamic-legacy-split "
            "after reading experiments/hierarchical_mic/README.md."
        )
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    config = HierarchicalMicConfig.load(
        config_path,
        repo_root,
        holdout_protocol=args.protocol,
        molecule_encoder_name=args.molecule_encoder,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
    )
    validate_paths(config, repo_root)
    prepared = prepare_hierarchical_mic_data(
        repo_root,
        mic_data_path=config.paths.mic_records,
        small_molecule_data_path=config.paths.small_molecule_records,
    )
    split = build_holdout_split(
        prepared,
        repo_root,
        args.protocol,
        adapter=config.holdout_adapter,
        group_names=config.holdout_group_names,
        tree_path=config.holdout_tree,
        num_clusters=config.holdout_clusters,
    )
    if split.group_names != config.holdout_group_names:
        raise ValueError(
            "Config/code holdout group names differ: "
            f"config={config.holdout_group_names}, code={split.group_names}"
        )
    if args.test_group < 0 or args.test_group >= len(split.group_names):
        raise SystemExit(
            f"--test-group must be between 0 and {len(split.group_names) - 1} "
            f"for {args.protocol}"
        )
    if args.dry_run:
        report = dry_run_report(
            config,
            prepared,
            split,
            group=args.test_group,
            repo_root=repo_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    configure_logging(
        config.paths.output_dir,
        legacy_group_token(config, split, args.test_group),
    )
    LOGGER.info("Start")
    LOGGER.info("Current test group: %s", split.group_names[args.test_group])
    features = load_runtime_features(config, device, repo_root)
    metrics = run_holdout(
        config,
        prepared,
        split,
        features,
        group=args.test_group,
        device=device,
    )
    return metrics


if __name__ == "__main__":
    main()
