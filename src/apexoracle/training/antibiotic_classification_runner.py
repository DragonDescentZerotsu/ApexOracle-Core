"""Unified runner for the paper-era three-strain classification family.

The three modes share target preparation, folds, feature loading, metrics,
checkpoint naming, and ensemble orchestration. Mode-specific behavior remains
explicit where the old scripts genuinely differ:

* ``strict-zero-shot`` never trains on the held target;
* ``fine-tune`` adds the target fold's training subset to the full fusion loop;
* ``molecule-only`` trains only a 768-dimensional classification head.

Several scientifically awkward historical choices are intentionally retained:
the held target selects checkpoints every epoch, the full-fusion classification
head stays in train mode during that selection, and checkpoint AUPRC can lag the
epoch that improved AUROC.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from apexoracle.data.antibiotic_classification import (
    AntibioticClassificationFrames,
    AntibioticGenomeTextDataset,
    AntibioticTextOnlyDataset,
    TARGET_STRAINS,
    collate_antibiotic_genome_text_classification,
    collate_antibiotic_genome_text_regression,
    collate_antibiotic_text_classification,
    collate_antibiotic_text_regression,
    legacy_target_folds,
    prepare_antibiotic_classification_frames,
)
from apexoracle.data.hierarchical_mic_preparation import (
    prepare_hierarchical_mic_data,
)
from apexoracle.evaluation.antibiotic_classification import (
    LegacyClassificationBestTracker,
    classification_metrics,
    ensemble_classification_predictions,
)
from apexoracle.features.precomputed import (
    load_all_embeddings,
    load_text_only_embeddings,
)
from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead
from apexoracle.training.antibiotic_classification import (
    ClassificationEvaluation,
    full_fusion_classification_forward,
    full_fusion_classification_step,
    legacy_full_fusion_checkpoint_payload,
    legacy_molecule_only_checkpoint_payload,
    molecule_only_forward,
    molecule_only_step,
    set_legacy_full_fusion_selection_modes,
    set_legacy_full_fusion_training_modes,
)
from apexoracle.training.hierarchical_mic import (
    build_legacy_cosine_scheduler,
    hierarchical_mic_optimizer_step,
    legacy_zip_longest_loaders,
)


LOGGER = logging.getLogger("apexoracle.antibiotic_classification_runner")
DEFAULT_CONFIG = Path("configs/antibiotic_classification/legacy_three_strain.yaml")
MODES = ("strict-zero-shot", "fine-tune", "molecule-only")


@dataclass(frozen=True)
class AntibioticClassificationPaths:
    genome_embeddings: Path
    atcc_text_embeddings: Path
    text_only_embeddings: Path
    small_molecule_records: Path
    peptide_embeddings: Path
    small_molecule_embeddings: Path
    output_dir: Path

    def required_inputs(self) -> tuple[Path, ...]:
        return (
            self.genome_embeddings,
            self.atcc_text_embeddings,
            self.text_only_embeddings,
            self.small_molecule_records,
            self.peptide_embeddings,
            self.small_molecule_embeddings,
        )


@dataclass(frozen=True)
class AntibioticClassificationConfig:
    protocol_family: str
    mode: str
    target_names: tuple[str, ...]
    target_modalities: tuple[str, ...]
    molecule_embedding_dim: int
    genome_embedding_dim: int
    text_embedding_dim: int
    attention_heads: int
    attention_dropout: float
    fusion_head_hidden_dims: tuple[int, int]
    molecule_only_head_hidden_dims: tuple[int, int]
    head_dropout: float
    ensembles: int
    ensemble_seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler_eta_min: float
    freeze_epochs: int
    genome_embedding_scale: float
    text_embedding_scale: float
    target_training: bool
    num_folds: int | None
    full_fusion: bool
    evidence: str
    paths: AntibioticClassificationPaths

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        mode: str,
        epochs: int | None = None,
        ensembles: int | None = None,
        weight_decay: float | None = None,
        output_dir: Path | None = None,
    ) -> "AntibioticClassificationConfig":
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if mode not in MODES:
            raise ValueError(f"Unsupported classification mode: {mode}")
        model = raw["model"]
        training = raw["training"]
        mode_values = raw["modes"][mode]

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else repo_root / path

        selected_output = output_dir or resolve(raw["outputs"][mode])
        if not selected_output.is_absolute():
            selected_output = repo_root / selected_output
        targets = raw["targets"]
        config = cls(
            protocol_family=str(raw["protocol_family"]),
            mode=mode,
            target_names=tuple(str(item["strain"]) for item in targets),
            target_modalities=tuple(str(item["strain_modality"]) for item in targets),
            molecule_embedding_dim=int(model["molecule_embedding_dim"]),
            genome_embedding_dim=int(model["genome_embedding_dim"]),
            text_embedding_dim=int(model["text_embedding_dim"]),
            attention_heads=int(model["attention_heads"]),
            attention_dropout=float(model["attention_dropout"]),
            fusion_head_hidden_dims=tuple(
                int(value) for value in model["fusion_head_hidden_dims"]
            ),
            molecule_only_head_hidden_dims=tuple(
                int(value) for value in model["molecule_only_head_hidden_dims"]
            ),
            head_dropout=float(model["head_dropout"]),
            ensembles=int(training["ensembles"] if ensembles is None else ensembles),
            ensemble_seeds=tuple(int(seed) for seed in training["ensemble_seeds"]),
            epochs=int(training["epochs"] if epochs is None else epochs),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(
                training["weight_decay"] if weight_decay is None else weight_decay
            ),
            scheduler_eta_min=float(training["scheduler_eta_min"]),
            freeze_epochs=int(training["freeze_epochs"]),
            genome_embedding_scale=float(training["genome_embedding_scale"]),
            text_embedding_scale=float(training["text_embedding_scale"]),
            target_training=bool(mode_values["target_training"]),
            num_folds=(
                None if mode_values["folds"] is None else int(mode_values["folds"])
            ),
            full_fusion=str(mode_values["model"]) == "strain_fusion",
            evidence=str(mode_values["evidence"]),
            paths=AntibioticClassificationPaths(
                genome_embeddings=resolve(raw["paths"]["genome_embeddings"]),
                atcc_text_embeddings=resolve(raw["paths"]["atcc_text_embeddings"]),
                text_only_embeddings=resolve(raw["paths"]["text_only_embeddings"]),
                small_molecule_records=resolve(
                    raw["paths"]["small_molecule_records"]
                ),
                peptide_embeddings=resolve(raw["paths"]["peptide_embeddings"]),
                small_molecule_embeddings=resolve(
                    raw["paths"]["small_molecule_embeddings"]
                ),
                output_dir=selected_output,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.protocol_family
            != "paper_legacy_three_strain_antibiotic_classification"
        ):
            raise ValueError(f"Unsupported protocol family: {self.protocol_family}")
        if self.target_names != TARGET_STRAINS:
            raise ValueError(
                f"Config target order differs from legacy order: {self.target_names}"
            )
        if self.target_modalities != ("genome_text", "genome_text", "text_only"):
            raise ValueError(f"Unexpected target modalities: {self.target_modalities}")
        if self.mode == "strict-zero-shot":
            if (
                self.target_training
                or self.num_folds is not None
                or not self.full_fusion
            ):
                raise ValueError("Invalid strict-zero-shot mode contract")
        elif self.mode == "fine-tune":
            if not self.target_training or self.num_folds != 5 or not self.full_fusion:
                raise ValueError("Invalid fine-tune mode contract")
        elif self.mode == "molecule-only":
            if not self.target_training or self.num_folds != 5 or self.full_fusion:
                raise ValueError("Invalid molecule-only mode contract")
        if self.ensembles < 1 or len(self.ensemble_seeds) < self.ensembles:
            raise ValueError("Invalid ensemble count or insufficient seeds")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.genome_embedding_dim % self.attention_heads:
            raise ValueError("genome dimension must be divisible by attention heads")
        if self.text_embedding_dim % self.attention_heads:
            raise ValueError("text dimension must be divisible by attention heads")
        if self.fusion_head_hidden_dims != (
            (self.genome_embedding_dim + self.text_embedding_dim) // 4,
            128,
        ):
            raise ValueError("Unexpected full-fusion head dimensions")
        if self.molecule_only_head_hidden_dims != (
            self.molecule_embedding_dim // 2,
            128,
        ):
            raise ValueError("Unexpected molecule-only head dimensions")


@dataclass
class RuntimeFeatures:
    genome_embeddings: Mapping[str, torch.Tensor]
    atcc_text_embeddings: Mapping[str, torch.Tensor]
    text_only_embeddings: Mapping[str, torch.Tensor]
    peptide_embeddings: Mapping[Any, torch.Tensor]
    small_molecule_embeddings: Mapping[Any, torch.Tensor]

    @property
    def all_text_embeddings(self) -> dict[str, torch.Tensor]:
        return dict(self.atcc_text_embeddings) | dict(self.text_only_embeddings)


@dataclass
class ClassificationLoaders:
    mic_genome_text_train: DataLoader
    mic_text_route_train: DataLoader
    auxiliary_genome_text_train: DataLoader | list[None]
    auxiliary_text_only_train: DataLoader | list[None]
    target_train: DataLoader | None
    target_test: DataLoader
    target_has_genome: bool
    target_dataset: AntibioticGenomeTextDataset | AntibioticTextOnlyDataset


@dataclass
class FullFusionModel:
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


@dataclass
class MoleculeOnlyModel:
    classification_head: nn.Module
    classification_criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    scaler: Any


def load_runtime_features(
    config: AntibioticClassificationConfig, device: torch.device
) -> RuntimeFeatures:
    paths = config.paths
    return RuntimeFeatures(
        genome_embeddings=load_all_embeddings(
            paths.genome_embeddings,
            config.genome_embedding_scale,
            device,
            "genome",
        ),
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
        peptide_embeddings=torch.load(paths.peptide_embeddings),
        small_molecule_embeddings=torch.load(paths.small_molecule_embeddings),
    )


def _genome_dataset(
    frame: pd.DataFrame,
    features: RuntimeFeatures,
    description: str,
) -> AntibioticGenomeTextDataset:
    return AntibioticGenomeTextDataset(
        frame,
        None,
        features.genome_embeddings,
        features.atcc_text_embeddings,
        description,
        features.peptide_embeddings,
        features.small_molecule_embeddings,
    )


def _text_dataset(
    frame: pd.DataFrame,
    features: RuntimeFeatures,
    description: str,
) -> AntibioticTextOnlyDataset:
    return AntibioticTextOnlyDataset(
        frame,
        None,
        features.all_text_embeddings,
        description,
        features.peptide_embeddings,
        features.small_molecule_embeddings,
    )


def _optional_loader(dataset, *, batch_size: int, collate_fn):
    if dataset is None:
        return [None]
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


def build_classification_loaders(
    config: AntibioticClassificationConfig,
    frames: AntibioticClassificationFrames,
    features: RuntimeFeatures,
    *,
    fold: int | None,
) -> tuple[ClassificationLoaders, int, int]:
    mic_genome = _genome_dataset(
        frames.mic_genome_text_train, features, "peptide genome-text training set"
    )
    mic_text = _text_dataset(
        frames.mic_text_route_train, features, "peptide text-route training set"
    )
    auxiliary_genome = (
        None
        if frames.auxiliary_genome_text_train is None
        else _genome_dataset(
            frames.auxiliary_genome_text_train,
            features,
            "other-target genome-text classification set",
        )
    )
    auxiliary_text = (
        None
        if frames.auxiliary_text_only_train is None
        else _text_dataset(
            frames.auxiliary_text_only_train,
            features,
            "other-target text-only classification set",
        )
    )
    target_dataset = (
        _genome_dataset(frames.target, features, "held-target classification set")
        if frames.target_has_genome
        else _text_dataset(frames.target, features, "held-target classification set")
    )
    target_collator = (
        collate_antibiotic_genome_text_classification
        if frames.target_has_genome
        else collate_antibiotic_text_classification
    )
    target_train = None
    target_test_dataset = target_dataset
    if config.target_training:
        if fold is None or config.num_folds is None:
            raise ValueError(f"Mode {config.mode} requires one explicit fold")
        folds = legacy_target_folds(len(target_dataset), num_folds=config.num_folds)
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"fold must be in [0, {len(folds) - 1}]")
        train_indices, test_indices = folds[fold]
        target_train = DataLoader(
            Subset(target_dataset, train_indices),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=target_collator,
        )
        target_test_dataset = Subset(target_dataset, test_indices)
    elif fold is not None:
        raise ValueError("strict-zero-shot does not accept a fold")

    loaders = ClassificationLoaders(
        mic_genome_text_train=DataLoader(
            mic_genome,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_antibiotic_genome_text_regression,
        ),
        mic_text_route_train=DataLoader(
            mic_text,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_antibiotic_text_regression,
        ),
        auxiliary_genome_text_train=_optional_loader(
            auxiliary_genome,
            batch_size=config.batch_size,
            collate_fn=collate_antibiotic_genome_text_classification,
        ),
        auxiliary_text_only_train=_optional_loader(
            auxiliary_text,
            batch_size=config.batch_size,
            collate_fn=collate_antibiotic_text_classification,
        ),
        target_train=target_train,
        target_test=DataLoader(
            target_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=target_collator,
        ),
        target_has_genome=frames.target_has_genome,
        target_dataset=target_dataset,
    )
    first = mic_genome[0]
    return loaders, first["genome_embedding"].shape[1], first["text_embedding"].shape[1]


def build_full_fusion_model(
    config: AntibioticClassificationConfig,
    *,
    genome_dim: int,
    text_dim: int,
    device: torch.device,
) -> FullFusionModel:
    if (genome_dim, text_dim) != (
        config.genome_embedding_dim,
        config.text_embedding_dim,
    ):
        raise ValueError(
            f"Runtime dimensions {(genome_dim, text_dim)} do not match config"
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
        config.fusion_head_hidden_dims[0],
        config.fusion_head_hidden_dims[1],
        1,
        config.head_dropout,
    ).to(device)
    classification_head = RegressionHead(
        fused_dim,
        config.fusion_head_hidden_dims[0],
        config.fusion_head_hidden_dims[1],
        1,
        config.head_dropout,
    ).to(device)
    missing_genome_embedding = nn.Parameter(torch.randn(1, genome_dim, device=device))
    optimizer = optim.Adam(
        genome_attention.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for parameters in (
        text_attention.parameters(),
        regression_head.parameters(),
        classification_head.parameters(),
        [missing_genome_embedding],
    ):
        optimizer.add_param_group(
            {
                "params": parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    return FullFusionModel(
        genome_attention=genome_attention,
        text_attention=text_attention,
        regression_head=regression_head,
        classification_head=classification_head,
        missing_genome_embedding=missing_genome_embedding,
        regression_criterion=nn.MSELoss(),
        classification_criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=build_legacy_cosine_scheduler(
            optimizer, num_epochs=config.epochs, min_lr=config.scheduler_eta_min
        ),
        scaler=torch.cuda.amp.GradScaler(enabled=device.type == "cuda"),
    )


def build_molecule_only_model(
    config: AntibioticClassificationConfig, *, device: torch.device
) -> MoleculeOnlyModel:
    classification_head = RegressionHead(
        config.molecule_embedding_dim,
        config.molecule_only_head_hidden_dims[0],
        config.molecule_only_head_hidden_dims[1],
        1,
        config.head_dropout,
    ).to(device)
    optimizer = optim.Adam(
        classification_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return MoleculeOnlyModel(
        classification_head=classification_head,
        classification_criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=build_legacy_cosine_scheduler(
            optimizer, num_epochs=config.epochs, min_lr=config.scheduler_eta_min
        ),
        scaler=torch.cuda.amp.GradScaler(enabled=device.type == "cuda"),
    )


def _train_full_shared_epoch(
    loaders: ClassificationLoaders,
    model: FullFusionModel,
    config: AntibioticClassificationConfig,
    *,
    device: torch.device,
    epoch: int,
) -> dict[str, list[float]]:
    losses = {"regression": [], "auxiliary": [], "target": []}
    set_legacy_full_fusion_training_modes(
        model.genome_attention,
        model.text_attention,
        model.regression_head,
        model.classification_head,
    )
    batches, total = legacy_zip_longest_loaders(
        loaders.mic_genome_text_train,
        loaders.mic_text_route_train,
        loaders.auxiliary_genome_text_train,
        loaders.auxiliary_text_only_train,
    )
    for genome_batch, text_batch, genome_cls_batch, text_cls_batch in tqdm(
        batches, total=total, leave=False, desc="shared MIC/auxiliary training"
    ):
        if genome_batch is not None:
            result = hierarchical_mic_optimizer_step(
                genome_batch,
                device=device,
                genome_attention=model.genome_attention,
                text_attention=model.text_attention,
                prediction_head=model.regression_head,
                legacy_regression_head_for_clipping=model.regression_head,
                criterion=model.regression_criterion,
                missing_genome_embedding=model.missing_genome_embedding,
                optimizer=model.optimizer,
                scaler=model.scaler,
                has_genome=True,
                reshape_outputs=True,
                autocast_enabled=device.type == "cuda",
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
            )
            losses["regression"].append(result.loss.item())
        if text_batch is not None:
            result = hierarchical_mic_optimizer_step(
                text_batch,
                device=device,
                genome_attention=model.genome_attention,
                text_attention=model.text_attention,
                prediction_head=model.regression_head,
                legacy_regression_head_for_clipping=model.regression_head,
                criterion=model.regression_criterion,
                missing_genome_embedding=model.missing_genome_embedding,
                optimizer=model.optimizer,
                scaler=model.scaler,
                has_genome=False,
                reshape_outputs=True,
                autocast_enabled=device.type == "cuda",
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
            )
            losses["regression"].append(result.loss.item())
        if genome_cls_batch is not None:
            result = full_fusion_classification_step(
                genome_cls_batch,
                device=device,
                genome_attention=model.genome_attention,
                text_attention=model.text_attention,
                classification_head=model.classification_head,
                regression_head=model.regression_head,
                criterion=model.classification_criterion,
                missing_genome_embedding=model.missing_genome_embedding,
                optimizer=model.optimizer,
                scaler=model.scaler,
                has_genome=True,
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
                autocast_enabled=device.type == "cuda",
                clip_text_attention=False,
                clip_missing_genome=False,
            )
            losses["auxiliary"].append(result.loss.item())
        if text_cls_batch is not None:
            result = full_fusion_classification_step(
                text_cls_batch,
                device=device,
                genome_attention=model.genome_attention,
                text_attention=model.text_attention,
                classification_head=model.classification_head,
                regression_head=model.regression_head,
                criterion=model.classification_criterion,
                missing_genome_embedding=model.missing_genome_embedding,
                optimizer=model.optimizer,
                scaler=model.scaler,
                has_genome=False,
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
                autocast_enabled=device.type == "cuda",
                clip_text_attention=True,
                clip_missing_genome=True,
            )
            losses["auxiliary"].append(result.loss.item())

    if loaders.target_train is not None:
        for batch in tqdm(
            loaders.target_train, leave=False, desc="target-fold training"
        ):
            result = full_fusion_classification_step(
                batch,
                device=device,
                genome_attention=model.genome_attention,
                text_attention=model.text_attention,
                classification_head=model.classification_head,
                regression_head=model.regression_head,
                criterion=model.classification_criterion,
                missing_genome_embedding=model.missing_genome_embedding,
                optimizer=model.optimizer,
                scaler=model.scaler,
                has_genome=loaders.target_has_genome,
                epoch=epoch,
                freeze_epochs=config.freeze_epochs,
                autocast_enabled=device.type == "cuda",
                clip_text_attention=True,
                clip_missing_genome=False,
            )
            losses["target"].append(result.loss.item())
    model.scheduler.step()
    return losses


def _train_molecule_only_epoch(
    loaders: ClassificationLoaders,
    model: MoleculeOnlyModel,
    *,
    device: torch.device,
) -> list[float]:
    if loaders.target_train is None:
        raise ValueError("molecule-only mode requires target-fold training data")
    model.classification_head.train()
    losses = []
    for batch in tqdm(loaders.target_train, leave=False, desc="molecule-only training"):
        result = molecule_only_step(
            batch,
            device=device,
            classification_head=model.classification_head,
            criterion=model.classification_criterion,
            optimizer=model.optimizer,
            scaler=model.scaler,
            autocast_enabled=device.type == "cuda",
        )
        losses.append(result.loss.item())
    model.scheduler.step()
    return losses


def evaluate_target(
    loaders: ClassificationLoaders,
    model: FullFusionModel | MoleculeOnlyModel,
    *,
    full_fusion: bool,
    device: torch.device,
    deterministic_checkpoint_inference: bool = False,
) -> ClassificationEvaluation:
    if full_fusion:
        assert isinstance(model, FullFusionModel)
        set_legacy_full_fusion_selection_modes(
            model.genome_attention,
            model.text_attention,
            model.regression_head,
            model.classification_head,
        )
        if deterministic_checkpoint_inference:
            model.classification_head.eval()
    else:
        assert isinstance(model, MoleculeOnlyModel)
        model.classification_head.eval()

    evaluation = ClassificationEvaluation([], [], [], [])
    with torch.no_grad():
        for batch in tqdm(
            loaders.target_test, leave=False, desc="held-target evaluation"
        ):
            if full_fusion:
                result = full_fusion_classification_forward(
                    batch,
                    device=device,
                    genome_attention=model.genome_attention,
                    text_attention=model.text_attention,
                    classification_head=model.classification_head,
                    criterion=model.classification_criterion,
                    missing_genome_embedding=model.missing_genome_embedding,
                    has_genome=loaders.target_has_genome,
                    autocast_enabled=device.type == "cuda",
                )
            else:
                result = molecule_only_forward(
                    batch,
                    device=device,
                    classification_head=model.classification_head,
                    criterion=model.classification_criterion,
                    autocast_enabled=device.type == "cuda",
                )
            evaluation.losses.append(result.loss.item())
            evaluation.labels.extend(result.labels.detach().cpu().flatten().tolist())
            evaluation.logits.extend(result.logits.detach().cpu().flatten().tolist())
            evaluation.molecule_ids.extend(batch["molecule_ids"])
    return evaluation


def checkpoint_filename(mode: str, group: int, ensemble: int, fold: int | None) -> str:
    prefix = "genome_text_learnable_emb_SM_outer_SM_best_AUROC"
    if mode == "strict-zero-shot":
        if fold is not None:
            raise ValueError("strict-zero-shot checkpoints do not have fold tokens")
        return f"{prefix}_group_{group}_ensemble_{ensemble}.pth"
    if fold is None:
        raise ValueError(f"{mode} checkpoints require a fold token")
    return f"{prefix}_group_{group}_ensemble_{ensemble}_fold_{fold}.pth"


def _save_checkpoint(
    config: AntibioticClassificationConfig,
    model: FullFusionModel | MoleculeOnlyModel,
    tracker: LegacyClassificationBestTracker,
    path: Path,
) -> None:
    if config.full_fusion:
        assert isinstance(model, FullFusionModel)
        payload = legacy_full_fusion_checkpoint_payload(
            auroc=tracker.best_auroc,
            checkpoint_auprc=tracker.checkpoint_auprc,
            optimizer=model.optimizer,
            regression_head=model.regression_head,
            classification_head=model.classification_head,
            genome_attention=model.genome_attention,
            text_attention=model.text_attention,
            missing_genome_embedding=model.missing_genome_embedding,
        )
    else:
        assert isinstance(model, MoleculeOnlyModel)
        payload = legacy_molecule_only_checkpoint_payload(
            auroc=tracker.best_auroc,
            checkpoint_auprc=tracker.checkpoint_auprc,
            optimizer=model.optimizer,
            classification_head=model.classification_head,
        )
    torch.save(payload, path)


def run_training(
    config: AntibioticClassificationConfig,
    loaders: ClassificationLoaders,
    *,
    group: int,
    fold: int | None,
    genome_dim: int,
    text_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    predictions_by_member = []
    final_evaluation: ClassificationEvaluation | None = None
    for ensemble in range(config.ensembles):
        seed = config.ensemble_seeds[ensemble]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        model: FullFusionModel | MoleculeOnlyModel
        if config.full_fusion:
            model = build_full_fusion_model(
                config, genome_dim=genome_dim, text_dim=text_dim, device=device
            )
        else:
            model = build_molecule_only_model(config, device=device)
        tracker = LegacyClassificationBestTracker()
        for epoch in range(config.epochs):
            if config.full_fusion:
                losses = _train_full_shared_epoch(
                    loaders, model, config, device=device, epoch=epoch
                )
            else:
                losses = {
                    "target": _train_molecule_only_epoch(loaders, model, device=device)
                }
            evaluation = evaluate_target(
                loaders,
                model,
                full_fusion=config.full_fusion,
                device=device,
            )
            final_evaluation = evaluation
            metrics = classification_metrics(evaluation.labels, evaluation.logits)
            improved = tracker.update_auroc(
                auroc=metrics["auroc"], predictions=evaluation.logits
            )
            if improved:
                _save_checkpoint(
                    config,
                    model,
                    tracker,
                    config.paths.output_dir
                    / checkpoint_filename(config.mode, group, ensemble, fold),
                )
            tracker.finish_epoch(auprc=metrics["auprc"])
            LOGGER.info(
                "ensemble=%d/%d epoch=%d/%d loss=%s AUROC=%.6f best=%.6f AUPRC=%.6f best=%.6f",
                ensemble + 1,
                config.ensembles,
                epoch + 1,
                config.epochs,
                {
                    key: float(np.mean(value)) if value else None
                    for key, value in losses.items()
                },
                metrics["auroc"],
                tracker.best_auroc,
                metrics["auprc"],
                tracker.best_auprc,
            )
        if tracker.best_predictions is not None:
            predictions_by_member.append(tracker.best_predictions)
    if final_evaluation is None or len(predictions_by_member) != config.ensembles:
        raise RuntimeError(
            "Training did not produce one selected prediction per ensemble"
        )
    predictions = ensemble_classification_predictions(predictions_by_member)
    metrics = classification_metrics(final_evaluation.labels, predictions)
    LOGGER.info(
        "ensemble target=%s AUROC=%.4f AUPRC=%.4f",
        TARGET_STRAINS[group],
        metrics["auroc"],
        metrics["auprc"],
    )
    return {
        "mode": config.mode,
        "target_group": group,
        "target_strain": TARGET_STRAINS[group],
        "fold": fold,
        "ensembles": config.ensembles,
        "metrics": metrics,
    }


def load_checkpoint_into_model(
    config: AntibioticClassificationConfig,
    model: FullFusionModel | MoleculeOnlyModel,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config.full_fusion:
        assert isinstance(model, FullFusionModel)
        inference_keys = {
            "auroc",
            "auprc",
            "re_head_state_dict",
            "cls_head_state_dict",
            "co_cross_attn_genome",
            "co_cross_attn_text",
            "learnable_embedding_weight",
        }
        accepted_schemas = (inference_keys, inference_keys | {"optimizer_state_dict"})
        if set(checkpoint) not in accepted_schemas:
            raise ValueError(
                "Unexpected full-fusion checkpoint keys; expected either the original "
                "training payload or the documented capsule payload without optimizer: "
                f"{sorted(checkpoint)}"
            )
        model.regression_head.load_state_dict(
            checkpoint["re_head_state_dict"], strict=True
        )
        model.classification_head.load_state_dict(
            checkpoint["cls_head_state_dict"], strict=True
        )
        model.genome_attention.load_state_dict(
            checkpoint["co_cross_attn_genome"], strict=True
        )
        model.text_attention.load_state_dict(
            checkpoint["co_cross_attn_text"], strict=True
        )
        missing = checkpoint["learnable_embedding_weight"]
        if isinstance(missing, nn.Parameter):
            missing = missing.detach()
        if tuple(missing.shape) != tuple(model.missing_genome_embedding.shape):
            raise ValueError("Unexpected missing-genome parameter shape")
        model.missing_genome_embedding.data.copy_(missing.to(device))
    else:
        assert isinstance(model, MoleculeOnlyModel)
        inference_keys = {"auroc", "auprc", "cls_head_state_dict"}
        accepted_schemas = (inference_keys, inference_keys | {"optimizer_state_dict"})
        if set(checkpoint) not in accepted_schemas:
            raise ValueError(
                f"Unexpected molecule-only checkpoint keys: {sorted(checkpoint)}"
            )
        model.classification_head.load_state_dict(
            checkpoint["cls_head_state_dict"], strict=True
        )
    return {"auroc": float(checkpoint["auroc"]), "auprc": float(checkpoint["auprc"])}


def evaluate_checkpoints(
    config: AntibioticClassificationConfig,
    loaders: ClassificationLoaders,
    *,
    group: int,
    fold: int | None,
    genome_dim: int,
    text_dim: int,
    device: torch.device,
    checkpoint_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    predictions_by_member = []
    reference: ClassificationEvaluation | None = None
    checkpoint_metrics = []
    for ensemble in range(config.ensembles):
        model: FullFusionModel | MoleculeOnlyModel
        if config.full_fusion:
            model = build_full_fusion_model(
                config, genome_dim=genome_dim, text_dim=text_dim, device=device
            )
        else:
            model = build_molecule_only_model(config, device=device)
        path = checkpoint_dir / checkpoint_filename(config.mode, group, ensemble, fold)
        checkpoint_metrics.append(
            load_checkpoint_into_model(config, model, path, device=device)
        )
        evaluation = evaluate_target(
            loaders,
            model,
            full_fusion=config.full_fusion,
            device=device,
            deterministic_checkpoint_inference=True,
        )
        if reference is None:
            reference = evaluation
        elif (
            reference.labels != evaluation.labels
            or reference.molecule_ids != evaluation.molecule_ids
        ):
            raise RuntimeError("Checkpoint evaluations produced different target order")
        predictions_by_member.append(evaluation.logits)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if reference is None:
        raise RuntimeError("No checkpoints were evaluated")
    predictions = ensemble_classification_predictions(predictions_by_member)
    metrics = classification_metrics(reference.labels, predictions)
    results_dir.mkdir(parents=True, exist_ok=True)
    token = f"group_{group}" if fold is None else f"group_{group}_fold_{fold}"
    prediction_path = (
        results_dir / f"antibiotic_classification_{config.mode}_{token}.csv"
    )
    pd.DataFrame(
        {
            "molecule_id": reference.molecule_ids,
            "target_strain": TARGET_STRAINS[group],
            "label": reference.labels,
            "prediction": predictions,
            "group": group,
            "fold": fold,
            "mode": config.mode,
        }
    ).to_csv(prediction_path, index=False)
    report = {
        "mode": config.mode,
        "target_group": group,
        "target_strain": TARGET_STRAINS[group],
        "fold": fold,
        "checkpoint_inference_module_mode": "eval",
        "training_time_selection_classification_head_mode": (
            "train" if config.full_fusion else "eval"
        ),
        "num_examples": len(reference.labels),
        "ensembles": config.ensembles,
        "metrics": metrics,
        "checkpoint_stored_metrics": checkpoint_metrics,
        "predictions": str(prediction_path),
    }
    metrics_path = results_dir / f"antibiotic_classification_{config.mode}_{token}.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _count_after_length_filter(frame: pd.DataFrame, max_length: int = 512) -> int:
    return int(
        frame["SMILES"]
        .apply(lambda value: len(ast.literal_eval(value)) <= max_length)
        .sum()
    )


def dry_run_report(
    config: AntibioticClassificationConfig,
    frames: AntibioticClassificationFrames,
    *,
    group: int,
    fold: int | None,
    repo_root: Path,
) -> dict[str, Any]:
    target_count = _count_after_length_filter(frames.target)
    fold_counts = None
    if config.target_training:
        if fold is None or config.num_folds is None:
            raise ValueError(f"Mode {config.mode} requires one explicit fold")
        folds = legacy_target_folds(target_count, num_folds=config.num_folds)
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"fold must be in [0, {len(folds) - 1}]")
        train_indices, test_indices = folds[fold]
        fold_counts = {"train": len(train_indices), "test": len(test_indices)}
    elif fold is not None:
        raise ValueError("strict-zero-shot does not accept a fold")

    def before_after(frame: pd.DataFrame | None) -> dict[str, int]:
        if frame is None:
            return {"before_512_filter": 0, "after_512_filter": 0}
        return {
            "before_512_filter": len(frame),
            "after_512_filter": _count_after_length_filter(frame),
        }

    return {
        "status": "dry_run_ok",
        "runner": "apexoracle.training.antibiotic_classification_runner",
        "legacy_driver_imported": False,
        "repo_root": str(repo_root),
        "mode": config.mode,
        "evidence": config.evidence,
        "target_group": group,
        "target_strain": TARGET_STRAINS[group],
        "target_has_genome": frames.target_has_genome,
        "target_training": config.target_training,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "fold": fold,
        "fold_counts": fold_counts,
        "ensembles": config.ensembles,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "record_counts": {
            "mic_genome_text_train": before_after(frames.mic_genome_text_train),
            "mic_text_route_train": before_after(frames.mic_text_route_train),
            "auxiliary_genome_text_train": before_after(
                frames.auxiliary_genome_text_train
            ),
            "auxiliary_text_only_train": before_after(frames.auxiliary_text_only_train),
            "target": before_after(frames.target),
        },
        "historical_contracts": {
            "mic_genome_rows_visited_again_via_text_route": True,
            "held_target_selects_checkpoint_each_epoch": True,
            "full_fusion_selection_cls_head_mode": "train",
            "checkpoint_auprc_can_lag_auroc_epoch": True,
            "two_genome_auxiliary_block_order_uses_legacy_set_iteration": True,
        },
        "output_dir": str(config.paths.output_dir),
    }


def validate_paths(config: AntibioticClassificationConfig, repo_root: Path) -> None:
    missing = [path for path in config.paths.required_inputs() if not path.exists()]
    preparation_inputs = (
        repo_root / "DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
        repo_root
        / "DataPrepare/Data/Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        repo_root / "DataPrepare/Data/Genome/old_to_new_NCBI_taxonomy.json",
    )
    missing.extend(path for path in preparation_inputs if not path.exists())
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Required classification resources are missing:\n{formatted}"
        )


def configure_logging(output_dir: Path, *, group: int, fold: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    token = f"log_group_{group}" if fold is None else f"log_group_{group}_fold_{fold}"
    file_handler = logging.FileHandler(output_dir / f"{token}.log", mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one behavior-frozen three-strain classification task."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--test-group", type=int, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ensembles", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/antibiotic_classification")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    config = AntibioticClassificationConfig.load(
        config_path,
        repo_root,
        mode=args.mode,
        epochs=args.epochs,
        ensembles=args.ensembles,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
    )
    if args.test_group < 0 or args.test_group >= len(TARGET_STRAINS):
        raise SystemExit(f"--test-group must be in [0, {len(TARGET_STRAINS) - 1}]")
    if config.target_training and args.fold is None:
        raise SystemExit(f"--fold is required for mode {config.mode}")
    if not config.target_training and args.fold is not None:
        raise SystemExit("--fold is not valid for strict-zero-shot")
    validate_paths(config, repo_root)
    prepared = prepare_hierarchical_mic_data(
        repo_root, small_molecule_data_path=config.paths.small_molecule_records
    )
    frames = prepare_antibiotic_classification_frames(prepared, args.test_group)
    if args.dry_run:
        report = dry_run_report(
            config,
            frames,
            group=args.test_group,
            fold=args.fold,
            repo_root=repo_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features = load_runtime_features(config, device)
    loaders, genome_dim, text_dim = build_classification_loaders(
        config, frames, features, fold=args.fold
    )
    if args.evaluate_checkpoints:
        checkpoint_dir = args.checkpoint_dir or config.paths.output_dir
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = repo_root / checkpoint_dir
        results_dir = args.results_dir
        if not results_dir.is_absolute():
            results_dir = repo_root / results_dir
        return evaluate_checkpoints(
            config,
            loaders,
            group=args.test_group,
            fold=args.fold,
            genome_dim=genome_dim,
            text_dim=text_dim,
            device=device,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
        )
    configure_logging(config.paths.output_dir, group=args.test_group, fold=args.fold)
    return run_training(
        config,
        loaders,
        group=args.test_group,
        fold=args.fold,
        genome_dim=genome_dim,
        text_dim=text_dim,
        device=device,
    )


if __name__ == "__main__":
    main()
