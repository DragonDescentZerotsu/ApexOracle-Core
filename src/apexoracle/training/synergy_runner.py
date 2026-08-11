"""Unified experimental runner for the paper-era synergy three-fold CV.

The code deliberately retains held-out-fold checkpoint selection, the highly
uneven legacy split, two optimizer steps per paired loader iteration, and the
initial train-mode evaluation because all can affect the saved ensemble.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from apexoracle.data.synergy import (
    SynergyFold,
    build_legacy_synergy_folds,
    filter_synergy_token_lengths,
    prepare_legacy_synergy_data,
)
from apexoracle.data.synergy_dataset import (
    SynergyPairDataset,
    collate_synergy_genome_text,
    collate_synergy_text_only,
)
from apexoracle.features.precomputed import (
    load_all_embeddings,
    load_text_only_embeddings,
)
from apexoracle.models.synergy_checkpoint import build_legacy_synergy_components
from apexoracle.training.synergy import (
    legacy_synergy_checkpoint_payload,
    synergy_pair_forward,
    synergy_pair_step,
)


LOGGER = logging.getLogger("apexoracle.synergy_runner")
DEFAULT_CONFIG = Path("configs/synergy/legacy_cv.yaml")
EXPECTED_FILTERED_ROWS = (
    (258, 1917, 653, 27),
    (2110, 65, 2345, 187),
    (2162, 13, 2422, 162),
)


@dataclass(frozen=True)
class SynergyPaths:
    source: Path
    molecule_embeddings: Path
    genome_embeddings: Path
    atcc_text_embeddings: Path
    text_only_embeddings: Path
    base_checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class SynergyConfig:
    source_sha256: str
    python_hash_seed: str
    tokenizer_model: str
    tokenizer_revision: str
    max_length: int
    molecule_dim: int
    genome_dim: int
    text_dim: int
    attention_heads: int
    lora_rank: int
    head_dimensions: tuple[int, ...]
    folds: int
    ensemble_members: int
    seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    eta_min: float
    genome_scale: float
    text_scale: float
    paths: SynergyPaths

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        epochs: int | None = None,
        ensemble_members: int | None = None,
        output_dir: Path | None = None,
    ) -> "SynergyConfig":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else repo_root / path

        data = raw["data"]
        features = raw["features"]
        model = raw["model"]
        training = raw["training"]
        selected_output = output_dir or repo_root / "results" / "synergy_legacy_cv"
        if not selected_output.is_absolute():
            selected_output = repo_root / selected_output
        config = cls(
            source_sha256=str(data["source_sha256"]),
            python_hash_seed=str(data["split_python_hash_seed"]),
            tokenizer_model=str(data["tokenizer"]["model"]),
            tokenizer_revision=str(data["tokenizer"]["revision"]),
            max_length=int(data["tokenizer"]["max_length"]),
            molecule_dim=int(model["molecule_dim"]),
            genome_dim=int(model["genome_dim"]),
            text_dim=int(model["text_dim"]),
            attention_heads=int(model["attention_heads"]),
            lora_rank=int(model["active_fusion_lora"]["rank"]),
            head_dimensions=tuple(
                int(value) for value in model["synergy_head"]["dimensions"]
            ),
            folds=int(training["folds"]),
            ensemble_members=int(
                training["ensemble_members"]
                if ensemble_members is None
                else ensemble_members
            ),
            seeds=tuple(int(seed) for seed in training["seeds"]),
            epochs=int(training["epochs"] if epochs is None else epochs),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            eta_min=float(training["eta_min"]),
            genome_scale=float(features["genome_scale"]),
            text_scale=float(features["text_scale"]),
            paths=SynergyPaths(
                source=resolve(data["source"]),
                molecule_embeddings=resolve(features["molecule"]),
                genome_embeddings=resolve(features["genome"]),
                atcc_text_embeddings=resolve(features["text_with_genome"]),
                text_only_embeddings=resolve(features["text_without_genome"]),
                base_checkpoint=resolve(model["base_checkpoint"]),
                output_dir=selected_output,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.folds != 3:
            raise ValueError("Legacy synergy protocol requires exactly three folds")
        if self.ensemble_members < 1 or len(self.seeds) < self.ensemble_members:
            raise ValueError("Invalid ensemble count or insufficient seeds")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.lora_rank != 1024:
            raise ValueError("Candidate CV checkpoints require fusion LoRA rank 1024")
        expected_head = (
            2 * (self.genome_dim + self.text_dim),
            (self.genome_dim + self.text_dim) // 4,
            128,
            1,
        )
        if self.head_dimensions != expected_head:
            raise ValueError(
                f"Unexpected synergy head {self.head_dimensions}; expected {expected_head}"
            )


@dataclass
class RuntimeFeatures:
    genomes: Mapping[str, torch.Tensor]
    atcc_text: Mapping[str, torch.Tensor]
    text_only: Mapping[str, torch.Tensor]
    molecules: Mapping

    @property
    def all_text(self) -> dict[str, torch.Tensor]:
        return dict(self.atcc_text) | dict(self.text_only)


@dataclass
class FoldLoaders:
    genome_train: DataLoader
    genome_test: DataLoader
    text_train: DataLoader
    text_test: DataLoader


@dataclass
class EvaluationOutput:
    losses: dict[str, list[float]]
    labels: dict[str, list[float]]
    probabilities: dict[str, list[float]]
    pair_keys: list[tuple]
    routes: list[str]

    def metrics(self, route: str = "all") -> dict[str, float]:
        return {
            "auroc": float(
                roc_auc_score(self.labels[route], self.probabilities[route])
            ),
            "auprc": float(
                average_precision_score(self.labels[route], self.probabilities[route])
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_filtered_folds(
    repo_root: Path,
    config: SynergyConfig,
    *,
    local_files_only: bool,
) -> list[SynergyFold]:
    import selfies as sf
    from transformers import AutoTokenizer

    if _sha256(config.paths.source) != config.source_sha256:
        raise ValueError("Synergy source SHA-256 differs from the frozen config")
    prepared = prepare_legacy_synergy_data(repo_root)
    folds = build_legacy_synergy_folds(prepared)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_model,
        revision=config.tokenizer_revision,
        local_files_only=local_files_only,
    )
    cache: dict[str, int] = {}
    filtered_folds = []
    for fold in folds:
        tables = []
        for table in (
            fold.genome_text_train,
            fold.genome_text_test,
            fold.text_only_train,
            fold.text_only_test,
        ):
            tables.append(
                filter_synergy_token_lengths(
                    table,
                    tokenizer=tokenizer,
                    selfies_encoder=sf.encoder,
                    max_length=config.max_length,
                    length_cache=cache,
                ).table
            )
        counts = tuple(len(table) for table in tables)
        if counts != EXPECTED_FILTERED_ROWS[fold.fold]:
            raise ValueError(
                f"Fold {fold.fold} filtered counts {counts} differ from legacy "
                f"{EXPECTED_FILTERED_ROWS[fold.fold]}"
            )
        filtered_folds.append(
            SynergyFold(
                fold=fold.fold,
                strain_for_train=fold.strain_for_train,
                strain_for_test=fold.strain_for_test,
                genome_text_train=tables[0],
                genome_text_test=tables[1],
                text_only_train=tables[2],
                text_only_test=tables[3],
            )
        )
    return filtered_folds


def load_runtime_features(
    config: SynergyConfig, device: torch.device
) -> RuntimeFeatures:
    return RuntimeFeatures(
        genomes=load_all_embeddings(
            config.paths.genome_embeddings, config.genome_scale, device, "genome"
        ),
        atcc_text=load_all_embeddings(
            config.paths.atcc_text_embeddings,
            config.text_scale,
            device,
            "text (with corresponding genome)",
        ),
        text_only=load_text_only_embeddings(
            config.paths.text_only_embeddings,
            config.text_scale,
            device,
            "text (without corresponding genome)",
        ),
        molecules=torch.load(
            config.paths.molecule_embeddings, map_location="cpu", weights_only=False
        ),
    )


def build_fold_loaders(
    fold: SynergyFold,
    features: RuntimeFeatures,
    *,
    batch_size: int,
) -> FoldLoaders:
    def genome_dataset(table):
        return SynergyPairDataset(
            table,
            molecule_embeddings=features.molecules,
            genome_embeddings=features.genomes,
            text_embeddings=features.atcc_text,
        )

    def text_dataset(table):
        return SynergyPairDataset(
            table,
            molecule_embeddings=features.molecules,
            text_embeddings=features.all_text,
        )

    return FoldLoaders(
        genome_train=DataLoader(
            genome_dataset(fold.genome_text_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_synergy_genome_text,
        ),
        genome_test=DataLoader(
            genome_dataset(fold.genome_text_test),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_synergy_genome_text,
        ),
        text_train=DataLoader(
            text_dataset(fold.text_only_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_synergy_text_only,
        ),
        text_test=DataLoader(
            text_dataset(fold.text_only_test),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_synergy_text_only,
        ),
    )


def _empty_evaluation() -> EvaluationOutput:
    routes = ("all", "genome_text", "text_only")
    return EvaluationOutput(
        losses={route: [] for route in routes},
        labels={route: [] for route in routes},
        probabilities={route: [] for route in routes},
        pair_keys=[],
        routes=[],
    )


def evaluate(
    loaders: FoldLoaders,
    *,
    components,
    device: torch.device,
    criterion: nn.Module,
    autocast_enabled: bool,
    selection_mode: bool,
    legacy_initial_probability_bug: bool = False,
) -> EvaluationOutput:
    if selection_mode:
        components.genome_attention.eval()
        components.text_attention.eval()
        components.prediction_head.eval()
    output = _empty_evaluation()
    with torch.no_grad():
        for genome_batch, text_batch in itertools.zip_longest(
            loaders.genome_test, loaders.text_test, fillvalue=None
        ):
            for route, batch, has_genome in (
                ("genome_text", genome_batch, True),
                ("text_only", text_batch, False),
            ):
                if batch is None:
                    continue
                result = synergy_pair_forward(
                    batch,
                    device=device,
                    genome_attention=components.genome_attention,
                    text_attention=components.text_attention,
                    prediction_head=components.prediction_head,
                    criterion=criterion,
                    missing_genome_embedding=components.missing_genome_embedding,
                    has_genome=has_genome,
                    autocast_enabled=autocast_enabled,
                )
                labels = result.labels.detach().cpu().flatten().tolist()
                if legacy_initial_probability_bug and has_genome:
                    probabilities = result.logits.detach().cpu().flatten().tolist()
                else:
                    probabilities = (
                        torch.sigmoid(result.logits).detach().cpu().flatten().tolist()
                    )
                for target in (route, "all"):
                    output.losses[target].append(result.loss.item())
                    output.labels[target].extend(labels)
                    output.probabilities[target].extend(probabilities)
                output.pair_keys.extend(result.pair_keys)
                output.routes.extend([route] * len(labels))
    return output


def train_epoch(
    loaders: FoldLoaders,
    *,
    components,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    autocast_enabled: bool,
    epoch: int,
) -> None:
    components.genome_attention.train()
    components.text_attention.train()
    components.prediction_head.train()
    for genome_batch, text_batch in itertools.zip_longest(
        loaders.genome_train, loaders.text_train, fillvalue=None
    ):
        for batch, has_genome in (
            (genome_batch, True),
            (text_batch, False),
        ):
            if batch is None:
                continue
            synergy_pair_step(
                batch,
                device=device,
                genome_attention=components.genome_attention,
                text_attention=components.text_attention,
                prediction_head=components.prediction_head,
                criterion=criterion,
                missing_genome_embedding=components.missing_genome_embedding,
                optimizer=optimizer,
                scaler=scaler,
                has_genome=has_genome,
                autocast_enabled=autocast_enabled,
                epoch=epoch,
                freeze_epochs=0,
            )


def run_member(
    *,
    config: SynergyConfig,
    loaders: FoldLoaders,
    fold: int,
    member: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[float], list[float], list[tuple], float]:
    seed = config.seeds[member]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    components = build_legacy_synergy_components(
        config.paths.base_checkpoint,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    optimizer = optim.Adam(
        [
            parameter
            for parameter in components.genome_attention.parameters()
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.add_param_group(
        {
            "params": [
                parameter
                for parameter in components.text_attention.parameters()
                if parameter.requires_grad
            ],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    optimizer.add_param_group(
        {
            "params": components.prediction_head.parameters(),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    autocast_enabled = device.type == "cuda"

    # This train-mode pass consumes dropout RNG exactly where the legacy script
    # performs its epoch-0 diagnostic evaluation.
    initial = evaluate(
        loaders,
        components=components,
        device=device,
        criterion=criterion,
        autocast_enabled=autocast_enabled,
        selection_mode=False,
        legacy_initial_probability_bug=True,
    )
    LOGGER.info("fold=%d member=%d initial=%s", fold, member, initial.metrics())

    best_auroc = -10.0
    best_predictions = None
    best_labels = None
    best_pair_keys = None
    for epoch in range(config.epochs):
        train_epoch(
            loaders,
            components=components,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            autocast_enabled=autocast_enabled,
            epoch=epoch,
        )
        scheduler.step()
        selected = evaluate(
            loaders,
            components=components,
            device=device,
            criterion=criterion,
            autocast_enabled=autocast_enabled,
            selection_mode=True,
        )
        metrics = selected.metrics()
        LOGGER.info(
            "fold=%d member=%d epoch=%d/%d metrics=%s",
            fold,
            member,
            epoch + 1,
            config.epochs,
            metrics,
        )
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_predictions = list(selected.probabilities["all"])
            best_labels = list(selected.labels["all"])
            best_pair_keys = list(selected.pair_keys)
            torch.save(
                legacy_synergy_checkpoint_payload(
                    auroc=best_auroc,
                    optimizer=optimizer,
                    prediction_head=components.prediction_head,
                    genome_attention=components.genome_attention,
                    text_attention=components.text_attention,
                    missing_genome_embedding=components.missing_genome_embedding,
                ),
                output_dir / f"fold_{fold}_ensemble_{member}.ckpt",
            )
    assert best_predictions is not None and best_labels is not None
    assert best_pair_keys is not None
    return best_predictions, best_labels, best_pair_keys, best_auroc


def run_fold(
    config: SynergyConfig,
    fold: SynergyFold,
    *,
    features: RuntimeFeatures,
    device: torch.device,
) -> dict:
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_fold_loaders(fold, features, batch_size=config.batch_size)
    members = []
    labels = pair_keys = None
    member_aurocs = []
    for member in range(config.ensemble_members):
        predictions, current_labels, current_keys, auroc = run_member(
            config=config,
            loaders=loaders,
            fold=fold.fold,
            member=member,
            device=device,
            output_dir=output_dir,
        )
        if labels is not None and (
            current_labels != labels or current_keys != pair_keys
        ):
            raise ValueError(
                "Held-out prediction order changed across ensemble members"
            )
        labels, pair_keys = current_labels, current_keys
        members.append(predictions)
        member_aurocs.append(auroc)
    ensemble = np.mean(np.asarray(members), axis=0)
    metrics = {
        "auroc": float(roc_auc_score(labels, ensemble)),
        "auprc": float(average_precision_score(labels, ensemble)),
    }
    prediction_rows = []
    for (first, second, strain), label, prediction in zip(pair_keys, labels, ensemble):
        prediction_rows.append(
            {
                "molecule_1_id": first,
                "molecule_2_id": second,
                "strain_name": strain,
                "fold": fold.fold,
                "label": label,
                "prediction": prediction,
            }
        )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / f"fold_{fold.fold}_predictions.csv", index=False
    )
    result = {
        "fold": fold.fold,
        "metrics": metrics,
        "member_best_aurocs": member_aurocs,
        "test_rows": len(labels),
    }
    (output_dir / f"fold_{fold.fold}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ensemble-members", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-experimental-protocol", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = SynergyConfig.load(
        config_path,
        repo_root,
        epochs=args.epochs,
        ensemble_members=args.ensemble_members,
        output_dir=args.output_dir,
    )
    actual_hash_seed = os.environ.get("PYTHONHASHSEED")
    if actual_hash_seed != config.python_hash_seed:
        raise SystemExit(
            f"Set PYTHONHASHSEED={config.python_hash_seed}; got {actual_hash_seed!r}"
        )
    folds = prepare_filtered_folds(
        repo_root, config, local_files_only=args.local_files_only
    )
    selected_ids = args.fold or [0, 1, 2]
    selected = [folds[index] for index in selected_ids]
    summary = {
        "status": "experimental_candidate",
        "fold_rows_after_token_filter": {
            str(fold.fold): [
                len(fold.genome_text_train),
                len(fold.genome_text_test),
                len(fold.text_only_train),
                len(fold.text_only_test),
            ]
            for fold in selected
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    if not args.confirm_experimental_protocol:
        raise SystemExit(
            "Full training requires --confirm-experimental-protocol because the "
            "candidate rank/base checkpoint differ from Methods."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    features = load_runtime_features(config, device)
    results = [
        run_fold(config, fold, features=features, device=device) for fold in selected
    ]
    summary["results"] = results
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
