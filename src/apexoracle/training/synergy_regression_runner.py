"""Legacy-compatible producer for prospective synergy regression members."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from apexoracle.data.synergy import (
    build_legacy_synergy_all_data_routes,
    filter_synergy_token_lengths,
    prepare_legacy_synergy_data,
    synergy_regression_target,
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
    legacy_synergy_regression_checkpoint_payload,
    synergy_pair_forward,
    synergy_pair_step,
)


LOGGER = logging.getLogger("apexoracle.synergy_regression_runner")
DEFAULT_CONFIG = Path("configs/synergy/legacy_regression_producer.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegressionProducerPaths:
    train_source: Path
    test_source: Path
    mapping: Path
    taxonomy_aliases: Path
    train_molecules: Path
    test_molecules: Path
    genome_embeddings: Path
    atcc_text_embeddings: Path
    text_only_embeddings: Path
    base_checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class RegressionProducerConfig:
    hashes: Mapping[str, str]
    expected_rows: Mapping[str, int]
    tokenizer_model: str
    tokenizer_revision: str
    max_length: int
    molecule_dim: int
    genome_dim: int
    text_dim: int
    attention_heads: int
    lora_rank: int
    head_dimensions: tuple[int, ...]
    ensemble_members: int
    seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    eta_min: float
    fixed_epoch_index: int
    genome_scale: float
    text_scale: float
    paths: RegressionProducerPaths

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        epochs: int | None = None,
        ensemble_members: int | None = None,
        output_dir: Path | None = None,
    ) -> "RegressionProducerConfig":
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else repo_root / path

        data = raw["data"]
        features = raw["features"]
        model = raw["model"]
        training = raw["training"]
        selected_output = output_dir or resolve(raw["output"]["default_dir"])
        if not selected_output.is_absolute():
            selected_output = repo_root / selected_output
        config = cls(
            hashes={
                "train_source": str(data["train_source_sha256"]),
                "test_source": str(data["test_source_sha256"]),
                "mapping": str(data["mapping_sha256"]),
                "taxonomy_aliases": str(data["taxonomy_aliases_sha256"]),
                "train_molecules": str(features["train_molecules_sha256"]),
                "test_molecules": str(features["test_molecules_sha256"]),
                "base_checkpoint": str(model["base_checkpoint_sha256"]),
            },
            expected_rows={
                key: int(value) for key, value in data["expected_rows"].items()
            },
            tokenizer_model=str(data["tokenizer"]["model"]),
            tokenizer_revision=str(data["tokenizer"]["revision"]),
            max_length=int(data["tokenizer"]["max_length"]),
            molecule_dim=int(model["molecule_dim"]),
            genome_dim=int(model["genome_dim"]),
            text_dim=int(model["text_dim"]),
            attention_heads=int(model["attention_heads"]),
            lora_rank=int(model["fusion_lora_rank"]),
            head_dimensions=tuple(int(value) for value in model["head_dimensions"]),
            ensemble_members=int(
                training["ensemble_members"]
                if ensemble_members is None
                else ensemble_members
            ),
            seeds=tuple(int(value) for value in training["seeds"]),
            epochs=int(training["epochs"] if epochs is None else epochs),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            eta_min=float(training["eta_min"]),
            fixed_epoch_index=int(training["fixed_checkpoint_epoch_index"]),
            genome_scale=float(features["genome_scale"]),
            text_scale=float(features["text_scale"]),
            paths=RegressionProducerPaths(
                train_source=resolve(data["train_source"]),
                test_source=resolve(data["test_source"]),
                mapping=resolve(data["mapping"]),
                taxonomy_aliases=resolve(data["taxonomy_aliases"]),
                train_molecules=resolve(features["train_molecules"]),
                test_molecules=resolve(features["test_molecules"]),
                genome_embeddings=resolve(features["genome"]),
                atcc_text_embeddings=resolve(features["text_with_genome"]),
                text_only_embeddings=resolve(features["text_without_genome"]),
                base_checkpoint=resolve(model["base_checkpoint"]),
                output_dir=selected_output.resolve(),
            ),
        )
        config.validate(repo_root)
        return config

    def validate(self, repo_root: Path) -> None:
        expected_head = (
            2 * (self.genome_dim + self.text_dim),
            (self.genome_dim + self.text_dim) // 4,
            128,
            1,
        )
        if self.lora_rank != 64 or self.head_dimensions != expected_head:
            raise ValueError("Unexpected prospective regression model contract")
        if not 1 <= self.ensemble_members <= len(self.seeds):
            raise ValueError("Invalid ensemble count")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        protected = (
            (repo_root / "DataPrepare" / "Data").resolve(),
            (repo_root / "Checkpoints").resolve(),
        )
        if any(
            self.paths.output_dir == root or root in self.paths.output_dir.parents
            for root in protected
        ):
            raise ValueError("Regression output must not overwrite data or checkpoints")


@dataclass(frozen=True)
class RegressionTables:
    train_genome_text: pd.DataFrame
    train_combined_text: pd.DataFrame
    test_genome_text: pd.DataFrame
    unique_smiles_tokenized: int


@dataclass
class RegressionRuntimeFeatures:
    genomes: Mapping[str, torch.Tensor]
    atcc_text: Mapping[str, torch.Tensor]
    text_only: Mapping[str, torch.Tensor]
    molecules: Mapping

    @property
    def all_text(self) -> dict[str, torch.Tensor]:
        return dict(self.atcc_text) | dict(self.text_only)


@dataclass(frozen=True)
class RegressionLoaders:
    genome_train: DataLoader
    text_train: DataLoader
    genome_test: DataLoader


@dataclass(frozen=True)
class RegressionEvaluation:
    losses: tuple[float, ...]
    labels: tuple[float, ...]
    predictions: tuple[float, ...]
    pair_keys: tuple[tuple, ...]

    def metrics(self) -> dict[str, float]:
        labels = np.asarray(self.labels)
        predictions = np.asarray(self.predictions)
        return {
            "r2": legacy_r2(labels, predictions),
            "spearman": float(spearmanr(labels, predictions)[0]),
            "pearson": float(pearsonr(labels, predictions)[0]),
            "mse": float(np.mean((labels - predictions) ** 2)),
        }


def validate_inputs(
    config: RegressionProducerConfig,
    *,
    verify_hashes: bool,
) -> dict[str, str]:
    files = {
        "train_source": config.paths.train_source,
        "test_source": config.paths.test_source,
        "mapping": config.paths.mapping,
        "taxonomy_aliases": config.paths.taxonomy_aliases,
        "train_molecules": config.paths.train_molecules,
        "test_molecules": config.paths.test_molecules,
        "base_checkpoint": config.paths.base_checkpoint,
    }
    missing = [path for path in files.values() if not path.is_file()]
    for directory in (
        config.paths.genome_embeddings,
        config.paths.atcc_text_embeddings,
        config.paths.text_only_embeddings,
    ):
        if not directory.is_dir():
            missing.append(directory)
    if missing:
        raise FileNotFoundError("Missing producer inputs:\n" + "\n".join(map(str, missing)))
    verified = {}
    if verify_hashes:
        for name, path in files.items():
            actual = _sha256(path)
            expected = config.hashes[name]
            if actual != expected:
                raise ValueError(
                    f"{name} hash changed: expected {expected}, got {actual}"
                )
            verified[name] = actual
    return verified


def _assert_row_count(
    config: RegressionProducerConfig,
    key: str,
    table: pd.DataFrame,
) -> None:
    expected = config.expected_rows[key]
    if len(table) != expected:
        raise ValueError(f"{key} has {len(table)} rows; expected {expected}")


def prepare_regression_tables(
    repo_root: Path,
    config: RegressionProducerConfig,
    *,
    local_files_only: bool,
) -> RegressionTables:
    import selfies as sf
    from transformers import AutoTokenizer

    train = build_legacy_synergy_all_data_routes(
        prepare_legacy_synergy_data(repo_root, source_path=config.paths.train_source)
    )
    test = build_legacy_synergy_all_data_routes(
        prepare_legacy_synergy_data(repo_root, source_path=config.paths.test_source)
    )
    _assert_row_count(
        config, "train_genome_text_before_filter", train.genome_text
    )
    _assert_row_count(
        config, "train_combined_text_before_filter", train.combined_text
    )
    _assert_row_count(config, "test_genome_text_before_filter", test.genome_text)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_model,
        revision=config.tokenizer_revision,
        local_files_only=local_files_only,
    )
    cache: dict[str, int] = {}

    def filtered(table: pd.DataFrame) -> pd.DataFrame:
        return filter_synergy_token_lengths(
            table,
            tokenizer=tokenizer,
            selfies_encoder=sf.encoder,
            max_length=config.max_length,
            length_cache=cache,
        ).table

    train_genome = filtered(train.genome_text)
    train_text = filtered(train.combined_text)
    test_genome = filtered(test.genome_text)
    _assert_row_count(config, "train_genome_text_after_filter", train_genome)
    _assert_row_count(config, "train_combined_text_after_filter", train_text)
    _assert_row_count(config, "test_genome_text_after_filter", test_genome)
    return RegressionTables(
        train_genome_text=train_genome,
        train_combined_text=train_text,
        test_genome_text=test_genome,
        unique_smiles_tokenized=len(cache),
    )


def load_runtime_features(
    config: RegressionProducerConfig,
    device: torch.device,
) -> RegressionRuntimeFeatures:
    train_molecules = torch.load(
        config.paths.train_molecules,
        map_location="cpu",
        weights_only=False,
    )
    test_molecules = torch.load(
        config.paths.test_molecules,
        map_location="cpu",
        weights_only=False,
    )
    return RegressionRuntimeFeatures(
        genomes=load_all_embeddings(
            config.paths.genome_embeddings,
            config.genome_scale,
            device,
            "genome",
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
        molecules=train_molecules | test_molecules,
    )


def build_regression_loaders(
    tables: RegressionTables,
    features: RegressionRuntimeFeatures,
    *,
    batch_size: int,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
) -> RegressionLoaders:
    genome_train = tables.train_genome_text.iloc[:max_train_rows]
    text_train = tables.train_combined_text.iloc[:max_train_rows]
    genome_test = tables.test_genome_text.iloc[:max_test_rows]

    def dataset(table, *, has_genome: bool) -> SynergyPairDataset:
        return SynergyPairDataset(
            table,
            molecule_embeddings=features.molecules,
            genome_embeddings=features.genomes if has_genome else None,
            text_embeddings=features.atcc_text if has_genome else features.all_text,
            target_transform=synergy_regression_target,
        )

    return RegressionLoaders(
        genome_train=DataLoader(
            dataset(genome_train, has_genome=True),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_synergy_genome_text,
        ),
        text_train=DataLoader(
            dataset(text_train, has_genome=False),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_synergy_text_only,
        ),
        genome_test=DataLoader(
            dataset(genome_test, has_genome=True),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_synergy_genome_text,
        ),
    )


def legacy_r2(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    total = np.sum((labels - np.mean(labels)) ** 2)
    residual = np.sum((labels - predictions) ** 2)
    return float(1 - residual / total)


def evaluate_regression(
    loader: DataLoader,
    *,
    components,
    device: torch.device,
    criterion: nn.Module,
    autocast_enabled: bool,
    selection_mode: bool,
) -> RegressionEvaluation:
    if selection_mode:
        components.genome_attention.eval()
        components.text_attention.eval()
        components.prediction_head.eval()
    losses: list[float] = []
    labels: list[float] = []
    predictions: list[float] = []
    pair_keys: list[tuple] = []
    with torch.no_grad():
        for batch in loader:
            result = synergy_pair_forward(
                batch,
                device=device,
                genome_attention=components.genome_attention,
                text_attention=components.text_attention,
                prediction_head=components.prediction_head,
                criterion=criterion,
                missing_genome_embedding=components.missing_genome_embedding,
                has_genome=True,
                autocast_enabled=autocast_enabled,
            )
            losses.append(result.loss.item())
            labels.extend(result.labels.detach().cpu().flatten().tolist())
            predictions.extend(result.logits.detach().cpu().flatten().tolist())
            pair_keys.extend(result.pair_keys)
    return RegressionEvaluation(
        losses=tuple(losses),
        labels=tuple(labels),
        predictions=tuple(predictions),
        pair_keys=tuple(pair_keys),
    )


def train_regression_epoch(
    loaders: RegressionLoaders,
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
        loaders.genome_train,
        loaders.text_train,
        fillvalue=None,
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


def build_regression_optimizer(config, components) -> optim.Adam:
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
    return optimizer


def run_member(
    config: RegressionProducerConfig,
    loaders: RegressionLoaders,
    *,
    member: int,
    device: torch.device,
) -> dict:
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
    optimizer = build_regression_optimizer(config, components)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.eta_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.MSELoss()
    autocast_enabled = device.type == "cuda"

    initial = evaluate_regression(
        loaders.genome_test,
        components=components,
        device=device,
        criterion=criterion,
        autocast_enabled=autocast_enabled,
        selection_mode=False,
    )
    LOGGER.info("member=%d initial=%s", member, initial.metrics())

    best_r2 = -10.0
    best: RegressionEvaluation | None = None
    history = []
    for epoch in range(config.epochs):
        train_regression_epoch(
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
        selected = evaluate_regression(
            loaders.genome_test,
            components=components,
            device=device,
            criterion=criterion,
            autocast_enabled=autocast_enabled,
            selection_mode=True,
        )
        metrics = selected.metrics()
        history.append(metrics)
        LOGGER.info(
            "member=%d epoch=%d/%d metrics=%s",
            member,
            epoch + 1,
            config.epochs,
            metrics,
        )
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best = selected
            torch.save(
                legacy_synergy_regression_checkpoint_payload(
                    r2=best_r2,
                    optimizer=optimizer,
                    prediction_head=components.prediction_head,
                    genome_attention=components.genome_attention,
                    text_attention=components.text_attention,
                    missing_genome_embedding=components.missing_genome_embedding,
                ),
                config.paths.output_dir
                / f"fold_0_ensemble_{member}_best_test.ckpt",
            )
        if epoch == config.fixed_epoch_index:
            torch.save(
                legacy_synergy_regression_checkpoint_payload(
                    r2=best_r2,
                    optimizer=optimizer,
                    prediction_head=components.prediction_head,
                    genome_attention=components.genome_attention,
                    text_attention=components.text_attention,
                    missing_genome_embedding=components.missing_genome_embedding,
                ),
                config.paths.output_dir
                / f"fold_0_ensemble_{member}_fixed_epoch.ckpt",
            )
    if best is None:
        raise RuntimeError("No best regression checkpoint was selected")
    return {
        "member": member,
        "seed": seed,
        "best_r2": best_r2,
        "labels": list(best.labels),
        "predictions": list(best.predictions),
        "pair_keys": [list(key) for key in best.pair_keys],
        "history": history,
    }


def run_regression_producer(
    config: RegressionProducerConfig,
    tables: RegressionTables,
    *,
    device: torch.device,
    max_train_rows: int | None,
    max_test_rows: int | None,
) -> dict:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    features = load_runtime_features(config, device)
    loaders = build_regression_loaders(
        tables,
        features,
        batch_size=config.batch_size,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )
    members = [
        run_member(config, loaders, member=member, device=device)
        for member in range(config.ensemble_members)
    ]
    labels = members[0]["labels"]
    pair_keys = members[0]["pair_keys"]
    if any(
        member["labels"] != labels or member["pair_keys"] != pair_keys
        for member in members[1:]
    ):
        raise ValueError("Held-out prediction order changed across members")
    ensemble = np.mean(
        np.asarray([member["predictions"] for member in members]),
        axis=0,
    )
    metrics = {
        "r2": legacy_r2(np.asarray(labels), ensemble),
        "spearman": float(spearmanr(labels, ensemble)[0]),
        "pearson": float(pearsonr(labels, ensemble)[0]),
    }
    prediction_table = pd.DataFrame(pair_keys, columns=["first_id", "second_id", "strain"])
    prediction_table["target"] = labels
    for member in members:
        prediction_table[f"member_{member['member']}"] = member["predictions"]
    prediction_table["ensemble"] = ensemble
    prediction_table.to_csv(
        config.paths.output_dir / "held_out_predictions.csv",
        index=False,
    )
    report = {
        "status": "complete",
        "purpose": "post_paper_prospective_regression",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "dynamic_legacy_row_order": True,
        "train_genome_text_rows": len(loaders.genome_train.dataset),
        "train_combined_text_rows": len(loaders.text_train.dataset),
        "test_genome_text_rows": len(loaders.genome_test.dataset),
        "ensemble_members": config.ensemble_members,
        "epochs": config.epochs,
        "metrics": metrics,
        "member_best_r2": [member["best_r2"] for member in members],
        "inhouse_test_used_for_checkpoint_selection": True,
        "source_mutated": False,
    }
    with (config.paths.output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def dry_run_report(
    config: RegressionProducerConfig,
    tables: RegressionTables,
    verified_hashes: Mapping[str, str],
) -> dict:
    return {
        "status": "dry_run_ok",
        "purpose": "post_paper_prospective_regression",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "dynamic_legacy_row_order": True,
        "train_genome_text_rows": len(tables.train_genome_text),
        "train_combined_text_rows": len(tables.train_combined_text),
        "test_genome_text_rows": len(tables.test_genome_text),
        "unique_smiles_tokenized": tables.unique_smiles_tokenized,
        "ensemble_members": config.ensemble_members,
        "epochs": config.epochs,
        "fixed_checkpoint_epoch_index": config.fixed_epoch_index,
        "inhouse_test_used_for_checkpoint_selection": True,
        "safe_output": str(config.paths.output_dir),
        "verified_input_hashes": dict(verified_hashes),
        "source_mutated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the legacy prospective synergy regression ensemble."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ensemble-members", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-post-paper-regression", action="store_true")
    parser.add_argument("--acknowledge-dynamic-legacy-order", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    for name in ("epochs", "ensemble_members", "max_train_rows", "max_test_rows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    config = RegressionProducerConfig.load(
        config_path,
        repo_root,
        epochs=args.epochs,
        ensemble_members=args.ensemble_members,
        output_dir=output_dir,
    )
    verified = validate_inputs(
        config,
        verify_hashes=not args.skip_hash_validation,
    )
    tables = prepare_regression_tables(
        repo_root,
        config,
        local_files_only=args.local_files_only,
    )
    if args.dry_run:
        report = dry_run_report(config, tables, verified)
        print(json.dumps(report, indent=2))
        return report
    if not args.confirm_post_paper_regression:
        raise SystemExit(
            "Refusing to train without --confirm-post-paper-regression; this "
            "producer is not the paper synergy CV."
        )
    if not args.acknowledge_dynamic_legacy_order:
        raise SystemExit(
            "Refusing to train without --acknowledge-dynamic-legacy-order; "
            "the historical Python hash seed was not recorded."
        )
    report = run_regression_producer(
        config,
        tables,
        device=torch.device(args.device),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    print(json.dumps(report, indent=2))
    return report
