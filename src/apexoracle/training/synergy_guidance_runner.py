"""Canonical runner for the post-paper all-data synergy guidance classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from apexoracle.data.synergy import (
    build_legacy_synergy_all_data_routes,
    filter_synergy_token_lengths,
    prepare_legacy_synergy_data,
)
from apexoracle.data.synergy_dataset import (
    TokenizedSynergyPairDataset,
    collate_tokenized_synergy_genome_text,
    collate_tokenized_synergy_text_only,
)
from apexoracle.features.precomputed import (
    load_all_embeddings,
    load_text_only_embeddings,
)
from apexoracle.models.legacy_mdlm_encoder import (
    build_frozen_legacy_mdlm_encoder,
)
from apexoracle.models.synergy_checkpoint import (
    SynergyComponents,
    build_legacy_synergy_components,
)
from apexoracle.training.synergy import (
    legacy_synergy_guidance_checkpoint_payload,
    synergy_guidance_pair_step,
)


LOGGER = logging.getLogger("apexoracle.synergy_guidance_runner")
DEFAULT_CONFIG = Path("configs/synergy/legacy_guidance.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GuidancePaths:
    source: Path
    genome_embeddings: Path
    atcc_text_embeddings: Path
    text_only_embeddings: Path
    base_mic_checkpoint: Path
    mdlm_root: Path
    mdlm_checkpoint: Path
    observed_output: Path
    output_dir: Path


@dataclass(frozen=True)
class GuidanceConfig:
    profile: str
    source_sha256: str
    tokenizer_model: str
    tokenizer_revision: str
    eligibility_max_length: int
    model_fixed_length: int
    molecule_dim: int
    genome_dim: int
    text_dim: int
    attention_heads: int
    lora_rank: int
    head_dimensions: tuple[int, ...]
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    eta_min: float
    checkpoint_every: int
    genome_scale: float
    text_scale: float
    paths: GuidancePaths

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        profile: str,
        epochs: int | None = None,
        output_dir: Path | None = None,
    ) -> "GuidanceConfig":
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if profile not in raw["profiles"]:
            raise ValueError(f"Unknown guidance profile: {profile}")

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else repo_root / path

        data = raw["data"]
        features = raw["features"]
        model = raw["model"]
        training = raw["training"]
        selected = raw["profiles"][profile]
        mdlm_root = Path(
            os.environ.get(
                str(model["mdlm_root_env"]),
                str(repo_root / str(model["mdlm_default_root"])),
            )
        ).resolve()
        selected_output = (
            output_dir or repo_root / "results" / "synergy_guidance" / profile
        )
        if not selected_output.is_absolute():
            selected_output = repo_root / selected_output
        config = cls(
            profile=profile,
            source_sha256=str(data["source_sha256"]),
            tokenizer_model=str(data["tokenizer"]["model"]),
            tokenizer_revision=str(data["tokenizer"]["revision"]),
            eligibility_max_length=int(
                data["tokenizer"]["eligibility_max_length"]
            ),
            model_fixed_length=int(data["tokenizer"]["model_fixed_length"]),
            molecule_dim=int(model["molecule_dim"]),
            genome_dim=int(model["genome_dim"]),
            text_dim=int(model["text_dim"]),
            attention_heads=int(model["attention_heads"]),
            lora_rank=int(model["fusion_lora"]["rank"]),
            head_dimensions=tuple(
                int(value) for value in model["guidance_head"]["dimensions"]
            ),
            seed=int(training["seed"]),
            epochs=int(selected["epochs"] if epochs is None else epochs),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            eta_min=float(training["eta_min"]),
            checkpoint_every=int(training["periodic_checkpoint_every_epochs"]),
            genome_scale=float(features["genome_scale"]),
            text_scale=float(features["text_scale"]),
            paths=GuidancePaths(
                source=resolve(data["source"]),
                genome_embeddings=resolve(features["genome"]),
                atcc_text_embeddings=resolve(features["text_with_genome"]),
                text_only_embeddings=resolve(features["text_without_genome"]),
                base_mic_checkpoint=resolve(model["base_mic_checkpoint"]),
                mdlm_root=mdlm_root,
                mdlm_checkpoint=mdlm_root / str(model["mdlm_checkpoint"]),
                observed_output=resolve(selected["observed_output"]),
                output_dir=selected_output,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.lora_rank != 64:
            raise ValueError("Guidance checkpoint contract requires fusion LoRA rank 64")
        expected_head = (
            2 * (self.genome_dim + self.text_dim),
            (self.genome_dim + self.text_dim) // 4,
            128,
            1,
        )
        if self.head_dimensions != expected_head:
            raise ValueError(
                f"Unexpected guidance head {self.head_dimensions}; expected {expected_head}"
            )
        if self.model_fixed_length != 1024:
            raise ValueError("Legacy guidance MDLM requires fixed length 1024")


@dataclass
class GuidanceFeatures:
    genomes: Mapping[str, torch.Tensor]
    atcc_text: Mapping[str, torch.Tensor]
    text_only: Mapping[str, torch.Tensor]

    @property
    def all_text(self) -> dict[str, torch.Tensor]:
        return dict(self.atcc_text) | dict(self.text_only)


def validate_paths(config: GuidanceConfig) -> None:
    required = (
        config.paths.source,
        config.paths.genome_embeddings,
        config.paths.atcc_text_embeddings,
        config.paths.text_only_embeddings,
        config.paths.base_mic_checkpoint,
        config.paths.mdlm_root / "configs",
        config.paths.mdlm_checkpoint,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing guidance resources:\n" + "\n".join(f"  - {p}" for p in missing)
        )
    actual = _sha256(config.paths.source)
    if actual != config.source_sha256:
        raise ValueError(
            f"Guidance source hash changed: expected {config.source_sha256}, got {actual}"
        )


def load_tokenizer(config: GuidanceConfig, *, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.tokenizer_model,
        revision=config.tokenizer_revision,
        local_files_only=local_files_only,
    )


def prepare_filtered_tables(config: GuidanceConfig, repo_root: Path, tokenizer):
    import selfies

    prepared = prepare_legacy_synergy_data(
        repo_root,
        source_path=config.paths.source,
    )
    routes = build_legacy_synergy_all_data_routes(prepared)
    cache: dict[str, int] = {}
    genome = filter_synergy_token_lengths(
        routes.genome_text,
        tokenizer=tokenizer,
        selfies_encoder=selfies.encoder,
        max_length=config.eligibility_max_length,
        length_cache=cache,
    )
    combined = filter_synergy_token_lengths(
        routes.combined_text,
        tokenizer=tokenizer,
        selfies_encoder=selfies.encoder,
        max_length=config.eligibility_max_length,
        length_cache=cache,
    )
    return routes, genome, combined


def dry_run_report(config: GuidanceConfig, repo_root: Path, tokenizer) -> dict:
    routes, genome, combined = prepare_filtered_tables(config, repo_root, tokenizer)
    return {
        "status": "dry_run_ok",
        "profile": config.profile,
        "source": str(config.paths.source),
        "source_sha256": config.source_sha256,
        "raw_rows": len(pd.read_csv(config.paths.source)),
        "eligible_before_token_filter": {
            "genome_text": len(routes.genome_text),
            "combined_text_route": len(routes.combined_text),
        },
        "rows_after_token_filter": {
            "genome_text": genome.retained_rows,
            "combined_text_route": combined.retained_rows,
        },
        "unique_smiles_tokenized": combined.unique_smiles_tokenized,
        "epochs": config.epochs,
        "observed_legacy_output": str(config.paths.observed_output),
        "safe_output": str(config.paths.output_dir),
        "legacy_row_order": {
            "uses_process_hash_order": True,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "exact_historical_process_order_recovered": False,
        },
        "source_mutated": False,
    }


def load_features(config: GuidanceConfig, device: torch.device) -> GuidanceFeatures:
    return GuidanceFeatures(
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
    )


def build_loaders(
    config: GuidanceConfig,
    repo_root: Path,
    tokenizer,
    features: GuidanceFeatures,
):
    import selfies

    prepared = prepare_legacy_synergy_data(repo_root, source_path=config.paths.source)
    routes = build_legacy_synergy_all_data_routes(prepared)
    genome_dataset = TokenizedSynergyPairDataset(
        routes.genome_text,
        tokenizer=tokenizer,
        selfies_encoder=selfies.encoder,
        genome_embeddings=features.genomes,
        text_embeddings=features.atcc_text,
        max_length=config.eligibility_max_length,
    )
    text_dataset = TokenizedSynergyPairDataset(
        routes.combined_text,
        tokenizer=tokenizer,
        selfies_encoder=selfies.encoder,
        text_embeddings=features.all_text,
        max_length=config.eligibility_max_length,
    )
    genome_collate = partial(
        collate_tokenized_synergy_genome_text,
        pad_token_id=tokenizer.pad_token_id,
        fixed_length=config.model_fixed_length,
    )
    text_collate = partial(
        collate_tokenized_synergy_text_only,
        pad_token_id=tokenizer.pad_token_id,
        fixed_length=config.model_fixed_length,
    )
    return (
        DataLoader(
            genome_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=genome_collate,
        ),
        DataLoader(
            text_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=text_collate,
        ),
    )


def _build_optimizer(config: GuidanceConfig, components: SynergyComponents):
    optimizer = optim.Adam(
        [p for p in components.genome_attention.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.add_param_group(
        {
            "params": [
                p for p in components.text_attention.parameters() if p.requires_grad
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


def run(
    config: GuidanceConfig,
    repo_root: Path,
    *,
    device: torch.device,
    tokenizer,
) -> dict:
    features = load_features(config, device)
    genome_loader, text_loader = build_loaders(
        config, repo_root, tokenizer, features
    )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    molecule_encoder = build_frozen_legacy_mdlm_encoder(
        mdlm_root=config.paths.mdlm_root,
        checkpoint_path=config.paths.mdlm_checkpoint,
        tokenizer=tokenizer,
        device=device,
    )
    components = build_legacy_synergy_components(
        config.paths.base_mic_checkpoint,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    optimizer = _build_optimizer(config, components)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.eta_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = -10.0
    history = []
    for epoch in range(config.epochs):
        components.genome_attention.train()
        components.text_attention.train()
        components.prediction_head.train()
        labels = {"all": [], "genome_text": [], "text": []}
        predictions = {"all": [], "genome_text": [], "text": []}
        losses = {"all": [], "genome_text": [], "text": []}
        for genome_batch, text_batch in itertools.zip_longest(
            genome_loader, text_loader, fillvalue=None
        ):
            for batch, has_genome, route in (
                (genome_batch, True, "genome_text"),
                (text_batch, False, "text"),
            ):
                if batch is None:
                    continue
                result = synergy_guidance_pair_step(
                    batch,
                    device=device,
                    molecule_encoder=molecule_encoder,
                    genome_attention=components.genome_attention,
                    text_attention=components.text_attention,
                    prediction_head=components.prediction_head,
                    criterion=criterion,
                    missing_genome_embedding=components.missing_genome_embedding,
                    optimizer=optimizer,
                    scaler=scaler,
                    has_genome=has_genome,
                    autocast_enabled=device.type == "cuda",
                )
                current_labels = result.labels.detach().cpu().flatten().tolist()
                current_predictions = (
                    torch.sigmoid(result.logits).detach().cpu().flatten().tolist()
                )
                for key in ("all", route):
                    labels[key].extend(current_labels)
                    predictions[key].extend(current_predictions)
                    losses[key].append(float(result.loss.item()))
        scheduler.step()
        metrics = {
            route: {
                "auroc": float(roc_auc_score(labels[route], predictions[route])),
                "auprc": float(
                    average_precision_score(labels[route], predictions[route])
                ),
                "loss": float(np.mean(losses[route])),
            }
            for route in ("all", "genome_text", "text")
        }
        history.append({"epoch": epoch + 1, "metrics": metrics})
        LOGGER.info("epoch=%d/%d metrics=%s", epoch + 1, config.epochs, metrics)
        payload = None
        if (
            (epoch + 1) % config.checkpoint_every == 0
            or metrics["all"]["auroc"] > best_auroc
        ):
            payload = legacy_synergy_guidance_checkpoint_payload(
                auroc=best_auroc,
                optimizer=optimizer,
                molecule_encoder=molecule_encoder,
                prediction_head=components.prediction_head,
                genome_attention=components.genome_attention,
                text_attention=components.text_attention,
                missing_genome_embedding=components.missing_genome_embedding,
            )
        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save(payload, output_dir / f"synergy_noise_clsfier_epoch_{epoch}.ckpt")
        if metrics["all"]["auroc"] > best_auroc:
            best_auroc = metrics["all"]["auroc"]
            payload["AUROC"] = best_auroc
            torch.save(payload, output_dir / "synergy_noise_clsfier_best.ckpt")
    summary = {
        "status": "complete",
        "profile": config.profile,
        "best_training_auroc": best_auroc,
        "history": history,
        "evidence_boundary": "post-paper all-data training metric; not held-out performance",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the post-paper all-data synergy guidance classifier."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        choices=["short_judger", "guidance_40epoch"],
        default="guidance_40epoch",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-post-paper-guidance", action="store_true")
    parser.add_argument("--acknowledge-dynamic-legacy-order", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    config = GuidanceConfig.load(
        config_path,
        repo_root,
        profile=args.profile,
        epochs=args.epochs,
        output_dir=output_dir,
    )
    validate_paths(config)
    tokenizer = load_tokenizer(config, local_files_only=args.local_files_only)
    if args.dry_run:
        report = dry_run_report(config, repo_root, tokenizer)
        print(json.dumps(report, indent=2))
        return report
    if not args.confirm_post_paper_guidance:
        raise SystemExit(
            "Refusing to train without --confirm-post-paper-guidance; this model "
            "selects checkpoints on training AUROC and is not paper CV evidence."
        )
    if not args.acknowledge_dynamic_legacy_order:
        raise SystemExit(
            "Refusing to train without --acknowledge-dynamic-legacy-order; the "
            "legacy driver grouped rows through Python sets, and its historical "
            "process hash order was not recorded."
        )
    return run(
        config,
        repo_root,
        device=torch.device(args.device),
        tokenizer=tokenizer,
    )
