"""Read-only legacy-compatible prospective synergy pair screening."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from apexoracle.data.synergy_dataset import (
    SynergyPairDataset,
    collate_synergy_genome_text,
)
from apexoracle.models.synergy_checkpoint import (
    build_legacy_synergy_regression_member,
)
from apexoracle.training.synergy import synergy_pair_forward


DEFAULT_CONFIG = Path("configs/synergy/legacy_screening.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScreeningConfig:
    profile: str
    smiles_pairs: Path
    sequence_pairs: Path
    molecule_embeddings: Path
    genome_embedding: Path
    text_embedding: Path
    checkpoint_paths: tuple[Path, ...]
    checkpoint_hashes: tuple[str, ...]
    output_path: Path
    hashes: dict[str, str]
    smiles_rows: int
    sequence_rows: int
    observed_output: Path
    observed_output_rows: int
    observed_output_sha256: str
    strain_id: str
    genome_scale: float
    text_scale: float
    molecule_dim: int
    genome_dim: int
    text_dim: int
    attention_heads: int
    lora_rank: int
    batch_size: int
    historical_world_size: int
    threshold: float

    @classmethod
    def load(
        cls,
        config_path: Path,
        repo_root: Path,
        *,
        profile: str,
        output_path: Path | None = None,
    ) -> "ScreeningConfig":
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if profile not in raw["profiles"]:
            raise ValueError(f"Unknown screening profile: {profile}")

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else repo_root / path

        data = raw["data"]
        strain = raw["strain"]
        model = raw["model"]
        selected = raw["profiles"][profile]
        checkpoint_dir = resolve(model["checkpoint_dir"])
        default_output = (
            resolve(raw["output"]["default_dir"])
            / f"filtered_pairs_{profile}.csv"
        )
        config = cls(
            profile=profile,
            smiles_pairs=resolve(data["smiles_pairs"]),
            sequence_pairs=resolve(data["sequence_pairs"]),
            molecule_embeddings=resolve(data["molecule_embeddings"]),
            genome_embedding=resolve(strain["genome_embedding"]),
            text_embedding=resolve(strain["text_embedding"]),
            checkpoint_paths=tuple(
                checkpoint_dir
                / str(selected["checkpoint_pattern"]).format(member=member)
                for member in range(int(model["ensemble_members"]))
            ),
            checkpoint_hashes=tuple(map(str, selected["checkpoint_sha256"])),
            output_path=(output_path or default_output).resolve(),
            hashes={
                "smiles_pairs": str(data["smiles_pairs_sha256"]),
                "sequence_pairs": str(data["sequence_pairs_sha256"]),
                "molecule_embeddings": str(data["molecule_embeddings_sha256"]),
                "genome_embedding": str(strain["genome_embedding_sha256"]),
                "text_embedding": str(strain["text_embedding_sha256"]),
            },
            smiles_rows=int(data["smiles_pair_rows"]),
            sequence_rows=int(data["sequence_pair_rows"]),
            observed_output=resolve(selected["observed_output"]),
            observed_output_rows=int(selected["observed_output_rows"]),
            observed_output_sha256=str(selected["observed_output_sha256"]),
            strain_id=str(strain["id"]),
            genome_scale=float(strain["genome_scale"]),
            text_scale=float(strain["text_scale"]),
            molecule_dim=int(model["molecule_dim"]),
            genome_dim=int(model["genome_dim"]),
            text_dim=int(model["text_dim"]),
            attention_heads=int(model["attention_heads"]),
            lora_rank=int(model["fusion_lora_rank"]),
            batch_size=int(model["batch_size"]),
            historical_world_size=int(model["historical_ddp_world_size"]),
            threshold=float(model["selection_threshold"]),
        )
        config.validate(repo_root)
        return config

    def validate(self, repo_root: Path) -> None:
        if (
            self.lora_rank != 64
            or len(self.checkpoint_paths) != 7
            or len(self.checkpoint_hashes) != 7
            or self.historical_world_size != 4
        ):
            raise ValueError("Legacy screening requires seven rank-64 members")
        data_root = (repo_root / "DataPrepare" / "Data").resolve()
        if self.output_path == data_root or data_root in self.output_path.parents:
            raise ValueError("Screening output must not be written inside original data")
        if self.batch_size < 1 or not 0 < self.threshold:
            raise ValueError("Batch size and selection threshold must be positive")


def validate_inputs(config: ScreeningConfig, *, verify_hashes: bool) -> dict:
    paths = {
        "smiles_pairs": config.smiles_pairs,
        "sequence_pairs": config.sequence_pairs,
        "molecule_embeddings": config.molecule_embeddings,
        "genome_embedding": config.genome_embedding,
        "text_embedding": config.text_embedding,
    }
    paths.update(
        {f"checkpoint_{index}": path for index, path in enumerate(config.checkpoint_paths)}
    )
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing screening inputs:\n" + "\n".join(map(str, missing)))
    verified = {}
    if verify_hashes:
        for name, expected in config.hashes.items():
            actual = _sha256(paths[name])
            if actual != expected:
                raise ValueError(f"{name} hash changed: expected {expected}, got {actual}")
            verified[name] = actual
        for index, (path, expected) in enumerate(
            zip(config.checkpoint_paths, config.checkpoint_hashes)
        ):
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"checkpoint {index} hash changed: expected {expected}, got {actual}"
                )
            verified[f"checkpoint_{index}"] = actual
    return verified


def inverse_fici_target(logits: torch.Tensor) -> torch.Tensor:
    """Undo the legacy ``-log10(FICI / 10)`` target transform."""

    return torch.pow(10.0, -logits) * 10.0


def select_legacy_positional_rows(
    sequence_rows: pd.DataFrame,
    predictions: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Reproduce the historical positional selection, including misalignment."""

    indices = np.flatnonzero(predictions < threshold)
    selected = sequence_rows.iloc[indices].copy()
    selected[selected.columns[-1]] = predictions[indices]
    return selected


def legacy_ddp_rank_block_order(
    predictions: np.ndarray,
    *,
    world_size: int,
) -> np.ndarray:
    """Reproduce ``DistributedSampler`` plus rank-major ``all_gather`` order."""

    if len(predictions) % world_size:
        raise ValueError(
            "Legacy rank-block order requires a row count divisible by world size"
        )
    return np.concatenate(
        [predictions[rank::world_size] for rank in range(world_size)]
    )


def split_legacy_rank_frames(
    frame: pd.DataFrame,
    *,
    world_size: int,
) -> tuple[pd.DataFrame, ...]:
    """Create the per-rank frames produced by ``DistributedSampler``."""

    if len(frame) % world_size:
        raise ValueError(
            "Legacy rank split requires a row count divisible by world size"
        )
    return tuple(
        frame.iloc[rank::world_size].reset_index(drop=True)
        for rank in range(world_size)
    )


def _load_screening_data(config: ScreeningConfig, max_rows: int | None):
    frame = pd.read_csv(config.smiles_pairs, nrows=max_rows)
    frame["strain_name"] = config.strain_id
    molecule_embeddings = torch.load(
        config.molecule_embeddings,
        map_location="cpu",
        weights_only=False,
    )
    return frame, molecule_embeddings


def _build_loader(
    config: ScreeningConfig,
    frame: pd.DataFrame,
    molecule_embeddings: dict,
    genome_embeddings: dict[str, torch.Tensor],
    text_embeddings: dict[str, torch.Tensor],
) -> DataLoader:
    dataset = SynergyPairDataset(
        frame,
        molecule_embeddings=molecule_embeddings,
        genome_embeddings=genome_embeddings,
        text_embeddings=text_embeddings,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_synergy_genome_text,
    )


def _predict_member(
    config: ScreeningConfig,
    checkpoint_path: Path,
    loaders: tuple[DataLoader, ...],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    components, stored_r2 = build_legacy_synergy_regression_member(
        checkpoint_path,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    predictions = []
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    result = synergy_pair_forward(
                        batch,
                        device=device,
                        genome_attention=components.genome_attention,
                        text_attention=components.text_attention,
                        prediction_head=components.prediction_head,
                        criterion=nn.MSELoss(),
                        missing_genome_embedding=components.missing_genome_embedding,
                        has_genome=True,
                        autocast_enabled=device.type == "cuda",
                    )
                    transformed = inverse_fici_target(result.logits)
                predictions.extend(transformed.detach().cpu().flatten().tolist())
    return np.asarray(predictions, dtype=np.float64), stored_r2


def run_screening(
    config: ScreeningConfig,
    *,
    device: torch.device,
    max_rows: int | None,
    max_members: int | None,
) -> dict:
    frame, molecule_embeddings = _load_screening_data(config, max_rows)
    genome_embeddings = {
        config.strain_id: torch.load(
            config.genome_embedding,
            weights_only=False,
        ).to(device)
        * config.genome_scale
    }
    text_embeddings = {
        config.strain_id: torch.load(
            config.text_embedding,
            weights_only=False,
        ).to(device)
        * config.text_scale
    }
    rank_frames = split_legacy_rank_frames(
        frame,
        world_size=config.historical_world_size,
    )
    loaders = tuple(
        _build_loader(
            config,
            rank_frame,
            molecule_embeddings,
            genome_embeddings,
            text_embeddings,
        )
        for rank_frame in rank_frames
    )
    prediction_sum = np.zeros(len(frame), dtype=np.float64)
    member_r2 = []
    checkpoint_paths = config.checkpoint_paths[:max_members]
    for checkpoint_path in checkpoint_paths:
        predictions, stored_r2 = _predict_member(
            config, checkpoint_path, loaders, device
        )
        prediction_sum += predictions
        member_r2.append(stored_r2)
    predictions = prediction_sum / len(checkpoint_paths)
    sequence_rows = pd.read_csv(config.sequence_pairs, nrows=max_rows)
    selected = select_legacy_positional_rows(
        sequence_rows,
        predictions,
        threshold=config.threshold,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(config.output_path, index=False)
    report = {
        "status": "complete",
        "profile": config.profile,
        "evaluated_rows": len(frame),
        "evaluated_members": len(checkpoint_paths),
        "selected_rows": len(selected),
        "stored_member_r2": member_r2,
        "output": str(config.output_path),
        "alignment": "legacy_ddp_rank_block_then_positional",
        "source_mutated": False,
    }
    report_path = config.output_path.with_suffix(".metrics.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def dry_run_report(config: ScreeningConfig, verified_hashes: dict) -> dict:
    return {
        "status": "dry_run_ok",
        "profile": config.profile,
        "smiles_pair_rows": config.smiles_rows,
        "sequence_pair_rows": config.sequence_rows,
        "row_count_difference": config.sequence_rows - config.smiles_rows,
        "alignment": "legacy_ddp_rank_block_then_positional",
        "historical_ddp_world_size": config.historical_world_size,
        "ensemble_members": len(config.checkpoint_paths),
        "observed_selected_rows": config.observed_output_rows,
        "observed_output_sha256": config.observed_output_sha256,
        "safe_output": str(config.output_path),
        "verified_input_hashes": verified_hashes,
        "source_mutated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce prospective BAA-3170 synergy pair screening."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        choices=["DBAASP_train_best", "inhouse_best"],
        default="DBAASP_train_best",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument(
        "--confirm-legacy-positional-alignment",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be positive")
    if args.max_members is not None and args.max_members < 1:
        raise SystemExit("--max-members must be positive")
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_path = args.output
    if output_path is not None and not output_path.is_absolute():
        output_path = repo_root / output_path
    config = ScreeningConfig.load(
        config_path,
        repo_root,
        profile=args.profile,
        output_path=output_path,
    )
    verified = validate_inputs(
        config,
        verify_hashes=not args.skip_hash_validation,
    )
    if args.dry_run:
        report = dry_run_report(config, verified)
        print(json.dumps(report, indent=2))
        return report
    if not args.confirm_legacy_positional_alignment:
        raise SystemExit(
            "Refusing to screen without --confirm-legacy-positional-alignment; "
            "the historical output selected sequence rows by positions from a "
            "shorter SMILES table."
        )
    report = run_screening(
        config,
        device=torch.device(args.device),
        max_rows=args.max_rows,
        max_members=args.max_members,
    )
    print(json.dumps(report, indent=2))
    return report
