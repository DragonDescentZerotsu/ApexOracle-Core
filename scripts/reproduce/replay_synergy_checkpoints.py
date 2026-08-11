#!/usr/bin/env python3
"""Replay the frozen high-confidence synergy checkpoint family.

This entrypoint does not retrain models. It evaluates the complete 3-fold x
7-member checkpoint grid against the fixed ``PYTHONHASHSEED=0`` legacy-codepath
candidate, writes privacy-minimized predictions, and records the difference
from the archived fold metrics. The candidate must not be described as proven
2025 membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from apexoracle.models.synergy_checkpoint import (  # noqa: E402
    build_legacy_synergy_components,
    load_legacy_synergy_member,
)
from apexoracle.training.synergy_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    SynergyConfig,
    build_fold_loaders,
    evaluate,
    load_runtime_features,
    prepare_filtered_folds,
)


DEFAULT_CHECKPOINT_DIR = Path(
    "Checkpoints/genome_text_learnable_emb/strain_wise_synergy/"
    "MDLM_3_fold_ensembles_1_base_model_cls"
)
DEFAULT_CHECKPOINT_MANIFEST = Path("experiments/synergy/checkpoint_file_manifest.csv")
DEFAULT_SPLIT_MANIFEST = Path("experiments/synergy/legacy_split_seed0.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def pair_identity(pair_key: tuple[object, ...]) -> str:
    encoded = "\0".join(str(value) for value in pair_key).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_checkpoint_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 22 or {row["role"] for row in rows} != {"base", "ensemble"}:
        raise ValueError("Expected one base and 21 ensemble checkpoint records")
    return rows


def selected_checkpoint_records(
    rows: list[dict[str, str]], selected_folds: tuple[int, ...]
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["role"] == "base" or int(row["fold"]) in selected_folds
    ]


def verify_checkpoint_records(
    rows: list[dict[str, str]], asset_root: Path
) -> list[dict[str, object]]:
    verified = []
    for row in rows:
        path = asset_root / row["path"]
        expected_size = int(row["size_bytes"])
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        verified.append(
            {
                "role": row["role"],
                "fold": None if row["fold"] == "" else int(row["fold"]),
                "member": None if row["member"] == "" else int(row["member"]),
                "filename": path.name,
                "size_bytes": expected_size,
                "sha256": observed,
            }
        )
    return verified


def build_prediction_table(
    *,
    fold: int,
    routes: list[str],
    pair_keys: list[tuple],
    labels: list[float],
    member_predictions: list[list[float]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    expected = len(labels)
    if len(routes) != expected or len(pair_keys) != expected:
        raise ValueError("Prediction metadata length mismatch")
    if not member_predictions or any(
        len(values) != expected for values in member_predictions
    ):
        raise ValueError("Member prediction length mismatch")
    matrix = np.asarray(member_predictions, dtype=np.float64)
    ensemble = matrix.mean(axis=0)
    pair_identities = [pair_identity(key) for key in pair_keys]
    occurrences: dict[str, int] = {}
    measurement_indices = []
    for identity in pair_identities:
        measurement_indices.append(occurrences.get(identity, 0))
        occurrences[identity] = occurrences.get(identity, 0) + 1
    data: dict[str, object] = {
        "pair_identity": pair_identities,
        "measurement_index": measurement_indices,
        "fold": [fold] * expected,
        "route": routes,
        "strain_id": [str(key[2]) for key in pair_keys],
        "label": [int(value) for value in labels],
    }
    for member, values in enumerate(matrix):
        data[f"member_{member}_probability"] = values
    data["ensemble_probability"] = ensemble
    table = pd.DataFrame(data)
    if table.duplicated(["pair_identity", "measurement_index"]).any():
        raise ValueError(f"Fold {fold} contains duplicate measurement identities")
    metrics = {
        "auroc": float(roc_auc_score(labels, ensemble)),
        "auprc": float(average_precision_score(labels, ensemble)),
    }
    return table, metrics


def replay(
    *,
    code_root: Path,
    asset_root: Path,
    config_path: Path,
    checkpoint_dir: Path,
    checkpoint_manifest: Path,
    split_manifest: Path,
    output_dir: Path,
    selected_folds: tuple[int, ...],
    device: torch.device,
    local_files_only: bool,
    verify_checkpoints: bool,
) -> dict[str, object]:
    config = SynergyConfig.load(config_path, asset_root)
    if os.environ.get("PYTHONHASHSEED") != config.python_hash_seed:
        raise RuntimeError(
            f"Set PYTHONHASHSEED={config.python_hash_seed}; "
            f"got {os.environ.get('PYTHONHASHSEED')!r}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    manifest_rows = read_checkpoint_manifest(checkpoint_manifest)
    selected_records = selected_checkpoint_records(manifest_rows, selected_folds)
    verified_records = (
        verify_checkpoint_records(selected_records, asset_root)
        if verify_checkpoints
        else [
            {
                "role": row["role"],
                "fold": None if row["fold"] == "" else int(row["fold"]),
                "member": None if row["member"] == "" else int(row["member"]),
                "filename": Path(row["path"]).name,
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
                "verification": "manifest_only",
            }
            for row in selected_records
        ]
    )

    folds = prepare_filtered_folds(
        asset_root, config, local_files_only=local_files_only
    )
    features = load_runtime_features(config, device)
    components = build_legacy_synergy_components(
        config.paths.base_checkpoint,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    criterion = nn.BCEWithLogitsLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    all_tables = []
    archived = {
        0: {"auroc": 0.6690, "auprc": 0.6159},
        1: {"auroc": 0.7614, "auprc": 0.6853},
        2: {"auroc": 0.8489, "auprc": 0.9307},
    }
    for fold_id in selected_folds:
        fold = folds[fold_id]
        loaders = build_fold_loaders(fold, features, batch_size=config.batch_size)
        member_predictions: list[list[float]] = []
        labels = routes = pair_keys = None
        saved_member_aurocs = []
        for member in range(config.ensemble_members):
            checkpoint = checkpoint_dir / f"fold_{fold_id}_ensemble_{member}.ckpt"
            saved_member_aurocs.append(
                load_legacy_synergy_member(checkpoint, components=components)
            )
            current = evaluate(
                loaders,
                components=components,
                device=device,
                criterion=criterion,
                autocast_enabled=device.type == "cuda",
                selection_mode=True,
            )
            current_labels = current.labels["all"]
            if labels is not None and (
                current_labels != labels
                or current.pair_keys != pair_keys
                or current.routes != routes
            ):
                raise ValueError("Held-out prediction order changed across members")
            labels = list(current_labels)
            pair_keys = list(current.pair_keys)
            routes = list(current.routes)
            member_predictions.append(list(current.probabilities["all"]))
        assert labels is not None and pair_keys is not None and routes is not None
        table, metrics = build_prediction_table(
            fold=fold_id,
            routes=routes,
            pair_keys=pair_keys,
            labels=labels,
            member_predictions=member_predictions,
        )
        prediction_path = output_dir / f"fold_{fold_id}_predictions.csv"
        table.to_csv(prediction_path, index=False, lineterminator="\n")
        all_tables.append(table)
        fold_summaries.append(
            {
                "fold": fold_id,
                "rows": len(table),
                "positive_rows": int(table["label"].sum()),
                "unique_pair_keys": int(table["pair_identity"].nunique()),
                "repeated_measurement_rows": int(
                    (table["measurement_index"] > 0).sum()
                ),
                "route_rows": {
                    key: int(value)
                    for key, value in table["route"].value_counts().items()
                },
                "metrics": metrics,
                "archived_log_metrics": archived[fold_id],
                "absolute_delta_from_archived_log": {
                    metric: abs(metrics[metric] - archived[fold_id][metric])
                    for metric in ("auroc", "auprc")
                },
                "saved_member_aurocs": saved_member_aurocs,
                "prediction_file": prediction_path.name,
                "prediction_sha256": sha256_file(prediction_path),
            }
        )

    combined = pd.concat(all_tables, ignore_index=True)
    combined_path = output_dir / "all_fold_predictions.csv"
    combined.to_csv(combined_path, index=False, lineterminator="\n")
    mean_metrics = {
        metric: float(np.mean([row["metrics"][metric] for row in fold_summaries]))
        for metric in ("auroc", "auprc")
    }
    summary = {
        "schema_version": 1,
        "status": "replayed",
        "historical_boundary": (
            "PYTHONHASHSEED=0 legacy-codepath candidate; not proven exact 2025 membership"
        ),
        "code_revision": git_revision(code_root),
        "source_sha256": config.source_sha256,
        "split_manifest_sha256": sha256_file(split_manifest),
        "checkpoint_hash_verification": (
            "computed" if verify_checkpoints else "manifest_only"
        ),
        "checkpoint_registry": verified_records,
        "folds": fold_summaries,
        "all_fold_rows": len(combined),
        "unweighted_fold_mean": mean_metrics,
        "paper_reported_mean": {"auroc": 0.7539, "auprc": 0.7454},
        "combined_prediction_file": combined_path.name,
        "combined_prediction_sha256": sha256_file(combined_path),
        "privacy_boundary": [
            "no exact FICI values",
            "no raw molecule identifiers or structures",
            "no embeddings, checkpoints, optimizer state, or absolute author paths",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-checkpoint-hash-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asset_root = args.asset_root.resolve()
    summary = replay(
        code_root=ROOT,
        asset_root=asset_root,
        config_path=args.config.resolve(),
        checkpoint_dir=(
            args.checkpoint_dir.resolve()
            if args.checkpoint_dir
            else asset_root / DEFAULT_CHECKPOINT_DIR
        ),
        checkpoint_manifest=(
            args.checkpoint_manifest.resolve()
            if args.checkpoint_manifest
            else asset_root / DEFAULT_CHECKPOINT_MANIFEST
        ),
        split_manifest=(
            args.split_manifest.resolve()
            if args.split_manifest
            else asset_root / DEFAULT_SPLIT_MANIFEST
        ),
        output_dir=args.output_dir.resolve(),
        selected_folds=tuple(args.fold or (0, 1, 2)),
        device=torch.device(args.device),
        local_files_only=args.local_files_only,
        verify_checkpoints=not args.skip_checkpoint_hash_verification,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
