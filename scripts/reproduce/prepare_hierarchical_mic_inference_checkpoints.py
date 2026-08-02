#!/usr/bin/env python3
"""Strip unused optimizer/classification payloads from hierarchical MIC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.models.hierarchical_mic_checkpoint import (  # noqa: E402
    inspect_checkpoint_contract,
)
from apexoracle.data.hierarchical_mic_preparation import HoldoutSplit  # noqa: E402
from apexoracle.training.hierarchical_mic_runner import (  # noqa: E402
    HierarchicalMicConfig,
    build_holdout_split,
    checkpoint_filename,
    prepare_hierarchical_mic_data,
)

DEFAULT_STRAIN_MANIFEST = (
    REPO_ROOT / "experiments/hierarchical_mic/strain/legacy_protocol_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_strain_candidate(path: Path) -> HoldoutSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = sorted(payload["folds"], key=lambda row: int(row["fold"]))
    return HoldoutSplit(
        protocol="strain",
        group_names=tuple(f"fold {int(row['fold']) + 1}" for row in folds),
        test_groups=tuple(tuple(map(str, row["test_strain_ids"])) for row in folds),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["strain", "species", "phylum"], required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/hierarchical_mic/legacy_mdlm.yaml")
    parser.add_argument(
        "--strain-manifest",
        type=Path,
        help="Optional frozen strain membership manifest used to resolve checkpoint names.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Source checkpoint directory; defaults to the configured historical output.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", type=int, action="append")
    parser.add_argument("--ensemble", type=int, action="append")
    args = parser.parse_args()

    config = HierarchicalMicConfig.load(
        args.config.resolve(), REPO_ROOT, holdout_protocol=args.protocol
    )
    prepared = prepare_hierarchical_mic_data(
        REPO_ROOT,
        mic_data_path=config.paths.mic_records,
        small_molecule_data_path=config.paths.small_molecule_records,
    )
    split = (
        load_frozen_strain_candidate(args.strain_manifest.resolve())
        if args.protocol == "strain" and args.strain_manifest
        else build_holdout_split(
            prepared,
            REPO_ROOT,
            args.protocol,
            adapter=config.holdout_adapter,
            group_names=config.holdout_group_names,
            tree_path=config.holdout_tree,
            num_clusters=config.holdout_clusters,
        )
    )
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir
        else config.paths.output_dir
    )
    groups = args.group or list(range(len(split.group_names)))
    ensembles = args.ensemble or list(range(config.ensembles_per_group))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group in groups:
        for ensemble in ensembles:
            source = checkpoint_dir / checkpoint_filename(
                config, split, group, ensemble
            )
            source_hash = sha256_file(source)
            checkpoint = torch.load(
                source, map_location="cpu", weights_only=False, mmap=True
            )
            inspect_checkpoint_contract(checkpoint)
            target = output_dir / checkpoint_filename(config, split, group, ensemble)
            payload = {
                "format": "apexoracle_hierarchical_mic_inference_v1",
                "source_checkpoint": str(source),
                "source_checkpoint_size": source.stat().st_size,
                "source_checkpoint_sha256": source_hash,
                "archived_r2": float(checkpoint["R2"]),
                "re_head_state_dict": checkpoint["re_head_state_dict"],
                "co_cross_attn_genome": checkpoint["co_cross_attn_genome"],
                "co_cross_attn_text": checkpoint["co_cross_attn_text"],
                "learnable_embedding_weight": checkpoint["learnable_embedding_weight"],
            }
            torch.save(payload, target)
            row = {
                "protocol": args.protocol,
                "group": group,
                "ensemble": ensemble,
                "source": str(source),
                "source_size": source.stat().st_size,
                "source_sha256": source_hash,
                "inference_checkpoint": str(target),
                "inference_size": target.stat().st_size,
                "inference_sha256": sha256_file(target),
            }
            manifest.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "checkpoints": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
