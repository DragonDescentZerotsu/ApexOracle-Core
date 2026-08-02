#!/usr/bin/env python3
"""Replay one hierarchical MIC checkpoint on full/seen/unseen peptide test rows."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.hierarchical_mic import (  # noqa: E402
    StrainEmbeddingDataset,
    TextOnlyStrainEmbeddingDataset,
    collate_genome_text_regression,
    collate_text_regression,
)
from apexoracle.data.hierarchical_mic_preparation import (  # noqa: E402
    HoldoutSplit,
    build_holdout_split,
    prepare_hierarchical_mic_data,
)
from apexoracle.evaluation.hierarchical_mic_molecule_overlap import (  # noqa: E402
    IDENTITY_MODEL_INPUT,
    add_molecule_identity,
    apply_legacy_token_length_filter,
    concatenate_routes,
)
from apexoracle.models.hierarchical_mic_checkpoint import (  # noqa: E402
    load_hierarchical_inference_checkpoint,
    load_legacy_hierarchical_checkpoint,
)
from apexoracle.training.hierarchical_mic import hierarchical_mic_batch_forward  # noqa: E402
from apexoracle.training.hierarchical_mic_runner import (  # noqa: E402
    HierarchicalMicConfig,
    checkpoint_filename,
    load_runtime_features,
    prepare_holdout_frames,
)

DEFAULT_STRAIN_MANIFEST = (
    REPO_ROOT / "experiments/hierarchical_mic/strain/legacy_protocol_manifest.json"
)


def load_frozen_strain_candidate(path: Path) -> HoldoutSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = sorted(payload["folds"], key=lambda row: int(row["fold"]))
    return HoldoutSplit(
        protocol="strain",
        group_names=tuple(f"fold {int(row['fold']) + 1}" for row in folds),
        test_groups=tuple(tuple(map(str, row["test_strain_ids"])) for row in folds),
    )


def eligible_identified(frame: pd.DataFrame, route: str, group: int) -> pd.DataFrame:
    output = add_molecule_identity(
        apply_legacy_token_length_filter(frame), IDENTITY_MODEL_INPUT
    )
    output["route"] = route
    output["row_key"] = [
        f"g{group}:{route}:{index}" for index in range(len(output))
    ]
    return output


def infer_route(
    frame: pd.DataFrame,
    *,
    route: str,
    features,
    components,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    common = (None, features.peptide_embeddings, features.small_molecule_embeddings)
    if route == "genome_text":
        dataset = StrainEmbeddingDataset(
            frame,
            common[0],
            features.genome_embeddings,
            features.atcc_text_embeddings,
            "molecule-disjoint genome-text test",
            common[1],
            common[2],
        )
        collate = collate_genome_text_regression
        has_genome = True
    elif route == "text_only":
        dataset = TextOnlyStrainEmbeddingDataset(
            frame,
            common[0],
            features.all_text_embeddings,
            "molecule-disjoint text-only test",
            common[1],
            common[2],
        )
        collate = collate_text_regression
        has_genome = False
    else:
        raise ValueError(route)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    criterion = nn.MSELoss()
    with torch.inference_mode():
        for batch in loader:
            result = hierarchical_mic_batch_forward(
                batch,
                device=device,
                genome_attention=components.genome_attention,
                text_attention=components.text_attention,
                prediction_head=components.regression_head,
                criterion=criterion,
                missing_genome_embedding=components.missing_genome_embedding,
                has_genome=has_genome,
                reshape_outputs=True,
                autocast_enabled=device.type == "cuda",
            )
            predictions.append(result.logits.detach().float().reshape(-1).cpu().numpy())
            labels.append(result.labels.detach().float().reshape(-1).cpu().numpy())
    metadata = dataset.dataframe.copy()
    metadata["label_z"] = np.concatenate(labels)
    metadata["prediction"] = np.concatenate(predictions)
    return metadata.drop(columns=["input_ids", "attn_mask"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["strain", "species", "phylum"], required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument(
        "--ensemble",
        type=int,
        action="append",
        required=True,
        help="Repeat to evaluate multiple ensemble members while reusing loaded features.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/hierarchical_mic/legacy_mdlm.yaml")
    parser.add_argument("--strain-manifest", type=Path, default=DEFAULT_STRAIN_MANIFEST)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = HierarchicalMicConfig.load(
        args.config.resolve(), REPO_ROOT, holdout_protocol=args.protocol
    )
    with redirect_stdout(StringIO()):
        prepared = prepare_hierarchical_mic_data(
            REPO_ROOT,
            mic_data_path=config.paths.mic_records,
            small_molecule_data_path=config.paths.small_molecule_records,
        )
        split = (
            load_frozen_strain_candidate(args.strain_manifest.resolve())
            if args.protocol == "strain"
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
    frames = prepare_holdout_frames(prepared, split, args.group)
    train = concatenate_routes(
        [
            apply_legacy_token_length_filter(frames.genome_text_train),
            apply_legacy_token_length_filter(frames.text_only_train),
        ]
    )
    train = add_molecule_identity(train, IDENTITY_MODEL_INPUT)
    train_seen = set(train["molecule_identity"])
    peptide_means = (
        train.assign(label_z=-np.log10(train["MIC"].astype(float) / 10.0))
        .groupby("molecule_identity")["label_z"]
        .mean()
    )
    global_train_mean = float(
        -np.log10(train["MIC"].astype(float) / 10.0).mean()
    )
    route_frames = [
        eligible_identified(frames.genome_text_test, "genome_text", args.group),
        eligible_identified(frames.text_only_test, "text_only", args.group),
    ]
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir
        else config.paths.output_dir
    )
    loader = (
        load_hierarchical_inference_checkpoint
        if args.inference_only
        else load_legacy_hierarchical_checkpoint
    )
    features = load_runtime_features(config, device, REPO_ROOT)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ensemble in args.ensemble:
        checkpoint = checkpoint_dir / checkpoint_filename(
            config, split, args.group, ensemble
        )
        components, contract = loader(
            checkpoint,
            device=device,
            num_heads=config.attention_heads,
            attention_dropout=config.attention_dropout,
            head_dropout=config.head_dropout,
        )
        components.eval()
        predictions = [
            infer_route(
                frame,
                route=route,
                features=features,
                components=components,
                device=device,
                batch_size=args.batch_size,
            )
            for frame, route in zip(route_frames, ("genome_text", "text_only"))
            if len(frame)
        ]
        output = pd.concat(predictions, ignore_index=True)
        output["train_seen_molecule"] = output["molecule_identity"].isin(train_seen)
        output["peptide_mean_baseline_prediction"] = (
            output["molecule_identity"].map(peptide_means).fillna(global_train_mean)
        )
        output["training_mean_baseline_prediction"] = global_train_mean
        output["MIC_um"] = output["MIC"].astype(float)
        output["protocol"] = args.protocol
        output["group_index"] = args.group
        output["group_name"] = split.group_names[args.group]
        output["ensemble"] = ensemble
        columns = [
            "row_key", "protocol", "group_index", "group_name", "ensemble", "route",
            "DBAASP_id", "molecule_identity", "strain_name", "MIC_um", "label_z",
            "train_seen_molecule", "prediction", "peptide_mean_baseline_prediction",
            "training_mean_baseline_prediction",
        ]
        output = output[columns]
        if output["row_key"].duplicated().any():
            raise RuntimeError("row_key is not unique")
        stem = f"{args.protocol}_group_{args.group}_ensemble_{ensemble}"
        output.to_csv(output_dir / f"{stem}.csv", index=False)
        report = {
            "schema_version": 1,
            "status": "completed",
            "task": stem,
            "checkpoint": str(checkpoint),
            "checkpoint_format": (
                "inference_only" if args.inference_only else "legacy_full"
            ),
            "archived_r2": components.archived_r2,
            "contract": contract,
            "evaluation_mode": "deterministic_eval",
            "membership_status": (
                "deterministic_candidate_not_exact_2025"
                if args.protocol == "strain"
                else "canonical_taxonomy_cluster_adapter"
            ),
            "identity_definition": IDENTITY_MODEL_INPUT,
            "test_measurements": len(output),
            "test_train_seen_measurements": int(
                output["train_seen_molecule"].sum()
            ),
            "test_train_unseen_measurements": int(
                (~output["train_seen_molecule"]).sum()
            ),
            "global_train_mean_z": global_train_mean,
        }
        (output_dir / f"{stem}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        del components
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
