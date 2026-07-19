#!/usr/bin/env python3
"""Validate one legacy synergy member against an inline reference forward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.synergy_dataset import (  # noqa: E402
    SynergyPairDataset,
    collate_synergy_genome_text,
    collate_synergy_text_only,
)
from apexoracle.models.synergy_checkpoint import (  # noqa: E402
    build_legacy_synergy_components,
    load_legacy_synergy_member,
)
from apexoracle.training.synergy import synergy_pair_forward  # noqa: E402
from apexoracle.training.synergy_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    SynergyConfig,
    prepare_filtered_folds,
)


def _embedding_key(path: Path) -> str:
    name = path.stem
    if "ATCC" not in name:
        return name
    components = name.split("ATCC")[-1].split("_")[1:]
    return "-".join(components) if len(components) == 2 else components[0]


def _load_atcc_embedding(folder: Path, strain: str, scale: float, device):
    for path in folder.iterdir():
        if path.is_file() and _embedding_key(path) == strain:
            return torch.load(path, map_location="cpu", weights_only=False).to(device) * scale
    raise KeyError(f"No embedding for {strain} in {folder}")


def _load_text_only_embedding(folder: Path, strain: str, scale: float, device):
    for path in folder.iterdir():
        name = path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        if path.is_file() and name == strain:
            return torch.load(path, map_location="cpu", weights_only=False).to(device) * scale
    raise KeyError(f"No text embedding for {strain} in {folder}")


def _inline_reference(batch, components, *, has_genome: bool, device):
    molecules = batch["mol_emb"].to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        if has_genome:
            genome_values = batch["padded_genome_embeddings"]
            genome_masks = batch["genome_attn_masks"]
        else:
            genome_values = components.missing_genome_embedding[:, None, :].expand(
                molecules.shape[0], 1, -1
            )
            genome_masks = torch.ones(
                (molecules.shape[0], 1), device=device, dtype=torch.int64
            )
        genome = components.genome_attention(
            mol_cls_emb=molecules,
            genome_embs=genome_values,
            key_padding_mask=1 - genome_masks,
        )
        text = components.text_attention(
            mol_cls_emb=molecules,
            genome_embs=batch["padded_text_embeddings"],
            key_padding_mask=1 - batch["text_attn_masks"],
        )
        fused = torch.cat(
            (
                genome.reshape(-1, genome_values.shape[-1]),
                text.reshape(-1, batch["padded_text_embeddings"].shape[-1]),
            ),
            dim=1,
        )
        first = torch.cat((fused[::2], fused[1::2]), dim=1)
        second = torch.cat((fused[1::2], fused[::2]), dim=1)
        return (
            components.prediction_head(first)
            + components.prediction_head(second)
        ) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--member", type=int, choices=range(7), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0")
    repo_root = REPO_ROOT
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = SynergyConfig.load(config_path, repo_root)
    fold = prepare_filtered_folds(
        repo_root, config, local_files_only=args.local_files_only
    )[args.fold]
    device = torch.device(args.device)
    molecules = torch.load(
        config.paths.molecule_embeddings, map_location="cpu", weights_only=False
    )

    genome_row = fold.genome_text_test.iloc[[0]].copy()
    genome_strain = genome_row.iloc[0]["strain_name"]
    genome_dataset = SynergyPairDataset(
        genome_row,
        molecule_embeddings=molecules,
        genome_embeddings={
            genome_strain: _load_atcc_embedding(
                config.paths.genome_embeddings,
                genome_strain,
                config.genome_scale,
                device,
            )
        },
        text_embeddings={
            genome_strain: _load_atcc_embedding(
                config.paths.atcc_text_embeddings,
                genome_strain,
                config.text_scale,
                device,
            )
        },
    )
    text_row = fold.text_only_test.iloc[[0]].copy()
    text_strain = text_row.iloc[0]["strain_name"]
    text_dataset = SynergyPairDataset(
        text_row,
        molecule_embeddings=molecules,
        text_embeddings={
            text_strain: _load_text_only_embedding(
                config.paths.text_only_embeddings,
                text_strain,
                config.text_scale,
                device,
            )
        },
    )
    genome_batch = collate_synergy_genome_text([genome_dataset[0]])
    text_batch = collate_synergy_text_only([text_dataset[0]])

    torch.manual_seed(config.seeds[args.member])
    torch.cuda.manual_seed(config.seeds[args.member])
    components = build_legacy_synergy_components(
        config.paths.base_checkpoint,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    checkpoint = (
        repo_root
        / "Checkpoints/genome_text_learnable_emb/strain_wise_synergy"
        / "MDLM_3_fold_ensembles_1_base_model_cls"
        / f"fold_{args.fold}_ensemble_{args.member}.ckpt"
    )
    saved_auroc = load_legacy_synergy_member(checkpoint, components=components)
    components.genome_attention.eval()
    components.text_attention.eval()
    components.prediction_head.eval()
    criterion = nn.BCEWithLogitsLoss()
    results = {}
    for route, batch, has_genome in (
        ("genome_text", genome_batch, True),
        ("text_only", text_batch, False),
    ):
        shared = synergy_pair_forward(
            batch,
            device=device,
            genome_attention=components.genome_attention,
            text_attention=components.text_attention,
            prediction_head=components.prediction_head,
            criterion=criterion,
            missing_genome_embedding=components.missing_genome_embedding,
            has_genome=has_genome,
            autocast_enabled=True,
        ).logits
        reference = _inline_reference(
            batch, components, has_genome=has_genome, device=device
        )
        torch.testing.assert_close(shared, reference, rtol=0, atol=0)
        results[route] = {
            "pair_key": [
                value.item() if hasattr(value, "item") else value
                for value in batch["pair_keys"][0]
            ],
            "logit": float(shared.item()),
            "exact_reference_match": True,
        }
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint.relative_to(repo_root)),
                "saved_best_auroc": saved_auroc,
                "device": str(device),
                "routes": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
