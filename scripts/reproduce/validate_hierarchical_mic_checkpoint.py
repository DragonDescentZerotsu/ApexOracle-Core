#!/usr/bin/env python3
"""Strictly load one hierarchical MIC checkpoint and run fixed inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.models.hierarchical_mic_checkpoint import (  # noqa: E402
    load_legacy_hierarchical_checkpoint,
    predict_genome_text,
    predict_text_only,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    device = torch.device(args.device)
    components, contract = load_legacy_hierarchical_checkpoint(
        args.checkpoint, device=device
    )
    components.eval()
    generator = torch.Generator(device="cpu").manual_seed(20260718)
    molecule = torch.randn(2, contract["molecule_dim"], generator=generator).to(device)
    genome = torch.randn(
        2, 2, contract["genome_dim"], generator=generator, dtype=torch.bfloat16
    ).to(device)
    text = torch.randn(
        2, 3, contract["text_dim"], generator=generator, dtype=torch.bfloat16
    ).to(device)
    genome_mask = torch.tensor([[False, False], [False, True]], device=device)
    text_mask = torch.tensor(
        [[False, False, False], [False, False, True]], device=device
    )
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        genome_text_prediction = predict_genome_text(
            components, molecule, genome, genome_mask, text, text_mask
        )
        text_only_prediction = predict_text_only(components, molecule, text, text_mask)
    resolved_checkpoint = args.checkpoint.resolve()
    try:
        checkpoint_display = str(resolved_checkpoint.relative_to(REPO_ROOT))
    except ValueError:
        checkpoint_display = str(resolved_checkpoint)
    result = {
        "checkpoint": checkpoint_display,
        "size_bytes": args.checkpoint.stat().st_size,
        "sha256": file_sha256(args.checkpoint) if args.sha256 else None,
        "archived_r2": components.archived_r2,
        "contract": contract,
        "synthetic_seed": 20260718,
        "genome_text_prediction": genome_text_prediction.float()
        .cpu()
        .flatten()
        .tolist(),
        "text_only_prediction": text_only_prediction.float().cpu().flatten().tolist(),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
