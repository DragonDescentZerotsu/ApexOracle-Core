"""Inference on versioned hierarchical-MIC embedding bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from apexoracle.models.hierarchical_mic_checkpoint import (
    load_hierarchical_inference_checkpoint,
    predict_genome_text,
    predict_text_only,
)

EXAMPLE_FORMAT = "apexoracle_hierarchical_mic_input_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batched(tensor: torch.Tensor, *, feature_dim: int, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim not in {2, 3} or tensor.shape[-1] != feature_dim:
        raise ValueError(
            f"{name} must end in feature dimension {feature_dim}; got {tuple(tensor.shape)}"
        )
    return tensor


def _sequence_batch(
    tensor: torch.Tensor, *, feature_dim: int, name: str
) -> torch.Tensor:
    tensor = _batched(tensor, feature_dim=feature_dim, name=name)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _padding_mask(bundle: dict[str, Any], key: str, sequence: torch.Tensor) -> torch.Tensor:
    mask = bundle.get(key)
    if mask is None:
        return torch.zeros(sequence.shape[:2], dtype=torch.bool)
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{key} must be a tensor")
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != tuple(sequence.shape[:2]):
        raise ValueError(
            f"{key} shape {tuple(mask.shape)} does not match {tuple(sequence.shape[:2])}"
        )
    return mask.to(dtype=torch.bool)


def predict_mic_bundle(
    checkpoint_path: Path,
    input_path: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Predict MIC from a released embedding bundle.

    The regression target is the paper transform ``z = -log10(MIC_um / 10)``;
    this function returns both ``z`` and the inverse-transformed MIC in µM.
    """

    resolved_device = torch.device(device)
    components, contract = load_hierarchical_inference_checkpoint(
        checkpoint_path, device=resolved_device
    )
    components.eval()
    bundle = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format") != EXAMPLE_FORMAT:
        raise ValueError(f"Input must use format {EXAMPLE_FORMAT}")

    molecule = bundle.get("molecule_embedding")
    text = bundle.get("text_embeddings")
    if not isinstance(molecule, torch.Tensor) or not isinstance(text, torch.Tensor):
        raise ValueError("Input requires molecule_embedding and text_embeddings tensors")
    molecule = _batched(
        molecule,
        feature_dim=contract["molecule_dim"],
        name="molecule_embedding",
    )
    if molecule.ndim != 2:
        raise ValueError("molecule_embedding must have shape [batch, features]")
    text = _sequence_batch(
        text, feature_dim=contract["text_dim"], name="text_embeddings"
    )
    text_mask = _padding_mask(bundle, "text_padding_mask", text)
    if molecule.shape[0] != text.shape[0]:
        raise ValueError("Molecule and text batch sizes differ")

    molecule = molecule.to(resolved_device)
    text = text.to(resolved_device)
    text_mask = text_mask.to(resolved_device)
    genome = bundle.get("genome_embeddings")

    autocast_enabled = resolved_device.type == "cuda"
    with torch.inference_mode(), torch.amp.autocast(
        resolved_device.type, enabled=autocast_enabled
    ):
        if genome is None:
            route = "text_only"
            prediction = predict_text_only(components, molecule, text, text_mask)
        else:
            if not isinstance(genome, torch.Tensor):
                raise ValueError("genome_embeddings must be a tensor")
            genome = _sequence_batch(
                genome,
                feature_dim=contract["genome_dim"],
                name="genome_embeddings",
            )
            genome_mask = _padding_mask(bundle, "genome_padding_mask", genome)
            if genome.shape[0] != molecule.shape[0]:
                raise ValueError("Molecule and genome batch sizes differ")
            route = "genome_text"
            prediction = predict_genome_text(
                components,
                molecule,
                genome.to(resolved_device),
                genome_mask.to(resolved_device),
                text,
                text_mask,
            )

    z = prediction.float().cpu().flatten()
    mic_um = 10.0 * torch.pow(torch.tensor(10.0), -z)
    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    return {
        "schema_version": 1,
        "route": route,
        "prediction_transform": "z = -log10(MIC_um / 10)",
        "prediction_z": z.tolist(),
        "predicted_mic_um": mic_um.tolist(),
        "archived_checkpoint_r2": components.archived_r2,
        "input_metadata": metadata,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a released ApexOracle hierarchical MIC checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checkpoint-sha256")
    parser.add_argument("--verify-input-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    for path, expected in (
        (args.checkpoint, args.verify_checkpoint_sha256),
        (args.input, args.verify_input_sha256),
    ):
        if expected is not None:
            observed = _sha256(path)
            if observed != expected:
                raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")
    result = predict_mic_bundle(args.checkpoint, args.input, device=args.device)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
