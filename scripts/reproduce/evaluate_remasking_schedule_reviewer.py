#!/usr/bin/env python
"""Evaluate raw remasking-schedule attempts with frozen clean models.

The generation runner deliberately saves every attempted token sequence. This
script adds two post-hoc measurements without changing the generation
denominator:

1. clean-input peptide probability from the same v1 classifier checkpoint used
   by the sampler; and
2. predicted MIC from the frozen clean (non-noisy) MIC regressor, evaluated
   only for RDKit-valid molecules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer


MODEL_NAME = "ibm-research/materials.selfies-ted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mdlm-root", type=Path, required=True)
    parser.add_argument("--synergy-root", type=Path, required=True)
    parser.add_argument("--peptide-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-mic-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--mic-batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate available completed tasks instead of requiring the manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class ClassificationHead(nn.Module):
    """Exact head architecture used by the historical v1 classifier."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, 384)
        self.dense_2 = nn.Linear(384, 128)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.activation(self.dense_1(features)))
        features = self.dropout(self.activation(self.dense_2(features)))
        return self.out_proj(features)


class V1PeptideClassifier(nn.Module):
    """Deployment-equivalent v1 classifier evaluated at clean input (t=0)."""

    def __init__(self, mdlm_root: Path, checkpoint_path: Path):
        super().__init__()
        if str(mdlm_root) not in sys.path:
            sys.path.insert(0, str(mdlm_root))
        models = importlib.import_module("models")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = OmegaConf.create(
            OmegaConf.to_container(
                checkpoint["hyper_parameters"]["config"], resolve=False
            )
        )
        self.config = config
        vocab_size = int(checkpoint["hyper_parameters"]["vocab_size"])
        self.backbone = models.dit.DIT(config, vocab_size=vocab_size)
        backbone_state = OrderedDict()
        prefix = "backbone.backbone."
        for key, value in checkpoint["state_dict"].items():
            if key.startswith(prefix):
                backbone_state[key.removeprefix(prefix)] = value
        missing, unexpected = self.backbone.load_state_dict(
            backbone_state, strict=False
        )
        material_missing = [
            key
            for key in missing
            if not key.startswith(("output_layer.", "regression."))
        ]
        if material_missing or unexpected:
            raise RuntimeError(
                "Classifier backbone mismatch: "
                f"missing={material_missing}, unexpected={unexpected}"
            )
        self.head = ClassificationHead(input_dim=int(config.model.hidden_size))
        head_state = OrderedDict(
            (
                key.removeprefix("ClsHead."),
                value,
            )
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("ClsHead.")
        )
        self.head.load_state_dict(head_state, strict=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sigma = torch.zeros(
            input_ids.shape[0], dtype=torch.float32, device=input_ids.device
        )
        hidden = self.backbone.vocab_embed(input_ids)
        conditioning = F.silu(self.backbone.sigma_map(sigma))
        rotary = self.backbone.rotary_emb(hidden)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in self.backbone.blocks:
                hidden = block(hidden, rotary, conditioning, seqlens=None)
        return self.head(hidden[:, 0, :].float()).squeeze(-1)


def collect_rows(
    manifest: dict,
    runs_root: Path,
    *,
    allow_incomplete: bool,
) -> tuple[list[dict], list[dict]]:
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    rows: list[dict] = []
    input_files: list[dict] = []
    missing = []
    for task_id, task in tasks.items():
        task_dir = runs_root / task_id
        completed = task_dir / "completed.json"
        if not completed.exists():
            missing.append(task_id)
            continue
        completion = json.loads(completed.read_text(encoding="utf-8"))
        if int(completion["attempted_samples"]) != int(
            task["attempted_samples"]
        ):
            raise RuntimeError(
                f"{task_id}: completion attempted-sample count differs "
                "from task manifest"
            )
        batch_paths = sorted((task_dir / "batches").glob("batch_*.jsonl"))
        if len(batch_paths) != int(task["num_batches"]):
            raise RuntimeError(
                f"{task_id}: expected {task['num_batches']} batches, "
                f"found {len(batch_paths)}"
            )
        completion_files = {
            item["path"]: item for item in completion["batch_files"]
        }
        if len(completion_files) != len(batch_paths):
            raise RuntimeError(
                f"{task_id}: completion marker contains "
                f"{len(completion_files)} batch entries for "
                f"{len(batch_paths)} files"
            )
        task_count = 0
        for batch_path in batch_paths:
            relative_path = str(batch_path.relative_to(task_dir))
            if relative_path not in completion_files:
                raise RuntimeError(
                    f"{task_id}: {relative_path} absent from completion marker"
                )
            actual_sha256 = sha256_file(batch_path)
            expected_file = completion_files[relative_path]
            if (
                int(expected_file["size"]) != batch_path.stat().st_size
                or expected_file["sha256"] != actual_sha256
            ):
                raise RuntimeError(
                    f"{task_id}: transferred batch identity mismatch for "
                    f"{relative_path}"
                )
            input_files.append(
                {
                    "path": str(batch_path.resolve()),
                    "size": batch_path.stat().st_size,
                    "sha256": actual_sha256,
                }
            )
            with batch_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    row.update(
                        {
                            "task_id": task_id,
                            "condition": task["condition"],
                            "t_on": task["t_on"],
                            "t_off": task["t_off"],
                            "gamma_peptide": task["gamma_peptide"],
                            "target_length": task["target_length"],
                            "generation_host": task["host"],
                        }
                    )
                    rows.append(row)
                    task_count += 1
        if task_count != int(task["attempted_samples"]):
            raise RuntimeError(
                f"{task_id}: expected {task['attempted_samples']} rows, "
                f"found {task_count}"
            )
    if missing and not allow_incomplete:
        raise RuntimeError(f"Incomplete generation tasks: {missing}")
    if not rows:
        raise RuntimeError("No completed generation attempts found")
    return rows, input_files


@torch.inference_mode()
def add_peptide_probabilities(
    rows: list[dict],
    *,
    mdlm_root: Path,
    checkpoint_path: Path,
    batch_size: int,
) -> None:
    model = V1PeptideClassifier(mdlm_root, checkpoint_path).cuda().eval()
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row["token_ids"])].append(index)
    for _, indices in sorted(groups.items()):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            input_ids = torch.tensor(
                [rows[index]["token_ids"] for index in batch_indices],
                dtype=torch.long,
                device="cuda",
            )
            probabilities = (
                torch.sigmoid(model(input_ids)).float().cpu().tolist()
            )
            for index, probability in zip(batch_indices, probabilities):
                rows[index]["peptide_probability"] = float(probability)
                rows[index]["peptide_classifier_positive"] = bool(
                    probability >= 0.5
                )
    del model
    torch.cuda.empty_cache()


def load_clean_mic_model(
    *,
    mdlm_root: Path,
    synergy_root: Path,
    checkpoint_path: Path,
    tokenizer,
):
    if str(mdlm_root) not in sys.path:
        sys.path.insert(0, str(mdlm_root))
    module = importlib.import_module("judge_generated_mols_MIC")
    module.current_directory = synergy_root
    module.tokenizer = tokenizer
    module.device = torch.device("cuda")
    model = module.MIC_regressor(
        module.config, str(checkpoint_path), module.device
    )
    return model.cuda().eval()


@torch.inference_mode()
def add_mic_predictions(
    rows: list[dict],
    *,
    mdlm_root: Path,
    synergy_root: Path,
    checkpoint_path: Path,
    tokenizer,
    batch_size: int,
) -> None:
    model = load_clean_mic_model(
        mdlm_root=mdlm_root,
        synergy_root=synergy_root,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
    )
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row["predicted_mic_uM"] = None
        if row["rdkit_valid"] and row["first_sep_index"] is not None:
            sequence_length = int(row["first_sep_index"]) + 1
            groups[(row["strain"], sequence_length)].append(index)
    for (strain, sequence_length), indices in sorted(groups.items()):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            input_ids = torch.tensor(
                [
                    rows[index]["token_ids"][:sequence_length]
                    for index in batch_indices
                ],
                dtype=torch.long,
                device="cuda",
            )
            logits = model(input_ids, strain).squeeze(-1).float()
            predictions = (torch.pow(10.0, -logits) * 10.0).cpu().tolist()
            for index, prediction in zip(batch_indices, predictions):
                rows[index]["predicted_mic_uM"] = float(prediction)
    del model
    torch.cuda.empty_cache()


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_group(group_rows: list[dict]) -> dict:
    attempted = len(group_rows)
    complete = sum(bool(row["complete"]) for row in group_rows)
    valid = sum(bool(row["rdkit_valid"]) for row in group_rows)
    amide = sum(bool(row["has_amide_bond"]) for row in group_rows)
    peptide = sum(
        bool(row["peptide_classifier_positive"]) for row in group_rows
    )
    valid_peptide = sum(
        bool(row["rdkit_valid"] and row["peptide_classifier_positive"])
        for row in group_rows
    )
    valid_small_molecule = sum(
        bool(row["rdkit_valid"] and not row["peptide_classifier_positive"])
        for row in group_rows
    )
    probabilities = [float(row["peptide_probability"]) for row in group_rows]
    mics = [
        float(row["predicted_mic_uM"])
        for row in group_rows
        if row["predicted_mic_uM"] is not None
        and math.isfinite(float(row["predicted_mic_uM"]))
    ]
    peptide_valid_mics = [
        float(row["predicted_mic_uM"])
        for row in group_rows
        if row["predicted_mic_uM"] is not None
        and row["peptide_classifier_positive"]
        and math.isfinite(float(row["predicted_mic_uM"]))
    ]
    return {
        "attempted": attempted,
        "complete": complete,
        "complete_proportion": complete / attempted,
        "rdkit_valid": valid,
        "rdkit_valid_proportion": valid / attempted,
        "amide_positive": amide,
        "amide_positive_proportion": amide / attempted,
        "peptide_classifier_positive": peptide,
        "peptide_classifier_positive_proportion": peptide / attempted,
        "rdkit_valid_peptide_classifier_positive": valid_peptide,
        "rdkit_valid_peptide_classifier_positive_proportion_of_attempts": (
            valid_peptide / attempted
        ),
        "rdkit_valid_peptide_classifier_positive_proportion_of_valid": (
            valid_peptide / valid if valid else None
        ),
        "rdkit_valid_classifier_negative_small_molecule": valid_small_molecule,
        "rdkit_valid_classifier_negative_small_molecule_proportion_of_valid": (
            valid_small_molecule / valid if valid else None
        ),
        "mean_peptide_probability": float(np.mean(probabilities)),
        "valid_predicted_mic_n": len(mics),
        "valid_predicted_mic_median_uM": quantile(mics, 0.5),
        "valid_predicted_mic_q1_uM": quantile(mics, 0.25),
        "valid_predicted_mic_q3_uM": quantile(mics, 0.75),
        "peptide_positive_valid_predicted_mic_n": len(peptide_valid_mics),
        "peptide_positive_valid_predicted_mic_median_uM": quantile(
            peptide_valid_mics, 0.5
        ),
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "task_id",
        "condition",
        "strain",
        "seed",
        "generation_host",
        "t_on",
        "t_off",
        "gamma_peptide",
        "target_length",
        "batch_index",
        "sample_index",
        "complete",
        "rdkit_valid",
        "has_amide_bond",
        "peptide_probability",
        "peptide_classifier_positive",
        "predicted_mic_uM",
        "canonical_smiles",
        "invalid_reason",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES before evaluation."
        )
    for name in (
        "task_manifest",
        "runs_root",
        "output_dir",
        "mdlm_root",
        "synergy_root",
        "peptide_classifier_checkpoint",
        "clean_mic_checkpoint",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.classifier_batch_size < 1 or args.mic_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    rows, input_files = collect_rows(
        manifest, args.runs_root, allow_incomplete=args.allow_incomplete
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    add_peptide_probabilities(
        rows,
        mdlm_root=args.mdlm_root,
        checkpoint_path=args.peptide_classifier_checkpoint,
        batch_size=args.classifier_batch_size,
    )
    add_mic_predictions(
        rows,
        mdlm_root=args.mdlm_root,
        synergy_root=args.synergy_root,
        checkpoint_path=args.clean_mic_checkpoint,
        tokenizer=tokenizer,
        batch_size=args.mic_batch_size,
    )

    write_rows_csv(args.output_dir / "evaluated_attempts.csv", rows)
    summary: dict[str, dict] = {}
    grouping_specs = {
        "condition": ("condition",),
        "condition_strain": ("condition", "strain"),
        "condition_strain_seed": ("condition", "strain", "seed"),
    }
    for grouping_name, fields in grouping_specs.items():
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[field] for field in fields)].append(row)
        summary[grouping_name] = {
            "|".join(str(value) for value in key): summarize_group(group_rows)
            for key, group_rows in sorted(grouped.items())
        }
    summary["overall"] = summarize_group(rows)
    atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    provenance = {
        "schema_version": 1,
        "created_unix": time.time(),
        "host": os.uname().nodename,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "task_manifest_sha256": sha256_file(args.task_manifest),
        "peptide_classifier_checkpoint": {
            "path": str(args.peptide_classifier_checkpoint),
            "size": args.peptide_classifier_checkpoint.stat().st_size,
            "sha256": sha256_file(args.peptide_classifier_checkpoint),
        },
        "clean_mic_checkpoint": {
            "path": str(args.clean_mic_checkpoint),
            "size": args.clean_mic_checkpoint.stat().st_size,
            "sha256": sha256_file(args.clean_mic_checkpoint),
        },
        "input_files": input_files,
        "row_count": len(rows),
        "outputs": {
            filename: {
                "size": (args.output_dir / filename).stat().st_size,
                "sha256": sha256_file(args.output_dir / filename),
            }
            for filename in ("evaluated_attempts.csv", "summary.json")
        },
    }
    atomic_write_text(
        args.output_dir / "provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
