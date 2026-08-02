#!/usr/bin/env python
"""Audit whether the remasking-reviewer peptide label is structurally credible.

This audit does not redefine peptide identity.  It checks three narrower
questions:

1. whether the saved probabilities came from the documented historical v1
   checkpoint;
2. how much scoring changes when tokens after the first ``[SEP]`` are replaced
   by padding, matching the molecule actually decoded by the generation runner;
3. how classifier-positive RDKit-valid molecules overlap two amide-bond QC
   definitions.

The row-level output remains local.  The compact JSON report is suitable for
reviewing the evidence and preserving provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce.evaluate_remasking_schedule_reviewer import (
    ClassificationHead,
    MODEL_NAME,
    V1PeptideClassifier,
)


EXPERIMENT = ROOT / "experiments" / "remasking_schedule_reviewer"
DEFAULT_CHECKPOINT = (
    ROOT.parent
    / "mdlm"
    / "cls-guide-pad-no-mask-checkpoints"
    / "epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt"
)
RUNNER_AMIDE_PATTERN = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluated-attempts",
        type=Path,
        default=EXPERIMENT / "analysis" / "evaluated_attempts.csv",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=EXPERIMENT / "runs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT / "analysis" / "peptide_structure_audit",
    )
    parser.add_argument("--mdlm-root", type=Path, default=ROOT.parent / "mdlm")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reviewer-retrain-root",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "peptide_classifier"
            / "reviewer_retrain"
            / "runs"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict) -> tuple[str, int, int]:
    return (
        str(row["task_id"]),
        int(row["batch_index"]),
        int(row["sample_index"]),
    )


def load_rows(
    evaluated_attempts: Path, runs_root: Path
) -> tuple[list[dict], dict[tuple[str, int, int], dict]]:
    with evaluated_attempts.open(encoding="utf-8", newline="") as handle:
        evaluated = list(csv.DictReader(handle))
    raw: dict[tuple[str, int, int], dict] = {}
    for path in sorted(runs_root.glob("*/batches/batch_*.jsonl")):
        task_id = path.parents[1].name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record["task_id"] = task_id
                key = row_key(record)
                if key in raw:
                    raise RuntimeError(f"Duplicate raw attempt key: {key}")
                raw[key] = record
    evaluated_keys = [row_key(row) for row in evaluated]
    if len(set(evaluated_keys)) != len(evaluated_keys):
        raise RuntimeError("Duplicate evaluated-attempt key")
    if set(evaluated_keys) != set(raw):
        raise RuntimeError(
            "Raw/evaluated key mismatch: "
            f"evaluated={len(evaluated_keys)}, raw={len(raw)}"
        )
    return evaluated, raw


def pad_after_first_sep(
    token_ids: list[int], first_sep_index: int | None, pad_token_id: int
) -> list[int]:
    cleaned = list(token_ids)
    if first_sep_index is not None:
        cleaned[first_sep_index + 1 :] = [pad_token_id] * (
            len(cleaned) - first_sep_index - 1
        )
    return cleaned


@torch.inference_mode()
def score_both_inputs(
    evaluated: list[dict],
    raw: dict[tuple[str, int, int], dict],
    *,
    mdlm_root: Path,
    checkpoint: Path,
    batch_size: int,
    device: str,
    reviewer_retrain_root: Path,
) -> tuple[list[float], list[float], dict[str, list[float]], int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_token_id = int(tokenizer.pad_token_id)
    model = V1PeptideClassifier(mdlm_root, checkpoint).to(device).eval()
    reviewer_heads: dict[str, ClassificationHead] = {}
    for path in sorted(reviewer_retrain_root.glob("seed_*/best.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        head = ClassificationHead(
            input_dim=int(model.config.model.hidden_size)
        )
        head.load_state_dict(payload["head_state_dict"], strict=True)
        reviewer_heads[path.parent.name] = head.to(device).eval()
    if not reviewer_heads:
        raise RuntimeError(
            f"No completed reviewer-retrained heads under {reviewer_retrain_root}"
        )
    original_probabilities: list[float] = []
    sep_padded_probabilities: list[float] = []
    reviewer_retrained_logits: dict[str, list[float]] = {
        name: [] for name in reviewer_heads
    }
    for start in range(0, len(evaluated), batch_size):
        batch = evaluated[start : start + batch_size]
        original = []
        sep_padded = []
        for row in batch:
            record = raw[row_key(row)]
            token_ids = [int(value) for value in record["token_ids"]]
            first_sep = record.get("first_sep_index")
            first_sep = None if first_sep is None else int(first_sep)
            original.append(token_ids)
            sep_padded.append(
                pad_after_first_sep(token_ids, first_sep, pad_token_id)
            )
        for token_batch, destination, is_sep_padded in (
            (original, original_probabilities, False),
            (sep_padded, sep_padded_probabilities, True),
        ):
            input_ids = torch.tensor(
                token_batch, dtype=torch.long, device=device
            )
            probabilities = torch.sigmoid(model(input_ids))
            destination.extend(probabilities.float().cpu().tolist())
            if is_sep_padded:
                sigma = torch.zeros(
                    input_ids.shape[0],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                hidden = model.backbone.vocab_embed(input_ids)
                conditioning = F.silu(model.backbone.sigma_map(sigma))
                rotary = model.backbone.rotary_emb(hidden)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    for block in model.backbone.blocks:
                        hidden = block(
                            hidden, rotary, conditioning, seqlens=None
                        )
                features = hidden[:, 0, :].float()
                for name, head in reviewer_heads.items():
                    reviewer_retrained_logits[name].extend(
                        head(features).squeeze(-1).float().cpu().tolist()
                    )
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return (
        original_probabilities,
        sep_padded_probabilities,
        reviewer_retrained_logits,
        pad_token_id,
    )


def quantiles(values: list[float | int]) -> dict[str, float] | None:
    if not values:
        return None
    result = np.quantile(np.asarray(values, dtype=np.float64), [0, 0.25, 0.5, 0.75, 1])
    return {
        name: float(value)
        for name, value in zip(("min", "q25", "median", "q75", "max"), result)
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["rdkit_valid"]]
    reparseable = [row for row in valid if row["structure_reparsed"]]
    original_positive = [
        row for row in valid if row["original_classifier_positive"]
    ]
    sep_padded_positive = [
        row for row in valid if row["sep_padded_classifier_positive"]
    ]
    reviewer_retrained_positive = [
        row
        for row in valid
        if row["reviewer_retrained_ensemble_positive"]
    ]
    general_amide = [
        row
        for row in reparseable
        if row["rdkit_amide_count"] is not None
        and row["rdkit_amide_count"] > 0
    ]

    def overlap(positive_rows: list[dict]) -> dict:
        structurally_evaluable = [
            row for row in positive_rows if row["structure_reparsed"]
        ]
        with_amide = sum(
            row["rdkit_amide_count"] is not None
            and row["rdkit_amide_count"] > 0
            for row in structurally_evaluable
        )
        return {
            "positive_count": len(positive_rows),
            "structure_reparsed_count": len(structurally_evaluable),
            "structure_reparse_failed_count": (
                len(positive_rows) - len(structurally_evaluable)
            ),
            "with_general_amide_count": with_amide,
            "with_general_amide_fraction": (
                with_amide / len(structurally_evaluable)
                if structurally_evaluable
                else None
            ),
            "without_general_amide_count": (
                len(structurally_evaluable) - with_amide
            ),
            "without_general_amide_fraction": (
                (len(structurally_evaluable) - with_amide)
                / len(structurally_evaluable)
                if structurally_evaluable
                else None
            ),
        }

    return {
        "attempt_count": len(rows),
        "rdkit_valid_count": len(valid),
        "structure_reparsed_count": len(reparseable),
        "structure_reparse_failed_count": len(valid) - len(reparseable),
        "original_classifier": overlap(original_positive),
        "sep_padded_classifier": overlap(sep_padded_positive),
        "reviewer_retrained_ensemble": overlap(
            reviewer_retrained_positive
        ),
        "original_positive_to_sep_padded_negative": sum(
            row["original_classifier_positive"]
            and not row["sep_padded_classifier_positive"]
            for row in valid
        ),
        "original_negative_to_sep_padded_positive": sum(
            not row["original_classifier_positive"]
            and row["sep_padded_classifier_positive"]
            for row in valid
        ),
        "runner_smarts_amide_positive_count": sum(
            row["runner_smarts_amide"] for row in valid
        ),
        "rdkit_general_amide_positive_count": len(general_amide),
        "runner_smarts_missed_general_amide_count": sum(
            row["rdkit_amide_count"] is not None
            and row["rdkit_amide_count"] > 0
            and not row["runner_smarts_amide"]
            for row in reparseable
        ),
        "general_amide_original_classifier_positive_count": sum(
            row["original_classifier_positive"] for row in general_amide
        ),
        "general_amide_sep_padded_classifier_positive_count": sum(
            row["sep_padded_classifier_positive"] for row in general_amide
        ),
        "original_probability_quantiles": quantiles(
            [row["original_probability"] for row in valid]
        ),
        "sep_padded_probability_quantiles": quantiles(
            [row["sep_padded_probability"] for row in valid]
        ),
        "reviewer_retrained_ensemble_probability_quantiles": quantiles(
            [
                row["reviewer_retrained_ensemble_probability"]
                for row in valid
            ]
        ),
        "nonpad_tokens_after_first_sep_quantiles": quantiles(
            [
                row["nonpad_tokens_after_first_sep"]
                for row in valid
                if row["nonpad_tokens_after_first_sep"] is not None
            ]
        ),
    }


def representation_consistency(rows: list[dict], label_field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["rdkit_valid"] and row["canonical_smiles"]:
            groups[row["canonical_smiles"]].append(row)
    duplicated = {
        smiles: group for smiles, group in groups.items() if len(group) > 1
    }
    disagreeing = {
        smiles: group
        for smiles, group in duplicated.items()
        if len({bool(row[label_field]) for row in group}) > 1
    }
    return {
        "unique_canonical_molecules": len(groups),
        "canonical_molecules_with_multiple_generated_rows": len(duplicated),
        "rows_in_multirow_canonical_groups": sum(
            len(group) for group in duplicated.values()
        ),
        "canonical_molecules_with_label_disagreement": len(disagreeing),
        "rows_in_label_disagreement_groups": sum(
            len(group) for group in disagreeing.values()
        ),
        "example_disagreements": [
            {
                "canonical_smiles": smiles,
                "row_count": len(group),
                "positive_count": sum(
                    bool(row[label_field]) for row in group
                ),
            }
            for smiles, group in sorted(
                disagreeing.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:20]
        ],
    }


def main() -> None:
    args = parse_args()
    evaluated, raw = load_rows(args.evaluated_attempts, args.runs_root)
    (
        original,
        sep_padded,
        reviewer_retrained_logits,
        pad_token_id,
    ) = score_both_inputs(
        evaluated,
        raw,
        mdlm_root=args.mdlm_root,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        reviewer_retrain_root=args.reviewer_retrain_root,
    )
    reviewer_retrained_names = sorted(reviewer_retrained_logits)
    reviewer_retrained_ensemble_logits = np.mean(
        np.asarray(
            [
                reviewer_retrained_logits[name]
                for name in reviewer_retrained_names
            ],
            dtype=np.float64,
        ),
        axis=0,
    )
    reviewer_retrained_ensemble_probabilities = (
        1.0 / (1.0 + np.exp(-reviewer_retrained_ensemble_logits))
    ).tolist()
    max_saved_difference = max(
        abs(float(row["peptide_probability"]) - probability)
        for row, probability in zip(evaluated, original)
    )
    if max_saved_difference > 1e-5:
        raise RuntimeError(
            "Recomputed v1 probabilities do not match saved evaluation: "
            f"max_abs_difference={max_saved_difference}"
        )

    audited: list[dict] = []
    for (
        row,
        original_probability,
        sep_padded_probability,
        reviewer_retrained_ensemble_probability,
    ) in zip(
        evaluated,
        original,
        sep_padded,
        reviewer_retrained_ensemble_probabilities,
    ):
        record = raw[row_key(row)]
        valid = row["rdkit_valid"].lower() == "true"
        smiles = row["canonical_smiles"] if valid else ""
        raw_smiles = str(record.get("smiles") or "")
        molecule = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
        structure_smiles_source = "raw_decoded_smiles"
        if molecule is None and smiles:
            molecule = Chem.MolFromSmiles(smiles)
            structure_smiles_source = "canonical_smiles"
        if molecule is None:
            structure_smiles_source = None
        first_sep = record.get("first_sep_index")
        first_sep = None if first_sep is None else int(first_sep)
        token_ids = [int(value) for value in record["token_ids"]]
        suffix_nonpad = (
            None
            if first_sep is None
            else sum(
                value != pad_token_id for value in token_ids[first_sep + 1 :]
            )
        )
        audited.append(
            {
                "task_id": row["task_id"],
                "batch_index": int(row["batch_index"]),
                "sample_index": int(row["sample_index"]),
                "condition": row["condition"],
                "strain": row["strain"],
                "rdkit_valid": valid,
                "canonical_smiles": smiles,
                "structure_reparsed": molecule is not None,
                "structure_smiles_source": structure_smiles_source,
                "first_sep_index": first_sep,
                "nonpad_tokens_after_first_sep": suffix_nonpad,
                "original_probability": original_probability,
                "original_classifier_positive": original_probability >= 0.5,
                "sep_padded_probability": sep_padded_probability,
                "sep_padded_classifier_positive": (
                    sep_padded_probability >= 0.5
                ),
                "reviewer_retrained_ensemble_probability": (
                    reviewer_retrained_ensemble_probability
                ),
                "reviewer_retrained_ensemble_positive": (
                    reviewer_retrained_ensemble_probability >= 0.5
                ),
                "runner_smarts_amide": bool(record["has_amide_bond"]),
                "rdkit_amide_count": (
                    int(rdMolDescriptors.CalcNumAmideBonds(molecule))
                    if molecule is not None
                    else None
                ),
                "molecular_weight": (
                    float(Descriptors.MolWt(molecule))
                    if molecule is not None
                    else None
                ),
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in audited:
        grouped[row["condition"]].append(row)
    examples = sorted(
        (
            row
            for row in audited
            if row["rdkit_valid"]
            and row["sep_padded_classifier_positive"]
            and row["rdkit_amide_count"] == 0
        ),
        key=lambda row: (
            -row["sep_padded_probability"],
            row["canonical_smiles"],
            row["task_id"],
        ),
    )[:20]

    reviewer_manifests = sorted(
        (
            ROOT
            / "experiments"
            / "peptide_classifier"
            / "reviewer_retrain"
            / "runs"
        ).glob("seed_*/run_manifest.json")
    )
    reviewer_backbone_hashes = sorted(
        {
            json.loads(path.read_text(encoding="utf-8"))[
                "backbone_checkpoint_sha256"
            ]
            for path in reviewer_manifests
        }
    )
    report = {
        "schema_version": 1,
        "purpose": (
            "Audit checkpoint identity, post-SEP token leakage, and amide-bond "
            "agreement; not a new peptide identity definition."
        ),
        "inputs": {
            "evaluated_attempts": {
                "path": str(args.evaluated_attempts.resolve()),
                "size": args.evaluated_attempts.stat().st_size,
                "sha256": sha256_file(args.evaluated_attempts),
            },
            "historical_v1_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "size": args.checkpoint.stat().st_size,
                "sha256": sha256_file(args.checkpoint),
            },
            "reviewer_retrain_backbone_checkpoint_sha256_values": (
                reviewer_backbone_hashes
            ),
            "reviewer_retrained_head_names": reviewer_retrained_names,
        },
        "checkpoint_identity": {
            "same_v1_checkpoint_file_used_as_reviewer_retrain_backbone": (
                reviewer_backbone_hashes
                == [sha256_file(args.checkpoint)]
            ),
            "important_model_difference": (
                "The reviewer retrain froze the v1 backbone but initialized "
                "and trained new classification heads. Its AUROC/AUPRC do not "
                "evaluate the historical deployed v1 head used here."
            ),
            "saved_probability_max_abs_difference": max_saved_difference,
        },
        "definitions": {
            "runner_smarts": "[NX3][CX3](=O)[#6]",
            "rdkit_general_amide": "rdMolDescriptors.CalcNumAmideBonds(mol) > 0",
            "classifier_threshold": 0.5,
            "sep_padded_input": (
                "All token positions strictly after the first [SEP] are "
                "replaced by the tokenizer PAD id before clean t=0 scoring."
            ),
        },
        "all_conditions": summarize(audited),
        "by_condition": {
            condition: summarize(rows)
            for condition, rows in sorted(grouped.items())
        },
        "high_confidence_sep_padded_positive_without_general_amide_examples": [
            {
                key: row[key]
                for key in (
                    "condition",
                    "strain",
                    "canonical_smiles",
                    "sep_padded_probability",
                    "original_probability",
                    "reviewer_retrained_ensemble_probability",
                    "first_sep_index",
                    "nonpad_tokens_after_first_sep",
                    "molecular_weight",
                )
            }
            for row in examples
        ],
        "canonical_representation_consistency": {
            "historical_v1_original_full_token_label": (
                representation_consistency(
                    audited, "original_classifier_positive"
                )
            ),
            "historical_v1_sep_padded_label": representation_consistency(
                audited, "sep_padded_classifier_positive"
            ),
            "reviewer_retrained_ensemble_sep_padded_label": (
                representation_consistency(
                    audited, "reviewer_retrained_ensemble_positive"
                )
            ),
        },
        "interpretation_boundaries": [
            (
                "The runner SMARTS is narrower than RDKit's general amide "
                "descriptor, so its negative result alone is not definitive."
            ),
            (
                "Even the general amide test is only a necessary-like QC for "
                "ordinary multi-residue peptides, not a sufficient peptide "
                "definition and not a biological ground truth."
            ),
            (
                "Classifier-positive must remain an operational model label "
                "until generated candidates are evaluated with an independent "
                "structure-based peptide criterion."
            ),
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row_path = args.output_dir / "audited_valid_attempts.csv"
    valid_rows = [row for row in audited if row["rdkit_valid"]]
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(valid_rows[0]))
        writer.writeheader()
        writer.writerows(valid_rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {report_path}")
    print(f"Wrote {row_path}")


if __name__ == "__main__":
    main()
