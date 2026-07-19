#!/usr/bin/env python3
"""Audit the complete 3-fold x 7-member strain-wise checkpoint family."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.models.strainwise_checkpoint import inspect_checkpoint_contract  # noqa: E402

FILE_PATTERN = re.compile(
    r"genome_text_learnable_emb_Strain_wise_best_R2_group_(\d+)_ensemble_(\d+)\.pth"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT
        / "Checkpoints"
        / "genome_text_learnable_emb"
        / "strain_wise_w_SM_b_attn"
        / "MDLM_MTR_fix_7_fold_ensembles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "fig2c_strainwise"
        / "checkpoint_family_audit.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = []
    for path in sorted(args.checkpoint_dir.glob("*.pth")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), int(match.group(2)), path))
    observed = {(group, ensemble) for group, ensemble, _ in files}
    expected = {(group, ensemble) for group in range(3) for ensemble in range(7)}
    if observed != expected:
        raise SystemExit(
            f"checkpoint grid mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )

    records = []
    canonical_contract = None
    for group, ensemble, path in files:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        contract = inspect_checkpoint_contract(checkpoint)
        structural_contract = {
            key: value
            for key, value in contract.items()
            if key not in {"checkpoint_keys", "optional_payloads"}
        }
        if canonical_contract is None:
            canonical_contract = structural_contract
        elif structural_contract != canonical_contract:
            raise SystemExit(
                f"contract mismatch in {path.name}: {structural_contract} != {canonical_contract}"
            )
        records.append(
            {
                "group": group,
                "ensemble": ensemble,
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "archived_best_r2": float(checkpoint["R2"]),
                "contract": contract,
                "checkpoint_variant": (
                    "with_saved_frozen_mdlm_backbone"
                    if contract["optional_payloads"]
                    else "fusion_and_heads_only"
                ),
            }
        )
        del checkpoint
        gc.collect()

    result = {
        "schema_version": 1,
        "status": "complete_grid_common_consumed_contract_verified_two_top_level_variants",
        "checkpoint_count": len(records),
        "expected_groups": 3,
        "expected_ensembles_per_group": 7,
        "common_contract": canonical_contract,
        "sha256_status": "representative_only_remaining_files_pending",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
