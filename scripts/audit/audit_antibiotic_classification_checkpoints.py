#!/usr/bin/env python3
"""Inventory the three legacy antibiotic-classification checkpoint families.

This is deliberately metadata-only: multi-gigabyte checkpoints are never
deserialized. The report distinguishes grid completeness from evidence that a
training log actually reached its final ensemble summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


FAMILIES = {
    "strict-zero-shot": {
        "directory": "MDLM_fix_cls_sm_all_test_10_fold_ensembles",
        "pattern": re.compile(
            r"genome_text_learnable_emb_SM_outer_SM_best_AUROC_"
            r"group_(\d+)_ensemble_(\d+)\.pth"
        ),
        "expected": {(group, ensemble) for group in range(3) for ensemble in range(10)},
        "completion_pattern": re.compile(r"Ensemble AUPRC of"),
    },
    "fine-tune": {
        "directory": "MDLM_fix_cls_10_fold_ensembles",
        "pattern": re.compile(
            r"genome_text_learnable_emb_SM_outer_SM_best_AUROC_"
            r"group_(\d+)_ensemble_(\d+)_fold_(\d+)\.pth"
        ),
        "expected": {
            (group, ensemble, fold)
            for group in range(3)
            for ensemble in range(10)
            for fold in range(5)
        },
        "completion_pattern": re.compile(r"5 Fold mean Ensembled AUPRC"),
    },
    "molecule-only": {
        "directory": "MDLM_fix_cls_wo_sand_10_fold_ensembles",
        "pattern": re.compile(
            r"genome_text_learnable_emb_SM_outer_SM_best_AUROC_"
            r"group_(\d+)_ensemble_(\d+)_fold_(\d+)\.pth"
        ),
        "expected": {
            (group, ensemble, fold)
            for group in range(3)
            for ensemble in range(10)
            for fold in range(5)
        },
        "completion_pattern": re.compile(r"5 Fold mean Ensembled AUPRC"),
    },
}


def audit_family(root: Path, name: str, specification: dict) -> dict:
    directory = root / specification["directory"]
    parsed = {}
    unmatched = []
    sizes = Counter()
    for path in directory.glob("*.pth") if directory.exists() else []:
        match = specification["pattern"].fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        key = tuple(int(value) for value in match.groups())
        parsed[key] = path.name
        sizes[path.stat().st_size] += 1
    expected = specification["expected"]
    logs = []
    complete_logs = []
    for path in sorted(directory.glob("*.log")) if directory.exists() else []:
        logs.append(path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
        if specification["completion_pattern"].search(text):
            complete_logs.append(path.name)
    return {
        "family": name,
        "directory": str(
            Path(
                "Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare"
            )
            / specification["directory"]
        ),
        "expected_checkpoint_count": len(expected),
        "observed_checkpoint_count": len(parsed),
        "checkpoint_grid_complete": set(parsed) == expected,
        "missing_grid_members": [list(key) for key in sorted(expected - set(parsed))],
        "unexpected_grid_members": [
            list(key) for key in sorted(set(parsed) - expected)
        ],
        "unmatched_checkpoint_filenames": sorted(unmatched),
        "checkpoint_size_bytes": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "log_count": len(logs),
        "complete_log_count": len(complete_logs),
        "complete_logs": complete_logs,
        "evidence_interpretation": (
            "checkpoint_grid_and_completion_logs_complete"
            if set(parsed) == expected and len(complete_logs) == len(logs) and logs
            else "checkpoint_presence_does_not_establish_complete_historical_run"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = (
        repo_root / "Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare"
    )
    report = {
        "schema_version": 1,
        "audit_method": "filename_grid_stat_and_log_completion_pattern_only",
        "checkpoint_tensors_loaded": False,
        "families": {
            name: audit_family(root, name, specification)
            for name, specification in FAMILIES.items()
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
