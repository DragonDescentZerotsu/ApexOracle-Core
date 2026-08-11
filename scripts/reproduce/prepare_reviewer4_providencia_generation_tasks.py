#!/usr/bin/env python
"""Freeze the historical length grid for ATCC 29914 guided generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "reviewer4_unseen_targets"
    / "providencia_stuartii_atcc_29914"
    / "generation_task_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-batches", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seed or [1]
    lengths = list(range(256, 417, 4))
    tasks = []
    for seed in seeds:
        for length in lengths:
            task_id = f"atcc29914_len{length}_seed{seed}"
            tasks.append(
                {
                    "task_id": task_id,
                    "condition": "paper_guided",
                    "strain": "29914",
                    "target_length": length,
                    "seed": seed,
                    "host": "unassigned",
                    "t_on": 0.55,
                    "t_off": 0.45,
                    "gamma_peptide": 15.0,
                    "gamma_mic": 15.0,
                    "target_mic": 1.0,
                    "steps": 256,
                    "eta": 0.02,
                    "alpha_on": 0.5,
                    "batch_size": args.batch_size,
                    "num_batches": args.num_batches,
                    "attempted_samples": args.batch_size * args.num_batches,
                }
            )
    manifest = {
        "schema_version": 1,
        "target": "Providencia stuartii ATCC 29914",
        "sampler_key": "29914",
        "length_grid": lengths,
        "seeds": seeds,
        "attempted_samples_total": sum(t["attempted_samples"] for t in tasks),
        "runner": "scripts/reproduce/run_remasking_schedule_reviewer.py",
        "evaluator": "scripts/reproduce/evaluate_remasking_schedule_reviewer.py",
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
