#!/usr/bin/env python
"""Freeze the minimal Reviewer 1/2 remasking generation task matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = [
    {"name": "earlier", "t_on": 0.75, "t_off": 0.65, "gamma_peptide": 15.0},
    {"name": "current", "t_on": 0.55, "t_off": 0.45, "gamma_peptide": 15.0},
    {"name": "later", "t_on": 0.35, "t_off": 0.25, "gamma_peptide": 15.0},
    {"name": "narrower", "t_on": 0.525, "t_off": 0.475, "gamma_peptide": 15.0},
    {"name": "wider", "t_on": 0.55, "t_off": 0.25, "gamma_peptide": 15.0},
    {
        "name": "no_peptide_correction",
        "t_on": 0.55,
        "t_off": 0.45,
        "gamma_peptide": 0.0,
    },
]
STRAINS = [
    {"name": "BAA-3170", "target_length": 368},
    {"name": "BAA-3197", "target_length": 232},
]
SEEDS = [20260728, 20260729, 20260730]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--local-gpus", default="0,3,1,2")
    parser.add_argument("--node002-gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    if args.samples_per_task % args.batch_size:
        raise ValueError("samples-per-task must be divisible by batch-size")
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)

    local_gpus = parse_gpus(args.local_gpus)
    node002_gpus = parse_gpus(args.node002_gpus)
    if not local_gpus or not node002_gpus:
        raise ValueError("Both local and node002 GPU lists are required")

    tasks = []
    local_task_index = 0
    node002_task_index = 0
    for condition_index, condition in enumerate(CONDITIONS):
        # Exactly one seed for each strain is local in every condition. The
        # selected seed rotates across conditions, so condition is not
        # confounded with host while all four local GPUs receive three tasks.
        local_seed_index = condition_index % len(SEEDS)
        for strain in STRAINS:
            for seed_index, seed in enumerate(SEEDS):
                if seed_index == local_seed_index:
                    host = "local"
                    gpu = local_gpus[local_task_index % len(local_gpus)]
                    local_task_index += 1
                else:
                    host = "node002"
                    gpu = node002_gpus[
                        node002_task_index % len(node002_gpus)
                    ]
                    node002_task_index += 1
                task_id = (
                    f"{condition['name']}__{strain['name'].lower()}__seed{seed}"
                )
                tasks.append(
                    {
                        "task_id": task_id,
                        "condition": condition["name"],
                        "t_on": condition["t_on"],
                        "t_off": condition["t_off"],
                        "gamma_peptide": condition["gamma_peptide"],
                        "strain": strain["name"],
                        "target_length": strain["target_length"],
                        "seed": seed,
                        "host": host,
                        "gpu": gpu,
                        "batch_size": args.batch_size,
                        "num_batches": args.samples_per_task // args.batch_size,
                        "attempted_samples": args.samples_per_task,
                    }
                )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "steps": 256,
            "eta": 0.02,
            "alpha_on": 0.5,
            "gamma_mic": 15.0,
            "target_mic": 1.0,
            "conditions": CONDITIONS,
            "strains": STRAINS,
            "seeds": SEEDS,
            "samples_per_task": args.samples_per_task,
            "batch_size": args.batch_size,
            "total_tasks": len(tasks),
            "total_attempted_samples": sum(
                task["attempted_samples"] for task in tasks
            ),
        },
        "allocation": {
            "local_gpus": local_gpus,
            "node002_gpus": node002_gpus,
            "method": (
                "two_local_tasks_per_condition_one_per_strain_with_rotating_"
                "seed_then_round_robin_within_host"
            ),
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": len(tasks),
                "attempts": payload["protocol"]["total_attempted_samples"],
                "batches_per_task": math.ceil(
                    args.samples_per_task / args.batch_size
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
