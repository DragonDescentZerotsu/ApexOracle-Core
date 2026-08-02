#!/usr/bin/env python
"""Run host-local remasking tasks concurrently, one sequential queue per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--host", choices=["local", "node002"], required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--synergy-root", type=Path, required=True)
    parser.add_argument("--mdlm-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--guidance-regressor-checkpoint", type=Path)
    parser.add_argument("--peptide-classifier-checkpoint", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    tasks = [task for task in manifest["tasks"] if task["host"] == args.host]
    if not tasks:
        raise RuntimeError(f"No tasks assigned to {args.host}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / f"queue_status_{args.host}.json"
    state = {
        "host": args.host,
        "started_unix": time.time(),
        "tasks": {
            task["task_id"]: {"status": "pending", "gpu": task["gpu"]}
            for task in tasks
        },
    }
    lock = threading.Lock()
    atomic_json(status_path, state)

    queues: dict[int, list[dict]] = {}
    for task in tasks:
        queues.setdefault(int(task["gpu"]), []).append(task)

    def update(task_id: str, **updates) -> None:
        with lock:
            state["tasks"][task_id].update(updates)
            atomic_json(status_path, state)

    def run_queue(gpu: int, gpu_tasks: list[dict]) -> None:
        for task in gpu_tasks:
            task_id = task["task_id"]
            task_dir = args.output_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            if (task_dir / "completed.json").exists():
                update(task_id, status="skipped_complete")
                continue
            command = [
                str(args.python),
                "-u",
                str(args.runner),
                "--producer-root",
                str(args.producer_root),
                "--synergy-root",
                str(args.synergy_root),
                "--mdlm-root",
                str(args.mdlm_root),
                "--output-dir",
                str(task_dir),
                "--strain",
                task["strain"],
                "--target-length",
                str(task["target_length"]),
                "--seed",
                str(task["seed"]),
                "--t-on",
                str(task["t_on"]),
                "--t-off",
                str(task["t_off"]),
                "--gamma-peptide",
                str(task["gamma_peptide"]),
                "--num-batches",
                str(task["num_batches"]),
                "--batch-size",
                str(task["batch_size"]),
                "--resume",
            ]
            optional_paths = [
                ("--diffusion-checkpoint", args.diffusion_checkpoint),
                (
                    "--guidance-regressor-checkpoint",
                    args.guidance_regressor_checkpoint,
                ),
                (
                    "--peptide-classifier-checkpoint",
                    args.peptide_classifier_checkpoint,
                ),
            ]
            for option, path in optional_paths:
                if path is not None:
                    command.extend([option, str(path)])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["TOKENIZERS_PARALLELISM"] = "false"
            environment["PYTHONHASHSEED"] = str(task["seed"])
            update(
                task_id,
                status="running",
                started_unix=time.time(),
                command=command,
            )
            with (task_dir / "launcher.log").open(
                "a", encoding="utf-8"
            ) as log_handle:
                result = subprocess.run(
                    command,
                    cwd=args.synergy_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            update(
                task_id,
                status="completed" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                ended_unix=time.time(),
            )

    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [
            executor.submit(run_queue, gpu, gpu_tasks)
            for gpu, gpu_tasks in sorted(queues.items())
        ]
        for future in futures:
            future.result()

    state["ended_unix"] = time.time()
    atomic_json(status_path, state)
    failed = [
        task_id
        for task_id, task_state in state["tasks"].items()
        if task_state["status"] == "failed"
    ]
    if failed:
        raise RuntimeError(f"Failed tasks: {failed}")


if __name__ == "__main__":
    main()
