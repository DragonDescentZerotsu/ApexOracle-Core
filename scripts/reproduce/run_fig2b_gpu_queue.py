#!/usr/bin/env python3
"""Run a recorded sequence of GPU jobs after an optional sentinel appears."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, nargs="+", default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_path = args.queue.resolve()
    tasks = json.loads(queue_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("queue JSON must be a list of task objects")

    if args.wait_for is not None:
        sentinels = [path.resolve() for path in args.wait_for]
        print(f"[gpu-queue] GPU {args.gpu} waiting for {sentinels}", flush=True)
        while not all(path.exists() for path in sentinels):
            time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    for position, task in enumerate(tasks, start=1):
        name = str(task["name"])
        command = [str(value) for value in task["command"]]
        cwd = Path(task.get("cwd", queue_path.parent)).resolve()
        expected = task.get("expected_output")
        if expected is not None and Path(expected).resolve().exists():
            print(f"[gpu-queue] GPU {args.gpu} skip {name}: output exists", flush=True)
            continue
        print(
            f"[gpu-queue] GPU {args.gpu} task {position}/{len(tasks)} start {name}: "
            + " ".join(command),
            flush=True,
        )
        started = time.time()
        completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
        elapsed = time.time() - started
        print(
            f"[gpu-queue] GPU {args.gpu} task {name} exit={completed.returncode} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        if completed.returncode != 0:
            return completed.returncode
        if expected is not None and not Path(expected).resolve().exists():
            raise FileNotFoundError(f"task {name} did not create expected output {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
