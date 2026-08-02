#!/usr/bin/env python
"""Relay completed classifier training runs into test and final bootstrap.

This script does not alter the scientific protocol. It monitors the tasks
included by ``task_manifest.json``, starts evaluation only after a training
task has cleanly completed, synchronizes node-local results, and runs the final
multi-seed summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def remote_command(node: str, command: list[str]) -> list[str]:
    return ["ssh", node, shlex.join(command)]


def read_json(path: Path, *, node: str | None = None) -> dict | None:
    if node is None:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    result = run(
        ["ssh", node, f"test -f {shlex.quote(str(path))} && cat {shlex.quote(str(path))}"],
        check=False,
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)


def file_exists(path: Path, *, node: str | None = None) -> bool:
    if node is None:
        return path.exists()
    return (
        run(
            ["ssh", node, f"test -f {shlex.quote(str(path))}"],
            check=False,
        ).returncode
        == 0
    )


def tmux_session_exists(server: str, session: str, *, node: str | None) -> bool:
    command = ["tmux", "-L", server, "has-session", "-t", session]
    if node is not None:
        command = remote_command(node, command)
    return run(command, check=False).returncode == 0


def training_complete(
    output_dir: Path,
    *,
    node: str | None,
    final_epoch: int,
    steps_per_epoch: int,
) -> bool:
    history = read_json(output_dir / "history.json", node=node)
    if history is None or not history.get("evaluations"):
        return False
    last = history["evaluations"][-1]
    return (
        int(last["epoch"]) == final_epoch
        and int(last["step_in_epoch"]) == steps_per_epoch
        and file_exists(output_dir / "best.pt", node=node)
        and file_exists(output_dir / "last.pt", node=node)
    )


def evaluation_complete(output_dir: Path, *, node: str | None) -> bool:
    return file_exists(
        output_dir / "test_metrics.json", node=node
    ) and file_exists(output_dir / "test_predictions.npz", node=node)


def launch_evaluation(
    task: dict,
    *,
    node: str | None,
    project_root: Path,
    tmux_server: str,
) -> None:
    seed = int(task["seed"])
    output_dir = Path(task["output_dir"])
    if node is None:
        python = "/home/tianang/anaconda3/envs/mdlm/bin/python"
        dataset = project_root / "DataPrepare/MDLM/Data/hf_pep_SM_cls_1024"
        split_dir = project_root / "experiments/peptide_classifier/reviewer_retrain"
        producer = Path("/data2/tianang/projects/mdlm")
        checkpoint = producer / (
            "cls-guide-pad-no-mask-checkpoints/"
            "epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt"
        )
    else:
        python = "/local/tianang/peptide_classifier/mdlm-env/bin/python"
        dataset = Path("/local/tianang/peptide_classifier/hf_pep_SM_cls_1024")
        split_dir = Path("/local/tianang/peptide_classifier/reviewer_retrain")
        producer = Path("/data1/tianang/Projects/mdlm")
        checkpoint = producer / (
            "cls-guide-pad-no-mask-checkpoints/"
            "epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt"
        )
    visible = ",".join(str(gpu) for gpu in task["gpus"])
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={visible}",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "PYTHONPATH=src",
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "scripts/reproduce/run_peptide_classifier_reviewer.py",
        "--dataset-dir",
        str(dataset),
        "--split-dir",
        str(split_dir),
        "--producer-root",
        str(producer),
        "--v1-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--eval-batch-size",
        "225",
        "--num-workers",
        "8",
        "--noise-replicates",
        "10",
        "--evaluate-only",
        "--checkpoint",
        str(output_dir / "best.pt"),
    ]
    shell_command = (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.join(command)} >> "
        f"{shlex.quote(str(output_dir / 'evaluation.log'))} 2>&1"
    )
    tmux = [
        "tmux",
        "-L",
        tmux_server,
        "new-session",
        "-d",
        "-s",
        f"eval-seed{seed}",
        shell_command,
    ]
    if node is not None:
        tmux = remote_command(node, tmux)
    result = run(tmux, check=False)
    if result.returncode:
        raise RuntimeError(
            f"Failed to launch evaluation for seed {seed}: {result.stdout}"
        )


def atomic_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/peptide_classifier/reviewer_retrain/task_manifest.json",
    )
    parser.add_argument("--node", default="node001")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--tmux-server", default="peptide-classifier")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.task_manifest.read_text())
    included_tasks = [
        task for task in manifest["tasks"] if task.get("include_in_summary", True)
    ]
    if not included_tasks:
        raise RuntimeError("Task manifest contains no included reviewer runs")
    split_manifest = json.loads(
        (
            PROJECT_ROOT
            / "experiments/peptide_classifier/reviewer_retrain/split_manifest.json"
        ).read_text()
    )
    steps_per_epoch = math.floor(
        math.ceil(int(split_manifest["split_counts"]["train"]) / 4)
        / int(included_tasks[0]["batch_size_per_gpu"])
    )
    final_epoch = int(manifest["max_epochs"]) - 1
    status_path = args.task_manifest.parent / "pipeline_status.json"
    launched: set[int] = set()
    started = time.time()

    while True:
        status: dict[str, object] = {
            "elapsed_seconds": time.time() - started,
            "expected_steps_per_epoch": steps_per_epoch,
            "tasks": {},
        }
        all_evaluated = True
        for task in manifest["tasks"]:
            seed = int(task["seed"])
            if not task.get("include_in_summary", True):
                status["tasks"][f"seed_{seed}"] = {
                    "excluded": True,
                    "exclusion_reason": task.get("exclusion_reason"),
                    "host": task["host"],
                }
                continue
            node = None if seed == 0 else args.node
            output_dir = Path(task["output_dir"])
            trained = training_complete(
                output_dir,
                node=node,
                final_epoch=final_epoch,
                steps_per_epoch=steps_per_epoch,
            )
            evaluated = evaluation_complete(output_dir, node=node)
            train_session = tmux_session_exists(
                args.tmux_server, f"seed{seed}", node=node
            )
            eval_session = tmux_session_exists(
                args.tmux_server, f"eval-seed{seed}", node=node
            )
            status["tasks"][f"seed_{seed}"] = {
                "evaluation_complete": evaluated,
                "evaluation_session_running": eval_session,
                "host": task["host"],
                "training_complete": trained,
                "training_session_running": train_session,
            }
            if not trained and not train_session:
                atomic_status(status_path, status)
                raise RuntimeError(f"Seed {seed} training stopped before completion")
            if trained and not train_session and not evaluated and not eval_session:
                launch_evaluation(
                    task,
                    node=node,
                    project_root=(
                        PROJECT_ROOT
                        if node is None
                        else Path("/data1/tianang/Projects/Synergy")
                    ),
                    tmux_server=args.tmux_server,
                )
                launched.add(seed)
                eval_session = True
            if seed in launched and not evaluated and not eval_session:
                atomic_status(status_path, status)
                raise RuntimeError(f"Seed {seed} evaluation stopped before completion")
            all_evaluated &= evaluated
        atomic_status(status_path, status)
        if all_evaluated:
            break
        time.sleep(args.poll_seconds)

    local_runs = args.task_manifest.parent / "runs"
    for task in included_tasks:
        seed = int(task["seed"])
        if seed == 0:
            continue
        run(
            [
                "rsync",
                "-a",
                f"{args.node}:/local/tianang/peptide_classifier/runs/seed_{seed}/",
                str(local_runs / f"seed_{seed}") + "/",
            ]
        )
    summary_dir = args.task_manifest.parent / "summary"
    summary_command = [
        "/home/tianang/anaconda3/bin/python",
        str(
            PROJECT_ROOT
            / "scripts/reproduce/summarize_peptide_classifier_reviewer.py"
        ),
    ]
    for task in included_tasks:
        seed = int(task["seed"])
        summary_command.extend(["--run-dir", str(local_runs / f"seed_{seed}")])
    summary_command.extend(
        [
            "--output-dir",
            str(summary_dir),
            "--bootstrap",
            str(args.bootstrap),
            "--jobs",
            str(args.jobs),
        ]
    )
    result = run(summary_command)
    (args.task_manifest.parent / "summary.log").write_text(result.stdout)
    status["summary_complete"] = True
    status["summary_path"] = str(summary_dir / "summary.json")
    atomic_status(status_path, status)


if __name__ == "__main__":
    main()
