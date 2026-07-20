#!/usr/bin/env python3
"""实时汇总 Fig. 1b reviewer 实验在本机、node001 与 node002 的进度。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


EPOCH_RE = re.compile(r"INFO: ensemble=\d+/10 epoch=(\d+)/(\d+)")
TASK_RE = re.compile(r"group_(\d+)_fold_(\d+)_member_(\d+)$")
GPU_FIELDS = (
    "index,name,temperature.gpu,temperature.memory,power.draw,power.limit,"
    "memory.used,utilization.gpu,clocks_throttle_reasons.hw_thermal_slowdown,"
    "clocks_throttle_reasons.sw_thermal_slowdown"
)
EXPECTED_TASKS = {
    *(f"1:{fold}:{member}" for fold in (2, 3, 4) for member in range(1, 10)),
    *(f"2:{fold}:{member}" for fold in (3, 4) for member in range(1, 10)),
}
EPOCHS_PER_TASK = 25


def _elapsed_seconds(value: str) -> int | None:
    """Parse tqdm 的 ``MM:SS`` 或 ``HH:MM:SS`` elapsed time。"""
    try:
        fields = [int(part) for part in value.split(":")]
    except ValueError:
        return None
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    return None


def _epoch_durations(text: str) -> list[int]:
    """Extract completed shared-training epoch durations from tqdm output."""
    durations = []
    matches = re.findall(
        r"shared MIC/auxiliary training: 100%[^\r\n]*?\[([0-9:]+)<", text
    )
    for value in matches:
        if seconds := _elapsed_seconds(value):
            durations.append(seconds)
    return durations


def _gpu_status() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return [{"error": result.stderr.strip() or "nvidia-smi failed"}]
    names = GPU_FIELDS.split(",")
    return [
        dict(zip(names, (field.strip() for field in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _fig1b_sessions() -> list[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [name for name in result.stdout.splitlines() if name.startswith("fig1b_")]


def collect(repo_root: Path) -> dict[str, object]:
    reconstruction = (
        repo_root / "results/fig1b_revision/full_ensemble_reconstruction"
    )
    tasks: dict[str, dict[str, object]] = {}
    if reconstruction.exists():
        for task_dir in reconstruction.glob("group_*_fold_*_member_*"):
            match = TASK_RE.fullmatch(task_dir.name)
            if not match:
                continue
            key = ":".join(match.groups())
            if key not in EXPECTED_TASKS:
                continue
            prediction = next(task_dir.glob("predictions/*.json"), None)
            log = task_dir / "driver.log"
            epoch = 0
            total = EPOCHS_PER_TASK
            seconds_per_epoch = None
            updated = None
            if log.exists():
                text = log.read_text(errors="replace")
                matches = EPOCH_RE.findall(text)
                if matches:
                    epoch, total = (int(value) for value in matches[-1])
                durations = _epoch_durations(text)
                if durations:
                    seconds_per_epoch = statistics.median(durations[-5:])
                updated = log.stat().st_mtime
            tasks[key] = {
                "epoch": EPOCHS_PER_TASK if prediction else epoch,
                "total": total,
                "complete": prediction is not None,
                "seconds_per_epoch": seconds_per_epoch,
                "updated": updated,
                "interrupted": (task_dir / ".interrupted").exists(),
            }

    baseline_root = (
        repo_root / "results/fig1b_revision/baselines_full_ensemble_no_rdkit"
    )
    baseline_complete = 0
    baseline_running = 0
    for group in range(3):
        for fold in range(5):
            fold_dir = baseline_root / f"group_{group}" / f"fold_{fold}"
            baseline_complete += int((fold_dir / "metrics.json").exists())
            baseline_running += int(
                (fold_dir / "baseline_driver.log").exists()
                and not (fold_dir / "metrics.json").exists()
            )

    disk = shutil.disk_usage(repo_root)
    checkpoint_files = list(reconstruction.glob("group_*_fold_*_member_*/*.pth"))
    return {
        "tasks": tasks,
        "baseline_complete": baseline_complete,
        "baseline_running": baseline_running,
        "disk_free": disk.free,
        "checkpoint_count": len(checkpoint_files),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "gpus": _gpu_status(),
        "sessions": _fig1b_sessions(),
    }


def _remote_collect(
    host: str, root: Path, script: Path, *, runtime_only: bool = False
) -> dict[str, object]:
    command = [
        "ssh",
        "-o",
        "ConnectTimeout=8",
        host,
        "python3",
        str(script),
        "--collect",
        "--repo-root",
        str(root),
    ]
    if runtime_only:
        command.append("--runtime-only")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "remote collection failed")
    return json.loads(result.stdout)


def _human_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _gpu_lines(label: str, payload: dict[str, object]) -> list[str]:
    lines = [f"{label} GPU:"]
    for gpu in payload["gpus"]:
        if "error" in gpu:
            lines.append(f"  无法读取: {gpu['error']}")
            continue
        throttle = any(
            gpu[field] == "Active"
            for field in (
                "clocks_throttle_reasons.hw_thermal_slowdown",
                "clocks_throttle_reasons.sw_thermal_slowdown",
            )
        )
        warning = "  !!热降频" if throttle else ""
        lines.append(
            f"  GPU {gpu['index']}: {gpu['utilization.gpu']:>3}% | "
            f"{gpu['temperature.gpu']:>2}C/mem {gpu['temperature.memory']:>2}C | "
            f"{gpu['power.draw']:>6}W | {gpu['memory.used']:>6} MiB | "
            f"{gpu['name']}{warning}"
        )
    return lines


def render(
    local: dict[str, object],
    remote: dict[str, object] | None,
    extra_runtimes: list[tuple[str, dict[str, object]]] | None = None,
) -> str:
    extra_runtimes = extra_runtimes or []
    hosts = [local] + ([remote] if remote else [])
    merged: dict[str, dict[str, object]] = {}
    for key in EXPECTED_TASKS:
        candidates = [host["tasks"].get(key) for host in hosts]
        candidates = [candidate for candidate in candidates if candidate]
        if not candidates:
            merged[key] = {"epoch": 0, "complete": False}
            continue
        merged[key] = max(
            candidates, key=lambda item: (bool(item["complete"]), int(item["epoch"]))
        )

    complete = sum(bool(task["complete"]) for task in merged.values())
    epoch_units = sum(int(task["epoch"]) for task in merged.values())
    total_units = len(EXPECTED_TASKS) * EPOCHS_PER_TASK

    now = datetime.now().timestamp()
    rates_by_host = []
    for host in hosts:
        active = [
            task
            for task in host["tasks"].values()
            if not task["complete"]
            and not task.get("interrupted", False)
            and task["updated"]
            and now - float(task["updated"]) < 40 * 60
        ]
        measured = [
            float(task["seconds_per_epoch"])
            for task in active
            if task["seconds_per_epoch"]
        ]
        fallback = statistics.median(measured) if measured else None
        rates_by_host.extend(
            1.0 / float(task["seconds_per_epoch"] or fallback)
            for task in active
            if task["seconds_per_epoch"] or fallback
        )
    rate = sum(rates_by_host)
    eta_hours = (total_units - epoch_units) / rate / 3600 if rate else None

    lines = [
        f"Fig. 1b reviewer 实验监控  {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Apex 缺失 ensemble: 完成 {complete}/45；epoch 进度 "
        f"{epoch_units}/{total_units} ({epoch_units / total_units:.1%})",
        (
            f"Apex 粗略 ETA: {eta_hours:.1f} 小时（按当前活跃 worker 的实测速度）"
            if eta_hours is not None
            else "Apex 粗略 ETA: 至少一个 worker 完成首个 epoch 后显示"
        ),
    ]
    baseline_complete = sum(int(host["baseline_complete"]) for host in hosts)
    baseline_running = sum(int(host["baseline_running"]) for host in hosts)
    lines.append(
        f"Chemprop no-RDKit baseline: 完成 {baseline_complete}/15 folds；"
        f"运行中 {baseline_running}（Apex 阶段后自动启动）"
    )
    if not baseline_complete:
        lines.append("整体 ETA: 当前至少等于 Apex ETA；baseline 首个 fold 完成后才能可靠外推。")

    lines.extend(["", *_gpu_lines("本机", local)])
    if remote:
        lines.extend(["", *_gpu_lines("node002", remote)])
    else:
        lines.extend(["", "node002: 状态读取失败；本机任务不受影响。"])
    for label, runtime in extra_runtimes:
        lines.extend(["", *_gpu_lines(label, runtime)])

    lines.extend(
        [
            "",
            "存储:",
            f"  本机 free {_human_bytes(int(local['disk_free']))}; "
            f"new ckpt {local['checkpoint_count']} / "
            f"{_human_bytes(int(local['checkpoint_bytes']))}",
        ]
    )
    if remote:
        lines.append(
            f"  node002 free {_human_bytes(int(remote['disk_free']))}; "
            f"new ckpt {remote['checkpoint_count']} / "
            f"{_human_bytes(int(remote['checkpoint_bytes']))}"
        )
    session_fields = [
        f"本机 {len(local['sessions'])}",
        f"node002 {len(remote['sessions']) if remote else '?'}",
        *(f"{label} {len(runtime['sessions'])}" for label, runtime in extra_runtimes),
    ]
    lines.append(f"活跃 tmux: {'; '.join(session_fields)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--node-host", default="node002")
    parser.add_argument(
        "--extra-node-host",
        action="append",
        default=["node001"],
        help="只读取 GPU/tmux 状态的额外共享文件系统节点；可重复指定",
    )
    parser.add_argument(
        "--node-root", type=Path, default=Path("/data1/tianang/Projects/Synergy_release")
    )
    parser.add_argument("--collect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runtime_only:
        print(json.dumps({"gpus": _gpu_status(), "sessions": _fig1b_sessions()}))
        return
    local = collect(args.repo_root.resolve())
    if args.collect:
        print(json.dumps(local))
        return
    remote_script = args.node_root / "scripts/reproduce" / Path(__file__).name
    try:
        remote = _remote_collect(args.node_host, args.node_root, remote_script)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
        remote = None
    extra_runtimes = []
    for host in args.extra_node_host:
        try:
            runtime = _remote_collect(
                host, args.node_root, remote_script, runtime_only=True
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue
        extra_runtimes.append((host, runtime))
    print(render(local, remote, extra_runtimes))


if __name__ == "__main__":
    main()
