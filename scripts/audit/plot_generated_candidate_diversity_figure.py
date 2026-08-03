#!/usr/bin/env python3
"""Plot the three-panel Supplementary Fig. C5 diversity analysis."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GROUPS = [
    ("BAA-3197", "linear", 12, 66),
    ("BAA-3197", "cyclic", 3, 3),
    ("BAA-3170", "linear", 6, 15),
    ("BAA-3170", "cyclic", 3, 3),
]
DATASET_ORDER = ["Peptide candidate pool", "Guided generation outputs"]
COLORS = {"BAA-3197": "#2C6EAA", "BAA-3170": "#56B4E9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid-csv",
        type=Path,
        default=REPO_ROOT
        / "experiments/generated_candidate_diversity/selected_peptides_24/"
        "pairwise_sequence_similarity.csv",
    )
    parser.add_argument(
        "--tanimoto-histogram-csv",
        type=Path,
        default=REPO_ROOT
        / "experiments/generated_candidate_diversity/"
        "tanimoto_histogram_plotted_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments/generated_candidate_diversity",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_pid_groups(
    rows: list[dict[str, str]],
) -> list[np.ndarray]:
    grouped: list[np.ndarray] = []
    for strain, topology, expected_peptides, expected_pairs in EXPECTED_GROUPS:
        selected = [
            row
            for row in rows
            if row["left_target_strain"] == strain
            and row["right_target_strain"] == strain
            and row["topology"] == topology
        ]
        ids = {
            apexoracle_id
            for row in selected
            for apexoracle_id in (row["left_id"], row["right_id"])
        }
        if len(selected) != expected_pairs or len(ids) != expected_peptides:
            raise RuntimeError(
                f"Unexpected {strain}/{topology} group: {len(ids)} peptides, "
                f"{len(selected)} pairs"
            )
        grouped.append(
            np.asarray([100.0 * float(row["pid"]) for row in selected], dtype=float)
        )
    if sum(values.size for values in grouped) != 87:
        raise RuntimeError("Expected 87 within-strain, within-topology PID pairs")
    return grouped


def validate_histogram(rows: list[dict[str, str]]) -> None:
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset"] == dataset]
        if len(selected) != 20:
            raise RuntimeError(f"Expected 20 histogram bins for {dataset}")
        fraction_sum = sum(float(row["fraction"]) for row in selected)
        if not np.isclose(fraction_sum, 1.0, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"Histogram fractions for {dataset} sum to {fraction_sum}")


def plot_pid_panel(axis: Any, grouped_values: list[np.ndarray]) -> None:
    positions = np.arange(len(EXPECTED_GROUPS), dtype=float)
    for index, ((strain, topology, _, _), values) in enumerate(
        zip(EXPECTED_GROUPS, grouped_values, strict=True)
    ):
        if values.size >= 5:
            violin = axis.violinplot(
                values,
                positions=[positions[index]],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.25,
            )
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[strain])
                body.set_edgecolor("#2F2F2F")
                body.set_linewidth(0.75)
                body.set_alpha(0.72)

        rng = np.random.default_rng(20260810 + index)
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        axis.scatter(
            positions[index] + jitter,
            values,
            s=18 if values.size <= 3 else 10,
            facecolor="#FFFFFF",
            edgecolor="#2F2F2F",
            linewidth=0.4,
            alpha=0.82 if values.size <= 3 else 0.58,
            zorder=3,
        )
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        axis.vlines(
            positions[index], q25, q75, color="#2F2F2F", linewidth=4.5, zorder=4
        )
        axis.scatter(
            [positions[index]],
            [median],
            s=30,
            facecolor="#FFFFFF",
            edgecolor="#2F2F2F",
            linewidth=0.85,
            zorder=5,
        )
        label_x = positions[index] if index == 3 else positions[index] + 0.14
        label_y = median + 8.0 if index == 3 else median + 2.0
        axis.text(
            label_x,
            label_y,
            f"{median:.1f}%",
            ha="center" if index == 3 else "left",
            va="bottom",
            fontsize=9.0,
            color="#2F2F2F",
        )

    labels = [
        f"{'Lin.' if topology == 'linear' else 'Cyc.'}\nn={n_pairs}"
        for _, topology, _, n_pairs in EXPECTED_GROUPS
    ]
    axis.set_title(
        "Selected peptide candidates\nWithin-target pairwise PID",
        fontsize=12.5,
        pad=8,
    )
    axis.set_ylabel("Pairwise PID (%)", fontsize=11.0)
    axis.set_xlim(-0.55, 3.55)
    axis.set_ylim(0.0, 100.0)
    axis.set_xticks(positions, labels, fontsize=9.5)
    axis.set_yticks(np.arange(0.0, 101.0, 20.0))
    axis.tick_params(axis="y", labelsize=10.0)
    axis.axvline(1.5, color="#C7C7C7", linewidth=0.7, linestyle="--")
    for x_position, strain in (
        (0.5, r"$\it{P.\ aeruginosa}$" + "\nPA5257"),
        (2.5, r"$\it{E.\ coli}$" + "\nAR-0349"),
    ):
        axis.text(
            x_position,
            90.0,
            strain,
            ha="center",
            va="center",
            fontsize=10.0,
            color="#2F2F2F",
        )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.0,
        1.03,
        "a",
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=14,
    )


def plot_tanimoto_panel(
    axis: Any,
    hist_rows: list[dict[str, str]],
    *,
    dataset: str,
    title: str,
    color: str,
    panel: str,
    show_ylabel: bool,
) -> None:
    selected = [row for row in hist_rows if row["dataset"] == dataset]
    centers = np.asarray([float(row["bin_center"]) for row in selected])
    fractions = np.asarray([float(row["fraction"]) for row in selected])
    widths = np.asarray(
        [float(row["bin_right"]) - float(row["bin_left"]) for row in selected]
    )
    axis.bar(
        centers,
        fractions,
        width=widths * 0.92,
        color=color,
        edgecolor="#333333",
        linewidth=0.45,
    )
    axis.set_title(title, fontsize=12.5, pad=8)
    axis.set_xlabel("Pairwise Tanimoto similarity", fontsize=11.0)
    if show_ylabel:
        axis.set_ylabel("Fraction of molecule pairs", fontsize=11.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 0.195)
    axis.set_xticks(np.arange(0.0, 1.01, 0.2))
    axis.set_yticks(np.arange(0.0, 0.176, 0.025))
    axis.tick_params(axis="both", labelsize=10.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        -0.13,
        1.03,
        panel,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=14,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pid_rows = read_csv(args.pid_csv)
    hist_rows = read_csv(args.tanimoto_histogram_csv)
    grouped_values = select_pid_groups(pid_rows)
    validate_histogram(hist_rows)

    fig = plt.figure(figsize=(10.8, 3.65))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.42, 1.0, 1.0), wspace=0.32)
    pid_axis = fig.add_subplot(grid[0, 0])
    candidate_axis = fig.add_subplot(grid[0, 1])
    generation_axis = fig.add_subplot(grid[0, 2], sharey=candidate_axis)
    plot_pid_panel(pid_axis, grouped_values)
    plot_tanimoto_panel(
        candidate_axis,
        hist_rows,
        dataset="Peptide candidate pool",
        title="Peptide candidate pool\n73 candidates",
        color="#2C6EAA",
        panel="b",
        show_ylabel=True,
    )
    plot_tanimoto_panel(
        generation_axis,
        hist_rows,
        dataset="Guided generation outputs",
        title="Guided generation outputs\n84,226 outputs",
        color="#56B4E9",
        panel="c",
        show_ylabel=False,
    )
    generation_axis.tick_params(labelleft=False)

    outputs: list[Path] = []
    stem = "generated_candidate_diversity_three_panel"
    for suffix in ("pdf", "svg", "png"):
        output = args.output_dir / f"{stem}.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)

    manifest_path = args.output_dir / f"{stem}_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (args.pid_csv, args.tanimoto_histogram_csv)
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
        "panel_order": {
            "a": "24 selected peptide PID, within target strain and topology",
            "b": "73-candidate-pool Morgan/Tanimoto distribution",
            "c": "84,226-output Morgan/Tanimoto distribution",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
