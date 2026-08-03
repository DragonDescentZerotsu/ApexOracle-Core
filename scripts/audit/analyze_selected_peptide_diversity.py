#!/usr/bin/env python3
"""Quantify pairwise sequence and structural diversity among 24 final peptides."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.evaluation.generated_candidate_diversity import (  # noqa: E402
    FINGERPRINT_BITS,
    FINGERPRINT_INCLUDE_CHIRALITY,
    FINGERPRINT_RADIUS,
    SEQUENCE_SCORING_SCHEME,
    all_pairwise_tanimoto,
    best_topology_aware_sequence_alignment,
    morgan_fingerprints,
    sha256_file,
    summarize_similarities,
)


EXPECTED_PEPTIDES = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=REPO_ROOT
        / "experiments/generated_candidate_diversity/canonical_candidates/"
        "final_peptide_mapping.csv",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=REPO_ROOT
        / "experiments/generated_candidate_diversity/canonical_candidates/"
        "candidates_73.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "experiments/generated_candidate_diversity/selected_peptides_24",
    )
    return parser.parse_args()


def numeric_id(apexoracle_id: str) -> int:
    return int(apexoracle_id.rsplit("-", 1)[1])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected)
        writer.writeheader()
        writer.writerows(rows)


def load_final_peptides(
    mapping_path: Path, candidate_path: Path
) -> list[dict[str, Any]]:
    mapping_rows = read_csv(mapping_path)
    candidate_rows = read_csv(candidate_path)
    if len(mapping_rows) != EXPECTED_PEPTIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_PEPTIDES} final peptides, found {len(mapping_rows)}"
        )
    candidates_by_row = {
        int(row["candidate_row"]): row for row in candidate_rows
    }
    peptides: list[dict[str, Any]] = []
    for row in sorted(mapping_rows, key=lambda item: numeric_id(item["apexoracle_id"])):
        candidate_indices = [
            int(value) for value in row["matching_candidate_rows"].split(";")
        ]
        matched = [candidates_by_row[index] for index in candidate_indices]
        smiles = {
            item["corrected_canonical_isomeric_smiles"] for item in matched
        }
        strains = {item["strain"] for item in matched}
        if len(smiles) != 1:
            raise RuntimeError(
                f"{row['apexoracle_id']} maps to inconsistent canonical structures"
            )
        if len(strains) != 1:
            raise RuntimeError(
                f"{row['apexoracle_id']} maps to inconsistent target strains"
            )
        peptides.append(
            {
                "apexoracle_id": row["apexoracle_id"],
                "sequence": row["final_sequence"],
                "is_cyclic": row["is_cyclic"].lower() == "true",
                "target_strain": next(iter(strains)),
                "canonical_isomeric_smiles": next(iter(smiles)),
                "candidate_rows": row["matching_candidate_rows"],
            }
        )
    expected_ids = [f"ApexOracle-{index}" for index in range(1, 25)]
    observed_ids = [row["apexoracle_id"] for row in peptides]
    if observed_ids != expected_ids:
        raise RuntimeError(f"Unexpected final peptide IDs: {observed_ids}")
    return peptides


def build_sequence_pairs(peptides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(peptides[:-1]):
        for right in peptides[left_index + 1 :]:
            alignment = best_topology_aware_sequence_alignment(
                left["sequence"],
                right["sequence"],
                left_is_cyclic=left["is_cyclic"],
                right_is_cyclic=right["is_cyclic"],
            )
            if alignment is None:
                continue
            metrics = alignment.metrics
            rows.append(
                {
                    "left_id": left["apexoracle_id"],
                    "right_id": right["apexoracle_id"],
                    "left_target_strain": left["target_strain"],
                    "right_target_strain": right["target_strain"],
                    "topology": "cyclic" if left["is_cyclic"] else "linear",
                    "left_sequence": left["sequence"],
                    "right_sequence": right["sequence"],
                    "left_rotation": alignment.left_rotation,
                    "right_rotation": alignment.right_rotation,
                    "orientation_swapped": alignment.orientation_swapped,
                    "scoring_scheme": SEQUENCE_SCORING_SCHEME,
                    "matches": metrics.matches,
                    "aligned_positions_including_gaps": (
                        metrics.aligned_positions_including_gaps
                    ),
                    "pid": metrics.pid,
                    "max_len_identity": metrics.max_len_identity,
                    "gapped_left": metrics.gapped_target,
                    "gapped_right": metrics.gapped_query,
                }
            )
    return rows


def build_tanimoto_pairs(
    peptides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fingerprints = morgan_fingerprints(
        [row["canonical_isomeric_smiles"] for row in peptides]
    )
    left, right, values = all_pairwise_tanimoto(fingerprints)
    rows: list[dict[str, Any]] = []
    for left_index, right_index, value in zip(left, right, values, strict=True):
        left_row = peptides[int(left_index)]
        right_row = peptides[int(right_index)]
        rows.append(
            {
                "left_id": left_row["apexoracle_id"],
                "right_id": right_row["apexoracle_id"],
                "left_target_strain": left_row["target_strain"],
                "right_target_strain": right_row["target_strain"],
                "left_topology": "cyclic" if left_row["is_cyclic"] else "linear",
                "right_topology": "cyclic" if right_row["is_cyclic"] else "linear",
                "tanimoto": float(value),
            }
        )
    return rows


def write_matrix(
    path: Path,
    peptides: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    metric: str,
    *,
    blank_cross_topology: bool,
) -> None:
    ids = [row["apexoracle_id"] for row in peptides]
    values: dict[frozenset[str], Any] = {
        frozenset((row["left_id"], row["right_id"])): row[metric]
        for row in pair_rows
    }
    output: list[dict[str, Any]] = []
    topology = {row["apexoracle_id"]: row["is_cyclic"] for row in peptides}
    for left_id in ids:
        matrix_row: dict[str, Any] = {"apexoracle_id": left_id}
        for right_id in ids:
            if left_id == right_id:
                matrix_row[right_id] = 1.0
            elif blank_cross_topology and topology[left_id] != topology[right_id]:
                matrix_row[right_id] = ""
            else:
                matrix_row[right_id] = values[frozenset((left_id, right_id))]
        output.append(matrix_row)
    write_csv(path, output, ["apexoracle_id", *ids])


def nearest_neighbors(
    peptides: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    tanimoto_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for peptide in peptides:
        apexoracle_id = peptide["apexoracle_id"]
        selected_sequence = [
            row
            for row in sequence_rows
            if apexoracle_id in (row["left_id"], row["right_id"])
        ]
        selected_tanimoto = [
            row
            for row in tanimoto_rows
            if apexoracle_id in (row["left_id"], row["right_id"])
        ]
        best_pid = max(float(row["pid"]) for row in selected_sequence)
        best_tanimoto = max(float(row["tanimoto"]) for row in selected_tanimoto)

        def partners(rows: list[dict[str, Any]], metric: str, best: float) -> str:
            ids = []
            for row in rows:
                if np.isclose(float(row[metric]), best, rtol=0.0, atol=1e-12):
                    ids.append(
                        row["right_id"]
                        if row["left_id"] == apexoracle_id
                        else row["left_id"]
                    )
            return ";".join(sorted(ids, key=numeric_id))

        output.append(
            {
                "apexoracle_id": apexoracle_id,
                "target_strain": peptide["target_strain"],
                "topology": "cyclic" if peptide["is_cyclic"] else "linear",
                "sequence_nearest_ids": partners(
                    selected_sequence, "pid", best_pid
                ),
                "max_pairwise_pid": best_pid,
                "structural_nearest_ids": partners(
                    selected_tanimoto, "tanimoto", best_tanimoto
                ),
                "max_pairwise_tanimoto": best_tanimoto,
            }
        )
    return output


def plot_pairwise_violin(
    sequence_values: np.ndarray,
    tanimoto_values: np.ndarray,
    output_dir: Path,
    *,
    sequence_top_label: str,
    tanimoto_top_label: str,
) -> list[Path]:
    """Plot the two frozen pairwise distributions on a shared 0--1 scale."""

    datasets = [
        (
            sequence_values,
            "Sequence identity (PID)",
            "168 topology-matched pairs",
            "#2C6EAA",
            20260803,
            sequence_top_label,
        ),
        (
            tanimoto_values,
            "Morgan Tanimoto similarity",
            "276 structural pairs",
            "#56B4E9",
            20260804,
            tanimoto_top_label,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for axis, (values, title, subtitle, color, seed, top_label), panel in zip(
        axes, datasets, ("a", "b"), strict=True
    ):
        violin = axis.violinplot(
            values,
            positions=[0.0],
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#2F2F2F")
            body.set_linewidth(0.8)
            body.set_alpha(0.72)

        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        axis.scatter(
            jitter,
            values,
            s=9,
            facecolor="#FFFFFF",
            edgecolor="#2F2F2F",
            linewidth=0.35,
            alpha=0.55,
            zorder=3,
        )
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        axis.vlines(0.0, q25, q75, color="#2F2F2F", linewidth=5.0, zorder=4)
        axis.scatter(
            [0.0],
            [median],
            s=34,
            facecolor="#FFFFFF",
            edgecolor="#2F2F2F",
            linewidth=0.9,
            zorder=5,
        )
        axis.text(
            0.03,
            0.96,
            f"Median {median:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#2F2F2F",
        )
        maximum = float(np.max(values))
        axis.annotate(
            top_label,
            xy=(0.0, maximum),
            xytext=(0.17, min(0.93, maximum + 0.055)),
            fontsize=7.5,
            color="#2F2F2F",
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "linewidth": 0.7,
            },
        )
        axis.set_title(f"{title}\n{subtitle}", fontsize=10, pad=8)
        axis.set_xlim(-0.58, 0.58)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([])
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.text(
            -0.13,
            1.03,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
        )
    axes[0].set_ylabel("Pairwise similarity")
    fig.tight_layout(w_pad=2.3)
    outputs: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        output = output_dir / f"selected_peptide_pairwise_similarity_violin.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)
    caption_path = output_dir / "selected_peptide_pairwise_similarity_violin_caption.md"
    caption_path.write_text(
        "**Pairwise similarity distributions among the 24 synthesized peptide "
        "candidates.** **a,** Sequence percent identity (PID) for 168 "
        "topology-matched pairs. Linear peptides were compared with linear "
        "peptides, and cyclic peptides with cyclic peptides using exhaustive "
        "circular shifts; the 108 linear--cyclic pairs were not assigned a PID. "
        "**b,** Morgan-fingerprint Tanimoto similarity for all 276 unordered "
        "peptide pairs. Dots represent individual pairs, thick vertical bars "
        "show the interquartile range, and white circles mark the median.\n",
        encoding="utf-8",
    )
    outputs.append(caption_path)
    return outputs


def summarize_within_strain_topology(
    peptides: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    tanimoto_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize pairs without mixing target strain or peptide topology."""

    group_order = [
        ("BAA-3197", "linear"),
        ("BAA-3197", "cyclic"),
        ("BAA-3170", "linear"),
        ("BAA-3170", "cyclic"),
    ]
    peptide_counts = {
        (strain, topology): sum(
            row["target_strain"] == strain
            and ("cyclic" if row["is_cyclic"] else "linear") == topology
            for row in peptides
        )
        for strain, topology in group_order
    }
    output: list[dict[str, Any]] = []
    for metric, rows, value_key in (
        ("sequence_pid", sequence_rows, "pid"),
        ("morgan_tanimoto", tanimoto_rows, "tanimoto"),
    ):
        for strain, topology in group_order:
            selected = []
            for row in rows:
                if (
                    row["left_target_strain"] != strain
                    or row["right_target_strain"] != strain
                ):
                    continue
                if metric == "sequence_pid":
                    same_topology = row["topology"] == topology
                else:
                    same_topology = (
                        row["left_topology"] == topology
                        and row["right_topology"] == topology
                    )
                if same_topology:
                    selected.append(float(row[value_key]))
            values = np.asarray(selected, dtype=np.float64)
            expected_pairs = peptide_counts[(strain, topology)] * (
                peptide_counts[(strain, topology)] - 1
            ) // 2
            if values.size != expected_pairs:
                raise RuntimeError(
                    f"Expected {expected_pairs} {metric} pairs for "
                    f"{strain}/{topology}, found {values.size}"
                )
            output.append(
                {
                    "metric": metric,
                    "target_strain": strain,
                    "topology": topology,
                    "n_peptides": peptide_counts[(strain, topology)],
                    **summarize_similarities(values),
                }
            )
    return output


def plot_pairwise_by_strain_topology(
    peptides: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    tanimoto_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot similarity distributions after both strain and topology stratification."""

    groups = [
        ("BAA-3197", "linear"),
        ("BAA-3197", "cyclic"),
        ("BAA-3170", "linear"),
        ("BAA-3170", "cyclic"),
    ]
    peptide_counts = {
        group: sum(
            row["target_strain"] == group[0]
            and ("cyclic" if row["is_cyclic"] else "linear") == group[1]
            for row in peptides
        )
        for group in groups
    }
    colors = {"BAA-3197": "#2C6EAA", "BAA-3170": "#56B4E9"}
    datasets: list[tuple[str, list[np.ndarray]]] = []
    for title, rows, key in (
        ("Sequence identity (PID)", sequence_rows, "pid"),
        ("Morgan Tanimoto similarity", tanimoto_rows, "tanimoto"),
    ):
        grouped_values: list[np.ndarray] = []
        for strain, topology in groups:
            selected: list[float] = []
            for row in rows:
                if (
                    row["left_target_strain"] != strain
                    or row["right_target_strain"] != strain
                ):
                    continue
                if key == "pid":
                    same_topology = row["topology"] == topology
                else:
                    same_topology = (
                        row["left_topology"] == topology
                        and row["right_topology"] == topology
                    )
                if same_topology:
                    selected.append(float(row[key]))
            grouped_values.append(np.asarray(selected, dtype=np.float64))
        datasets.append((title, grouped_values))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    positions = np.arange(len(groups), dtype=float)
    for axis, (title, grouped_values), panel_index in zip(
        axes, datasets, range(2), strict=True
    ):
        for index, ((strain, topology), values) in enumerate(
            zip(groups, grouped_values, strict=True)
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
                    body.set_facecolor(colors[strain])
                    body.set_edgecolor("#2F2F2F")
                    body.set_linewidth(0.8)
                    body.set_alpha(0.72 if topology == "linear" else 0.42)

            rng = np.random.default_rng(20260805 + panel_index * 10 + index)
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            axis.scatter(
                positions[index] + jitter,
                values,
                s=18 if values.size <= 3 else 11,
                facecolor="#FFFFFF",
                edgecolor="#2F2F2F",
                linewidth=0.45,
                alpha=0.8 if values.size <= 3 else 0.58,
                zorder=3,
            )
            q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
            axis.vlines(
                positions[index], q25, q75, color="#2F2F2F", linewidth=4.5, zorder=4
            )
            axis.scatter(
                [positions[index]],
                [median],
                s=31,
                facecolor="#FFFFFF",
                edgecolor="#2F2F2F",
                linewidth=0.9,
                zorder=5,
            )
            axis.text(
                positions[index] + 0.14,
                median + 0.018,
                f"{median:.3f}",
                ha="left",
                va="bottom",
                fontsize=7.2,
                color="#2F2F2F",
            )

        labels = [
            f"{strain}\n{topology.capitalize()}\n"
            f"{peptide_counts[(strain, topology)]} pep., {len(values)} pairs"
            for (strain, topology), values in zip(groups, grouped_values, strict=True)
        ]
        axis.set_title(
            f"{title}\nwithin-strain, within-topology pairs", fontsize=10, pad=8
        )
        axis.set_xlim(-0.55, 3.55)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(positions, labels, fontsize=7.5)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.axvline(1.5, color="#C7C7C7", linewidth=0.7, linestyle="--")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.10,
            1.03,
            "a" if panel_index == 0 else "b",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
        )
    axes[0].set_ylabel("Pairwise similarity")
    fig.tight_layout(w_pad=2.4)

    outputs: list[Path] = []
    stem = "selected_peptide_pairwise_similarity_by_strain_topology"
    for suffix in ("pdf", "svg", "png"):
        output = output_dir / f"{stem}.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        outputs.append(output)
    plt.close(fig)
    caption_path = output_dir / f"{stem}_caption.md"
    caption_path.write_text(
        "**Pairwise similarity distributions among synthesized peptide candidates, "
        "stratified by generation target and peptide topology.** Only pairs sharing "
        "both the target strain and topology were compared. **a,** Sequence percent "
        "identity (PID) using the topology-aware alignment protocol. **b,** Morgan-"
        "fingerprint Tanimoto similarity for the same within-group pairs. The four "
        "groups contain 66, 3, 15 and 3 pairs, respectively (87 pairs total); 81 "
        "cross-strain same-topology pairs and 108 linear--cyclic pairs were excluded. "
        "Dots represent individual pairs, thick vertical bars show the interquartile "
        "range, and white circles mark the median. Violin densities are omitted for "
        "the cyclic groups because each contains only three pairs.\n",
        encoding="utf-8",
    )
    outputs.append(caption_path)
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peptides = load_final_peptides(args.mapping_csv, args.candidate_csv)
    sequence_rows = build_sequence_pairs(peptides)
    tanimoto_rows = build_tanimoto_pairs(peptides)

    expected_sequence_pairs = 18 * 17 // 2 + 6 * 5 // 2
    expected_tanimoto_pairs = EXPECTED_PEPTIDES * (EXPECTED_PEPTIDES - 1) // 2
    if len(sequence_rows) != expected_sequence_pairs:
        raise RuntimeError(
            f"Expected {expected_sequence_pairs} same-topology pairs, "
            f"found {len(sequence_rows)}"
        )
    if len(tanimoto_rows) != expected_tanimoto_pairs:
        raise RuntimeError(
            f"Expected {expected_tanimoto_pairs} structural pairs, "
            f"found {len(tanimoto_rows)}"
        )

    sequence_path = args.output_dir / "pairwise_sequence_similarity.csv"
    tanimoto_path = args.output_dir / "pairwise_tanimoto.csv"
    sequence_matrix_path = args.output_dir / "pairwise_sequence_pid_matrix.csv"
    tanimoto_matrix_path = args.output_dir / "pairwise_tanimoto_matrix.csv"
    nearest_path = args.output_dir / "nearest_neighbors.csv"
    stratified_summary_path = (
        args.output_dir / "pairwise_similarity_by_strain_topology_summary.csv"
    )
    summary_path = args.output_dir / "summary.json"
    manifest_path = args.output_dir / "manifest.json"

    write_csv(sequence_path, sequence_rows)
    write_csv(tanimoto_path, tanimoto_rows)
    write_matrix(
        sequence_matrix_path,
        peptides,
        sequence_rows,
        "pid",
        blank_cross_topology=True,
    )
    write_matrix(
        tanimoto_matrix_path,
        peptides,
        tanimoto_rows,
        "tanimoto",
        blank_cross_topology=False,
    )
    nearest_rows = nearest_neighbors(peptides, sequence_rows, tanimoto_rows)
    write_csv(nearest_path, nearest_rows)
    stratified_summary_rows = summarize_within_strain_topology(
        peptides, sequence_rows, tanimoto_rows
    )
    write_csv(stratified_summary_path, stratified_summary_rows)

    sequence_values = np.asarray(
        [float(row["pid"]) for row in sequence_rows], dtype=np.float64
    )
    tanimoto_values = np.asarray(
        [float(row["tanimoto"]) for row in tanimoto_rows], dtype=np.float64
    )
    structure_count = len(
        {row["canonical_isomeric_smiles"] for row in peptides}
    )
    summary = {
        "protocol": {
            "peptides": EXPECTED_PEPTIDES,
            "linear_peptides": 18,
            "cyclic_peptides": 6,
            "sequence_similarity": {
                "scoring_scheme": SEQUENCE_SCORING_SCHEME,
                "alignment": "global",
                "gap_open_score": -10.0,
                "gap_extension_score": -0.5,
                "pid_denominator": "aligned_positions_including_gaps",
                "matches": "case-sensitive exact residue equality excluding gaps",
                "topology_rule": (
                    "linear-linear and cyclic-cyclic only; linear-cyclic pairs are "
                    "not comparable under the frozen paper protocol"
                ),
                "cyclic_search": "all_left_rotations_by_all_right_rotations",
                "unordered_pair_symmetrization": (
                    "evaluate both target-query orientations and retain the higher "
                    "PID; this resolves first-optimal-alignment ties without making "
                    "an unordered result depend on peptide ID order"
                ),
            },
            "structural_similarity": {
                "fingerprint": "RDKit Morgan bit fingerprint",
                "radius": FINGERPRINT_RADIUS,
                "n_bits": FINGERPRINT_BITS,
                "include_chirality": FINGERPRINT_INCLUDE_CHIRALITY,
                "similarity": "Tanimoto",
            },
        },
        "structurally_distinct_canonical_isomeric_structures": structure_count,
        "unique_structure_fraction": structure_count / EXPECTED_PEPTIDES,
        "sequence_pairwise_pid": summarize_similarities(sequence_values),
        "structural_pairwise_tanimoto": summarize_similarities(tanimoto_values),
        "highest_sequence_similarity_pair": max(
            sequence_rows, key=lambda row: float(row["pid"])
        ),
        "highest_structural_similarity_pair": max(
            tanimoto_rows, key=lambda row: float(row["tanimoto"])
        ),
        "within_target_strain_and_topology": stratified_summary_rows,
        "within_target_strain_and_topology_pair_count": 87,
        "excluded_cross_strain_same_topology_sequence_pairs": 81,
        "excluded_linear_cyclic_sequence_pairs": (
            expected_tanimoto_pairs - expected_sequence_pairs
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    figure_paths = plot_pairwise_violin(
        sequence_values,
        tanimoto_values,
        args.output_dir,
        sequence_top_label="ApexOracle-14/23\nPID 0.895",
        tanimoto_top_label="ApexOracle-5/11\nTanimoto 0.856",
    )
    stratified_figure_paths = plot_pairwise_by_strain_topology(
        peptides, sequence_rows, tanimoto_rows, args.output_dir
    )

    output_paths = [
        sequence_path,
        tanimoto_path,
        sequence_matrix_path,
        tanimoto_matrix_path,
        nearest_path,
        stratified_summary_path,
        summary_path,
        *figure_paths,
        *stratified_figure_paths,
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_code": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                REPO_ROOT
                / "src/apexoracle/evaluation/generated_candidate_diversity.py",
                REPO_ROOT
                / "src/apexoracle/evaluation/sequence_similarity/alignment.py",
            )
        ],
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (args.mapping_csv, args.candidate_csv)
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
