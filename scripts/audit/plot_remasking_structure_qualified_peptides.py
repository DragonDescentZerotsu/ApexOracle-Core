#!/usr/bin/env python3
"""Plot the preliminary structure-qualified peptide counts.

This figure is separate from the historical four-panel reviewer figure. It
recomputes a narrow structure screen from the frozen valid attempts and shows
the SEP-padded historical-v1 classifier only as a supporting subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "remasking_schedule_reviewer"
DEFAULT_AUDIT_ROWS = (
    EXPERIMENT
    / "analysis"
    / "peptide_structure_audit"
    / "audited_valid_attempts.csv"
)
DEFAULT_SUMMARY = EXPERIMENT / "analysis" / "summary.json"
DEFAULT_EVALUATED_ATTEMPTS = (
    EXPERIMENT / "analysis" / "evaluated_attempts.csv"
)
DEFAULT_RUNS_ROOT = EXPERIMENT / "runs"
DEFAULT_OUTPUT_DIR = EXPERIMENT / "figures"

WINDOW_CONDITIONS = ["earlier", "current", "later", "narrower", "wider"]
WINDOW_LABELS = {
    "earlier": "Earlier\n0.75–0.65",
    "current": "Current\n0.55–0.45",
    "later": "Later\n0.35–0.25",
    "narrower": "Narrower\n0.525–0.475",
    "wider": "Wider\n0.55–0.25",
    "no_peptide_correction": "No peptide\nguidance",
}
COMPACT_WINDOW_LABELS = {
    "earlier": "Earlier\n.75–.65",
    "current": "Current\n.55–.45",
    "later": "Later\n.35–.25",
    "narrower": "Narrower\n.525–.475",
    "wider": "Wider\n.55–.25",
}

# B and halogens are explicitly allowed. Common/biologically plausible metals
# enter a separate manual-review bucket and are never auto-accepted. All other
# elements, including exotic metals, fail this narrow screen.
ALLOWED_ELEMENTS = frozenset(
    {"C", "H", "N", "O", "S", "P", "Se", "B", "F", "Cl", "Br", "I"}
)
MANUAL_REVIEW_METALS = frozenset(
    {"Li", "Na", "K", "Mg", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mo"}
)
RESIDUE_PATTERN = Chem.MolFromSmarts("[N;X3,X4+]-[C;X4]-[C;X3](=O)")

INK = "#263238"
MUTED = "#697782"
GRID = "#DCE2E7"
BLUE = "#2878B5"
BLUE_LIGHT = "#BFD9EB"
GREY = "#8995A0"
GREY_LIGHT = "#E1E6EA"
GOLD = "#C68A1B"

FIGURE_CAPTION = """\
**Figure X | Remasking-window sensitivity and peptide-guidance effectiveness.**
**a,** Peptide yield across five remasking windows. Bars are pooled rates from
600 attempts per window; vertical error bars are the sample s.d. across three
seed-level pooled rates (200 attempts per seed). **b,** Pooled median predicted
MIC among all RDKit-valid outputs from the clean MIC model; vertical error
bars are the sample s.d. across the three seed-level pooled median MIC values.
Lower values indicate stronger predicted activity. **c,** Current peptide
guidance (blue circles; gamma_peptide=15) versus no peptide guidance (open
squares; gamma_peptide=0) under the same 0.55–0.45 window. The three rows show
RDKit-valid yield, peptide yield, and the pooled median clean-model predicted
MIC among RDKit-valid outputs; row labels specify percentage or micromolar
units. The numerical x axis omits the unused 5–25 interval, as indicated by
the axis break. Horizontal error bars are the sample s.d. across three
seed-level pooled rates for yields and across three seed-level pooled median
MIC values for predicted MIC. Predicted MIC is model-based and is not a wet-lab
measurement.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-rows", type=Path, default=DEFAULT_AUDIT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--evaluated-attempts",
        type=Path,
        default=DEFAULT_EVALUATED_ATTEMPTS,
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--layout",
        choices=("with-context", "yield-only"),
        default="with-context",
        help=(
            "Default three-panel figure retains the original MIC and direct-"
            "control context. yield-only reproduces the earlier two-panel "
            "structure-screen figure."
        ),
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Defaults to a non-overlapping stem selected from --layout.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def parse_seed(task_id: str) -> str:
    marker = "__seed"
    if marker not in task_id:
        raise ValueError(f"Cannot parse seed from task_id={task_id!r}")
    return task_id.rsplit(marker, 1)[1]


def load_raw_smiles(
    runs_root: Path,
) -> dict[tuple[str, int, int], str]:
    raw_smiles: dict[tuple[str, int, int], str] = {}
    for path in sorted(runs_root.glob("*/batches/batch_*.jsonl")):
        task_id = path.parents[1].name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (
                    task_id,
                    int(row["batch_index"]),
                    int(row["sample_index"]),
                )
                raw_smiles[key] = str(row["smiles"])
    return raw_smiles


def molecule_for_row(
    row: dict[str, str],
    raw_smiles: dict[tuple[str, int, int], str],
) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(row["canonical_smiles"])
    if molecule is None:
        key = (
            row["task_id"],
            int(row["batch_index"]),
            int(row["sample_index"]),
        )
        molecule = Chem.MolFromSmiles(raw_smiles[key])
    if molecule is None:
        raise RuntimeError(
            "A row marked structurally reparsed could not be reconstructed: "
            f"{row['task_id']}:{row['batch_index']}:{row['sample_index']}"
        )
    return molecule


def structural_bucket(molecule: Chem.Mol) -> str:
    """Return clean, metal_review, or reject for the narrow screen."""
    has_required_backbone = (
        len(Chem.GetMolFrags(molecule)) == 1
        and not any(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
        and rdMolDescriptors.CalcNumAmideBonds(molecule) >= 1
        and len(molecule.GetSubstructMatches(RESIDUE_PATTERN)) >= 2
    )
    if not has_required_backbone:
        return "reject"

    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    extra_elements = elements - ALLOWED_ELEMENTS
    if not extra_elements:
        return "clean"
    if extra_elements <= MANUAL_REVIEW_METALS:
        return "metal_review"
    return "reject"


def denominators_by_condition_seed(
    summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, int]]:
    result: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "rdkit_valid": 0}
    )
    for key, values in summary["condition_strain_seed"].items():
        condition, _strain, seed = key.split("|")
        result[(condition, seed)]["attempted"] += int(values["attempted"])
        result[(condition, seed)]["rdkit_valid"] += int(
            values["rdkit_valid"]
        )
    return dict(result)


def compute_counts(
    audit_rows: Path,
    raw_smiles: dict[tuple[str, int, int], str],
    denominators: dict[tuple[str, str], dict[str, int]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "structure_qualified": 0,
            "classifier_supported": 0,
            "metal_review": 0,
        }
    )
    with audit_rows.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            seed = parse_seed(row["task_id"])
            molecule = molecule_for_row(row, raw_smiles)
            bucket = structural_bucket(molecule)
            if bucket == "clean":
                counts[(condition, seed)]["structure_qualified"] += 1
                if bool_value(row["sep_padded_classifier_positive"]):
                    counts[(condition, seed)]["classifier_supported"] += 1
            elif bucket == "metal_review":
                counts[(condition, seed)]["metal_review"] += 1

    rows: list[dict[str, Any]] = []
    for (condition, seed), denominator in sorted(denominators.items()):
        values = counts[(condition, seed)]
        row: dict[str, Any] = {
            "condition": condition,
            "seed": seed,
            **denominator,
            **values,
        }
        row["rdkit_valid_pct_attempted"] = (
            100 * denominator["rdkit_valid"] / denominator["attempted"]
        )
        for metric in (
            "structure_qualified",
            "classifier_supported",
            "metal_review",
        ):
            row[f"{metric}_pct_attempted"] = (
                100 * values[metric] / denominator["attempted"]
            )
            row[f"{metric}_pct_rdkit_valid"] = (
                100 * values[metric] / denominator["rdkit_valid"]
            )
        rows.append(row)
    return rows


def pooled_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[row["condition"]].append(row)

    pooled: list[dict[str, Any]] = []
    for condition, rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "condition": condition,
            "seed": "pooled",
            "attempted": sum(row["attempted"] for row in rows),
            "rdkit_valid": sum(row["rdkit_valid"] for row in rows),
        }
        output["rdkit_valid_pct_attempted"] = (
            100 * output["rdkit_valid"] / output["attempted"]
        )
        output["rdkit_valid_pct_attempted_seed_sd"] = float(
            np.std(
                [row["rdkit_valid_pct_attempted"] for row in rows],
                ddof=1,
            )
        )
        for metric in (
            "structure_qualified",
            "classifier_supported",
            "metal_review",
        ):
            output[metric] = sum(row[metric] for row in rows)
            for denominator_name in ("attempted", "rdkit_valid"):
                rate_key = f"{metric}_pct_{denominator_name}"
                output[rate_key] = (
                    100 * output[metric] / output[denominator_name]
                )
                output[f"{rate_key}_seed_sd"] = float(
                    np.std([row[rate_key] for row in rows], ddof=1)
                )
        pooled.append(output)
    return pooled


def add_context_metrics(
    pooled: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    seed_rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for seed_row in seed_rows:
        seed_rows_by_condition[seed_row["condition"]].append(seed_row)
    for row in pooled:
        condition = summary["condition"][row["condition"]]
        row["v1_classifier_positive"] = int(
            condition["peptide_classifier_positive"]
        )
        row["v1_classifier_positive_pct_attempted"] = (
            100
            * condition["peptide_classifier_positive"]
            / condition["attempted"]
        )
        row["valid_predicted_mic_n"] = int(
            condition["valid_predicted_mic_n"]
        )
        row["valid_predicted_mic_median_uM"] = float(
            condition["valid_predicted_mic_median_uM"]
        )
        row["valid_predicted_mic_seed_median_sd_uM"] = float(
            np.std(
                [
                    seed_row["valid_predicted_mic_median_uM"]
                    for seed_row in seed_rows_by_condition[row["condition"]]
                ],
                ddof=1,
            )
        )


def add_seed_mic_metrics(
    seed_rows: list[dict[str, Any]],
    evaluated_attempts: Path,
) -> None:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with evaluated_attempts.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                bool_value(row["rdkit_valid"])
                and row["predicted_mic_uM"].strip()
            ):
                values[(row["condition"], row["seed"])].append(
                    float(row["predicted_mic_uM"])
                )
    for row in seed_rows:
        mic_values = values[(row["condition"], row["seed"])]
        if not mic_values:
            raise RuntimeError(
                "No finite RDKit-valid MIC values for "
                f"{row['condition']} seed {row['seed']}"
            )
        row["valid_predicted_mic_n"] = len(mic_values)
        row["valid_predicted_mic_median_uM"] = float(
            np.median(np.asarray(mic_values, dtype=np.float64))
        )


def write_plotted_data(
    path: Path,
    seed_rows: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
) -> None:
    rows = pooled + seed_rows
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)


def plot_panel(
    axis: plt.Axes,
    pooled_by_condition: dict[str, dict[str, Any]],
    conditions: list[str],
    *,
    denominator: str,
    title: str,
    panel_label: str,
) -> None:
    x = np.arange(len(conditions), dtype=float)
    width = 0.31
    outer_values = np.asarray(
        [
            pooled_by_condition[name][
                f"structure_qualified_pct_{denominator}"
            ]
            for name in conditions
        ]
    )
    inner_values = np.asarray(
        [
            pooled_by_condition[name][
                f"classifier_supported_pct_{denominator}"
            ]
            for name in conditions
        ]
    )
    metal_values = np.asarray(
        [
            pooled_by_condition[name][f"metal_review_pct_{denominator}"]
            for name in conditions
        ]
    )
    outer_sd = np.asarray(
        [
            pooled_by_condition[name][
                f"structure_qualified_pct_{denominator}_seed_sd"
            ]
            for name in conditions
        ]
    )
    inner_sd = np.asarray(
        [
            pooled_by_condition[name][
                f"classifier_supported_pct_{denominator}_seed_sd"
            ]
            for name in conditions
        ]
    )

    outer_colors = [
        BLUE_LIGHT if name == "current" else GREY_LIGHT
        for name in conditions
    ]
    inner_colors = [
        BLUE if name == "current" else GREY for name in conditions
    ]
    axis.bar(
        x - width / 2,
        outer_values,
        width,
        color=outer_colors,
        edgecolor=INK,
        linewidth=0.7,
        yerr=outer_sd,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": MUTED},
        label="Structure-qualified",
        zorder=3,
    )
    axis.bar(
        x + width / 2,
        inner_values,
        width,
        color=inner_colors,
        edgecolor=INK,
        linewidth=0.7,
        yerr=inner_sd,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": MUTED},
        label="+ SEP-padded classifier positive",
        zorder=3,
    )
    axis.scatter(
        x,
        metal_values,
        marker="D",
        s=31,
        facecolors="white",
        edgecolors=GOLD,
        linewidths=1.4,
        label="Metal review (excluded)",
        zorder=5,
    )

    for index, name in enumerate(conditions):
        values = pooled_by_condition[name]
        label_y = max(
            outer_values[index] + outer_sd[index],
            inner_values[index] + inner_sd[index],
        )
        axis.annotate(
            (
                f"{values['structure_qualified']}/"
                f"{values['classifier_supported']}"
            ),
            (x[index], label_y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=INK,
        )
        if values["metal_review"]:
            axis.annotate(
                f"n={values['metal_review']}",
                (x[index], metal_values[index]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=GOLD,
            )

    axis.set_xticks(x, [WINDOW_LABELS[name] for name in conditions])
    axis.set_ylabel(
        "% of all attempts"
        if denominator == "attempted"
        else "% of RDKit-valid outputs",
        color=INK,
    )
    axis.set_title(
        title,
        loc="left",
        color=INK,
        fontsize=11.5,
        y=1.12,
        pad=0,
    )
    axis.text(
        -0.10,
        1.12,
        panel_label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )
    style_axis(axis)


def make_yield_only_figure(
    pooled: list[dict[str, Any]],
    output_base: Path,
) -> list[Path]:
    pooled_by_condition = {row["condition"]: row for row in pooled}
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 5.2),
        gridspec_kw={"width_ratios": [1.75, 1.0]},
    )
    plot_panel(
        axes[0],
        pooled_by_condition,
        WINDOW_CONDITIONS,
        denominator="attempted",
        title="Window sensitivity",
        panel_label="a",
    )
    plot_panel(
        axes[1],
        pooled_by_condition,
        ["current", "no_peptide_correction"],
        denominator="rdkit_valid",
        title="Current setting vs. no peptide guidance",
        panel_label="b",
    )
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    legend_order = [
        "Structure-qualified",
        "+ SEP-padded classifier positive",
        "Metal review (excluded)",
    ]
    figure.legend(
        [handles_by_label[label] for label in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.925),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Structure-qualified peptide yield across remasking settings",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=INK,
    )
    figure.text(
        0.07,
        0.935,
        (
            "Labels above bars: structure-qualified / classifier-supported "
            "counts. Bars show pooled rates; error bars show sample s.d. "
            "across three seed-level pooled rates."
        ),
        ha="left",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )
    figure.text(
        0.07,
        0.018,
        (
            "Narrow structural screen: ≥1 general amide, ≥2 "
            "N–Cα–C(=O) motifs, one component, and no radicals; B and "
            "halogens are allowed. Common-metal candidates are shown "
            "separately and excluded pending manual review."
        ),
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.80,
        bottom=0.20,
        wspace=0.32,
    )

    outputs: list[Path] = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output_base.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_mic_panel(
    axis: plt.Axes,
    pooled_by_condition: dict[str, dict[str, Any]],
) -> None:
    x = np.arange(len(WINDOW_CONDITIONS), dtype=float)
    values = np.asarray(
        [
            pooled_by_condition[name]["valid_predicted_mic_median_uM"]
            for name in WINDOW_CONDITIONS
        ]
    )
    standard_deviations = np.asarray(
        [
            pooled_by_condition[name][
                "valid_predicted_mic_seed_median_sd_uM"
            ]
            for name in WINDOW_CONDITIONS
        ]
    )
    colors = [
        BLUE if name == "current" else GREY_LIGHT
        for name in WINDOW_CONDITIONS
    ]
    bars = axis.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor=INK,
        linewidth=0.7,
        yerr=standard_deviations,
        capsize=2.5,
        error_kw={"elinewidth": 0.9, "ecolor": MUTED},
        zorder=3,
    )
    for bar, value, standard_deviation in zip(
        bars,
        values,
        standard_deviations,
    ):
        axis.annotate(
            f"{value:.1f}",
            (
                bar.get_x() + bar.get_width() / 2,
                value + standard_deviation,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
    axis.set_xticks(
        x,
        [COMPACT_WINDOW_LABELS[name] for name in WINDOW_CONDITIONS],
    )
    axis.tick_params(axis="x", labelsize=7.6)
    axis.set_ylim(0, max(values + standard_deviations) * 1.16)
    axis.set_ylabel("Median predicted MIC (µM)", color=INK)
    axis.set_title(
        "Predicted MIC across guidance windows",
        loc="center",
        color=INK,
        fontsize=11.5,
        y=1.08,
        pad=0,
    )
    axis.text(
        -0.10,
        1.08,
        "b",
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )
    style_axis(axis)


def plot_strict_yield_panel(
    axis: plt.Axes,
    pooled_by_condition: dict[str, dict[str, Any]],
) -> None:
    x = np.arange(len(WINDOW_CONDITIONS), dtype=float)
    values = np.asarray(
        [
            pooled_by_condition[name][
                "classifier_supported_pct_attempted"
            ]
            for name in WINDOW_CONDITIONS
        ]
    )
    standard_deviations = np.asarray(
        [
            pooled_by_condition[name][
                "classifier_supported_pct_attempted_seed_sd"
            ]
            for name in WINDOW_CONDITIONS
        ]
    )
    colors = [
        BLUE if name == "current" else GREY_LIGHT
        for name in WINDOW_CONDITIONS
    ]
    axis.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor=INK,
        linewidth=0.7,
        yerr=standard_deviations,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": MUTED},
        zorder=3,
    )
    for position, name, value, standard_deviation in zip(
        x,
        WINDOW_CONDITIONS,
        values,
        standard_deviations,
    ):
        axis.annotate(
            (
                f"{value:.1f}%\n"
                f"(n={pooled_by_condition[name]['classifier_supported']})"
            ),
            (position, value + standard_deviation),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=INK,
        )
    axis.set_xticks(
        x,
        [COMPACT_WINDOW_LABELS[name] for name in WINDOW_CONDITIONS],
    )
    axis.tick_params(axis="x", labelsize=7.6)
    axis.set_ylim(
        0,
        max(values + standard_deviations) + 2.8,
    )
    axis.set_ylabel("Peptide yield (% of all attempts)", color=INK)
    axis.set_title(
        "Peptide yield across guidance windows",
        loc="center",
        color=INK,
        fontsize=11.5,
        y=1.08,
        pad=0,
    )
    axis.text(
        -0.10,
        1.08,
        "a",
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )
    style_axis(axis)


def plot_control_panel(
    axis: plt.Axes,
    pooled_by_condition: dict[str, dict[str, Any]],
) -> None:
    current = pooled_by_condition["current"]
    control = pooled_by_condition["no_peptide_correction"]
    metric_rows = [
        (
            "RDKit-valid\n(% of all attempts)",
            current["rdkit_valid_pct_attempted"],
            control["rdkit_valid_pct_attempted"],
            current["rdkit_valid_pct_attempted_seed_sd"],
            control["rdkit_valid_pct_attempted_seed_sd"],
        ),
        (
            "Peptide yield\n(% of all attempts)",
            current["classifier_supported_pct_attempted"],
            control["classifier_supported_pct_attempted"],
            current["classifier_supported_pct_attempted_seed_sd"],
            control["classifier_supported_pct_attempted_seed_sd"],
        ),
        (
            "Median predicted MIC\n(μM; RDKit-valid outputs)",
            current["valid_predicted_mic_median_uM"],
            control["valid_predicted_mic_median_uM"],
            current["valid_predicted_mic_seed_median_sd_uM"],
            control["valid_predicted_mic_seed_median_sd_uM"],
        ),
    ]
    y = np.arange(len(metric_rows))
    current_values = np.asarray([row[1] for row in metric_rows])
    control_values = np.asarray([row[2] for row in metric_rows])
    current_standard_deviations = np.asarray(
        [row[3] for row in metric_rows]
    )
    control_standard_deviations = np.asarray(
        [row[4] for row in metric_rows]
    )

    # Compress the unused 5--25 numerical interval while preserving a linear
    # scale within both displayed segments. Row labels carry the metric units.
    break_left = 5.0
    break_right = 25.0
    displayed_gap_end = 8.0

    def compressed_x(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if np.any((values > break_left) & (values < break_right)):
            raise ValueError(
                "A panel-c value falls inside the declared 5--25 axis break"
            )
        return np.where(
            values <= break_left,
            values,
            values - break_right + displayed_gap_end,
        )

    current_display = compressed_x(current_values)
    control_display = compressed_x(control_values)

    for position, current_value, control_value in zip(
        y, current_display, control_display
    ):
        axis.plot(
            [control_value, current_value],
            [position, position],
            color=GREY_LIGHT,
            linewidth=2.2,
            zorder=1,
        )
    vertical_offset = 0.075
    axis.errorbar(
        current_display,
        y - vertical_offset,
        xerr=current_standard_deviations,
        fmt="o",
        markersize=6.8,
        color=BLUE,
        ecolor=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.8,
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.2,
        label="Current guidance",
        zorder=3,
    )
    axis.errorbar(
        control_display,
        y + vertical_offset,
        xerr=control_standard_deviations,
        fmt="s",
        markersize=6.5,
        color=INK,
        ecolor=INK,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=1.0,
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.2,
        label="No peptide guidance",
        zorder=3,
    )
    for (
        position,
        current_value,
        control_value,
        current_position,
        control_position,
    ) in zip(
        y,
        current_values,
        control_values,
        current_display,
        control_display,
    ):
        axis.text(
            current_position,
            position - 0.22,
            f"{current_value:.1f}",
            fontsize=8.2,
            color=BLUE,
            ha="center",
            va="center",
        )
        axis.text(
            control_position,
            position + 0.22,
            f"{control_value:.1f}",
            fontsize=8.2,
            color=INK,
            ha="center",
            va="center",
        )
    axis.set_yticks(y, [row[0] for row in metric_rows])
    axis.invert_yaxis()
    axis.set_ylim(len(metric_rows) - 0.55, -0.55)
    axis.set_xlim(0, 57.0)
    tick_values = np.asarray([0, 5, 25, 40, 60, 75], dtype=float)
    axis.set_xticks(
        compressed_x(tick_values),
        [f"{value:g}" for value in tick_values],
    )
    axis.set_xlabel("Value (row-specific units)", color=INK)
    axis.set_title(
        "With vs. without peptide guidance",
        loc="center",
        color=INK,
        fontsize=11.5,
        y=1.08,
        pad=0,
    )
    axis.text(
        -0.14,
        1.08,
        "c",
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=7.7,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    style_axis(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    break_center = (break_left + displayed_gap_end) / 2
    transform = axis.get_xaxis_transform()
    for offset in (-0.36, 0.36):
        center = break_center + offset
        axis.plot(
            [center - 0.20, center + 0.20],
            [-0.025, 0.025],
            transform=transform,
            color=INK,
            linewidth=1.0,
            clip_on=False,
            zorder=6,
        )


def make_context_figure(
    pooled: list[dict[str, Any]],
    output_base: Path,
) -> list[Path]:
    pooled_by_condition = {row["condition"]: row for row in pooled}
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.5),
        gridspec_kw={
            "width_ratios": [1.08, 1.08, 1.18],
            "wspace": 0.42,
        },
    )
    axis_a, axis_b, axis_c = axes

    plot_strict_yield_panel(axis_a, pooled_by_condition)
    plot_mic_panel(axis_b, pooled_by_condition)
    plot_control_panel(axis_c, pooled_by_condition)
    axis_a.set_ylim(bottom=0)

    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.80,
        bottom=0.20,
    )

    outputs: list[Path] = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output_base.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    denominators = denominators_by_condition_seed(summary)
    raw_smiles = load_raw_smiles(args.runs_root)
    seed_rows = compute_counts(args.audit_rows, raw_smiles, denominators)
    add_seed_mic_metrics(seed_rows, args.evaluated_attempts)
    pooled = pooled_rows(seed_rows)
    add_context_metrics(pooled, seed_rows, summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or (
        "remasking_structure_qualified_peptides_with_mic_control"
        if args.layout == "with-context"
        else "remasking_structure_qualified_peptides"
    )
    output_base = args.output_dir / stem
    plotted_data = args.output_dir / f"{stem}_plotted_data.csv"
    write_plotted_data(plotted_data, seed_rows, pooled)
    figure_paths = (
        make_context_figure(pooled, output_base)
        if args.layout == "with-context"
        else make_yield_only_figure(pooled, output_base)
    )
    caption_path = args.output_dir / f"{stem}_caption.md"
    caption_path.write_text(FIGURE_CAPTION, encoding="utf-8")

    script_path = Path(__file__).resolve()
    manifest_path = args.output_dir / f"{stem}_manifest.json"
    manifest = {
        "schema_version": 1,
        "layout": args.layout,
        "artifact_status": (
            "canonical_final_reviewer_figure"
            if args.layout == "with-context"
            else "legacy_yield_only_figure"
        ),
        "approval_date": (
            "2026-07-31" if args.layout == "with-context" else None
        ),
        "canonical_revision": (
            "Panel c uses one three-row dot-and-interval axis for two yields "
            "and all-RDKit-valid median predicted MIC, with row-specific "
            "units and a 5--25 numerical axis break."
            if args.layout == "with-context"
            else None
        ),
        "scientific_status": (
            "Preliminary narrow structural screen; not yet validated as a "
            "general peptide ground-truth definition."
        ),
        "definitions": {
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "manual_review_metals": sorted(MANUAL_REVIEW_METALS),
            "general_amide_minimum": 1,
            "residue_pattern": "[N;X3,X4+]-[C;X4]-[C;X3](=O)",
            "residue_pattern_minimum": 2,
            "single_component": True,
            "radicals_allowed": False,
            "classifier_support": (
                "Historical v1 score >=0.5 after replacing all token "
                "positions strictly after the first [SEP] with PAD."
            ),
        },
        "inputs": {
            "audit_rows": {
                "path": str(args.audit_rows.resolve()),
                "sha256": sha256(args.audit_rows),
            },
            "summary": {
                "path": str(args.summary.resolve()),
                "sha256": sha256(args.summary),
            },
            "evaluated_attempts": {
                "path": str(args.evaluated_attempts.resolve()),
                "sha256": sha256(args.evaluated_attempts),
            },
        },
        "script": {
            "path": str(script_path),
            "sha256": sha256(script_path),
        },
        "outputs": {},
    }
    for path in [plotted_data, caption_path, *figure_paths]:
        manifest["outputs"][path.name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pooled_by_condition = {row["condition"]: row for row in pooled}
    compact = {
        name: {
            key: pooled_by_condition[name][key]
            for key in (
                "attempted",
                "rdkit_valid",
                "structure_qualified",
                "classifier_supported",
                "metal_review",
                "structure_qualified_pct_attempted",
                "classifier_supported_pct_attempted",
                "structure_qualified_pct_rdkit_valid",
                "classifier_supported_pct_rdkit_valid",
            )
        }
        for name in [*WINDOW_CONDITIONS, "no_peptide_correction"]
    }
    print(json.dumps(compact, indent=2))
    print(f"Wrote {plotted_data}")
    for path in figure_paths:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
