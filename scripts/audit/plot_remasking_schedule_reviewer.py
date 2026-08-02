#!/usr/bin/env python3
"""Plot the frozen ReMDM schedule reviewer experiment.

The bar version consumes only the compact, reviewed summary JSON. The violin
version additionally reads the frozen evaluated-attempt CSV to display the
existing predicted-MIC distribution; neither version recomputes peptide labels
or predicted MIC values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT
    / "experiments"
    / "remasking_schedule_reviewer"
    / "analysis"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "remasking_schedule_reviewer" / "figures"
)
DEFAULT_EVALUATED_ATTEMPTS = (
    ROOT
    / "experiments"
    / "remasking_schedule_reviewer"
    / "analysis"
    / "evaluated_attempts.csv"
)

WINDOW_CONDITIONS = ["earlier", "current", "later", "narrower", "wider"]
WINDOW_LABELS = {
    "earlier": "Earlier\n0.75–0.65",
    "current": "Current\n0.55–0.45",
    "later": "Later\n0.35–0.25",
    "narrower": "Narrower\n0.525–0.475",
    "wider": "Wider\n0.55–0.25",
}

INK = "#263238"
MUTED = "#6B7785"
GRID = "#DCE2E7"
BLUE = "#2878B5"
BLUE_DARK = "#195581"
NEUTRAL = "#C8D0D7"
NEUTRAL_LIGHT = "#E9EDF0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--evaluated-attempts",
        type=Path,
        default=DEFAULT_EVALUATED_ATTEMPTS,
        help="Required only for --panel-b-style violin.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--panel-b-style",
        choices=("bar", "violin"),
        default="bar",
        help="Keep the pooled-median bar or show the full valid-MIC distribution.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output stem. Defaults to a style-specific non-overlapping name.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=INK, labelsize=9)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.17,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )


def pooled_seed_yields(
    condition_strain_seed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, list[int]]] = {}
    for key, values in condition_strain_seed.items():
        condition_name, _strain, seed = key.split("|")
        pair = counts.setdefault(condition_name, {}).setdefault(seed, [0, 0])
        pair[0] += int(values["rdkit_valid_peptide_classifier_positive"])
        pair[1] += int(values["attempted"])
    return {
        condition_name: {
            seed: 100 * numerator / denominator
            for seed, (numerator, denominator) in sorted(seed_counts.items())
        }
        for condition_name, seed_counts in counts.items()
    }


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    differences_array = np.asarray(differences, dtype=float)
    observed = abs(float(differences_array.mean()))
    permutation_statistics = []
    for mask in range(1 << len(differences_array)):
        signs = np.array(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(differences_array))]
        )
        permutation_statistics.append(abs(float((signs * differences_array).mean())))
    return float(
        np.mean(np.asarray(permutation_statistics) >= observed - np.finfo(float).eps)
    )


def paired_control_pvalues(
    condition_strain_seed: dict[str, dict[str, Any]],
) -> dict[str, float]:
    metric_keys = {
        "classifier_positive": "peptide_classifier_positive_proportion",
        "rdkit_valid": "rdkit_valid_proportion",
        "valid_peptide_yield": (
            "rdkit_valid_peptide_classifier_positive_proportion_of_attempts"
        ),
    }
    pvalues: dict[str, float] = {}
    for metric, key in metric_keys.items():
        differences = []
        for suffix in sorted(
            entry.removeprefix("current|")
            for entry in condition_strain_seed
            if entry.startswith("current|")
        ):
            current = condition_strain_seed[f"current|{suffix}"][key]
            control = condition_strain_seed[f"no_peptide_correction|{suffix}"][key]
            differences.append(float(current) - float(control))
        pvalues[metric] = exact_sign_flip_pvalue(differences)
    return pvalues


def load_valid_mic_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in WINDOW_CONDITIONS:
                continue
            if row["rdkit_valid"].lower() != "true" or not row["predicted_mic_uM"]:
                continue
            records.append(
                {
                    "condition": row["condition"],
                    "strain": row["strain"],
                    "seed": row["seed"],
                    "task_id": row["task_id"],
                    "sample_id": f'{row["batch_index"]}:{row["sample_index"]}',
                    "predicted_mic_uM": float(row["predicted_mic_uM"]),
                }
            )
    return records


def write_figure_data(
    path: Path,
    summary: dict[str, Any],
    panel_b_style: str,
    seed_yields: dict[str, dict[str, float]],
    pvalues: dict[str, float],
    mic_records: list[dict[str, Any]],
) -> None:
    condition = summary["condition"]
    condition_strain = summary["condition_strain"]
    fields = [
        "panel",
        "condition",
        "window",
        "strain",
        "seed",
        "sample_id",
        "metric",
        "value",
        "numerator",
        "denominator",
        "unit",
        "uncertainty_or_test",
        "uncertainty_value_or_p",
    ]
    rows: list[dict[str, Any]] = []

    for name in WINDOW_CONDITIONS:
        values = condition[name]
        seed_values = list(seed_yields[name].values())
        rows.append(
            {
                "panel": "a",
                "condition": name,
                "window": WINDOW_LABELS[name].replace("\n", " "),
                "strain": "pooled",
                "metric": "valid_peptide_yield",
                "value": 100
                * values[
                    "rdkit_valid_peptide_classifier_positive_proportion_of_attempts"
                ],
                "numerator": values["rdkit_valid_peptide_classifier_positive"],
                "denominator": values["attempted"],
                "unit": "percent_of_attempts",
                "uncertainty_or_test": "sample_sd_across_3_seed_pooled_rates",
                "uncertainty_value_or_p": float(np.std(seed_values, ddof=1)),
            }
        )
        for seed, seed_value in seed_yields[name].items():
            rows.append(
                {
                    "panel": "a",
                    "condition": name,
                    "window": WINDOW_LABELS[name].replace("\n", " "),
                    "strain": "pooled",
                    "seed": seed,
                    "metric": "valid_peptide_yield_seed_rate",
                    "value": seed_value,
                    "denominator": 200,
                    "unit": "percent_of_attempts",
                }
            )
        rows.append(
            {
                "panel": "b",
                "condition": name,
                "window": WINDOW_LABELS[name].replace("\n", " "),
                "strain": "pooled",
                "metric": (
                    "predicted_mic_valid_distribution"
                    if panel_b_style == "violin"
                    else "median_predicted_mic_valid"
                ),
                "value": values["valid_predicted_mic_median_uM"],
                "numerator": values["valid_predicted_mic_n"],
                "denominator": values["rdkit_valid"],
                "unit": "micromolar",
            }
        )

    if panel_b_style == "violin":
        for record in mic_records:
            rows.append(
                {
                    "panel": "b",
                    "condition": record["condition"],
                    "window": WINDOW_LABELS[record["condition"]].replace("\n", " "),
                    "strain": record["strain"],
                    "seed": record["seed"],
                    "sample_id": f'{record["task_id"]}|{record["sample_id"]}',
                    "metric": "predicted_mic_valid_observation",
                    "value": record["predicted_mic_uM"],
                    "unit": "micromolar",
                }
            )

    for metric, key in [
        ("classifier_positive", "peptide_classifier_positive_proportion"),
        ("rdkit_valid", "rdkit_valid_proportion"),
        (
            "valid_peptide_yield",
            "rdkit_valid_peptide_classifier_positive_proportion_of_attempts",
        ),
    ]:
        for name in ["current", "no_peptide_correction"]:
            values = condition[name]
            rows.append(
                {
                    "panel": "c",
                    "condition": name,
                    "window": "0.55–0.45",
                    "strain": "pooled",
                    "metric": metric,
                    "value": 100 * values[key],
                    "numerator": "",
                    "denominator": values["attempted"],
                    "unit": "percent_of_attempts",
                    "uncertainty_or_test": (
                        "two_sided_exact_paired_sign_flip_6_matched_strain_seed_tasks"
                        if name == "current"
                        else ""
                    ),
                    "uncertainty_value_or_p": (
                        pvalues[metric] if name == "current" else ""
                    ),
                }
            )

    composition_groups = [
        ("pooled", condition["current"]),
        ("BAA-3170", condition_strain["current|BAA-3170"]),
        ("BAA-3197", condition_strain["current|BAA-3197"]),
    ]
    for strain, values in composition_groups:
        for metric, count_key, proportion_key in [
            (
                "peptide",
                "rdkit_valid_peptide_classifier_positive",
                "rdkit_valid_peptide_classifier_positive_proportion_of_valid",
            ),
            (
                "classifier_negative_small_molecule",
                "rdkit_valid_classifier_negative_small_molecule",
                "rdkit_valid_classifier_negative_small_molecule_proportion_of_valid",
            ),
        ]:
            rows.append(
                {
                    "panel": "d",
                    "condition": "current",
                    "window": "0.55–0.45",
                    "strain": strain,
                    "metric": metric,
                    "value": 100 * values[proportion_key],
                    "numerator": values[count_key],
                    "denominator": values["rdkit_valid"],
                    "unit": "percent_of_rdkit_valid",
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    evaluated_attempts_path = args.evaluated_attempts.resolve()
    output_dir = args.output_dir.resolve()
    stem = args.stem or (
        "remasking_schedule_reviewer"
        if args.panel_b_style == "bar"
        else "remasking_schedule_reviewer_violin"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    condition = summary["condition"]
    condition_strain = summary["condition_strain"]
    condition_strain_seed = summary["condition_strain_seed"]
    seed_yields = pooled_seed_yields(condition_strain_seed)
    pvalues = paired_control_pvalues(condition_strain_seed)
    mic_records = (
        load_valid_mic_records(evaluated_attempts_path)
        if args.panel_b_style == "violin"
        else []
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.3, 8.2))
    axis_a, axis_b, axis_c, axis_d = axes.flatten()
    x = np.arange(len(WINDOW_CONDITIONS))
    colors = [BLUE if name == "current" else NEUTRAL for name in WINDOW_CONDITIONS]
    edge_colors = [
        BLUE_DARK if name == "current" else MUTED for name in WINDOW_CONDITIONS
    ]

    yields = [
        100
        * condition[name][
            "rdkit_valid_peptide_classifier_positive_proportion_of_attempts"
        ]
        for name in WINDOW_CONDITIONS
    ]
    bars = axis_a.bar(
        x,
        yields,
        width=0.68,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.9,
        yerr=[
            float(np.std(list(seed_yields[name].values()), ddof=1))
            for name in WINDOW_CONDITIONS
        ],
        capsize=3,
        error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0},
    )
    seed_jitter = np.array([-0.09, 0.0, 0.09])
    yield_sd = []
    for position, name, value in zip(x, WINDOW_CONDITIONS, yields):
        seed_values = np.array(list(seed_yields[name].values()))
        standard_deviation = float(np.std(seed_values, ddof=1))
        yield_sd.append(standard_deviation)
        axis_a.scatter(
            position + seed_jitter,
            seed_values,
            s=18,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )
        axis_a.text(
            position,
            value + standard_deviation + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis_a.set_xticks(x, [WINDOW_LABELS[name] for name in WINDOW_CONDITIONS])
    axis_a.set_ylim(0, max(np.asarray(yields) + np.asarray(yield_sd)) + 8)
    axis_a.set_ylabel("Valid peptide yield (% of attempts)")
    axis_a.set_title(
        "Valid peptide yield across remasking windows",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        y=1.12,
        pad=0,
    )
    axis_a.text(
        0,
        1.025,
        "Bars: pooled rate; error bars: sample s.d. across 3 seeds (200 attempts/seed)",
        transform=axis_a.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    style_axis(axis_a)
    add_panel_label(axis_a, "a")

    mic = [condition[name]["valid_predicted_mic_median_uM"] for name in WINDOW_CONDITIONS]
    if args.panel_b_style == "bar":
        bars = axis_b.bar(
            x,
            mic,
            width=0.68,
            color=colors,
            edgecolor=edge_colors,
            linewidth=0.9,
        )
        axis_b.bar_label(
            bars, labels=[f"{value:.1f}" for value in mic], padding=3, fontsize=9
        )
        axis_b.set_ylim(0, 62)
        axis_b.set_ylabel("Median predicted MIC (µM)")
        panel_b_title = "Median predicted MIC among RDKit-valid molecules"
        panel_b_subtitle = (
            "Pooled median from the clean MIC model; lower values indicate stronger activity"
        )
    else:
        mic_by_condition = [
            np.asarray(
                [
                    record["predicted_mic_uM"]
                    for record in mic_records
                    if record["condition"] == name
                ]
            )
            for name in WINDOW_CONDITIONS
        ]
        log_mic = [np.log10(values) for values in mic_by_condition]
        violins = axis_b.violinplot(
            log_mic,
            positions=x,
            widths=0.76,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=200,
        )
        for body, color, edge_color in zip(
            violins["bodies"], colors, edge_colors
        ):
            body.set_facecolor(color)
            body.set_edgecolor(edge_color)
            body.set_linewidth(0.9)
            body.set_alpha(0.90)
        for position, values, median in zip(x, mic_by_condition, mic):
            q1, q3 = np.quantile(values, [0.25, 0.75])
            axis_b.vlines(
                position,
                np.log10(q1),
                np.log10(q3),
                color=INK,
                linewidth=4.0,
                zorder=3,
            )
            axis_b.scatter(
                position,
                np.log10(median),
                s=34,
                facecolor="white",
                edgecolor=INK,
                linewidth=0.8,
                zorder=4,
            )
            axis_b.text(
                position + 0.09,
                np.log10(median),
                f"{median:.1f}",
                fontsize=8,
                va="center",
                color=INK,
            )
        mic_ticks = np.array([1, 4, 16, 64, 256, 1024, 4096])
        axis_b.set_yticks(np.log10(mic_ticks), [f"{value:g}" for value in mic_ticks])
        axis_b.set_ylim(np.log10(0.6), np.log10(4500))
        axis_b.set_ylabel("Predicted MIC (µM, log scale)")
        panel_b_title = "Distribution of predicted MIC among RDKit-valid molecules"
        panel_b_subtitle = "Violin density on log scale; white dots: median; thick lines: IQR"
    axis_b.set_xticks(
        x,
        [
            (
                f"{WINDOW_LABELS[name]}\n(n={condition[name]['rdkit_valid']})"
                if args.panel_b_style == "violin"
                else WINDOW_LABELS[name]
            )
            for name in WINDOW_CONDITIONS
        ],
    )
    axis_b.set_title(
        panel_b_title,
        loc="left",
        fontsize=11,
        fontweight="semibold",
        y=1.12,
        pad=0,
    )
    axis_b.text(
        0,
        1.025,
        panel_b_subtitle,
        transform=axis_b.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    style_axis(axis_b)
    add_panel_label(axis_b, "b")

    comparison_metrics = [
        (
            "Classifier-positive\n(% of attempts)",
            "peptide_classifier_positive_proportion",
            "classifier_positive",
        ),
        ("RDKit-valid\n(% of attempts)", "rdkit_valid_proportion", "rdkit_valid"),
        (
            "Valid peptide yield\n(% of attempts)",
            "rdkit_valid_peptide_classifier_positive_proportion_of_attempts",
            "valid_peptide_yield",
        ),
    ]
    y = np.arange(len(comparison_metrics))
    current_values = np.array(
        [100 * condition["current"][key] for _, key, _ in comparison_metrics]
    )
    control_values = np.array(
        [
            100 * condition["no_peptide_correction"][key]
            for _, key, _ in comparison_metrics
        ]
    )
    for position, current_value, control_value in zip(
        y, current_values, control_values
    ):
        axis_c.plot(
            [control_value, current_value],
            [position, position],
            color=NEUTRAL,
            linewidth=2.0,
            zorder=1,
        )
    axis_c.scatter(
        current_values,
        y,
        s=62,
        marker="o",
        color=BLUE,
        edgecolor=BLUE_DARK,
        linewidth=0.9,
        label=r"Current ($\gamma_{\mathrm{peptide}}=15$)",
        zorder=3,
    )
    axis_c.scatter(
        control_values,
        y,
        s=58,
        marker="s",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.1,
        label=r"No peptide correction ($\gamma_{\mathrm{peptide}}=0$)",
        zorder=3,
    )
    for position, current_value, control_value in zip(
        y, current_values, control_values
    ):
        axis_c.text(
            current_value + 1.2,
            position - 0.10,
            f"{current_value:.1f}",
            fontsize=8.5,
            color=BLUE_DARK,
            va="center",
        )
        axis_c.text(
            control_value + 1.2,
            position + 0.13,
            f"{control_value:.1f}",
            fontsize=8.5,
            color=INK,
            va="center",
        )
    axis_c.set_yticks(y, [label for label, _, _ in comparison_metrics])
    axis_c.invert_yaxis()
    axis_c.set_xlim(25, 75)
    axis_c.set_xlabel("Percentage")
    axis_c.set_title(
        "Current window with and without peptide correction",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        y=1.12,
        pad=0,
    )
    axis_c.text(
        0,
        1.025,
        "Same 0.55–0.45 window; 600 raw attempts per condition",
        transform=axis_c.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    axis_c.legend(
        loc="lower right",
        frameon=False,
        fontsize=8.2,
        handletextpad=0.5,
    )
    style_axis(axis_c, grid_axis="x")
    add_panel_label(axis_c, "c")

    composition_groups = [
        ("Pooled", condition["current"]),
        ("BAA-3170", condition_strain["current|BAA-3170"]),
        ("BAA-3197", condition_strain["current|BAA-3197"]),
    ]
    group_labels = [name for name, _ in composition_groups]
    peptide_share = np.array(
        [
            100
            * values[
                "rdkit_valid_peptide_classifier_positive_proportion_of_valid"
            ]
            for _, values in composition_groups
        ]
    )
    small_share = 100 - peptide_share
    y = np.arange(len(composition_groups))
    axis_d.barh(
        y,
        peptide_share,
        color=BLUE,
        edgecolor=BLUE_DARK,
        linewidth=0.9,
        label="Peptide classifier-positive",
    )
    axis_d.barh(
        y,
        small_share,
        left=peptide_share,
        color=NEUTRAL_LIGHT,
        edgecolor=MUTED,
        linewidth=0.9,
        label="Classifier-negative small molecule",
    )
    for position, (_, values), peptide, small in zip(
        y, composition_groups, peptide_share, small_share
    ):
        peptide_count = values["rdkit_valid_peptide_classifier_positive"]
        small_count = values["rdkit_valid_classifier_negative_small_molecule"]
        peptide_name = "Peptide\n" if position == 0 else ""
        small_name = "Small molecule\n" if position == 0 else ""
        axis_d.text(
            peptide / 2,
            position,
            f"{peptide_name}{peptide:.1f}%\n(n={peptide_count})",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
            fontweight="semibold",
        )
        axis_d.text(
            peptide + small / 2,
            position,
            f"{small_name}{small:.1f}%\n(n={small_count})",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    axis_d.set_yticks(y, group_labels)
    axis_d.invert_yaxis()
    axis_d.set_xlim(0, 100)
    axis_d.set_xlabel("Composition among RDKit-valid candidates (%)")
    axis_d.set_title(
        "Candidate composition under the current window",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        y=1.12,
        pad=0,
    )
    axis_d.text(
        0,
        1.025,
        "Peptide defined by the generation-time v1 classifier at p ≥ 0.5",
        transform=axis_d.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    axis_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        fontsize=8.2,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    style_axis(axis_d, grid_axis="x")
    add_panel_label(axis_d, "d")

    figure.suptitle(
        "Remasking-window sensitivity and peptide-correction effectiveness",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.06,
        0.012,
        "Descriptive reviewer analysis. Peptide status and predicted MIC are model-based; "
        "predicted MIC is not a wet-lab measurement.",
        fontsize=8.2,
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.86,
        bottom=0.14,
        wspace=0.34,
        hspace=0.72,
    )

    output_paths = {
        suffix: output_dir / f"{stem}.{suffix}"
        for suffix in ("pdf", "svg", "png")
    }
    figure.savefig(output_paths["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(output_paths["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(
        output_paths["png"],
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
    )
    plt.close(figure)

    figure_data_path = output_dir / f"{stem}_data.csv"
    write_figure_data(
        figure_data_path,
        summary,
        args.panel_b_style,
        seed_yields,
        pvalues,
        mic_records,
    )

    script_path = Path(__file__).resolve()
    manifest = {
        "figure_kind": f"descriptive_reviewer_analysis_panel_b_{args.panel_b_style}",
        "source": {
            "path": str(summary_path),
            "sha256": sha256(summary_path),
        },
        "script": {
            "path": str(script_path),
            "sha256": sha256(script_path),
        },
        "outputs": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [*output_paths.values(), figure_data_path]
        ],
        "definitions": {
            "peptide": "v1 peptide classifier clean-input probability >= 0.5",
            "valid_peptide_yield_denominator": "all raw attempted generations",
            "composition_denominator": "RDKit-valid generated candidates",
            "predicted_mic": "clean MIC reporting model; micromolar; not wet-lab MIC",
            "panel_a_error_bars": (
                "sample standard deviation across 3 seed-level rates, each pooling "
                "2 strains and 200 attempts"
            ),
            "panel_c_tests": (
                "two-sided exact paired sign-flip tests across 6 matched "
                "strain-by-seed tasks; no multiple-testing correction"
            ),
        },
        "statistics": {
            "panel_a_seed_sd_percent": {
                name: float(np.std(list(seed_yields[name].values()), ddof=1))
                for name in WINDOW_CONDITIONS
            },
            "panel_c_exact_pvalues": pvalues,
        },
    }
    if args.panel_b_style == "violin":
        manifest["evaluated_attempts_source"] = {
            "path": str(evaluated_attempts_path),
            "sha256": sha256(evaluated_attempts_path),
        }
        manifest["definitions"]["panel_b_violin"] = (
            "kernel density of all existing finite predicted MIC values among "
            "RDKit-valid candidates; rendered on log10 scale"
        )
    manifest_path = output_dir / f"{stem}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    for path in [*output_paths.values(), figure_data_path, manifest_path]:
        print(path)


if __name__ == "__main__":
    main()
