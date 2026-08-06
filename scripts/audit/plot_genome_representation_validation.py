#!/usr/bin/env python3
"""Plot encoded-fragment variation and AMR/MGE linear-probe validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import sha256_file  # noqa: E402

DEFAULT_FRAGMENT_DIR = (
    REPO_ROOT
    / "experiments/genome_condition_reviewer/fragment_variation/all_embeddings"
)
DEFAULT_PROBE_DIR = (
    REPO_ROOT / "experiments/genome_condition_reviewer/historical_probe/analysis"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/genome_condition_reviewer/figures"

BLUE = "#356FA8"
BLUE_LIGHT = "#A9C5DF"
ORANGE = "#C66A2B"
ORANGE_LIGHT = "#E5B590"
INK = "#252525"
MID_GREY = "#777777"
LIGHT_GREY = "#D9D9D9"


def prepare_fragment_plot_data(
    frame: pd.DataFrame, *, maximum_divergence: float
) -> pd.DataFrame:
    """Return the exact homologous-fragment rows used by panel a."""
    required = {
        "pair_id",
        "species",
        "whole_genome_ani",
        "fragment_a_index",
        "fragment_b_index",
        "global_sequence_divergence",
        "cosine_distance",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fragment table missing columns: {missing}")
    output = frame.loc[
        (frame["global_sequence_divergence"] > 0)
        & (frame["global_sequence_divergence"] <= maximum_divergence)
    ].copy()
    if output.empty:
        raise ValueError("No fragments pass the maximum divergence threshold")
    if not np.isfinite(
        output[["global_sequence_divergence", "cosine_distance"]].to_numpy()
    ).all():
        raise ValueError("Fragment plot inputs must be finite")
    output["sequence_divergence_percent"] = 100.0 * output["global_sequence_divergence"]
    if (output["cosine_distance"] <= 0).any():
        raise ValueError("Variable-fragment cosine distances must be positive")
    return output


def prepare_probe_plot_data(
    probe_dir: Path, summary: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the exact five-fold probe metrics and pooled OOF summaries."""
    paths = {
        "AMR-associated": probe_dir / "amr_associated_probe_fold_metrics.csv",
        "Mobile-element-associated": (
            probe_dir / "mge_associated_probe_fold_metrics.csv"
        ),
    }
    fold_frames = []
    for label, path in paths.items():
        frame = pd.read_csv(path)
        if len(frame) != 5:
            raise ValueError(f"Expected five folds for {label}, found {len(frame)}")
        frame = frame.copy()
        frame["display_label"] = label
        fold_frames.append(frame)
    folds = pd.concat(fold_frames, ignore_index=True)
    if not np.isfinite(folds[["test_prevalence", "auprc", "auroc"]].to_numpy()).all():
        raise ValueError("Probe fold metrics must be finite")

    display_names = {
        "AMR": "AMR-associated",
        "Mobile element": "Mobile-element-associated",
    }
    rows = []
    for probe in summary["linear_probes"]:
        rows.append(
            {
                "display_label": display_names[str(probe["label"])],
                "probe_cohort_fragments": int(probe["probe_cohort_fragments"]),
                "probe_cohort_genomes": int(probe["probe_cohort_genomes"]),
                "positive_fragments": int(probe["probe_cohort_positive_fragments"]),
                "prevalence": float(probe["probe_cohort_prevalence"]),
                "oof_auprc": float(probe["oof_auprc"]),
                "oof_auroc": float(probe["oof_auroc"]),
                "fold_auprc_mean": float(probe["fold_auprc_mean"]),
                "fold_auprc_sample_sd": float(probe["fold_auprc_sample_sd"]),
                "fold_auroc_mean": float(probe["fold_auroc_mean"]),
                "fold_auroc_sample_sd": float(probe["fold_auroc_sample_sd"]),
            }
        )
    pooled = pd.DataFrame(rows)
    if set(pooled["display_label"]) != set(paths):
        raise ValueError("Probe summary labels do not match fold metric labels")
    return folds, pooled


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(colors=INK, labelsize=8)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)


def plot_fragment_panel(
    axis: plt.Axes,
    fragments: pd.DataFrame,
    fragment_summary: dict[str, object],
) -> None:
    close_mask = fragments["whole_genome_ani"] >= 99.0
    other_fragments = fragments.loc[~close_mask]
    close_fragments = fragments.loc[close_mask]
    axis.scatter(
        other_fragments["sequence_divergence_percent"],
        other_fragments["cosine_distance"],
        s=8,
        marker="x",
        color=MID_GREY,
        alpha=0.28,
        linewidths=0.45,
        label=f"95% ≤ ANI <99% ($n$={len(other_fragments):,})",
        rasterized=True,
    )
    axis.scatter(
        close_fragments["sequence_divergence_percent"],
        close_fragments["cosine_distance"],
        s=7,
        color=BLUE,
        alpha=0.10,
        linewidths=0,
        label=f"ANI ≥99% ($n$={len(close_fragments):,})",
        rasterized=True,
    )
    all_result = fragment_summary["results"]["all_pairs"]
    close_result = fragment_summary["results"]["whole_genome_ani_ge_99"]
    axis.text(
        0.98,
        0.04,
        f"All variable fragments ($n$={len(fragments):,}): $\\rho$ = "
        f"{all_result['pooled_spearman_sequence_divergence_vs_cosine_distance']:.3f}\n"
        f"ANI ≥99% subset ($n$={len(close_fragments):,}): $\\rho$ = "
        f"{close_result['pooled_spearman_sequence_divergence_vs_cosine_distance']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK,
    )
    axis.set_xlim(-0.03, 5.05)
    axis.set_yscale("log")
    axis.set_ylim(1.0e-8, 6.0e-3)
    axis.set_xlabel("Sequence divergence within homologous fragment (%)", fontsize=8.5)
    axis.set_ylabel("Embedding cosine distance", fontsize=8.5)
    axis.set_title("Variable homologous fragments", fontsize=9.5, fontweight="normal")
    style_axis(axis)
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        handletextpad=0.4,
        borderaxespad=0.4,
        markerscale=1.7,
    )


def plot_probe_panel(
    axis: plt.Axes,
    folds: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    metric: str,
) -> None:
    labels = ["AMR-associated", "Mobile-element-associated"]
    colors = [ORANGE, BLUE]
    light_colors = [ORANGE_LIGHT, BLUE_LIGHT]
    positions = np.arange(len(labels), dtype=float)
    offsets = np.linspace(-0.10, 0.10, 5)
    for position, label, color, light_color in zip(
        positions, labels, colors, light_colors
    ):
        fold = folds[folds["display_label"] == label].sort_values("fold")
        row = pooled[pooled["display_label"] == label].iloc[0]
        values = fold[metric].to_numpy(dtype=float)
        mean = float(row[f"fold_{metric}_mean"])
        sample_sd = float(row[f"fold_{metric}_sample_sd"])
        axis.scatter(
            position + offsets,
            values,
            s=23,
            facecolor=light_color,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        axis.errorbar(
            position,
            mean,
            yerr=sample_sd,
            fmt="D",
            markersize=4.5,
            color=INK,
            markerfacecolor="white",
            markeredgewidth=0.9,
            capsize=3,
            linewidth=1.0,
            zorder=4,
        )
        if metric == "auprc":
            baseline = float(row["prevalence"])
            axis.hlines(
                baseline,
                position - 0.22,
                position + 0.22,
                color=MID_GREY,
                linestyle="--",
                linewidth=1.0,
                zorder=2,
            )
            label_value = float(row["oof_auprc"])
        else:
            label_value = float(row["oof_auroc"])
        axis.text(
            position,
            mean + sample_sd + 0.025,
            f"OOF {label_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=INK,
        )
    if metric == "auroc":
        axis.axhline(0.5, color=MID_GREY, linestyle="--", linewidth=1.0)
        axis.text(
            1.40,
            0.505,
            "Random = 0.5",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=MID_GREY,
        )
        axis.set_ylim(0.45, 0.82)
        axis.set_ylabel("Held-out AUROC", fontsize=8.5)
        axis.set_title(
            "Fragment annotation probes: AUROC", fontsize=9.5, fontweight="normal"
        )
    else:
        axis.set_ylim(0.12, 0.56)
        axis.set_ylabel("Held-out AUPRC", fontsize=8.5)
        axis.set_title(
            "Fragment annotation probes: AUPRC", fontsize=9.5, fontweight="normal"
        )
        axis.text(
            0.98,
            0.04,
            "Dashed: evaluation-set prevalence",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=MID_GREY,
        )
    axis.set_xlim(-0.45, 1.45)
    axis.set_xticks(positions, ["AMR", "Mobile\nelement"])
    style_axis(axis)


def build_caption() -> str:
    """Return the manuscript caption for the canonical three-panel figure."""
    return (
        "**Encoded genome-fragment representations retain sub-species variation "
        "and annotation-associated signals.**\n"
        "**a,** Relationship between global sequence divergence and cosine distance "
        "for 4,649 variable mutual-best homologous 11-kb fragment pairs from 165 "
        "nearest same-species strain pairs. Blue circles denote 4,156 fragments from "
        "116 strain pairs with whole-genome average nucleotide identity (ANI) ≥99%, "
        "and grey crosses denote the remaining 493 fragments. Identical fragment "
        "pairs are omitted, and no binned "
        "summary or fitted trend line is shown. The logarithmic y-axis starts at "
        "1e-8, below the smallest observed variable-fragment distance. Spearman "
        "correlations were 0.695 across all variable fragments and 0.714 within the "
        "ANI ≥99% subset. "
        "**b,** Five-fold held-out AUPRC for simple linear readouts of AMR-associated "
        "and mobile-element-associated fragment annotations; all fragments from each "
        "genome were kept in the same fold. Points denote individual folds, diamonds "
        "the fold mean ± sample s.d., and dashed lines the evaluation-set prevalence. "
        "Pooled out-of-fold AUPRC was 0.203 and 0.446, respectively. **c,** AUROC for "
        "the same held-out predictions. Points and diamonds are defined as in **b**; "
        "the dashed line denotes random AUROC of 0.5. Pooled out-of-fold AUROC was "
        "0.578 and 0.741, respectively. Labels were derived from existing GenBank "
        "annotations and do not constitute complete resistome or mobile-element "
        "catalogues.\n"
    )


def normalize_svg_whitespace(path: Path) -> None:
    """Remove generator-added line-end spaces for clean Git diffs."""

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    fragment_dir = args.fragment_dir.resolve()
    probe_dir = args.probe_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_summary_path = fragment_dir / "summary.json"
    probe_summary_path = probe_dir / "summary.json"
    fragment_summary = json.loads(fragment_summary_path.read_text(encoding="utf-8"))
    probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    maximum_divergence = float(
        fragment_summary["protocol"]["maximum_global_sequence_divergence"]
    )
    fragment_input_path = fragment_dir / "fragment_pairs.csv"
    fragments = prepare_fragment_plot_data(
        pd.read_csv(fragment_input_path), maximum_divergence=maximum_divergence
    )
    folds, pooled = prepare_probe_plot_data(probe_dir, probe_summary)

    fragment_plotted_path = output_dir / "fragment_scatter_plotted_data.csv"
    probe_fold_path = output_dir / "probe_fold_plotted_data.csv"
    probe_summary_plotted_path = output_dir / "probe_summary_plotted_data.csv"
    fragments.to_csv(fragment_plotted_path, index=False)
    folds.to_csv(probe_fold_path, index=False)
    pooled.to_csv(probe_summary_plotted_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.35),
        gridspec_kw={"width_ratios": [1.55, 1.0, 1.0], "wspace": 0.42},
    )
    plot_fragment_panel(axes[0], fragments, fragment_summary)
    plot_probe_panel(axes[1], folds, pooled, metric="auprc")
    plot_probe_panel(axes[2], folds, pooled, metric="auroc")
    for label, axis in zip("abc", axes):
        axis.text(
            -0.16,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            color=INK,
        )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.89)

    stem = "genome_representation_validation"
    figure_paths = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=400, bbox_inches="tight")
        if suffix == "svg":
            normalize_svg_whitespace(path)
        figure_paths.append(path)
    panel_a_y_limits = [float(value) for value in axes[0].get_ylim()]
    plt.close(figure)

    caption_path = output_dir / f"{stem}_caption.md"
    caption_path.write_text(build_caption(), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "figure": stem,
        "panel_a_y_scale": "log",
        "panel_a_y_limits": panel_a_y_limits,
        "panel_a_observed_cosine_distance_range": [
            float(fragments["cosine_distance"].min()),
            float(fragments["cosine_distance"].max()),
        ],
        "panel_contract": {
            "a": (
                "4,649 variable homologous 11-kb fragment pairs; cosine distance "
                "shown on a log scale from 1e-8; ANI >=99% fragments are blue "
                "circles and the remaining fragments are grey crosses; no binned "
                "summaries or fitted trend lines"
            ),
            "b": (
                "five held-out folds per annotation, with every genome kept entirely "
                "within one fold; small points are "
                "fold AUPRC, diamonds/error bars are fold mean +/- sample SD, and "
                "short dashed lines are evaluation-set prevalence"
            ),
            "c": (
                "five held-out folds per annotation, with every genome kept entirely "
                "within one fold; small points are "
                "fold AUROC, diamonds/error bars are fold mean +/- sample SD, and "
                "the dashed line is random AUROC=0.5"
            ),
        },
        "source_sha256": {
            "fragment_pairs": sha256_file(fragment_input_path),
            "fragment_summary": sha256_file(fragment_summary_path),
            "probe_summary": sha256_file(probe_summary_path),
            "amr_fold_metrics": sha256_file(
                probe_dir / "amr_associated_probe_fold_metrics.csv"
            ),
            "mge_fold_metrics": sha256_file(
                probe_dir / "mge_associated_probe_fold_metrics.csv"
            ),
            "plot_script": sha256_file(Path(__file__).resolve()),
        },
        "plotted_data_sha256": {
            path.name: sha256_file(path)
            for path in (
                fragment_plotted_path,
                probe_fold_path,
                probe_summary_plotted_path,
            )
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in (*figure_paths, caption_path)
        },
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
