#!/usr/bin/env python3
"""Plot the MIC distribution of actual held-out hierarchical test measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "experiments/hierarchical_mic/fixed_strain_retrain/"
    "analysis/ensemble_predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path("experiments/hierarchical_mic/mic_distribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pooled and fold-level MIC distributions from the fixed "
            "strain-wise held-out prediction table."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-mic-threshold", type=float, default=16.0)
    parser.add_argument("--bin-width-log2", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_measurements(path: Path) -> pd.DataFrame:
    required = {
        "row_key",
        "protocol",
        "group_index",
        "group_name",
        "MIC_um",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in required)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if frame["row_key"].duplicated().any():
        raise ValueError("Expected one ensemble row per held-out measurement")
    if set(frame["protocol"]) != {"strain"}:
        raise ValueError(
            f"Expected only strain protocol rows, got {sorted(frame['protocol'].unique())}"
        )
    frame["MIC_um"] = pd.to_numeric(frame["MIC_um"], errors="raise")
    if (~np.isfinite(frame["MIC_um"])).any() or (frame["MIC_um"] <= 0).any():
        raise ValueError("MIC_um must contain only finite positive values")
    return frame.sort_values(["group_index", "row_key"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = [("Overall", frame)]
    cohorts.extend(
        (str(group_name), group)
        for (_, group_name), group in frame.groupby(
            ["group_index", "group_name"], sort=True
        )
    )
    for cohort, group in cohorts:
        mic = group["MIC_um"].to_numpy(dtype=float)
        low_count = int((mic <= threshold).sum())
        rows.append(
            {
                "cohort": cohort,
                "measurements": len(group),
                "low_mic_threshold_um": threshold,
                "low_mic_measurements": low_count,
                "low_mic_fraction": low_count / len(group),
                "minimum_um": float(np.min(mic)),
                "q1_um": float(np.quantile(mic, 0.25)),
                "median_um": float(np.median(mic)),
                "q3_um": float(np.quantile(mic, 0.75)),
                "maximum_um": float(np.max(mic)),
            }
        )
    return pd.DataFrame(rows)


def histogram_table(
    frame: pd.DataFrame, *, threshold: float, bin_width: float
) -> pd.DataFrame:
    log_mic = np.log2(frame["MIC_um"].to_numpy(dtype=float))
    lower = math.floor(float(log_mic.min()) / bin_width) * bin_width
    upper = math.ceil(float(log_mic.max()) / bin_width) * bin_width
    edges = np.arange(lower, upper + 1.5 * bin_width, bin_width)
    counts, _ = np.histogram(log_mic, bins=edges)
    return pd.DataFrame(
        {
            "bin_left_log2_um": edges[:-1],
            "bin_right_log2_um": edges[1:],
            "bin_left_um": np.power(2.0, edges[:-1]),
            "bin_right_um": np.power(2.0, edges[1:]),
            "measurements": counts,
            "fraction": counts / len(frame),
            "low_mic_threshold_um": threshold,
        }
    )


def _mic_tick_label(log2_value: float) -> str:
    value = 2.0**log2_value
    if value < 0.01:
        return f"{value:.3g}"
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:,.0f}"


def plot_distribution(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    threshold: float,
    bin_width: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    log_mic = np.log2(frame["MIC_um"].to_numpy(dtype=float))
    lower = math.floor(float(log_mic.min()) / bin_width) * bin_width
    upper = math.ceil(float(log_mic.max()) / bin_width) * bin_width
    edges = np.arange(lower, upper + 1.5 * bin_width, bin_width)
    threshold_log2 = math.log2(threshold)

    colors = ["#22577A", "#D97706", "#8A6D1D"]
    line_styles = ["-", "--", ":"]
    ink = "#20262E"
    grid = "#D9DEE5"
    low_fill = "#DCEAF3"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.25),
        gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=True,
    )

    weights = np.full(len(log_mic), 100.0 / len(log_mic))
    axes[0].hist(
        log_mic,
        bins=edges,
        weights=weights,
        color="#4F86A6",
        edgecolor="#214B65",
        linewidth=0.45,
    )
    axes[0].axvspan(lower, threshold_log2, color=low_fill, alpha=0.55, zorder=0)
    axes[0].axvline(
        threshold_log2,
        color=ink,
        linestyle="--",
        linewidth=1.3,
        label=f"Low-MIC threshold ({threshold:g} µM)",
    )
    overall = summary.loc[summary["cohort"] == "Overall"].iloc[0]
    axes[0].text(
        0.025,
        0.95,
        (
            f"n = {int(overall['measurements']):,}\n"
            f"MIC ≤ {threshold:g} µM: {overall['low_mic_fraction']:.2%}\n"
            f"Median: {overall['median_um']:.2f} µM"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=ink,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": grid,
            "alpha": 0.95,
        },
    )
    axes[0].set_title("Pooled eligible measurements", loc="center", weight="normal")
    axes[0].set_title("a", loc="left", weight="bold")
    axes[0].set_ylabel("Measurements per bin (%)")

    group_rows = summary.loc[summary["cohort"] != "Overall"].set_index("cohort")
    for index, ((_, group_name), group) in enumerate(
        frame.groupby(["group_index", "group_name"], sort=True)
    ):
        values = np.sort(np.log2(group["MIC_um"].to_numpy(dtype=float)))
        ecdf = np.arange(1, len(values) + 1) / len(values)
        low_fraction = float(group_rows.loc[str(group_name), "low_mic_fraction"])
        axes[1].plot(
            values,
            ecdf,
            color=colors[index],
            linestyle=line_styles[index],
            linewidth=2.0,
            label=(
                f"{group_name} (n={len(group):,}; "
                f"≤{threshold:g} µM={low_fraction:.2%})"
            ),
        )
        axes[1].plot(
            threshold_log2,
            low_fraction,
            marker="o",
            markersize=5.5,
            color=colors[index],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
    axes[1].axvline(
        threshold_log2, color=ink, linestyle="--", linewidth=1.3, zorder=0
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks(np.linspace(0, 1, 6))
    axes[1].set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    axes[1].set_title(
        "Fixed strain-wise test folds", loc="center", weight="normal"
    )
    axes[1].set_title("b", loc="left", weight="bold")
    axes[1].set_ylabel("Cumulative fraction of measurements")
    axes[1].legend(
        frameon=False,
        fontsize=8.2,
        loc="lower right",
        handlelength=2.8,
    )

    tick_start = math.ceil(lower / 4) * 4
    ticks = np.arange(tick_start, upper + 0.1, 4)
    if threshold_log2 not in ticks:
        ticks = np.sort(np.append(ticks, threshold_log2))
    for axis in axes:
        axis.set_xlim(lower, upper)
        axis.set_xticks(ticks)
        axis.set_xticklabels([_mic_tick_label(value) for value in ticks])
        axis.set_xlabel("MIC (µM; log₂ scale)")
        axis.grid(axis="y", color=grid, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#6B7280")
        axis.spines["bottom"].set_color("#6B7280")
        axis.tick_params(colors=ink, labelsize=8.5)

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.low_mic_threshold <= 0:
        raise ValueError("--low-mic-threshold must be positive")
    if args.bin_width_log2 <= 0:
        raise ValueError("--bin-width-log2 must be positive")

    frame = load_measurements(args.input_csv)
    summary = summarize(frame, args.low_mic_threshold)
    bins = histogram_table(
        frame,
        threshold=args.low_mic_threshold,
        bin_width=args.bin_width_log2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "mic_distribution_summary.csv"
    bins_path = args.output_dir / "mic_distribution_bins.csv"
    png_path = args.output_dir / "fixed_strain_test_mic_distribution.png"
    pdf_path = args.output_dir / "fixed_strain_test_mic_distribution.pdf"
    manifest_path = args.output_dir / "manifest.json"

    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    plot_distribution(
        frame,
        summary,
        threshold=args.low_mic_threshold,
        bin_width=args.bin_width_log2,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    manifest = {
        "schema_version": 1,
        "input_csv": str(args.input_csv),
        "input_size_bytes": args.input_csv.stat().st_size,
        "input_sha256": sha256_file(args.input_csv),
        "measurement_rows": len(frame),
        "low_mic_threshold_um": args.low_mic_threshold,
        "bin_width_log2": args.bin_width_log2,
        "outputs": {
            "summary_csv": str(summary_path),
            "bins_csv": str(bins_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
