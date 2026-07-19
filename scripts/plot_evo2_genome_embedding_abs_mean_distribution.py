#!/usr/bin/env python3
"""Plot per-genome mean absolute values for saved Evo-2 genome embeddings.

The default mode reproduces the rebuttal figure for the ApexOracle manuscript:
it resolves strains from the AMP MIC, small-molecule binary, and synergy FICI
tables to the saved genome embedding files, computes one mean(abs(E)) value per
matched genome embedding, and writes a PNG/PDF histogram plus a CSV summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import (  # noqa: E402
    compute_abs_mean_rows,
    genome_embedding_paths,
    matched_genome_ids,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot the distribution of per-genome mean absolute "
            "values for saved Evo-2 genome embeddings."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "DataPrepare" / "Data",
        help="Directory containing Genome_embs and strain data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "paper_figs",
        help="Directory for the generated PNG, PDF, and CSV.",
    )
    parser.add_argument(
        "--prefix",
        default="evo2_genome_embedding_abs_mean_distribution",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--all-embeddings",
        action="store_true",
        help="Use every file in Genome_embs instead of only dataset-matched genome IDs.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1e14,
        help="Scale factor shown on the secondary x-axis.",
    )
    return parser.parse_args()


def print_summary(table: pd.DataFrame, scale: float) -> None:
    quantiles = table["abs_mean"].quantile([0.05, 0.5, 0.95])
    std_quantiles = table["std"].quantile([0.05, 0.5, 0.95])
    print(f"n = {len(table)} genome embeddings")
    print(
        "abs_mean: "
        f"median={quantiles.loc[0.5]:.6e}, "
        f"p05={quantiles.loc[0.05]:.6e}, "
        f"p95={quantiles.loc[0.95]:.6e}"
    )
    print(
        "std: "
        f"median={std_quantiles.loc[0.5]:.6e}, "
        f"p05={std_quantiles.loc[0.05]:.6e}, "
        f"p95={std_quantiles.loc[0.95]:.6e}"
    )
    print(f"scaled median abs_mean with scale={scale:g}: {quantiles.loc[0.5] * scale:.6g}")


def plot_distribution(table: pd.DataFrame, output_dir: Path, prefix: str, scale: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    values = table["abs_mean"]
    q05 = float(values.quantile(0.05))
    median = float(values.quantile(0.50))
    q95 = float(values.quantile(0.95))
    x_values = values.to_numpy() / 1e-15

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.15), dpi=300)

    ax.hist(x_values, bins=28, color="#5B7FA4", edgecolor="white", linewidth=0.7, alpha=0.95)
    ax.axvspan(q05 / 1e-15, q95 / 1e-15, color="#F4A261", alpha=0.20, label="5th-95th percentile")
    ax.axvline(median / 1e-15, color="#C62828", linewidth=2.0, label="Median")
    ax.plot(
        x_values,
        [-0.7] * len(x_values),
        "|",
        color="#34495E",
        markersize=3.2,
        markeredgewidth=0.45,
        alpha=0.35,
        clip_on=False,
    )

    ax.set_title("Distribution of Evo-2 Genome Embedding Magnitudes")
    ax.set_xlabel(r"Per-genome mean absolute value, mean$(|E|)$ ($\times 10^{-15}$)")
    ax.set_ylabel("Genome embeddings")
    ax.set_xlim(1.35, 3.65)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    summary = (
        f"n = {len(values)} matched genome embeddings\n"
        f"median = {median / 1e-15:.2f} x 10^-15\n"
        f"5th-95th = {q05 / 1e-15:.2f}-{q95 / 1e-15:.2f} x 10^-15\n"
        f"after x10^14: median = {median * scale:.2f}"
    )
    ax.text(
        0.035,
        0.955,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.34", "facecolor": "white", "edgecolor": "#C9D1D9", "alpha": 0.96},
        linespacing=1.25,
    )

    top_scale = scale * 1e-15
    secax = ax.secondary_xaxis("top", functions=(lambda value: value * top_scale, lambda value: value / top_scale))
    secax.set_xlabel(r"Mean absolute value after fixed $10^{14}$ scaling")
    secax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

    fig.tight_layout(pad=1.0)
    png_path = output_dir / f"{prefix}.png"
    pdf_path = output_dir / f"{prefix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    emb_paths = genome_embedding_paths(args.data_dir / "Genome_embs")
    genome_ids = set(emb_paths) if args.all_embeddings else matched_genome_ids(args.data_dir, emb_paths)

    table = compute_abs_mean_rows(genome_ids, emb_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.prefix}.csv"
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    print_summary(table, args.scale)
    plot_distribution(table, args.output_dir, args.prefix, args.scale)


if __name__ == "__main__":
    main()
