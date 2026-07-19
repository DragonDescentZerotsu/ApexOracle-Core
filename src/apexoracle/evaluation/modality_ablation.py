"""Paper-figure contract for the ApexOracle modality ablation.

This module reproduces the published plot from its frozen result table.  It
does not claim to recompute those values from the incomplete legacy checkpoint
families.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_COLUMNS = ("holdout", "series", "r2")


@dataclass(frozen=True)
class ModalityAblationRecord:
    holdout: str
    series: str
    r2: float


@dataclass(frozen=True)
class ModalityAblationPlotConfig:
    holdout_order: tuple[str, ...]
    series_order: tuple[str, ...]
    base_colors: tuple[str, ...]
    markers: Mapping[str, str]
    marker_sizes: Mapping[str, float]
    figure_size: tuple[float, float]
    title: str

    @classmethod
    def load(cls, path: Path) -> "ModalityAblationPlotConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        chart = raw["chart"]
        config = cls(
            holdout_order=tuple(chart["holdout_order"]),
            series_order=tuple(chart["series_order"]),
            base_colors=tuple(chart["base_colors"]),
            markers=dict(chart["markers"]),
            marker_sizes={
                key: float(value) for key, value in chart["marker_sizes"].items()
            },
            figure_size=tuple(float(value) for value in chart["figure_size"]),
            title=str(chart["title"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.holdout_order) != len(set(self.holdout_order)):
            raise ValueError("holdout_order contains duplicates")
        if len(self.series_order) != len(set(self.series_order)):
            raise ValueError("series_order contains duplicates")
        if len(self.base_colors) != len(self.series_order):
            raise ValueError("base_colors must have one entry per series")
        if set(self.markers) != set(self.series_order):
            raise ValueError("markers must cover exactly series_order")
        if set(self.marker_sizes) != set(self.series_order):
            raise ValueError("marker_sizes must cover exactly series_order")
        if len(self.figure_size) != 2 or any(value <= 0 for value in self.figure_size):
            raise ValueError("figure_size must contain two positive values")


def load_records(path: Path) -> tuple[ModalityAblationRecord, ...]:
    """Load and validate the frozen paper values."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Expected CSV columns {EXPECTED_COLUMNS}, got {reader.fieldnames}"
            )
        records = tuple(
            ModalityAblationRecord(
                holdout=str(row["holdout"]),
                series=str(row["series"]),
                r2=float(row["r2"]),
            )
            for row in reader
        )
    if not records:
        raise ValueError("The modality-ablation value table is empty")
    if any(not math.isfinite(record.r2) for record in records):
        raise ValueError("All R2 values must be finite")
    keys = [(record.holdout, record.series) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate holdout/series rows in value table")
    return records


def validate_complete_grid(
    records: Sequence[ModalityAblationRecord],
    config: ModalityAblationPlotConfig,
) -> None:
    expected = {
        (holdout, series)
        for series in config.series_order
        for holdout in config.holdout_order
    }
    observed = {(record.holdout, record.series) for record in records}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Incomplete value grid; missing={missing}, extra={extra}")


def ordered_values(
    records: Sequence[ModalityAblationRecord],
    config: ModalityAblationPlotConfig,
) -> tuple[tuple[float, ...], ...]:
    """Return values ordered as series × holdout, matching the notebook cell."""

    validate_complete_grid(records, config)
    lookup = {(record.holdout, record.series): record.r2 for record in records}
    return tuple(
        tuple(lookup[(holdout, series)] for holdout in config.holdout_order)
        for series in config.series_order
    )


def plot_modality_ablation(
    records: Sequence[ModalityAblationRecord],
    config: ModalityAblationPlotConfig,
    *,
    pdf_path: Path,
    png_path: Path | None = None,
) -> None:
    """Render the paper plot without reading or modifying experimental data."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import pandas as pd
    import seaborn as sns

    values = ordered_values(records, config)
    table = pd.DataFrame(
        {
            "holdout": list(config.holdout_order) * len(config.series_order),
            "r2": [value for series_values in values for value in series_values],
            "series": [
                series
                for series in config.series_order
                for _ in config.holdout_order
            ],
        }
    )
    sns.set_theme(style="whitegrid")
    color_map = LinearSegmentedColormap.from_list(
        "modality_ablation", config.base_colors, N=256
    )
    palette = color_map(np.linspace(0, 1, len(config.series_order)))

    figure, axis = plt.subplots(figsize=config.figure_size)
    for series, color in zip(config.series_order, palette):
        subset = table[table["series"] == series]
        sns.lineplot(
            data=subset,
            x="holdout",
            y="r2",
            marker=config.markers[series],
            markersize=config.marker_sizes[series],
            linewidth=1.4,
            color=color,
            label=series,
            ax=axis,
        )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.6)
    axis.grid(axis="x", linestyle="--", alpha=0.35, linewidth=1.6)
    axis.text(
        -0.15,
        0.5,
        r"$R^2$",
        transform=axis.transAxes,
        fontsize=11,
        verticalalignment="top",
    )
    axis.set_title(config.title, fontsize=14)
    axis.legend(loc="upper left", frameon=False, facecolor="none", edgecolor="none")
    figure.tight_layout()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen paper modality-ablation plot."
    )
    parser.add_argument(
        "--values",
        type=Path,
        default=Path("experiments/modality_ablation/paper_values.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/modality_ablation/paper_plot.json"),
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    values_path = args.values.resolve()
    config_path = args.config.resolve()
    output_paths = [args.output_pdf.resolve()]
    if args.output_png is not None:
        output_paths.append(args.output_png.resolve())
    if values_path in output_paths or config_path in output_paths:
        raise ValueError("Output paths must not overwrite the value table or config")
    records = load_records(args.values)
    config = ModalityAblationPlotConfig.load(args.config)
    plot_modality_ablation(
        records,
        config,
        pdf_path=args.output_pdf,
        png_path=args.output_png,
    )


if __name__ == "__main__":
    main()
