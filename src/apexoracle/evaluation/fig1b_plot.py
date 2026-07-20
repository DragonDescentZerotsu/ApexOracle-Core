"""Create the revised three-strain AUPRC panel from the paired-analysis report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import json
import numpy as np
import pandas as pd


TARGET_LABELS = {
    0: r"$\it{E.\ coli}$ BW25113",
    1: r"$\it{A.\ baumannii}$ ATCC 17978",
    2: r"$\it{S.\ aureus}$ RN4220",
}
MODE_LABELS = {
    "fine_tune_vs_baseline": "ApexOracle, fine-tuned",
    "strict_zero_shot_vs_baseline": "ApexOracle, zero-shot",
}
BASELINE_LABEL = "Chemprop baseline (common folds)"
BASELINE_CI_FAMILY = "strict_zero_shot_vs_baseline"


def build_plot_rows(report: dict[str, Any]) -> pd.DataFrame:
    """Return one AUPRC row per target and displayed method."""

    by_key = {
        (item["family"], int(item["group"])): item
        for item in report["results"]
        if item["metric"] == "auprc"
    }
    rows = []
    for group, target_label in TARGET_LABELS.items():
        baseline_references: list[dict[str, Any]] = []
        for family, method_label in MODE_LABELS.items():
            try:
                item = by_key[(family, group)]
            except KeyError as error:
                raise ValueError(
                    f"Missing AUPRC result for family={family}, group={group}"
                ) from error
            rows.append(
                {
                    "group": group,
                    "target": target_label,
                    "method": method_label,
                    "value": item["candidate"],
                    "ci_low": item["candidate_95ci"][0],
                    "ci_high": item["candidate_95ci"][1],
                }
            )
            baseline_references.append(item)
        baseline_values = [item["baseline"] for item in baseline_references]
        if not np.allclose(baseline_values, baseline_values[0]):
            raise ValueError(f"Baseline point estimates disagree for group {group}")
        # The two paired comparisons use independent bootstrap random streams,
        # so their Monte Carlo interval endpoints need not be bit-identical even
        # though the baseline predictions and point estimate are identical. Use
        # the predeclared strict-zero-shot comparison as the display interval.
        baseline_reference = by_key[(BASELINE_CI_FAMILY, group)]
        rows.append(
            {
                "group": group,
                "target": target_label,
                "method": BASELINE_LABEL,
                "value": baseline_reference["baseline"],
                "ci_low": baseline_reference["baseline_95ci"][0],
                "ci_high": baseline_reference["baseline_95ci"][1],
            }
        )
    return pd.DataFrame(rows)


def plot_rows(frame: pd.DataFrame, output_prefix: Path) -> None:
    """Render PDF and PNG outputs with asymmetric bootstrap intervals."""

    import matplotlib.pyplot as plt

    methods = list(MODE_LABELS.values()) + [BASELINE_LABEL]
    colors = ["#FFFDD0", "#F7CFE1", "#B49EDE"]
    groups = list(TARGET_LABELS)
    width = 0.24
    x = np.arange(len(groups), dtype=float)
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    for method_index, (method, color) in enumerate(zip(methods, colors, strict=True)):
        subset = frame[frame["method"] == method].set_index("group").loc[groups]
        values = subset["value"].to_numpy(dtype=float)
        yerr = np.vstack(
            (
                values - subset["ci_low"].to_numpy(dtype=float),
                subset["ci_high"].to_numpy(dtype=float) - values,
            )
        )
        positions = x + (method_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width,
            color=color,
            edgecolor="white",
            label=method,
            yerr=yerr,
            capsize=3,
            error_kw={"linewidth": 1.1},
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=5,
            fontsize=9,
        )
    axis.set_xticks(x, [TARGET_LABELS[group] for group in groups])
    axis.tick_params(axis="both", labelsize=9)
    axis.set_ylabel("AUPRC", fontsize=10)
    axis.set_title("Small-molecule antibiotic classification", fontsize=12, pad=48)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    axis.tick_params(axis="x", length=0)
    upper = min(1.0, float(frame["ci_high"].max()) + 0.14)
    axis.set_ylim(0, upper)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    frame = build_plot_rows(report)
    plot_rows(frame, args.output_prefix)
    table = args.table or args.output_prefix.with_suffix(".csv")
    table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, index=False)
    return frame


if __name__ == "__main__":
    main()
