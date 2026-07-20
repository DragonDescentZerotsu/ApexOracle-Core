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
    "fine_tune_vs_baseline": "ApexOracle, fine-tuned (1 model/fold)",
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
                    "comparison_family": family,
                    "holm_p": item.get(
                        "holm_adjusted_p_within_family_and_metric"
                    ),
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
                "comparison_family": None,
                "holm_p": None,
            }
        )
    return pd.DataFrame(rows)


def format_adjusted_p(p_value: float) -> str:
    """Format the Holm-adjusted paired-test p value shown above a bracket."""

    if p_value < 1e-4:
        return r"Holm $p < 0.0001$"
    return rf"Holm $p = {p_value:.4f}$"


def draw_significance(
    axis,
    *,
    x1: float,
    x2: float,
    y_top: float,
    text: str,
    drop: float = 0.012,
    text_offset: float = 0.008,
) -> float:
    """Draw a Fig. 3a-style significance bracket and return its text top."""

    axis.plot(
        [x1, x1, x2, x2],
        [y_top - drop, y_top, y_top, y_top - drop],
        color="#222222",
        linewidth=1.1,
        solid_capstyle="butt",
        clip_on=False,
        zorder=4,
    )
    text_y = y_top + text_offset
    axis.text(
        (x1 + x2) / 2,
        text_y,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
    )
    return text_y


def plot_rows(frame: pd.DataFrame, output_prefix: Path) -> None:
    """Render PDF and PNG outputs with asymmetric bootstrap intervals."""

    import matplotlib.pyplot as plt

    methods = list(MODE_LABELS.values()) + [BASELINE_LABEL]
    colors = ["#FFFDD0", "#F7CFE1", "#B49EDE"]
    groups = list(TARGET_LABELS)
    width = 0.24
    x = np.arange(len(groups), dtype=float)
    fig, axis = plt.subplots(figsize=(10.4, 5.4))
    method_positions: dict[tuple[int, str], float] = {}
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
        method_positions.update(
            {
                (group, method): float(position)
                for group, position in zip(groups, positions)
            }
        )
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

    annotation_tops = []
    # Match the bracket convention used in the Fig. 3a plotting code. The
    # shorter zero-shot comparison is drawn first; the wider fine-tune
    # comparison is stacked above it. These are paired prediction-swap tests,
    # not independent-sample tests.
    family_levels = (
        ("strict_zero_shot_vs_baseline", 0),
        ("fine_tune_vs_baseline", 1),
    )
    for group in groups:
        group_frame = frame[frame["group"] == group]
        local_top = float(group_frame["ci_high"].max())
        for family, level in family_levels:
            candidate = group_frame[
                group_frame["comparison_family"] == family
            ]
            if len(candidate) != 1 or pd.isna(candidate.iloc[0]["holm_p"]):
                continue
            method = str(candidate.iloc[0]["method"])
            y_top = local_top + 0.045 + 0.085 * level
            annotation_tops.append(
                draw_significance(
                    axis,
                    x1=method_positions[(group, method)],
                    x2=method_positions[(group, BASELINE_LABEL)],
                    y_top=y_top,
                    text=format_adjusted_p(float(candidate.iloc[0]["holm_p"])),
                )
            )
    axis.set_xticks(x, [TARGET_LABELS[group] for group in groups])
    axis.tick_params(axis="both", labelsize=9)
    axis.set_ylabel("AUPRC", fontsize=10)
    axis.set_title("Small-molecule antibiotic classification", fontsize=12, pad=54)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    axis.tick_params(axis="x", length=0)
    plotted_top = max(annotation_tops, default=float(frame["ci_high"].max()))
    upper = min(1.0, plotted_top + 0.055)
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
