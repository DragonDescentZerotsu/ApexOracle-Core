#!/usr/bin/env python3
"""Recompute the reported Fig. 2b mean R² and sample SD from frozen fold metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_MODELS = {
    "dlm_mtr_dlm",
    "chemberta_mtr",
    "apex",
    "peptideclm",
    "dlm_only",
    "molformer",
    "chemberta_mlm",
}


def load_fold_metrics(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["model"]].append(
                {
                    "fold": int(row["fold"]),
                    "best_epoch": int(row["best_epoch"]),
                    "best_r2_mean": float(row["best_r2_mean"]),
                    "train_size": int(row["train_size"]),
                    "test_size": int(row["test_size"]),
                }
            )
    if set(grouped) != EXPECTED_MODELS:
        raise ValueError(f"Unexpected model set: {sorted(grouped)}")
    for model, rows in grouped.items():
        if sorted(row["fold"] for row in rows) != [1, 2, 3, 4, 5]:
            raise ValueError(f"{model} does not contain exactly folds 1-5")
        if sum(row["test_size"] for row in rows) != 10886:
            raise ValueError(f"{model} test folds do not cover 10,886 molecules")
    return grouped


def recompute(grouped: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        model: {
            "fold_r2": [row["best_r2_mean"] for row in rows],
            "mean_r2": statistics.mean(row["best_r2_mean"] for row in rows),
            "sample_sd": statistics.stdev(row["best_r2_mean"] for row in rows),
        }
        for model, rows in grouped.items()
    }


def verify_published_summary(recomputed: dict[str, dict], summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["protocol"] != "fig2b-shared-native-intersection-v2":
        raise ValueError("Unexpected Fig. 2b protocol")
    if summary["missing"]:
        raise ValueError(f"Published summary records missing jobs: {summary['missing']}")
    for record in summary["summary"]:
        values = recomputed[record["model"]]
        if abs(values["mean_r2"] - record["shared_mean_r2"]) > 1e-12:
            raise ValueError(f"Mean R² mismatch for {record['model']}")
        if abs(values["sample_sd"] - record["shared_sample_sd"]) > 1e-12:
            raise ValueError(f"Sample SD mismatch for {record['model']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    source = args.data_root / "fig2b_shared_original_protocol"
    grouped = load_fold_metrics(source / "fold_metrics.csv")
    recomputed = recompute(grouped)
    verify_published_summary(recomputed, source / "comparison_summary.json")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "fig2b_shared_metrics_verified.json"
    output.write_text(
        json.dumps(
            {
                "protocol": "fig2b-shared-native-intersection-v2",
                "shared_molecules": 10886,
                "folds_per_model": 5,
                "models": recomputed,
                "published_summary_verified": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
