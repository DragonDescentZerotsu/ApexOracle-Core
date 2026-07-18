#!/usr/bin/env python3
"""Aggregate shared-data Fig. 2b folds and compare them with original runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = (
    "dlm_mtr_dlm",
    "chemberta_mtr",
    "dlm_only",
    "chemberta_mlm",
    "peptideclm",
    "molformer",
    "apex",
)
DISPLAY_NAMES = {
    "dlm_mtr_dlm": "DLM MTR+DLM",
    "chemberta_mtr": "ChemBERTa MTR",
    "dlm_only": "DLM MLM",
    "chemberta_mlm": "ChemBERTa MLM",
    "peptideclm": "PeptideCLM",
    "molformer": "MolFormer",
    "apex": "APEX",
}

# The five non-DLM values come from faithful native-retained-set reruns already
# saved in Checkpoints/fig2b_baselines_online_5fold, except APEX which uses the
# original-dropout seed-0 run matching the paper bar. DLM-only is reconstructed
# from its five archived log maxima; MTR+DLM uses the archived cached rerun.
ORIGINAL_RERUN_REFERENCE = {
    "dlm_mtr_dlm": 0.5207119213907342,
    "chemberta_mtr": 0.41969915440208033,
    "dlm_only": 0.40834,
    "chemberta_mlm": 0.23017377853393556,
    "peptideclm": 0.37667842821071024,
    "molformer": 0.3726220676773473,
    "apex": 0.4049862874181647,
}
PAPER_FIGURE_APPROX = {
    "dlm_mtr_dlm": 0.530,
    "chemberta_mtr": 0.417,
    "dlm_only": 0.408,
    "chemberta_mlm": 0.226,
    "peptideclm": 0.376,
    "molformer": 0.371,
    "apex": 0.403,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/fig2b_shared_original_protocol"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_fold_results(results_dir: Path, model: str) -> list[dict]:
    if model in {"dlm_mtr_dlm", "dlm_only"}:
        paths = [results_dir / model / f"fold_{fold}" / "metrics.json" for fold in range(1, 6)]
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]

    rows = []
    for fold in range(1, 6):
        path = results_dir / "jobs" / f"{model}_fold_{fold}" / model / "metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("folds", [])) != 1:
            raise ValueError(f"expected exactly one fold in {path}")
        rows.append(payload["folds"][0])
    return rows


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summaries = []
    fold_rows = []
    missing = []
    for model in MODEL_ORDER:
        folds = sorted(load_fold_results(results_dir, model), key=lambda row: int(row["fold"]))
        observed = {int(row["fold"]) for row in folds}
        if observed != set(range(1, 6)):
            missing.append({"model": model, "observed_folds": sorted(observed)})
            if not args.allow_incomplete:
                continue
        scores = [float(row["best_r2_mean"]) for row in folds]
        if not scores:
            continue
        for row in folds:
            fold_rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "fold": int(row["fold"]),
                    "best_epoch": row.get("best_epoch", row.get("epoch")),
                    "best_r2_mean": float(row["best_r2_mean"]),
                    "train_size": int(row["train_size"]),
                    "test_size": int(row["test_size"]),
                }
            )
        shared_mean = float(np.mean(scores))
        original = ORIGINAL_RERUN_REFERENCE[model]
        paper = PAPER_FIGURE_APPROX[model]
        summaries.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "completed_folds": len(scores),
                "shared_mean_r2": shared_mean,
                "shared_sample_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "original_rerun_r2": original,
                "absolute_change_vs_original_rerun": shared_mean - original,
                "relative_change_vs_original_rerun_percent": 100 * (shared_mean - original) / abs(original),
                "paper_figure_approx_r2": paper,
                "absolute_change_vs_paper_figure": shared_mean - paper,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(results_dir / "fold_metrics.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(results_dir / "comparison_summary.csv", index=False)
    payload = {
        "protocol": "fig2b-shared-native-intersection-v2",
        "shared_molecules": 10886,
        "summary": summaries,
        "missing": missing,
        "reference_notes": {
            "original_rerun": "native retained sets; source details are encoded in this script",
            "paper_figure": "approximate values read from the current Fig. 2b bars",
        },
    }
    (results_dir / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Fig. 2b 共同数据五折结果",
        "",
        "正式修订版只统一 10,886 个 molecule IDs 和五个 folds；模型、head、epoch、batch size、optimizer 与原 checkpoint-selection 行为保持不变。",
        "",
        "| 模型 | 新 R²（mean ± SD） | 原 retained-set rerun | 绝对变化 | 相对变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        sd = row["shared_sample_sd"]
        new_value = f"{row['shared_mean_r2']:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        lines.append(
            f"| {row['display_name']} | {new_value} | {row['original_rerun_r2']:.4f} | "
            f"{row['absolute_change_vs_original_rerun']:+.4f} | "
            f"{row['relative_change_vs_original_rerun_percent']:+.1f}% |"
        )
    if missing:
        lines.extend(("", "未完成：`" + json.dumps(missing, ensure_ascii=False) + "`"))
    (results_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if args.allow_incomplete or not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
