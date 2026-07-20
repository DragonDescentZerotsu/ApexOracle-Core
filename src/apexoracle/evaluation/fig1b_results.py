"""Assemble provenance-aware Fig. 1b fold/member predictions into OOF tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_source(path: Path, source_index: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = pd.read_csv(path)
    required = {"molecule_id", "label", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["molecule_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate molecule IDs")
    frame = frame.sort_values("molecule_id").reset_index(drop=True)
    member_columns = [
        column for column in frame.columns if column.startswith("prediction_ensemble_")
    ]
    if member_columns:
        members = {
            column.removeprefix("prediction_"): frame[column].to_numpy(dtype=float)
            for column in member_columns
        }
    else:
        members = {
            f"source_{source_index}": frame["prediction"].to_numpy(dtype=float)
        }
    return frame, members


def combine_fold_sources(
    paths: Sequence[Path], *, group: int, fold: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not paths:
        raise ValueError(f"No prediction sources for group {group}, fold {fold}")
    reference: pd.DataFrame | None = None
    members: dict[str, np.ndarray] = {}
    source_paths = []
    for source_index, path in enumerate(paths):
        frame, source_members = _load_source(path, source_index)
        if reference is None:
            reference = frame[["molecule_id", "label"]]
        elif not reference.equals(frame[["molecule_id", "label"]]):
            raise ValueError(f"Prediction IDs or labels differ in {path}")
        for name, values in source_members.items():
            unique_name = name
            if unique_name in members:
                if np.array_equal(members[unique_name], values):
                    continue
                unique_name = f"{name}_source_{source_index}"
            members[unique_name] = values
        source_paths.append(str(path))
    assert reference is not None
    output = reference.copy()
    output["prediction"] = np.mean(np.stack(list(members.values())), axis=0)
    output["group"] = group
    output["fold"] = fold
    report = {
        "group": group,
        "fold": fold,
        "num_examples": int(len(output)),
        "num_members": len(members),
        "member_names": sorted(members),
        "sources": source_paths,
        "metrics": {
            "auroc": float(roc_auc_score(output["label"], output["prediction"])),
            "auprc": float(
                average_precision_score(output["label"], output["prediction"])
            ),
        },
    }
    return output, report


def assemble_config(config_path: Path, *, repo_root: Path, output_root: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    reports = []
    for group_config in config["groups"]:
        group = int(group_config["group"])
        group_dir = output_root / f"group_{group}"
        group_dir.mkdir(parents=True, exist_ok=True)
        folds = []
        fold_reports = []
        for fold_config in group_config["folds"]:
            fold = int(fold_config["fold"])
            paths = []
            for value in fold_config["sources"]:
                path = Path(value)
                paths.append(path if path.is_absolute() else repo_root / path)
            combined, report = combine_fold_sources(paths, group=group, fold=fold)
            combined.to_csv(group_dir / f"fold_{fold}_predictions.csv", index=False)
            folds.append(combined)
            fold_reports.append(report)
        oof_all = pd.concat(folds, ignore_index=True)
        if oof_all["molecule_id"].duplicated().any():
            raise ValueError(f"Group {group} OOF table contains duplicate IDs")
        all_eligible_metrics = {
            "auroc": float(roc_auc_score(oof_all["label"], oof_all["prediction"])),
            "auprc": float(
                average_precision_score(oof_all["label"], oof_all["prediction"])
            ),
        }
        oof = oof_all
        exclusions: list[str] = []
        reference_value = group_config.get("reference_predictions")
        if reference_value is not None:
            reference_path = Path(reference_value)
            if not reference_path.is_absolute():
                reference_path = repo_root / reference_path
            reference = pd.read_csv(reference_path)[["molecule_id", "label"]]
            if reference["molecule_id"].duplicated().any():
                raise ValueError(f"{reference_path} contains duplicate molecule IDs")
            missing = set(reference["molecule_id"]) - set(oof_all["molecule_id"])
            if missing:
                raise ValueError(
                    f"Group {group} is missing {len(missing)} reference molecule IDs"
                )
            exclusions = sorted(set(oof_all["molecule_id"]) - set(reference["molecule_id"]))
            oof = reference.merge(
                oof_all,
                on="molecule_id",
                how="left",
                validate="one_to_one",
                suffixes=("_reference", ""),
            )
            if oof["prediction"].isna().any():
                raise ValueError(f"Group {group} produced missing aligned predictions")
            if not np.array_equal(oof["label_reference"], oof["label"]):
                raise ValueError(f"Group {group} reference labels differ")
            oof = oof.drop(columns="label_reference")
        if exclusions:
            oof_all.to_csv(group_dir / "oof_predictions_all_eligible.csv", index=False)
        oof.to_csv(group_dir / "oof_predictions.csv", index=False)
        aligned_fold_metrics = [
            {
                "auroc": float(roc_auc_score(frame["label"], frame["prediction"])),
                "auprc": float(
                    average_precision_score(frame["label"], frame["prediction"])
                ),
            }
            for _, frame in oof.groupby("fold", sort=True)
        ]
        report = {
            "group": group,
            "folds": fold_reports,
            "num_examples": int(len(oof)),
            "num_examples_all_eligible": int(len(oof_all)),
            "excluded_molecule_ids": exclusions,
            "num_members_by_fold": {
                str(item["fold"]): item["num_members"] for item in fold_reports
            },
            "pooled_oof_metrics": {
                "auroc": float(roc_auc_score(oof["label"], oof["prediction"])),
                "auprc": float(
                    average_precision_score(oof["label"], oof["prediction"])
                ),
            },
            "pooled_oof_metrics_all_eligible": all_eligible_metrics,
            "fold_mean": {
                metric: float(
                    np.mean([item[metric] for item in aligned_fold_metrics])
                )
                for metric in ("auroc", "auprc")
            },
            "fold_sample_sd": {
                metric: float(
                    np.std(
                        [item[metric] for item in aligned_fold_metrics], ddof=1
                    )
                )
                for metric in ("auroc", "auprc")
            },
            "fold_mean_all_eligible": {
                metric: float(
                    np.mean([item["metrics"][metric] for item in fold_reports])
                )
                for metric in ("auroc", "auprc")
            },
            "fold_sample_sd_all_eligible": {
                metric: float(
                    np.std(
                        [item["metrics"][metric] for item in fold_reports], ddof=1
                    )
                )
                for metric in ("auroc", "auprc")
            },
            "predictions": str(group_dir / "oof_predictions.csv"),
        }
        (group_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        reports.append(report)
    final = {
        "schema_version": 1,
        "description": config["description"],
        "groups": reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    return final


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    report = assemble_config(config, repo_root=root, output_root=output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
