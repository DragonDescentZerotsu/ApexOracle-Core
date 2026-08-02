#!/usr/bin/env python3
"""Aggregate ensemble predictions and cluster-bootstrap molecule-disjoint metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METRICS = ("r2", "spearman", "pearson")


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(labels) < 2:
        return {metric: float("nan") for metric in METRICS}
    denominator = np.square(labels - labels.mean()).sum()
    r2 = (
        float("nan")
        if denominator == 0
        else float(1.0 - np.square(labels - predictions).sum() / denominator)
    )
    if denominator == 0 or np.all(predictions == predictions[0]):
        spearman = float("nan")
        pearson = float("nan")
    else:
        spearman = float(spearmanr(labels, predictions).statistic)
        pearson = float(pearsonr(labels, predictions).statistic)
    return {
        "r2": r2,
        "spearman": spearman,
        "pearson": pearson,
    }


def load_ensemble(prediction_dir: Path, protocol: str, group: int, members: int) -> pd.DataFrame:
    frames = []
    for ensemble in range(members):
        path = prediction_dir / f"{protocol}_group_{group}_ensemble_{ensemble}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("row_key").reset_index(drop=True)
        if "training_mean_baseline_prediction" not in frame:
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame["training_mean_baseline_prediction"] = float(
                metadata["global_train_mean_z"]
            )
        frames.append(frame)
    reference = frames[0].drop(columns=["prediction", "ensemble"])
    predictions = []
    for ensemble, frame in enumerate(frames):
        comparable = frame.drop(columns=["prediction", "ensemble"])
        if list(reference.columns) != list(comparable.columns):
            raise ValueError(
                f"Metadata columns mismatch in group {group}, ensemble {ensemble}"
            )
        for column in reference.columns:
            left = reference[column]
            right = comparable[column]
            matches = (
                np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                )
                if pd.api.types.is_numeric_dtype(left)
                else left.equals(right)
            )
            if not matches:
                raise ValueError(
                    f"Metadata mismatch in group {group}, ensemble {ensemble}, "
                    f"column {column}"
                )
        predictions.append(frame["prediction"].to_numpy(dtype=float))
    output = reference.copy()
    output["prediction"] = np.mean(predictions, axis=0)
    output["ensemble_members"] = members
    return output


def cohort_frame(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "full":
        return frame
    seen = frame["train_seen_molecule"].astype(bool)
    if cohort == "train_seen":
        return frame.loc[seen]
    if cohort == "train_unseen":
        return frame.loc[~seen]
    raise ValueError(cohort)


def metric_rows(frame: pd.DataFrame, protocol: str, group_label: str) -> list[dict]:
    rows = []
    high_mic_z = -np.log10(512.0 / 10.0)
    for cohort in ("full", "train_seen", "train_unseen"):
        selected = cohort_frame(frame, cohort)
        labels = selected["label_z"].to_numpy(dtype=float)
        predictors = {
            "apexoracle_ensemble": selected["prediction"].to_numpy(dtype=float),
            "peptide_mean_or_train_mean": selected[
                "peptide_mean_baseline_prediction"
            ].to_numpy(dtype=float),
            "training_mean_constant": selected[
                "training_mean_baseline_prediction"
            ].to_numpy(dtype=float),
            "always_512um_constant": np.full(len(selected), high_mic_z),
        }
        for model, predictions in predictors.items():
            row = {
                "protocol": protocol,
                "group": group_label,
                "cohort": cohort,
                "model": model,
                "measurements": len(selected),
                "unique_molecules": selected["molecule_identity"].nunique(),
                "pathogens": selected["strain_name"].nunique(),
                "low_mic_16um_measurements": int((selected["MIC_um"] <= 16).sum()),
                "low_mic_16um_fraction": float(
                    (selected["MIC_um"] <= 16).mean()
                ) if len(selected) else float("nan"),
            }
            row.update(calculate_metrics(labels, predictions))
            rows.append(row)
    return rows


def clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    clusters = {
        identity: indices.to_numpy()
        for identity, indices in frame.groupby("molecule_identity").groups.items()
    }
    identities = np.asarray(list(clusters), dtype=object)
    if len(identities) < 2:
        return []
    model_samples = {metric: [] for metric in METRICS}
    baseline_samples = {metric: [] for metric in METRICS}
    label_values = frame["label_z"].to_numpy(dtype=float)
    model_values = frame["prediction"].to_numpy(dtype=float)
    baseline_values = frame["peptide_mean_baseline_prediction"].to_numpy(dtype=float)
    for _ in range(iterations):
        selected_ids = rng.choice(identities, size=len(identities), replace=True)
        indices = np.concatenate([clusters[identity] for identity in selected_ids])
        model_metrics = calculate_metrics(label_values[indices], model_values[indices])
        baseline_metrics = calculate_metrics(
            label_values[indices], baseline_values[indices]
        )
        for metric in METRICS:
            model_samples[metric].append(model_metrics[metric])
            baseline_samples[metric].append(baseline_metrics[metric])
    rows = []
    for metric in METRICS:
        model = np.asarray(model_samples[metric], dtype=float)
        baseline = np.asarray(baseline_samples[metric], dtype=float)
        delta = model - baseline
        finite = np.isfinite(model) & np.isfinite(baseline)
        rows.append(
            {
                "metric": metric,
                "bootstrap_iterations": iterations,
                "finite_iterations": int(finite.sum()),
                "model_ci_low": float(np.nanquantile(model, 0.025)),
                "model_ci_high": float(np.nanquantile(model, 0.975)),
                "baseline_ci_low": float(np.nanquantile(baseline, 0.025)),
                "baseline_ci_high": float(np.nanquantile(baseline, 0.975)),
                "delta_model_minus_baseline": float(np.nanmean(delta)),
                "delta_ci_low": float(np.nanquantile(delta, 0.025)),
                "delta_ci_high": float(np.nanquantile(delta, 0.975)),
                "paired_bootstrap_probability_delta_le_zero": float(
                    np.mean(delta[finite] <= 0)
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["strain", "species", "phylum"], required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--members", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-all-cohorts",
        action="store_true",
        help="Bootstrap every group/cohort; default is the primary pooled train-unseen cohort.",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        load_ensemble(prediction_dir, args.protocol, group, args.members)
        for group in range(args.groups)
    ]
    ensemble = pd.concat(groups, ignore_index=True)
    ensemble.to_csv(output_dir / "ensemble_predictions.csv", index=False)
    rows = []
    for group, frame in enumerate(groups):
        rows.extend(metric_rows(frame, args.protocol, str(group)))
    rows.extend(metric_rows(ensemble, args.protocol, "all_groups_pooled"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    bootstrap_rows = []
    bootstrap_scopes = (
        [
            (group_label, cohort, frame)
            for group_label, frame in [
                *[(str(group), frame) for group, frame in enumerate(groups)],
                ("all_groups_pooled", ensemble),
            ]
            for cohort in ("full", "train_seen", "train_unseen")
        ]
        if args.bootstrap_all_cohorts
        else [("all_groups_pooled", "train_unseen", ensemble)]
    )
    for group_label, cohort, frame in bootstrap_scopes:
        selected = cohort_frame(frame, cohort).reset_index(drop=True)
        for row in clustered_bootstrap(
            selected,
            iterations=args.bootstrap_iterations,
            seed=args.seed + len(bootstrap_rows),
        ):
            row.update(
                {
                    "protocol": args.protocol,
                    "group": group_label,
                    "cohort": cohort,
                    "measurements": len(selected),
                    "unique_molecules": selected["molecule_identity"].nunique(),
                }
            )
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "cluster_bootstrap.csv", index=False)
    report = {
        "schema_version": 1,
        "status": "completed",
        "protocol": args.protocol,
        "groups": args.groups,
        "ensemble_members": args.members,
        "bootstrap": {
            "unit": "exact stored-token molecule identity",
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "method": "paired cluster bootstrap retaining all MIC rows per sampled molecule",
            "scope": (
                "all group/cohort combinations"
                if args.bootstrap_all_cohorts
                else "primary pooled train-unseen cohort"
            ),
        },
        "outputs": {
            "ensemble_predictions": str(output_dir / "ensemble_predictions.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "cluster_bootstrap": str(output_dir / "cluster_bootstrap.csv"),
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
