"""Paired uncertainty and randomization tests for binary-classification scores."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


Metric = Callable[[np.ndarray, np.ndarray], float]
METRICS: dict[str, Metric] = {
    "auprc": average_precision_score,
    "auroc": roc_auc_score,
}


def align_predictions(
    candidate_path: Path,
    baseline_path: Path,
    *,
    candidate_may_be_superset: bool = False,
) -> pd.DataFrame:
    required = {"molecule_id", "label", "prediction"}
    frames = []
    for name, path in (("candidate", candidate_path), ("baseline", baseline_path)):
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["molecule_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate molecule IDs")
        frames.append(
            frame[["molecule_id", "label", "prediction"]].rename(
                columns={
                    "label": f"label_{name}",
                    "prediction": f"prediction_{name}",
                }
            )
        )
    candidate_ids = set(frames[0]["molecule_id"])
    baseline_ids = set(frames[1]["molecule_id"])
    missing_candidate_ids = baseline_ids - candidate_ids
    excluded_candidate_ids = sorted(candidate_ids - baseline_ids)
    if missing_candidate_ids:
        raise ValueError(
            f"Candidate is missing {len(missing_candidate_ids)} baseline molecule IDs"
        )
    if excluded_candidate_ids and not candidate_may_be_superset:
        raise ValueError("Candidate and baseline molecule-ID sets differ")
    merged = frames[0].merge(
        frames[1], on="molecule_id", how="inner", validate="one_to_one"
    )
    if not np.array_equal(
        merged["label_candidate"].to_numpy(), merged["label_baseline"].to_numpy()
    ):
        raise ValueError("Candidate and baseline labels differ after ID alignment")
    aligned = merged.rename(columns={"label_candidate": "label"}).drop(
        columns="label_baseline"
    )
    aligned.attrs["excluded_candidate_molecule_ids"] = excluded_candidate_ids
    return aligned


def _stratified_resample_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    return np.concatenate(
        (
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        )
    )


def paired_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: Metric,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    rng = np.random.default_rng(seed)
    values = {"candidate": [], "baseline": [], "difference": []}
    for _ in range(iterations):
        indices = _stratified_resample_indices(labels, rng)
        candidate_value = float(metric(labels[indices], candidate[indices]))
        baseline_value = float(metric(labels[indices], baseline[indices]))
        values["candidate"].append(candidate_value)
        values["baseline"].append(baseline_value)
        values["difference"].append(candidate_value - baseline_value)
    return values


def paired_randomization_test(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: Metric,
    iterations: int,
    seed: int,
) -> float:
    """Two-sided paired prediction-swap test with a finite-sample correction."""

    if iterations < 1:
        raise ValueError("Permutation iterations must be positive")
    observed = float(metric(labels, candidate) - metric(labels, baseline))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(iterations):
        swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
        permuted_candidate = np.where(swap, baseline, candidate)
        permuted_baseline = np.where(swap, candidate, baseline)
        difference = float(
            metric(labels, permuted_candidate)
            - metric(labels, permuted_baseline)
        )
        exceedances += abs(difference) >= abs(observed)
    return float((exceedances + 1) / (iterations + 1))


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    ]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""

    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        current = min(1.0, (count - rank) * float(p_values[original_index]))
        running = max(running, current)
        adjusted[original_index] = running
    return adjusted.tolist()


def analyze_comparison(
    candidate_path: Path,
    baseline_path: Path,
    *,
    metric_name: str,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    candidate_may_be_superset: bool = False,
) -> dict:
    frame = align_predictions(
        candidate_path,
        baseline_path,
        candidate_may_be_superset=candidate_may_be_superset,
    )
    labels = frame["label"].to_numpy(dtype=int)
    candidate = frame["prediction_candidate"].to_numpy(dtype=float)
    baseline = frame["prediction_baseline"].to_numpy(dtype=float)
    metric = METRICS[metric_name]
    candidate_value = float(metric(labels, candidate))
    baseline_value = float(metric(labels, baseline))
    bootstrap = paired_bootstrap(
        labels,
        candidate,
        baseline,
        metric=metric,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    return {
        "metric": metric_name,
        "num_examples": int(len(frame)),
        "num_positive": int(labels.sum()),
        "excluded_candidate_molecule_ids": frame.attrs[
            "excluded_candidate_molecule_ids"
        ],
        "candidate": candidate_value,
        "baseline": baseline_value,
        "difference": candidate_value - baseline_value,
        "candidate_95ci": percentile_interval(bootstrap["candidate"]),
        "baseline_95ci": percentile_interval(bootstrap["baseline"]),
        "difference_95ci": percentile_interval(bootstrap["difference"]),
        "paired_prediction_swap_p": paired_randomization_test(
            labels,
            candidate,
            baseline,
            metric=metric,
            iterations=permutation_iterations,
            seed=seed + 1,
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
    }


def _analyze_task(arguments: dict) -> dict:
    return analyze_comparison(**arguments)


def run_config(
    config_path: Path,
    *,
    repo_root: Path,
    output_path: Path,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    workers: int = 1,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = []
    metadata = []
    for comparison_index, comparison in enumerate(config["comparisons"]):
        candidate = Path(comparison["candidate_predictions"])
        baseline = Path(comparison["baseline_predictions"])
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not baseline.is_absolute():
            baseline = repo_root / baseline
        for metric_name in comparison.get("metrics", ["auprc", "auroc"]):
            tasks.append(
                {
                    "candidate_path": candidate,
                    "baseline_path": baseline,
                    "metric_name": metric_name,
                    "bootstrap_iterations": bootstrap_iterations,
                    "permutation_iterations": permutation_iterations,
                    "seed": seed + 10 * comparison_index,
                    "candidate_may_be_superset": bool(
                        comparison.get("candidate_may_be_superset", False)
                    ),
                }
            )
            metadata.append(
                {
                    "comparison": comparison["name"],
                    "family": comparison["family"],
                    "group": int(comparison["group"]),
                    "candidate_name": comparison["candidate_name"],
                    "baseline_name": comparison["baseline_name"],
                    "candidate_predictions": str(candidate),
                    "baseline_predictions": str(baseline),
                }
            )
    if workers == 1:
        analyzed = [_analyze_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            analyzed = list(executor.map(_analyze_task, tasks))
    results = []
    for result, item_metadata in zip(analyzed, metadata, strict=True):
        result.update(item_metadata)
        results.append(result)
    families = sorted({item["family"] for item in results})
    for family in families:
        for metric_name in METRICS:
            selected = [
                item
                for item in results
                if item["family"] == family and item["metric"] == metric_name
            ]
            adjusted = holm_adjust(
                [item["paired_prediction_swap_p"] for item in selected]
            )
            for item, value in zip(selected, adjusted, strict=True):
                item["holm_adjusted_p_within_family_and_metric"] = value
    report = {
        "schema_version": 1,
        "inference": (
            "paired stratified bootstrap confidence intervals and two-sided "
            "paired prediction-swap randomization tests"
        ),
        "multiple_testing": "Holm correction within each model-mode and metric family",
        "workers": workers,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--permutation-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent metric comparisons to evaluate concurrently.",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    report = run_config(
        config,
        repo_root=root,
        output_path=output,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
