#!/usr/bin/env python
"""Summarize three classifier seeds and molecule bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    predicted = scores >= 0.0
    positive = labels == 1
    negative = ~positive
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
        float(np.mean(predicted[positive])),
        float(np.mean(~predicted[negative])),
    )


def one_bootstrap(
    seed: int,
    labels: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    sampled = np.r_[
        rng.choice(positive, len(positive), replace=True),
        rng.choice(negative, len(negative), replace=True),
    ]
    return (
        *metrics(labels[sampled], clean[sampled]),
        *metrics(labels[sampled], noisy[sampled]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    payloads = [np.load(path / "test_predictions.npz") for path in args.run_dir]
    molecules = payloads[0]["molecule_hash"]
    labels = payloads[0]["label"]
    for payload in payloads[1:]:
        if not np.array_equal(molecules, payload["molecule_hash"]):
            raise RuntimeError("Seed prediction molecule sets differ")
        if not np.array_equal(labels, payload["label"]):
            raise RuntimeError("Seed prediction labels differ")
    clean_by_seed = np.stack([payload["clean_logit"] for payload in payloads])
    noisy_by_seed = np.stack([payload["t_0_5_logit"] for payload in payloads])
    clean_ensemble = clean_by_seed.mean(axis=0)
    noisy_ensemble = noisy_by_seed.mean(axis=0)
    seeds = np.random.SeedSequence(args.seed).generate_state(args.bootstrap)
    bootstraps = np.asarray(
        Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(one_bootstrap)(
                int(seed), labels, clean_ensemble, noisy_ensemble
            )
            for seed in seeds
        )
    )
    names = [
        "auroc",
        "auprc",
        "peptide_recall_at_probability_0_5",
        "non_peptide_specificity_at_probability_0_5",
    ]
    result = {
        "bootstrap_method": (
            "stratified nonparametric bootstrap with canonical molecule as the "
            "resampling unit"
        ),
        "bootstrap_replicates": args.bootstrap,
        "molecules": int(len(labels)),
        "negative_molecules": int(np.count_nonzero(labels == 0)),
        "positive_molecules": int(np.count_nonzero(labels == 1)),
        "seed_metrics": {},
    }
    for index, run_dir in enumerate(args.run_dir):
        clean_values = metrics(labels, clean_by_seed[index])
        noisy_values = metrics(labels, noisy_by_seed[index])
        result["seed_metrics"][run_dir.name] = {
            "clean": dict(zip(names, clean_values)),
            "t_0_5": dict(zip(names, noisy_values)),
        }
    result["ensemble"] = {}
    for condition, offset, scores in [
        ("clean", 0, clean_ensemble),
        ("t_0_5", 4, noisy_ensemble),
    ]:
        point = metrics(labels, scores)
        result["ensemble"][condition] = {
            name: {
                "estimate": point[index],
                "bootstrap_95_ci": [
                    float(np.quantile(bootstraps[:, offset + index], 0.025)),
                    float(np.quantile(bootstraps[:, offset + index], 0.975)),
                ],
            }
            for index, name in enumerate(names)
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.save(args.output_dir / "bootstrap_metrics.npy", bootstraps)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
