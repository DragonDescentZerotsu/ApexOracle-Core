#!/usr/bin/env python3
"""Stream the exact all-pairs Tanimoto distribution from a fingerprint cache."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rdkit
from rdkit import DataStructs


FINGERPRINTS: list[DataStructs.ExplicitBitVect] = []
FINE_BINS = 1_000
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cumulative_pairs_before(row: int, population_size: int) -> int:
    return row * (2 * population_size - row - 1) // 2


def balanced_ranges(population_size: int, workers: int) -> list[tuple[int, int]]:
    total_pairs = population_size * (population_size - 1) // 2
    cumulative = [
        cumulative_pairs_before(row, population_size)
        for row in range(population_size + 1)
    ]
    boundaries = [0]
    for worker_index in range(1, workers):
        target = total_pairs * worker_index // workers
        boundaries.append(bisect.bisect_left(cumulative, target))
    boundaries.append(population_size - 1)
    ranges = [
        (start, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop > start
    ]
    return ranges


def compute_range(bounds: tuple[int, int]) -> dict[str, Any]:
    start, stop = bounds
    histogram = np.zeros(FINE_BINS, dtype=np.int64)
    threshold_counts = np.zeros(len(THRESHOLDS), dtype=np.int64)
    pair_count = 0
    value_sum = 0.0
    value_sum_squares = 0.0
    minimum = 1.0
    maximum = 0.0
    for query_index in range(start, stop):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                FINGERPRINTS[query_index], FINGERPRINTS[query_index + 1 :]
            ),
            dtype=np.float64,
        )
        if similarities.size == 0:
            continue
        bin_indices = np.minimum(
            (similarities * FINE_BINS).astype(np.int64), FINE_BINS - 1
        )
        histogram += np.bincount(bin_indices, minlength=FINE_BINS)
        threshold_counts += np.asarray(
            [np.count_nonzero(similarities >= threshold) for threshold in THRESHOLDS],
            dtype=np.int64,
        )
        pair_count += int(similarities.size)
        value_sum += float(np.sum(similarities, dtype=np.float64))
        value_sum_squares += float(
            np.sum(similarities * similarities, dtype=np.float64)
        )
        minimum = min(minimum, float(np.min(similarities)))
        maximum = max(maximum, float(np.max(similarities)))
    return {
        "start": start,
        "stop": stop,
        "pair_count": pair_count,
        "histogram": histogram,
        "threshold_counts": threshold_counts,
        "sum": value_sum,
        "sum_squares": value_sum_squares,
        "min": minimum,
        "max": maximum,
    }


def histogram_quantile(histogram: np.ndarray, quantile: float) -> dict[str, float]:
    total = int(np.sum(histogram))
    target = quantile * (total - 1)
    bin_index = int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))
    return {
        "estimate": (bin_index + 0.5) / FINE_BINS,
        "lower": bin_index / FINE_BINS,
        "upper": (bin_index + 1) / FINE_BINS,
    }


def write_histogram(path: Path, histogram: np.ndarray) -> None:
    total = int(np.sum(histogram))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("bin_left", "bin_right", "bin_center", "count", "fraction"),
        )
        writer.writeheader()
        for index, count in enumerate(histogram):
            writer.writerow(
                {
                    "bin_left": index / FINE_BINS,
                    "bin_right": (index + 1) / FINE_BINS,
                    "bin_center": (index + 0.5) / FINE_BINS,
                    "count": int(count),
                    "fraction": float(count / total),
                }
            )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    started = time.perf_counter()
    matrix = np.load(args.fingerprint_cache, allow_pickle=False)["fingerprints"]
    if matrix.ndim != 2 or matrix.dtype != np.uint8:
        raise ValueError("fingerprint cache must be a 2D uint8 matrix")
    global FINGERPRINTS
    FINGERPRINTS = [
        DataStructs.CreateFromBinaryText(row.tobytes()) for row in matrix
    ]
    population_size = len(FINGERPRINTS)
    expected_pairs = population_size * (population_size - 1) // 2
    ranges = balanced_ranges(population_size, args.workers)
    load_seconds = time.perf_counter() - started

    compute_started = time.perf_counter()
    with mp.get_context("fork").Pool(len(ranges)) as pool:
        results = pool.map(compute_range, ranges)
    compute_seconds = time.perf_counter() - compute_started

    histogram = np.sum([row["histogram"] for row in results], axis=0)
    threshold_counts = np.sum(
        [row["threshold_counts"] for row in results], axis=0
    )
    pair_count = sum(int(row["pair_count"]) for row in results)
    if pair_count != expected_pairs or int(np.sum(histogram)) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairs, observed {pair_count}"
        )
    value_sum = sum(float(row["sum"]) for row in results)
    value_sum_squares = sum(float(row["sum_squares"]) for row in results)
    mean = value_sum / pair_count
    sample_variance = (
        value_sum_squares - pair_count * mean * mean
    ) / (pair_count - 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = args.output_dir / "generation_all_pairs_tanimoto_histogram.csv"
    write_histogram(histogram_path, histogram)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "population_size": population_size,
        "unordered_nonself_pairs": pair_count,
        "workers": len(ranges),
        "fine_histogram_bins": FINE_BINS,
        "fingerprint_cache": str(args.fingerprint_cache.resolve()),
        "fingerprint_cache_sha256": sha256_file(args.fingerprint_cache),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "load_seconds": load_seconds,
        "compute_seconds": compute_seconds,
        "total_seconds": time.perf_counter() - started,
        "throughput_pairs_per_second": pair_count / compute_seconds,
        "mean": mean,
        "sample_sd": sample_variance**0.5,
        "min": min(float(row["min"]) for row in results),
        "max": max(float(row["max"]) for row in results),
        "q05_histogram": histogram_quantile(histogram, 0.05),
        "q25_histogram": histogram_quantile(histogram, 0.25),
        "median_histogram": histogram_quantile(histogram, 0.50),
        "q75_histogram": histogram_quantile(histogram, 0.75),
        "q95_histogram": histogram_quantile(histogram, 0.95),
        "threshold_counts": {
            str(threshold): int(count)
            for threshold, count in zip(THRESHOLDS, threshold_counts, strict=True)
        },
        "threshold_fractions": {
            str(threshold): float(count / pair_count)
            for threshold, count in zip(THRESHOLDS, threshold_counts, strict=True)
        },
    }
    summary_path = args.output_dir / "generation_all_pairs_tanimoto_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
