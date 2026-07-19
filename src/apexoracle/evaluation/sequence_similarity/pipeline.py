"""Deterministic linear and exhaustive cyclic similarity computation."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from Bio import Align

from .alignment import (
    SCORING_SCHEMES,
    build_aligner,
    compute_alignment,
    ranking_key,
    rotate_sequence,
)
from .io import (
    CYCLIC_FIELDNAMES,
    CYCLIC_SUMMARY_FIELDNAMES,
    LINEAR_FIELDNAMES,
    load_query_peptides,
    load_training_cache,
)


WORKER_ALIGNER: Align.PairwiseAligner | None = None
WORKER_SCHEME: str | None = None


@dataclass(frozen=True)
class SimilarityOutputs:
    linear: Path
    cyclic_rotations: Path
    cyclic_best_pid: Path
    cyclic_best_max_len: Path
    manifest: Path


def init_worker(scoring_scheme: str) -> None:
    global WORKER_ALIGNER, WORKER_SCHEME
    WORKER_SCHEME = scoring_scheme
    WORKER_ALIGNER = build_aligner(scoring_scheme)


def chunk_records(records: list[dict], chunk_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def _metric_values(
    target_sequence: str, query_sequence: str
) -> tuple[int, int, float, float]:
    if WORKER_ALIGNER is None or WORKER_SCHEME is None:
        raise RuntimeError("Worker aligner has not been initialized")
    metrics = compute_alignment(
        target_sequence,
        query_sequence,
        WORKER_SCHEME,
        aligner=WORKER_ALIGNER,
    )
    return (
        metrics.matches,
        metrics.aligned_positions_including_gaps,
        metrics.pid,
        metrics.max_len_identity,
    )


def linear_worker(task: tuple[dict, list[dict]]) -> list[dict]:
    query, training_chunk = task
    rows: list[dict] = []
    for training_record in training_chunk:
        matches, aligned_positions, pid, max_len_identity = _metric_values(
            query["query_sequence"], training_record["train_sequence"]
        )
        rows.append(
            {
                "query_peptide_id": query["query_peptide_id"],
                "query_sequence": query["query_sequence"],
                "query_length": query["query_length"],
                "train_dbaasp_id": training_record["train_dbaasp_id"],
                "train_sequence": training_record["train_sequence"],
                "train_length": training_record["train_length"],
                "scoring_scheme": WORKER_SCHEME,
                "matches": matches,
                "aligned_positions_including_gaps": aligned_positions,
                "pid": pid,
                "max_len_identity": max_len_identity,
            }
        )
    return rows


def cyclic_worker(task: tuple[dict, list[dict]]) -> list[dict]:
    query, training_chunk = task
    rows: list[dict] = []
    query_sequence = query["query_sequence"]
    for training_record in training_chunk:
        train_sequence = training_record["train_sequence"]
        for query_rotation in range(len(query_sequence)):
            rotated_query = rotate_sequence(query_sequence, query_rotation)
            for train_rotation in range(len(train_sequence)):
                rotated_train = rotate_sequence(train_sequence, train_rotation)
                matches, aligned_positions, pid, max_len_identity = _metric_values(
                    rotated_query, rotated_train
                )
                rows.append(
                    {
                        "query_peptide_id": query["query_peptide_id"],
                        "query_sequence": query_sequence,
                        "query_length": query["query_length"],
                        "train_dbaasp_id": training_record["train_dbaasp_id"],
                        "train_sequence": train_sequence,
                        "train_length": training_record["train_length"],
                        "query_rotation": query_rotation,
                        "train_rotation": train_rotation,
                        "scoring_scheme": WORKER_SCHEME,
                        "matches": matches,
                        "aligned_positions_including_gaps": aligned_positions,
                        "pid": pid,
                        "max_len_identity": max_len_identity,
                    }
                )
    return rows


def _update_best(
    best_rows: dict[tuple[str, str, str], dict],
    row: dict,
    primary_metric: str,
) -> None:
    key = (row["query_peptide_id"], row["train_dbaasp_id"], row["scoring_scheme"])
    current = best_rows.get(key)
    if current is None or ranking_key(
        row, primary_metric, include_rotations=True
    ) > ranking_key(current, primary_metric, include_rotations=True):
        best_rows[key] = row.copy()


def _executor_rows(
    worker,
    query: dict,
    records: list[dict],
    scoring_scheme: str,
    processes: int,
    chunk_size: int,
):
    tasks = [(query, chunk) for chunk in chunk_records(records, chunk_size)]
    with ProcessPoolExecutor(
        max_workers=processes,
        initializer=init_worker,
        initargs=(scoring_scheme,),
    ) as executor:
        yield from executor.map(worker, tasks)


def _selected_queries(
    queries: list[dict], query_ids: Sequence[str] | None
) -> list[dict]:
    if not query_ids:
        return queries
    requested = set(query_ids)
    selected = [query for query in queries if query["query_peptide_id"] in requested]
    missing = requested - {query["query_peptide_id"] for query in selected}
    if missing:
        raise ValueError(f"Unknown query IDs: {sorted(missing)}")
    return selected


def run_similarity(
    *,
    query_csv: Path,
    linear_cache: Path,
    cyclic_cache: Path,
    outputs: SimilarityOutputs,
    processes: int | None = None,
    linear_chunk_size: int = 500,
    cyclic_chunk_size: int = 40,
    query_ids: Sequence[str] | None = None,
) -> dict:
    """Run the legacy-compatible CSV pipeline and return its manifest."""

    worker_count = processes or max(1, min(8, os.cpu_count() or 1))
    if worker_count < 1 or linear_chunk_size < 1 or cyclic_chunk_size < 1:
        raise ValueError("Process and chunk counts must be positive")

    queries = _selected_queries(load_query_peptides(query_csv), query_ids)
    linear_queries = [query for query in queries if query["cyclic"] == "No"]
    cyclic_queries = [query for query in queries if query["cyclic"] == "Yes"]
    linear_training = load_training_cache(linear_cache)
    cyclic_training = load_training_cache(cyclic_cache)
    for path in vars(outputs).values():
        path.parent.mkdir(parents=True, exist_ok=True)

    linear_count = 0
    with outputs.linear.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINEAR_FIELDNAMES)
        writer.writeheader()
        for query in linear_queries:
            for scoring_scheme in SCORING_SCHEMES:
                for chunk_rows in _executor_rows(
                    linear_worker,
                    query,
                    linear_training,
                    scoring_scheme,
                    worker_count,
                    linear_chunk_size,
                ):
                    writer.writerows(chunk_rows)
                    linear_count += len(chunk_rows)

    cyclic_count = 0
    best_pid: dict[tuple[str, str, str], dict] = {}
    best_max_len: dict[tuple[str, str, str], dict] = {}
    with outputs.cyclic_rotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CYCLIC_FIELDNAMES)
        writer.writeheader()
        for query in cyclic_queries:
            for scoring_scheme in SCORING_SCHEMES:
                for chunk_rows in _executor_rows(
                    cyclic_worker,
                    query,
                    cyclic_training,
                    scoring_scheme,
                    worker_count,
                    cyclic_chunk_size,
                ):
                    writer.writerows(chunk_rows)
                    cyclic_count += len(chunk_rows)
                    for row in chunk_rows:
                        _update_best(best_pid, row, "pid")
                        _update_best(best_max_len, row, "max_len_identity")

    def write_best(path: Path, rows: dict, metric: str) -> int:
        ordered = [rows[key] for key in sorted(rows)]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CYCLIC_SUMMARY_FIELDNAMES)
            writer.writeheader()
            for row in ordered:
                writer.writerow({**row, "selection_metric": metric})
        return len(ordered)

    best_pid_count = write_best(outputs.cyclic_best_pid, best_pid, "pid")
    best_max_len_count = write_best(
        outputs.cyclic_best_max_len, best_max_len, "max_len_identity"
    )

    linear_expected = len(linear_queries) * len(linear_training) * len(SCORING_SCHEMES)
    rotations_per_scheme = sum(
        query["query_length"] * record["train_length"]
        for query in cyclic_queries
        for record in cyclic_training
    )
    cyclic_expected = rotations_per_scheme * len(SCORING_SCHEMES)
    cyclic_summary_expected = (
        len(cyclic_queries) * len(cyclic_training) * len(SCORING_SCHEMES)
    )
    observed = (linear_count, cyclic_count, best_pid_count, best_max_len_count)
    expected = (
        linear_expected,
        cyclic_expected,
        cyclic_summary_expected,
        cyclic_summary_expected,
    )
    if observed != expected:
        raise RuntimeError(
            f"Output count mismatch: expected {expected}, observed {observed}"
        )

    manifest = {
        "schema_version": 1,
        "protocol": "paper_sequence_similarity_legacy_compatible",
        "inputs": {
            "query_csv": str(query_csv),
            "linear_cache": str(linear_cache),
            "cyclic_cache": str(cyclic_cache),
        },
        "outputs": {key: str(value) for key, value in vars(outputs).items()},
        "runtime": {
            "processes": worker_count,
            "linear_chunk_size": linear_chunk_size,
            "cyclic_chunk_size": cyclic_chunk_size,
        },
        "queries": [query["query_peptide_id"] for query in queries],
        "counts": {
            "linear_queries": len(linear_queries),
            "cyclic_queries": len(cyclic_queries),
            "linear_training_records": len(linear_training),
            "cyclic_training_records": len(cyclic_training),
            "linear_rows_written": linear_count,
            "cyclic_rotation_rows_written": cyclic_count,
            "cyclic_best_pid_rows": best_pid_count,
            "cyclic_best_max_len_rows": best_max_len_count,
        },
        "scoring_schemes": list(SCORING_SCHEMES),
        "identity": {
            "pid_denominator": "aligned_positions_including_gaps",
            "matches": "case-sensitive exact residue equality excluding gaps",
        },
        "cyclic_search": "all_query_rotations_by_all_training_rotations",
        "blosum62_normalization": {
            "case_sensitive_d_amino_acids": True,
            "mixed_chirality_positive_scores_clamped_to": 0.0,
            "uppercase_noncanonical": "X",
            "lowercase_noncanonical": "x",
        },
    }
    outputs.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest
