#!/usr/bin/env python3
"""Audit canonical sequence-similarity outputs against saved paper artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.evaluation.sequence_similarity.alignment import (  # noqa: E402
    compute_alignment,
    rotate_sequence,
)


RAW_FILENAMES = (
    "linear_similarity_results.csv",
    "cyclic_rotation_similarity_results.csv",
    "cyclic_best_by_pid.csv",
    "cyclic_best_by_max_len_identity.csv",
)
PAPER_PID = {
    "ApexOracle-3": 0.36666666666666664,
    "ApexOracle-12": 0.35714285714285715,
    "ApexOracle-23": 0.3684210526315789,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def compare_raw_outputs(historical_dir: Path, recomputed_raw_dir: Path) -> dict:
    files = {}
    for filename in RAW_FILENAMES:
        historical = historical_dir / filename
        recomputed = recomputed_raw_dir / filename
        old_hash = sha256(historical)
        new_hash = sha256(recomputed)
        files[filename] = {
            "historical_path": relative(historical),
            "canonical_recomputed_path": f"<recomputed-raw-dir>/{filename}",
            "historical_sha256": old_hash,
            "canonical_recomputed_sha256": new_hash,
            "byte_identical": old_hash == new_hash
            and historical.stat().st_size == recomputed.stat().st_size,
            "bytes": historical.stat().st_size,
        }
    return {
        "files": files,
        "all_byte_identical": all(item["byte_identical"] for item in files.values()),
    }


def compare_caches(historical_dir: Path, prepared_cache_dir: Path) -> dict:
    files = {}
    for filename in ("train_linear_peptides.csv", "train_cyclic_peptides.csv"):
        historical = historical_dir / filename
        prepared = prepared_cache_dir / filename
        old_hash = sha256(historical)
        new_hash = sha256(prepared)
        files[filename] = {
            "historical_sha256": old_hash,
            "canonical_prepared_sha256": new_hash,
            "byte_identical": old_hash == new_hash,
        }
    return {
        "normalization": "uppercase training sequences",
        "files": files,
        "all_byte_identical": all(item["byte_identical"] for item in files.values()),
    }


def recompute_saved_samples(path: Path, per_bucket: int) -> dict:
    selected: dict[tuple[str, str], list[dict]] = {}
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            key = (row["query_peptide_id"], row["scoring_scheme"])
            bucket = selected.setdefault(key, [])
            if len(bucket) < per_bucket:
                bucket.append(row)
    mismatches = []
    for key, rows in selected.items():
        for row in rows:
            query = row["query_sequence"]
            train = row["train_sequence"]
            if "query_rotation" in row:
                query = rotate_sequence(query, int(row["query_rotation"]))
                train = rotate_sequence(train, int(row["train_rotation"]))
            observed = compute_alignment(query, train, row["scoring_scheme"])
            expected = (
                int(row["matches"]),
                int(row["aligned_positions_including_gaps"]),
                float(row["pid"]),
                float(row["max_len_identity"]),
            )
            actual = (
                observed.matches,
                observed.aligned_positions_including_gaps,
                observed.pid,
                observed.max_len_identity,
            )
            if actual != expected:
                mismatches.append(
                    {
                        "bucket": list(key),
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "observed": list(actual),
                        "saved": list(expected),
                    }
                )
    return {
        "path": relative(path),
        "total_rows": total,
        "sampled_rows": sum(map(len, selected.values())),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def main_pid_rows(report_dir: Path) -> tuple[dict, dict]:
    selected: dict[str, dict] = {}
    for filename in (
        "linear_top_similarity_hits.csv",
        "cyclic_top_similarity_hits.csv",
    ):
        with (report_dir / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["selection_metric"] == "pid"
                    and row["scoring_scheme"] == "blosum62_needle"
                ):
                    selected[row["query_peptide_id"]] = {
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "train_sequence": row["train_sequence"],
                        "matches": int(row["matches"]),
                        "aligned_positions": int(
                            row["aligned_positions_including_gaps"]
                        ),
                        "pid": float(row["pid"]),
                        "paper_pid": PAPER_PID[row["query_peptide_id"]],
                        "matches_paper_pid": float(row["pid"])
                        == PAPER_PID[row["query_peptide_id"]],
                    }
    return selected, {
        "all_three_present": set(selected) == set(PAPER_PID),
        "all_match_paper_pid": all(
            item["matches_paper_pid"] for item in selected.values()
        ),
    }


def apex12_ties(linear_results: Path) -> list[dict]:
    rows = []
    with linear_results.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["query_peptide_id"] == "ApexOracle-12"
                and row["scoring_scheme"] == "blosum62_needle"
                and float(row["pid"]) == PAPER_PID["ApexOracle-12"]
            ):
                rows.append(
                    {
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "train_sequence": row["train_sequence"],
                        "matches": int(row["matches"]),
                        "aligned_positions": int(
                            row["aligned_positions_including_gaps"]
                        ),
                        "max_len_identity": float(row["max_len_identity"]),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-dir", type=Path, default=Path("DataPrepare/get_similarity")
    )
    parser.add_argument("--recomputed-raw-dir", type=Path, required=True)
    parser.add_argument("--prepared-cache-dir", type=Path, required=True)
    parser.add_argument("--full-output-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical = (REPO_ROOT / args.historical_dir).resolve()
    recomputed_raw = args.recomputed_raw_dir.resolve()
    prepared_cache = args.prepared_cache_dir.resolve()
    full_output = args.full_output_dir.resolve()
    lead_rows, paper_check = main_pid_rows(full_output / "reports")
    report = {
        "schema_version": 1,
        "historical_output_equivalence": compare_raw_outputs(
            historical, recomputed_raw
        ),
        "training_cache_rebuild_equivalence": compare_caches(
            historical, prepared_cache
        ),
        "canonical_sample_recomputation": {
            "linear": recompute_saved_samples(
                historical / "linear_similarity_results.csv", args.per_bucket
            ),
            "cyclic": recompute_saved_samples(
                historical / "cyclic_rotation_similarity_results.csv", args.per_bucket
            ),
        },
        "three_lead_paper_pid": lead_rows,
        "three_lead_check": paper_check,
        "apexoracle_12_complete_ties": apex12_ties(
            full_output / "raw/linear_similarity_results.csv"
        ),
        "evidence_boundary": {
            "ApexOracle-3": "historical full output and canonical recomputation",
            "ApexOracle-12": "recomputed from canonical algorithm; absent from saved main output",
            "ApexOracle-23": "historical full output and canonical recomputation",
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
