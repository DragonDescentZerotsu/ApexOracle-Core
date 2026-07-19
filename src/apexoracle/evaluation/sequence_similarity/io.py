"""Typed CSV I/O contracts for sequence-similarity analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


LINEAR_FIELDNAMES = [
    "query_peptide_id",
    "query_sequence",
    "query_length",
    "train_dbaasp_id",
    "train_sequence",
    "train_length",
    "scoring_scheme",
    "matches",
    "aligned_positions_including_gaps",
    "pid",
    "max_len_identity",
]
CYCLIC_FIELDNAMES = LINEAR_FIELDNAMES[:]
CYCLIC_FIELDNAMES.insert(6, "query_rotation")
CYCLIC_FIELDNAMES.insert(7, "train_rotation")
CYCLIC_SUMMARY_FIELDNAMES = CYCLIC_FIELDNAMES + ["selection_metric"]


def normalize_sequence(raw_sequence: str | None) -> str:
    return (raw_sequence or "").strip()


def load_query_peptides(query_csv: Path) -> list[dict]:
    queries: list[dict] = []
    with query_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"peptide_id", "cyclic", "sequence"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Query CSV must contain {sorted(required)}")
        for row in reader:
            sequence = normalize_sequence(row["sequence"])
            queries.append(
                {
                    "query_peptide_id": row["peptide_id"].strip(),
                    "cyclic": row["cyclic"].strip(),
                    "query_sequence": sequence,
                    "query_length": len(sequence),
                }
            )
    labels = sorted({query["cyclic"] for query in queries} - {"No", "Yes"})
    if labels:
        raise ValueError(f"Unsupported query cyclic labels: {labels}")
    if len({query["query_peptide_id"] for query in queries}) != len(queries):
        raise ValueError("Query peptide IDs must be unique")
    return queries


def load_training_cache(cache_csv: Path) -> list[dict]:
    records: list[dict] = []
    with cache_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dbaasp_id", "sequence", "length"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Training cache must contain {sorted(required)}")
        for row in reader:
            sequence = normalize_sequence(row["sequence"])
            declared_length = int(row["length"])
            if declared_length != len(sequence):
                raise ValueError(
                    f"Length mismatch for DBAASP {row['dbaasp_id']}: "
                    f"declared {declared_length}, observed {len(sequence)}"
                )
            records.append(
                {
                    "train_dbaasp_id": row["dbaasp_id"].strip(),
                    "train_sequence": sequence,
                    "train_length": declared_length,
                }
            )
    return records


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count
