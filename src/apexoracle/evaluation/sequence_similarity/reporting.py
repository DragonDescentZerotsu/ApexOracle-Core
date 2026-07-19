"""Top-hit extraction and invariant checks for similarity outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .alignment import compute_alignment, ranking_key, rotate_sequence


SUMMARY_FIELDNAMES = [
    "peptide_type",
    "query_peptide_id",
    "selection_metric",
    "scoring_scheme",
    "train_dbaasp_id",
    "query_sequence",
    "train_sequence",
    "query_length",
    "train_length",
    "query_rotation",
    "train_rotation",
    "rotated_query_sequence",
    "rotated_train_sequence",
    "matches",
    "aligned_positions_including_gaps",
    "pid",
    "max_len_identity",
    "gapped_query_sequence",
    "alignment_midline",
    "gapped_train_sequence",
]


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_rows_by_metric(rows: Iterable[dict], metric: str) -> list[dict]:
    """Keep the first row on an exact tie, matching the legacy script."""

    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["query_peptide_id"], row["scoring_scheme"])
        current = best.get(key)
        include_rotations = "query_rotation" in row
        if current is None or ranking_key(
            row, metric, include_rotations=include_rotations
        ) > ranking_key(current, metric, include_rotations=include_rotations):
            best[key] = row
    return [best[key] for key in sorted(best)]


def enrich_row(row: dict, peptide_type: str, metric: str) -> dict:
    query_rotation = int(row.get("query_rotation", 0) or 0)
    train_rotation = int(row.get("train_rotation", 0) or 0)
    rotated_query = rotate_sequence(row["query_sequence"], query_rotation)
    rotated_train = rotate_sequence(row["train_sequence"], train_rotation)
    alignment = compute_alignment(
        rotated_query,
        rotated_train,
        row["scoring_scheme"],
        include_gapped=True,
    )
    gapped_query = alignment.gapped_target or ""
    gapped_train = alignment.gapped_query or ""
    midline = "".join(
        "|" if left == right and left != "-" else " "
        for left, right in zip(gapped_query, gapped_train)
    )
    return {
        "peptide_type": peptide_type,
        "query_peptide_id": row["query_peptide_id"],
        "selection_metric": metric,
        "scoring_scheme": row["scoring_scheme"],
        "train_dbaasp_id": row["train_dbaasp_id"],
        "query_sequence": row["query_sequence"],
        "train_sequence": row["train_sequence"],
        "query_length": row["query_length"],
        "train_length": row["train_length"],
        "query_rotation": query_rotation,
        "train_rotation": train_rotation,
        "rotated_query_sequence": rotated_query,
        "rotated_train_sequence": rotated_train,
        "matches": row["matches"],
        "aligned_positions_including_gaps": row["aligned_positions_including_gaps"],
        "pid": row["pid"],
        "max_len_identity": row["max_len_identity"],
        "gapped_query_sequence": gapped_query,
        "alignment_midline": midline,
        "gapped_train_sequence": gapped_train,
    }


def extract_top_hits(
    *,
    linear_results: Path,
    cyclic_best_pid: Path,
    cyclic_best_max_len: Path,
    linear_summary_output: Path,
    cyclic_summary_output: Path,
    alignment_report_output: Path,
) -> tuple[list[dict], list[dict]]:
    linear_rows = load_csv_rows(linear_results)
    cyclic_pid_rows = load_csv_rows(cyclic_best_pid)
    cyclic_max_len_rows = load_csv_rows(cyclic_best_max_len)
    linear_summary = [
        enrich_row(row, "linear", metric)
        for metric in ("pid", "max_len_identity")
        for row in best_rows_by_metric(linear_rows, metric)
    ]
    cyclic_summary = [
        *[
            enrich_row(row, "cyclic", "pid")
            for row in best_rows_by_metric(cyclic_pid_rows, "pid")
        ],
        *[
            enrich_row(row, "cyclic", "max_len_identity")
            for row in best_rows_by_metric(cyclic_max_len_rows, "max_len_identity")
        ],
    ]
    for rows in (linear_summary, cyclic_summary):
        rows.sort(
            key=lambda row: (
                row["query_peptide_id"],
                row["selection_metric"],
                row["scoring_scheme"],
            )
        )
    _write_summary(linear_summary_output, linear_summary)
    _write_summary(cyclic_summary_output, cyclic_summary)
    _write_alignment_report(alignment_report_output, linear_summary, cyclic_summary)
    return linear_summary, cyclic_summary


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_alignment_report(
    path: Path, linear_rows: list[dict], cyclic_rows: list[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for section, rows in (("Linear", linear_rows), ("Cyclic", cyclic_rows)):
            handle.write(f"{section} Top Hits\n{'=' * (len(section) + 9)}\n\n")
            for row in rows:
                handle.write(
                    f"Query: {row['query_peptide_id']} | Type: {row['peptide_type']} | "
                    f"Metric: {row['selection_metric']} | Scoring: {row['scoring_scheme']}\n"
                    f"Best training DBAASP_id: {row['train_dbaasp_id']} | "
                    f"matches={row['matches']} | "
                    f"aligned_positions={row['aligned_positions_including_gaps']} | "
                    f"pid={row['pid']} | max_len_identity={row['max_len_identity']}\n"
                    f"Original query:  {row['query_sequence']}\n"
                    f"Original train:  {row['train_sequence']}\n"
                    f"Rotated query:   {row['rotated_query_sequence']} "
                    f"(rotation={row['query_rotation']})\n"
                    f"Rotated train:   {row['rotated_train_sequence']} "
                    f"(rotation={row['train_rotation']})\n"
                    f"{row['gapped_query_sequence']}\n"
                    f"{row['alignment_midline']}\n"
                    f"{row['gapped_train_sequence']}\n\n"
                )


def validate_rows(path: Path, expected_selection_metric: str | None = None) -> dict:
    checks = {
        "row_count": 0,
        "formula_mismatches": 0,
        "pid_gt_max_len_identity": 0,
        "aligned_positions_lt_max_len": 0,
        "matches_gt_shorter_len": 0,
        "selection_metric_mismatches": 0,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            checks["row_count"] += 1
            query_length = int(row["query_length"])
            train_length = int(row["train_length"])
            matches = int(row["matches"])
            aligned_positions = int(row["aligned_positions_including_gaps"])
            pid = float(row["pid"])
            max_len_identity = float(row["max_len_identity"])
            if (
                abs(pid - (matches / aligned_positions if aligned_positions else 0.0))
                > 1e-12
                or abs(
                    max_len_identity
                    - (
                        matches / max(query_length, train_length)
                        if max(query_length, train_length)
                        else 0.0
                    )
                )
                > 1e-12
            ):
                checks["formula_mismatches"] += 1
            if pid > max_len_identity + 1e-12:
                checks["pid_gt_max_len_identity"] += 1
            if aligned_positions < max(query_length, train_length):
                checks["aligned_positions_lt_max_len"] += 1
            if matches > min(query_length, train_length):
                checks["matches_gt_shorter_len"] += 1
            if (
                expected_selection_metric
                and row.get("selection_metric") != expected_selection_metric
            ):
                checks["selection_metric_mismatches"] += 1
    checks["ok"] = all(
        value == 0 for key, value in checks.items() if key != "row_count"
    )
    return checks
