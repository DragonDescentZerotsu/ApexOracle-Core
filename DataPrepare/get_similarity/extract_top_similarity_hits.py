from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compute_percent_identity import (
    build_aligner,
    reconstruct_gapped_sequences,
    rotate_sequence,
    sanitize_sequence_for_scheme,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_LINEAR_RESULTS = CURRENT_DIR / "linear_similarity_results.csv"
DEFAULT_CYCLIC_BEST_PID = CURRENT_DIR / "cyclic_best_by_pid.csv"
DEFAULT_CYCLIC_BEST_MAXLEN = CURRENT_DIR / "cyclic_best_by_max_len_identity.csv"
DEFAULT_LINEAR_SUMMARY = CURRENT_DIR / "linear_top_similarity_hits.csv"
DEFAULT_CYCLIC_SUMMARY = CURRENT_DIR / "cyclic_top_similarity_hits.csv"
DEFAULT_ALIGNMENT_REPORT = CURRENT_DIR / "top_similarity_alignments.txt"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top linear/cyclic similarity hits and show their alignments."
    )
    parser.add_argument("--linear-results", type=Path, default=DEFAULT_LINEAR_RESULTS)
    parser.add_argument("--cyclic-best-pid", type=Path, default=DEFAULT_CYCLIC_BEST_PID)
    parser.add_argument("--cyclic-best-maxlen", type=Path, default=DEFAULT_CYCLIC_BEST_MAXLEN)
    parser.add_argument("--linear-summary-output", type=Path, default=DEFAULT_LINEAR_SUMMARY)
    parser.add_argument("--cyclic-summary-output", type=Path, default=DEFAULT_CYCLIC_SUMMARY)
    parser.add_argument("--alignment-report-output", type=Path, default=DEFAULT_ALIGNMENT_REPORT)
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ranking_key(row: dict, selection_metric: str) -> tuple[float, float, int, int, int]:
    secondary_metric = "max_len_identity" if selection_metric == "pid" else "pid"
    return (
        float(row[selection_metric]),
        float(row[secondary_metric]),
        int(row["matches"]),
        -int(row.get("query_rotation", 0) or 0),
        -int(row.get("train_rotation", 0) or 0),
    )


def best_rows_by_metric(rows: list[dict], selection_metric: str) -> list[dict]:
    best_by_query_and_scheme: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["query_peptide_id"], row["scoring_scheme"])
        current = best_by_query_and_scheme.get(key)
        if current is None or ranking_key(row, selection_metric) > ranking_key(current, selection_metric):
            best_by_query_and_scheme[key] = row
    return [best_by_query_and_scheme[key] for key in sorted(best_by_query_and_scheme)]


def build_alignment_strings(
    scoring_scheme: str,
    rotated_query_sequence: str,
    rotated_train_sequence: str,
) -> tuple[str, str, str]:
    aligner = build_aligner(scoring_scheme)
    sanitized_query = sanitize_sequence_for_scheme(rotated_query_sequence, scoring_scheme)
    sanitized_train = sanitize_sequence_for_scheme(rotated_train_sequence, scoring_scheme)

    alignment = aligner.align(sanitized_query, sanitized_train)[0]
    gapped_query, gapped_train = reconstruct_gapped_sequences(
        alignment,
        rotated_query_sequence,
        rotated_train_sequence,
    )
    midline = "".join(
        "|" if query_residue == train_residue and query_residue != "-" else " "
        for query_residue, train_residue in zip(gapped_query, gapped_train)
    )
    return gapped_query, midline, gapped_train


def enrich_row(row: dict, peptide_type: str, selection_metric: str) -> dict:
    query_sequence = row["query_sequence"]
    train_sequence = row["train_sequence"]
    query_rotation = int(row.get("query_rotation", 0) or 0)
    train_rotation = int(row.get("train_rotation", 0) or 0)
    rotated_query_sequence = rotate_sequence(query_sequence, query_rotation)
    rotated_train_sequence = rotate_sequence(train_sequence, train_rotation)
    gapped_query, midline, gapped_train = build_alignment_strings(
        row["scoring_scheme"],
        rotated_query_sequence,
        rotated_train_sequence,
    )
    return {
        "peptide_type": peptide_type,
        "query_peptide_id": row["query_peptide_id"],
        "selection_metric": selection_metric,
        "scoring_scheme": row["scoring_scheme"],
        "train_dbaasp_id": row["train_dbaasp_id"],
        "query_sequence": query_sequence,
        "train_sequence": train_sequence,
        "query_length": row["query_length"],
        "train_length": row["train_length"],
        "query_rotation": query_rotation,
        "train_rotation": train_rotation,
        "rotated_query_sequence": rotated_query_sequence,
        "rotated_train_sequence": rotated_train_sequence,
        "matches": row["matches"],
        "aligned_positions_including_gaps": row["aligned_positions_including_gaps"],
        "pid": row["pid"],
        "max_len_identity": row["max_len_identity"],
        "gapped_query_sequence": gapped_query,
        "alignment_midline": midline,
        "gapped_train_sequence": gapped_train,
    }


def write_summary_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_alignment_report(output_path: Path, linear_rows: list[dict], cyclic_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for section_name, rows in (("Linear", linear_rows), ("Cyclic", cyclic_rows)):
            handle.write(f"{section_name} Top Hits\n")
            handle.write("=" * (len(section_name) + 9))
            handle.write("\n\n")
            for row in rows:
                handle.write(
                    f"Query: {row['query_peptide_id']} | Type: {row['peptide_type']} | "
                    f"Metric: {row['selection_metric']} | Scoring: {row['scoring_scheme']}\n"
                )
                handle.write(
                    f"Best training DBAASP_id: {row['train_dbaasp_id']} | "
                    f"matches={row['matches']} | aligned_positions={row['aligned_positions_including_gaps']} | "
                    f"pid={row['pid']} | max_len_identity={row['max_len_identity']}\n"
                )
                handle.write(
                    f"Original query:  {row['query_sequence']}\n"
                    f"Original train:  {row['train_sequence']}\n"
                    f"Rotated query:   {row['rotated_query_sequence']} (rotation={row['query_rotation']})\n"
                    f"Rotated train:   {row['rotated_train_sequence']} (rotation={row['train_rotation']})\n"
                )
                handle.write(f"{row['gapped_query_sequence']}\n")
                handle.write(f"{row['alignment_midline']}\n")
                handle.write(f"{row['gapped_train_sequence']}\n\n")


def main() -> None:
    args = parse_args()

    linear_rows = load_csv_rows(args.linear_results)
    cyclic_best_pid_rows = load_csv_rows(args.cyclic_best_pid)
    cyclic_best_maxlen_rows = load_csv_rows(args.cyclic_best_maxlen)

    linear_summary_rows = []
    for selection_metric in ("pid", "max_len_identity"):
        for row in best_rows_by_metric(linear_rows, selection_metric):
            linear_summary_rows.append(enrich_row(row, peptide_type="linear", selection_metric=selection_metric))

    cyclic_summary_rows = []
    for row in best_rows_by_metric(cyclic_best_pid_rows, "pid"):
        cyclic_summary_rows.append(enrich_row(row, peptide_type="cyclic", selection_metric="pid"))
    for row in best_rows_by_metric(cyclic_best_maxlen_rows, "max_len_identity"):
        cyclic_summary_rows.append(
            enrich_row(row, peptide_type="cyclic", selection_metric="max_len_identity")
        )

    linear_summary_rows.sort(key=lambda row: (row["query_peptide_id"], row["selection_metric"], row["scoring_scheme"]))
    cyclic_summary_rows.sort(key=lambda row: (row["query_peptide_id"], row["selection_metric"], row["scoring_scheme"]))

    write_summary_csv(args.linear_summary_output, linear_summary_rows)
    write_summary_csv(args.cyclic_summary_output, cyclic_summary_rows)
    write_alignment_report(args.alignment_report_output, linear_summary_rows, cyclic_summary_rows)

    print(f"Wrote {len(linear_summary_rows)} linear top-hit rows to {args.linear_summary_output}")
    print(f"Wrote {len(cyclic_summary_rows)} cyclic top-hit rows to {args.cyclic_summary_output}")
    print(f"Wrote alignment report to {args.alignment_report_output}")


if __name__ == "__main__":
    main()
