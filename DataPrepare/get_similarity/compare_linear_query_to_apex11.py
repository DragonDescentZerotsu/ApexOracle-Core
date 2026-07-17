from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compute_percent_identity import (
    LINEAR_FIELDNAMES,
    build_aligner,
    load_query_peptides,
    reconstruct_gapped_sequences,
    sanitize_sequence_for_scheme,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_CSV = CURRENT_DIR / "query_peptides.csv"
DEFAULT_APEX11_CSV = CURRENT_DIR.parent / "Data" / "APEX 1.1 Data.csv"
DEFAULT_RESULTS_CSV = CURRENT_DIR / "apex11_linear_similarity_results.csv"
DEFAULT_TOP_HITS_CSV = CURRENT_DIR / "apex11_linear_top_hits.csv"

SCORING_SCHEMES = ("blosum62_needle", "exact_match_needle")
RESULT_FIELDNAMES = ["apex_row_index"] + LINEAR_FIELDNAMES
TOP_HIT_FIELDNAMES = RESULT_FIELDNAMES + [
    "selection_metric",
    "gapped_query_sequence",
    "alignment_midline",
    "gapped_train_sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare linear query peptides to all peptides in APEX 1.1 Data.csv."
    )
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--apex11-csv", type=Path, default=DEFAULT_APEX11_CSV)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--top-hits-csv", type=Path, default=DEFAULT_TOP_HITS_CSV)
    return parser.parse_args()
def compute_alignment_details(query_sequence: str, apex_sequence: str, scoring_scheme: str) -> tuple[int, int, float, float, str, str, str]:
    aligner = build_aligner(scoring_scheme)
    sanitized_query = sanitize_sequence_for_scheme(query_sequence, scoring_scheme)
    sanitized_apex = sanitize_sequence_for_scheme(apex_sequence, scoring_scheme)
    alignment = aligner.align(sanitized_query, sanitized_apex)[0]
    gapped_query, gapped_apex = reconstruct_gapped_sequences(alignment, query_sequence, apex_sequence)
    matches = sum(
        1
        for query_residue, apex_residue in zip(gapped_query, gapped_apex)
        if query_residue == apex_residue and query_residue != "-"
    )
    aligned_positions = len(gapped_query)
    max_len_identity = matches / max(len(query_sequence), len(apex_sequence))
    pid = matches / aligned_positions if aligned_positions else 0.0
    midline = "".join(
        "|" if query_residue == apex_residue and query_residue != "-" else " "
        for query_residue, apex_residue in zip(gapped_query, gapped_apex)
    )
    return matches, aligned_positions, pid, max_len_identity, gapped_query, midline, gapped_apex


def load_apex11_peptides(apex11_csv: Path) -> list[dict]:
    peptides: list[dict] = []
    with apex11_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            sequence = (row.get("Peptide") or "").strip()
            if not sequence:
                continue
            peptides.append({"apex_row_index": idx, "train_sequence": sequence, "train_length": len(sequence)})
    return peptides


def ranking_key(row: dict, metric: str) -> tuple[float, float, int]:
    secondary_metric = "max_len_identity" if metric == "pid" else "pid"
    return (float(row[metric]), float(row[secondary_metric]), int(row["matches"]))


def main() -> None:
    args = parse_args()

    queries = [query for query in load_query_peptides(args.query_csv) if query["cyclic"] == "No"]
    apex_peptides = load_apex11_peptides(args.apex11_csv)

    all_rows: list[dict] = []
    top_hit_rows: list[dict] = []

    for query in queries:
        for scoring_scheme in SCORING_SCHEMES:
            best_by_metric: dict[str, dict] = {}
            for apex_record in apex_peptides:
                matches, aligned_positions, pid, max_len_identity, gapped_query, midline, gapped_apex = compute_alignment_details(
                    query["query_sequence"],
                    apex_record["train_sequence"],
                    scoring_scheme,
                )
                row = {
                    "apex_row_index": apex_record["apex_row_index"],
                    "query_peptide_id": query["query_peptide_id"],
                    "query_sequence": query["query_sequence"],
                    "query_length": query["query_length"],
                    "train_dbaasp_id": f"APEX11_row_{apex_record['apex_row_index']}",
                    "train_sequence": apex_record["train_sequence"],
                    "train_length": apex_record["train_length"],
                    "scoring_scheme": scoring_scheme,
                    "matches": matches,
                    "aligned_positions_including_gaps": aligned_positions,
                    "pid": pid,
                    "max_len_identity": max_len_identity,
                }
                all_rows.append(row)

                for metric in ("pid", "max_len_identity"):
                    current_best = best_by_metric.get(metric)
                    if current_best is None or ranking_key(row, metric) > ranking_key(current_best, metric):
                        best_by_metric[metric] = {
                            **row,
                            "selection_metric": metric,
                            "gapped_query_sequence": gapped_query,
                            "alignment_midline": midline,
                            "gapped_train_sequence": gapped_apex,
                        }

            for metric in ("pid", "max_len_identity"):
                top_hit_rows.append(best_by_metric[metric])

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    with args.top_hits_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOP_HIT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            sorted(top_hit_rows, key=lambda row: (row["query_peptide_id"], row["selection_metric"], row["scoring_scheme"]))
        )

    print(f"Wrote {len(all_rows)} rows to {args.results_csv}")
    print(f"Wrote {len(top_hit_rows)} top-hit rows to {args.top_hits_csv}")


if __name__ == "__main__":
    main()
