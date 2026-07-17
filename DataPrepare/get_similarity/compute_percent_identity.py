from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_CSV = CURRENT_DIR / "query_peptides.csv"
DEFAULT_LINEAR_CACHE = CURRENT_DIR / "train_linear_peptides.csv"
DEFAULT_CYCLIC_CACHE = CURRENT_DIR / "train_cyclic_peptides.csv"
DEFAULT_LINEAR_OUTPUT = CURRENT_DIR / "linear_similarity_results.csv"
DEFAULT_CYCLIC_ROTATION_OUTPUT = CURRENT_DIR / "cyclic_rotation_similarity_results.csv"
DEFAULT_CYCLIC_BEST_PID_OUTPUT = CURRENT_DIR / "cyclic_best_by_pid.csv"
DEFAULT_CYCLIC_BEST_MAXLEN_OUTPUT = CURRENT_DIR / "cyclic_best_by_max_len_identity.csv"
DEFAULT_RUN_MANIFEST = CURRENT_DIR / "similarity_run_manifest.json"

SCORING_SCHEMES = ("blosum62_needle", "exact_match_needle")
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
SUMMARY_FIELDNAMES = CYCLIC_FIELDNAMES + ["selection_metric"]

WORKER_ALIGNER: Align.PairwiseAligner | None = None
WORKER_SCHEME: str | None = None
BASE_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def build_chirality_aware_blosum62() -> Array:
    extended_alphabet = BASE_BLOSUM62.alphabet + "".join(
        residue.lower()
        for residue in BASE_BLOSUM62.alphabet
        if residue.isalpha() and residue.lower() not in BASE_BLOSUM62.alphabet
    )
    matrix = Array(alphabet=extended_alphabet, dims=2)
    for left_residue in extended_alphabet:
        for right_residue in extended_alphabet:
            left_base = left_residue.upper()
            right_base = right_residue.upper()
            base_score = float(BASE_BLOSUM62[left_base, right_base])
            if (
                left_residue.isalpha()
                and right_residue.isalpha()
                and left_residue.islower() != right_residue.islower()
            ):
                matrix[left_residue, right_residue] = min(0.0, base_score)
            else:
                matrix[left_residue, right_residue] = base_score
    return matrix


CHIRALITY_AWARE_BLOSUM62 = build_chirality_aware_blosum62()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute percent identity between query peptides and training peptides."
    )
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--linear-cache", type=Path, default=DEFAULT_LINEAR_CACHE)
    parser.add_argument("--cyclic-cache", type=Path, default=DEFAULT_CYCLIC_CACHE)
    parser.add_argument("--linear-output", type=Path, default=DEFAULT_LINEAR_OUTPUT)
    parser.add_argument("--cyclic-rotation-output", type=Path, default=DEFAULT_CYCLIC_ROTATION_OUTPUT)
    parser.add_argument("--cyclic-best-pid-output", type=Path, default=DEFAULT_CYCLIC_BEST_PID_OUTPUT)
    parser.add_argument("--cyclic-best-maxlen-output", type=Path, default=DEFAULT_CYCLIC_BEST_MAXLEN_OUTPUT)
    parser.add_argument("--run-manifest-output", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Worker process count for alignment computation.",
    )
    parser.add_argument(
        "--linear-chunk-size",
        type=int,
        default=500,
        help="Training peptides per worker chunk for linear queries.",
    )
    parser.add_argument(
        "--cyclic-chunk-size",
        type=int,
        default=40,
        help="Training peptides per worker chunk for cyclic queries.",
    )
    return parser.parse_args()


def normalize_sequence(raw_sequence: str) -> str:
    return raw_sequence.strip()


def load_query_peptides(query_csv: Path) -> list[dict]:
    queries: list[dict] = []
    with query_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
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
    return queries


def load_training_cache(cache_csv: Path) -> list[dict]:
    records: list[dict] = []
    with cache_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "train_dbaasp_id": row["dbaasp_id"].strip(),
                    "train_sequence": normalize_sequence(row["sequence"]),
                    "train_length": int(row["length"]),
                }
            )
    return records


def build_aligner(scheme_name: str) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner(mode="global")
    if scheme_name == "blosum62_needle":
        aligner.substitution_matrix = CHIRALITY_AWARE_BLOSUM62
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5
    elif scheme_name == "exact_match_needle":
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = 0.0
        aligner.extend_gap_score = 0.0
    else:
        raise ValueError(f"Unsupported scoring scheme: {scheme_name}")
    return aligner


def init_worker(scheme_name: str) -> None:
    global WORKER_ALIGNER, WORKER_SCHEME
    WORKER_SCHEME = scheme_name
    WORKER_ALIGNER = build_aligner(scheme_name)


def sanitize_sequence_for_scheme(sequence: str, scoring_scheme: str) -> str:
    if scoring_scheme != "blosum62_needle":
        return sequence
    sanitized_residues: list[str] = []
    chirality_aware_alphabet = set(CHIRALITY_AWARE_BLOSUM62.alphabet)
    for residue in sequence:
        if residue in chirality_aware_alphabet:
            sanitized_residues.append(residue)
        elif residue.islower():
            sanitized_residues.append("x")
        else:
            sanitized_residues.append("X")
    return "".join(sanitized_residues)


def sanitize_for_scheme(sequence: str) -> str:
    assert WORKER_SCHEME is not None
    return sanitize_sequence_for_scheme(sequence, WORKER_SCHEME)


def chunk_records(records: list[dict], chunk_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def rotate_sequence(sequence: str, rotation: int) -> str:
    if not sequence:
        return sequence
    rotation %= len(sequence)
    return sequence[rotation:] + sequence[:rotation]


def reconstruct_gapped_sequences(alignment: Align.Alignment, target: str, query: str) -> tuple[str, str]:
    target_blocks, query_blocks = alignment.aligned
    target_parts: list[str] = []
    query_parts: list[str] = []
    target_position = 0
    query_position = 0

    for (target_start, target_end), (query_start, query_end) in zip(target_blocks, query_blocks):
        if target_position < target_start:
            gap_segment = target[target_position:target_start]
            target_parts.append(gap_segment)
            query_parts.append("-" * len(gap_segment))
            target_position = target_start
        if query_position < query_start:
            gap_segment = query[query_position:query_start]
            target_parts.append("-" * len(gap_segment))
            query_parts.append(gap_segment)
            query_position = query_start

        aligned_target = target[target_start:target_end]
        aligned_query = query[query_start:query_end]
        target_parts.append(aligned_target)
        query_parts.append(aligned_query)
        target_position = target_end
        query_position = query_end

    if target_position < len(target):
        gap_segment = target[target_position:]
        target_parts.append(gap_segment)
        query_parts.append("-" * len(gap_segment))
    if query_position < len(query):
        gap_segment = query[query_position:]
        target_parts.append("-" * len(gap_segment))
        query_parts.append(gap_segment)

    gapped_target = "".join(target_parts)
    gapped_query = "".join(query_parts)
    if len(gapped_target) != len(gapped_query):
        raise ValueError("Aligned sequences must have the same gapped length")
    return gapped_target, gapped_query


def compute_alignment_metrics(target_sequence: str, query_sequence: str) -> tuple[int, int, float, float]:
    assert WORKER_ALIGNER is not None
    sanitized_target = sanitize_for_scheme(target_sequence)
    sanitized_query = sanitize_for_scheme(query_sequence)
    alignment = WORKER_ALIGNER.align(sanitized_target, sanitized_query)[0]
    gapped_target, gapped_query = reconstruct_gapped_sequences(alignment, target_sequence, query_sequence)
    matches = sum(
        1
        for target_residue, query_residue in zip(gapped_target, gapped_query)
        if target_residue == query_residue and target_residue != "-"
    )
    aligned_positions = len(gapped_target)
    max_length = max(len(target_sequence), len(query_sequence))
    pid = matches / aligned_positions if aligned_positions else 0.0
    max_len_identity = matches / max_length if max_length else 0.0
    return matches, aligned_positions, pid, max_len_identity


def linear_worker(task: tuple[dict, list[dict]]) -> list[dict]:
    query, training_chunk = task
    rows: list[dict] = []
    for training_record in training_chunk:
        matches, aligned_positions, pid, max_len_identity = compute_alignment_metrics(
            query["query_sequence"],
            training_record["train_sequence"],
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
                matches, aligned_positions, pid, max_len_identity = compute_alignment_metrics(
                    rotated_query,
                    rotated_train,
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


def row_key(row: dict, primary_metric: str) -> tuple[float, float, int, int, int]:
    secondary_metric = "max_len_identity" if primary_metric == "pid" else "pid"
    return (
        float(row[primary_metric]),
        float(row[secondary_metric]),
        int(row["matches"]),
        -int(row["query_rotation"]),
        -int(row["train_rotation"]),
    )


def update_best_row(best_rows: dict[tuple[str, str, str], dict], row: dict, primary_metric: str) -> None:
    key = (row["query_peptide_id"], row["train_dbaasp_id"], row["scoring_scheme"])
    current = best_rows.get(key)
    if current is None or row_key(row, primary_metric) > row_key(current, primary_metric):
        best_rows[key] = row.copy()


def write_summary_rows(output_path: Path, rows: Iterable[dict], selection_metric: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            summary_row = row.copy()
            summary_row["selection_metric"] = selection_metric
            writer.writerow(summary_row)


def progress_bar(total: int, description: str):
    if tqdm is None:
        return _NullProgressBar(total=total, description=description)
    return tqdm(total=total, desc=description, unit="align", file=sys.stdout, dynamic_ncols=True)


class _NullProgressBar:
    def __init__(self, total: int, description: str) -> None:
        self.total = total
        self.description = description
        self.current = 0
        print(f"{self.description}: 0/{self.total}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        print(f"{self.description}: {self.current}/{self.total}")

    def update(self, value: int) -> None:
        self.current += value
        print(f"{self.description}: {self.current}/{self.total}")


def run_linear_similarity(
    query: dict,
    training_records: list[dict],
    scoring_scheme: str,
    output_writer: csv.DictWriter,
    processes: int,
    chunk_size: int,
) -> int:
    tasks = [(query, chunk) for chunk in chunk_records(training_records, chunk_size)]
    row_count = 0
    with ProcessPoolExecutor(max_workers=processes, initializer=init_worker, initargs=(scoring_scheme,)) as executor:
        description = f"{query['query_peptide_id']} | {scoring_scheme}"
        with progress_bar(total=len(training_records), description=description) as bar:
            for chunk_rows in executor.map(linear_worker, tasks):
                for row in chunk_rows:
                    output_writer.writerow(row)
                    row_count += 1
                bar.update(len(chunk_rows))
    return row_count


def run_cyclic_similarity(
    query: dict,
    training_records: list[dict],
    scoring_scheme: str,
    output_writer: csv.DictWriter,
    processes: int,
    chunk_size: int,
) -> tuple[int, dict[tuple[str, str, str], dict], dict[tuple[str, str, str], dict]]:
    tasks = [(query, chunk) for chunk in chunk_records(training_records, chunk_size)]
    row_count = 0
    best_by_pid: dict[tuple[str, str, str], dict] = {}
    best_by_maxlen: dict[tuple[str, str, str], dict] = {}
    total_alignments = sum(query["query_length"] * record["train_length"] for record in training_records)

    with ProcessPoolExecutor(max_workers=processes, initializer=init_worker, initargs=(scoring_scheme,)) as executor:
        description = f"{query['query_peptide_id']} | {scoring_scheme}"
        with progress_bar(total=total_alignments, description=description) as bar:
            for chunk_rows in executor.map(cyclic_worker, tasks):
                for row in chunk_rows:
                    output_writer.writerow(row)
                    row_count += 1
                    update_best_row(best_by_pid, row, "pid")
                    update_best_row(best_by_maxlen, row, "max_len_identity")
                bar.update(len(chunk_rows))

    return row_count, best_by_pid, best_by_maxlen


def main() -> None:
    args = parse_args()

    queries = load_query_peptides(args.query_csv)
    linear_training = load_training_cache(args.linear_cache)
    cyclic_training = load_training_cache(args.cyclic_cache)

    linear_queries = [query for query in queries if query["cyclic"] == "No"]
    cyclic_queries = [query for query in queries if query["cyclic"] == "Yes"]
    unsupported_query_types = sorted({query["cyclic"] for query in queries if query["cyclic"] not in {"No", "Yes"}})
    if unsupported_query_types:
        raise ValueError(f"Unsupported query cyclic labels: {unsupported_query_types}")

    for output_path in (
        args.linear_output,
        args.cyclic_rotation_output,
        args.cyclic_best_pid_output,
        args.cyclic_best_maxlen_output,
        args.run_manifest_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(linear_queries)} linear queries and {len(cyclic_queries)} cyclic queries.")
    print(f"Loaded {len(linear_training)} linear training peptides and {len(cyclic_training)} cyclic training peptides.")
    print(f"Using {args.processes} worker processes.")

    linear_row_count = 0
    with args.linear_output.open("w", newline="", encoding="utf-8") as handle:
        linear_writer = csv.DictWriter(handle, fieldnames=LINEAR_FIELDNAMES)
        linear_writer.writeheader()
        for query in linear_queries:
            for scoring_scheme in SCORING_SCHEMES:
                print(
                    f"Computing linear similarity for query={query['query_peptide_id']} "
                    f"scheme={scoring_scheme} ..."
                )
                linear_row_count += run_linear_similarity(
                    query=query,
                    training_records=linear_training,
                    scoring_scheme=scoring_scheme,
                    output_writer=linear_writer,
                    processes=args.processes,
                    chunk_size=args.linear_chunk_size,
                )

    cyclic_row_count = 0
    best_pid_rows: dict[tuple[str, str, str], dict] = {}
    best_maxlen_rows: dict[tuple[str, str, str], dict] = {}
    with args.cyclic_rotation_output.open("w", newline="", encoding="utf-8") as handle:
        cyclic_writer = csv.DictWriter(handle, fieldnames=CYCLIC_FIELDNAMES)
        cyclic_writer.writeheader()
        for query in cyclic_queries:
            for scoring_scheme in SCORING_SCHEMES:
                print(
                    f"Computing cyclic similarity for query={query['query_peptide_id']} "
                    f"scheme={scoring_scheme} ..."
                )
                row_count, scheme_best_pid, scheme_best_maxlen = run_cyclic_similarity(
                    query=query,
                    training_records=cyclic_training,
                    scoring_scheme=scoring_scheme,
                    output_writer=cyclic_writer,
                    processes=args.processes,
                    chunk_size=args.cyclic_chunk_size,
                )
                cyclic_row_count += row_count
                best_pid_rows.update(scheme_best_pid)
                best_maxlen_rows.update(scheme_best_maxlen)

    sorted_best_pid_rows = [
        best_pid_rows[key]
        for key in sorted(best_pid_rows, key=lambda item: (item[0], item[1], item[2]))
    ]
    sorted_best_maxlen_rows = [
        best_maxlen_rows[key]
        for key in sorted(best_maxlen_rows, key=lambda item: (item[0], item[1], item[2]))
    ]
    write_summary_rows(args.cyclic_best_pid_output, sorted_best_pid_rows, "pid")
    write_summary_rows(args.cyclic_best_maxlen_output, sorted_best_maxlen_rows, "max_len_identity")

    cyclic_rotation_count_per_scheme = sum(
        query["query_length"] * training_record["train_length"]
        for query in cyclic_queries
        for training_record in cyclic_training
    )
    linear_expected_rows = len(linear_queries) * len(linear_training) * len(SCORING_SCHEMES)
    cyclic_expected_rows = cyclic_rotation_count_per_scheme * len(SCORING_SCHEMES)
    cyclic_summary_expected_rows = len(cyclic_queries) * len(cyclic_training) * len(SCORING_SCHEMES)

    if linear_row_count != linear_expected_rows:
        raise RuntimeError(f"Linear row count mismatch: expected {linear_expected_rows}, got {linear_row_count}")
    if cyclic_row_count != cyclic_expected_rows:
        raise RuntimeError(f"Cyclic row count mismatch: expected {cyclic_expected_rows}, got {cyclic_row_count}")
    if len(sorted_best_pid_rows) != cyclic_summary_expected_rows:
        raise RuntimeError(
            f"Cyclic best-by-pid row count mismatch: expected {cyclic_summary_expected_rows}, "
            f"got {len(sorted_best_pid_rows)}"
        )
    if len(sorted_best_maxlen_rows) != cyclic_summary_expected_rows:
        raise RuntimeError(
            f"Cyclic best-by-max_len_identity row count mismatch: expected {cyclic_summary_expected_rows}, "
            f"got {len(sorted_best_maxlen_rows)}"
        )

    manifest = {
        "inputs": {
            "query_csv": str(args.query_csv),
            "linear_cache": str(args.linear_cache),
            "cyclic_cache": str(args.cyclic_cache),
        },
        "outputs": {
            "linear_output": str(args.linear_output),
            "cyclic_rotation_output": str(args.cyclic_rotation_output),
            "cyclic_best_pid_output": str(args.cyclic_best_pid_output),
            "cyclic_best_maxlen_output": str(args.cyclic_best_maxlen_output),
        },
        "runtime": {
            "processes": args.processes,
            "linear_chunk_size": args.linear_chunk_size,
            "cyclic_chunk_size": args.cyclic_chunk_size,
        },
        "counts": {
            "linear_queries": len(linear_queries),
            "cyclic_queries": len(cyclic_queries),
            "linear_training_records": len(linear_training),
            "cyclic_training_records": len(cyclic_training),
            "linear_rows_written": linear_row_count,
            "cyclic_rotation_rows_written": cyclic_row_count,
            "cyclic_best_pid_rows": len(sorted_best_pid_rows),
            "cyclic_best_max_len_rows": len(sorted_best_maxlen_rows),
            "linear_expected_rows": linear_expected_rows,
            "cyclic_expected_rows": cyclic_expected_rows,
            "cyclic_summary_expected_rows": cyclic_summary_expected_rows,
        },
        "scoring_schemes": list(SCORING_SCHEMES),
        "blosum62_normalization": {
            "case_sensitive_d_amino_acids": True,
            "mixed_chirality_positive_scores_clamped_to": 0.0,
            "replacement": {
                "uppercase_noncanonical": "X",
                "lowercase_noncanonical": "x",
            },
        },
    }

    with args.run_manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {linear_row_count} linear rows to {args.linear_output}")
    print(f"Wrote {cyclic_row_count} cyclic rotation rows to {args.cyclic_rotation_output}")
    print(f"Wrote {len(sorted_best_pid_rows)} cyclic best-by-pid rows to {args.cyclic_best_pid_output}")
    print(
        f"Wrote {len(sorted_best_maxlen_rows)} cyclic best-by-max_len_identity rows "
        f"to {args.cyclic_best_maxlen_output}"
    )
    print(f"Wrote run manifest to {args.run_manifest_output}")


if __name__ == "__main__":
    main()
