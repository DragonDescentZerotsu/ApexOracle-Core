from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_LINEAR_RESULTS = CURRENT_DIR / "linear_similarity_results.csv"
DEFAULT_CYCLIC_ROTATION_RESULTS = CURRENT_DIR / "cyclic_rotation_similarity_results.csv"
DEFAULT_CYCLIC_BEST_PID = CURRENT_DIR / "cyclic_best_by_pid.csv"
DEFAULT_CYCLIC_BEST_MAXLEN = CURRENT_DIR / "cyclic_best_by_max_len_identity.csv"
DEFAULT_REPORT_JSON = CURRENT_DIR / "similarity_validation_report.json"
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate saved similarity CSV outputs and check PID/max_len_identity invariants."
    )
    parser.add_argument("--linear-results", type=Path, default=DEFAULT_LINEAR_RESULTS)
    parser.add_argument("--cyclic-rotation-results", type=Path, default=DEFAULT_CYCLIC_ROTATION_RESULTS)
    parser.add_argument("--cyclic-best-pid", type=Path, default=DEFAULT_CYCLIC_BEST_PID)
    parser.add_argument("--cyclic-best-maxlen", type=Path, default=DEFAULT_CYCLIC_BEST_MAXLEN)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def validate_file(csv_path: Path, expected_selection_metric: str | None = None) -> dict:
    checks = {
        "row_count": 0,
        "formula_mismatches": 0,
        "pid_gt_max_len_identity": 0,
        "aligned_positions_lt_max_len": 0,
        "matches_gt_shorter_len": 0,
        "selection_metric_mismatches": 0,
        "first_formula_mismatch": None,
        "first_pid_gt_max_len_identity": None,
        "first_aligned_positions_lt_max_len": None,
        "first_matches_gt_shorter_len": None,
        "first_selection_metric_mismatch": None,
    }

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            checks["row_count"] += 1
            row_number = checks["row_count"] + 1

            query_length = int(row["query_length"])
            train_length = int(row["train_length"])
            matches = int(row["matches"])
            aligned_positions = int(row["aligned_positions_including_gaps"])
            pid = float(row["pid"])
            max_len_identity = float(row["max_len_identity"])
            max_length = max(query_length, train_length)
            shorter_length = min(query_length, train_length)

            expected_pid = matches / aligned_positions if aligned_positions else 0.0
            expected_max_len_identity = matches / max_length if max_length else 0.0

            if abs(pid - expected_pid) > TOLERANCE or abs(max_len_identity - expected_max_len_identity) > TOLERANCE:
                checks["formula_mismatches"] += 1
                if checks["first_formula_mismatch"] is None:
                    checks["first_formula_mismatch"] = {
                        "row_number": row_number,
                        "query_peptide_id": row["query_peptide_id"],
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "pid": pid,
                        "expected_pid": expected_pid,
                        "max_len_identity": max_len_identity,
                        "expected_max_len_identity": expected_max_len_identity,
                    }

            if pid > max_len_identity + TOLERANCE:
                checks["pid_gt_max_len_identity"] += 1
                if checks["first_pid_gt_max_len_identity"] is None:
                    checks["first_pid_gt_max_len_identity"] = {
                        "row_number": row_number,
                        "query_peptide_id": row["query_peptide_id"],
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "pid": pid,
                        "max_len_identity": max_len_identity,
                    }

            if aligned_positions < max_length:
                checks["aligned_positions_lt_max_len"] += 1
                if checks["first_aligned_positions_lt_max_len"] is None:
                    checks["first_aligned_positions_lt_max_len"] = {
                        "row_number": row_number,
                        "query_peptide_id": row["query_peptide_id"],
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "aligned_positions_including_gaps": aligned_positions,
                        "max_length": max_length,
                    }

            if matches > shorter_length:
                checks["matches_gt_shorter_len"] += 1
                if checks["first_matches_gt_shorter_len"] is None:
                    checks["first_matches_gt_shorter_len"] = {
                        "row_number": row_number,
                        "query_peptide_id": row["query_peptide_id"],
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "matches": matches,
                        "shorter_length": shorter_length,
                    }

            if expected_selection_metric is not None and row.get("selection_metric") != expected_selection_metric:
                checks["selection_metric_mismatches"] += 1
                if checks["first_selection_metric_mismatch"] is None:
                    checks["first_selection_metric_mismatch"] = {
                        "row_number": row_number,
                        "query_peptide_id": row["query_peptide_id"],
                        "train_dbaasp_id": row["train_dbaasp_id"],
                        "selection_metric": row.get("selection_metric"),
                        "expected_selection_metric": expected_selection_metric,
                    }

    checks["ok"] = all(
        checks[key] == 0
        for key in (
            "formula_mismatches",
            "pid_gt_max_len_identity",
            "aligned_positions_lt_max_len",
            "matches_gt_shorter_len",
            "selection_metric_mismatches",
        )
    )
    return checks


def main() -> None:
    args = parse_args()

    report = {
        "linear_results": validate_file(args.linear_results),
        "cyclic_rotation_results": validate_file(args.cyclic_rotation_results),
        "cyclic_best_by_pid": validate_file(args.cyclic_best_pid, expected_selection_metric="pid"),
        "cyclic_best_by_max_len_identity": validate_file(
            args.cyclic_best_maxlen, expected_selection_metric="max_len_identity"
        ),
    }
    report["overall_ok"] = all(section["ok"] for section in report.values())

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
