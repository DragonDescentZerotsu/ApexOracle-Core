from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from Bio.Align import substitution_matrices


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_CSV = CURRENT_DIR.parent / "Data" / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv"
DEFAULT_ALL_PEPTIDES_JSON = CURRENT_DIR.parent / "Data" / "all_peptides_data.json"
DEFAULT_LINEAR_OUTPUT = CURRENT_DIR / "train_linear_peptides.csv"
DEFAULT_CYCLIC_OUTPUT = CURRENT_DIR / "train_cyclic_peptides.csv"
DEFAULT_MANIFEST_OUTPUT = CURRENT_DIR / "training_peptide_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated training peptides and split them into linear/cyclic caches."
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=DEFAULT_TRAINING_CSV,
        help="CSV with the DBAASP_id column used to define the training set.",
    )
    parser.add_argument(
        "--all-peptides-json",
        type=Path,
        default=DEFAULT_ALL_PEPTIDES_JSON,
        help="JSON exported from DBAASP with the top-level data list.",
    )
    parser.add_argument(
        "--linear-output",
        type=Path,
        default=DEFAULT_LINEAR_OUTPUT,
        help="Output CSV for linear training peptides.",
    )
    parser.add_argument(
        "--cyclic-output",
        type=Path,
        default=DEFAULT_CYCLIC_OUTPUT,
        help="Output CSV for cyclic training peptides.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
        help="Output JSON manifest with counts and skipped IDs.",
    )
    return parser.parse_args()


def load_training_ids(training_csv: Path) -> tuple[int, list[str]]:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    total_rows = 0
    with training_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            raw_id = str(row["DBAASP_id"]).strip()
            if raw_id and raw_id not in seen:
                seen.add(raw_id)
                ordered_ids.append(raw_id)
    return total_rows, ordered_ids


def load_all_peptides(all_peptides_json: Path) -> list[dict]:
    with all_peptides_json.open(encoding="utf-8") as handle:
        blob = json.load(handle)
    if isinstance(blob, dict):
        data = blob.get("data")
        if not isinstance(data, list):
            raise ValueError("Expected top-level 'data' list in all_peptides_data.json")
        return data
    if isinstance(blob, list):
        return blob
    raise ValueError("Unsupported all_peptides_data.json payload")


def classify_peptide(item: dict) -> str:
    bonds = item.get("intrachainBonds") or []
    if bonds == []:
        return "linear"
    if len(bonds) == 1:
        chain_name = ((bonds[0].get("chainParticipating") or {}).get("name") or "").strip()
        if chain_name == "MMB":
            return "cyclic"
    return "skip"


def normalize_sequence(raw_sequence: str | None) -> str:
    return (raw_sequence or "").strip()


def write_training_cache(output_path: Path, records: Iterable[tuple[str, str, int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dbaasp_id", "sequence", "length"])
        writer.writerows(records)


def main() -> None:
    args = parse_args()

    print(f"Loading training IDs from {args.training_csv} ...")
    total_training_rows, unique_training_ids = load_training_ids(args.training_csv)
    print(f"  Total rows: {total_training_rows}")
    print(f"  Unique training IDs: {len(unique_training_ids)}")

    print(f"Loading peptide JSON from {args.all_peptides_json} ...")
    all_peptides = load_all_peptides(args.all_peptides_json)
    items_by_id = {str(item.get('id')): item for item in all_peptides}
    print(f"  JSON peptide items: {len(all_peptides)}")

    blosum62_alphabet = set(substitution_matrices.load("BLOSUM62").alphabet)

    linear_records: list[tuple[str, str, int]] = []
    cyclic_records: list[tuple[str, str, int]] = []
    skipped_rule_ids: list[str] = []
    missing_sequence_ids: list[str] = []
    unmatched_training_ids: list[str] = []
    blosum62_unsupported_ids: dict[str, list[str]] = {}

    for dbaasp_id in unique_training_ids:
        item = items_by_id.get(dbaasp_id)
        if item is None:
            unmatched_training_ids.append(dbaasp_id)
            continue

        sequence = normalize_sequence(item.get("sequence"))
        if not sequence:
            missing_sequence_ids.append(dbaasp_id)
            continue

        unsupported_residues = sorted({residue for residue in sequence if residue not in blosum62_alphabet})
        if unsupported_residues:
            blosum62_unsupported_ids[dbaasp_id] = unsupported_residues

        peptide_type = classify_peptide(item)
        record = (dbaasp_id, sequence, len(sequence))
        if peptide_type == "linear":
            linear_records.append(record)
        elif peptide_type == "cyclic":
            cyclic_records.append(record)
        else:
            skipped_rule_ids.append(dbaasp_id)

    write_training_cache(args.linear_output, linear_records)
    write_training_cache(args.cyclic_output, cyclic_records)

    manifest = {
        "inputs": {
            "training_csv": str(args.training_csv),
            "all_peptides_json": str(args.all_peptides_json),
        },
        "outputs": {
            "linear_output": str(args.linear_output),
            "cyclic_output": str(args.cyclic_output),
            "manifest_output": str(args.manifest_output),
        },
        "counts": {
            "training_csv_rows": total_training_rows,
            "unique_training_ids": len(unique_training_ids),
            "json_items": len(all_peptides),
            "matched_training_ids": len(unique_training_ids) - len(unmatched_training_ids),
            "linear_training_peptides": len(linear_records),
            "cyclic_training_peptides": len(cyclic_records),
            "skipped_rule_ids": len(skipped_rule_ids),
            "missing_sequence_ids": len(missing_sequence_ids),
            "unmatched_training_ids": len(unmatched_training_ids),
            "blosum62_unsupported_ids": len(blosum62_unsupported_ids),
        },
        "blosum62_normalization": {
            "replacement": {"O": "X"},
            "unsupported_ids": blosum62_unsupported_ids,
        },
        "unmatched_training_ids": unmatched_training_ids,
        "missing_sequence_ids": missing_sequence_ids,
        "skipped_rule_ids": skipped_rule_ids,
    }

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {len(linear_records)} linear peptides to {args.linear_output}")
    print(f"Wrote {len(cyclic_records)} cyclic peptides to {args.cyclic_output}")
    print(f"Wrote manifest to {args.manifest_output}")


if __name__ == "__main__":
    main()
