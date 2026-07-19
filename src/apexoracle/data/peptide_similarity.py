"""Build the linear/cyclic training caches used by sequence similarity."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from Bio.Align import substitution_matrices


def load_training_ids(training_csv: Path) -> tuple[int, list[str]]:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    total_rows = 0
    with training_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "DBAASP_id" not in (reader.fieldnames or ()):
            raise ValueError("Training CSV must contain DBAASP_id")
        for row in reader:
            total_rows += 1
            dbaasp_id = str(row["DBAASP_id"]).strip()
            if dbaasp_id and dbaasp_id not in seen:
                seen.add(dbaasp_id)
                ordered_ids.append(dbaasp_id)
    return total_rows, ordered_ids


def load_all_peptides(all_peptides_json: Path) -> list[dict]:
    with all_peptides_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise ValueError("Expected a list or a top-level 'data' list")
    return payload


def classify_peptide(item: dict) -> str:
    """Preserve the paper cache rule based only on ``intrachainBonds``."""

    bonds = item.get("intrachainBonds") or []
    if bonds == []:
        return "linear"
    if len(bonds) == 1:
        chain_name = (
            (bonds[0].get("chainParticipating") or {}).get("name") or ""
        ).strip()
        if chain_name == "MMB":
            return "cyclic"
    return "skip"


def _write_cache(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dbaasp_id", "sequence", "length"])
        writer.writerows(rows)


def build_training_caches(
    *,
    training_csv: Path,
    all_peptides_json: Path,
    linear_output: Path,
    cyclic_output: Path,
    manifest_output: Path,
    sequence_case: str = "preserve",
) -> dict:
    if sequence_case not in {"preserve", "uppercase"}:
        raise ValueError("sequence_case must be 'preserve' or 'uppercase'")
    total_rows, training_ids = load_training_ids(training_csv)
    all_peptides = load_all_peptides(all_peptides_json)
    items_by_id = {str(item.get("id")): item for item in all_peptides}
    blosum_alphabet = set(substitution_matrices.load("BLOSUM62").alphabet)

    linear: list[tuple[str, str, int]] = []
    cyclic: list[tuple[str, str, int]] = []
    skipped: list[str] = []
    missing_sequence: list[str] = []
    unmatched: list[str] = []
    unsupported: dict[str, list[str]] = {}
    for dbaasp_id in training_ids:
        item = items_by_id.get(dbaasp_id)
        if item is None:
            unmatched.append(dbaasp_id)
            continue
        sequence = (item.get("sequence") or "").strip()
        if sequence_case == "uppercase":
            sequence = sequence.upper()
        if not sequence:
            missing_sequence.append(dbaasp_id)
            continue
        residues = sorted(
            {residue for residue in sequence if residue not in blosum_alphabet}
        )
        if residues:
            unsupported[dbaasp_id] = residues
        record = (dbaasp_id, sequence, len(sequence))
        peptide_type = classify_peptide(item)
        if peptide_type == "linear":
            linear.append(record)
        elif peptide_type == "cyclic":
            cyclic.append(record)
        else:
            skipped.append(dbaasp_id)

    _write_cache(linear_output, linear)
    _write_cache(cyclic_output, cyclic)
    manifest = {
        "schema_version": 1,
        "inputs": {
            "training_csv": str(training_csv),
            "all_peptides_json": str(all_peptides_json),
        },
        "outputs": {
            "linear_output": str(linear_output),
            "cyclic_output": str(cyclic_output),
            "manifest_output": str(manifest_output),
        },
        "counts": {
            "training_csv_rows": total_rows,
            "unique_training_ids": len(training_ids),
            "json_items": len(all_peptides),
            "matched_training_ids": len(training_ids) - len(unmatched),
            "linear_training_peptides": len(linear),
            "cyclic_training_peptides": len(cyclic),
            "skipped_rule_ids": len(skipped),
            "missing_sequence_ids": len(missing_sequence),
            "unmatched_training_ids": len(unmatched),
            "blosum62_unsupported_ids": len(unsupported),
        },
        "blosum62_normalization": {
            "sequence_case": sequence_case,
            "uppercase_noncanonical": "X",
            "lowercase_noncanonical": "x",
            "unsupported_ids": unsupported,
        },
        "unmatched_training_ids": unmatched,
        "missing_sequence_ids": missing_sequence,
        "skipped_rule_ids": skipped,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest
