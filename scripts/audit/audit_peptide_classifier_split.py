#!/usr/bin/env python
"""Audit the frozen Reviewer 2 peptide-classifier split artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from apexoracle.data.peptide_classifier import split_codes_for_digests


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("experiments/peptide_classifier/reviewer_retrain"),
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()
    manifest = json.loads((args.split_dir / "split_manifest.json").read_text())
    rows = int(manifest["row_count"])
    files = manifest["files"]
    for filename, expected in files.items():
        path = args.split_dir / filename
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Size mismatch: {path}")
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")

    digests = np.memmap(
        args.split_dir / "canonical_digest_128.bin",
        mode="r",
        dtype="V16",
        shape=(rows,),
    )
    stored_splits = np.memmap(
        args.split_dir / "split_codes.u1", mode="r", dtype=np.uint8, shape=(rows,)
    )
    stored_molecules = np.memmap(
        args.split_dir / "molecule_hashes.u8",
        mode="r",
        dtype=np.uint64,
        shape=(rows,),
    )
    real_rows = (args.split_dir / "real_sequence_roots.u8").stat().st_size // 8
    real_canonicals = np.memmap(
        args.split_dir / "real_canonical_digest_128.bin",
        mode="r",
        dtype="V16",
        shape=(real_rows,),
    )
    real_roots = np.memmap(
        args.split_dir / "real_sequence_roots.u8",
        mode="r",
        dtype=np.uint64,
        shape=(real_rows,),
    )
    sortable_real = real_canonicals.view("S16")
    if np.any(sortable_real[1:] <= sortable_real[:-1]):
        raise RuntimeError("Real-peptide canonical map is not strictly sorted/unique")

    counts = np.zeros(3, dtype=np.int64)
    for begin in range(0, rows, args.chunk_size):
        end = min(rows, begin + args.chunk_size)
        expected_splits, expected_molecules = split_codes_for_digests(
            digests[begin:end],
            seed=int(manifest["seed"]),
            real_canonicals=real_canonicals,
            real_sequence_roots=real_roots,
        )
        if not np.array_equal(expected_splits, stored_splits[begin:end]):
            raise RuntimeError(f"Split recomputation failed at rows {begin}:{end}")
        if not np.array_equal(expected_molecules, stored_molecules[begin:end]):
            raise RuntimeError(f"Molecule hash recomputation failed at rows {begin}:{end}")
        counts += np.bincount(expected_splits, minlength=3)

    expected_counts = manifest["split_counts"]
    observed = {
        "train": int(counts[0]),
        "validation": int(counts[1]),
        "test": int(counts[2]),
    }
    if observed != expected_counts:
        raise RuntimeError(f"Split counts differ: observed={observed}, expected={expected_counts}")
    output = {
        "audit_status": "passed",
        "canonical_and_sequence_group_assignment_recomputed": True,
        "file_hashes_verified": True,
        "row_count": rows,
        "split_counts": observed,
    }
    output_path = args.split_dir / "split_audit.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
