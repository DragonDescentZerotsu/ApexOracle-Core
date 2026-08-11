#!/usr/bin/env python
"""Prepare and validate exact-target assets for P. stuartii ATCC 29914."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from Bio import SeqIO


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    REPO_ROOT
    / "experiments"
    / "reviewer4_unseen_targets"
    / "providencia_stuartii_atcc_29914"
)
STEM = "Providencia_stuartii_ATCC_29914"
EXPECTED_LENGTH = 4_438_675


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atcc-fasta", type=Path)
    parser.add_argument("--atcc-genbank", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--length-tolerance", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    """Prefer portable repository-relative paths in the public manifest."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def normalize_historical_text(text: str) -> str:
    """Apply the exact strain-name replacement used by the historical encoder."""
    return text.replace(STEM.replace("_", " "), "This strain")


def validate_fasta(path: Path, tolerance: int) -> dict:
    records = list(SeqIO.parse(path, "fasta"))
    lengths = [len(record.seq) for record in records]
    total = sum(lengths)
    if len(records) != 1:
        raise ValueError(
            f"Expected the ATCC portal single-contig assembly; found {len(records)} records"
        )
    if abs(total - EXPECTED_LENGTH) > tolerance:
        raise ValueError(
            f"Expected {EXPECTED_LENGTH} bp for ATCC 29914; found {total} bp. "
            "Do not substitute GCF_029075745.1."
        )
    return {
        "path": manifest_path(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "record_ids": [record.id for record in records],
        "lengths_bp": lengths,
        "total_length_bp": total,
    }


def validate_genbank(path: Path, tolerance: int) -> tuple[dict, str]:
    records = list(SeqIO.parse(path, "genbank"))
    lengths = [len(record.seq) for record in records]
    total = sum(lengths)
    if len(records) != 1:
        raise ValueError(
            f"Expected one ATCC annotation record; found {len(records)} records"
        )
    if abs(total - EXPECTED_LENGTH) > tolerance:
        raise ValueError(
            f"Expected {EXPECTED_LENGTH} annotated bp for ATCC 29914; found {total}."
        )
    metadata = {
        "path": manifest_path(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "record_ids": [record.id for record in records],
        "lengths_bp": lengths,
        "total_length_bp": total,
        "feature_count": sum(len(record.features) for record in records),
    }
    return metadata, str(records[0].seq).upper()


def copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)


def main() -> None:
    args = parse_args()
    source_text = EXPERIMENT_DIR / "strain_description.txt"
    text = source_text.read_text(encoding="utf-8").strip() + "\n"
    normalized = normalize_historical_text(text)
    normalized_path = EXPERIMENT_DIR / "med_llama3_input.txt"
    normalized_path.write_text(normalized, encoding="utf-8")

    fasta_metadata = None
    fasta_sequence = None
    if args.atcc_fasta is not None:
        fasta_path = args.atcc_fasta.resolve()
        fasta_metadata = validate_fasta(fasta_path, args.length_tolerance)
        fasta_sequence = str(next(SeqIO.parse(fasta_path, "fasta")).seq).upper()

    genbank_metadata = None
    annotation_sequence_matches_fasta = None
    if args.atcc_genbank is not None:
        genbank_metadata, genbank_sequence = validate_genbank(
            args.atcc_genbank.resolve(), args.length_tolerance
        )
        if fasta_sequence is not None:
            annotation_sequence_matches_fasta = genbank_sequence == fasta_sequence
            if not annotation_sequence_matches_fasta:
                raise ValueError("ATCC FASTA and GenBank annotation sequences differ")

    installed = {}
    if args.install:
        text_target = (
            REPO_ROOT
            / "DataPrepare"
            / "Data"
            / "Text_Description"
            / "ATCC"
            / "Text"
            / f"{STEM}.txt"
        )
        copy_if_needed(source_text, text_target)
        installed["text"] = manifest_path(text_target)
        if args.atcc_fasta is not None:
            fasta_target = (
                REPO_ROOT / "DataPrepare" / "Data" / "Genome" / "ATCC" / f"{STEM}.fasta"
            )
            copy_if_needed(args.atcc_fasta, fasta_target)
            installed["fasta"] = manifest_path(fasta_target)
        if args.atcc_genbank is not None:
            genbank_target = (
                REPO_ROOT
                / "DataPrepare"
                / "Data"
                / "Genome_annotation"
                / "ATCC"
                / f"{STEM}.gbk"
            )
            copy_if_needed(args.atcc_genbank, genbank_target)
            installed["genbank"] = manifest_path(genbank_target)

    manifest = {
        "schema_version": 1,
        "source_text": {
            "path": str(source_text.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
        "med_llama3_input": {
            "path": str(normalized_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "normalization": "historical full strain-name replacement with 'This strain'",
            "model": "YBXL/Med-LLaMA3-8B",
            "hidden_state": -2,
        },
        "fasta": fasta_metadata,
        "genbank": genbank_metadata,
        "annotation_sequence_matches_fasta": annotation_sequence_matches_fasta,
        "installed": installed,
        "blocked": fasta_metadata is None,
        "block_reason": (
            "Exact ATCC Genome Portal FASTA is not available locally"
            if fasta_metadata is None
            else None
        ),
    }
    (EXPERIMENT_DIR / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
