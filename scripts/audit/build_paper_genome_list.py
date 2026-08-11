#!/usr/bin/env python3
"""Build the compact genome list used by ApexOracle paper datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import (  # noqa: E402
    PAPER_DATASET_COLUMNS,
    build_paper_genome_list,
    genome_embedding_paths,
    matched_genome_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=REPO_ROOT / "experiments/evo2_genome_embeddings/file_manifest.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "experiments/evo2_genome_embeddings/paper_genome_list.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=REPO_ROOT
        / "experiments/evo2_genome_embeddings/paper_genome_list_manifest.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_csv = args.output_csv.resolve()
    output_manifest = args.output_manifest.resolve()
    for output in (output_csv, output_manifest):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    embedding_manifest_path = args.embedding_manifest.resolve()
    embedding_manifest = pd.read_csv(embedding_manifest_path, dtype={"genome_id": str})
    embedding_paths = genome_embedding_paths(data_dir / "Genome_embs")
    dataset_ids = {
        column: matched_genome_ids(data_dir, embedding_paths, (relative_path,))
        for relative_path, column in PAPER_DATASET_COLUMNS.items()
    }
    table = build_paper_genome_list(
        embedding_manifest, data_dir / "Genome" / "ATCC", dataset_ids
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False, lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "scope": "genomes used by at least one ApexOracle paper MIC, classification, or synergy dataset",
        "paper_genome_count": len(table),
        "task_counts": {
            key.removeprefix("used_by_"): int(table[key].sum())
            for key in sorted(dataset_ids)
        },
        "source_embedding_manifest": {
            "path": str(embedding_manifest_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(embedding_manifest_path),
        },
        "paper_genome_list": {
            "path": str(output_csv.relative_to(REPO_ROOT)),
            "size_bytes": output_csv.stat().st_size,
            "sha256": sha256_file(output_csv),
        },
        "boundaries": [
            "current_fasta_* identifies the filename-matched FASTA archive available during the release audit",
            "the original extraction logs were not recovered, so current FASTA byte identity with the original producer inputs is not asserted",
            "not_recovered is retained where an exact external accession could not be verified",
            "no sequences, embeddings, MIC labels, molecule structures, or private assay rows are included",
        ],
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
