#!/usr/bin/env python3
"""Build AMR/MGE labels for the genomic segments in saved tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import sha256_file  # noqa: E402
from apexoracle.evaluation import genome_condition_reviewer  # noqa: E402
from apexoracle.evaluation import genome_fragment_validation  # noqa: E402
from apexoracle.evaluation.genome_fragment_validation import (  # noqa: E402
    build_fragment_annotation_manifest,
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/genome_condition_reviewer/historical_probe/manifests"
)


def directory_contract_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "DataPrepare/Data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    genomes, fragments, summary = build_fragment_annotation_manifest(
        data_dir=data_dir, output_dir=output_dir
    )
    if genomes["genome_id"].duplicated().any():
        raise RuntimeError("Probe manifest contains duplicate genome IDs")
    if fragments.duplicated(["genome_id", "fragment_index"]).any():
        raise RuntimeError("Probe manifest contains duplicate fragment keys")
    if int(genomes["fragments"].sum()) != len(fragments):
        raise RuntimeError("Genome and fragment manifest row counts differ")

    genomes_path = output_dir / "compatible_genomes.csv"
    labels_path = output_dir / "fragment_annotation_labels.csv"
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "scope": (
            "all paper-dataset-matched bacterial embeddings with exact FASTA/GenBank "
            "sequence order and saved-tensor-compatible window reconstruction"
        ),
        "summary": summary,
        "source_sha256": {
            "window_reconstruction_module": sha256_file(
                Path(genome_condition_reviewer.__file__)
            ),
            "fragment_validation_module": sha256_file(
                Path(genome_fragment_validation.__file__)
            ),
            "entrypoint": sha256_file(Path(__file__)),
        },
        "asset_directory_contract_sha256": directory_contract_hash(
            list((data_dir / "Genome_embs").iterdir())
        ),
        "output_sha256": {
            "compatible_genomes": sha256_file(genomes_path),
            "fragment_annotation_labels": sha256_file(labels_path),
        },
        "outputs": {
            "compatible_genomes": genomes_path.name,
            "fragment_annotation_labels": labels_path.name,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
