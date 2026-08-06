#!/usr/bin/env python3
"""Run held-out AMR/MGE probes with each genome confined to one fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import sha256_file  # noqa: E402
from apexoracle.evaluation import genome_fragment_validation  # noqa: E402
from apexoracle.evaluation.genome_fragment_validation import (  # noqa: E402
    ProbeConfig,
    parse_boolean_series,
    run_linear_probe,
)

DEFAULT_MANIFEST_DIR = (
    REPO_ROOT / "experiments/genome_condition_reviewer/historical_probe/manifests"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/genome_condition_reviewer/historical_probe/analysis"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=REPO_ROOT / "DataPrepare/Data/Genome_embs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = manifest_dir / "fragment_annotation_labels.csv"
    compatible_path = manifest_dir / "compatible_genomes.csv"
    preparation_manifest = manifest_dir / "manifest.json"
    labels = pd.read_csv(labels_path, dtype={"genome_id": str})
    for column in ("amr_associated", "mge_associated"):
        labels[column] = parse_boolean_series(labels[column])
    probes = [
        run_linear_probe(
            labels,
            config=ProbeConfig(label_column="amr_associated", display_name="AMR"),
            embedding_dir=args.embedding_dir.resolve(),
            output_dir=output_dir,
        ),
        run_linear_probe(
            labels,
            config=ProbeConfig(
                label_column="mge_associated", display_name="Mobile element"
            ),
            embedding_dir=args.embedding_dir.resolve(),
            output_dir=output_dir,
        ),
    ]
    summary = {
        "schema_version": 1,
        "status": "completed",
        "scope": "saved bacterial fragment tensors with compatible annotations",
        "linear_probes": probes,
        "interpretation_boundary": (
            "Tests linear decodability of conservative existing annotations; not a "
            "complete resistome or mobile-element catalogue."
        ),
        "source_sha256": {
            "compatible_genomes": sha256_file(compatible_path),
            "fragment_labels": sha256_file(labels_path),
            "preparation_manifest": sha256_file(preparation_manifest),
            "fragment_validation_module": sha256_file(
                Path(genome_fragment_validation.__file__)
            ),
            "entrypoint": sha256_file(Path(__file__)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
