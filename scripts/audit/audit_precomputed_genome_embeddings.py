#!/usr/bin/env python3
"""Build a read-only identity manifest for precomputed Evo-2 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import (  # noqa: E402
    build_file_manifest,
    genome_embedding_paths,
    manifest_identity,
    matched_genome_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    embeddings = genome_embedding_paths(data_dir / "Genome_embs")
    matched = matched_genome_ids(data_dir, embeddings)
    manifest = build_file_manifest(embeddings, matched)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    if data_dir in output.parents:
        raise ValueError("Manifest output must be outside the source data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, lineterminator="\n")
    report = {
        "files": len(manifest),
        "bytes": int(manifest["bytes"].sum()),
        "matched_by_paper_datasets": int(manifest["used_by_paper_datasets"].sum()),
        "unused_ids": manifest.loc[
            ~manifest["used_by_paper_datasets"], "genome_id"
        ].tolist(),
        "manifest_sha256": manifest_identity(manifest),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
