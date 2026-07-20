#!/usr/bin/env python3
"""Create a read-only identity manifest for precomputed k-mer tensors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.features.kmer import inspect_tensor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.embedding_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    if source in output.parents:
        raise ValueError("Manifest must be written outside the source tensor directory")

    rows = [
        asdict(inspect_tensor(path))
        for path in sorted(source.glob("*.pt"))
    ]
    if not rows:
        raise FileNotFoundError(f"No .pt tensors found in {source}")
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n")
    print(
        json.dumps(
            {
                "files": len(table),
                "bytes": int(table["bytes"].sum()),
                "dtypes": sorted(table["dtype"].unique()),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
