#!/usr/bin/env python3
"""Export the compact reviewer-facing paper strain mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.strain_mapping_release import (  # noqa: E402
    build_paper_strain_mapping,
    write_paper_strain_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the paper-era DataPrepare/Data directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "assets" / "manifests" / "paper_strain_mapping.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_paper_strain_mapping(args.data_root)
    write_paper_strain_mapping(document, args.output)
    summary = document["summary"]
    print(
        f"wrote {args.output}: {summary['mapped_source_strain_count']} source "
        f"strains, {summary['mapping_row_count']} condition routes, "
        f"{summary['mapped_mic_route_record_count']} routed MIC records"
    )


if __name__ == "__main__":
    main()
