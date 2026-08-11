#!/usr/bin/env python
"""Materialize non-destructive filter tiers for ATCC 29914 generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated-attempts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--peplink-root",
        type=Path,
        help="Optional source checkout; otherwise import installed PepLink==0.1.2.",
    )
    parser.add_argument("--mic-threshold", type=float, default=15.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def annotate_row(row: dict[str, str], peplink, mic_threshold: float) -> dict:
    canonical = row.get("canonical_smiles") or ""
    rdkit_valid = as_bool(row.get("rdkit_valid", ""))
    predicted = row.get("predicted_mic_uM") or ""
    finite_mic = False
    mic = None
    try:
        mic = float(predicted)
        finite_mic = math.isfinite(mic)
    except ValueError:
        pass
    parsed = peplink.smiles_to_aa_seqs(canonical) if rdkit_valid else None
    peplink_standard = bool(
        parsed is not None
        and parsed.sequence is not None
        and parsed.unsupported_reason is None
    )
    row.update(
        {
            "finite_clean_mic": finite_mic,
            "mic_le_15": bool(finite_mic and mic <= mic_threshold),
            "peplink_standard_peptide": peplink_standard,
            "peplink_sequence": parsed.sequence if peplink_standard else "",
            "peplink_topology": parsed.cyclization if peplink_standard else "",
            "peplink_unsupported_reason": (
                parsed.unsupported_reason if parsed is not None else "not_run"
            ),
            "strict_candidate": bool(
                finite_mic and mic <= mic_threshold and peplink_standard
            ),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.peplink_root is not None:
        peplink_root = str(args.peplink_root.resolve())
        if peplink_root not in sys.path:
            sys.path.insert(0, peplink_root)
    peplink = importlib.import_module("PepLink")
    if peplink.__version__ != "0.1.2":
        raise RuntimeError(f"Expected PepLink 0.1.2, found {peplink.__version__}")
    with args.evaluated_attempts.open(newline="", encoding="utf-8") as handle:
        rows = [
            annotate_row(dict(row), peplink, args.mic_threshold)
            for row in csv.DictReader(handle)
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "all_attempts_with_filter_flags.csv"
    strict_path = args.output_dir / "strict_candidates_mic_le_15_peplink.csv"
    fields = list(rows[0]) if rows else []
    write_csv(all_path, rows, fields)
    write_csv(
        strict_path,
        [row for row in rows if row["strict_candidate"]],
        fields,
    )

    tiers = {
        "attempted": lambda row: True,
        "complete": lambda row: as_bool(row.get("complete", "")),
        "rdkit_valid": lambda row: as_bool(row.get("rdkit_valid", "")),
        "legacy_amide_positive": lambda row: as_bool(row.get("has_amide_bond", "")),
        "finite_clean_mic": lambda row: bool(row["finite_clean_mic"]),
        "mic_le_15": lambda row: bool(row["mic_le_15"]),
        "peplink_standard_peptide": lambda row: bool(row["peplink_standard_peptide"]),
        "strict_candidate": lambda row: bool(row["strict_candidate"]),
    }
    tier_dir = args.output_dir / "tiers"
    tier_dir.mkdir(exist_ok=True)
    tier_paths = {}
    for index, (name, predicate) in enumerate(tiers.items()):
        tier_path = tier_dir / f"{index:02d}_{name}.csv"
        write_csv(tier_path, [row for row in rows if predicate(row)], fields)
        tier_paths[name] = tier_path
    counts = Counter(
        {name: sum(predicate(row) for row in rows) for name, predicate in tiers.items()}
    )
    manifest = {
        "schema_version": 1,
        "input": {
            "path": str(args.evaluated_attempts.resolve()),
            "sha256": sha256_file(args.evaluated_attempts),
        },
        "peplink": {
            "version": peplink.__version__,
            "source": (
                str(args.peplink_root.resolve())
                if args.peplink_root is not None
                else "installed_environment"
            ),
        },
        "mic_threshold_uM": args.mic_threshold,
        "tier_counts": dict(counts),
        "contract": (
            "All attempts remain in all_attempts_with_filter_flags.csv; tiers are boolean flags, "
            "and the strict export is MIC<=15 inclusive plus PepLink v0.1.2 reliable reverse parsing "
            "of a standard linear or head-to-tail cyclic peptide."
        ),
        "outputs": {
            str(path.relative_to(args.output_dir)): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (all_path, strict_path, *tier_paths.values())
        },
    }
    (args.output_dir / "filter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
