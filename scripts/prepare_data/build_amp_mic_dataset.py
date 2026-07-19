#!/usr/bin/env python3
"""Rebuild the frozen paper AMP MIC table into an independent output path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.amp_mic import (  # noqa: E402
    PAPER_PROTOCOL,
    build_paper_mic_table,
    load_paper_mic_inputs,
    sha256_file,
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_paths(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {_resolved(path) for path in inputs}
    for output in outputs:
        resolved = _resolved(output)
        if resolved in input_paths:
            raise ValueError(f"Refusing to overwrite an input file: {resolved}")
        if resolved.exists():
            raise FileExistsError(
                f"Output already exists: {resolved}. Choose a new output directory."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only reconstruction of the paper AMP MIC dataset."
    )
    parser.add_argument("--dbaasp-json", type=Path, required=True)
    parser.add_argument("--smiles-csv", type=Path, required=True)
    parser.add_argument(
        "--molecular-weight-smiles-overrides",
        type=Path,
        help=(
            "Optional pre-correction SMILES used only for historical µg/ml-to-µM "
            "conversion. The displayed output SMILES still comes from --smiles-csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", default=PAPER_PROTOCOL, choices=[PAPER_PROTOCOL])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolved(args.output_dir)
    table_path = output_dir / "DBAASP_id_bact_name_SMILES_MIC_Evo.csv"
    counts_path = output_dir / "MIC_data_count.json"
    manifest_path = output_dir / "dataset_manifest.json"
    inputs = [_resolved(args.dbaasp_json), _resolved(args.smiles_csv)]
    if args.molecular_weight_smiles_overrides is not None:
        inputs.append(_resolved(args.molecular_weight_smiles_overrides))
    outputs = [table_path, counts_path, manifest_path]
    validate_paths(inputs, outputs)

    records, smiles = load_paper_mic_inputs(*inputs[:2])
    weight_overrides = pd.read_csv(inputs[2]) if len(inputs) == 3 else None
    result = build_paper_mic_table(
        records,
        smiles,
        molecular_weight_smiles_overrides=weight_overrides,
        protocol=args.protocol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(table_path, index=False)
    counts_path.write_text(
        json.dumps(result.strain_counts, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    manifest = {
        "protocol": args.protocol,
        "inputs": {
            "dbaasp_json": {"path": str(inputs[0]), "sha256": sha256_file(inputs[0])},
            "smiles_csv": {"path": str(inputs[1]), "sha256": sha256_file(inputs[1])},
        },
        "output": {
            "path": str(table_path),
            "rows": len(result.table),
            "unique_dbaasp_ids": int(result.table["DBAASP_id"].nunique()),
            "sha256": sha256_file(table_path),
        },
        "unusual_concentration_occurrences": len(result.unusual_concentrations),
    }
    if len(inputs) == 3:
        manifest["inputs"]["molecular_weight_smiles_overrides"] = {
            "path": str(inputs[2]),
            "sha256": sha256_file(inputs[2]),
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
