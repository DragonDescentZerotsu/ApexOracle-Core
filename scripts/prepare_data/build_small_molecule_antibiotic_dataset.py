#!/usr/bin/env python3
"""Read-only build and token-filter commands for the paper small molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.amp_mic import sha256_file  # noqa: E402
from apexoracle.data.amp_training_data import tokenize_and_filter_smiles  # noqa: E402
from apexoracle.data.small_molecule_antibiotics import (  # noqa: E402
    format_abaumannii_atcc17978,
    format_ecoli_bw25113,
    format_saureus_rn4220,
    merge_paper_small_molecule_tables,
    summarize_small_molecule_table,
)


def safe_output_path(output: Path, inputs: list[Path]) -> Path:
    resolved = output.expanduser().resolve()
    if resolved in {path.expanduser().resolve() for path in inputs}:
        raise ValueError(f"Refusing to overwrite an input file: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"Output already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build and explicitly order 3 sources")
    build.add_argument("--ecoli", type=Path, required=True)
    build.add_argument("--abaumannii", type=Path, required=True)
    build.add_argument("--saureus", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    tokenize = commands.add_parser("tokenize", help="SELFIES-tokenize and filter")
    tokenize.add_argument("--input", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)
    tokenize.add_argument(
        "--model-name", default="ibm-research/materials.selfies-ted"
    )
    tokenize.add_argument("--revision", required=True)
    tokenize.add_argument("--max-length", type=int, default=1024)
    tokenize.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        inputs = [
            args.ecoli.resolve(),
            args.abaumannii.resolve(),
            args.saureus.resolve(),
        ]
        output = safe_output_path(args.output, inputs)
        table = merge_paper_small_molecule_tables(
            format_ecoli_bw25113(pd.read_csv(inputs[0])),
            format_abaumannii_atcc17978(pd.read_csv(inputs[1])),
            format_saureus_rn4220(pd.read_csv(inputs[2])),
        )
        table.to_csv(output, index=False)
        summary = summarize_small_molecule_table(table)
        report = {
            "operation": "build",
            "input_sha256": [sha256_file(path) for path in inputs],
            "rows": summary.rows,
            "positives": summary.positives,
            "counts_by_strain": summary.counts_by_strain,
            "output_sha256": sha256_file(output),
        }
    else:
        import selfies
        from transformers import AutoTokenizer

        input_path = args.input.resolve()
        output = safe_output_path(args.output, [input_path])
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
        result = tokenize_and_filter_smiles(
            pd.read_csv(input_path),
            selfies_encoder=selfies.encoder,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        result.table.to_csv(output, index=False)
        report = {
            "operation": "tokenize",
            "input_sha256": sha256_file(input_path),
            "model_name": args.model_name,
            "revision": args.revision,
            "max_length": args.max_length,
            "rows": len(result.table),
            "unique_smiles_tokenized": result.unique_smiles_tokenized,
            "excluded": {
                "invalid_smiles": result.excluded_invalid_smiles,
                "too_long": result.excluded_too_long,
                "unknown_token": result.excluded_unknown_token,
            },
            "output_sha256": sha256_file(output),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
