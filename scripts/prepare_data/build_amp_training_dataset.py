#!/usr/bin/env python3
"""Read-only merge and token-filter commands for the paper AMP data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.amp_mic import sha256_file  # noqa: E402
from apexoracle.data.amp_training_data import (  # noqa: E402
    format_inhouse_mic_table,
    merge_mic_tables,
    tokenize_and_filter_smiles,
)


def safe_output_path(output: Path, inputs: list[Path]) -> Path:
    resolved = output.expanduser().resolve()
    input_paths = {path.expanduser().resolve() for path in inputs}
    if resolved in input_paths:
        raise ValueError(f"Refusing to overwrite an input file: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"Output already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="Append frozen in-house MIC rows")
    merge.add_argument("--dbaasp-mic", type=Path, required=True)
    merge.add_argument("--inhouse-mic", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    inhouse = subparsers.add_parser(
        "format-inhouse", help="Convert the APEX wide matrix through PepLink"
    )
    inhouse.add_argument("--input", type=Path, required=True)
    inhouse.add_argument("--output", type=Path, required=True)

    tokenize = subparsers.add_parser("tokenize", help="SELFIES-tokenize and filter")
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
    if args.command == "format-inhouse":
        from apexoracle.data.peplink_adapter import sequence_to_smiles

        input_path = args.input.resolve()
        output = safe_output_path(args.output, [input_path])
        source = pd.read_csv(input_path)
        sequences = source["Peptide"].astype(str).drop_duplicates()
        smiles_by_sequence = {
            sequence: sequence_to_smiles(sequence) for sequence in sequences
        }
        table = format_inhouse_mic_table(source, smiles_by_sequence)
        table.to_csv(output, index=False)
        report = {
            "operation": "format-inhouse",
            "input_sha256": sha256_file(input_path),
            "peplink_version": "0.1.1",
            "unique_sequences": len(sequences),
            "rows": len(table),
            "output_sha256": sha256_file(output),
        }
    elif args.command == "merge":
        inputs = [args.dbaasp_mic.resolve(), args.inhouse_mic.resolve()]
        output = safe_output_path(args.output, inputs)
        table = merge_mic_tables(*(pd.read_csv(path) for path in inputs))
        table.to_csv(output, index=False)
        report = {
            "operation": "merge",
            "input_sha256": [sha256_file(path) for path in inputs],
            "rows": len(table),
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
