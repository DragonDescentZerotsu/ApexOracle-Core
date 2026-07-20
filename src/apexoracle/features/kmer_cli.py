"""Command-line entrypoint for non-overwriting k-mer tensor extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from apexoracle.features.kmer import extract_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "windowed"), required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=(4, 5, 6))
    parser.add_argument("--window-length", type=int, default=11_000)
    parser.add_argument("--step-length", type=int, default=10_000)
    parser.add_argument("--output-dtype", choices=("float32", "bfloat16"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dtype = None if args.output_dtype is None else getattr(torch, args.output_dtype)
    outputs = extract_folder(
        args.genome_dir,
        args.output_dir,
        mode=args.mode,
        k_values=args.k_values,
        window_length=args.window_length,
        step_length=args.step_length,
        output_dtype=dtype,
    )
    print(f"wrote {len(outputs)} tensors to {args.output_dir.resolve()}")
