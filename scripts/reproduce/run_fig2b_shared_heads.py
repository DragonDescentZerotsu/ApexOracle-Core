#!/usr/bin/env python3
"""Train the common Fig. 2b head on one validated frozen-encoder cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.data import load_shared_benchmark
from apexoracle.benchmarks.molecule_encoders.feature_cache import load_feature_cache
from apexoracle.benchmarks.molecule_encoders.training import (
    HeadTrainingConfig,
    train_shared_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_shared_v1"),
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_shared_benchmark(args.data_dir)
    cache = load_feature_cache(
        args.feature_cache,
        benchmark,
        expected_encoder=args.encoder_name,
    )
    config = HeadTrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    metrics = train_shared_heads(
        benchmark,
        cache,
        args.output_dir,
        config=config,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "encoder": metrics["encoder"],
                "outer_test_macro_r2_mean": metrics["outer_test_macro_r2_mean"],
                "outer_test_macro_r2_sample_sd": metrics["outer_test_macro_r2_sample_sd"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
