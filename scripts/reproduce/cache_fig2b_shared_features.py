#!/usr/bin/env python3
"""Build a strict shared-ID feature cache for one Fig. 2b encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.data import load_shared_benchmark
from apexoracle.benchmarks.molecule_encoders.assets import apex_aaindex_path
from apexoracle.benchmarks.molecule_encoders.encoders import (
    HF_ENCODERS,
    extract_apex_features,
    extract_hf_features,
)
from apexoracle.benchmarks.molecule_encoders.feature_cache import save_feature_cache
from apexoracle.resources import resolve_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=sorted((*HF_ENCODERS, "apex")))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_shared_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--apex-aaindex",
        type=Path,
        default=None,
        help="Override the canonical local APEX aaindex1.csv asset.",
    )
    parser.add_argument(
        "--apex-checkpoint",
        type=Path,
        default=None,
        help="Override the fig2b_apex_encoder manifest entry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_shared_benchmark(args.data_dir)
    if args.encoder == "apex":
        aaindex = args.apex_aaindex or apex_aaindex_path(REPO_ROOT)
        checkpoint = args.apex_checkpoint or resolve_weight(
            "fig2b_apex_encoder", repo_root=REPO_ROOT
        )
        cache = extract_apex_features(
            benchmark,
            aaindex_path=aaindex,
            checkpoint_path=checkpoint,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        cache = extract_hf_features(
            benchmark,
            HF_ENCODERS[args.encoder],
            repo_root=REPO_ROOT,
            device=args.device,
            batch_size=args.batch_size,
        )
    save_feature_cache(
        args.output,
        encoder_name=cache.encoder_name,
        molecule_ids=cache.molecule_ids,
        features=cache.features,
        metadata=cache.metadata,
    )
    print(
        f"saved {cache.encoder_name}: {cache.features.shape[0]} molecules, "
        f"feature_dim={cache.features.shape[1]} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
