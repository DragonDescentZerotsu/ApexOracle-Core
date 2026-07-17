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
from apexoracle.benchmarks.molecule_encoders.encoders import (
    HF_ENCODERS,
    extract_apex_features,
    extract_hf_features,
)
from apexoracle.benchmarks.molecule_encoders.feature_cache import save_feature_cache


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
    parser.add_argument("--apex-root", type=Path, default=Path("compare_APEX"))
    parser.add_argument(
        "--apex-checkpoint",
        type=Path,
        default=Path("compare_APEX/APEX_ckpt/APEX_pretrained_encoder_state_dict_best.ckpt"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_shared_benchmark(args.data_dir)
    if args.encoder == "apex":
        cache = extract_apex_features(
            benchmark,
            apex_root=args.apex_root,
            checkpoint_path=args.apex_checkpoint,
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
