#!/usr/bin/env python3
"""Stable transitional entrypoint for the final strain-wise training protocol.

The paper-era training loop remains in its audited source file for this migration
step. That driver now consumes the shared Dataset/collate/fusion/head modules.
This wrapper deliberately forwards only arguments supported by the legacy run.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DRIVER = (
    REPO_ROOT
    / "DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_strains_MDLM_MTR_fix.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-compatible three-fold, seven-member strain-wise benchmark."
    )
    parser.add_argument("--test-group", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--acknowledge-dynamic-legacy-split",
        action="store_true",
        help=(
            "Required because the archived process hash seeds were not recorded; "
            "a fresh training run follows the legacy split code but is not claimed "
            "to recover the exact 2025 fold membership."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_dynamic_legacy_split:
        raise SystemExit(
            "Refusing an ambiguous rerun. Pass --acknowledge-dynamic-legacy-split "
            "after reading experiments/fig2c_strainwise/README.md."
        )
    sys.argv = [
        str(LEGACY_DRIVER),
        "--parallel",
        "--test_group",
        str(args.test_group),
        "--device",
        str(args.device),
        "--epoch",
        str(args.epochs),
        "--weight_decay",
        str(args.weight_decay),
    ]
    runpy.run_path(str(LEGACY_DRIVER), run_name="__main__")


if __name__ == "__main__":
    main()
