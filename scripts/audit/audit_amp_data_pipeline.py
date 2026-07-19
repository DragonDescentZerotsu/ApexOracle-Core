#!/usr/bin/env python3
"""Compare rebuilt AMP pipeline artifacts with frozen paper data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.amp_mic import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-mic", type=Path, required=True)
    parser.add_argument("--rebuilt-mic", type=Path, required=True)
    parser.add_argument("--frozen-merged", type=Path, required=True)
    parser.add_argument("--rebuilt-merged", type=Path, required=True)
    parser.add_argument("--frozen-tokens", type=Path, required=True)
    parser.add_argument("--rebuilt-tokens", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = pd.read_csv(args.frozen_mic, dtype=str, keep_default_na=False)
    rebuilt = pd.read_csv(args.rebuilt_mic, dtype=str, keep_default_na=False)
    numeric_difference = np.abs(
        frozen["MIC"].astype(float).to_numpy()
        - rebuilt["MIC"].astype(float).to_numpy()
    )
    report = {
        "mic": {
            "rows": len(frozen),
            "exact_non_mic_columns": all(
                frozen[column].equals(rebuilt[column])
                for column in ("DBAASP_id", "strain_name", "SMILES")
            ),
            "mic_text_mismatches": int((frozen["MIC"] != rebuilt["MIC"]).sum()),
            "mic_values_outside_atol_1e-12": int(
                np.count_nonzero(numeric_difference > 1e-12)
            ),
            "mic_max_absolute_difference": float(numeric_difference.max()),
        },
        "merged": {
            "frozen_sha256": sha256_file(args.frozen_merged),
            "rebuilt_sha256": sha256_file(args.rebuilt_merged),
        },
        "tokens": {
            "frozen_sha256": sha256_file(args.frozen_tokens),
            "rebuilt_sha256": sha256_file(args.rebuilt_tokens),
        },
    }
    report["merged"]["byte_exact"] = (
        report["merged"]["frozen_sha256"] == report["merged"]["rebuilt_sha256"]
    )
    report["tokens"]["byte_exact"] = (
        report["tokens"]["frozen_sha256"] == report["tokens"]["rebuilt_sha256"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
