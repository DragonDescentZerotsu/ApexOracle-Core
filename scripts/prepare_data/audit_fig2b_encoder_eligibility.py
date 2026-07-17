#!/usr/bin/env python3
"""Build the native-processability intersection for the revised Fig. 2b."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.eligibility import audit_encoder_eligibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mic-csv",
        type=Path,
        default=Path("DataPrepare/Data/DBAASP_id_SMILES_bact_MICs.csv"),
    )
    parser.add_argument(
        "--selfies-csv",
        type=Path,
        default=Path("DataPrepare/Data/DBAASP_id_SELFIES_bact_MICs.csv"),
    )
    parser.add_argument(
        "--dbaasp-records-json",
        type=Path,
        default=Path("DataPrepare/Data/all_peptides_data.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_encoder_eligibility.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_encoder_eligibility_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = audit_encoder_eligibility(
        mic_csv=args.mic_csv,
        selfies_csv=args.selfies_csv,
        dbaasp_records_json=args.dbaasp_records_json,
        repo_root=REPO_ROOT,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
