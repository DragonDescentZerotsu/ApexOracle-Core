#!/usr/bin/env python3
"""Recalculate hierarchical MIC metrics under alternate >V point encodings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.evaluation.hierarchical_mic_censor_workflow import (  # noqa: E402
    CensorSensitivityInputs,
    run_censor_sensitivity_analysis,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/hierarchical_mic/censor_multiplier_sensitivity/analysis"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data = REPO_ROOT / "DataPrepare/Data"
    parser = argparse.ArgumentParser(
        description=(
            "Audit DBAASP >V prevalence and recalculate frozen hierarchical MIC "
            "prediction metrics under alternate finite label encodings."
        )
    )
    parser.add_argument(
        "--dbaasp-json", type=Path, default=data / "all_peptides_data.json"
    )
    parser.add_argument(
        "--smiles-csv", type=Path, default=data / "DBAASP_id_SMILES_merged.csv"
    )
    parser.add_argument(
        "--frozen-dbaasp-mic",
        type=Path,
        default=data / "DBAASP_id_bact_name_SMILES_MIC_Evo.csv",
    )
    parser.add_argument(
        "--mic-records",
        type=Path,
        default=data / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
    )
    parser.add_argument(
        "--small-molecule-records",
        type=Path,
        default=data
        / "small_molecule/processed/small_molecule_Evo_binary_data_SELFIES.csv",
    )
    parser.add_argument(
        "--strain-predictions",
        type=Path,
        default=REPO_ROOT
        / "experiments/hierarchical_mic/fixed_strain_retrain/analysis/ensemble_predictions.csv",
    )
    parser.add_argument(
        "--phylum-predictions",
        type=Path,
        default=REPO_ROOT
        / "experiments/hierarchical_mic/molecule_disjoint/phylum_analysis/ensemble_predictions.csv",
    )
    parser.add_argument(
        "--strain-manifest",
        type=Path,
        default=REPO_ROOT
        / "experiments/hierarchical_mic/strain/legacy_protocol_manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/hierarchical_mic/legacy_mdlm.yaml",
    )
    parser.add_argument(
        "--right-censor-multiplier",
        type=float,
        action="append",
        dest="multipliers",
        help="Repeat to replace the default ordinary >V grid of 1, 2, and 4.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this workflow's declared derived outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = CensorSensitivityInputs(
        dbaasp_json=args.dbaasp_json,
        smiles_csv=args.smiles_csv,
        frozen_dbaasp_mic=args.frozen_dbaasp_mic,
        mic_records=args.mic_records,
        small_molecule_records=args.small_molecule_records,
        strain_predictions=args.strain_predictions,
        phylum_predictions=args.phylum_predictions,
        strain_manifest=args.strain_manifest,
        config=args.config,
    )
    manifest = run_censor_sensitivity_analysis(
        REPO_ROOT,
        inputs,
        output_dir=args.output_dir,
        multipliers=args.multipliers or (1.0, 2.0, 4.0),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
