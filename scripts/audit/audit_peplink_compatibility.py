#!/usr/bin/env python3
"""Compare independent PepLink output with the frozen paper correction table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/data_pipeline/peplink_v0.1.1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def fragment_parent_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid historical SMILES: {smiles}")
    cleaned = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(cleaned)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)


def load_csv_mapping(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["DBAASP_id"]): row["SMILES"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--peplink-source",
        type=Path,
        help="Optional source checkout used for an audit before installing PepLink.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.peplink_source:
        sys.path.insert(0, str(args.peplink_source.resolve()))
    from PepLink import aa_seqs_to_smiles, from_dbaasp_record

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = config["paper_inputs"]
    source_paths = {
        name: resolve(spec["path"])
        for name, spec in inputs.items()
        if isinstance(spec, dict) and "path" in spec
    }
    verified_hashes = {}
    for name, path in source_paths.items():
        observed_hash = sha256(path)
        verified_hashes[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "expected_sha256": inputs[name].get("sha256"),
            "observed_sha256": observed_hash,
            "matches": inputs[name].get("sha256") in {None, observed_hash},
        }

    records = json.loads(source_paths["dbaasp_json"].read_text(encoding="utf-8"))
    by_id = {int(record["id"]): record for record in records}
    offered = load_csv_mapping(source_paths["correction_input"])
    historical = load_csv_mapping(source_paths["correction_output"])
    special = set(
        config["legacy_correction_contract"]["preserve_offered_smiles_for_ids"]
    )

    comparisons = []
    mode_counts = {"peplink": 0, "preserved_special": 0, "preserved_non_monomer": 0}
    for dbaasp_id, historical_smiles in historical.items():
        record = by_id[dbaasp_id]
        if dbaasp_id in special:
            observed = offered[dbaasp_id]
            mode = "preserved_special"
        elif (record.get("complexity") or {}).get("name") != "Monomer":
            observed = offered[dbaasp_id]
            mode = "preserved_non_monomer"
        else:
            peptide = from_dbaasp_record(record)
            observed = aa_seqs_to_smiles(**peptide.to_api_kwargs())
            mode = "peplink"
        mode_counts[mode] += 1
        exact = observed == historical_smiles
        parent_equivalent = fragment_parent_smiles(historical_smiles) == observed
        comparisons.append(
            {
                "dbaasp_id": dbaasp_id,
                "mode": mode,
                "exact": exact,
                "fragment_parent_equivalent": parent_equivalent,
                "historical_smiles": historical_smiles if not exact else None,
                "peplink_v0_1_1_smiles": observed if not exact else None,
            }
        )

    exception_ids = {row["dbaasp_id"] for row in comparisons if not row["exact"]}
    row_impact = {}
    for name in ("final_dbaasp_mic", "final_merged_mic", "final_token_cache"):
        frame = pd.read_csv(source_paths[name], usecols=["DBAASP_id"])
        ids = frame["DBAASP_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        row_impact[name] = {
            str(dbaasp_id): int((ids == str(dbaasp_id)).sum())
            for dbaasp_id in sorted(exception_ids)
        }

    report = {
        "schema_version": 1,
        "peplink": config["dependency"],
        "integration": config["integration"],
        "source_hashes": verified_hashes,
        "comparison": {
            "total": len(comparisons),
            "mode_counts": mode_counts,
            "exact": sum(row["exact"] for row in comparisons),
            "fragment_parent_equivalent": sum(
                row["fragment_parent_equivalent"] for row in comparisons
            ),
            "mismatches": [row for row in comparisons if not row["exact"]],
        },
        "paper_row_impact": row_impact,
        "conclusion": {
            "paper_reproduction": "use frozen paper CSVs identified by SHA-256",
            "new_data": "use PepLink 0.1.1 output",
            "submodule_required": False,
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
