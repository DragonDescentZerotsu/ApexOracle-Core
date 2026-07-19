#!/usr/bin/env python3
"""Freeze the data/split contract used by the final strain-wise checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.strainwise_preparation import (  # noqa: E402
    fold_record_counts,
    prepare_legacy_strainwise_data,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "fig2c_strainwise"
        / "legacy_protocol_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None:
        raise SystemExit(
            "Set PYTHONHASHSEED explicitly before building a strain-wise candidate "
            "manifest (the checked-in candidate uses PYTHONHASHSEED=0)."
        )
    repo_root = args.repo_root.resolve()
    prepared = prepare_legacy_strainwise_data(repo_root)
    data_root = repo_root / "DataPrepare" / "Data"
    source_paths = {
        "mic_records": data_root / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
        "small_molecule_auxiliary": data_root
        / "small_molecule"
        / "processed"
        / "small_molecule_Evo_binary_data_SELFIES.csv",
        "strain_mapping": data_root
        / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        "taxonomy_aliases": data_root / "Genome" / "old_to_new_NCBI_taxonomy.json",
    }
    archived_log_counts = [
        {
            "genome_text_train": {"before_length_filter": 7103, "after_length_filter": 6828},
            "genome_text_test": {"before_length_filter": 72801, "after_length_filter": 70544},
            "text_only_train": {"before_length_filter": 18553, "after_length_filter": 17698},
            "text_only_test": {"before_length_filter": 968, "after_length_filter": 876},
        },
        {
            "genome_text_train": {"before_length_filter": 77418, "after_length_filter": 75002},
            "genome_text_test": {"before_length_filter": 2486, "after_length_filter": 2370},
            "text_only_train": {"before_length_filter": 83761, "after_length_filter": 80968},
            "text_only_test": {"before_length_filter": 6075, "after_length_filter": 5780},
        },
        {
            "genome_text_train": {"before_length_filter": 77253, "after_length_filter": 74807},
            "genome_text_test": {"before_length_filter": 2651, "after_length_filter": 2565},
            "text_only_train": {"before_length_filter": 85168, "after_length_filter": 82316},
            "text_only_test": {"before_length_filter": 4503, "after_length_filter": 4237},
        },
    ]
    manifest = {
        "schema_version": 1,
        "generated_on": str(date.today()),
        "protocol": "deterministic_legacy_codepath_candidate",
        "python_hash_seed": python_hash_seed,
        "historical_membership_status": "not_fully_recovered",
        "compatibility_notes": [
            "taxonomy alias lists are intentionally mutated across fold construction",
            "the three archived GPU jobs were independent processes with unrecorded hash seeds",
            "membership below freezes this generated candidate and is not claimed to be the exact 2025 membership",
            "archived per-fold counts remain authoritative for the 2025 checkpoint runs",
            "token length filtering parses the stored SELFIES token-id list and keeps length <= 512",
        ],
        "sources": {
            name: {
                "relative_path": str(path.relative_to(repo_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "record_counts": {
            "genome_text_before_length_filter": len(prepared.genome_text_records),
            "genome_or_text_before_length_filter": len(prepared.genome_or_text_records),
            "small_molecule_auxiliary": len(prepared.small_molecule_records),
        },
        "folds": [],
    }
    for fold in range(3):
        manifest["folds"].append(
            {
                "fold": fold,
                "train_strain_ids": sorted(set(prepared.train_groups[fold])),
                "test_strain_ids": sorted(set(prepared.test_groups[fold])),
                "candidate_record_counts": fold_record_counts(prepared, fold),
                "archived_log_record_counts": archived_log_counts[fold],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
