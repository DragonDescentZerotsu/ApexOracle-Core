#!/usr/bin/env python3
"""Audit exact-peptide overlap in hierarchical MIC pathogen holdouts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.hierarchical_mic_preparation import (  # noqa: E402
    HoldoutSplit,
    build_holdout_split,
    prepare_hierarchical_mic_data,
)
from apexoracle.evaluation.hierarchical_mic_molecule_overlap import (  # noqa: E402
    IDENTITY_DEFINITIONS,
    aggregate_group_summaries,
    apply_legacy_token_length_filter,
    concatenate_routes,
    summarize_overlap,
)
from apexoracle.training.hierarchical_mic_runner import (  # noqa: E402
    HierarchicalMicConfig,
    prepare_holdout_frames,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "hierarchical_mic" / "legacy_mdlm.yaml"
DEFAULT_STRAIN_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "hierarchical_mic"
    / "strain"
    / "legacy_protocol_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "hierarchical_mic" / "molecule_overlap"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_strain_candidate(path: Path) -> HoldoutSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["historical_membership_status"] != "not_fully_recovered":
        raise ValueError("Unexpected strain manifest provenance status")
    folds = sorted(payload["folds"], key=lambda row: int(row["fold"]))
    return HoldoutSplit(
        protocol="strain",
        group_names=tuple(f"fold {int(row['fold']) + 1}" for row in folds),
        test_groups=tuple(tuple(map(str, row["test_strain_ids"])) for row in folds),
    )


def audit_protocol(
    protocol: str,
    *,
    config_path: Path,
    strain_manifest: Path,
    low_mic_threshold_um: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = HierarchicalMicConfig.load(
        config_path, REPO_ROOT, holdout_protocol=protocol
    )
    with redirect_stdout(StringIO()):
        prepared = prepare_hierarchical_mic_data(
            REPO_ROOT,
            mic_data_path=config.paths.mic_records,
            small_molecule_data_path=config.paths.small_molecule_records,
        )
        split = (
            load_frozen_strain_candidate(strain_manifest)
            if protocol == "strain"
            else build_holdout_split(
                prepared,
                REPO_ROOT,
                protocol,
                adapter=config.holdout_adapter,
                group_names=config.holdout_group_names,
                tree_path=config.holdout_tree,
                num_clusters=config.holdout_clusters,
            )
        )

    rows: list[dict[str, Any]] = []
    for group_index, group_name in enumerate(split.group_names):
        frames = prepare_holdout_frames(prepared, split, group_index)
        train = concatenate_routes(
            [
                apply_legacy_token_length_filter(frames.genome_text_train),
                apply_legacy_token_length_filter(frames.text_only_train),
            ]
        )
        test = concatenate_routes(
            [
                apply_legacy_token_length_filter(frames.genome_text_test),
                apply_legacy_token_length_filter(frames.text_only_test),
            ]
        )
        for identity_definition in IDENTITY_DEFINITIONS:
            row = summarize_overlap(
                train,
                test,
                identity_definition=identity_definition,
                low_mic_threshold_um=low_mic_threshold_um,
            )
            row.update(
                {
                    "protocol": protocol,
                    "group_index": group_index,
                    "group_name": group_name,
                    "aggregation": "group",
                    "membership_status": (
                        "deterministic_candidate_not_exact_2025"
                        if protocol == "strain"
                        else "canonical_taxonomy_cluster_adapter"
                    ),
                }
            )
            rows.append(row)

    for identity_definition in IDENTITY_DEFINITIONS:
        selected = [
            row
            for row in rows
            if row["identity_definition"] == identity_definition
        ]
        total = aggregate_group_summaries(selected)
        total.update(
            {
                "protocol": protocol,
                "group_index": -1,
                "group_name": "all_groups",
                "aggregation": "fold_or_group_measurement_instances",
                "membership_status": selected[0]["membership_status"],
            }
        )
        rows.append(total)

    provenance = {
        "protocol": protocol,
        "group_names": list(split.group_names),
        "membership_status": rows[0]["membership_status"],
        "mic_records": {
            "path": str(config.paths.mic_records.relative_to(REPO_ROOT)),
            "sha256": sha256_file(config.paths.mic_records),
        },
        "strain_manifest": (
            {
                "path": str(strain_manifest.relative_to(REPO_ROOT)),
                "sha256": sha256_file(strain_manifest),
            }
            if protocol == "strain"
            else None
        ),
        "max_model_input_tokens": 512,
        "low_mic_threshold_um": low_mic_threshold_um,
    }
    return rows, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit exact-molecule overlap in hierarchical MIC holdouts."
    )
    parser.add_argument(
        "--protocol",
        choices=["strain", "species", "phylum", "all"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--strain-manifest", type=Path, default=DEFAULT_STRAIN_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-mic-threshold-um", type=float, default=16.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocols = (
        ["strain", "species", "phylum"]
        if args.protocol == "all"
        else [args.protocol]
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    provenance = []
    for protocol in protocols:
        rows, protocol_provenance = audit_protocol(
            protocol,
            config_path=args.config.resolve(),
            strain_manifest=args.strain_manifest.resolve(),
            low_mic_threshold_um=args.low_mic_threshold_um,
        )
        all_rows.extend(rows)
        provenance.append(protocol_provenance)

    table = pd.DataFrame(all_rows)
    table.to_csv(output_dir / "overlap_by_group.csv", index=False)
    report = {
        "schema_version": 1,
        "status": "completed",
        "identity_definitions": {
            "dbaasp_id": "Database record identity; reproduces the legacy overlap question.",
            "model_input_token_sha256": (
                "SHA-256 of the exact stored molecular token sequence consumed by "
                "the frozen precomputed-embedding model."
            ),
        },
        "aggregation_note": (
            "Strain totals are fold-level measurement instances, not unique rows "
            "across folds. Group-level unique-molecule counts are additive only as "
            "fold/group molecule instances because train exposure changes by group."
        ),
        "protocols": provenance,
        "rows": all_rows,
    }
    (output_dir / "overlap_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"Wrote {output_dir / 'overlap_by_group.csv'}")
    print(f"Wrote {output_dir / 'overlap_audit.json'}")


if __name__ == "__main__":
    main()
