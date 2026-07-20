#!/usr/bin/env python3
"""Build the compact Fig. 2b shared-benchmark audit capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


RESULT_FILES = (
    "fold_metrics.csv",
    "comparison_summary.csv",
    "comparison_summary.json",
    "REPORT.md",
)

CANONICAL_SOURCE_FILES = (
    "configs/model_weights.yaml",
    "experiments/fig2b_molecule_encoders/README.md",
    "experiments/fig2b_molecule_encoders/results_shared_5fold.md",
    "experiments/fig2b_molecule_encoders/apex_legacy_migration_audit.json",
    "scripts/prepare_data/audit_fig2b_encoder_eligibility.py",
    "scripts/prepare_data/build_fig2b_shared_dataset.py",
    "scripts/reproduce/run_fig2b_gpu_queue.py",
    "scripts/reproduce/run_fig2b_shared_mdlm_online.py",
    "scripts/reproduce/summarize_fig2b_shared_results.py",
    "scripts/reproduce_fig2b_baselines_online_5fold.py",
    "src/apexoracle/benchmarks/molecule_encoders/apex_adapter.py",
    "src/apexoracle/benchmarks/molecule_encoders/apex_model.py",
    "src/apexoracle/benchmarks/molecule_encoders/assets.py",
    "src/apexoracle/benchmarks/molecule_encoders/eligibility.py",
    "src/apexoracle/benchmarks/molecule_encoders/legacy_training.py",
    "src/apexoracle/benchmarks/molecule_encoders/protocol.py",
    "src/apexoracle/resources/model_weights.py",
    "reproducibility/release_cleanup_2026-07-19.json",
    "reproducibility/pre_generation_cleanup_2026-07-20.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, overwrite: bool) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        status = "exists"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        status = "written"
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copy identity check failed: {source} -> {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "status": status,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    capsule_data = args.capsule_data.resolve()
    source_results = repo_root / "results/fig2b_shared_original_protocol"
    result_destination = capsule_data / "fig2b_shared_original_protocol"

    manifest = {
        "schema_version": 2,
        "description": (
            "Compact audit resources for the formal 10,886-molecule, "
            "seven-model, five-fold Fig. 2b benchmark."
        ),
        "protocol": "fig2b-shared-native-intersection-v2",
        "result_files": [],
        "canonical_source_files": [],
        "excluded_legacy_sources": {
            "root_fix_drivers": "recoverable from legacy-code-snapshot-2026-07-17",
            "chemberta_mlm_mean": "diagnostic pooling variant; not a reported model",
            "external_mdlm_driver_copy": "canonical thin runner is packaged instead",
        },
    }

    for relative in RESULT_FILES:
        manifest["result_files"].append(
            copy_file(
                source_results / relative,
                result_destination / relative,
                args.overwrite,
            )
        )

    source_destination = capsule_data / "source"
    for relative in CANONICAL_SOURCE_FILES:
        source = repo_root / relative
        manifest["canonical_source_files"].append(
            copy_file(source, source_destination / relative, args.overwrite)
        )

    manifest_path = capsule_data / "fig2b_resource_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--capsule-data", type=Path, default=Path("capsule_fig2/data"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
