"""Stable CLI orchestration for lead-peptide sequence similarity."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from apexoracle.data.peptide_similarity import build_training_caches

from .pipeline import SimilarityOutputs, run_similarity
from .reporting import extract_top_hits, validate_rows


DEFAULT_CONFIG = Path("configs/sequence_similarity/paper_leads.yaml")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("protocol") != "paper_lead_peptide_sequence_similarity":
        raise ValueError(f"Unsupported protocol: {config.get('protocol')}")
    queries = config.get("queries") or []
    required = {"peptide_id", "cyclic", "sequence"}
    if not queries or any(not required.issubset(query) for query in queries):
        raise ValueError(
            "Every configured query needs peptide_id, cyclic, and sequence"
        )
    if len({query["peptide_id"] for query in queries}) != len(queries):
        raise ValueError("Configured query IDs must be unique")
    if {query["cyclic"] for query in queries} - {"Yes", "No"}:
        raise ValueError("Configured cyclic values must be Yes or No")
    return config


def write_query_csv(path: Path, queries: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["peptide_id", "cyclic", "sequence"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(queries)
    path.write_text(buffer.getvalue().removesuffix("\n"), encoding="utf-8")


def output_paths(output_dir: Path) -> tuple[SimilarityOutputs, dict[str, Path]]:
    raw_dir = output_dir / "raw"
    report_dir = output_dir / "reports"
    outputs = SimilarityOutputs(
        linear=raw_dir / "linear_similarity_results.csv",
        cyclic_rotations=raw_dir / "cyclic_rotation_similarity_results.csv",
        cyclic_best_pid=raw_dir / "cyclic_best_by_pid.csv",
        cyclic_best_max_len=raw_dir / "cyclic_best_by_max_len_identity.csv",
        manifest=report_dir / "similarity_run_manifest.json",
    )
    reports = {
        "linear_summary": report_dir / "linear_top_similarity_hits.csv",
        "cyclic_summary": report_dir / "cyclic_top_similarity_hits.csv",
        "alignments": report_dir / "top_similarity_alignments.txt",
        "validation": report_dir / "similarity_validation_report.json",
    }
    return outputs, reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the ApexOracle-3/12/23 sequence-similarity analysis."
    )
    parser.add_argument(
        "action", choices=("prepare", "compute", "summarize", "validate", "all")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--query-csv", type=Path)
    parser.add_argument("--linear-cache", type=Path)
    parser.add_argument("--cyclic-cache", type=Path)
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--linear-chunk-size", type=int)
    parser.add_argument("--cyclic-chunk-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[4]
    config_path = _resolve(repo_root, args.config)
    config = load_config(config_path)
    configured_output = _resolve(repo_root, config["output_dir"])
    output_dir = (
        _resolve(repo_root, args.output_dir) if args.output_dir else configured_output
    )
    cache_dir = output_dir / "cache"
    query_csv = (
        _resolve(repo_root, args.query_csv)
        if args.query_csv
        else cache_dir / "query_peptides.csv"
    )
    linear_cache = (
        _resolve(repo_root, args.linear_cache)
        if args.linear_cache
        else cache_dir / "train_linear_peptides.csv"
    )
    cyclic_cache = (
        _resolve(repo_root, args.cyclic_cache)
        if args.cyclic_cache
        else cache_dir / "train_cyclic_peptides.csv"
    )
    training_manifest = output_dir / "reports/training_peptide_manifest.json"
    outputs, reports = output_paths(output_dir)

    if args.action in {"prepare", "all"}:
        write_query_csv(query_csv, config["queries"])
        manifest = build_training_caches(
            training_csv=_resolve(repo_root, config["sources"]["training_csv"]),
            all_peptides_json=_resolve(
                repo_root, config["sources"]["all_peptides_json"]
            ),
            linear_output=linear_cache,
            cyclic_output=cyclic_cache,
            manifest_output=training_manifest,
            sequence_case=str(
                config["sources"].get("training_sequence_case", "preserve")
            ),
        )
        print(json.dumps(manifest["counts"], indent=2))

    if args.action in {"compute", "all"}:
        runtime = config["runtime"]
        manifest = run_similarity(
            query_csv=query_csv,
            linear_cache=linear_cache,
            cyclic_cache=cyclic_cache,
            outputs=outputs,
            processes=args.processes or int(runtime["processes"]),
            linear_chunk_size=args.linear_chunk_size
            or int(runtime["linear_chunk_size"]),
            cyclic_chunk_size=args.cyclic_chunk_size
            or int(runtime["cyclic_chunk_size"]),
            query_ids=args.query_ids,
        )
        print(json.dumps(manifest["counts"], indent=2))

    if args.action in {"summarize", "all"}:
        linear, cyclic = extract_top_hits(
            linear_results=outputs.linear,
            cyclic_best_pid=outputs.cyclic_best_pid,
            cyclic_best_max_len=outputs.cyclic_best_max_len,
            linear_summary_output=reports["linear_summary"],
            cyclic_summary_output=reports["cyclic_summary"],
            alignment_report_output=reports["alignments"],
        )
        print(f"Wrote {len(linear)} linear and {len(cyclic)} cyclic top-hit rows")

    if args.action in {"validate", "all"}:
        report = {
            "linear_results": validate_rows(outputs.linear),
            "cyclic_rotation_results": validate_rows(outputs.cyclic_rotations),
            "cyclic_best_by_pid": validate_rows(outputs.cyclic_best_pid, "pid"),
            "cyclic_best_by_max_len_identity": validate_rows(
                outputs.cyclic_best_max_len, "max_len_identity"
            ),
        }
        report["overall_ok"] = all(section["ok"] for section in report.values())
        reports["validation"].parent.mkdir(parents=True, exist_ok=True)
        reports["validation"].write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
