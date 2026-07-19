#!/usr/bin/env python3
"""Freeze the candidate paper-era synergy strain split without loading tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.synergy import (  # noqa: E402
    build_legacy_synergy_folds,
    filter_synergy_token_lengths,
    prepare_legacy_synergy_data,
    synergy_label,
)


TOKENIZER_MODEL = "ibm-research/materials.selfies-ted"
TOKENIZER_REVISION = "55e83392264cb998f7aa5014847df29868aefeb8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-token-filter", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _table_summary(table):
    labels = table["FICI"].map(synergy_label)
    return {
        "rows_before_token_filter": len(table),
        "strains": int(table["strain_name"].nunique()),
        "positive_fici_lt_0_5": int(labels.sum()),
        "negative_fici_gte_0_5": int(len(labels) - labels.sum()),
    }


def main() -> None:
    args = parse_args()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        raise SystemExit("Set PYTHONHASHSEED explicitly; legacy splits use unordered sets.")
    output = args.output.resolve()
    data_dir = args.repo_root.resolve() / "DataPrepare" / "Data"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if data_dir in output.parents:
        raise ValueError("Output must be outside the original data directory")

    prepared = prepare_legacy_synergy_data(args.repo_root.resolve())
    folds = build_legacy_synergy_folds(prepared)
    length_cache = {}

    if args.skip_token_filter:
        tokenizer = selfies_encoder = None
    else:
        import selfies as sf
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL,
            revision=TOKENIZER_REVISION,
            local_files_only=args.local_files_only,
        )
        selfies_encoder = sf.encoder

    def partition_summary(table):
        summary = _table_summary(table)
        if tokenizer is not None:
            filtered = filter_synergy_token_lengths(
                table,
                tokenizer=tokenizer,
                selfies_encoder=selfies_encoder,
                length_cache=length_cache,
            ).table
            labels = filtered["FICI"].map(synergy_label)
            summary["rows_after_token_filter"] = len(filtered)
            summary["positive_after_token_filter"] = int(labels.sum())
            summary["negative_after_token_filter"] = int(len(labels) - labels.sum())
        return summary

    manifest = {
        "protocol": "paper_legacy_candidate",
        "python_hash_seed": hash_seed,
        "source": {
            "path": "DataPrepare/Data/synergistic_pairs_Evo.csv",
            "sha256": sha256_file(data_dir / "synergistic_pairs_Evo.csv"),
            "rows": 4285,
        },
        "token_filter": {
            "model": TOKENIZER_MODEL,
            "revision": TOKENIZER_REVISION,
            "max_length": 512,
            "applied": not args.skip_token_filter,
        },
        "prepared": {
            "genome_text": _table_summary(prepared.genome_text),
            "text_only": _table_summary(prepared.text_only),
            "combined": _table_summary(prepared.combined),
        },
        "folds": [
            {
                "fold": fold.fold,
                "strain_for_train": list(fold.strain_for_train),
                "strain_for_test": list(fold.strain_for_test),
                "genome_text_train": partition_summary(fold.genome_text_train),
                "genome_text_test": partition_summary(fold.genome_text_test),
                "text_only_train": partition_summary(fold.text_only_train),
                "text_only_test": partition_summary(fold.text_only_test),
            }
            for fold in folds
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "python_hash_seed": hash_seed,
                "prepared": manifest["prepared"],
                "fold_rows": [
                    {
                        key: {
                            "before": fold[key]["rows_before_token_filter"],
                            "after": fold[key].get("rows_after_token_filter"),
                        }
                        for key in (
                            "genome_text_train",
                            "genome_text_test",
                            "text_only_train",
                            "text_only_test",
                        )
                    }
                    for fold in manifest["folds"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
