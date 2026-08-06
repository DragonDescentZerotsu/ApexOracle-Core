#!/usr/bin/env python3
"""Freeze nearest same-species pairs across all eligible bacterial embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.genome_embeddings import (  # noqa: E402
    genome_embedding_paths,
    parse_embedding_id,
    sha256_file,
)
from apexoracle.data.strain_mapping import (  # noqa: E402
    get_atcc_id_to_species_name_map,
)
from apexoracle.evaluation.genome_condition_reviewer import (  # noqa: E402
    MIN_DONOR_ALIGNED_FRACTION,
    MIN_DONOR_ANI,
    NON_BACTERIAL_SPECIES,
    parse_skani_sparse,
    select_nearest_same_species_neighbors,
)

DEFAULT_SKANI = REPO_ROOT / ".external/envs/genome_condition_reviewer/bin/skani"
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/genome_condition_reviewer/fragment_variation/manifests"
)


def fasta_id(path: Path) -> str:
    return parse_embedding_id(path.name.replace(".fasta", ".pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "DataPrepare/Data")
    parser.add_argument("--skani", type=Path, default=DEFAULT_SKANI)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = genome_embedding_paths(data_dir / "Genome_embs")
    fasta_by_id = {
        fasta_id(path): path
        for path in (data_dir / "Genome/ATCC").glob("*.fasta")
        if path.is_file()
    }
    species_by_id, _ = get_atcc_id_to_species_name_map(data_dir / "Genome/ATCC")
    asset_ids = set(embeddings) & set(fasta_by_id) & set(species_by_id)
    bacterial_ids = {
        genome_id
        for genome_id in asset_ids
        if species_by_id[genome_id] not in NON_BACTERIAL_SPECIES
    }
    species_counts = pd.Series(
        [species_by_id[genome_id] for genome_id in bacterial_ids], dtype="object"
    ).value_counts()
    eligible_ids = {
        genome_id
        for genome_id in bacterial_ids
        if int(species_counts[species_by_id[genome_id]]) >= 2
    }

    fasta_list = output_dir / "eligible_bacterial_fastas.txt"
    fasta_list.write_text(
        "".join(
            f"{fasta_by_id[genome_id].resolve()}\n"
            for genome_id in sorted(eligible_ids)
        ),
        encoding="utf-8",
    )
    raw_path = output_dir / "all_bacterial_skani_sparse.tsv"
    command = [
        str(args.skani.resolve()),
        "triangle",
        "-E",
        "-t",
        str(args.threads),
        "-l",
        str(fasta_list.resolve()),
        "-o",
        str(raw_path.resolve()),
    ]
    subprocess.run(command, check=True)
    raw = pd.read_csv(raw_path, sep="\t")
    pairs = parse_skani_sparse(raw)
    directed = select_nearest_same_species_neighbors(
        pairs,
        eligible_ids=eligible_ids,
        species_by_id=species_by_id,
        min_ani=MIN_DONOR_ANI,
        min_aligned_fraction=MIN_DONOR_ALIGNED_FRACTION,
    )
    directed.to_csv(output_dir / "nearest_pairs_directed.csv", index=False)
    unordered = directed.copy()
    unordered["genome_a"] = unordered[["target_id", "donor_id"]].min(axis=1)
    unordered["genome_b"] = unordered[["target_id", "donor_id"]].max(axis=1)
    unordered = (
        unordered.sort_values(["genome_a", "genome_b", "target_id"], kind="mergesort")
        .groupby(["genome_a", "genome_b"], as_index=False)
        .agg(
            species=("species", "first"),
            ani=("ani", "first"),
            minimum_aligned_fraction=("minimum_aligned_fraction", "min"),
            directed_selections=("target_id", "size"),
        )
    )
    unordered.to_csv(output_dir / "nearest_pairs_unordered.csv", index=False)
    directed_path = output_dir / "nearest_pairs_directed.csv"
    unordered_path = output_dir / "nearest_pairs_unordered.csv"

    summary = {
        "schema_version": 1,
        "status": "completed",
        "scope": "all eligible bacterial frozen genome embeddings",
        "eligibility": {
            "embedding_fasta_species_intersection": len(asset_ids),
            "bacterial_assets": len(bacterial_ids),
            "excluded_non_bacterial_assets": len(asset_ids - bacterial_ids),
            "excluded_singleton_species_bacterial_assets": len(
                bacterial_ids - eligible_ids
            ),
            "eligible_bacterial_assets": len(eligible_ids),
            "eligible_species": len(
                {species_by_id[genome_id] for genome_id in eligible_ids}
            ),
        },
        "selection": {
            "minimum_ani": MIN_DONOR_ANI,
            "minimum_bidirectional_aligned_fraction": MIN_DONOR_ALIGNED_FRACTION,
            "directed_targets_selected": len(directed),
            "directed_species": int(directed["species"].nunique()),
            "unordered_nearest_pairs": len(unordered),
            "unordered_species": int(unordered["species"].nunique()),
            "ani_minimum": float(unordered["ani"].min()),
            "ani_median": float(unordered["ani"].median()),
            "ani_maximum": float(unordered["ani"].max()),
        },
        "tools": {
            "skani": subprocess.check_output(
                [str(args.skani.resolve()), "--version"], text=True
            ).strip()
        },
        "command": command,
        "source_sha256": {
            "fasta_list": sha256_file(fasta_list),
            "skani_sparse": sha256_file(raw_path),
        },
        "output_sha256": {
            "nearest_pairs_directed": sha256_file(directed_path),
            "nearest_pairs_unordered": sha256_file(unordered_path),
        },
        "outputs": {
            "directed": "nearest_pairs_directed.csv",
            "unordered": "nearest_pairs_unordered.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
