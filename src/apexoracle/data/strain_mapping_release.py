"""Build the compact, public strain-name map used by paper MIC workflows.

The historical training table contains author-facing strain labels, while the
runtime consumes either an ATCC-style genome/text key or an exact text-only
key.  This module exports that relationship without copying MIC labels,
molecule structures, embedding tensors, or private assay records.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from apexoracle.data.strain_mapping import (
    get_atcc_id_to_species_name_map,
    get_original_strain_names_with_genome_embedding,
)
from apexoracle.features.precomputed import get_embedded_genome_ids


SCHEMA_COLUMNS = (
    "source_strain_name",
    "canonical_strain_id",
    "species_name",
    "condition_route",
    "genome_embedding_file",
    "text_embedding_file",
    "mapping_reason",
    "mic_record_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atcc_file_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        stem = path.stem
        if "ATCC" not in stem:
            continue
        components = stem.split("ATCC", 1)[1].split("_")[1:]
        strain_id = "-".join(components) if len(components) == 2 else components[0]
        if strain_id in mapping:
            raise ValueError(f"Duplicate ATCC condition key: {strain_id}")
        mapping[strain_id] = path.name
    return mapping


def _text_only_file_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        strain_name = path.stem.replace("～", " ").replace("^", "/")
        if strain_name in mapping:
            raise ValueError(f"Duplicate text-only condition key: {strain_name}")
        mapping[strain_name] = path.name
    return mapping


def build_paper_strain_mapping(data_root: Path) -> dict[str, object]:
    """Return a deterministic mapping for strains present in the MIC table."""

    data_root = data_root.resolve()
    paths = {
        "mic_records": data_root / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
        "legacy_mapping": data_root
        / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        "genome_embeddings": data_root / "Genome_embs",
        "genome_fastas": data_root / "Genome" / "ATCC",
        "atcc_text_embeddings": data_root / "Text_Description" / "ATCC" / "embeddings",
        "text_only_embeddings": data_root
        / "Text_Description"
        / "wo_ATCC"
        / "embeddings",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing strain-mapping inputs: {missing}")

    embedded_ids, genome_id_to_species_first_name = get_embedded_genome_ids(
        paths["genome_embeddings"]
    )
    handcrafted_names, direct_atcc_names, origin_to_standard = (
        get_original_strain_names_with_genome_embedding(
            paths["legacy_mapping"], embedded_ids
        )
    )
    handcrafted_names = set(handcrafted_names)
    direct_atcc_names = set(direct_atcc_names)
    atcc_species, _ = get_atcc_id_to_species_name_map(paths["genome_fastas"])
    genome_files = _atcc_file_map(paths["genome_embeddings"])
    atcc_text_files = _atcc_file_map(paths["atcc_text_embeddings"])
    text_only_files = _text_only_file_map(paths["text_only_embeddings"])

    frame = pd.read_csv(paths["mic_records"], usecols=["strain_name"])
    # Preserve the paper-era deletion-mutant exclusion exactly.  The historical
    # code used a substring check, so this also excludes a small number of
    # taxon names containing ``del``; changing it would alter the frozen cohort.
    counts = Counter(
        str(value) for value in frame["strain_name"].dropna() if "del" not in str(value)
    )
    rows: list[dict[str, object]] = []
    for source_name in sorted(counts):
        canonical_id = origin_to_standard.get(source_name)
        mapping_reason = None
        if source_name in handcrafted_names:
            mapping_reason = "handcrafted_alias"
        elif source_name in direct_atcc_names:
            expected_prefix = genome_id_to_species_first_name.get(canonical_id)
            if expected_prefix is None or expected_prefix in source_name:
                mapping_reason = "direct_atcc"

        if mapping_reason is not None and canonical_id is not None:
            genome_file = genome_files.get(canonical_id)
            text_file = atcc_text_files.get(canonical_id)
            if genome_file is None or text_file is None:
                raise ValueError(
                    f"Incomplete genome/text condition for {source_name!r} -> "
                    f"{canonical_id!r}"
                )
            rows.append(
                {
                    "source_strain_name": source_name,
                    "canonical_strain_id": canonical_id,
                    "species_name": atcc_species.get(canonical_id, ""),
                    "condition_route": "genome_text",
                    "genome_embedding_file": genome_file,
                    "text_embedding_file": text_file,
                    "mapping_reason": mapping_reason,
                    "mic_record_count": counts[source_name],
                }
            )

        if (
            len(source_name.split(" ")) > 1
            and source_name.split(" ")[1] not in {"sp.", "spp.", "group"}
            and source_name in text_only_files
        ):
            rows.append(
                {
                    "source_strain_name": source_name,
                    "canonical_strain_id": source_name,
                    "species_name": " ".join(source_name.split(" ")[:2]),
                    "condition_route": "text_only",
                    "genome_embedding_file": None,
                    "text_embedding_file": text_only_files[source_name],
                    "mapping_reason": "exact_text_only_name",
                    "mic_record_count": counts[source_name],
                }
            )

    route_counts = Counter(str(row["condition_route"]) for row in rows)
    mapped_record_count = sum(int(row["mic_record_count"]) for row in rows)
    mapped_source_names = {str(row["source_strain_name"]) for row in rows}
    return {
        "schema_version": 1,
        "scope": "paper MIC strains with a model-consumable genome+text or text-only condition",
        "excludes": [
            "MIC labels",
            "molecule structures",
            "embedding tensors",
            "private assay records",
        ],
        "columns": list(SCHEMA_COLUMNS),
        "summary": {
            "source_strain_count": len(counts),
            "mapped_source_strain_count": len(mapped_source_names),
            "mapping_row_count": len(rows),
            "mapped_mic_route_record_count": mapped_record_count,
            "route_counts": dict(sorted(route_counts.items())),
        },
        "sources": {
            name: {
                "relative_path": str(path.relative_to(data_root)),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "records": rows,
    }


def write_paper_strain_mapping(document: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
