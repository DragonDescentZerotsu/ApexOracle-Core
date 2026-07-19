"""Read-only contracts for precomputed Evo-2 genome embeddings.

This module intentionally contains no genome embedding extraction code.  Evo-2
is an external producer; ApexOracle consumes versioned ``.pt`` tensors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASETS = (
    "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv",
    "small_molecule/processed/small_molecule_Evo_binary_data_SELFIES.csv",
    "synergy_DBAASP_inhouse_Evo.csv",
)


def parse_embedding_id(file_name: str) -> str:
    stem = file_name.split(".")[0]
    if "ATCC" not in stem:
        return stem
    components = stem.split("ATCC")[-1].split("_")[1:]
    if len(components) == 2:
        return "-".join(components)
    return components[0]


def genome_embedding_paths(embeddings_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in embeddings_dir.iterdir():
        if not path.is_file():
            continue
        genome_id = parse_embedding_id(path.name)
        if genome_id in paths:
            raise ValueError(
                f"Duplicate parsed genome ID {genome_id}: {paths[genome_id]} and {path}"
            )
        paths[genome_id] = path
    return paths


def build_origin_to_standard_map(mapping_path: Path) -> dict[str, str]:
    with mapping_path.open("r", encoding="utf-8") as handle:
        strain_count_data = json.load(handle)

    pairs: list[tuple[str, str]] = []
    for name in strain_count_data:
        if "*" in name:
            original_name, standard_name = name.split("*", 1)
            standard_name = (
                standard_name.split("ATCC")[-1].strip()
                if "ATCC" in standard_name
                else standard_name.strip()
            )
            pairs.append((original_name.strip(), standard_name))
            continue
        if "ATCC" not in name:
            continue
        atcc_id = name.split("ATCC")[-1].strip()
        if "BAA" in name:
            atcc_id = atcc_id.replace(" ", "-")
        if "MY" in name:
            atcc_id = atcc_id.replace(" ", "")
        if "MAY" in name:
            atcc_id = atcc_id.replace("MAY", "MYA")
        for separator in ("D", "T", "s", " "):
            if separator in name:
                atcc_id = atcc_id.split(separator)[0]
        pairs.append((name.strip(), atcc_id))
    return dict(pairs)


def fallback_atcc_id(strain_name: str) -> str | None:
    if "ATCC" not in strain_name:
        return None
    atcc_id = strain_name.split("ATCC")[-1].strip()
    if "BAA" in strain_name:
        atcc_id = atcc_id.replace(" ", "-")
    if "MY" in strain_name:
        atcc_id = atcc_id.replace(" ", "")
    if "MAY" in strain_name:
        atcc_id = atcc_id.replace("MAY", "MYA")
    for separator in ("D", "T", "s", " "):
        if separator in atcc_id:
            atcc_id = atcc_id.split(separator)[0]
    return atcc_id


def resolve_genome_id(
    strain_name: str,
    embedded_ids: set[str],
    origin_to_standard: dict[str, str],
) -> str | None:
    if strain_name in embedded_ids:
        return strain_name
    standard_id = origin_to_standard.get(strain_name)
    if standard_id in embedded_ids:
        return standard_id
    atcc_id = fallback_atcc_id(strain_name)
    return atcc_id if atcc_id in embedded_ids else None


def matched_genome_ids(
    data_dir: Path,
    embedding_paths: dict[str, Path],
    dataset_paths: Iterable[str] = DEFAULT_DATASETS,
) -> set[str]:
    origin_to_standard = build_origin_to_standard_map(
        data_dir
        / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json"
    )
    embedded_ids = set(embedding_paths)
    matched: set[str] = set()
    for relative_path in dataset_paths:
        dataframe = pd.read_csv(data_dir / relative_path, usecols=["strain_name"])
        for strain_name in dataframe["strain_name"].dropna().unique():
            genome_id = resolve_genome_id(
                str(strain_name), embedded_ids, origin_to_standard
            )
            if genome_id is not None:
                matched.add(genome_id)
    return matched


def load_embedding(path: Path):
    """Load a tensor on CPU without allowing arbitrary pickle objects."""

    import torch

    embedding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(embedding).__name__}")
    return embedding


def compute_abs_mean_rows(
    genome_ids: set[str], embedding_paths: dict[str, Path]
) -> pd.DataFrame:
    rows = []
    for genome_id in sorted(genome_ids):
        path = embedding_paths[genome_id]
        embedding = load_embedding(path)
        values = embedding.float().flatten()
        rows.append(
            {
                "genome_id": genome_id,
                "file": path.name,
                "shape": "x".join(str(dim) for dim in embedding.shape),
                "dtype": str(embedding.dtype),
                "numel": int(embedding.numel()),
                "abs_mean": float(values.abs().mean()),
                "std": float(values.std(unbiased=False)),
                "abs_max": float(values.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(
    embedding_paths: dict[str, Path], matched_ids: set[str]
) -> pd.DataFrame:
    rows = []
    for genome_id, path in sorted(embedding_paths.items()):
        rows.append(
            {
                "genome_id": genome_id,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "used_by_paper_datasets": genome_id in matched_ids,
            }
        )
    return pd.DataFrame(rows)


def manifest_identity(table: pd.DataFrame) -> str:
    required = ["genome_id", "file", "bytes", "sha256", "used_by_paper_datasets"]
    if list(table.columns) != required:
        raise ValueError(f"Unexpected manifest columns: {list(table.columns)}")
    payload = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
