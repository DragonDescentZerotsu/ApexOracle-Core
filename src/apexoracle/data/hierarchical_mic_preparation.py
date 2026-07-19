"""Shared data preparation and holdout adapters for hierarchical MIC protocols."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from apexoracle.data.strain_mapping import (
    exclude_wrong_species_atcc_map,
    get_atcc_id_to_species_name_map,
    get_original_strain_id_to_species_name_map,
    get_original_strain_names_with_genome_embedding,
    merge_strain_maps,
)
from apexoracle.data.strainwise_protocol import build_legacy_three_fold_groups
from apexoracle.features.precomputed import get_embedded_genome_ids


@dataclass
class PreparedHierarchicalMicData:
    columns: list[str]
    genome_text_records: np.ndarray
    genome_or_text_records: np.ndarray
    small_molecule_records: np.ndarray
    genome_text_groups: dict[str, np.ndarray]
    genome_or_text_groups: dict[str, np.ndarray]
    small_molecule_groups: dict[str, np.ndarray]
    atcc_id_to_species: dict[str, str]
    original_strain_to_species: dict[str, str]
    species_to_strains: dict[str, list[str]]
    taxonomy_aliases: dict[str, str]


@dataclass(frozen=True)
class HoldoutSplit:
    """The only protocol-specific input consumed by the shared runner."""

    protocol: str
    group_names: tuple[str, ...]
    test_groups: tuple[tuple[str, ...], ...]


def _text_only_embedding_names(folder: Path) -> set[str]:
    return {
        path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        for path in folder.iterdir()
        if path.is_file()
    }


def _group_records(records: np.ndarray) -> dict[str, np.ndarray]:
    names = set(records[:, 1])
    return {name: records[np.where(records[:, 1] == name)[0]] for name in names}


def prepare_hierarchical_mic_data(
    repo_root: Path,
    *,
    mic_data_path: Path | None = None,
    small_molecule_data_path: Path | None = None,
) -> PreparedHierarchicalMicData:
    """Reproduce common legacy filtering without choosing a holdout strategy."""

    data_root = repo_root / "DataPrepare" / "Data"
    genome_folder = data_root / "Genome_embs"
    text_only_folder = data_root / "Text_Description" / "wo_ATCC" / "embeddings"

    embedded_ids, genome_id_to_species = get_embedded_genome_ids(genome_folder)
    handcrafted_names, original_names, origin_to_standard = (
        get_original_strain_names_with_genome_embedding(
            data_root
            / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
            embedded_ids,
        )
    )

    mic_frame = pd.read_csv(
        mic_data_path
        if mic_data_path is not None
        else data_root / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv"
    )
    columns = mic_frame.columns.tolist()
    all_mic_records = mic_frame.values
    small_molecule_records = pd.read_csv(
        small_molecule_data_path
        if small_molecule_data_path is not None
        else data_root
        / "small_molecule"
        / "processed"
        / "small_molecule_Evo_binary_data_SELFIES.csv"
    ).values

    all_mic_records = np.array(
        [line for line in all_mic_records if "del" not in line[1]], dtype=object
    )
    handcrafted_records = np.array(
        [line for line in all_mic_records if line[1] in handcrafted_names], dtype=object
    )
    original_records = np.array(
        [line for line in all_mic_records if line[1] in original_names], dtype=object
    )
    original_records = exclude_wrong_species_atcc_map(
        original_records, genome_id_to_species
    )
    genome_text_records = np.concatenate((handcrafted_records, original_records))

    # Match the in-place standard-name replacement in the legacy DataFrame path.
    standardized_records = genome_text_records.copy()
    for line in standardized_records:
        line[1] = origin_to_standard[line[1]]

    text_only_names = _text_only_embedding_names(text_only_folder)
    text_only_records = np.array(
        [
            line
            for line in all_mic_records
            if len(line[1].split(" ")) > 1
            and line[1].split(" ")[1] not in ["sp.", "spp.", "group"]
            and line[1] in text_only_names
        ],
        dtype=object,
    )
    genome_or_text_records = np.concatenate((standardized_records, text_only_records))

    genome_text_groups = _group_records(standardized_records)
    genome_or_text_groups = _group_records(genome_or_text_records)

    atcc_id_to_species, species_to_atcc = get_atcc_id_to_species_name_map(
        data_root / "Genome" / "ATCC"
    )
    original_to_species, species_to_original = (
        get_original_strain_id_to_species_name_map(text_only_folder)
    )
    merged_species_to_strains = merge_strain_maps(species_to_atcc, species_to_original)
    with open(
        data_root / "Genome" / "old_to_new_NCBI_taxonomy.json",
        "r",
        encoding="utf-8",
    ) as handle:
        old_to_new = json.load(handle)
    new_to_old = {value: key for key, value in old_to_new.items()}
    two_way_taxonomy_map = new_to_old | old_to_new
    return PreparedHierarchicalMicData(
        columns=columns,
        genome_text_records=standardized_records,
        genome_or_text_records=genome_or_text_records,
        small_molecule_records=small_molecule_records,
        genome_text_groups=genome_text_groups,
        genome_or_text_groups=genome_or_text_groups,
        small_molecule_groups=_group_records(small_molecule_records),
        atcc_id_to_species=atcc_id_to_species,
        original_strain_to_species=original_to_species,
        species_to_strains=merged_species_to_strains,
        taxonomy_aliases=two_way_taxonomy_map,
    )


def _cluster_species(
    tree_path: Path, taxonomy_path: Path, *, num_clusters: int
) -> list[np.ndarray]:
    """Run the exact tree-distance clustering used by both legacy holdouts."""

    tree = Phylo.read(tree_path, "newick")
    species = np.array([clade.name for clade in tree.get_terminals()])
    with open(taxonomy_path, "r", encoding="utf-8") as handle:
        old_to_new = json.load(handle)

    size = len(species)
    distance_matrix = np.zeros((size, size))
    for i in tqdm(range(size), desc=" computing distance matrix"):
        for j in range(i, size):
            distance_matrix[i, j] = tree.distance(species[i], species[j])
    distance_matrix = distance_matrix + distance_matrix.T
    labels = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance_matrix)

    for old_name, new_name in old_to_new.items():
        species[species == new_name] = old_name
    return [species[np.where(labels == label)[0]] for label in set(labels)]


def _cluster_holdout_split(
    prepared: PreparedHierarchicalMicData,
    *,
    protocol: str,
    num_clusters: int,
    tree_path: Path,
    taxonomy_path: Path,
    group_names: tuple[str, ...],
) -> HoldoutSplit:
    species_groups = _cluster_species(
        tree_path,
        taxonomy_path,
        num_clusters=num_clusters,
    )
    strain_groups: list[list[str]] = []
    for species_group in tqdm(
        species_groups, desc=" Grouping strains by clustered species"
    ):
        strains: list[str] = []
        for species in species_group:
            if species in prepared.taxonomy_aliases:
                first = prepared.species_to_strains.get(species)
                second = prepared.species_to_strains.get(
                    prepared.taxonomy_aliases[species]
                )
                if first is not None:
                    strains.extend(first)
                if second is not None:
                    strains.extend(second)
            else:
                strains.extend(prepared.species_to_strains[species])
        strain_groups.append(strains)

    combined = sorted(zip(species_groups, strain_groups), key=lambda item: len(item[0]))
    _, sorted_strain_groups = zip(*combined)
    if len(group_names) != len(sorted_strain_groups):
        raise ValueError(
            f"Expected {len(group_names)} {protocol} groups, got {len(sorted_strain_groups)}"
        )
    return HoldoutSplit(
        protocol=protocol,
        group_names=group_names,
        test_groups=tuple(tuple(group) for group in sorted_strain_groups),
    )


def build_holdout_split(
    prepared: PreparedHierarchicalMicData,
    repo_root: Path,
    protocol: str,
    *,
    adapter: str,
    group_names: tuple[str, ...],
    tree_path: Path | None,
    num_clusters: int | None,
) -> HoldoutSplit:
    """Select only the legacy split adapter; model and training stay shared."""

    if protocol == "strain":
        if adapter != "legacy_within_species_three_fold":
            raise ValueError(f"Unsupported strain holdout adapter: {adapter}")
        _, test_groups = build_legacy_three_fold_groups(
            prepared.species_to_strains, prepared.taxonomy_aliases
        )
        return HoldoutSplit(
            protocol=protocol,
            group_names=group_names,
            test_groups=tuple(tuple(group) for group in test_groups),
        )
    if protocol in {"species", "phylum"}:
        if adapter != "taxonomy_tree_agglomerative_clusters":
            raise ValueError(f"Unsupported {protocol} holdout adapter: {adapter}")
        if tree_path is None or num_clusters is None:
            raise ValueError(f"{protocol} holdout requires tree_path and num_clusters")
        return _cluster_holdout_split(
            prepared,
            protocol=protocol,
            num_clusters=num_clusters,
            tree_path=tree_path,
            taxonomy_path=(
                repo_root
                / "DataPrepare"
                / "Data"
                / "Genome"
                / "old_to_new_NCBI_taxonomy.json"
            ),
            group_names=group_names,
        )
    raise ValueError(f"Unknown hierarchical holdout protocol: {protocol}")


def _filtered_length(records: pd.DataFrame, max_length: int = 512) -> int:
    return int(
        records["SMILES"]
        .apply(lambda value: len(ast.literal_eval(value)) <= max_length)
        .sum()
    )


def holdout_record_counts(
    prepared: PreparedHierarchicalMicData, split: HoldoutSplit, group: int
) -> dict:
    all_genome_names = set(prepared.genome_text_records[:, 1])
    all_names = set(prepared.genome_or_text_records[:, 1])
    test_names = set(split.test_groups[group])
    genome_test_names = test_names & all_genome_names
    genome_train_names = all_genome_names - genome_test_names
    text_test_names = (test_names & all_names) - genome_test_names
    text_train_names = all_names - text_test_names - genome_test_names

    def concatenate(groups: dict[str, np.ndarray], names: set[str]) -> pd.DataFrame:
        return pd.DataFrame(
            np.concatenate([groups[name] for name in names]),
            columns=prepared.columns,
        )

    frames = {
        "genome_text_train": concatenate(
            prepared.genome_text_groups, genome_train_names
        ),
        "genome_text_test": concatenate(prepared.genome_text_groups, genome_test_names),
        "text_only_train": concatenate(
            prepared.genome_or_text_groups, text_train_names
        ),
        "text_only_test": concatenate(prepared.genome_or_text_groups, text_test_names),
    }
    return {
        name: {
            "before_length_filter": len(frame),
            "after_length_filter": _filtered_length(frame),
        }
        for name, frame in frames.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
