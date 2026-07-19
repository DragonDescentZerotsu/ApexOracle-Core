"""Preparation of the final paper-era strain-wise MIC protocol."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
class PreparedStrainwiseData:
    columns: list[str]
    genome_text_records: np.ndarray
    genome_or_text_records: np.ndarray
    small_molecule_records: np.ndarray
    genome_text_groups: dict[str, np.ndarray]
    genome_or_text_groups: dict[str, np.ndarray]
    train_groups: list[list[str]]
    test_groups: list[list[str]]
    atcc_id_to_species: dict[str, str]
    original_strain_to_species: dict[str, str]


def _text_only_embedding_names(folder: Path) -> set[str]:
    return {
        path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        for path in folder.iterdir()
        if path.is_file()
    }


def _group_records(records: np.ndarray) -> dict[str, np.ndarray]:
    names = set(records[:, 1])
    return {name: records[np.where(records[:, 1] == name)[0]] for name in names}


def prepare_legacy_strainwise_data(repo_root: Path) -> PreparedStrainwiseData:
    """Reproduce legacy filtering and strain-fold membership without loading tensors."""

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

    mic_frame = pd.read_csv(data_root / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv")
    columns = mic_frame.columns.tolist()
    all_mic_records = mic_frame.values
    small_molecule_records = pd.read_csv(
        data_root
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
    genome_or_text_records = np.concatenate(
        (standardized_records, text_only_records)
    )

    genome_text_groups = _group_records(standardized_records)
    genome_or_text_groups = _group_records(genome_or_text_records)

    atcc_id_to_species, species_to_atcc = get_atcc_id_to_species_name_map(
        data_root / "Genome" / "ATCC"
    )
    original_to_species, species_to_original = (
        get_original_strain_id_to_species_name_map(text_only_folder)
    )
    merged_species_to_strains = merge_strain_maps(
        species_to_atcc, species_to_original
    )
    with open(
        data_root / "Genome" / "old_to_new_NCBI_taxonomy.json",
        "r",
        encoding="utf-8",
    ) as handle:
        old_to_new = json.load(handle)
    new_to_old = {value: key for key, value in old_to_new.items()}
    two_way_taxonomy_map = new_to_old | old_to_new
    train_groups, test_groups = build_legacy_three_fold_groups(
        merged_species_to_strains, two_way_taxonomy_map
    )

    return PreparedStrainwiseData(
        columns=columns,
        genome_text_records=standardized_records,
        genome_or_text_records=genome_or_text_records,
        small_molecule_records=small_molecule_records,
        genome_text_groups=genome_text_groups,
        genome_or_text_groups=genome_or_text_groups,
        train_groups=train_groups,
        test_groups=test_groups,
        atcc_id_to_species=atcc_id_to_species,
        original_strain_to_species=original_to_species,
    )


def _filtered_length(records: pd.DataFrame, max_length: int = 512) -> int:
    return int(
        records["SMILES"].apply(lambda value: len(ast.literal_eval(value)) <= max_length).sum()
    )


def fold_record_counts(prepared: PreparedStrainwiseData, fold: int) -> dict:
    all_genome_names = set(prepared.genome_text_records[:, 1])
    all_names = set(prepared.genome_or_text_records[:, 1])
    test_names = set(prepared.test_groups[fold])
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
        "genome_text_train": concatenate(prepared.genome_text_groups, genome_train_names),
        "genome_text_test": concatenate(prepared.genome_text_groups, genome_test_names),
        "text_only_train": concatenate(prepared.genome_or_text_groups, text_train_names),
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
