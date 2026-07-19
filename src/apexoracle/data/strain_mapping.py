"""Legacy-compatible strain and taxonomy mapping utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


def get_original_strain_names_with_genome_embedding(
    evo_mic_count_file_path: Path,
    embedded_genome_ids: Iterable[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Map original DBAASP strain names to stored genome identifiers."""

    with open(evo_mic_count_file_path, "r", encoding="utf-8") as handle:
        strain_count_data = json.load(handle)

    embedded_genome_ids = set(embedded_genome_ids)
    handcrafted: list[tuple[str, str]] = []
    dbaasp_original: list[tuple[str, str]] = []
    for name in strain_count_data:
        if "*" in name:
            original_name, standard_name = name.split("*")
            if "ATCC" in standard_name:
                standard_name = standard_name.split("ATCC")[-1].strip()
            else:
                standard_name = standard_name.strip()
            handcrafted.append((original_name.strip(), standard_name))
        elif "ATCC" in name:
            original_name = name
            atcc_id = name.split("ATCC")[-1].strip()
            if "BAA" in name:
                atcc_id = atcc_id.replace(" ", "-")
            if "MY" in name:
                atcc_id = atcc_id.replace(" ", "")
            if "MAY" in name:
                atcc_id = atcc_id.replace("MAY", "MYA")
            if "D" in name:
                atcc_id = atcc_id.split("D")[0]
            if "T" in name:
                atcc_id = atcc_id.split("T")[0]
            if "s" in name:
                atcc_id = atcc_id.split("s")[0]
            if " " in name:
                atcc_id = atcc_id.split(" ")[0]
            dbaasp_original.append((original_name.strip(), atcc_id))

    combined = np.array(handcrafted + dbaasp_original)
    handcrafted_names = [
        original_name
        for original_name, standard_name in handcrafted
        if standard_name in embedded_genome_ids
    ]
    original_names = [
        original_name
        for original_name, standard_name in dbaasp_original
        if standard_name in embedded_genome_ids
    ]
    return handcrafted_names, original_names, dict(combined)


def exclude_wrong_species_atcc_map(
    data_with_genome_embedding: np.ndarray,
    genome_id_to_species_first_name: Mapping[str, str],
) -> np.ndarray:
    """Apply the paper-era ATCC/species consistency filter without alteration."""

    original_length = len(data_with_genome_embedding)
    marked_atcc_ids: set[str] = set()
    cleaned_data = []
    for line in data_with_genome_embedding:
        name = line[1]
        if "ATCC" not in name:
            cleaned_data.append(line)
            continue

        atcc_id = name.split("ATCC")[-1].strip()
        if "BAA" in name:
            atcc_id = atcc_id.replace(" ", "-")
        if "MY" in name:
            atcc_id = atcc_id.replace(" ", "")
        if "MAY" in name:
            atcc_id = atcc_id.replace("MAY", "MYA")
        if "D" in name:
            atcc_id = atcc_id.split("D")[0]
        if "T" in name:
            atcc_id = atcc_id.split("T")[0]
        if "s" in name:
            atcc_id = atcc_id.split("s")[0]
        if " " in name:
            atcc_id = atcc_id.split(" ")[0]

        if genome_id_to_species_first_name.get(atcc_id) is None:
            cleaned_data.append(line)
            marked_atcc_ids.add(atcc_id)
        elif genome_id_to_species_first_name[atcc_id] in name:
            cleaned_data.append(line)

    cleaned_array = np.array(cleaned_data)
    wrong_atcc_numbers = set(data_with_genome_embedding[:, 1]) - set(
        cleaned_array[:, 1]
    )
    print(f"\n wrong strain names: {wrong_atcc_numbers}")
    print(f"\n double marked_ATCC_IDs: {marked_atcc_ids}")
    print(
        f'\n original data length (no "*", no manual modification) {original_length}'
        f"\n cleaned data length {len(cleaned_array)}\n"
    )
    return cleaned_array


def get_atcc_id_to_species_name_map(
    atcc_fasta_folder_path: Path,
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    file_names = [path.name for path in atcc_fasta_folder_path.iterdir() if path.is_file()]
    atcc_ids: list[str] = []
    species_names: list[str] = []
    for file_name in file_names:
        atcc_id = file_name.split(".")[0].split("ATCC")[-1].strip()
        atcc_id = atcc_id.replace("_", " ").strip().replace(" ", "-")
        atcc_ids.append(atcc_id)

        species_file_name = file_name.split("ATCC")[0]
        if "subsp" in species_file_name.split("_"):
            species_file_name = species_file_name.split("subsp")[0]
        if "pathovar" in species_file_name.split("_"):
            species_file_name = species_file_name.split("pathovar")[0]
        if "var" in species_file_name.split("_"):
            species_file_name = species_file_name.split("var")[0]
        if "sp" in species_file_name.split("_"):
            species_file_name = species_file_name.split("_sp")[0]
        species_names.append(species_file_name.replace("_", " ").strip())

    id_array = np.array(atcc_ids)
    species_array = np.array(species_names)
    species_to_ids = {
        species_name: id_array[species_array == species_name]
        for species_name in set(species_array)
    }
    return dict(zip(atcc_ids, species_names)), species_to_ids


def get_original_strain_id_to_species_name_map(
    original_text_emb_folder_path: Path,
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    file_names = [path.name for path in original_text_emb_folder_path.iterdir() if path.is_file()]
    strain_names: list[str] = []
    species_names: list[str] = []
    for file_name in file_names:
        strain_name = file_name.split(".pt")[0].replace("～", " ").replace("^", "/")
        strain_names.append(strain_name)
        species_names.append(" ".join(strain_name.split(" ")[:2]))

    strain_array = np.array(strain_names)
    species_array = np.array(species_names)
    species_to_strains = {
        species_name: strain_array[species_array == species_name]
        for species_name in set(species_array)
    }
    return dict(zip(strain_names, species_names)), species_to_strains


def merge_strain_maps(
    first: Mapping[str, Iterable[str]],
    second: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: list(value) for key, value in first.items()}
    for key, value in second.items():
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = list(value)
    return merged


# Legacy spellings retained for compatibility with paper-era drivers.
get_original_strain_name_with_genome_embedding = get_original_strain_names_with_genome_embedding
exclude_wrong_species_ATCC_map = exclude_wrong_species_atcc_map
get_ATCC_ID_to_species_name_map = get_atcc_id_to_species_name_map
get_original_strain_ID_to_species_name_map = get_original_strain_id_to_species_name_map
merge_dict = merge_strain_maps
