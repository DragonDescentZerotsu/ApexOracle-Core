"""Legacy-compatible preparation and strain-wise splitting for synergy CV."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


SYNERGY_COLUMNS = (
    "DBAASP_id",
    "antibio_id_or_name",
    "strain_name",
    "AMP_smiles",
    "antibiotic_smiles",
    "FICI",
)


@dataclass(frozen=True)
class SynergyTokenFilterResult:
    table: pd.DataFrame
    original_rows: int
    retained_rows: int
    unique_smiles_tokenized: int


@dataclass(frozen=True)
class PreparedSynergyData:
    genome_text: pd.DataFrame
    text_only: pd.DataFrame
    combined: pd.DataFrame
    standard_strain_groups: Mapping[str, pd.DataFrame]
    all_strain_groups: Mapping[str, pd.DataFrame]
    species_to_strains: Mapping[str, list[str]]
    taxonomy_aliases: Mapping[str, str]


@dataclass(frozen=True)
class SynergyFold:
    fold: int
    strain_for_train: tuple[str, ...]
    strain_for_test: tuple[str, ...]
    genome_text_train: pd.DataFrame
    genome_text_test: pd.DataFrame
    text_only_train: pd.DataFrame
    text_only_test: pd.DataFrame


@dataclass(frozen=True)
class SynergyAllDataRoutes:
    genome_text: pd.DataFrame
    combined_text: pd.DataFrame
    strain_order: tuple[str, ...]


def filter_synergy_token_lengths(
    table: pd.DataFrame,
    *,
    tokenizer: Any,
    selfies_encoder: Callable[[str], str],
    max_length: int = 512,
    length_cache: MutableMapping[str, int] | None = None,
) -> SynergyTokenFilterResult:
    """Apply the legacy two-SMILES SELFIES token-length filter.

    The original Dataset tokenized both molecule columns before discarding rows
    and used the resulting tensors only for this length check. This helper
    preserves that selection while keeping precomputed molecule features as the
    actual model input.
    """

    if tuple(table.columns) != SYNERGY_COLUMNS:
        raise ValueError(f"Unexpected synergy columns: {tuple(table.columns)}")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    cache = length_cache if length_cache is not None else {}

    def token_length(smiles: str) -> int:
        if smiles not in cache:
            selfies = selfies_encoder(smiles).replace("][", "] [")
            tokenized = tokenizer(
                selfies,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            cache[smiles] = len(tokenized["input_ids"].squeeze(0))
        return cache[smiles]

    amp_lengths = [token_length(str(value)) for value in table["AMP_smiles"]]
    antibiotic_lengths = [
        token_length(str(value)) for value in table["antibiotic_smiles"]
    ]
    keep = np.asarray(amp_lengths) <= max_length
    keep &= np.asarray(antibiotic_lengths) <= max_length
    filtered = table.loc[keep].reset_index(drop=True).copy()
    return SynergyTokenFilterResult(
        table=filtered,
        original_rows=len(table),
        retained_rows=len(filtered),
        unique_smiles_tokenized=len(cache),
    )


def synergy_label(fici: float) -> float:
    """Paper label: FICI < 0.5 is positive synergy."""

    return 1.0 if float(fici) < 0.5 else 0.0


def _parse_atcc_id(name: str) -> str:
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
    return atcc_id


def embedded_ids_and_species(folder: Path) -> tuple[list[str], dict[str, str]]:
    embedded_ids: list[str] = []
    species_by_id: dict[str, str] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.name.split(".")[0]
        components = stem.split("ATCC")[-1].split("_")[1:]
        embedded_id = "-".join(components) if len(components) == 2 else components[0]
        embedded_ids.append(embedded_id)
        species_by_id[embedded_id] = stem.split("_")[0]
    return embedded_ids, species_by_id


def legacy_strain_mapping_groups(
    mapping_path: Path, embedded_ids: Sequence[str]
) -> tuple[list[str], list[str], dict[str, str]]:
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_data = json.load(handle)
    handcrafted: list[tuple[str, str]] = []
    original: list[tuple[str, str]] = []
    for name in mapping_data:
        if "*" in name:
            original_name, standard_name = name.split("*", 1)
            standard_name = (
                standard_name.split("ATCC")[-1].strip()
                if "ATCC" in standard_name
                else standard_name.strip()
            )
            handcrafted.append((original_name.strip(), standard_name))
        elif "ATCC" in name:
            original.append((name.strip(), _parse_atcc_id(name)))
    embedded = set(embedded_ids)
    handcrafted_names = [name for name, target in handcrafted if target in embedded]
    original_names = [name for name, target in original if target in embedded]
    return handcrafted_names, original_names, dict(handcrafted + original)


def exclude_wrong_species_mappings(
    table: pd.DataFrame, species_by_embedding_id: Mapping[str, str]
) -> pd.DataFrame:
    keep = []
    for _, row in table.iterrows():
        name = str(row["strain_name"])
        if "ATCC" not in name:
            keep.append(row)
            continue
        atcc_id = _parse_atcc_id(name)
        expected_species = species_by_embedding_id.get(atcc_id)
        if expected_species is None or expected_species in name:
            keep.append(row)
    return pd.DataFrame(keep, columns=table.columns).reset_index(drop=True)


def _text_only_strains(folder: Path) -> list[str]:
    return [
        path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        for path in folder.iterdir()
        if path.is_file()
    ]


def _atcc_species_groups(folder: Path) -> dict[str, list[str]]:
    ids: list[str] = []
    species: list[str] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        file_name = path.name
        atcc_id = file_name.split(".")[0].split("ATCC")[-1].strip()
        ids.append(atcc_id.replace("_", " ").strip().replace(" ", "-"))
        prefix = file_name.split("ATCC")[0]
        components = prefix.split("_")
        for marker in ("subsp", "pathovar", "var"):
            if marker in components:
                prefix = prefix.split(marker)[0]
                components = prefix.split("_")
        if "sp" in components:
            prefix = prefix.split("_sp")[0]
        species.append(prefix.replace("_", " ").strip())
    ids_array = np.asarray(ids)
    species_array = np.asarray(species)
    return {
        name: ids_array[species_array == name].tolist() for name in set(species_array)
    }


def _text_species_groups(folder: Path) -> dict[str, list[str]]:
    strains = _text_only_strains(folder)
    species = [" ".join(name.split(" ")[:2]) for name in strains]
    strain_array = np.asarray(strains)
    species_array = np.asarray(species)
    return {
        name: strain_array[species_array == name].tolist() for name in set(species_array)
    }


def _merge_species_groups(
    first: Mapping[str, Iterable[str]], second: Mapping[str, Iterable[str]]
) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in first.items()}
    for key, value in second.items():
        merged.setdefault(key, []).extend(value)
    return merged


def _group_by_strain(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        strain: table.loc[table["strain_name"] == strain].copy()
        for strain in set(table["strain_name"])
    }


def prepare_legacy_synergy_data(
    repo_root: Path, *, source_path: Path | None = None
) -> PreparedSynergyData:
    data_dir = repo_root / "DataPrepare" / "Data"
    raw = pd.read_csv(
        source_path if source_path is not None else data_dir / "synergistic_pairs_Evo.csv"
    )
    if tuple(raw.columns) != SYNERGY_COLUMNS:
        raise ValueError(f"Unexpected synergy columns: {tuple(raw.columns)}")
    filtered = raw.loc[
        ~raw["strain_name"].astype(str).str.contains("del", regex=False)
    ].copy()

    embedded_ids, species_by_id = embedded_ids_and_species(data_dir / "Genome_embs")
    handcrafted, original, mapping = legacy_strain_mapping_groups(
        data_dir
        / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        embedded_ids,
    )
    handcrafted_table = filtered.loc[filtered["strain_name"].isin(handcrafted)].copy()
    original_table = exclude_wrong_species_mappings(
        filtered.loc[filtered["strain_name"].isin(original)].copy(), species_by_id
    )
    genome_text = pd.concat([handcrafted_table, original_table], ignore_index=True)
    genome_text["strain_name"] = genome_text["strain_name"].map(mapping)

    text_folder = data_dir / "Text_Description" / "wo_ATCC" / "embeddings"
    text_strains = set(_text_only_strains(text_folder))
    eligible_text_strains = {
        name
        for name in text_strains
        if len(name.split(" ")) > 1
        and name.split(" ")[1] not in {"sp.", "spp.", "group"}
    }
    text_only = filtered.loc[filtered["strain_name"].isin(eligible_text_strains)].copy()
    combined = pd.concat([genome_text, text_only], ignore_index=True)

    species_groups = _merge_species_groups(
        _atcc_species_groups(data_dir / "Genome" / "ATCC"),
        _text_species_groups(text_folder),
    )
    with (data_dir / "Genome" / "old_to_new_NCBI_taxonomy.json").open(
        "r", encoding="utf-8"
    ) as handle:
        old_to_new = json.load(handle)
    aliases = {value: key for key, value in old_to_new.items()} | old_to_new
    return PreparedSynergyData(
        genome_text=genome_text,
        text_only=text_only,
        combined=combined,
        standard_strain_groups=_group_by_strain(genome_text),
        all_strain_groups=_group_by_strain(combined),
        species_to_strains=species_groups,
        taxonomy_aliases=aliases,
    )


def _concat_groups(
    groups: Mapping[str, pd.DataFrame], strain_ids: Iterable[str], columns: Sequence[str]
) -> pd.DataFrame:
    selected = [groups[strain] for strain in strain_ids]
    if not selected:
        return pd.DataFrame(columns=columns)
    return pd.concat(selected, ignore_index=True)


def build_legacy_synergy_folds(prepared: PreparedSynergyData) -> list[SynergyFold]:
    """Preserve the three-fold script behavior, including alias-list mutation."""

    species_groups = {
        species: list(strains)
        for species, strains in prepared.species_to_strains.items()
    }
    train_groups: list[list[str]] = [[], [], []]
    test_groups: list[list[str]] = [[], [], []]
    for fold in range(3):
        repeated_species: list[str] = []
        for species, corresponding_strains in species_groups.items():
            if species in repeated_species:
                continue
            merged_strains = corresponding_strains
            if species in prepared.taxonomy_aliases:
                repeated_species.append(prepared.taxonomy_aliases[species])
                aliases = species_groups.get(prepared.taxonomy_aliases[species])
                if aliases is not None:
                    merged_strains.extend(aliases)
            merged_strains.sort()
            if len(merged_strains) >= 6:
                merged_strains[1], merged_strains[2] = merged_strains[2], merged_strains[1]
            if len(merged_strains) == 1:
                train_groups[fold].extend(merged_strains)
            elif len(merged_strains) == 2:
                train_groups[fold].append(merged_strains[fold % 2])
                test_groups[fold].append(merged_strains[(fold + 1) % 2])
            else:
                chunk_length = len(merged_strains) // 3
                test = merged_strains[fold * chunk_length : (fold + 1) * chunk_length]
                train = list(set(merged_strains) - set(test))
                train_groups[fold].extend(train)
                test_groups[fold].extend(test)

    standard_strains = set(prepared.standard_strain_groups)
    all_strains = set(prepared.all_strain_groups)
    folds: list[SynergyFold] = []
    for fold, (strain_train, strain_test) in enumerate(zip(train_groups, test_groups)):
        genome_train_ids = set(strain_train) & standard_strains
        genome_test_ids = standard_strains - genome_train_ids
        text_test_ids = (set(strain_test) & all_strains) - genome_test_ids
        text_train_ids = all_strains - text_test_ids - genome_test_ids
        folds.append(
            SynergyFold(
                fold=fold,
                strain_for_train=tuple(strain_train),
                strain_for_test=tuple(strain_test),
                genome_text_train=_concat_groups(
                    prepared.standard_strain_groups, genome_train_ids, SYNERGY_COLUMNS
                ),
                genome_text_test=_concat_groups(
                    prepared.standard_strain_groups, genome_test_ids, SYNERGY_COLUMNS
                ),
                text_only_train=_concat_groups(
                    prepared.all_strain_groups, text_train_ids, SYNERGY_COLUMNS
                ),
                text_only_test=_concat_groups(
                    prepared.all_strain_groups, text_test_ids, SYNERGY_COLUMNS
                ),
            )
        )
    return folds


def build_legacy_synergy_all_data_routes(
    prepared: PreparedSynergyData,
) -> SynergyAllDataRoutes:
    """Preserve the all-data driver's alias merge and set-ordered row blocks."""

    species_groups = {
        species: list(strains)
        for species, strains in prepared.species_to_strains.items()
    }
    repeated_species: list[str] = []
    strain_for_train: list[str] = []
    for species, corresponding_strains in species_groups.items():
        if species in repeated_species:
            continue
        merged_strains = corresponding_strains
        if species in prepared.taxonomy_aliases:
            repeated_species.append(prepared.taxonomy_aliases[species])
            aliases = species_groups.get(prepared.taxonomy_aliases[species])
            if aliases is not None:
                merged_strains.extend(aliases)
        strain_for_train.extend(merged_strains)

    standard_names = set(prepared.standard_strain_groups)
    all_names = set(prepared.all_strain_groups)
    genome_names = set(strain_for_train) & standard_names
    text_names = set(strain_for_train) & all_names
    return SynergyAllDataRoutes(
        genome_text=_concat_groups(
            prepared.standard_strain_groups,
            genome_names,
            SYNERGY_COLUMNS,
        ),
        combined_text=_concat_groups(
            prepared.all_strain_groups,
            text_names,
            SYNERGY_COLUMNS,
        ),
        strain_order=tuple(strain_for_train),
    )
