"""Paper-compatible strain-wise split construction.

The legacy split function mutates lists in ``merged_species_to_strains`` when it
joins taxonomy aliases. This is scientifically awkward but is part of the code
path that generated the published checkpoints. The compatibility function below
preserves it explicitly; a corrected split must use a differently named protocol.
"""

from __future__ import annotations

from typing import Mapping


def build_legacy_three_fold_groups(
    merged_species_to_strains: dict[str, list[str]],
    two_way_taxonomy_map: Mapping[str, str],
) -> tuple[list[list[str]], list[list[str]]]:
    train_groups = [[], [], []]
    test_groups = [[], [], []]

    for fold_index in range(len(train_groups)):
        repeated_species_names: list[str] = []
        for species_name, corresponding_ids in merged_species_to_strains.items():
            if species_name in repeated_species_names:
                continue

            # Intentionally alias rather than copy: this preserves the mutation
            # in the paper-era implementation.
            merged_corresponding_ids = corresponding_ids
            if species_name in two_way_taxonomy_map:
                repeated_species_names.append(two_way_taxonomy_map[species_name])
                second_ids = merged_species_to_strains.get(
                    two_way_taxonomy_map[species_name]
                )
                if second_ids is not None:
                    merged_corresponding_ids.extend(second_ids)

            merged_corresponding_ids.sort()
            if len(merged_corresponding_ids) >= 6:
                merged_corresponding_ids[1], merged_corresponding_ids[2] = (
                    merged_corresponding_ids[2],
                    merged_corresponding_ids[1],
                )

            if len(merged_corresponding_ids) == 1:
                train_groups[fold_index].extend(merged_corresponding_ids)
            elif len(merged_corresponding_ids) == 2:
                train_groups[fold_index].append(
                    merged_corresponding_ids[fold_index % 2]
                )
                test_groups[fold_index].append(
                    merged_corresponding_ids[(fold_index + 1) % 2]
                )
            else:
                chunk_length = len(merged_corresponding_ids) // len(train_groups)
                test_ids = merged_corresponding_ids[
                    fold_index * chunk_length : (fold_index + 1) * chunk_length
                ]
                train_ids = list(set(merged_corresponding_ids) - set(test_ids))
                train_groups[fold_index].extend(train_ids)
                test_groups[fold_index].extend(test_ids)

    return train_groups, test_groups
