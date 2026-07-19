"""Compatibility facade for the former strain-wise preparation module.

New code must use :mod:`apexoracle.data.hierarchical_mic_preparation`.  This
facade keeps the audited manifest builder and downstream imports working while
all three hierarchical holdouts share one implementation.
"""

from __future__ import annotations

from pathlib import Path

from apexoracle.data.hierarchical_mic_preparation import (
    HoldoutSplit,
    PreparedHierarchicalMicData,
    holdout_record_counts,
    prepare_hierarchical_mic_data,
    sha256_file,
)
from apexoracle.data.strainwise_protocol import build_legacy_three_fold_groups

PreparedStrainwiseData = PreparedHierarchicalMicData


def prepare_legacy_strainwise_data(repo_root: Path) -> PreparedHierarchicalMicData:
    prepared = prepare_hierarchical_mic_data(repo_root)
    # Preserve the old facade's mutation timing and attributes for its manifest
    # consumer. The canonical shared representation does not store a split.
    train_groups, test_groups = build_legacy_three_fold_groups(
        prepared.species_to_strains, prepared.taxonomy_aliases
    )
    prepared.train_groups = train_groups  # type: ignore[attr-defined]
    prepared.test_groups = test_groups  # type: ignore[attr-defined]
    return prepared


def fold_record_counts(prepared: PreparedHierarchicalMicData, fold: int) -> dict:
    split = HoldoutSplit(
        protocol="strain",
        group_names=("fold 1", "fold 2", "fold 3"),
        test_groups=tuple(tuple(group) for group in prepared.test_groups),  # type: ignore[attr-defined]
    )
    return holdout_record_counts(prepared, split, fold)


__all__ = [
    "PreparedStrainwiseData",
    "fold_record_counts",
    "prepare_legacy_strainwise_data",
    "sha256_file",
]
