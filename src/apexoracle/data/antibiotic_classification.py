"""Behavior-frozen data adapters for the Fig. 1b three-strain classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from apexoracle.data.hierarchical_mic import (
    StrainEmbeddingDataset,
    TextOnlyStrainEmbeddingDataset,
    collate_genome_text_classification,
    collate_genome_text_regression,
    collate_text_classification,
    collate_text_regression,
)
from apexoracle.data.hierarchical_mic_preparation import PreparedHierarchicalMicData


TARGET_STRAINS = (
    "#004",
    "17978",
    "Staphylococcus aureus RN4220",
)
GENOME_TEXT_TARGETS = frozenset({"#004", "17978"})


@dataclass(frozen=True)
class AntibioticClassificationFrames:
    """The five active frames constructed by all three paper-era drivers."""

    mic_genome_text_train: pd.DataFrame
    mic_text_route_train: pd.DataFrame
    auxiliary_genome_text_train: pd.DataFrame | None
    auxiliary_text_only_train: pd.DataFrame | None
    target: pd.DataFrame
    target_has_genome: bool


def _frame(records: Sequence, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=columns)


def _concat_named_groups(
    groups: Mapping[str, np.ndarray], names: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame | None:
    blocks = [groups[name] for name in names if name in groups]
    if not blocks:
        return None
    return _frame(np.concatenate(blocks), columns)


def prepare_antibiotic_classification_frames(
    prepared: PreparedHierarchicalMicData,
    target_group: int,
) -> AntibioticClassificationFrames:
    """Reproduce the active DataFrame construction in the three MDLM scripts.

    The second MIC frame deliberately contains the standardized genome-backed
    rows again, followed by text-only rows. The legacy training loop therefore
    visits genome-backed MIC records once through the genome route and a second
    time through the learned-missing-genome/text route. This duplication is a
    historical behavior, not a data-cleaning mistake in this adapter.
    """

    if target_group < 0 or target_group >= len(TARGET_STRAINS):
        raise ValueError(f"target_group must be in [0, {len(TARGET_STRAINS) - 1}]")
    target = TARGET_STRAINS[target_group]
    # Preserve the old set-difference iteration. For the text-only target this
    # leaves two genome-backed auxiliary blocks whose order depends on the
    # process hash seed, exactly as in the archived driver.
    genome_train_names = list(set(TARGET_STRAINS[:2]) - {target})
    text_train_names = list({TARGET_STRAINS[2]} - {target})
    target_records = prepared.small_molecule_groups.get(target)
    if target_records is None or len(target_records) == 0:
        raise ValueError(f"No small-molecule records found for target {target!r}")

    return AntibioticClassificationFrames(
        mic_genome_text_train=_frame(
            prepared.genome_text_records.copy(), prepared.columns
        ),
        mic_text_route_train=_frame(
            prepared.genome_or_text_records.copy(), prepared.columns
        ),
        auxiliary_genome_text_train=_concat_named_groups(
            prepared.small_molecule_groups, genome_train_names, prepared.columns
        ),
        auxiliary_text_only_train=_concat_named_groups(
            prepared.small_molecule_groups, text_train_names, prepared.columns
        ),
        target=_frame(target_records.copy(), prepared.columns),
        target_has_genome=target in GENOME_TEXT_TARGETS,
    )


def legacy_target_folds(
    num_records: int, *, num_folds: int = 5
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the exact shuffled KFold indices used after the 512-token filter."""

    if num_records < num_folds:
        raise ValueError(
            f"Cannot split {num_records} target records into {num_folds} folds"
        )
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    return tuple(splitter.split(np.arange(num_records)))


class AntibioticGenomeTextDataset(StrainEmbeddingDataset):
    """Shared legacy dataset with molecule IDs retained for prediction export."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["molecule_id"] = self.dataframe.iloc[idx]["DBAASP_id"]
        return item


class AntibioticTextOnlyDataset(TextOnlyStrainEmbeddingDataset):
    """Text-only counterpart retaining molecule IDs for prediction export."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["molecule_id"] = self.dataframe.iloc[idx]["DBAASP_id"]
        return item


def _with_molecule_ids(batch: list[dict], collator) -> dict:
    output = collator(batch)
    output["molecule_ids"] = [item["molecule_id"] for item in batch]
    return output


def collate_antibiotic_genome_text_regression(batch: list[dict]) -> dict:
    return _with_molecule_ids(batch, collate_genome_text_regression)


def collate_antibiotic_text_regression(batch: list[dict]) -> dict:
    return _with_molecule_ids(batch, collate_text_regression)


def collate_antibiotic_genome_text_classification(batch: list[dict]) -> dict:
    return _with_molecule_ids(batch, collate_genome_text_classification)


def collate_antibiotic_text_classification(batch: list[dict]) -> dict:
    return _with_molecule_ids(batch, collate_text_classification)
