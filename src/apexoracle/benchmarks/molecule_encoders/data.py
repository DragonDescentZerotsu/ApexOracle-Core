"""Validated loading and splitting for the shared Fig. 2b benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .protocol import DEFAULT_TARGET_COLUMNS


@dataclass(frozen=True)
class SharedBenchmarkData:
    """In-memory view of the immutable shared molecule benchmark."""

    molecule_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    apex_sequences: tuple[str, ...]
    labels: np.ndarray
    label_mask: np.ndarray
    folds: np.ndarray
    target_columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.molecule_ids)

    def outer_fold_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return full-data indices for one frozen outer train/test split."""

        available = set(int(value) for value in np.unique(self.folds))
        if fold not in available:
            raise ValueError(f"fold {fold} is not present; choices are {sorted(available)}")
        test_indices = np.flatnonzero(self.folds == fold)
        train_indices = np.flatnonzero(self.folds != fold)
        return train_indices, test_indices

def transform_mic_labels(
    raw_labels: np.ndarray,
    *,
    missing_sentinel: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper's ``-log10(MIC / 10)`` transform to observed labels."""

    values = np.asarray(raw_labels, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"raw_labels must be a 2D array; received shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("raw_labels contains NaN or infinite values")
    mask = values != missing_sentinel
    if np.any(values[mask] <= 0):
        raise ValueError("observed MIC labels must be positive")
    transformed = np.zeros_like(values, dtype=np.float32)
    transformed[mask] = -np.log10(values[mask] / 10.0)
    return transformed, mask


def _validate_unique_ids(values: Sequence[str], source_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{source_name} contains duplicate dbaasp_id values")


def load_shared_benchmark(data_dir: Path) -> SharedBenchmarkData:
    """Load the shared table and reject any ID or fold drift."""

    data_dir = Path(data_dir)
    shared_path = data_dir / "shared_molecules.csv"
    folds_path = data_dir / "folds.csv"
    shared = pd.read_csv(shared_path, dtype={"dbaasp_id": "string"})
    folds = pd.read_csv(folds_path, dtype={"dbaasp_id": "string"})

    required_shared = ("dbaasp_id", "smiles", "apex_sequence", *DEFAULT_TARGET_COLUMNS)
    missing_shared = [column for column in required_shared if column not in shared.columns]
    if missing_shared:
        raise ValueError(f"shared_molecules.csv is missing columns: {missing_shared}")
    if list(folds.columns) != ["dbaasp_id", "fold"]:
        raise ValueError("folds.csv must contain exactly dbaasp_id and fold columns")

    shared_ids = [str(value).strip() for value in shared["dbaasp_id"]]
    fold_ids = [str(value).strip() for value in folds["dbaasp_id"]]
    _validate_unique_ids(shared_ids, "shared_molecules.csv")
    _validate_unique_ids(fold_ids, "folds.csv")
    if set(shared_ids) != set(fold_ids):
        missing_in_folds = sorted(set(shared_ids) - set(fold_ids))[:10]
        missing_in_shared = sorted(set(fold_ids) - set(shared_ids))[:10]
        raise ValueError(
            "shared/fold ID sets differ: "
            f"missing_in_folds={missing_in_folds}, missing_in_shared={missing_in_shared}"
        )

    shared = shared.assign(dbaasp_id=shared_ids).set_index("dbaasp_id", drop=False)
    folds = folds.assign(dbaasp_id=fold_ids).set_index("dbaasp_id")
    shared = shared.sort_index()
    aligned_folds = pd.to_numeric(folds.loc[shared.index, "fold"], errors="raise").to_numpy()
    if not np.equal(aligned_folds, aligned_folds.astype(np.int64)).all():
        raise ValueError("fold values must be integers")
    aligned_folds = aligned_folds.astype(np.int64)
    fold_values = sorted(int(value) for value in np.unique(aligned_folds))
    if fold_values != list(range(len(fold_values))):
        raise ValueError(f"fold values must be contiguous from zero; received {fold_values}")

    raw_labels = shared.loc[:, DEFAULT_TARGET_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy()
    labels, label_mask = transform_mic_labels(raw_labels)

    return SharedBenchmarkData(
        molecule_ids=tuple(shared.index),
        smiles=tuple(str(value) for value in shared["smiles"]),
        apex_sequences=tuple(str(value) for value in shared["apex_sequence"]),
        labels=labels,
        label_mask=label_mask,
        folds=aligned_folds,
        target_columns=tuple(DEFAULT_TARGET_COLUMNS),
    )
