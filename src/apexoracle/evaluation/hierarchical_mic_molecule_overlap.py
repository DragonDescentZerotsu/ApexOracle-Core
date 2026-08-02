"""Exact-molecule overlap helpers for hierarchical MIC reviewer analyses.

The paper-era hierarchical split is defined on pathogen groups.  This module
adds a second, evaluation-only molecule axis without changing that split or the
training cohort.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Sequence
from functools import lru_cache

import pandas as pd


IDENTITY_DBAASP_ID = "dbaasp_id"
IDENTITY_MODEL_INPUT = "model_input_token_sha256"
IDENTITY_DEFINITIONS = (IDENTITY_DBAASP_ID, IDENTITY_MODEL_INPUT)


@lru_cache(maxsize=None)
def _parsed_model_input_tokens(serialized: str) -> tuple[int, ...]:
    tokens = ast.literal_eval(serialized)
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError("Expected SMILES column to contain a Python list of integer token IDs")
    return tuple(tokens)


def normalized_model_input_tokens(value: object) -> str:
    """Return a whitespace-independent serialization of stored token IDs."""

    return json.dumps(_parsed_model_input_tokens(str(value)), separators=(",", ":"))


def model_input_identity(value: object) -> str:
    """Hash the exact molecular token sequence consumed by the frozen model."""

    return hashlib.sha256(normalized_model_input_tokens(value).encode("utf-8")).hexdigest()


def apply_legacy_token_length_filter(
    frame: pd.DataFrame, *, max_length: int = 512
) -> pd.DataFrame:
    """Apply the same stored-token length eligibility rule as the legacy Dataset."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    required = {"DBAASP_id", "strain_name", "SMILES", "MIC"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MIC frame is missing required columns: {sorted(missing)}")
    keep = frame["SMILES"].map(
        lambda value: len(_parsed_model_input_tokens(str(value))) <= max_length
    )
    return frame.loc[keep].copy().reset_index(drop=True)


def concatenate_routes(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine mutually exclusive genome-text and text-only MIC routes."""

    if not frames:
        raise ValueError("At least one frame is required")
    return pd.concat(list(frames), ignore_index=True)


def add_molecule_identity(
    frame: pd.DataFrame, identity_definition: str
) -> pd.DataFrame:
    """Add the requested exact-molecule identity as ``molecule_identity``."""

    output = frame.copy()
    if identity_definition == IDENTITY_DBAASP_ID:
        output["molecule_identity"] = output["DBAASP_id"].map(str)
    elif identity_definition == IDENTITY_MODEL_INPUT:
        output["molecule_identity"] = output["SMILES"].map(model_input_identity)
    else:
        raise ValueError(
            f"Unknown identity definition {identity_definition!r}; "
            f"expected one of {IDENTITY_DEFINITIONS}"
        )
    return output


def partition_test_by_train_molecules(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    identity_definition: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full, train-seen, and train-unseen test cohorts."""

    train_identified = add_molecule_identity(train, identity_definition)
    test_identified = add_molecule_identity(test, identity_definition)
    train_identities = set(train_identified["molecule_identity"])
    seen_mask = test_identified["molecule_identity"].isin(train_identities)
    full = test_identified.reset_index(drop=True)
    seen = test_identified.loc[seen_mask].copy().reset_index(drop=True)
    unseen = test_identified.loc[~seen_mask].copy().reset_index(drop=True)
    if set(unseen["molecule_identity"]) & train_identities:
        raise AssertionError("Molecule-disjoint test cohort still overlaps training")
    return full, seen, unseen


def summarize_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    identity_definition: str,
    low_mic_threshold_um: float = 16.0,
) -> dict[str, int | float | str]:
    """Summarize exact-molecule overlap at measurement and molecule grain."""

    if low_mic_threshold_um <= 0:
        raise ValueError("low_mic_threshold_um must be positive")
    train_identified = add_molecule_identity(train, identity_definition)
    full, seen, unseen = partition_test_by_train_molecules(
        train, test, identity_definition=identity_definition
    )
    test_count = len(full)
    seen_count = len(seen)
    unseen_count = len(unseen)
    unseen_low_mic = int((unseen["MIC"].astype(float) <= low_mic_threshold_um).sum())
    return {
        "identity_definition": identity_definition,
        "train_measurement_instances": len(train_identified),
        "test_measurement_instances": test_count,
        "test_train_seen_measurement_instances": seen_count,
        "test_train_seen_fraction": seen_count / test_count,
        "test_train_unseen_measurement_instances": unseen_count,
        "train_unique_molecules": train_identified["molecule_identity"].nunique(),
        "test_unique_molecules": full["molecule_identity"].nunique(),
        "test_train_seen_unique_molecules": seen["molecule_identity"].nunique(),
        "test_train_unseen_unique_molecules": unseen["molecule_identity"].nunique(),
        "test_pathogens": full["strain_name"].nunique(),
        "test_train_unseen_pathogens": unseen["strain_name"].nunique(),
        "test_train_unseen_low_mic_threshold_um": float(low_mic_threshold_um),
        "test_train_unseen_low_mic_measurement_instances": unseen_low_mic,
        "test_train_unseen_low_mic_fraction": (
            unseen_low_mic / unseen_count if unseen_count else float("nan")
        ),
    }


def aggregate_group_summaries(
    summaries: Iterable[dict[str, int | float | str]],
) -> dict[str, int | float | str]:
    """Aggregate group rows using measurement-weighted numerators/denominators."""

    rows = list(summaries)
    if not rows:
        raise ValueError("No group summaries supplied")
    identities = {str(row["identity_definition"]) for row in rows}
    if len(identities) != 1:
        raise ValueError(f"Cannot aggregate mixed identity definitions: {identities}")
    test_count = sum(int(row["test_measurement_instances"]) for row in rows)
    seen_count = sum(
        int(row["test_train_seen_measurement_instances"]) for row in rows
    )
    unseen_count = sum(
        int(row["test_train_unseen_measurement_instances"]) for row in rows
    )
    unseen_low_mic = sum(
        int(row["test_train_unseen_low_mic_measurement_instances"]) for row in rows
    )
    return {
        "identity_definition": identities.pop(),
        "train_measurement_instances": sum(
            int(row["train_measurement_instances"]) for row in rows
        ),
        "test_measurement_instances": test_count,
        "test_train_seen_measurement_instances": seen_count,
        "test_train_seen_fraction": seen_count / test_count,
        "test_train_unseen_measurement_instances": unseen_count,
        # These are group-level molecule instances because a molecule can have a
        # different seen/unseen status under a different training fold.
        "train_unique_molecules": sum(int(row["train_unique_molecules"]) for row in rows),
        "test_unique_molecules": sum(int(row["test_unique_molecules"]) for row in rows),
        "test_train_seen_unique_molecules": sum(
            int(row["test_train_seen_unique_molecules"]) for row in rows
        ),
        "test_train_unseen_unique_molecules": sum(
            int(row["test_train_unseen_unique_molecules"]) for row in rows
        ),
        "test_pathogens": sum(int(row["test_pathogens"]) for row in rows),
        "test_train_unseen_pathogens": sum(
            int(row["test_train_unseen_pathogens"]) for row in rows
        ),
        "test_train_unseen_low_mic_threshold_um": float(
            rows[0]["test_train_unseen_low_mic_threshold_um"]
        ),
        "test_train_unseen_low_mic_measurement_instances": unseen_low_mic,
        "test_train_unseen_low_mic_fraction": (
            unseen_low_mic / unseen_count if unseen_count else float("nan")
        ),
    }
