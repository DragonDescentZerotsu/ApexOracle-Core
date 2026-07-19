"""Pure table operations for the paper AMP training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .amp_mic import MIC_COLUMNS


@dataclass(frozen=True)
class TokenFilterResult:
    table: pd.DataFrame
    excluded_invalid_smiles: int
    excluded_too_long: int
    excluded_unknown_token: int
    unique_smiles_tokenized: int


def validate_mic_table(table: pd.DataFrame, *, name: str) -> None:
    if tuple(table.columns) != MIC_COLUMNS:
        raise ValueError(
            f"{name} must have columns {MIC_COLUMNS}, got {tuple(table.columns)}"
        )
    if table[list(MIC_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains missing values")


def format_inhouse_mic_table(
    source: pd.DataFrame,
    smiles_by_sequence: Mapping[str, str],
) -> pd.DataFrame:
    """Convert the APEX wide MIC matrix using the frozen row-ID convention."""

    if len(source.columns) < 3 or source.columns[0] != "Peptide":
        raise ValueError("In-house source must begin with a Peptide column")
    strain_columns = list(source.columns[1:-1])
    rows: list[tuple[str, str, str, float]] = []
    for row_index, row in source.iterrows():
        sequence = str(row.iloc[0])
        try:
            smiles = smiles_by_sequence[sequence]
        except KeyError as exc:
            raise KeyError(f"Missing SMILES for in-house sequence {sequence}") from exc
        for strain in strain_columns:
            value = float(row[strain])
            if value < -500:
                value = -1.0
            elif np.isposinf(value):
                value = 512.0
            if value > -1:
                rows.append((f"#{row_index}", strain, smiles, value))
    return pd.DataFrame(rows, columns=MIC_COLUMNS)


def merge_mic_tables(dbaasp: pd.DataFrame, inhouse: pd.DataFrame) -> pd.DataFrame:
    """Append the frozen in-house table without modifying either input."""

    validate_mic_table(dbaasp, name="DBAASP MIC table")
    validate_mic_table(inhouse, name="in-house MIC table")
    return pd.concat([dbaasp, inhouse], ignore_index=True)


def tokenize_and_filter_smiles(
    table: pd.DataFrame,
    *,
    selfies_encoder: Callable[[str], str],
    tokenizer: Any,
    max_length: int = 1024,
) -> TokenFilterResult:
    """Apply the paper SELFIES/token filter, caching repeated SMILES safely."""

    validate_mic_table(table, name="AMP MIC table")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    unknown_id = tokenizer.unk_token_id
    cache: dict[str, tuple[Sequence[int] | None, str | None]] = {}
    valid_indices: list[int] = []
    token_lists: list[Sequence[int]] = []
    reason_counts = {"invalid_smiles": 0, "too_long": 0, "unknown_token": 0}

    for index, smiles in zip(table.index, table["SMILES"].astype(str)):
        cached = cache.get(smiles)
        if cached is None:
            try:
                selfies = selfies_encoder(smiles).replace("][", "] [")
            except Exception:
                cached = (None, "invalid_smiles")
            else:
                ids = tokenizer(selfies, add_special_tokens=True)["input_ids"]
                if len(ids) > max_length:
                    cached = (None, "too_long")
                elif unknown_id in ids:
                    cached = (None, "unknown_token")
                else:
                    cached = (tuple(ids), None)
            cache[smiles] = cached

        ids, reason = cached
        if reason is not None:
            reason_counts[reason] += 1
            continue
        assert ids is not None
        valid_indices.append(index)
        token_lists.append(ids)

    filtered = table.loc[valid_indices].copy()
    # Lists, rather than tuples, preserve the legacy CSV representation.
    filtered["SMILES"] = [list(ids) for ids in token_lists]
    return TokenFilterResult(
        table=filtered,
        excluded_invalid_smiles=reason_counts["invalid_smiles"],
        excluded_too_long=reason_counts["too_long"],
        excluded_unknown_token=reason_counts["unknown_token"],
        unique_smiles_tokenized=len(cache),
    )
