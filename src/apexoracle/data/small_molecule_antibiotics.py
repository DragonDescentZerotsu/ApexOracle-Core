"""Pure builders for the paper three-strain small-molecule dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .amp_mic import MIC_COLUMNS


ECOLI_STRAIN = "#004"
ABAUMANNII_STRAIN = "17978"
SAUREUS_STRAIN = "Staphylococcus aureus RN4220"
PAPER_STRAIN_ORDER = (ECOLI_STRAIN, ABAUMANNII_STRAIN, SAUREUS_STRAIN)
PAPER_COUNTS = {
    ECOLI_STRAIN: (2335, 120),
    ABAUMANNII_STRAIN: (7684, 480),
    SAUREUS_STRAIN: (39312, 512),
}


@dataclass(frozen=True)
class SmallMoleculeSummary:
    rows: int
    positives: int
    counts_by_strain: Mapping[str, tuple[int, int]]


def _paper_table(
    source: pd.DataFrame,
    *,
    id_prefix: str,
    strain: str,
    smiles_column: str,
    labels: pd.Series,
) -> pd.DataFrame:
    if smiles_column not in source:
        raise ValueError(f"Missing SMILES column: {smiles_column}")
    if len(labels) != len(source):
        raise ValueError("Labels must have one value per source row")
    return pd.DataFrame(
        {
            "DBAASP_id": [f"{id_prefix}_{index}" for index in range(len(source))],
            "strain_name": strain,
            "SMILES": source[smiles_column].to_numpy(copy=True),
            "MIC": labels.astype(int).to_numpy(copy=True),
        },
        columns=MIC_COLUMNS,
    )


def format_ecoli_bw25113(source: pd.DataFrame) -> pd.DataFrame:
    """Apply the paper's ``Activity == 'Active'`` label rule."""

    if "Activity" not in source:
        raise ValueError("E. coli source is missing Activity")
    return _paper_table(
        source,
        id_prefix="ce",
        strain=ECOLI_STRAIN,
        smiles_column="SMILES",
        labels=source["Activity"].eq("Active"),
    )


def format_abaumannii_atcc17978(source: pd.DataFrame) -> pd.DataFrame:
    """Label growth below the sample mean minus one sample standard deviation."""

    if "Mean" not in source:
        raise ValueError("A. baumannii source is missing Mean")
    relative_growth = pd.to_numeric(source["Mean"], errors="raise")
    threshold = relative_growth.mean() - relative_growth.std(ddof=1)
    return _paper_table(
        source,
        id_prefix="ch",
        strain=ABAUMANNII_STRAIN,
        smiles_column="SMILES",
        labels=relative_growth.lt(threshold),
    )


def format_saureus_rn4220(source: pd.DataFrame) -> pd.DataFrame:
    """Copy the source ``ACTIVITY`` labels used by the paper notebook."""

    if "ACTIVITY" not in source:
        raise ValueError("S. aureus source is missing ACTIVITY")
    labels = pd.to_numeric(source["ACTIVITY"], errors="raise")
    return _paper_table(
        source,
        id_prefix="na",
        strain=SAUREUS_STRAIN,
        smiles_column="SMILES",
        labels=labels,
    )


def summarize_small_molecule_table(table: pd.DataFrame) -> SmallMoleculeSummary:
    """Validate the common schema and return binary-label counts."""

    if tuple(table.columns) != MIC_COLUMNS:
        raise ValueError(
            f"Small-molecule table must have columns {MIC_COLUMNS}, "
            f"got {tuple(table.columns)}"
        )
    if table[list(MIC_COLUMNS)].isna().any().any():
        raise ValueError("Small-molecule table contains missing values")
    labels = pd.to_numeric(table["MIC"], errors="raise")
    if not labels.isin((0, 1)).all():
        raise ValueError("Small-molecule labels must be binary")

    counts: dict[str, tuple[int, int]] = {}
    for strain, frame in table.groupby("strain_name", sort=False):
        strain_labels = pd.to_numeric(frame["MIC"], errors="raise")
        counts[str(strain)] = (len(frame), int(strain_labels.sum()))
    return SmallMoleculeSummary(
        rows=len(table),
        positives=int(labels.sum()),
        counts_by_strain=counts,
    )


def merge_paper_small_molecule_tables(
    ecoli: pd.DataFrame,
    abaumannii: pd.DataFrame,
    saureus: pd.DataFrame,
    *,
    require_paper_counts: bool = True,
) -> pd.DataFrame:
    """Merge in the frozen paper order without modifying any input table."""

    blocks = (ecoli, abaumannii, saureus)
    for expected_strain, block in zip(PAPER_STRAIN_ORDER, blocks):
        summary = summarize_small_molecule_table(block)
        if tuple(summary.counts_by_strain) != (expected_strain,):
            raise ValueError(f"Expected only strain {expected_strain!r}")
    merged = pd.concat(blocks, ignore_index=True)
    summary = summarize_small_molecule_table(merged)
    if tuple(summary.counts_by_strain) != PAPER_STRAIN_ORDER:
        raise ValueError("Small-molecule blocks are not in the frozen paper order")
    if require_paper_counts and dict(summary.counts_by_strain) != PAPER_COUNTS:
        raise ValueError(
            f"Paper row/positive counts differ: {dict(summary.counts_by_strain)}"
        )
    return merged
