from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from apexoracle.data.small_molecule_antibiotics import (
    ABAUMANNII_STRAIN,
    ECOLI_STRAIN,
    SAUREUS_STRAIN,
    format_abaumannii_atcc17978,
    format_ecoli_bw25113,
    format_saureus_rn4220,
    merge_paper_small_molecule_tables,
    summarize_small_molecule_table,
)


def test_three_sources_preserve_legacy_labels_ids_and_order() -> None:
    ecoli = format_ecoli_bw25113(
        pd.DataFrame({"SMILES": ["CC", "NN"], "Activity": ["Active", "Inactive"]})
    )
    abaumannii = format_abaumannii_atcc17978(
        pd.DataFrame({"SMILES": ["AA", "BB", "CC"], "Mean": [0.0, 1.0, 2.0]})
    )
    saureus = format_saureus_rn4220(
        pd.DataFrame({"SMILES": ["OO", "SS"], "ACTIVITY": [0, 1]})
    )
    merged = merge_paper_small_molecule_tables(
        ecoli, abaumannii, saureus, require_paper_counts=False
    )
    assert merged["DBAASP_id"].tolist() == [
        "ce_0",
        "ce_1",
        "ch_0",
        "ch_1",
        "ch_2",
        "na_0",
        "na_1",
    ]
    assert merged["strain_name"].drop_duplicates().tolist() == [
        ECOLI_STRAIN,
        ABAUMANNII_STRAIN,
        SAUREUS_STRAIN,
    ]
    # For [0, 1, 2], mean - sample SD is exactly 0 and the rule is strict <.
    assert merged["MIC"].tolist() == [1, 0, 0, 0, 0, 0, 1]


def test_merge_does_not_modify_inputs() -> None:
    source = pd.DataFrame({"SMILES": ["CC"], "Activity": ["Active"]})
    ecoli = format_ecoli_bw25113(source)
    abaumannii = format_abaumannii_atcc17978(
        pd.DataFrame({"SMILES": ["NN", "OO"], "Mean": [0.0, 1.0]})
    )
    saureus = format_saureus_rn4220(
        pd.DataFrame({"SMILES": ["SS"], "ACTIVITY": [0]})
    )
    originals = [frame.copy(deep=True) for frame in (ecoli, abaumannii, saureus)]
    merge_paper_small_molecule_tables(
        ecoli, abaumannii, saureus, require_paper_counts=False
    )
    for frame, original in zip((ecoli, abaumannii, saureus), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_summary_rejects_nonbinary_labels() -> None:
    table = pd.DataFrame(
        [["x", ECOLI_STRAIN, "CC", 2]],
        columns=["DBAASP_id", "strain_name", "SMILES", "MIC"],
    )
    with pytest.raises(ValueError, match="binary"):
        summarize_small_molecule_table(table)


def test_real_paper_sources_reconstruct_frozen_tables() -> None:
    root = Path(__file__).resolve().parents[1] / "DataPrepare/Data/small_molecule"
    raw = root / "raw"
    processed = root / "processed"
    builders = (
        (
            format_ecoli_bw25113,
            "cell~Escherichia_coli_BW25113~#004.csv",
        ),
        (
            format_abaumannii_atcc17978,
            "chem_bio_relative_growth~Acinetobacter_baumannii_ATCC_17978.csv",
        ),
        (
            format_saureus_rn4220,
            "nature_1_positive~Staphylococcus_aureus_RN4220.csv",
        ),
    )
    rebuilt = []
    for builder, filename in builders:
        actual = builder(pd.read_csv(raw / filename))
        pd.testing.assert_frame_equal(
            pd.read_csv(StringIO(actual.to_csv(index=False))),
            pd.read_csv(processed / filename),
        )
        rebuilt.append(actual)
    merged = merge_paper_small_molecule_tables(*rebuilt)
    pd.testing.assert_frame_equal(
        merged,
        pd.read_csv(processed / "small_molecule_Evo_binary_data.csv"),
    )
