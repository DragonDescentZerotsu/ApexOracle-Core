from __future__ import annotations

import pandas as pd
import pytest

from apexoracle.data.amp_mic import (
    build_paper_mic_table,
    collect_strain_measurements,
    parse_concentration,
    parse_inhibition_percentage,
    select_measurement,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4.0),
        (">4", 8.0),
        (">>4", 12.0),
        ("4-8", 6.0),
        ("4–8", 6.0),
        ("4±1", 4.0),
        ("4,5", 4.5),
        ("4->8", 6.0),
        ("4 - =>8", 6.0),
    ],
)
def test_parse_concentration_matches_legacy(raw: str, expected: float) -> None:
    assert parse_concentration(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("95%", 95.0), (">=95%", 95.0), ("95-99%", 97.0), ("95±2%", 95.0)],
)
def test_parse_inhibition_percentage_matches_legacy(raw: str, expected: float) -> None:
    assert parse_inhibition_percentage(raw) == expected


def test_selection_prefers_mic_and_preserves_inhibition_unit_bug() -> None:
    assert select_measurement({("MIC", "µM"): " 2 ", ("MIC", "µg/ml"): "1"}).concentration == "2"
    selected = select_measurement(
        {("99% inhibition", "µM"): "8", ("95% inhibition", "µg/ml"): "4"}
    )
    assert selected is not None
    assert selected.concentration == "8"
    assert selected.convert_micrograms_per_ml is True


def test_collection_keeps_first_duplicate_measurement() -> None:
    activities = [
        {
            "targetSpecies": {"name": "strain"},
            "unit": {"name": "µM"},
            "activityMeasureValue": "MIC",
            "concentration": "2",
        },
        {
            "targetSpecies": {"name": "strain"},
            "unit": {"name": "µM"},
            "activityMeasureValue": "MIC",
            "concentration": "4",
        },
    ]
    assert collect_strain_measurements(activities) == {"strain": {("MIC", "µM"): "2"}}


def test_build_includes_atcc_and_skips_unmatched_ids() -> None:
    records = [
        {
            "id": 1,
            "targetActivities": [
                {
                    "targetSpecies": {"name": "Example ATCC 1"},
                    "unit": {"name": "µM"},
                    "activityMeasureValue": "MIC",
                    "concentration": ">4",
                }
            ],
        },
        {"id": 2, "targetActivities": []},
    ]
    result = build_paper_mic_table(
        records, pd.DataFrame({"DBAASP_id": [1], "SMILES": ["CC"]})
    )
    assert result.table.to_dict("records") == [
        {
            "DBAASP_id": 1,
            "strain_name": "Example ATCC 1",
            "SMILES": "CC",
            "MIC": 8.0,
        }
    ]


def test_weight_override_does_not_change_displayed_smiles() -> None:
    records = [
        {
            "id": 1,
            "targetActivities": [
                {
                    "targetSpecies": {"name": "strain"},
                    "unit": {"name": "µg/ml"},
                    "activityMeasureValue": "MIC",
                    "concentration": "1",
                }
            ],
        }
    ]
    displayed = pd.DataFrame({"DBAASP_id": [1], "SMILES": ["CC"]})
    original = pd.DataFrame({"DBAASP_id": [1], "SMILES": ["C"]})
    current = build_paper_mic_table(records, displayed)
    historical = build_paper_mic_table(
        records, displayed, molecular_weight_smiles_overrides=original
    )
    assert historical.table.loc[0, "SMILES"] == "CC"
    assert historical.table.loc[0, "MIC"] != current.table.loc[0, "MIC"]
