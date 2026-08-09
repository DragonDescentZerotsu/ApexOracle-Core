from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexoracle.evaluation.hierarchical_mic_censor_sensitivity import (
    CENSOR_LEFT,
    CENSOR_NONE,
    CENSOR_RIGHT,
    CENSOR_RIGHT_DOUBLE,
    SCENARIO_ALL_CENSORED_EXCLUDED,
    SCENARIO_PAPER_LEGACY,
    SCENARIO_RIGHT_EXCLUDED,
    alternative_label_values,
    annotate_training_table,
    calculate_metrics,
    classify_censor_rule,
    evaluate_scenarios,
    validate_lineage_against_frozen_table,
)
from apexoracle.evaluation.hierarchical_mic_censor_workflow import (
    OUTPUT_FILENAMES,
    prepare_outputs,
    validate_multipliers,
)


@pytest.mark.parametrize(
    ("raw", "censor_class", "legacy_multiplier"),
    [
        (">100", CENSOR_RIGHT, 2.0),
        (">=64", CENSOR_RIGHT, 2.0),
        ("≥ 800", CENSOR_RIGHT, 1.0),
        (">>500±10", CENSOR_RIGHT_DOUBLE, 3.0),
        ("<1", CENSOR_LEFT, 1.0),
        ("64->128", CENSOR_NONE, 1.0),
        ("64 - >=128", CENSOR_NONE, 1.0),
        ("16", CENSOR_NONE, 1.0),
    ],
)
def test_classify_censor_rule_preserves_frozen_parser_behavior(
    raw: str, censor_class: str, legacy_multiplier: float
) -> None:
    rule = classify_censor_rule(raw)
    assert rule.censor_class == censor_class
    assert rule.legacy_multiplier == legacy_multiplier


def test_lineage_validation_requires_exact_frozen_row_order() -> None:
    lineage = pd.DataFrame(
        {
            "DBAASP_id": ["1", "2"],
            "strain_name": ["a", "b"],
            "raw_concentration": [">8", "4"],
            "raw_unit": ["micromolar", "micromolar"],
            "censor_class": [CENSOR_RIGHT, CENSOR_NONE],
            "legacy_multiplier": [2.0, 1.0],
        }
    )
    frozen = pd.DataFrame(
        {
            "DBAASP_id": [1, 2],
            "strain_name": ["a", "b"],
            "MIC": [16.0, 4.0],
        }
    )
    validated = validate_lineage_against_frozen_table(lineage, frozen)
    assert validated["paper_MIC_um"].tolist() == [16.0, 4.0]
    with pytest.raises(ValueError, match="row-order mismatch"):
        validate_lineage_against_frozen_table(
            lineage, frozen.iloc[::-1].reset_index(drop=True)
        )


def test_training_annotation_marks_inhouse_and_checks_mic_values() -> None:
    lineage = pd.DataFrame(
        {
            "DBAASP_id": ["1"],
            "strain_name": ["a"],
            "raw_concentration": [">8"],
            "raw_unit": ["micromolar"],
            "censor_class": [CENSOR_RIGHT],
            "legacy_multiplier": [2.0],
            "paper_MIC_um": [16.0],
            "source_row_index": [0],
        }
    )
    mic = pd.DataFrame(
        {
            "DBAASP_id": [1, "inhouse-1"],
            "strain_name": ["a", "b"],
            "SMILES": ["[1]", "[2]"],
            "MIC": [16.0, 8.0],
        }
    )
    annotated = annotate_training_table(mic, lineage)
    assert annotated["measurement_source"].tolist() == ["DBAASP", "in_house"]
    assert annotated.loc[1, "censor_class"] == "inhouse_not_applicable"
    bad = mic.copy()
    bad.loc[0, "MIC"] = 15.0
    with pytest.raises(ValueError, match="maximum absolute error"):
        annotate_training_table(bad, lineage)


def sensitivity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol": ["strain"] * 5,
            "group_index": [0, 0, 1, 1, 1],
            "group_name": ["fold 1", "fold 1", "fold 2", "fold 2", "fold 2"],
            "molecule_identity": ["a", "b", "c", "d", "e"],
            "strain_name": ["x", "x", "y", "y", "z"],
            "MIC_um": [16.0, 8.0, 24.0, 4.0, 2.0],
            "prediction": -np.log10(np.asarray([8.0, 8.0, 8.0, 4.0, 2.0]) / 10),
            "censor_class": [
                CENSOR_RIGHT,
                CENSOR_NONE,
                CENSOR_RIGHT_DOUBLE,
                CENSOR_LEFT,
                CENSOR_NONE,
            ],
            "legacy_multiplier": [2.0, 1.0, 3.0, 1.0, 1.0],
        }
    )


def test_alternative_label_scenarios_filter_and_rescale_as_declared() -> None:
    frame = sensitivity_frame()
    legacy_rows, legacy_labels = alternative_label_values(frame, SCENARIO_PAPER_LEGACY)
    assert len(legacy_rows) == 5
    assert legacy_labels[0] == pytest.approx(-np.log10(16 / 10))

    multiplier_rows, multiplier_labels = alternative_label_values(
        frame, "right_censored_multiplier_1"
    )
    assert len(multiplier_rows) == 4
    assert CENSOR_RIGHT_DOUBLE not in set(multiplier_rows["censor_class"])
    assert multiplier_labels[0] == pytest.approx(-np.log10(8 / 10))

    right_excluded, _ = alternative_label_values(frame, SCENARIO_RIGHT_EXCLUDED)
    assert set(right_excluded["censor_class"]) == {
        CENSOR_NONE,
        CENSOR_LEFT,
    }
    uncensored, _ = alternative_label_values(frame, SCENARIO_ALL_CENSORED_EXCLUDED)
    assert set(uncensored["censor_class"]) == {CENSOR_NONE}


def test_metrics_and_group_mean_are_recomputed_from_paired_rows() -> None:
    exact = np.asarray([0.0, 1.0, 2.0])
    metrics = calculate_metrics(exact, exact)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)

    evaluated = evaluate_scenarios(
        sensitivity_frame(),
        protocol="strain",
        scenarios=[SCENARIO_PAPER_LEGACY, "right_censored_multiplier_2"],
    )
    assert set(evaluated["aggregation"]) == {
        "group",
        "pooled",
        "mean_across_groups",
    }
    group_rows = evaluated.loc[
        evaluated["scenario"].eq(SCENARIO_PAPER_LEGACY)
        & evaluated["aggregation"].eq("group")
    ]
    mean_row = evaluated.loc[
        evaluated["scenario"].eq(SCENARIO_PAPER_LEGACY)
        & evaluated["aggregation"].eq("mean_across_groups")
    ].iloc[0]
    assert mean_row["r2"] == pytest.approx(group_rows["r2"].mean())
    assert mean_row["r2_sample_sd"] == pytest.approx(group_rows["r2"].std(ddof=1))


def test_workflow_validates_multiplier_grid_and_closed_output_contract(
    tmp_path: Path,
) -> None:
    assert validate_multipliers([1, 2, 4]) == (1.0, 2.0, 4.0)
    with pytest.raises(ValueError, match="must include the paper multiplier 2"):
        validate_multipliers([1, 4])
    with pytest.raises(ValueError, match="must be unique"):
        validate_multipliers([1, 2, 2])

    output_dir = tmp_path / "analysis"
    paths = prepare_outputs(output_dir, overwrite=False)
    assert set(paths) == set(OUTPUT_FILENAMES)
    paths["metrics"].write_text("metric\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        prepare_outputs(output_dir, overwrite=False)
    assert prepare_outputs(output_dir, overwrite=True) == paths
