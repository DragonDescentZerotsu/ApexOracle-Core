"""Evaluation-label sensitivity for paper-era censored MIC imputations.

The historical training table converts right-censored DBAASP measurements into
finite point labels.  This module reconstructs that row-level lineage and
recalculates metrics from frozen predictions.  It never retrains a model and
must therefore be described as an evaluation-label sensitivity analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from apexoracle.data.amp_mic import (
    collect_strain_measurements,
    select_measurement,
)


CENSOR_NONE = "not_censored"
CENSOR_LEFT = "left_censored"
CENSOR_RIGHT = "right_censored"
CENSOR_RIGHT_DOUBLE = "right_censored_double_angle"
CENSOR_INHOUSE = "inhouse_not_applicable"

SCENARIO_PAPER_LEGACY = "paper_legacy"
SCENARIO_RIGHT_EXCLUDED = "right_censored_excluded"
SCENARIO_ALL_CENSORED_EXCLUDED = "all_censored_excluded"
SCENARIO_MULTIPLIER_TEMPLATE = "right_censored_multiplier_{:g}"

METRIC_NAMES = ("r2", "mae", "rmse", "spearman", "pearson")


@dataclass(frozen=True)
class CensorRule:
    censor_class: str
    legacy_multiplier: float


def classify_censor_rule(raw_concentration: str) -> CensorRule:
    """Classify censoring while preserving the exact legacy multiplier behavior."""

    raw = str(raw_concentration)
    range_separators = (" - =>", "->=", " - >=", "->")
    if any(separator in raw for separator in range_separators):
        return CensorRule(CENSOR_NONE, 1.0)
    if ">>" in raw:
        return CensorRule(CENSOR_RIGHT_DOUBLE, 3.0)
    if ">" in raw or ">=" in raw:
        return CensorRule(CENSOR_RIGHT, 2.0)
    if "≥" in raw:
        # The paper table grouped this glyph with ASCII >/>=, but the frozen
        # parser stripped it without applying the later ASCII-only multiplier.
        return CensorRule(CENSOR_RIGHT, 1.0)
    if "<" in raw or "≤" in raw:
        return CensorRule(CENSOR_LEFT, 1.0)
    return CensorRule(CENSOR_NONE, 1.0)


def build_dbaasp_censor_lineage(
    records: Iterable[Mapping[str, object]],
    smiles_ids: set[str],
) -> pd.DataFrame:
    """Reconstruct selected raw concentration and censor rule per DBAASP row."""

    rows: list[dict[str, object]] = []
    for record in records:
        dbaasp_id = str(record["id"])
        if dbaasp_id not in smiles_ids:
            continue
        measurements_by_strain = collect_strain_measurements(
            record.get("targetActivities")  # type: ignore[arg-type]
        )
        for strain_name, measurements in measurements_by_strain.items():
            selected = select_measurement(measurements)
            if selected is None or not selected.concentration:
                continue
            rule = classify_censor_rule(selected.concentration)
            rows.append(
                {
                    "DBAASP_id": dbaasp_id,
                    "strain_name": strain_name,
                    "raw_concentration": selected.concentration,
                    "raw_unit": (
                        "micrograms_per_ml"
                        if selected.convert_micrograms_per_ml
                        else "micromolar"
                    ),
                    "censor_class": rule.censor_class,
                    "legacy_multiplier": rule.legacy_multiplier,
                }
            )
    output = pd.DataFrame(rows)
    keys = ["DBAASP_id", "strain_name"]
    if output.duplicated(keys).any():
        duplicates = output.loc[output.duplicated(keys, keep=False), keys]
        raise ValueError(
            "DBAASP censor lineage is not unique by peptide/strain: "
            f"{duplicates.head().to_dict(orient='records')}"
        )
    return output


def validate_lineage_against_frozen_table(
    lineage: pd.DataFrame,
    frozen_dbaasp_mic: pd.DataFrame,
) -> pd.DataFrame:
    """Attach frozen MIC values after exact row-order and identity validation."""

    required = {"DBAASP_id", "strain_name", "MIC"}
    missing = required - set(frozen_dbaasp_mic.columns)
    if missing:
        raise ValueError(f"Frozen DBAASP MIC table is missing {sorted(missing)}")
    frozen = frozen_dbaasp_mic.copy()
    frozen["DBAASP_id"] = frozen["DBAASP_id"].map(str)
    if len(lineage) != len(frozen):
        raise ValueError(
            f"Lineage/frozen row count mismatch: {len(lineage)} != {len(frozen)}"
        )
    for column in ("DBAASP_id", "strain_name"):
        left = lineage[column].astype(str).reset_index(drop=True)
        right = frozen[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            first = int(np.flatnonzero(left.to_numpy() != right.to_numpy())[0])
            raise ValueError(
                f"Lineage/frozen row-order mismatch at row {first}, column {column}: "
                f"{left.iloc[first]!r} != {right.iloc[first]!r}"
            )
    output = lineage.copy()
    output["paper_MIC_um"] = frozen["MIC"].astype(float).to_numpy()
    output["source_row_index"] = np.arange(len(output), dtype=int)
    return output


def annotate_training_table(
    mic_frame: pd.DataFrame,
    dbaasp_lineage: pd.DataFrame,
) -> pd.DataFrame:
    """Attach public-source censor metadata; mark unmatched in-house rows explicitly."""

    output = mic_frame.copy()
    output["DBAASP_id"] = output["DBAASP_id"].map(str)
    lineage = dbaasp_lineage.copy()
    lineage["DBAASP_id"] = lineage["DBAASP_id"].map(str)
    lineage_columns = [
        "DBAASP_id",
        "strain_name",
        "raw_concentration",
        "raw_unit",
        "censor_class",
        "legacy_multiplier",
        "paper_MIC_um",
        "source_row_index",
    ]
    output = output.merge(
        lineage[lineage_columns],
        on=["DBAASP_id", "strain_name"],
        how="left",
        validate="one_to_one",
    )
    is_dbaasp = output["censor_class"].notna()
    if is_dbaasp.any():
        observed = output.loc[is_dbaasp, "MIC"].astype(float).to_numpy()
        expected = output.loc[is_dbaasp, "paper_MIC_um"].astype(float).to_numpy()
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
            maximum = float(np.max(np.abs(observed - expected)))
            raise ValueError(
                "Tokenized training MIC does not match frozen DBAASP lineage; "
                f"maximum absolute error={maximum}"
            )
    output["measurement_source"] = np.where(is_dbaasp, "DBAASP", "in_house")
    output.loc[~is_dbaasp, "censor_class"] = CENSOR_INHOUSE
    output.loc[~is_dbaasp, "legacy_multiplier"] = 1.0
    output.loc[~is_dbaasp, "raw_unit"] = "not_available"
    return output


def alternative_label_values(
    frame: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select scenario rows and return transformed labels for frozen predictions."""

    output = frame.copy()
    censor_class = output["censor_class"].astype(str)
    right = censor_class.eq(CENSOR_RIGHT)
    double = censor_class.eq(CENSOR_RIGHT_DOUBLE)
    left = censor_class.eq(CENSOR_LEFT)

    if scenario == SCENARIO_PAPER_LEGACY:
        selected = output
        mic = selected["MIC_um"].astype(float).to_numpy()
    elif scenario == SCENARIO_RIGHT_EXCLUDED:
        selected = output.loc[~(right | double)].copy()
        mic = selected["MIC_um"].astype(float).to_numpy()
    elif scenario == SCENARIO_ALL_CENSORED_EXCLUDED:
        selected = output.loc[~(right | double | left)].copy()
        mic = selected["MIC_um"].astype(float).to_numpy()
    elif scenario.startswith("right_censored_multiplier_"):
        multiplier = float(scenario.rsplit("_", 1)[-1])
        if multiplier <= 0:
            raise ValueError("Right-censor multiplier must be positive")
        # The non-standard >> rows are excluded from the ordinary >V grid.
        selected = output.loc[~double].copy()
        selected_right = selected["censor_class"].astype(str).eq(CENSOR_RIGHT)
        mic_series = selected["MIC_um"].astype(float).copy()
        legacy = selected.loc[selected_right, "legacy_multiplier"].astype(float)
        mic_series.loc[selected_right] = (
            mic_series.loc[selected_right] * multiplier / legacy
        )
        mic = mic_series.to_numpy()
    else:
        raise ValueError(f"Unknown sensitivity scenario: {scenario}")
    if len(selected) == 0 or np.any(~np.isfinite(mic)) or np.any(mic <= 0):
        raise ValueError(f"Scenario {scenario} produced invalid MIC values")
    return selected.reset_index(drop=True), -np.log10(mic / 10.0)


def calculate_metrics(labels: Sequence[float], predictions: Sequence[float]) -> dict:
    """Calculate regression metrics on the paper transformed-label scale."""

    label_values = np.asarray(labels, dtype=float)
    prediction_values = np.asarray(predictions, dtype=float)
    if len(label_values) != len(prediction_values) or len(label_values) < 2:
        raise ValueError("At least two paired labels/predictions are required")
    residual = label_values - prediction_values
    denominator = np.square(label_values - label_values.mean()).sum()
    r2 = (
        float("nan")
        if denominator == 0
        else float(1.0 - np.square(residual).sum() / denominator)
    )
    constant_prediction = np.all(prediction_values == prediction_values[0])
    correlation_defined = denominator != 0 and not constant_prediction
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "spearman": (
            float(spearmanr(label_values, prediction_values).statistic)
            if correlation_defined
            else float("nan")
        ),
        "pearson": (
            float(pearsonr(label_values, prediction_values).statistic)
            if correlation_defined
            else float("nan")
        ),
    }


def evaluate_scenarios(
    frame: pd.DataFrame,
    *,
    protocol: str,
    scenarios: Sequence[str],
) -> pd.DataFrame:
    """Evaluate group, pooled, and unweighted mean-across-group metrics."""

    rows: list[dict[str, object]] = []
    group_values = sorted(frame["group_index"].astype(int).unique())
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("group", str(group), frame.loc[frame["group_index"].astype(int).eq(group)])
        for group in group_values
    ]
    scopes.append(("pooled", "all_groups_pooled", frame))
    for scenario in scenarios:
        scenario_group_rows: list[dict[str, object]] = []
        for aggregation, group, scope in scopes:
            selected, labels = alternative_label_values(scope, scenario)
            predictions = selected["prediction"].astype(float).to_numpy()
            row: dict[str, object] = {
                "protocol": protocol,
                "scenario": scenario,
                "aggregation": aggregation,
                "group": group,
                "measurements": len(selected),
                "unique_molecules": selected["molecule_identity"].nunique(),
                "pathogens": selected["strain_name"].nunique(),
                "right_censored_measurements": int(
                    selected["censor_class"].astype(str).eq(CENSOR_RIGHT).sum()
                ),
                "left_censored_measurements": int(
                    selected["censor_class"].astype(str).eq(CENSOR_LEFT).sum()
                ),
                "double_angle_measurements": int(
                    selected["censor_class"].astype(str).eq(CENSOR_RIGHT_DOUBLE).sum()
                ),
            }
            row.update(calculate_metrics(labels, predictions))
            rows.append(row)
            if aggregation == "group":
                scenario_group_rows.append(row)
        group_frame = pd.DataFrame(scenario_group_rows)
        mean_row: dict[str, object] = {
            "protocol": protocol,
            "scenario": scenario,
            "aggregation": "mean_across_groups",
            "group": "mean_across_groups",
            "measurements": int(group_frame["measurements"].sum()),
            "unique_molecules": float("nan"),
            "pathogens": float("nan"),
            "right_censored_measurements": int(
                group_frame["right_censored_measurements"].sum()
            ),
            "left_censored_measurements": int(
                group_frame["left_censored_measurements"].sum()
            ),
            "double_angle_measurements": int(
                group_frame["double_angle_measurements"].sum()
            ),
        }
        for metric in METRIC_NAMES:
            mean_row[metric] = float(group_frame[metric].mean())
            mean_row[f"{metric}_sample_sd"] = float(group_frame[metric].std(ddof=1))
        rows.append(mean_row)
    return pd.DataFrame(rows)


def load_json_records(path: Path) -> list[Mapping[str, object]]:
    """Load the frozen DBAASP JSON with a typed public boundary."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected DBAASP source JSON to contain a list")
    return payload
