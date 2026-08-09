"""Workflow orchestration for hierarchical MIC multiplier sensitivity.

This module keeps reconstruction, frozen-prediction alignment, output tables,
and manifest generation out of the command-line entrypoint.  It is deliberately
read-only with respect to model assets: the only writes are the declared
analysis outputs under the caller-provided output directory.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, fields
from io import StringIO
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from apexoracle.data.amp_mic import sha256_file
from apexoracle.data.hierarchical_mic_preparation import (
    HoldoutSplit,
    PreparedHierarchicalMicData,
    build_holdout_split,
    prepare_hierarchical_mic_data,
)
from apexoracle.evaluation.hierarchical_mic_censor_sensitivity import (
    METRIC_NAMES,
    SCENARIO_ALL_CENSORED_EXCLUDED,
    SCENARIO_MULTIPLIER_TEMPLATE,
    SCENARIO_PAPER_LEGACY,
    SCENARIO_RIGHT_EXCLUDED,
    annotate_training_table,
    build_dbaasp_censor_lineage,
    evaluate_scenarios,
    load_json_records,
    validate_lineage_against_frozen_table,
)
from apexoracle.evaluation.hierarchical_mic_molecule_overlap import (
    apply_legacy_token_length_filter,
    model_input_identity,
)
from apexoracle.training.hierarchical_mic_runner import (
    HierarchicalMicConfig,
    prepare_holdout_frames,
)


@dataclass(frozen=True)
class CensorSensitivityInputs:
    """Complete frozen input contract for one sensitivity run."""

    dbaasp_json: Path
    smiles_csv: Path
    frozen_dbaasp_mic: Path
    mic_records: Path
    small_molecule_records: Path
    strain_predictions: Path
    phylum_predictions: Path
    strain_manifest: Path
    config: Path

    def resolved(self) -> "CensorSensitivityInputs":
        return CensorSensitivityInputs(
            **{
                field.name: getattr(self, field.name).resolve()
                for field in fields(self)
            }
        )

    def as_dict(self) -> dict[str, Path]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


OUTPUT_FILENAMES = {
    "row_assignments": "row_censor_assignments.csv",
    "source_censor_rules": "source_censor_rules.csv",
    "eligible_censor_counts": "eligible_censor_counts.csv",
    "metrics": "metrics.csv",
    "metric_deltas": "metric_deltas.csv",
    "duplicate_prediction_audit": "duplicate_prediction_normalization_audit.csv",
    "manifest": "analysis_manifest.json",
}

# Preserve the canonical artifact's original provenance date across deterministic
# reruns.  This is intentionally not the wall-clock date of the latest execution.
CANONICAL_ANALYSIS_DATE = "2026-08-07"

ROW_ASSIGNMENT_COLUMNS = (
    "row_key",
    "protocol",
    "group_index",
    "group_name",
    "route",
    "DBAASP_id",
    "molecule_identity",
    "strain_name",
    "MIC_um",
    "label_z",
    "prediction",
    "saved_prediction",
    "prediction_duplicate_normalization_delta",
    "ensemble_members",
    "raw_concentration",
    "raw_unit",
    "censor_class",
    "legacy_multiplier",
    "measurement_source",
    "source_row_index",
    "lineage_join_multiplicity",
    "lineage_join_occurrence",
    "lineage_exact_row_assignment",
)


def validate_multipliers(multipliers: Sequence[float]) -> tuple[float, ...]:
    """Return a stable multiplier tuple after validating the paper baseline."""

    values = tuple(float(value) for value in multipliers)
    if not values or any(value <= 0 for value in values):
        raise ValueError("Right-censor multipliers must be positive")
    if len(set(values)) != len(values):
        raise ValueError("Right-censor multipliers must be unique")
    if 2.0 not in values:
        raise ValueError("The sensitivity grid must include the paper multiplier 2")
    return values


def load_frozen_strain_split(path: Path) -> HoldoutSplit:
    """Load the frozen three-fold strain membership used by the completed replay."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = sorted(payload["folds"], key=lambda row: int(row["fold"]))
    return HoldoutSplit(
        protocol="strain",
        group_names=tuple(f"fold {int(row['fold']) + 1}" for row in folds),
        test_groups=tuple(tuple(map(str, row["test_strain_ids"])) for row in folds),
    )


def output_paths(output_dir: Path) -> dict[str, Path]:
    """Resolve the complete, closed set of files written by this workflow."""

    return {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}


def prepare_outputs(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    """Create the output directory and reject accidental partial overwrites."""

    paths = output_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Derived outputs already exist; pass --overwrite to replace only:\n"
            f"{formatted}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def eligible_route_frame(
    frame: pd.DataFrame,
    *,
    route: str,
    group: int,
) -> pd.DataFrame:
    """Apply frozen eligibility and attach deterministic route-level identities."""

    output = apply_legacy_token_length_filter(frame)
    output["route"] = route
    output["molecule_identity"] = output["SMILES"].map(model_input_identity)
    output["row_key"] = [f"g{group}:{route}:{index}" for index in range(len(output))]
    return output


def reconstruct_assignments(
    prepared: PreparedHierarchicalMicData,
    split: HoldoutSplit,
    *,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align reconstructed censor lineage to one frozen protocol prediction table."""

    frames = []
    for group in range(len(split.test_groups)):
        heldout = prepare_holdout_frames(prepared, split, group)
        routes = (
            (heldout.genome_text_test, "genome_text"),
            (heldout.text_only_test, "text_only"),
        )
        for route_frame, route in routes:
            eligible = eligible_route_frame(route_frame, route=route, group=group)
            eligible["group_index"] = group
            frames.append(eligible)
    reconstructed = pd.concat(frames, ignore_index=True)

    required_prediction_columns = {
        "row_key",
        "protocol",
        "group_index",
        "group_name",
        "route",
        "DBAASP_id",
        "molecule_identity",
        "strain_name",
        "MIC_um",
        "label_z",
        "prediction",
        "ensemble_members",
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing {sorted(missing)}")
    if predictions["row_key"].duplicated().any():
        raise ValueError("Prediction row_key values are not unique")
    if reconstructed["row_key"].duplicated().any():
        raise ValueError("Reconstructed row_key values are not unique")

    metadata_columns = [
        "row_key",
        "group_index",
        "route",
        "DBAASP_id",
        "molecule_identity",
        "strain_name",
        "MIC",
        "raw_concentration",
        "raw_unit",
        "censor_class",
        "legacy_multiplier",
        "measurement_source",
        "source_row_index",
    ]
    reconstructed = reconstructed[metadata_columns].copy()
    reconstructed["DBAASP_id"] = reconstructed["DBAASP_id"].map(str)
    reconstructed = reconstructed.rename(columns={"MIC": "MIC_um"})
    prediction_frame = predictions.copy()
    prediction_frame["DBAASP_id"] = prediction_frame["DBAASP_id"].map(str)
    join_keys = [
        "group_index",
        "route",
        "DBAASP_id",
        "molecule_identity",
        "strain_name",
        "MIC_um",
    ]
    reconstructed_counts = reconstructed.groupby(join_keys, dropna=False).size()
    prediction_counts = prediction_frame.groupby(join_keys, dropna=False).size()
    if not reconstructed_counts.equals(prediction_counts):
        comparison = pd.concat(
            [
                reconstructed_counts.rename("reconstructed"),
                prediction_counts.rename("predictions"),
            ],
            axis=1,
        ).fillna(0)
        mismatch = comparison.loc[
            comparison["reconstructed"].ne(comparison["predictions"])
        ]
        raise ValueError(
            "Prediction/reconstruction stable-key multiplicities do not match: "
            f"{mismatch.head().reset_index().to_dict(orient='records')}"
        )

    prediction_frame["saved_prediction"] = prediction_frame["prediction"].astype(float)
    prediction_frame["prediction"] = prediction_frame.groupby(join_keys, dropna=False)[
        "saved_prediction"
    ].transform("mean")
    prediction_frame["prediction_duplicate_normalization_delta"] = (
        prediction_frame["prediction"] - prediction_frame["saved_prediction"]
    )
    reconstructed["lineage_join_multiplicity"] = reconstructed.groupby(
        join_keys, dropna=False
    )["row_key"].transform("size")
    prediction_frame["lineage_join_multiplicity"] = prediction_frame.groupby(
        join_keys, dropna=False
    )["row_key"].transform("size")
    reconstructed["lineage_join_occurrence"] = reconstructed.groupby(
        join_keys, dropna=False
    ).cumcount()
    prediction_frame["lineage_join_occurrence"] = prediction_frame.groupby(
        join_keys, dropna=False
    ).cumcount()
    occurrence_keys = join_keys + [
        "lineage_join_multiplicity",
        "lineage_join_occurrence",
    ]
    merged = prediction_frame.merge(
        reconstructed.drop(columns=["row_key"]),
        on=occurrence_keys,
        how="left",
        validate="one_to_one",
    )
    if merged["censor_class"].isna().any():
        raise ValueError("Some prediction rows lack reconstructed censor lineage")
    recalculated_labels = -np.log10(merged["MIC_um"].astype(float) / 10.0)
    if not np.allclose(
        merged["label_z"].astype(float),
        recalculated_labels,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Saved prediction labels do not match reconstructed MIC")
    if set(merged["ensemble_members"].astype(int)) != {7}:
        raise ValueError("Expected every prediction to be a 7-member ensemble mean")
    merged["lineage_exact_row_assignment"] = merged["lineage_join_multiplicity"].eq(1)
    ambiguous = ~merged["lineage_exact_row_assignment"]
    merged.loc[ambiguous, "raw_concentration"] = "multiple_equivalent_records"
    merged.loc[ambiguous, "source_row_index"] = np.nan
    return merged


def source_rule_counts(lineage: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw selected DBAASP measurements by finite-label rule."""

    grouped = (
        lineage.groupby(["censor_class", "legacy_multiplier"], dropna=False)
        .size()
        .rename("measurements")
        .reset_index()
    )
    grouped["fraction"] = grouped["measurements"] / len(lineage)
    return grouped.sort_values(["censor_class", "legacy_multiplier"]).reset_index(
        drop=True
    )


def eligible_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize eligible held-out rows by protocol, group, and censor class."""

    rows = []
    for protocol, protocol_frame in assignments.groupby("protocol", sort=True):
        scopes = [
            ("pooled", "all_groups_pooled", protocol_frame),
            *[
                (
                    "group",
                    str(group),
                    protocol_frame.loc[
                        protocol_frame["group_index"].astype(int).eq(group)
                    ],
                )
                for group in sorted(protocol_frame["group_index"].astype(int).unique())
            ],
        ]
        for aggregation, group, scope in scopes:
            counts = scope["censor_class"].astype(str).value_counts()
            for censor_class, count in counts.sort_index().items():
                rows.append(
                    {
                        "protocol": protocol,
                        "aggregation": aggregation,
                        "group": group,
                        "censor_class": censor_class,
                        "measurements": int(count),
                        "total_measurements": len(scope),
                        "fraction": float(count / len(scope)),
                    }
                )
    return pd.DataFrame(rows)


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach deltas from the exact frozen parser and harmonized 2V scenarios."""

    key = ["protocol", "aggregation", "group"]
    legacy = metrics.loc[
        metrics["scenario"].eq(SCENARIO_PAPER_LEGACY), key + list(METRIC_NAMES)
    ].rename(columns={metric: f"{metric}_paper_legacy" for metric in METRIC_NAMES})
    multiplier_two = metrics.loc[
        metrics["scenario"].eq(SCENARIO_MULTIPLIER_TEMPLATE.format(2)),
        key + list(METRIC_NAMES),
    ].rename(columns={metric: f"{metric}_multiplier_2" for metric in METRIC_NAMES})
    output = metrics.merge(legacy, on=key, validate="many_to_one").merge(
        multiplier_two, on=key, validate="many_to_one"
    )
    for metric in METRIC_NAMES:
        output[f"delta_{metric}_vs_paper_legacy"] = (
            output[metric] - output[f"{metric}_paper_legacy"]
        )
        output[f"delta_{metric}_vs_multiplier_2"] = (
            output[metric] - output[f"{metric}_multiplier_2"]
        )
    return output


def duplicate_prediction_audit(assignments: pd.DataFrame) -> pd.DataFrame:
    """Quantify metric effects from stabilizing indistinguishable duplicate rows."""

    normalized = []
    saved = []
    for protocol, frame in assignments.groupby("protocol", sort=True):
        normalized.append(
            evaluate_scenarios(
                frame.reset_index(drop=True),
                protocol=protocol,
                scenarios=[SCENARIO_PAPER_LEGACY],
            )
        )
        saved_frame = frame.copy()
        saved_frame["prediction"] = saved_frame["saved_prediction"]
        saved.append(
            evaluate_scenarios(
                saved_frame.reset_index(drop=True),
                protocol=protocol,
                scenarios=[SCENARIO_PAPER_LEGACY],
            )
        )
    key = ["protocol", "scenario", "aggregation", "group"]
    normalized_frame = pd.concat(normalized, ignore_index=True)
    saved_frame = pd.concat(saved, ignore_index=True)
    rows = []
    for metric in METRIC_NAMES:
        left = normalized_frame[key + [metric]].rename(
            columns={metric: "normalized_prediction_metric"}
        )
        right = saved_frame[key + [metric]].rename(
            columns={metric: "saved_prediction_metric"}
        )
        merged = left.merge(right, on=key, validate="one_to_one")
        merged["metric"] = metric
        merged["delta_normalized_minus_saved"] = (
            merged["normalized_prediction_metric"] - merged["saved_prediction_metric"]
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)[
        key
        + [
            "metric",
            "normalized_prediction_metric",
            "saved_prediction_metric",
            "delta_normalized_minus_saved",
        ]
    ]


def hash_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_censor_sensitivity_analysis(
    repo_root: Path,
    inputs: CensorSensitivityInputs,
    *,
    output_dir: Path,
    multipliers: Sequence[float] = (1.0, 2.0, 4.0),
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the complete frozen-prediction sensitivity workflow."""

    multiplier_values = validate_multipliers(multipliers)
    resolved_inputs = inputs.resolved()
    input_paths = resolved_inputs.as_dict()
    missing_inputs = [path for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing inputs: {missing_inputs}")
    paths = prepare_outputs(output_dir.resolve(), overwrite=overwrite)

    records = load_json_records(input_paths["dbaasp_json"])
    smiles_ids = set(
        pd.read_csv(
            input_paths["smiles_csv"],
            usecols=["DBAASP_id"],
            dtype={"DBAASP_id": str},
        )["DBAASP_id"]
    )
    raw_lineage = build_dbaasp_censor_lineage(records, smiles_ids)
    frozen_dbaasp = pd.read_csv(input_paths["frozen_dbaasp_mic"])
    lineage = validate_lineage_against_frozen_table(raw_lineage, frozen_dbaasp)
    mic_frame = pd.read_csv(input_paths["mic_records"])
    annotated_mic = annotate_training_table(mic_frame, lineage)

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        prepared = prepare_hierarchical_mic_data(
            repo_root,
            mic_frame=annotated_mic,
            small_molecule_data_path=input_paths["small_molecule_records"],
        )
    phylum_config = HierarchicalMicConfig.load(
        input_paths["config"], repo_root, holdout_protocol="phylum"
    )
    strain_split = load_frozen_strain_split(input_paths["strain_manifest"])
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        phylum_split = build_holdout_split(
            prepared,
            repo_root,
            "phylum",
            adapter=phylum_config.holdout_adapter,
            group_names=phylum_config.holdout_group_names,
            tree_path=phylum_config.holdout_tree,
            num_clusters=phylum_config.holdout_clusters,
        )

    prediction_inputs = {
        "strain": pd.read_csv(input_paths["strain_predictions"]),
        "phylum": pd.read_csv(input_paths["phylum_predictions"]),
    }
    assignments = []
    for protocol, split in (("strain", strain_split), ("phylum", phylum_split)):
        prediction_frame = prediction_inputs[protocol]
        if set(prediction_frame["protocol"].astype(str)) != {protocol}:
            raise ValueError(f"Prediction protocol mismatch for {protocol}")
        assignments.append(
            reconstruct_assignments(prepared, split, predictions=prediction_frame)
        )
    all_assignments = pd.concat(assignments, ignore_index=True)
    scenarios = [
        SCENARIO_PAPER_LEGACY,
        *[
            SCENARIO_MULTIPLIER_TEMPLATE.format(multiplier)
            for multiplier in multiplier_values
        ],
        SCENARIO_RIGHT_EXCLUDED,
        SCENARIO_ALL_CENSORED_EXCLUDED,
    ]
    metrics = pd.concat(
        [
            evaluate_scenarios(
                frame.reset_index(drop=True),
                protocol=protocol,
                scenarios=scenarios,
            )
            for protocol, frame in all_assignments.groupby("protocol", sort=True)
        ],
        ignore_index=True,
    )
    source_rules = source_rule_counts(lineage)
    counts = eligible_counts(all_assignments)
    deltas = metric_deltas(metrics)
    prediction_audit = duplicate_prediction_audit(all_assignments)

    all_assignments[list(ROW_ASSIGNMENT_COLUMNS)].to_csv(
        paths["row_assignments"], index=False
    )
    source_rules.to_csv(paths["source_censor_rules"], index=False)
    counts.to_csv(paths["eligible_censor_counts"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    deltas.to_csv(paths["metric_deltas"], index=False)
    prediction_audit.to_csv(paths["duplicate_prediction_audit"], index=False)
    output_hashes = {
        name: hash_record(path) for name, path in paths.items() if name != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "analysis_type": "evaluation_label_sensitivity_without_retraining",
        "generated_on": CANONICAL_ANALYSIS_DATE,
        "right_censor_multipliers": list(multiplier_values),
        "scenarios": scenarios,
        "protocols": {
            "strain": {
                "membership": "fixed_PYTHONHASHSEED_0_reconstruction",
                "groups": 3,
                "ensemble_members": 7,
                "measurements": int(all_assignments["protocol"].eq("strain").sum()),
            },
            "phylum": {
                "membership": "canonical_taxonomy_cluster_adapter",
                "groups": 3,
                "ensemble_members": 7,
                "measurements": int(all_assignments["protocol"].eq("phylum").sum()),
            },
        },
        "lineage_contract": {
            "dbaasp_selected_measurements": len(lineage),
            "tokenized_training_rows": len(annotated_mic),
            "right_censored_definition": (
                "ordinary >, >=, or unicode >=; range-like arrows excluded"
            ),
            "double_angle_policy": "excluded from ordinary multiplier grid",
            "all_censored_excluded_policy": (
                "all <, <=, >, >=, unicode censor, and >> rows excluded"
            ),
            "metric_scale": "-log10(MIC_um / 10)",
            "duplicate_key_prediction_policy": (
                "Rows indistinguishable by protocol/group/route/model-input/"
                "normalized-strain/MIC use their within-key mean saved prediction "
                "so censor-lineage occurrence order cannot affect metrics."
            ),
            "duplicate_key_rows": int(
                all_assignments["lineage_join_multiplicity"].gt(1).sum()
            ),
            "maximum_absolute_prediction_normalization_delta": float(
                all_assignments["prediction_duplicate_normalization_delta"].abs().max()
            ),
            "maximum_absolute_metric_normalization_delta": float(
                prediction_audit["delta_normalized_minus_saved"].abs().max()
            ),
        },
        "inputs": {name: hash_record(path) for name, path in input_paths.items()},
        "outputs": output_hashes,
        "claim_boundary": (
            "This analysis tests reported metrics against alternate evaluation-label "
            "encodings using frozen predictions. It does not test retraining under a "
            "censor-aware or alternatively imputed objective."
        ),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
