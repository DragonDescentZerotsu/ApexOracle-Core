"""Protocol guards for the frozen ReMDM reviewer experiment."""

from __future__ import annotations

import json
import importlib.util
import statistics
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "experiments" / "remasking_schedule_reviewer" / "task_manifest.json"
)
SUMMARY = (
    ROOT
    / "experiments"
    / "remasking_schedule_reviewer"
    / "analysis"
    / "summary.json"
)
PLOT_SCRIPT = (
    ROOT / "scripts" / "audit" / "plot_remasking_schedule_reviewer.py"
)
STRUCTURE_PLOT_SCRIPT = (
    ROOT
    / "scripts"
    / "audit"
    / "plot_remasking_structure_qualified_peptides.py"
)
STRUCTURE_AUDIT_SUMMARY = (
    ROOT
    / "experiments"
    / "remasking_schedule_reviewer"
    / "analysis"
    / "peptide_structure_audit"
    / "summary.json"
)
EVALUATED_ATTEMPTS = (
    ROOT
    / "experiments"
    / "remasking_schedule_reviewer"
    / "analysis"
    / "evaluated_attempts.csv"
)
EVALUATOR_SCRIPT = (
    ROOT / "scripts" / "reproduce" / "evaluate_remasking_schedule_reviewer.py"
)


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_remasking_schedule_reviewer", PLOT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_structure_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_remasking_structure_qualified_peptides",
        STRUCTURE_PLOT_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remasking_protocol_has_only_predeclared_conditions() -> None:
    manifest = load_manifest()
    conditions = {
        condition["name"]: (
            condition["t_on"],
            condition["t_off"],
            condition["gamma_peptide"],
        )
        for condition in manifest["protocol"]["conditions"]
    }
    assert conditions == {
        "earlier": (0.75, 0.65, 15.0),
        "current": (0.55, 0.45, 15.0),
        "later": (0.35, 0.25, 15.0),
        "narrower": (0.525, 0.475, 15.0),
        "wider": (0.55, 0.25, 15.0),
        "no_peptide_correction": (0.55, 0.45, 0.0),
    }
    assert manifest["protocol"]["seeds"] == [20260728, 20260729, 20260730]
    assert manifest["protocol"]["total_tasks"] == 36
    assert manifest["protocol"]["total_attempted_samples"] == 3600


def test_remasking_tasks_cover_full_factorial_once() -> None:
    tasks = load_manifest()["tasks"]
    keys = [
        (task["condition"], task["strain"], task["seed"]) for task in tasks
    ]
    assert len(keys) == 36
    assert len(set(keys)) == 36
    assert Counter(task["condition"] for task in tasks) == {
        "earlier": 6,
        "current": 6,
        "later": 6,
        "narrower": 6,
        "wider": 6,
        "no_peptide_correction": 6,
    }
    assert Counter(task["strain"] for task in tasks) == {
        "BAA-3170": 18,
        "BAA-3197": 18,
    }
    assert all(task["attempted_samples"] == 100 for task in tasks)
    assert all(task["batch_size"] == 25 for task in tasks)
    assert all(task["num_batches"] == 4 for task in tasks)


def test_remasking_allocation_balances_every_condition_across_hosts() -> None:
    tasks = load_manifest()["tasks"]
    for condition in {
        task["condition"] for task in tasks
    }:
        condition_tasks = [
            task for task in tasks if task["condition"] == condition
        ]
        assert Counter(task["host"] for task in condition_tasks) == {
            "local": 2,
            "node002": 4,
        }
        for strain in ("BAA-3170", "BAA-3197"):
            strain_hosts = {
                task["host"]
                for task in condition_tasks
                if task["strain"] == strain
            }
            assert strain_hosts == {"local", "node002"}
    assert Counter(
        (task["host"], task["gpu"]) for task in tasks
    ) == {
        ("local", 0): 3,
        ("local", 1): 3,
        ("local", 2): 3,
        ("local", 3): 3,
        ("node002", 0): 3,
        ("node002", 1): 3,
        ("node002", 2): 3,
        ("node002", 3): 3,
        ("node002", 4): 3,
        ("node002", 5): 3,
        ("node002", 6): 3,
        ("node002", 7): 3,
    }


def test_reviewer_figure_seed_error_bars_use_three_balanced_seed_rates() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    module = load_plot_module()
    seed_yields = module.pooled_seed_yields(
        summary["condition_strain_seed"]
    )
    assert seed_yields["current"] == {
        "20260728": 34.0,
        "20260729": 36.5,
        "20260730": 36.0,
    }
    assert all(len(values) == 3 for values in seed_yields.values())


def test_reviewer_figure_control_tests_are_exact_task_paired_tests() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    module = load_plot_module()
    pvalues = module.paired_control_pvalues(
        summary["condition_strain_seed"]
    )
    assert pvalues == {
        "classifier_positive": 1.0,
        "rdkit_valid": 0.1875,
        "valid_peptide_yield": 0.59375,
    }


def test_structure_audit_records_classifier_and_amide_disagreement() -> None:
    summary = json.loads(
        STRUCTURE_AUDIT_SUMMARY.read_text(encoding="utf-8")
    )
    assert summary["checkpoint_identity"][
        "same_v1_checkpoint_file_used_as_reviewer_retrain_backbone"
    ]
    assert (
        summary["checkpoint_identity"][
            "saved_probability_max_abs_difference"
        ]
        < 1e-5
    )
    current = summary["by_condition"]["current"]
    assert current["rdkit_valid_count"] == 395
    assert current["rdkit_general_amide_positive_count"] == 104
    assert current["original_classifier"]["positive_count"] == 213
    assert current["original_classifier"]["without_general_amide_count"] == 125
    assert current["sep_padded_classifier"]["positive_count"] == 191
    assert (
        current["reviewer_retrained_ensemble"]["positive_count"] == 220
    )


def test_narrow_structure_plot_element_policy_is_explicit() -> None:
    module = load_structure_plot_module()
    assert {"B", "F", "Cl", "Br", "I"} <= module.ALLOWED_ELEMENTS
    assert module.ALLOWED_ELEMENTS.isdisjoint(module.MANUAL_REVIEW_METALS)
    assert "Fe" in module.MANUAL_REVIEW_METALS
    assert "Pu" not in module.MANUAL_REVIEW_METALS


def test_mic_error_bar_uses_three_seed_pooled_median_sd() -> None:
    if not EVALUATED_ATTEMPTS.is_file():
        pytest.skip("ignored evaluated-attempt rows are not installed")
    module = load_structure_plot_module()
    rows = [
        {"condition": "current", "seed": str(seed)}
        for seed in (20260728, 20260729, 20260730)
    ]
    module.add_seed_mic_metrics(rows, EVALUATED_ATTEMPTS)
    assert [row["valid_predicted_mic_n"] for row in rows] == [133, 131, 131]
    medians = [row["valid_predicted_mic_median_uM"] for row in rows]
    assert abs(statistics.stdev(medians) - 6.337521397261419) < 1e-12


def test_mic_evaluator_uses_canonical_mdlm_scoring_api() -> None:
    source = EVALUATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'import_module("judge_generated_mols_MIC")' not in source
    assert "load_candidate_mic_regressor" in source
    assert "load_condition_embedding_banks" in source
    assert 'mdlm_source_root = mdlm_root / "src"' in source
    assert "runtime_root=mdlm_root" in source
