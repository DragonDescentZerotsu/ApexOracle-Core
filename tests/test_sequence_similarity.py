from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from apexoracle.data.peptide_similarity import build_training_caches, classify_peptide
from apexoracle.evaluation.sequence_similarity.alignment import (
    build_aligner,
    compute_alignment,
    rotate_sequence,
    sanitize_sequence_for_scheme,
)
from apexoracle.evaluation.sequence_similarity.pipeline import (
    SimilarityOutputs,
    run_similarity,
)
from apexoracle.evaluation.sequence_similarity.reporting import (
    best_rows_by_metric,
    extract_top_hits,
    validate_rows,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_alignment_matches_saved_paper_examples() -> None:
    linear = compute_alignment(
        "MKILKKRRLVSVTLARDITTLENRL",
        "LKKLKGRVSRSFLFFVKLRPAKRTLKKRIL",
        "blosum62_needle",
    )
    assert linear.matches == 11
    assert linear.aligned_positions_including_gaps == 30
    assert linear.pid == pytest.approx(0.36666666666666664)

    cyclic = compute_alignment(
        rotate_sequence("VLKAAFHMRKLFRGHWVWW", 18),
        rotate_sequence("FPVKLKFPKVKL", 4),
        "blosum62_needle",
    )
    assert cyclic.matches == 7
    assert cyclic.aligned_positions_including_gaps == 19
    assert cyclic.pid == pytest.approx(0.3684210526315789)


def test_d_amino_acid_is_case_sensitive_and_unknowns_keep_case() -> None:
    assert sanitize_sequence_for_scheme("AkO?", "blosum62_needle") == "AkXX"
    assert sanitize_sequence_for_scheme("Ak?", "exact_match_needle") == "Ak?"
    aligner = build_aligner("blosum62_needle")
    assert aligner.substitution_matrix["K", "K"] > 0
    assert aligner.substitution_matrix["K", "k"] == 0
    assert compute_alignment("K", "k", "blosum62_needle").matches == 0


def test_peptide_cache_classification_preserves_legacy_rule() -> None:
    assert classify_peptide({"intrachainBonds": []}) == "linear"
    assert (
        classify_peptide({"intrachainBonds": [{"chainParticipating": {"name": "MMB"}}]})
        == "cyclic"
    )
    assert classify_peptide({"intrachainBonds": [{}, {}]}) == "skip"


def test_paper_cache_can_freeze_training_sequences_to_uppercase(tmp_path: Path) -> None:
    training_csv = tmp_path / "training.csv"
    peptides_json = tmp_path / "peptides.json"
    _write_csv(training_csv, ["DBAASP_id"], [{"DBAASP_id": "1"}])
    peptides_json.write_text(
        json.dumps({"data": [{"id": 1, "sequence": "fPk", "intrachainBonds": []}]}),
        encoding="utf-8",
    )
    linear = tmp_path / "linear.csv"
    manifest = build_training_caches(
        training_csv=training_csv,
        all_peptides_json=peptides_json,
        linear_output=linear,
        cyclic_output=tmp_path / "cyclic.csv",
        manifest_output=tmp_path / "manifest.json",
        sequence_case="uppercase",
    )
    assert list(csv.DictReader(linear.open()))[0]["sequence"] == "FPK"
    assert manifest["blosum62_normalization"]["sequence_case"] == "uppercase"


def test_complete_metric_tie_keeps_first_input_row() -> None:
    rows = [
        {
            "query_peptide_id": "lead",
            "scoring_scheme": "blosum62_needle",
            "train_dbaasp_id": dbaasp_id,
            "pid": "0.5",
            "max_len_identity": "0.5",
            "matches": "2",
        }
        for dbaasp_id in ("first", "second")
    ]
    assert best_rows_by_metric(rows, "pid")[0]["train_dbaasp_id"] == "first"


def test_small_end_to_end_pipeline_counts_order_and_validation(tmp_path: Path) -> None:
    query_csv = tmp_path / "queries.csv"
    linear_cache = tmp_path / "linear.csv"
    cyclic_cache = tmp_path / "cyclic.csv"
    _write_csv(
        query_csv,
        ["peptide_id", "cyclic", "sequence"],
        [
            {"peptide_id": "cyclic", "cyclic": "Yes", "sequence": "AK"},
            {"peptide_id": "linear", "cyclic": "No", "sequence": "AK"},
        ],
    )
    cache_fields = ["dbaasp_id", "sequence", "length"]
    _write_csv(
        linear_cache,
        cache_fields,
        [
            {"dbaasp_id": "1", "sequence": "AK", "length": 2},
            {"dbaasp_id": "2", "sequence": "AA", "length": 2},
        ],
    )
    _write_csv(
        cyclic_cache,
        cache_fields,
        [{"dbaasp_id": "3", "sequence": "KA", "length": 2}],
    )
    outputs = SimilarityOutputs(
        linear=tmp_path / "raw/linear.csv",
        cyclic_rotations=tmp_path / "raw/cyclic.csv",
        cyclic_best_pid=tmp_path / "raw/best_pid.csv",
        cyclic_best_max_len=tmp_path / "raw/best_max.csv",
        manifest=tmp_path / "reports/manifest.json",
    )
    manifest = run_similarity(
        query_csv=query_csv,
        linear_cache=linear_cache,
        cyclic_cache=cyclic_cache,
        outputs=outputs,
        processes=1,
        linear_chunk_size=1,
        cyclic_chunk_size=1,
    )
    assert manifest["counts"] == {
        "linear_queries": 1,
        "cyclic_queries": 1,
        "linear_training_records": 2,
        "cyclic_training_records": 1,
        "linear_rows_written": 4,
        "cyclic_rotation_rows_written": 8,
        "cyclic_best_pid_rows": 2,
        "cyclic_best_max_len_rows": 2,
    }
    with outputs.linear.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["train_dbaasp_id"], row["scoring_scheme"]) for row in rows] == [
        ("1", "blosum62_needle"),
        ("2", "blosum62_needle"),
        ("1", "exact_match_needle"),
        ("2", "exact_match_needle"),
    ]
    assert validate_rows(outputs.linear)["ok"]
    linear_summary = tmp_path / "reports/linear_top.csv"
    cyclic_summary = tmp_path / "reports/cyclic_top.csv"
    report = tmp_path / "reports/alignments.txt"
    linear_top, cyclic_top = extract_top_hits(
        linear_results=outputs.linear,
        cyclic_best_pid=outputs.cyclic_best_pid,
        cyclic_best_max_len=outputs.cyclic_best_max_len,
        linear_summary_output=linear_summary,
        cyclic_summary_output=cyclic_summary,
        alignment_report_output=report,
    )
    assert len(linear_top) == 4
    assert len(cyclic_top) == 4
    assert "Linear Top Hits" in report.read_text(encoding="utf-8")
    assert json.loads(outputs.manifest.read_text())["queries"] == [
        "cyclic",
        "linear",
    ]
