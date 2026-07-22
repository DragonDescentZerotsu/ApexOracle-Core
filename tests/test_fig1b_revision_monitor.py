from __future__ import annotations

import json
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/reproduce/monitor_fig1b_revision.py"
)
SPEC = importlib.util.spec_from_file_location("fig1b_revision_monitor", SCRIPT)
assert SPEC and SPEC.loader
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_epoch_duration_parser_accepts_tqdm_formats() -> None:
    text = (
        "shared MIC/auxiliary training: 100%|x| [06:59<00:00]\n"
        "shared MIC/auxiliary training: 100%|x| [1:02:03<00:00]\n"
    )
    assert MONITOR._epoch_durations(text) == [419, 3723]


def test_expected_reconstruction_grid_has_45_unique_members() -> None:
    assert len(MONITOR.EXPECTED_TASKS) == 45
    assert "1:2:1" in MONITOR.EXPECTED_TASKS
    assert "2:4:9" in MONITOR.EXPECTED_TASKS
    assert "0:0:0" not in MONITOR.EXPECTED_TASKS
    assert MONITOR.FINE_TUNE_GRID_SIZE == 150
    assert MONITOR.AVAILABLE_BEFORE_CURRENT_RUN == 105


def test_baseline_completion_rejects_stale_20_member_result(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"ensemble_size": 20}), encoding="utf-8")
    assert MONITOR._baseline_is_complete(metrics) is False
    metrics.write_text(json.dumps({"ensemble_size": 10}), encoding="utf-8")
    assert MONITOR._baseline_is_complete(metrics) is True


def test_collect_reports_baseline_fold_identity_and_ignores_completed_log(
    tmp_path,
) -> None:
    fold = (
        tmp_path
        / "results/fig1b_revision/baselines_full_ensemble_no_rdkit/group_2/fold_1"
    )
    fold.mkdir(parents=True)
    (fold / "baseline_driver.log").write_text("finished\n", encoding="utf-8")
    (fold / "metrics.json").write_text(
        json.dumps({"ensemble_size": 10}), encoding="utf-8"
    )
    report = MONITOR.collect(tmp_path)
    assert report["baseline_complete_keys"] == ["2:1"]
    assert report["baseline_running_keys"] == []
