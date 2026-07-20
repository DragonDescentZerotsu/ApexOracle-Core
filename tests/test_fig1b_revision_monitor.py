from __future__ import annotations

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
