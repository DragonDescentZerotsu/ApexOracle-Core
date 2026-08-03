from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit"
    / "compute_exact_generation_tanimoto.py"
)
SPEC = importlib.util.spec_from_file_location("exact_generation_tanimoto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_ranges_cover_each_query_once() -> None:
    ranges = MODULE.balanced_ranges(100, 7)
    covered = [row for start, stop in ranges for row in range(start, stop)]
    assert covered == list(range(99))
    pair_count = sum(
        MODULE.cumulative_pairs_before(stop, 100)
        - MODULE.cumulative_pairs_before(start, 100)
        for start, stop in ranges
    )
    assert pair_count == 100 * 99 // 2


def test_histogram_quantile_reports_containing_bin() -> None:
    histogram = np.zeros(1_000, dtype=np.int64)
    histogram[100] = 2
    histogram[500] = 3
    result = MODULE.histogram_quantile(histogram, 0.5)
    assert result == {"estimate": 0.5005, "lower": 0.5, "upper": 0.501}
