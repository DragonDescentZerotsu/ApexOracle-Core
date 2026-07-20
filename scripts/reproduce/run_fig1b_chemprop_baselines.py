#!/usr/bin/env python3
"""Prepare or run the common-fold Fig. 1b Chemprop baselines."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.benchmarks.fig1b_baselines import main  # noqa: E402


if __name__ == "__main__":
    main()
