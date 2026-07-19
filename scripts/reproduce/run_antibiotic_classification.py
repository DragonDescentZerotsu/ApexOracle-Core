#!/usr/bin/env python3
"""Stable entrypoint for the three Fig. 1b antibiotic-classification modes."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.training.antibiotic_classification_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
