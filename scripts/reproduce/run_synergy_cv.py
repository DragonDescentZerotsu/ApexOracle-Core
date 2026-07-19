#!/usr/bin/env python3
"""Stable entrypoint for the experimental synergy three-fold CV candidate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.training.synergy_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
