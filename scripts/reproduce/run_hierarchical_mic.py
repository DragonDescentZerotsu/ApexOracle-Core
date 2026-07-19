#!/usr/bin/env python3
"""Single stable entrypoint for strain-, species-, and phylum-wise MIC runs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.training.hierarchical_mic_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
