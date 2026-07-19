#!/usr/bin/env python3
"""CLI wrapper for the prospective synergy regression producer."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apexoracle.training.synergy_regression_runner import main


if __name__ == "__main__":
    main()
