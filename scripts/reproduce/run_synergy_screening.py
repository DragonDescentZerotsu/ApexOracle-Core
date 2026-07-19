#!/usr/bin/env python3
"""Run the canonical prospective synergy screening entrypoint."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.evaluation.synergy_screening import main  # noqa: E402


if __name__ == "__main__":
    main()
