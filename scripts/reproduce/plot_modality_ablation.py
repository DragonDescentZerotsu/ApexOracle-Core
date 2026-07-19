#!/usr/bin/env python3
"""Stable wrapper for the frozen paper modality-ablation figure."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.evaluation.modality_ablation import main  # noqa: E402


if __name__ == "__main__":
    main()
