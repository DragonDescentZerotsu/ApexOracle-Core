#!/usr/bin/env python3
"""Repository wrapper for the canonical strain-text embedding CLI."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.features.strain_text_cli import main  # noqa: E402


if __name__ == "__main__":
    main()
