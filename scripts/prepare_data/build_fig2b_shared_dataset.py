#!/usr/bin/env python3
"""Repository-local entry point for the revised Fig. 2b data protocol."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.protocol import main


if __name__ == "__main__":
    raise SystemExit(main())
