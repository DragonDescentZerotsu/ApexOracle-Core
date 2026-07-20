"""Reference-asset paths for molecule-encoder benchmarks."""

from __future__ import annotations

from pathlib import Path


APEX_AAINDEX_RELATIVE_PATH = Path("resources/reference/apex/aaindex1.csv")


def apex_aaindex_path(repo_root: Path | None = None) -> Path:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    path = root / APEX_AAINDEX_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"APEX AAindex asset is not installed: {path}. "
            "See resources/reference/apex/README.md."
        )
    return path
