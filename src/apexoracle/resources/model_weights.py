"""Resolve local model binaries through the canonical weight manifest."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml


DEFAULT_MANIFEST = Path("configs/model_weights.yaml")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weight(
    weight_id: str,
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    verify: bool = True,
) -> Path:
    """Resolve one local weight by manifest ID and optionally verify its identity.

    ``APEXORACLE_WEIGHTS_DIR`` overrides the repository-local ``weights`` root.
    If it is unset, the manifest's current path is preferred and the declared
    future relative path is used as a fallback.
    """

    root = Path(repo_root).resolve() if repo_root is not None else repository_root()
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / DEFAULT_MANIFEST
    )
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    try:
        entry = manifest["weights"][weight_id]
    except KeyError as exc:
        raise KeyError(f"Unknown weight manifest ID: {weight_id}") from exc

    relative = Path(entry["future_storage"]["relative_path"])
    override = os.environ.get("APEXORACLE_WEIGHTS_DIR")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser() / relative)
    else:
        current = Path(entry["source"]["current_path"])
        candidates.append(current if current.is_absolute() else root / current)
        default_root = Path(manifest["storage_policy"]["default_local_root"])
        candidates.append(root / default_root / relative)

    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        locations = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"Weight {weight_id!r} is not installed. Checked:\n{locations}"
        )

    if verify:
        expected_size = int(entry["source"]["size_bytes"])
        expected_sha = str(entry["source"]["sha256"])
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Weight {weight_id!r} has size {path.stat().st_size}; "
                f"expected {expected_size}: {path}"
            )
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Weight {weight_id!r} has SHA-256 {actual_sha}; "
                f"expected {expected_sha}: {path}"
            )
    return path
