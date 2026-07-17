"""Portable feature-cache contract for frozen molecule encoders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .data import SharedBenchmarkData
from .protocol import PROTOCOL_VERSION


FEATURE_CACHE_VERSION = "fig2b-feature-cache-v1"


@dataclass(frozen=True)
class FeatureCache:
    encoder_name: str
    features: np.ndarray
    molecule_ids: tuple[str, ...]
    metadata: dict[str, Any]


def save_feature_cache(
    path: Path,
    *,
    encoder_name: str,
    molecule_ids: Sequence[str],
    features: np.ndarray,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a non-pickle NPZ cache with explicit protocol metadata."""

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("feature cache path must end in .npz")
    ids = tuple(str(value) for value in molecule_ids)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 2D; received shape {matrix.shape}")
    if matrix.shape[0] != len(ids):
        raise ValueError("feature row count does not match molecule ID count")
    if len(ids) != len(set(ids)):
        raise ValueError("molecule_ids contains duplicates")
    if not np.isfinite(matrix).all():
        raise ValueError("features contains NaN or infinite values")

    cache_metadata = dict(metadata or {})
    cache_metadata.update(
        {
            "cache_version": FEATURE_CACHE_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "encoder_name": encoder_name,
            "number_of_molecules": len(ids),
            "feature_dim": int(matrix.shape[1]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=matrix,
        molecule_ids=np.asarray(ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(cache_metadata, sort_keys=True)),
    )


def load_feature_cache(
    path: Path,
    benchmark: SharedBenchmarkData,
    *,
    expected_encoder: Optional[str] = None,
) -> FeatureCache:
    """Load, validate and reorder a cache to the canonical benchmark ID order."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"features", "molecule_ids", "metadata"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"feature cache is missing arrays: {sorted(missing)}")
        features = np.asarray(payload["features"], dtype=np.float32)
        ids = tuple(str(value) for value in payload["molecule_ids"].tolist())
        metadata = json.loads(str(payload["metadata"].item()))

    if metadata.get("cache_version") != FEATURE_CACHE_VERSION:
        raise ValueError("feature cache version is missing or incompatible")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("feature cache was not built for the frozen shared-data protocol")
    encoder_name = str(metadata.get("encoder_name", ""))
    if not encoder_name:
        raise ValueError("feature cache does not identify its encoder")
    if expected_encoder is not None and encoder_name != expected_encoder:
        raise ValueError(f"expected encoder {expected_encoder}, cache contains {encoder_name}")
    if features.ndim != 2 or features.shape[0] != len(ids):
        raise ValueError("feature cache matrix shape does not match its ID list")
    if len(ids) != len(set(ids)):
        raise ValueError("feature cache contains duplicate molecule IDs")
    if not np.isfinite(features).all():
        raise ValueError("feature cache contains NaN or infinite values")

    expected_ids = set(benchmark.molecule_ids)
    actual_ids = set(ids)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)[:10]
        extra_ids = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            "feature cache must contain exactly the shared molecule IDs: "
            f"missing={missing_ids}, extra={extra_ids}"
        )
    row_by_id = {molecule_id: index for index, molecule_id in enumerate(ids)}
    order = np.asarray([row_by_id[molecule_id] for molecule_id in benchmark.molecule_ids])
    aligned = np.ascontiguousarray(features[order], dtype=np.float32)
    return FeatureCache(
        encoder_name=encoder_name,
        features=aligned,
        molecule_ids=benchmark.molecule_ids,
        metadata=metadata,
    )
