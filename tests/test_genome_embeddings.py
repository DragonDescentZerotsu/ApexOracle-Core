from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apexoracle.data.genome_embeddings import (
    fallback_atcc_id,
    genome_embedding_paths,
    manifest_identity,
    parse_embedding_id,
    resolve_genome_id,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Escherichia_coli_ATCC_25922.pt", "25922"),
        ("Escherichia_coli_ATCC_BAA_3170.pt", "BAA-3170"),
        ("custom.pt", "custom"),
    ],
)
def test_parse_embedding_id_preserves_legacy_mapping(name: str, expected: str) -> None:
    assert parse_embedding_id(name) == expected


def test_duplicate_parsed_ids_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "a_ATCC_1.pt").touch()
    (tmp_path / "b_ATCC_1.bin").touch()
    with pytest.raises(ValueError, match="Duplicate parsed genome ID"):
        genome_embedding_paths(tmp_path)


def test_resolve_prefers_exact_then_mapping_then_fallback() -> None:
    ids = {"custom", "BAA-3170", "25922"}
    assert resolve_genome_id("custom", ids, {}) == "custom"
    assert resolve_genome_id("alias", ids, {"alias": "BAA-3170"}) == "BAA-3170"
    assert resolve_genome_id("E. coli ATCC 25922", ids, {}) == "25922"
    assert fallback_atcc_id("E. coli") is None


def test_manifest_identity_is_deterministic() -> None:
    table = pd.DataFrame(
        [
            {
                "genome_id": "1",
                "file": "one.pt",
                "bytes": 10,
                "sha256": "a" * 64,
                "used_by_paper_datasets": True,
            }
        ]
    )
    assert manifest_identity(table) == manifest_identity(table.copy())
