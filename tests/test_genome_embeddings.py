from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from apexoracle.data.genome_embeddings import (
    build_paper_genome_list,
    fasta_source,
    fallback_atcc_id,
    genome_embedding_paths,
    manifest_identity,
    parse_embedding_id,
    parse_species_label,
    resolve_genome_id,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_species_and_source_labels_do_not_invent_accessions() -> None:
    assert parse_species_label("Escherichia_coli_ATCC_25922.pt") == "Escherichia coli"
    assert fasta_source("25922") == ("ATCC FASTA archive", "ATCC 25922")
    assert fasta_source("#001") == (
        "NCBI RefSeq assembly",
        "GCF_000212715.2",
    )
    assert fasta_source("#003") == ("project custom FASTA", "not_recovered")


def test_paper_genome_list_uses_only_task_union(tmp_path: Path) -> None:
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    (fasta_dir / "Escherichia_coli_ATCC_1.fasta").write_text(
        ">record\nACGT\n", encoding="utf-8"
    )
    manifest = pd.DataFrame(
        [
            {
                "genome_id": "1",
                "file": "Escherichia_coli_ATCC_1.pt",
                "bytes": 10,
                "sha256": "a" * 64,
                "used_by_paper_datasets": True,
            },
            {
                "genome_id": "2",
                "file": "Escherichia_coli_ATCC_2.pt",
                "bytes": 20,
                "sha256": "b" * 64,
                "used_by_paper_datasets": False,
            },
        ]
    )
    table = build_paper_genome_list(
        manifest,
        fasta_dir,
        {
            "used_by_classification": {"1"},
            "used_by_mic": {"1"},
            "used_by_synergy": set(),
        },
    )
    assert list(table["genome_id"]) == ["1"]
    assert table.loc[0, "species_label"] == "Escherichia coli"
    assert bool(table.loc[0, "used_by_mic"])
    assert not bool(table.loc[0, "used_by_synergy"])


def test_released_paper_genome_list_matches_manifest() -> None:
    csv_path = ROOT / "experiments/evo2_genome_embeddings/paper_genome_list.csv"
    manifest = json.loads(
        (
            ROOT / "experiments/evo2_genome_embeddings/paper_genome_list_manifest.json"
        ).read_text(encoding="utf-8")
    )
    table = pd.read_csv(csv_path, dtype={"genome_id": str})
    assert len(table) == table["genome_id"].nunique() == 563
    assert int(table["used_by_mic"].sum()) == 563
    assert int(table["used_by_classification"].sum()) == 2
    assert int(table["used_by_synergy"].sum()) == 100
    assert (
        hashlib.sha256(csv_path.read_bytes()).hexdigest()
        == manifest["paper_genome_list"]["sha256"]
    )
    assert not any(
        value.startswith(("/data", "/home"))
        for column in table.select_dtypes(include="object")
        for value in table[column].dropna().astype(str)
    )
