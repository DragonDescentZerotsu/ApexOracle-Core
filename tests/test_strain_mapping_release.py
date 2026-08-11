import json
from pathlib import Path

import pandas as pd

from apexoracle.data.strain_mapping_release import (
    build_paper_strain_mapping,
    write_paper_strain_mapping,
)


def _touch(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).touch()


def test_build_paper_strain_mapping_routes_and_filters(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    _touch(data_root / "Genome_embs", "Species_one_ATCC_123.pt")
    _touch(data_root / "Genome_embs", "Species_two_ATCC_BAA_456.pt")
    _touch(data_root / "Genome" / "ATCC", "Species_one_ATCC_123.fasta")
    _touch(data_root / "Genome" / "ATCC", "Species_two_ATCC_BAA_456.fasta")
    _touch(
        data_root / "Text_Description" / "ATCC" / "embeddings",
        "Species_one_ATCC_123.pt",
    )
    _touch(
        data_root / "Text_Description" / "ATCC" / "embeddings",
        "Species_two_ATCC_BAA_456.pt",
    )
    _touch(
        data_root / "Text_Description" / "wo_ATCC" / "embeddings",
        "Species～three～strain.pt",
    )
    mapping = {
        "Species one alias*ATCC 123": 2,
        "Species two ATCC BAA 456": 1,
        "Wrong genus ATCC BAA 456": 1,
    }
    (
        data_root
        / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json"
    ).write_text(json.dumps(mapping), encoding="utf-8")
    pd.DataFrame(
        {
            "strain_name": [
                "Species one alias",
                "Species one alias",
                "Species two ATCC BAA 456",
                "Wrong genus ATCC BAA 456",
                "Species three strain",
                "Unknown species",
            ]
        }
    ).to_csv(data_root / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv", index=False)

    document = build_paper_strain_mapping(data_root)
    assert document["summary"] == {
        "source_strain_count": 5,
        "mapped_source_strain_count": 3,
        "mapping_row_count": 3,
        "mapped_mic_route_record_count": 4,
        "route_counts": {"genome_text": 2, "text_only": 1},
    }
    records = document["records"]
    assert [record["source_strain_name"] for record in records] == [
        "Species one alias",
        "Species three strain",
        "Species two ATCC BAA 456",
    ]
    assert records[0]["canonical_strain_id"] == "123"
    assert records[0]["mapping_reason"] == "handcrafted_alias"
    assert records[1]["condition_route"] == "text_only"
    assert records[2]["canonical_strain_id"] == "BAA-456"

    output = tmp_path / "mapping.json"
    write_paper_strain_mapping(document, output)
    assert json.loads(output.read_text(encoding="utf-8")) == document
