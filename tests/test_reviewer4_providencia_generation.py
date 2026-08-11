from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / "reproduce" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_historical_text_normalization_replaces_exact_strain_name():
    module = load_script("prepare_reviewer4_providencia_assets.py")
    normalized = module.normalize_historical_text(
        "Providencia stuartii ATCC 29914 is a type strain."
    )
    assert normalized == "This strain is a type strain."


def test_fasta_validator_rejects_multiple_contigs(tmp_path: Path):
    module = load_script("prepare_reviewer4_providencia_assets.py")
    fasta = tmp_path / "wrong.fasta"
    fasta.write_text(">one\nACGT\n>two\nACGT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="single-contig"):
        module.validate_fasta(fasta, module.EXPECTED_LENGTH)


def test_genbank_validator_records_annotation_contract(tmp_path: Path):
    module = load_script("prepare_reviewer4_providencia_assets.py")
    module.EXPECTED_LENGTH = 4
    genbank = tmp_path / "target.gbk"
    record = SeqRecord(Seq("ACGT"), id="annotation-1", description="target")
    record.annotations["molecule_type"] = "DNA"
    SeqIO.write([record], genbank, "genbank")
    metadata, sequence = module.validate_genbank(genbank, tolerance=0)
    assert metadata["records"] == 1
    assert metadata["total_length_bp"] == 4
    assert sequence == "ACGT"


def test_generation_manifest_freezes_full_historical_length_grid(tmp_path, monkeypatch):
    module = load_script("prepare_reviewer4_providencia_generation_tasks.py")
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        ["prepare", "--output", str(output), "--batch-size", "2", "--num-batches", "3"],
    )
    module.main()
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["length_grid"] == list(range(256, 417, 4))
    assert len(manifest["tasks"]) == 41
    assert manifest["attempted_samples_total"] == 41 * 6
    assert {task["strain"] for task in manifest["tasks"]} == {"29914"}
    assert {task["gamma_mic"] for task in manifest["tasks"]} == {15.0}


def test_strict_filter_is_inclusive_and_requires_peplink_standard():
    module = load_script("filter_reviewer4_providencia_candidates.py")

    class FakePepLink:
        @staticmethod
        def smiles_to_aa_seqs(smiles):
            if smiles == "standard":
                return SimpleNamespace(
                    sequence="AA", cyclization="linear", unsupported_reason=None
                )
            return SimpleNamespace(
                sequence=None,
                cyclization=None,
                unsupported_reason="unsupported",
            )

    accepted = module.annotate_row(
        {
            "rdkit_valid": "True",
            "canonical_smiles": "standard",
            "predicted_mic_uM": "15.0",
        },
        FakePepLink,
        15.0,
    )
    rejected_mic = module.annotate_row(
        {
            "rdkit_valid": "True",
            "canonical_smiles": "standard",
            "predicted_mic_uM": "15.0001",
        },
        FakePepLink,
        15.0,
    )
    rejected_structure = module.annotate_row(
        {
            "rdkit_valid": "True",
            "canonical_smiles": "other",
            "predicted_mic_uM": "1",
        },
        FakePepLink,
        15.0,
    )
    assert accepted["strict_candidate"] is True
    assert rejected_mic["strict_candidate"] is False
    assert rejected_structure["strict_candidate"] is False


def test_filter_materializes_every_tier(tmp_path: Path, monkeypatch):
    module = load_script("filter_reviewer4_providencia_candidates.py")

    class FakePepLink:
        __version__ = "0.1.2"

        @staticmethod
        def smiles_to_aa_seqs(smiles):
            return SimpleNamespace(
                sequence="AA", cyclization="linear", unsupported_reason=None
            )

    source = tmp_path / "evaluated.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "complete",
                "rdkit_valid",
                "has_amide_bond",
                "canonical_smiles",
                "predicted_mic_uM",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "complete": True,
                "rdkit_valid": True,
                "has_amide_bond": True,
                "canonical_smiles": "standard",
                "predicted_mic_uM": 15,
            }
        )
    output = tmp_path / "filters"
    monkeypatch.setitem(sys.modules, "PepLink", FakePepLink)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter",
            "--evaluated-attempts",
            str(source),
            "--output-dir",
            str(output),
        ],
    )
    module.main()
    manifest = json.loads((output / "filter_manifest.json").read_text())
    assert manifest["tier_counts"]["strict_candidate"] == 1
    assert len(list((output / "tiers").glob("*.csv"))) == 8
    assert (output / "tiers" / "07_strict_candidate.csv").exists()
