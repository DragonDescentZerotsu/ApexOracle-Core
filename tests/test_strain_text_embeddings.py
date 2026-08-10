from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from apexoracle.features.strain_text import (
    DEFAULT_MODEL_REVISION,
    decode_strain_name,
    embed_prepared_text,
    embed_strain_text_directory,
    prepare_strain_text,
    replace_strain_name,
)
from apexoracle.features.strain_text_cli import (
    build_parser,
    resolve_device,
    resolve_model_source,
)


class StubTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, *, return_tensors: str):
        assert return_tensors == "pt"
        self.texts.append(text)
        return {"input_ids": torch.tensor([[3, 4, 5]])}


class StubModel:
    def __init__(self) -> None:
        self.device = None
        self.training = True
        self.calls = 0

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, *, output_hidden_states: bool, use_cache: bool):
        assert output_hidden_states is True
        assert use_cache is False
        assert input_ids.shape == (1, 3)
        self.calls += 1
        hidden_states = (
            torch.zeros((1, 3, 2)),
            torch.arange(6, dtype=torch.float64).reshape(1, 3, 2),
            torch.full((1, 3, 2), 9.0),
        )
        return SimpleNamespace(hidden_states=hidden_states)


def test_filename_decoding_preserves_both_legacy_rules() -> None:
    assert (
        decode_strain_name("Escherichia_coli_ATCC_25922")
        == "Escherichia coli ATCC 25922"
    )
    assert (
        decode_strain_name("Lactobacillus～casei^paracasei～A1")
        == "Lactobacillus casei/paracasei A1"
    )


def test_two_pass_subspecies_replacement_is_exact() -> None:
    transformed, name, count = replace_strain_name(
        "Bacillus subtilis subsp. spizizenii ATCC 6633 is studied. "
        "Bacillus subtilis subsp spizizenii ATCC 6633 is archived.",
        "Bacillus_subtilis_subsp_spizizenii_ATCC_6633",
        "atcc",
    )
    assert name == "Bacillus subtilis subsp spizizenii ATCC 6633"
    assert transformed == "This strain is studied. This strain is archived."
    assert count == 2


def test_penultimate_layer_contract_and_float32_save_dtype(tmp_path: Path) -> None:
    source = tmp_path / "Klebsiella_pneumoniae_ATCC_1.txt"
    source.write_text(
        "Klebsiella pneumoniae ATCC 1 is a type strain.", encoding="utf-8"
    )
    prepared = prepare_strain_text(source)
    tokenizer = StubTokenizer()
    model = StubModel()
    tensor = embed_prepared_text(prepared, tokenizer, model, "cpu", -2)
    assert tokenizer.texts == ["This strain is a type strain."]
    assert torch.equal(tensor, torch.arange(6).reshape(3, 2).float())
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 2)


def test_directory_manifest_skip_and_error_policies(tmp_path: Path) -> None:
    input_dir = tmp_path / "text"
    output_dir = tmp_path / "embeddings"
    input_dir.mkdir()
    (input_dir / "B～strain.txt").write_text("B strain grows.", encoding="utf-8")
    (input_dir / "A_strain.txt").write_text("A strain grows.", encoding="utf-8")
    tokenizer = StubTokenizer()
    model = StubModel()

    first = embed_strain_text_directory(
        input_dir,
        output_dir,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
        existing_policy="error",
    )
    assert first["model"]["requested_revision"] == DEFAULT_MODEL_REVISION
    assert [record["source"]["path"] for record in first["records"]] == [
        "A_strain.txt",
        "B～strain.txt",
    ]
    assert first["summary"] == {
        "sources": 2,
        "written": 2,
        "skipped_existing": 0,
        "without_name_replacement": 0,
        "short_text_warnings": 0,
    }
    assert model.calls == 2
    before = {path.name: path.read_bytes() for path in output_dir.glob("*.pt")}

    skipped = embed_strain_text_directory(
        input_dir,
        output_dir,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
        existing_policy="skip",
    )
    assert skipped["summary"]["skipped_existing"] == 2
    assert model.calls == 2
    assert before == {path.name: path.read_bytes() for path in output_dir.glob("*.pt")}

    with pytest.raises(FileExistsError, match="A_strain.pt"):
        embed_strain_text_directory(
            input_dir,
            output_dir,
            tokenizer=tokenizer,
            model=model,
            device="cpu",
            existing_policy="error",
        )


def test_cli_contract_and_device_validation() -> None:
    args = build_parser().parse_args(
        ["--input-dir", "in", "--output-dir", "out", "--device", "cpu"]
    )
    assert args.revision == DEFAULT_MODEL_REVISION
    assert args.hidden_state_index == -2
    assert args.existing == "skip"
    assert resolve_device("cpu") == torch.device("cpu")


def test_local_snapshot_resolution_ignores_conflicting_cache_root(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-cache"
    available = tmp_path / "available-cache"
    snapshot = (
        available / "models--YBXL--Med-LLaMA3-8B" / "snapshots" / DEFAULT_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    assert resolve_model_source(
        "YBXL/Med-LLaMA3-8B",
        DEFAULT_MODEL_REVISION,
        local_files_only=True,
        cache_roots=[missing, available],
    ) == str(snapshot.resolve())


def test_local_snapshot_resolution_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pinned local snapshot not found"):
        resolve_model_source(
            "YBXL/Med-LLaMA3-8B",
            DEFAULT_MODEL_REVISION,
            local_files_only=True,
            cache_roots=[tmp_path],
        )
