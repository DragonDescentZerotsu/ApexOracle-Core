from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from apexoracle.features.kmer import (
    LegacyFrozenKmerProjection,
    extract_folder,
    global_kmer_embedding,
    load_legacy_projected_embeddings,
    windowed_kmer_embedding,
)
from apexoracle.training.hierarchical_mic import (
    legacy_hierarchical_checkpoint_payload,
)
from apexoracle.training.hierarchical_mic_runner import HierarchicalMicConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_global_counts_do_not_cross_contig_boundaries() -> None:
    values = global_kmer_embedding(["ACGTN", "AC"], (1, 2))
    assert values.shape == (1, 20)
    torch.testing.assert_close(
        values[0, :4], torch.tensor([2 / 6, 2 / 6, 1 / 6, 1 / 6])
    )
    expected_pairs = torch.zeros(16)
    expected_pairs[[1, 6, 11]] = torch.tensor([0.5, 0.25, 0.25])
    torch.testing.assert_close(values[0, 4:], expected_pairs)


def test_window_offsets_reset_for_each_fasta_record() -> None:
    values = windowed_kmer_embedding(
        ["AAAAAA", "CCCC"],
        (1,),
        window_length=4,
        step_length=3,
    )
    assert values.shape == (4, 4)
    torch.testing.assert_close(values.sum(dim=1), torch.ones(4))
    torch.testing.assert_close(values[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # The windowed legacy alphabet is A,T,G,C, so C occupies the last bin.
    torch.testing.assert_close(values[-1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_extractor_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    (genomes / "one.fasta").write_text(">one\nACGTACGT\n", encoding="utf-8")
    output = tmp_path / "features"
    paths = extract_folder(genomes, output, mode="global", k_values=(1,))
    assert len(paths) == 1
    values = torch.load(paths[0], map_location="cpu", weights_only=True)
    assert values.shape == (1, 4)
    assert values.dtype == torch.bfloat16
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extract_folder(genomes, output, mode="global", k_values=(1,))


def test_frozen_projection_matches_inline_reconstruction() -> None:
    torch.manual_seed(17)
    shared = LegacyFrozenKmerProjection(4, 8)
    torch.manual_seed(17)
    inline = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 8))
    for module in inline:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    for key, value in shared.projection.state_dict().items():
        torch.testing.assert_close(value, inline.state_dict()[key])
    assert all(not parameter.requires_grad for parameter in shared.parameters())


def test_projected_loader_and_checkpoint_keep_projection_state(tmp_path: Path) -> None:
    torch.save(torch.ones(1, 4), tmp_path / "custom.pt")
    torch.manual_seed(23)
    embeddings, projection = load_legacy_projected_embeddings(
        tmp_path,
        input_dim=4,
        output_dim=8,
        scale=1.0,
        device=torch.device("cpu"),
    )
    assert embeddings["custom"].shape == (1, 8)
    assert embeddings["custom"].dtype == torch.bfloat16

    genome_attention = nn.Linear(2, 2)
    text_attention = nn.Linear(2, 2)
    regression = nn.Linear(2, 1)
    classification = nn.Linear(2, 1)
    missing = nn.Parameter(torch.zeros(1, 2))
    optimizer = torch.optim.Adam(genome_attention.parameters())
    payload = legacy_hierarchical_checkpoint_payload(
        r2=0.1,
        optimizer=optimizer,
        regression_head=regression,
        classification_head=classification,
        genome_attention=genome_attention,
        text_attention=text_attention,
        missing_genome_embedding=missing,
        genome_embedding_adapter=projection,
    )
    assert payload["kmer_projection_state_dict"].keys() == projection.state_dict().keys()


def test_kmer_reconstruction_profile_is_strain_only(monkeypatch) -> None:
    embedding_dir = Path("/data/fangping/kmer_embeddings/k456_global")
    monkeypatch.setenv("APEXORACLE_KMER_EMBEDDINGS_DIR", str(embedding_dir))
    config = HierarchicalMicConfig.load(
        REPO_ROOT / "configs/hierarchical_mic/legacy_kmer_reconstruction.yaml",
        REPO_ROOT,
        holdout_protocol="strain",
    )
    assert config.ensembles_per_group == 7
    assert config.epochs == 25
    assert config.genome_embedding_scale == 1.0
    assert config.genome_embedding_adapter is not None
    assert config.genome_embedding_adapter.input_dim == 5_376
    assert config.genome_embedding_adapter.output_dim == 8_192
    assert config.paths.genome_embeddings == embedding_dir
