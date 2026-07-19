from __future__ import annotations

import pandas as pd
import torch

from apexoracle.data.synergy import (
    SYNERGY_COLUMNS,
    exclude_wrong_species_mappings,
    filter_synergy_token_lengths,
    synergy_label,
)
from apexoracle.data.synergy_dataset import (
    SynergyPairDataset,
    collate_synergy_genome_text,
    collate_synergy_text_only,
)


def test_synergy_label_preserves_strict_threshold_and_direction() -> None:
    assert synergy_label(0.49) == 1.0
    assert synergy_label(0.5) == 0.0
    assert synergy_label(1.0) == 0.0


def test_wrong_species_filter_preserves_unmapped_and_matching_rows() -> None:
    rows = [
        (1, "a", "Escherichia coli ATCC 1", "C", "N", 0.2),
        (2, "a", "Staphylococcus aureus ATCC 1", "C", "N", 0.8),
        (3, "a", "custom strain", "C", "N", 0.4),
        (4, "a", "Unknown species ATCC 2", "C", "N", 0.4),
    ]
    table = pd.DataFrame(rows, columns=SYNERGY_COLUMNS)
    result = exclude_wrong_species_mappings(table, {"1": "Escherichia"})
    assert result["DBAASP_id"].tolist() == [1, 3, 4]


class _LengthTokenizer:
    def __call__(self, value, **kwargs):
        del kwargs
        return {"input_ids": torch.arange(int(value))[None, :]}


def test_pair_token_filter_checks_both_columns_and_preserves_rows() -> None:
    table = pd.DataFrame(
        [
            ["a", "b", "s1", "2", "3", 0.25],
            ["c", "d", "s2", "5", "2", 0.75],
            ["e", "f", "s3", "2", "6", 0.5],
        ],
        columns=SYNERGY_COLUMNS,
    )
    cache = {}
    result = filter_synergy_token_lengths(
        table,
        tokenizer=_LengthTokenizer(),
        selfies_encoder=lambda value: value,
        max_length=4,
        length_cache=cache,
    )
    assert result.original_rows == 3
    assert result.retained_rows == 1
    assert result.table["strain_name"].tolist() == ["s1"]
    assert set(cache) == {"2", "3", "5", "6"}


def test_pair_dataset_and_collators_preserve_legacy_duplication_contract() -> None:
    table = pd.DataFrame(
        [["a", "b", "s1", "C", "N", 0.49]], columns=SYNERGY_COLUMNS
    )
    molecule = {
        "a": torch.arange(6, dtype=torch.float32)[None, :],
        "b": (torch.arange(6, dtype=torch.float32) + 10)[None, :],
    }
    genome = {"s1": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}
    text = {"s1": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)}
    dataset = SynergyPairDataset(
        table,
        molecule_embeddings=molecule,
        genome_embeddings=genome,
        text_embeddings=text,
    )
    batch = collate_synergy_genome_text([dataset[0]])
    assert batch["label"].tolist() == [1.0]
    assert batch["mol_emb"].shape == (2, 6)
    torch.testing.assert_close(batch["mol_emb"][0], molecule["a"].squeeze())
    torch.testing.assert_close(batch["mol_emb"][1], molecule["b"].squeeze())
    assert batch["padded_genome_embeddings"].shape == (2, 2, 4)
    assert batch["padded_text_embeddings"].shape == (2, 3, 4)
    assert batch["genome_attn_masks"].tolist() == [[1, 1], [1, 1]]
    assert batch["strain_names"] == ["s1"]

    text_dataset = SynergyPairDataset(
        table,
        molecule_embeddings=molecule,
        text_embeddings=text,
    )
    text_batch = collate_synergy_text_only([text_dataset[0]])
    assert "padded_genome_embeddings" not in text_batch
    torch.testing.assert_close(text_batch["mol_emb"], batch["mol_emb"])
