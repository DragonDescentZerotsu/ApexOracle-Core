"""Datasets and collators for the paper-era synergy pair classifier."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .synergy import SYNERGY_COLUMNS, synergy_label


class SynergyPairDataset(Dataset):
    """Read paired precomputed molecule features and strain embeddings."""

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        molecule_embeddings: Mapping,
        text_embeddings: Mapping[str, torch.Tensor],
        genome_embeddings: Mapping[str, torch.Tensor] | None = None,
        target_transform: Callable[[float], float] = synergy_label,
    ) -> None:
        if tuple(table.columns) != SYNERGY_COLUMNS:
            raise ValueError(f"Unexpected synergy columns: {tuple(table.columns)}")
        self.table = table.reset_index(drop=True).copy()
        self.molecule_embeddings = molecule_embeddings
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        strain = row["strain_name"]
        item = {
            "label": torch.tensor(
                self.target_transform(row["FICI"]), dtype=torch.float
            ),
            "text_embedding": self.text_embeddings[strain],
            "strain_name": strain,
            "pair_key": (
                row["DBAASP_id"],
                row["antibio_id_or_name"],
                strain,
            ),
            "mol_emb_1": self.molecule_embeddings[row["DBAASP_id"]].squeeze(),
            "mol_emb_2": self.molecule_embeddings[
                row["antibio_id_or_name"]
            ].squeeze(),
        }
        if self.genome_embeddings is not None:
            item["genome_embedding"] = self.genome_embeddings[strain]
        return item


class TokenizedSynergyPairDataset(Dataset):
    """Online SELFIES-tokenized pairs used by the all-data guidance classifier."""

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        tokenizer: Any,
        selfies_encoder: Callable[[str], str],
        text_embeddings: Mapping[str, torch.Tensor],
        genome_embeddings: Mapping[str, torch.Tensor] | None = None,
        max_length: int = 512,
    ) -> None:
        if tuple(table.columns) != SYNERGY_COLUMNS:
            raise ValueError(f"Unexpected synergy columns: {tuple(table.columns)}")
        self.tokenizer = tokenizer
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings
        frame = table.reset_index(drop=True).copy()

        def tokenize(smiles: str) -> torch.Tensor:
            encoded = selfies_encoder(str(smiles)).replace("][", "] [")
            return tokenizer(
                encoded,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )["input_ids"].squeeze(0)

        frame["input_ids_1"] = frame["AMP_smiles"].map(tokenize)
        frame["input_ids_2"] = frame["antibiotic_smiles"].map(tokenize)
        frame = frame.loc[
            frame["input_ids_1"].map(len).le(max_length)
            & frame["input_ids_2"].map(len).le(max_length)
        ].reset_index(drop=True)
        self.original_rows = len(table)
        self.table = frame

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        strain = row["strain_name"]
        item = {
            "input_ids_1": row["input_ids_1"],
            "input_ids_2": row["input_ids_2"],
            "label": torch.tensor(synergy_label(row["FICI"]), dtype=torch.float),
            "text_embedding": self.text_embeddings[strain],
            "strain_name": strain,
            "pair_key": (
                row["DBAASP_id"],
                row["antibio_id_or_name"],
                strain,
            ),
        }
        if self.genome_embeddings is not None:
            item["genome_embedding"] = self.genome_embeddings[strain]
        return item


def _pad_duplicated_embeddings(embeddings: list[torch.Tensor]) -> tuple:
    duplicated = [embedding for item in embeddings for embedding in (item, item)]
    max_length = max(len(embedding) for embedding in duplicated)
    padded = []
    masks = []
    for embedding in duplicated:
        length, width = embedding.shape
        mask = torch.zeros(
            max_length, device=embedding.device, dtype=torch.uint8
        )
        values = torch.zeros(
            (max_length, width), dtype=torch.bfloat16, device=embedding.device
        )
        values[:length] = embedding
        mask[:length] = 1
        padded.append(values)
        masks.append(mask)
    return torch.stack(padded), torch.stack(masks)


def _common_pair_batch(batch: list[dict]) -> dict:
    labels = torch.from_numpy(np.array([item["label"] for item in batch]))
    molecule_embeddings = torch.stack(
        [
            embedding
            for item in batch
            for embedding in (item["mol_emb_1"], item["mol_emb_2"])
        ]
    )
    padded_text, text_masks = _pad_duplicated_embeddings(
        [item["text_embedding"] for item in batch]
    )
    return {
        "label": labels,
        "padded_text_embeddings": padded_text,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "pair_keys": [item["pair_key"] for item in batch],
        "mol_emb": molecule_embeddings,
    }


def collate_synergy_genome_text(batch: list[dict]) -> dict:
    result = _common_pair_batch(batch)
    padded_genome, genome_masks = _pad_duplicated_embeddings(
        [item["genome_embedding"] for item in batch]
    )
    result["padded_genome_embeddings"] = padded_genome
    result["genome_attn_masks"] = genome_masks
    return result


def collate_synergy_text_only(batch: list[dict]) -> dict:
    return _common_pair_batch(batch)


def _collate_tokenized_synergy(
    batch: list[dict],
    *,
    pad_token_id: int,
    fixed_length: int,
    has_genome: bool,
) -> dict:
    tokens = [
        token_ids
        for item in batch
        for token_ids in (item["input_ids_1"], item["input_ids_2"])
    ]
    input_ids = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=pad_token_id,
    )
    if input_ids.shape[1] > fixed_length:
        raise ValueError(
            f"Tokenized pair length {input_ids.shape[1]} exceeds fixed length {fixed_length}"
        )
    fixed_input_ids = torch.full(
        (input_ids.shape[0], fixed_length),
        pad_token_id,
        dtype=input_ids.dtype,
    )
    fixed_input_ids[:, : input_ids.shape[1]] = input_ids
    padded_text, text_masks = _pad_duplicated_embeddings(
        [item["text_embedding"] for item in batch]
    )
    result = {
        "input_ids": fixed_input_ids,
        "label": torch.from_numpy(np.array([item["label"] for item in batch])),
        "padded_text_embeddings": padded_text,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "pair_keys": [item["pair_key"] for item in batch],
    }
    if has_genome:
        padded_genome, genome_masks = _pad_duplicated_embeddings(
            [item["genome_embedding"] for item in batch]
        )
        result["padded_genome_embeddings"] = padded_genome
        result["genome_attn_masks"] = genome_masks
    return result


def collate_tokenized_synergy_genome_text(
    batch: list[dict], *, pad_token_id: int, fixed_length: int = 1024
) -> dict:
    return _collate_tokenized_synergy(
        batch,
        pad_token_id=pad_token_id,
        fixed_length=fixed_length,
        has_genome=True,
    )


def collate_tokenized_synergy_text_only(
    batch: list[dict], *, pad_token_id: int, fixed_length: int = 1024
) -> dict:
    return _collate_tokenized_synergy(
        batch,
        pad_token_id=pad_token_id,
        fixed_length=fixed_length,
        has_genome=False,
    )
