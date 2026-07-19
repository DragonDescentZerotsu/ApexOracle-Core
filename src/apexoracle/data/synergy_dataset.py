"""Datasets and collators for the paper-era synergy pair classifier."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    ) -> None:
        if tuple(table.columns) != SYNERGY_COLUMNS:
            raise ValueError(f"Unexpected synergy columns: {tuple(table.columns)}")
        self.table = table.reset_index(drop=True).copy()
        self.molecule_embeddings = molecule_embeddings
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        strain = row["strain_name"]
        item = {
            "label": torch.tensor(synergy_label(row["FICI"]), dtype=torch.float),
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
