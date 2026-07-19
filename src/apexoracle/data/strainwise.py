"""Datasets and collators for the paper-era strain-wise experiments.

The uint8 masks, bfloat16 padding, MIC transform, and molecular embedding lookup
order are deliberate compatibility requirements. Do not modernize them in place.
"""

from __future__ import annotations

import ast
import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class StrainEmbeddingDataset(Dataset):
    """Molecule/strain records with both genome and text embeddings."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        embeddings_dict: Optional[Mapping[str, torch.Tensor]],
        text_embeddings_dict: Mapping[str, torch.Tensor],
        set_desc: str,
        pep_emb_dict: Mapping,
        sm_emb_dict: Mapping,
        max_length: int = 512,
    ) -> None:
        self.dataframe = dataframe
        self.original_length = len(self.dataframe)
        self.tokenizer = tokenizer
        self.embeddings_dict = embeddings_dict
        self.text_embeddings_dict = text_embeddings_dict
        self.max_length = max_length
        self.pep_emb_dict = pep_emb_dict
        self.SM_emb_dict = sm_emb_dict
        self.target_columns = "MIC"
        self.remove_long_smiles()
        LOGGER.info(
            "\n %s:\n original length: %d\n after SMILES length limitation length: %d",
            set_desc,
            self.original_length,
            len(self.dataframe),
        )

    @staticmethod
    def tokenize_smiles(smiles):
        input_ids = torch.from_numpy(np.array(ast.literal_eval(smiles)))
        attn_mask = torch.ones_like(input_ids)
        return input_ids, attn_mask

    def remove_long_smiles(self) -> None:
        tokenized_cols = self.dataframe["SMILES"].apply(
            lambda value: pd.Series(
                self.tokenize_smiles(value), index=["input_ids", "attn_mask"]
            )
        )
        self.dataframe = pd.concat([self.dataframe, tokenized_cols], axis=1)
        self.dataframe = self.dataframe[
            self.dataframe["input_ids"].apply(len) <= self.max_length
        ]
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.dataframe.drop(columns=["SMILES"], inplace=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        dbaasp_id = self.dataframe.iloc[idx]["DBAASP_id"]
        if self.pep_emb_dict.get(dbaasp_id) is not None:
            mol_emb = self.pep_emb_dict[dbaasp_id]
        else:
            mol_emb = self.SM_emb_dict[dbaasp_id]
        strain_name = self.dataframe.iloc[idx]["strain_name"]
        target = self.dataframe.iloc[idx][self.target_columns]
        return {
            "label": torch.tensor(target, dtype=torch.float),
            "genome_embedding": self.embeddings_dict[strain_name],
            "text_embedding": self.text_embeddings_dict[strain_name],
            "strain_name": strain_name,
            "mol_emb": mol_emb.squeeze(),
        }


class TextOnlyStrainEmbeddingDataset(StrainEmbeddingDataset):
    """Molecule/strain records for strains without a stored genome embedding."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        text_embeddings_dict: Mapping[str, torch.Tensor],
        set_desc: str,
        pep_emb_dict: Mapping,
        sm_emb_dict: Mapping,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            dataframe,
            tokenizer,
            embeddings_dict=None,
            text_embeddings_dict=text_embeddings_dict,
            set_desc=set_desc,
            pep_emb_dict=pep_emb_dict,
            sm_emb_dict=sm_emb_dict,
            max_length=max_length,
        )

    def __getitem__(self, idx: int) -> dict:
        dbaasp_id = self.dataframe.iloc[idx]["DBAASP_id"]
        if self.pep_emb_dict.get(dbaasp_id) is not None:
            mol_emb = self.pep_emb_dict[dbaasp_id]
        else:
            mol_emb = self.SM_emb_dict[dbaasp_id]
        strain_name = self.dataframe.iloc[idx]["strain_name"]
        target = self.dataframe.iloc[idx][self.target_columns]
        return {
            "label": torch.tensor(target, dtype=torch.float),
            "text_embedding": self.text_embeddings_dict[strain_name],
            "strain_name": strain_name,
            "mol_emb": mol_emb.squeeze(),
        }


def _pad_embeddings(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = 0
    for embedding in embeddings:
        if len(embedding) > max_length:
            max_length = len(embedding)

    padded_embeddings = []
    attention_masks = []
    for embedding in embeddings:
        length, dimension = embedding.shape
        attention_mask = torch.zeros(
            max_length, device=embedding.device, dtype=torch.uint8
        )
        padding = torch.zeros(
            (max_length, dimension),
            dtype=torch.bfloat16,
            device=embedding.device,
        )
        padding[:length] = embedding
        attention_mask[:length] = 1
        padded_embeddings.append(padding)
        attention_masks.append(attention_mask)

    return torch.stack(padded_embeddings), torch.stack(attention_masks)


def _labels_as_legacy_tensor(batch: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.array([item["label"] for item in batch]))


def collate_genome_text_regression(batch: list[dict]) -> dict:
    genome_embeddings, genome_masks = _pad_embeddings(
        [item["genome_embedding"] for item in batch]
    )
    text_embeddings, text_masks = _pad_embeddings(
        [item["text_embedding"] for item in batch]
    )
    labels = -torch.log10(_labels_as_legacy_tensor(batch) / 10)
    return {
        "label": labels,
        "padded_genome_embeddings": genome_embeddings,
        "genome_attn_masks": genome_masks,
        "padded_text_embeddings": text_embeddings,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "mol_emb": torch.stack([item["mol_emb"] for item in batch]),
    }


def collate_text_regression(batch: list[dict]) -> dict:
    text_embeddings, text_masks = _pad_embeddings(
        [item["text_embedding"] for item in batch]
    )
    labels = -torch.log10(_labels_as_legacy_tensor(batch) / 10)
    return {
        "label": labels,
        "padded_text_embeddings": text_embeddings,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "mol_emb": torch.stack([item["mol_emb"] for item in batch]),
    }


def collate_genome_text_classification(batch: list[dict]) -> dict:
    genome_embeddings, genome_masks = _pad_embeddings(
        [item["genome_embedding"] for item in batch]
    )
    text_embeddings, text_masks = _pad_embeddings(
        [item["text_embedding"] for item in batch]
    )
    return {
        "label": _labels_as_legacy_tensor(batch),
        "padded_genome_embeddings": genome_embeddings,
        "genome_attn_masks": genome_masks,
        "padded_text_embeddings": text_embeddings,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "mol_emb": torch.stack([item["mol_emb"] for item in batch]),
    }


def collate_text_classification(batch: list[dict]) -> dict:
    text_embeddings, text_masks = _pad_embeddings(
        [item["text_embedding"] for item in batch]
    )
    return {
        "label": _labels_as_legacy_tensor(batch),
        "padded_text_embeddings": text_embeddings,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "mol_emb": torch.stack([item["mol_emb"] for item in batch]),
    }


# Legacy names used by transitional drivers.
SMILESDataset_with_genome_and_text = StrainEmbeddingDataset
SMILESDataset_with_text_only = TextOnlyStrainEmbeddingDataset
collate_fn = collate_genome_text_regression
collate_fn_text_only = collate_text_regression
collate_fn_cls = collate_genome_text_classification
collate_fn_text_only_cls = collate_text_classification
