"""Published APEX encoder architecture used by the Fig. 2b comparator.

Names and operations intentionally match ``compare_APEX/APEX_models.py`` so
the original checkpoint loads strictly and produces identical features.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ApexPeptideEmbeddings(nn.Module):
    def __init__(self, embedding: np.ndarray) -> None:
        super().__init__()
        self.aa_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding, dtype=torch.float32),
            padding_idx=0,
            freeze=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.aa_embedding(token_ids)


class ApexEncoder(nn.Module):
    """Unmodified 3-layer bidirectional-GRU APEX feature encoder."""

    def __init__(
        self,
        embedding: np.ndarray,
        embedding_dim: int,
        *,
        num_rnn_layers: int = 3,
        hidden_dim: int = 128,
        max_length: int = 52,
    ) -> None:
        super().__init__()
        self.peptideEmb = ApexPeptideEmbeddings(embedding)
        self.dim_emb = embedding_dim
        self.dim_h = hidden_dim
        self.dropout = 0.1
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.attn1 = nn.Linear(hidden_dim * 2 + embedding_dim, max_length)
        self.attn2 = nn.Linear(hidden_dim * 2, 1)
        self.fc0 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.peptideEmb(token_ids)
        recurrent, _ = self.rnn(embedded)
        recurrent = self.layernorm(recurrent)
        attention_1 = functional.softmax(
            self.attn1(torch.cat((recurrent, embedded), dim=2)), dim=2
        )
        attended = torch.bmm(attention_1, recurrent)
        attention_2 = functional.softmax(self.attn2(attended), dim=1)
        pooled = torch.sum(attention_2 * attended, dim=1)
        return self.fc0(pooled)


def load_aaindex_embedding(
    path: Path, vocabulary: Mapping[str, int]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the legacy AAindex CSV with the original row/column semantics."""

    matrix_rows: list[np.ndarray] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"AAindex file is empty: {path}")
        header = np.asarray(first)[1:].tolist()
        for row in reader:
            values = []
            for value in np.asarray(row)[1:]:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(0.0)
            matrix_rows.append(np.asarray(values))
    matrix = np.asarray(matrix_rows)
    aaindex = {symbol: matrix[:, index] for index, symbol in enumerate(header)}
    embedding = np.zeros((len(vocabulary), matrix.shape[0]))
    for symbol, index in vocabulary.items():
        if symbol in aaindex:
            embedding[index] = aaindex[symbol]
    return embedding, aaindex
