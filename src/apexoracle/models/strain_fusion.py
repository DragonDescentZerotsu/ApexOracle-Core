"""Fusion and prediction heads used by the final strain-aware paper models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Paper-era two-layer GELU/dropout regression or classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 384,
        hidden_dim_2: int = 128,
        num_targets: int = 19,
        pooler_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


class FirstTokenAttentionGenome(nn.Module):
    """Molecule-query cross-attention used for both genome and text sequences."""

    def __init__(
        self,
        mol_cls_embed_dim: int,
        genome_embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, genome_embed_dim)
        self.key_value_projection = nn.Linear(genome_embed_dim, genome_embed_dim * 2)
        self.mha = nn.MultiheadAttention(genome_embed_dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(genome_embed_dim)
        self.norm1 = nn.LayerNorm(genome_embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(genome_embed_dim, genome_embed_dim),
            nn.GELU(),
            nn.Linear(genome_embed_dim, genome_embed_dim),
        )
        self.norm2 = nn.LayerNorm(genome_embed_dim)

    def forward(self, mol_cls_emb, genome_embs, key_padding_mask):
        genome_embs_dim = genome_embs.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :]
        query = query.transpose(0, 1)
        key_value = self.key_value_projection(
            genome_embs.reshape(-1, genome_embs.shape[-1])
        ).reshape([genome_embs.shape[0], genome_embs.shape[1], -1])
        key_value = key_value.transpose(0, 1)
        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        attn_output, _ = self.mha(
            query_norm,
            key_value[:, :, :genome_embs_dim],
            key_value[:, :, genome_embs_dim:],
            key_padding_mask=key_padding_mask.to(torch.bool),
        )
        # ``squeeze`` without a dimension is intentionally preserved, including
        # the legacy batch-size-one behavior.
        query = self.norm1(query.squeeze() + attn_output.squeeze())
        ffn_output = self.ffn(query)
        return self.norm2(query + ffn_output)


FirstTokenAttention_genome = FirstTokenAttentionGenome


def fuse_genome_text_embeddings(
    molecule_embedding,
    genome_embeddings,
    genome_attention_mask,
    text_embeddings,
    text_attention_mask,
    genome_attention,
    text_attention,
    *,
    reshape_outputs: bool,
):
    """Run the paper-era dual cross-attention and concatenate its outputs."""

    genome = genome_attention(
        molecule_embedding, genome_embeddings, 1 - genome_attention_mask
    )
    text = text_attention(
        molecule_embedding, text_embeddings, 1 - text_attention_mask
    )
    if reshape_outputs:
        genome = genome.reshape(-1, genome_embeddings.shape[-1])
        text = text.reshape(-1, text_embeddings.shape[-1])
    return torch.cat((genome, text), dim=1)


def fuse_text_only_embeddings(
    molecule_embedding,
    text_embeddings,
    text_attention_mask,
    missing_genome_embedding,
    genome_attention,
    text_attention,
    *,
    reshape_outputs: bool,
):
    """Use the learned one-token genome placeholder exactly as the legacy path."""

    batch_size = molecule_embedding.shape[0]
    genome_embeddings = missing_genome_embedding[:, None, :].expand(
        batch_size, 1, -1
    )
    genome_attention_mask = torch.from_numpy(np.array([1]))[None, :].expand(
        batch_size, -1
    ).to(molecule_embedding.device)
    return fuse_genome_text_embeddings(
        molecule_embedding,
        genome_embeddings,
        genome_attention_mask,
        text_embeddings,
        text_attention_mask,
        genome_attention,
        text_attention,
        reshape_outputs=reshape_outputs,
    )
