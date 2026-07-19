"""Behavior-frozen single-batch training for the strain-wise paper protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from apexoracle.models.strain_fusion import (
    fuse_genome_text_embeddings,
    fuse_text_only_embeddings,
)


@dataclass
class StrainwiseBatchResult:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    strain_names: list[str]


def strainwise_batch_forward(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    has_genome: bool,
    reshape_outputs: bool,
    autocast_enabled: bool,
) -> StrainwiseBatchResult:
    """Compute one legacy batch without changing optimizer or module modes."""

    labels = batch["label"].to(device)
    molecule_embedding = batch["mol_emb"].to(device)
    text_embeddings = batch["padded_text_embeddings"]
    text_attention_mask = batch["text_attn_masks"]

    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        if has_genome:
            fused = fuse_genome_text_embeddings(
                molecule_embedding,
                batch["padded_genome_embeddings"],
                batch["genome_attn_masks"],
                text_embeddings,
                text_attention_mask,
                genome_attention,
                text_attention,
                reshape_outputs=reshape_outputs,
            )
        else:
            fused = fuse_text_only_embeddings(
                molecule_embedding,
                text_embeddings,
                text_attention_mask,
                missing_genome_embedding,
                genome_attention,
                text_attention,
                reshape_outputs=reshape_outputs,
            )
        logits = prediction_head(fused).squeeze()
        loss = criterion(logits, labels.squeeze())

    return StrainwiseBatchResult(
        loss=loss,
        logits=logits,
        labels=labels,
        strain_names=batch["strain_names"],
    )


def strainwise_optimizer_step(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    legacy_regression_head_for_clipping: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scaler,
    has_genome: bool,
    reshape_outputs: bool,
    autocast_enabled: bool,
    epoch: int,
    freeze_epochs: int,
) -> StrainwiseBatchResult:
    """Run the exact zero-grad/backward/optional-clip/step legacy sequence.

    The historical classification branches clip ``reg_head`` rather than
    ``cls_head`` after ``freeze_epochs``. The explicit legacy argument preserves
    that contract and prevents a future cleanup from silently changing it.
    """

    optimizer.zero_grad()
    result = strainwise_batch_forward(
        batch,
        device=device,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=prediction_head,
        criterion=criterion,
        missing_genome_embedding=missing_genome_embedding,
        has_genome=has_genome,
        reshape_outputs=reshape_outputs,
        autocast_enabled=autocast_enabled,
    )
    scaler.scale(result.loss).backward()
    if epoch >= freeze_epochs:
        scaler.unscale_(optimizer)
        if not has_genome:
            torch.nn.utils.clip_grad_norm_([missing_genome_embedding], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(genome_attention.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            legacy_regression_head_for_clipping.parameters(), max_norm=1.0
        )
    scaler.step(optimizer)
    scaler.update()
    return result
