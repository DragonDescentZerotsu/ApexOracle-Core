"""Behavior-frozen model operations for the synergy pair classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from apexoracle.models.strain_fusion import (
    fuse_genome_text_embeddings,
    fuse_text_only_embeddings,
)


@dataclass
class SynergyBatchResult:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    strain_names: list[str]
    pair_keys: list[tuple]


def symmetric_pair_logits(
    fused_molecule_embeddings: torch.Tensor,
    prediction_head: nn.Module,
) -> torch.Tensor:
    """Average predictions for both molecule orders, as in the legacy driver."""

    if fused_molecule_embeddings.shape[0] % 2:
        raise ValueError("Expected exactly two molecule embeddings per pair")
    first = fused_molecule_embeddings[::2]
    second = fused_molecule_embeddings[1::2]
    logits_forward = prediction_head(torch.cat((first, second), dim=1))
    logits_reverse = prediction_head(torch.cat((second, first), dim=1))
    return (logits_forward + logits_reverse) / 2


def synergy_pair_forward(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    has_genome: bool,
    autocast_enabled: bool,
) -> SynergyBatchResult:
    labels = batch["label"].to(device)
    molecule_embeddings = batch["mol_emb"].to(device)
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        if has_genome:
            fused = fuse_genome_text_embeddings(
                molecule_embeddings,
                batch["padded_genome_embeddings"],
                batch["genome_attn_masks"],
                batch["padded_text_embeddings"],
                batch["text_attn_masks"],
                genome_attention,
                text_attention,
                reshape_outputs=True,
            )
        else:
            fused = fuse_text_only_embeddings(
                molecule_embeddings,
                batch["padded_text_embeddings"],
                batch["text_attn_masks"],
                missing_genome_embedding,
                genome_attention,
                text_attention,
                reshape_outputs=True,
            )
        logits = symmetric_pair_logits(fused, prediction_head)
        loss = criterion(logits.squeeze(), labels.squeeze())
    return SynergyBatchResult(
        loss=loss,
        logits=logits,
        labels=labels,
        strain_names=batch["strain_names"],
        pair_keys=batch["pair_keys"],
    )


def synergy_pair_step(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scaler,
    has_genome: bool,
    autocast_enabled: bool,
    epoch: int,
    freeze_epochs: int,
) -> SynergyBatchResult:
    optimizer.zero_grad()
    result = synergy_pair_forward(
        batch,
        device=device,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=prediction_head,
        criterion=criterion,
        missing_genome_embedding=missing_genome_embedding,
        has_genome=has_genome,
        autocast_enabled=autocast_enabled,
    )
    scaler.scale(result.loss).backward()
    if epoch >= freeze_epochs:
        scaler.unscale_(optimizer)
        if not has_genome:
            torch.nn.utils.clip_grad_norm_([missing_genome_embedding], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(genome_attention.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(text_attention.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(prediction_head.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return result


def legacy_synergy_checkpoint_payload(
    *,
    auroc: float,
    optimizer: torch.optim.Optimizer,
    prediction_head: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    missing_genome_embedding: nn.Parameter,
) -> dict:
    """Preserve the six-key paper checkpoint schema and LoRA-only adapters."""

    return {
        "AUROC": auroc,
        "optimizer_state_dict": optimizer.state_dict(),
        "re_head_state_dict": prediction_head.state_dict(),
        "co_cross_attn_genome": {
            key: value
            for key, value in genome_attention.state_dict().items()
            if "lora" in key
        },
        "co_cross_attn_text": {
            key: value
            for key, value in text_attention.state_dict().items()
            if "lora" in key
        },
        "learnable_embedding_weight": missing_genome_embedding,
    }
