"""Behavior-frozen primitives for the Fig. 1b classification family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from apexoracle.training.hierarchical_mic import (
    HierarchicalMicBatchResult,
    hierarchical_mic_batch_forward,
)


@dataclass
class ClassificationEvaluation:
    losses: list[float]
    labels: list[float]
    logits: list[float]
    molecule_ids: list


def set_legacy_full_fusion_training_modes(
    genome_attention: nn.Module,
    text_attention: nn.Module,
    regression_head: nn.Module,
    classification_head: nn.Module,
) -> None:
    """Set only the three modules explicitly toggled by the legacy scripts.

    ``classification_head`` is deliberately untouched. It starts in train mode
    and remains there during held-target checkpoint selection, so its dropout
    is active in the historical AUROC/AUPRC values.
    """

    genome_attention.train()
    text_attention.train()
    regression_head.train()


def set_legacy_full_fusion_selection_modes(
    genome_attention: nn.Module,
    text_attention: nn.Module,
    regression_head: nn.Module,
    classification_head: nn.Module,
) -> None:
    """Apply the legacy held-target modes without touching ``cls_head``."""

    genome_attention.eval()
    text_attention.eval()
    regression_head.eval()


def full_fusion_classification_step(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    classification_head: nn.Module,
    regression_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scaler,
    has_genome: bool,
    epoch: int,
    freeze_epochs: int,
    autocast_enabled: bool,
    clip_text_attention: bool,
    clip_missing_genome: bool,
) -> HierarchicalMicBatchResult:
    optimizer.zero_grad()
    result = full_fusion_classification_forward(
        batch,
        device=device,
        genome_attention=genome_attention,
        text_attention=text_attention,
        classification_head=classification_head,
        criterion=criterion,
        missing_genome_embedding=missing_genome_embedding,
        has_genome=has_genome,
        autocast_enabled=autocast_enabled,
    )
    scaler.scale(result.loss).backward()
    if epoch >= freeze_epochs:
        scaler.unscale_(optimizer)
        if clip_missing_genome:
            torch.nn.utils.clip_grad_norm_([missing_genome_embedding], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(genome_attention.parameters(), max_norm=1.0)
        if clip_text_attention:
            torch.nn.utils.clip_grad_norm_(text_attention.parameters(), max_norm=1.0)
        # The paper-era scripts consistently clip reg_head, not cls_head.
        torch.nn.utils.clip_grad_norm_(regression_head.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return result


def full_fusion_classification_forward(
    batch: dict,
    *,
    device: torch.device,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    classification_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    has_genome: bool,
    autocast_enabled: bool,
) -> HierarchicalMicBatchResult:
    return hierarchical_mic_batch_forward(
        batch,
        device=device,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=classification_head,
        criterion=criterion,
        missing_genome_embedding=missing_genome_embedding,
        has_genome=has_genome,
        reshape_outputs=False,
        autocast_enabled=autocast_enabled,
    )


def molecule_only_forward(
    batch: dict,
    *,
    device: torch.device,
    classification_head: nn.Module,
    criterion: nn.Module,
    autocast_enabled: bool,
) -> HierarchicalMicBatchResult:
    labels = batch["label"].to(device)
    molecule_embeddings = batch["mol_emb"].to(device)
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        logits = classification_head(molecule_embeddings).squeeze()
        loss = criterion(logits, labels.squeeze())
    return HierarchicalMicBatchResult(
        loss=loss,
        logits=logits,
        labels=labels,
        strain_names=batch["strain_names"],
    )


def molecule_only_step(
    batch: dict,
    *,
    device: torch.device,
    classification_head: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    autocast_enabled: bool,
) -> HierarchicalMicBatchResult:
    optimizer.zero_grad()
    result = molecule_only_forward(
        batch,
        device=device,
        classification_head=classification_head,
        criterion=criterion,
        autocast_enabled=autocast_enabled,
    )
    scaler.scale(result.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return result


def legacy_full_fusion_checkpoint_payload(
    *,
    auroc: float,
    checkpoint_auprc: float,
    optimizer: torch.optim.Optimizer,
    regression_head: nn.Module,
    classification_head: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    missing_genome_embedding: nn.Parameter,
) -> dict:
    return {
        "auroc": auroc,
        "auprc": checkpoint_auprc,
        "optimizer_state_dict": optimizer.state_dict(),
        "re_head_state_dict": regression_head.state_dict(),
        "cls_head_state_dict": classification_head.state_dict(),
        "co_cross_attn_genome": genome_attention.state_dict(),
        "co_cross_attn_text": text_attention.state_dict(),
        "learnable_embedding_weight": missing_genome_embedding,
    }


def legacy_molecule_only_checkpoint_payload(
    *,
    auroc: float,
    checkpoint_auprc: float,
    optimizer: torch.optim.Optimizer,
    classification_head: nn.Module,
) -> dict:
    return {
        "auroc": auroc,
        "auprc": checkpoint_auprc,
        "optimizer_state_dict": optimizer.state_dict(),
        "cls_head_state_dict": classification_head.state_dict(),
    }
