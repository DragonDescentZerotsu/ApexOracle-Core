"""Behavior-frozen batch training shared by hierarchical MIC protocols."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch
import torch.nn as nn

from apexoracle.models.strain_fusion import (
    fuse_genome_text_embeddings,
    fuse_text_only_embeddings,
)


@dataclass
class HierarchicalMicBatchResult:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    strain_names: list[str]


def legacy_zip_longest_loaders(*loaders):
    """Return the historical longest-loader iterator and tqdm total."""

    return itertools.zip_longest(*loaders, fillvalue=None), max(
        len(loader) for loader in loaders
    )


def build_legacy_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    min_lr: float = 1e-10,
):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_lr
    )


def hierarchical_mic_batch_forward(
    batch: dict,
    *,
    device: torch.device,
    molecule_encoder: nn.Module | None = None,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    has_genome: bool,
    reshape_outputs: bool,
    autocast_enabled: bool,
) -> HierarchicalMicBatchResult:
    """Compute one legacy batch without changing optimizer or module modes."""

    labels = batch["label"].to(device)
    text_embeddings = batch["padded_text_embeddings"]
    text_attention_mask = batch["text_attn_masks"]

    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        if molecule_encoder is None:
            molecule_embedding = batch["mol_emb"].to(device)
        else:
            outputs = molecule_encoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            molecule_embedding = outputs.last_hidden_state[:, 0, :]
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

    return HierarchicalMicBatchResult(
        loss=loss,
        logits=logits,
        labels=labels,
        strain_names=batch["strain_names"],
    )


def hierarchical_mic_optimizer_step(
    batch: dict,
    *,
    device: torch.device,
    molecule_encoder: nn.Module | None = None,
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
) -> HierarchicalMicBatchResult:
    """Run the exact zero-grad/backward/optional-clip/step legacy sequence.

    The historical classification branches clip ``reg_head`` rather than
    ``cls_head`` after ``freeze_epochs``. The explicit legacy argument preserves
    that contract and prevents a future cleanup from silently changing it.
    """

    optimizer.zero_grad()
    result = hierarchical_mic_batch_forward(
        batch,
        device=device,
        molecule_encoder=molecule_encoder,
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
        if molecule_encoder is not None:
            torch.nn.utils.clip_grad_norm_(molecule_encoder.parameters(), max_norm=1.0)
        if not has_genome:
            torch.nn.utils.clip_grad_norm_([missing_genome_embedding], max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(genome_attention.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            legacy_regression_head_for_clipping.parameters(), max_norm=1.0
        )
    scaler.step(optimizer)
    scaler.update()
    return result


def legacy_hierarchical_checkpoint_payload(
    *,
    r2: float,
    optimizer: torch.optim.Optimizer,
    regression_head: nn.Module,
    classification_head: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    missing_genome_embedding: nn.Parameter,
    molecule_encoder: nn.Module | None = None,
    molecule_encoder_state_key: str | None = None,
    genome_embedding_adapter: nn.Module | None = None,
) -> dict:
    """Build the legacy payload, including an optional frozen input adapter."""

    payload = {
        "R2": r2,
        "optimizer_state_dict": optimizer.state_dict(),
        "re_head_state_dict": regression_head.state_dict(),
        "cls_head_state_dict": classification_head.state_dict(),
        "co_cross_attn_genome": genome_attention.state_dict(),
        "co_cross_attn_text": text_attention.state_dict(),
        "learnable_embedding_weight": missing_genome_embedding,
    }
    if molecule_encoder is not None:
        if not molecule_encoder_state_key:
            raise ValueError(
                "molecule_encoder_state_key is required when saving an online encoder"
            )
        payload[molecule_encoder_state_key] = molecule_encoder.state_dict()
    if genome_embedding_adapter is not None:
        payload["kmer_projection_state_dict"] = (
            genome_embedding_adapter.state_dict()
        )
    return payload


# Compatibility names used by the audited legacy driver and earlier tests.
StrainwiseBatchResult = HierarchicalMicBatchResult
strainwise_batch_forward = hierarchical_mic_batch_forward
strainwise_optimizer_step = hierarchical_mic_optimizer_step
legacy_strainwise_checkpoint_payload = legacy_hierarchical_checkpoint_payload
