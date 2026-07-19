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


def synergy_guidance_pair_forward(
    batch: dict,
    *,
    device: torch.device,
    molecule_encoder: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    has_genome: bool,
    autocast_enabled: bool,
) -> SynergyBatchResult:
    """Forward for the all-data guidance model's online, two-call MDLM path."""

    labels = batch["label"].to(device)
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        molecule_embeddings = molecule_encoder.encode_pairs(
            batch["input_ids"].to(device)
        )
        first, second = molecule_embeddings[::2], molecule_embeddings[1::2]
        if has_genome:
            # The order is observable because both attention modules contain
            # training-mode dropout: genome(1), text(1), genome(2), text(2).
            fused_parts = [
                fuse_genome_text_embeddings(
                    current,
                    batch["padded_genome_embeddings"][offset::2],
                    batch["genome_attn_masks"][offset::2],
                    batch["padded_text_embeddings"][offset::2],
                    batch["text_attn_masks"][offset::2],
                    genome_attention,
                    text_attention,
                    reshape_outputs=True,
                )
                for offset, current in ((0, first), (1, second))
            ]
        else:
            # The text-only legacy branch used a different observable order:
            # genome(1), genome(2), text(1), text(2).
            missing = missing_genome_embedding[:, None, :].expand(
                molecule_embeddings.shape[0], 1, -1
            )
            missing_mask = torch.ones(
                molecule_embeddings.shape[0],
                1,
                dtype=torch.uint8,
                device=device,
            )
            genome_parts = [
                genome_attention(
                    mol_cls_emb=current,
                    genome_embs=missing[offset::2],
                    key_padding_mask=1 - missing_mask[offset::2],
                ).reshape(-1, missing_genome_embedding.shape[-1])
                for offset, current in ((0, first), (1, second))
            ]
            text_parts = [
                text_attention(
                    mol_cls_emb=current,
                    genome_embs=batch["padded_text_embeddings"][offset::2],
                    key_padding_mask=1 - batch["text_attn_masks"][offset::2],
                ).reshape(-1, batch["padded_text_embeddings"].shape[-1])
                for offset, current in ((0, first), (1, second))
            ]
            fused_parts = [
                torch.cat((genome, text), dim=1)
                for genome, text in zip(genome_parts, text_parts)
            ]
        logits_forward = prediction_head(torch.cat(fused_parts, dim=1))
        logits_reverse = prediction_head(
            torch.cat((fused_parts[1], fused_parts[0]), dim=1)
        )
        logits = (logits_forward + logits_reverse) / 2
        loss = criterion(logits.squeeze(), labels.squeeze())
    return SynergyBatchResult(
        loss=loss,
        logits=logits,
        labels=labels,
        strain_names=batch["strain_names"],
        pair_keys=batch["pair_keys"],
    )


def synergy_guidance_pair_step(
    batch: dict,
    *,
    device: torch.device,
    molecule_encoder: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    prediction_head: nn.Module,
    criterion: nn.Module,
    missing_genome_embedding: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scaler,
    has_genome: bool,
    autocast_enabled: bool,
) -> SynergyBatchResult:
    optimizer.zero_grad()
    # Dead in value, but not in behavior: the legacy driver consumed one CPU
    # RNG draw before every modality step, which affects later loader shuffles.
    _legacy_noise_input = torch.randn(1)[0].item() < 0.0
    del _legacy_noise_input
    result = synergy_guidance_pair_forward(
        batch,
        device=device,
        molecule_encoder=molecule_encoder,
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=prediction_head,
        criterion=criterion,
        missing_genome_embedding=missing_genome_embedding,
        has_genome=has_genome,
        autocast_enabled=autocast_enabled,
    )
    scaler.scale(result.loss).backward()
    scaler.unscale_(optimizer)
    if not has_genome:
        torch.nn.utils.clip_grad_norm_([missing_genome_embedding], max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(genome_attention.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(text_attention.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(prediction_head.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return result


def legacy_synergy_guidance_checkpoint_payload(
    *,
    auroc: float,
    optimizer: torch.optim.Optimizer,
    molecule_encoder: nn.Module,
    prediction_head: nn.Module,
    genome_attention: nn.Module,
    text_attention: nn.Module,
    missing_genome_embedding: nn.Parameter,
) -> dict:
    """Preserve the seven-key all-data guidance checkpoint schema."""

    return {
        "AUROC": auroc,
        "optimizer_state_dict": optimizer.state_dict(),
        "mdlm_model_state_dict": molecule_encoder.state_dict(),
        "re_head_state_dict": prediction_head.state_dict(),
        "co_cross_attn_genome": genome_attention.state_dict(),
        "co_cross_attn_text": text_attention.state_dict(),
        "learnable_embedding_weight": missing_genome_embedding,
    }
