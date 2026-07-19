"""Construction and strict loading of the paper-era synergy classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .strain_fusion import FirstTokenAttentionGenome, RegressionHead


LEGACY_LORA_TARGETS = (
    "mol_to_genome_dim",
    "key_value_projection",
    "mha.out_proj",
    "ffn.0",
    "ffn.2",
)


@dataclass
class SynergyComponents:
    genome_attention: nn.Module
    text_attention: nn.Module
    prediction_head: RegressionHead
    missing_genome_embedding: nn.Parameter


def build_legacy_synergy_components(
    base_checkpoint_path: Path,
    *,
    device: torch.device,
    molecule_dim: int = 768,
    genome_dim: int = 8192,
    text_dim: int = 4096,
    attention_heads: int = 4,
    lora_rank: int = 1024,
) -> SynergyComponents:
    """Initialize the CV model from the complete-data MIC base checkpoint."""

    from peft import LoraConfig, TaskType, get_peft_model

    state = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    genome_attention = FirstTokenAttentionGenome(
        molecule_dim, genome_dim, attention_heads, 0.1
    )
    text_attention = FirstTokenAttentionGenome(
        molecule_dim, text_dim, attention_heads, 0.1
    )
    genome_attention.load_state_dict(state["co_cross_attn_genome"], strict=True)
    text_attention.load_state_dict(state["co_cross_attn_text"], strict=True)

    prediction_head = RegressionHead(
        (genome_dim + text_dim) * 2,
        (genome_dim + text_dim) // 4,
        128,
        1,
        0.2,
    )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=list(LEGACY_LORA_TARGETS),
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_dropout=0.1,
        bias="none",
    )
    genome_attention = get_peft_model(genome_attention, lora).to(device)
    text_attention = get_peft_model(text_attention, lora).to(device)
    prediction_head = prediction_head.to(device)
    missing = nn.Parameter(
        state["learnable_embedding_weight"].to(device).detach(),
        requires_grad=False,
    )
    del state
    return SynergyComponents(
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=prediction_head,
        missing_genome_embedding=missing,
    )


def load_legacy_synergy_member(
    checkpoint_path: Path,
    *,
    components: SynergyComponents,
) -> float:
    """Load a saved LoRA/head member onto base-initialized components."""

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    components.prediction_head.load_state_dict(
        state["re_head_state_dict"], strict=True
    )
    for module, key in (
        (components.genome_attention, "co_cross_attn_genome"),
        (components.text_attention, "co_cross_attn_text"),
    ):
        result = module.load_state_dict(state[key], strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter keys in {checkpoint_path}: {result}")
    components.missing_genome_embedding.data.copy_(
        state["learnable_embedding_weight"].to(
            components.missing_genome_embedding.device
        )
    )
    return float(state["AUROC"])
