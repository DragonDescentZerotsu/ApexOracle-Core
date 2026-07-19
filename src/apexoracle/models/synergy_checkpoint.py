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
GUIDANCE_CHECKPOINT_KEYS = {
    "AUROC",
    "optimizer_state_dict",
    "mdlm_model_state_dict",
    "re_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
}
REGRESSION_MEMBER_CHECKPOINT_KEYS = {
    "R2",
    "optimizer_state_dict",
    "re_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
}


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


def inspect_synergy_guidance_checkpoint(state: dict) -> dict:
    if set(state) != GUIDANCE_CHECKPOINT_KEYS:
        raise ValueError(
            "synergy guidance checkpoint keys changed: "
            f"expected={sorted(GUIDANCE_CHECKPOINT_KEYS)}, got={sorted(state)}"
        )
    genome_state = state["co_cross_attn_genome"]
    text_state = state["co_cross_attn_text"]
    genome_lora = next(
        value for key, value in genome_state.items() if "lora_A" in key
    )
    text_lora = next(value for key, value in text_state.items() if "lora_A" in key)
    head_state = state["re_head_state_dict"]
    return {
        "auroc": float(state["AUROC"]),
        "mdlm_key_count": len(state["mdlm_model_state_dict"]),
        "genome_attention_key_count": len(genome_state),
        "text_attention_key_count": len(text_state),
        "fusion_lora_rank": int(genome_lora.shape[0]),
        "text_fusion_lora_rank": int(text_lora.shape[0]),
        "head_dimensions": [
            int(head_state["dense_1.weight"].shape[1]),
            int(head_state["dense_1.weight"].shape[0]),
            int(head_state["dense_2.weight"].shape[0]),
            int(head_state["out_proj.weight"].shape[0]),
        ],
        "missing_genome_shape": list(state["learnable_embedding_weight"].shape),
    }


def load_synergy_guidance_checkpoint(
    checkpoint_path: Path,
    *,
    molecule_encoder: nn.Module,
    components: SynergyComponents,
) -> dict:
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    contract = inspect_synergy_guidance_checkpoint(state)
    molecule_encoder.load_state_dict(state["mdlm_model_state_dict"], strict=True)
    components.genome_attention.load_state_dict(
        state["co_cross_attn_genome"], strict=True
    )
    components.text_attention.load_state_dict(
        state["co_cross_attn_text"], strict=True
    )
    components.prediction_head.load_state_dict(
        state["re_head_state_dict"], strict=True
    )
    components.missing_genome_embedding.data.copy_(
        state["learnable_embedding_weight"].to(
            components.missing_genome_embedding.device
        )
    )
    return contract


def build_legacy_synergy_regression_member(
    checkpoint_path: Path,
    *,
    device: torch.device,
    molecule_dim: int = 768,
    genome_dim: int = 8192,
    text_dim: int = 4096,
    attention_heads: int = 4,
    lora_rank: int = 64,
) -> tuple[SynergyComponents, float]:
    """Strictly load one prospective-screening regression member."""

    from peft import LoraConfig, TaskType, get_peft_model

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if set(state) != REGRESSION_MEMBER_CHECKPOINT_KEYS:
        raise ValueError(
            f"Unexpected regression checkpoint keys in {checkpoint_path}: "
            f"{sorted(state)}"
        )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=list(LEGACY_LORA_TARGETS),
        task_type=TaskType.FEATURE_EXTRACTION,
        lora_dropout=0.1,
        bias="none",
    )
    genome_attention = get_peft_model(
        FirstTokenAttentionGenome(
            molecule_dim, genome_dim, attention_heads, 0.1
        ),
        lora,
    ).to(device)
    text_attention = get_peft_model(
        FirstTokenAttentionGenome(
            molecule_dim, text_dim, attention_heads, 0.1
        ),
        lora,
    ).to(device)
    prediction_head = RegressionHead(
        (genome_dim + text_dim) * 2,
        (genome_dim + text_dim) // 4,
        128,
        1,
        0.2,
    ).to(device)
    genome_attention.load_state_dict(state["co_cross_attn_genome"], strict=True)
    text_attention.load_state_dict(state["co_cross_attn_text"], strict=True)
    prediction_head.load_state_dict(state["re_head_state_dict"], strict=True)
    missing = nn.Parameter(
        state["learnable_embedding_weight"].to(device).detach(),
        requires_grad=False,
    )
    for module in (genome_attention, text_attention, prediction_head):
        module.eval()
    components = SynergyComponents(
        genome_attention=genome_attention,
        text_attention=text_attention,
        prediction_head=prediction_head,
        missing_genome_embedding=missing,
    )
    return components, float(state["R2"])
