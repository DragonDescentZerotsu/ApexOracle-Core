"""Strict loader for final hierarchical MIC paper checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from apexoracle.models.strain_fusion import FirstTokenAttentionGenome, RegressionHead

REQUIRED_CHECKPOINT_KEYS = {
    "R2",
    "optimizer_state_dict",
    "re_head_state_dict",
    "cls_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
}
OPTIONAL_LEGACY_CHECKPOINT_KEYS = {"ChemBERTa_state_dict"}
INFERENCE_CHECKPOINT_KEYS = {
    "format",
    "source_checkpoint",
    "source_checkpoint_size",
    "source_checkpoint_sha256",
    "archived_r2",
    "re_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
}


@dataclass
class HierarchicalMicCheckpointComponents:
    regression_head: RegressionHead
    classification_head: RegressionHead
    genome_attention: FirstTokenAttentionGenome
    text_attention: FirstTokenAttentionGenome
    missing_genome_embedding: nn.Parameter
    archived_r2: float

    def eval(self) -> "HierarchicalMicCheckpointComponents":
        self.regression_head.eval()
        self.classification_head.eval()
        self.genome_attention.eval()
        self.text_attention.eval()
        return self

    def to(self, device: torch.device) -> "HierarchicalMicCheckpointComponents":
        self.regression_head.to(device)
        self.classification_head.to(device)
        self.genome_attention.to(device)
        self.text_attention.to(device)
        self.missing_genome_embedding = nn.Parameter(
            self.missing_genome_embedding.detach().to(device)
        )
        return self


def inspect_checkpoint_contract(checkpoint: dict) -> dict:
    missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    unexpected = (
        set(checkpoint) - REQUIRED_CHECKPOINT_KEYS - OPTIONAL_LEGACY_CHECKPOINT_KEYS
    )
    if missing or unexpected:
        raise ValueError(
            f"hierarchical MIC checkpoint contract mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    genome_state = checkpoint["co_cross_attn_genome"]
    text_state = checkpoint["co_cross_attn_text"]
    regression_state = checkpoint["re_head_state_dict"]
    molecule_dim = int(genome_state["mol_to_genome_dim.weight"].shape[1])
    genome_dim = int(genome_state["mol_to_genome_dim.weight"].shape[0])
    text_dim = int(text_state["mol_to_genome_dim.weight"].shape[0])
    head_input_dim = int(regression_state["dense_1.weight"].shape[1])
    hidden_dim_1 = int(regression_state["dense_1.weight"].shape[0])
    hidden_dim_2 = int(regression_state["dense_2.weight"].shape[0])
    num_targets = int(regression_state["out_proj.weight"].shape[0])
    if head_input_dim != genome_dim + text_dim:
        raise ValueError(
            f"head input {head_input_dim} != genome {genome_dim} + text {text_dim}"
        )
    missing_embedding_shape = tuple(checkpoint["learnable_embedding_weight"].shape)
    if missing_embedding_shape != (1, genome_dim):
        raise ValueError(
            f"missing-genome embedding shape {missing_embedding_shape} != {(1, genome_dim)}"
        )
    optional_payloads = {}
    if "ChemBERTa_state_dict" in checkpoint:
        backbone_state = checkpoint["ChemBERTa_state_dict"]
        if "backbone.vocab_embed.embedding" in backbone_state:
            interpretation = "historically_misnamed_mdlm_backbone_state_dict"
            vocabulary_shape = list(
                backbone_state["backbone.vocab_embed.embedding"].shape
            )
        elif "embeddings.word_embeddings.weight" in backbone_state:
            interpretation = "online_huggingface_molecule_encoder_state_dict"
            vocabulary_shape = list(
                backbone_state["embeddings.word_embeddings.weight"].shape
            )
        else:
            raise ValueError(
                "Unrecognized ChemBERTa_state_dict payload; it is neither the "
                "archived MDLM backbone nor a supported online HF encoder"
            )
        optional_payloads["ChemBERTa_state_dict"] = {
            "interpretation": interpretation,
            "key_count": len(backbone_state),
            "vocab_embedding_shape": vocabulary_shape,
        }
    return {
        "molecule_dim": molecule_dim,
        "genome_dim": genome_dim,
        "text_dim": text_dim,
        "head_input_dim": head_input_dim,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "num_targets": num_targets,
        "checkpoint_keys": sorted(checkpoint),
        "optional_payloads": optional_payloads,
    }


def load_legacy_hierarchical_checkpoint(
    path: Path,
    *,
    device: torch.device = torch.device("cpu"),
    num_heads: int = 4,
    attention_dropout: float = 0.1,
    head_dropout: float = 0.2,
    mmap: bool = True,
) -> tuple[HierarchicalMicCheckpointComponents, dict]:
    """Load model tensors strictly while leaving the legacy optimizer unused.

    These are project-owned checkpoints and contain a NumPy scalar for ``R2``;
    PyTorch's restricted weights-only unpickler therefore cannot load them. The
    loader explicitly uses ``weights_only=False`` and must not be used on
    untrusted files.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )
    contract = inspect_checkpoint_contract(checkpoint)
    genome_attention = FirstTokenAttentionGenome(
        contract["molecule_dim"],
        contract["genome_dim"],
        num_heads,
        attention_dropout,
    )
    text_attention = FirstTokenAttentionGenome(
        contract["molecule_dim"],
        contract["text_dim"],
        num_heads,
        attention_dropout,
    )
    regression_head = RegressionHead(
        contract["head_input_dim"],
        contract["hidden_dim_1"],
        contract["hidden_dim_2"],
        contract["num_targets"],
        head_dropout,
    )
    classification_head = RegressionHead(
        contract["head_input_dim"],
        contract["hidden_dim_1"],
        contract["hidden_dim_2"],
        contract["num_targets"],
        head_dropout,
    )
    genome_attention.load_state_dict(checkpoint["co_cross_attn_genome"], strict=True)
    text_attention.load_state_dict(checkpoint["co_cross_attn_text"], strict=True)
    regression_head.load_state_dict(checkpoint["re_head_state_dict"], strict=True)
    classification_head.load_state_dict(checkpoint["cls_head_state_dict"], strict=True)
    components = HierarchicalMicCheckpointComponents(
        regression_head=regression_head,
        classification_head=classification_head,
        genome_attention=genome_attention,
        text_attention=text_attention,
        missing_genome_embedding=nn.Parameter(
            checkpoint["learnable_embedding_weight"].detach().clone()
        ),
        archived_r2=float(checkpoint["R2"]),
    ).to(device)
    return components, contract


def load_hierarchical_inference_checkpoint(
    path: Path,
    *,
    device: torch.device = torch.device("cpu"),
    num_heads: int = 4,
    attention_dropout: float = 0.1,
    head_dropout: float = 0.2,
    mmap: bool = True,
) -> tuple[HierarchicalMicCheckpointComponents, dict]:
    """Load a provenance-bearing inference-only copy of a legacy checkpoint."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )
    missing = INFERENCE_CHECKPOINT_KEYS - set(checkpoint)
    unexpected = set(checkpoint) - INFERENCE_CHECKPOINT_KEYS
    if missing or unexpected:
        raise ValueError(
            "hierarchical MIC inference checkpoint contract mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if checkpoint["format"] != "apexoracle_hierarchical_mic_inference_v1":
        raise ValueError(f"Unsupported inference checkpoint format: {checkpoint['format']}")

    synthetic = {
        "R2": checkpoint["archived_r2"],
        "optimizer_state_dict": {},
        "re_head_state_dict": checkpoint["re_head_state_dict"],
        "cls_head_state_dict": checkpoint["re_head_state_dict"],
        "co_cross_attn_genome": checkpoint["co_cross_attn_genome"],
        "co_cross_attn_text": checkpoint["co_cross_attn_text"],
        "learnable_embedding_weight": checkpoint["learnable_embedding_weight"],
    }
    contract = inspect_checkpoint_contract(synthetic)
    genome_attention = FirstTokenAttentionGenome(
        contract["molecule_dim"],
        contract["genome_dim"],
        num_heads,
        attention_dropout,
    )
    text_attention = FirstTokenAttentionGenome(
        contract["molecule_dim"],
        contract["text_dim"],
        num_heads,
        attention_dropout,
    )
    regression_head = RegressionHead(
        contract["head_input_dim"],
        contract["hidden_dim_1"],
        contract["hidden_dim_2"],
        contract["num_targets"],
        head_dropout,
    )
    classification_head = RegressionHead(
        contract["head_input_dim"],
        contract["hidden_dim_1"],
        contract["hidden_dim_2"],
        contract["num_targets"],
        head_dropout,
    )
    genome_attention.load_state_dict(checkpoint["co_cross_attn_genome"], strict=True)
    text_attention.load_state_dict(checkpoint["co_cross_attn_text"], strict=True)
    regression_head.load_state_dict(checkpoint["re_head_state_dict"], strict=True)
    classification_head.load_state_dict(checkpoint["re_head_state_dict"], strict=True)
    components = HierarchicalMicCheckpointComponents(
        regression_head=regression_head,
        classification_head=classification_head,
        genome_attention=genome_attention,
        text_attention=text_attention,
        missing_genome_embedding=nn.Parameter(
            checkpoint["learnable_embedding_weight"].detach().clone()
        ),
        archived_r2=float(checkpoint["archived_r2"]),
    ).to(device)
    contract["inference_checkpoint_provenance"] = {
        key: checkpoint[key]
        for key in (
            "source_checkpoint",
            "source_checkpoint_size",
            "source_checkpoint_sha256",
        )
    }
    return components, contract


def predict_genome_text(
    components: HierarchicalMicCheckpointComponents,
    molecule_embedding: torch.Tensor,
    genome_embeddings: torch.Tensor,
    genome_padding_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_padding_mask: torch.Tensor,
) -> torch.Tensor:
    genome = components.genome_attention(
        molecule_embedding, genome_embeddings, genome_padding_mask
    )
    text = components.text_attention(
        molecule_embedding, text_embeddings, text_padding_mask
    )
    fused = torch.cat(
        (genome.reshape(-1, genome.shape[-1]), text.reshape(-1, text.shape[-1])), dim=1
    )
    return components.regression_head(fused)


def predict_text_only(
    components: HierarchicalMicCheckpointComponents,
    molecule_embedding: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_padding_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = molecule_embedding.shape[0]
    genome_embeddings = components.missing_genome_embedding[:, None, :].expand(
        batch_size, 1, -1
    )
    genome_padding_mask = torch.zeros(
        (batch_size, 1), dtype=torch.bool, device=molecule_embedding.device
    )
    return predict_genome_text(
        components,
        molecule_embedding,
        genome_embeddings,
        genome_padding_mask,
        text_embeddings,
        text_padding_mask,
    )


StrainwiseCheckpointComponents = HierarchicalMicCheckpointComponents
load_legacy_strainwise_checkpoint = load_legacy_hierarchical_checkpoint
