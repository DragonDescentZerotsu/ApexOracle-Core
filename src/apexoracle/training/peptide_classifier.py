"""Deployment-consistent peptide-classifier model and deterministic noise."""

from __future__ import annotations

import importlib
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def deterministic_mask(
    molecule_hashes: np.ndarray,
    length: int,
    *,
    replicate: int,
    move_chance: float,
) -> np.ndarray:
    """Create stateless masks invariant to batch size and DDP partitioning."""
    molecule_hashes = np.asarray(molecule_hashes, dtype=np.uint64).reshape(-1, 1)
    positions = np.arange(length, dtype=np.uint64).reshape(1, -1)
    replicate_key = np.uint64(
        ((replicate + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    )
    values = (
        molecule_hashes
        ^ (positions * np.uint64(0xD1B54A32D192ED03))
        ^ replicate_key
    )
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    threshold = np.uint64(move_chance * float(2**64 - 1))
    return values < threshold


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, 384)
        self.dense_2 = nn.Linear(384, 128)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.activation(self.dense_1(features)))
        features = self.dropout(self.activation(self.dense_2(features)))
        return self.out_proj(features)


class FrozenBackbonePeptideClassifier(nn.Module):
    """Exact v1 DDiT backbone plus a newly initialized binary head.

    The backbone weights and architecture come directly from the published v1
    classifier checkpoint. Unlike the historical training wrapper, sigma is
    retained so training matches the generation configuration.
    """

    def __init__(self, producer_root: Path, v1_classifier_checkpoint: Path):
        super().__init__()
        producer_root = producer_root.resolve()
        if str(producer_root) not in sys.path:
            sys.path.insert(0, str(producer_root))
        from omegaconf import OmegaConf

        models = importlib.import_module("models")
        noise_schedule = importlib.import_module("noise_schedule")
        checkpoint = torch.load(
            v1_classifier_checkpoint, map_location="cpu", weights_only=False
        )
        config = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
        config.time_conditioning = True
        self.config = config
        self.mask_index = int(checkpoint["hyper_parameters"]["mask_index"])
        self.pad_index = 3
        self.backbone = models.dit.DIT(config, vocab_size=3160)
        backbone_state = OrderedDict()
        prefix = "backbone.backbone."
        for key, value in checkpoint["state_dict"].items():
            if key.startswith(prefix):
                backbone_state[key.removeprefix(prefix)] = value
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        material_missing = [
            key for key in missing if not key.startswith(("output_layer.", "regression."))
        ]
        if material_missing or unexpected:
            raise RuntimeError(
                f"Backbone state mismatch: missing={material_missing}, unexpected={unexpected}"
            )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.noise = noise_schedule.get_noise(config)
        self.head = ClassificationHead(input_dim=int(config.model.hidden_size))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode(self, input_ids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                hidden = self.backbone.vocab_embed(input_ids)
                conditioning = F.silu(self.backbone.sigma_map(sigma.float()))
                rotary = self.backbone.rotary_emb(hidden)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    for block in self.backbone.blocks:
                        hidden = block(hidden, rotary, conditioning, seqlens=None)
        return hidden[:, 0, :].float()

    def logits_at_t(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        *,
        explicit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma, _ = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        if explicit_mask is None:
            mask = torch.rand(input_ids.shape, device=input_ids.device) < move_chance
        else:
            mask = explicit_mask
        mask &= input_ids != self.pad_index
        noised = torch.where(mask, self.mask_index, input_ids)
        return self.head(self._encode(noised, sigma)).squeeze(-1)

    def training_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        t = torch.rand(input_ids.shape[0], device=input_ids.device)
        t = 0.001 + 0.999 * t
        return self.logits_at_t(input_ids, t)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.training_logits(input_ids)


def move_chance_at_t(t: float, eps: float = 1e-3) -> float:
    sigma = -math.log1p(-(1 - eps) * t)
    return 1 - math.exp(-sigma)
