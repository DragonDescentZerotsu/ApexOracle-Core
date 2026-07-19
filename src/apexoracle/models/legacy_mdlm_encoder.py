"""Thin adapter for the external MDLM backbone used by synergy guidance."""

from __future__ import annotations

from collections import OrderedDict
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_legacy_mdlm_config(mdlm_root: Path):
    """Load the external repository's default paper-era Hydra config."""

    try:
        from hydra import compose, initialize_config_dir
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MDLM execution requires the external repository's mdlm conda environment"
        ) from error

    with initialize_config_dir(
        version_base=None,
        config_dir=str(mdlm_root / "configs"),
    ):
        return compose(config_name="config")


class LegacyMDLMEncoder(nn.Module):
    """Frozen MDLM encoder preserving the legacy no-noise RNG consumption."""

    def __init__(
        self,
        *,
        mdlm_root: Path,
        config,
        vocab_size: int,
        checkpoint_path: Path,
        mask_token_id: int,
    ) -> None:
        super().__init__()
        if str(mdlm_root) not in sys.path:
            sys.path.insert(0, str(mdlm_root))
        import models
        import noise_schedule

        self.config = config
        self.mask_token_id = mask_token_id
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.backbone = models.dit.DIT(config, vocab_size=vocab_size)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        stripped = OrderedDict(
            (
                key[len("backbone.") :] if key.startswith("backbone.") else key,
                value,
            )
            for key, value in checkpoint["state_dict"].items()
        )
        incompatible = self.backbone.load_state_dict(stripped, strict=False)
        self.missing_keys = tuple(incompatible.missing_keys)
        self.unexpected_keys = tuple(incompatible.unexpected_keys)
        self.noise = noise_schedule.get_noise(config)

    def _process_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def _sample_t(
        self, batch_size: int, device: torch.device, *, noise_input: bool
    ) -> torch.Tensor:
        sampling_eps = 1e-3
        values = torch.rand(batch_size, device=device)
        values = (1 - sampling_eps) * values + sampling_eps
        return values if noise_input else values * 0

    def _apply_forward_noise(
        self, input_ids: torch.Tensor, move_chance: torch.Tensor
    ) -> torch.Tensor:
        move_indices = torch.rand(*input_ids.shape, device=input_ids.device) < move_chance
        noisy = torch.where(move_indices, self.mask_token_id, input_ids)
        noisy[input_ids == 3] = 3
        return noisy

    def forward(
        self, input_ids: torch.Tensor, *, noise_input: bool = False
    ) -> torch.Tensor:
        times = self._sample_t(
            input_ids.shape[0], input_ids.device, noise_input=noise_input
        )
        sigma, _ = self.noise(times)
        conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])
        tokens = self._apply_forward_noise(input_ids, move_chance)
        sigma = self._process_sigma(conditioning)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden = self.backbone.vocab_embed(tokens)
            conditioning_embedding = F.silu(self.backbone.sigma_map(sigma))
            rotary = self.backbone.rotary_emb(hidden)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for block in self.backbone.blocks:
                    hidden = block(
                        hidden,
                        rotary,
                        conditioning_embedding,
                        seqlens=None,
                    )
        return hidden

    def encode_pairs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode first then second molecules, matching the two legacy calls."""

        first = self(input_ids[::2], noise_input=False)[:, 0, :]
        second = self(input_ids[1::2], noise_input=False)[:, 0, :]
        interleaved = torch.empty(
            (input_ids.shape[0], first.shape[1]),
            dtype=first.dtype,
            device=first.device,
        )
        interleaved[::2] = first
        interleaved[1::2] = second
        return interleaved


def build_frozen_legacy_mdlm_encoder(
    *,
    mdlm_root: Path,
    checkpoint_path: Path,
    tokenizer,
    device: torch.device,
) -> LegacyMDLMEncoder:
    config = load_legacy_mdlm_config(mdlm_root)
    model = LegacyMDLMEncoder(
        mdlm_root=mdlm_root,
        config=config,
        vocab_size=len(tokenizer.get_vocab()),
        checkpoint_path=checkpoint_path,
        mask_token_id=tokenizer.mask_token_id,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
