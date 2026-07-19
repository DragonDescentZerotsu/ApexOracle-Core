"""Online Hugging Face molecule encoders used by Fig. 2c comparators."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn


@dataclass(frozen=True)
class HFMoleculeEncoderConfig:
    """Behavioral contract for one legacy online molecule encoder."""

    name: str
    model_name: str
    revision: str
    hidden_size: int
    tokenizer_kind: str
    trust_remote_code: bool
    initial_mode: str
    pooling: str
    max_length: int
    optimizer_learning_rate: float
    optimizer_weight_decay_multiplier: float
    checkpoint_state_key: str

    def validate(self) -> None:
        if self.initial_mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported initial_mode: {self.initial_mode}")
        if self.pooling != "first_token":
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        if self.tokenizer_kind not in {"auto", "vendored_peptideclm"}:
            raise ValueError(f"Unsupported tokenizer_kind: {self.tokenizer_kind}")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")


def load_legacy_tokenizer(config: HFMoleculeEncoderConfig, repo_root: Path):
    """Load the exact tokenizer family used by the legacy comparator."""

    config.validate()
    if config.tokenizer_kind == "vendored_peptideclm":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from PeptideCLM.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

        tokenizer_root = repo_root / "PeptideCLM" / "tokenizer"
        return SMILES_SPE_Tokenizer(
            tokenizer_root / "new_vocab.txt",
            tokenizer_root / "new_splits.txt",
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )


def load_legacy_molecule_encoder(
    config: HFMoleculeEncoderConfig,
    *,
    device,
) -> nn.Module:
    """Load, mode-set, and freeze one online backbone exactly as legacy code."""

    config.validate()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        config.model_name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if int(model.config.hidden_size) != config.hidden_size:
        raise ValueError(
            f"{config.name} hidden size changed: "
            f"expected {config.hidden_size}, got {model.config.hidden_size}"
        )
    model.to(device)
    if config.initial_mode == "eval":
        model.eval()
    else:
        model.train()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def unfreeze_legacy_molecule_encoder(model: nn.Module) -> None:
    """Match the legacy epoch-boundary requires_grad transition."""

    for parameter in model.parameters():
        parameter.requires_grad = True
