"""Minimal PeptideCLM SMILES SPE tokenizer bundle."""

from pathlib import Path

from .tokenizer import SMILES_SPE_Tokenizer


RESOURCE_ROOT = Path(__file__).resolve().parent


def load_tokenizer() -> SMILES_SPE_Tokenizer:
    return SMILES_SPE_Tokenizer(
        RESOURCE_ROOT / "new_vocab.txt",
        RESOURCE_ROOT / "new_splits.txt",
    )


__all__ = ["SMILES_SPE_Tokenizer", "load_tokenizer"]
