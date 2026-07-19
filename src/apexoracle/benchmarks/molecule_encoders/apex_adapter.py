"""Deterministic APEX input adapter for the shared Fig. 2b benchmark."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


PAD_TOKEN = "0"
START_TOKEN = "1"
END_TOKEN = "2"
UNKNOWN_RESIDUE = "X"
CANONICAL_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
APEX_MAX_LENGTH = 52
APEX_MAX_CONTENT_LENGTH = APEX_MAX_LENGTH - 2


def build_apex_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    """Return the original, unchanged 23-token APEX vocabulary."""

    tokens = (PAD_TOKEN, START_TOKEN, END_TOKEN, *CANONICAL_RESIDUES)
    word_to_index = {token: index for index, token in enumerate(tokens)}
    index_to_word = {index: token for token, index in word_to_index.items()}
    return word_to_index, index_to_word


def encode_apex_sequences(
    sequences: Sequence[str],
    *,
    word_to_index: Optional[Mapping[str, int]] = None,
    max_length: int = APEX_MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode projected sequences with the original APEX behavior.

    Returns token IDs and an attention mask. Content is deterministically
    truncated to ``max_length - 2`` so start and end tokens are always present.
    APEX has no X token: X and every other unknown symbol remain index 0, exactly
    as in ``compare_APEX/utils.py::onehot_encoding``.  Keeping this behavior is
    necessary for comparison with the published, unmodified APEX model.
    """

    if max_length < 3:
        raise ValueError("max_length must leave room for start, content and end tokens")
    vocabulary = dict(word_to_index or build_apex_vocabulary()[0])
    required_tokens = (PAD_TOKEN, START_TOKEN, END_TOKEN)
    missing = [token for token in required_tokens if token not in vocabulary]
    if missing:
        raise ValueError(f"APEX vocabulary is missing required tokens: {missing}")

    token_ids = np.full(
        (len(sequences), max_length),
        fill_value=vocabulary[PAD_TOKEN],
        dtype=np.int64,
    )
    attention_mask = np.zeros((len(sequences), max_length), dtype=np.int64)
    content_limit = max_length - 2

    for row_index, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"sequence at index {row_index} is empty")
        content = []
        for symbol in sequence[:content_limit].upper():
            if symbol in CANONICAL_RESIDUES:
                content.append(vocabulary[symbol])
            else:
                content.append(vocabulary[PAD_TOKEN])
        encoded = [vocabulary[START_TOKEN], *content, vocabulary[END_TOKEN]]
        token_ids[row_index, : len(encoded)] = encoded
        attention_mask[row_index, : len(encoded)] = 1

    return token_ids, attention_mask


def legacy_onehot_encoding(
    sequences: Sequence[str], max_length: int, word_to_index: Mapping[str, int]
) -> np.ndarray:
    """Compatibility wrapper for the original ``onehot_encoding`` signature."""

    token_ids, _ = encode_apex_sequences(
        sequences,
        word_to_index=word_to_index,
        max_length=max_length,
    )
    return token_ids
