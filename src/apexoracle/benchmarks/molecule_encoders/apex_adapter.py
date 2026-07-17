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
    """Return the legacy APEX vocabulary extended by an explicit X token."""

    tokens = (PAD_TOKEN, START_TOKEN, END_TOKEN, *CANONICAL_RESIDUES, UNKNOWN_RESIDUE)
    word_to_index = {token: index for index, token in enumerate(tokens)}
    index_to_word = {index: token for token, index in word_to_index.items()}
    return word_to_index, index_to_word


def extend_aaindex_with_unknown(legacy_embedding: np.ndarray) -> np.ndarray:
    """Append an X vector equal to the mean of canonical AAindex vectors.

    The historical matrix has rows 0--2 for pad/start/end followed by the 20
    canonical residues. It is kept frozen by APEX, so an explicit deterministic
    X vector is required before training.
    """

    embedding = np.asarray(legacy_embedding)
    expected_rows = 3 + len(CANONICAL_RESIDUES)
    if embedding.ndim != 2 or embedding.shape[0] != expected_rows:
        raise ValueError(
            f"legacy APEX embedding must have shape ({expected_rows}, feature_dim); "
            f"received {embedding.shape}"
        )
    unknown = embedding[3:expected_rows].mean(axis=0, keepdims=True)
    return np.concatenate((embedding, unknown), axis=0)


def encode_apex_sequences(
    sequences: Sequence[str],
    *,
    word_to_index: Optional[Mapping[str, int]] = None,
    max_length: int = APEX_MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode projected sequences without silently dropping or padding X.

    Returns token IDs and an attention mask. Content is deterministically
    truncated to ``max_length - 2`` so start and end tokens are always present.
    Any symbol outside the canonical vocabulary is treated as X.
    """

    if max_length < 3:
        raise ValueError("max_length must leave room for start, content and end tokens")
    vocabulary = dict(word_to_index or build_apex_vocabulary()[0])
    required_tokens = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_RESIDUE)
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
                content.append(vocabulary[UNKNOWN_RESIDUE])
        encoded = [vocabulary[START_TOKEN], *content, vocabulary[END_TOKEN]]
        token_ids[row_index, : len(encoded)] = encoded
        attention_mask[row_index, : len(encoded)] = 1

    return token_ids, attention_mask
