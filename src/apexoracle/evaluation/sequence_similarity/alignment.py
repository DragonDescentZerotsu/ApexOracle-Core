"""Pairwise-alignment primitives for the paper sequence-similarity analysis.

The implementation intentionally retains the legacy conventions: global
alignment, the first optimal alignment returned by Biopython, exact-character
matches in the final gapped alignment, and case-sensitive D-amino acids.
"""

from __future__ import annotations

from dataclasses import dataclass

from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Align.substitution_matrices import Array


SCORING_SCHEMES = ("blosum62_needle", "exact_match_needle")
BASE_BLOSUM62 = substitution_matrices.load("BLOSUM62")


@dataclass(frozen=True)
class AlignmentMetrics:
    matches: int
    aligned_positions_including_gaps: int
    pid: float
    max_len_identity: float
    gapped_target: str | None = None
    gapped_query: str | None = None


def build_chirality_aware_blosum62() -> Array:
    """Extend BLOSUM62 with lowercase residues representing D-amino acids."""

    extended_alphabet = BASE_BLOSUM62.alphabet + "".join(
        residue.lower()
        for residue in BASE_BLOSUM62.alphabet
        if residue.isalpha() and residue.lower() not in BASE_BLOSUM62.alphabet
    )
    matrix = Array(alphabet=extended_alphabet, dims=2)
    for left_residue in extended_alphabet:
        for right_residue in extended_alphabet:
            base_score = float(
                BASE_BLOSUM62[left_residue.upper(), right_residue.upper()]
            )
            mixed_chirality = (
                left_residue.isalpha()
                and right_residue.isalpha()
                and left_residue.islower() != right_residue.islower()
            )
            matrix[left_residue, right_residue] = (
                min(0.0, base_score) if mixed_chirality else base_score
            )
    return matrix


CHIRALITY_AWARE_BLOSUM62 = build_chirality_aware_blosum62()
CHIRALITY_AWARE_ALPHABET = frozenset(CHIRALITY_AWARE_BLOSUM62.alphabet)


def build_aligner(scheme_name: str) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner(mode="global")
    if scheme_name == "blosum62_needle":
        aligner.substitution_matrix = CHIRALITY_AWARE_BLOSUM62
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5
    elif scheme_name == "exact_match_needle":
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = 0.0
        aligner.extend_gap_score = 0.0
    else:
        raise ValueError(f"Unsupported scoring scheme: {scheme_name}")
    return aligner


def sanitize_sequence_for_scheme(sequence: str, scoring_scheme: str) -> str:
    if scoring_scheme != "blosum62_needle":
        return sequence
    return "".join(
        (
            residue
            if residue in CHIRALITY_AWARE_ALPHABET
            else "x" if residue.islower() else "X"
        )
        for residue in sequence
    )


def rotate_sequence(sequence: str, rotation: int) -> str:
    if not sequence:
        return sequence
    rotation %= len(sequence)
    return sequence[rotation:] + sequence[:rotation]


def reconstruct_gapped_sequences(
    alignment: Align.Alignment,
    target: str,
    query: str,
) -> tuple[str, str]:
    """Reconstruct the two strings without parsing Biopython display text."""

    target_blocks, query_blocks = alignment.aligned
    target_parts: list[str] = []
    query_parts: list[str] = []
    target_position = 0
    query_position = 0

    for (target_start, target_end), (query_start, query_end) in zip(
        target_blocks, query_blocks
    ):
        if target_position < target_start:
            segment = target[target_position:target_start]
            target_parts.append(segment)
            query_parts.append("-" * len(segment))
            target_position = target_start
        if query_position < query_start:
            segment = query[query_position:query_start]
            target_parts.append("-" * len(segment))
            query_parts.append(segment)
            query_position = query_start

        target_parts.append(target[target_start:target_end])
        query_parts.append(query[query_start:query_end])
        target_position = target_end
        query_position = query_end

    if target_position < len(target):
        segment = target[target_position:]
        target_parts.append(segment)
        query_parts.append("-" * len(segment))
    if query_position < len(query):
        segment = query[query_position:]
        target_parts.append("-" * len(segment))
        query_parts.append(segment)

    gapped_target = "".join(target_parts)
    gapped_query = "".join(query_parts)
    if len(gapped_target) != len(gapped_query):
        raise ValueError("Aligned sequences must have the same gapped length")
    return gapped_target, gapped_query


def compute_alignment(
    target_sequence: str,
    query_sequence: str,
    scoring_scheme: str,
    *,
    aligner: Align.PairwiseAligner | None = None,
    include_gapped: bool = False,
) -> AlignmentMetrics:
    selected_aligner = aligner or build_aligner(scoring_scheme)
    sanitized_target = sanitize_sequence_for_scheme(target_sequence, scoring_scheme)
    sanitized_query = sanitize_sequence_for_scheme(query_sequence, scoring_scheme)
    alignment = selected_aligner.align(sanitized_target, sanitized_query)[0]
    gapped_target, gapped_query = reconstruct_gapped_sequences(
        alignment, target_sequence, query_sequence
    )
    matches = sum(
        1
        for target_residue, query_residue in zip(gapped_target, gapped_query)
        if target_residue == query_residue and target_residue != "-"
    )
    aligned_positions = len(gapped_target)
    max_length = max(len(target_sequence), len(query_sequence))
    return AlignmentMetrics(
        matches=matches,
        aligned_positions_including_gaps=aligned_positions,
        pid=matches / aligned_positions if aligned_positions else 0.0,
        max_len_identity=matches / max_length if max_length else 0.0,
        gapped_target=gapped_target if include_gapped else None,
        gapped_query=gapped_query if include_gapped else None,
    )


def ranking_key(
    row: dict,
    primary_metric: str,
    *,
    include_rotations: bool,
) -> tuple[float, ...]:
    secondary_metric = "max_len_identity" if primary_metric == "pid" else "pid"
    values: tuple[float, ...] = (
        float(row[primary_metric]),
        float(row[secondary_metric]),
        float(row["matches"]),
    )
    if include_rotations:
        values += (-float(row["query_rotation"]), -float(row["train_rotation"]))
    return values
