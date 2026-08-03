"""Shared helpers for guided-generation structural-diversity audits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from apexoracle.evaluation.sequence_similarity.alignment import (
    AlignmentMetrics,
    build_aligner,
    compute_alignment,
    ranking_key,
    rotate_sequence,
)


FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
FINGERPRINT_INCLUDE_CHIRALITY = True
SEQUENCE_SCORING_SCHEME = "blosum62_needle"


@dataclass(frozen=True)
class ReplacementRule:
    apexoracle_id: str
    precursor_sequence: str
    expected_occurrences: int


@dataclass(frozen=True)
class TopologyAwareSequenceAlignment:
    metrics: AlignmentMetrics
    left_rotation: int
    right_rotation: int
    orientation_swapped: bool


REPLACEMENT_RULES: tuple[ReplacementRule, ...] = (
    ReplacementRule("ApexOracle-2", "FImKCWYYWMVMKQRIMYRGT", 1),
    ReplacementRule("ApexOracle-5", "MIKLLIKLAIGYLRLQRGQPLLNPGKGAR", 1),
    ReplacementRule("ApexOracle-7", "KKKKLVAIENRKKYTVVLRNLAISRRGV", 1),
    ReplacementRule("ApexOracle-9", "IQKLKFLRLAAQAQKLLLKLGLARRSLASK", 1),
    ReplacementRule("ApexOracle-10", "MNLAAFFFIKNPPSKWKYKR", 1),
    ReplacementRule("ApexOracle-14", "cyclo-VLKAARHMRKLFRGHWVW", 2),
    ReplacementRule("ApexOracle-19", "IILLGVKKLQKNNVLQKEIKNANGKALVVA", 1),
    ReplacementRule("ApexOracle-21", "MRLSRRLLEWRRRLRIAIA", 1),
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_legacy_arg(sequence: str) -> str:
    """Translate the legacy arginine-tautomer token B to final-table R."""

    return sequence.replace("B", "R")


def split_topology(sequence: str) -> tuple[str, bool]:
    is_cyclic = sequence.startswith("cyclo-")
    return (sequence.removeprefix("cyclo-"), is_cyclic)


def cyclic_rotations(sequence: str) -> set[str]:
    if not sequence:
        return {sequence}
    return {sequence[offset:] + sequence[:offset] for offset in range(len(sequence))}


def topology_aware_equal(left: str, right: str) -> bool:
    left_body, left_cyclic = split_topology(left)
    right_body, right_cyclic = split_topology(right)
    if left_cyclic != right_cyclic:
        return False
    if left_cyclic:
        return left_body in cyclic_rotations(right_body)
    return left_body == right_body


def best_topology_aware_sequence_alignment(
    left_sequence: str,
    right_sequence: str,
    *,
    left_is_cyclic: bool,
    right_is_cyclic: bool,
) -> TopologyAwareSequenceAlignment | None:
    """Apply the paper sequence-similarity protocol to one peptide pair.

    Linear and cyclic peptides are not compared to each other. Linear pairs are
    aligned once. For cyclic pairs, all rotations of both sequences are aligned
    and ranked by PID using the same deterministic tie-breaking rule as the
    lead-versus-training-set analysis.
    """

    if left_is_cyclic != right_is_cyclic:
        return None

    def directional(
        target_sequence: str, query_sequence: str
    ) -> TopologyAwareSequenceAlignment:
        aligner = build_aligner(SEQUENCE_SCORING_SCHEME)
        if not left_is_cyclic:
            metrics = compute_alignment(
                target_sequence,
                query_sequence,
                SEQUENCE_SCORING_SCHEME,
                aligner=aligner,
                include_gapped=True,
            )
            return TopologyAwareSequenceAlignment(metrics, 0, 0, False)

        best_row: dict[str, float | int] | None = None
        for target_rotation in range(len(target_sequence)):
            rotated_target = rotate_sequence(target_sequence, target_rotation)
            for query_rotation in range(len(query_sequence)):
                rotated_query = rotate_sequence(query_sequence, query_rotation)
                metrics = compute_alignment(
                    rotated_target,
                    rotated_query,
                    SEQUENCE_SCORING_SCHEME,
                    aligner=aligner,
                )
                row: dict[str, float | int] = {
                    "pid": metrics.pid,
                    "max_len_identity": metrics.max_len_identity,
                    "matches": metrics.matches,
                    "query_rotation": target_rotation,
                    "train_rotation": query_rotation,
                }
                if best_row is None or ranking_key(
                    row, "pid", include_rotations=True
                ) > ranking_key(best_row, "pid", include_rotations=True):
                    best_row = row

        if best_row is None:
            raise ValueError("Cyclic peptide sequences must be non-empty")
        target_rotation = int(best_row["query_rotation"])
        query_rotation = int(best_row["train_rotation"])
        metrics = compute_alignment(
            rotate_sequence(target_sequence, target_rotation),
            rotate_sequence(query_sequence, query_rotation),
            SEQUENCE_SCORING_SCHEME,
            aligner=aligner,
            include_gapped=True,
        )
        return TopologyAwareSequenceAlignment(
            metrics, target_rotation, query_rotation, False
        )

    forward = directional(left_sequence, right_sequence)
    reverse = directional(right_sequence, left_sequence)
    forward_key = (
        forward.metrics.pid,
        forward.metrics.max_len_identity,
        forward.metrics.matches,
    )
    reverse_key = (
        reverse.metrics.pid,
        reverse.metrics.max_len_identity,
        reverse.metrics.matches,
    )
    if forward_key >= reverse_key:
        return forward
    reverse_metrics = AlignmentMetrics(
        matches=reverse.metrics.matches,
        aligned_positions_including_gaps=(
            reverse.metrics.aligned_positions_including_gaps
        ),
        pid=reverse.metrics.pid,
        max_len_identity=reverse.metrics.max_len_identity,
        gapped_target=reverse.metrics.gapped_query,
        gapped_query=reverse.metrics.gapped_target,
    )
    return TopologyAwareSequenceAlignment(
        reverse_metrics,
        reverse.right_rotation,
        reverse.left_rotation,
        True,
    )


def match_replacement_rule(sequence: str) -> ReplacementRule | None:
    normalized = normalize_legacy_arg(sequence)
    matches = [
        rule
        for rule in REPLACEMENT_RULES
        if topology_aware_equal(normalized, rule.precursor_sequence)
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous replacement mapping for {sequence!r}: {matches}")
    return matches[0] if matches else None


def canonical_isomeric_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def morgan_fingerprints(
    canonical_smiles: Sequence[str],
    *,
    radius: int = FINGERPRINT_RADIUS,
    n_bits: int = FINGERPRINT_BITS,
    include_chirality: bool = FINGERPRINT_INCLUDE_CHIRALITY,
) -> list[DataStructs.ExplicitBitVect]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    fingerprints: list[DataStructs.ExplicitBitVect] = []
    for row_index, smiles in enumerate(canonical_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse canonical SMILES at row {row_index}")
        fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints


def morgan_fingerprints_from_mols(
    molecules: Sequence[Chem.Mol],
    *,
    radius: int = FINGERPRINT_RADIUS,
    n_bits: int = FINGERPRINT_BITS,
    include_chirality: bool = FINGERPRINT_INCLUDE_CHIRALITY,
) -> list[DataStructs.ExplicitBitVect]:
    """Fingerprint already parsed molecules without a lossy SMILES reparse."""

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    return [generator.GetFingerprint(mol) for mol in molecules]


def all_pairwise_tanimoto(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    values: list[float] = []
    for query_index in range(len(fingerprints) - 1):
        targets = fingerprints[query_index + 1 :]
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[query_index], targets
        )
        for offset, similarity in enumerate(similarities, start=query_index + 1):
            left.append(query_index)
            right.append(offset)
            values.append(float(similarity))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def sample_distinct_ordered_pairs(
    population_size: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample ordered non-self pairs, with replacement."""

    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, population_size, size=sample_size, dtype=np.int64)
    compact_right = rng.integers(
        0, population_size - 1, size=sample_size, dtype=np.int64
    )
    right = compact_right + (compact_right >= left)
    return left, right


def sampled_tanimoto(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("left and right index arrays must have identical shapes")
    values = np.empty(left.size, dtype=np.float64)
    for index, (left_index, right_index) in enumerate(zip(left, right, strict=True)):
        values[index] = DataStructs.TanimotoSimilarity(
            fingerprints[int(left_index)], fingerprints[int(right_index)]
        )
    return values


def summarize_similarities(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        raise ValueError("At least one similarity value is required")
    summary: dict[str, float | int] = {
        "n_pairs": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
        key = f"fraction_ge_{str(threshold).replace('.', '_')}"
        summary[key] = float(np.mean(values >= threshold))
    return summary


def histogram_rows(
    values: np.ndarray,
    *,
    edges: Iterable[float] | None = None,
) -> list[dict[str, float | int]]:
    selected_edges = np.asarray(
        list(edges) if edges is not None else np.linspace(0.0, 1.0, 21),
        dtype=np.float64,
    )
    counts, selected_edges = np.histogram(values, bins=selected_edges)
    total = int(np.sum(counts))
    rows: list[dict[str, float | int]] = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "bin_left": float(selected_edges[index]),
                "bin_right": float(selected_edges[index + 1]),
                "bin_center": float(
                    (selected_edges[index] + selected_edges[index + 1]) / 2
                ),
                "count": int(count),
                "fraction": float(count / total),
            }
        )
    return rows
