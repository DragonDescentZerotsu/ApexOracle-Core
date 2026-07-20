"""K-mer genome features used by the strain-wise ablation.

The producer is deliberately separate from the training runner.  It reads FASTA
files and writes new ``.pt`` tensors, while the legacy consumer applies the
paper-era frozen random projection needed by the 8192-dimensional fusion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from apexoracle.data.genome_embeddings import parse_embedding_id


GLOBAL_ALPHABET = "ACGT"
WINDOWED_LEGACY_ALPHABET = "ATGC"
DEFAULT_K_VALUES = (4, 5, 6)


def feature_dimension(k_values: Sequence[int]) -> int:
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates")
    return sum(4**k for k in k_values)


@lru_cache(maxsize=2)
def _base_lookup(alphabet: str) -> np.ndarray:
    if len(alphabet) != 4 or set(alphabet) != set("ACGT"):
        raise ValueError("alphabet must be a permutation of ACGT")
    lookup = np.full(256, -1, dtype=np.int16)
    for index, base in enumerate(alphabet):
        lookup[ord(base)] = index
        lookup[ord(base.lower())] = index
    return lookup


def count_kmers(sequence: str, k: int, *, alphabet: str) -> np.ndarray:
    """Count overlapping canonical k-mers using base-4 lexical indices."""

    if k < 1:
        raise ValueError("k must be positive")
    encoded = np.frombuffer(
        sequence.encode("ascii", errors="ignore"), dtype=np.uint8
    )
    codes = _base_lookup(alphabet)[encoded]
    counts = np.zeros(4**k, dtype=np.int64)
    if codes.size < k:
        return counts

    window_count = codes.size - k + 1
    indices = np.zeros(window_count, dtype=np.int64)
    valid = np.ones(window_count, dtype=bool)
    for offset in range(k):
        values = codes[offset : offset + window_count]
        current_valid = values >= 0
        valid &= current_valid
        indices = indices * 4 + np.where(current_valid, values, 0)
    if valid.any():
        counts += np.bincount(indices[valid], minlength=4**k)
    return counts


def normalized_kmer_vector(
    sequence: str, k_values: Sequence[int], *, alphabet: str
) -> torch.Tensor:
    """Return concatenated per-k frequency vectors as float32."""

    feature_dimension(k_values)
    features = []
    for k in k_values:
        counts = count_kmers(sequence, k, alphabet=alphabet)
        total = int(counts.sum())
        values = (
            counts.astype(np.float32) / float(total)
            if total
            else np.zeros(4**k, dtype=np.float32)
        )
        features.append(values)
    return torch.from_numpy(np.concatenate(features))


def global_kmer_embedding(
    records: Iterable[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    *,
    alphabet: str = GLOBAL_ALPHABET,
) -> torch.Tensor:
    """Build one genome-wide vector without forming k-mers across contigs."""

    feature_dimension(k_values)
    totals = {k: np.zeros(4**k, dtype=np.int64) for k in k_values}
    for sequence in records:
        for k in k_values:
            totals[k] += count_kmers(sequence, k, alphabet=alphabet)

    features = []
    for k in k_values:
        counts = totals[k]
        total = int(counts.sum())
        features.append(
            counts.astype(np.float32) / float(total)
            if total
            else np.zeros(4**k, dtype=np.float32)
        )
    return torch.from_numpy(np.concatenate(features)).unsqueeze(0)


def windowed_kmer_embedding(
    records: Iterable[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    *,
    window_length: int = 11_000,
    step_length: int = 10_000,
    alphabet: str = WINDOWED_LEGACY_ALPHABET,
) -> torch.Tensor:
    """Build per-window vectors, resetting the window offset for each contig."""

    if window_length < 1 or step_length < 1:
        raise ValueError("window_length and step_length must be positive")
    rows = []
    for sequence in records:
        for start in range(0, len(sequence), step_length):
            rows.append(
                normalized_kmer_vector(
                    sequence[start : start + window_length],
                    k_values,
                    alphabet=alphabet,
                )
            )
    if not rows:
        return torch.zeros((1, feature_dimension(k_values)), dtype=torch.float32)
    return torch.stack(rows)


def fasta_records(path: Path) -> list[str]:
    """Read FASTA records while preserving their file order."""

    from Bio import SeqIO

    return [str(record.seq) for record in SeqIO.parse(path, "fasta")]


def extract_fasta_embedding(
    path: Path,
    *,
    mode: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    window_length: int = 11_000,
    step_length: int = 10_000,
) -> torch.Tensor:
    records = fasta_records(path)
    if mode == "global":
        return global_kmer_embedding(records, k_values)
    if mode == "windowed":
        return windowed_kmer_embedding(
            records,
            k_values,
            window_length=window_length,
            step_length=step_length,
        )
    raise ValueError(f"Unsupported k-mer mode: {mode}")


def extract_folder(
    genome_dir: Path,
    output_dir: Path,
    *,
    mode: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    window_length: int = 11_000,
    step_length: int = 10_000,
    output_dtype: torch.dtype | None = None,
) -> list[Path]:
    """Extract a new tensor set without overwriting any existing output."""

    genome_dir = genome_dir.resolve()
    output_dir = output_dir.resolve()
    if genome_dir == output_dir or genome_dir in output_dir.parents:
        raise ValueError("output_dir must be outside the source genome directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")

    genome_files = sorted((*genome_dir.glob("*.fasta"), *genome_dir.glob("*.fa")))
    if not genome_files:
        raise FileNotFoundError(f"No FASTA files found in {genome_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = output_dtype or (
        torch.bfloat16 if mode == "global" else torch.float32
    )
    outputs = []
    for genome_path in genome_files:
        tensor = extract_fasta_embedding(
            genome_path,
            mode=mode,
            k_values=k_values,
            window_length=window_length,
            step_length=step_length,
        ).to(dtype=dtype)
        output_path = output_dir / f"{genome_path.stem}.pt"
        torch.save(tensor, output_path)
        outputs.append(output_path)
    return outputs


class LegacyFrozenKmerProjection(nn.Module):
    """The untrained 5376->8192 projection used by the 2026 reconstruction."""

    def __init__(self, input_dim: int = 5_376, output_dim: int = 8_192):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        self.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


def load_legacy_projected_embeddings(
    embedding_dir: Path,
    *,
    input_dim: int,
    output_dim: int,
    scale: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], LegacyFrozenKmerProjection]:
    """Preserve the reconstruction's one-time frozen random projection."""

    projection = LegacyFrozenKmerProjection(input_dim, output_dim).to(
        device=device, dtype=torch.bfloat16
    )
    projection.eval()
    embeddings = {}
    with torch.no_grad():
        for path in (path for path in embedding_dir.iterdir() if path.is_file()):
            values = torch.load(path, map_location=device, weights_only=True)
            if not isinstance(values, torch.Tensor) or values.ndim != 2:
                raise TypeError(f"Expected a rank-2 tensor in {path}")
            if values.shape[1] != input_dim:
                raise ValueError(
                    f"Unexpected k-mer dimension in {path}: {values.shape[1]}"
                )
            embeddings[parse_embedding_id(path.name)] = (
                projection(values.to(dtype=torch.bfloat16)) * scale
            ).to(dtype=torch.bfloat16)
    return embeddings, projection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class KmerTensorIdentity:
    genome_id: str
    file: str
    bytes: int
    sha256: str
    shape: str
    dtype: str
    per_row_sum_min: float
    per_row_sum_max: float


def inspect_tensor(path: Path) -> KmerTensorIdentity:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise TypeError(f"Expected a rank-2 tensor in {path}")
    row_sums = values.float().sum(dim=1)
    return KmerTensorIdentity(
        genome_id=parse_embedding_id(path.name),
        file=path.name,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        shape="x".join(str(dim) for dim in values.shape),
        dtype=str(values.dtype),
        per_row_sum_min=float(row_sums.min()),
        per_row_sum_max=float(row_sums.max()),
    )
