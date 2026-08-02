"""Deterministic split utilities for the reviewer peptide-classifier experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


TRAIN = np.uint8(0)
VALIDATION = np.uint8(1)
TEST = np.uint8(2)


def stable_digest(value: str | bytes, *, digest_size: int = 16) -> bytes:
    """Return a version-stable BLAKE2b digest."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.blake2b(value, digest_size=digest_size).digest()


def split_from_group(group: bytes, *, seed: int = 20260726) -> np.uint8:
    """Map an indivisible group to the deterministic 98/1/1 split."""
    digest = hashlib.blake2b(
        group,
        digest_size=8,
        person=b"pepcls-v1",
        salt=seed.to_bytes(16, "little", signed=False),
    ).digest()
    bucket = int.from_bytes(digest, "little") % 10_000
    if bucket < 100:
        return TEST
    if bucket < 200:
        return VALIDATION
    return TRAIN


def source_from_mol_id(mol_id: str) -> str:
    if mol_id.startswith("SmProt2_"):
        return "smprot2"
    if mol_id.startswith("uni_"):
        return "uniprot_uniref"
    if mol_id.startswith("Generated_pep_CLM_"):
        return "generated_peptideclm"
    if mol_id.startswith("pubchem_"):
        return "pubchem"
    raise ValueError(f"Unknown v1 molecule ID prefix: {mol_id}")


def sequence_index_from_mol_id(mol_id: str) -> tuple[str, int] | None:
    if mol_id.startswith("SmProt2_"):
        return "smprot2", int(mol_id.removeprefix("SmProt2_"))
    if mol_id.startswith("uni_"):
        return "uniprot_uniref", int(mol_id.removeprefix("uni_"))
    return None


@dataclass
class UnionFind:
    """Compact union-find used to merge sequence clusters sharing a molecule."""

    parent: np.ndarray
    rank: np.ndarray

    @classmethod
    def create(cls, size: int) -> "UnionFind":
        return cls(
            parent=np.arange(size, dtype=np.int64),
            rank=np.zeros(size, dtype=np.uint8),
        )

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            nxt = int(parent[item])
            parent[item] = root
            item = nxt
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


def molecule_u64(canonical_digest: bytes) -> np.uint64:
    if len(canonical_digest) < 8:
        raise ValueError("Canonical digest must contain at least eight bytes")
    return np.uint64(int.from_bytes(canonical_digest[:8], "little"))


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).copy()
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def split_codes_for_digests(
    digests: np.ndarray,
    *,
    seed: int,
    real_canonicals: np.ndarray,
    real_sequence_roots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return split codes and molecule hashes for a chunk of canonical digests."""
    digests = np.asarray(digests, dtype="V16")
    molecule_ids = digests.view("<u8").reshape(-1, 2)[:, 0].copy()
    group_values = molecule_ids ^ np.uint64(seed)
    positions = np.searchsorted(real_canonicals, digests)
    valid = positions < len(real_canonicals)
    matched = np.zeros(len(digests), dtype=bool)
    matched[valid] = real_canonicals[positions[valid]] == digests[valid]
    group_values[matched] = (
        real_sequence_roots[positions[matched]]
        ^ np.uint64(seed)
        ^ np.uint64(0xD1B54A32D192ED03)
    )
    buckets = splitmix64_array(group_values) % np.uint64(10_000)
    splits = np.full(len(digests), TRAIN, dtype=np.uint8)
    splits[buckets < 200] = VALIDATION
    splits[buckets < 100] = TEST
    return splits, molecule_ids
