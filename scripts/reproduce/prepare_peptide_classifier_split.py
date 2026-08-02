#!/usr/bin/env python
"""Build a deterministic molecule/sequence-disjoint v1 classifier split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
from rdkit import Chem, RDLogger, rdBase

from apexoracle.data.peptide_classifier import (
    TEST,
    TRAIN,
    VALIDATION,
    UnionFind,
    molecule_u64,
    sequence_index_from_mol_id,
    split_codes_for_digests,
    source_from_mol_id,
    stable_digest,
)

RDLogger.DisableLog("rdApp.*")


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def extract_sequences(args: argparse.Namespace) -> None:
    output = args.output_dir / "real_peptides.fasta"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counts = Counter()
    with output.open("w") as dst:
        with args.uniprot_sequences.open() as src:
            for index, line in enumerate(src):
                sequence = line.strip().upper()
                if sequence:
                    dst.write(f">uni_{index}\n{sequence}\n")
                    counts["uniprot_uniref"] += 1
        with args.smprot_csv.open(newline="") as src:
            for index, row in enumerate(csv.DictReader(src)):
                sequence = row["seq"].strip().upper()
                if sequence:
                    dst.write(f">SmProt2_{index}\n{sequence}\n")
                    counts["smprot2"] += 1
    _write_json(
        args.output_dir / "sequence_extraction.json",
        {
            "counts": dict(counts),
            "elapsed_seconds": time.time() - started,
            "output": str(output),
            "output_sha256": _sha256(output),
            "smprot_csv": str(args.smprot_csv),
            "smprot_csv_sha256": _sha256(args.smprot_csv),
            "uniprot_sequences": str(args.uniprot_sequences),
            "uniprot_sequences_sha256": _sha256(args.uniprot_sequences),
        },
    )


def cluster_sequences(args: argparse.Namespace) -> None:
    fasta = args.output_dir / "real_peptides.fasta"
    prefix = args.output_dir / "mmseqs_clusters"
    tmp = args.output_dir / "mmseqs_tmp"
    command = [
        str(args.mmseqs),
        "easy-linclust",
        str(fasta),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        str(args.min_seq_id),
        "-c",
        str(args.coverage),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "2",
        "--threads",
        str(args.threads),
        "--remove-tmp-files",
        "1",
    ]
    started = time.time()
    subprocess.run(command, check=True)
    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    _write_json(
        args.output_dir / "sequence_clustering.json",
        {
            "cluster_tsv": str(cluster_tsv),
            "cluster_tsv_sha256": _sha256(cluster_tsv),
            "command": command,
            "coverage": args.coverage,
            "elapsed_seconds": time.time() - started,
            "min_sequence_identity": args.min_seq_id,
            "mmseqs_version": subprocess.check_output(
                [str(args.mmseqs), "version"], text=True
            ).strip(),
        },
    )


def _arrow_shards(dataset_dir: Path) -> list[Path]:
    state = json.loads((dataset_dir / "state.json").read_text())
    return [dataset_dir / item["filename"] for item in state["_data_files"]]


def _iter_final_rows(
    dataset_dir: Path, *, wait_for_shards_seconds: int = 0
) -> Iterator[tuple[str, int]]:
    for shard in _arrow_shards(dataset_dir):
        deadline = time.time() + wait_for_shards_seconds
        while not shard.exists():
            if wait_for_shards_seconds <= 0 or time.time() >= deadline:
                raise FileNotFoundError(shard)
            time.sleep(5)
        with pa.memory_map(str(shard), "r") as source:
            for batch in pa.ipc.open_stream(source):
                ids = batch.column(batch.schema.get_field_index("mol_ids")).to_pylist()
                labels = batch.column(batch.schema.get_field_index("labels")).to_numpy()
                if len(ids) != len(labels):
                    raise RuntimeError(f"Arrow ID/label length mismatch in {shard}")
                yield from zip(ids, labels)


def _iter_final_batches(
    dataset_dir: Path, *, wait_for_shards_seconds: int = 0
) -> Iterator[tuple[list[str], np.ndarray]]:
    for shard in _arrow_shards(dataset_dir):
        deadline = time.time() + wait_for_shards_seconds
        while not shard.exists():
            if wait_for_shards_seconds <= 0 or time.time() >= deadline:
                raise FileNotFoundError(shard)
            time.sleep(5)
        with pa.memory_map(str(shard), "r") as source:
            for batch in pa.ipc.open_stream(source):
                ids = batch.column(batch.schema.get_field_index("mol_ids")).to_pylist()
                labels = batch.column(batch.schema.get_field_index("labels")).to_numpy()
                if len(ids) != len(labels):
                    raise RuntimeError(f"Arrow ID/label length mismatch in {shard}")
                yield ids, labels


def _canonical_digest(smiles: str) -> bytes | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return stable_digest(canonical)


def _matched_smiles(
    raw_csv: Path,
    dataset_dir: Path,
    limit: int | None,
    wait_for_shards_seconds: int = 0,
) -> Iterator[tuple[str, int, str]]:
    final = _iter_final_rows(
        dataset_dir, wait_for_shards_seconds=wait_for_shards_seconds
    )
    try:
        wanted_id, wanted_label = next(final)
    except StopIteration:
        return
    emitted = 0
    with raw_csv.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["ID"] != wanted_id:
                continue
            label = int(row["label"])
            if label != int(wanted_label):
                raise RuntimeError(
                    f"Label mismatch for {wanted_id}: raw={label}, Arrow={wanted_label}"
                )
            yield wanted_id, label, row["SMILES"]
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            try:
                wanted_id, wanted_label = next(final)
            except StopIteration:
                return
    raise RuntimeError(f"Final Arrow ID {wanted_id!r} was not found in raw CSV order")


def _load_cluster_arrays(
    cluster_tsv: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    max_smprot = -1
    max_uni = -1
    representatives: dict[str, int] = {}
    with cluster_tsv.open() as source:
        for line in source:
            representative, member = line.rstrip("\n").split("\t")
            if member.startswith("SmProt2_"):
                max_smprot = max(max_smprot, int(member.removeprefix("SmProt2_")))
            elif member.startswith("uni_"):
                max_uni = max(max_uni, int(member.removeprefix("uni_")))
            else:
                raise ValueError(f"Unexpected sequence ID: {member}")
            if representative not in representatives:
                representatives[representative] = len(representatives)
    missing = np.uint32(2**32 - 1)
    smprot = np.full(max_smprot + 1, missing, dtype=np.uint32)
    uni = np.full(max_uni + 1, missing, dtype=np.uint32)
    with cluster_tsv.open() as source:
        for line in source:
            representative, member = line.rstrip("\n").split("\t")
            cluster = np.uint32(representatives[representative])
            if member.startswith("SmProt2_"):
                smprot[int(member.removeprefix("SmProt2_"))] = cluster
            else:
                uni[int(member.removeprefix("uni_"))] = cluster
    names = [None] * len(representatives)
    for name, index in representatives.items():
        names[index] = name
    return smprot, uni, names


def _cluster_for_id(mol_id: str, smprot: np.ndarray, uni: np.ndarray) -> int | None:
    parsed = sequence_index_from_mol_id(mol_id)
    if parsed is None:
        return None
    source, index = parsed
    array = smprot if source == "smprot2" else uni
    if index >= len(array) or array[index] == np.uint32(2**32 - 1):
        raise RuntimeError(f"No MMseqs cluster assignment for {mol_id}")
    return int(array[index])


SOURCE_CAPACITIES = {
    "smprot2": 825_631,
    "uniprot_uniref": 3_105_732,
    "generated_peptideclm": 9_999_999,
    "pubchem": 111_378_206,
}


def _source_index(mol_id: str) -> tuple[str, int]:
    source = source_from_mol_id(mol_id)
    prefixes = {
        "smprot2": "SmProt2_",
        "uniprot_uniref": "uni_",
        "generated_peptideclm": "Generated_pep_CLM_",
        "pubchem": "pubchem_",
    }
    index = int(mol_id.removeprefix(prefixes[source]))
    if not 0 <= index < SOURCE_CAPACITIES[source]:
        raise RuntimeError(
            f"{mol_id} exceeds frozen v1 source capacity {SOURCE_CAPACITIES[source]}"
        )
    return source, index


def assign_splits(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest_path = args.output_dir / "canonical_digest_128.bin"
    label_path = args.output_dir / "labels.u1"
    source_path = args.output_dir / "sources.u1"
    split_path = args.output_dir / "split_codes.u1"
    molecule_path = args.output_dir / "molecule_hashes.u8"
    real_canonical_path = args.output_dir / "real_canonical_digest_128.bin"
    real_root_path = args.output_dir / "real_sequence_roots.u8"
    smprot, uni, representatives = _load_cluster_arrays(args.cluster_tsv)
    union_find = UnionFind.create(len(representatives))
    canonical_to_cluster: dict[bytes, int] = {}
    counts = Counter()
    started = time.time()
    workers = args.workers or max(1, len(os.sched_getaffinity(0)) - 2)
    context = mp.get_context(args.mp_start_method)
    lookup_dir = args.output_dir / "raw_canonical_lookup"
    lookup_dir.mkdir(parents=True, exist_ok=True)
    lookups = {
        source: np.memmap(
            lookup_dir / f"{source}.v16",
            mode="w+",
            dtype="V16",
            shape=(capacity,),
        )
        for source, capacity in SOURCE_CAPACITIES.items()
    }
    fallback_lookups = {
        source: np.memmap(
            lookup_dir / f"{source}.fallback.u1",
            mode="w+",
            dtype=np.uint8,
            shape=(capacity,),
        )
        for source, capacity in SOURCE_CAPACITIES.items()
    }
    with (
        context.Pool(workers) as pool,
    ):
        def consume_raw(batch: list[tuple[str, int, str]]) -> None:
            smiles = [item[2] for item in batch]
            digests = pool.map(
                _canonical_digest,
                smiles,
                chunksize=args.canonical_chunk_size,
            )
            grouped_indices: dict[str, list[int]] = {
                source: [] for source in SOURCE_CAPACITIES
            }
            grouped_digests: dict[str, list[bytes]] = {
                source: [] for source in SOURCE_CAPACITIES
            }
            grouped_fallback_indices: dict[str, list[int]] = {
                source: [] for source in SOURCE_CAPACITIES
            }
            for (mol_id, label, raw_smiles), digest in zip(batch, digests):
                fallback = digest is None
                if digest is None:
                    counts["raw_rdkit_parse_failures"] += 1
                    digest = stable_digest(b"raw:" + raw_smiles.encode("utf-8"))
                source, index = _source_index(mol_id)
                expected_label = 0 if source == "pubchem" else 1
                if label != expected_label:
                    raise RuntimeError(
                        f"v1 source/label mismatch for {mol_id}: {label}"
                    )
                grouped_indices[source].append(index)
                grouped_digests[source].append(digest)
                if fallback:
                    grouped_fallback_indices[source].append(index)
                counts[f"raw_source_{source}"] += 1
            counts["raw_rows_seen"] += len(batch)
            for source in SOURCE_CAPACITIES:
                if grouped_indices[source]:
                    lookups[source][
                        np.asarray(grouped_indices[source], dtype=np.int64)
                    ] = np.asarray(grouped_digests[source], dtype="V16")
                if grouped_fallback_indices[source]:
                    fallback_lookups[source][
                        np.asarray(grouped_fallback_indices[source], dtype=np.int64)
                    ] = 1
            completed = counts["raw_rows_seen"]
            print(json.dumps({"raw_rows": completed}), flush=True)

        batch: list[tuple[str, int, str]] = []
        with args.raw_csv.open(newline="") as source:
            for row in csv.DictReader(source):
                batch.append((row["ID"], int(row["label"]), row["SMILES"]))
                if args.limit is not None and (
                    counts["raw_rows_seen"] + len(batch) >= args.limit
                ):
                    break
                if len(batch) >= args.canonical_work_batch_size:
                    consume_raw(batch)
                    batch.clear()
        if batch:
            consume_raw(batch)
    for lookup in lookups.values():
        lookup.flush()
    for lookup in fallback_lookups.values():
        lookup.flush()

    source_code_by_name = {
        "smprot2": 0,
        "uniprot_uniref": 1,
        "generated_peptideclm": 2,
        "pubchem": 3,
    }
    final_rows = 0
    with (
        digest_path.open("wb") as digest_out,
        label_path.open("wb") as label_out,
        source_path.open("wb") as source_out,
    ):
        for mol_ids, arrow_labels in _iter_final_batches(
            args.dataset_dir,
            wait_for_shards_seconds=args.wait_for_shards_seconds,
        ):
            if args.limit is not None:
                remaining = args.limit - final_rows
                if remaining <= 0:
                    break
                mol_ids = mol_ids[:remaining]
                arrow_labels = arrow_labels[:remaining]
            digests = np.empty(len(mol_ids), dtype="V16")
            source_codes = np.empty(len(mol_ids), dtype=np.uint8)
            for offset, mol_id in enumerate(mol_ids):
                source, index = _source_index(mol_id)
                digest = lookups[source][index]
                if bytes(digest) == b"\0" * 16:
                    raise RuntimeError(f"No raw canonical lookup for final ID {mol_id}")
                if fallback_lookups[source][index]:
                    counts["final_raw_identity_fallback_rows"] += 1
                digests[offset] = digest
                source_codes[offset] = source_code_by_name[source]
                label = int(arrow_labels[offset])
                expected_label = 0 if source == "pubchem" else 1
                if label != expected_label:
                    raise RuntimeError(
                        f"Final source/label mismatch for {mol_id}: {label}"
                    )
                counts[f"source_{source}"] += 1
                counts[f"label_{label}"] += 1
                cluster = _cluster_for_id(mol_id, smprot, uni)
                if cluster is not None:
                    digest_bytes = bytes(digest)
                    previous = canonical_to_cluster.get(digest_bytes)
                    if previous is None:
                        canonical_to_cluster[digest_bytes] = cluster
                    else:
                        union_find.union(previous, cluster)
            digest_out.write(digests.tobytes())
            label_out.write(np.asarray(arrow_labels, dtype=np.uint8).tobytes())
            source_out.write(source_codes.tobytes())
            final_rows += len(mol_ids)
            if final_rows % 1_000_000 < len(mol_ids):
                print(json.dumps({"final_rows": final_rows}), flush=True)

    row_count = counts["label_0"] + counts["label_1"]
    if digest_path.stat().st_size != row_count * 16:
        raise RuntimeError("Canonical digest file has an unexpected size")

    roots = np.fromiter(
        (union_find.find(i) for i in range(len(representatives))),
        dtype=np.int64,
        count=len(representatives),
    )
    positive_digests = np.memmap(digest_path, mode="r", dtype="V16", shape=(row_count,))
    labels = np.memmap(label_path, mode="r", dtype=np.uint8, shape=(row_count,))
    positive_unique = np.unique(positive_digests[labels == 1])
    conflict_digests: list[np.void] = []
    chunk = 2_000_000
    for begin in range(0, row_count, chunk):
        end = min(row_count, begin + chunk)
        local_digests = positive_digests[begin:end]
        local_labels = labels[begin:end]
        negative = np.unique(local_digests[local_labels == 0])
        positions = np.searchsorted(positive_unique, negative)
        in_range = positions < len(positive_unique)
        matches = np.zeros(len(negative), dtype=bool)
        matches[in_range] = positive_unique[positions[in_range]] == negative[in_range]
        conflict_digests.extend(negative[matches])
    conflict_unique = np.unique(np.asarray(conflict_digests, dtype="V16"))

    split_counts = Counter()
    source_counts = np.zeros((4, 3), dtype=np.int64)
    label_counts = np.zeros((2, 3), dtype=np.int64)
    conflict_rows = 0
    real_canonicals = np.asarray(list(canonical_to_cluster), dtype="V16")
    real_clusters = np.asarray(
        [roots[canonical_to_cluster[bytes(item)]] for item in real_canonicals],
        dtype=np.uint64,
    )
    order = np.argsort(real_canonicals)
    real_canonicals = real_canonicals[order]
    real_clusters = real_clusters[order]
    real_canonicals.tofile(real_canonical_path)
    real_clusters.tofile(real_root_path)

    with split_path.open("wb") as split_out, molecule_path.open("wb") as molecule_out:
        source_codes = np.memmap(source_path, mode="r", dtype=np.uint8, shape=(row_count,))
        for begin in range(0, row_count, chunk):
            end = min(row_count, begin + chunk)
            digests = positive_digests[begin:end]
            splits, molecule_ids = split_codes_for_digests(
                digests,
                seed=args.seed,
                real_canonicals=real_canonicals,
                real_sequence_roots=real_clusters,
            )
            split_out.write(splits.tobytes())
            molecule_out.write(molecule_ids.tobytes())
            for split in range(3):
                selected = splits == split
                split_counts[split] += int(np.count_nonzero(selected))
                for source in range(4):
                    source_counts[source, split] += int(
                        np.count_nonzero(selected & (source_codes[begin:end] == source))
                    )
                for label in range(2):
                    label_counts[label, split] += int(
                        np.count_nonzero(selected & (labels[begin:end] == label))
                    )
        if len(conflict_unique):
            positions = np.searchsorted(conflict_unique, positive_digests)
            valid = positions < len(conflict_unique)
            conflict_rows = int(
                np.count_nonzero(
                    valid
                    & (
                        conflict_unique[
                            np.minimum(positions, len(conflict_unique) - 1)
                        ]
                        == positive_digests
                    )
                )
            )

    split_names = ["train", "validation", "test"]
    source_names = ["smprot2", "uniprot_uniref", "generated_peptideclm", "pubchem"]
    manifest = {
        "canonical_identity": (
            "RDKit canonical isomeric SMILES, BLAKE2b-128; raw:SMILES "
            "namespace fallback only when RDKit cannot parse a raw row"
        ),
        "group_assignment": (
            "SplitMix64((canonical_digest_u64 xor seed) for ordinary molecules; "
            "(sequence_component_root xor seed xor 0xD1B54A32D192ED03) for "
            "real-peptide sequence components), modulo 10000"
        ),
        "cluster_count_before_molecule_union": len(representatives),
        "conflicting_label_molecule_count": len(conflict_unique),
        "conflicting_label_row_count": conflict_rows,
        "dataset_dir": str(args.dataset_dir),
        "elapsed_seconds": time.time() - started,
        "files": {},
        "label_by_split": {
            str(label): {
                split_names[split]: int(label_counts[label, split]) for split in range(3)
            }
            for label in range(2)
        },
        "row_count": row_count,
        "raw_rdkit_parse_failures": int(counts["raw_rdkit_parse_failures"]),
        "final_raw_identity_fallback_rows": int(
            counts["final_raw_identity_fallback_rows"]
        ),
        "rdkit_version": rdBase.rdkitVersion,
        "seed": args.seed,
        "sequence_cluster": {
            "coverage": args.coverage,
            "min_sequence_identity": args.min_seq_id,
            "scope": ["SmProt2", "UniProt/UniRef"],
        },
        "source_by_split": {
            source_names[source]: {
                split_names[split]: int(source_counts[source, split]) for split in range(3)
            }
            for source in range(4)
        },
        "split_code": {"0": "train", "1": "validation", "2": "test"},
        "split_counts": {
            split_names[split]: int(split_counts[split]) for split in range(3)
        },
    }
    for path in [
        digest_path,
        label_path,
        source_path,
        split_path,
        molecule_path,
        real_canonical_path,
        real_root_path,
    ]:
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json(args.output_dir / "split_manifest.json", manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "phase", choices=["extract-sequences", "cluster-sequences", "assign-splits"]
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--uniprot-sequences", type=Path)
    result.add_argument("--smprot-csv", type=Path)
    result.add_argument("--mmseqs", type=Path)
    result.add_argument("--threads", type=int, default=192)
    result.add_argument("--min-seq-id", type=float, default=0.4)
    result.add_argument("--coverage", type=float, default=0.8)
    result.add_argument("--cluster-tsv", type=Path)
    result.add_argument("--raw-csv", type=Path)
    result.add_argument("--dataset-dir", type=Path)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--canonical-chunk-size", type=int, default=256)
    result.add_argument("--canonical-work-batch-size", type=int, default=100_000)
    result.add_argument("--seed", type=int, default=20260726)
    result.add_argument("--limit", type=int)
    result.add_argument("--wait-for-shards-seconds", type=int, default=0)
    result.add_argument(
        "--mp-start-method", choices=["spawn", "fork", "forkserver"], default="spawn"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.phase == "extract-sequences":
        if args.uniprot_sequences is None or args.smprot_csv is None:
            raise SystemExit("extract-sequences requires --uniprot-sequences and --smprot-csv")
        extract_sequences(args)
    elif args.phase == "cluster-sequences":
        if args.mmseqs is None:
            raise SystemExit("cluster-sequences requires --mmseqs")
        cluster_sequences(args)
    else:
        required = [args.cluster_tsv, args.raw_csv, args.dataset_dir]
        if any(value is None for value in required):
            raise SystemExit(
                "assign-splits requires --cluster-tsv, --raw-csv, and --dataset-dir"
            )
        assign_splits(args)


if __name__ == "__main__":
    main()
