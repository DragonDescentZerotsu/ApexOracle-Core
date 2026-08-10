#!/usr/bin/env python3
"""Test whether frozen Evo-2 fragment embeddings retain strain variation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from Bio import SeqIO
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
EDLIB_PATH = REPO_ROOT / ".external/python/genome_condition_reviewer"
sys.path.insert(0, str(EDLIB_PATH))

from apexoracle.data.genome_embeddings import (  # noqa: E402
    genome_embedding_paths,
    parse_embedding_id,
    sha256_file,
)
from apexoracle.evaluation import genome_condition_reviewer  # noqa: E402
from apexoracle.evaluation.genome_condition_reviewer import (  # noqa: E402
    build_saved_tensor_windows,
)

DEFAULT_PAIRS = (
    REPO_ROOT / "experiments/genome_condition_reviewer/fragment_variation/manifests/"
    "nearest_pairs_unordered.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/genome_condition_reviewer/fragment_variation/all_embeddings"
)
DEFAULT_MINIMAP2 = REPO_ROOT / ".external/envs/genome_condition_reviewer/bin/minimap2"

PAF_COLUMNS = [
    "query",
    "query_length",
    "query_start",
    "query_end",
    "strand",
    "target",
    "target_length",
    "target_start",
    "target_end",
    "matching_bases",
    "alignment_block_length",
    "mapq",
]


def load_edlib():
    """Load the optional alignment dependency only for full fragment analysis."""

    try:
        return importlib.import_module("edlib")
    except ImportError as error:
        raise RuntimeError(
            "Genome-fragment edit-distance analysis requires edlib>=1.3.9. "
            "Install that optional dependency before running this audit."
        ) from error


def fasta_id(path: Path) -> str:
    return parse_embedding_id(path.name.replace(".fasta", ".pt"))


def load_encoded_fragments(fasta_path: Path) -> tuple[list[str], list[dict[str, int]]]:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    windows = build_saved_tensor_windows([len(record.seq) for record in records])
    sequences = [
        str(
            records[window["contig_index"]].seq[window["start"] : window["end"]]
        ).upper()
        for window in windows
    ]
    return sequences, windows


def write_fragment_fasta(path: Path, sequences: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for fragment_index, sequence in enumerate(sequences):
            handle.write(f">f{fragment_index}\n{sequence}\n")


def read_best_alignments(
    path: Path,
    *,
    minimum_coverage: float,
    minimum_mapq: int,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=range(12),
        names=PAF_COLUMNS,
    )
    frame["query_index"] = frame["query"].str[1:].astype(int)
    frame["target_index"] = frame["target"].str[1:].astype(int)
    frame["query_coverage"] = (frame["query_end"] - frame["query_start"]) / frame[
        "query_length"
    ]
    frame["target_coverage"] = (frame["target_end"] - frame["target_start"]) / frame[
        "target_length"
    ]
    frame["local_identity"] = frame["matching_bases"] / frame["alignment_block_length"]
    frame = frame[
        (frame["strand"] == "+")
        & (frame["query_length"] == 11_000)
        & (frame["target_length"] == 11_000)
        & (frame["query_coverage"] >= minimum_coverage)
        & (frame["target_coverage"] >= minimum_coverage)
        & (frame["mapq"] >= minimum_mapq)
    ].copy()
    return (
        frame.sort_values(
            ["query_index", "matching_bases", "local_identity", "mapq"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .drop_duplicates("query_index", keep="first")
        .reset_index(drop=True)
    )


def mutual_best_alignments(
    forward: pd.DataFrame, reverse: pd.DataFrame
) -> pd.DataFrame:
    if forward.empty or reverse.empty:
        return pd.DataFrame()
    return forward.merge(
        reverse[["query_index", "target_index"]],
        left_on=["query_index", "target_index"],
        right_on=["target_index", "query_index"],
        how="inner",
        suffixes=("", "_reverse"),
        validate="one_to_one",
    )


def analyze_pair(task: dict[str, object]) -> pd.DataFrame:
    torch.set_num_threads(1)
    genome_a = str(task["genome_a"])
    genome_b = str(task["genome_b"])
    fasta_a = Path(str(task["fasta_a"]))
    fasta_b = Path(str(task["fasta_b"]))
    embedding_a = Path(str(task["embedding_a"]))
    embedding_b = Path(str(task["embedding_b"]))
    minimap2 = Path(str(task["minimap2"]))
    sequences_a, windows_a = load_encoded_fragments(fasta_a)
    sequences_b, windows_b = load_encoded_fragments(fasta_b)
    tensor_a = torch.load(embedding_a, map_location="cpu", weights_only=True, mmap=True)
    tensor_b = torch.load(embedding_b, map_location="cpu", weights_only=True, mmap=True)
    if len(sequences_a) != int(tensor_a.shape[0]):
        raise RuntimeError(
            f"Saved-tensor window mismatch for {genome_a}: "
            f"{len(sequences_a)} != {int(tensor_a.shape[0])}"
        )
    if len(sequences_b) != int(tensor_b.shape[0]):
        raise RuntimeError(
            f"Saved-tensor window mismatch for {genome_b}: "
            f"{len(sequences_b)} != {int(tensor_b.shape[0])}"
        )

    with tempfile.TemporaryDirectory(prefix=f"evo2_frag_{genome_a}_{genome_b}_") as raw:
        work = Path(raw)
        fragments_a = work / "a.fasta"
        fragments_b = work / "b.fasta"
        forward_path = work / "a_to_b.paf"
        reverse_path = work / "b_to_a.paf"
        write_fragment_fasta(fragments_a, sequences_a)
        write_fragment_fasta(fragments_b, sequences_b)
        commands = [
            (fragments_b, fragments_a, forward_path),
            (fragments_a, fragments_b, reverse_path),
        ]
        for reference, query, output in commands:
            with output.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [
                        str(minimap2),
                        "-x",
                        "asm5",
                        "-c",
                        "--eqx",
                        "-N",
                        "5",
                        "-t",
                        "1",
                        str(reference),
                        str(query),
                    ],
                    check=True,
                    stdout=handle,
                    stderr=subprocess.DEVNULL,
                )
        forward = read_best_alignments(
            forward_path,
            minimum_coverage=float(task["minimum_alignment_coverage"]),
            minimum_mapq=int(task["minimum_mapq"]),
        )
        reverse = read_best_alignments(
            reverse_path,
            minimum_coverage=float(task["minimum_alignment_coverage"]),
            minimum_mapq=int(task["minimum_mapq"]),
        )
    matches = mutual_best_alignments(forward, reverse)
    if matches.empty:
        return matches

    a_indices = matches["query_index"].to_numpy(dtype=int)
    b_indices = matches["target_index"].to_numpy(dtype=int)
    edlib = load_edlib()
    edit_distances = np.asarray(
        [
            edlib.align(
                sequences_a[a_index],
                sequences_b[b_index],
                mode="NW",
                task="distance",
            )["editDistance"]
            for a_index, b_index in zip(a_indices, b_indices)
        ],
        dtype=int,
    )
    # Scaling prevents cosine norm underflow for the frozen bfloat16 tensors.
    vectors_a = tensor_a[a_indices].double() * 1e14
    vectors_b = tensor_b[b_indices].double() * 1e14
    cosine_distance = (
        1
        - torch.nn.functional.cosine_similarity(vectors_a, vectors_b, dim=1, eps=1e-12)
    ).numpy()
    relative_l2 = (
        torch.linalg.vector_norm(vectors_a - vectors_b, dim=1)
        / (
            (
                torch.linalg.vector_norm(vectors_a, dim=1)
                + torch.linalg.vector_norm(vectors_b, dim=1)
            )
            / 2
        )
    ).numpy()

    seed = int(
        hashlib.sha256(f"{genome_a}|{genome_b}|20260805".encode()).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    random_b_indices = rng.integers(0, len(sequences_b), size=len(matches))
    if len(sequences_b) > 1:
        collision = random_b_indices == b_indices
        random_b_indices[collision] = (random_b_indices[collision] + 1) % len(
            sequences_b
        )
    random_vectors_b = tensor_b[random_b_indices].double() * 1e14
    random_cosine_distance = (
        1
        - torch.nn.functional.cosine_similarity(
            vectors_a, random_vectors_b, dim=1, eps=1e-12
        )
    ).numpy()

    output = pd.DataFrame(
        {
            "pair_id": f"{genome_a}|{genome_b}",
            "genome_a": genome_a,
            "genome_b": genome_b,
            "species": str(task["species"]),
            "whole_genome_ani": float(task["ani"]),
            "fragment_a_index": a_indices,
            "fragment_b_index": b_indices,
            "fragment_a_contig": [
                windows_a[index]["contig_index"] for index in a_indices
            ],
            "fragment_b_contig": [
                windows_b[index]["contig_index"] for index in b_indices
            ],
            "local_alignment_identity": matches["local_identity"].to_numpy(),
            "query_coverage": matches["query_coverage"].to_numpy(),
            "target_coverage": matches["target_coverage"].to_numpy(),
            "global_edit_distance": edit_distances,
            "global_sequence_divergence": edit_distances / 11_000.0,
            "cosine_distance": cosine_distance,
            "relative_l2_distance": relative_l2,
            "random_donor_fragment_index": random_b_indices,
            "random_donor_cosine_distance": random_cosine_distance,
        }
    )
    return output


def finite_spearman(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def summarize_scope(frame: pd.DataFrame) -> dict[str, object]:
    identical = frame[frame["global_edit_distance"] == 0]
    variable = frame[frame["global_edit_distance"] > 0]
    pair_correlations = []
    for _, pair in variable.groupby("pair_id"):
        rho = finite_spearman(
            pair["global_sequence_divergence"], pair["cosine_distance"]
        )
        if rho is not None and len(pair) >= 10:
            pair_correlations.append(rho)
    return {
        "strain_pairs": int(frame["pair_id"].nunique()),
        "species": int(frame["species"].nunique()),
        "homologous_fragment_pairs": len(frame),
        "identical_fragment_pairs": len(identical),
        "variable_fragment_pairs": len(variable),
        "pooled_spearman_sequence_divergence_vs_cosine_distance": finite_spearman(
            variable["global_sequence_divergence"], variable["cosine_distance"]
        ),
        "per_strain_pair_spearman": {
            "eligible_pairs": len(pair_correlations),
            "median": (
                float(np.median(pair_correlations)) if pair_correlations else None
            ),
            "q1": (
                float(np.quantile(pair_correlations, 0.25))
                if pair_correlations
                else None
            ),
            "q3": (
                float(np.quantile(pair_correlations, 0.75))
                if pair_correlations
                else None
            ),
            "positive_fraction": (
                float(np.mean(np.asarray(pair_correlations) > 0))
                if pair_correlations
                else None
            ),
        },
        "identical_cosine_distance": {
            "median": (
                float(identical["cosine_distance"].median()) if len(identical) else None
            ),
            "maximum": (
                float(identical["cosine_distance"].max()) if len(identical) else None
            ),
        },
        "variable_cosine_distance": {
            "median": (
                float(variable["cosine_distance"].median()) if len(variable) else None
            ),
            "q1": (
                float(variable["cosine_distance"].quantile(0.25))
                if len(variable)
                else None
            ),
            "q3": (
                float(variable["cosine_distance"].quantile(0.75))
                if len(variable)
                else None
            ),
            "nonzero_fraction": (
                float((variable["cosine_distance"] > 1e-12).mean())
                if len(variable)
                else None
            ),
        },
        "homologous_closer_than_random_donor_fraction": float(
            (frame["cosine_distance"] < frame["random_donor_cosine_distance"]).mean()
        ),
        "homologous_cosine_distance_median": float(frame["cosine_distance"].median()),
        "random_donor_cosine_distance_median": float(
            frame["random_donor_cosine_distance"].median()
        ),
    }


def normalize_pair_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical unordered or legacy directed pair manifests."""
    pairs = frame.copy()
    if {"genome_a", "genome_b"}.issubset(pairs.columns):
        required = {"genome_a", "genome_b", "species", "ani"}
    elif {"target_id", "donor_id"}.issubset(pairs.columns):
        required = {"target_id", "donor_id", "species", "ani"}
        pairs["genome_a"] = pairs[["target_id", "donor_id"]].min(axis=1)
        pairs["genome_b"] = pairs[["target_id", "donor_id"]].max(axis=1)
    else:
        raise ValueError(
            "Pair manifest must contain genome_a/genome_b or target_id/donor_id"
        )
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair manifest missing columns: {missing}")
    pairs["genome_a"] = pairs["genome_a"].astype(str)
    pairs["genome_b"] = pairs["genome_b"].astype(str)
    return (
        pairs.sort_values(["genome_a", "genome_b"], kind="mergesort")
        .drop_duplicates(["genome_a", "genome_b"], keep="first")
        .reset_index(drop=True)
    )


def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    pairs = normalize_pair_manifest(
        pd.read_csv(
            args.pair_manifest.resolve(),
            dtype={
                "target_id": str,
                "donor_id": str,
                "genome_a": str,
                "genome_b": str,
            },
        )
    )
    embeddings = genome_embedding_paths(args.embedding_dir.resolve())
    fastas = {
        fasta_id(path): path
        for path in args.fasta_dir.resolve().glob("*.fasta")
        if path.is_file()
    }
    required = set(pairs["genome_a"]) | set(pairs["genome_b"])
    missing = (required - set(embeddings)) | (required - set(fastas))
    if missing:
        raise FileNotFoundError(f"Missing FASTA/tensor assets: {sorted(missing)}")
    tasks = []
    for row in pairs.itertuples(index=False):
        tasks.append(
            {
                "genome_a": row.genome_a,
                "genome_b": row.genome_b,
                "species": row.species,
                "ani": row.ani,
                "fasta_a": fastas[row.genome_a],
                "fasta_b": fastas[row.genome_b],
                "embedding_a": embeddings[row.genome_a],
                "embedding_b": embeddings[row.genome_b],
                "minimap2": args.minimap2.resolve(),
                "minimum_alignment_coverage": args.minimum_alignment_coverage,
                "minimum_mapq": args.minimum_mapq,
            }
        )
    return tasks


def audit_saved_tensor_shapes(
    fasta_dir: Path, embedding_dir: Path
) -> dict[str, object]:
    embeddings = genome_embedding_paths(embedding_dir.resolve())
    fastas = {
        fasta_id(path): path
        for path in fasta_dir.resolve().glob("*.fasta")
        if path.is_file()
    }
    common = sorted(set(embeddings) & set(fastas))
    mismatches = []
    multi_record_genomes = 0
    for genome_id in common:
        record_lengths = [
            len(record.seq) for record in SeqIO.parse(fastas[genome_id], "fasta")
        ]
        multi_record_genomes += int(len(record_lengths) > 1)
        window_count = len(build_saved_tensor_windows(record_lengths))
        tensor_rows = int(
            torch.load(
                embeddings[genome_id],
                map_location="cpu",
                weights_only=True,
                mmap=True,
            ).shape[0]
        )
        if window_count != tensor_rows:
            mismatches.append(
                {
                    "genome_id": genome_id,
                    "reconstructed_windows": window_count,
                    "tensor_rows": tensor_rows,
                }
            )
    return {
        "fasta_tensor_pairs": len(common),
        "multi_record_genomes": multi_record_genomes,
        "exact_shape_matches": len(common) - len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-manifest",
        "--donor-manifest",
        dest="pair_manifest",
        type=Path,
        default=DEFAULT_PAIRS,
    )
    parser.add_argument(
        "--fasta-dir", type=Path, default=REPO_ROOT / "DataPrepare/Data/Genome/ATCC"
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=REPO_ROOT / "DataPrepare/Data/Genome_embs",
    )
    parser.add_argument("--minimap2", type=Path, default=DEFAULT_MINIMAP2)
    parser.add_argument("--minimum-alignment-coverage", type=float, default=0.80)
    parser.add_argument("--minimum-mapq", type=int, default=20)
    parser.add_argument("--maximum-global-divergence", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.minimap2.exists():
        raise FileNotFoundError(args.minimap2)
    shape_audit = audit_saved_tensor_shapes(args.fasta_dir, args.embedding_dir)
    if shape_audit["mismatches"]:
        raise RuntimeError(
            f"Saved-tensor window audit failed: {shape_audit['mismatches'][:5]}"
        )
    tasks = build_tasks(args)
    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_pair, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
            print(f"completed {completed}/{len(tasks)} strain pairs", flush=True)
    if not frames:
        raise RuntimeError("No mutual homologous fragment pairs passed the filters")
    raw = pd.concat(frames, ignore_index=True).sort_values(
        ["pair_id", "fragment_a_index", "fragment_b_index"], kind="mergesort"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "fragment_pairs.csv", index=False)

    homologous = raw[
        raw["global_sequence_divergence"] <= args.maximum_global_divergence
    ].copy()
    if homologous.empty:
        raise RuntimeError("No fragment pairs passed maximum global divergence")
    pair_rows = []
    for pair_id, frame in homologous.groupby("pair_id"):
        variable = frame[frame["global_edit_distance"] > 0]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "genome_a": frame["genome_a"].iloc[0],
                "genome_b": frame["genome_b"].iloc[0],
                "species": frame["species"].iloc[0],
                "whole_genome_ani": frame["whole_genome_ani"].iloc[0],
                "homologous_fragments": len(frame),
                "identical_fragments": int((frame["global_edit_distance"] == 0).sum()),
                "variable_fragments": len(variable),
                "spearman_divergence_cosine": finite_spearman(
                    variable["global_sequence_divergence"],
                    variable["cosine_distance"],
                ),
                "homologous_closer_than_random_fraction": float(
                    (
                        frame["cosine_distance"] < frame["random_donor_cosine_distance"]
                    ).mean()
                ),
            }
        )
    pair_summary = pd.DataFrame(pair_rows)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)

    scopes = {"all_pairs": summarize_scope(homologous)}
    closely_related = homologous[homologous["whole_genome_ani"] >= 99.0]
    if not closely_related.empty:
        scopes["whole_genome_ani_ge_99"] = summarize_scope(closely_related)
    included_pairs = homologous.drop_duplicates("pair_id")
    task_ani = pd.Series([float(task["ani"]) for task in tasks], dtype=float)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "protocol": {
            "scope": "all eligible bacterial nearest same-species strain pairs",
            "representation_scope": (
                "only genomic segments represented by the saved fragment tensors"
            ),
            "window_length_nt": 11_000,
            "window_step_nt": 10_000,
            "mutual_best_same_orientation_only": True,
            "minimum_bidirectional_alignment_coverage": (
                args.minimum_alignment_coverage
            ),
            "minimum_mapq": args.minimum_mapq,
            "maximum_global_sequence_divergence": args.maximum_global_divergence,
            "global_distance": "edlib Needleman-Wunsch edit distance / 11000",
            "embedding_distance": (
                "cosine distance after 1e14 scaling and float64 conversion"
            ),
            "random_control": (
                "one deterministic non-homologous fragment from the same donor genome"
            ),
        },
        "cohort": {
            "input_unordered_strain_pairs": len(tasks),
            "input_pair_whole_genome_ani": {
                "minimum": float(task_ani.min()),
                "median": float(task_ani.median()),
                "maximum": float(task_ani.max()),
            },
            "pairs_with_homologous_fragments": int(homologous["pair_id"].nunique()),
            "species": int(homologous["species"].nunique()),
            "included_pair_whole_genome_ani": {
                "minimum": float(included_pairs["whole_genome_ani"].min()),
                "median": float(included_pairs["whole_genome_ani"].median()),
                "maximum": float(included_pairs["whole_genome_ani"].max()),
            },
            "raw_mutual_fragment_pairs": len(raw),
            "analysis_homologous_fragment_pairs": len(homologous),
        },
        "saved_tensor_shape_audit": shape_audit,
        "results": scopes,
        "tools": {
            "minimap2": subprocess.check_output(
                [str(args.minimap2.resolve()), "--version"], text=True
            ).strip(),
            "edlib": "1.3.9.post1",
        },
        "source_sha256": {
            "pair_manifest": sha256_file(args.pair_manifest.resolve()),
            "window_reconstruction_module": sha256_file(
                Path(genome_condition_reviewer.__file__)
            ),
            "entrypoint": sha256_file(Path(__file__)),
        },
        "interpretation_boundary": (
            "This tests whether sequence differences within saved homologous fragment "
            "conditions change frozen embeddings. It does not test sequence outside "
            "the saved condition, causal function, or MIC-head use."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
