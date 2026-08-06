"""Annotation and linear-probe helpers for saved genome fragment tensors."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from itertools import zip_longest
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch

from apexoracle.data.genome_embeddings import (
    genome_embedding_paths,
    matched_genome_ids,
    parse_embedding_id,
    sha256_file,
)
from apexoracle.evaluation.genome_condition_reviewer import (
    build_saved_tensor_windows,
    classify_annotation,
    overlapping_fragment_indices,
)


EMBEDDING_WIDTH = 8_192
EMBEDDING_SCALE = 1.0e14


@dataclass(frozen=True)
class ProbeConfig:
    """Frozen settings for one simple fragment-annotation readout."""

    label_column: str
    display_name: str
    negative_ratio: int = 5
    seed: int = 20260804


@dataclass
class _CompatibleGenomeAsset:
    """Validated records and tensor metadata for one saved genome condition."""

    fasta_path: Path
    genbank_path: Path
    embedding_path: Path
    fasta_records: list[SeqRecord]
    genbank_records: list[SeqRecord]
    windows: list[dict[str, int]]


def _read_genbank_lineage(path: Path) -> str:
    lines: list[str] = []
    in_lineage = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("  ORGANISM"):
                in_lineage = True
                continue
            if in_lineage:
                if line.startswith("            "):
                    lines.append(line.strip())
                    continue
                break
            if line.startswith("FEATURES"):
                break
    return " ".join(lines)


def _sequences_match(fasta_records, genbank_records) -> bool:
    for fasta_record, genbank_record in zip_longest(fasta_records, genbank_records):
        if fasta_record is None or genbank_record is None:
            return False
        if str(fasta_record.seq).upper() != str(genbank_record.seq).upper():
            return False
    return True


def _load_compatible_genome_asset(
    *,
    embedding_path: Path,
    fasta_path: Path | None,
    genbank_dir: Path,
) -> tuple[_CompatibleGenomeAsset | None, str | None]:
    """Validate one FASTA/GenBank/tensor triplet or return its exclusion reason."""

    if fasta_path is None:
        return None, "missing_matching_fasta"
    genbank_path = genbank_dir / f"{fasta_path.stem}.gbk"
    if not genbank_path.exists():
        return None, "missing_matching_genbank"
    if "Bacteria" not in _read_genbank_lineage(genbank_path):
        return None, "not_annotated_as_bacteria"

    fasta_records = list(SeqIO.parse(fasta_path, "fasta"))
    genbank_records = list(SeqIO.parse(genbank_path, "genbank"))
    if not _sequences_match(fasta_records, genbank_records):
        return None, "fasta_genbank_sequence_or_order_mismatch"

    windows = build_saved_tensor_windows([len(record.seq) for record in fasta_records])
    embedding = torch.load(
        embedding_path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise TypeError(f"Unexpected embedding payload: {embedding_path}")
    if int(embedding.shape[1]) != EMBEDDING_WIDTH:
        raise ValueError(f"Unexpected embedding width: {embedding_path}")
    if len(windows) != int(embedding.shape[0]):
        return None, "reconstructed_window_count_mismatch"
    del embedding
    return (
        _CompatibleGenomeAsset(
            fasta_path=fasta_path,
            genbank_path=genbank_path,
            embedding_path=embedding_path,
            fasta_records=fasta_records,
            genbank_records=genbank_records,
            windows=windows,
        ),
        None,
    )


def _map_annotations_to_windows(
    genbank_records: list[SeqRecord], windows: list[dict[str, int]]
) -> dict[int, dict[str, object]]:
    """Map conservative GenBank annotations onto saved fragment indices."""

    labels: dict[int, dict[str, object]] = {
        int(window["fragment_index"]): {
            "amr": False,
            "mge": False,
            "matches": set(),
        }
        for window in windows
    }
    for contig_index, record in enumerate(genbank_records):
        for feature in record.features:
            is_amr, is_mge, matches = classify_annotation(
                feature.type, feature.qualifiers
            )
            if not is_amr and not is_mge:
                continue
            for part in feature.location.parts:
                for fragment_index in overlapping_fragment_indices(
                    windows,
                    contig_index=contig_index,
                    start=int(part.start),
                    end=int(part.end),
                ):
                    label = labels[fragment_index]
                    label["amr"] = bool(label["amr"]) or is_amr
                    label["mge"] = bool(label["mge"]) or is_mge
                    label["matches"].update(matches)  # type: ignore[union-attr]
    return labels


def _build_fragment_rows(
    *, genome_id: str, asset: _CompatibleGenomeAsset
) -> tuple[list[dict[str, object]], int, int]:
    """Build deterministic fragment rows and per-genome positive counts."""

    labels = _map_annotations_to_windows(asset.genbank_records, asset.windows)
    rows: list[dict[str, object]] = []
    amr_count = 0
    mge_count = 0
    for window in asset.windows:
        fragment_index = int(window["fragment_index"])
        label = labels[fragment_index]
        amr_count += int(bool(label["amr"]))
        mge_count += int(bool(label["mge"]))
        rows.append(
            {
                "genome_id": genome_id,
                "embedding_file": asset.embedding_path.name,
                **window,
                "amr_associated": bool(label["amr"]),
                "mge_associated": bool(label["mge"]),
                "matched_annotations": "; ".join(
                    sorted(label["matches"])  # type: ignore[arg-type]
                ),
            }
        )
    return rows, amr_count, mge_count


def build_fragment_annotation_manifest(
    *, data_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Map existing GenBank annotations to saved tensor-compatible windows."""

    fasta_dir = data_dir / "Genome/ATCC"
    genbank_dir = data_dir / "Genome_annotation/ATCC"
    embedding_dir = data_dir / "Genome_embs"
    embeddings = genome_embedding_paths(embedding_dir)
    matched_ids = matched_genome_ids(data_dir, embeddings)
    fasta_by_id = {
        parse_embedding_id(path.name.replace(".fasta", ".pt")): path
        for path in fasta_dir.glob("*.fasta")
        if path.is_file()
    }

    genome_rows: list[dict[str, object]] = []
    fragment_rows: list[dict[str, object]] = []
    excluded: dict[str, int] = {}

    def exclude(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for genome_id in sorted(matched_ids):
        embedding_path = embeddings[genome_id]
        asset, exclusion_reason = _load_compatible_genome_asset(
            embedding_path=embedding_path,
            fasta_path=fasta_by_id.get(genome_id),
            genbank_dir=genbank_dir,
        )
        if asset is None:
            if exclusion_reason is None:
                raise RuntimeError(f"Missing exclusion reason for genome {genome_id}")
            exclude(exclusion_reason)
            gc.collect()
            continue
        rows, amr_count, mge_count = _build_fragment_rows(
            genome_id=genome_id, asset=asset
        )
        fragment_rows.extend(rows)
        genome_rows.append(
            {
                "genome_id": genome_id,
                "fasta_file": asset.fasta_path.name,
                "genbank_file": asset.genbank_path.name,
                "embedding_file": asset.embedding_path.name,
                "contigs": len(asset.fasta_records),
                "total_nt": sum(len(record.seq) for record in asset.fasta_records),
                "fragments": len(asset.windows),
                "amr_positive_fragments": amr_count,
                "mge_positive_fragments": mge_count,
                "fasta_sha256": sha256_file(asset.fasta_path),
                "genbank_sha256": sha256_file(asset.genbank_path),
                "embedding_sha256": sha256_file(asset.embedding_path),
            }
        )
        del asset, rows
        gc.collect()

    genomes = pd.DataFrame(genome_rows)
    fragments = pd.DataFrame(fragment_rows)
    genomes.to_csv(output_dir / "compatible_genomes.csv", index=False)
    fragments.to_csv(output_dir / "fragment_annotation_labels.csv", index=False)
    summary: dict[str, object] = {
        "window_reconstruction": "saved_tensor_compatible",
        "paper_matched_embedding_ids": len(matched_ids),
        "compatible_bacterial_genomes": len(genomes),
        "compatible_fragments": len(fragments),
        "amr_positive_fragments": int(fragments["amr_associated"].sum()),
        "mge_positive_fragments": int(fragments["mge_associated"].sum()),
        "excluded": dict(sorted(excluded.items())),
    }
    return genomes, fragments, summary


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Read manifest booleans without treating the string ``False`` as truthy."""

    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"Unexpected boolean values: {unexpected}")
    return normalized.map({"true": True, "false": False}).astype(bool)


def deterministic_probe_cohort(
    labels: pd.DataFrame, *, config: ProbeConfig
) -> pd.DataFrame:
    """Keep all positives and reproducibly sample local negatives."""

    rng = np.random.default_rng(config.seed)
    selected: list[pd.DataFrame] = []
    for _, frame in labels.groupby("genome_id", sort=True):
        positive = frame[frame[config.label_column]].copy()
        if positive.empty:
            continue
        negative = frame[~frame[config.label_column]].copy()
        take = min(len(negative), config.negative_ratio * len(positive))
        if take:
            indices = rng.choice(negative.index.to_numpy(), size=take, replace=False)
            negative = negative.loc[indices]
        else:
            negative = negative.iloc[0:0]
        selected.extend([positive, negative])
    cohort = pd.concat(selected, ignore_index=True)
    return cohort.sort_values(["genome_id", "fragment_index"]).reset_index(drop=True)


def _load_probe_matrix(cohort: pd.DataFrame, embedding_dir: Path) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for (_, embedding_file), frame in cohort.groupby(
        ["genome_id", "embedding_file"], sort=True
    ):
        embedding = torch.load(
            embedding_dir / embedding_file,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        indices = torch.tensor(frame["fragment_index"].to_numpy(), dtype=torch.long)
        blocks.append((embedding[indices].float() * EMBEDDING_SCALE).numpy())
    return np.concatenate(blocks, axis=0)


def run_linear_probe(
    labels: pd.DataFrame,
    *,
    config: ProbeConfig,
    embedding_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run one fixed five-fold genome-disjoint L2 logistic readout."""

    cohort = deterministic_probe_cohort(labels, config=config)
    positives = int(cohort[config.label_column].sum())
    positive_genomes = int(
        cohort.loc[cohort[config.label_column], "genome_id"].nunique()
    )
    if positives < 50 or positive_genomes < 10:
        raise RuntimeError(
            f"{config.display_name} probe is underpowered: {positives} positives in "
            f"{positive_genomes} genomes"
        )
    matrix = _load_probe_matrix(cohort, embedding_dir)
    if len(matrix) != len(cohort):
        raise RuntimeError("Probe matrix and cohort row counts differ")
    target = cohort[config.label_column].astype(int).to_numpy()
    groups = cohort["genome_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.seed)
    probabilities = np.full(len(cohort), np.nan, dtype=float)
    fold_assignments = np.full(len(cohort), -1, dtype=int)
    fold_rows: list[dict[str, object]] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(matrix, target, groups)
    ):
        train_genomes = set(groups[train_indices])
        test_genomes = set(groups[test_indices])
        overlap = train_genomes & test_genomes
        if overlap:
            raise RuntimeError(
                f"Genome leakage in probe fold {fold}: {sorted(overlap)}"
            )
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=config.seed,
        )
        model.fit(matrix[train_indices], target[train_indices])
        fold_probability = model.predict_proba(matrix[test_indices])[:, 1]
        probabilities[test_indices] = fold_probability
        fold_assignments[test_indices] = fold
        fold_rows.append(
            {
                "label": config.display_name,
                "fold": fold,
                "test_fragments": len(test_indices),
                "train_genomes": len(train_genomes),
                "test_genomes": len(test_genomes),
                "test_positives": int(target[test_indices].sum()),
                "test_prevalence": float(target[test_indices].mean()),
                "auprc": average_precision_score(
                    target[test_indices], fold_probability
                ),
                "auroc": roc_auc_score(target[test_indices], fold_probability),
                "iterations": int(model.n_iter_[0]),
            }
        )
    if np.isnan(probabilities).any() or (fold_assignments < 0).any():
        raise RuntimeError("Probe did not produce a prediction for every cohort row")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = cohort[
        ["genome_id", "embedding_file", "fragment_index", config.label_column]
    ].copy()
    predictions["fold"] = fold_assignments
    predictions["probability"] = probabilities
    predictions.to_csv(
        output_dir / f"{config.label_column}_probe_predictions.csv", index=False
    )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(
        output_dir / f"{config.label_column}_probe_fold_metrics.csv", index=False
    )
    return {
        "label": config.display_name,
        "full_compatible_fragments": len(labels),
        "full_compatible_genomes": int(labels["genome_id"].nunique()),
        "full_positive_fragments": int(labels[config.label_column].sum()),
        "full_positive_genomes": int(
            labels.loc[labels[config.label_column], "genome_id"].nunique()
        ),
        "full_prevalence": float(labels[config.label_column].mean()),
        "probe_cohort_fragments": len(cohort),
        "probe_cohort_positive_fragments": positives,
        "probe_cohort_genomes": int(cohort["genome_id"].nunique()),
        "probe_cohort_positive_genomes": positive_genomes,
        "probe_cohort_prevalence": float(target.mean()),
        "oof_auprc": float(average_precision_score(target, probabilities)),
        "oof_auroc": float(roc_auc_score(target, probabilities)),
        "fold_auprc_mean": float(folds["auprc"].mean()),
        "fold_auprc_sample_sd": float(folds["auprc"].std(ddof=1)),
        "fold_auroc_mean": float(folds["auroc"].mean()),
        "fold_auroc_sample_sd": float(folds["auroc"].std(ddof=1)),
        "model": "L2 logistic regression; C=1; class_weight=balanced; liblinear",
        "split": "5-fold StratifiedGroupKFold grouped by genome ID",
        "negative_sampling": (
            f"all positives plus at most {config.negative_ratio} negatives per "
            "positive within each positive-bearing genome"
        ),
    }
