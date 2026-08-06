"""Shared deterministic contracts for genome-representation validation.

The public reviewer analyses use these helpers to reconstruct the coordinates
represented by saved fragment tensors, select nearest same-species genomes,
label conservative fragment annotations, and quantify paired effects. Archived
condition-swap diagnostics use the same contracts but are not part of the
formal manuscript evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re

import numpy as np
import pandas as pd


WINDOW_LENGTH_NT = 11_000
WINDOW_STEP_NT = 10_000
MIN_DONOR_ANI = 95.0
MIN_DONOR_ALIGNED_FRACTION = 50.0

# The frozen MIC genome-text pool contains these six fungal species in addition
# to bacteria. The reviewer experiment is explicitly bacterial-only. A curated
# species exclusion is used instead of requiring a matching GenBank lineage,
# because several valid bacterial FASTA/tensor assets lack a matching current
# GenBank file or lineage block.
NON_BACTERIAL_SPECIES = frozenset(
    {
        "Aspergillus fumigatus",
        "Candida albicans",
        "Candida krusei",
        "Candida tropicalis",
        "Cryptococcus neoformans",
        "Saccharomyces cerevisiae",
    }
)


AMR_PRODUCT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bantibiotic resistance\b",
        r"\bantimicrobial resistance\b",
        r"\bbeta[- ]lactamase\b",
        r"\bcarbapenemase\b",
        r"\bchloramphenicol acetyltransferase\b",
        r"\baminoglycoside (?:acetyltransferase|phosphotransferase|nucleotidyltransferase)\b",
        r"\b(?:methicillin|vancomycin|tetracycline|macrolide|quinolone|"
        r"sulfonamide|trimethoprim|colistin) resistance\b",
    )
)

AMR_GENE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^bla[A-Za-z0-9_-]+$",
        r"^mecA$",
        r"^mcr(?:-[0-9]+)?$",
        r"^van[AB]$",
        r"^tet[A-Za-z0-9()_-]+$",
        r"^erm[A-Za-z0-9()_-]+$",
        r"^qnr[A-Za-z0-9()_-]+$",
        r"^sul[123]$",
        r"^dfr[A-Za-z0-9()_-]+$",
        r"^(?:aac|aph|ant)[A-Za-z0-9()'_-]+$",
    )
)

MGE_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btransposase\b",
        r"\binsertion sequence\b",
        r"\bprophage\b",
        r"\bphage integrase\b",
        r"\bsite[- ]specific integrase\b",
        r"\bsite[- ]specific recombinase\b",
        r"\brelaxase\b",
        r"\bconjugative transfer protein\b",
        r"\bplasmid (?:replication|mobilization) protein\b",
    )
)


def build_fragment_windows(
    contig_lengths: Sequence[int],
    *,
    window_length: int = WINDOW_LENGTH_NT,
    step: int = WINDOW_STEP_NT,
) -> list[dict[str, int]]:
    """Return fragment rows in contig order and ascending start order."""

    if window_length <= 0 or step <= 0:
        raise ValueError("window_length and step must be positive")
    rows: list[dict[str, int]] = []
    fragment_index = 0
    for contig_index, raw_length in enumerate(contig_lengths):
        contig_length = int(raw_length)
        if contig_length < 0:
            raise ValueError("contig lengths cannot be negative")
        for start in range(0, contig_length, step):
            rows.append(
                {
                    "fragment_index": fragment_index,
                    "contig_index": contig_index,
                    "start": start,
                    "end": min(start + window_length, contig_length),
                }
            )
            fragment_index += 1
    return rows


def build_saved_tensor_windows(
    record_lengths: Sequence[int],
    *,
    window_length: int = WINDOW_LENGTH_NT,
    step: int = WINDOW_STEP_NT,
) -> list[dict[str, int]]:
    """Reconstruct coordinates using the indexing of the saved tensors.

    This compatibility mapping is frozen because it exactly matches the first
    dimension of all saved embedding tensors used by the validation analyses.
    """

    if window_length <= 0 or step <= 0:
        raise ValueError("window_length and step must be positive")
    rows: list[dict[str, int]] = []
    fragment_index = 0
    for record_index, raw_length in enumerate(record_lengths):
        record_length = int(raw_length)
        if record_length < 0:
            raise ValueError("record lengths cannot be negative")
        while fragment_index * step < record_length:
            start = fragment_index * step
            rows.append(
                {
                    "fragment_index": fragment_index,
                    "contig_index": record_index,
                    "start": start,
                    "end": min(start + window_length, record_length),
                }
            )
            fragment_index += 1
    return rows


def overlapping_fragment_indices(
    windows: Sequence[Mapping[str, int]],
    *,
    contig_index: int,
    start: int,
    end: int,
) -> list[int]:
    """Return half-open windows overlapping one half-open annotation interval."""

    if end <= start:
        return []
    return [
        int(row["fragment_index"])
        for row in windows
        if int(row["contig_index"]) == int(contig_index)
        and int(row["start"]) < int(end)
        and int(row["end"]) > int(start)
    ]


def _qualifier_values(
    qualifiers: Mapping[str, Iterable[str] | str], key: str
) -> list[str]:
    raw = qualifiers.get(key, [])
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw]


def classify_annotation(
    feature_type: str,
    qualifiers: Mapping[str, Iterable[str] | str],
) -> tuple[bool, bool, list[str]]:
    """Classify one annotation using the frozen conservative dictionaries."""

    genes = _qualifier_values(qualifiers, "gene")
    text_values: list[str] = []
    for key in ("product", "function", "note", "mobile_element_type"):
        text_values.extend(_qualifier_values(qualifiers, key))
    matched: list[str] = []

    amr = False
    for gene in genes:
        if any(pattern.search(gene.strip()) for pattern in AMR_GENE_PATTERNS):
            amr = True
            matched.append(f"amr_gene:{gene}")
    for value in text_values:
        if any(pattern.search(value) for pattern in AMR_PRODUCT_PATTERNS):
            amr = True
            matched.append(f"amr_text:{value}")

    mge = feature_type == "mobile_element"
    if mge:
        matched.append("mge_feature:mobile_element")
    for value in text_values:
        if any(pattern.search(value) for pattern in MGE_TEXT_PATTERNS):
            mge = True
            matched.append(f"mge_text:{value}")
    return amr, mge, sorted(set(matched))


def parse_skani_sparse(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize a ``skani triangle -E`` table to one row per unordered pair."""

    required = {
        "Ref_file",
        "Query_file",
        "ANI",
        "Align_fraction_ref",
        "Align_fraction_query",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"skani table is missing columns: {sorted(missing)}")
    output = table.copy()

    def genome_id(value: object) -> str:
        stem = str(value).split("/")[-1].split(".fasta")[0]
        if "ATCC" not in stem:
            return stem
        suffix = stem.split("ATCC", 1)[1].strip("_")
        return suffix.replace("_", "-")

    output["ref_id"] = output["Ref_file"].map(genome_id)
    output["query_id"] = output["Query_file"].map(genome_id)
    output = output[output["ref_id"] != output["query_id"]].copy()
    output["ANI"] = output["ANI"].astype(float)
    output["Align_fraction_ref"] = output["Align_fraction_ref"].astype(float)
    output["Align_fraction_query"] = output["Align_fraction_query"].astype(float)
    return output


def select_nearest_same_species_neighbors(
    ani_pairs: pd.DataFrame,
    *,
    eligible_ids: set[str],
    species_by_id: Mapping[str, str],
    min_ani: float = MIN_DONOR_ANI,
    min_aligned_fraction: float = MIN_DONOR_ALIGNED_FRACTION,
) -> pd.DataFrame:
    """Choose one deterministic nearest same-species neighbor per eligible ID."""

    candidates: list[dict[str, object]] = []
    for row in ani_pairs.to_dict("records"):
        ref_id = str(row["ref_id"])
        query_id = str(row["query_id"])
        if ref_id not in eligible_ids or query_id not in eligible_ids:
            continue
        if species_by_id.get(ref_id) != species_by_id.get(query_id):
            continue
        ani = float(row["ANI"])
        ref_af = float(row["Align_fraction_ref"])
        query_af = float(row["Align_fraction_query"])
        if ani < min_ani or min(ref_af, query_af) < min_aligned_fraction:
            continue
        candidates.extend(
            [
                {
                    "target_id": ref_id,
                    "donor_id": query_id,
                    "species": species_by_id[ref_id],
                    "ani": ani,
                    "aligned_fraction_target": ref_af,
                    "aligned_fraction_donor": query_af,
                },
                {
                    "target_id": query_id,
                    "donor_id": ref_id,
                    "species": species_by_id[query_id],
                    "ani": ani,
                    "aligned_fraction_target": query_af,
                    "aligned_fraction_donor": ref_af,
                },
            ]
        )
    if not candidates:
        return pd.DataFrame(
            columns=[
                "target_id",
                "donor_id",
                "species",
                "ani",
                "aligned_fraction_target",
                "aligned_fraction_donor",
            ]
        )
    frame = pd.DataFrame(candidates)
    frame["minimum_aligned_fraction"] = frame[
        ["aligned_fraction_target", "aligned_fraction_donor"]
    ].min(axis=1)
    frame = frame.sort_values(
        ["target_id", "ani", "minimum_aligned_fraction", "donor_id"],
        ascending=[True, False, False, True],
    )
    return frame.drop_duplicates("target_id", keep="first").reset_index(drop=True)


def paired_strain_bootstrap(
    strain_metrics: pd.DataFrame,
    *,
    value_column: str,
    iterations: int = 2_000,
    seed: int = 20260804,
) -> tuple[float, float]:
    """Bootstrap the mean paired effect with strains as the sampling unit."""

    values = strain_metrics[value_column].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("strain bootstrap requires at least one strain")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)
