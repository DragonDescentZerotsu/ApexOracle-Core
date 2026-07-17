"""Build the immutable shared-data protocol for the revised Fig. 2b.

The original comparator scripts applied model-specific filtering before their
independent KFold splits.  This module creates one molecule list and one split
file that every encoder must consume.  Encoder adapters may truncate their
inputs, but they must not silently remove a molecule after this protocol is
frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROTOCOL_VERSION = "fig2b-shared-native-intersection-v2"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
CHAIN_SEPARATORS = frozenset("-/;,.|:")

EXPECTED_ENCODERS = (
    "dlm_mtr_dlm",
    "dlm_only",
    "chemberta_mtr",
    "chemberta_mlm",
    "molformer",
    "peptideclm",
    "apex",
)

DEFAULT_TARGET_COLUMNS = (
    "Escherichia coli ATCC 25922",
    "Pseudomonas aeruginosa ATCC 27853",
    "Staphylococcus aureus ATCC 25923",
    "Staphylococcus aureus",
    "Staphylococcus aureus ATCC 29213",
    "Escherichia coli",
    "Pseudomonas aeruginosa",
    "Pseudomonas aeruginosa PAO1",
    "Enterococcus faecalis ATCC 29212",
    "Acinetobacter baumannii ATCC 19606",
    "Staphylococcus epidermidis ATCC 12228",
    "Candida albicans ATCC 10231",
    "Klebsiella pneumoniae ATCC 700603",
    "Staphylococcus aureus ATCC 43300",
    "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028",
    "Staphylococcus aureus ATCC 6538",
    "Pseudomonas aeruginosa ATCC 9027",
    "Candida albicans",
    "Klebsiella pneumoniae",
)


@dataclass(frozen=True)
class ApexProjection:
    """APEX-compatible lossy view of one DBAASP residue sequence."""

    sequence: str
    original_residue_count: int
    projected_residue_count: int
    contained_noncanonical: bool
    contained_d_residue: bool
    removed_topology_or_chain_marker: bool
    truncated: bool
    unusual_position_count: int


def _normalise_unusual_positions(values: Any) -> tuple[int, ...]:
    """Flatten DBAASP unusual-amino-acid metadata into 1-based positions."""

    positions: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            position = value.get("position")
            if position is not None:
                try:
                    positions.append(int(position))
                except (TypeError, ValueError):
                    pass
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    visit(values)
    return tuple(sorted(set(position for position in positions if position > 0)))


def project_apex_sequence(
    sequence: str,
    *,
    unusual_positions: Iterable[int] = (),
    has_topology_or_multiple_chains: bool = False,
    max_residues: int = 50,
) -> ApexProjection:
    """Project a DBAASP sequence into the fixed-vocabulary APEX input space.

    Rules agreed for the reviewer-requested shared benchmark:

    - lowercase D-residues are converted to their uppercase residue identity;
    - noncanonical or unknown residue symbols become ``X``;
    - ``cyclo-``, whitespace and common chain separators are removed, so cyclic
      and multichain records are represented as one deterministic linear order;
    - the projected sequence is truncated to the 50 content positions available
      in the historical APEX length-52 input (start and end occupy two slots).

    The historical APEX vocabulary has no X token. The sequence audit retains X
    explicitly, while the unchanged APEX encoder maps it to index 0 in the same
    way as every other unknown symbol in the published implementation.
    """

    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("APEX projection requires a non-empty residue sequence")
    if max_residues <= 0:
        raise ValueError("max_residues must be positive")

    text = sequence.strip()
    removed_marker = bool(has_topology_or_multiple_chains)
    if text.lower().startswith("cyclo-"):
        text = text[6:]
        removed_marker = True

    residues: list[str] = []
    contained_noncanonical = False
    contained_d_residue = False

    for symbol in text:
        if symbol.isspace() or symbol in CHAIN_SEPARATORS:
            removed_marker = True
            continue

        upper = symbol.upper()
        if upper in CANONICAL_AMINO_ACIDS:
            residues.append(upper)
            if symbol.islower():
                contained_d_residue = True
        elif upper == "X" or symbol.isalpha():
            residues.append("X")
            contained_noncanonical = True
        else:
            residues.append("X")
            contained_noncanonical = True

    if not residues:
        raise ValueError("APEX projection produced an empty residue sequence")

    normalised_unusual: list[int] = []
    for position in unusual_positions:
        try:
            parsed_position = int(position)
        except (TypeError, ValueError):
            continue
        if parsed_position > 0:
            normalised_unusual.append(parsed_position)
    unusual = tuple(sorted(set(normalised_unusual)))
    for position in unusual:
        if position <= len(residues):
            residues[position - 1] = "X"
            contained_noncanonical = True

    original_count = len(residues)
    truncated = original_count > max_residues
    projected = "".join(residues[:max_residues])

    return ApexProjection(
        sequence=projected,
        original_residue_count=original_count,
        projected_residue_count=len(projected),
        contained_noncanonical=contained_noncanonical,
        contained_d_residue=contained_d_residue,
        removed_topology_or_chain_marker=removed_marker,
        truncated=truncated,
        unusual_position_count=len(unusual),
    )


def assign_folds(
    molecule_ids: Iterable[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return an order-independent molecule-level KFold assignment."""

    ids = sorted(str(value) for value in molecule_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("molecule_ids must be unique")
    if len(ids) < n_splits:
        raise ValueError("number of molecules must be at least n_splits")

    assignments: dict[str, int] = {}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    id_array = np.asarray(ids, dtype=object)
    for fold, (_, test_indices) in enumerate(splitter.split(id_array)):
        for index in test_indices:
            assignments[str(id_array[index])] = fold

    return pd.DataFrame(
        {
            "dbaasp_id": ids,
            "fold": [assignments[molecule_id] for molecule_id in ids],
        }
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path while allowing external sources."""

    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _load_dbaasp_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("DBAASP record file must contain a JSON list")
    return {str(record.get("id")): record for record in records}


def _validate_mic_table(frame: pd.DataFrame) -> None:
    required = ("DBAASP_id", "SMILES", *DEFAULT_TARGET_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"MIC table is missing columns: {missing}")
    if frame["DBAASP_id"].isna().any():
        raise ValueError("MIC table contains missing DBAASP_id values")
    if frame["DBAASP_id"].duplicated().any():
        duplicates = frame.loc[frame["DBAASP_id"].duplicated(), "DBAASP_id"].tolist()[:10]
        raise ValueError(f"MIC table contains duplicate DBAASP IDs: {duplicates}")


def _load_eligibility(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"dbaasp_id": "string"})
    required = {"dbaasp_id", *EXPECTED_ENCODERS, "eligible_all"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"eligibility table is missing columns: {sorted(missing)}")
    frame["dbaasp_id"] = frame["dbaasp_id"].str.strip()
    if frame["dbaasp_id"].isna().any() or frame["dbaasp_id"].duplicated().any():
        raise ValueError("eligibility table IDs must be present and unique")
    for column in (*EXPECTED_ENCODERS, "eligible_all"):
        if frame[column].dtype != bool:
            normalised = frame[column].astype(str).str.strip().str.lower()
            if not normalised.isin(("true", "false")).all():
                raise ValueError(f"eligibility column {column} must contain booleans")
            frame[column] = normalised == "true"
    calculated = frame.loc[:, EXPECTED_ENCODERS].all(axis=1)
    if not calculated.equals(frame["eligible_all"]):
        raise ValueError("eligible_all does not equal the intersection of encoder flags")
    return frame.set_index("dbaasp_id")


def build_shared_dataset(
    mic_csv: Path,
    dbaasp_records_json: Path,
    eligibility_csv: Path,
    output_dir: Path,
    *,
    n_splits: int = 5,
    seed: int = 42,
    apex_max_residues: int = 50,
) -> dict[str, Any]:
    """Build shared molecule IDs, folds, projections, exclusions and manifest."""

    mic_csv = mic_csv.resolve()
    dbaasp_records_json = dbaasp_records_json.resolve()
    eligibility_csv = eligibility_csv.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(mic_csv, dtype={"DBAASP_id": "string"})
    frame["DBAASP_id"] = frame["DBAASP_id"].str.strip()
    _validate_mic_table(frame)
    records = _load_dbaasp_records(dbaasp_records_json)
    eligibility = _load_eligibility(eligibility_csv)
    source_ids = set(frame["DBAASP_id"].astype(str))
    if set(eligibility.index) != source_ids:
        raise ValueError("eligibility table IDs must exactly match the MIC source IDs")

    shared_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    for _, row in frame.iterrows():
        molecule_id = str(row["DBAASP_id"])
        eligibility_row = eligibility.loc[molecule_id]
        if not bool(eligibility_row["eligible_all"]):
            failed = [name for name in EXPECTED_ENCODERS if not bool(eligibility_row[name])]
            exclusions.append(
                {
                    "dbaasp_id": molecule_id,
                    "stage": "native_encoder_intersection",
                    "reason": ";".join(failed),
                }
            )
            continue
        record = records.get(molecule_id)
        if record is None:
            exclusions.append(
                {"dbaasp_id": molecule_id, "stage": "source_join", "reason": "missing_dbaasp_record"}
            )
            continue

        sequence = record.get("sequence")
        if not isinstance(sequence, str) or not sequence.strip():
            exclusions.append(
                {"dbaasp_id": molecule_id, "stage": "apex_projection", "reason": "missing_sequence"}
            )
            continue

        unusual_positions = _normalise_unusual_positions(record.get("unusualAminoAcids"))
        complexity = record.get("complexity")
        complexity_name = complexity.get("name") if isinstance(complexity, Mapping) else None
        has_topology_or_multiple_chains = bool(
            record.get("intrachainBonds")
            or record.get("interchainBonds")
            or record.get("coordinationBonds")
            or (complexity_name and complexity_name != "Monomer")
        )
        try:
            projection = project_apex_sequence(
                sequence,
                unusual_positions=unusual_positions,
                has_topology_or_multiple_chains=has_topology_or_multiple_chains,
                max_residues=apex_max_residues,
            )
        except ValueError as error:
            exclusions.append(
                {"dbaasp_id": molecule_id, "stage": "apex_projection", "reason": str(error)}
            )
            continue

        shared_row: dict[str, Any] = {
            "dbaasp_id": molecule_id,
            "smiles": row["SMILES"],
            "apex_sequence": projection.sequence,
        }
        shared_row.update({column: row[column] for column in DEFAULT_TARGET_COLUMNS})
        shared_rows.append(shared_row)
        projection_rows.append({"dbaasp_id": molecule_id, **asdict(projection)})

    if not shared_rows:
        raise ValueError("shared dataset is empty")

    shared = pd.DataFrame(shared_rows)
    projections = pd.DataFrame(projection_rows)
    folds = assign_folds(shared["dbaasp_id"], n_splits=n_splits, seed=seed)
    common_ids = folds[["dbaasp_id"]].copy()
    exclusions_frame = pd.DataFrame(exclusions, columns=("dbaasp_id", "stage", "reason"))

    shared.to_csv(output_dir / "shared_molecules.csv", index=False)
    common_ids.to_csv(output_dir / "common_molecule_ids.csv", index=False)
    folds.to_csv(output_dir / "folds.csv", index=False)
    projections.to_csv(output_dir / "apex_projection_audit.csv", index=False)
    exclusions_frame.to_csv(output_dir / "exclusions.csv", index=False)

    fold_counts = {
        str(int(fold)): int(count)
        for fold, count in folds["fold"].value_counts().sort_index().items()
    }
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "shared_ids_and_folds_frozen",
        "sources": {
            "mic_csv": _portable_path(mic_csv),
            "mic_csv_sha256": sha256_file(mic_csv),
            "dbaasp_records_json": _portable_path(dbaasp_records_json),
            "dbaasp_records_json_sha256": sha256_file(dbaasp_records_json),
            "eligibility_csv": _portable_path(eligibility_csv),
            "eligibility_csv_sha256": sha256_file(eligibility_csv),
        },
        "expected_encoders": list(EXPECTED_ENCODERS),
        "sample_policy": "native-processable ID intersection and one shared molecule-level 5-fold split",
        "label_policy": {
            "columns": list(DEFAULT_TARGET_COLUMNS),
            "missing_sentinel_in_source": -1,
            "training_transform": "-log10(MIC / 10) on observed labels only",
        },
        "split_policy": {
            "kind": "molecule-level KFold",
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": seed,
            "ids_sorted_before_split": True,
            "fold_counts": fold_counts,
        },
        "apex_projection": {
            "noncanonical_residue": "X",
            "d_residue": "uppercase residue identity",
            "cyclic_or_multichain": "remove topology/chain markers and use deterministic linear residue order",
            "max_content_residues": apex_max_residues,
            "x_encoder_index": 0,
            "x_encoder_note": "unmodified APEX maps X/unknown symbols to index 0",
            "records_with_noncanonical": int(projections["contained_noncanonical"].sum()),
            "records_with_d_residue": int(projections["contained_d_residue"].sum()),
            "records_with_linearized_topology_or_chain_marker": int(
                projections["removed_topology_or_chain_marker"].sum()
            ),
            "records_truncated": int(projections["truncated"].sum()),
        },
        "counts": {
            "source_molecules": int(len(frame)),
            "shared_molecules": int(len(shared)),
            "excluded_molecules": int(len(exclusions_frame)),
        },
        "outputs": {
            "shared_molecules": "shared_molecules.csv",
            "common_molecule_ids": "common_molecule_ids.csv",
            "folds": "folds.csv",
            "apex_projection_audit": "apex_projection_audit.csv",
            "exclusions": "exclusions.csv",
        },
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mic-csv",
        type=Path,
        default=Path("DataPrepare/Data/DBAASP_id_SMILES_bact_MICs.csv"),
    )
    parser.add_argument(
        "--dbaasp-records-json",
        type=Path,
        default=Path("DataPrepare/Data/all_peptides_data.json"),
    )
    parser.add_argument(
        "--eligibility-csv",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_encoder_eligibility.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DataPrepare/Data/fig2b_shared_v1"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apex-max-residues", type=int, default=50)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_shared_dataset(
        mic_csv=args.mic_csv,
        dbaasp_records_json=args.dbaasp_records_json,
        eligibility_csv=args.eligibility_csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        apex_max_residues=args.apex_max_residues,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
