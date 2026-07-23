#!/usr/bin/env python3
"""Audit PepLink AA/peptide -> SELFIES -> AA round-trip behavior.

The audit deliberately separates two claims:

1. structural round-trip: the SELFIES emitted from an annotation decodes to
   the same isomeric molecular graph as the corresponding SMILES;
2. annotation round-trip: PepLink's conservative reverse parser recovers the
   residue sequence and supported topology.

The selected PepLink version only promises annotation reverse parsing for standard L/D
amino-acid peptides with linear or head-to-tail backbones.  Non-canonical
residues, side-chain crosslinks, terminal modifications, multimers, and
coordination complexes remain visible as out-of-scope rows rather than being
misreported as round-trip failures.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DBAASP_JSON = REPO_ROOT / "DataPrepare/Data/all_peptides_data.json"
DEFAULT_CURATED_MIC = (
    REPO_ROOT / "DataPrepare/Data/DBAASP_id_bact_name_SMILES_MIC_Evo.csv"
)
DEFAULT_AA_MAPPING = (
    REPO_ROOT / "DataPrepare/Data/all_aa_smiles_new_handcrafted.csv"
)
DEFAULT_PEPLINK_SOURCE = Path(
    os.environ.get("PEPLINK_SOURCE", REPO_ROOT.parent / "PepLink")
)
STANDARD_AA_CODES_WITH_D = frozenset(
    "ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return f"external/{resolved.name}"


def canonical_smiles(text: str) -> str | None:
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def cyclic_rotations(sequence: str) -> set[str]:
    return {
        sequence[index:] + sequence[:index] for index in range(len(sequence))
    }


def is_head_to_tail(record: dict[str, Any]) -> bool:
    bonds = record.get("intrachainBonds") or []
    sequence = record.get("sequence") or ""
    if len(bonds) != 1 or not sequence:
        return False
    bond = bonds[0]
    return (
        (bond.get("type") or {}).get("name") == "AMD"
        and (bond.get("chainParticipating") or {}).get("name") == "MMB"
        and int(bond.get("position1", -1)) == 1
        and int(bond.get("position2", -1)) == len(sequence)
    )


def reverse_contract_scope(record: dict[str, Any]) -> str:
    sequence = (record.get("sequence") or "").strip()
    if (record.get("complexity") or {}).get("name") != "Monomer":
        return "out_of_scope_multimer"
    if not sequence:
        return "out_of_scope_empty_sequence"
    if any(code not in STANDARD_AA_CODES_WITH_D for code in sequence):
        return "out_of_scope_noncanonical_sequence"
    if record.get("unusualAminoAcids"):
        return "out_of_scope_noncanonical_residue"
    if record.get("interchainBonds"):
        return "out_of_scope_interchain_bond"
    if record.get("coordinationBonds"):
        return "out_of_scope_coordination_bond"
    if record.get("nTerminus") or record.get("cTerminus"):
        return "out_of_scope_terminal_modification"
    bonds = record.get("intrachainBonds") or []
    if not bonds:
        return "supported_linear"
    if is_head_to_tail(record):
        return "supported_head_to_tail"
    return "out_of_scope_other_intrachain_bond"


def topology_label(record: dict[str, Any]) -> str:
    if (record.get("complexity") or {}).get("name") != "Monomer":
        return "multimer"
    if record.get("coordinationBonds"):
        return "coordination_complex"
    if record.get("interchainBonds"):
        return "interchain_crosslinked"
    if is_head_to_tail(record):
        return "head_to_tail_cyclic"
    if record.get("intrachainBonds"):
        return "other_intrachain_crosslinked"
    return "linear"


def _hash_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_row(task: dict[str, Any]) -> dict[str, Any]:
    record = task["record"]
    sequence = record.get("sequence") or ""
    unusual = record.get("unusualAminoAcids") or []
    return {
        "record_kind": task["record_kind"],
        "source_key": task["source_key"],
        "dbaasp_id": task.get("dbaasp_id"),
        "used_in_curated_dbaasp": task.get("used_in_curated_dbaasp", True),
        "curated_occurrences": task.get("curated_occurrences", 0),
        "sequence_length": len(sequence),
        "topology": topology_label(record),
        "noncanonical_residue_count": len(unusual),
        "intrachain_bond_count": len(record.get("intrachainBonds") or []),
        "has_terminal_modification": bool(
            record.get("nTerminus") or record.get("cTerminus")
        ),
        "reverse_contract_scope": reverse_contract_scope(record),
    }


def audit_task(task: dict[str, Any]) -> dict[str, Any]:
    from PepLink import aa_seqs_to_smiles, from_dbaasp_record, smiles_to_aa_seqs
    import selfies as sf

    row = _base_row(task)
    record = task["record"]
    try:
        if task["record_kind"] == "dbaasp_peptide":
            peptide_input = from_dbaasp_record(record)
            kwargs = peptide_input.to_api_kwargs()
        else:
            kwargs = task["peplink_kwargs"]
        smiles = aa_seqs_to_smiles(**kwargs)
        selfies_text = aa_seqs_to_smiles(**kwargs, output_format="selfies")
        decoded_smiles = sf.decoder(selfies_text)
        normalized_smiles = canonical_smiles(smiles)
        normalized_decoded = canonical_smiles(decoded_smiles)
        row.update(
            {
                "forward_status": "success",
                "forward_error_class": None,
                "forward_error": None,
                "smiles_sha256": _hash_text(smiles),
                "selfies_sha256": _hash_text(selfies_text),
                "selfies_decodes_to_valid_smiles": normalized_decoded is not None,
                "selfies_structure_exact": normalized_smiles
                == normalized_decoded,
            }
        )
    except Exception as exc:  # Keep every unsupported/invalid source row visible.
        row.update(
            {
                "forward_status": "failed",
                "forward_error_class": type(exc).__name__,
                "forward_error": str(exc),
                "smiles_sha256": None,
                "selfies_sha256": None,
                "selfies_decodes_to_valid_smiles": False,
                "selfies_structure_exact": False,
                "reverse_smiles_status": "not_run_forward_failed",
                "reverse_selfies_status": "not_run_forward_failed",
                "annotation_roundtrip_pass": False,
            }
        )
        return row

    scope = row["reverse_contract_scope"]
    if not scope.startswith("supported_"):
        row.update(
            {
                "reverse_smiles_status": "not_run_out_of_contract",
                "reverse_selfies_status": "not_run_out_of_contract",
                "annotation_roundtrip_pass": None,
            }
        )
        return row

    expected_sequence = record["sequence"]
    expected_cyclic = scope == "supported_head_to_tail"

    def assess(parsed: Any) -> bool:
        if parsed.sequence is None or parsed.unsupported_reason is not None:
            return False
        sequence_match = (
            parsed.sequence in cyclic_rotations(expected_sequence)
            if expected_cyclic
            else parsed.sequence == expected_sequence
        )
        return sequence_match and parsed.is_cyclic == expected_cyclic

    parsed_smiles = smiles_to_aa_seqs(smiles, input_format="smiles")
    parsed_selfies = smiles_to_aa_seqs(selfies_text, input_format="selfies")
    smiles_pass = assess(parsed_smiles)
    selfies_pass = assess(parsed_selfies)
    row.update(
        {
            "reverse_smiles_status": "pass" if smiles_pass else "fail",
            "reverse_selfies_status": "pass" if selfies_pass else "fail",
            "annotation_roundtrip_pass": smiles_pass and selfies_pass,
        }
    )
    return row


def load_curated_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["DBAASP_id"]) for row in csv.DictReader(handle)}


def load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def residue_tasks(
    mapping: list[dict[str, str]], unusual_counts: Counter[str]
) -> list[dict[str, Any]]:
    tasks = []
    for row in mapping:
        name = row["aa"]
        if name in STANDARD_AA_CODES_WITH_D:
            record = {
                "sequence": name,
                "complexity": {"name": "Monomer"},
                "unusualAminoAcids": [],
                "intrachainBonds": [],
                "interchainBonds": [],
                "coordinationBonds": [],
                "nTerminus": None,
                "cTerminus": None,
            }
            kwargs = {"sequence": name}
            occurrences = 0
        else:
            unusual = {
                "position": 1,
                "name": name,
                "modificationType": {"name": name},
            }
            record = {
                "sequence": "X",
                "complexity": {"name": "Monomer"},
                "unusualAminoAcids": [unusual],
                "intrachainBonds": [],
                "interchainBonds": [],
                "coordinationBonds": [],
                "nTerminus": None,
                "cTerminus": None,
            }
            kwargs = {"sequence": "X", "unusual_amino_acids": [unusual]}
            occurrences = unusual_counts[name]
        tasks.append(
            {
                "record_kind": "residue_definition",
                "source_key": name,
                "dbaasp_id": None,
                "used_in_curated_dbaasp": occurrences > 0,
                "curated_occurrences": occurrences,
                "record": record,
                "peplink_kwargs": kwargs,
            }
        )
    return tasks


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [audit_task(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(audit_task, tasks, chunksize=16))


def count_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbaasp-json", type=Path, default=DEFAULT_DBAASP_JSON)
    parser.add_argument("--curated-mic", type=Path, default=DEFAULT_CURATED_MIC)
    parser.add_argument("--aa-mapping", type=Path, default=DEFAULT_AA_MAPPING)
    parser.add_argument(
        "--peplink-source", type=Path, default=DEFAULT_PEPLINK_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    sys.path.insert(0, str(args.peplink_source.resolve()))
    from PepLink import __version__ as peplink_version

    records = json.loads(args.dbaasp_json.read_text(encoding="utf-8"))
    by_id = {int(record["id"]): record for record in records}
    curated_ids = load_curated_ids(args.curated_mic)
    missing = sorted(curated_ids - by_id.keys())
    if missing:
        raise ValueError(f"Curated DBAASP IDs missing from source JSON: {missing[:10]}")

    unusual_counts: Counter[str] = Counter()
    peptide_tasks = []
    for dbaasp_id in sorted(curated_ids):
        record = by_id[dbaasp_id]
        for unusual in record.get("unusualAminoAcids") or []:
            name = unusual.get("name") or (
                unusual.get("modificationType") or {}
            ).get("name")
            if name:
                unusual_counts[name] += 1
        peptide_tasks.append(
            {
                "record_kind": "dbaasp_peptide",
                "source_key": str(dbaasp_id),
                "dbaasp_id": dbaasp_id,
                "record": record,
            }
        )
    definition_tasks = residue_tasks(load_mapping(args.aa_mapping), unusual_counts)

    peptide_rows = run_tasks(peptide_tasks, args.workers)
    definition_rows = run_tasks(definition_tasks, args.workers)
    rows = definition_rows + peptide_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "roundtrip_records.csv"
    fieldnames = list(rows[0])
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        structural_denominator = sum(
            row["forward_status"] == "success" for row in group
        )
        structural_pass = sum(
            row["forward_status"] == "success"
            and row["selfies_structure_exact"] is True
            for row in group
        )
        annotation_scope = [
            row
            for row in group
            if str(row["reverse_contract_scope"]).startswith("supported_")
        ]
        annotation_ran = [
            row for row in annotation_scope if row["forward_status"] == "success"
        ]
        return {
            "rows": len(group),
            "forward_status": count_by(group, "forward_status"),
            "forward_error_class": count_by(
                [row for row in group if row["forward_status"] == "failed"],
                "forward_error_class",
            ),
            "topology": count_by(group, "topology"),
            "reverse_contract_scope": count_by(group, "reverse_contract_scope"),
            "structural_roundtrip": {
                "denominator_forward_success": structural_denominator,
                "pass": structural_pass,
                "fail": structural_denominator - structural_pass,
            },
            "annotation_roundtrip": {
                "contract_scope": len(annotation_scope),
                "ran_after_forward_success": len(annotation_ran),
                "pass": sum(
                    row["annotation_roundtrip_pass"] is True
                    for row in annotation_ran
                ),
                "fail": sum(
                    row["annotation_roundtrip_pass"] is False
                    for row in annotation_ran
                ),
            },
        }

    summary = {
        "schema_version": 1,
        "claim_boundary": {
            "structural_roundtrip": (
                "AA/peptide annotation -> PepLink SMILES and SELFIES; SELFIES "
                "must decode to the identical canonical isomeric molecular graph."
            ),
            "annotation_roundtrip": (
                "PepLink reverse parser must recover the exact linear sequence or "
                "a cyclic rotation for head-to-tail peptides, with topology preserved."
            ),
            "reverse_scope": (
                f"PepLink {peplink_version}: standard L/D residues, linear and "
                "head-to-tail cyclic monomer peptides only."
            ),
        },
        "software": {
            "peplink_version": peplink_version,
            "peplink_source": display_path(args.peplink_source),
        },
        "sources": {
            "dbaasp_json": {
                "path": display_path(args.dbaasp_json),
                "sha256": sha256(args.dbaasp_json),
            },
            "curated_mic": {
                "path": display_path(args.curated_mic),
                "sha256": sha256(args.curated_mic),
                "unique_dbaasp_ids": len(curated_ids),
            },
            "aa_mapping": {
                "path": display_path(args.aa_mapping),
                "sha256": sha256(args.aa_mapping),
            },
        },
        "residue_definitions": summarize(definition_rows),
        "curated_dbaasp_peptides": summarize(peptide_rows),
        "detail_csv": display_path(detail_path),
    }
    summary_path = args.output_dir / "roundtrip_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
