#!/usr/bin/env python3
"""Build the auditable chemical-validation table for ChatGPT-o1/OPSIN AAs.

This script reconstructs the exact 169-row historical lineage, performs
machine-checkable structure and stereochemistry diagnostics, and merges a
separately curated row-by-row manual adjudication table.  Automated checks do
not substitute for the manual chemical decision.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import selfies as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "DataPrepare/Data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/peplink_validation"
FORMULA_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
ALLOWED_DECISIONS = {
    "verified",
    "verified_source_annotation_typo",
    "ambiguous_source_annotation",
    "incorrect_pipeline_output",
    "non_exact_polymer_proxy",
    "not_a_complete_amino_acid_definition",
}
SUPPORTED_DECISIONS = {"verified", "verified_source_annotation_typo"}
HANDCRAFTED_REPLACEMENTS = {
    "S-ALA-4-pen",
    "R-ALA-7-oct",
    "Me-PENT-GLY",
}


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
        return str(resolved)


def canonical_smiles(text: str) -> str | None:
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def formula_mentions(text: str) -> str:
    candidates = [
        token
        for token in FORMULA_PATTERN.findall(text)
        if any(character.isdigit() for character in token)
        and ("C" in token or "H" in token)
    ]
    return ";".join(candidates)


def chiral_summary(mol: Chem.Mol) -> tuple[int, int, str]:
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True
    )
    assigned = sum(label != "?" for _, label in centers)
    unassigned = sum(label == "?" for _, label in centers)
    return assigned, unassigned, ";".join(
        f"{atom_index}:{label}" for atom_index, label in centers
    )


def load_manual_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"aa", "decision", "confidence", "review_notes", "required_action"}
    missing_columns = required - set(rows[0] if rows else [])
    if missing_columns:
        raise ValueError(f"Manual review is missing columns: {sorted(missing_columns)}")
    duplicates = [key for key, count in Counter(row["aa"] for row in rows).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate manual adjudications: {duplicates}")
    invalid = sorted({row["decision"] for row in rows} - ALLOWED_DECISIONS)
    if invalid:
        raise ValueError(f"Unsupported manual decisions: {invalid}")
    return {row["aa"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-annotations",
        type=Path,
        default=DATA_DIR / "unusual_aa_names_wo_PubChem_smiles.json",
    )
    parser.add_argument(
        "--gpt-names",
        type=Path,
        default=DATA_DIR / "unusual_aa_text_transfered_by_GPT.txt",
    )
    parser.add_argument(
        "--opsin-smiles",
        type=Path,
        default=DATA_DIR / "unusual_aa_smiles_OPSIN_output.txt",
    )
    parser.add_argument(
        "--historical-pipeline-csv",
        type=Path,
        default=DATA_DIR / "unusual_aa_wo_PubChem_smiles.csv",
    )
    parser.add_argument(
        "--final-aa-mapping",
        type=Path,
        default=DATA_DIR / "all_aa_smiles_new_handcrafted.csv",
    )
    parser.add_argument(
        "--dbaasp-json",
        type=Path,
        default=DATA_DIR / "all_peptides_data.json",
    )
    parser.add_argument(
        "--curated-mic",
        type=Path,
        default=DATA_DIR / "DBAASP_id_bact_name_SMILES_MIC_Evo.csv",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "chatgpt_opsin_manual_review.csv",
    )
    parser.add_argument(
        "--roundtrip-records",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "peplink_0.1.2/roundtrip_records.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    annotations_json = json.loads(
        args.source_annotations.read_text(encoding="utf-8")
    )
    annotations = [next(iter(item.items())) for item in annotations_json]
    annotation_codes = [code for code, _ in annotations]
    gpt_names = args.gpt_names.read_text(encoding="utf-8").splitlines()
    opsin_smiles = args.opsin_smiles.read_text(encoding="utf-8").splitlines()
    with args.historical_pipeline_csv.open(newline="", encoding="utf-8") as handle:
        historical = list(csv.DictReader(handle))
    with args.final_aa_mapping.open(newline="", encoding="utf-8") as handle:
        final_mapping = {row["aa"]: row["SMILES"] for row in csv.DictReader(handle)}
    with args.curated_mic.open(newline="", encoding="utf-8") as handle:
        curated_mic_counts = Counter(
            int(row["DBAASP_id"]) for row in csv.DictReader(handle)
        )
    dbaasp_records = json.loads(args.dbaasp_json.read_text(encoding="utf-8"))
    curated_ids = set(curated_mic_counts)
    usage_ids: dict[str, set[int]] = {aa: set() for aa in annotation_codes}
    usage_occurrences: Counter[str] = Counter()
    for record in dbaasp_records:
        record_id = int(record["id"])
        if record_id not in curated_ids:
            continue
        for unusual in record.get("unusualAminoAcids") or []:
            name = unusual.get("name") or (
                unusual.get("modificationType") or {}
            ).get("name")
            if name in usage_ids:
                usage_ids[name].add(record_id)
                usage_occurrences[name] += 1

    lengths = {
        "source_annotations": len(annotations),
        "gpt_names": len(gpt_names),
        "opsin_smiles": len(opsin_smiles),
        "historical_pipeline_csv": len(historical),
    }
    if set(lengths.values()) != {169}:
        raise ValueError(f"Historical lineage is not the expected aligned 169 rows: {lengths}")
    if annotation_codes != [row["unusual_AA"] for row in historical]:
        raise ValueError("Historical pipeline CSV order does not match source annotations")
    if opsin_smiles != [row["SMILES"] for row in historical]:
        raise ValueError("Historical pipeline CSV SMILES do not match OPSIN output")
    missing_final = sorted(set(annotation_codes) - final_mapping.keys())
    if missing_final:
        raise ValueError(f"Pipeline AA codes missing from final mapping: {missing_final}")

    manual = load_manual_review(args.manual_review)
    unknown_manual = sorted(manual.keys() - set(annotation_codes))
    if unknown_manual:
        raise ValueError(f"Manual review contains unknown AA codes: {unknown_manual}")

    rows: list[dict[str, Any]] = []
    for index, ((aa, original), gpt_name, opsin) in enumerate(
        zip(annotations, gpt_names, opsin_smiles), start=1
    ):
        final_smiles = final_mapping[aa]
        mol = Chem.MolFromSmiles(opsin)
        final_mol = Chem.MolFromSmiles(final_smiles)
        opsin_valid = mol is not None
        final_valid = final_mol is not None
        if mol is not None:
            assigned, unassigned, centers = chiral_summary(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            exact_mass = Descriptors.ExactMolWt(mol)
            fragment_count = len(Chem.GetMolFrags(mol))
            formal_charge = Chem.GetFormalCharge(mol)
            isotope_atoms = sum(atom.GetIsotope() != 0 for atom in mol.GetAtoms())
            try:
                selfies_text = sf.encoder(
                    Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                )
                decoded = canonical_smiles(sf.decoder(selfies_text))
                selfies_structure_exact = decoded == canonical_smiles(opsin)
            except Exception:
                selfies_structure_exact = False
        else:
            assigned = unassigned = fragment_count = formal_charge = isotope_atoms = 0
            centers = formula = ""
            exact_mass = None
            selfies_structure_exact = False

        adjudication = manual.get(aa, {})
        rows.append(
            {
                "lineage_index": index,
                "aa": aa,
                "original_dbaasp_annotation": original,
                "gpt_o1_standardized_name": gpt_name,
                "opsin_smiles": opsin,
                "final_mapping_smiles": final_smiles,
                "used_in_curated_dbaasp": bool(usage_ids[aa]),
                "curated_peptide_count": len(usage_ids[aa]),
                "curated_residue_occurrence_count": usage_occurrences[aa],
                "curated_mic_row_count": sum(
                    curated_mic_counts[record_id] for record_id in usage_ids[aa]
                ),
                "opsin_rdkit_valid": opsin_valid,
                "final_mapping_rdkit_valid": final_valid,
                "opsin_selfies_structure_exact": selfies_structure_exact,
                "opsin_formula": formula,
                "source_formula_mentions": formula_mentions(original),
                "exact_mass": exact_mass,
                "fragment_count": fragment_count,
                "formal_charge": formal_charge,
                "isotope_atom_count": isotope_atoms,
                "assigned_chiral_centers": assigned,
                "unassigned_chiral_centers": unassigned,
                "chiral_centers": centers,
                "final_exactly_matches_opsin": final_smiles == opsin,
                "final_structure_matches_opsin": canonical_smiles(final_smiles)
                == canonical_smiles(opsin),
                "manual_decision": adjudication.get("decision"),
                "manual_confidence": adjudication.get("confidence"),
                "manual_review_notes": adjudication.get("review_notes"),
                "required_action": adjudication.get("required_action"),
            }
        )

    if manual and set(manual) != set(annotation_codes):
        missing = sorted(set(annotation_codes) - manual.keys())
        raise ValueError(
            f"Manual adjudication is incomplete: {len(missing)} missing, first={missing[:10]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "chatgpt_opsin_chemical_validation.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    decision_by_aa = {row["aa"]: row["manual_decision"] for row in rows}
    flagged_codes = {
        aa
        for aa, decision in decision_by_aa.items()
        if decision not in SUPPORTED_DECISIONS
    }
    final_risk_codes = flagged_codes - HANDCRAFTED_REPLACEMENTS - {
        aa
        for aa, decision in decision_by_aa.items()
        if decision == "not_a_complete_amino_acid_definition"
    }
    exclusion_rows = []
    for record in dbaasp_records:
        record_id = int(record["id"])
        if record_id not in curated_ids:
            continue
        record_codes = []
        for unusual in record.get("unusualAminoAcids") or []:
            name = unusual.get("name") or (
                unusual.get("modificationType") or {}
            ).get("name")
            if name in flagged_codes:
                record_codes.append(name)
        if not record_codes:
            continue
        unique_codes = sorted(set(record_codes))
        exclusion_rows.append(
            {
                "dbaasp_id": record_id,
                "dbaasp_accession": record.get("dbaaspId") or "",
                "peptide_name": record.get("name") or "",
                "sequence": record.get("sequence") or "",
                "flagged_aa_codes": ";".join(unique_codes),
                "decision_classes": ";".join(
                    sorted({decision_by_aa[aa] for aa in unique_codes})
                ),
                "contains_handcrafted_replacement": any(
                    aa in HANDCRAFTED_REPLACEMENTS for aa in unique_codes
                ),
                "contains_unresolved_final_risk": any(
                    aa in final_risk_codes for aa in unique_codes
                ),
                "curated_mic_row_count": curated_mic_counts[record_id],
            }
        )
    exclusion_path = args.output_dir / "flagged_peptide_exclusions.csv"
    with exclusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(exclusion_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(exclusion_rows)

    chemical_by_id = {
        int(row["dbaasp_id"]): row for row in exclusion_rows
    }
    with args.roundtrip_records.open(newline="", encoding="utf-8") as handle:
        roundtrip_rows = [
            row
            for row in csv.DictReader(handle)
            if row["record_kind"] == "dbaasp_peptide"
        ]
    forward_failure_by_id = {
        int(row["dbaasp_id"]): row
        for row in roundtrip_rows
        if row["forward_status"] == "failed"
    }
    sensitivity_ids = sorted(set(chemical_by_id) | set(forward_failure_by_id))
    sensitivity_rows = []
    for record_id in sensitivity_ids:
        chemical = chemical_by_id.get(record_id, {})
        forward = forward_failure_by_id.get(record_id, {})
        sensitivity_rows.append(
            {
                "dbaasp_id": record_id,
                "chemical_adjudication_flag": bool(chemical),
                "flagged_aa_codes": chemical.get("flagged_aa_codes", ""),
                "decision_classes": chemical.get("decision_classes", ""),
                "forward_generation_failed": bool(forward),
                "forward_error_class": forward.get("forward_error_class", ""),
                "forward_error": forward.get("forward_error", ""),
                "curated_mic_row_count": curated_mic_counts[record_id],
            }
        )
    sensitivity_path = (
        args.output_dir / "peplink_validation_sensitivity_exclusions.csv"
    )
    with sensitivity_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(sensitivity_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    source_paths = {
        "source_annotations": args.source_annotations,
        "gpt_names": args.gpt_names,
        "opsin_smiles": args.opsin_smiles,
        "historical_pipeline_csv": args.historical_pipeline_csv,
        "final_aa_mapping": args.final_aa_mapping,
        "dbaasp_json": args.dbaasp_json,
        "curated_mic": args.curated_mic,
    }
    summary = {
        "schema_version": 2,
        "scope_notice": (
            "This file summarizes the complete 169-row chemical audit, not the "
            "reviewer-facing training-impact scope. Use "
            "reviewer_response_scope_summary.json for the source-aware "
            "56-peptide/219-row scope."
        ),
        "lineage_rows": len(rows),
        "source_hashes": {
            name: {"path": display_path(path), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "automated_checks": {
            "opsin_rdkit_valid": sum(row["opsin_rdkit_valid"] for row in rows),
            "final_mapping_rdkit_valid": sum(
                row["final_mapping_rdkit_valid"] for row in rows
            ),
            "opsin_selfies_structure_exact": sum(
                row["opsin_selfies_structure_exact"] for row in rows
            ),
            "final_exactly_matches_opsin": sum(
                row["final_exactly_matches_opsin"] for row in rows
            ),
            "final_structure_matches_opsin": sum(
                row["final_structure_matches_opsin"] for row in rows
            ),
            "charged_opsin_structures": sum(
                row["formal_charge"] != 0 for row in rows
            ),
            "structures_with_unassigned_stereocenters": sum(
                row["unassigned_chiral_centers"] > 0 for row in rows
            ),
        },
        "manual_review": {
            "complete": set(manual) == set(annotation_codes),
            "reviewed_rows": len(manual),
            "decision_counts": dict(
                sorted(Counter(row["manual_decision"] for row in rows).items(), key=lambda item: str(item[0]))
            ),
            "used_aa_by_decision": dict(
                sorted(
                    Counter(
                        row["manual_decision"]
                        for row in rows
                        if row["used_in_curated_dbaasp"]
                    ).items(),
                    key=lambda item: str(item[0]),
                )
            ),
            "affected_unique_peptides_by_decision": {
                decision: len(
                    set().union(
                        *(
                            usage_ids[row["aa"]]
                            for row in rows
                            if row["manual_decision"] == decision
                        )
                    )
                )
                for decision in sorted(
                    {row["manual_decision"] for row in rows}, key=str
                )
            },
        },
        "detail_csv": display_path(detail_path),
        "conservative_exclusion": {
            "csv": display_path(exclusion_path),
            "flagged_aa_definitions": len(flagged_codes),
            "affected_unique_peptides": len(exclusion_rows),
            "affected_mic_rows": sum(
                row["curated_mic_row_count"] for row in exclusion_rows
            ),
            "contains_handcrafted_replacement_peptides": sum(
                row["contains_handcrafted_replacement"] for row in exclusion_rows
            ),
            "contains_unresolved_final_risk_peptides": sum(
                row["contains_unresolved_final_risk"] for row in exclusion_rows
            ),
        },
        "deprecated_exploratory_exclusion_do_not_use_for_reviewer_response": {
            "csv": display_path(sensitivity_path),
            "affected_unique_peptides": len(sensitivity_rows),
            "affected_mic_rows": sum(
                row["curated_mic_row_count"] for row in sensitivity_rows
            ),
            "chemical_flag_peptides": len(chemical_by_id),
            "forward_generation_failed_peptides": len(forward_failure_by_id),
            "overlap_peptides": len(
                set(chemical_by_id) & set(forward_failure_by_id)
            ),
            "status": "retired_after_source_lineage_audit",
            "reason": (
                "Forward-construction failure is not equivalent to a frozen "
                "training-structure error; many such records use DBAASP-linked "
                "whole-peptide PubChem structures."
            ),
        },
    }
    summary_path = args.output_dir / "chatgpt_opsin_chemical_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
