#!/usr/bin/env python3
"""Recalculate the source-aware reviewer scope for local peptide structures.

Coordination-bond omission is intentionally reported as an accepted preprocessing
decision, not as a chemical error.  The confirmed-error union contains only
local residue-definition, residue-template, and record-assembly issues.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from apexoracle.data.hierarchical_mic_preparation import prepare_hierarchical_mic_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "DataPrepare" / "Data"
OUTPUT_DIR = REPO_ROOT / "experiments" / "peplink_validation"

MAIN_BRANCH_ERROR_DEFINITIONS = {
    "N-TYR",
    "LYS-C18",
    "3-Me-Trp",
    "2-OH-Me-SER",
    "NNar",
    "D-3-OH-ASN",
    "IAA-Cys",
    "6F-LEU",
    "HCha",
    "D-Me-Trp",
    "BisHomo-Pra",
}
INCOMPLETE_TEMPLATE_DEFINITIONS = {"Aic", "Agb", "Nae", "MIM", "Cl-Th2CA"}
SECONDARY_BRANCH_ERROR_DEFINITIONS = {"D-End", "D-IGln"}
CATEGORY_LABELS_ZH = {
    "incorrect_main_169_definitions": "主169支路错误AA定义",
    "incomplete_or_nonpolymerizable_templates": "非完整或不可聚合残基模板",
    "incorrect_secondary_44_definitions": "二次44支路错误AA定义",
}

def unusual_names(record: dict) -> set[str]:
    return {
        name
        for unusual in record.get("unusualAminoAcids") or []
        if (
            name := unusual.get("name")
            or (unusual.get("modificationType") or {}).get("name")
        )
    }


def eligible_rows(records: np.ndarray, curated_ids: set[int]) -> list[tuple]:
    result = []
    for row in records:
        try:
            peptide_id = int(row[0])
            token_count = len(ast.literal_eval(row[2]))
        except (TypeError, ValueError, SyntaxError):
            continue
        if peptide_id in curated_ids and token_count <= 512:
            result.append(tuple(row))
    return result


def dbaasp_rows(records: np.ndarray, curated_ids: set[int]) -> list[tuple]:
    result = []
    for row in records:
        try:
            peptide_id = int(row[0])
        except (TypeError, ValueError):
            continue
        if peptide_id in curated_ids:
            result.append(tuple(row))
    return result


def pool_summary(
    rows: Iterable[tuple], category_ids: dict[str, set[int]], *, deduplicate: bool
) -> dict:
    selected = list(dict.fromkeys(rows)) if deduplicate else list(rows)
    counts = Counter(int(row[0]) for row in selected)
    peptide_ids = set(counts)
    categories = {}
    for name, ids in category_ids.items():
        overlap = peptide_ids & ids
        categories[name] = {
            "unique_peptides": len(overlap),
            "mic_rows": sum(counts[peptide_id] for peptide_id in overlap),
        }
    strict_category_names = {
        "incorrect_main_169_definitions",
        "incomplete_or_nonpolymerizable_templates",
        "incorrect_secondary_44_definitions",
    }
    reviewer_ids = (
        set().union(*(category_ids[name] for name in strict_category_names))
        & peptide_ids
    )
    reviewer_rows = sum(counts[peptide_id] for peptide_id in reviewer_ids)
    return {
        "unique_peptides": len(peptide_ids),
        "mic_rows": sum(counts.values()),
        "reviewer_facing_local_conversion_error_union": {
            "unique_peptides": len(reviewer_ids),
            "mic_rows": reviewer_rows,
            "peptide_fraction": len(reviewer_ids) / len(peptide_ids),
            "mic_row_fraction": reviewer_rows / sum(counts.values()),
        },
        "categories": categories,
    }


def main() -> None:
    curated = pd.read_csv(DATA_DIR / "DBAASP_id_bact_name_SMILES_MIC_Evo.csv")
    curated["DBAASP_id"] = curated["DBAASP_id"].astype(int)
    curated_ids = set(curated["DBAASP_id"])
    local_ids = set(
        pd.read_csv(
            DATA_DIR / "DBAASP_id_wo_existing_smiles_intra_linked_smiles.csv"
        )["DBAASP_id"].astype(int)
    )
    roundtrip = pd.read_csv(OUTPUT_DIR / "peplink_0.1.2/roundtrip_records.csv")
    validation_rows = roundtrip[
        roundtrip["forward_error_class"].eq("ValidationError")
        & roundtrip["dbaasp_id"].notna()
    ].copy()
    validation_rows["dbaasp_id"] = validation_rows["dbaasp_id"].astype(int)
    validation_rows = validation_rows[validation_rows["dbaasp_id"].isin(local_ids)]
    position_mismatch_ids = set(
        validation_rows.loc[
            validation_rows["forward_error"].str.startswith(
                "unusual_amino_acids positions"
            ),
            "dbaasp_id",
        ]
    )
    count_mismatch_ids = set(
        validation_rows.loc[
            validation_rows["forward_error"].str.startswith("sequence contains"),
            "dbaasp_id",
        ]
    )
    if (len(position_mismatch_ids), len(count_mismatch_ids)) != (9, 9):
        raise ValueError(
            "Expected 9 local position mismatches and 9 local count mismatches; "
            f"got {len(position_mismatch_ids)} and {len(count_mismatch_ids)}"
        )
    records = json.loads((DATA_DIR / "all_peptides_data.json").read_text())

    category_ids = {
        "incorrect_main_169_definitions": set(),
        "incomplete_or_nonpolymerizable_templates": set(),
        "incorrect_secondary_44_definitions": set(),
    }
    record_names: dict[int, set[str]] = {}
    for record in records:
        peptide_id = int(record["id"])
        if peptide_id not in curated_ids or peptide_id not in local_ids:
            continue
        names = unusual_names(record)
        record_names[peptide_id] = names
        if names & MAIN_BRANCH_ERROR_DEFINITIONS:
            category_ids["incorrect_main_169_definitions"].add(peptide_id)
        if names & INCOMPLETE_TEMPLATE_DEFINITIONS:
            category_ids["incomplete_or_nonpolymerizable_templates"].add(peptide_id)
        if names & SECONDARY_BRANCH_ERROR_DEFINITIONS:
            category_ids["incorrect_secondary_44_definitions"].add(peptide_id)

    prepared = prepare_hierarchical_mic_data(REPO_ROOT)
    genome_rows = eligible_rows(prepared.genome_text_records, curated_ids)
    text_rows = eligible_rows(prepared.genome_or_text_records, curated_ids)
    frozen_rows = [
        (row.DBAASP_id, row.strain_name, row.SMILES, row.MIC)
        for row in curated.itertuples(index=False)
    ]

    output = {
        "schema_version": 1,
        "scope_decision": {
            "coordination_bond_omission": "accepted_preprocessing_not_counted_as_error",
            "dbaasp_annotation_inconsistency": (
                "excluded_upstream_source_data_quality_issue_outside_reviewer_question"
            ),
            "polymer_proxy": "excluded_by_author_decision",
            "whole_peptide_pubchem": "excluded_from_local_builder_error_scope",
        },
        "excluded_upstream_dbaasp_annotation_data_quality": {
            "position_mismatch_ids": sorted(position_mismatch_ids),
            "count_mismatch_ids": sorted(count_mismatch_ids),
            "note": (
                "These records document inconsistent DBAASP sequence/annotation "
                "fields and historical producer fallback behavior. They are not "
                "ChatGPT-o1/OPSIN or PepLink conversion errors and are excluded "
                "from the reviewer-facing error scope."
            ),
        },
        "confirmed_error_categories": {
            name: sorted(ids) for name, ids in category_ids.items()
        },
        "category_overlaps": {
            f"{left}__AND__{right}": sorted(category_ids[left] & category_ids[right])
            for index, left in enumerate(category_ids)
            for right in list(category_ids)[index + 1 :]
            if category_ids[left] & category_ids[right]
        },
        "data_pools": {
            "frozen_dbaasp_mic": pool_summary(
                frozen_rows, category_ids, deduplicate=False
            ),
            "canonical_genome_text_le_512": pool_summary(
                genome_rows, category_ids, deduplicate=False
            ),
            "canonical_genome_or_text_before_length_filter": pool_summary(
                dbaasp_rows(prepared.genome_or_text_records, curated_ids),
                category_ids,
                deduplicate=False,
            ),
            "canonical_genome_or_text_le_512_with_path_duplicates": pool_summary(
                text_rows, category_ids, deduplicate=False
            ),
            "canonical_genome_or_text_le_512_unique_records": pool_summary(
                text_rows, category_ids, deduplicate=True
            ),
        },
        "interpretation": (
            "The genome-or-text array can contain the same assay row through more than "
            "one modality path. Use its unique-record view for prevalence and retain the "
            "non-deduplicated view only to describe loader exposure."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "recalculated_local_error_scope.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    csv_path = OUTPUT_DIR / "recalculated_local_error_peptides.csv"
    confirmed_union = set().union(*category_ids.values())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["DBAASP_id", "问题类别", "非标准残基名称"],
            lineterminator="\n",
        )
        writer.writeheader()
        for peptide_id in sorted(confirmed_union):
            writer.writerow(
                {
                    "DBAASP_id": peptide_id,
                    "问题类别": ";".join(
                        CATEGORY_LABELS_ZH[name]
                        for name, ids in category_ids.items()
                        if peptide_id in ids
                    ),
                    "非标准残基名称": ";".join(
                        sorted(record_names.get(peptide_id, set()))
                    ),
                }
            )

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
