"""Audit native encoder processability before freezing the shared ID set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .encoders import HF_ENCODERS, load_hf_tokenizer
from .protocol import project_apex_sequence, sha256_file


ELIGIBILITY_VERSION = "fig2b-native-intersection-v1"


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {str(record.get("id")): record for record in records}


def _token_length_eligible(tokenizer, text: str, *, reject_unknown: bool) -> bool:
    token_ids = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    if len(token_ids) > 512:
        return False
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    return not (reject_unknown and unknown_id is not None and unknown_id in token_ids)


def audit_encoder_eligibility(
    *,
    mic_csv: Path,
    selfies_csv: Path,
    dbaasp_records_json: Path,
    repo_root: Path,
    output_csv: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Write native processability flags and their exact all-encoder intersection."""

    mic_csv = Path(mic_csv).resolve()
    selfies_csv = Path(selfies_csv).resolve()
    dbaasp_records_json = Path(dbaasp_records_json).resolve()
    repo_root = Path(repo_root).resolve()
    source = pd.read_csv(mic_csv, dtype={"DBAASP_id": "string"})
    source["DBAASP_id"] = source["DBAASP_id"].str.strip()
    if source["DBAASP_id"].isna().any() or source["DBAASP_id"].duplicated().any():
        raise ValueError("MIC source IDs must be present and unique")
    ids = source["DBAASP_id"].astype(str).tolist()
    smiles_by_id = dict(zip(ids, source["SMILES"].astype(str)))

    flags: dict[str, dict[str, bool]] = {molecule_id: {} for molecule_id in ids}
    for encoder_name, spec in HF_ENCODERS.items():
        tokenizer = load_hf_tokenizer(spec, repo_root)
        for molecule_id in ids:
            flags[molecule_id][encoder_name] = _token_length_eligible(
                tokenizer,
                smiles_by_id[molecule_id],
                reject_unknown=False,
            )

    from transformers import AutoTokenizer

    dlm_source = pd.read_csv(selfies_csv, dtype={"DBAASP_id": "string"})
    dlm_source["DBAASP_id"] = dlm_source["DBAASP_id"].str.strip()
    if dlm_source["DBAASP_id"].duplicated().any():
        raise ValueError("DLM SELFIES source contains duplicate IDs")
    selfies_by_id = dict(zip(dlm_source["DBAASP_id"].astype(str), dlm_source["SMILES"].astype(str)))
    if set(selfies_by_id) != set(ids):
        raise ValueError("DLM SELFIES source IDs do not match the MIC source IDs")
    dlm_tokenizer = AutoTokenizer.from_pretrained("ibm-research/materials.selfies-ted")
    for molecule_id in ids:
        eligible = _token_length_eligible(
            dlm_tokenizer,
            selfies_by_id[molecule_id].replace("][", "] ["),
            reject_unknown=True,
        )
        flags[molecule_id]["dlm_mtr_dlm"] = eligible
        flags[molecule_id]["dlm_only"] = eligible

    records = _load_records(dbaasp_records_json)
    for molecule_id in ids:
        record = records.get(molecule_id)
        eligible = False
        if record is not None and isinstance(record.get("sequence"), str) and record["sequence"].strip():
            projection = project_apex_sequence(record["sequence"], max_residues=52)
            eligible = projection.original_residue_count <= 52
        flags[molecule_id]["apex"] = eligible

    encoder_columns = (
        "dlm_mtr_dlm",
        "dlm_only",
        "chemberta_mtr",
        "chemberta_mlm",
        "molformer",
        "peptideclm",
        "apex",
    )
    rows = []
    for molecule_id in ids:
        row: dict[str, Any] = {"dbaasp_id": molecule_id}
        row.update({name: flags[molecule_id][name] for name in encoder_columns})
        row["eligible_all"] = all(row[name] for name in encoder_columns)
        rows.append(row)
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_manifest = Path(output_manifest)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    counts = {name: int(result[name].sum()) for name in encoder_columns}
    counts["eligible_all"] = int(result["eligible_all"].sum())
    manifest = {
        "eligibility_version": ELIGIBILITY_VERSION,
        "policy": "intersection of IDs accepted by every original native preprocessing path",
        "rules": {
            "chemberta_mtr": "original tokenizer length <= 512; UNK allowed",
            "chemberta_mlm": "original tokenizer length <= 512; UNK allowed",
            "molformer": "original tokenizer length <= 512; UNK allowed",
            "peptideclm": "original tokenizer length <= 512; UNK allowed",
            "dlm_mtr_dlm": "SELFIES tokenizer length <= 512 and no UNK",
            "dlm_only": "SELFIES tokenizer length <= 512 and no UNK",
            "apex": "original 23-token model; projected residue sequence length <= 52",
        },
        "counts": counts,
        "sources": {
            "mic_csv_sha256": sha256_file(mic_csv),
            "selfies_csv_sha256": sha256_file(selfies_csv),
            "dbaasp_records_json_sha256": sha256_file(dbaasp_records_json),
        },
    }
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest
