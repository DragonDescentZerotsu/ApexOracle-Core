import json

import pandas as pd
import pytest

from apexoracle.benchmarks.molecule_encoders.apex_adapter import (
    build_apex_vocabulary,
    encode_apex_sequences,
)
from apexoracle.benchmarks.molecule_encoders.protocol import (
    DEFAULT_TARGET_COLUMNS,
    EXPECTED_ENCODERS,
    assign_folds,
    build_shared_dataset,
    project_apex_sequence,
)


def test_apex_adapter_preserves_original_unknown_symbol_behavior():
    vocabulary, _ = build_apex_vocabulary()
    token_ids, attention_mask = encode_apex_sequences(["AXD", "A" * 60])

    assert "X" not in vocabulary
    assert len(vocabulary) == 23
    assert token_ids[0, :5].tolist() == [1, 3, 0, 5, 2]
    assert attention_mask[0, :5].tolist() == [1, 1, 1, 1, 1]
    assert token_ids[1, -1] == 2
    assert attention_mask[1].sum() == 52


def test_apex_projection_records_every_lossy_rule():
    projection = project_apex_sequence(
        "cyclo-AcX O/d",
        unusual_positions=(2,),
        has_topology_or_multiple_chains=True,
        max_residues=4,
    )

    assert projection.sequence == "AXXX"
    assert projection.original_residue_count == 5
    assert projection.contained_noncanonical is True
    assert projection.contained_d_residue is True
    assert projection.removed_topology_or_chain_marker is True
    assert projection.truncated is True
    assert projection.unusual_position_count == 1


def test_apex_projection_rejects_empty_sequence():
    with pytest.raises(ValueError, match="non-empty"):
        project_apex_sequence("  ")


def test_fold_assignment_is_order_independent_and_balanced():
    ids = [str(value) for value in range(23)]
    first = assign_folds(ids, n_splits=5, seed=42)
    second = assign_folds(reversed(ids), n_splits=5, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["dbaasp_id"].nunique() == len(ids)
    counts = first["fold"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_shared_dataset_writes_ids_folds_exclusions_and_manifest(tmp_path):
    mic_path = tmp_path / "mic.csv"
    records_path = tmp_path / "records.json"
    eligibility_path = tmp_path / "eligibility.csv"
    output_dir = tmp_path / "output"

    rows = []
    for index in range(6):
        row = {"DBAASP_id": str(index), "SMILES": f"SMILES-{index}"}
        row.update({column: -1.0 for column in DEFAULT_TARGET_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(mic_path, index=False)

    records = [
        {"id": 0, "sequence": "ACD", "unusualAminoAcids": []},
        {"id": 1, "sequence": "AcD", "unusualAminoAcids": []},
        {"id": 2, "sequence": "AXD", "unusualAminoAcids": []},
        {"id": 3, "sequence": "ACD", "unusualAminoAcids": [{"position": 2}]},
        {"id": 4, "sequence": "cyclo-ACD", "unusualAminoAcids": []},
        {"id": 5, "sequence": None, "unusualAminoAcids": []},
    ]
    records_path.write_text(json.dumps(records), encoding="utf-8")
    eligibility_rows = []
    for index in range(6):
        row = {"dbaasp_id": str(index)}
        row.update({encoder: index < 5 for encoder in EXPECTED_ENCODERS})
        row["eligible_all"] = index < 5
        eligibility_rows.append(row)
    pd.DataFrame(eligibility_rows).to_csv(eligibility_path, index=False)

    manifest = build_shared_dataset(
        mic_csv=mic_path,
        dbaasp_records_json=records_path,
        eligibility_csv=eligibility_path,
        output_dir=output_dir,
        n_splits=5,
        seed=42,
        apex_max_residues=50,
    )

    assert manifest["counts"] == {
        "source_molecules": 6,
        "shared_molecules": 5,
        "excluded_molecules": 1,
    }
    assert len(pd.read_csv(output_dir / "common_molecule_ids.csv")) == 5
    assert len(pd.read_csv(output_dir / "folds.csv")) == 5
    exclusions = pd.read_csv(output_dir / "exclusions.csv", dtype={"dbaasp_id": str})
    assert exclusions.to_dict("records") == [
        {
            "dbaasp_id": "5",
            "stage": "native_encoder_intersection",
            "reason": ";".join(EXPECTED_ENCODERS),
        }
    ]
