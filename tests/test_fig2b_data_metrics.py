import numpy as np
import pandas as pd
import pytest

from apexoracle.benchmarks.molecule_encoders.data import (
    load_shared_benchmark,
    transform_mic_labels,
)
from apexoracle.benchmarks.molecule_encoders.metrics import (
    finite_macro_mean,
    masked_r2_per_task,
)
from apexoracle.benchmarks.molecule_encoders.protocol import DEFAULT_TARGET_COLUMNS


def _write_shared_fixture(tmp_path, number_of_rows=20):
    rows = []
    folds = []
    for index in range(number_of_rows):
        row = {
            "dbaasp_id": str(index),
            "smiles": f"SMILES-{index}",
            "apex_sequence": "AXD",
        }
        row.update({column: float(index + 1) for column in DEFAULT_TARGET_COLUMNS})
        rows.append(row)
        folds.append({"dbaasp_id": str(index), "fold": index % 5})
    pd.DataFrame(reversed(rows)).to_csv(tmp_path / "shared_molecules.csv", index=False)
    pd.DataFrame(folds).to_csv(tmp_path / "folds.csv", index=False)


def test_transform_mic_labels_masks_missing_values():
    transformed, mask = transform_mic_labels(np.array([[10.0, -1.0], [0.1, 100.0]]))

    np.testing.assert_allclose(transformed, [[0.0, 0.0], [2.0, -1.0]], atol=1e-6)
    assert mask.tolist() == [[True, False], [True, True]]


def test_shared_loader_aligns_ids_and_keeps_test_out_of_validation(tmp_path):
    _write_shared_fixture(tmp_path)
    data = load_shared_benchmark(tmp_path)

    assert data.molecule_ids == tuple(sorted(data.molecule_ids))
    outer_train, outer_test = data.outer_fold_indices(2)
    inner_train, validation = data.train_validation_indices(2, seed=17)
    assert set(inner_train).isdisjoint(validation)
    assert set(inner_train) | set(validation) == set(outer_train)
    assert (set(inner_train) | set(validation)).isdisjoint(outer_test)


def test_shared_loader_rejects_fold_id_drift(tmp_path):
    _write_shared_fixture(tmp_path)
    folds = pd.read_csv(tmp_path / "folds.csv", dtype={"dbaasp_id": str})
    folds.loc[0, "dbaasp_id"] = "missing-id"
    folds.to_csv(tmp_path / "folds.csv", index=False)

    with pytest.raises(ValueError, match="ID sets differ"):
        load_shared_benchmark(tmp_path)


def test_masked_r2_and_macro_mean_handle_undefined_tasks():
    labels = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    predictions = np.array([[0.0, 9.0], [1.0, 9.0], [2.0, 9.0]])
    mask = np.ones_like(labels, dtype=bool)

    per_task = masked_r2_per_task(labels, predictions, mask)
    assert per_task == [1.0, None]
    assert finite_macro_mean(per_task) == 1.0
