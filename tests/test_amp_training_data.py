from __future__ import annotations

import numpy as np
import pandas as pd

from apexoracle.data.amp_training_data import (
    format_inhouse_mic_table,
    merge_mic_tables,
    tokenize_and_filter_smiles,
)


class FakeTokenizer:
    unk_token_id = 99

    def __call__(self, value: str, *, add_special_tokens: bool):
        assert add_special_tokens is True
        if "unknown" in value:
            return {"input_ids": [1, 99, 2]}
        return {"input_ids": [1] + list(range(len(value))) + [2]}


def _table(smiles: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DBAASP_id": [str(i) for i in range(len(smiles))],
            "strain_name": ["strain"] * len(smiles),
            "SMILES": smiles,
            "MIC": [1.0] * len(smiles),
        }
    )


def test_format_inhouse_preserves_row_ids_and_sentinels() -> None:
    source = pd.DataFrame(
        {
            "Peptide": ["AA", "BB"],
            "strain-a": [np.inf, -1000],
            "strain-b": [2, 4],
            "Group": ["CV", "CV"],
        }
    )
    result = format_inhouse_mic_table(source, {"AA": "CC", "BB": "NN"})
    assert result[["DBAASP_id", "strain_name", "MIC"]].to_dict("records") == [
        {"DBAASP_id": "#0", "strain_name": "strain-a", "MIC": 512.0},
        {"DBAASP_id": "#0", "strain_name": "strain-b", "MIC": 2.0},
        {"DBAASP_id": "#1", "strain_name": "strain-b", "MIC": 4.0},
    ]


def test_merge_preserves_order_and_inputs() -> None:
    first = _table(["CC"])
    second = _table(["NN"])
    result = merge_mic_tables(first, second)
    assert result["SMILES"].tolist() == ["CC", "NN"]
    assert len(first) == len(second) == 1


def test_token_filter_caches_smiles_and_counts_rows() -> None:
    calls: list[str] = []

    def encode(value: str) -> str:
        calls.append(value)
        if value == "bad":
            raise ValueError("invalid")
        return value

    result = tokenize_and_filter_smiles(
        _table(["ok", "ok", "bad", "unknown", "long"]),
        selfies_encoder=encode,
        tokenizer=FakeTokenizer(),
        max_length=5,
    )
    assert calls == ["ok", "bad", "unknown", "long"]
    assert len(result.table) == 2
    assert result.excluded_invalid_smiles == 1
    assert result.excluded_unknown_token == 1
    assert result.excluded_too_long == 1
