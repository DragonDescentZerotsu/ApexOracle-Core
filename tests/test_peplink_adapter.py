from __future__ import annotations

from types import SimpleNamespace

import pytest

from apexoracle.data import peplink_adapter


def test_dependency_error_explains_pinned_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(peplink_adapter, "import_module", fail_import)
    with pytest.raises(peplink_adapter.PepLinkDependencyError, match="PepLink==0.1.1"):
        peplink_adapter.load_peplink()


def test_dbaasp_adapter_uses_only_public_peplink_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Input:
        def to_api_kwargs(self):
            return {"sequence": "AC", "n_terminal": "ACT"}

    fake = SimpleNamespace(
        from_dbaasp_record=lambda record: calls.append(("adapt", record)) or Input(),
        aa_seqs_to_smiles=lambda **kwargs: calls.append(("convert", kwargs))
        or "SMILES",
    )
    monkeypatch.setattr(peplink_adapter, "load_peplink", lambda: fake)
    record = {"id": 11}
    assert peplink_adapter.dbaasp_record_to_smiles(record) == "SMILES"
    assert calls == [
        ("adapt", record),
        ("convert", {"sequence": "AC", "n_terminal": "ACT"}),
    ]


def test_release_provenance_is_frozen() -> None:
    assert peplink_adapter.PEPLINK_VERSION == "0.1.1"
    assert len(peplink_adapter.PEPLINK_GIT_COMMIT) == 40


def test_sequence_adapter_uses_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake = SimpleNamespace(
        aa_seqs_to_smiles=lambda **kwargs: calls.append(kwargs) or "SMILES"
    )
    monkeypatch.setattr(peplink_adapter, "load_peplink", lambda: fake)
    assert peplink_adapter.sequence_to_smiles("AC") == "SMILES"
    assert calls == [{"sequence": "AC"}]
