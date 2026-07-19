from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "reproducibility/release_cleanup_2026-07-19.json"


def test_release_cleanup_manifest_is_complete_and_targets_stay_absent() -> None:
    audit = json.loads(MANIFEST.read_text(encoding="utf-8"))
    deleted = [
        path
        for paths in audit["categories"].values()
        for path in paths
    ]

    assert len(deleted) == audit["summary"]["deleted_files"] == 59
    assert len(deleted) == len(set(deleted))
    assert all(not (REPO_ROOT / path).exists() for path in deleted)
    assert audit["summary"]["original_or_paper_data_deleted"] == 0
    assert audit["summary"]["checkpoint_or_weight_files_deleted"] == 0
    assert audit["summary"]["result_files_deleted"] == 0
