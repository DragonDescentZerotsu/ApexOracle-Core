from __future__ import annotations

import json
from pathlib import Path

import pytest

from apexoracle.resources import resolve_weight
from apexoracle.vendor.peptideclm_tokenizer import load_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "reproducibility/pre_generation_cleanup_2026-07-20.json"


def test_cleanup_manifest_and_removed_legacy_entries() -> None:
    audit = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert audit["status"] == "completed"
    assert audit["verified_facts"]["original_paper_data_modified"] is False
    assert audit["verified_facts"]["paper_checkpoint_deleted"] is False
    assert all(not (REPO_ROOT / item["path"]).exists() for item in audit["root_drivers"])
    assert not (REPO_ROOT / "train_on_all_data.py").exists()
    assert not (REPO_ROOT / "compare_APEX").exists()
    assert not (REPO_ROOT / "PeptideCLM").exists()
    assert not (REPO_ROOT / "capsule").exists()


def test_vendored_peptideclm_tokenizer_contract() -> None:
    tokenizer = load_tokenizer()
    assert tokenizer("CCO")["input_ids"] == [2, 32, 3]
    assert tokenizer("AXD")["input_ids"] == [2, 1, 3]


@pytest.mark.skipif(
    not (REPO_ROOT / "weights/molecule_encoders/apex/APEX_pretrained_encoder_state_dict_best.ckpt").exists(),
    reason="ignored APEX paper checkpoint is not installed",
)
def test_canonical_apex_weight_is_resolved_by_manifest_id() -> None:
    checkpoint = resolve_weight("fig2b_apex_encoder", repo_root=REPO_ROOT)
    assert checkpoint.name == "APEX_pretrained_encoder_state_dict_best.ckpt"
