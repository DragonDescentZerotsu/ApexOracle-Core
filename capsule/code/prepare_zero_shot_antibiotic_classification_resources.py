#!/usr/bin/env python3
"""Prepare capsule resources for zero-shot antibiotic classification.

This copies only the resources needed by the inference-only reproduction script
and writes optimizer-free copies of the trained 10-ensemble checkpoints.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm


RESOURCE_DIRS = [
    ("DataPrepare/Data/Genome_embs", "DataPrepare/Data/Genome_embs"),
    ("DataPrepare/Data/Text_Description/ATCC/embeddings", "DataPrepare/Data/Text_Description/ATCC/embeddings"),
    ("DataPrepare/Data/Text_Description/wo_ATCC/embeddings", "DataPrepare/Data/Text_Description/wo_ATCC/embeddings"),
]

RESOURCE_FILES = [
    ("DataPrepare/Data/Pep_emb_dict_cls_wo_pad_eval.pt", "DataPrepare/Data/Pep_emb_dict_cls_wo_pad_eval.pt"),
    ("DataPrepare/Data/SM_emb_dict_cls_wo_pad_eval.pt", "DataPrepare/Data/SM_emb_dict_cls_wo_pad_eval.pt"),
    (
        "DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data_SELFIES.csv",
        "DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data_SELFIES.csv",
    ),
    (
        "antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py",
        "source/antibiotic_3_strain_compare_MDLM_fix_cls_wo_pad_all_test.py",
    ),
]

CHECKPOINT_SRC = Path(
    "Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare/"
    "MDLM_fix_cls_sm_all_test_10_fold_ensembles"
)
CHECKPOINT_DST = CHECKPOINT_SRC
KEEP_CHECKPOINT_KEYS = [
    "auroc",
    "auprc",
    "re_head_state_dict",
    "cls_head_state_dict",
    "co_cross_attn_genome",
    "co_cross_attn_text",
    "learnable_embedding_weight",
]


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def strip_checkpoint(src: Path, dst: Path, overwrite: bool) -> dict:
    if dst.exists() and not overwrite:
        return {"file": str(dst), "status": "exists", "bytes": dst.stat().st_size}

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    stripped = {key: checkpoint[key] for key in KEEP_CHECKPOINT_KEYS if key in checkpoint}
    if "optimizer_state_dict" in stripped:
        raise RuntimeError("optimizer_state_dict unexpectedly kept")
    missing = sorted(set(KEEP_CHECKPOINT_KEYS) - set(stripped))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, dst)
    return {
        "file": str(dst),
        "status": "written",
        "source_bytes": src.stat().st_size,
        "bytes": dst.stat().st_size,
        "kept_keys": sorted(stripped.keys()),
        "missing_keys": missing,
    }


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    capsule_data = args.capsule_data.resolve()
    manifest = {"resource_dirs": [], "resource_files": [], "checkpoints": []}

    for src_rel, dst_rel in RESOURCE_DIRS:
        src = repo_root / src_rel
        dst = capsule_data / dst_rel
        print(f"Copying directory {src} -> {dst}")
        copy_tree(src, dst, args.overwrite)
        manifest["resource_dirs"].append({"source": src_rel, "destination": dst_rel, "bytes": directory_size(dst)})

    for src_rel, dst_rel in RESOURCE_FILES:
        src = repo_root / src_rel
        dst = capsule_data / dst_rel
        print(f"Copying file {src} -> {dst}")
        copy_file(src, dst, args.overwrite)
        manifest["resource_files"].append({"source": src_rel, "destination": dst_rel, "bytes": dst.stat().st_size})

    src_ckpt_dir = repo_root / CHECKPOINT_SRC
    dst_ckpt_dir = capsule_data / CHECKPOINT_DST
    for log_file in sorted(src_ckpt_dir.glob("*.log")):
        copy_file(log_file, dst_ckpt_dir / log_file.name, args.overwrite)

    checkpoint_files = sorted(src_ckpt_dir.glob("*.pth"))
    expected_count = args.expected_checkpoints
    if expected_count is not None and len(checkpoint_files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} checkpoints, found {len(checkpoint_files)} in {src_ckpt_dir}")

    for checkpoint_file in tqdm(checkpoint_files, desc="Stripping checkpoint optimizers"):
        info = strip_checkpoint(checkpoint_file, dst_ckpt_dir / checkpoint_file.name, args.overwrite)
        manifest["checkpoints"].append(info)

    manifest_path = capsule_data / "zero_shot_antibiotic_classification_resource_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--capsule-data", type=Path, default=Path("capsule/data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-checkpoints", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
