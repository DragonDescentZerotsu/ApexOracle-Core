#!/usr/bin/env python3
"""Prepare capsule resources for Fig. 2b MIC regression eval.

The Fig. 2b capsule path is inference-only. This script copies the frozen
feature cache and already trained 5-fold regression heads produced locally,
plus small provenance files needed to audit how those resources were built.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


FIG2B_ROOT = Path("fig2b_mic_regression")
MODEL_RESOURCES = {
    "mdlm_dlm_mtr": {
        "source_dir": Path("Checkpoints/fig2b_mdlm_cached_5fold"),
        "feature_file": "fig2b_mdlm_best_first_token_embeddings.pt",
        "extra_files": ["metrics.json", "embedding_cache_manifest.json", "run.log"],
        "description": "DLM/MTR frozen first-token features with 5-fold regression heads.",
    },
    "chemberta_mtr": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/chemberta_mtr"),
        "checkpoint_source_dir": Path("Checkpoints/fig2b_baselines_online_5fold/chemberta_mtr"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "ChemBERTa-77M-MTR frozen eval-mode first-token features with faithful online-trained 5-fold regression heads.",
    },
    "chemberta_mlm": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/chemberta_mlm"),
        "checkpoint_source_dir": Path("Checkpoints/fig2b_baselines_online_5fold/chemberta_mlm"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "ChemBERTa-77M-MLM frozen eval-mode first-token features with faithful online-trained 5-fold regression heads.",
    },
    "chemberta_mlm_mean": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/chemberta_mlm_mean"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "ChemBERTa-77M-MLM frozen mean-pooled features with 5-fold regression heads.",
    },
    "molformer": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/molformer"),
        "feature_source_dir": Path("Checkpoints/fig2b_baselines_online_5fold/molformer"),
        "checkpoint_source_dir": Path("Checkpoints/fig2b_baselines_online_5fold/molformer"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "MoLFormer-XL frozen eval-mode first-token features with faithful online-trained 5-fold regression heads.",
    },
    "peptideclm": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/peptideclm"),
        "checkpoint_source_dir": Path("Checkpoints/fig2b_baselines_online_5fold/peptideclm"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "PeptideCLM frozen eval-mode first-token features with faithful online-trained 5-fold regression heads.",
    },
    "apex": {
        "source_dir": Path("Checkpoints/fig2b_baselines_cached_5fold/apex"),
        "checkpoint_source_dir": Path("Checkpoints/fig2b_apex_original_dropout_eval_seed_search/seed_2"),
        "feature_file": "features.pt",
        "extra_files": ["metrics.json"],
        "description": "APEX frozen eval-mode encoder features with original-script dropout-validation 5-fold regression heads.",
    },
}


def copy_file(src: Path, dst: Path, overwrite: bool) -> dict:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        return {"source": str(src), "destination": str(dst), "status": "exists", "bytes": dst.stat().st_size}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": str(src), "destination": str(dst), "status": "written", "bytes": dst.stat().st_size}


def validate_feature_cache(path: Path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"features", "labels", "label_masks", "target_columns", "dbaasp_ids"}
    missing = sorted(required_keys - set(cache))
    if missing:
        raise RuntimeError(f"{path} is missing required keys: {missing}")
    features = cache["features"]
    labels = cache["labels"]
    masks = cache["label_masks"]
    return {
        "num_examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_targets": int(labels.shape[1]),
        "num_observed_labels": int(masks.sum().item()),
        "target_columns": list(cache["target_columns"]),
    }


def validate_head_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer_state_dict" in checkpoint:
        raise RuntimeError(f"optimizer_state_dict should not be included in {path}")
    if "head_state_dict" not in checkpoint:
        raise RuntimeError(f"{path} does not contain head_state_dict")
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "keys": sorted(checkpoint.keys()),
        "best_r2_mean": checkpoint.get("best_r2_mean"),
        "epoch": checkpoint.get("epoch"),
        "input_dim": checkpoint.get("input_dim"),
        "num_targets": checkpoint.get("num_targets"),
    }


def copy_model_resources(model_name: str, config: dict, repo_root: Path, capsule_data: Path, overwrite: bool) -> dict:
    source_dir = repo_root / config["source_dir"]
    feature_source_dir = repo_root / config.get("feature_source_dir", config["source_dir"])
    checkpoint_source_dir = repo_root / config.get("checkpoint_source_dir", config["source_dir"])
    model_dst = capsule_data / FIG2B_ROOT / model_name
    model_manifest = {
        "model": model_name,
        "description": config["description"],
        "resource_root": str(FIG2B_ROOT / model_name),
        "files": [],
        "fold_checkpoints": [],
    }

    features_dst = model_dst / "features.pt"
    model_manifest["files"].append(copy_file(feature_source_dir / config["feature_file"], features_dst, overwrite))
    model_manifest["feature_cache"] = validate_feature_cache(features_dst)

    for rel_name in config["extra_files"]:
        src = checkpoint_source_dir / rel_name
        if src.exists():
            model_manifest["files"].append(copy_file(src, model_dst / rel_name, overwrite))

    for fold_idx in range(1, 6):
        src = checkpoint_source_dir / f"fold_{fold_idx}" / "best_head.pt"
        dst = model_dst / f"fold_{fold_idx}" / "best_head.pt"
        model_manifest["files"].append(copy_file(src, dst, overwrite))
        model_manifest["fold_checkpoints"].append(validate_head_checkpoint(dst))

    return model_manifest


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    mdlm_root = args.mdlm_root.resolve()
    capsule_data = args.capsule_data.resolve()
    manifest = {
        "description": "Fig. 2b frozen-feature 5-fold MIC regression eval resources.",
        "models": {},
        "source_files": [],
    }

    selected_models = list(MODEL_RESOURCES) if args.models == ["all"] else args.models
    for model_name in selected_models:
        if model_name not in MODEL_RESOURCES:
            raise ValueError(f"Unknown model {model_name}. Choices: {sorted(MODEL_RESOURCES)}")
        manifest["models"][model_name] = copy_model_resources(
            model_name=model_name,
            config=MODEL_RESOURCES[model_name],
            repo_root=repo_root,
            capsule_data=capsule_data,
            overwrite=args.overwrite,
        )

    source_dst = capsule_data / "source"
    source_files = [
        (repo_root / "scripts/reproduce_fig2b_mdlm_cached_5fold.py", source_dst / "reproduce_fig2b_mdlm_cached_5fold.py"),
        (repo_root / "scripts/reproduce_fig2b_baselines_cached_5fold.py", source_dst / "reproduce_fig2b_baselines_cached_5fold.py"),
        (repo_root / "scripts/reproduce_fig2b_baselines_online_5fold.py", source_dst / "reproduce_fig2b_baselines_online_5fold.py"),
        (repo_root / "scripts/reproduce_fig2b_apex_original_5fold.py", source_dst / "reproduce_fig2b_apex_original_5fold.py"),
        (repo_root / "scripts/cache_fig2b_molformer_fold_eval_features.py", source_dst / "cache_fig2b_molformer_fold_eval_features.py"),
        (mdlm_root / "DBAASP_MLM_MDLM.py", source_dst / "DBAASP_MLM_MDLM.py"),
        (repo_root / "fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py", source_dst / "fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py"),
        (repo_root / "fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py", source_dst / "fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py"),
        (repo_root / "fix_ChemBERTa_MLM_mean_emb_on_DBAASP_SMILES_5_fold_mean_MIC.py", source_dst / "fix_ChemBERTa_MLM_mean_emb_on_DBAASP_SMILES_5_fold_mean_MIC.py"),
        (repo_root / "fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py", source_dst / "fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py"),
        (repo_root / "fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py", source_dst / "fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py"),
        (repo_root / "src/apexoracle/benchmarks/molecule_encoders/apex_adapter.py", source_dst / "apex_adapter.py"),
        (repo_root / "src/apexoracle/benchmarks/molecule_encoders/apex_model.py", source_dst / "apex_model.py"),
        (repo_root / "src/apexoracle/benchmarks/molecule_encoders/legacy_training.py", source_dst / "legacy_training.py"),
    ]
    for src, dst in source_files:
        if src.exists():
            manifest["source_files"].append(copy_file(src, dst, args.overwrite))

    baseline_metrics = {}
    for model_name in MODEL_RESOURCES:
        if model_name == "mdlm_dlm_mtr":
            continue
        metrics_path = capsule_data / FIG2B_ROOT / model_name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline_metrics[model_name] = metrics["best_mean_R2_across_folds"]

    baseline_summary_path = capsule_data / FIG2B_ROOT / "baseline_metrics_summary.json"
    baseline_summary_path.write_text(
        json.dumps({"best_mean_R2_across_folds": baseline_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["source_files"].append(
        {
            "source": "generated from capsule fig2b_mic_regression/*/metrics.json",
            "destination": str(baseline_summary_path),
            "status": "written",
            "bytes": baseline_summary_path.stat().st_size,
        }
    )

    manifest_path = capsule_data / "fig2b_mic_regression_resource_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mdlm-root",
        type=Path,
        default=Path(
            os.environ.get("APEXORACLE_MDLM_DIR", str(Path.cwd().parent / "mdlm"))
        ),
    )
    parser.add_argument("--capsule-data", type=Path, default=Path("capsule/data"))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
