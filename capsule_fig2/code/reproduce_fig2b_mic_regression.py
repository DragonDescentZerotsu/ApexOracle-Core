#!/usr/bin/env python
"""Inference-only reproduction for Fig. 2b MIC regression.

The capsule resources contain frozen molecular feature caches and trained
5-fold regression-head checkpoints. This script reloads those heads and
recomputes the fold-wise R2 metrics without training or running upstream
backbone models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from tqdm import tqdm


DEFAULT_MODELS = [
    "mdlm_dlm_mtr",
    "chemberta_mtr",
    "molformer",
    "apex",
    "peptideclm",
    "chemberta_mlm_mean",
    "chemberta_mlm",
]


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 384,
        hidden_dim_2: int = 128,
        num_targets: int = 19,
        pooler_dropout: float = 0.2,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features):
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


def calculate_r2_per_task(labels, preds, masks):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    masks = np.asarray(masks)
    r2_per_task = []
    for task_idx in range(labels.shape[1]):
        mask = masks[:, task_idx].astype(bool)
        y_true = labels[mask, task_idx]
        y_pred = preds[mask, task_idx]
        if len(y_true) == 0:
            r2_per_task.append(None)
            continue
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2_per_task.append(float(1 - (ss_residual / ss_total)))
    return r2_per_task


def finite_mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_head(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", 1024))
    num_targets = int(checkpoint.get("num_targets", 19))
    hyper = checkpoint.get("hyperparameters", {})
    head = RegressionHead(
        input_dim=input_dim,
        hidden_dim_1=int(hyper.get("hidden_dim_1", 384)),
        hidden_dim_2=int(hyper.get("hidden_dim_2", 128)),
        num_targets=num_targets,
        pooler_dropout=float(hyper.get("dropout", 0.2)),
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device)
    head.eval()
    return head, checkpoint


def predict_in_batches(head, features, device, batch_size):
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device)
            preds.append(head(batch).detach().cpu())
    return torch.cat(preds, dim=0)


def prepare_checkpoint_eval_mode(head, checkpoint, device, default_batch_size):
    hyper = checkpoint.get("hyperparameters", {})
    if hyper.get("head_training_mode_during_validation") and checkpoint.get("eval_rng_state"):
        head.train()
        rng_state = checkpoint["eval_rng_state"]
        torch.set_rng_state(rng_state["cpu"])
        if device.type == "cuda" and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state(rng_state["cuda"], device)
        return int(hyper.get("validation_batch_size", default_batch_size))
    head.eval()
    return default_batch_size


def evaluate_model(model_name: str, data_root: Path, results_dir: Path, device, batch_size, write_predictions):
    model_root = data_root / "fig2b_mic_regression" / model_name
    feature_path = model_root / "features.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature cache for {model_name}: {feature_path}")

    cache = torch_load(feature_path, map_location="cpu")
    features = cache["features"].float()
    labels = cache["labels"].float().numpy()
    masks = cache["label_masks"].float().numpy()
    target_columns = list(cache["target_columns"])
    dbaasp_ids = [str(x) for x in cache.get("dbaasp_ids", list(range(features.shape[0])))]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    prediction_rows = []

    for fold_idx, (_, test_idx) in enumerate(
        tqdm(list(kf.split(np.arange(features.shape[0]))), desc=f"{model_name} folds"),
        start=1,
    ):
        ckpt_path = model_root / f"fold_{fold_idx}" / "best_head.pt"
        head, checkpoint = load_head(ckpt_path, device)
        fold_features = features[test_idx]
        predict_batch_size = prepare_checkpoint_eval_mode(head, checkpoint, device, batch_size)
        fold_preds = predict_in_batches(head, fold_features, device, predict_batch_size).numpy()
        fold_labels = labels[test_idx]
        fold_masks = masks[test_idx]
        r2_per_task = calculate_r2_per_task(fold_labels, fold_preds, fold_masks)
        r2_mean = finite_mean(r2_per_task)
        fold_results.append(
            {
                "fold": fold_idx,
                "r2_mean": r2_mean,
                "r2_per_task": r2_per_task,
                "checkpoint_recorded_best_r2_mean": checkpoint.get("best_r2_mean"),
                "checkpoint_recorded_epoch": checkpoint.get("epoch"),
                "test_size": int(len(test_idx)),
            }
        )

        if write_predictions:
            for local_row, global_idx in enumerate(test_idx):
                for task_idx, task_name in enumerate(target_columns):
                    if fold_masks[local_row, task_idx] <= 0:
                        continue
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "fold": fold_idx,
                            "DBAASP_id": dbaasp_ids[global_idx],
                            "task": task_name,
                            "label": float(fold_labels[local_row, task_idx]),
                            "prediction": float(fold_preds[local_row, task_idx]),
                        }
                    )

    metrics = {
        "model": model_name,
        "feature_cache": str(feature_path.relative_to(data_root)),
        "num_examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "best_mean_R2_across_folds": float(np.mean([fold["r2_mean"] for fold in fold_results])),
        "folds": fold_results,
        "target_columns": target_columns,
    }

    metrics_path = results_dir / f"fig2b_{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    if write_predictions:
        predictions_path = results_dir / f"fig2b_{model_name}_predictions.csv"
        with predictions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "fold", "DBAASP_id", "task", "label", "prediction"],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--memory-report", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.memory_report and device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    all_metrics = {}
    for model_name in args.models:
        metrics = evaluate_model(
            model_name=model_name,
            data_root=args.data_root,
            results_dir=args.results_dir,
            device=device,
            batch_size=args.batch_size,
            write_predictions=args.write_predictions,
        )
        all_metrics[model_name] = metrics["best_mean_R2_across_folds"]

    summary_path = args.results_dir / "fig2b_mic_regression_summary.json"
    summary = {"best_mean_R2_across_folds": all_metrics}
    if args.memory_report and device.type == "cuda":
        summary["cuda_memory"] = {
            "device": str(device),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
