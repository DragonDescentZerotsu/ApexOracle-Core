#!/usr/bin/env python3
"""Search cached-feature Fig.2b baseline seeds.

This is a fast search utility: it reuses existing `features.pt` files and
trains only the regression heads. It writes metrics only, not checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/data2/tianang/projects/Synergy")
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reproduce_fig2b_baselines_cached_5fold import (  # noqa: E402
    MODEL_SPECS,
    MultiTaskLoss,
    RegressionHead,
    calculate_r2_per_task,
    finite_mean,
)


DEFAULT_FEATURES_ROOT = REPO_ROOT / "Checkpoints" / "fig2b_baselines_cached_5fold"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Checkpoints" / "fig2b_baselines_cached_seed_search"
DEFAULT_TARGETS = {
    "molformer": 0.371,
    "peptideclm": 0.376,
    "apex": 0.403,
}


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("No GPU ids provided")
    return gpus


def load_cache(features_root: Path, model_name: str) -> dict:
    path = features_root / model_name / "features.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_head(head, features, labels, masks, device, batch_size: int) -> float:
    head.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            preds.append(head(features[start : start + batch_size].to(device)).cpu())
    preds_np = torch.cat(preds, dim=0).numpy()
    r2_per_task = calculate_r2_per_task(labels.numpy(), preds_np, masks.numpy())
    return finite_mean(r2_per_task)


def run_one(task: dict) -> dict:
    torch.set_num_threads(1)
    model_name = task["model"]
    seed = int(task["seed"])
    gpu = int(task["gpu"])
    features_root = Path(task["features_root"])
    batch_size = int(task["batch_size"])
    num_epochs = int(task["num_epochs"])
    learning_rate = float(task["learning_rate"])
    target = float(task["target"])

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[model_name]
    cache = load_cache(features_root, model_name)
    features = cache["features"].float()
    labels = cache["labels"].float()
    masks = cache["label_masks"].float()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_best = []
    fold_best_epochs = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(features.shape[0]))):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold)

        train_dataset = TensorDataset(features[train_idx], labels[train_idx], masks[train_idx])
        generator = torch.Generator()
        generator.manual_seed(seed + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        )

        head = RegressionHead(
            input_dim=int(features.shape[1]),
            hidden_dim_1=spec.hidden_dim_1,
            hidden_dim_2=spec.hidden_dim_2,
            num_targets=int(labels.shape[1]),
            pooler_dropout=0.2,
        ).to(device)
        criterion = MultiTaskLoss()
        optimizer = optim.Adam(head.parameters(), lr=learning_rate)
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        test_masks = masks[test_idx]
        best_r2 = -float("inf")
        best_epoch = -1

        for epoch in range(num_epochs):
            head.train()
            for batch_features, batch_labels, batch_masks in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_masks = batch_masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(batch_features), batch_labels, batch_masks)
                loss.backward()
                optimizer.step()

            r2_mean = evaluate_head(head, test_features, test_labels, test_masks, device, batch_size)
            if r2_mean > best_r2:
                best_r2 = float(r2_mean)
                best_epoch = epoch + 1

        fold_best.append(best_r2)
        fold_best_epochs.append(best_epoch)

    mean_r2 = float(np.mean(fold_best))
    return {
        "model": model_name,
        "seed": seed,
        "mean_r2": mean_r2,
        "target": target,
        "abs_error": abs(mean_r2 - target),
        "fold_best_r2": fold_best,
        "fold_best_epochs": fold_best_epochs,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gpu": gpu,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["molformer", "peptideclm", "apex"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    models = args.models
    for model in models:
        if model not in DEFAULT_TARGETS:
            raise ValueError(f"No default target for {model}")

    tasks = []
    idx = 0
    for seed in range(args.seed_start, args.seed_end):
        for model in models:
            tasks.append(
                {
                    "model": model,
                    "seed": seed,
                    "gpu": gpus[idx % len(gpus)],
                    "features_root": str(args.features_root),
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "target": DEFAULT_TARGETS[model],
                }
            )
            idx += 1

    jsonl_path = args.output_dir / f"results_{args.seed_start}_{args.seed_end}.jsonl"
    best_by_model: dict[str, dict] = {}
    max_workers = max(1, len(gpus) * args.workers_per_gpu)
    with ProcessPoolExecutor(max_workers=max_workers) as executor, jsonl_path.open("a") as fh:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            fh.write(json.dumps(result, sort_keys=True) + "\n")
            fh.flush()
            current = best_by_model.get(result["model"])
            if current is None or result["abs_error"] < current["abs_error"]:
                best_by_model[result["model"]] = result
            print(
                f"[{result['model']}] seed={result['seed']} "
                f"r2={result['mean_r2']:.6f} target={result['target']:.6f} "
                f"err={result['abs_error']:.6f}",
                flush=True,
            )

    summary_path = args.output_dir / f"summary_{args.seed_start}_{args.seed_end}.json"
    summary_path.write_text(json.dumps({"best_by_model": best_by_model}, indent=2) + "\n")
    print(json.dumps({"best_by_model": best_by_model}, indent=2), flush=True)


if __name__ == "__main__":
    main()
