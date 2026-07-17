#!/usr/bin/env python
"""Cache Fig.2b DLM embeddings and train 5-fold MIC regression heads.

This script mirrors the downstream setup in
/data2/tianang/projects/mdlm/DBAASP_MLM_MDLM.py, but separates the frozen DLM
backbone pass from the regression-head training so the expensive feature
extraction can use multiple GPUs and be reused.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer


SYNERGY_DIR = Path("/data2/tianang/projects/Synergy")
MDLM_DIR = Path("/data2/tianang/projects/mdlm")
DEFAULT_DATA_PATH = SYNERGY_DIR / "DataPrepare" / "Data" / "DBAASP_id_SELFIES_bact_MICs.csv"
DEFAULT_DLM_CKPT = MDLM_DIR / "Checkpoints_fangping" / "best.ckpt"
DEFAULT_OUTPUT_DIR = SYNERGY_DIR / "Checkpoints" / "fig2b_mdlm_cached_5fold"
MODEL_NAME = "ibm-research/materials.selfies-ted"


def parse_gpus(value: str) -> list[int]:
    if value == "auto":
        return list(range(torch.cuda.device_count()))
    gpus = [int(part) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("No GPU ids were provided.")
    return gpus


class TokenizedMoleculeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.original_length = len(self.dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_columns = self.dataframe.columns.tolist()[2:]
        self._tokenize_and_filter()

    def _tokenize_and_filter(self) -> None:
        kept_rows = []
        token_ids = []
        lengths = []
        unk_id = self.tokenizer.unk_token_id

        for _, row in tqdm(
            self.dataframe.iterrows(),
            total=len(self.dataframe),
            desc="Tokenizing/filtering Fig.2b molecules",
        ):
            selfies = row["SMILES"].replace("][", "] [")
            inputs = self.tokenizer(
                selfies,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            ids = inputs["input_ids"].squeeze(0).to(torch.long)
            if ids.numel() > self.max_length:
                continue
            if unk_id is not None and unk_id in ids.tolist():
                continue
            kept_rows.append(row)
            token_ids.append(ids)
            lengths.append(int(ids.numel()))

        self.dataframe = pd.DataFrame(kept_rows).reset_index(drop=True)
        self.token_ids = token_ids
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        target = row[self.target_columns].values.astype(np.float32)
        return {
            "input_ids": self.token_ids[idx],
            "length": self.lengths[idx],
            "label": target,
            "dbaasp_id": str(row["DBAASP_id"]),
        }


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


class MultiTaskLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, mask):
        loss = (y_pred - y_true) ** 2
        masked_loss = loss * mask
        if self.reduction == "mean":
            return masked_loss.sum() / (mask.sum() + 1e-8)
        if self.reduction == "sum":
            return masked_loss.sum()
        return masked_loss


def prepare_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = raw_labels.astype(np.float32).copy()
    mask = (labels >= -0.5).astype(np.float32)
    valid = mask.astype(bool)
    labels[valid] = -np.log10(labels[valid] / 10.0)
    return labels, mask


def calculate_r2_per_task(all_labels, all_preds, all_label_masks) -> list[float | None]:
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_label_masks = np.array(all_label_masks)
    r2_per_task = []
    for task_idx in range(all_labels.shape[1]):
        mask = all_label_masks[:, task_idx].astype(bool)
        y_true = all_labels[mask, task_idx]
        y_pred = all_preds[mask, task_idx]
        if len(y_true) == 0:
            r2_per_task.append(None)
            continue
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2_per_task.append(float(1 - (ss_residual / ss_total)))
    return r2_per_task


def finite_mean(values: Iterable[float | None]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_mdlm_config():
    if str(MDLM_DIR) not in sys.path:
        sys.path.insert(0, str(MDLM_DIR))
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(MDLM_DIR / "configs")):
        return compose(config_name="config", overrides=["model=medium"])


def load_dit_backbone(config, vocab_size: int, ckpt_path: Path):
    if str(MDLM_DIR) not in sys.path:
        sys.path.insert(0, str(MDLM_DIR))
    import models

    backbone = models.dit.DIT(config, vocab_size=vocab_size)
    lightning_ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = lightning_ckpt["state_dict"]

    new_sd = OrderedDict()
    for key, value in state_dict.items():
        new_sd[key[len("backbone.") :] if key.startswith("backbone.") else key] = value

    if "ema" in lightning_ckpt and "shadow_params" in lightning_ckpt["ema"]:
        ema_sd = OrderedDict()
        pass_flag = False
        shadow_params = lightning_ckpt["ema"]["shadow_params"]
        for idx, (key, value) in enumerate(new_sd.items()):
            if key != "rotary_emb.inv_freq":
                ema_sd[key] = shadow_params[idx if not pass_flag else idx - 1]
            else:
                ema_sd[key] = value
                pass_flag = True
        backbone.load_state_dict(ema_sd, strict=False)
    else:
        backbone.load_state_dict(new_sd, strict=False)

    return backbone


class DLMFirstTokenExtractor(nn.Module):
    def __init__(self, config, vocab_size: int, ckpt_path: Path):
        super().__init__()
        if str(MDLM_DIR) not in sys.path:
            sys.path.insert(0, str(MDLM_DIR))
        import noise_schedule

        self.config = config
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.backbone = load_dit_backbone(config, vocab_size, ckpt_path)
        self.noise = noise_schedule.get_noise(config)

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n: int, device):
        sampling_eps = 1e-3
        eps_t = torch.rand(n, device=device) * 0
        return (1 - sampling_eps) * eps_t + sampling_eps

    def forward(self, input_ids):
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, _ = self.noise(t)
        sigma = self._process_sigma(sigma[:, None])
        x = self.backbone.vocab_embed(input_ids)
        c = F.silu(self.backbone.sigma_map(sigma))
        rotary_cos_sin = self.backbone.rotary_emb(x)
        for block in self.backbone.blocks:
            x = block(x, rotary_cos_sin, c, seqlens=None)
        return x[:, 0, :]


def make_batches(items, batch_size: int, pad_to: str, pad_token_id: int):
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        positions = [item["position"] for item in batch]
        ids = [item["input_ids"] for item in batch]
        if pad_to == "fixed1024":
            padded = torch.full((len(ids), 1024), pad_token_id, dtype=torch.long)
            for row_idx, token_ids in enumerate(ids):
                padded[row_idx, : token_ids.numel()] = token_ids[:1024]
        else:
            padded = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        yield positions, padded


def extraction_worker(
    gpu_id: int,
    worker_items: list[dict],
    args_dict: dict,
    shard_path: str,
) -> str:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    config = load_mdlm_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DLMFirstTokenExtractor(config, len(tokenizer.get_vocab()), Path(args_dict["dlm_ckpt"]))
    model.to(device)
    if args_dict["backbone_mode"] == "train":
        model.train()
    else:
        model.eval()

    worker_items = sorted(worker_items, key=lambda item: item["length"])
    all_positions = []
    all_features = []
    with torch.inference_mode():
        for positions, input_ids in tqdm(
            make_batches(
                worker_items,
                args_dict["extract_batch_size"],
                args_dict["pad_to"],
                tokenizer.pad_token_id,
            ),
            total=math.ceil(len(worker_items) / args_dict["extract_batch_size"]),
            desc=f"GPU {gpu_id} embedding",
            position=gpu_id,
        ):
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                features = model(input_ids)
            all_positions.extend(positions)
            all_features.append(features.detach().float().cpu())

    torch.save(
        {
            "positions": torch.tensor(all_positions, dtype=torch.long),
            "features": torch.cat(all_features, dim=0),
            "gpu": gpu_id,
        },
        shard_path,
    )
    return shard_path


def build_or_load_cache(args) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / args.cache_name
    manifest_path = output_dir / "embedding_cache_manifest.json"
    if cache_path.exists() and not args.force_cache:
        print(f"Using existing embedding cache: {cache_path}")
        return cache_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data = pd.read_csv(args.data_path)
    if args.limit_rows is not None:
        data = data.head(args.limit_rows).copy()
    dataset = TokenizedMoleculeDataset(data, tokenizer, max_length=args.max_length)

    gpus = parse_gpus(args.gpus)
    if len(gpus) == 0:
        raise RuntimeError("No CUDA GPUs are available for DLM extraction.")

    order = sorted(range(len(dataset)), key=lambda idx: dataset.lengths[idx], reverse=True)
    shards = [[] for _ in gpus]
    for rank, idx in enumerate(order):
        item = dataset[idx]
        item["position"] = idx
        shards[rank % len(gpus)].append(item)

    shard_dir = output_dir / "embedding_shards"
    shard_dir.mkdir(exist_ok=True)
    args_dict = {
        "dlm_ckpt": str(args.dlm_ckpt),
        "extract_batch_size": args.extract_batch_size,
        "pad_to": args.pad_to,
        "backbone_mode": args.backbone_mode,
    }

    ctx = mp.get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx) as executor:
        for gpu_id, shard_items in zip(gpus, shards):
            shard_path = shard_dir / f"features_gpu{gpu_id}.pt"
            futures.append(
                executor.submit(extraction_worker, gpu_id, shard_items, args_dict, str(shard_path))
            )
        shard_paths = [future.result() for future in as_completed(futures)]

    features = None
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        if features is None:
            features = torch.empty(len(dataset), shard["features"].shape[1], dtype=torch.float32)
        features[shard["positions"]] = shard["features"]

    raw_labels = np.stack([dataset[idx]["label"] for idx in range(len(dataset))], axis=0)
    labels, masks = prepare_labels(raw_labels)
    dbaasp_ids = [dataset[idx]["dbaasp_id"] for idx in range(len(dataset))]
    lengths = [dataset[idx]["length"] for idx in range(len(dataset))]

    torch.save(
        {
            "features": features,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "label_masks": torch.tensor(masks, dtype=torch.float32),
            "dbaasp_ids": dbaasp_ids,
            "lengths": lengths,
            "target_columns": dataset.target_columns,
        },
        cache_path,
    )
    manifest = {
        "cache_path": str(cache_path),
        "source_script": str(Path(__file__).resolve()),
        "reference_script": str(MDLM_DIR / "DBAASP_MLM_MDLM.py"),
        "data_path": str(args.data_path),
        "dlm_checkpoint": str(args.dlm_ckpt),
        "model_name": MODEL_NAME,
        "model_config_override": "model=medium",
        "pooling": "first_token",
        "pad_to": args.pad_to,
        "backbone_mode": args.backbone_mode,
        "max_length": args.max_length,
        "original_rows": int(dataset.original_length),
        "filtered_rows": int(len(dataset)),
        "feature_dim": int(features.shape[1]),
        "gpus": gpus,
        "extract_batch_size": args.extract_batch_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved embedding cache: {cache_path}")
    print(f"Saved embedding manifest: {manifest_path}")
    return cache_path


def train_fold_worker(task: dict) -> dict:
    gpu_id = task["gpu_id"]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    cache = torch.load(task["cache_path"], map_location="cpu")
    features = cache["features"]
    labels = cache["labels"]
    masks = cache["label_masks"]

    fold = task["fold"]
    train_idx = task["train_idx"]
    test_idx = task["test_idx"]
    seed = task["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(features, labels, masks)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=task["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=task["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    head = RegressionHead(features.shape[1], num_targets=labels.shape[1]).to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(head.parameters(), lr=task["learning_rate"])

    best_r2_mean = -float("inf")
    best_epoch = -1
    best_r2_per_task = None
    fold_dir = Path(task["output_dir"]) / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_head.pt"
    final_path = fold_dir / "final_head.pt"

    for epoch in range(task["num_epochs"]):
        head.train()
        train_losses = []
        for batch_features, batch_labels, batch_masks in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels, batch_masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        head.eval()
        all_labels = []
        all_preds = []
        all_masks = []
        with torch.no_grad():
            for batch_features, batch_labels, batch_masks in test_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                logits = head(batch_features)
                all_labels.extend(batch_labels.numpy())
                all_preds.extend(logits.detach().cpu().numpy())
                all_masks.extend(batch_masks.numpy())

        r2_per_task = calculate_r2_per_task(all_labels, all_preds, all_masks)
        r2_mean = finite_mean(r2_per_task)
        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_epoch = epoch
            best_r2_per_task = r2_per_task
            torch.save(
                {
                    "fold": fold + 1,
                    "epoch": epoch,
                    "best_r2_mean": best_r2_mean,
                    "r2_per_task": best_r2_per_task,
                    "head_state_dict": head.state_dict(),
                    "input_dim": int(features.shape[1]),
                    "num_targets": int(labels.shape[1]),
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "hyperparameters": {
                        "learning_rate": task["learning_rate"],
                        "batch_size": task["batch_size"],
                        "num_epochs": task["num_epochs"],
                        "dropout": 0.2,
                        "hidden_dim_1": 384,
                        "hidden_dim_2": 128,
                        "seed": seed,
                    },
                },
                best_path,
            )

        if (epoch + 1) % task["log_every"] == 0 or epoch == 0:
            print(
                f"fold {fold + 1} gpu {gpu_id} epoch {epoch + 1}/{task['num_epochs']} "
                f"loss={np.mean(train_losses):.6f} r2_mean={r2_mean:.6f} best={best_r2_mean:.6f}",
                flush=True,
            )

    torch.save(
        {
            "fold": fold + 1,
            "epoch": task["num_epochs"] - 1,
            "head_state_dict": head.state_dict(),
            "best_r2_mean": best_r2_mean,
            "best_epoch": best_epoch,
            "best_r2_per_task": best_r2_per_task,
            "input_dim": int(features.shape[1]),
            "num_targets": int(labels.shape[1]),
        },
        final_path,
    )
    return {
        "fold": fold + 1,
        "gpu": gpu_id,
        "best_epoch": best_epoch + 1,
        "best_r2_mean": best_r2_mean,
        "best_r2_per_task": best_r2_per_task,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }


def train_heads(args, cache_path: Path) -> dict:
    cache = torch.load(cache_path, map_location="cpu")
    n = cache["features"].shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gpus = parse_gpus(args.gpus)
    random_seeds = [42, 2024, 2025, 2077, 2012]
    tasks = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        tasks.append(
            {
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "gpu_id": gpus[fold % len(gpus)],
                "seed": random_seeds[fold],
                "cache_path": str(cache_path),
                "output_dir": str(args.output_dir),
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "log_every": args.log_every,
            }
        )

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=min(len(gpus), len(tasks)), mp_context=ctx) as executor:
        futures = [executor.submit(train_fold_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda row: row["fold"])
    metrics = {
        "cache_path": str(cache_path),
        "num_folds": 5,
        "best_mean_R2_across_folds": float(np.mean([row["best_r2_mean"] for row in results])),
        "folds": results,
        "hyperparameters": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kfold": {"n_splits": 5, "shuffle": True, "random_state": 42},
        },
        "target_columns": cache["target_columns"],
    }
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics: {metrics_path}")
    print(f"best_mean_R2_across_folds={metrics['best_mean_R2_across_folds']:.6f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "cache", "train"], default="all")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dlm-ckpt", type=Path, default=DEFAULT_DLM_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-name", default="fig2b_mdlm_best_first_token_embeddings.pt")
    parser.add_argument("--gpus", default="auto", help="'auto' or comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--extract-batch-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pad-to", choices=["batch_max", "fixed1024"], default="batch_max")
    parser.add_argument("--backbone-mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug/smoke-test only.")
    return parser.parse_args()


def main():
    args = parse_args()
    if str(MDLM_DIR) not in sys.path:
        sys.path.insert(0, str(MDLM_DIR))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.output_dir) / args.cache_name
    if args.mode in {"all", "cache"}:
        cache_path = build_or_load_cache(args)
    if args.mode in {"all", "train"}:
        if not cache_path.exists():
            raise FileNotFoundError(f"Embedding cache does not exist: {cache_path}")
        train_heads(args, cache_path)


if __name__ == "__main__":
    main()
