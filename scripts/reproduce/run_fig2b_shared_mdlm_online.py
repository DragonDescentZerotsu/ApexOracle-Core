#!/usr/bin/env python3
"""Run one paper-compatible shared-data Fig. 2b DLM fold.

This keeps the online frozen-backbone behavior of
the external ``mdlm/DBAASP_MLM_MDLM.py`` implementation. The only scientific
protocol changes are the reviewer-requested common molecule IDs and common
fold assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hydra import compose, initialize_config_dir
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer


SYNERGY_DIR = Path(__file__).resolve().parents[2]
MDLM_DIR = Path(
    os.environ.get("APEXORACLE_MDLM_DIR", str(SYNERGY_DIR.parent / "mdlm"))
).resolve()
SHARED_DIR = SYNERGY_DIR / "DataPrepare" / "Data" / "fig2b_shared_v1"
SELFIES_SOURCE = SYNERGY_DIR / "DataPrepare" / "Data" / "DBAASP_id_SELFIES_bact_MICs.csv"
TOKENIZER_NAME = "ibm-research/materials.selfies-ted"
MODEL_SPECS = {
    "dlm_only": {
        "config": "small",
        "checkpoint": MDLM_DIR / "Checkpoints_fangping" / "best_2.ckpt",
        "paper_label": "DLM MLM",
    },
    "dlm_mtr_dlm": {
        "config": "medium",
        "checkpoint": MDLM_DIR / "Checkpoints_fangping" / "best.ckpt",
        "paper_label": "DLM MTR+DLM",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(model_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(MDLM_DIR / "configs")):
        return compose(config_name="config", overrides=[f"model={model_name}"])


def load_backbone(config, vocab_size: int, checkpoint_path: Path):
    if str(MDLM_DIR) not in sys.path:
        sys.path.insert(0, str(MDLM_DIR))
    import models

    backbone = models.dit.DIT(config, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    stripped = OrderedDict()
    for key, value in state_dict.items():
        stripped[key[len("backbone.") :] if key.startswith("backbone.") else key] = value

    if "ema" in checkpoint and "shadow_params" in checkpoint["ema"]:
        ema_state = OrderedDict()
        passed_rotary = False
        shadow = checkpoint["ema"]["shadow_params"]
        for index, (key, value) in enumerate(stripped.items()):
            if key == "rotary_emb.inv_freq":
                ema_state[key] = value
                passed_rotary = True
            else:
                ema_state[key] = shadow[index if not passed_rotary else index - 1]
        incompatible = backbone.load_state_dict(ema_state, strict=False)
        source = "ema.shadow_params"
    else:
        incompatible = backbone.load_state_dict(stripped, strict=False)
        source = "state_dict"
    return backbone, source, list(incompatible.missing_keys), list(incompatible.unexpected_keys)


class SharedSelfiesDataset(Dataset):
    def __init__(self, shared_dir: Path, selfies_source: Path, tokenizer, max_length: int = 512):
        shared = pd.read_csv(shared_dir / "shared_molecules.csv", dtype={"dbaasp_id": "string"})
        folds = pd.read_csv(shared_dir / "folds.csv", dtype={"dbaasp_id": "string"})
        selfies = pd.read_csv(selfies_source, dtype={"DBAASP_id": "string"})
        for frame, column in ((shared, "dbaasp_id"), (folds, "dbaasp_id"), (selfies, "DBAASP_id")):
            frame[column] = frame[column].str.strip()
        if shared["dbaasp_id"].duplicated().any() or folds["dbaasp_id"].duplicated().any():
            raise ValueError("shared IDs must be unique")

        selfies_by_id = selfies.set_index("DBAASP_id")["SMILES"]
        fold_by_id = folds.set_index("dbaasp_id")["fold"]
        if not set(shared["dbaasp_id"]).issubset(selfies_by_id.index):
            raise ValueError("SELFIES source is missing shared IDs")
        if set(shared["dbaasp_id"]) != set(fold_by_id.index):
            raise ValueError("shared molecule and fold ID sets differ")

        self.ids = shared["dbaasp_id"].astype(str).tolist()
        self.folds = shared["dbaasp_id"].map(fold_by_id).to_numpy(dtype=np.int64)
        self.target_columns = [column for column in shared.columns if column not in {"dbaasp_id", "smiles", "apex_sequence"}]
        self.raw_labels = shared[self.target_columns].to_numpy(dtype=np.float32)
        self.token_ids: list[torch.Tensor] = []
        unk_id = tokenizer.unk_token_id
        for molecule_id in self.ids:
            text = str(selfies_by_id.loc[molecule_id]).replace("][", "] [")
            ids = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"].squeeze(0).to(torch.long)
            if ids.numel() > max_length or (unk_id is not None and unk_id in ids.tolist()):
                raise ValueError(f"shared intersection contains DLM-ineligible ID {molecule_id}")
            self.token_ids.append(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        return {
            "input_ids": self.token_ids[index],
            "label": torch.tensor(self.raw_labels[index], dtype=torch.float32),
            "dbaasp_id": self.ids[index],
        }


def make_collate(pad_token_id: int):
    def collate(batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = input_ids.ne(pad_token_id).to(torch.long)
        labels = torch.stack([item["label"] for item in batch])
        mask = labels >= -0.5
        processed = labels.clone()
        processed[mask] = -torch.log10(labels[mask] / 10)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": processed,
            "label_mask": mask.to(torch.int32),
            "dbaasp_ids": [item["dbaasp_id"] for item in batch],
        }

    return collate


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, num_targets: int = 19):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, 384)
        self.dense_2 = nn.Linear(384, 128)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.out_proj = nn.Linear(128, num_targets)

    def forward(self, features):
        features = self.dropout(self.activation_fn(self.dense_1(features)))
        features = self.dropout(self.activation_fn(self.dense_2(features)))
        return self.out_proj(features)


class DLMRegressionModel(nn.Module):
    def __init__(self, config, vocab_size: int, checkpoint_path: Path, num_targets: int):
        super().__init__()
        if str(MDLM_DIR) not in sys.path:
            sys.path.insert(0, str(MDLM_DIR))
        import noise_schedule

        self.config = config
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.backbone, self.weight_source, self.missing_keys, self.unexpected_keys = load_backbone(
            config,
            vocab_size,
            checkpoint_path,
        )
        self.classifier = RegressionHead(int(config.model.hidden_size), num_targets=num_targets)
        self.noise = noise_schedule.get_noise(config)

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def encode(self, input_ids, attention_mask):
        del attention_mask  # The published DLM downstream script does not use it.
        sampling_eps = 1e-3
        t = torch.rand(input_ids.shape[0], device=input_ids.device) * 0
        t = (1 - sampling_eps) * t + sampling_eps
        sigma, _ = self.noise(t)
        sigma = self._process_sigma(sigma[:, None])
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden = self.backbone.vocab_embed(input_ids)
            conditioning = F.silu(self.backbone.sigma_map(sigma))
            rotary = self.backbone.rotary_emb(hidden)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for block in self.backbone.blocks:
                    hidden = block(hidden, rotary, conditioning, seqlens=None)
        return hidden[:, 0, :]

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encode(input_ids, attention_mask))


class MultiTaskLoss(nn.Module):
    def forward(self, predictions, labels, mask):
        return (((predictions - labels) ** 2) * mask).sum() / (mask.sum() + 1e-8)


def r2_per_task(labels: np.ndarray, predictions: np.ndarray, masks: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for task in range(labels.shape[1]):
        valid = masks[:, task].astype(bool)
        y_true = labels[valid, task]
        y_pred = predictions[valid, task]
        if len(y_true) == 0:
            values.append(None)
            continue
        total = np.sum((y_true - np.mean(y_true)) ** 2)
        residual = np.sum((y_true - y_pred) ** 2)
        values.append(float(1 - residual / total))
    return values


def evaluate(model, loader, device):
    model.eval()
    labels, predictions, masks, ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            labels.extend(batch["label"].numpy())
            predictions.extend(logits.float().cpu().numpy())
            masks.extend(batch["label_mask"].numpy())
            ids.extend(batch["dbaasp_ids"])
    task_values = r2_per_task(np.asarray(labels), np.asarray(predictions), np.asarray(masks))
    mean_value = float(np.mean(task_values))
    return task_values, mean_value, ids, np.asarray(predictions, dtype=np.float32)


def cache_held_out_features(model, loader, device):
    """Cache the deterministic frozen-backbone eval output once per fold."""

    model.eval()
    features, labels, masks, ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            features.append(
                model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float().cpu()
            )
            labels.append(batch["label"].float().cpu())
            masks.append(batch["label_mask"].float().cpu())
            ids.extend(batch["dbaasp_ids"])
    return torch.cat(features), torch.cat(labels), torch.cat(masks), ids


def evaluate_cached_head(head, cache, device, batch_size: int):
    features, labels, masks, ids = cache
    head.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            predictions.append(head(features[start : start + batch_size].to(device)).float().cpu())
    prediction_array = torch.cat(predictions).numpy()
    task_values = r2_per_task(labels.numpy(), prediction_array, masks.numpy())
    return task_values, float(np.mean(task_values)), ids, prediction_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True, help="Zero-based fold index")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shared-dir", type=Path, default=SHARED_DIR)
    parser.add_argument("--selfies-source", type=Path, default=SELFIES_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=SYNERGY_DIR / "results" / "fig2b_shared_original_protocol")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    initial_torch_seed = int(torch.initial_seed())
    device = torch.device(args.device)
    spec = MODEL_SPECS[args.model]
    checkpoint_path = Path(spec["checkpoint"])
    config = load_config(str(spec["config"]))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = SharedSelfiesDataset(args.shared_dir, args.selfies_source, tokenizer)
    train_indices = np.flatnonzero(dataset.folds != args.fold)
    test_indices = np.flatnonzero(dataset.folds == args.fold)
    collate = make_collate(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = DLMRegressionModel(config, len(tokenizer.get_vocab()), checkpoint_path, len(dataset.target_columns))
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    held_out_cache = cache_held_out_features(model, test_loader, device)
    best_r2 = 0.0
    best_epoch = None
    best_task_r2 = None
    best_ids = None
    best_predictions = None
    final_task_r2 = None
    final_ids = None
    final_predictions = None
    started = time.time()

    fold_dir = args.output_dir / args.model / f"fold_{args.fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        last_loss = None
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device), batch["label_mask"].to(device))
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        task_values, mean_value, ids, predictions = evaluate_cached_head(
            model.classifier,
            held_out_cache,
            device,
            args.batch_size,
        )
        final_task_r2 = task_values
        final_ids = ids
        final_predictions = predictions
        if mean_value > best_r2:
            best_r2 = mean_value
            best_epoch = epoch + 1
            best_task_r2 = task_values
            best_ids = ids
            best_predictions = predictions
            torch.save(
                {
                    "head_state_dict": model.classifier.state_dict(),
                    "epoch": best_epoch,
                    "best_r2_mean": best_r2,
                    "r2_per_task": best_task_r2,
                },
                fold_dir / "best_head.pt",
            )
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"[{args.model}] fold={args.fold + 1} epoch={epoch + 1}/{args.epochs} "
                f"loss={last_loss:.6f} r2={mean_value:.6f} best={best_r2:.6f}",
                flush=True,
            )

    selected_positive_checkpoint = best_ids is not None and best_predictions is not None
    if not selected_positive_checkpoint:
        best_ids = final_ids
        best_predictions = final_predictions
        best_task_r2 = final_task_r2
    pd.DataFrame(
        {"dbaasp_id": best_ids, **{f"prediction_{i}": best_predictions[:, i] for i in range(best_predictions.shape[1])}}
    ).to_csv(fold_dir / "best_predictions.csv", index=False)
    metrics = {
        "model": args.model,
        "paper_label": spec["paper_label"],
        "fold": args.fold + 1,
        "best_epoch": best_epoch,
        "best_r2_mean": best_r2,
        "r2_per_task": best_task_r2,
        "selected_positive_checkpoint": selected_positive_checkpoint,
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
        "elapsed_seconds": time.time() - started,
        "initial_torch_seed": initial_torch_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_config": spec["config"],
        "weight_source": model.weight_source,
        "missing_keys": model.missing_keys,
        "unexpected_keys": model.unexpected_keys,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "head": [384, 128, 19],
            "head_dropout": 0.2,
            "backbone_frozen": True,
            "backbone_train_mode_during_training": True,
            "backbone_eval_mode_during_held_out_selection": True,
            "explicit_training_seed": args.seed,
        },
        "protocol": "fig2b-shared-native-intersection-v2",
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "fold": args.fold + 1, "best_r2_mean": best_r2}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
