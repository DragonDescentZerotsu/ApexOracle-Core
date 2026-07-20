#!/usr/bin/env python3
"""Generate Fig. 2b baseline resources for capsule eval.

This mirrors the fixed-backbone Fig. 2b baseline scripts with the same 5-fold
split and downstream-head hyperparameters, while caching frozen eval-mode
features so the capsule can later do inference-only metric recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.apex_adapter import (
    build_apex_vocabulary,
    legacy_onehot_encoding,
)
from apexoracle.benchmarks.molecule_encoders.apex_model import (
    ApexEncoder,
    load_aaindex_embedding,
)
from apexoracle.benchmarks.molecule_encoders.assets import apex_aaindex_path
from apexoracle.benchmarks.molecule_encoders.legacy_training import (
    LegacyMaskedMSELoss,
    finite_mean_or_nan,
    legacy_r2_per_task,
)
from apexoracle.resources import resolve_weight
from apexoracle.vendor.peptideclm_tokenizer import load_tokenizer as load_peptideclm_tokenizer

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Checkpoints" / "fig2b_baselines_cached_5fold"
SMILES_DATA = REPO_ROOT / "DataPrepare" / "Data" / "DBAASP_id_SMILES_bact_MICs.csv"
APEX_DATA = REPO_ROOT / "DataPrepare" / "Data" / "DBAASP_id_same_as_SMILES_AAseqs_bact_MICs_512_limit.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    source_script: str
    hf_model: str | None = None
    trust_remote_code: bool = False
    pooling: str = "cls"
    feature_dim: int | None = None
    hidden_dim_1: int = 384
    hidden_dim_2: int = 128
    data_path: Path = SMILES_DATA


MODEL_SPECS = {
    "chemberta_mtr": ModelSpec(
        name="chemberta_mtr",
        kind="hf_smiles",
        hf_model="DeepChem/ChemBERTa-77M-MTR",
        source_script="legacy-code-snapshot-2026-07-17:fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "chemberta_mlm": ModelSpec(
        name="chemberta_mlm",
        kind="hf_smiles",
        hf_model="DeepChem/ChemBERTa-77M-MLM",
        source_script="legacy-code-snapshot-2026-07-17:fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "molformer": ModelSpec(
        name="molformer",
        kind="hf_smiles",
        hf_model="ibm/MoLFormer-XL-both-10pct",
        trust_remote_code=True,
        source_script="legacy-code-snapshot-2026-07-17:fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "peptideclm": ModelSpec(
        name="peptideclm",
        kind="peptideclm",
        hf_model="aaronfeller/PeptideCLM-23M-all",
        source_script="legacy-code-snapshot-2026-07-17:fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "apex": ModelSpec(
        name="apex",
        kind="apex",
        feature_dim=128,
        hidden_dim_1=512,
        hidden_dim_2=256,
        data_path=APEX_DATA,
        source_script="legacy-code-snapshot-2026-07-17:compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py",
    ),
}


class TokenizedSmilesDataset(Dataset):
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
        attention_masks = []
        for _, row in tqdm(
            self.dataframe.iterrows(),
            total=len(self.dataframe),
            desc="Tokenizing SMILES",
            leave=False,
        ):
            inputs = self.tokenizer(str(row["SMILES"]), return_tensors="pt", padding=False, truncation=False)
            ids = inputs["input_ids"].squeeze(0).to(torch.long)
            if ids.numel() > self.max_length:
                continue
            kept_rows.append(row)
            token_ids.append(ids)
            attention_masks.append(inputs["attention_mask"].squeeze(0).to(torch.long))
        self.dataframe = pd.DataFrame(kept_rows).reset_index(drop=True)
        self.token_ids = token_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        return {
            "input_ids": self.token_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": row[self.target_columns].values.astype(np.float32),
            "dbaasp_id": str(row["DBAASP_id"]),
        }


class ApexDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_length: int, word2idx: dict[str, int], onehot_encoding):
        self.dataframe = dataframe.copy()
        self.original_length = len(self.dataframe)
        self.max_length = max_length
        self.target_columns = self.dataframe.columns.tolist()[2:]
        self.dataframe = self.dataframe[self.dataframe["AAseqs"].apply(lambda value: len(str(value)) <= max_length)]
        self.dataframe = self.dataframe.reset_index(drop=True)
        self.seqs = onehot_encoding(self.dataframe["AAseqs"].tolist(), max_length, word2idx)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        return {
            "input_ids": torch.from_numpy(self.seqs[idx]).long(),
            "label": row[self.target_columns].values.astype(np.float32),
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


def collate_tokens(batch, pad_token_id: int):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor(np.stack([item["label"] for item in batch]), dtype=torch.float32)
    dbaasp_ids = [item["dbaasp_id"] for item in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "dbaasp_ids": dbaasp_ids}


def collate_apex(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.tensor(np.stack([item["label"] for item in batch]), dtype=torch.float32)
    dbaasp_ids = [item["dbaasp_id"] for item in batch]
    return {"input_ids": input_ids, "labels": labels, "dbaasp_ids": dbaasp_ids}


def prepare_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = raw_labels.astype(np.float32).copy()
    mask = (labels >= -0.5).astype(np.float32)
    valid = mask.astype(bool)
    labels[valid] = -np.log10(labels[valid] / 10.0)
    return labels, mask


def get_peptideclm_tokenizer():
    return load_peptideclm_tokenizer()


def get_pad_token_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "pad_token", 0)
    if isinstance(value, str):
        return int(value) if value.isdigit() else 0
    return int(value)


def extract_hf_features(spec: ModelSpec, device: torch.device, batch_size: int, limit_rows: int | None):
    tokenizer = get_peptideclm_tokenizer() if spec.kind == "peptideclm" else AutoTokenizer.from_pretrained(
        spec.hf_model,
        trust_remote_code=spec.trust_remote_code,
    )
    data = pd.read_csv(spec.data_path)
    if limit_rows is not None:
        data = data.head(limit_rows)
    dataset = TokenizedSmilesDataset(data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_tokens(batch, get_pad_token_id(tokenizer)),
    )
    model = AutoModel.from_pretrained(spec.hf_model, trust_remote_code=spec.trust_remote_code)
    model.to(device)
    model.eval()

    all_features = []
    raw_labels = []
    dbaasp_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{spec.name} feature extraction"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if spec.pooling == "mean":
                expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                summed = torch.sum(outputs.last_hidden_state * expanded, dim=1)
                denom = torch.clamp(expanded.sum(dim=1), min=1e-9)
                features = summed / denom
            else:
                features = outputs.last_hidden_state[:, 0, :]
            all_features.append(features.detach().cpu().float())
            raw_labels.append(batch["labels"].numpy())
            dbaasp_ids.extend(batch["dbaasp_ids"])

    labels, masks = prepare_labels(np.concatenate(raw_labels, axis=0))
    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "label_masks": torch.tensor(masks, dtype=torch.float32),
        "target_columns": list(dataset.target_columns),
        "dbaasp_ids": dbaasp_ids,
        "original_length": int(dataset.original_length),
        "filtered_length": int(len(dataset)),
    }


def extract_apex_features(spec: ModelSpec, device: torch.device, batch_size: int, limit_rows: int | None):
    word2idx, _ = build_apex_vocabulary()
    emb, _ = load_aaindex_embedding(apex_aaindex_path(REPO_ROOT), word2idx)
    data = pd.read_csv(spec.data_path)
    if limit_rows is not None:
        data = data.head(limit_rows)
    dataset = ApexDataset(
        data,
        max_length=52,
        word2idx=word2idx,
        onehot_encoding=legacy_onehot_encoding,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_apex)
    model = ApexEncoder(
        emb, np.shape(emb)[1], num_rnn_layers=3, hidden_dim=128
    )
    checkpoint = torch.load(
        resolve_weight("fig2b_apex_encoder", repo_root=REPO_ROOT),
        map_location="cpu",
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    all_features = []
    raw_labels = []
    dbaasp_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{spec.name} feature extraction"):
            input_ids = batch["input_ids"].to(device)
            features = model(input_ids)
            all_features.append(features.detach().cpu().float())
            raw_labels.append(batch["labels"].numpy())
            dbaasp_ids.extend(batch["dbaasp_ids"])

    labels, masks = prepare_labels(np.concatenate(raw_labels, axis=0))
    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "label_masks": torch.tensor(masks, dtype=torch.float32),
        "target_columns": list(dataset.target_columns),
        "dbaasp_ids": dbaasp_ids,
        "original_length": int(dataset.original_length),
        "filtered_length": int(len(dataset)),
    }


def load_or_extract_features(spec: ModelSpec, output_dir: Path, device: torch.device, args: argparse.Namespace):
    feature_path = output_dir / "features.pt"
    if feature_path.exists() and not args.force_cache:
        return torch.load(feature_path, map_location="cpu", weights_only=False)
    if spec.kind == "apex":
        cache = extract_apex_features(spec, device, args.extract_batch_size, args.limit_rows)
    else:
        cache = extract_hf_features(spec, device, args.extract_batch_size, args.limit_rows)
    cache.update(
        {
            "model": spec.name,
            "source_script": spec.source_script,
            "pooling": spec.pooling,
            "feature_dim": int(cache["features"].shape[1]),
            "hyperparameters": {
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "kfold_splits": 5,
                "kfold_shuffle": True,
                "kfold_random_state": 42,
                "hidden_dim_1": spec.hidden_dim_1,
                "hidden_dim_2": spec.hidden_dim_2,
                "dropout": 0.2,
            },
        }
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, feature_path)
    return cache


def evaluate_head(head, features, labels, masks, device: torch.device, batch_size: int):
    head.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device)
            preds.append(head(batch).detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    r2_per_task = legacy_r2_per_task(labels.numpy(), preds, masks.numpy())
    return r2_per_task, finite_mean_or_nan(r2_per_task)


def train_fold(spec: ModelSpec, cache: dict, fold: int, train_idx, test_idx, output_dir: Path, device: torch.device, args):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    features = cache["features"].float()
    labels = cache["labels"].float()
    masks = cache["label_masks"].float()
    train_dataset = TensorDataset(features[train_idx], labels[train_idx], masks[train_idx])
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    head = RegressionHead(
        input_dim=int(features.shape[1]),
        hidden_dim_1=spec.hidden_dim_1,
        hidden_dim_2=spec.hidden_dim_2,
        num_targets=int(labels.shape[1]),
        pooler_dropout=0.2,
    ).to(device)
    criterion = LegacyMaskedMSELoss()
    optimizer = optim.Adam(head.parameters(), lr=args.learning_rate)
    test_features = features[test_idx]
    test_labels = labels[test_idx]
    test_masks = masks[test_idx]
    best_r2 = -float("inf")
    best_info = None
    fold_dir = output_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        head.train()
        last_loss = None
        for batch_features, batch_labels, batch_masks in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad()
            logits = head(batch_features)
            loss = criterion(logits, batch_labels, batch_masks)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        r2_per_task, r2_mean = evaluate_head(head, test_features, test_labels, test_masks, device, args.batch_size)
        if r2_mean > best_r2:
            best_r2 = r2_mean
            best_info = {
                "epoch": epoch,
                "best_r2_mean": best_r2,
                "r2_per_task": r2_per_task,
                "loss": last_loss,
            }
            torch.save(
                {
                    "fold": fold + 1,
                    "epoch": epoch,
                    "best_r2_mean": best_r2,
                    "r2_per_task": r2_per_task,
                    "head_state_dict": head.state_dict(),
                    "input_dim": int(features.shape[1]),
                    "num_targets": int(labels.shape[1]),
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "hyperparameters": {
                        "hidden_dim_1": spec.hidden_dim_1,
                        "hidden_dim_2": spec.hidden_dim_2,
                        "dropout": 0.2,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "num_epochs": args.num_epochs,
                        "seed": args.seed,
                    },
                },
                fold_dir / "best_head.pt",
            )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"[{spec.name}] fold={fold + 1} epoch={epoch + 1}/{args.num_epochs} r2={r2_mean:.6f} best={best_r2:.6f}", flush=True)

    torch.save(
        {
            "fold": fold + 1,
            "epoch": args.num_epochs - 1,
            "head_state_dict": head.state_dict(),
            "input_dim": int(features.shape[1]),
            "num_targets": int(labels.shape[1]),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "hyperparameters": {
                "hidden_dim_1": spec.hidden_dim_1,
                "hidden_dim_2": spec.hidden_dim_2,
                "dropout": 0.2,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "seed": args.seed,
            },
        },
        fold_dir / "final_head.pt",
    )
    return {
        "fold": fold + 1,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        **(best_info or {}),
        "best_checkpoint": str(fold_dir / "best_head.pt"),
        "final_checkpoint": str(fold_dir / "final_head.pt"),
    }


def run_model(task: dict) -> dict:
    spec = MODEL_SPECS[task["model_name"]]
    args = argparse.Namespace(**task["args"])
    gpu = int(task["gpu"])
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.name}] starting on {device}", flush=True)

    cache = load_or_extract_features(spec, output_dir, device, args)
    features = cache["features"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(features.shape[0]))):
        fold_results.append(train_fold(spec, cache, fold, train_idx, test_idx, output_dir, device, args))

    metrics = {
        "model": spec.name,
        "source_script": spec.source_script,
        "feature_cache": str(output_dir / "features.pt"),
        "num_examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "best_mean_R2_across_folds": float(np.mean([item["best_r2_mean"] for item in fold_results])),
        "folds": fold_results,
        "target_columns": list(cache["target_columns"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[{spec.name}] completed mean_r2={metrics['best_mean_R2_across_folds']:.6f}", flush=True)
    return metrics


def parse_models(value: list[str]) -> list[str]:
    if value == ["all"]:
        return list(MODEL_SPECS)
    models = []
    for name in value:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model {name}. Choices: {sorted(MODEL_SPECS)}")
        models.append(name)
    return models


def parse_gpus(value: str) -> list[int]:
    if value == "auto":
        return list(range(torch.cuda.device_count()))
    gpus = [int(part) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("No GPU ids provided")
    return gpus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract-batch-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = parse_models(args.models)
    gpus = parse_gpus(args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_args = vars(args).copy()
    task_args["output_dir"] = str(args.output_dir)
    tasks = [{"model_name": model_name, "gpu": gpus[idx % len(gpus)], "args": task_args} for idx, model_name in enumerate(model_names)]
    all_metrics = {}
    with ProcessPoolExecutor(max_workers=min(len(gpus), len(tasks))) as executor:
        futures = [executor.submit(run_model, task) for task in tasks]
        for future in as_completed(futures):
            metrics = future.result()
            all_metrics[metrics["model"]] = metrics["best_mean_R2_across_folds"]
    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps({"best_mean_R2_across_folds": all_metrics}, indent=2) + "\n")
    print(json.dumps({"best_mean_R2_across_folds": all_metrics}, indent=2))


if __name__ == "__main__":
    main()
