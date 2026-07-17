#!/usr/bin/env python3
"""Faithful online-training reproduction for Fig. 2b frozen-backbone baselines.

Unlike the cached-feature resource builder, this script keeps the original
baseline training semantics: every training batch runs the frozen backbone in
``model.train()`` mode, and every validation pass runs it in ``model.eval()``
mode. Only the downstream regression head is optimized and saved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path("/data2/tianang/projects/Synergy")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apexoracle.benchmarks.molecule_encoders.protocol import DEFAULT_TARGET_COLUMNS


DEFAULT_OUTPUT_DIR = REPO_ROOT / "Checkpoints" / "fig2b_baselines_online_5fold"
DEFAULT_SHARED_DIR = REPO_ROOT / "DataPrepare" / "Data" / "fig2b_shared_v1"
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
    hidden_dim_1: int = 384
    hidden_dim_2: int = 128
    data_path: Path = SMILES_DATA


MODEL_SPECS = {
    "chemberta_mtr": ModelSpec(
        name="chemberta_mtr",
        kind="hf_smiles",
        hf_model="DeepChem/ChemBERTa-77M-MTR",
        source_script="fix_ChemBERTa_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "chemberta_mlm": ModelSpec(
        name="chemberta_mlm",
        kind="hf_smiles",
        hf_model="DeepChem/ChemBERTa-77M-MLM",
        source_script="fix_ChemBERTa_MLM_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "chemberta_mlm_mean": ModelSpec(
        name="chemberta_mlm_mean",
        kind="hf_smiles",
        hf_model="DeepChem/ChemBERTa-77M-MLM",
        pooling="mean",
        source_script="fix_ChemBERTa_MLM_mean_emb_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "molformer": ModelSpec(
        name="molformer",
        kind="hf_smiles",
        hf_model="ibm/MoLFormer-XL-both-10pct",
        trust_remote_code=True,
        source_script="fix_MolFormer_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "peptideclm": ModelSpec(
        name="peptideclm",
        kind="peptideclm",
        hf_model="aaronfeller/PeptideCLM-23M-all",
        source_script="fix_PeptideCLM_on_DBAASP_SMILES_5_fold_mean_MIC.py",
    ),
    "apex": ModelSpec(
        name="apex",
        kind="apex",
        hidden_dim_1=512,
        hidden_dim_2=256,
        data_path=APEX_DATA,
        source_script="compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py",
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
        target = row[self.target_columns].values.astype(np.float32)
        return {
            "input_ids": self.token_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": torch.tensor(target, dtype=torch.float32),
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
        target = row[self.target_columns].values.astype(np.float32)
        return {
            "input_ids": torch.from_numpy(self.seqs[idx]).long(),
            "label": torch.tensor(target, dtype=torch.float32),
            "dbaasp_id": str(row["DBAASP_id"]),
        }


def load_shared_frame_and_folds(shared_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the reviewer-requested common IDs and their frozen folds."""

    shared = pd.read_csv(shared_dir / "shared_molecules.csv", dtype={"dbaasp_id": "string"})
    folds = pd.read_csv(shared_dir / "folds.csv", dtype={"dbaasp_id": "string"})
    shared["dbaasp_id"] = shared["dbaasp_id"].str.strip()
    folds["dbaasp_id"] = folds["dbaasp_id"].str.strip()
    if shared["dbaasp_id"].duplicated().any() or folds["dbaasp_id"].duplicated().any():
        raise ValueError("shared benchmark IDs must be unique")
    fold_by_id = folds.set_index("dbaasp_id")["fold"]
    if set(shared["dbaasp_id"]) != set(fold_by_id.index):
        raise ValueError("shared_molecules.csv and folds.csv contain different ID sets")
    fold_ids = shared["dbaasp_id"].map(fold_by_id).to_numpy(dtype=np.int64)
    if set(fold_ids.tolist()) != set(range(5)):
        raise ValueError("shared benchmark must contain folds 0 through 4")
    return shared, fold_ids


def explicit_fold_indices(fold_ids: np.ndarray):
    for fold in range(5):
        test_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"fold {fold} is empty")
        yield train_idx, test_idx


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


class HFRegressionModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.bert = AutoModel.from_pretrained(spec.hf_model, trust_remote_code=spec.trust_remote_code)
        self.classifier = RegressionHead(
            input_dim=self.bert.config.hidden_size,
            hidden_dim_1=spec.hidden_dim_1,
            hidden_dim_2=spec.hidden_dim_2,
            num_targets=19,
            pooler_dropout=0.2,
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.spec.pooling == "mean":
            expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * expanded, dim=1)
            features = summed / torch.clamp(expanded.sum(dim=1), min=1e-9)
        else:
            features = outputs.last_hidden_state[:, 0, :]
        return features

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encode(input_ids, attention_mask))


class MultiTaskLoss(nn.Module):
    def forward(self, y_pred, y_true, mask):
        loss = (y_pred - y_true) ** 2
        return (loss * mask).sum() / (mask.sum() + 1e-8)


def get_peptideclm_tokenizer():
    from PeptideCLM.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

    pepclm_path = REPO_ROOT / "PeptideCLM"
    return SMILES_SPE_Tokenizer(pepclm_path / "tokenizer" / "new_vocab.txt", pepclm_path / "tokenizer" / "new_splits.txt")


def get_pad_token_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "pad_token", 0)
    if isinstance(value, str):
        return int(value) if value.isdigit() else 0
    return int(value)


def collate_hf(batch, pad_token_id: int):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    mask = labels >= -0.5
    labels_processed = labels.clone()
    labels_processed[mask] = -torch.log10(labels[mask] / 10)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels_processed,
        "label_mask": mask.int(),
        "dbaasp_ids": [item["dbaasp_id"] for item in batch],
    }


def collate_apex(batch):
    labels = torch.stack([item["label"] for item in batch], dim=0)
    mask = labels >= -0.5
    labels_processed = labels.clone()
    labels_processed[mask] = -torch.log10(labels[mask] / 10)
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "label": labels_processed,
        "label_mask": mask.int(),
        "dbaasp_ids": [item["dbaasp_id"] for item in batch],
    }


def calculate_r2_per_task(labels, preds, masks) -> list[float | None]:
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


def finite_mean(values: Iterable[float | None]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def build_hf_dataset(spec: ModelSpec, limit_rows: int | None, shared_dir: Path | None = None):
    tokenizer = get_peptideclm_tokenizer() if spec.kind == "peptideclm" else AutoTokenizer.from_pretrained(
        spec.hf_model,
        trust_remote_code=spec.trust_remote_code,
    )
    if shared_dir is None:
        data = pd.read_csv(spec.data_path)
    else:
        shared, _ = load_shared_frame_and_folds(shared_dir)
        data = shared.rename(columns={"dbaasp_id": "DBAASP_id", "smiles": "SMILES"})
        data = data[["DBAASP_id", "SMILES", *DEFAULT_TARGET_COLUMNS]]
    if limit_rows is not None:
        data = data.head(limit_rows)
    dataset = TokenizedSmilesDataset(data, tokenizer)
    return dataset, tokenizer


def build_apex_components(
    spec: ModelSpec,
    device: torch.device,
    limit_rows: int | None,
    shared_dir: Path | None = None,
):
    sys.path.insert(0, str(REPO_ROOT / "compare_APEX"))
    from APEX_models import AMP_model_fix
    from utils import AAindex, make_vocab, onehot_encoding

    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(REPO_ROOT / "compare_APEX" / "aaindex1.csv"), word2idx)
    if shared_dir is None:
        data = pd.read_csv(spec.data_path)
    else:
        shared, _ = load_shared_frame_and_folds(shared_dir)
        data = shared.rename(columns={"dbaasp_id": "DBAASP_id", "apex_sequence": "AAseqs"})
        data = data[["DBAASP_id", "AAseqs", *DEFAULT_TARGET_COLUMNS]]
    if limit_rows is not None:
        data = data.head(limit_rows)
    dataset = ApexDataset(data, max_length=52, word2idx=word2idx, onehot_encoding=onehot_encoding)
    backbone = AMP_model_fix(emb, np.shape(emb)[1], num_rnn_layers=3, dim_h=128)
    checkpoint = torch.load(REPO_ROOT / "compare_APEX" / "APEX_ckpt" / "APEX_pretrained_encoder_state_dict_best.ckpt", map_location="cpu")
    backbone.load_state_dict(checkpoint)
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    head = RegressionHead(128, spec.hidden_dim_1, spec.hidden_dim_2, 19).to(device)
    return dataset, backbone, head


def evaluate_hf(model: HFRegressionModel, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels, all_preds, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_labels.extend(batch["label"].numpy())
            all_preds.extend(logits.detach().cpu().numpy())
            all_masks.extend(batch["label_mask"].numpy())
    r2_per_task = calculate_r2_per_task(all_labels, all_preds, all_masks)
    return r2_per_task, finite_mean(r2_per_task)


def evaluate_apex(backbone, head, loader: DataLoader, device: torch.device, *, original_head_mode: bool):
    backbone.eval()
    # The published APEX driver never calls cls_head.eval(); its dropout remains
    # active during held-out-fold checkpoint selection. Preserve that behavior
    # for the paper-compatible shared-data run.
    if original_head_mode:
        head.train()
    else:
        head.eval()
    all_labels, all_preds, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            features = backbone(batch["input_ids"].to(device))
            logits = head(features)
            all_labels.extend(batch["label"].numpy())
            all_preds.extend(logits.detach().cpu().numpy())
            all_masks.extend(batch["label_mask"].numpy())
    r2_per_task = calculate_r2_per_task(all_labels, all_preds, all_masks)
    return r2_per_task, finite_mean(r2_per_task)


def cache_hf_held_out_features(model: HFRegressionModel, loader: DataLoader, device: torch.device):
    """Cache the deterministic frozen-backbone eval pass once per fold."""

    model.eval()
    features, labels, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            features.append(encoded.float().cpu())
            labels.append(batch["label"].float().cpu())
            masks.append(batch["label_mask"].float().cpu())
    return torch.cat(features), torch.cat(labels), torch.cat(masks)


def cache_apex_held_out_features(backbone, loader: DataLoader, device: torch.device):
    backbone.eval()
    features, labels, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            features.append(backbone(batch["input_ids"].to(device)).float().cpu())
            labels.append(batch["label"].float().cpu())
            masks.append(batch["label_mask"].float().cpu())
    return torch.cat(features), torch.cat(labels), torch.cat(masks)


def evaluate_cached_head(head, cache, device: torch.device, batch_size: int, *, train_mode: bool):
    features, labels, masks = cache
    head.train(mode=train_mode)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            predictions.append(head(features[start : start + batch_size].to(device)).float().cpu())
    task_r2 = calculate_r2_per_task(labels.numpy(), torch.cat(predictions).numpy(), masks.numpy())
    return task_r2, finite_mean(task_r2)


def save_head_checkpoint(path: Path, spec: ModelSpec, fold: int, epoch: int, best_r2: float, r2_per_task, head, train_size: int, test_size: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold + 1,
            "epoch": epoch,
            "best_r2_mean": best_r2,
            "r2_per_task": r2_per_task,
            "head_state_dict": head.state_dict(),
            "input_dim": next(head.parameters()).shape[1],
            "num_targets": 19,
            "train_size": int(train_size),
            "test_size": int(test_size),
            "hyperparameters": {
                "hidden_dim_1": spec.hidden_dim_1,
                "hidden_dim_2": spec.hidden_dim_2,
                "dropout": 0.2,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "backbone_training_mode_during_train": True,
                "backbone_eval_mode_during_validation": True,
            },
        },
        path,
    )


def train_hf_model(spec: ModelSpec, output_dir: Path, device: torch.device, args):
    shared_dir = Path(args.shared_dir) if args.shared_dir is not None else None
    dataset, tokenizer = build_hf_dataset(spec, args.limit_rows, shared_dir)
    if shared_dir is None:
        split_iterator = KFold(n_splits=5, shuffle=True, random_state=42).split(dataset)
    else:
        _, fold_ids = load_shared_frame_and_folds(shared_dir)
        if len(fold_ids) != len(dataset):
            raise ValueError(f"{spec.name} dropped IDs after the shared intersection was frozen")
        split_iterator = explicit_fold_indices(fold_ids)
    fold_results = []
    pad_id = get_pad_token_id(tokenizer)
    selected_folds = set(args.folds if args.folds is not None else range(5))
    for fold, (train_idx, test_idx) in enumerate(split_iterator):
        if fold not in selected_folds:
            continue
        model = HFRegressionModel(spec).to(device)
        for param in model.bert.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
        criterion = MultiTaskLoss()
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_hf(batch, pad_id),
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_hf(batch, pad_id),
        )
        best_r2 = 0.0 if shared_dir is not None else -float("inf")
        best_epoch = None
        best_task_r2 = None
        fold_dir = output_dir / f"fold_{fold + 1}"
        held_out_cache = cache_hf_held_out_features(model, test_loader, device)

        for epoch in range(args.num_epochs):
            model.train()
            last_loss = None
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["label"].to(device), batch["label_mask"].to(device))
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            r2_per_task, r2_mean = evaluate_cached_head(
                model.classifier,
                held_out_cache,
                device,
                args.batch_size,
                train_mode=False,
            )
            if r2_mean > best_r2:
                best_r2 = r2_mean
                best_epoch = epoch
                best_task_r2 = r2_per_task
                save_head_checkpoint(
                    fold_dir / "best_head.pt",
                    spec,
                    fold,
                    epoch,
                    best_r2,
                    r2_per_task,
                    model.classifier,
                    len(train_idx),
                    len(test_idx),
                    args,
                )
            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                print(f"[{spec.name}] fold={fold + 1} epoch={epoch + 1}/{args.num_epochs} r2={r2_mean:.6f} best={best_r2:.6f}", flush=True)

        save_head_checkpoint(
            fold_dir / "final_head.pt",
            spec,
            fold,
            args.num_epochs - 1,
            best_r2,
            best_task_r2,
            model.classifier,
            len(train_idx),
            len(test_idx),
            args,
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "epoch": best_epoch,
                "best_r2_mean": best_r2,
                "r2_per_task": best_task_r2,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "best_checkpoint": str(fold_dir / "best_head.pt"),
                "final_checkpoint": str(fold_dir / "final_head.pt"),
            }
        )
    return dataset, fold_results


def train_apex_model(spec: ModelSpec, output_dir: Path, device: torch.device, args):
    shared_dir = Path(args.shared_dir) if args.shared_dir is not None else None
    dataset, backbone, _ = build_apex_components(spec, device, args.limit_rows, shared_dir)
    if shared_dir is None:
        split_iterator = KFold(n_splits=5, shuffle=True, random_state=42).split(dataset)
    else:
        _, fold_ids = load_shared_frame_and_folds(shared_dir)
        if len(fold_ids) != len(dataset):
            raise ValueError("APEX dropped IDs after the shared intersection was frozen")
        split_iterator = explicit_fold_indices(fold_ids)
    fold_results = []
    selected_folds = set(args.folds if args.folds is not None else range(5))
    for fold, (train_idx, test_idx) in enumerate(split_iterator):
        if fold not in selected_folds:
            continue
        _, backbone, head = build_apex_components(spec, device, args.limit_rows, shared_dir)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, head.parameters()), lr=args.learning_rate)
        criterion = MultiTaskLoss()
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_apex)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_apex)
        best_r2 = 0.0 if shared_dir is not None else -float("inf")
        best_epoch = None
        best_task_r2 = None
        fold_dir = output_dir / f"fold_{fold + 1}"
        held_out_cache = cache_apex_held_out_features(backbone, test_loader, device)

        for epoch in range(args.num_epochs):
            backbone.train()
            head.train()
            for batch in train_loader:
                optimizer.zero_grad()
                features = backbone(batch["input_ids"].to(device))
                logits = head(features)
                loss = criterion(logits, batch["label"].to(device), batch["label_mask"].to(device))
                loss.backward()
                optimizer.step()

            r2_per_task, r2_mean = evaluate_cached_head(
                head,
                held_out_cache,
                device,
                args.batch_size,
                train_mode=shared_dir is not None,
            )
            if r2_mean > best_r2:
                best_r2 = r2_mean
                best_epoch = epoch
                best_task_r2 = r2_per_task
                save_head_checkpoint(fold_dir / "best_head.pt", spec, fold, epoch, best_r2, r2_per_task, head, len(train_idx), len(test_idx), args)
            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                print(f"[{spec.name}] fold={fold + 1} epoch={epoch + 1}/{args.num_epochs} r2={r2_mean:.6f} best={best_r2:.6f}", flush=True)

        save_head_checkpoint(fold_dir / "final_head.pt", spec, fold, args.num_epochs - 1, best_r2, best_task_r2, head, len(train_idx), len(test_idx), args)
        fold_results.append(
            {
                "fold": fold + 1,
                "epoch": best_epoch,
                "best_r2_mean": best_r2,
                "r2_per_task": best_task_r2,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "best_checkpoint": str(fold_dir / "best_head.pt"),
                "final_checkpoint": str(fold_dir / "final_head.pt"),
            }
        )
    return dataset, fold_results


def run_model(task: dict):
    spec = MODEL_SPECS[task["model_name"]]
    args = argparse.Namespace(**task["args"])
    gpu = int(task["gpu"])
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.name}] starting faithful online training on {device}", flush=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    initial_torch_seed = int(torch.initial_seed())

    if spec.kind == "apex":
        dataset, fold_results = train_apex_model(spec, output_dir, device, args)
    else:
        dataset, fold_results = train_hf_model(spec, output_dir, device, args)

    metrics = {
        "model": spec.name,
        "source_script": spec.source_script,
        "num_examples": int(len(dataset)),
        "original_length": int(dataset.original_length),
        "best_mean_R2_across_folds": float(np.mean([item["best_r2_mean"] for item in fold_results])),
        "folds": fold_results,
        "target_columns": list(dataset.target_columns),
        "training_semantics": (
            "online frozen backbone; train uses model.train(); held-out fold uses model.eval(); "
            "published APEX head dropout remains active during held-out-fold selection"
            if spec.kind == "apex" and args.shared_dir is not None
            else "online frozen backbone; train uses model.train(); validation uses model.eval()"
        ),
        "shared_protocol_dir": str(args.shared_dir) if args.shared_dir is not None else None,
        "initial_torch_seed": initial_torch_seed,
        "hyperparameters": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kfold_splits": 5,
            "kfold_shuffle": True,
            "kfold_random_state": 42,
            "seed": args.seed,
            "selected_folds_zero_based": args.folds,
            "held_out_frozen_backbone_eval_cached_once": args.shared_dir is not None,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[{spec.name}] completed mean_r2={metrics['best_mean_R2_across_folds']:.6f}", flush=True)
    return metrics


def parse_models(values: list[str]) -> list[str]:
    if values == ["all"]:
        return ["chemberta_mtr", "molformer", "apex", "peptideclm", "chemberta_mlm"]
    for value in values:
        if value not in MODEL_SPECS:
            raise ValueError(f"Unknown model {value}. Choices: {sorted(MODEL_SPECS)}")
    return values


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
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=range(5),
        default=None,
        help="Optional zero-based folds to run; defaults to all five.",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=DEFAULT_SHARED_DIR,
        help="Common-ID/fold directory. Pass an empty value only for historical native-filter runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = parse_models(args.models)
    gpus = parse_gpus(args.gpus)
    task_args = vars(args).copy()
    task_args["output_dir"] = str(args.output_dir)
    tasks = [{"model_name": model_name, "gpu": gpus[idx % len(gpus)], "args": task_args} for idx, model_name in enumerate(model_names)]
    all_metrics = {}
    with ProcessPoolExecutor(max_workers=min(len(gpus), len(tasks))) as executor:
        futures = [executor.submit(run_model, task) for task in tasks]
        for future in as_completed(futures):
            metrics = future.result()
            all_metrics[metrics["model"]] = metrics["best_mean_R2_across_folds"]
    summary = {"best_mean_R2_across_folds": all_metrics}
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
