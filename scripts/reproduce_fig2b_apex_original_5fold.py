#!/usr/bin/env python3
"""Re-run the original APEX Fig. 2b 5-fold protocol and save head checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
APEX_ROOT = REPO_ROOT / "compare_APEX"
sys.path.insert(0, str(APEX_ROOT))

from APEX_models import AMP_model_fix  # noqa: E402
from fine_tune_on_DBAASP_SMILES import MultiTaskLoss, calculate_r2_per_task  # noqa: E402
from utils import AAindex, make_vocab, onehot_encoding  # noqa: E402


class AAseqsDataset(Dataset):
    def __init__(self, dataframe, max_length, word2vec):
        self.dataframe = dataframe
        self.original_length = len(self.dataframe)
        self.max_length = max_length
        self.target_columns = self.dataframe.columns.tolist()[2:]
        self.remove_long_smiles()
        self.seqs = onehot_encoding(self.dataframe["AAseqs"].tolist(), max_length, word2vec)

    def remove_long_smiles(self):
        self.dataframe = self.dataframe[self.dataframe["AAseqs"].apply(lambda x: len(x) <= self.max_length)]
        self.dataframe = self.dataframe.reset_index(drop=True)
        return self.dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        target = self.dataframe.loc[idx, self.target_columns].values.tolist()
        return {
            "input_ids": self.seqs[idx],
            "label": torch.tensor(target, dtype=torch.float),
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    labels = torch.stack(labels, dim=0)
    mask = labels >= -0.5
    labels_processed = labels.clone()
    labels_processed[mask] = -torch.log10(labels[mask] / 10)
    mask = mask.int()
    return {
        "input_ids": torch.from_numpy(np.array(input_ids)),
        "label": labels_processed,
        "label_mask": mask,
    }


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim_1=384,
        hidden_dim_2=128,
        num_targets=19,
        pooler_dropout: float = 0.2,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_head(
    path: Path,
    fold: int,
    epoch: int,
    best_r2: float,
    r2_per_task,
    head,
    train_size: int,
    test_size: int,
    args,
    eval_rng_state: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold + 1,
            "epoch": epoch,
            "best_r2_mean": float(best_r2),
            "r2_per_task": [float(x) for x in r2_per_task],
            "head_state_dict": head.state_dict(),
            "input_dim": 128,
            "num_targets": 19,
            "train_size": int(train_size),
            "test_size": int(test_size),
            "hyperparameters": {
                "hidden_dim_1": 512,
                "hidden_dim_2": 256,
                "dropout": 0.2,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "seed": args.seed,
                "backbone_training_mode_during_train": True,
                "backbone_eval_mode_during_validation": True,
                "head_training_mode_during_validation": True,
                "validation_batch_size": args.batch_size,
                "source_protocol": "compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py",
            },
            "eval_rng_state": eval_rng_state,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "DataPrepare/Data/DBAASP_id_same_as_SMILES_AAseqs_bact_MICs_512_limit.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_len = 52
    word2idx, _ = make_vocab()
    emb, _ = AAindex(str(APEX_ROOT / "aaindex1.csv"), word2idx)
    emb_size = np.shape(emb)[1]
    data = pd.read_csv(args.data_path)
    dataset = AAseqsDataset(data, max_len, word2idx)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    criterion = MultiTaskLoss()
    pretrained_path = APEX_ROOT / "APEX_ckpt/APEX_pretrained_encoder_state_dict_best.ckpt"

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model = AMP_model_fix(emb, emb_size, num_rnn_layers=3, dim_h=128)
        model.load_state_dict(state_dict)
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False

        cls_head = RegressionHead(128, 512, 256, 19).to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, cls_head.parameters()), lr=args.learning_rate)
        best_r2 = -float("inf")
        best_epoch = None
        best_task_r2 = None

        for epoch in range(args.num_epochs):
            model.train()
            cls_head.train()
            last_loss = None
            for batch in tqdm(train_loader, desc=f"seed={args.seed} fold={fold + 1} epoch={epoch + 1}/{args.num_epochs} train", leave=False):
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                label_masks = batch["label_mask"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids)
                logits = cls_head(outputs)
                loss = criterion(logits, labels, label_masks)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            model.eval()
            # The original script does not call cls_head.eval() before validation,
            # so dropout in the regression head remains active.
            eval_rng_state = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            all_labels, all_preds, all_label_masks = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(batch["input_ids"].to(device))
                    logits = cls_head(outputs)
                    all_labels.extend(batch["label"].cpu().numpy())
                    all_preds.extend(logits.cpu().numpy())
                    all_label_masks.extend(batch["label_mask"].cpu().numpy())
            r2_per_task = calculate_r2_per_task(all_labels, all_preds, all_label_masks)
            r2_mean = float(np.array(r2_per_task).mean())
            if r2_mean > best_r2:
                best_r2 = r2_mean
                best_epoch = epoch
                best_task_r2 = r2_per_task
                save_head(
                    args.output_dir / f"fold_{fold + 1}" / "best_head.pt",
                    fold,
                    epoch,
                    best_r2,
                    r2_per_task,
                    cls_head,
                    len(train_idx),
                    len(test_idx),
                    args,
                    eval_rng_state=eval_rng_state,
                )
            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                print(
                    f"[apex-original] seed={args.seed} fold={fold + 1} epoch={epoch + 1}/{args.num_epochs} "
                    f"loss={last_loss:.6f} r2={r2_mean:.6f} best={best_r2:.6f}",
                    flush=True,
                )

        save_head(args.output_dir / f"fold_{fold + 1}" / "final_head.pt", fold, args.num_epochs - 1, best_r2, best_task_r2, cls_head, len(train_idx), len(test_idx), args)
        fold_results.append(
            {
                "fold": fold + 1,
                "epoch": int(best_epoch),
                "best_r2_mean": float(best_r2),
                "r2_per_task": [float(x) for x in best_task_r2],
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "best_checkpoint": str(args.output_dir / f"fold_{fold + 1}" / "best_head.pt"),
            }
        )

    metrics = {
        "model": "apex",
        "source_script": "compare_APEX/APEX_fix_train_DBAASP_MIC_5_fold_mean.py",
        "num_examples": int(len(dataset)),
        "original_length": int(dataset.original_length),
        "best_mean_R2_across_folds": float(np.mean([item["best_r2_mean"] for item in fold_results])),
        "folds": fold_results,
        "target_columns": list(dataset.target_columns),
        "hyperparameters": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "kfold_splits": 5,
            "kfold_shuffle": True,
            "kfold_random_state": 42,
            "seed": args.seed,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_mean_R2_across_folds": metrics["best_mean_R2_across_folds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
