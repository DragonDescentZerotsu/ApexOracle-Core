#!/usr/bin/env python3
"""Cache MoLFormer fold-eval embeddings for Fig. 2b inference-only replay.

MoLFormer outputs can vary with the dynamically padded batch shape. The
faithful online run evaluates each fold through that fold's validation loader,
so this cache fills each row with the embedding produced in the same fold/test
batch context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from reproduce_fig2b_baselines_online_5fold import (
    MODEL_SPECS,
    TokenizedSmilesDataset,
    collate_hf,
    get_pad_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("Checkpoints/fig2b_baselines_online_5fold/molformer/features.pt"))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def prepare_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = raw_labels >= -0.5
    labels = raw_labels.copy()
    labels[masks] = -np.log10(labels[masks] / 10)
    return labels, masks.astype(np.float32)


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS["molformer"]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model, trust_remote_code=spec.trust_remote_code)
    data = pd.read_csv(spec.data_path)
    if args.limit_rows is not None:
        data = data.head(args.limit_rows)
    dataset = TokenizedSmilesDataset(data, tokenizer)
    pad_id = get_pad_token_id(tokenizer)

    backbone = AutoModel.from_pretrained(spec.hf_model, trust_remote_code=spec.trust_remote_code).to(device)
    backbone.eval()

    features = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    with torch.no_grad():
        for fold_idx, (_, test_idx) in enumerate(kf.split(dataset), start=1):
            loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_hf(batch, pad_id),
            )
            cursor = 0
            for batch in tqdm(loader, desc=f"molformer fold {fold_idx} eval-cache"):
                outputs = backbone(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                batch_features = outputs.last_hidden_state[:, 0, :].detach().cpu().float()
                if features is None:
                    features = torch.empty((len(dataset), batch_features.shape[1]), dtype=torch.float32)
                rows = test_idx[cursor : cursor + batch_features.shape[0]]
                features[rows] = batch_features
                cursor += batch_features.shape[0]

    if features is None:
        raise RuntimeError("No features were extracted")

    labels, masks = prepare_labels(dataset.dataframe[dataset.target_columns].values.astype(np.float32))
    cache = {
        "features": features,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "label_masks": torch.tensor(masks, dtype=torch.float32),
        "target_columns": list(dataset.target_columns),
        "dbaasp_ids": [str(value) for value in dataset.dataframe["DBAASP_id"].tolist()],
        "original_length": int(dataset.original_length),
        "filtered_length": int(len(dataset)),
        "model": spec.name,
        "source_script": spec.source_script,
        "pooling": spec.pooling,
        "feature_dim": int(features.shape[1]),
        "feature_cache_semantics": "fold-specific validation-loader dynamic-padding cache",
        "hyperparameters": {
            "batch_size": args.batch_size,
            "kfold_splits": 5,
            "kfold_shuffle": True,
            "kfold_random_state": 42,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    manifest = {
        "feature_cache": str(args.output),
        "num_examples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_cache_semantics": cache["feature_cache_semantics"],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
