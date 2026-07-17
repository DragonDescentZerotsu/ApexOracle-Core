"""One shared regression-head trainer for every frozen molecule encoder."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SharedBenchmarkData
from .feature_cache import FeatureCache
from .metrics import finite_macro_mean, masked_r2_per_task


@dataclass(frozen=True)
class HeadTrainingConfig:
    hidden_dim_1: int = 384
    hidden_dim_2: int = 128
    dropout: float = 0.2
    learning_rate: float = 1e-4
    batch_size: int = 200
    max_epochs: int = 200
    patience: int = 20
    validation_fraction: float = 0.1
    seed: int = 42


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        *,
        hidden_dim_1: int = 384,
        hidden_dim_2: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.dense_1(features)))
        hidden = self.dropout(self.activation(self.dense_2(hidden)))
        return self.output(hidden)


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = label_mask.to(dtype=predictions.dtype)
        denominator = weights.sum()
        if denominator.item() == 0:
            raise ValueError("training batch contains no observed labels")
        return (((predictions - labels) ** 2) * weights).sum() / denominator


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(labels[indices]),
        torch.from_numpy(mask[indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _predict(
    head: RegressionHead,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    head.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = torch.from_numpy(features[batch_indices]).to(device)
            outputs.append(head(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _evaluate(
    head: RegressionHead,
    data: SharedBenchmarkData,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[Any], float]:
    predictions = _predict(head, features, indices, device=device, batch_size=batch_size)
    per_task = masked_r2_per_task(
        data.labels[indices],
        predictions,
        data.label_mask[indices],
    )
    return predictions, per_task, finite_macro_mean(per_task)


def train_shared_heads(
    data: SharedBenchmarkData,
    cache: FeatureCache,
    output_dir: Path,
    *,
    config: HeadTrainingConfig = HeadTrainingConfig(),
    device: str = "cpu",
) -> dict[str, Any]:
    """Train all outer folds and evaluate each untouched test fold once."""

    if cache.molecule_ids != data.molecule_ids:
        raise ValueError("feature cache is not aligned to the shared benchmark")
    if config.max_epochs <= 0 or config.patience <= 0 or config.batch_size <= 0:
        raise ValueError("max_epochs, patience and batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    features = np.asarray(cache.features, dtype=np.float32)
    fold_values = sorted(int(value) for value in np.unique(data.folds))
    criterion = MaskedMSELoss()
    fold_results = []
    prediction_rows = []

    for outer_fold in fold_values:
        fold_seed = config.seed + outer_fold
        _set_seed(fold_seed)
        train_indices, validation_indices = data.train_validation_indices(
            outer_fold,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
        _, test_indices = data.outer_fold_indices(outer_fold)
        head = RegressionHead(
            features.shape[1],
            len(data.target_columns),
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            dropout=config.dropout,
        ).to(torch_device)
        optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)
        loader = _make_loader(
            features,
            data.labels,
            data.label_mask,
            train_indices,
            batch_size=config.batch_size,
            shuffle=True,
            seed=fold_seed,
        )

        best_validation_r2 = -float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(config.max_epochs):
            head.train()
            for batch_features, batch_labels, batch_mask in loader:
                optimizer.zero_grad(set_to_none=True)
                predictions = head(batch_features.to(torch_device))
                loss = criterion(
                    predictions,
                    batch_labels.to(torch_device),
                    batch_mask.to(torch_device),
                )
                loss.backward()
                optimizer.step()

            _, _, validation_r2 = _evaluate(
                head,
                data,
                features,
                validation_indices,
                device=torch_device,
                batch_size=config.batch_size,
            )
            if validation_r2 > best_validation_r2:
                best_validation_r2 = validation_r2
                best_epoch = epoch
                best_state = copy.deepcopy(head.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        if best_state is None:
            raise RuntimeError(f"outer fold {outer_fold} did not produce a valid checkpoint")
        head.load_state_dict(best_state)
        test_predictions, test_per_task, test_macro_r2 = _evaluate(
            head,
            data,
            features,
            test_indices,
            device=torch_device,
            batch_size=config.batch_size,
        )
        checkpoint_dir = output_dir / f"fold_{outer_fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": best_state,
                "encoder_name": cache.encoder_name,
                "outer_fold": outer_fold,
                "best_epoch": best_epoch,
                "validation_macro_r2": best_validation_r2,
                "test_macro_r2": test_macro_r2,
                "input_dim": int(features.shape[1]),
                "num_targets": len(data.target_columns),
                "training_config": asdict(config),
            },
            checkpoint_dir / "best_head.pt",
        )
        fold_results.append(
            {
                "fold": outer_fold,
                "best_epoch": best_epoch,
                "validation_macro_r2": best_validation_r2,
                "test_macro_r2": test_macro_r2,
                "test_r2_per_task": test_per_task,
                "train_size": int(len(train_indices)),
                "validation_size": int(len(validation_indices)),
                "test_size": int(len(test_indices)),
            }
        )
        for local_index, global_index in enumerate(test_indices):
            for task_index, task_name in enumerate(data.target_columns):
                if not data.label_mask[global_index, task_index]:
                    continue
                prediction_rows.append(
                    {
                        "encoder": cache.encoder_name,
                        "fold": outer_fold,
                        "dbaasp_id": data.molecule_ids[global_index],
                        "task": task_name,
                        "label": float(data.labels[global_index, task_index]),
                        "prediction": float(test_predictions[local_index, task_index]),
                    }
                )

    test_scores = np.asarray([result["test_macro_r2"] for result in fold_results])
    metrics = {
        "encoder": cache.encoder_name,
        "number_of_molecules": len(data),
        "feature_dim": int(features.shape[1]),
        "training_config": asdict(config),
        "outer_test_macro_r2_mean": float(test_scores.mean()),
        "outer_test_macro_r2_sample_sd": float(test_scores.std(ddof=1)),
        "folds": fold_results,
        "feature_metadata": cache.metadata,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("encoder", "fold", "dbaasp_id", "task", "label", "prediction"),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    return metrics
