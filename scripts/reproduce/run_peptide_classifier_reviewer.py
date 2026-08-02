#!/usr/bin/env python
"""Train/evaluate the v1-source-label peptide classifier for Reviewer 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from apexoracle.training.peptide_classifier import (
    FrozenBackbonePeptideClassifier,
    deterministic_mask,
    move_chance_at_t,
)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class IndexedClassifierDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices: np.ndarray,
        molecule_hashes: np.memmap,
    ):
        self.dataset = dataset
        self.indices = indices
        self.molecule_hashes = molecule_hashes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        original_index = int(self.indices[item])
        row = self.dataset[original_index]
        molecule = np.asarray(
            self.molecule_hashes[original_index], dtype=np.uint64
        ).view(np.int64).item()
        return row["input_ids"], row["labels"], molecule


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def partition_for_evaluation(indices: np.ndarray, rank: int, world: int) -> np.ndarray:
    return indices[rank::world]


def gather_numpy(array: np.ndarray, rank: int, world: int):
    if world == 1:
        return [array]
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(array, gathered, dst=0)
    return gathered


def aggregate_molecules(
    labels: np.ndarray, logits: np.ndarray, molecules: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    order = np.argsort(molecules, kind="stable")
    molecules = molecules[order]
    labels = labels[order]
    logits = logits[order]
    starts = np.r_[0, np.flatnonzero(molecules[1:] != molecules[:-1]) + 1]
    ends = np.r_[starts[1:], len(molecules)]
    group_labels = np.add.reduceat(labels.astype(np.int64), starts)
    group_sizes = ends - starts
    consistent = (group_labels == 0) | (group_labels == group_sizes)
    binary_labels = (group_labels[consistent] > 0).astype(np.uint8)
    group_logits = (
        np.add.reduceat(logits.astype(np.float64), starts)[consistent]
        / group_sizes[consistent]
    )
    return (
        molecules[starts][consistent],
        binary_labels,
        group_logits.astype(np.float32),
        int(np.count_nonzero(~consistent)),
    )


def molecule_metrics(
    labels: np.ndarray, logits: np.ndarray, molecules: np.ndarray
) -> tuple[dict, dict[str, np.ndarray]]:
    molecule_ids, binary_labels, group_logits, excluded = aggregate_molecules(
        labels, logits, molecules
    )
    both_classes = np.unique(binary_labels).size == 2
    predicted = group_logits >= 0.0
    positive = binary_labels == 1
    negative = ~positive
    recall = float(np.mean(predicted[positive])) if np.any(positive) else None
    specificity = (
        float(np.mean(~predicted[negative])) if np.any(negative) else None
    )
    metrics = {
        "auprc": (
            float(average_precision_score(binary_labels, group_logits))
            if both_classes
            else None
        ),
        "auroc": (
            float(roc_auc_score(binary_labels, group_logits))
            if both_classes
            else None
        ),
        "excluded_conflicting_molecules": excluded,
        "molecules": int(len(binary_labels)),
        "negative_molecules": int(np.count_nonzero(binary_labels == 0)),
        "non_peptide_specificity_at_probability_0_5": specificity,
        "peptide_recall_at_probability_0_5": recall,
        "positive_molecules": int(np.count_nonzero(binary_labels == 1)),
    }
    if recall is not None and specificity is not None:
        metrics["balanced_accuracy_at_probability_0_5"] = (
            recall + specificity
        ) / 2
    predictions = {
        "molecule_hash": molecule_ids,
        "label": binary_labels,
        "logit": group_logits,
    }
    return metrics, predictions


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    rank: int,
    world: int,
    device: torch.device,
    t: float,
    replicates: int,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    model.eval()
    local_logits = []
    local_labels = []
    local_molecules = []
    chance = move_chance_at_t(t)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    for input_ids, labels, molecule_hashes_signed in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        molecule_hashes = molecule_hashes_signed.numpy().view(np.uint64)
        replicate_logits = []
        for replicate in range(replicates):
            if t == 0:
                mask = torch.zeros_like(input_ids, dtype=torch.bool)
            else:
                mask_np = deterministic_mask(
                    molecule_hashes,
                    input_ids.shape[1],
                    replicate=replicate,
                    move_chance=chance,
                )
                mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
            time_tensor = torch.full(
                (input_ids.shape[0],), t, dtype=torch.float32, device=device
            )
            logits = raw_model.logits_at_t(input_ids, time_tensor, explicit_mask=mask)
            replicate_logits.append(logits.float().cpu().numpy())
        local_logits.append(np.mean(replicate_logits, axis=0))
        local_labels.append(labels.numpy())
        local_molecules.append(molecule_hashes.copy())
    arrays = {
        "logits": np.concatenate(local_logits),
        "labels": np.concatenate(local_labels),
        "molecules": np.concatenate(local_molecules),
    }
    gathered = {
        name: gather_numpy(value, rank, world) for name, value in arrays.items()
    }
    if rank != 0:
        return None
    combined = {
        name: np.concatenate(parts) for name, parts in gathered.items()
    }
    return molecule_metrics(
        combined["labels"], combined["logits"], combined["molecules"]
    )


def broadcast_metrics(metrics: dict | None, rank: int, world: int) -> dict:
    if world == 1:
        assert metrics is not None
        return metrics
    payload = [metrics]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=225)
    parser.add_argument("--eval-batch-size", type=int, default=225)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--scheduler-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise-replicates", type=int, default=10)
    parser.add_argument("--validation-interval-steps", type=int, default=45_000)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    rank, world, local_rank = setup_distributed()
    seed_everything(args.seed, rank)
    device = torch.device("cuda", local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads((args.split_dir / "split_manifest.json").read_text())
    row_count = int(split_manifest["row_count"])
    splits = np.memmap(
        args.split_dir / "split_codes.u1", mode="r", dtype=np.uint8, shape=(row_count,)
    )
    molecules = np.memmap(
        args.split_dir / "molecule_hashes.u8",
        mode="r",
        dtype=np.uint64,
        shape=(row_count,),
    )
    train_indices = np.flatnonzero(splits == 0)
    validation_indices = np.flatnonzero(splits == 1)
    test_indices = np.flatnonzero(splits == 2)
    dataset = load_from_disk(str(args.dataset_dir))
    if len(dataset) != row_count:
        raise RuntimeError(f"Dataset has {len(dataset)} rows; split has {row_count}")
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    model = FrozenBackbonePeptideClassifier(
        args.producer_root, args.v1_checkpoint
    ).to(device)
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = torch.optim.Adam(raw_model.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.scheduler_epochs, eta_min=1e-9
    )
    label_counts = split_manifest["label_by_split"]
    train_positive = int(label_counts["1"]["train"])
    train_negative = int(label_counts["0"]["train"])
    pos_weight = torch.tensor(
        train_negative / train_positive, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        raw_model.head.load_state_dict(saved["head_state_dict"])

    validation_local = partition_for_evaluation(validation_indices, rank, world)
    validation_loader = DataLoader(
        IndexedClassifierDataset(dataset, validation_local, molecules),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_local = partition_for_evaluation(test_indices, rank, world)
    test_loader = DataLoader(
        IndexedClassifierDataset(dataset, test_local, molecules),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    provenance = {
        "args": vars(args) | {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
        "backbone_checkpoint_sha256": sha256(args.v1_checkpoint),
        "dataset_rows": row_count,
        "global_batch_size": args.batch_size * world,
        "split_manifest_sha256": sha256(args.split_dir / "split_manifest.json"),
        "time_conditioning": True,
        "world_size": world,
    }
    if rank == 0:
        manifest_name = (
            "evaluation_manifest.json" if args.evaluate_only else "run_manifest.json"
        )
        atomic_json(args.output_dir / manifest_name, provenance)

    if args.evaluate_only:
        if args.checkpoint is None:
            raise SystemExit("--evaluate-only requires --checkpoint")
        clean_result = evaluate(
            model,
            test_loader,
            rank=rank,
            world=world,
            device=device,
            t=0.0,
            replicates=1,
        )
        noisy_result = evaluate(
            model,
            test_loader,
            rank=rank,
            world=world,
            device=device,
            t=0.5,
            replicates=args.noise_replicates,
        )
        if rank == 0:
            clean, clean_predictions = clean_result
            noisy, noisy_predictions = noisy_result
            if not np.array_equal(
                clean_predictions["molecule_hash"],
                noisy_predictions["molecule_hash"],
            ) or not np.array_equal(
                clean_predictions["label"], noisy_predictions["label"]
            ):
                raise RuntimeError("Clean/noisy molecule prediction sets differ")
            np.savez_compressed(
                args.output_dir / "test_predictions.npz",
                molecule_hash=clean_predictions["molecule_hash"],
                label=clean_predictions["label"],
                clean_logit=clean_predictions["logit"],
                t_0_5_logit=noisy_predictions["logit"],
            )
            atomic_json(
                args.output_dir / "test_metrics.json",
                {
                    "clean": clean,
                    "t_0_5": noisy,
                    "t_0_5_mask_replicates": args.noise_replicates,
                },
            )
        if world > 1:
            dist.destroy_process_group()
        return

    train_dataset = IndexedClassifierDataset(dataset, train_indices, molecules)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    best_auprc = -float("inf")
    history = []
    global_step = 0

    def validate_and_checkpoint(
        *,
        epoch: int,
        step_in_epoch: int,
        running_loss: float,
        started: float,
        include_noisy: bool,
    ) -> None:
        nonlocal best_auprc
        clean_result = evaluate(
            model,
            validation_loader,
            rank=rank,
            world=world,
            device=device,
            t=0.0,
            replicates=1,
        )
        clean = broadcast_metrics(
            clean_result[0] if rank == 0 else None, rank, world
        )
        noisy = None
        if include_noisy:
            noisy_result = evaluate(
                model,
                validation_loader,
                rank=rank,
                world=world,
                device=device,
                t=0.5,
                replicates=1,
            )
            noisy = broadcast_metrics(
                noisy_result[0] if rank == 0 else None, rank, world
            )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "seconds": time.time() - started,
            "train_loss": running_loss / step_in_epoch,
            "validation_clean": clean,
            "validation_t_0_5_one_mask": noisy,
        }
        history.append(record)
        if rank == 0:
            atomic_json(args.output_dir / "history.json", {"evaluations": history})
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "head_state_dict": raw_model.head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "record": record,
                "run_manifest": provenance,
            }
            torch.save(payload, args.output_dir / "last.pt")
            if clean["auprc"] > best_auprc:
                best_auprc = clean["auprc"]
                torch.save(payload, args.output_dir / "best.pt")
                atomic_json(
                    args.output_dir / "best_validation_metrics.json", record
                )
        if world > 1:
            dist.barrier()
        model.train()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        started = time.time()
        for step, (input_ids, labels, _) in enumerate(train_loader, start=1):
            global_step += 1
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            if rank == 0 and step % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "mean_loss": running_loss / step,
                            "seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
            if (
                args.validation_interval_steps > 0
                and step % args.validation_interval_steps == 0
                and (len(train_loader) - step)
                > max(1, args.validation_interval_steps // 4)
            ):
                validate_and_checkpoint(
                    epoch=epoch,
                    step_in_epoch=step,
                    running_loss=running_loss,
                    started=started,
                    include_noisy=False,
                )
        scheduler.step()
        validate_and_checkpoint(
            epoch=epoch,
            step_in_epoch=len(train_loader),
            running_loss=running_loss,
            started=started,
            include_noisy=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
