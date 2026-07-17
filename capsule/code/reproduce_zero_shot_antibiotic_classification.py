#!/usr/bin/env python3
"""Inference-only reproduction for zero-shot antibiotic classification.

This evaluates the three zero-shot target strains from Fig. 1b using the
trained all-test 10-ensemble checkpoints. It does not train or fine-tune.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


GENOME_SCALE = 1e14
TEXT_SCALE = 1.0
GROUP_NAMES = ["#004", "17978", "Staphylococcus aureus RN4220"]
DISPLAY_GROUP_NAMES = {
    "#004": "E. coli BW25113",
    "17978": "A. baumannii ATCC 17978",
}
GENOME_TEXT_GROUPS = {0, 1}
CHECKPOINT_DIR = Path(
    "Checkpoints/genome_text_learnable_emb/antibiotic_3_strain_compare/"
    "MDLM_fix_cls_sm_all_test_10_fold_ensembles"
)
OUTPUT_PREFIX = "zero_shot_antibiotic_classification"


def parse_atcc_id_from_embedding_name(file_name: str) -> str:
    stem = file_name.split(".")[0]
    if "ATCC" not in stem:
        return stem
    suffix = stem.split("ATCC")[-1]
    components = suffix.split("_")[1:]
    if len(components) == 2:
        return "-".join(components)
    return components[0]


def genome_embedding_paths(folder_path: Path) -> dict[str, Path]:
    paths = {}
    for path in folder_path.iterdir():
        if path.is_file():
            paths[parse_atcc_id_from_embedding_name(path.name)] = path
    return paths


def text_wo_genome_paths(folder_path: Path) -> dict[str, Path]:
    paths = {}
    for path in folder_path.iterdir():
        if path.is_file():
            strain_name = path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
            paths[strain_name] = path
    return paths


def load_embedding(path: Path, scale: float) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False) * scale


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_embeddings: dict[str, torch.Tensor],
        peptide_embeddings: dict[str, torch.Tensor],
        small_molecule_embeddings: dict[str, torch.Tensor],
        genome_embeddings: dict[str, torch.Tensor] | None = None,
        max_length: int = 512,
    ):
        self.dataframe = dataframe.copy()
        self.text_embeddings = text_embeddings
        self.genome_embeddings = genome_embeddings
        self.peptide_embeddings = peptide_embeddings
        self.small_molecule_embeddings = small_molecule_embeddings
        self.max_length = max_length
        self._remove_long_tokenized_molecules()

    def _remove_long_tokenized_molecules(self) -> None:
        self.dataframe["input_len"] = self.dataframe["SMILES"].apply(lambda x: len(ast.literal_eval(x)))
        self.dataframe = self.dataframe[self.dataframe["input_len"] <= self.max_length].reset_index(drop=True)
        self.dataframe.drop(columns=["SMILES", "input_len"], inplace=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[idx]
        molecule_id = row["DBAASP_id"]
        mol_emb = self.peptide_embeddings.get(molecule_id)
        if mol_emb is None:
            mol_emb = self.small_molecule_embeddings[molecule_id]
        strain_name = row["strain_name"]
        item = {
            "label": torch.tensor(row["MIC"], dtype=torch.float32),
            "text_embedding": self.text_embeddings[strain_name],
            "strain_name": strain_name,
            "molecule_id": molecule_id,
            "row_index": int(row["_source_row_index"]),
            "mol_emb": mol_emb.squeeze(),
        }
        if self.genome_embeddings is not None:
            item["genome_embedding"] = self.genome_embeddings[strain_name]
        return item


def pad_embedding_list(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(embedding) for embedding in embeddings)
    dim = embeddings[0].shape[1]
    padded = []
    masks = []
    for embedding in embeddings:
        length = len(embedding)
        mask = torch.zeros(max_length, dtype=torch.uint8)
        pad = torch.zeros((max_length, dim), dtype=torch.bfloat16)
        pad[:length] = embedding.to(torch.bfloat16)
        mask[:length] = 1
        padded.append(pad)
        masks.append(mask)
    return torch.stack(padded), torch.stack(masks)


def collate_classification(batch: list[dict]) -> dict:
    labels = torch.stack([item["label"] for item in batch]).to(torch.float32)
    mol_emb = torch.stack([item["mol_emb"] for item in batch])
    padded_text, text_masks = pad_embedding_list([item["text_embedding"] for item in batch])
    output = {
        "label": labels,
        "mol_emb": mol_emb,
        "padded_text_embeddings": padded_text,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
        "molecule_ids": [item["molecule_id"] for item in batch],
        "row_indices": [item["row_index"] for item in batch],
    }
    if "genome_embedding" in batch[0]:
        padded_genome, genome_masks = pad_embedding_list([item["genome_embedding"] for item in batch])
        output["padded_genome_embeddings"] = padded_genome
        output["genome_attn_masks"] = genome_masks
    return output


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, num_targets: int, dropout: float):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation_fn(self.dense_1(features)))
        x = self.dropout(self.activation_fn(self.dense_2(x)))
        return self.out_proj(x)


class FirstTokenAttentionGenome(nn.Module):
    def __init__(self, mol_cls_embed_dim: int, genome_embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, genome_embed_dim)
        self.key_value_projection = nn.Linear(genome_embed_dim, genome_embed_dim * 2)
        self.mha = nn.MultiheadAttention(genome_embed_dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(genome_embed_dim)
        self.norm1 = nn.LayerNorm(genome_embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(genome_embed_dim, genome_embed_dim),
            nn.GELU(),
            nn.Linear(genome_embed_dim, genome_embed_dim),
        )
        self.norm2 = nn.LayerNorm(genome_embed_dim)

    def forward(self, mol_cls_emb: torch.Tensor, genome_embs: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        genome_dim = genome_embs.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :].transpose(0, 1)
        key_value = self.key_value_projection(genome_embs.reshape(-1, genome_dim)).reshape(
            genome_embs.shape[0], genome_embs.shape[1], -1
        )
        key_value = key_value.transpose(0, 1)
        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        attn_output, _ = self.mha(
            query_norm,
            key_value[:, :, :genome_dim],
            key_value[:, :, genome_dim:],
            key_padding_mask=key_padding_mask.to(torch.bool),
        )
        query = self.norm1(query.squeeze() + attn_output.squeeze())
        return self.norm2(query + self.ffn(query))


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype == torch.bfloat16:
                moved[key] = value.to(device=device, dtype=torch.float32)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_model(checkpoint: dict, mol_dim: int, genome_dim: int, text_dim: int, device: torch.device) -> tuple:
    co_cross_attn_genome = FirstTokenAttentionGenome(mol_dim, genome_dim, 4, 0.1).to(device)
    co_cross_attn_text = FirstTokenAttentionGenome(mol_dim, text_dim, 4, 0.1).to(device)
    cls_head = RegressionHead(genome_dim + text_dim, (genome_dim + text_dim) // 4, 128, 1, 0.2).to(device)

    co_cross_attn_genome.load_state_dict(checkpoint["co_cross_attn_genome"])
    co_cross_attn_text.load_state_dict(checkpoint["co_cross_attn_text"])
    cls_head.load_state_dict(checkpoint["cls_head_state_dict"])
    learnable_embedding_weight = checkpoint["learnable_embedding_weight"]
    if isinstance(learnable_embedding_weight, nn.Parameter):
        learnable_embedding_weight = learnable_embedding_weight.detach()
    learnable_embedding_weight = learnable_embedding_weight.to(device)

    co_cross_attn_genome.eval()
    co_cross_attn_text.eval()
    cls_head.eval()
    return co_cross_attn_genome, co_cross_attn_text, cls_head, learnable_embedding_weight


def predict_loader(
    loader: DataLoader,
    has_genome: bool,
    models: tuple,
    device: torch.device,
    max_batches: int | None,
    desc: str,
) -> tuple[list[float], list[float], list[str], list[str], list[int]]:
    co_cross_attn_genome, co_cross_attn_text, cls_head, learnable_embedding_weight = models
    labels_all = []
    logits_all = []
    strains_all = []
    molecule_ids_all = []
    row_indices_all = []

    with torch.no_grad():
        total = len(loader) if max_batches is None else min(len(loader), max_batches)
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, total=total, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]
            mol_cls_embedding = batch["mol_emb"]
            padded_text_embeddings = batch["padded_text_embeddings"]
            text_attn_masks = batch["text_attn_masks"]

            if has_genome:
                padded_genome_embeddings = batch["padded_genome_embeddings"]
                genome_attn_masks = batch["genome_attn_masks"]
            else:
                padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                genome_attn_masks = torch.ones((mol_cls_embedding.shape[0], 1), dtype=torch.uint8, device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                genome_part = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
                text_part = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
                features = torch.cat((genome_part, text_part), dim=1)
                logits = cls_head(features)

            labels_all.extend(labels.detach().cpu().flatten().tolist())
            logits_all.extend(logits.detach().cpu().flatten().tolist())
            strains_all.extend(batch["strain_names"])
            molecule_ids_all.extend(batch["molecule_ids"])
            row_indices_all.extend(batch["row_indices"])

    return labels_all, logits_all, strains_all, molecule_ids_all, row_indices_all


def safe_classification_metrics(labels: list[float], logits: np.ndarray) -> dict[str, float | None]:
    try:
        auroc = float(roc_auc_score(labels, logits))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(labels, logits))
    except ValueError:
        auprc = None
    return {"AUROC": auroc, "AUPRC": auprc}


def as_json_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def collect_molecule_embeddings(dataframe: pd.DataFrame, peptide_path: Path, small_molecule_path: Path) -> tuple[dict, dict]:
    needed_ids = set(dataframe["DBAASP_id"].tolist())
    peptide_all = torch.load(peptide_path, map_location="cpu", weights_only=False)
    small_molecule_all = torch.load(small_molecule_path, map_location="cpu", weights_only=False)
    peptide = {key: value for key, value in peptide_all.items() if key in needed_ids}
    small_molecule = {key: value for key, value in small_molecule_all.items() if key in needed_ids}
    missing = sorted(needed_ids - set(peptide) - set(small_molecule))
    if missing:
        raise KeyError(f"Missing {len(missing)} molecule embeddings; first missing id: {missing[0]}")
    return peptide, small_molecule


def build_group_dataset(data_root: Path, group_idx: int) -> tuple[Dataset, bool, int, int, int]:
    data_dir = data_root / "DataPrepare" / "Data"
    target_strain = GROUP_NAMES[group_idx]
    sm_data = pd.read_csv(data_dir / "small_molecule" / "processed" / "small_molecule_Evo_binary_data_SELFIES.csv")
    sm_data = sm_data.reset_index().rename(columns={"index": "_source_row_index"})
    test_data = sm_data[sm_data["strain_name"] == target_strain].reset_index(drop=True)
    if test_data.empty:
        raise RuntimeError(f"No small-molecule test data found for {target_strain}")

    genome_paths = genome_embedding_paths(data_dir / "Genome_embs")
    text_atcc_paths = genome_embedding_paths(data_dir / "Text_Description" / "ATCC" / "embeddings")
    text_wo_paths = text_wo_genome_paths(data_dir / "Text_Description" / "wo_ATCC" / "embeddings")

    peptide_embeddings, small_molecule_embeddings = collect_molecule_embeddings(
        test_data,
        data_dir / "Pep_emb_dict_cls_wo_pad_eval.pt",
        data_dir / "SM_emb_dict_cls_wo_pad_eval.pt",
    )
    mol_dim = next(iter((peptide_embeddings or small_molecule_embeddings).values())).squeeze().shape[-1]

    if group_idx in GENOME_TEXT_GROUPS:
        genome_embeddings = {target_strain: load_embedding(genome_paths[target_strain], GENOME_SCALE)}
        text_embeddings = {target_strain: load_embedding(text_atcc_paths[target_strain], TEXT_SCALE)}
        genome_dim = genome_embeddings[target_strain].shape[1]
        has_genome = True
    else:
        genome_embeddings = None
        text_embeddings = {target_strain: load_embedding(text_wo_paths[target_strain], TEXT_SCALE)}
        genome_dim = 8192
        has_genome = False

    text_dim = text_embeddings[target_strain].shape[1]
    dataset = ClassificationDataset(
        test_data,
        text_embeddings=text_embeddings,
        peptide_embeddings=peptide_embeddings,
        small_molecule_embeddings=small_molecule_embeddings,
        genome_embeddings=genome_embeddings,
    )
    return dataset, has_genome, mol_dim, genome_dim, text_dim


def checkpoint_path(checkpoint_root: Path, group_idx: int, ensemble_idx: int) -> Path:
    return checkpoint_root / f"genome_text_learnable_emb_SM_outer_SM_best_AUROC_group_{group_idx}_ensemble_{ensemble_idx}.pth"


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    checkpoint_root = data_root / CHECKPOINT_DIR
    group_indices = list(range(len(GROUP_NAMES))) if args.group is None else [args.group]
    all_metrics = {}
    prediction_frames = []

    for group_idx in tqdm(group_indices, desc="groups"):
        target_strain = GROUP_NAMES[group_idx]
        dataset, has_genome, mol_dim, genome_dim, text_dim = build_group_dataset(data_root, group_idx)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_classification)

        ensemble_logits = []
        base_labels = None
        base_strains = None
        base_molecule_ids = None
        base_row_indices = None
        checkpoint_metrics = []

        for ensemble_idx in tqdm(range(args.num_ensembles), desc=f"{target_strain} ensembles", leave=False):
            path = checkpoint_path(checkpoint_root, group_idx, ensemble_idx)
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(checkpoint, mol_dim, genome_dim, text_dim, device)
            labels, logits, strains, molecule_ids, row_indices = predict_loader(
                loader,
                has_genome=has_genome,
                models=model,
                device=device,
                max_batches=args.max_batches,
                desc=f"{target_strain} ensemble {ensemble_idx}",
            )
            ensemble_logits.append(np.asarray(logits, dtype=np.float64))
            base_labels = labels
            base_strains = strains
            base_molecule_ids = molecule_ids
            base_row_indices = row_indices
            checkpoint_metrics.append(
                {
                    "ensemble": ensemble_idx,
                    "checkpoint": str(path.relative_to(data_root)),
                    "checkpoint_AUROC": as_json_float(checkpoint.get("auroc")),
                    "checkpoint_AUPRC": as_json_float(checkpoint.get("auprc")),
                }
            )
            del checkpoint, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble_logits_array = np.stack(ensemble_logits, axis=0)
        mean_logits = ensemble_logits_array.mean(axis=0)
        probabilities = 1.0 / (1.0 + np.exp(-mean_logits))
        metrics = safe_classification_metrics(base_labels, mean_logits)
        metrics.update(
            {
                "target_strain": DISPLAY_GROUP_NAMES.get(target_strain, target_strain),
                "group": group_idx,
                "has_genome": has_genome,
                "num_examples": len(base_labels),
                "num_ensembles": args.num_ensembles,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "checkpoint_metrics": checkpoint_metrics,
            }
        )
        all_metrics[f"group_{group_idx}"] = metrics

        prediction_frames.append(
            pd.DataFrame(
                {
                    "group": group_idx,
                    "target_strain": target_strain,
                    "source_row_index": base_row_indices,
                    "molecule_id": base_molecule_ids,
                    "label": base_labels,
                    "ensemble_logit_mean": mean_logits,
                    "ensemble_probability": probabilities,
                    "strain_name": base_strains,
                }
            )
        )

    if device.type == "cuda":
        all_metrics["_cuda_peak_memory"] = {
            "allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }

    metrics_path = results_dir / f"{OUTPUT_PREFIX}_metrics.json"
    predictions_path = results_dir / f"{OUTPUT_PREFIX}_predictions.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-ensembles", type=int, default=10)
    parser.add_argument("--group", type=int, choices=range(len(GROUP_NAMES)))
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
