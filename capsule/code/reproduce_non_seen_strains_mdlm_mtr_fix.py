#!/usr/bin/env python3
"""Inference-only reproduction for the non-seen-strain MDLM-MTR 7-ensemble result.

This script reconstructs the original train/test split from the raw data shipped
in the capsule, loads the trained model checkpoints, and evaluates the held-out
strains. It does not train or fine-tune any model.
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
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


GENOME_SCALE = 1e14
TEXT_SCALE = 1.0
RANDOM_SEEDS = [
    42, 2024, 2025, 2077, 2012, 1973, 2002, 2001, 2020, 2019, 31, 13, 55,
    11, 12, 58, 72, 2010, 2008, 2001, 1717, 1313, 99, 83, 29, 1001, 1002,
    1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027,
]


def parse_atcc_id_from_embedding_name(file_name: str) -> str:
    stem = file_name.split(".")[0]
    if "ATCC" not in stem:
        return stem
    stem = stem.split("ATCC")[-1]
    components = stem.split("_")[1:]
    if len(components) == 2:
        return "-".join(components)
    return components[0]


def genome_embedding_paths(folder_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {}
    genome_id_to_species_first_name = {}
    for path in folder_path.iterdir():
        if not path.is_file():
            continue
        genome_id = parse_atcc_id_from_embedding_name(path.name)
        paths[genome_id] = path
        genome_id_to_species_first_name[genome_id] = path.name.split(".")[0].split("_")[0]
    return paths, genome_id_to_species_first_name


def text_wo_genome_paths(folder_path: Path) -> dict[str, Path]:
    paths = {}
    for path in folder_path.iterdir():
        if not path.is_file():
            continue
        strain_name = path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        paths[strain_name] = path
    return paths


def load_embedding(path: Path, scale: float) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False) * scale


def get_original_strain_name_with_genome_embedding(
    evo_mic_count_file_path: Path, embedded_genome_ids: set[str]
) -> tuple[list[str], list[str], dict[str, str]]:
    with open(evo_mic_count_file_path, "r", encoding="utf-8") as f:
        strain_count_data = json.load(f)

    handcrafted = []
    dbaasp_original = []
    for name in strain_count_data:
        if "*" in name:
            original_name, standard_name = name.split("*")
            if "ATCC" in standard_name:
                standard_name = standard_name.split("ATCC")[-1].strip()
            else:
                standard_name = standard_name.strip()
            handcrafted.append((original_name.strip(), standard_name))
        elif "ATCC" in name:
            atcc_id = name.split("ATCC")[-1].strip()
            if "BAA" in name:
                atcc_id = atcc_id.replace(" ", "-")
            if "MY" in name:
                atcc_id = atcc_id.replace(" ", "")
            if "MAY" in name:
                atcc_id = atcc_id.replace("MAY", "MYA")
            if "D" in name:
                atcc_id = atcc_id.split("D")[0]
            if "T" in name:
                atcc_id = atcc_id.split("T")[0]
            if "s" in name:
                atcc_id = atcc_id.split("s")[0]
            if " " in name:
                atcc_id = atcc_id.split(" ")[0]
            dbaasp_original.append((name.strip(), atcc_id))

    handcrafted_names = [orig for orig, std in handcrafted if std in embedded_genome_ids]
    dbaasp_names = [orig for orig, std in dbaasp_original if std in embedded_genome_ids]
    return handcrafted_names, dbaasp_names, dict(handcrafted + dbaasp_original)


def exclude_wrong_species_atcc_map(
    evo_mic_data_with_genome_embedding: np.ndarray,
    genome_id_to_species_first_name: dict[str, str],
) -> np.ndarray:
    cleaned_data = []
    for line in evo_mic_data_with_genome_embedding:
        name = line[1]
        if "ATCC" not in name:
            cleaned_data.append(line)
            continue

        atcc_id = name.split("ATCC")[-1].strip()
        if "BAA" in name:
            atcc_id = atcc_id.replace(" ", "-")
        if "MY" in name:
            atcc_id = atcc_id.replace(" ", "")
        if "MAY" in name:
            atcc_id = atcc_id.replace("MAY", "MYA")
        if "D" in name:
            atcc_id = atcc_id.split("D")[0]
        if "T" in name:
            atcc_id = atcc_id.split("T")[0]
        if "s" in name:
            atcc_id = atcc_id.split("s")[0]
        if " " in name:
            atcc_id = atcc_id.split(" ")[0]

        if genome_id_to_species_first_name.get(atcc_id) is None:
            cleaned_data.append(line)
        elif genome_id_to_species_first_name[atcc_id] in name:
            cleaned_data.append(line)
    return np.array(cleaned_data)


def get_atcc_id_to_species_name_map(atcc_fasta_folder_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    atcc_ids = []
    species_names = []
    for path in atcc_fasta_folder_path.iterdir():
        if not path.is_file():
            continue
        file_name = path.name
        atcc_id = file_name.split(".")[0].split("ATCC")[-1].strip()
        atcc_id = atcc_id.replace("_", " ").strip().replace(" ", "-")
        species_name = file_name.split("ATCC")[0]
        if "subsp" in species_name.split("_"):
            species_name = species_name.split("subsp")[0]
        if "pathovar" in species_name.split("_"):
            species_name = species_name.split("pathovar")[0]
        if "var" in species_name.split("_"):
            species_name = species_name.split("var")[0]
        if "sp" in species_name.split("_"):
            species_name = species_name.split("_sp")[0]
        species_name = species_name.replace("_", " ").strip()
        atcc_ids.append(atcc_id)
        species_names.append(species_name)

    atcc_to_species = dict(zip(atcc_ids, species_names))
    species_to_atcc = {}
    atcc_array = np.array(atcc_ids)
    species_array = np.array(species_names)
    for species_name in set(species_array):
        species_to_atcc[species_name] = list(atcc_array[species_array == species_name])
    return atcc_to_species, species_to_atcc


def get_original_strain_id_to_species_name_map(
    original_text_emb_folder_path: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    strain_names = []
    species_names = []
    for path in original_text_emb_folder_path.iterdir():
        if not path.is_file():
            continue
        strain_name = path.name.split(".pt")[0].replace("～", " ").replace("^", "/")
        species_name = " ".join(strain_name.split(" ")[:2])
        strain_names.append(strain_name)
        species_names.append(species_name)

    strain_to_species = dict(zip(strain_names, species_names))
    species_to_strain = {}
    strain_array = np.array(strain_names)
    species_array = np.array(species_names)
    for species_name in set(species_array):
        species_to_strain[species_name] = list(strain_array[species_array == species_name])
    return strain_to_species, species_to_strain


def merge_dict(dict_1: dict[str, list[str]], dict_2: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in dict_1.items()}
    for key, value in dict_2.items():
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = list(value)
    return merged


class MoleculeDatasetWithGenomeAndText(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        genome_embeddings: dict[str, torch.Tensor],
        text_embeddings: dict[str, torch.Tensor],
        peptide_embeddings: dict[str, torch.Tensor],
        small_molecule_embeddings: dict[str, torch.Tensor],
        max_length: int = 512,
    ):
        self.dataframe = dataframe.copy()
        self.genome_embeddings = genome_embeddings
        self.text_embeddings = text_embeddings
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.dataframe.iloc[idx]
        dbaasp_id = row["DBAASP_id"]
        mol_emb = self.peptide_embeddings.get(dbaasp_id)
        if mol_emb is None:
            mol_emb = self.small_molecule_embeddings[dbaasp_id]
        strain_name = row["strain_name"]
        return {
            "label": torch.tensor(row["MIC"], dtype=torch.float32),
            "genome_embedding": self.genome_embeddings[strain_name],
            "text_embedding": self.text_embeddings[strain_name],
            "strain_name": strain_name,
            "mol_emb": mol_emb.squeeze(),
        }


class MoleculeDatasetWithTextOnly(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_embeddings: dict[str, torch.Tensor],
        peptide_embeddings: dict[str, torch.Tensor],
        small_molecule_embeddings: dict[str, torch.Tensor],
        max_length: int = 512,
    ):
        self.dataframe = dataframe.copy()
        self.text_embeddings = text_embeddings
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.dataframe.iloc[idx]
        dbaasp_id = row["DBAASP_id"]
        mol_emb = self.peptide_embeddings.get(dbaasp_id)
        if mol_emb is None:
            mol_emb = self.small_molecule_embeddings[dbaasp_id]
        return {
            "label": torch.tensor(row["MIC"], dtype=torch.float32),
            "text_embedding": self.text_embeddings[row["strain_name"]],
            "strain_name": row["strain_name"],
            "mol_emb": mol_emb.squeeze(),
        }


def collate_genome_text(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    labels = torch.stack([item["label"] for item in batch]).to(torch.float32)
    labels = -torch.log10(labels / 10)
    mol_emb = torch.stack([item["mol_emb"] for item in batch])
    genome_embeddings = [item["genome_embedding"] for item in batch]
    text_embeddings = [item["text_embedding"] for item in batch]

    padded_genome, genome_masks = pad_embedding_list(genome_embeddings)
    padded_text, text_masks = pad_embedding_list(text_embeddings)
    return {
        "label": labels,
        "mol_emb": mol_emb,
        "padded_genome_embeddings": padded_genome,
        "genome_attn_masks": genome_masks,
        "padded_text_embeddings": padded_text,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
    }


def collate_text_only(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    labels = torch.stack([item["label"] for item in batch]).to(torch.float32)
    labels = -torch.log10(labels / 10)
    mol_emb = torch.stack([item["mol_emb"] for item in batch])
    padded_text, text_masks = pad_embedding_list([item["text_embedding"] for item in batch])
    return {
        "label": labels,
        "mol_emb": mol_emb,
        "padded_text_embeddings": padded_text,
        "text_attn_masks": text_masks,
        "strain_names": [item["strain_name"] for item in batch],
    }


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


def calculate_r2(labels: list[float] | np.ndarray, preds: list[float] | np.ndarray) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    return float(1 - (np.sum((labels - preds) ** 2) / np.sum((labels - np.mean(labels)) ** 2)))


def compute_metrics(labels: list[float], preds: list[float]) -> dict[str, float]:
    return {
        "R2": calculate_r2(labels, preds),
        "spearman": float(spearmanr(labels, preds)[0]),
        "pearson": float(pearsonr(labels, preds)[0]),
    }


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


def build_model(checkpoint: dict, mol_dim: int, genome_dim: int, text_dim: int, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, torch.Tensor]:
    co_cross_attn_genome = FirstTokenAttentionGenome(mol_dim, genome_dim, 4, 0.1).to(device)
    co_cross_attn_text = FirstTokenAttentionGenome(mol_dim, text_dim, 4, 0.1).to(device)
    reg_head = RegressionHead(genome_dim + text_dim, (genome_dim + text_dim) // 4, 128, 1, 0.2).to(device)

    co_cross_attn_genome.load_state_dict(checkpoint["co_cross_attn_genome"])
    co_cross_attn_text.load_state_dict(checkpoint["co_cross_attn_text"])
    reg_head.load_state_dict(checkpoint["re_head_state_dict"])
    learnable_embedding_weight = checkpoint["learnable_embedding_weight"]
    if isinstance(learnable_embedding_weight, nn.Parameter):
        learnable_embedding_weight = learnable_embedding_weight.detach()
    learnable_embedding_weight = learnable_embedding_weight.to(device)

    co_cross_attn_genome.eval()
    co_cross_attn_text.eval()
    reg_head.eval()
    return co_cross_attn_genome, co_cross_attn_text, reg_head, learnable_embedding_weight


def predict_loader(
    loader: DataLoader,
    has_genome: bool,
    models: tuple[nn.Module, nn.Module, nn.Module, torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
    desc: str = "predicting",
) -> tuple[list[float], list[float], list[str]]:
    co_cross_attn_genome, co_cross_attn_text, reg_head, learnable_embedding_weight = models
    labels_all = []
    preds_all = []
    strains_all = []

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
                features = torch.cat((genome_part.reshape(-1, 8192), text_part.reshape(-1, 4096)), dim=1)
                logits = reg_head(features)

            labels_all.extend(labels.detach().cpu().flatten().tolist())
            preds_all.extend(logits.detach().cpu().flatten().tolist())
            strains_all.extend(batch["strain_names"])
    return labels_all, preds_all, strains_all


def build_splits(data_root: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Path], dict[str, Path], dict[str, Path]]:
    data_dir = data_root / "DataPrepare" / "Data"
    genome_paths, genome_id_to_species = genome_embedding_paths(data_dir / "Genome_embs")
    text_atcc_paths, _ = genome_embedding_paths(data_dir / "Text_Description" / "ATCC" / "embeddings")
    text_wo_paths = text_wo_genome_paths(data_dir / "Text_Description" / "wo_ATCC" / "embeddings")

    handcrafted, dbaasp_original, origin_to_standard = get_original_strain_name_with_genome_embedding(
        data_dir / "Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json",
        set(genome_paths),
    )

    all_evo_mic = pd.read_csv(data_dir / "DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv")
    columns = all_evo_mic.columns
    all_evo_mic_values = all_evo_mic.values
    sm_evo_binary = pd.read_csv(data_dir / "small_molecule" / "processed" / "small_molecule_Evo_binary_data_SELFIES.csv").values

    all_evo_mic_values = np.array([line for line in all_evo_mic_values if "del" not in line[1]], dtype=object)
    with_genome_handcrafted = [line for line in all_evo_mic_values if line[1] in handcrafted]
    with_genome_dbaasp = [line for line in all_evo_mic_values if line[1] in dbaasp_original]
    with_genome_dbaasp = exclude_wrong_species_atcc_map(np.array(with_genome_dbaasp, dtype=object), genome_id_to_species)
    with_genome = np.concatenate((np.array(with_genome_handcrafted, dtype=object), with_genome_dbaasp))

    text_only = []
    text_only_names = set(text_wo_paths)
    for line in all_evo_mic_values:
        if len(line[1].split(" ")) <= 1:
            continue
        if line[1].split(" ")[1] not in ["sp.", "spp.", "group"] and line[1] in text_only_names:
            text_only.append(line)

    with_genome_standard = []
    for line in with_genome:
        line = line.copy()
        line[1] = origin_to_standard[line[1]]
        with_genome_standard.append(line)
    with_genome_standard = np.array(with_genome_standard, dtype=object)
    with_genome_or_text = np.concatenate((with_genome_standard, np.array(text_only, dtype=object)))

    all_name_set = set(with_genome_or_text[:, 1])
    all_strain_line_group = {
        strain: with_genome_or_text[np.where(with_genome_or_text[:, 1] == strain)[0]]
        for strain in all_name_set
    }
    all_standard_name_set = set(with_genome_standard[:, 1])
    standard_strain_line_group = {
        strain: with_genome_standard[np.where(with_genome_standard[:, 1] == strain)[0]]
        for strain in all_standard_name_set
    }

    _, species_to_atcc = get_atcc_id_to_species_name_map(data_dir / "Genome" / "ATCC")
    _, species_to_original_strain = get_original_strain_id_to_species_name_map(
        data_dir / "Text_Description" / "wo_ATCC" / "embeddings"
    )
    merged_species_to_strain = merge_dict(species_to_atcc, species_to_original_strain)

    with open(data_dir / "Genome" / "old_to_new_NCBI_taxonomy.json", "r", encoding="utf-8") as f:
        old_to_new = json.load(f)
    new_to_old = {value: key for key, value in old_to_new.items()}
    two_way_taxonomy = new_to_old | old_to_new

    train_groups = [[], [], []]
    test_groups = [[], [], []]
    for fold_idx in range(len(train_groups)):
        repeated_species = []
        for species_name, corresponding_ids in merged_species_to_strain.items():
            if species_name in repeated_species:
                continue
            merged_ids = list(corresponding_ids)
            if species_name in two_way_taxonomy:
                repeated_species.append(two_way_taxonomy[species_name])
                extra = merged_species_to_strain.get(two_way_taxonomy[species_name])
                if extra is not None:
                    merged_ids.extend(extra)

            merged_ids.sort()
            if len(merged_ids) >= 6:
                merged_ids[1], merged_ids[2] = merged_ids[2], merged_ids[1]

            if len(merged_ids) == 1:
                train_groups[fold_idx].extend(merged_ids)
            elif len(merged_ids) == 2:
                train_groups[fold_idx].append(merged_ids[fold_idx % 2])
                test_groups[fold_idx].append(merged_ids[(fold_idx + 1) % 2])
            else:
                chunk_length = len(merged_ids) // len(train_groups)
                test_ids = merged_ids[fold_idx * chunk_length : (fold_idx + 1) * chunk_length]
                train_ids = list(set(merged_ids) - set(test_ids))
                train_groups[fold_idx].extend(train_ids)
                test_groups[fold_idx].extend(test_ids)

    gt_test_frames = []
    text_only_test_frames = []
    for strain_for_test in test_groups:
        gt_strain_for_test = set(strain_for_test) & all_standard_name_set
        gt_test = [standard_strain_line_group[strain] for strain in gt_strain_for_test]
        gt_test_df = pd.DataFrame(np.concatenate(gt_test), columns=columns)

        text_strain_for_test = (set(strain_for_test) & all_name_set) - gt_strain_for_test
        text_test = [all_strain_line_group[strain] for strain in text_strain_for_test]
        text_test_df = pd.DataFrame(np.concatenate(text_test), columns=columns)

        gt_test_frames.append(gt_test_df)
        text_only_test_frames.append(text_test_df)

    return gt_test_frames, text_only_test_frames, genome_paths, text_atcc_paths, text_wo_paths


def collect_molecule_embeddings(
    dataframes: list[pd.DataFrame],
    peptide_path: Path,
    small_molecule_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    needed_ids = set()
    for dataframe in dataframes:
        needed_ids.update(dataframe["DBAASP_id"].tolist())
    peptide_all = torch.load(peptide_path, map_location="cpu", weights_only=False)
    small_molecule_all = torch.load(small_molecule_path, map_location="cpu", weights_only=False)
    peptide = {key: value for key, value in peptide_all.items() if key in needed_ids}
    small_molecule = {key: value for key, value in small_molecule_all.items() if key in needed_ids}
    return peptide, small_molecule


def load_fold_embeddings(
    gt_test: pd.DataFrame,
    text_test: pd.DataFrame,
    genome_paths_map: dict[str, Path],
    text_atcc_paths_map: dict[str, Path],
    text_wo_paths_map: dict[str, Path],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    genome_embeddings = {}
    text_embeddings = {}
    for strain in set(gt_test["strain_name"]):
        genome_embeddings[strain] = load_embedding(genome_paths_map[strain], GENOME_SCALE)
        text_embeddings[strain] = load_embedding(text_atcc_paths_map[strain], TEXT_SCALE)
    for strain in set(text_test["strain_name"]):
        text_embeddings[strain] = load_embedding(text_wo_paths_map[strain], TEXT_SCALE)
    return genome_embeddings, text_embeddings


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    checkpoint_dir = data_root / "Checkpoints" / "genome_text_learnable_emb" / "strain_wise_w_SM_b_attn" / "MDLM_MTR_fix_7_fold_ensembles"
    data_dir = data_root / "DataPrepare" / "Data"
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    gt_test_frames, text_test_frames, genome_paths_map, text_atcc_paths_map, text_wo_paths_map = build_splits(data_root)
    peptide_embeddings, small_molecule_embeddings = collect_molecule_embeddings(
        gt_test_frames + text_test_frames,
        data_dir / "Pep_emb_dict.pt",
        data_dir / "SM_emb_dict.pt",
    )
    mol_dim = next(iter(peptide_embeddings.values())).squeeze().shape[-1]

    summary = {}
    prediction_rows = []

    fold_items = list(enumerate(zip(gt_test_frames, text_test_frames)))
    for fold_idx, (gt_test, text_test) in tqdm(fold_items, desc="Folds"):
        if args.fold is not None and fold_idx != args.fold:
            continue
        genome_embeddings, text_embeddings = load_fold_embeddings(
            gt_test, text_test, genome_paths_map, text_atcc_paths_map, text_wo_paths_map
        )
        genome_dim = next(iter(genome_embeddings.values())).shape[1]
        text_dim = next(iter(text_embeddings.values())).shape[1]

        gt_dataset = MoleculeDatasetWithGenomeAndText(
            gt_test, genome_embeddings, text_embeddings, peptide_embeddings, small_molecule_embeddings
        )
        text_dataset = MoleculeDatasetWithTextOnly(
            text_test, text_embeddings, peptide_embeddings, small_molecule_embeddings
        )
        gt_loader = DataLoader(gt_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_genome_text)
        text_loader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text_only)

        ensemble_predictions = []
        fold_labels = None
        fold_strains = None

        for ensemble_idx in tqdm(range(args.num_ensembles), desc=f"Fold {fold_idx} ensembles", leave=False):
            checkpoint_path = checkpoint_dir / f"genome_text_learnable_emb_Strain_wise_best_R2_group_{fold_idx}_ensemble_{ensemble_idx}.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            models = build_model(checkpoint, mol_dim, genome_dim, text_dim, device)
            gt_labels, gt_preds, gt_strains = predict_loader(
                gt_loader,
                True,
                models,
                device,
                args.max_batches,
                desc=f"Fold {fold_idx} ensemble {ensemble_idx} genome+text",
            )
            text_labels, text_preds, text_strains = predict_loader(
                text_loader,
                False,
                models,
                device,
                args.max_batches,
                desc=f"Fold {fold_idx} ensemble {ensemble_idx} text-only",
            )
            labels = gt_labels + text_labels
            preds = gt_preds + text_preds
            strains = gt_strains + text_strains

            if fold_labels is None:
                fold_labels = labels
                fold_strains = strains
            ensemble_predictions.append(preds)

            del checkpoint, models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble_mean = np.mean(np.asarray(ensemble_predictions), axis=0)
        metrics = compute_metrics(fold_labels, ensemble_mean.tolist())
        summary[f"group_{fold_idx}"] = {
            **metrics,
            "num_examples": len(fold_labels),
            "num_ensembles": args.num_ensembles,
        }

        for row_idx, (strain, label, pred) in enumerate(zip(fold_strains, fold_labels, ensemble_mean.tolist())):
            prediction_rows.append(
                {
                    "group": fold_idx,
                    "row_index": row_idx,
                    "strain_name": strain,
                    "label_neg_log10_mic_over_10": label,
                    "prediction": pred,
                }
            )

    with open(results_dir / "non_seen_strains_mdlm_mtr_fix_metrics.json", "w", encoding="utf-8") as f:
        if device.type == "cuda":
            summary["_cuda_peak_memory"] = {
                "allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
            }
        json.dump(summary, f, indent=2)
    pd.DataFrame(prediction_rows).to_csv(
        results_dir / "non_seen_strains_mdlm_mtr_fix_predictions.csv", index=False
    )

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root / "data")
    parser.add_argument("--results-dir", type=Path, default=default_root / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--num-ensembles", type=int, default=7)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit per loader.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
