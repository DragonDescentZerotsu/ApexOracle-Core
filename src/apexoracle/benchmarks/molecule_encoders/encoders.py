"""Frozen encoder adapters that obey the shared Fig. 2b ID contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .apex_adapter import (
    build_apex_vocabulary,
    encode_apex_sequences,
)
from .data import SharedBenchmarkData
from .feature_cache import FeatureCache


@dataclass(frozen=True)
class HFEncoderSpec:
    name: str
    model_name: str
    revision: str
    trust_remote_code: bool = False
    tokenizer_kind: str = "auto"
    pooling: str = "first_token"
    max_length: int = 512


HF_ENCODERS = {
    "chemberta_mtr": HFEncoderSpec(
        name="chemberta_mtr",
        model_name="DeepChem/ChemBERTa-77M-MTR",
        revision="66b895cab8adebea0cb59a8effa66b2020f204ca",
    ),
    "chemberta_mlm": HFEncoderSpec(
        name="chemberta_mlm",
        model_name="DeepChem/ChemBERTa-77M-MLM",
        revision="ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
    ),
    "molformer": HFEncoderSpec(
        name="molformer",
        model_name="ibm/MoLFormer-XL-both-10pct",
        revision="7b12d946c181a37f6012b9dc3b002275de070314",
        trust_remote_code=True,
    ),
    "peptideclm": HFEncoderSpec(
        name="peptideclm",
        model_name="aaronfeller/PeptideCLM-23M-all",
        revision="a0847d8231d236645a2c4f629590118716c6fdda",
        tokenizer_kind="vendored_peptideclm",
    ),
}


def load_hf_tokenizer(spec: HFEncoderSpec, repo_root: Path):
    if spec.tokenizer_kind == "vendored_peptideclm":
        from apexoracle.vendor.peptideclm_tokenizer import load_tokenizer

        return load_tokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        spec.model_name,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )


def _tokenize_without_dropping(tokenizer, texts: tuple[str, ...], max_length: int):
    token_ids = []
    records_truncated = 0
    records_with_unk = 0
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        full = tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(full) > max_length:
            records_truncated += 1
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if unknown_id is not None and unknown_id in encoded:
            records_with_unk += 1
        token_ids.append(encoded)
    return token_ids, records_truncated, records_with_unk


def extract_hf_features(
    benchmark: SharedBenchmarkData,
    spec: HFEncoderSpec,
    *,
    repo_root: Path,
    device: str,
    batch_size: int,
) -> FeatureCache:
    """Extract eval-mode first-token features for every shared SMILES."""

    if spec.pooling != "first_token":
        raise ValueError(f"unsupported pooling method: {spec.pooling}")
    import torch
    from transformers import AutoModel

    tokenizer = load_hf_tokenizer(spec, Path(repo_root))
    token_ids, truncated_count, unknown_count = _tokenize_without_dropping(
        tokenizer,
        benchmark.smiles,
        spec.max_length,
    )
    model = AutoModel.from_pretrained(
        spec.model_name,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    feature_batches = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), batch_size):
            batch_ids = token_ids[start : start + batch_size]
            padded = tokenizer.pad(
                {"input_ids": batch_ids},
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                "input_ids": padded["input_ids"].to(torch_device),
                "attention_mask": padded["attention_mask"].to(torch_device),
            }
            outputs = model(**inputs)
            feature_batches.append(outputs.last_hidden_state[:, 0, :].float().cpu().numpy())
    features = np.ascontiguousarray(np.concatenate(feature_batches, axis=0), dtype=np.float32)
    if features.shape[0] != len(benchmark):
        raise RuntimeError("encoder did not return one feature row per shared molecule")
    return FeatureCache(
        encoder_name=spec.name,
        features=features,
        molecule_ids=benchmark.molecule_ids,
        metadata={
            "model_name": spec.model_name,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "pooling": spec.pooling,
            "max_length": spec.max_length,
            "overflow_policy": "truncate_and_record",
            "records_truncated": truncated_count,
            "records_with_unk": unknown_count,
            "encoder_mode": "eval",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_apex_features(
    benchmark: SharedBenchmarkData,
    *,
    aaindex_path: Path,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
) -> FeatureCache:
    """Extract features from the published, unmodified APEX encoder."""

    import torch

    aaindex_path = Path(aaindex_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    from apexoracle.benchmarks.molecule_encoders.apex_model import (
        ApexEncoder,
        load_aaindex_embedding,
    )

    legacy_vocabulary, _ = build_apex_vocabulary()
    legacy_embedding, _ = load_aaindex_embedding(aaindex_path, legacy_vocabulary)
    embedding = np.asarray(legacy_embedding)
    vocabulary, _ = build_apex_vocabulary()
    token_ids, _ = encode_apex_sequences(
        benchmark.apex_sequences,
        word_to_index=vocabulary,
    )

    model = ApexEncoder(
        embedding, embedding.shape[1], num_rnn_layers=3, hidden_dim=128
    )
    state_dict = _torch_load(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), batch_size):
            batch = torch.from_numpy(token_ids[start : start + batch_size]).to(torch_device)
            features.append(model(batch).float().cpu().numpy())
    matrix = np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float32)
    if matrix.shape[0] != len(benchmark):
        raise RuntimeError("APEX did not return one feature row per shared molecule")
    return FeatureCache(
        encoder_name="apex",
        features=matrix,
        molecule_ids=benchmark.molecule_ids,
        metadata={
            "checkpoint_reference": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "aaindex_reference": str(aaindex_path),
            "aaindex_sha256": _sha256(aaindex_path),
            "pooling": "APEX pretrained encoder output",
            "encoder_mode": "eval",
            "max_length": 52,
            "max_content_residues": 50,
            "x_token_index": 0,
            "x_behavior": "original APEX unknown symbol behavior; encoded as index 0",
        },
    )
