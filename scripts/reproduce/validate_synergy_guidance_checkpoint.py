#!/usr/bin/env python3
"""Strictly validate an observed all-data synergy guidance checkpoint."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.synergy import prepare_legacy_synergy_data  # noqa: E402
from apexoracle.data.synergy_dataset import (  # noqa: E402
    TokenizedSynergyPairDataset,
    collate_tokenized_synergy_genome_text,
)
from apexoracle.models.legacy_mdlm_encoder import (  # noqa: E402
    build_frozen_legacy_mdlm_encoder,
)
from apexoracle.models.synergy_checkpoint import (  # noqa: E402
    build_legacy_synergy_components,
    load_synergy_guidance_checkpoint,
)
from apexoracle.training.synergy import synergy_guidance_pair_forward  # noqa: E402
from apexoracle.training.synergy_guidance_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    GuidanceConfig,
    load_features,
    load_tokenizer,
    validate_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=["short_judger", "guidance_40epoch"],
        default="guidance_40epoch",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = GuidanceConfig.load(
        config_path,
        REPO_ROOT,
        profile=args.profile,
    )
    validate_paths(config)
    checkpoint = args.checkpoint or (
        config.paths.observed_output / "synergy_noise_clsfier_best.ckpt"
    )
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    device = torch.device(args.device)
    tokenizer = load_tokenizer(config, local_files_only=args.local_files_only)
    molecule_encoder = build_frozen_legacy_mdlm_encoder(
        mdlm_root=config.paths.mdlm_root,
        checkpoint_path=config.paths.mdlm_checkpoint,
        tokenizer=tokenizer,
        device=device,
    )
    components = build_legacy_synergy_components(
        config.paths.base_mic_checkpoint,
        device=device,
        molecule_dim=config.molecule_dim,
        genome_dim=config.genome_dim,
        text_dim=config.text_dim,
        attention_heads=config.attention_heads,
        lora_rank=config.lora_rank,
    )
    contract = load_synergy_guidance_checkpoint(
        checkpoint,
        molecule_encoder=molecule_encoder,
        components=components,
    )
    features = load_features(config, device)
    prepared = prepare_legacy_synergy_data(
        REPO_ROOT,
        source_path=config.paths.source,
    )
    import selfies

    dataset = TokenizedSynergyPairDataset(
        prepared.genome_text.head(1),
        tokenizer=tokenizer,
        selfies_encoder=selfies.encoder,
        genome_embeddings=features.genomes,
        text_embeddings=features.atcc_text,
        max_length=config.eligibility_max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(
            collate_tokenized_synergy_genome_text,
            pad_token_id=tokenizer.pad_token_id,
            fixed_length=config.model_fixed_length,
        ),
    )
    components.genome_attention.eval()
    components.text_attention.eval()
    components.prediction_head.eval()
    batch = next(iter(loader))
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed(123)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with torch.no_grad():
        result = synergy_guidance_pair_forward(
            batch,
            device=device,
            molecule_encoder=molecule_encoder,
            genome_attention=components.genome_attention,
            text_attention=components.text_attention,
            prediction_head=components.prediction_head,
            criterion=nn.BCEWithLogitsLoss(),
            missing_genome_embedding=components.missing_genome_embedding,
            has_genome=True,
            autocast_enabled=device.type == "cuda",
        )
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=device.type == "cuda"
    ):
        input_ids = batch["input_ids"].to(device)
        first = molecule_encoder(input_ids[::2], noise_input=False)[:, 0, :]
        second = molecule_encoder(input_ids[1::2], noise_input=False)[:, 0, :]
        fused = []
        for offset, molecule in ((0, first), (1, second)):
            genome = components.genome_attention(
                mol_cls_emb=molecule,
                genome_embs=batch["padded_genome_embeddings"][offset::2],
                key_padding_mask=1 - batch["genome_attn_masks"][offset::2],
            )
            text = components.text_attention(
                mol_cls_emb=molecule,
                genome_embs=batch["padded_text_embeddings"][offset::2],
                key_padding_mask=1 - batch["text_attn_masks"][offset::2],
            )
            fused.append(
                torch.cat(
                    (genome.reshape(-1, 8192), text.reshape(-1, 4096)),
                    dim=1,
                )
            )
        inline_logits = (
            components.prediction_head(torch.cat(fused, dim=1))
            + components.prediction_head(torch.cat(fused[::-1], dim=1))
        ) / 2
    torch.testing.assert_close(result.logits, inline_logits, rtol=0, atol=0)
    report = {
        "status": "strict_load_and_forward_ok",
        "profile": args.profile,
        "checkpoint": str(checkpoint),
        "contract": contract,
        "mdlm_initial_missing_keys": list(molecule_encoder.missing_keys),
        "mdlm_initial_unexpected_keys": list(molecule_encoder.unexpected_keys),
        "fixed_batch_logit": float(result.logits.item()),
        "fixed_batch_loss": float(result.loss.item()),
        "inline_legacy_max_abs_logit_difference": float(
            (result.logits - inline_logits).abs().max().item()
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
