#!/usr/bin/env python
"""Run one isolated ApexOracle remasking reviewer-generation task.

This wrapper intentionally leaves the historical sampler checkout unchanged. It
imports the frozen producer, applies explicit CLI overrides, and writes every
raw attempted sequence to an experiment-specific batch file before any
validity or peptide filtering.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem
import selfies as sf
from transformers import AutoTokenizer


MODEL_NAME = "ibm-research/materials.selfies-ted"
AMIDE_PATTERN = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--synergy-root", type=Path, required=True)
    parser.add_argument("--mdlm-root", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--guidance-regressor-checkpoint", type=Path)
    parser.add_argument("--peptide-classifier-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strain", required=True)
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--t-on", type=float, required=True)
    parser.add_argument("--t-off", type=float, required=True)
    parser.add_argument("--gamma-peptide", type=float, required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--alpha-on", type=float, default=0.5)
    parser.add_argument("--gamma-mic", type=float, default=15.0)
    parser.add_argument("--target-mic", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def register_resolvers() -> None:
    resolvers = {
        "cwd": os.getcwd,
        "device_count": torch.cuda.device_count,
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
        "if_then_else": lambda condition, x, y: x if condition else y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_config(args: argparse.Namespace):
    register_resolvers()
    config_dir = args.producer_root / "configs"
    overrides = [
        "mode=guide_sample",
        "trainer.devices=1",
        f"seed={args.seed}",
        f"loader.global_batch_size={args.batch_size}",
        f"loader.eval_global_batch_size={args.batch_size}",
        f"sampling.batch_size={args.batch_size}",
        f"sampling.num_sample_batches={args.num_batches}",
        f"sampling.steps={args.steps}",
        f"sampling.strain={args.strain}",
        f"sampling.target_MIC={args.target_mic}",
        f"sampling.target_MIC_max={args.target_mic}",
        f"sampling.target_length={args.target_length}",
        f"sampling.remdm.eta={args.eta}",
        f"sampling.remdm.t_on={args.t_on}",
        f"sampling.remdm.t_off={args.t_off}",
        f"sampling.remdm.alpha_on={args.alpha_on}",
        f"guidance.var_gamma.gamma_l={args.gamma_mic}",
        f"guidance.var_gamma.gamma_s={args.gamma_peptide}",
        "guidance.use_approx=true",
        "guidance.noise=true",
        "sampling.peptide_only=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="config", overrides=overrides)
    path_updates = {
        "sampling.pretrained_ckpt_path": (
            args.diffusion_checkpoint
            or args.mdlm_root / "Checkpoints_fangping" / "last_reg_v1.ckpt"
        ),
        "sampling.pretrain_backbone_config": (
            args.producer_root / "configs" / "config_mdlm_cls.yaml"
        ),
        "sampling.genome_test_emb_dir_path.ATCC_genome_emb": (
            args.synergy_root / "DataPrepare" / "Data" / "Genome_embs"
        ),
        "sampling.genome_test_emb_dir_path.ATCC_text_emb": (
            args.synergy_root
            / "DataPrepare"
            / "Data"
            / "Text_Description"
            / "ATCC"
            / "embeddings"
        ),
        "sampling.genome_test_emb_dir_path.only_text_emb": (
            args.synergy_root
            / "DataPrepare"
            / "Data"
            / "Text_Description"
            / "wo_ATCC"
            / "embeddings"
        ),
        "guidance.regressor_checkpoint_path": (
            args.guidance_regressor_checkpoint
            or (
                args.synergy_root
                / "Checkpoints"
                / "genome_text_learnable_emb"
                / "guidance_regressor_pad_no_mask"
                / "noise_guidance_best_R2_all_peptide_epoch_100.pth"
            )
        ),
        "guidance.classifier_checkpoint_path": (
            args.peptide_classifier_checkpoint
            or (
                args.mdlm_root
                / "cls-guide-pad-no-mask-checkpoints"
                / "epoch-epoch=1-step-step=134000-train_loss-train_loss=0.008.ckpt"
            )
        ),
    }
    for key, value in path_updates.items():
        OmegaConf.update(config, key, str(value), merge=False)
    OmegaConf.resolve(config)
    return config


def validate_assets(config) -> list[Path]:
    paths = [
        Path(config.sampling.pretrained_ckpt_path),
        Path(config.sampling.pretrain_backbone_config),
        Path(config.sampling.genome_test_emb_dir_path.ATCC_genome_emb),
        Path(config.sampling.genome_test_emb_dir_path.ATCC_text_emb),
        Path(config.sampling.genome_test_emb_dir_path.only_text_emb),
        Path(config.guidance.regressor_checkpoint_path),
        Path(config.guidance.classifier_checkpoint_path),
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")
    return paths


def load_sampler(config, tokenizer, producer_root: Path):
    if str(producer_root) not in sys.path:
        sys.path.insert(0, str(producer_root))
    import classifier
    import diffusion
    import models

    sampler = diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")
    checkpoint = torch.load(
        config.sampling.pretrained_ckpt_path,
        map_location="cuda",
        weights_only=False,
    )
    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        state_dict[key.removeprefix("backbone.")] = value
    sampler.backbone = models.dit.DIT(
        config, vocab_size=len(tokenizer.get_vocab())
    )
    sampler.backbone.load_state_dict(state_dict, strict=False)
    sampler = sampler.to("cuda").eval()

    guidance_model = classifier.Classifier(
        config, tokenizer=tokenizer, pretrained_backbone=None
    )
    guidance_model.eval()
    guidance_model.classifier_model.load_pretrained_weight()
    guidance_model = guidance_model.to("cuda")
    return sampler, guidance_model


def interpret_sample(
    token_ids: list[int],
    *,
    tokenizer,
    strain: str,
    seed: int,
    batch_index: int,
    sample_index: int,
) -> dict:
    sep_positions = [
        index for index, token in enumerate(token_ids)
        if token == tokenizer.sep_token_id
    ]
    first_sep = sep_positions[0] if sep_positions else None
    starts_with_cls = bool(token_ids and token_ids[0] == tokenizer.cls_token_id)
    complete = first_sep is not None and starts_with_cls
    contains_mask_before_sep = None
    contains_pad_before_sep = None
    selfies_value = None
    smiles_value = None
    canonical_smiles = None
    rdkit_valid = False
    has_amide_bond = False
    invalid_reason = None

    if complete:
        interior = token_ids[1:first_sep]
        contains_mask_before_sep = tokenizer.mask_token_id in interior
        contains_pad_before_sep = tokenizer.pad_token_id in interior
        if contains_mask_before_sep:
            invalid_reason = "mask_before_sep"
        elif contains_pad_before_sep:
            invalid_reason = "pad_before_sep"
        else:
            selfies_value = tokenizer.decode(
                interior, skip_special_tokens=False
            ).replace("] [", "][")
            try:
                smiles_value = sf.decoder(selfies_value)
            except Exception as error:
                invalid_reason = f"selfies_decode:{type(error).__name__}"
            if smiles_value is not None:
                molecule = Chem.MolFromSmiles(smiles_value)
                if molecule is None:
                    invalid_reason = "rdkit_parse"
                else:
                    rdkit_valid = True
                    canonical_smiles = Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    )
                    has_amide_bond = bool(
                        AMIDE_PATTERN is not None
                        and molecule.HasSubstructMatch(AMIDE_PATTERN)
                    )
    else:
        invalid_reason = "missing_sep_or_cls"

    return {
        "strain": strain,
        "seed": seed,
        "batch_index": batch_index,
        "sample_index": sample_index,
        "token_ids": token_ids,
        "first_sep_index": first_sep,
        "starts_with_cls": starts_with_cls,
        "complete": complete,
        "contains_mask_before_sep": contains_mask_before_sep,
        "contains_pad_before_sep": contains_pad_before_sep,
        "selfies": selfies_value,
        "smiles": smiles_value,
        "canonical_smiles": canonical_smiles,
        "rdkit_valid": rdkit_valid,
        "has_amide_bond": has_amide_bond,
        "invalid_reason": invalid_reason,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.t_off < args.t_on < 1:
        raise ValueError("Expected 0 < t_off < t_on < 1")
    if args.num_batches < 1 or args.batch_size < 1:
        raise ValueError("num-batches and batch-size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES before launching."
        )

    args.producer_root = args.producer_root.resolve()
    args.synergy_root = args.synergy_root.resolve()
    args.mdlm_root = args.mdlm_root.resolve()
    for optional_path in (
        "diffusion_checkpoint",
        "guidance_regressor_checkpoint",
        "peptide_classifier_checkpoint",
    ):
        value = getattr(args, optional_path)
        if value is not None:
            setattr(args, optional_path, value.resolve())
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    completed_path = args.output_dir / "completed.json"
    if completed_path.exists():
        if args.resume:
            return
        raise FileExistsError(f"Task already completed: {completed_path}")

    config = build_config(args)
    assets = validate_assets(config)
    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    config_path = args.output_dir / "resolved_config.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_yaml:
        raise RuntimeError("Existing resolved config does not match requested task")
    atomic_write_text(config_path, config_yaml)

    metadata = {
        "schema_version": 1,
        "created_unix": time.time(),
        "host": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_config_sha256": hashlib.sha256(
            config_yaml.encode("utf-8")
        ).hexdigest(),
        "assets": [
            {
                "path": str(path),
                "size": path.stat().st_size if path.is_file() else None,
            }
            for path in assets
        ],
    }
    atomic_write_text(
        args.output_dir / "task_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )

    L.seed_everything(args.seed, workers=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    log_path = args.output_dir / "sampler.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
            log_handle
        ):
            sampler, guidance_model = load_sampler(
                config, tokenizer, args.producer_root
            )
            for batch_index in range(args.num_batches):
                batch_path = batch_dir / f"batch_{batch_index:05d}.jsonl"
                if batch_path.exists():
                    if args.resume:
                        continue
                    raise FileExistsError(batch_path)
                started = time.time()
                samples = sampler._diffusion_sample(
                    classifier_model=guidance_model,
                    cond=None,
                    eps=1e-5,
                )
                rows = []
                for sample_index, sample in enumerate(samples.detach().cpu()):
                    rows.append(
                        interpret_sample(
                            [int(value) for value in sample.tolist()],
                            tokenizer=tokenizer,
                            strain=args.strain,
                            seed=args.seed,
                            batch_index=batch_index,
                            sample_index=sample_index,
                        )
                    )
                payload = "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                )
                atomic_write_text(batch_path, payload)
                print(
                    json.dumps(
                        {
                            "event": "batch_complete",
                            "batch_index": batch_index,
                            "rows": len(rows),
                            "seconds": time.time() - started,
                            "sha256": sha256_file(batch_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    batch_paths = sorted(batch_dir.glob("batch_*.jsonl"))
    if len(batch_paths) != args.num_batches:
        raise RuntimeError(
            f"Expected {args.num_batches} completed batches, found {len(batch_paths)}"
        )
    completed = {
        "completed_unix": time.time(),
        "num_batches": len(batch_paths),
        "attempted_samples": args.batch_size * len(batch_paths),
        "batch_files": [
            {
                "path": str(path.relative_to(args.output_dir)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in batch_paths
        ],
    }
    atomic_write_text(
        completed_path, json.dumps(completed, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
