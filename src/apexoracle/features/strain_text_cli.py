"""Generate frozen Med-LLaMA3 strain-text embeddings and a hash manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from apexoracle.features.strain_text import (
    DEFAULT_HIDDEN_STATE_INDEX,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    embed_strain_text_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--filename-encoding",
        choices=("auto", "atcc", "text-only"),
        default="auto",
    )
    parser.add_argument(
        "--hidden-state-index", type=int, default=DEFAULT_HIDDEN_STATE_INDEX
    )
    parser.add_argument(
        "--existing",
        choices=("error", "skip", "overwrite"),
        default="skip",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {requested}"
        )
    return device


def resolve_model_source(
    model_id: str,
    revision: str,
    *,
    local_files_only: bool,
    cache_roots: Sequence[Path] | None = None,
) -> str:
    """Resolve a pinned local snapshot even when legacy cache env vars conflict."""

    explicit = Path(model_id).expanduser()
    if explicit.exists():
        return str(explicit.resolve())
    if not local_files_only:
        return model_id
    if cache_roots is None:
        roots: list[Path] = []
        if os.environ.get("HF_HUB_CACHE"):
            roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
        if os.environ.get("TRANSFORMERS_CACHE"):
            roots.append(Path(os.environ["TRANSFORMERS_CACHE"]).expanduser())
        if os.environ.get("HF_HOME"):
            roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    else:
        roots = [Path(root) for root in cache_roots]

    repository_name = "models--" + model_id.replace("/", "--")
    for root in dict.fromkeys(root.resolve() for root in roots):
        repository = root / repository_name
        candidates = [repository / "snapshots" / revision]
        reference = repository / "refs" / revision
        if reference.is_file():
            candidates.append(
                repository / "snapshots" / reference.read_text(encoding="utf-8").strip()
            )
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                return str(candidate.resolve())
    raise FileNotFoundError(
        f"Pinned local snapshot not found for {model_id}@{revision}. "
        "Populate the Hugging Face cache or omit --local-files-only."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install ApexOracle with the data-preparation extra to embed strain text"
        ) from error

    device = resolve_device(args.device)
    model_source = resolve_model_source(
        args.model_id,
        args.revision,
        local_files_only=args.local_files_only,
    )
    source_is_model_id = model_source == args.model_id
    source_kwargs = {
        "local_files_only": args.local_files_only,
        **({"revision": args.revision} if source_is_model_id else {}),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **source_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **source_kwargs,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    manifest = embed_strain_text_directory(
        args.input_dir,
        args.output_dir,
        tokenizer=tokenizer,
        model=model,
        model_id=args.model_id,
        requested_revision=args.revision,
        resolved_revision=resolved_revision,
        device=device,
        filename_encoding=args.filename_encoding,
        hidden_state_index=args.hidden_state_index,
        existing_policy=args.existing,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
