"""Reproducible strain-description embeddings for ApexOracle consumers.

The paper-era producers used two filename encodings but shared one model contract:
replace the exact strain name with ``This strain`` and save the penultimate hidden
state from ``YBXL/Med-LLaMA3-8B`` as a token-by-feature float32 tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch


DEFAULT_MODEL_ID = "YBXL/Med-LLaMA3-8B"
DEFAULT_MODEL_REVISION = "567e7e71d8b6b433d8bc494f8112176bec4afccf"
DEFAULT_HIDDEN_STATE_INDEX = -2

FilenameEncoding = Literal["auto", "atcc", "text-only"]
ExistingPolicy = Literal["error", "skip", "overwrite"]


@dataclass(frozen=True)
class PreparedStrainText:
    """A source description after the historical strain-name replacement."""

    source_path: Path
    stem: str
    decoded_strain_name: str
    text: str
    replacement_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_strain_name(stem: str, encoding: FilenameEncoding = "auto") -> str:
    """Decode one paper-era strain filename stem into its displayed name."""

    if encoding not in {"auto", "atcc", "text-only"}:
        raise ValueError(f"Unsupported filename encoding: {encoding}")
    selected = encoding
    if selected == "auto":
        selected = "text-only" if ("～" in stem or "^" in stem) else "atcc"
    if selected == "text-only":
        return stem.replace("～", " ").replace("^", "/")
    return stem.replace("_", " ")


def replace_strain_name(
    text: str,
    stem: str,
    encoding: FilenameEncoding = "auto",
) -> tuple[str, str, int]:
    """Apply the exact two-pass replacement used by both historical producers."""

    decoded_name = decode_strain_name(stem, encoding)
    dotted_subspecies_name = decoded_name.replace("subsp", "subsp.")
    replacements = text.count(dotted_subspecies_name)
    transformed = text.replace(dotted_subspecies_name, "This strain")
    second_count = transformed.count(decoded_name)
    transformed = transformed.replace(decoded_name, "This strain")
    return transformed, decoded_name, replacements + second_count


def prepare_strain_text(
    source_path: Path,
    encoding: FilenameEncoding = "auto",
) -> PreparedStrainText:
    """Read and normalize one UTF-8 ``.txt`` strain description."""

    source_path = Path(source_path)
    if source_path.suffix != ".txt":
        raise ValueError(f"Expected a .txt source file: {source_path}")
    original = source_path.read_text(encoding="utf-8")
    transformed, decoded_name, replacement_count = replace_strain_name(
        original, source_path.stem, encoding
    )
    return PreparedStrainText(
        source_path=source_path,
        stem=source_path.stem,
        decoded_strain_name=decoded_name,
        text=transformed,
        replacement_count=replacement_count,
    )


def discover_text_sources(input_dir: Path) -> list[Path]:
    """Return deterministic top-level ``.txt`` inputs from ``input_dir``."""

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    sources = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".txt"
    )
    if not sources:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    return sources


def embed_prepared_text(
    prepared: PreparedStrainText,
    tokenizer: Any,
    model: Any,
    device: torch.device | str,
    hidden_state_index: int = DEFAULT_HIDDEN_STATE_INDEX,
) -> torch.Tensor:
    """Encode one prepared description under the frozen token-level contract."""

    tokenized = tokenizer(prepared.text, return_tensors="pt")
    input_ids = (
        tokenized["input_ids"]
        if isinstance(tokenized, Mapping)
        else tokenized.input_ids
    ).to(device)
    with torch.inference_mode():
        result = model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states: Sequence[torch.Tensor] = result.hidden_states
    try:
        embedding = hidden_states[hidden_state_index]
    except IndexError as error:
        raise ValueError(
            f"Model returned {len(hidden_states)} hidden states; "
            f"index {hidden_state_index} is invalid"
        ) from error
    if embedding.ndim != 3 or embedding.shape[0] != 1:
        raise ValueError(
            "Expected hidden state shape [1, tokens, width], got "
            f"{tuple(embedding.shape)}"
        )
    return embedding.squeeze(0).detach().cpu().float().contiguous()


def _existing_tensor_record(output_path: Path) -> dict[str, Any]:
    tensor = torch.load(output_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Existing output must be a rank-2 tensor: {output_path}")
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _atomic_torch_save(tensor: torch.Tensor, output_path: Path) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def embed_strain_text_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    tokenizer: Any,
    model: Any,
    model_id: str = DEFAULT_MODEL_ID,
    requested_revision: str = DEFAULT_MODEL_REVISION,
    resolved_revision: str | None = None,
    device: torch.device | str = "cpu",
    filename_encoding: FilenameEncoding = "auto",
    hidden_state_index: int = DEFAULT_HIDDEN_STATE_INDEX,
    existing_policy: ExistingPolicy = "skip",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Embed a directory and atomically write tensors plus a provenance manifest."""

    if existing_policy not in {"error", "skip", "overwrite"}:
        raise ValueError(f"Unsupported existing policy: {existing_policy}")
    sources = discover_text_sources(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = output_dir / "strain_text_embedding_manifest.json"
    else:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    existing = [output_dir / f"{source.stem}.pt" for source in sources]
    existing = [path for path in existing if path.exists()]
    if existing_policy == "error" and existing:
        raise FileExistsError(
            "Output tensors already exist: " + ", ".join(path.name for path in existing)
        )

    model.to(device)
    model.eval()
    records: list[dict[str, Any]] = []
    for source_path in sources:
        prepared = prepare_strain_text(source_path, filename_encoding)
        output_path = output_dir / f"{prepared.stem}.pt"
        source_record = {
            "path": source_path.name,
            "sha256": sha256_file(source_path),
            "decoded_strain_name": prepared.decoded_strain_name,
            "replacement_count": prepared.replacement_count,
            "normalized_text_sha256": hashlib.sha256(
                prepared.text.encode("utf-8")
            ).hexdigest(),
            "short_text_warning": len(prepared.text) <= 4,
        }
        if output_path.exists() and existing_policy == "skip":
            records.append(
                {
                    "source": source_record,
                    "status": "skipped-existing",
                    "output": _existing_tensor_record(output_path),
                }
            )
            continue
        embedding = embed_prepared_text(
            prepared,
            tokenizer,
            model,
            device,
            hidden_state_index,
        )
        _atomic_torch_save(embedding, output_path)
        records.append(
            {
                "source": source_record,
                "status": "written",
                "output": _existing_tensor_record(output_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "filename_encoding": filename_encoding,
            "replacement": "exact decoded strain name -> This strain",
            "hidden_state_index": hidden_state_index,
            "saved_dtype": "torch.float32",
            "tensor_layout": "tokens_by_features",
        },
        "model": {
            "id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision or requested_revision,
        },
        "device": str(device),
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "records": records,
        "summary": {
            "sources": len(records),
            "written": sum(record["status"] == "written" for record in records),
            "skipped_existing": sum(
                record["status"] == "skipped-existing" for record in records
            ),
            "without_name_replacement": sum(
                record["source"]["replacement_count"] == 0 for record in records
            ),
            "short_text_warnings": sum(
                record["source"]["short_text_warning"] for record in records
            ),
        },
    }
    _atomic_json_dump(manifest, manifest_path)
    return manifest
