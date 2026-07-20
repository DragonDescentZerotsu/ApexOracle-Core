#!/usr/bin/env python3
"""Create a non-destructive manifest and source archive for a legacy checkout.

The legacy tree is only read.  All generated files must live outside that tree.
Large experiment assets are inventoried by path, size, and modification time but
are deliberately not hashed or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


SOURCE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {".gitattributes", ".gitignore", "Dockerfile", "LICENSE", "Makefile"}
DEFAULT_EXCLUDED_PREFIXES = {
    ".git",
    ".git-state",
    "Checkpoints",
    "DataPrepare/Data",
    "capsule",
    "capsule_fig2",
    "results",
    "wandb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=32 * 1024 * 1024,
        help="Do not hash/archive source-looking files larger than this limit.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Additional legacy-root-relative prefix excluded from the source archive.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Generate manifests without the compressed source archive.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        for name in file_names:
            path = base / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            yield path, stat


def has_excluded_prefix(relative: Path, excluded: set[str]) -> bool:
    text = relative.as_posix()
    return any(text == prefix or text.startswith(f"{prefix}/") for prefix in excluded)


def is_source_file(
    relative: Path,
    stat: os.stat_result,
    excluded: set[str],
    max_source_bytes: int,
) -> bool:
    if has_excluded_prefix(relative, excluded):
        return False
    if not relative.name or relative.name.startswith(".") and relative.name not in SOURCE_NAMES:
        return False
    if stat.st_size > max_source_bytes:
        return False
    return relative.name in SOURCE_NAMES or relative.suffix.lower() in SOURCE_SUFFIXES


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    legacy_root = args.legacy_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if is_relative_to(output_dir, legacy_root):
        raise SystemExit("output_dir must be outside legacy_root")

    output_dir.mkdir(parents=True, exist_ok=False)
    excluded = DEFAULT_EXCLUDED_PREFIXES | {
        Path(item).as_posix().strip("/") for item in args.exclude_prefix
    }

    source_rows: list[list[object]] = []
    asset_rows: list[list[object]] = []
    source_paths: list[Path] = []
    errors: list[dict[str, str]] = []
    top_level = defaultdict(lambda: {"files": 0, "bytes": 0})

    for path, stat in iter_files(legacy_root):
        relative = path.relative_to(legacy_root)
        bucket = relative.parts[0] if relative.parts else "."
        top_level[bucket]["files"] += 1
        top_level[bucket]["bytes"] += stat.st_size
        file_type = "symlink" if path.is_symlink() else "file"

        if file_type == "file" and is_source_file(
            relative, stat, excluded, args.max_source_bytes
        ):
            try:
                digest = sha256(path)
            except OSError as error:
                errors.append({"path": relative.as_posix(), "error": str(error)})
                continue
            source_rows.append(
                [digest, stat.st_size, stat.st_mtime_ns, relative.as_posix()]
            )
            source_paths.append(relative)
        else:
            link_target = os.readlink(path) if file_type == "symlink" else ""
            asset_rows.append(
                [file_type, stat.st_size, stat.st_mtime_ns, relative.as_posix(), link_target]
            )

    source_rows.sort(key=lambda row: str(row[-1]))
    asset_rows.sort(key=lambda row: str(row[-2]))
    source_paths.sort(key=lambda path: path.as_posix())

    source_manifest = output_dir / "source_files.sha256.tsv"
    asset_manifest = output_dir / "asset_inventory.tsv"
    summary_manifest = output_dir / "top_level_summary.tsv"
    write_tsv(
        source_manifest,
        ["sha256", "size_bytes", "mtime_ns", "relative_path"],
        source_rows,
    )
    write_tsv(
        asset_manifest,
        ["type", "size_bytes", "mtime_ns", "relative_path", "link_target"],
        asset_rows,
    )
    write_tsv(
        summary_manifest,
        ["top_level_entry", "file_count", "size_bytes"],
        [
            [name, values["files"], values["bytes"]]
            for name, values in sorted(top_level.items())
        ],
    )

    archive_path: Path | None = None
    if not args.no_archive:
        archive_path = output_dir / "legacy_source.tar.gz"
        with tarfile.open(archive_path, mode="w:gz", dereference=False) as archive:
            for relative in source_paths:
                archive.add(
                    legacy_root / relative,
                    arcname=relative.as_posix(),
                    recursive=False,
                )

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "legacy_root": str(legacy_root),
        "legacy_root_stat": {
            "device": legacy_root.stat().st_dev,
            "inode": legacy_root.stat().st_ino,
            "mtime_ns": legacy_root.stat().st_mtime_ns,
        },
        "source_selection": {
            "suffixes": sorted(SOURCE_SUFFIXES),
            "names": sorted(SOURCE_NAMES),
            "excluded_prefixes": sorted(excluded),
            "max_source_bytes": args.max_source_bytes,
        },
        "counts": {
            "source_files": len(source_rows),
            "asset_entries": len(asset_rows),
            "scan_errors": len(errors),
        },
        "bytes": {
            "source_files": sum(int(row[1]) for row in source_rows),
            "asset_entries": sum(int(row[1]) for row in asset_rows),
        },
        "generated_files": {
            source_manifest.name: sha256(source_manifest),
            asset_manifest.name: sha256(asset_manifest),
            summary_manifest.name: sha256(summary_manifest),
        },
        "errors": errors,
    }
    if archive_path is not None:
        metadata["generated_files"][archive_path.name] = sha256(archive_path)

    with (output_dir / "snapshot_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
