"""Legacy-compatible loaders for precomputed genome and text embeddings.

These functions intentionally preserve the filename parsing, eager device transfer,
dtype, and scaling behavior of the paper-era strain-wise scripts.
"""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from tqdm import tqdm


def get_embedded_genome_ids(folder_path: Path) -> Tuple[list[str], Dict[str, str]]:
    """Return stored strain IDs and their species-name prefixes.

    This is a behavior-preserving extraction of
    ``get_embedded_genome_IDs`` from the final strain-wise legacy script.
    Files are intentionally consumed in ``Path.iterdir()`` order.
    """

    stored_genome_ids: list[str] = []
    genome_id_to_species_first_name: Dict[str, str] = {}
    files = [path.name for path in folder_path.iterdir() if path.is_file()]
    for file_name in files:
        stem = file_name.split(".")[0]
        file_name_temp = stem.split("ATCC")[-1]
        components = file_name_temp.split("_")[1:]
        if len(components) == 2:
            strain_id = "-".join(components)
        else:
            strain_id = components[0]
        stored_genome_ids.append(strain_id)
        genome_id_to_species_first_name[strain_id] = stem.split("_")[0]

    return stored_genome_ids, genome_id_to_species_first_name


def _embedding_files(folder_path: Path) -> Iterable[Path]:
    # Do not sort: paper-era scripts inherited the filesystem iteration order.
    return [folder_path / path.name for path in folder_path.iterdir() if path.is_file()]


def load_all_embeddings(
    embeddings_folder_path: Path,
    scale: float,
    device: torch.device,
    desc_str: str,
) -> Dict[str, torch.Tensor]:
    """Eagerly load ATCC-style embeddings onto ``device`` and apply ``scale``."""

    embeddings: Dict[str, torch.Tensor] = {}
    for file_path in tqdm(
        _embedding_files(embeddings_folder_path),
        desc=f" loading {desc_str} embeddings ... ",
    ):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split(".")[0]
        if "ATCC" in file_name:
            file_name = file_name.split("ATCC")[-1]
            components = file_name.split("_")[1:]
            if len(components) == 2:
                strain_id = "-".join(components)
            else:
                strain_id = components[0]
        else:
            strain_id = file_name
        embeddings[strain_id] = embedding * scale
    return embeddings


def load_text_only_embeddings(
    embeddings_folder_path: Path,
    scale: float,
    device: torch.device,
    desc_str: str,
) -> Dict[str, torch.Tensor]:
    """Load text embeddings whose filenames encode the original strain name."""

    embeddings: Dict[str, torch.Tensor] = {}
    for file_path in tqdm(
        _embedding_files(embeddings_folder_path),
        desc=f" loading {desc_str} embeddings ... ",
    ):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split(".pt")[0]
        strain_name = file_name.replace("～", " ").replace("^", "/")
        embeddings[strain_name] = embedding * scale
    return embeddings


# Paper-era public names retained for thin compatibility wrappers.
get_embedded_genome_IDs = get_embedded_genome_ids
load_all_genome_embeddings = load_all_embeddings
load_text_wo_genome_embeddings = load_text_only_embeddings
