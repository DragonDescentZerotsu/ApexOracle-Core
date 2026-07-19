"""Thin integration boundary for the independently released PepLink package.

PepLink is intentionally not vendored into ApexOracle.  This module keeps the
optional dependency error explicit and adapts a DBAASP record to PepLink's
public API without duplicating peptide-chemistry implementation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


PEPLINK_PACKAGE = "PepLink"
PEPLINK_VERSION = "0.1.1"
PEPLINK_GIT_COMMIT = "cec2a02427766e4ba95806924801af31bdcc9939"
PEPLINK_REPOSITORY = "https://github.com/DragonDescentZerotsu/PepLink"


class PepLinkDependencyError(ImportError):
    """Raised when the optional, versioned PepLink dependency is unavailable."""


def load_peplink() -> Any:
    try:
        return import_module(PEPLINK_PACKAGE)
    except ImportError as exc:
        raise PepLinkDependencyError(
            "Peptide structure construction requires the independent "
            f"{PEPLINK_PACKAGE}=={PEPLINK_VERSION} package. Install ApexOracle "
            "with its 'peplink' extra or install PepLink directly."
        ) from exc


def dbaasp_record_to_smiles(record: Mapping[str, Any]) -> str:
    """Convert one supported DBAASP record using PepLink's public API."""

    peplink = load_peplink()
    peptide_input = peplink.from_dbaasp_record(record)
    return peplink.aa_seqs_to_smiles(**peptide_input.to_api_kwargs())
