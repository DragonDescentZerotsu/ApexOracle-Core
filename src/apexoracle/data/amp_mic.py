"""Read-only reconstruction of the paper AMP MIC table.

The ``paper_legacy`` protocol intentionally preserves the decisions made by
``DataPrepare/concentration_unit_transfer_Evo.py``.  In particular, inhibition
records use the unit of the last retained inhibition measurement.  That is a
historical implementation detail, not a recommended rule for new datasets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PAPER_PROTOCOL = "paper_legacy"
MIC_COLUMNS = ("DBAASP_id", "strain_name", "SMILES", "MIC")
_NUMBER_PATTERN = re.compile(
    r"^(>|>=|<|<=)?\d+(\.\d+)?(±\d+(\.\d+)?)?(\-\d+(\.\d+)?)?$"
)


@dataclass(frozen=True)
class SelectedMeasurement:
    concentration: str
    convert_micrograms_per_ml: bool


@dataclass(frozen=True)
class MicBuildResult:
    table: pd.DataFrame
    strain_counts: Mapping[str, int]
    unusual_concentrations: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_simple_concentration(value: str) -> bool:
    return bool(_NUMBER_PATTERN.fullmatch(value))


def parse_inhibition_percentage(value_name: str) -> float:
    """Parse the inhibition percentage exactly as the paper-era converter."""

    parsed = value_name
    if "%" not in parsed:
        parsed = parsed.split(" ")[0] + "%"
    percentage = parsed.split("%")[0]
    if "(" in percentage:
        percentage = parsed.split(")")[0].split("(")[-1]
    if "±" in percentage:
        percentage = percentage.split("±")[0]
    if ">=" in percentage:
        percentage = percentage.split(">=")[-1]
    if "<" in percentage:
        percentage = percentage.split("<")[0 if percentage.endswith("<") else -1]
    if ">" in percentage:
        percentage = percentage.split(">")[0 if percentage.endswith(">") else -1]
    if "e" in percentage:
        percentage = str(float(percentage.split("±")[0]))
    if "-" in percentage:
        parts = [float(part) for part in percentage.split("-")]
        return sum(parts) / len(parts)
    return float(percentage)


def select_measurement(
    measurements: Mapping[tuple[str, str], str],
    *,
    protocol: str = PAPER_PROTOCOL,
) -> SelectedMeasurement | None:
    """Select a strain MIC or >=95% inhibition measurement.

    Dictionary insertion order is part of the frozen ``paper_legacy``
    contract.  Duplicate measurement/unit pairs have already been discarded.
    """

    if protocol != PAPER_PROTOCOL:
        raise ValueError(f"Unsupported MIC protocol: {protocol}")
    if not measurements:
        return None
    if ("MIC", "µM") in measurements:
        return SelectedMeasurement(measurements[("MIC", "µM")].strip(), False)
    if ("MIC", "µg/ml") in measurements:
        return SelectedMeasurement(measurements[("MIC", "µg/ml")].strip(), True)

    maximum = 0.0
    maximum_key: tuple[str, str] | None = None
    last_unit: str | None = None
    for key in measurements:
        value_name, last_unit = key
        percentage = parse_inhibition_percentage(value_name)
        if percentage > maximum:
            maximum = percentage
            maximum_key = key
    if maximum < 95 or maximum_key is None:
        return None
    if last_unit not in {"µg/ml", "µM"}:
        raise ValueError(f"Unsupported paper-era inhibition unit: {last_unit}")

    # Historical behavior: the converter inspected the final iterated unit,
    # rather than maximum_key[1].  Preserve it for paper data reconstruction.
    return SelectedMeasurement(
        measurements[maximum_key].strip(), last_unit == "µg/ml"
    )


def parse_concentration(value: str) -> float:
    """Parse one concentration and apply the paper-era censor multipliers."""

    if not value:
        raise ValueError("Empty concentration")
    raw = value
    parsed = value
    for separator in (" - =>", "->=", " - >="):
        if separator in parsed:
            parts = [float(part) for part in parsed.split(separator)]
            parsed = str(sum(parts) / len(parts))
    if " " in parsed.strip():
        parsed = parsed.replace(" ", "")
    if "->" in parsed:
        parts = [float(part) for part in parsed.split("->")]
        parsed = str(sum(parts) / len(parts))
    for separator in ("≥", ">=", "<=", ">>", ">"):
        if separator in parsed:
            parsed = parsed.split(separator)[-1]
    if "<" in parsed:
        parsed = parsed.split("<")[0 if parsed.endswith("<") else -1]
    if "±" in parsed:
        parsed = parsed.split("±")[0]
    if "," in parsed:
        parsed = parsed.replace(",", ".")
    for separator in ("-", "–"):
        if separator in parsed:
            parts = [float(part) for part in parsed.split(separator)]
            parsed = str(sum(parts) / len(parts))
    if len(parsed.split(".")) > 2:
        parsed = ".".join(parsed.split(".")[:2])

    concentration = float(parsed)
    if (">" in raw or ">=" in raw) and all(
        separator not in raw for separator in ("->", " - =>", ">>")
    ):
        concentration *= 2
    if ">>" in raw:
        concentration *= 3
    return concentration


def collect_strain_measurements(
    target_activities: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[tuple[str, str], str]]:
    """Collect the first accepted value for every strain/measure/unit key."""

    collected: dict[str, dict[tuple[str, str], str]] = {}
    for activity in target_activities or ():
        species = activity.get("targetSpecies")
        if species is None:
            continue
        strain = species["name"]
        strain_values = collected.setdefault(strain, {})
        unit = activity.get("unit")
        if unit is None:
            continue
        measure = activity["activityMeasureValue"]
        key = (measure, unit["name"])
        accepted = (
            measure == "MIC"
            or "inhibition" in measure
            or "inhibiton" in measure
            or "Inhibition" in measure
        )
        if accepted and key not in strain_values:
            strain_values[key] = activity["concentration"]
    return collected


def _first_smiles_by_id(smiles_table: pd.DataFrame) -> dict[str, str]:
    missing = {"DBAASP_id", "SMILES"}.difference(smiles_table.columns)
    if missing:
        raise ValueError(f"SMILES table is missing columns: {sorted(missing)}")
    result: dict[str, str] = {}
    for dbaasp_id, smiles in smiles_table[["DBAASP_id", "SMILES"]].itertuples(
        index=False, name=None
    ):
        result.setdefault(str(dbaasp_id), str(smiles))
    return result


def build_paper_mic_table(
    records: Iterable[Mapping[str, Any]],
    smiles_table: pd.DataFrame,
    *,
    molecular_weight_smiles_overrides: pd.DataFrame | None = None,
    protocol: str = PAPER_PROTOCOL,
) -> MicBuildResult:
    """Build the complete all-strain paper MIC table without mutating inputs."""

    if protocol != PAPER_PROTOCOL:
        raise ValueError(f"Unsupported MIC protocol: {protocol}")
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("AMP MIC reconstruction requires RDKit") from exc

    smiles_by_id = _first_smiles_by_id(smiles_table)
    weight_smiles_by_id = (
        _first_smiles_by_id(molecular_weight_smiles_overrides)
        if molecular_weight_smiles_overrides is not None
        else {}
    )
    molecular_weights: dict[str, float] = {}
    rows: list[tuple[str, str, str, str]] = []
    counts: Counter[str] = Counter()
    unusual: list[str] = []

    for record in records:
        dbaasp_id = str(record["id"])
        smiles = smiles_by_id.get(dbaasp_id)
        if smiles is None:
            continue
        measurements_by_strain = collect_strain_measurements(
            record.get("targetActivities")
        )
        for strain, measurements in measurements_by_strain.items():
            selected = select_measurement(measurements, protocol=protocol)
            if selected is None or not selected.concentration:
                continue
            raw = selected.concentration
            if not is_simple_concentration(raw):
                unusual.append(raw)
            concentration = parse_concentration(raw)
            if selected.convert_micrograms_per_ml:
                molecular_weight = molecular_weights.get(dbaasp_id)
                if molecular_weight is None:
                    # The frozen table was converted before the later structure
                    # correction replaced 179 displayed SMILES in place.  The
                    # optional overrides make that two-step lineage explicit.
                    weight_smiles = weight_smiles_by_id.get(dbaasp_id, smiles)
                    molecule = Chem.MolFromSmiles(weight_smiles)
                    if molecule is None:
                        raise ValueError(
                            f"Invalid molecular-weight SMILES for DBAASP {dbaasp_id}"
                        )
                    molecular_weight = rdMolDescriptors.CalcExactMolWt(molecule)
                    molecular_weights[dbaasp_id] = molecular_weight
                concentration = concentration / molecular_weight * 1000

            counts[strain] += 1
            # The first converter stored mixed NumPy arrays, which rendered MIC
            # as text.  Structure correction then re-read that CSV as float and
            # wrote it again.  This in-memory round trip preserves the frozen
            # numeric representation without creating or overwriting an
            # intermediate data file.
            legacy_mic = np.array(
                (dbaasp_id, strain, smiles, concentration)
            ).tolist()[3]
            rows.append((dbaasp_id, strain, smiles, legacy_mic))

    table = pd.DataFrame(rows, columns=MIC_COLUMNS)
    # ``read_csv`` and ``to_numeric`` do not make identical last-bit choices.
    # Use an in-memory CSV to reproduce the actual correction script's parse.
    legacy_csv = io.StringIO()
    table.to_csv(legacy_csv, index=False)
    legacy_csv.seek(0)
    table = pd.read_csv(legacy_csv)
    ordered_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    return MicBuildResult(table, ordered_counts, tuple(unusual))


def load_paper_mic_inputs(
    dbaasp_json: str | Path, smiles_csv: str | Path
) -> tuple[list[Mapping[str, Any]], pd.DataFrame]:
    with Path(dbaasp_json).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return records, pd.read_csv(smiles_csv)
