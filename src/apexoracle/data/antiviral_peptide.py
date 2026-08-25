"""DRAVP antiviral-peptide potency parsing for the viral ApexOracle task.

DRAVP does not ship a numeric activity column. Potency lives inside a free-text
``Activity`` sentence, for example::

    [Ref.20950663]Hepatitis C virus (HCV): inhibition of viral infection in
    Huh7.5.1 cells(IC50 = 7.62 µg/ml (5.0 µM)).

This module turns that text into censored, unit-normalised potency records so
the antiviral task can reuse the MIC conventions rather than inventing new ones:

* range, ``>``/``>=``/``>>`` and ``<`` handling is delegated to
  :func:`apexoracle.data.amp_mic.parse_concentration`, so an AVP range collapses
  to its mean and a right-censored value gets the same 2x/3x paper multipliers
  the bacterial pipeline applies;
* mass units are converted to micromolar using the DRAVP ``Mass`` column, giving
  the same µM scale the MIC labels use;
* qualitative rows of the form ``>50% inhibition at 100 µg/ml`` are recovered as
  left-censored IC50 values, mirroring the ``>=95% inhibition`` fallback that
  :func:`apexoracle.data.amp_mic.select_measurement` already uses for MIC.

Nothing here applies the ``-log10(x/10)`` transform; that stays in the dataset
layer so the stored table keeps physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from .amp_mic import parse_concentration

METRICS = ("IC50", "EC50")

# Unit matching is case sensitive on purpose: a case-insensitive pattern cannot
# tell mM from µM once the prefix is folded.
_UNIT_TO_MICROMOLAR = {
    "nM": 1e-3,
    "uM": 1.0,
    "µM": 1.0,
    "μM": 1.0,
    "mM": 1e3,
    "M": 1e6,
}
_MASS_UNIT_TO_UG_PER_ML = {}
# DRAVP writes the volume as ml, mL and mL interchangeably, and uses both the
# micro sign and Greek mu. Enumerate the spellings rather than folding case,
# because case folding would also merge mM into µM.
for _prefix, _factor in (("ng", 1e-3), ("ug", 1.0), ("µg", 1.0), ("μg", 1.0), ("mg", 1e3)):
    for _volume in ("ml", "mL", "ML"):
        _MASS_UNIT_TO_UG_PER_ML[f"{_prefix}/{_volume}"] = _factor
for _prefix, _factor in (("ng", 1e-6), ("ug", 1e-3), ("µg", 1e-3), ("μg", 1e-3), ("mg", 1.0)):
    for _volume in ("l", "L"):
        _MASS_UNIT_TO_UG_PER_ML[f"{_prefix}/{_volume}"] = _factor

_UNIT_ALTERNATION = "|".join(
    sorted(
        (re.escape(unit) for unit in (*_UNIT_TO_MICROMOLAR, *_MASS_UNIT_TO_UG_PER_ML)),
        key=len,
        reverse=True,
    )
)

# "IC50 = 7.62 µg/ml", "EC50= 2.76 µM", "IC50 = 0.06-0.15 µg/ml", "IC50 > 100 µM"
_POTENCY_PATTERN = re.compile(
    r"\b(?P<metric>IC50|EC50)\b\s*[=:是]?\s*"
    r"(?P<value>[<>≥~]{0,2}\s*\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?(?:\s*±\s*\d+(?:\.\d+)?)?)"
    rf"\s*(?P<unit>{_UNIT_ALTERNATION})(?![A-Za-z/])",
    re.IGNORECASE if False else 0,
)

# ">50% inhibition at 100 µg/ml", "50 % inhibition at 12 µM"
_INHIBITION_PATTERN = re.compile(
    r"(?P<percent>[<>≥]?\s*\d+(?:\.\d+)?)\s*%\s*inhibition[^.;]{0,40}?\bat\b\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>" + _UNIT_ALTERNATION + r")(?![A-Za-z/])",
    re.IGNORECASE,
)


# "7.62 µg/ml (5.0 µM)": the mass reading followed by the molar reading in
# parentheses, which is DRAVP's own statement of the same measurement.
_MASS_UNIT_ALTERNATION = "|".join(
    sorted((re.escape(unit) for unit in _MASS_UNIT_TO_UG_PER_ML), key=len, reverse=True)
)
_MOLAR_UNIT_ALTERNATION = "|".join(
    sorted((re.escape(unit) for unit in _UNIT_TO_MICROMOLAR), key=len, reverse=True)
)
_DUAL_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + _MASS_UNIT_ALTERNATION + r")\s*"
    r"\(\s*(\d+(?:\.\d+)?)\s*(" + _MOLAR_UNIT_ALTERNATION + r")\s*\)"
)

_LEFT_CENSOR_MIN_PERCENT = 50.0


@dataclass(frozen=True)
class PotencyRecord:
    """One potency value recovered from a DRAVP ``Activity`` sentence."""

    metric: str
    value_micromolar: float
    censoring: str  # "none", "right" or "left"
    source_unit: str
    source_text: str
    from_inhibition: bool = False


def _censoring_of(raw_value: str) -> str:
    stripped = raw_value.replace(" ", "")
    if any(token in stripped for token in (">>", ">=", "≥", ">")):
        return "right"
    if "<" in stripped:
        return "left"
    return "none"


def _as_weight(molecular_weight: object) -> float | None:
    """Coerce the DRAVP ``Mass`` cell, which is a string in some rows."""

    if molecular_weight is None:
        return None
    try:
        weight = float(str(molecular_weight).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return weight if weight > 0 else None


def to_micromolar(value: float, unit: str, molecular_weight: object) -> float | None:
    """Convert one concentration to µM, returning ``None`` when it cannot be."""

    if unit in _UNIT_TO_MICROMOLAR:
        return value * _UNIT_TO_MICROMOLAR[unit]
    factor = _MASS_UNIT_TO_UG_PER_ML.get(unit)
    if factor is None:
        return None
    weight = _as_weight(molecular_weight)
    if weight is None:
        return None
    # c[µM] = c[µg/mL] * 1000 / MW[g/mol]
    return value * factor * 1000.0 / weight


def parse_activity(
    activity: str,
    molecular_weight: object,
    *,
    recover_inhibition: bool = True,
) -> list[PotencyRecord]:
    """Extract every potency value from one DRAVP ``Activity`` field."""

    if not activity:
        return []

    records: list[PotencyRecord] = []
    for match in _POTENCY_PATTERN.finditer(activity):
        raw_value = match.group("value")
        unit = match.group("unit")
        try:
            magnitude = parse_concentration(raw_value.replace(" ", ""))
        except (ValueError, IndexError):
            continue
        micromolar = to_micromolar(magnitude, unit, molecular_weight)
        if micromolar is None or micromolar <= 0:
            continue
        records.append(
            PotencyRecord(
                metric=match.group("metric").upper(),
                value_micromolar=micromolar,
                censoring=_censoring_of(raw_value),
                source_unit=unit,
                source_text=match.group(0).strip(),
            )
        )

    if records or not recover_inhibition:
        return records

    # No explicit IC50/EC50. A statement that the peptide inhibits at least half
    # of the virus at concentration C bounds IC50 from above: IC50 <= C.
    for match in _INHIBITION_PATTERN.finditer(activity):
        percent_text = match.group("percent").replace(" ", "").lstrip("<>≥")
        try:
            percent = float(percent_text)
        except ValueError:
            continue
        if percent < _LEFT_CENSOR_MIN_PERCENT:
            continue
        unit = match.group("unit")
        micromolar = to_micromolar(float(match.group("value")), unit, molecular_weight)
        if micromolar is None or micromolar <= 0:
            continue
        records.append(
            PotencyRecord(
                metric="IC50",
                value_micromolar=micromolar,
                censoring="left",
                source_unit=unit,
                source_text=match.group(0).strip(),
                from_inhibition=True,
            )
        )
    return records


def aggregate_metric(values: Sequence[float]) -> float:
    """Combine repeated measurements of one metric.

    Potency spans orders of magnitude and the training label is a log transform,
    so the geometric mean is the arithmetic mean of the labels themselves.
    """

    if not values:
        raise ValueError("no values to aggregate")
    product_log = sum(_log10(value) for value in values) / len(values)
    return 10.0**product_log


def _log10(value: float) -> float:
    from math import log10

    if value <= 0:
        raise ValueError(f"non-positive potency: {value}")
    return log10(value)


def cross_check_dual_unit(activity: str, molecular_weight: object) -> float | None:
    """Ratio between the molar and mass readings when DRAVP prints both.

    DRAVP often writes ``IC50 = 7.62 µg/ml (5.0 µM)``. Converting the mass value
    with the ``Mass`` column must reproduce the molar value, so the ratio is a
    free correctness check on the molecular weights.
    """

    pairs = _DUAL_UNIT_PATTERN.findall(activity)
    if len(pairs) != 1:
        return None
    mass_value, mass_unit, molar_value, molar_unit = pairs[0]
    converted = to_micromolar(float(mass_value), mass_unit, molecular_weight)
    stated = to_micromolar(float(molar_value), molar_unit, molecular_weight)
    if converted is None or not stated:
        return None
    return converted / stated
