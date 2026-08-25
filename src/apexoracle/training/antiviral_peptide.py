"""Two-head antiviral-peptide potency regression.

DRAVP reports IC50 for some peptides and EC50 for others, and almost never both:
of the 1,585 single-target training rows, 1,239 carry an IC50, 365 carry an EC50
and only 19 carry both. A two-head model therefore sees a label matrix that is
about half empty, and the emptiness is structural rather than incidental.

That makes masking mandatory rather than a refinement. The MIC pipeline predicts
one target per row and every row has it, so ``nn.MSELoss`` over the whole tensor
is correct there and no mask exists in that code path. Reusing it here would
either propagate NaN into every gradient, or, if the gaps were filled with zero,
teach the EC50 head that every IC50-only peptide has an EC50 of exactly 10 µM,
because the label transform sends 10 µM to 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from ..data.antiviral_peptide import METRICS

# Same transform the hierarchical MIC pipeline applies, so both tasks share a
# scale: -log10(x / 10) with x in micromolar.
MIC_TRANSFORM_DIVISOR = 10.0


def potency_to_label(values: torch.Tensor) -> torch.Tensor:
    """Apply the shared ``-log10(x/10)`` potency transform, preserving NaN."""

    return -torch.log10(values / MIC_TRANSFORM_DIVISOR)


def label_to_potency(labels: torch.Tensor) -> torch.Tensor:
    """Invert :func:`potency_to_label` back to micromolar."""

    return torch.pow(10.0, -labels) * MIC_TRANSFORM_DIVISOR


def build_label_matrix(frame: pd.DataFrame) -> torch.Tensor:
    """Return the ``[n_rows, len(METRICS)]`` label matrix with NaN for gaps.

    NaN is the mask carrier. Do not fill it: zero is a real label value here.
    """

    columns = [f"{metric}_uM" for metric in METRICS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing potency columns: {missing}")
    values = torch.tensor(frame[columns].to_numpy(dtype="float64"), dtype=torch.float32)
    return potency_to_label(values)


@dataclass(frozen=True)
class MaskedLossResult:
    loss: torch.Tensor
    observed: torch.Tensor
    per_head_observed: torch.Tensor


def masked_mse_loss(logits: torch.Tensor, labels: torch.Tensor) -> MaskedLossResult:
    """Mean squared error over observed labels only.

    The denominator is the number of observed entries, not ``numel``. Dividing by
    ``numel`` would halve every real gradient because half the matrix is empty.
    """

    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} labels {tuple(labels.shape)}")
    mask = torch.isfinite(labels)
    observed = mask.sum()
    if observed == 0:
        # An all-empty batch contributes nothing, but must stay connected to the
        # graph so distributed training does not desynchronise.
        return MaskedLossResult(
            loss=(logits * 0.0).sum(),
            observed=observed,
            per_head_observed=mask.sum(dim=0),
        )
    squared = (logits - torch.nan_to_num(labels)).pow(2) * mask
    return MaskedLossResult(
        loss=squared.sum() / observed,
        observed=observed,
        per_head_observed=mask.sum(dim=0),
    )


def per_head_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Report R^2 per head over observed labels, for checkpoint selection."""

    results: dict[str, float] = {}
    mask = torch.isfinite(labels)
    for index, metric in enumerate(METRICS):
        column = mask[:, index]
        count = int(column.sum())
        results[f"{metric}_n"] = float(count)
        if count < 2:
            results[f"{metric}_r2"] = float("nan")
            continue
        target = labels[column, index]
        predicted = logits[column, index]
        residual = torch.sum((target - predicted) ** 2)
        total = torch.sum((target - target.mean()) ** 2)
        results[f"{metric}_r2"] = float("nan") if total == 0 else float(1.0 - residual / total)
    return results
