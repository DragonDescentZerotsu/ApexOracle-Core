"""Shared-data molecule encoder benchmark used for the revised Fig. 2b."""

from .apex_adapter import (
    build_apex_vocabulary,
    encode_apex_sequences,
    extend_aaindex_with_unknown,
)
from .data import SharedBenchmarkData, load_shared_benchmark, transform_mic_labels
from .metrics import finite_macro_mean, masked_r2_per_task
from .protocol import (
    DEFAULT_TARGET_COLUMNS,
    EXPECTED_ENCODERS,
    ApexProjection,
    assign_folds,
    build_shared_dataset,
    project_apex_sequence,
)

__all__ = [
    "DEFAULT_TARGET_COLUMNS",
    "EXPECTED_ENCODERS",
    "ApexProjection",
    "SharedBenchmarkData",
    "assign_folds",
    "build_apex_vocabulary",
    "build_shared_dataset",
    "encode_apex_sequences",
    "extend_aaindex_with_unknown",
    "finite_macro_mean",
    "load_shared_benchmark",
    "masked_r2_per_task",
    "project_apex_sequence",
    "transform_mic_labels",
]
