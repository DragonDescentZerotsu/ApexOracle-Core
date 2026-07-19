from __future__ import annotations

import csv
from pathlib import Path

import pytest

from apexoracle.evaluation.modality_ablation import (
    ModalityAblationPlotConfig,
    load_records,
    main,
    ordered_values,
    validate_complete_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "experiments/modality_ablation/paper_values.csv"
CONFIG = REPO_ROOT / "configs/modality_ablation/paper_plot.json"


def test_frozen_paper_values_match_final_notebook_cell() -> None:
    records = load_records(VALUES)
    config = ModalityAblationPlotConfig.load(CONFIG)

    assert config.holdout_order == (
        "phylum-wise",
        "species-wise",
        "strain-wise",
    )
    assert ordered_values(records, config) == (
        (0.2382, 0.3289, 0.4514),
        (0.2130, 0.3441, 0.4376),
        (0.2670, 0.3462, 0.5184),
        (0.2674, 0.4010, 0.4890),
    )


def test_value_grid_rejects_a_missing_paper_point() -> None:
    records = load_records(VALUES)
    config = ModalityAblationPlotConfig.load(CONFIG)

    with pytest.raises(ValueError, match="Incomplete value grid"):
        validate_complete_grid(records[:-1], config)


def test_loader_rejects_duplicate_holdout_series(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    with VALUES.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    with duplicate.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerows(rows + [rows[1]])

    with pytest.raises(ValueError, match="Duplicate holdout/series"):
        load_records(duplicate)


def test_cli_refuses_to_overwrite_frozen_values() -> None:
    with pytest.raises(ValueError, match="must not overwrite"):
        main(
            [
                "--values",
                str(VALUES),
                "--config",
                str(CONFIG),
                "--output-pdf",
                str(VALUES),
            ]
        )


def test_cli_writes_pdf_and_png_without_touching_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    values_before = VALUES.read_bytes()
    config_before = CONFIG.read_bytes()
    pdf = tmp_path / "modality-ablation.pdf"
    png = tmp_path / "modality-ablation.png"
    monkeypatch.setenv("MPLBACKEND", "Agg")

    main(
        [
            "--values",
            str(VALUES),
            "--config",
            str(CONFIG),
            "--output-pdf",
            str(pdf),
            "--output-png",
            str(png),
        ]
    )

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert VALUES.read_bytes() == values_before
    assert CONFIG.read_bytes() == config_before
