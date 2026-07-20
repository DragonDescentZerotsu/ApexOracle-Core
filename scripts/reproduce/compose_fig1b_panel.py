#!/usr/bin/env python3
"""Replace Fig. 1b by overlaying a frozen panel on the pre-revision Fig. 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pypdf import PdfReader, PdfWriter, Transformation


def compose(
    *,
    base_figure: Path,
    panel: Path,
    output: Path,
    x: float,
    y: float,
    target_width: float,
) -> None:
    """Overlay the one-page panel at a fixed paper-figure coordinate."""

    base_reader = PdfReader(base_figure)
    panel_reader = PdfReader(panel)
    if len(base_reader.pages) != 1 or len(panel_reader.pages) != 1:
        raise ValueError("The base figure and replacement panel must be one-page PDFs")
    base_page = base_reader.pages[0]
    panel_page = panel_reader.pages[0]
    panel_width = float(panel_page.mediabox.width)
    if panel_width <= 0:
        raise ValueError("Replacement panel has a non-positive width")
    scale = target_width / panel_width
    base_page.merge_transformed_page(
        panel_page,
        Transformation().scale(scale).translate(x, y),
        over=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(base_page)
    with output.open("wb") as handle:
        writer.write(handle)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-figure", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", type=float, default=600.0)
    parser.add_argument("--y", type=float, default=1320.0)
    parser.add_argument("--target-width", type=float, default=935.0)
    args = parser.parse_args(argv)
    compose(
        base_figure=args.base_figure,
        panel=args.panel,
        output=args.output,
        x=args.x,
        y=args.y,
        target_width=args.target_width,
    )


if __name__ == "__main__":
    main()
