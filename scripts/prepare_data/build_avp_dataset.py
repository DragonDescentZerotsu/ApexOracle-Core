#!/usr/bin/env python3
"""Build the antiviral-peptide regression dataset for the viral ApexOracle task.

Joins three things:

* DRAVP antiviral peptides, whose potency is parsed out of the free-text
  ``Activity`` field by :mod:`apexoracle.data.antiviral_peptide`;
* the canonical genome chosen for each DRAVP target;
* the genome/description file stem the encoder assets are keyed by.

Only single-target rows enter the training table. A DRAVP row that names several
viruses reports its potency values in one prose field without saying which value
belongs to which virus, so attributing them would be guesswork. Those rows are
written to a separate file so sentence-level attribution can be added later
without re-deriving anything.

The emitted labels stay in micromolar. The ``-log10(x/10)`` transform is applied
in the dataset layer, matching the MIC pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apexoracle.data.antiviral_peptide import (  # noqa: E402
    METRICS,
    aggregate_metric,
    parse_activity,
)

DEFAULT_DATA = REPO_ROOT / "DataPrepare/Data/DRAVP_2"
DEFAULT_OUTPUT = DEFAULT_DATA / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dravp", type=Path, default=DEFAULT_DATA / "raw/antiviral_peptides.xlsx")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_DATA / "processed/avp_canonical_pairs.tsv.gz")
    parser.add_argument(
        "--target-to-genome",
        type=Path,
        default=REPO_ROOT / "DataPrepare/Data/virus_handoff_manifests/target_to_genome.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dravp = pd.read_excel(args.dravp)
    pairs = pd.read_csv(args.pairs, sep="\t")
    stems = pd.read_csv(args.target_to_genome, sep="\t")[["target_id", "file_stem", "display_name"]]

    targets_per_row = pairs.groupby("DRAVP_ID")["target_id"].agg(list)

    single_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    for _, row in dravp.iterrows():
        records = parse_activity(str(row.get("Activity") or ""), row.get("Mass"))
        if not records:
            continue
        targets = targets_per_row.get(row["DRAVP_ID"], [])
        if not targets:
            continue

        payload = {
            "DRAVP_ID": row["DRAVP_ID"],
            "Sequence": row["Sequence"],
            "Sequence_Length": row.get("Sequence_Length"),
            "Mass": row.get("Mass"),
            "Pubmed_ID": row.get("Pubmed_ID"),
        }
        for metric in METRICS:
            hits = [r for r in records if r.metric == metric]
            payload[f"{metric}_uM"] = aggregate_metric([h.value_micromolar for h in hits]) if hits else None
            payload[f"{metric}_n"] = len(hits)
            payload[f"{metric}_censoring"] = (
                "none"
                if not hits
                else ("none" if any(h.censoring == "none" for h in hits) else hits[0].censoring)
            )
            payload[f"{metric}_from_inhibition"] = bool(hits) and all(h.from_inhibition for h in hits)

        if len(targets) == 1:
            single_rows.append({**payload, "target_id": targets[0]})
        else:
            ambiguous_rows.append({**payload, "target_ids": ";".join(targets)})

    single = pd.DataFrame(single_rows).merge(stems, on="target_id", how="left")
    ambiguous = pd.DataFrame(ambiguous_rows)

    missing_stem = int(single["file_stem"].isna().sum())
    if missing_stem:
        raise ValueError(f"{missing_stem} training rows have no genome file stem")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "avp_potency_dataset.tsv.gz"
    ambiguous_path = args.output_dir / "avp_potency_multi_target.tsv.gz"
    single.to_csv(train_path, sep="\t", index=False, compression="gzip")
    ambiguous.to_csv(ambiguous_path, sep="\t", index=False, compression="gzip")

    both = int((single["IC50_uM"].notna() & single["EC50_uM"].notna()).sum())
    print(
        f"training rows={len(single)}  genomes={single.file_stem.nunique()}  "
        f"IC50={int(single.IC50_uM.notna().sum())}  EC50={int(single.EC50_uM.notna().sum())}  "
        f"both={both}  multi_target_held_out={len(ambiguous)}"
    )
    print(f"  -> {train_path}")
    print(f"  -> {ambiguous_path}")


if __name__ == "__main__":
    main()
