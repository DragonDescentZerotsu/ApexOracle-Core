# Genome representation validation figure

Canonical entry:

```bash
MPLBACKEND=Agg PYTHONPATH=src python \
  scripts/audit/plot_genome_representation_validation.py
```

The script reads only frozen outputs from `fragment_variation/all_embeddings/` and
`historical_probe/analysis/`. It does not retrain a probe or recompute Evo-2 embeddings. The canonical
stem is `genome_representation_validation`; PDF, SVG and PNG exports share the same
content. `*_manifest.json` records panel definitions and SHA-256 provenance. The three exact
plotted-data CSVs are compact, explicitly allowlisted release artifacts; raw alignment and
probe-prediction tables remain local under the repository-wide storage policy.

Panel a omits identical fragments and shows the 4,649 variable homologous fragments on
a log cosine-distance axis starting at `1e-8`, below the smallest observed distance
(`5.86e-8`). The 4,156 fragments from whole-genome ANI `>=99%` strain pairs are blue
circles; the remaining 493 fragments are grey crosses. The panel has no bins or fitted
trend line.
Panels b/c show all five held-out folds, with every genome kept entirely within one fold,
together with the fold mean ± sample s.d. and the relevant random baseline. The figure does
not include the abandoned genome-swap diagnostic.

Validation: `PYTHONPATH=src python -m pytest -q tests/test_genome_condition_reviewer.py`
completed with 13 passed; the full repository completed with 184 passed and 14 existing
dependency/runtime warnings. The canonical figure received visual QA; Black, JSON and
`git diff --check` also passed.
