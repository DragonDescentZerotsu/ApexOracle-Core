# Generated-candidate diversity publication handoff

## Scope

This capsule closes the reviewer questions about generated-candidate diversity and peptide/small-molecule
selection. It contains the canonical code, compact reviewer-facing results and provenance needed to reproduce
the reported 24-selected-peptide, 73-candidate-pool and 84,226-output analyses. The formal manuscript and
response-letter sources remain in the external `ApexOracle_cleaned` document directory and are not copied into
this Git repository.

## Canonical entrypoints

```bash
# Build/audit the corrected 73-row pool and generation-level diversity inputs.
MPLBACKEND=Agg PYTHONPATH=src /home/tianang/anaconda3/bin/python \
  scripts/audit/analyze_generated_candidate_diversity.py

# Recompute the 24 selected-peptide sequence/structure analysis.
MPLBACKEND=Agg PYTHONPATH=src /home/tianang/anaconda3/bin/python \
  scripts/audit/analyze_selected_peptide_diversity.py

# Stream the exact 84,226 choose 2 Tanimoto distribution from the local cache.
PYTHONPATH=src /home/tianang/anaconda3/bin/python \
  scripts/audit/compute_exact_generation_tanimoto.py \
  --fingerprint-cache experiments/generated_candidate_diversity/local_cache/generation_morgan_r2_2048_chiral.npz \
  --output-dir experiments/generated_candidate_diversity --workers 64

# Rebuild the canonical three-panel Supplementary Fig. C5.
MPLBACKEND=Agg /home/tianang/anaconda3/bin/python \
  scripts/audit/plot_generated_candidate_diversity_figure.py
```

Detailed inputs, denominators, outputs and interpretation boundaries are documented in `README.md`; the shared
alignment/fingerprint implementation is in `src/apexoracle/evaluation/generated_candidate_diversity.py`.

## Git release boundary

Versioned artifacts include the four entrypoints, shared evaluation module, focused tests, compact CSV/JSON
summaries and manifests, reviewer/audit Markdown, and the canonical PDF/SVG/PNG three-panel figure. The release
does not include raw sampler outputs, the external peptide parser/PepLink checkout, MolPort source assets,
workbooks, the response DOCX or manuscript TeX/PDF.

The following reproducible or superseded artifacts remain local under `.gitignore`:

- the 21 MiB generation fingerprint cache;
- 81 per-length SELFIES provenance files reconstructed from the external read-only generation outputs;
- the superseded two-panel Tanimoto figure;
- exploratory selected-peptide violin/stratified figures and their local captions;
- Python caches and temporary compilation/rendering directories.

No local evidence file was deleted. Tightening the release whitelist changes only what Git publishes; it does
not change analysis behavior or outputs.

## Validation

- Focused tests: `13 passed` for generated diversity, exact Tanimoto and sequence similarity.
- Full repository: `171 passed` (`14` pre-existing dependency/runtime warnings).
- A clean temporary rerun of `analyze_selected_peptide_diversity.py` reproduced all seven canonical compact
  CSV/JSON outputs byte-for-byte.
- A clean temporary rerun of the C5 plotting entry reproduced the canonical PNG byte-for-byte; PDF/SVG visual
  content is stable but container metadata/element identifiers are not byte-deterministic across invocations.
- The revised manuscript source compiled independently to 32 pages; the reorganized Methods and Supplementary
  Fig. C5 pages were visually checked. The formal manuscript PDF was not overwritten.

## Publication record

The user authorized direct publication to `DragonDescentZerotsu/Synergy` `main`. Final commit identifiers and
post-push alignment are recorded in the repository-level `AGENTS.md` after publication.
