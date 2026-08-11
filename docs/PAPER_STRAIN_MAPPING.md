# Paper strain mapping

`assets/manifests/paper_strain_mapping.json` is the compact public bridge
between strain labels in the paper-era MIC table and the condition keys loaded
by ApexOracle Core. It contains no MIC values, molecule structures, embedding
tensors, or private assay rows.

## Contents

Each mapping row records:

- the source strain label;
- the canonical condition key;
- species name;
- whether the paper workflow used a `genome_text` or `text_only` condition;
- the corresponding embedding filenames;
- whether the mapping came from a handcrafted alias, a direct ATCC label, or
  an exact text-only name;
- the number of MIC rows routed through that mapping.

The frozen export contains 1,766 unique source strain labels and 1,769
condition routes. Three historical source labels had both a handcrafted
genome/text alias and an exact text-only condition, so they intentionally
appear twice. The routed count is 92,322 MIC records: 79,904 through the
genome+text path and 12,418 through the text-only path. These values exactly
match `prepare_hierarchical_mic_data` before token-length filtering.

## Rebuild

The source data remain external to Git. Rebuild from the paper data directory:

```bash
python scripts/prepare_data/export_paper_strain_mapping.py \
  --data-root /path/to/DataPrepare/Data \
  --output assets/manifests/paper_strain_mapping.json
```

The JSON embeds input paths relative to the data root, file sizes, SHA-256
hashes for file inputs, route counts, and the deterministic records. Directory
inputs are identified by relative path; embedding binaries themselves are not
hashed or copied into this compact mapping.

The exporter preserves the historical paper cohort exactly, including its
substring-based exclusion of deletion-mutant labels. That legacy rule also
excludes a small number of taxon labels containing the same substring. The
export records the model's frozen data contract; it does not silently redefine
or repair the published cohort.

Validation:

```bash
PYTHONPATH=src python -m pytest -q tests/test_strain_mapping_release.py
```
