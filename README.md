# ApexOracle Core

ApexOracle Core contains the shared data contracts, feature interfaces, fusion models, checkpoint loaders,
training/evaluation runners, and reproducibility audits used by ApexOracle. It is the Core submodule of the
multi-repository [ApexOracle](https://github.com/DragonDescentZerotsu/ApexOracle) release.

The molecular DLM pretraining producer, downstream MDLM guidance/scoring, Evo-2 genome extraction, and guided
generation sampler remain independent modules with their own environments. Core consumes their versioned outputs;
it does not duplicate those implementations.

## Included workflows

- hierarchical MIC prediction with strain-, species-, and phylum-level holdouts;
- three-strain antibiotic classification and matched benchmark interfaces;
- strain-wise synergy cross-validation;
- Fig. 2b molecular-representation benchmarks and sequence-similarity evaluation;
- reusable precomputed molecule, genome, and strain-text feature contracts;
- reviewer-facing reproducibility and plotting workflows with frozen manifests.

Large datasets, model weights, embeddings, raw generations, caches, and experiment outputs are intentionally not
stored in Git.

## Installation

```bash
git clone https://github.com/DragonDescentZerotsu/ApexOracle-Core.git
cd ApexOracle-Core
python -m pip install -e .
```

Install only the optional capabilities you need:

```bash
# Data builders and strain-text embedding
python -m pip install -e ".[data-preparation]"

# MIC/classification benchmark and synergy runtimes
python -m pip install -e ".[benchmark,synergy]"

# Sequence-similarity and figure workflows
python -m pip install -e ".[similarity,figures]"

# Development tests
python -m pip install -e ".[test,benchmark,synergy,data-preparation,similarity,figures]"
```

PepLink is a separate optional package. Peptide conversion workflows use the fixed public release
`PepLink==0.1.2` rather than a copied source tree.

## Command-line entrypoints

Inspect the stable runners without loading a checkpoint:

```bash
apexoracle-run-hierarchical-mic --help
apexoracle-run-antibiotic-classification --help
apexoracle-run-synergy-cv --help
```

Repository wrappers and frozen paper configs live under `scripts/reproduce/` and `configs/`. For example:

```bash
PYTHONHASHSEED=0 python scripts/reproduce/run_hierarchical_mic.py \
  --protocol strain \
  --test-group 0 \
  --acknowledge-dynamic-legacy-split \
  --dry-run
```

The canonical strain-text producer preserves the paper tensor contract while using explicit paths and a pinned
model revision:

```bash
python scripts/prepare_data/embed_strain_texts.py \
  --input-dir /path/to/descriptions \
  --output-dir /path/to/text_embeddings \
  --device cuda:0 \
  --local-files-only
```

It writes one float32 `[tokens, features]` tensor per description plus a SHA-256 provenance manifest. See
[`scripts/prepare_data/README.md`](scripts/prepare_data/README.md) for the filename/replacement contract and parity
evidence.

## Data and model assets

Runtime assets are external to the source release. Do not infer redistribution permission from this repository's
MIT license.

- historical weight identities and local provenance: [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md);
- machine-readable weight registry: [`configs/model_weights.yaml`](configs/model_weights.yaml);
- compute, environment, and private/public asset boundaries:
  [`docs/COMPUTE_AND_ASSET_MAP.md`](docs/COMPUTE_AND_ASSET_MAP.md);
- unified five-module release architecture:
  [`docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md`](docs/UNIFIED_APEXORACLE_RELEASE_PLAN.md).

Paper/reviewer workflows document their required source tables, checkpoints, outputs, and validation commands in
the corresponding directory under [`experiments/`](experiments/README.md).

## Testing

The project tests synthetic contracts without private assets and conditionally exercises installed paper assets:

```bash
PYTHONPATH=src python -m pytest -q
```

Tests that require ignored source tables, large outputs, checkpoints, or locally installed compiled tools are
skipped when those assets are absent. A clean source clone must still complete all remaining tests, package builds,
imports, and CLI help checks.

## Legacy recovery

The active tree contains canonical, parameterized implementations rather than paper-era machine-specific scripts.
Deleted source remains recoverable from the annotated tags:

- `legacy-code-snapshot-2026-07-17`
- `core-pre-public-cleanup-2026-08-10`

The file-by-file role and disposition of the former `DataPrepare/` source tree are recorded in
[`docs/DATAPREPARE_LEGACY_LEDGER.md`](docs/DATAPREPARE_LEGACY_LEDGER.md). Ignored local datasets and embeddings
were not deleted as part of source cleanup.

## License

Core source is released under the [MIT License](LICENSE). Vendored third-party notices are summarized in
[`NOTICE`](NOTICE); external datasets, pretrained models, and optional dependencies retain their own terms.
