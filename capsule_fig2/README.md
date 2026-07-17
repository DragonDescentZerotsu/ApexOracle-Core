# ApexOracle Fig. 2b Capsule Reproduction

This capsule contains the inference-only resources needed to reproduce the
Fig. 2b MIC regression evaluation. It reloads cached frozen molecular features
and trained 5-fold regression-head checkpoints, then recomputes the fold-wise
R2 metrics. It does not train any model or run backbone feature extraction.

This Code Ocean capsule is intentionally limited to Fig. 2b. Code Ocean provides
at most 20 GB of persistent storage for this submission and a single Tesla T4
GPU, which is not sufficient to package and run the other inference-only
experiments. Those experiments require tens to hundreds of GB of trained
checkpoint and embedding resources even after removing optimizer states, so they
cannot be reproduced on this platform under the available storage and GPU
constraints.

## Run

From the capsule root:

```bash
code/run
```

Equivalent explicit form:

```bash
code/run fig2b
```

Outputs are written to `results/`:

- `fig2b_mdlm_dlm_mtr_metrics.json`
- `fig2b_chemberta_mtr_metrics.json`
- `fig2b_molformer_metrics.json`
- `fig2b_apex_metrics.json`
- `fig2b_peptideclm_metrics.json`
- `fig2b_chemberta_mlm_mean_metrics.json`
- `fig2b_chemberta_mlm_metrics.json`
- `fig2b_mic_regression_summary.json`

Optional prediction CSVs can be generated with:

```bash
code/run --write-predictions
```

## Included Resources

The required runtime resources are under `data/fig2b_mic_regression/`:

- one `features.pt` frozen feature cache per model
- five optimizer-free `fold_*/best_head.pt` regression-head checkpoints per model
- per-model metrics/provenance files

The source scripts used to build the caches and heads are included under
`data/source/` for auditability. The Code Ocean runtime entrypoint only needs
`code/run`, `code/reproduce_fig2b_mic_regression.py`, and
`data/fig2b_mic_regression/`.
