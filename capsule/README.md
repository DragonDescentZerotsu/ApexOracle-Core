# ApexOracle Capsule Reproduction

This capsule includes inference-only reproduction paths for:

- the trained non-seen-strain MDLM-MTR 7-ensemble experiment
- the Fig. 1b zero-shot antibiotic classification experiment
- the Fig. 2b DLM/MTR MIC regression experiment

These paths reconstruct or reload the required evaluation data from `data/`,
load optimizer-free trained checkpoints, and write metrics to `results/`.

## Run

From the capsule root:

```bash
code/run
```

The default inference batch size is `128`, chosen to use most of a 15 GB T4
while leaving some margin for variable genome/text sequence lengths. If a
smaller GPU runs out of memory, use `code/run --batch-size 96` or
`code/run --batch-size 64`.

Useful development checks:

```bash
code/run --fold 0 --num-ensembles 1 --max-batches 2
```

Outputs:

- `results/non_seen_strains_mdlm_mtr_fix_metrics.json`
- `results/non_seen_strains_mdlm_mtr_fix_predictions.csv`

For zero-shot antibiotic classification:

```bash
code/run zero-shot
```

Useful development checks:

```bash
code/run zero-shot --group 0 --num-ensembles 1 --max-batches 2
```

The zero-shot path defaults to `--batch-size 64` for the 15 GB T4 target. If it
runs out of memory, use `code/run zero-shot --batch-size 32`.

Outputs:

- `results/zero_shot_antibiotic_classification_metrics.json`
- `results/zero_shot_antibiotic_classification_predictions.csv`

For Fig. 2b MIC regression:

```bash
code/run fig2b
```

This path is eval-only. It reloads cached frozen first-token DLM features and
the cached baseline features with their five trained regression heads, then
recomputes the fold-wise R2 metrics. It does not run backbone feature extraction
or train any head inside the capsule.

Outputs:

- `results/fig2b_mdlm_dlm_mtr_metrics.json`
- `results/fig2b_chemberta_mtr_metrics.json`
- `results/fig2b_molformer_metrics.json`
- `results/fig2b_apex_metrics.json`
- `results/fig2b_peptideclm_metrics.json`
- `results/fig2b_chemberta_mlm_mean_metrics.json`
- `results/fig2b_chemberta_mlm_metrics.json`
- `results/fig2b_mic_regression_summary.json`

## Resource Notes

The original full raw genome/text/molecule embeddings are included under
`data/DataPrepare/Data/`. The formal 7-ensemble checkpoints are included under
`data/Checkpoints/.../MDLM_MTR_fix_7_fold_ensembles/` after removing only the
optimizer state from each checkpoint.

Zero-shot antibiotic classification uses the checkpoints under
`data/Checkpoints/.../MDLM_fix_cls_sm_all_test_10_fold_ensembles/`. These are
also stored without optimizer state.

Fig. 2b MIC regression uses resources under
`data/fig2b_mic_regression/<model>/`: a frozen feature cache and five
optimizer-free `best_head.pt` regression-head checkpoints for DLM/MTR,
ChemBERTa-MTR, ChemBERTa-MLM, ChemBERTa-MLM mean pooling, MoLFormer, PeptideCLM,
and APEX.
