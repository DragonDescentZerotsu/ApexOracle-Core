# Remasking-window sensitivity

This experiment compares five remasking windows and a peptide-guidance control
under a fixed generation and evaluation protocol.

## Protocol

- 36 tasks: six conditions × two strains × three seeds; 100 raw attempts per
  task (four batches of 25), for 3,600 attempts in total.
- Seeds: `20260728`, `20260729`, `20260730`. Strains: BAA-3170 (length 368)
  and BAA-3197 (length 232).
- Windows: earlier `0.75–0.65`, current `0.55–0.45`, later `0.35–0.25`,
  narrower `0.525–0.475`, and wider `0.55–0.25`.
- The control uses the current window with `gamma_peptide=0`.
- Shared parameters: 256 steps, `r_t=0.02`, `alpha_on=0.5`, `gamma_MIC=15`,
  and target MIC 1 µM. The frozen specification is `task_manifest.json`.

## Entry points

Run from the Core repository root with the MDLM and Generation dependencies
installed. Use each command's `--help` for explicit asset paths and device options.

```bash
python scripts/reproduce/prepare_remasking_schedule_reviewer_tasks.py --help
python scripts/reproduce/run_remasking_schedule_reviewer.py --help
python scripts/reproduce/orchestrate_remasking_schedule_reviewer.py --help
python scripts/reproduce/evaluate_remasking_schedule_reviewer.py --help
MPLBACKEND=Agg python scripts/audit/plot_remasking_structure_qualified_peptides.py
PYTHONPATH=src python -m pytest -q tests/test_remasking_schedule_reviewer.py
```

Each task writes its own `runs/<task_id>/` directory, preserving all attempted
tokens, decoding outcomes, resolved configuration and batch hashes. Raw attempts,
checkpoints and row-level evaluation tables remain outside Git. Some tests require
the local `analysis/evaluated_attempts.csv` evaluation table.

## Evaluation and interpretation

Generation completed 36/36 tasks and 144/144 batches. Evaluation obtained 3,600
v1 clean-input classifier probabilities and 2,355 RDKit-valid finite MIC predictions.
MIC scoring uses the independent fixed-`t=1e-3`, non-padding checkpoint through
`apexoracle_mdlm.scoring`; it does not replace the noisy sampler guidance model.

A v1 classifier-positive output is not a verified peptide. The canonical figure
combines a structure screen with the SEP-padded classifier criterion; this remains
a preliminary operational definition, not universal peptide ground truth.
Predicted MIC is not measured antimicrobial activity. All raw attempts remain in
yield denominators; MIC summaries use valid molecules only.

The canonical figure is
`figures/remasking_structure_qualified_peptides_with_mic_control*`.
Yield uncertainty is the sample standard deviation across three seed-level pooled
rates; MIC uncertainty is the sample standard deviation across three seed-level
pooled medians. `--layout yield-only` reproduces the earlier figure layout.
See [results](RESULTS.md), [structure audit](STRUCTURE_AUDIT.md), and the
[figure index](figures/README.md) for numerical results, limitations and outputs.
