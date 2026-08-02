# Fixed strain-wise exact-peptide sensitivity analysis

## Scope and status

This analysis retrains the original strain-wise hierarchical MIC task on the frozen
`PYTHONHASHSEED=0` candidate membership in
`experiments/hierarchical_mic/strain/legacy_protocol_manifest.json`. It preserves the
legacy model, four training routes, batch size 80, 25 epochs, seven seeds, train-mode
held-out evaluation, and highest-held-out-\(R^2\) checkpoint selection.

This is a new fixed-split reconstruction. It is not the unrecovered exact membership
used by the 2025 checkpoints, because the historical split used unordered Python
`set` iteration and did not record `PYTHONHASHSEED`.

All \(3\times7=21\) training tasks and all 21 deterministic `eval()` checkpoint
replays completed. Exact peptide identity is the SHA-256 of the stored model-input
token sequence. The sensitivity cohort is formed after training by retaining only
test measurements whose exact peptide was absent from that fold's training data.

## Verified results

| Cohort | Measurements | Distinct exact peptides | MIC <=16 µM | R² | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full test | 86,358 | 16,560 | 47.01% | 0.4638 | 0.6681 | 0.6826 |
| Train-seen exact peptide | 60,086 (69.58%) | 9,248 | 44.34% | 0.5672 | 0.7499 | 0.7555 |
| Train-unseen exact peptide | 26,272 (30.42%) | 8,259 | 53.12% | 0.0942 | 0.4070 | 0.4130 |

The table reports measurement-level pooled metrics. For direct comparison with the
paper's mean-across-folds headline \(R^2\), the full, train-seen and train-unseen
fold-mean values are 0.5814, 0.6283 and 0.1089, respectively. The full-test value
is consistent with the reported historical strain-wise mean of 0.5793.

The 2,000-replicate exact-peptide cluster-bootstrap 95% CI for unseen \(R^2\) is
`[0.0687, 0.1191]`. The corresponding CIs are `[0.3899, 0.4240]` for Spearman and
`[0.3951, 0.4308]` for Pearson.

Per-fold unseen results are:

| Fold | Unseen measurements | Distinct exact peptides | MIC <=16 µM | R² | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 25,645 | 7,732 | 53.25% | 0.0923 | 0.4048 | 0.4108 |
| 1 | 363 | 332 | 57.02% | 0.0690 | 0.4411 | 0.4420 |
| 2 | 264 | 218 | 35.23% | 0.1653 | 0.4985 | 0.4830 |

The pooled result is dominated by fold 0; folds 1 and 2 contain relatively few
unseen measurements and should not be interpreted independently as precise
fold-level estimates.

## Interpretation

The result does not support claiming that exact-peptide reuse has no effect.
Calibrated performance falls substantially from \(R^2=0.5672\) on train-seen
peptides to \(R^2=0.0942\) on train-unseen peptides. However, the strict unseen
cohort is not small overall, contains many low-MIC observations, has positive
\(R^2\), and retains moderate ranking correlations. The defensible conclusion is:

> Exact-peptide overlap materially contributes to the original pathogen-holdout
> performance, while the model retains a weaker but measurable signal on peptides
> not observed during training.

The main benchmark should continue to be described as generalization to held-out
pathogens, not as generalization to novel peptide molecules. This sensitivity
analysis should be reported separately and should not replace the pathogen-holdout
benchmark.

## Concise reviewer-response draft

We thank the reviewer for highlighting this distinction. We agree that the same
peptide can occur in training measurements for other strains; accordingly, the
main split evaluates generalization to held-out pathogens rather than to entirely
novel peptide molecules. We have now quantified this effect using a fixed,
deterministic reconstruction of the original threefold strain-wise protocol,
retaining the same model, training procedure and seven-member ensembles. Among
86,358 eligible test measurements, 60,086 (69.58%) involved an exact peptide
observed in that fold's training data, whereas 26,272 (30.42%; 8,259 distinct exact
peptides) were train-unseen. The unseen cohort remained well represented at the
active end of the regression range: 53.12% of its measurements had MIC <=16 µM.

The mean-across-folds \(R^2\) values on the full, train-seen and train-unseen
cohorts were 0.5814, 0.6283 and 0.1089, respectively. Within the train-unseen
cohort, the mean-across-folds Spearman and Pearson correlations were 0.4481 and
0.4452, respectively. Together, the positive \(R^2\) and correlation values
indicate that the model retained measurable regression predictive ability as well
as meaningful ranking ability under the more stringent joint condition of unseen
peptides and held-out strains. Thus, exact-peptide reuse materially contributes to
calibrated performance, but the model retains a weaker, measurable signal on
peptides absent from training. This computational result is further supported by
our prospective experimental validation: ApexOracle generated three representative
peptides active against the training-unseen strains *P. aeruginosa* PA5257
or *E. coli* AR-0349, with MIC values of 32, 64 and 8 \(\mu\)M,
respectively. Their maximum sequence identities to any training peptide were only
36.7%, 35.7% and 36.8%, respectively. These results provide complementary
experimental evidence that ApexOracle can generate active, structurally novel
peptides in the challenging setting where both the peptide and target strain are
unseen. We have revised the manuscript to define the split unit and scope
explicitly and report this stricter molecule-disjoint test-subset analysis as a
sensitivity analysis.

## Reproduction and artifacts

- Task and machine ownership:
  `experiments/hierarchical_mic/fixed_strain_retrain/task_manifest.json`
- Row-level replay output (local, not versioned):
  `experiments/hierarchical_mic/fixed_strain_retrain/predictions/`
- Ensemble metrics:
  `experiments/hierarchical_mic/fixed_strain_retrain/analysis/metrics.csv`
- Cluster bootstrap:
  `experiments/hierarchical_mic/fixed_strain_retrain/analysis/cluster_bootstrap.csv`
- Analysis manifest:
  `experiments/hierarchical_mic/fixed_strain_retrain/analysis/analysis_manifest.json`

Validation:

```bash
PYTHONPATH=src python scripts/reproduce/summarize_hierarchical_mic_molecule_disjoint.py \
  --protocol strain --groups 3 --members 7 --bootstrap-iterations 2000 \
  --prediction-dir experiments/hierarchical_mic/fixed_strain_retrain/predictions \
  --output-dir experiments/hierarchical_mic/fixed_strain_retrain/analysis
```
