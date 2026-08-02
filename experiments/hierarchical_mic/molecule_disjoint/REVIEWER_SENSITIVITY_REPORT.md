# Hierarchical MIC exact-molecule-disjoint sensitivity

Status: completed post hoc sensitivity analysis for reviewer discussion. This
file does not by itself modify the manuscript or response-letter DOCX.

## Question answered

The historical hierarchical MIC benchmarks split by pathogen group, not by
peptide. They therefore measure transfer to held-out pathogen contexts and
allow the same peptide to have been assayed against training pathogens. This
analysis asks how the archived models perform after removing from each test
group every measurement whose exact peptide model input occurred in that
group's training partition.

Exact molecule identity was defined as SHA-256 of the complete stored token
sequence consumed by the frozen molecular encoder. This deliberately merges
different database IDs with identical model inputs. The original pathogen
holdout and archived model parameters were not changed.

## Primary result: fixed phylum membership

The phylum-wise benchmark is the primary sensitivity because its three
taxonomic group memberships are deterministic and all 21 final MDLM
checkpoints are available.

- Full test: 85,824 measurements.
- Train-seen exact peptide: 79,309 measurements (92.41%).
- Strict train-unseen exact peptide: 6,515 measurements (7.59%), comprising
  3,491 exact molecules and 440 pathogen instances.
- Of the retained measurements, 3,792 (58.20%) had MIC <= 16 micromolar.
- Retained measurements by group: Fungi 600, Pseudomonadati 3,449, and
  Bacillati 2,466.

Seven archived members were averaged at the sample level. The deterministic
`eval()` replay produced:

| Test cohort | Group | R2 | Spearman | Pearson |
| --- | --- | ---: | ---: | ---: |
| Full | Fungi | 0.309 | 0.580 | 0.634 |
| Full | Pseudomonadati | 0.417 | 0.672 | 0.679 |
| Full | Bacillati | 0.438 | 0.641 | 0.663 |
| Exact-peptide unseen | Fungi | 0.011 | 0.389 | 0.385 |
| Exact-peptide unseen | Pseudomonadati | 0.075 | 0.370 | 0.370 |
| Exact-peptide unseen | Bacillati | -0.159 | 0.198 | 0.214 |
| Exact-peptide unseen | All groups pooled | 0.014 | 0.333 | 0.340 |

The macro-average group R2 decreased from 0.388 on the full test cohorts to
-0.024 on the strict unseen-peptide subsets. Because R2 is nonlinear, the
pooled sample-level R2 (0.014) is not equal to the macro-average group R2 and
both should be distinguished if reported.

In 2,000 paired cluster-bootstrap replicates that resampled exact molecules and
retained all their MIC rows, the pooled strict-unseen estimates were:

| Metric | ApexOracle 95% CI | Train-mean baseline 95% CI | Paired delta 95% CI |
| --- | ---: | ---: | ---: |
| R2 | -0.043 to 0.060 | -0.124 to -0.074 | 0.060 to 0.156 |
| Spearman | 0.293 to 0.369 | 0.090 to 0.167 | 0.156 to 0.247 |
| Pearson | 0.300 to 0.374 | 0.082 to 0.159 | 0.168 to 0.264 |

The pooled model-versus-baseline paired delta was positive in all 2,000
replicates. However, the model R2 interval itself included zero, and Bacillati
had negative R2. The defensible interpretation is therefore that ranking
signal remains for exact unseen peptides, while calibrated variance explanation
decreases markedly.

## Secondary result: strain reconstruction

The frozen deterministic strain candidate retained 26,272 strict-unseen
measurements (8,259 exact molecules), of which 53.12% had MIC <= 16
micromolar. Its pooled 7-member R2/Spearman/Pearson were
0.088/0.403/0.409, with an R2 cluster-bootstrap interval of 0.062--0.112.

This is useful internal robustness evidence, but it must not be presented as
the formal primary sensitivity: the exact 2025 strain membership was not
recorded because the historical processes did not log `PYTHONHASHSEED`.
Consequently the candidate-defined training exposure cannot be guaranteed to
equal the archived checkpoint's exact training exposure.

## Interpretation and manuscript consequence

1. The reviewer is correct that the hierarchical split unit was pathogen, not
   peptide. The response should state this directly.
2. The main benchmark remains a valid test of held-out pathogen-context
   transfer for a peptide library that includes many previously assayed
   peptides. It is not a joint unseen-pathogen/unseen-peptide benchmark.
3. Exact-peptide reuse contributes materially to the headline R2. We should not
   claim that removing overlap leaves performance unchanged.
4. The phylum sensitivity should be reported as post hoc supplementary
   evidence, including all three group R2 values rather than only the pooled
   value.
5. A genuinely prospective joint-disjoint benchmark would require a new
   training/validation/test design. Filtering archived test predictions is not
   equivalent to retraining under a joint-disjoint split.
6. The archived hierarchical protocol used held-out performance for checkpoint
   selection. Therefore this sensitivity is an evaluation subset of archived
   checkpoints, not a newly untouched outer test set; avoid the phrase
   "independent outer test."

## Reproducibility

- Overlap audit:
  `scripts/audit/audit_hierarchical_mic_molecule_overlap.py`
- Per-member replay:
  `scripts/reproduce/evaluate_hierarchical_mic_molecule_disjoint.py`
- Ensemble metrics and molecule-cluster bootstrap:
  `scripts/reproduce/summarize_hierarchical_mic_molecule_disjoint.py`
- Compact outputs:
  `phylum_analysis/metrics.csv`,
  `phylum_analysis/cluster_bootstrap.csv`, and
  `phylum_analysis/analysis_manifest.json`

The strain inference-only checkpoint exporter records source and derived
SHA-256 values. A representative source/derived pair was also loaded and every
inference tensor was verified with exact `torch.equal`.
