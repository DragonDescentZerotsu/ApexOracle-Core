# Reviewer response and manuscript implementation record: multiplier for MIC values reported as `>V`

> **Status (2026-08-08): implemented in the formal reviewer response and
> manuscript.** The historical filename is retained so existing documentation
> links remain stable.

> *MIC multipliers (>V -> 2V) are somewhat arbitrary. Were they sensitivity tested?*

Thank you for raising this point. We agree that replacing an MIC reported only
as `>V` with a finite value is an operational heuristic rather than a recovery
of the unknown true MIC. We used `2V` to represent one additional step on the
twofold-dilution scale commonly used for MIC measurements. However, the true
MIC is only known to be greater than `V`, so `2V` remains a practical
approximation rather than a fixed value for all greater than V records.

We have now performed a post hoc sensitivity analysis using the frozen
seven-member ensemble predictions. For measurements reported as `>V`,
we recalculated the held-out metrics after assigning `V`, `2V`, or `4V`, and
after excluding these measurements altogether. This multiplier sweep
directly affected 14,939 of 86,358 strain-wise held-out measurement instances
(17.30%) and 15,264 of 85,824 phylum-wise instances (17.79%). In the strain-wise
benchmark, the mean-across-fold R² values were 0.5785, 0.5813, and 0.5634 for
`V`, `2V`, and `4V`, respectively (Spearman rho: 0.7478, 0.7478, and 0.7415);
excluding the `>V` measurements gave R² = 0.5699 and rho = 0.7251. In the
phylum-wise benchmark, the corresponding R² values were 0.3804, 0.3879, and
0.3748 (rho: 0.6282, 0.6309, and 0.6269), while exclusion gave R² = 0.3491 and
rho = 0.5829. The mean absolute error varied more visibly with the point
encoding, as expected: for example, it was 0.4469, 0.4644, and 0.5088 across
the three strain-wise multiplier settings.

Thus, the precise finite multiplier affects absolute error estimates, and we
have revised the text to identify `2V` explicitly as a heuristic. However, the
held-out R² and rank-correlation conclusions were similar across the `V`--`4V`
range and remained positive after the `>V` rows were excluded, indicating that
the reported predictive signal is not dependent on the unique choice of a
`2V` multiplier.

## Manuscript text implemented beside Table `DBAASP_MIC`

**Placement used:** extend the current paragraph immediately before Table
`DBAASP_MIC` with the sensitivity-analysis sentence below, and place the compact
result paragraph immediately after the table and before the `PepLink` paragraph.

### Paragraph before the table

All MIC values originally reported in micrograms per milliliter were converted
to micromolar units. Values annotated with special operators (e.g., `<=`, `>`)
were handled according to the procedures summarized in Table `DBAASP_MIC`. For
model training, each MIC measurement was further transformed as
`-log10(MIC/10)`. Measurements reported as `>V` were by far the largest class
requiring a multiplier-based finite assignment (22,158 DBAASP measurements,
compared with 19 reported as `>>V` and assigned `3V`), and were therefore used
as the representative case for multiplier sensitivity analysis. Held-out metrics were
recalculated from the same frozen ensemble predictions after assigning `V`,
`2V`, or `4V` to the `>V` measurements, and after excluding these measurements;
models were not retrained.

### Compact result paragraph after the table

Measurements reported as `>V` comprised 14,939 of 86,358 strain-wise held-out
measurement instances (17.30%) and 15,264 of 85,824 phylum-wise instances
(17.79%). For the `V`, `2V`, and `4V` assignments, respectively, the
mean-across-group R² values were 0.5785, 0.5813, and 0.5634 in the strain-wise
benchmark and 0.3804, 0.3879, and 0.3748 in the phylum-wise benchmark. Excluding
the `>V` measurements gave R² values of 0.5699 and 0.3491, respectively.
Spearman correlations varied from 0.7415 to 0.7478 and from 0.6269 to 0.6309
across the three assignments, whereas MAE ranged from 0.4469 to 0.5088 and from
0.5165 to 0.5967 in the strain-wise and phylum-wise benchmarks, respectively.
These results support the use of `2V` as a practical heuristic and show that the
main R² and rank-correlation conclusions were not materially altered by other
reasonable assignments, although absolute-error estimates were more sensitive
to the assigned value.
