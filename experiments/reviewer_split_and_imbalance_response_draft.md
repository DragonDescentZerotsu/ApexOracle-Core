# Reviewer response draft: train/test splitting and outcome imbalance

Status: synchronized to the revised manuscript and response-letter DOCX on 2026-07-28.

## Reviewer comment

> How the train/test split was defined. A random split is less challenging than a scaffold split. Was a stratified split strategy used? As antibacterial activity datasets are typically highly imbalanced, it is important to ensure that the proportion of actives (or low MIC values for regression) is sufficient in test sets. Otherwise, always predicting inactive (or high MIC) can lead to artificially high performance metrics.

## Proposed response

We thank the reviewer for this important comment. We have clarified the data-partitioning strategies and evaluation metrics in the revised Methods.

We first examined whether the regression test sets contained sufficient low-MIC measurements. To provide a deliberately stringent stress test of test-set composition, we used MIC \(\leq 16\,\mu\mathrm{M}\)—a cutoff fourfold lower than the MIC \(\leq 64\,\mu\mathrm{M}\) criterion used to interpret activity in the prospective wet-lab experiments—as a descriptive cutoff. In the shared Fig. 2b benchmark, 15,547 of 30,991 observed test MIC labels (50.17%) were \(\leq 16\,\mu\mathrm{M}\), with consistent proportions across the five folds (49.66%–51.11%). Every pathogen-target–fold combination contained at least 41 such measurements. In the fixed strain-wise evaluation, 40,596 of 86,358 held-out measurements (47.01%) had MIC \(\leq 16\,\mu\mathrm{M}\), with fold-wise proportions ranging from 41.59% to 51.02% (Supplementary Fig. C3). Across the 85,824 evaluation-eligible measurements in the species- and phylum-level pathogen-holdout benchmarks, 39,450 (45.97%) were \(\leq 16\,\mu\mathrm{M}\). The proportions ranged from 30.98% to 63.70% across the eleven species-level groups and from 41.89% to 49.84% across the three phylum-level groups. Thus, even under this stringent cutoff, the regression test sets contained substantial numbers of low-MIC measurements and were not dominated by uniformly high MIC values.

The binary Fig. 1b test folds likewise contained active compounds in every fold. The active-class proportions ranged from 3.65% to 8.14% for *E. coli*, from 5.27% to 6.77% for *A. baumannii*, and from 1.23% to 1.41% for RN4220. Accuracy was not reported; AUPRC, calculated as average precision, was used as the primary classification metric. An always-inactive constant-score classifier has an AUPRC equal to the active-class prevalence and would therefore obtain only prevalence-level performance. By comparison, fine-tuned ApexOracle achieved mean AUPRC values of 0.712, 0.434 and 0.401 for the three respective targets. Thus, the reported classification performance cannot be explained by an always-inactive predictor.

MIC regression was performed using the continuous transformed label

\[
z_i=-\log_{10}\left(\frac{\mathrm{MIC}_i}{10}\right),
\]

and \(R^2\) was calculated as

\[
R^2=1-\frac{\sum_i(z_i-\hat z_i)^2}
{\sum_i(z_i-\bar z_{\mathrm{test}})^2}.
\]

For any constant prediction \(c\),

\[
R^2(c)=-
\frac{n(c-\bar z_{\mathrm{test}})^2}
{\sum_i(z_i-\bar z_{\mathrm{test}})^2}\leq 0.
\]

Therefore, even the optimal constant predictor—the test-set mean—has \(R^2=0\), while an always-high-MIC predictor has \(R^2<0\). Constant predictions also yield undefined, rather than high, Pearson and Spearman correlations. In contrast, the headline mean \(R^2\) values were positive: 0.5386 in the shared molecular-encoder benchmark, 0.5814 in the fixed strain-wise evaluation, 0.3809 in the species-wise evaluation and 0.3744 in the phylum-wise evaluation. The reported regression performance therefore cannot arise from an always-high-MIC predictor.

Taken together, the observed test-set composition and the mathematical behavior of AUPRC and \(R^2\) rule out the specific failure mode raised by the reviewer: neither an always-inactive classifier nor an always-high-MIC regressor could obtain artificially elevated values for the metrics reported here. Outcome-value stratification was therefore not required to prevent this particular artifact. For the hierarchical MIC experiments, there is an additional design reason not to impose MIC-value stratification: the indivisible splitting unit was an entire pathogen strain or taxonomic group, and forcing a predetermined MIC distribution would require selecting or subdividing pathogen groups according to their outcomes, thereby compromising the intended pathogen-level separation.

For the hierarchical MIC experiments in Figs. 1a, 2c, 2f and 2g, the split was not performed randomly at the level of individual peptide–pathogen measurements. Instead, the intended evaluation axis was pathogen-level generalization: entire pathogen strains, species-level clusters or broader taxonomic groups were held out, and all peptide–MIC measurements associated with a held-out pathogen group were excluded from gradient-based model fitting.

A conventional molecular scaffold split addresses a different generalization question—generalization to unseen molecular chemotypes—whereas the hierarchical experiments were designed specifically to evaluate generalization to unseen pathogen contexts. We have clarified this scope and do not interpret these experiments as a joint unseen-pathogen and unseen-peptide-family benchmark.

The molecular-encoder regression benchmark in Fig. 2b used one predefined shuffled molecule-level fivefold partition with random seed 42. All seven encoders received exactly the same 10,886 molecule IDs and the same train and held-fold assignments. This benchmark was not scaffold-based or MIC-stratified; its purpose was to provide a controlled, like-for-like comparison of molecular representations on identical data. The empirical test-set composition reported above shows that its folds nevertheless contained comparable and substantial proportions of low-MIC measurements.

Similarly, Fig. 1b used fixed ordinary shuffled molecule-level KFold partitions with random seed 42 rather than label-stratified or scaffold-based outer folds. All methods were evaluated on the same fold assignments, and every fold contained active compounds.

## Internal notes — remove before submission

- The Fig. 2b and species/phylum composition numbers above were recomputed from the frozen evaluation-eligible data and split artifacts.
- The fixed strain-wise composition values and Supplementary Fig. C3 use the completed fixed evaluation that supplies the revised strain-wise result.
- The classification code computes average precision and reports it as AUPRC. A constant-score classifier therefore has AUPRC equal to class prevalence.
- Avoid the phrase “used only for testing.” The verified statement is “excluded from gradient-based model fitting,” because held-fold performance was used for legacy checkpoint selection.
- The revised Methods includes the concise statistical description and cites Supplementary Fig. C3 for the fixed strain-wise outcome-composition audit.

---

## Related Reviewer 2 comment: unit of splitting

> Given that the same peptide may have been tested against multiple strains, it is important to confirm that splitting was done by peptide (not by peptide-strain entries), otherwise the same molecule could appear in both train and test sets against different strains, inflating performance metrics.

## Proposed response

We thank the reviewer for raising this important point. We confirm that the primary hierarchical MIC partitions were defined by held-out pathogen groups, not by peptide. Consequently, a peptide tested against a held-out pathogen could also have appeared in the training data against another pathogen. We have revised the Methods and Results to state the scope more precisely: these benchmarks evaluate transfer to unseen pathogen contexts within the available peptide library; they are not joint unseen-pathogen and unseen-peptide benchmarks.

We quantified this overlap using the exact stored molecular token sequence consumed by the model, thereby treating different database IDs with identical model inputs as the same peptide. In the phylum-wise benchmark, 79,309 of 85,824 eligible held-out measurements (92.41%) involved a peptide present in the corresponding training partition. We therefore performed a post hoc exact-molecule-disjoint sensitivity analysis by retaining only test measurements whose peptide model input was absent from that training partition. This left 6,515 measurements (7.59%; 3,491 exact peptides) across all three held-out phylum groups, including 600 Fungi, 3,449 Pseudomonadati and 2,466 Bacillati measurements. The retained subset was not depleted of low-MIC observations: 58.20% had MIC \(\leq16\,\mu\mathrm{M}\).

Using the same archived seven-member models without retraining, the exact-peptide-unseen subsets yielded \(R^2\) values of 0.011, 0.075 and -0.159 for Fungi, Pseudomonadati and Bacillati, respectively, with corresponding Spearman correlations of 0.389, 0.370 and 0.198. Across all retained sample-level predictions, pooled \(R^2\), Spearman and Pearson values were 0.014, 0.333 and 0.340. In 2,000 paired cluster-bootstrap replicates that resampled exact peptides, the pooled 95% intervals were -0.043 to 0.060 for \(R^2\), 0.293 to 0.369 for Spearman and 0.300 to 0.374 for Pearson. Relative to the group-specific training-mean baseline, the paired \(R^2\) improvement had a 95% interval of 0.060 to 0.156.

These results show that a ranking signal remains for exact unseen peptides, but they also confirm the reviewer's concern that performance—particularly calibrated \(R^2\)—decreases substantially when train-seen peptides are removed. We now report this analysis as a post hoc Supplementary sensitivity and have softened the manuscript language so that the primary pathogen-holdout results are not interpreted as molecule-disjoint generalization.

## Suggested concise Methods addition

As a post hoc molecule-overlap sensitivity analysis, peptide identity was defined by the exact stored molecular token sequence used by the model. Within each phylum-wise pathogen holdout, test measurements whose peptide identity occurred in the corresponding training partition were removed without retraining or reselecting the archived seven model members. Metrics were recalculated from sample-level ensemble-mean predictions on the retained exact-peptide-unseen measurements. Uncertainty was estimated using 2,000 paired cluster-bootstrap replicates that resampled peptide identities and retained all MIC measurements associated with each sampled peptide.

## Additional internal notes — remove before submission

- The phylum analysis is the primary reportable sensitivity because its taxonomic membership is fixed and its 21 final MDLM checkpoints are complete.
- Do not use the stronger strain reconstruction as the primary evidence. Its exact 2025 fold membership is unrecovered because `PYTHONHASHSEED` was not logged.
- Do not describe the sensitivity subset as an independent outer test. The archived hierarchical protocol selected checkpoints using held-out performance.
- The analysis is exact-molecule-disjoint, not peptide-family-, sequence-cluster- or scaffold-disjoint.
- The substantial full-to-unseen \(R^2\) decrease should not be hidden. The defensible conclusion is retained ranking signal with weaker calibration, not unchanged performance.
