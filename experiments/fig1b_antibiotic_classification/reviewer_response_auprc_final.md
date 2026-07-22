# Fig. 1b reviewer response：最终 10-member AUPRC 版本

## 可直接粘贴到 response letter 的英文回复

**Response:** We thank the reviewer for highlighting the inconsistency in the original Fig. 1b and agree that AUPRC is the more informative primary metric for these imbalanced antibacterial-activity datasets. We have therefore revised Fig. 1b to report AUPRC consistently for all three target strains, rather than mixing AUPRC and AUROC across strains.

Because Stokes et al. did not report E. coli AUPRC, we reproduced the released Chemprop/RDKit model profile on the same fixed fivefold molecule partitions used for the ApexOracle comparison. We applied the same common-fold procedure to the Liu and Wong model profiles.

To make the comparison like-for-like, we used the same fixed five outer molecule folds for every method. Within each fold, the fine-tuned ApexOracle model and the corresponding Chemprop baseline were each evaluated as a 10-model ensemble. Strict zero-shot ApexOracle was also a fixed 10-model ensemble and was evaluated on the same five test partitions without using target-strain samples for gradient updates. For the *A. baumannii* comparison, we used the no-RDKit-feature version reported by Liu et al., because ApexOracle likewise does not use RDKit descriptor augmentation. Bars in the revised figure show the mean AUPRC across the five outer folds, and error bars show the sample standard deviation across folds.

For *E. coli* BW25113, the Chemprop baseline, zero-shot ApexOracle and fine-tuned ApexOracle achieved AUPRC values of 0.545 ± 0.133, 0.585 ± 0.137 and 0.712 ± 0.147, respectively. For *A. baumannii* ATCC 17978, the corresponding values were 0.304 ± 0.022, 0.325 ± 0.066 and 0.434 ± 0.053. For *S. aureus* RN4220, they were 0.366 ± 0.048, 0.166 ± 0.053 and 0.401 ± 0.035. Values are mean ± sample standard deviation across the five fixed outer folds.

We additionally compared methods on identical sample-level out-of-fold predictions using 5,000 class-stratified paired bootstrap resamples and 5,000 two-sided paired prediction-swap randomizations, with Holm correction across the three strains within each model mode. Fine-tuned ApexOracle improved pooled out-of-fold AUPRC over the corresponding baseline by 0.147 for *E. coli* (95% CI, 0.055 to 0.245; Holm-adjusted p = 0.0264), 0.124 for *A. baumannii* (95% CI, 0.081 to 0.162; p = 0.0006), and 0.060 for RN4220 (95% CI, 0.021 to 0.097; p = 0.0264). In the strict zero-shot setting, the differences were 0.044 for *E. coli* (95% CI, −0.065 to 0.151; p = 0.5427), 0.029 for *A. baumannii* (95% CI, −0.016 to 0.073; p = 0.5427), and −0.169 for RN4220 (95% CI, −0.214 to −0.126; p = 0.0006). Thus, fine-tuned ApexOracle significantly outperformed the matched baseline for all three strains, whereas zero-shot performance was target-dependent.

## 建议替换的 Fig. 1b 图注

**b,** Small-molecule antibiotic classification evaluated consistently by AUPRC for all three target strains. We reproduced each baseline using the model configuration and training procedure reported in its original study and evaluated all methods on the same fixed fivefold molecule partitions. Bars show mean AUPRC and error bars show sample standard deviation across the five folds. Brackets show Holm-adjusted p values from paired prediction-swap tests.

## 建议替换的 Results 段落

Beyond peptides, we evaluated small-molecule antibiotic activity prediction against three strain-specific Chemprop baselines, reproducing the model configuration and training procedure reported in each original study and evaluating all methods on the same fixed fivefold molecule partitions (Fig. 1b). AUPRC was used as the primary metric because the active class was sparse in all three datasets. For *E. coli* BW25113, *A. baumannii* ATCC 17978 and *S. aureus* RN4220, respectively, fine-tuned ApexOracle achieved mean AUPRC values of 0.712, 0.434 and 0.401, compared with baseline values of 0.545, 0.304 and 0.366. Fine-tuned ApexOracle significantly outperformed the matched baseline for all three targets (Holm-adjusted p = 0.0264, 0.0006 and 0.0264, respectively). Strict zero-shot ApexOracle achieved mean AUPRC values of 0.585, 0.325 and 0.166. Its performance was not significantly different from baseline for *E. coli* or *A. baumannii* (both Holm-adjusted p = 0.5427), but was significantly lower for RN4220 (p = 0.0006), demonstrating that zero-shot transfer remains target-dependent.

## 已同步到论文和 response letter 的修改

以下内容已同步到 `sn-article.tex` 的概括句、Fig. 1b 图注和 Results，以及 response letter 中所有仍引用旧 single-model 结果的段落；Methods 保持作者当前版本，不加入额外 Fig. 1b 协议：

1. Fig. 1b 从混合报告 AUPRC/AUROC 改为三个菌株统一报告 AUPRC-only；AUROC 可移到补充表。
2. 图中 point estimate 从旧的 pooled OOF/single-member sensitivity 改为五折 10-member ensemble 的 mean ± sample s.d.；不能再把 error bar 写成 paired-bootstrap 95% CI。
3. Results 中旧数值 `baseline 0.508/0.320/0.329`、`zero-shot 0.587/0.321/0.166` 和 `single-model fine-tuned 0.667/0.353/0.345` 已替换为本文件中的最终 fold-mean 数值。
4. 显著性结论更新为：fine-tuned 的三株 AUPRC 均显著高于 baseline。
5. 删除或收窄正文中 “zero-shot ApexOracle can match or exceed dedicated models” 的概括，改为 zero-shot transfer is target-dependent。
6. 图注、Results 和 response letter 已统一到最终 10-member ensemble 口径。
