# Fig. 1b reviewer response：最终 10-member AUPRC 版本

## 可直接粘贴到 response letter 的英文回复

**Response:** We thank the reviewer for highlighting the inconsistency in the original Fig. 1b and
agree that AUPRC is the more informative primary metric for these imbalanced antibacterial-activity
datasets. We have therefore revised Fig. 1b to report AUPRC consistently for all three target strains,
rather than mixing AUPRC and AUROC across strains.

To make the comparison like-for-like, we used the same fixed five outer molecule folds for every
method. Within each fold, the fine-tuned ApexOracle model and the corresponding Chemprop baseline
were each evaluated as a 10-model ensemble. Strict zero-shot ApexOracle was also a fixed 10-model
ensemble and was evaluated on the same five test partitions without using target-strain samples for
gradient updates. For the *A. baumannii* comparison, we used the no-RDKit-feature version reported
by Liu et al., because ApexOracle likewise does not use RDKit descriptor augmentation. Bars in the
revised figure show the mean AUPRC across the five outer folds, and error bars show the sample standard
deviation across folds.

We also clarified an important limitation of the paper-era ApexOracle protocol. Although target-strain
samples were excluded from gradient updates in the zero-shot setting, held-target AUROC was used to
select the best ApexOracle checkpoint. The fine-tuned ApexOracle runs likewise used the held outer fold
for epoch-level checkpoint selection, whereas each Chemprop baseline used an internal validation split
drawn only from the outer training data. We retained this behavior to reproduce the original analysis,
but now disclose it explicitly and do not describe the comparison as a fully label-free prospective
evaluation or as having identical model-selection procedures.

For *E. coli* BW25113, the Chemprop baseline, zero-shot ApexOracle and fine-tuned ApexOracle achieved
AUPRC values of 0.545 ± 0.133, 0.585 ± 0.137 and 0.712 ± 0.147, respectively. For *A. baumannii*
ATCC 17978, the corresponding values were 0.304 ± 0.022, 0.325 ± 0.066 and 0.434 ± 0.053. For
*S. aureus* RN4220, they were 0.366 ± 0.048, 0.166 ± 0.053 and 0.401 ± 0.035. Values are mean ±
sample standard deviation across the five fixed outer folds.

We additionally compared methods on identical sample-level out-of-fold predictions using 5,000
class-stratified paired bootstrap resamples and 5,000 two-sided paired prediction-swap randomizations,
with Holm correction across the three strains within each model mode. Fine-tuned ApexOracle improved
pooled out-of-fold AUPRC over the corresponding baseline by 0.147 for *E. coli* (95% CI,
0.055–0.245; Holm-adjusted p = 0.0264), 0.124 for *A. baumannii* (95% CI, 0.081–0.162;
p = 0.0006), and 0.060 for RN4220 (95% CI, 0.021–0.097; p = 0.0264). In the strict zero-shot
setting, the differences were 0.044 for *E. coli* (95% CI, −0.065–0.151; p = 0.5427), 0.029 for
*A. baumannii* (95% CI, −0.016–0.073; p = 0.5427), and −0.169 for RN4220 (95% CI,
−0.214–−0.126; p = 0.0006). Thus, fine-tuned ApexOracle significantly outperformed the matched
baseline for all three strains, whereas zero-shot performance was target-dependent: it was not
significantly different from baseline for *E. coli* or *A. baumannii* and was significantly lower for
RN4220. We have revised the Results and figure legend accordingly and removed the previous blanket
claim that zero-shot ApexOracle generally matched or exceeded all dedicated strain-specific models.

## 建议替换的 Fig. 1b 图注

**b,** Small-molecule antibiotic classification evaluated consistently by AUPRC for all three target
strains. Bars show mean AUPRC and error bars show sample standard deviation across five fixed outer
molecule folds. ApexOracle fine-tuned and each matched Chemprop baseline use 10-model ensembles per
fold; ApexOracle zero-shot uses a fixed 10-model ensemble evaluated on the same five test partitions
without target-strain samples in gradient updates. Under the frozen paper-era ApexOracle protocol,
held-target AUROC was used for checkpoint selection; this differs from the baseline internal-validation
procedure. Brackets show Holm-adjusted p values from two-sided paired
prediction-swap tests on identical pooled out-of-fold samples. The *A. baumannii* baseline uses the
no-RDKit-feature Liu et al. profile to match the absence of RDKit descriptor augmentation in
ApexOracle.

## 建议替换的 Results 段落

Beyond peptides, we evaluated small-molecule antibiotic activity prediction against three matched
strain-specific Chemprop baselines using identical fixed fivefold molecule partitions and 10-model
ensembles (Fig. 1b). AUPRC was used as the primary metric because the active class was sparse in all
three datasets. For *E. coli* BW25113, *A. baumannii* ATCC 17978 and *S. aureus* RN4220,
respectively, fine-tuned ApexOracle achieved mean AUPRC values of 0.712, 0.434 and 0.401, compared
with baseline values of 0.545, 0.304 and 0.366. Fine-tuned ApexOracle significantly outperformed the
matched baseline for all three targets (Holm-adjusted p = 0.0264, 0.0006 and 0.0264, respectively).
Strict zero-shot ApexOracle achieved mean AUPRC values of 0.585, 0.325 and 0.166. Its performance was
not significantly different from baseline for *E. coli* or *A. baumannii* (both Holm-adjusted
p = 0.5427), but was significantly lower for RN4220 (p = 0.0006), demonstrating that zero-shot
transfer remains target-dependent.

## Methods 中需要替换的实验口径

The small-molecule benchmark used one fixed fivefold molecule-level partition per strain
(KFold with shuffling and random state 42), with identical outer test membership for all methods.
Within each outer fold, each Chemprop baseline was trained as a 10-model ensemble; a stratified 12.5%
subset of the outer training data was used for model selection, and the outer test fold was never used
for baseline checkpoint selection. Fine-tuned ApexOracle likewise used 10 members per outer fold.
Strict zero-shot ApexOracle used a fixed 10-model ensemble; target-strain samples were excluded from
gradient updates, and the same zero-shot ensemble was
evaluated separately on each of the five outer test partitions to estimate fold-to-fold variability.
To preserve the paper-era ApexOracle behavior, held-target AUROC was used for epoch-level best-checkpoint
selection in both the zero-shot and fine-tuned ApexOracle runs. This differs from the Chemprop baselines,
which used only the outer-training internal validation subset for checkpoint selection, and is reported
as a limitation. The Liu et al. comparison used its reported no-RDKit-feature profile
because ApexOracle did not use RDKit descriptor augmentation. Bars report the mean and sample standard
deviation of AUPRC across the five outer folds. Statistical comparisons used pooled sample-level
out-of-fold predictions: 5,000 class-stratified paired bootstrap resamples estimated confidence
intervals, and 5,000 two-sided paired prediction-swap randomizations tested differences. P values were
Holm-corrected across the three strains separately within each model mode.

## 相对当前论文草稿必须同步的修改

以下是根据现有 `sn-article.tex` 和 response letter 检查得到的事实，不表示这些文件已被本脚本自动改写：

1. Fig. 1b 从混合报告 AUPRC/AUROC 改为三个菌株统一报告 AUPRC-only；AUROC 可移到补充表。
2. 图中 point estimate 从旧的 pooled OOF/single-member sensitivity 改为五折 10-member ensemble
   的 mean ± sample s.d.；不能再把 error bar 写成 paired-bootstrap 95% CI。
3. Results 中旧数值 `baseline 0.508/0.320/0.329`、`zero-shot 0.587/0.321/0.166` 和
   `single-model fine-tuned 0.667/0.353/0.345` 必须替换为本文件中的最终 fold-mean 数值。
4. Methods 中的 “one Chemprop model per outer fold” 和 “exactly one ApexOracle ensemble member
   per fold” 必须改成每个 outer fold 固定 10-member ensemble；single-member 结果只保留为
   sensitivity，不再作为 Fig. 1b 主结果。
5. 明确披露 *A. baumannii* 使用 Liu et al. 的 no-RDKit-feature profile，而不是其 RDKit descriptor
   增强主模型。
6. 显著性结论更新为：fine-tuned 的三株 AUPRC 均显著高于 baseline；zero-shot 在 *E. coli* 和
   *A. baumannii* 与 baseline 无显著差异，在 RN4220 显著更低。
7. 删除或收窄正文中 “zero-shot ApexOracle can match or exceed dedicated models” 的概括，改为
   zero-shot transfer is target-dependent。
8. 图注、Results、Methods 和 response letter 必须同时更新，避免继续混用旧 sensitivity 与最终
   10-member ensemble 口径。
9. 不能继续写成 zero-shot “完全没有使用目标标签”：目标样本没有参与梯度更新，但历史 runner
   用 held-target AUROC 选择 checkpoint；fine-tuned 同样用 outer held fold 做 epoch selection，
   而 baseline 使用 outer-train 内部 validation。该非对称 model-selection protocol 必须作为限制披露。
