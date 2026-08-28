# Peptide-classifier 论文修改记录

> 状态：**正式 TeX 已落实；正式 response letter 尚未同步**
> 更新日期：2026-08-03

## 已确认范围

- 只在 Methods 增加 peptide-classifier training/evaluation 描述；
- 不修改 Results；
- 不新增 Supplementary Table 或 figure；
- 新增文字使用 `\rev{}` 标记；
- 正式论文 PDF 不覆盖。

## 正式插入位置

正式论文：

`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`

Methods 的局部顺序已按作者确认调整为：

1. `Remask during generation`；
2. `Sampling Strategy`；
3. `Remasking-window sensitivity analysis`；
4. `Peptide-classifier training and evaluation`；
5. `Implementation Details`。

本轮先把两个相关段落移到 `Sampling Strategy` 与 `Implementation Details` 之间，随后按作者确认
将 `Remasking-window sensitivity analysis` 放在 classifier 段之前，使 window comparison 紧接
其 sampling 定义；两段均为完整移动，没有改写内容。

## 已写入的 Methods 文字

> **Peptide-classifier training and evaluation.** Because exhaustive structure-level peptide
> annotation was unavailable at this scale, classifier labels were assigned by data source. The
> positive class comprised 677,323 SmProt2 records, 3,105,732 UniProt/UniRef records and 6,654,492
> PeptideCLM-generated records, whereas 72,357,504 PubChem records were used as a broad
> general-chemistry background class. PubChem was not assumed to be strictly peptide-free;
> therefore, these source-based labels constitute operational weak supervision and may contain
> label noise. The classifier was used only to guide sampling toward the peptide-enriched training
> domain, rather than to definitively determine whether a generated structure was a peptide. For
> performance evaluation, we constructed one fixed 98%/1%/1% train/validation/test split containing
> 81,137,648, 830,022 and 827,381 records, respectively. Exact canonical molecules were restricted
> to a single split. Peptide sequences were clustered with MMseqs2 at 40% sequence identity and 80%
> coverage, and each complete sequence cluster was assigned to only one split. Together, these
> constraints remove near-duplicate leakage across splits and allow the classifier to be evaluated
> on peptides dissimilar to those used for training. The DDiT backbone
> was held fixed, and the binary classification head was trained independently in two replicate
> runs. The held-out test set contained 741,260 non-conflicting canonical molecules, including
> 101,051 positive and 640,209 negative molecules. The positive test cohort included records from
> all three positive sources, including PeptideCLM-generated records. At \(t=0\), corresponding to
> clean inputs, both replicas achieved AUROC, AUPRC and positive-class recall of 1.000, with
> negative-class specificities of 0.999991 and 0.999992. At \(t=0.5\), using ten fixed masking
> replicates per molecule, both replicas again achieved AUROC, AUPRC and positive-class recall of
> 1.000, with negative-class specificities of 0.999989 and 0.999991.

## 验证结果

- LaTeX 必要转义已加入：`98\%/1\%/1\%`、`40\%`、`80\%`；英文内容未改写。
- 段落顺序调整后，在独立临时输出目录使用
  `latexmk -pdf -interaction=nonstopmode -halt-on-error` 重新编译成功。
- 临时 PDF 为 32 页；相关段落位于第 15--16 页附近。
- 编译未出现 LaTeX error；正式 `sn-article.pdf` 未覆盖。

## 尚待执行

- 正式 response letter 尚未同步回指新增 Methods；
- response 中残留的中文内部备注尚未删除；
- response 的 PubChem 和指标术语尚未与本段统一。
