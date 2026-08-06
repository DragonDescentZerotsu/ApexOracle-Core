# Genome-condition reviewer results

## 正式结论

本轮用于 reviewer response 和论文修订的证据由两个 representation-level validation 构成：

1. homologous-fragment variation：检验近缘同种菌株的同源 11-kb fragments 中，真实序列变异是否
   会在冻结的 Evo-2 fragment vectors 中留下可测差异；
2. fragment linear probes：检验 nucleotide-level mean pooling 后，AMR-associated 与
   mobile-element-associated annotation signal 是否仍可由最简单的线性 readout 恢复。

这些实验支持 frozen fragment representation 保留 encoded fragment level 的 sub-species sequence
variation 和部分 annotation-associated information。它们不证明完整 genome coverage、完整 ARG/MGE
catalogue、single-gene attribution、功能因果或 MIC prediction head 对每个 fragment 的使用。

## Homologous-fragment variation

### Cohort 与协议

在全部 567 个 embedding/FASTA/species-compatible assets 中，539 个为 bacterial genomes；排除
singleton-species genomes 后，379 个 genomes 来自 71 个 multi-strain species。360 个 genomes 在
whole-genome ANI `>=95%`、两个方向的 aligned fraction 均 `>=50%` 的条件下找到同物种最近邻，去重
后得到 255 个 unordered strain pairs（67 species）。这些 pair 的 whole-genome ANI 中位数为
99.36%（范围 95.05%--100%）。

Fragment comparison 只分析 saved tensor 对应的 genomic segments。Full-length 11-kb windows 经过
双向 alignment 后，仅保留 mutual-best、same-orientation、两端 coverage `>=80%`、MAPQ `>=20` 且
global edit fraction `<=5%` 的同源 pair。最终 166 个 strain pairs、53 species 产生 6,625 个
homologous fragment pairs，其中 1,976 个序列相同、4,649 个含序列变异。

### 结果

| Scope | Strain pairs | Fragment pairs | Variable pairs | Pooled Spearman: sequence divergence vs cosine distance | Median per-pair Spearman | Positive per-pair correlations | Homologous closer than within-donor reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All included pairs | 166 | 6,625 | 4,649 | 0.695 | 0.377 | 89.4% (84/94) | 99.94% |
| Whole-genome ANI `>=99%` | 117 | 6,132 | 4,156 | 0.714 | 0.370 | 89.3% (67/75) | 99.93% |

Sequence-identical pairs 的 cosine distance 为数值零（median `3.3e-16`）；4,649 个 variable pairs
全部为非零距离，median 为 `7.39e-06`。同源 fragment 的 median cosine distance 为 `3.90e-06`，
而打破同源对应关系、改与同一 donor genome 中另一 fragment 比较时，median distance 为
`9.95e-03`。

正式图的 panel a 只展示 4,649 个 variable-fragment points，不使用 divergence bins 或 fitted trend
line。连续 divergence 值上的 Spearman correlation 是主要统计量。

### 解释边界

该实验直接说明：在实际保存的近缘菌株同源 fragments 中，序列差异会引起 frozen mean-pooled Evo-2
vectors 的可测变化，而且该关系在 whole-genome ANI `>=99%` 的高度近缘子集中仍成立。因此结果不是
只有 species-level separation。该实验不是 whole-strain classifier，也不评价未进入 saved condition
的 sequence 或 downstream MIC use。

## Fragment annotation probes

### Cohort 与协议

264 个 bacterial genomes 同时满足 saved windows、FASTA/GenBank sequence 与 record order、embedding
tensor shape 的精确兼容条件，共包含 96,716 个 fragments。保守冻结词典根据既有 GenBank feature
type、gene name 和 product description 标记 AMR-associated 与 mobile-element-associated fragments。

每个 annotation 使用独立的固定超参数 L2 logistic regression（`C=1`、
`class_weight=balanced`、`liblinear`）。Embeddings 保持冻结，不进行 hidden-layer training、降维或
调参。五折评估中，同一 genome 的所有 fragments 始终处于同一个 fold，避免同一 genome 的重叠
windows 同时进入 train 和 test。每个含阳性的 genome 纳入全部 positives，并最多按 5:1 纳入
negatives。

### 结果

| Annotation | Full positives / prevalence | Evaluation positives / prevalence | OOF AUPRC | OOF AUROC | Fold AUPRC mean +/- sample s.d. | Fold AUROC mean +/- sample s.d. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMR-associated | 1,217 / 1.26% | 1,217 / 16.67% | 0.2033 | 0.5775 | 0.2068 +/- 0.0080 | 0.5790 +/- 0.0193 |
| Mobile-element-associated | 8,843 / 9.14% | 8,843 / 19.77% | 0.4456 | 0.7415 | 0.4509 +/- 0.0605 | 0.7445 +/- 0.0371 |

AUPRC 的随机基线为实际 evaluation set 的 positive prevalence（16.67%/19.77%）。因此
AMR-associated signal 较弱但高于基线，mobile-element-associated signal 更明显。标签来自已有
annotation 的保守词典，不构成完整 resistome 或 mobile-element catalogue。

## Figure 与正式文稿

Canonical 三面板图、caption、exact plotted data 与 source/output SHA-256 manifest 位于
`figures/`。Panel a 展示 sequence divergence 与 embedding cosine distance；panels b/c 分别展示
AUPRC 与 AUROC 的五折 held-out evaluation 和随机基线。

正式 reviewer response、Methods、Results、caption 和新增引用已经按上述结果更新；当前落稿记录见
`REVIEWER_RESPONSE_AND_MANUSCRIPT_DRAFT.md`。对外表述统一限定为：fragment representation
preserves encoded sub-species sequence variation and annotation-associated information。

## 弃用诊断的维护边界

早期 genome-swap、genome/text factorial swap、162-genome probe pilot 和 strain-wise fragment pilot
只保留为本地审计历史，不进入 reviewer response、论文、正式三面板图或 GitHub release。Swap
diagnostics 无法构造已知的 donor-strain MIC 反事实，也没有提供可归因于 genome channel 的有效
证据；因此不得用它们声称 swap 降低预测性能或当前 MIC predictor 强依赖正确近缘 genome。
