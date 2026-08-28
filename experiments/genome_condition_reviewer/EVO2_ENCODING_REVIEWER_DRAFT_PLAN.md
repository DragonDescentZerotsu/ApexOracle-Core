# Evo-2 genomic encoding：最小修改 draft plan

> **状态：讨论稿；截至 2026-08-28，正式 TeX 已包含 window motivation 和未做 systematic
> layer sweep 两项修改，但 pooling 与 Evo-2 example 的关系仍未写入，DOCX 仍待同步。**
>
> 原则：以现有 reviewer response 和正式正文为基础，只改 reviewer 明确追问但正文尚未明确回答的
> 内容；非必要句子全部保持不动。本计划不新增实验、不重复 Reviewer 1/Supplementary Fig. C6 的
> 结果，也不重写现有段落。

## 1. 最小修改结论

只做以下修改：

1. Reviewer 2 的 Evo-2 主回复只改 pooling 段中的 **1 句话**。
2. 正文 `Genome Embeddings` 只做 **3 处句级修改**：
   - 替换 window-motivation 的第 1 句；
   - 插入“未做 systematic layer sweep”的 1 句；
   - 在 mean pooling 句前明确其与 Evo-2 feature-extraction example 的关系。
3. Reviewer 2 下一条总括回复只替换 **2 句话**，消除“没有 ApexOracle-specific changes”与前文
   fragmentation/pooling/fusion 描述之间的矛盾。
4. 其余 reviewer response、正文 equations、Supplementary Fig. C6 和 validation paragraphs
   全部不改。

## 2. Reviewer 2 Evo-2 主回复

### 2.1 完整拟提交版本

下面保留现有回复，只将 pooling 段的第二句替换为最小澄清版本。为便于直接复制，这里给出完整回复。

> We thank the reviewer for asking us to clarify the Evo-2 genome-embedding design. We have
> revised the Methods section, “Genome Embeddings”, to distinguish the design choices
> inherited from Evo-2 from ApexOracle-specific choices and to motivate the fragmentation,
> pooling, and layer-selection decisions.
>
> The sliding-window fragmentation was not used because Evo-2-40B lacks long-context
> capacity. Evo-2-40B can process sequences up to 1 million nucleotides, but our downstream
> objective was not to collapse an entire bacterial contig into a single coarse
> representation. ApexOracle uses genome embeddings as a fragment-indexed key–value bank in
> cross-attention, and the interpretability analysis depends on retaining fragment-level
> genomic resolution. We therefore chose 11,000-nt windows with a 10,000-nt step to preserve
> approximately gene-cluster-scale resolution, with a 1,000-nt overlap to reduce gene
> truncation at window boundaries. In principle, single-gene-scale chunks (~1,000 nt) would
> offer higher resolution, but this would increase the number of fragments by roughly an
> order of magnitude and was computationally infeasible at our scale for downstream
> cross-attention.
>
> For pooling, Evo-2 provides nucleotide-level hidden states from selected layers;
> ApexOracle then uses length-normalized mean pooling over nucleotides to summarize each
> fragment into one fixed-size fragment vector. **This within-fragment mean pooling follows
> the sequence-embedding procedure in the Evo-2 feature-extraction example; the
> ApexOracle-specific choice is that we apply it to 11-kb fragments and do not pool across
> fragments.** The full genome condition remains an \(M\times d_g\) fragment matrix and is
> used directly as the genome-derived key–value bank in cross-attention.
>
> For layer selection, the choice was guided by the Evo-2 feature-extraction tutorial rather
> than by an ApexOracle-specific validation-set sweep. The tutorial extracts embeddings from
> a late, non-final Hyena-short block in the 7B model. We therefore used the same principle
> for Evo-2-40B and selected layer 46, a near-top, non-final Hyena-short layer, to obtain
> context-dependent sequence features while avoiding both static input embeddings and final
> logits. We did not perform a systematic layer search because extracting and storing
> multiple Evo-2-40B layers for hundreds of complete bacterial genomes would have
> substantially increased the computational cost.

### 2.2 唯一修改的句子

**现有句子：**

> This pooling is a downstream representation choice consistent with using Evo-2 layer
> outputs as embeddings, whereas the key ApexOracle-specific point is that we do not pool
> across fragments.

**替换为：**

> This within-fragment mean pooling follows the sequence-embedding procedure in the Evo-2
> feature-extraction example; the ApexOracle-specific choice is that we apply it to 11-kb
> fragments and do not pool across fragments.

**其余主回复不修改。**

## 3. 正文 `Genome Embeddings` 的三处最小修改

正式文件：

`/data2/tianang/projects/ApexOracle_cleaned/docs/ApexOracle_Nat_Biotech/sn-article.tex`

### 3.1 修改 1：window motivation

位置：当前 `Genome Embeddings` 第二段第一句。

**现有句子：**

> The 11-kb window spans approximately ten average-length bacterial genes and was selected as
> a computationally tractable gene-cluster-scale condition.

**替换为：**

> Although Evo-2-40B supports sequences of up to 1 million nucleotides, the 11-kb window,
> spanning approximately ten average-length bacterial genes, was selected to preserve a
> computationally tractable gene-cluster-scale condition rather than collapse a long contig
> into one coarse representation.

**本段其余句子不修改。** 现有后续内容已经解释约 1-kb windows 会使 fragment key–value pairs、
storage 和 downstream cross-attention cost 增加约一个数量级。

### 3.2 修改 2：明确未做 systematic layer sweep

位置：当前 layer 46 justification 之后、mean pooling 之前。

在下面现有句子之后：

> Consistent with the empirical practice that near-top, non-final hidden states often provide
> higher-quality general-purpose representations than final output representations, we
> selected the analogous near-top, non-final Hyena-short layer in Evo-2-40B.

**只插入这一句：**

> No ApexOracle-specific systematic layer sweep was performed because extracting and storing
> multiple Evo-2-40B layers for hundreds of bacterial genomes would have substantially
> increased the computational cost.

### 3.3 修改 3：明确 pooling 与 Evo-2 example 的关系

位置：紧接修改 2。

**现有句子：**

> The nucleotide-level hidden states were length-normalized mean-pooled to obtain one
> \(d_g=8{,}192\)-dimensional vector per fragment.

**替换为：**

> Following the Evo-2 feature-extraction example, the nucleotide-level hidden states were
> length-normalized mean-pooled to obtain one \(d_g=8{,}192\)-dimensional vector per fragment.

下一句保持不动：

> No pooling was performed across fragment vectors.

### 3.4 三处修改后的完整正文段落

以下仅用于整体检查；正式修改时仍按上面三处逐句操作。

```tex
\textbf{Genome Embeddings.} \rev{Genome assemblies were supplied as FASTA files and processed
using a sliding-window implementation with a window length of 11,000 nt and a step of
10,000 nt. The 1,000-nt overlap reduces the likelihood that a feature crossing a window
boundary is represented only as a truncated sequence.}

\rev{Although Evo-2-40B supports sequences of up to 1 million nucleotides, the 11-kb window,
spanning approximately ten average-length bacterial genes, was selected to preserve a
computationally tractable gene-cluster-scale condition rather than collapse a long contig
into one coarse representation. Reducing the window length to approximately 1 kb would
increase the number of fragment key--value pairs by roughly an order of magnitude. Because
no pooling is performed across fragments, this would increase key--value storage and the
principal downstream genome cross-attention compute and memory costs by approximately the
same order of magnitude. This representation therefore targets fragment-level biological
context and is not intended for single-gene attribution.}

\rev{Each encoded fragment was passed independently through Evo-2-40B\cite{evo2}, and
nucleotide-level hidden states were extracted from layer 46. The choice of layer 46 was
guided by the Evo-2 feature-extraction tutorial, which uses a late, non-final Hyena-short
block from the 7B model for embedding extraction. Consistent with the empirical practice
that near-top, non-final hidden states often provide higher-quality general-purpose
representations than final output representations, we selected the analogous near-top,
non-final Hyena-short layer in Evo-2-40B. No ApexOracle-specific systematic layer sweep was
performed because extracting and storing multiple Evo-2-40B layers for hundreds of bacterial
genomes would have substantially increased the computational cost. Following the Evo-2
feature-extraction example, the nucleotide-level hidden states were length-normalized
mean-pooled to obtain one $d_g=8{,}192$-dimensional vector per fragment. No pooling was
performed across fragment vectors.}
```

现有 equations、\(M\times d_g\) matrix、cross-attention key–value bank、固定
\(10^{14}\) rescaling 和 `Genome-representation validation` 均保持不动。

## 4. Reviewer 2 下一条总括回复的两处最小修改

Reviewer comment：

> More generally, for each component borrowed from an existing model (Evo2, Me-LLaMA,
> MDLM), it would be helpful to clearly distinguish which design choices are adopted directly
> from the original approach and which are adjusted for ApexOracle, with justification and if
> possible sensitivity analysis for the latter.

### 4.1 修改第 1 句

**现有句子：**

> Thus, there are no ApexOracle-specific changes to the borrowed Evo-2 or Me-LLaMA model
> components that would require a separate sensitivity analysis.

**替换为：**

> Thus, the pretrained Evo-2 and Me-LLaMA backbones are unchanged, while the fragmentation,
> pooling, and downstream fusion described above are ApexOracle-specific processing choices.

### 4.2 修改紧接着的第 2 句

**现有句子：**

> Among the borrowed model components, the only adjustment made for ApexOracle is in the
> MDLM/DLM pretraining procedure.

**替换为：**

> Among the borrowed model backbones, the only component modified during training is the
> MDLM/DLM.

### 4.3 两句修改后的完整回复

> We thank the reviewer for this suggestion. We clarify the distinction as follows: the
> pretrained Evo-2 and Me-LLaMA components are adopted directly as frozen feature extractors
> and are not re-trained or architecturally modified for ApexOracle. Evo-2 is used to obtain
> genome hidden-state embeddings, and Me-LLaMA is used to obtain text embeddings from strain
> descriptions; ApexOracle-specific processing occurs downstream when these embeddings are
> fused with molecular representations. **Thus, the pretrained Evo-2 and Me-LLaMA backbones
> are unchanged, while the fragmentation, pooling, and downstream fusion described above are
> ApexOracle-specific processing choices. Among the borrowed model backbones, the only
> component modified during training is the MDLM/DLM.** Specifically, while we retain the
> MDLM absorbing-state masked diffusion formulation and SUBS parameterization, we adapt the
> molecular model to SELFIES inputs and add the multi-target regression objective on RDKit
> molecular descriptors so that the learned molecular embeddings support both generation
> and MIC/synergy prediction. This ApexOracle-specific regression component, its motivation,
> and its effect on downstream MIC prediction are already described in detail in the Results
> and Methods sections, including Fig. 2a,b and the DLM training objective.

## 5. 明确不修改的内容

- 不增加新的实验或 sensitivity analysis。
- 不向 Reviewer 2 主回复重复 Reviewer 1/Supplementary Fig. C6 的详细结果。
- 不修改 `Hyena-short` 表述。
- 不在对外回复中展开具体 GPU 或设备限制。
- 不修改 Reviewer 1 已完成的正式回复。
- 不修改正文 equations、rescaling、genome-representation validation 或 figure/caption。
