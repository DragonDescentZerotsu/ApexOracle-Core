# Genome-representation reviewer validation

## 正式范围

本目录发布两个只消费现有 genome assets 的 representation-level 实验；不重训 ApexOracle，也不
重跑 Evo-2。

1. **Homologous-fragment variation**：在全部合格 bacterial embeddings 中为每个 genome 选择
   同物种 ANI 最近邻，比较 mutual-best homologous 11-kb fragments 的连续 sequence divergence
   与 embedding cosine distance，并单列 whole-genome `ANI >=99%` 子集。
2. **Fragment annotation probes**：把现有 GenBank annotations 保守地映射到 saved-tensor-compatible
   fragment windows，再以固定超参数的 L2 logistic regression 检验 AMR-associated 和
   mobile-element-associated signals 是否可由 frozen 8,192-dimensional vectors 线性恢复。

早期 genome/text condition-swap controls 和 strain-wise fragment pilot 已被作者明确排除在 reviewer
response 与论文之外。它们的本地取证产物继续保留，但代码、逐 member completion markers 和
superseded summaries 不进入公共 reviewer capsule，避免与正式证据混淆。

## 代码结构

| 层级 | Canonical 文件 | 职责 |
| --- | --- | --- |
| 共享协议 | `src/apexoracle/evaluation/genome_condition_reviewer.py` | window、annotation dictionary、ANI nearest-neighbor 与通用确定性 contracts |
| Annotation/probe | `src/apexoracle/evaluation/genome_fragment_validation.py` | FASTA/GenBank/tensor compatibility、fragment labels、固定五折 linear readouts |
| Pair preparation | `scripts/audit/prepare_all_genome_fragment_variation_pairs.py` | 运行 skani 并冻结 directed/unordered nearest same-species pairs |
| Fragment analysis | `scripts/audit/analyze_genome_fragment_variation.py` | minimap2 mutual-best alignment、edlib global divergence、embedding distances 与汇总 |
| Probe preparation | `scripts/audit/prepare_historical_genome_annotation_probes.py` | 生成 264-genome compatible fragment manifest；文件名保留为冻结产物的 legacy 名称 |
| Probe evaluation | `scripts/audit/run_historical_genome_annotation_probes.py` | 运行两个固定 L2 probes 并导出 OOF predictions/metrics |
| Figure | `scripts/audit/plot_genome_representation_validation.py` | 只读生成 Supplementary Fig. C6 三面板图和 exact plotted-data manifest |
| Tests | `tests/test_genome_condition_reviewer.py` | window、ANI selection、alignment、annotation 与 figure contracts |

正式入口不再依赖弃用的 condition-swap scripts，也不需要本机绝对路径下的外部 producer 源码。
所有 saved-window compatibility 逻辑均来自受测试的 `src/` 模块。

## Canonical 命令

准备 nearest-pair manifest（CPU；需要 skani）：

```bash
PYTHONPATH=src python scripts/audit/prepare_all_genome_fragment_variation_pairs.py \
  --skani /path/to/skani --threads 64
```

分析同源片段（CPU；需要 minimap2 和 Python `edlib`）：

```bash
PYTHONPATH=src python scripts/audit/analyze_genome_fragment_variation.py \
  --minimap2 /path/to/minimap2 --workers 16
```

准备并运行 fragment annotation probes：

```bash
PYTHONPATH=src python scripts/audit/prepare_historical_genome_annotation_probes.py
PYTHONPATH=src python scripts/audit/run_historical_genome_annotation_probes.py
```

从冻结结果重建三面板图：

```bash
MPLBACKEND=Agg PYTHONPATH=src python \
  scripts/audit/plot_genome_representation_validation.py
```

验证：

```bash
PYTHONPATH=src python -m pytest -q tests/test_genome_condition_reviewer.py
```

## 冻结协议

### Homologous-fragment variation

- bacterial genomes 必须同时具有 embedding、FASTA 和 species mapping，且同物种至少有两个合格
  genomes。
- nearest neighbor 要求 whole-genome ANI `>=95%`，并且两个方向 aligned fraction 都 `>=50%`；
  先逐 genome 选择 ANI 最大的 same-species neighbor，再把 reciprocal choices 去重为 unordered pairs。
- saved-window reconstruction 已与 567/567 tensor first-dimension shapes 核对一致；分析范围仅为 saved
  fragment condition，不外推到其余 sequence。
- full-length 11-kb windows 使用 minimap2 `asm5 -c --eqx -N 5` 双向比对；保留 same-orientation、
  mutual-best one-to-one matches、双端 coverage `>=80%` 和 MAPQ `>=20`。
- global divergence 为 edlib Needleman--Wunsch edit distance / 11,000；正式分析保留 divergence
  `<=5%` 的 pairs。
- embedding difference 为固定 `1e14` rescaling 后的 cosine distance；Spearman 只对 variable
  fragments 计算。每个 homolog 另与 donor genome 中一个确定性选择的 different-index fragment
  比较，作为距离尺度 control。

### Fragment annotation probes

- FASTA 与 GenBank 的 record 数、顺序和 sequence 必须完全一致，重建 window 数必须等于 tensor
  第一维，embedding width 必须为 8,192。
- labels 来自冻结保守词典应用于现有 GenBank feature type、gene、product、function、note 和
  `mobile_element_type`；不等同于完整 ARG/MGE catalogue。
- 每个 label 保留全部 positives，并在每个 positive-bearing genome 内至多采样每个 positive 五个
  negatives。
- 模型固定为 `LogisticRegression(C=1, class_weight="balanced", solver="liblinear")`；输入乘
  `1e14`，不做降维、MLP 或调参。
- 五折 `StratifiedGroupKFold` 把同一 genome 的所有 fragments 保持在同一 fold；报告 pooled OOF
  AUPRC/AUROC、evaluation-set prevalence 及 fold mean ± sample s.d.

## 正式结果与解释边界

- 255 个 nearest unordered strain pairs 中，166 pairs/53 species 产生 6,625 个合格 homologous
  fragments；4,649 个 variable fragments 的 pooled Spearman 为 `0.6954`。
- whole-genome `ANI >=99%` 子集的 pooled Spearman 为 `0.7137`；99.94% true homologous pairs
  比对应的 within-genome different-index reference 更近。
- 264 个 bacterial genomes、96,716 fragments 通过 annotation compatibility。AMR/MGE probe 的
  OOF AUPRC 为 `0.2033/0.4456`，evaluation prevalence 为 `0.1667/0.1977`；OOF AUROC 为
  `0.5775/0.7415`。
- 可支持的结论是 saved fragment representation 保留 strain-level sequence variation、较弱的
  AMR-associated signal 和更明显的 mobile-element-associated signal。不得升级为 complete-genome
  coverage、完整 resistome/MGE catalogue、single-gene attribution、功能因果或 MIC-head use。

完整数字和弃用内部 controls 的解释边界见 `RESULTS.md`。正式 reviewer 文稿实施记录见
`REVIEWER_RESPONSE_AND_MANUSCRIPT_DRAFT.md`。

## 输出与发布边界

- `fragment_variation/manifests/summary.json`：nearest-pair preparation contract。
- `fragment_variation_task_manifest.json`：正式 fragment analysis 命令、owner 与完成条件。
- `fragment_variation/all_embeddings/summary.json`：fragment analysis protocol 与结果。
- `historical_probe/manifests/manifest.json`：annotation-compatible cohort 与 output hashes。
- `historical_probe/task_manifest.json`、`historical_probe/analysis/summary.json`：probe protocol、
  completion contract 与正式 metrics。
- `figures/`：canonical PDF/SVG/PNG、caption、exact plotted-data 和 SHA-256 manifest；公共 release
  显式纳入三种 canonical figure files 和 compact exact plotted-data，raw prediction/fragment CSV
  仍受 `.gitignore` 保护。

本地 raw/replay/probe-prediction tables 约 249 MB，均可由上述入口或冻结 assets 重建，不进入 Git。
本轮公共 capsule 在提交前必须通过 100 MB 单文件检查、凭据扫描、focused/full tests、Black、Ruff、
JSON parse 和 `git diff --check`。

## 正式文稿落稿（2026-08-06）

Supplementary Fig. C6 已写入正式 manuscript；Methods、Results、figure caption、三条软件引用和
reviewer response 已同步。三条引用 DOI 为 `10.1038/s41592-023-02018-3`、
`10.1093/bioinformatics/bty191` 和 `10.1093/bioinformatics/btw753`。论文独立编译为 34 页，C6
位于第 26 页且无 undefined citation/reference；response DOCX 独立渲染为 31 页，目标回复位于
第 7--9 页。正式 `sn-article.pdf` 未覆盖，修改前文件均已备份。
