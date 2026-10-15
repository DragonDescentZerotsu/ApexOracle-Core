# Guided-generation candidate diversity

本目录分析生成候选的多样性、独立分子结构数量和 Tanimoto similarity distribution。分析只使用最终候选对应的两个 MIC-guided targets
`BAA-3170` 与 `BAA-3197`，不加入 unguided 或 `gamma_peptide=0` control。

Canonical 入口：

```bash
MPLBACKEND=Agg PYTHONPATH=src \
  python \
  scripts/audit/analyze_generated_candidate_diversity.py
```

24 条最终合成 peptides 的 pairwise sequence/structure diversity 入口：

```bash
PYTHONPATH=src python \
  scripts/audit/analyze_selected_peptide_diversity.py
```

验证命令：

```bash
PYTHONPATH=src python -m pytest -q tests/test_generated_candidate_diversity.py
```

该入口输出到 `selected_peptides_24/`。Sequence PID 复用论文 lead-versus-training 的
chirality-aware BLOSUM62 global alignment、gap penalties、case-sensitive matches 和 cyclic
rotation 协议；无序 pair 额外对称化 target/query orientation，以消除 first-optimal alignment tie
对 ID 顺序的依赖。Linear--cyclic pairs 按原论文协议不作 sequence 比较；Tanimoto 覆盖全部 pairs。
该入口同时输出 `selected_peptide_pairwise_similarity_violin.{pdf,png,svg}`：左侧为 168 个
同拓扑 PID pairs，右侧为全部 276 个 Morgan/Tanimoto pairs，并以相同的 0--1 纵轴展示。
另输出 `selected_peptide_pairwise_similarity_by_strain_topology.{pdf,png,svg}`，严格只比较同一
generation target 且同一 topology 的 87 个 pairs。四组的 pair 数为 66、3、15、3；原来 168 个
同拓扑 PID pairs 中的 81 个跨 target-strain pairs 在该视图中排除。

## 两个冻结分析集合

### Candidate level

本地 `canonical_candidates/selfies_files/` 是历史
`generated_mol_SELFIES_w_mic-new/` 的 73-row 分析副本，保留原始
`strain × target length × source line` 布局。它可由 canonical audit 入口从外部只读输入重建，
8 个最近生成
前体被最终合成序列替换；其中 ApexOracle-14 的前体在两个 target lengths 重复出现，因此实际
替换 9 行。其余 64 行保持原始生成结构不变。

`canonical_candidates/candidates_73.csv` 同时保存 original/corrected sequence、SELFIES、
canonical isomeric SMILES、replacement status 和逐文件 SHA-256。所有后续 candidate-level
统计指标必须消费这份副本，不再直接消费历史目录。24 条最终合成 peptide 均通过
topology-aware exact mapping 在副本中出现；环肽允许 cyclic rotation。

Candidate-level 主分析保留全部 73 行，包括重复候选，并计算全部
`73 choose 2 = 2,628` 个无序非自身 pair。另报告 canonical isomeric structure 去重数。

### Candidate screening 血缘边界

**已由历史代码验证的事实：** `/path/to/ApexOracle-MDLM/judge_mol_mic_with_fig.py`
使用 clean 13-epoch MIC regressor 对 target strain 评分，通过 `if mic > 15: continue` 保留
predicted MIC <=15 µM 的 outputs；随后要求结构能够解析为 peptide sequence 且不含 unresolved
`X` residue，并要求结构可由 RDKit/SELFIES 继续处理。predicted MIC 因而是 post-generation
computational prioritization criterion，不是实验 MIC。

**根据现有资产作出的推断：** `generated_mol_SELFIES_w_mic-new/` 的 73 条 target-strain rows
是该 screening stage 保存的 peptide candidate pool；其 target、length 布局与原始 84,226 条
MIC-guided outputs 以及上述历史筛选代码一致。当前没有找到冻结的 producer command/log 或逐条
predicted-MIC table，因此无法确认现有脚本是这 73 条结果的 byte-exact producer；目前可核验的是
筛选逻辑与分母关系，逐条历史 prediction 尚未恢复。

从 73 条 pool 选择 24 条合成时，按 structural/sequence diversity 与 practical synthesis
feasibility 进行人工 prioritization，并降低过度疏水或含 aggregation-prone motifs 的候选优先级；
predicted MIC 不得写成实验 activity。

### Generation level

Generation-level 分析只读消费原始 `generated_mol_SELFIES-new/` 中 `BAA-3170` 与
`BAA-3197` 的全部 target lengths，共 84,226 条保存输出；不同长度直接 pooled。目录中另有
107 条 `BAA-1556` 输出，不属于本次候选对应 targets，已由显式 strain allowlist 排除。

对 84,226 条输出精确计算 canonical isomeric structure 去重数。全体两两组合超过 35 亿，
正式 Tanimoto distribution 已在 使用 64 workers 流式穷举全部
`84,226 choose 2 = 3,546,967,425` 个无序非自身 pairs；只累计 1,000-bin histogram、矩和阈值
计数，不保存 13.2 GiB 的逐 pair float32 数组。固定 seed `20260802` 的 1,000,000-pair 初步抽样
仍保留在 `summary.json` 作为抽样稳定性检查，主结果采用全量 pair 计算。

## 指纹与输出

两个层级使用相同的 RDKit Morgan bit fingerprint：radius 2、2,048 bits、
`includeChirality=True`，相似度为 Tanimoto。主要产物：

- `summary.json`：集合大小、唯一结构数、分位数和高相似度 pair 比例；
- `candidate_pairwise_tanimoto.csv`：73 行候选的全部 2,628 个 pair；
- `generation_all_pairs_tanimoto_histogram.csv`：84,226 条 outputs 的全部 3,546,967,425 pairs，
  以 0.001-wide bins 保存 exact counts；
- `generation_all_pairs_tanimoto_summary.json`：exact moments、quantile-containing bins、阈值计数、
  运行环境 和 cache/runner SHA-256；
- `tanimoto_histogram_plotted_data.csv`：图中精确 bin counts/fractions；
- `generated_candidate_diversity_three_panel.{pdf,svg,png}`：正式 Supplementary Fig. C5，包含
  selected-24 PID、candidate-pool Tanimoto 和 generation-output Tanimoto；
- `canonical_candidates/replacement_audit.csv`：9 个替换行；
- `canonical_candidates/final_peptide_mapping.csv`：24 条最终合成 peptide 的副本映射；
- `manifest.json`：全部输入、输出、外部 parser/PepLink 与 SHA-256 血缘。

## 当前已验证结果

- Peptide candidate pool：73 rows、71 个 structurally distinct canonical isomeric
  structures（97.26% unique）；全部 2,628 pairs 的 median Tanimoto 为 0.375，95th percentile
  为 0.695，`Tanimoto >= 0.8` 为 0.685%。
- Guided generation outputs：84,226 rows、83,433 个 distinct structures（99.06% unique）；
  全部 3,546,967,425 pairs 的 mean Tanimoto 为 0.2079，median 位于 `[0.189, 0.190)`，
  95th percentile 位于 `[0.411, 0.412)`，`Tanimoto >= 0.5` 为 0.8252%。
- 24 条最终实验 peptides：24 个 distinct canonical isomeric structures。168 个同拓扑 sequence
  pairs 的 median PID 为 `0.1719`，167/168 低于 `0.5`；唯一高相似 sequence pair 为
  ApexOracle-14/ApexOracle-23（`17/19 = 0.8947`）。全部 276 个结构 pairs 的 median Tanimoto
  为 `0.4633`，15/276 为 `>=0.7`、2/276 为 `>=0.8`、0/276 为 `>=0.9`；完整解释边界见
  `selected_peptides_24/REPORT.md`，双 panel 分布图见
  `selected_peptides_24/selected_peptide_pairwise_similarity_violin.pdf`。
- Target-stratified 24-peptide view：按 `target strain × topology` 分组的 87 个 PID pairs 全部
  `<0.5`，组内最大值为 `0.3438`；四组 median PID 分别为 `0.1667/0.2500/0.2258/0.1818`。
  pooled view 中 PID `0.8947` 的 ApexOracle-14/23 来自不同 target strains。

这些结果支持“没有观察到明显 exact-structure collapse”；all-pairs distribution 仍不是
nearest-neighbor distribution，也不能把 84,226 条历史弱结构过滤输出全部称为严格 peptide。
candidate level 使用 `peptide candidate pool`；generation level 应使用中性的
`guided generation outputs`。

## Exact all-pairs 运行记录

本机生成的 21 MiB fingerprint cache SHA-256 为
`a2b0a9789670057665ad124cb5c0fd9105c70030a809909c1a0b907074c764bf`。运行使用 64 workers
在 21.72 秒内完成全部 3.547 billion comparisons；含 cache 加载总计 22.62 秒，实际吞吐约
163.3 million pairs/s。该计算只执行 precomputed bit-vector Tanimoto，不重新生成 fingerprint；
fingerprint 仍由本机冻结的 RDKit 2024.09.6 protocol 产生。

初步 1,000,000-pair sample 与 exact 结果一致：mean 差 `-7.66e-5`；sample median 和 q95
分别落入 exact 0.001-wide quantile bins；`Tanimoto >= 0.5` 的 sample/exact 比例为
0.8107%/0.8252%，差值 0.0145 percentage points，位于预期 95% binomial half-width
0.0177 percentage points 内。这说明一百万 pairs 本身已足以估计总体分布，但正式结果采用
exact all-pairs 避免抽样误差。


## Figure

The canonical three-panel figure is `generated_candidate_diversity_three_panel.pdf`.
Panel a shows within-target sequence identity for the 24 synthesized peptides;
panels b/c show structural similarity for the 73-candidate pool and 84,226
guided outputs. See `figure_caption.md` and the figure manifest for the exact scope.
