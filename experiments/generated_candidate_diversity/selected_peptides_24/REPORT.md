# 24 条最终 peptide candidates 的 pairwise diversity

## 已由代码和冻结输入验证的事实

- 输入为 `canonical_candidates/final_peptide_mapping.csv` 中的 24 条最终合成序列，并通过
  `canonical_candidates/candidates_73.csv` 中的映射行取得对应 final canonical isomeric
  structures。24 条对应 24 个不同 canonical isomeric structures。
- Sequence similarity 复用论文三条 lead 对 training set 的 Biopython global-alignment 口径：
  chirality-aware BLOSUM62、gap-open `-10`、gap-extension `-0.5`，PID 为 case-sensitive exact
  matches 除以含 gaps 的最终 alignment length。Linear 只与 linear 比，cyclic 只与 cyclic 比；
  cyclic pairs 穷举两条序列的全部 rotations。由于这里是无序 pairs，额外对调 target/query
  orientation 并保留更高 PID，避免 Biopython first-optimal tie 使结果依赖 ApexOracle ID 顺序。
- 18 条 linear 和 6 条 cyclic peptides 产生 168 个同拓扑无序 pairs；108 个 linear--cyclic pairs
  按冻结论文口径不可比。168 个 PID 的 median 为 `0.1719`，95th percentile 为 `0.2857`。
  167/168 pairs 的 PID 小于 `0.5`。唯一高相似 pair 是 ApexOracle-14/ApexOracle-23：
  `17/19 = 0.8947`；其余 pairs 的最高 PID 为 `0.3438`。
- Structural similarity 对全部 `24 choose 2 = 276` 个无序 pairs 使用 RDKit Morgan radius-2、
  2,048-bit、`includeChirality=True` fingerprints 和 Tanimoto coefficient。Median 为 `0.4633`，
  95th percentile 为 `0.7033`，15/276（5.43%）为 `>=0.7`，2/276（0.72%）为 `>=0.8`，没有
  pair 为 `>=0.9`。最高值为 ApexOracle-5/ApexOracle-11 的 `0.8560`。
- 按 generation target 与 topology 同时分层后，BAA-3197 linear、BAA-3197 cyclic、BAA-3170
  linear、BAA-3170 cyclic 分别有 12、3、6、3 条 peptides，对应 66、3、15、3 个组内 pairs，
  共 87 个。PID median 分别为 `0.1667`、`0.2500`、`0.2258`、`0.1818`；87/87 均低于
  `0.5`，组内最高 PID 为 `0.3438`。原 pooled same-topology 分析中唯一的高 PID pair
  ApexOracle-14/ApexOracle-23 跨越 BAA-3197 与 BAA-3170，因此不进入 target-stratified 分析。
  对相同 87 个组内 pairs，Morgan/Tanimoto median 分别为 `0.5448`、`0.4493`、`0.5175`、
  `0.5035`。

## 根据现有结果作出的解释

- 这些结果支持“最终实验 panel 不是 duplicate-dominated 或普遍由高相似 analogues 构成”：全部
  结构均唯一，sequence PID 除一个明确的 cyclic near-neighbor pair 外普遍较低，Morgan
  Tanimoto 也没有达到 `0.9` 的 pairs。
- ApexOracle-14/ApexOracle-23 应明确披露而不是隐藏；它们代表 panel 中一个局部 cyclic
  analogue pair，但二者来自不同 generation targets。若论点是每个 target 内部 selection 的
  diversity，应以 87-pair target-stratified 结果为主；若描述合并后的完整 24-peptide panel，仍可
  同时披露 pooled 168-pair 结果。

## 不能由本分析单独证明的事项

- 该分析描述 selected panel 的 diversity，但不能单独证明 diversity-aware selection 相对随机
  selection 或按 predicted MIC 排名前 24 条提高了 diversity。24 条 selected peptides 的 median
  Tanimoto（`0.4633`）高于含重复行的 73-row candidate pool（`0.375`），因此不能写成“selection
  lowered median Tanimoto relative to the candidate pool”。
- 当前没有冻结的 73 条逐条 predicted-MIC table，无法可靠重建 predicted-MIC top-24 comparator。
  对外表述应使用“the selected panel remained structurally and sequence diverse”，而不是声称该
  人工 selection procedure 已由 causal comparator 证明提升 diversity。

## 产物

- `pairwise_sequence_similarity.csv`：168 个同拓扑 pairs 的完整 alignment/PID；
- `pairwise_tanimoto.csv`：276 个结构 pairs；
- `pairwise_sequence_pid_matrix.csv`：24 × 24 matrix，linear--cyclic cells 留空；
- `pairwise_tanimoto_matrix.csv`：完整 24 × 24 matrix；
- `nearest_neighbors.csv`：每条 peptide 的 sequence 和 structure nearest neighbor；
- `selected_peptide_pairwise_similarity_violin.{pdf,png,svg}`：共享 0--1 纵轴的 PID/Tanimoto
  双 panel violin plot，叠加全部 pairwise observations、IQR、median 和最高相似 pair；
- `selected_peptide_pairwise_similarity_violin_caption.md`：图注草稿与 linear--cyclic PID 边界；
- `pairwise_similarity_by_strain_topology_summary.csv`：四个 `target strain × topology` 组的
  peptide 数、pair 数和 PID/Tanimoto summary；
- `selected_peptide_pairwise_similarity_by_strain_topology.{pdf,png,svg}`：不混合 target strain
  或 topology 的双 panel 分布图；只有 3 pairs 的 cyclic 组不绘制平滑 violin；
- `selected_peptide_pairwise_similarity_by_strain_topology_caption.md`：分组图图注；
- `summary.json` 与 `manifest.json`：统计摘要和文件 SHA-256 血缘。
