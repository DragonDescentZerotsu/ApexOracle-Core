# Guided-generation candidate diversity

本目录回答 reviewer 对 generated-candidate diversity、structurally distinct candidates 和
Tanimoto similarity distribution 的问题。分析只使用最终候选对应的两个 MIC-guided targets
`BAA-3170` 与 `BAA-3197`，不加入 unguided 或 `gamma_peptide=0` control。

Canonical 入口：

```bash
MPLBACKEND=Agg PYTHONPATH=src \
  /home/tianang/anaconda3/bin/python \
  scripts/audit/analyze_generated_candidate_diversity.py
```

24 条最终合成 peptides 的 pairwise sequence/structure diversity 入口：

```bash
PYTHONPATH=src /home/tianang/anaconda3/bin/python \
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
`generated_mol_SELFIES_w_mic-new/` 的 73-row reviewer 文件布局副本，保留原始
`strain × target length × source line` 布局。它可由 canonical audit 入口从外部只读输入重建，
因此作为 provenance intermediate 受 `.gitignore` 保护，不进入 Git。根据作者确认，8 个最近生成
前体被最终合成序列替换；其中 ApexOracle-14 的前体在两个 target lengths 重复出现，因此实际
替换 9 行。其余 64 行保持原始生成结构不变。

`canonical_candidates/candidates_73.csv` 同时保存 original/corrected sequence、SELFIES、
canonical isomeric SMILES、replacement status 和逐文件 SHA-256。所有后续 candidate-level
reviewer 指标必须消费这份副本，不再直接消费历史目录。24 条最终合成 peptide 均通过
topology-aware exact mapping 在副本中出现；环肽允许 cyclic rotation。

Candidate-level 主分析保留全部 73 行，包括重复候选，并计算全部
`73 choose 2 = 2,628` 个无序非自身 pair。另报告 canonical isomeric structure 去重数。

### Candidate screening 血缘边界

**已由历史代码验证的事实：** `/data2/tianang/projects/mdlm/judge_mol_mic_with_fig.py`
使用 clean 13-epoch MIC regressor 对 target strain 评分，通过 `if mic > 15: continue` 保留
predicted MIC <=15 µM 的 outputs；随后要求结构能够解析为 peptide sequence 且不含 unresolved
`X` residue，并要求结构可由 RDKit/SELFIES 继续处理。predicted MIC 因而是 post-generation
computational prioritization criterion，不是实验 MIC。

**根据现有资产作出的推断：** `generated_mol_SELFIES_w_mic-new/` 的 73 条 target-strain rows
是该 screening stage 保存的 peptide candidate pool；其 target、length 布局与原始 84,226 条
MIC-guided outputs 以及上述历史筛选代码一致。当前没有找到冻结的 producer command/log 或逐条
predicted-MIC table，因此内部复现记录不能把现有脚本 checkout 声称为这 73 条的 byte-exact
producer。Reviewer 文稿只陈述已确认的筛选逻辑与分母链，不声称恢复了逐条历史 prediction。

从 73 条 pool 选择 24 条合成时，按 structural/sequence diversity 与 practical synthesis
feasibility 进行人工 prioritization，并降低过度疏水或含 aggregation-prone motifs 的候选优先级；
predicted MIC 不得写成实验 activity。

### Generation level

Generation-level 分析只读消费原始 `generated_mol_SELFIES-new/` 中 `BAA-3170` 与
`BAA-3197` 的全部 target lengths，共 84,226 条保存输出；不同长度直接 pooled。目录中另有
107 条 `BAA-1556` 输出，不属于本次候选对应 targets，已由显式 strain allowlist 排除。

对 84,226 条输出精确计算 canonical isomeric structure 去重数。全体两两组合超过 35 亿，
正式 Tanimoto distribution 已在 node002 用 64 workers 流式穷举全部
`84,226 choose 2 = 3,546,967,425` 个无序非自身 pairs；只累计 1,000-bin histogram、矩和阈值
计数，不保存 13.2 GiB 的逐 pair float32 数组。固定 seed `20260802` 的 1,000,000-pair 初步抽样
仍保留在 `summary.json` 作为内部稳定性检查，但不再作为 reviewer-facing 主结果。

## 指纹与输出

两个层级使用相同的 RDKit Morgan bit fingerprint：radius 2、2,048 bits、
`includeChirality=True`，相似度为 Tanimoto。主要产物：

- `summary.json`：集合大小、唯一结构数、分位数和高相似度 pair 比例；
- `candidate_pairwise_tanimoto.csv`：73 行候选的全部 2,628 个 pair；
- `generation_all_pairs_tanimoto_histogram.csv`：84,226 条 outputs 的全部 3,546,967,425 pairs，
  以 0.001-wide bins 保存 exact counts；
- `generation_all_pairs_tanimoto_summary.json`：exact moments、quantile-containing bins、阈值计数、
  node002 runtime/environment 和 cache/runner SHA-256；
- `tanimoto_histogram_plotted_data.csv`：图中精确 bin counts/fractions；
- `generated_candidate_diversity_three_panel.{pdf,svg,png}`：正式 Supplementary Fig. C5，包含
  selected-24 PID、candidate-pool Tanimoto 和 generation-output Tanimoto；
- `canonical_candidates/replacement_audit.csv`：9 个替换行；
- `canonical_candidates/final_peptide_mapping.csv`：24 条最终合成 peptide 的副本映射；
- `manifest.json`：全部输入、输出、外部 parser/PepLink 与 SHA-256 血缘。
- `reviewer_response_draft.md`：只使用上述 canonical 副本和冻结分母的英文回复草稿。
- `molport_selection_audit.md`：18 条 purchased compounds 的 44,608-entry screen、MolPort
  exact-match、structural-alert、cluster 与 19-to-18 procurement 血缘；明确区分已验证事实、推断和
  尚无证据支持的 delivery-time 说法。

早期两 panel `guided_generation_tanimoto_distributions.*` 和
`selected_peptides_24/selected_peptide_pairwise_similarity_*` 继续保留为本地诊断产物，不进入
Git；它们的 exact CSV、summary、caption/manifest 血缘和生成代码仍保留，功能没有删除。

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
Reviewer 文稿中，candidate level 可称为 `peptide candidate pool`；generation level 应使用中性的
`guided generation outputs`。

## Exact all-pairs 运行记录

本机生成的 21 MiB fingerprint cache SHA-256 为
`a2b0a9789670057665ad124cb5c0fd9105c70030a809909c1a0b907074c764bf`。node002 使用 64 workers
在 21.72 秒内完成全部 3.547 billion comparisons；含 cache 加载总计 22.62 秒，实际吞吐约
163.3 million pairs/s。node002 只执行 precomputed bit-vector Tanimoto，不重新生成 fingerprint；
fingerprint 仍由本机冻结的 RDKit 2024.09.6 protocol 产生。

初步 1,000,000-pair sample 与 exact 结果一致：mean 差 `-7.66e-5`；sample median 和 q95
分别落入 exact 0.001-wide quantile bins；`Tanimoto >= 0.5` 的 sample/exact 比例为
0.8107%/0.8252%，差值 0.0145 percentage points，位于预期 95% binomial half-width
0.0177 percentage points 内。这说明一百万 pairs 本身已足以估计总体分布，但正式结果采用
exact all-pairs 以消除 reviewer 对抽样规模的疑问。

## 正式文稿落稿（2026-08-02）

- 正式 TeX 已按作者确认的 plan 修改 Results、Methods 和 Supplementary Information；原
  `spanning a range of predicted MIC values` 已替换为 `<=15 µM` threshold 加
  diversity/feasibility prioritization，并加入 peptide/small-molecule intended-target hit rates。
- 合作者原回复中需要保留的三点已明确进入正文与 `reviewer_response_draft.md`：diversity-aware
  selection 用于避免近重复 top analogues；Fig. 3a 是 predicted candidate distributions 的
  *in silico* comparison；prospective MIC measurements 才是 selected candidates 的直接实验评价。
- 图已复制为正式文稿目录的 `Fig_SI_generated_candidate_diversity.pdf`，与 canonical PDF 的
  SHA-256 均为 `024d95d9d776b3988100827f4a2c94d866b0a977efe7854eeaccaf7539f2d296`。
- 修改前 TeX 备份为 `sn-article_before_generated_diversity_selection_20260802.tex`。独立临时编译
  为 32 页，Supplementary Fig. C3/C4/C5 与 Appendix D 编号已核验；正式 `sn-article.pdf` 未覆盖。
- 两条 selection comments 使用同一组 manuscript revisions，但保留各自完整回复：Reviewer 4
  额外回答为什么不只选择预测最强者和 exact hit-rate 分布。正式
  `Response to reviewers letter.docx` 已合并 diversity 和两条 selection/hit-rate 回复；修改前备份为
  `Response to reviewers letter_before_generated_diversity_selection_20260802.docx`。DOCX 独立渲染为 30 页，
  相关页面格式已核验。
- Supplementary Fig. B2 已替换为全部 24 条 tested peptides 和 18 个 tested small molecules
  在 20 个 strains 上的完整 MIC heatmaps。正式 `Fig_SI_heatmap_re.pdf` 来自作者提供的
  Mac v4 PDF，SHA-256 为
  `5ac3bd00e52958ecb06bc066e29de4752863b0f31d851e18df531954c1ae2693`；旧图备份为
  `Fig_SI_heatmap_re_before_all_candidates_20260802.pdf`。TeX 改为双栏 `figure*`/`\textwidth`，
  独立编译为 32 页，B2 位于第 23 页且 C3--C5 编号不变。两条 selection
  reviewer replies 已增加 complete B2 matrices 的回指；最终 reviewer DOCX SHA-256 为
  `5da0fc9894c861733917438b07904cb22777f800cceece7185b067b0464473bc`。

## Supplementary Fig. C5 selected-panel 扩展（2026-08-03）

- `scripts/audit/plot_generated_candidate_diversity_figure.py` 将 selected-panel within-target PID
  与原两组 Tanimoto histograms 组成单行三 panel 图：panel a 为 24 条 synthesized peptides，
  panels b/c 分别为 73-candidate pool 和 84,226 guided outputs。引用顺序为 24→73→84,226，
  因此 PID 是新 panel a，原 a/b 顺延为 b/c。
- Canonical 图为 `generated_candidate_diversity_three_panel.pdf`，SHA-256 为
  `72c50e1649720233a3a45ba46d57d21df8fa68ffb2660aba886b3a4f491a8ab8`；正式论文目录的
  `Fig_SI_generated_candidate_diversity.pdf` 与其一致。旧两 panel 正式图备份为
  `Fig_SI_generated_candidate_diversity_before_selected_pid_20260803.pdf`，SHA-256 为
  `024d95d9d776b3988100827f4a2c94d866b0a977efe7854eeaccaf7539f2d296`。
- Panel a 使用完整 0--100\% PID 纵轴；`P. aeruginosa PA5257` 和 `E. coli AR-0349` 放在
  40\% 以上的空白区，x 轴只保留 Lin./Cyc. 与 pair 数。最右侧 18.2\% median label 在 cyclic
  group 上方居中，不越过 panel a/b 边界。
- 正式 TeX Results 新增 87/87 within-group PID `<50%` 及四组 median，Methods 新增
  selected-peptide sequence-diversity protocol；caption 明确正式 strain 编号、四组 peptide/pair 数
  和 Lin./Cyc. 定义，0--100\% PID 轴与排除的 comparison scope 分别保留在图和 Methods，不在
  caption 重复。Reviewer
  diversity 与两条 selection 回复均加入 C5a 的 87-pair 定量结果，并将整体组织为 selected-panel、
  candidate-pool 和 generation-output 三个互补层级。正式 response DOCX 修改前备份为
  `Response to reviewers letter_before_selected_pid_c5_20260803.docx`，更新后 SHA-256 为
  `5da0fc9894c861733917438b07904cb22777f800cceece7185b067b0464473bc`。TeX 独立编译为
  32 页，C5 位于第 24 页并与 C4 同页；response DOCX 独立渲染为 30 页；两者相关页面均已
  目视核验，正式 `sn-article.pdf` 未覆盖。
- 2026-08-03 按作者要求重组新增 Methods：guided generation/remasking 继续留在
  `ApexOracle Architecture and Training`；peptide prioritization 与 small-molecule screen 合并到新
  `Candidate prioritization and virtual screening` subsection；原 `Sequence similarity analysis`
  扩展为 `Sequence and structural diversity analyses`，先统一定义 alignment/PID，再依次描述
  lead-to-training、selected-24 sequence 和 73/84,226 structural analyses。协议与数值均未改变；
  修改前备份为 `sn-article_before_methods_reorganization_20260803.tex`。独立编译仍为 32 页，
  两个新 subsection 位于第 16--17 页并完成目视核验，正式 `sn-article.pdf` 未覆盖。
